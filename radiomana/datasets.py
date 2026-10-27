"""loaders for public datasets"""

import os
from collections import Counter
from pathlib import Path

import lightning as L
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2

from .transforms import BenignSpurs, LogNoise, RandomFreqFlip, RandomFreqShift, RandomTimeCrop, RandomTimeRoll, SyntheticJammer

DSET_ENV_LUT = {
    # lookup table for environment variable to dataset source URL
    "DSET_FIOT_HIGHWAY1": "https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/FIOT_highway",
    "DSET_FIOT_HIGHWAY2": "https://gitlab.cc-asp.fraunhofer.de/darcy_gnss/fiot_highway2",
}


def get_dataset_path(some_env_var: str) -> Path:
    """
    Given an environment variable name, return the Path object for the dataset directory.
    If the environment variable is not set, raise an error.
    """
    dataset_path = os.getenv(some_env_var)
    if not dataset_path:
        raise ValueError(
            f"Environment variable {some_env_var} is not set. Please clone the dataset from {DSET_ENV_LUT.get(some_env_var, 'the appropriate source')} and set the environment variable accordingly."
        )
    return Path(dataset_path)


class Highway2Dataset(Dataset):
    """
    Dataset class for the FIOT Highway2 dataset

    This 30GB dataset contains PSDs of shape (freq, time) = (512, 243).

    working @ dataset git rev 6246f6

    items are indexed by text files in the root directory and contain rows with "folder/file class_label"
    """

    def __init__(
        self,
        root_dir: str = "DSET_FIOT_HIGHWAY2",
        subset: str = "train",
        transform=None,
    ):
        if subset not in ["train", "test"]:
            raise ValueError("subset must be 'train' or 'test'")
        self.root_dir = get_dataset_path(root_dir)
        self.subset = subset
        self.items = self.load_items()
        self.transform = transform

        # metadata from dataset readme
        self.sample_rate_hz = 62.5e6
        self.sample_duration_s = 0.02
        self.class_labels = [
            "None (background a)",
            "None (background b)",
            "None (background c)",
            "None (background d)",
            "Chirp, high distance",
            "Chirp, medium distance",
            "Chirp, small distance",
            "Cigarette lighter 1",
            "Cigarette lighter 2",
        ]

    def load_items(self) -> list:
        labels_path = self.root_dir / f"{self.subset}.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"Items file not found at {labels_path}")
        items = []
        with open(labels_path) as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) == 2:
                    items.append((parts[0], int(parts[1])))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        Get a single PSD and its label.

        PSDs are in dB units, with noise floor around -90 dB.

        Note this function could be accelerated significantly with stochastic caching and shared memory between workers.
        https://charl-ai.github.io/blog/dataloaders/
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
        file_path, label = self.items[idx]
        # load the data from the file (implementation depends on file format)
        data = torch.from_numpy(np.load(self.root_dir / file_path))
        # apply any transforms
        if self.transform:
            data = self.transform(data)
        # stored as float64, but precision is overkill, so cast to float32
        data = data.type(torch.float32)
        return data, label


class PairTransformDataset(Dataset):
    """Wrap a dataset so transforms see and may change the (data, label) pair, e.g. SyntheticJammer relabelling."""

    def __init__(self, dataset: Dataset, transforms: list):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class BinaryRelabelDataset(Dataset):
    """
    Map another site's labels onto Highway2's: jammed samples (label >= jammed_min) become jammed_label, the rest clean_label.

    Used to train on Highway1, whose 11-class taxonomy doesn't match Highway2's 9 classes, so only jammed / clean is kept.
    """

    def __init__(self, dataset: Dataset, jammed_min: int = 3, jammed_label: int = 4, clean_label: int = 0):
        self.dataset = dataset
        self.jammed_min = jammed_min
        self.jammed_label = jammed_label
        self.clean_label = clean_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, self.jammed_label if label >= self.jammed_min else self.clean_label


class HighwayDataModule(L.LightningDataModule):
    """
    Lightning datamodule for the FIOT Highway2 dataset

    Implements 5 key methods
    - prepare_data: things to do on 1 accelerator only (download, tokenize, etc)
    - setup: things to do on every accelerator (split dataset, etc)
    - train_dataloader: return the training dataloader
    - val_dataloader: return the validation dataloader
    - test_dataloader: return the test dataloader
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_cross_site_augmentation: bool = False,
        highway1_train_root: str | None = None,
        highway1_jam_weight: float = 8.0,
    ):
        """
        use_cross_site_augmentation can be set here because Trainer.fit calls setup() with only the stage

        highway1_train_root : str, optional
            environment variable of a Highway1 dataset (e.g. "DSET_FIOT_HIGHWAY1_SUBSET"); its train.txt split
            is added to training with labels mapped to jammed (class 4) / clean (class 0). Its test.txt split
            stays held out. Only 1 in 64 Highway1 samples is jammed, so those are drawn highway1_jam_weight
            times more often than other samples.
        """
        super().__init__()
        # this allows access to all hparams via self.hparams
        self.save_hyperparameters(logger=False)

    def setup(
        self,
        stage=None,
        root_dir: str = "DSET_FIOT_HIGHWAY2",
        use_oversampling: bool = False,
        use_augmentation: bool = False,
        use_cross_site_augmentation: bool = False,
    ):
        """
        called on every process in DDP

        Parameters
        ----------
        use_oversampling : bool, optional
            whether to apply oversampling to the training set, by default False
        use_augmentation : bool, optional
            whether to apply data augmentation to the training set, by default False
        use_cross_site_augmentation : bool, optional
            whether to apply the augmentations that let a Highway2-trained model transfer to Highway1:
            random time roll / frequency shift / frequency flip, synthetic tone, pulsed and noise jammers
            (relabelled to class 4), and faint benign spurs on clean samples, by default False
        """
        augmentations = v2.Compose(
            [
                v2.RandomErasing(p=1, value=-90),
                RandomTimeCrop(crop_width=211),
                v2.RandomChoice(
                    [
                        v2.Identity(),
                        LogNoise(noise_power_db=-110, p=1),
                        LogNoise(noise_power_db=-90, p=1),
                        LogNoise(noise_power_db=-70, p=1),
                    ],
                ),
            ]
        )
        use_cross_site_augmentation = use_cross_site_augmentation or self.hparams.use_cross_site_augmentation
        train_transforms = []
        if use_augmentation:
            train_transforms.append(augmentations)
        if use_cross_site_augmentation:
            train_transforms += [RandomTimeRoll(), RandomFreqShift(max_fraction=0.1), RandomFreqFlip(p=0.5)]
        data_train = Highway2Dataset(
            root_dir=root_dir,
            subset="train",
            transform=v2.Compose(train_transforms) if train_transforms else None,
        )
        data_val = Highway2Dataset(root_dir=root_dir, subset="train", transform=None)
        self.data_test = Highway2Dataset(root_dir=root_dir, subset="test", transform=None)

        # split train into train/val, stratified by class label
        labels = [label for _, label in data_train.items]
        train_indices, val_indices = train_test_split(
            np.arange(len(data_train)),
            test_size=0.1,
            stratify=labels,
            random_state=0xD00D1E,
        )

        if use_oversampling:
            # apply oversampling to training set only
            train_labels = [labels[idx] for idx in train_indices]

            # get counts for ALL classes first to find true majority
            all_train_counts = Counter(train_labels)
            print(f"full training set distribution: {all_train_counts}")

            # only oversample classes with few examples
            non_none_mask = np.array(train_labels) >= 3
            interference_indices = train_indices[non_none_mask]
            interference_labels = np.array(train_labels)[non_none_mask]

            orig_counts = Counter(interference_labels)
            print(f"original interference class distribution: {orig_counts}")

            # oversample interference classes to balance them
            # note: for PSD data, we use RandomOverSampler instead of SMOTE
            # since SMOTE creates synthetic samples which may not be realistic for PSDs

            # Option 1: Set specific target counts for each class
            # sampling_strategy = {4: 200, 5: 200, 6: 200, 7: 200, 8: 200}  # Equal counts

            # Option 2: Oversample to percentage of TRUE majority class (including None classes)
            max_count = max(all_train_counts.values())
            target_ratio = 0.1  # 10% of true majority class
            target_count = int(target_ratio * max_count)
            sampling_strategy = {cls: max(count, target_count) for cls, count in orig_counts.items()}

            # Option 3: Set maximum samples per class (prevents over-oversampling)
            # max_samples_per_class = 500
            # sampling_strategy = {cls: min(count, max_samples_per_class) for cls, count in orig_counts.items()}

            print(f"target sampling strategy: {sampling_strategy}")
            print(f"true majority class has {max_count} samples, targeting {target_count} samples ({target_ratio * 100:.0f}%)")

            oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0xC0FFEE)

            # reshape for imblearn (needs 2D)
            interference_indices_reshaped = interference_indices.reshape(-1, 1)
            resampled_indices_2d, resampled_labels = oversampler.fit_resample(interference_indices_reshaped, interference_labels)
            resampled_indices = np.array(resampled_indices_2d).flatten()

            print(f"resampled interference class distribution: {Counter(resampled_labels)}")

            # combine none classes with oversampled interference classes
            none_indices = train_indices[~non_none_mask]
            final_train_indices = np.concatenate([none_indices, resampled_indices])

            # shuffle final indices
            np.random.seed(0xD00D1E)
            np.random.shuffle(final_train_indices)

            self.data_train = Subset(data_train, final_train_indices.tolist())
        else:
            self.data_train = Subset(data_train, train_indices.tolist())

        self.train_weights = None
        if self.hparams.highway1_train_root:
            data_hw1 = Highway2Dataset(
                root_dir=self.hparams.highway1_train_root,
                subset="train",
                transform=v2.Compose(train_transforms) if train_transforms else None,
            )
            hw1_jammed = np.array([label >= 3 for _, label in data_hw1.items])
            self.train_weights = np.concatenate([np.ones(len(self.data_train)), np.where(hw1_jammed, self.hparams.highway1_jam_weight, 1.0)])
            self.data_train = ConcatDataset([self.data_train, BinaryRelabelDataset(data_hw1)])

        if use_cross_site_augmentation:
            self.data_train = PairTransformDataset(self.data_train, [SyntheticJammer(p=0.15, min_db=12.0), BenignSpurs(p=0.3)])

        self.data_val = Subset(data_val, val_indices.tolist())

    def train_dataloader(self):
        sampler = None
        if self.train_weights is not None:
            sampler = WeightedRandomSampler(torch.from_numpy(self.train_weights), num_samples=len(self.train_weights), replacement=True)
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
