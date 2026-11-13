"""Frozen manifests for synthetic experiments: never resplit augmented data."""

from pathlib import Path

from .datasets import Highway2Dataset, HighwayDataModule


class ManifestHighway2Dataset(Highway2Dataset):
    """Reuse Highway2Dataset loading/metadata with explicit frozen manifest."""

    def __init__(self, root, manifest, transform=None):
        # Avoid mutating process environment to initialize repository metadata.
        self.root_dir = Path(root).resolve()
        self.subset = manifest
        self.transform = transform
        self.sample_rate_hz = 62.5e6
        self.sample_duration_s = 0.02
        from .highway2_synthetic import LABELS

        self.class_labels = LABELS
        self.items = []
        for line in (self.root_dir / manifest).read_text().splitlines():
            p, c = line.split()
            if int(c) not in range(9):
                raise ValueError("Invalid label")
            self.items.append((p, int(c)))


class SyntheticHighwayDataModule(HighwayDataModule):
    """Same DataLoaders as RadioManA, but with train/val/test already frozen."""

    def __init__(self, root, augmented=True, batch_size=32, num_workers=0):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        self.manifest_root = Path(root)
        self.augmented = augmented

    def setup(self, stage=None):
        if not (self.manifest_root / "COMPLETE").exists():
            raise ValueError("Generation incomplete")
        self.data_train = ManifestHighway2Dataset(self.manifest_root, "train.txt" if self.augmented else "train_original.txt")
        self.data_val = ManifestHighway2Dataset(self.manifest_root, "val.txt")
        self.data_test = ManifestHighway2Dataset(self.manifest_root, "test.txt")
        self.train_weights = None
