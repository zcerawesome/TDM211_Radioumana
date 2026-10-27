"""unit tests for datasets"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from radiomana.datasets import BinaryRelabelDataset, HighwayDataModule


def make_fake_dataset(root: Path, labels: list):
    """write tiny (512, 243) PSDs and a train.txt / test.txt index in the FIOT layout"""
    (root / "data").mkdir(parents=True)
    for subset in ["train", "test"]:
        with open(root / f"{subset}.txt", "w") as handle:
            for idx, label in enumerate(labels):
                name = f"data/{subset}_{idx}.npy"
                np.save(root / name, np.full((512, 243), -90.0, dtype=np.float32))
                handle.write(f"{name} {label}\n")


class TestHighway1Training(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_fake_dataset(root / "hw2", [0] * 10 + [4] * 10)
        make_fake_dataset(root / "hw1", [1] * 8 + [5, 7])
        os.environ["DSET_TEST_HW2"] = str(root / "hw2")
        os.environ["DSET_TEST_HW1"] = str(root / "hw1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_binary_relabel(self):
        """Highway1 classes 3-10 become Highway2 class 4, the rest class 0"""
        data = BinaryRelabelDataset([(None, label) for label in range(11)])
        self.assertEqual([data[idx][1] for idx in range(11)], [0, 0, 0] + [4] * 8)

    def test_highway1_train_split_added(self):
        """train.txt of Highway1 is appended with mapped labels and oversampled jammed samples"""
        dmodule = HighwayDataModule(batch_size=4, num_workers=0, highway1_train_root="DSET_TEST_HW1", highway1_jam_weight=8.0)
        dmodule.setup(root_dir="DSET_TEST_HW2")
        self.assertEqual(len(dmodule.data_train), 18 + 10)
        hw1_labels = [dmodule.data_train[idx][1] for idx in range(18, 28)]
        self.assertEqual(hw1_labels, [0] * 8 + [4, 4])
        self.assertEqual(list(dmodule.train_weights[18:]), [1.0] * 8 + [8.0, 8.0])
        x, y = next(iter(dmodule.train_dataloader()))
        self.assertEqual(tuple(x.shape), (4, 512, 243))

    def test_default_has_no_highway1(self):
        dmodule = HighwayDataModule(batch_size=4, num_workers=0)
        dmodule.setup(root_dir="DSET_TEST_HW2")
        self.assertEqual(len(dmodule.data_train), 18)
        self.assertIsNone(dmodule.train_weights)


if __name__ == "__main__":
    unittest.main()
