"""Revalidate saved artifacts through the unmodified RadioManA loader."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from radiomana.datasets import Highway2Dataset, HighwayDataModule
from radiomana.highway2_synthetic import augment, digest
from radiomana.spectral import bin_frequencies, spectral_stats
from radiomana.synthetic_data import SyntheticHighwayDataModule


def main(out):
    out = Path(out).resolve()
    summary = json.loads((out / "summary.json").read_text())
    os.environ["TASK1_OUTPUT"] = str(out)
    os.environ["TASK1_ORIGINAL"] = summary["root"]
    original = HighwayDataModule(num_workers=0)
    original.setup(root_dir="TASK1_ORIGINAL")
    frozen = SyntheticHighwayDataModule(out)
    frozen.setup()
    for split in ["val", "test"]:
        reference = getattr(original, "data_" + split)
        actual = getattr(frozen, "data_" + split)
        ref_items = [reference.dataset.items[i] for i in reference.indices] if split == "val" else reference.items
        assert [(str(Path(summary["root"]) / p), c) for p, c in ref_items] == actual.items
    ds = Highway2Dataset(root_dir="TASK1_OUTPUT")
    provenance = list(csv.DictReader((out / "provenance.csv").open()))
    fit_items = {original.data_train.dataset.items[i] for i in original.data_train.indices}
    expected_original = [(str(Path(summary["root"]) / p), c) for p, c in [original.data_train.dataset.items[i] for i in original.data_train.indices]]
    assert ds.items[: len(expected_original)] == expected_original
    assert len(ds) == sum(summary["augmented"].values())
    metrics = []
    parent_counts = Counter()
    start = sum(summary["original"].values())
    for i, row in enumerate(provenance):
        assert (row["parent"], int(row["label"])) in fit_items
        x = np.load(Path(summary["root"]) / row["parent"], allow_pickle=False)
        y, label = ds[start + i]
        assert label == int(row["label"]) and tuple(y.shape) == (512, 243)
        assert torch.isfinite(y).all() and y.dtype == torch.float32
        assert digest(y.numpy()) == row["sha256"]
        reproduced, _ = augment(x, int(row["seed"]))
        assert np.array_equal(y.numpy(), reproduced)
        stats = spectral_stats(torch.stack([torch.from_numpy(x), y]), bin_frequencies(ds))
        metrics.append({k: abs(float(v[1] - v[0])) for k, v in stats.items()})
        parent_counts[row["parent"]] += 1
    assert max(parent_counts.values()) <= summary["max_variants"]
    batch, _labels = next(iter(frozen.train_dataloader()))
    assert batch.shape == (32, 512, 243)
    results = {
        "samples_checked": len(provenance),
        "exact_seed_reproductions": len(provenance),
        "validation_matches_original": True,
        "all_parents_in_original_fitting_split": True,
        "test_matches_original": True,
        "maximum_variants_per_parent": max(parent_counts.values()),
        "maximum_spectral_deltas": {k: max(m[k] for m in metrics) for k in metrics[0]},
        "batch_shape": list(batch.shape),
    }
    (out / "loader_validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
