"""Task 1 regression tests, independent of the private/local dataset."""

from collections import Counter

import numpy as np
import pytest

from radiomana.highway2_synthetic import augment, plan, read_manifest, run, validate_array
from radiomana.synthetic_data import ManifestHighway2Dataset


def signal():
    rng = np.random.default_rng(23)
    return rng.normal(-40, 5, (512, 243)).astype("float32")


def test_positive_power_and_reproducibility():
    x = signal()
    saved = x.copy()
    a, m = augment(x, 42)
    b, _ = augment(x, 42)
    c, _ = augment(x, 43)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.array_equal(x, saved)
    assert a.dtype == x.dtype and a.shape == x.shape
    assert np.all(a >= x + m["gain_db"] - 1e-5)
    assert m["correlation"] > 0.995 and m["rmse_db"] < 1


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "inf"])
def test_invalid_arrays(kind):
    x = signal()
    if kind == "shape":
        x = x[:5]
    if kind == "dtype":
        x = x.astype("int32")
    if kind == "nan":
        x[0, 0] = np.nan
    if kind == "inf":
        x[0, 0] = np.inf
    with pytest.raises(ValueError):
        validate_array(x)


def test_budget_includes_rare_background_and_handles_absence():
    threshold, counts = plan(Counter({0: 5000, 3: 43, 4: 158, 5: 259, 6: 302, 7: 144, 8: 115}))
    assert threshold == 500
    assert counts == {0: 0, 1: 0, 2: 0, 3: 86, 4: 316, 5: 241, 6: 198, 7: 288, 8: 230}


def test_manifest_traversal(tmp_path):
    (tmp_path / "train.txt").write_text("../escape.npy 0\n")
    with pytest.raises(ValueError):
        read_manifest(tmp_path, "train.txt")


def test_original_loader_compatibility(tmp_path):
    x = signal()
    np.save(tmp_path / "sample.npy", x)
    (tmp_path / "train.txt").write_text("sample.npy 3\n")
    ds = ManifestHighway2Dataset(tmp_path, "train.txt")
    y, label = ds[0]
    assert label == 3 and np.array_equal(y.numpy(), x)


def test_overlap_blocks_generation(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    x = signal()
    # All labels have sufficient members for stratification; exact copies leak.
    rows = []
    for c in range(9):
        for i in range(10):
            p = f"{c}_{i}.npy"
            np.save(root / p, x + c)
            rows.append(f"{p} {c}\n")
    (root / "train.txt").write_text("".join(rows))
    (root / "test.txt").write_text("0_0.npy 0\n")
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate content"):
        run(root, out)
    assert not (out / "COMPLETE").exists()
    assert not list((out / "synthetic").glob("*.npy"))


def test_output_protects_source(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "generated")


def test_documented_cli_generation_and_portable_relocation(tmp_path):
    """Run documented CLIs on independent fixtures, never regenerate Highway2."""
    import csv
    import hashlib
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from radiomana.synthetic_data import SyntheticHighwayDataModule

    repo = Path(__file__).resolve().parents[1]
    source, out = tmp_path / "source", tmp_path / "generated"
    source.mkdir()
    rng = np.random.default_rng(20261112)
    for split in ("train", "test"):
        rows = []
        for c in range(9):
            n = (20 if c == 0 else 10) if split == "train" else 1
            for i in range(n):
                name = f"{split}_{c}_{i}.npy"
                np.save(source / name, rng.normal(-40, 5, (512, 243)).astype("float32"))
                rows.append(f"{name} {c}\n")
        (source / f"{split}.txt").write_text("".join(rows))
    env = {**os.environ, "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    subprocess.run(
        [sys.executable, "-m", "radiomana.highway2_synthetic", "--root", str(source), "--out", str(out), "--ratio", "1"], cwd=repo, env=env, check=True
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["synthetic_count"] == 72
    frozen = (out / "val.txt").read_text()
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "synthetic").glob("*.npy")}
    # Move the originals: their old absolute paths are now genuinely unavailable.
    moved = tmp_path / "teammate_data"
    source.rename(moved)
    relocated = tmp_path / "relocated"
    subprocess.run(
        [sys.executable, "examples/task1_relocate.py", "--input", str(out), "--data-root", str(moved), "--out", str(relocated)], cwd=repo, env=env, check=True
    )
    assert (out / "val.txt").read_text() == frozen
    assert (relocated / "val.txt").read_text() == frozen.replace(str(source), str(moved))
    assert (relocated / "synthetic").is_symlink()
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (relocated / "synthetic").glob("*.npy")}
    dm = SyntheticHighwayDataModule(relocated, batch_size=3)
    dm.setup()
    assert len(dm.data_train) == 162 and len(dm.data_val) == 10 and len(dm.data_test) == 9
    x, y = next(iter(dm.train_dataloader()))
    assert tuple(x.shape) == (3, 512, 243) and set(y.tolist()) <= set(range(9))
    subprocess.run([sys.executable, "examples/validate_task1.py", "--root", str(relocated)], cwd=repo, env=env, check=True)
    provenance = list(csv.DictReader((relocated / "provenance.csv").open()))
    assert len(provenance) == 72


def test_relocation_rejects_different_source(tmp_path):
    import json

    from examples.task1_relocate import relocate

    source, data = tmp_path / "result", tmp_path / "data"
    source.mkdir()
    data.mkdir()
    (source / "COMPLETE").touch()
    (source / "synthetic").mkdir()
    (source / "summary.json").write_text(json.dumps({"root": "/old/source"}))
    (source / "audit.json").write_text(json.dumps({"original_train_manifest_sha256": "wrong"}))
    (data / "train.txt").write_text("different.npy 0\n")
    with pytest.raises(ValueError, match="manifest differs"):
        relocate(source, data, tmp_path / "out")
    assert not (tmp_path / "out").exists()
