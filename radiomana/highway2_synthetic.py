"""Reproducible, training-only augmentation of Highway2 log-power arrays.

Run: python -m radiomana.highway2_synthetic --root DATA --out NEW_DIRECTORY
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

LABELS = [
    "Background a",
    "Background b",
    "Background c",
    "Background d",
    "Chirp high distance",
    "Chirp medium distance",
    "Chirp small distance",
    "Cigarette lighter 1",
    "Cigarette lighter 2",
]
SPLIT_SEED = 0xD00D1E


def read_manifest(root, name):
    items = []
    for line in (root / name).read_text().splitlines():
        if not line.strip():
            continue
        path, label = line.split()
        label = int(label)
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()) or label not in range(9):
            raise ValueError(f"Unsafe path or invalid label: {line}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        items.append((path, label))
    if len({p for p, _ in items}) != len(items):
        raise ValueError("Duplicate paths in manifest")
    return items


def validate_array(x):
    if x.shape != (512, 243) or x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"Unexpected shape/dtype: {x.shape}, {x.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite array")


def digest(x):
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


def augment(x, seed):
    """Small receiver gain and positive averaged noise power; no axis changes.

    Gamma(shape=16) models an averaged noise periodogram, not waveform AWGN.
    The averaging factor is a conservative augmentation assumption, not known
    acquisition metadata. Noise is 14–20 dB below the sample's 1st percentile.
    """
    validate_array(x)
    rng = np.random.default_rng(seed)
    gain = float(rng.uniform(-0.25, 0.25))
    noise_db = float(np.percentile(x, 1) - rng.uniform(14, 20))
    # logaddexp avoids underflow/overflow and never creates negative power.
    k = np.log(10) / 10
    noise_log = noise_db * k + np.log(rng.gamma(16, 1 / 16, x.shape))
    y = (np.logaddexp((x.astype(np.float64) + gain) * k, noise_log) / k).astype(x.dtype)
    validate_array(y)
    delta = y.astype(float) - x
    corr = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    rmse = float(np.sqrt(np.mean(delta**2)))
    power_delta = float((np.logaddexp.reduce(y.ravel().astype(float) * k) - np.logaddexp.reduce(x.ravel().astype(float) * k)) / k)
    if not np.isfinite(corr) or corr < 0.995 or rmse > 1 or abs(power_delta) > 0.3:
        raise ValueError("Augmentation failed conservative signal quality bounds")
    if np.array_equal(x, y):
        raise ValueError("Unchanged synthetic sample")
    return y, {"gain_db": gain, "noise_power_db": noise_db, "correlation": corr, "rmse_db": rmse, "total_power_change_db": power_delta}


def plan(counts, ratio=0.1, max_variants=2):
    if not 0 < ratio <= 1 or max_variants < 1:
        raise ValueError("Invalid balancing parameters")
    target = int(np.ceil(max(counts.values()) * ratio))
    return target, {c: min(max(0, target - counts.get(c, 0)), max_variants * counts.get(c, 0)) for c in range(9)}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path, items):
    path.write_text("".join(f"{p} {c}\n" for p, c in items))


def distribution(out, name, counts):
    write_csv(out / f"{name}.csv", [{"label": c, "name": LABELS[c], "count": counts.get(c, 0)} for c in range(9)])
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(9), [counts.get(c, 0) for c in range(9)])
    ax.bar_label(bars, fontsize=9)
    ax.set(xticks=range(9), xlabel="Class label", ylabel="Samples", title=name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


def compare(out, c, x, y):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    lo, hi = np.percentile(x, [1, 99])
    for ax, a, title in zip(axes[:2], [x, y], ["Real training source", "Synthetic variant"]):
        im = ax.imshow(a, origin="lower", aspect="auto", vmin=lo, vmax=hi)
        ax.set(title=title, xlabel="Time bin", ylabel="Frequency bin")
    fig.colorbar(im, ax=list(axes[:2]), label="Power (dB)")
    axes[2].hist((y - x).ravel(), bins=60, log=True)
    axes[2].set(xlabel="Change (dB)", ylabel="Bin count", title="Perturbation distribution")
    fig.suptitle(f"{c}: {LABELS[c]}")
    fig.savefig(out / f"comparison_class_{c}.png", dpi=150)
    plt.close(fig)


def run(root, out, seed=20261112, ratio=0.1, max_variants=2):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Output must be separate from original data")
    out.mkdir(parents=True, exist_ok=False)
    (out / "synthetic").mkdir()
    train = read_manifest(root, "train.txt")
    test = read_manifest(root, "test.txt")
    ti, vi = train_test_split(np.arange(len(train)), test_size=0.1, stratify=[c for _, c in train], random_state=SPLIT_SEED)
    fit, val = [train[i] for i in ti], [train[i] for i in vi]
    splits = {"train": fit, "val": val, "test": test}
    # Full corpus integrity pass, bounded to one sample at a time.
    hashes = defaultdict(list)
    audit_rows = []
    all_hashes = set()
    for split, items in splits.items():
        for p, c in items:
            x = np.load(root / p, allow_pickle=False)
            validate_array(x)
            h = digest(x)
            all_hashes.add(h)
            hashes[h].append((split, p, c))
            stat = (root / p).stat()
            audit_rows.append(
                {
                    "split": split,
                    "path": p,
                    "label": c,
                    "sha256": h,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "minimum_db": float(x.min()),
                    "maximum_db": float(x.max()),
                }
            )
        print(f"Audited {split}: {len(items)}", flush=True)
    overlaps = [v for v in hashes.values() if len({r[0] for r in v}) > 1]
    audit = {
        "exact_cross_split_duplicate_groups": overlaps,
        "within_split_duplicate_groups": [v for v in hashes.values() if len(v) > 1 and len({r[0] for r in v}) == 1],
        "session_metadata_available": False,
        "session_separation": "Unverifiable: only numeric filenames and class labels supplied.",
        "original_train_manifest_sha256": hashlib.sha256((root / "train.txt").read_bytes()).hexdigest(),
        "original_test_manifest_sha256": hashlib.sha256((root / "test.txt").read_bytes()).hexdigest(),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    write_csv(out / "original_integrity.csv", audit_rows)
    if overlaps:
        raise ValueError("Cross-split duplicate content detected; see audit.json. Generation blocked.")
    before = Counter(c for _, c in fit)
    target, counts = plan(before, ratio, max_variants)
    distribution(out, "original_class_distribution", before)
    distribution(out, "official_train_distribution", Counter(c for _, c in train))
    # Original paths are absolute references, never copies or writes.
    for split, items in splits.items():
        write_manifest(out / f"{split}_original.txt", [(str(root / p), c) for p, c in items])
    write_manifest(out / "val.txt", [(str(root / p), c) for p, c in val])
    write_manifest(out / "test.txt", [(str(root / p), c) for p, c in test])
    generated = []
    provenance = []
    rng = np.random.default_rng(seed)
    for c in range(9):
        parents = [p for p, label in fit if label == c]
        order = list(rng.permutation(parents))
        for j in range(counts[c]):
            p = order[j % len(order)]
            item_seed = int(rng.integers(0, 2**63 - 1))
            x = np.load(root / p, allow_pickle=False)
            y, metrics = augment(x, item_seed)
            h = digest(y)
            if h in all_hashes:
                raise ValueError("Synthetic duplicate detected")
            all_hashes.add(h)
            dest = f"synthetic/class_{c}_{j:05d}.npy"
            np.save(out / dest, y, allow_pickle=False)
            reloaded = np.load(out / dest, allow_pickle=False)
            if not np.array_equal(y, reloaded):
                raise ValueError("Serialization mismatch")
            generated.append((dest, c))
            provenance.append(dict(path=dest, label=c, parent=p, seed=item_seed, sha256=h, **metrics))
            if j == 0:
                compare(out, c, x, y)
        print(f"Class {c}: {before[c]} + {counts[c]} synthetic", flush=True)
    if provenance:
        write_csv(out / "provenance.csv", provenance)
    write_manifest(out / "synthetic.txt", generated)
    write_manifest(out / "train.txt", [(str(root / p), c) for p, c in fit] + generated)
    after = before + Counter(c for _, c in generated)
    distribution(out, "augmented_class_distribution", after)
    fig, ax = plt.subplots(figsize=(11, 5))
    pos = np.arange(9)
    ax.bar(pos - 0.2, [before[c] for c in pos], 0.4, label="Original fitting split")
    ax.bar(pos + 0.2, [after[c] for c in pos], 0.4, label="Augmented fitting split")
    ax.set(xticks=pos, xlabel="Class label", ylabel="Samples", title="Training distribution before and after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "before_after.png", dpi=160)
    plt.close(fig)
    # Ensure no original file changed during the run, including held-out arrays.
    for row in audit_rows:
        path = root / row["path"]
        if digest(np.load(path, allow_pickle=False)) != row["sha256"]:
            raise ValueError("Original data changed during execution")
    for split in ("train", "test"):
        if hashlib.sha256((root / f"{split}.txt").read_bytes()).hexdigest() != audit[f"original_{split}_manifest_sha256"]:
            raise ValueError("Original manifest changed")
    summary = {
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "root": str(root),
        "threshold": target,
        "ratio": ratio,
        "max_variants": max_variants,
        "original": dict(before),
        "generated": counts,
        "augmented": dict(after),
        "validation": len(val),
        "test": len(test),
        "original_integrity_verified": True,
        "synthetic_count": len(generated),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "COMPLETE").write_text("Generation and integrity checks completed.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--max-variants", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.out, args.seed, args.ratio, args.max_variants), indent=2))
