"""Rebind an existing Task 1 result folder to a teammate's local data paths.

Only metadata is copied. The existing synthetic directory is symlinked;
original/synthetic arrays and frozen split membership are never modified.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

MANIFESTS = ("train.txt", "val.txt", "test.txt", "train_original.txt", "val_original.txt", "test_original.txt", "synthetic.txt")


def relocate(source, data_root, output):
    source, data_root, output = (Path(p).resolve() for p in (source, data_root, output))
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(data_root):
        raise ValueError("Output must be a new directory outside both inputs")
    if not (source / "COMPLETE").is_file() or not (source / "synthetic").is_dir():
        raise ValueError("Input must be a complete generated result folder")
    summary = json.loads((source / "summary.json").read_text())
    old_root = Path(summary["root"])
    audit = json.loads((source / "audit.json").read_text())
    for split in ("train", "test"):
        checksum = hashlib.sha256((data_root / f"{split}.txt").read_bytes()).hexdigest()
        if checksum != audit[f"original_{split}_manifest_sha256"]:
            raise ValueError(f"{split} manifest differs from the generation source")
    rebound = {}
    for name in MANIFESTS:
        rows = []
        for line in (source / name).read_text().splitlines():
            path, label = line.split()
            if int(label) not in range(9):
                raise ValueError("Invalid label")
            p = Path(path)
            if p.is_absolute():
                relative = p.relative_to(old_root)  # reject unrelated absolute paths
                target = (data_root / relative).resolve()
                if not target.is_relative_to(data_root):
                    raise ValueError("Original path escapes data root")
                replacement = str(target)
            else:
                target = (source / p).resolve()
                if not target.is_relative_to((source / "synthetic").resolve()):
                    raise ValueError("Unexpected relative sample path")
                replacement = path
            if not target.is_file():
                raise FileNotFoundError(target)
            if any(c.isspace() for c in replacement):
                raise ValueError("Upstream manifest format cannot represent whitespace in paths")
            rows.append(f"{replacement} {label}\n")
        rebound[name] = "".join(rows)
    output.mkdir(parents=True, exist_ok=False)
    # Copy metadata only; never copy or modify any sample arrays.
    for p in source.iterdir():
        if p.is_file() and p.suffix in {".json", ".csv", ".png", ".md", ".txt"} and p.name not in MANIFESTS:
            shutil.copyfile(p, output / p.name)
    (output / "synthetic").symlink_to(source / "synthetic", target_is_directory=True)
    for name, text in rebound.items():
        (output / name).write_text(text)
    summary["root"] = str(data_root)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "COMPLETE").write_text("Metadata relocated; original generation complete. Run validation on this local view.\n")
    print(f"Relocated metadata to {output}; synthetic arrays remain at {source / 'synthetic'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    relocate(args.input, args.data_root, args.out)
