"""Pre-resize both FIOT datasets to 224x224 float16 so ViT experiments don't re-read 15 GB of .npy files each epoch."""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_one(path):
    x = torch.from_numpy(np.load(path)).float()[None, None]  # (1, 1, 512, 243) dB
    # area interpolation averages bins instead of skipping them, so 1-bin tones survive the downsample
    x = F.interpolate(x, size=(224, 224), mode="area")
    return x[0, 0].numpy().astype(np.float16)


def build(root: Path, out: Path):
    for subset in ["train", "test"]:
        items = [line.split() for line in open(root / f"{subset}.txt") if line.strip()]
        paths = [root / p for p, _ in items]
        labels = np.array([int(lab) for _, lab in items], dtype=np.int64)
        arr = np.lib.format.open_memmap(out / f"{root.name}_{subset}_x.npy", mode="w+", dtype=np.float16, shape=(len(paths), 224, 224))
        with Pool(12) as pool:
            for idx, x in enumerate(pool.imap(load_one, paths, chunksize=32)):
                arr[idx] = x
        arr.flush()
        np.save(out / f"{root.name}_{subset}_y.npy", labels)
        print(root.name, subset, arr.shape, flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for root in sys.argv[2:]:
        build(Path(root), out)
