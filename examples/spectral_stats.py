"""
Spectral statistics for the Highway2 dataset, aggregated per class.

For each PSD we collapse the time axis to a mean power spectrum, then compute
the first two power-weighted moments of that spectrum:

    centroid = sum(f * P) / sum(P)              "the frequency of the signal"
    spread   = sqrt(sum((f - centroid)^2 * P) / sum(P))   "its standard deviation"

Moments must be taken on LINEAR power, not dB, so the stored dB values are
converted with P = 10 ** (psd_db / 10) first. Averaging dB directly would
weight the noise floor far too heavily.
"""

import argparse
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

import radiomana
from radiomana import spectral_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    datamodule = radiomana.HighwayDataModule(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)
    datamodule.setup()
    data = {"train": datamodule.data_train, "val": datamodule.data_val, "test": datamodule.data_test}[args.split]

    # Highway2 PSDs are baseband, so bins span -fs/2 .. +fs/2
    meta = radiomana.Highway2Dataset()
    freqs_hz = radiomana.bin_frequencies(meta)
    class_labels = meta.class_labels

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    per_class = defaultdict(lambda: defaultdict(list))
    for psds, labels in loader:
        stats = spectral_stats(psds, freqs_hz)
        for key, values in stats.items():
            for label, value in zip(labels.tolist(), values.tolist()):
                per_class[label][key].append(value)

    header = f"{'label':<28}{'n':>6}{'centroid MHz':>16}{'spread MHz':>16}{'peak MHz':>11}{'dB std':>9}"
    print(f"\n{args.split} split: spectral statistics\n")
    print(header)
    print("-" * len(header))

    for label in sorted(per_class):
        stat = per_class[label]
        name = f"{class_labels[label]} ({label})"
        centroid = np.array(stat["centroid_hz"]) / 1e6
        spread = np.array(stat["spread_hz"]) / 1e6
        peak = np.array(stat["peak_hz"]) / 1e6
        db_std = np.array(stat["db_std"])
        print(
            f"{name:<28}{len(centroid):>6}"
            f"{centroid.mean():>10.3f} ±{centroid.std():<5.3f}"
            f"{spread.mean():>10.3f} ±{spread.std():<5.3f}"
            f"{peak.mean():>11.3f}{db_std.mean():>9.2f}"
        )

    everything = {key: np.array([v for label in per_class for v in per_class[label][key]]) for key in ["centroid_hz", "spread_hz", "db_std"]}
    print("-" * len(header))
    print(
        f"{'ALL':<28}{len(everything['centroid_hz']):>6}"
        f"{everything['centroid_hz'].mean() / 1e6:>10.3f} ±{everything['centroid_hz'].std() / 1e6:<5.3f}"
        f"{everything['spread_hz'].mean() / 1e6:>10.3f} ±{everything['spread_hz'].std() / 1e6:<5.3f}"
        f"{'':>11}{everything['db_std'].mean():>9.2f}"
    )
    print("\ncentroid/spread columns are mean ± std across samples of that class\n")


if __name__ == "__main__":
    main()
