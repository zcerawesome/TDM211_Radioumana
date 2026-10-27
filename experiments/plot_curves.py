"""
Plot train vs Highway1 test curves per epoch from cross_site_vit.py result files.

    python experiments/plot_curves.py runs/E_seed1.json runs/G_hw1train_seed1.json -o curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("-o", "--out", type=Path, default=Path("train_test_curves.png"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd0"]
    for color, path in zip(colors, args.results):
        result = json.loads(path.read_text())
        history = [record for record in result["history"] if "curve_train" in record]
        if not history:
            # runs from before per-epoch curves were logged: show the final Highway1 test F1 as a single point
            axes[1].plot([len(result["history"])], [result["hw1_test"]["f1"]], marker="*", ms=14, ls="", color=color, label=f"{path.stem} Highway1 test (final)")
            continue
        epochs = [record["epoch"] + 1 for record in history]
        for ax, metric in zip(axes, ["bce", "f1"]):
            ax.plot(epochs, [r["curve_train"][metric] for r in history], marker="o", color=color, label=f"{path.stem} train")
            ax.plot(epochs, [r["curve_hw1_test"][metric] for r in history], marker="s", ls="--", color=color, label=f"{path.stem} Highway1 test")

    axes[0].set_ylabel("binary cross-entropy on P(jammed)")
    axes[0].set_title("Loss: train vs Highway1 test")
    axes[1].set_ylabel("jammed F1 (threshold 0.5)")
    axes[1].set_title("F1: train vs Highway1 test")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
