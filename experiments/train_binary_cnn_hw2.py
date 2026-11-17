"""Highway2 binary CNN: baseline vs class-targeted RF PSD augmentation.

Run from repository root after installing package and setting DSET_FIOT_HIGHWAY2.
Original labels 0-3 -> clean (0), 4-8 -> jammed (1).
The official test split is never augmented, sampled, or used for model selection.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from radiomana.datasets import Highway2Dataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shift_frequency_no_wrap(x, shift, fill):
    """Move RF frequency bins along axis 0, filling rather than wrapping edges."""
    if shift == 0:
        return x
    out = torch.full_like(x, fill)
    if abs(shift) >= x.shape[0]:
        return out
    if shift > 0:
        out[shift:, :] = x[:-shift, :]
    else:
        out[:shift, :] = x[-shift:, :]
    return out


def augment_db_psd(x):
    """Conservative operations in *dB* space; verify RF plausibility."""
    if torch.rand(()) < 0.5:
        x = x + torch.randn_like(x) * 0.5  # small dB jitter
    if torch.rand(()) < 0.5:
        x = x + float(torch.empty(()).uniform_(-1, 1))  # dB offset, not x*scale
    if torch.rand(()) < 0.3:
        shift = int(torch.randint(-4, 5, ()).item())
        x = shift_frequency_no_wrap(x, shift, fill=float(x.median()))
    return x


class BinarySubset(Dataset):
    def __init__(self, base, indices, augmented=False, augment_classes=(4, 8)):
        self.base = base
        self.indices = list(map(int, indices))
        self.augmented = augmented
        self.augment_classes = set(augment_classes)
        self.original_labels = [int(base.items[i][1]) for i in self.indices]
        self.binary_labels = [int(y >= 4) for y in self.original_labels]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, original = self.base[self.indices[idx]]
        x = x.clone().float()
        if self.augmented and original in self.augment_classes:
            x = augment_db_psd(x)
        return x, torch.tensor(int(original >= 4), dtype=torch.long)


class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(p=0.2), nn.Linear(64, 2),
        )

    def forward(self, x):
        # x is (batch, 512, 243), PSD values in dB.
        # Match repository's median-relative preprocessing rationale.
        x = x.float()
        x = x - x.flatten(1).median(dim=1).values[:, None, None]
        x = ((x.clamp(-10, 50) + 10) / 60).unsqueeze(1)
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            true.extend(y.tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
    cm = confusion_matrix(true, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "jammed_precision": float(precision_score(true, pred, zero_division=0)),
        "jammed_recall": float(recall_score(true, pred, zero_division=0)),
        "jammed_f1": float(f1_score(true, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }, true, pred


def train_one(name, train_set, val_loader, test_loader, test_set, args, device, out):
    seed_all(args.seed)  # same initialization per arm
    model = BinaryCNN().to(device)
    train_sampler = None
    if args.balance:
        counts = Counter(train_set.binary_labels)
        if len(counts) != 2:
            raise ValueError("Training must include both binary classes")
        # Cap minority oversampling for stability; equal sampling may overfit tiny classes.
        minority = min(counts, key=counts.get)
        ratio = min(counts[1 - minority] / counts[minority], args.max_weight)

        # Start with the existing binary balancing
        weights = [
            ratio if y == minority else 1.0
            for y in train_set.binary_labels
        ]

        # Increase sampling probability for classes 7 and 8
        for i, original_label in enumerate(train_set.original_labels):
            if original_label in [7, 8]:
                weights[i] *= 4.0

        gen = torch.Generator().manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=gen)
    loader = DataLoader(train_set, batch_size=args.batch_size,
                        shuffle=train_sampler is None, sampler=train_sampler,
                        num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = -1
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val, _, _ = evaluate(model, val_loader, device)
        line = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items() if k != "confusion_matrix"}}
        history.append(line)
        print(f"{name}: epoch {epoch}/{args.epochs}, train loss {line['train_loss']:.4f}, val Jammed F1 {val['jammed_f1']:.4f}", flush=True)
        if val["jammed_f1"] > best:
            best = val["jammed_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, out / f"{name}_best.pt")
    test, truth, predicted = evaluate(model, test_loader, device)
    details = {}
    for original_class in sorted(set(test_set.original_labels)):
        matching = [i for i, y in enumerate(test_set.original_labels) if y == original_class]
        correct = sum(predicted[i] == int(original_class >= 4) for i in matching)
        details[str(original_class)] = {"n": len(matching), "binary_detection_rate": correct / len(matching)}
    result = {"experiment": name, "best_epoch": best_epoch, "best_val_jammed_f1": best,
              "test": test, "per_original_class": details, "history": history}
    (out / f"{name}_results.json").write_text(json.dumps(result, indent=2))
    with (out / f"{name}_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--augment-classes", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--balance", action="store_true", help="use binary weighted sampling in both runs")
    ap.add_argument("--max-weight", type=float, default=5.0)
    ap.add_argument("--output", default="runs/binary_cnn_hw2")
    args = ap.parse_args()
    if not set(args.augment_classes).issubset({4, 5, 6, 7, 8}):
        ap.error("--augment-classes must be jammed Highway2 labels 4 through 8")
    seed_all(args.seed)
    root = os.environ.get("DSET_FIOT_HIGHWAY2")
    if not root or not (Path(root) / "train.txt").exists() or not (Path(root) / "test.txt").exists():
        ap.error("Set DSET_FIOT_HIGHWAY2 to your Highway2 folder containing train.txt and test.txt")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_base = Highway2Dataset(subset="train")
    test_base = Highway2Dataset(subset="test")
    indices = np.arange(len(train_base))
    labels = [int(y) for _, y in train_base.items]
    train_idx, val_idx = train_test_split(indices, test_size=args.val_fraction,
                                           stratify=labels, random_state=args.seed)
    val_set = BinarySubset(train_base, val_idx)
    test_set = BinarySubset(test_base, range(len(test_base)))
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print("Device:", device, "| Train raw labels:", dict(Counter(labels)), flush=True)
    print("Augment originals:", args.augment_classes, "| training subset:", len(train_idx),
          "| val:", len(val_idx), "| official test:", len(test_set), flush=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)
    results = []
    for name, use_aug in [("baseline", False), ("targeted_augmentation", True)]:
        train_set = BinarySubset(train_base, train_idx, augmented=use_aug, augment_classes=args.augment_classes)
        results.append(train_one(name, train_set, val_loader, test_loader, test_set, args, device, out))
    with (out / "comparison.csv").open("w", newline="") as f:
        keys = ["experiment", "best_epoch", "accuracy", "balanced_accuracy", "jammed_precision", "jammed_recall", "jammed_f1", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            cm = r["test"]["confusion_matrix"]
            writer.writerow({"experiment": r["experiment"], "best_epoch": r["best_epoch"],
                             **{k: r["test"][k] for k in keys[2:7]},
                             "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1]})
    print("\nOfficial Highway2 test comparison:")
    for r in results:
        print(r["experiment"], r["test"])
    print("\nSaved metrics + checkpoints under", out)


if __name__ == "__main__":
    main()
