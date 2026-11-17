"""Model F: Model E SpatialCNN + saved synthetic PSD samples + weighted loss.

Run from the repository root after generating the 598 synthetic samples:
  python experiments/train_multiclass_cnn_hw2_model_f.py train --epochs 10 --batch-size 16

Reuses the generated manifest from Model B/C, recreates the original
stratified validation split, and does not access the official test set.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from radiomana.datasets import Highway2Dataset
from train_multiclass_cnn_hw2_model_e import SpatialCNN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(base, seed):
    labels = np.array([int(item[1]) for item in base.items])
    return train_test_split(
        np.arange(len(base)), test_size=0.1, stratify=labels,
        random_state=seed)


def make_synthetic_psd(x, rng):
    """Small changes in PSD dB space; check physical validity with RF team.

    Avoid uniform dB offsets: the model subtracts the sample median, which
    would mostly cancel them. Do not flip/rotate or wrap frequency bins.
    """
    x = x.clone().float()
    if rng.random() < 0.9:
        # Per-bin noise with scale 0.15--0.50 dB.
        sigma = float(rng.uniform(0.15, 0.50))
        x += torch.from_numpy(rng.normal(0, sigma, size=x.shape).astype('float32'))
    if rng.random() < 0.5:
        # Shift frequency axis by at most two bins; pad with original median.
        shift = int(rng.choice([-2, -1, 1, 2]))
        shifted = torch.full_like(x, float(x.median()))
        if shift > 0:
            shifted[shift:, :] = x[:-shift, :]
        else:
            shifted[:shift, :] = x[-shift:, :]
        x = shifted
    return x


class OriginalSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x.float(), int(y)


class SyntheticTrainingSet(Dataset):
    def __init__(self, base, train_indices, records, folder):
        self.base_set = OriginalSubset(base, train_indices)
        self.records = records
        self.folder = folder

    def __len__(self):
        return len(self.base_set) + len(self.records)

    def __getitem__(self, i):
        if i < len(self.base_set):
            return self.base_set[i]
        row = self.records[i - len(self.base_set)]
        return torch.from_numpy(np.load(self.folder / row['file']).copy()).float(), int(row['class'])


def generate(args):
    seed_everything(args.seed)
    base = Highway2Dataset(subset='train')
    train_idx, val_idx = split_indices(base, args.seed)
    train_labels = [int(base.items[int(i)][1]) for i in train_idx]
    counts = Counter(train_labels)
    rng = np.random.default_rng(args.seed + 100)
    output = Path(args.output)
    data_folder = output / 'synthetic_npy'
    # Avoid silently mixing existing synthetic files from different settings.
    if (output / 'manifest.csv').exists():
        raise SystemExit(f'{output / "manifest.csv"} already exists. Use a new --output directory.')
    data_folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in args.classes:
        original_ids = [int(i) for i in train_idx if int(base.items[int(i)][1]) == cls]
        if not original_ids:
            raise ValueError(f'No training examples for class {cls}')
        extra = max(0, args.target_count - len(original_ids))
        for j in range(extra):
            source_idx = int(rng.choice(original_ids))
            x, y = base[source_idx]
            synthetic = make_synthetic_psd(x, rng)
            filename = f'class_{cls}_{j:04d}.npy'
            np.save(data_folder / filename, synthetic.numpy().astype(np.float32))
            rows.append({'file': filename, 'class': cls, 'source_train_index': source_idx})
        print(f'Class {cls}: {len(original_ids)} original train + {extra} synthetic = {len(original_ids)+extra}', flush=True)
    with (output / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file','class','source_train_index'])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'seed': args.seed, 'target_count': args.target_count,
        'classes': args.classes, 'train_count': len(train_idx),
        'validation_count': len(val_idx), 'synthetic_count': len(rows),
        'original_train_class_counts': dict(sorted(counts.items())),
        'warning': 'Validate RF plausibility; synthetic examples come only from train split.'
    }
    (output / 'generation_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Generated {len(rows)} synthetic files in {data_folder}')
    print('No validation or official test examples were augmented.')


def predict(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for x, y in loader:
            scores = model(x.to(device))
            actual.extend(y.tolist())
            predicted.extend(scores.argmax(1).cpu().tolist())
    return actual, predicted


def train(args):
    seed_everything(args.seed)
    folder = Path(args.output)
    meta_file = folder / 'generation_metadata.json'
    if not meta_file.is_file():
        raise SystemExit('Generate synthetic data first with the same --output path.')
    meta = json.loads(meta_file.read_text())
    if int(meta['seed']) != args.seed:
        raise SystemExit('Seed differs from synthetic generation seed; would change train/validation split.')
    if int(meta.get('synthetic_count', -1)) != 598 or set(meta.get('classes', [])) != {3, 7, 8}:
        raise SystemExit('Expected the existing 598 synthetic samples for classes 3, 7, 8. Check --output.')
    with (folder / 'manifest.csv').open(newline='') as f:
        records = list(csv.DictReader(f))
    base = Highway2Dataset(subset='train')
    train_idx, val_idx = split_indices(base, args.seed)
    allowed = set(map(int, train_idx))
    if any(int(row['source_train_index']) not in allowed for row in records):
        raise SystemExit('Synthetic manifest has sources outside the training split.')
    train_dataset = SyntheticTrainingSet(base, train_idx, records, folder / 'synthetic_npy')
    val_dataset = OriginalSubset(base, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    model = SpatialCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Use ORIGINAL training frequencies to avoid treating augmented copies as independent.
    original_counts = Counter(int(base.items[int(i)][1]) for i in train_idx)
    largest = max(original_counts.values())
    class_weights = torch.tensor([
        min(args.max_class_weight, (largest / original_counts[c]) ** 0.5)
        for c in range(9)
    ], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print('Model F: spatial 4x4 CNN + synthetic training data + weighted loss', flush=True)
    print('Class weights:', class_weights.detach().cpu().tolist(), flush=True)
    results_folder = Path(args.results_output) if args.results_output else folder
    results_folder.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_epoch = None
    best_state = None
    history = []
    print(f'Device: {device}; {len(train_idx)} originals + {len(records)} synthetic; validation: {len(val_idx)} originals', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        cumulative, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            cumulative += loss.item() * len(y)
            seen += len(y)
        truth, preds = predict(model, val_loader, device)
        score = float(f1_score(truth, preds, labels=list(range(9)), average='macro', zero_division=0))
        history.append({'epoch': epoch, 'train_loss': cumulative / seen, 'val_macro_f1': score})
        print(f'Epoch {epoch}/{args.epochs}: loss={cumulative/seen:.4f}, validation macro F1={score:.4f}', flush=True)
        if score > best:
            best = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, results_folder / 'model_f_best.pt')
    truth, preds = predict(model, val_loader, device)
    report = classification_report(truth, preds, labels=list(range(9)), output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, preds, labels=list(range(9)))
    with (results_folder / 'per_class_validation.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for cls in range(9):
            r = report[str(cls)]
            writer.writerow([cls, r['precision'], r['recall'], r['f1-score'], r['support']])
    with (results_folder / 'confusion_matrix.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual\\predicted'] + list(range(9)))
        for i, row in enumerate(cm):
            writer.writerow([i] + row.tolist())
    with (results_folder / 'training_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_macro_f1'])
        writer.writeheader()
        writer.writerows(history)
    (results_folder / 'results.json').write_text(json.dumps({
        'seed': args.seed, 'model': 'F_spatial_synthetic_weighted', 'weighted_loss': True, 'class_weights': class_weights.detach().cpu().tolist(), 'best_epoch': best_epoch, 'best_validation_macro_f1': best,
        'per_class_validation': {str(i): report[str(i)] for i in range(9)},
        'confusion_matrix': cm.tolist()
    }, indent=2))
    print(f'\nBest epoch: {best_epoch}; validation macro F1: {best:.4f}')
    print('Class | Precision | Recall | F1')
    for cls in range(9):
        r = report[str(cls)]
        print(f'{cls:>5} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["f1-score"]:.3f}')
    baseline_file = Path('runs/multiclass_cnn_hw2_baseline/results.json')
    if baseline_file.exists():
        previous = json.loads(baseline_file.read_text())
        print(f'\nBaseline validation macro F1: {previous["best_validation_macro_f1"]:.4f}')
        print(f'Synthetic validation macro F1: {best:.4f}')
        print('Per-class baseline -> synthetic F1:')
        for cls in range(9):
            old = previous['per_class_validation'][str(cls)]['f1-score']
            new = report[str(cls)]['f1-score']
            print(f'  {cls}: {old:.3f} -> {new:.3f}')
    prior_e = Path('runs/multiclass_cnn_hw2_model_e/results.json')
    if prior_e.is_file():
        old_e = json.loads(prior_e.read_text())
        print('Model E validation macro F1:', round(old_e['best_validation_macro_f1'], 4))
        print('Model F validation macro F1:', round(best, 4))
    print(f'\nSaved Model F + results under {results_folder}; official test split untouched.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'train'])
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 7, 8])
    parser.add_argument('--target-count', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='runs/multiclass_cnn_hw2_synthetic_378',
                        help='Location of previously generated synthetic samples and manifest')
    parser.add_argument('--results-output', default='runs/multiclass_cnn_hw2_model_f',
                        help='Separate folder for checkpoints and metrics; preserves prior results')
    parser.add_argument('--max-class-weight', type=float, default=5.0)

    args = parser.parse_args()
    if not set(args.classes).issubset(set(range(9))):
        parser.error('--classes must be original Highway2 labels 0..8')
    if args.action == 'generate':
        raise SystemExit('Model F reuses Model B/C generated data. Run action=train.')
    train(args)


if __name__ == '__main__':
    main()
