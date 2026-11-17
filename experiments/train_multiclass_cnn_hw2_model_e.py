"""Highway2 Model E: spatially aware nine-class CNN. No augmentation or weighting.

Train/validation are stratified portions of the official training split.
The official test split is intentionally not loaded: use validation to decide
which classes require synthetic data, reserving test for final assessment.
Run from repository root: python experiments/train_multiclass_cnn_hw2_model_e.py
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


class SpatialCNN(nn.Module):
    """Same feature extractor as Model A, but retain a 4x4 spatial grid."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Dropout(p=0.2), nn.Linear(64 * 4 * 4, 9),
        )

    def forward(self, x):
        # Match Model A's PSD preprocessing exactly.
        x = x.float()
        x = x - x.flatten(1).median(dim=1).values[:, None, None]
        x = ((x.clamp(-10, 50) + 10) / 60).unsqueeze(1)
        return self.net(x)


class OriginalLabelSubset(Dataset):
    """Keep nine original Highway2 labels rather than converting to binary."""
    def __init__(self, base, indices):
        self.base = base
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, label = self.base[self.indices[index]]
        return x.float(), int(label)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_predictions(model, loader, device):
    model.eval()
    true_labels, predictions = [], []
    with torch.inference_mode():
        for x, y in loader:
            output = model(x.to(device))
            true_labels.extend(y.cpu().tolist())
            predictions.extend(output.argmax(dim=1).cpu().tolist())
    return true_labels, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', default='runs/multiclass_cnn_hw2_model_e')
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The repository's Highway2Dataset reads DSET_FIOT_HIGHWAY2.
    base = Highway2Dataset(subset='train')
    original_labels = np.array([int(item[1]) for item in base.items])
    all_indices = np.arange(len(base))
    train_indices, val_indices = train_test_split(
        all_indices, test_size=0.10, stratify=original_labels,
        random_state=args.seed,
    )
    train_labels = original_labels[train_indices]
    val_labels = original_labels[val_indices]
    train_loader = DataLoader(OriginalLabelSubset(base, train_indices),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(OriginalLabelSubset(base, val_indices),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:', device, flush=True)
    print('Train counts:', dict(sorted(Counter(train_labels.tolist()).items())), flush=True)
    print('Validation counts:', dict(sorted(Counter(val_labels.tolist()).items())), flush=True)
    print('Model E: 4x4 spatial pooling; no augmentation, oversampling, or class weights. Official test untouched.', flush=True)

    # Model E: only architectural change vs. Model A is 4x4 spatial pooling.
    model = SpatialCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_epoch = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            total += len(y)

        y_true, y_pred = get_predictions(model, val_loader, device)
        macro_f1 = float(f1_score(y_true, y_pred, labels=list(range(9)),
                                  average='macro', zero_division=0))
        avg_loss = running_loss / total
        history.append({'epoch': epoch, 'train_loss': avg_loss,
                        'val_macro_f1': macro_f1})
        print(f'Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}, validation macro F1={macro_f1:.4f}', flush=True)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # Load best validation checkpoint, not final epoch.
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / 'multiclass_best.pt')
    y_true, y_pred = get_predictions(model, val_loader, device)
    report = classification_report(y_true, y_pred, labels=list(range(9)),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))

    with (out_dir / 'per_class_validation.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'train_count', 'validation_count',
                         'precision', 'recall', 'f1', 'support'])
        for cls in range(9):
            m = report[str(cls)]
            writer.writerow([cls, int(sum(train_labels == cls)),
                             int(sum(val_labels == cls)), m['precision'],
                             m['recall'], m['f1-score'], m['support']])

    with (out_dir / 'confusion_matrix.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actual\\predicted'] + list(range(9)))
        for cls, row in enumerate(cm):
            writer.writerow([cls] + row.tolist())

    with (out_dir / 'training_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_macro_f1'])
        writer.writeheader()
        writer.writerows(history)

    summary = {'model': 'E_spatial_4x4', 'seed': args.seed, 'best_epoch': best_epoch,
               'best_validation_macro_f1': best_f1,
               'per_class_validation': {str(i): report[str(i)] for i in range(9)},
               'confusion_matrix': cm.tolist()}
    (out_dir / 'results.json').write_text(json.dumps(summary, indent=2))

    print('\nBEST EPOCH:', best_epoch, '| VALIDATION MACRO F1:', round(best_f1, 4))
    print('Class | Train count | Val count | Precision | Recall | F1')
    for cls in range(9):
        r = report[str(cls)]
        print(f'{cls:>5} | {sum(train_labels == cls):>11} | {sum(val_labels == cls):>9} | '
              f'{r["precision"]:>9.3f} | {r["recall"]:>6.3f} | {r["f1-score"]:>5.3f}')
    print('\nConfusion matrix (rows=actual, columns=predicted):\n', cm)
    print('\nSaved results to:', out_dir)
    print('Choose difficult classes from validation metrics, NOT official test data.')


if __name__ == '__main__':
    main()
