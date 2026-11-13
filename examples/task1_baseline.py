"""Small CPU baseline on frozen Task 1 splits; no CNN or test-set tuning.

Features: 32 frequency-block means/stds, 27 time-block means/stds and
seven global dB quantiles. Standardization fitted independently on each
training arm only. Fixed logistic regression hyperparameters, same test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radiomana.synthetic_data import ManifestHighway2Dataset


def features(x):
    f = x.reshape(32, 16, 243)
    t = x.reshape(512, 27, 9)
    return np.concatenate([f.mean((1, 2)), f.std((1, 2)), t.mean((0, 2)), t.std((0, 2)), np.percentile(x, [1, 5, 25, 50, 75, 95, 99])])


def main(root):
    root = Path(root)
    if not (root / "COMPLETE").exists():
        raise ValueError("Incomplete generation")
    cache = {}
    data = {}
    for split, manifest in [("original", "train_original.txt"), ("augmented", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
        ds = ManifestHighway2Dataset(root, manifest)
        xs = []
        ys = []
        for p, c in ds.items:
            path = str((root / p).resolve())
            if path not in cache:
                cache[path] = features(np.load(path, allow_pickle=False))
            xs.append(cache[path])
            ys.append(c)
        data[split] = (np.stack(xs), np.array(ys))
        print("Features ready:", split, len(ys), flush=True)
    results = {}
    for arm in ["original", "augmented"]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=20261112))
        model.fit(*data[arm])
        results[arm] = {}
        for split in ["val", "test"]:
            x, y = data[split]
            pred = model.predict(x)
            results[arm][split] = {
                "accuracy": accuracy_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro"),
                "confusion_matrix": confusion_matrix(y, pred, labels=range(9)).tolist(),
                "per_class": classification_report(y, pred, labels=range(9), output_dict=True, zero_division=0),
            }
        results[arm]["iterations"] = model[-1].n_iter_.tolist()
        print(arm, results[arm]["test"]["macro_f1"], flush=True)
    (root / "baseline_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
