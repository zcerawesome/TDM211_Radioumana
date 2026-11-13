"""Read-only corpus audit and real Lightning/SqueezeNet training smoke test.

Run from the repository with PYTHONPATH=. No data generation or model saving.
The targeted sampler makes all nine original classes and all six synthetic
classes participate in five small batches; the underlying dataset stays whole.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from radiomana.datasets import HighwayDataModule
from radiomana.highway2_synthetic import digest
from radiomana.models import HighwayBaselineModel
from radiomana.synthetic_data import SyntheticHighwayDataModule


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(items, root):
    return [(str((root / p).resolve()), c) for p, c in items]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(root, module):
    summary = json.loads((root / "summary.json").read_text())
    source = Path(summary["root"])
    os.environ["TASK1_SMOKE_ORIGINAL"] = str(source)
    original = HighwayDataModule(num_workers=0)
    original.setup(root_dir="TASK1_SMOKE_ORIGINAL")
    expected = {}
    for split in ("train", "val"):
        subset = getattr(original, "data_" + split)
        expected[split] = canonical([subset.dataset.items[i] for i in subset.indices], source)
    expected["test"] = canonical(original.data_test.items, source)
    provenance = list(csv.DictReader((root / "provenance.csv").open()))
    synthetic = canonical([(r["path"], int(r["label"])) for r in provenance], root)
    actual = {s: canonical(getattr(module, "data_" + s).items, root) for s in expected}
    require(actual["train"] == expected["train"] + synthetic, "Training manifest mismatch")
    require(actual["val"] == expected["val"] and actual["test"] == expected["test"], "Held-out split changed")
    require(
        [len(expected["train"]), len(synthetic), len(actual["train"]), len(actual["val"]), len(actual["test"])] == [11865, 1359, 13224, 1319, 3296],
        "Unexpected Task 1 sample counts",
    )
    memberships = {s: {p for p, _ in items} for s, items in actual.items()}
    require(not (memberships["train"] & (memberships["val"] | memberships["test"])), "Training/held-out overlap")
    require(not (memberships["val"] & memberships["test"]), "Validation/test overlap")
    fit = set(expected["train"])
    for row in provenance:
        require((str((source / row["parent"]).resolve()), int(row["label"])) in fit, "Invalid synthetic parent/label")
    expected_hashes = {str((source / r["path"]).resolve()): r["sha256"] for r in csv.DictReader((root / "original_integrity.csv").open())}
    expected_hashes.update({str((root / r["path"]).resolve()): r["sha256"] for r in provenance})
    evidence, snapshots = {}, {}
    synthetic_paths = {p for p, _ in synthetic}
    for split, items in actual.items():
        for path, label in items:
            p = Path(path)
            x = np.load(p, allow_pickle=False)
            require(x.shape == (512, 243) and x.dtype == np.float32 and np.isfinite(x).all(), f"Invalid array {p}")
            h = digest(x)
            require(h == expected_hashes[path], f"Content changed since generation: {p}")
            require(h not in evidence, "Duplicate content across corpus")
            evidence[h] = {"path": path, "label": label, "split": split, "origin": "synthetic" if path in synthetic_paths else "original"}
            snapshots[path] = file_hash(p)
        print(f"Full read/hash audit: {split} {len(items)}", flush=True)
    for name in ("train.txt", "test.txt"):
        snapshots[str(source / name)] = file_hash(source / name)
    for p in root.glob("*.txt"):
        snapshots[str(p)] = file_hash(p)
    counts = Counter(c for _, c in actual["train"])
    require(counts == Counter({int(k): v for k, v in summary["augmented"].items()}), "Class counts mismatch")
    require(set(counts) == set(range(9)), "Missing or invalid class labels")
    # Exercise the inherited, shuffled loader too, before targeted training.
    x, y = next(iter(module.train_dataloader()))
    require(x.shape == (3, 512, 243) and x.dtype == torch.float32 and y.dtype == torch.int64, "Inherited DataLoader batch contract failed")
    return evidence, snapshots, counts


class SmokeDataModule(SyntheticHighwayDataModule):
    """Use frozen production datasets, restricting only training sampler length."""

    smoke_indices = None

    def train_dataloader(self):
        if self.smoke_indices is None:
            return super().train_dataloader()
        return DataLoader(self.data_train, batch_size=3, sampler=self.smoke_indices, num_workers=0)


def select_indices(items):
    originals, synthetic = {}, {}
    for i, (path, label) in enumerate(items):
        (synthetic if path.startswith("synthetic/") else originals).setdefault(label, i)
    return [
        originals[0],
        synthetic[3],
        originals[1],
        originals[2],
        synthetic[4],
        originals[3],
        originals[4],
        synthetic[5],
        originals[5],
        originals[6],
        synthetic[6],
        originals[7],
        originals[8],
        synthetic[7],
        synthetic[8],
    ]


class TrainingEvidence(L.Callback):
    def __init__(self, evidence):
        self.evidence = evidence
        self.batches = []
        self.validation_batches = 0
        self.forward_records = []

    def forward_hook(self, model, inputs, output):
        x = inputs[0]
        require(output.shape == (len(x), 9) and torch.isfinite(output).all(), "Invalid logits")
        records = [self.evidence[digest(a.detach().cpu().numpy())] for a in x]
        require(all(r["split"] == ("train" if model.training else "val") for r in records), "Wrong split reached model")
        self.forward_records.append({"phase": "train" if model.training else "val", "samples": records})

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        x, y = batch
        require(x.dtype == torch.float32 and y.dtype == torch.int64 and x.device.type == "cpu", "Batch dtype/device error")
        records = [self.evidence[digest(a.numpy())] for a in x]
        require(y.tolist() == [r["label"] for r in records], "Labels do not match observed tensors")
        require({r["origin"] for r in records} == {"original", "synthetic"}, "Batch must contain both origins")
        self.current = {"batch": batch_idx, "samples": records}
        self.before = [p.detach().clone() for p in model.parameters()]
        self.gradient_norm = None

    def on_before_optimizer_step(self, trainer, model, optimizer):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        require(grads and all(torch.isfinite(g).all() for g in grads), "Missing/nonfinite gradients")
        self.gradient_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        require(self.gradient_norm > 0, "Zero gradients")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        loss = float(outputs["loss"].detach())
        require(np.isfinite(loss) and self.gradient_norm is not None, "Loss/backpropagation failed")
        delta = max(float((p.detach() - before).abs().max()) for p, before in zip(model.parameters(), self.before))
        require(delta > 0 and all(torch.isfinite(p).all() for p in model.parameters()), "Optimizer did not update finite parameters")
        self.batches.append({**self.current, "loss": loss, "gradient_l2_norm": self.gradient_norm, "max_parameter_update": delta})

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        require(outputs is not None and torch.isfinite(outputs), "Invalid validation loss")
        self.validation_batches += 1


def main(root, result):
    root, result = Path(root).resolve(), Path(result).resolve()
    require(not result.exists(), "Choose a new result path; existing evidence is not overwritten")
    L.seed_everything(20261112, workers=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    module = SmokeDataModule(root, batch_size=3, num_workers=0)
    module.setup()
    evidence, snapshots, counts = audit(root, module)
    module.smoke_indices = select_indices(module.data_train.items)
    model = HighwayBaselineModel(num_classes=9)
    callback = TrainingEvidence(evidence)
    hook = model.register_forward_hook(callback.forward_hook)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )
    trainer.fit(model, datamodule=module)
    hook.remove()
    seen = [r for batch in callback.forward_records if batch["phase"] == "train" for r in batch["samples"]]
    origins = Counter(r["origin"] for r in seen)
    require(origins == Counter(original=9, synthetic=6), "Actual forward-pass participation mismatch")
    require({r["label"] for r in seen} == set(range(9)), "Not all classes reached the model")
    require(trainer.global_step == 5 and len(callback.batches) == 5 and callback.validation_batches == 1, "Training/validation step counts mismatch")
    require(len(module.data_train) == 13224, "Training dataset was truncated")
    for path, checksum in snapshots.items():
        require(file_hash(Path(path)) == checksum, f"Data or manifest modified: {path}")
    result.write_text(
        json.dumps(
            {
                "status": "passed",
                "architecture": "HighwayBaselineModel / torchvision SqueezeNet 1.1",
                "seed": 20261112,
                "device": "cpu",
                "precision": "float32",
                "optimizer": "repository AdamW lr=0.001 weight_decay=0.05",
                "dataset_counts": {"original_fitting": 11865, "synthetic": 1359, "augmented_fitting": 13224, "validation": 1319, "test": 3296},
                "augmented_class_counts": dict(counts),
                "actual_training_origins": dict(origins),
                "optimizer_updates": trainer.global_step,
                "validation_batches": callback.validation_batches,
                "validation_loss": float(trainer.callback_metrics["val_loss"]),
                "all_arrays_and_manifests_unchanged": True,
                "files_rehashed_after_training": len(snapshots),
                "inherited_dataloader_batch_verified": True,
                "no_synthetic_in_validation_or_test": True,
                "training_batches": callback.batches,
                "forward_pass_records": callback.forward_records,
            },
            indent=2,
        )
    )
    print(
        f"PASSED: five optimizer updates, originals={origins['original']}, synthetic={origins['synthetic']}, one validation batch. Evidence: {result}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    main(args.root, args.result)
