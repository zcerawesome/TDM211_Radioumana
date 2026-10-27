#!/usr/bin/env python3
"""
Train HighwayViTModel on Highway2 and evaluate binary jammed / not-jammed transfer to Highway1.

Requires DSET_FIOT_HIGHWAY2 and DSET_FIOT_HIGHWAY1_SUBSET to be set. Highway1 is only used for testing.
"""

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from torch.utils.data import ConcatDataset, DataLoader

from radiomana.datasets import Highway2Dataset, HighwayDataModule
from radiomana.models import HighwayViTModel

HW2_JAMMED_MIN = 4  # Highway2 classes 4-8 are interference
HW1_JAMMED_MIN = 3  # Highway1 classes 3-10 are interference


@torch.no_grad()
def evaluate_highway1(model, batch_size: int = 64, num_workers: int = 4):
    """binary jammed / not-jammed metrics on every Highway1 subset sample"""
    data = ConcatDataset([Highway2Dataset(root_dir="DSET_FIOT_HIGHWAY1_SUBSET", subset=subset) for subset in ["train", "test"]])
    model.eval()
    p_jam, labels = [], []
    for x, y in DataLoader(data, batch_size=batch_size, num_workers=num_workers):
        with torch.autocast("cuda", dtype=torch.float16):
            probs = torch.softmax(model(x.to(model.device)).float(), dim=1)
        p_jam.append(probs[:, HW2_JAMMED_MIN:].sum(dim=1).cpu())
        labels.append(y)
    pred = torch.cat(p_jam).numpy() >= 0.5
    true = torch.cat(labels).numpy() >= HW1_JAMMED_MIN

    tp, tn = int((pred & true).sum()), int((~pred & ~true).sum())
    fp, fn = int((pred & ~true).sum()), int((~pred & true).sum())
    recall, precision = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    print(f"highway1 confusion  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"accuracy:     {(tp + tn) / len(true):.2%}")
    print(f"balanced acc: {(recall + tn / max(tn + fp, 1)) / 2:.3f}")
    print(f"jammed precision {precision:.3f}  recall {recall:.3f}  f1 {2 * precision * recall / max(precision + recall, 1e-9):.3f}")


def train():
    dmodule = HighwayDataModule(batch_size=32, num_workers=8, use_cross_site_augmentation=True)
    model = HighwayViTModel()  # defaults: ImageNet weights, per-sample normalisation, layer-wise lr decay

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=7,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[RichProgressBar()],
    )
    trainer.fit(model, datamodule=dmodule)
    results = trainer.test(model, datamodule=dmodule)[0]
    print(f"highway2 test acc: {results['test_acc']:8.3%}  f1: {results['test_f1']:0.3f}")

    evaluate_highway1(model)

    filename = "vit_hw2_to_hw1.pt"
    torch.save(model.state_dict(), filename)
    print(f"saved model weights to {filename}")


if __name__ == "__main__":
    np.random.seed(0)
    train()
