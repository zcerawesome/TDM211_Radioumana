#!/usr/bin/env python3

import math

import lightning as L
import torch
import torchmetrics
from einops import repeat, rearrange
from torchinfo import summary
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_16_Weights, squeezenet1_1, vit_b_16


class ModelBaseClass(L.LightningModule):
    """Example model architecture for the FIOT datasets"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch, batch_idx):
        x, y_true = batch
        y_hat = self(x)
        return self.criterion(y_hat, y_true)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def on_test_start(self):
        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=self.num_classes, task="multiclass").to(self.device)
        self.test_f1 = torchmetrics.F1Score(num_classes=self.num_classes, average="macro", task="multiclass").to(self.device)

    def test_step(self, batch, batch_idx):
        """similar to step, but we also update confusion matrix and F1 score"""
        x, y_true = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.test_confmat.update(preds, y_true)
        self.test_f1.update(preds, y_true)
        loss = self.criterion(y_hat, y_true)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        self.confmat = self.test_confmat.compute()
        self.log(
            "test_acc",
            torch.sum(torch.diagonal(self.confmat)) / torch.sum(self.confmat).item(),
        )
        self.log("test_f1", self.test_f1.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        return optimizer


class HighwayBaselineModel(ModelBaseClass):
    """Example model architecture for the FIOT datasets"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        # submodel selection
        self.submodel = squeezenet1_1(num_classes=num_classes)

    def forward(self, x):
        # add channel dimension and repeat to 3 channels in one step
        x = repeat(x, "batch height width -> batch 3 height width")
        x = self.submodel(x)
        return x


class HighwayViTModel(ModelBaseClass):
    """
    ViT-B/16 on PSD spectrograms.

    The backbone expects 3-channel 224x224 images normalised with ImageNet
    statistics, so forward() maps the (512, 243) dB PSD into that space:
    clip to a dB window, rescale to [0, 1], repeat to 3 channels,
    resize, then normalise. The classification head is replaced with a
    fresh linear layer for num_classes.

    normalization="absolute" clips to the fixed [db_min, db_max] window.
    normalization="per_sample" first subtracts each PSD's median, so the
    window is relative to the local noise floor; this is what lets a model
    trained on one site (Highway2) transfer to another (Highway1) whose
    receivers sit at a different absolute power level.

    freeze_blocks freezes the patch embedding plus the first N encoder
    blocks (freeze_backbone=True freezes everything except the head).

    The defaults are the configuration that transferred best from Highway2
    to Highway1: ImageNet weights, per-sample normalisation with a
    [-10, 50] dB window above the median, and layer-wise lr decay (the top
    block trains at backbone_lr, each block below at layer_decay times the
    one above). Pair it with HighwayDataModule's use_cross_site_augmentation.
    """

    def __init__(
        self,
        num_classes: int = 9,
        freeze_backbone: bool = False,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        db_min: float = -10.0,
        db_max: float = 50.0,
        pretrained: bool = True,
        freeze_blocks: int = 0,
        normalization: str = "per_sample",
        layer_decay: float = 0.75,
        weight_decay: float = 0.05,
        warmup_steps: int = 300,
    ):
        super().__init__(num_classes)
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        if normalization not in ("absolute", "per_sample"):
            raise ValueError("normalization must be 'absolute' or 'per_sample'")

        self.submodel = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.submodel.heads.head = nn.Linear(self.submodel.hidden_dim, num_classes)

        if freeze_backbone:
            for name, param in self.submodel.named_parameters():
                param.requires_grad = name.startswith("heads.")
        elif freeze_blocks > 0:
            frozen = ["conv_proj.", "class_token", "encoder.pos_embedding"]
            frozen += [f"encoder.layers.encoder_layer_{idx}." for idx in range(freeze_blocks)]
            for name, param in self.submodel.named_parameters():
                if any(name.startswith(prefix) for prefix in frozen):
                    param.requires_grad = False

        # ImageNet normalisation constants, as buffers so they follow .to(device)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        # x: (batch, freq, time) in dB, noise floor ~ -90
        x = x.float()
        if self.hparams.normalization == "per_sample":
            x = x - x.flatten(1).median(dim=1).values.view(-1, 1, 1)
        lo, hi = self.hparams.db_min, self.hparams.db_max
        x = (x.clamp(lo, hi) - lo) / (hi - lo)  # -> [0, 1]
        x = repeat(x, "batch height width -> batch 3 height width")
        if x.shape[-2:] != (224, 224):
            # area averages neighbouring bins, so 1-bin-wide tones are not skipped over
            x = F.interpolate(x, size=(224, 224), mode="area")
        return (x - self.img_mean) / self.img_std

    def forward(self, x):
        return self.submodel(self.preprocess(x))

    def configure_optimizers(self):
        # layer-wise lr decay: fresh head at head_lr, top block at backbone_lr, lower blocks and embeddings smaller
        n_layers = len(self.submodel.encoder.layers)
        groups = {}
        for name, param in self.submodel.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("heads."):
                key, lr = "head", self.hparams.head_lr
            elif name.startswith("encoder.layers.encoder_layer_"):
                layer = int(name.split(".")[2].rsplit("_", 1)[1])
                key, lr = f"layer{layer}", self.hparams.backbone_lr * self.hparams.layer_decay ** (n_layers - layer)
            elif name.startswith("encoder.ln"):
                key, lr = "final_ln", self.hparams.backbone_lr
            else:
                key, lr = "embed", self.hparams.backbone_lr * self.hparams.layer_decay ** (n_layers + 1)
            # no weight decay on norms, biases, positional embedding and class token
            no_wd = param.ndim <= 1 or name.endswith("pos_embedding") or name.endswith("class_token")
            key += "_no_wd" if no_wd else ""
            groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_wd else self.hparams.weight_decay})
            groups[key]["params"].append(param)
        optimizer = torch.optim.AdamW(list(groups.values()))

        # linear warmup, then cosine decay to zero
        total = self.trainer.estimated_stepping_batches
        warmup = min(self.hparams.warmup_steps, total // 10)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class StudentModel(ModelBaseClass):
    """Unfinished student model template"""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        # create layers (unfinished)
        self.layers = nn.Identity()

    def forward(self, x):
        # x will be of shape (batchsize, 512, 243)
        x = rearrange(x, "batchsize height width -> batchsize 1 height width")  # add channel dimension
        # x is now of shape (batchsize, 1, 512, 243)
        x = self.layers(x)
        # x should be of shape (batchsize, num_classes)
        return x


if __name__ == "__main__":
    model = HighwayBaselineModel()
    summary(model, input_data=torch.randn(1, 512, 243), device="cpu")
