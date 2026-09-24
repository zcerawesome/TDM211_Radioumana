#!/usr/bin/env python3

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
    ImageNet-pretrained ViT-B/16 fine-tuned on PSD spectrograms.

    The backbone expects 3-channel 224x224 images normalised with ImageNet
    statistics, so forward() maps the (512, 243) dB PSD into that space:
    clip to a fixed dB window, rescale to [0, 1], repeat to 3 channels,
    resize, then normalise. The classification head is replaced with a
    fresh linear layer for num_classes.
    """

    def __init__(
        self,
        num_classes: int = 9,
        freeze_backbone: bool = False,
        backbone_lr: float = 2e-5,
        head_lr: float = 1e-3,
        db_min: float = -90.0,
        db_max: float = -10.0,
    ):
        super().__init__(num_classes)
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.submodel = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.submodel.heads.head = nn.Linear(self.submodel.hidden_dim, num_classes)

        if freeze_backbone:
            for name, param in self.submodel.named_parameters():
                param.requires_grad = name.startswith("heads.")

        # ImageNet normalisation constants, as buffers so they follow .to(device)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x: (batch, 512, 243) in dB, noise floor ~ -90
        lo, hi = self.hparams.db_min, self.hparams.db_max
        x = (x.clamp(lo, hi) - lo) / (hi - lo)  # -> [0, 1]
        x = repeat(x, "batch height width -> batch 3 height width")
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        return self.submodel(x)

    def configure_optimizers(self):
        # small lr for the pretrained backbone, larger for the fresh head
        head_params = [p for n, p in self.submodel.named_parameters() if n.startswith("heads.") and p.requires_grad]
        backbone_params = [p for n, p in self.submodel.named_parameters() if not n.startswith("heads.") and p.requires_grad]
        groups = [{"params": head_params, "lr": self.hparams.head_lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.hparams.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches)
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
