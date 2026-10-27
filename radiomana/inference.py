"""
Run the trained Highway2 -> Highway1 ViT jamming detector.

    from radiomana import predict_jamming

    result = predict_jamming("fiot_highway1_subset/data/40504.npy")
    result.jammed      # array([False])
    result.p_jammed    # array([0.03])

Also usable from the shell:

    python -m radiomana.inference fiot_highway1_subset/data/40504.npy more.npy ...
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .models import HighwayViTModel

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "vit_hw2_to_hw1.pt"
JAMMED_MIN_CLASS = 4  # Highway2 classes 4-8 are interference
CLASS_LABELS = [
    "None (background a)",
    "None (background b)",
    "None (background c)",
    "None (background d)",
    "Chirp, high distance",
    "Chirp, medium distance",
    "Chirp, small distance",
    "Cigarette lighter 1",
    "Cigarette lighter 2",
]


@dataclass
class JammingPrediction:
    """Per-sample outputs of predict_jamming; every field has one entry per input PSD."""

    p_jammed: np.ndarray  # (N,) probability of interference, summed over Highway2 classes 4-8
    jammed: np.ndarray  # (N,) bool, p_jammed >= threshold
    class_probs: np.ndarray  # (N, 9) Highway2 class probabilities
    predicted_class: np.ndarray  # (N,) most likely Highway2 class index

    @property
    def predicted_label(self) -> list:
        return [CLASS_LABELS[idx] for idx in self.predicted_class]

    def __len__(self):
        return len(self.p_jammed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def load_jamming_detector(weights_path: str | Path = DEFAULT_WEIGHTS, device: str | None = None) -> HighwayViTModel:
    """
    Load the trained ViT in eval mode. Cached, so repeated calls with the same arguments are free.

    Parameters
    ----------
    weights_path : str or Path
        State dict saved from HighwayViTModel, by default models/vit_hw2_to_hw1.pt in the repo.
    device : str, optional
        "cuda" or "cpu"; picks cuda when available.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"model weights not found at {weights_path}; train one with examples/train_vit_cross_site.py")
    # pretrained=False: the saved state dict replaces every weight, so skip loading ImageNet ones first
    model = HighwayViTModel(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.to(device or default_device()).eval()


def _as_batch(psds) -> torch.Tensor:
    """accept a .npy path, a list of paths, or an array / tensor of shape (freq, time) or (N, freq, time)"""
    if isinstance(psds, (str, Path)):
        psds = [psds]
    if isinstance(psds, (list, tuple)) and psds and isinstance(psds[0], (str, Path)):
        psds = np.stack([np.load(path) for path in psds])
    batch = torch.as_tensor(np.asarray(psds) if not isinstance(psds, torch.Tensor) else psds).float()
    if batch.ndim == 2:
        batch = batch.unsqueeze(0)
    if batch.ndim != 3:
        raise ValueError(f"expected PSDs of shape (freq, time) or (N, freq, time), got {tuple(batch.shape)}")
    return batch


@torch.no_grad()
def predict_jamming(
    psds,
    model: HighwayViTModel | None = None,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> JammingPrediction:
    """
    Classify log PSDs (dB, e.g. (512, 243) FIOT samples) as jammed / not jammed.

    Parameters
    ----------
    psds : path, list of paths, np.ndarray or torch.Tensor
        One or more PSDs; arrays may be (freq, time) or (N, freq, time).
    model : HighwayViTModel, optional
        Defaults to load_jamming_detector().
    threshold : float, default 0.5
        Decision threshold on p_jammed; raise it for fewer false alarms, lower it to catch more jammers.
    batch_size : int, default 64
        PSDs per forward pass; lower it if the GPU runs out of memory.
    """
    model = model or load_jamming_detector()
    batch = _as_batch(psds)
    probs = []
    for start in range(0, len(batch), batch_size):
        x = batch[start : start + batch_size].to(model.device)
        with torch.autocast(model.device.type, enabled=model.device.type == "cuda"):
            probs.append(torch.softmax(model(x).float(), dim=1).cpu())
    class_probs = torch.cat(probs).numpy()
    p_jammed = class_probs[:, JAMMED_MIN_CLASS:].sum(axis=1)
    return JammingPrediction(
        p_jammed=p_jammed,
        jammed=p_jammed >= threshold,
        class_probs=class_probs,
        predicted_class=class_probs.argmax(axis=1),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m radiomana.inference file.npy [file.npy ...]")
    paths = sys.argv[1:]
    result = predict_jamming(paths)
    for path, p_jam, jammed, label in zip(paths, result.p_jammed, result.jammed, result.predicted_label):
        print(f"{path}  {'JAMMED' if jammed else 'clean '}  p_jammed={p_jam:.3f}  ({label})")
