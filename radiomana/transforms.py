"""Transforms compatible with 2D log PSD tensors"""

import torch
from torch import Tensor
from torchvision.transforms import v2


class LogNoise(v2.Transform):
    """
    Transform to add noise to a log-scaled PSD tensor.
    """

    def __init__(self, p: float = 1, noise_power_db: float = -90):
        """
        Parameters
        ----------
        p : float, default 1
            Probability of the psd being noised.
        """
        super().__init__()
        self.p = p
        self.noise_power_db = noise_power_db

    def forward(self, *inputs):
        """Consume inputs in torchvision v2 style."""
        if len(inputs) == 1:
            # check if single input is a tuple (psd, label)
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) == 2:
                psd, label = inputs[0]
                return self._apply(psd), label
            else:
                return self._apply(inputs[0])
        elif len(inputs) == 2:
            label, psd = inputs
            return label, self._apply(psd)
        else:
            return super().forward(*inputs)

    def _apply(self, log_psd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        log_psd : Tensor
            Log PSD to add noise to of shape (freq, time) or (bs, freq, time).

        Returns
        -------
        Tensor
            Noised log PSD of same shape as input.
        """
        if torch.rand(1).item() < self.p:
            noise_power_linear = 10 ** (self.noise_power_db / 10)
            # convert data from dB to linear
            psd_linear = 10 ** (log_psd / 10)
            # add noise in linear domain
            noise = torch.randn_like(psd_linear) * noise_power_linear**0.5
            noisy_psd_linear = psd_linear + noise
            # ensure no negative or zero values before log10
            noisy_psd_linear = torch.clamp(noisy_psd_linear, min=1e-12)
            log_psd = 10 * torch.log10(noisy_psd_linear)
        return log_psd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_power_db={self.noise_power_db}, p={self.p})"


class RandomTimeCrop(v2.Transform):
    """
    Transform to randomly crop a fixed-width time segment from a PSD tensor.
    Crops to a fixed width with random offset to ensure consistent batch dimensions.
    """

    def __init__(self, crop_width: int = 211):
        """
        Parameters
        ----------
        crop_width : int, default 211
            Fixed width to crop to in time dimension.
        """
        super().__init__()
        self.crop_width = crop_width

    def forward(self, *inputs):
        """Consume inputs in torchvision v2 style."""
        if len(inputs) == 1:
            # check if single input is a tuple (psd, label)
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) == 2:
                psd, label = inputs[0]
                return self._apply(psd), label
            else:
                return self._apply(inputs[0])
        elif len(inputs) == 2:
            label, psd = inputs
            return label, self._apply(psd)
        else:
            return super().forward(*inputs)

    def _apply(self, log_psd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        log_psd : Tensor
            Log PSD tensor of shape (freq, time) or (bs, freq, time).

        Returns
        -------
        Tensor
            Cropped log PSD tensor with fixed width in time dimension.
        """
        # handle both 2D (freq, time) and 3D (bs, freq, time) tensors
        if len(log_psd.shape) == 2:
            # shape: (freq, time)
            time_dim = log_psd.shape[1]
            if time_dim <= self.crop_width:
                # if input is already smaller than crop width, return as-is
                return log_psd
            # randomly choose starting position
            max_start = time_dim - self.crop_width
            start = torch.randint(0, max_start + 1, (1,)).item()
            return log_psd[:, start : start + self.crop_width]
        elif len(log_psd.shape) == 3:
            # shape: (bs, freq, time)
            time_dim = log_psd.shape[2]
            if time_dim <= self.crop_width:
                # if input is already smaller than crop width, return as-is
                return log_psd
            # randomly choose starting position
            max_start = time_dim - self.crop_width
            start = torch.randint(0, max_start + 1, (1,)).item()
            return log_psd[:, :, start : start + self.crop_width]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {log_psd.shape}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_width={self.crop_width})"


class _LogPsdTransform(v2.Transform):
    """Shared input handling for transforms that act on a log PSD of shape (freq, time) or (bs, freq, time)."""

    def forward(self, *inputs):
        """Consume inputs in torchvision v2 style."""
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) == 2:
                psd, label = inputs[0]
                return self._apply(psd), label
            return self._apply(inputs[0])
        elif len(inputs) == 2:
            label, psd = inputs
            return label, self._apply(psd)
        return super().forward(*inputs)


class RandomTimeRoll(_LogPsdTransform):
    """
    Circularly shift a PSD along time by a random offset.

    Jammers can start anywhere in a snapshot, so where an emitter appears in time carries no label information.
    """

    def _apply(self, log_psd: Tensor) -> Tensor:
        shift = int(torch.randint(0, log_psd.shape[-1], (1,)))
        return torch.roll(log_psd, shifts=shift, dims=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFreqShift(_LogPsdTransform):
    """
    Circularly shift a PSD along frequency by up to max_fraction of the band.

    Another site's jammers sit at other centre frequencies, so the model should not key on absolute bin position.
    """

    def __init__(self, max_fraction: float = 0.1):
        super().__init__()
        self.max_fraction = max_fraction

    def _apply(self, log_psd: Tensor) -> Tensor:
        max_shift = int(self.max_fraction * log_psd.shape[-2])
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        return torch.roll(log_psd, shifts=shift, dims=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_fraction={self.max_fraction})"


class RandomFreqFlip(_LogPsdTransform):
    """Mirror a PSD along frequency with probability p."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def _apply(self, log_psd: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return log_psd.flip(-2)
        return log_psd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class SyntheticJammer:
    """
    Turn a fraction of clean PSDs into synthetic CW tone, pulsed, or wideband noise interference.

    Highway2 only contains chirp and cigarette-lighter jammers, while other sites (e.g. Highway1) also see
    tones and pulsed jammers. Painting simple versions of those onto clean Highway2 backgrounds teaches the
    model that a strong structured emitter above the noise floor means jammed.

    Takes and returns a (log_psd, label) pair of a single (freq, time) PSD; synthetic samples are relabelled
    to jammed_label. Sizes are fractions of the axes, so it works at native (512, 243) or resized resolution.
    """

    def __init__(self, p: float = 0.15, min_db: float = 12.0, range_db: float = 25.0, jammed_label: int = 4):
        """
        Parameters
        ----------
        p : float, default 0.15
            Probability that a clean sample is given a synthetic jammer.
        min_db, range_db : float
            Jammer power is drawn uniformly from [min_db, min_db + range_db] dB above the reference level
            (the row's own median for tones, the PSD median otherwise).
        jammed_label : int, default 4
            Label given to synthetic samples; also the first label treated as already jammed.
        """
        self.p = p
        self.min_db = min_db
        self.range_db = range_db
        self.jammed_label = jammed_label

    def __call__(self, sample):
        log_psd, label = sample
        if label >= self.jammed_label or torch.rand(1).item() >= self.p:
            return log_psd, label
        return self.inject(log_psd.clone()), self.jammed_label

    def inject(self, log_psd: Tensor) -> Tensor:
        n_freq, n_time = log_psd.shape
        bin_scale = max(1, round(n_freq / 224))  # parameters were tuned on 224 frequency bins
        floor = log_psd.median()
        offset = self.min_db + self.range_db * torch.rand(1).item()
        kind = int(torch.randint(0, 3, (1,)))
        if kind == 0:  # CW tone: narrow horizontal line, full or partial duration
            width = int(torch.randint(1, 4, (1,))) * bin_scale
            f0 = int(torch.randint(0, n_freq - width, (1,)))
            t0 = int(torch.randint(0, n_time // 2, (1,))) if torch.rand(1).item() < 0.3 else 0
            band = log_psd[f0 : f0 + width, t0:]
            power = log_psd[f0 : f0 + width].median() + offset
            log_psd[f0 : f0 + width, t0:] = torch.maximum(band, power + torch.randn_like(band) * 2)
        elif kind == 1:  # pulsed: periodic broadband vertical lines
            period = int(torch.randint(4, 40, (1,)))
            phase = int(torch.randint(0, period, (1,)))
            f_lo = int(torch.randint(0, n_freq // 2, (1,)))
            f_hi = int(torch.randint(f_lo + n_freq // 4, n_freq + 1, (1,)))
            cols = torch.arange(phase, n_time, period)
            band = log_psd[f_lo:f_hi][:, cols]
            log_psd[f_lo:f_hi, cols] = torch.maximum(band, floor + offset - 5 + torch.randn_like(band) * 2)
        else:  # wideband noise: raise a contiguous band for part or all of the snapshot
            width = int(torch.randint(n_freq // 16, n_freq // 3, (1,)))
            f0 = int(torch.randint(0, n_freq - width, (1,)))
            t0 = int(torch.randint(0, n_time // 2, (1,))) if torch.rand(1).item() < 0.5 else 0
            band = log_psd[f0 : f0 + width, t0:]
            log_psd[f0 : f0 + width, t0:] = torch.maximum(band, floor + offset - 5 + torch.randn_like(band) * 3)
        return log_psd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, min_db={self.min_db}, range_db={self.range_db}, jammed_label={self.jammed_label})"


class BenignSpurs:
    """
    Add 1-2 faint, full-duration narrowband spurs to clean PSDs without changing the label.

    Receivers at other sites pick up weak constant spurs (clock harmonics etc.) that are not jammers.
    Highway2 backgrounds have almost none, so a model trained with SyntheticJammer alone mistakes them
    for tones. Takes and returns a (log_psd, label) pair of a single (freq, time) PSD.
    """

    def __init__(self, p: float = 0.3, min_db: float = 2.0, max_db: float = 7.0, jammed_label: int = 4):
        self.p = p
        self.min_db = min_db
        self.max_db = max_db
        self.jammed_label = jammed_label

    def __call__(self, sample):
        log_psd, label = sample
        if label >= self.jammed_label or torch.rand(1).item() >= self.p:
            return log_psd, label
        log_psd = log_psd.clone()
        n_freq = log_psd.shape[0]
        width = max(1, round(n_freq / 224))
        for _ in range(int(torch.randint(1, 3, (1,)))):
            f0 = int(torch.randint(0, n_freq - width + 1, (1,)))
            rows = log_psd[f0 : f0 + width]
            level = rows.median() + self.min_db + (self.max_db - self.min_db) * torch.rand(1).item()
            spur = level + torch.randn_like(rows) * 0.9
            # add powers in the linear domain
            log_psd[f0 : f0 + width] = 10 * torch.log10(10 ** (rows / 10) + 10 ** (spur / 10))
        return log_psd, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, min_db={self.min_db}, max_db={self.max_db})"
