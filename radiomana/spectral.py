"""spectral statistics for PSD data"""

import torch


def spectral_stats(psd_db: torch.Tensor, freqs_hz: torch.Tensor) -> dict:
    """
    Compute per-sample spectral statistics for a batch of PSDs.

    Collapses the time axis to a mean power spectrum, then takes the first two
    power-weighted moments of that spectrum:

        centroid = sum(f * P) / sum(P)
        spread   = sqrt(sum((f - centroid)^2 * P) / sum(P))

    Moments are taken on LINEAR power, so the stored dB values are converted
    with P = 10 ** (psd_db / 10) first. Averaging dB directly would weight the
    noise floor far too heavily.

    Parameters
    ----------
    psd_db : torch.Tensor
        PSDs in dB, shape (batch, freq, time).
    freqs_hz : torch.Tensor
        Frequency of each bin in Hz, shape (freq,).

    Returns
    -------
    dict of torch.Tensor, each shape (batch,)
        centroid_hz : power-weighted mean frequency
        spread_hz   : power-weighted standard deviation about the centroid
        peak_hz     : frequency of the strongest bin
        db_std      : standard deviation of the raw dB values
    """
    power = torch.pow(10.0, psd_db.double() / 10.0)  # dB -> linear
    spectrum = power.mean(dim=2)  # collapse time -> (batch, freq)

    total = spectrum.sum(dim=1, keepdim=True).clamp(min=1e-30)
    weights = spectrum / total

    freqs = freqs_hz.double().unsqueeze(0)
    centroid = (weights * freqs).sum(dim=1)
    variance = (weights * (freqs - centroid.unsqueeze(1)) ** 2).sum(dim=1)

    return {
        "centroid_hz": centroid,
        "spread_hz": torch.sqrt(variance),
        "peak_hz": freqs_hz.double()[spectrum.argmax(dim=1)],
        "db_std": psd_db.double().flatten(1).std(dim=1),
    }


def bin_frequencies(dataset) -> torch.Tensor:
    """Frequency of each PSD bin in Hz, baseband, for a Highway2Dataset."""
    return torch.linspace(-dataset.sample_rate_hz / 2, dataset.sample_rate_hz / 2, 512)
