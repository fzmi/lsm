import os
import math
from typing import Any, Mapping, Sequence, Tuple, Union, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.image import StructuralSimilarityIndexMeasure


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def filesize(filepath: str):
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


# ---- Fast helpers (avoid per-call overhead) ----
_SSIM_CACHE: dict = {}
_DEFAULT_MAX_VALUE: float = 65535.0
_LOG20_DEFAULT_MAX: float = 20.0 * math.log10(_DEFAULT_MAX_VALUE)


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure a 4D NCHW tensor with minimal checks; used only when SSIM is requested."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x  # assume already NxCxHxW


def _get_ssim_measure(device: torch.device, data_range: float):
    key = (str(device), float(data_range))
    m = _SSIM_CACHE.get(key)
    if m is None:
        m = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        _SSIM_CACHE[key] = m
    return m


def calc_metrics_patch(
    recon: torch.Tensor,
    orig: torch.Tensor,
    psnr: bool = True,
    ssim: bool = True,
    max_value: float = 65535.0,
    eps: float = 1e-8,
):
    """Calculate image quality metrics between two patches.

    Args:
        recon (torch.Tensor): The reconstructed image tensor.
        orig (torch.Tensor): The original image tensor.
        psnr (bool, optional): Whether to compute PSNR. Defaults to True.
        ssim (bool, optional): Whether to compute SSIM. Defaults to True.
        max_value (float, optional): The maximum pixel value. Defaults to 65535.0.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-8.
    """
    results = {}
    se = (recon - orig) ** 2
    mean_se = se.mean().item()
    results["mse"] = float(mean_se)

    # note: whole image PSNR is not average of patch-wise PSNRs
    if psnr:
        if max_value == _DEFAULT_MAX_VALUE:
            results["psnr_patch"] = float(_LOG20_DEFAULT_MAX - 10.0 * math.log10(mean_se + eps))
        else:
            results["psnr_patch"] = float(20.0 * math.log10(max_value) - 10.0 * math.log10(mean_se + eps))

    if ssim:
        x = _ensure_nchw(recon)
        y = _ensure_nchw(orig)
        measure = _get_ssim_measure(x.device, max_value)
        results["ssim_patch"] = float(measure(x, y).item())

    return results


def bjontegaard_delta(bpp_anchor, metric_anchor, bpp_test, metric_test):
    """Bjontegaard Delta metric (BD-PSNR or BD-SSIM).

    Computes the average metric gain of the test RD curve over the anchor
    across their overlapping log-rate range using cubic polynomial fitting.
    Positive value means the test method is better than the anchor.

    Args:
        bpp_anchor: bit-per-pixel values for the anchor/reference method.
        metric_anchor: PSNR or SSIM values for the anchor method.
        bpp_test: bit-per-pixel values for the test method.
        metric_test: PSNR or SSIM values for the test method.

    Returns:
        float: BD-PSNR (dB) or BD-SSIM delta. nan if curves don't overlap
               or have too few points.
    """
    bpp_a = np.asarray(bpp_anchor, dtype=np.float64).ravel()
    met_a = np.asarray(metric_anchor, dtype=np.float64).ravel()
    bpp_t = np.asarray(bpp_test, dtype=np.float64).ravel()
    met_t = np.asarray(metric_test, dtype=np.float64).ravel()

    def _clean(bpp, met):
        mask = np.isfinite(bpp) & (bpp > 0) & np.isfinite(met)
        bpp, met = bpp[mask], met[mask]
        idx = np.argsort(bpp)
        return bpp[idx], met[idx]

    bpp_a, met_a = _clean(bpp_a, met_a)
    bpp_t, met_t = _clean(bpp_t, met_t)

    if len(bpp_a) < 2 or len(bpp_t) < 2:
        return float("nan")

    log_a = np.log(bpp_a)
    log_t = np.log(bpp_t)
    lo = max(log_a.min(), log_t.min())
    hi = min(log_a.max(), log_t.max())
    if hi <= lo:
        return float("nan")

    p_a = np.polyfit(log_a, met_a, min(3, len(bpp_a) - 1))
    p_t = np.polyfit(log_t, met_t, min(3, len(bpp_t) - 1))
    p_a_int = np.polyint(p_a)
    p_t_int = np.polyint(p_t)
    int_a = np.polyval(p_a_int, hi) - np.polyval(p_a_int, lo)
    int_t = np.polyval(p_t_int, hi) - np.polyval(p_t_int, lo)
    return float((int_t - int_a) / (hi - lo))


def plot(input, original, output=None, other=None, input_psnr=None, output_psnr=None, other_psnr=None, save_path=None):
    num_of_subplots = 2 + (output is not None) + (other is not None)
    fig, axs = plt.subplots(1, num_of_subplots, figsize=(4 * num_of_subplots, 4))
    images = [
        (input, "LR Input" + (f" ({input_psnr:.2f}dB)" if input_psnr is not None else "")),
        (original, "HR Original"),
    ]
    if output is not None:
        images.append((output, "SR Output" + (f" ({output_psnr:.2f}dB)" if output_psnr is not None else "")))
    if other is not None:
        images.append((other, "Bicubic" + (f" ({other_psnr:.2f}dB)" if other_psnr is not None else "")))

    for ax, (img, title) in zip(axs, images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=6000)
        ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
