from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import siegelslopes

MIN_PLAUSIBLE_SCALE = 0.1
MAX_PLAUSIBLE_SCALE = 10.0


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class ScaleShiftFit:
    scale: float
    shift: float
    used_pixels: int
    rms_residual_m: float


def valid_lidar_mask(
    lidar_depth: np.ndarray,
    confidence: np.ndarray,
    min_confidence: int,
    max_depth: float,
) -> np.ndarray:
    return (confidence >= min_confidence) & (lidar_depth > 0) & (lidar_depth <= max_depth)


def fit_scale_shift(
    mono_depth: np.ndarray,
    lidar_depth: np.ndarray,
    confidence: np.ndarray,
    min_confidence: int = 1,
    max_depth: float = 8.0,
    min_samples: int = 100,
) -> ScaleShiftFit:
    if mono_depth.shape != lidar_depth.shape or mono_depth.shape != confidence.shape:
        raise AlignmentError(
            f"mono_depth {mono_depth.shape}, lidar_depth {lidar_depth.shape}, and confidence "
            f"{confidence.shape} must all have the same shape"
        )

    mask = valid_lidar_mask(lidar_depth, confidence, min_confidence, max_depth) & (mono_depth > 0)
    used_pixels = int(mask.sum())
    if used_pixels < min_samples:
        raise AlignmentError(
            f"Only {used_pixels} valid pixels for scale/shift fit, need at least {min_samples}"
        )

    x = mono_depth[mask].astype(np.float64)
    y = lidar_depth[mask].astype(np.float64)
    scale, shift = siegelslopes(y, x)

    if not (MIN_PLAUSIBLE_SCALE <= scale <= MAX_PLAUSIBLE_SCALE):
        raise AlignmentError(
            f"Fitted scale {scale:.4f} is outside the plausible range "
            f"[{MIN_PLAUSIBLE_SCALE}, {MAX_PLAUSIBLE_SCALE}] - fit is unreliable"
        )

    residuals = y - (scale * x + shift)
    rms_residual_m = float(np.sqrt(np.mean(residuals**2)))

    return ScaleShiftFit(
        scale=float(scale),
        shift=float(shift),
        used_pixels=used_pixels,
        rms_residual_m=rms_residual_m,
    )
