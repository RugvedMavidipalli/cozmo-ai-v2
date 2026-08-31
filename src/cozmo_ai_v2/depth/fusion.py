from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .align import valid_lidar_mask

MIN_COVERAGE_WEIGHT = 1e-3


@dataclass(frozen=True)
class FusionResult:
    fused_depth_m: np.ndarray
    residual_m: np.ndarray
    covered_fraction: float


def fuse_local_residual(
    color: np.ndarray,
    corrected_mono: np.ndarray,
    lidar_depth: np.ndarray,
    confidence: np.ndarray,
    min_confidence: int = 1,
    max_depth: float = 8.0,
    radius: int = 20,
    eps: float = 100.0,
) -> FusionResult:
    height, width = color.shape[:2]

    lidar_full = cv2.resize(lidar_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    confidence_full = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST)

    valid = valid_lidar_mask(lidar_full, confidence_full, min_confidence, max_depth)
    mask = valid.astype(np.float32)

    sparse_residual = np.zeros_like(corrected_mono, dtype=np.float32)
    sparse_residual[valid] = (lidar_full[valid] - corrected_mono[valid]).astype(np.float32)

    guide = color if color.dtype == np.uint8 else color.astype(np.uint8)
    numerator = cv2.ximgproc.guidedFilter(guide, sparse_residual * mask, radius, eps)
    denominator = cv2.ximgproc.guidedFilter(guide, mask, radius, eps)

    covered = denominator > MIN_COVERAGE_WEIGHT
    dense_residual = np.zeros_like(corrected_mono, dtype=np.float32)
    dense_residual[covered] = numerator[covered] / denominator[covered]

    fused_depth = np.clip(corrected_mono + dense_residual, 0, None)
    covered_fraction = float(covered.mean())

    return FusionResult(
        fused_depth_m=fused_depth,
        residual_m=dense_residual,
        covered_fraction=covered_fraction,
    )
