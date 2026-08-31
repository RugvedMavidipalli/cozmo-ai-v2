import cv2
import numpy as np
import pytest

from cozmo_ai_v2.depth.fusion import fuse_local_residual

HEIGHT, WIDTH = 120, 160


def _flat_color(value):
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def _sparse_lidar_grid(true_depth, spacing=10, confidence_value=2):
    lidar = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    confidence = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    ys = np.arange(0, HEIGHT, spacing)
    xs = np.arange(0, WIDTH, spacing)
    for y in ys:
        for x in xs:
            lidar[y, x] = true_depth[y, x]
            confidence[y, x] = confidence_value
    return lidar, confidence


def test_correction_matches_truth_at_sample_locations():
    true_depth = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
    corrected_mono = true_depth - 0.5  # uniformly wrong by 0.5m
    lidar, confidence = _sparse_lidar_grid(true_depth, spacing=8)
    color = _flat_color(128)

    result = fuse_local_residual(color, corrected_mono, lidar, confidence, min_confidence=1, max_depth=10.0)

    sample_mask = confidence >= 1
    error_before = np.abs(corrected_mono[sample_mask] - true_depth[sample_mask]).mean()
    error_after = np.abs(result.fused_depth_m[sample_mask] - true_depth[sample_mask]).mean()

    assert error_after < error_before * 0.2
    assert result.covered_fraction > 0.0


def test_falls_back_to_corrected_mono_far_from_samples():
    true_depth = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
    corrected_mono = true_depth - 0.5
    color = _flat_color(128)

    # A single LiDAR sample in the top-left corner only.
    lidar = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    confidence = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    lidar[0, 0] = true_depth[0, 0]
    confidence[0, 0] = 2

    result = fuse_local_residual(color, corrected_mono, lidar, confidence, min_confidence=1, max_depth=10.0, radius=5)

    far_corner = result.fused_depth_m[-1, -1]
    assert far_corner == pytest.approx(corrected_mono[-1, -1], abs=1e-3)
    assert result.covered_fraction < 0.5


def test_edge_awareness_attenuates_across_color_boundary():
    true_depth = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
    corrected_mono = true_depth.copy()
    corrected_mono[:, : WIDTH // 2] -= 0.8  # only the left half is wrong

    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    color[:, : WIDTH // 2] = (0, 0, 0)
    color[:, WIDTH // 2 :] = (255, 255, 255)  # sharp edge at the midline

    lidar, confidence = _sparse_lidar_grid(true_depth, spacing=8)
    # Only keep LiDAR samples on the left (wrong) half.
    lidar[:, WIDTH // 2 :] = 0
    confidence[:, WIDTH // 2 :] = 0

    guided_result = fuse_local_residual(color, corrected_mono, lidar, confidence, min_confidence=1, max_depth=10.0, radius=15)

    # A plain (non-edge-aware) Gaussian normalized convolution of the same
    # sparse residual, for comparison.
    mask = ((confidence >= 1) & (lidar > 0)).astype(np.float32)
    residual = np.zeros_like(corrected_mono)
    residual[mask > 0] = lidar[mask > 0] - corrected_mono[mask > 0]
    ksize = (31, 31)
    gaussian_num = cv2.GaussianBlur(residual * mask, ksize, 0)
    gaussian_den = cv2.GaussianBlur(mask, ksize, 0)
    gaussian_residual = np.divide(gaussian_num, gaussian_den, out=np.zeros_like(gaussian_num), where=gaussian_den > 1e-3)

    right_half = slice(None), slice(WIDTH // 2, None)
    guided_leak = np.abs(guided_result.residual_m[right_half]).mean()
    gaussian_leak = np.abs(gaussian_residual[right_half]).mean()

    assert guided_leak < gaussian_leak
