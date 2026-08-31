import numpy as np
import pytest

from cozmo_ai_v2.depth.align import AlignmentError, fit_scale_shift

RNG = np.random.default_rng(0)


def _grid(h=40, w=40):
    mono = RNG.uniform(1.0, 5.0, size=(h, w)).astype(np.float64)
    confidence = np.full((h, w), 2, dtype=np.uint8)
    return mono, confidence


def test_exact_linear_relationship_recovers_scale_shift():
    mono, confidence = _grid()
    true_scale, true_shift = 2.0, 0.1
    lidar = true_scale * mono + true_shift

    fit = fit_scale_shift(mono, lidar, confidence, min_confidence=1, max_depth=20.0)

    assert fit.scale == pytest.approx(true_scale, abs=1e-6)
    assert fit.shift == pytest.approx(true_shift, abs=1e-6)
    assert fit.used_pixels == mono.size
    assert fit.rms_residual_m == pytest.approx(0.0, abs=1e-6)


def test_low_confidence_pixels_excluded():
    mono, confidence = _grid()
    true_scale, true_shift = 2.0, 0.1
    lidar = true_scale * mono + true_shift

    # Corrupt a block of pixels but mark them low-confidence - they should
    # not affect the fit at all.
    lidar[:10, :10] = 9999.0
    confidence[:10, :10] = 0

    fit = fit_scale_shift(mono, lidar, confidence, min_confidence=1, max_depth=20.0)

    assert fit.scale == pytest.approx(true_scale, abs=1e-6)
    assert fit.shift == pytest.approx(true_shift, abs=1e-6)
    assert fit.used_pixels == mono.size - 100


def test_robust_to_outliers():
    mono, confidence = _grid(h=50, w=50)
    true_scale, true_shift = 1.5, 0.2
    lidar = true_scale * mono + true_shift

    # 35% of high-confidence pixels are wild outliers (e.g. a moving object).
    flat_lidar = lidar.reshape(-1)
    outlier_idx = RNG.choice(flat_lidar.size, size=int(0.35 * flat_lidar.size), replace=False)
    flat_lidar[outlier_idx] = RNG.uniform(50, 100, size=outlier_idx.size)
    lidar = flat_lidar.reshape(lidar.shape)

    fit = fit_scale_shift(mono, lidar, confidence, min_confidence=1, max_depth=200.0)

    assert fit.scale == pytest.approx(true_scale, rel=0.1)
    assert fit.shift == pytest.approx(true_shift, abs=0.2)


def test_too_few_samples_raises():
    mono, confidence = _grid(h=5, w=5)
    lidar = 2.0 * mono

    with pytest.raises(AlignmentError):
        fit_scale_shift(mono, lidar, confidence, min_confidence=1, max_depth=20.0, min_samples=100)


def test_implausible_scale_raises():
    mono, confidence = _grid()
    # A wildly wrong relationship - not linear at all - should not sneak past
    # the plausibility check as a "successful" fit.
    lidar = np.full_like(mono, 0.001)

    with pytest.raises(AlignmentError):
        fit_scale_shift(mono, lidar, confidence, min_confidence=1, max_depth=20.0)


def test_shape_mismatch_raises():
    mono, confidence = _grid(h=10, w=10)
    lidar = np.ones((5, 5))

    with pytest.raises(AlignmentError):
        fit_scale_shift(mono, lidar, confidence[:5, :5], min_confidence=1, max_depth=20.0)
