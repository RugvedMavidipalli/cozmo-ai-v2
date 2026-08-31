"""Coverage for the capture reader, which every later stage sits on top of.

`iter_frames` and `iter_raw_frames` share one walk over the capture
directory, so these pin the behaviour each caller depends on: the
reconstruction path wants colour at depth resolution with low-confidence
samples zeroed, densification wants the raw samples untouched.
"""

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.ingest import (
    CONFIDENCE_HIGH,
    iter_frames,
    iter_raw_frames,
    load_capture,
)

VIDEO_W, VIDEO_H = 64, 48
DEPTH_W, DEPTH_H = 16, 12


def test_load_capture_reads_metadata(stray_capture):
    bundle = load_capture(stray_capture)

    assert len(bundle) == 2
    assert bundle.depth_size == (DEPTH_W, DEPTH_H)
    assert bundle.has_depth
    assert bundle.poses.shape == (2, 4, 4)
    # Camera translates one metre per frame along +x.
    np.testing.assert_allclose(bundle.poses[1][:3, 3], [1.0, 0.0, 0.0])


def test_load_capture_scales_intrinsics_to_depth_resolution(stray_capture):
    bundle = load_capture(stray_capture)

    # camera_matrix.csv is calibrated at video resolution; the bundle's
    # intrinsics must describe the depth image instead.
    scale = DEPTH_W / VIDEO_W
    assert bundle.intrinsics[0, 0] == pytest.approx(517.3 * scale)
    assert bundle.intrinsics[1, 1] == pytest.approx(516.5 * scale)
    assert bundle.intrinsics[0, 2] == pytest.approx(318.6 * scale)
    assert bundle.intrinsics[1, 2] == pytest.approx(255.3 * scale)


def test_load_capture_rejects_non_capture(tmp_path):
    with pytest.raises(FileNotFoundError, match="odometry.csv"):
        load_capture(tmp_path)


def test_load_capture_rejects_capture_without_depth(stray_capture):
    for png in (stray_capture / "depth").glob("*.png"):
        png.unlink()

    with pytest.raises(FileNotFoundError, match="depth"):
        load_capture(stray_capture)


def test_iter_frames_resizes_colour_and_converts_depth(stray_capture):
    bundle = load_capture(stray_capture)

    frames = list(iter_frames(bundle))

    assert [f.index for f in frames] == [0, 1]
    for frame in frames:
        # Colour comes back at depth resolution so the two line up pixelwise.
        assert frame.color.shape == (DEPTH_H, DEPTH_W, 3)
        assert frame.depth.shape == (DEPTH_H, DEPTH_W)
        assert frame.color_full is None
    # Depth arrives in millimetres on disk and must leave in metres.
    assert frames[0].depth.max() == pytest.approx(2.2, abs=1e-3)


def test_iter_frames_can_keep_full_resolution_colour(stray_capture):
    bundle = load_capture(stray_capture)

    frame = next(iter_frames(bundle, [0], include_full_res=True))

    assert frame.color.shape == (DEPTH_H, DEPTH_W, 3)
    assert frame.color_full.shape == (VIDEO_H, VIDEO_W, 3)


def test_iter_frames_zeroes_low_confidence_samples(stray_capture):
    import cv2

    # Drop the whole first frame to the lowest confidence level.
    cv2.imwrite(
        str(stray_capture / "confidence" / "000000.png"),
        np.zeros((DEPTH_H, DEPTH_W), np.uint8),
    )
    bundle = load_capture(stray_capture)

    frame = next(iter_frames(bundle, [0], min_confidence=1))

    # Zero is this pipeline's marker for "no usable measurement here".
    assert (frame.depth == 0).all()


def test_iter_frames_zeroes_samples_beyond_max_depth(stray_capture):
    bundle = load_capture(stray_capture)

    frame = next(iter_frames(bundle, [0], max_depth=1.9))

    assert frame.depth.max() <= 1.9
    assert (frame.depth == 0).any()


def test_iter_frames_skips_frames_with_no_depth_file(stray_capture):
    (stray_capture / "depth" / "000001.png").unlink()
    bundle = load_capture(stray_capture)

    assert [f.index for f in iter_frames(bundle)] == [0]


def test_iter_frames_trusts_frames_missing_a_confidence_file(stray_capture):
    (stray_capture / "confidence" / "000000.png").unlink()
    bundle = load_capture(stray_capture)

    frame = next(iter_frames(bundle, [0], min_confidence=CONFIDENCE_HIGH))

    # No confidence file means "assume trustworthy", not "discard".
    assert (frame.confidence == CONFIDENCE_HIGH).all()
    assert frame.depth.max() > 0


def test_iter_raw_frames_reports_missing_images_as_none(stray_capture):
    (stray_capture / "depth" / "000001.png").unlink()

    raw = {i: (d, c) for i, _bgr, d, c in iter_raw_frames(stray_capture)}

    assert raw[0][0] is not None
    assert raw[1][0] is None      # callers decide whether to skip
    assert raw[1][1] is not None


def test_iter_raw_frames_honours_requested_indices(stray_capture):
    assert [i for i, *_ in iter_raw_frames(stray_capture, [1])] == [1]
    assert [i for i, *_ in iter_raw_frames(stray_capture, [])] == []


def test_iter_raw_frames_rejects_unreadable_video(tmp_path):
    (tmp_path / "rgb.mp4").write_bytes(b"not a video")

    with pytest.raises(FileNotFoundError):
        list(iter_raw_frames(tmp_path))
