import pytest

from cozmo_ai_v2.depth.capture import LidarCaptureError, iter_capture_frames, require_lidar_capture


def test_require_lidar_capture_accepts_complete_dataset(lidar_stray_scanner_dataset):
    capture = require_lidar_capture(lidar_stray_scanner_dataset)

    assert capture.video_path == lidar_stray_scanner_dataset / "rgb.mp4"
    assert capture.camera_matrix_path == lidar_stray_scanner_dataset / "camera_matrix.csv"
    assert capture.depth_dir == lidar_stray_scanner_dataset / "depth"
    assert capture.confidence_dir == lidar_stray_scanner_dataset / "confidence"


def test_require_lidar_capture_rejects_missing_depth(stray_scanner_dataset):
    with pytest.raises(LidarCaptureError, match="depth"):
        require_lidar_capture(stray_scanner_dataset)


def test_require_lidar_capture_rejects_missing_confidence(lidar_stray_scanner_dataset):
    import shutil

    shutil.rmtree(lidar_stray_scanner_dataset / "confidence")

    with pytest.raises(LidarCaptureError, match="confidence"):
        require_lidar_capture(lidar_stray_scanner_dataset)


def test_iter_capture_frames_yields_correct_data(lidar_stray_scanner_dataset):
    capture = require_lidar_capture(lidar_stray_scanner_dataset)

    frames = list(iter_capture_frames(capture))

    assert len(frames) == 2
    assert frames[0].index == 0
    assert frames[1].index == 1
    assert frames[0].color.shape == (48, 64, 3)
    assert frames[0].depth_m.shape == (12, 16)
    assert frames[0].confidence.shape == (12, 16)
    assert frames[0].depth_m.min() > 1.0  # sanity: meters, not millimeters
    assert (frames[0].confidence == 2).all()


def test_iter_capture_frames_skips_missing_depth_file(lidar_stray_scanner_dataset):
    (lidar_stray_scanner_dataset / "depth" / "000001.png").unlink()

    capture = require_lidar_capture(lidar_stray_scanner_dataset)
    frames = list(iter_capture_frames(capture))

    assert [f.index for f in frames] == [0]
