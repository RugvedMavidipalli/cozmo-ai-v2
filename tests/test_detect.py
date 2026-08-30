import pytest

from cozmo_ai_v2.detect import InputDetectionError, InputKind, detect_input


def test_stray_scanner_folder_detected(stray_scanner_dataset):
    detected = detect_input(stray_scanner_dataset)

    assert detected.kind is InputKind.STRAY_SCANNER
    assert detected.video_path == stray_scanner_dataset / "rgb.mp4"
    assert detected.camera_matrix_path == stray_scanner_dataset / "camera_matrix.csv"


def test_missing_camera_matrix_raises(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "rgb.mp4").write_bytes(b"fake")

    with pytest.raises(InputDetectionError, match="camera_matrix.csv"):
        detect_input(dataset_dir)


def test_missing_rgb_video_raises(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "camera_matrix.csv").write_text("1,0,0\n0,1,0\n0,0,1\n")

    with pytest.raises(InputDetectionError, match="rgb.mp4"):
        detect_input(dataset_dir)


def test_unrelated_folder_raises(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "notes.txt").write_text("hello")

    with pytest.raises(InputDetectionError):
        detect_input(dataset_dir)


def test_plain_video_file_detected(synthetic_video):
    detected = detect_input(synthetic_video)

    assert detected.kind is InputKind.PLAIN_VIDEO
    assert detected.video_path == synthetic_video
    assert detected.camera_matrix_path is None


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a video")

    with pytest.raises(InputDetectionError):
        detect_input(path)


def test_nonexistent_path_raises(tmp_path):
    with pytest.raises(InputDetectionError):
        detect_input(tmp_path / "does-not-exist")
