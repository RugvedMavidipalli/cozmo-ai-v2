from __future__ import annotations

import cv2
import numpy as np
import pytest

VIDEO_WIDTH = 64
VIDEO_HEIGHT = 48

CAMERA_MATRIX = np.array(
    [
        [517.3, 0.0, 318.6],
        [0.0, 516.5, 255.3],
        [0.0, 0.0, 1.0],
    ]
)


def _write_synthetic_video(path, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, n_frames=2, fps=30):
    # 'mp4v' fails to open under macOS's AVFoundation backend (no FFMPEG in
    # opencv-python-headless); 'avc1' (H.264) is what AVFoundation can encode.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "test.mp4"
    _write_synthetic_video(path)
    return path


@pytest.fixture
def stray_scanner_dataset(tmp_path, synthetic_video):
    dataset_dir = tmp_path / "stray_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "rgb.mp4").write_bytes(synthetic_video.read_bytes())
    np.savetxt(dataset_dir / "camera_matrix.csv", CAMERA_MATRIX, delimiter=",")
    return dataset_dir
