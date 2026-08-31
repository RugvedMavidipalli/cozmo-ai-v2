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


DEPTH_WIDTH = 16
DEPTH_HEIGHT = 12

# Depth in mm for each of the 2 synthetic video frames (a simple ramp across
# columns, so there's real spatial structure to fit/fuse against).
FRAME_DEPTHS_MM = [
    np.tile(np.linspace(1800, 2200, DEPTH_WIDTH, dtype=np.uint16), (DEPTH_HEIGHT, 1)),
    np.tile(np.linspace(2300, 2700, DEPTH_WIDTH, dtype=np.uint16), (DEPTH_HEIGHT, 1)),
]


@pytest.fixture
def lidar_stray_scanner_dataset(stray_scanner_dataset):
    depth_dir = stray_scanner_dataset / "depth"
    confidence_dir = stray_scanner_dataset / "confidence"
    depth_dir.mkdir()
    confidence_dir.mkdir()

    for index, depth_mm in enumerate(FRAME_DEPTHS_MM):
        cv2.imwrite(str(depth_dir / f"{index:06d}.png"), depth_mm)
        confidence = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), 2, dtype=np.uint8)
        cv2.imwrite(str(confidence_dir / f"{index:06d}.png"), confidence)

    return stray_scanner_dataset


@pytest.fixture
def stray_capture(lidar_stray_scanner_dataset):
    """A complete Stray Scanner capture: the LiDAR fixture plus the
    odometry and IMU files that `pipeline.ingest.load_capture` requires."""
    root = lidar_stray_scanner_dataset
    fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
    cx, cy = CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]

    lines = [
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,"
        " distortion_center_x, distortion_center_y"
    ]
    for i in range(len(FRAME_DEPTHS_MM)):
        # Identity rotation, camera translating along +x one metre per frame.
        lines.append(
            f"{100.0 + i / 30.0},{i:06d},{float(i)},0.0,0.0,"
            f"0.0,0.0,0.0,1.0,{fx},{fy},{cx},{cy},,"
        )
    (root / "odometry.csv").write_text("\n".join(lines) + "\n")

    # Constant accelerometer reading: gravity along device -x, sampled
    # densely enough that _imu_gravity has something to average.
    imu = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    for i in range(20):
        imu.append(f"{100.0 + i / 100.0},-1.0,0.0,0.0,0.0,0.0,0.0")
    (root / "imu.csv").write_text("\n".join(imu) + "\n")

    return root
