"""Capture ingest: parse a recording directory into a CaptureBundle.

Coordinate conventions
----------------------
Stray Scanner writes odometry as camera-to-world poses whose rotation already
uses the OpenCV camera convention (camera looks down +Z, +Y down in image
space), in a gravity-aligned world frame.  This was established empirically,
not assumed: `tools/conv_test.py` scores every plausible convention by how well it
aligns nearby frames' point clouds, and the identity mapping wins by 6x
(4.2 cm median nearest-neighbour vs 25.9 cm for the flipped alternatives).
Poses are therefore passed through unmodified.

The world frame's up axis is likewise measured rather than assumed --
`_imu_gravity()` here recovers its direction from the accelerometer and
`geometry.estimate_gravity()` refines it against the floor and ceiling,
because a wrong up axis silently produces a floor plan of a wall.

`Frame.pose` is camera-to-world.  Code wanting world-to-camera (Open3D
integration, projection) takes `np.linalg.inv(frame.pose)`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Kept so `tools/conv_test.py` can still score the alternative it rules out.
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# ARKit's confidence raster: 0 = low, 1 = medium, 2 = high.  Low-confidence
# returns are where the sensor gives up: glass, mirrors, dark or wet surfaces,
# grazing incidence.  Those are exactly the surfaces a damaged room is made of,
# so the threshold is a first-class pipeline knob rather than a constant.
CONFIDENCE_LOW = 0
CONFIDENCE_MEDIUM = 1
CONFIDENCE_HIGH = 2

DEPTH_SCALE = 1000.0  # Stray Scanner stores depth as uint16 millimetres.

# The IMU reports in the device body frame, which is rotated from the camera
# raster frame.  Of the candidate mappings, only this one turns the
# accelerometer into a constant world vector: it scores 0.99 directional
# consistency and unit magnitude, versus 0.59-0.87 for the alternatives (see
# `tools/grav_test.py`).  Anything but the true mapping leaves the walking motion
# uncancelled and the mean drops well below 1 g.
DEVICE_TO_CAMERA = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Frame:
    """One synchronised RGB + depth + pose sample."""

    index: int
    timestamp: float
    pose: np.ndarray  # 4x4 camera-to-world, OpenCV convention
    color: np.ndarray  # HxWx3 uint8 RGB, at depth resolution
    depth: np.ndarray  # HxW float32 metres, 0 where invalid
    confidence: np.ndarray  # HxW uint8 in {0,1,2}

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]


@dataclass
class CaptureBundle:
    """A parsed capture, plus everything needed to re-read its frames."""

    root: Path
    name: str
    intrinsics: np.ndarray  # 3x3 for the depth-resolution image
    depth_size: tuple[int, int]  # (width, height)
    timestamps: np.ndarray  # (N,)
    poses: np.ndarray  # (N,4,4) camera-to-world, OpenCV convention
    has_depth: bool
    fps: float
    gravity_up: np.ndarray  # unit world vector opposing gravity, from the IMU
    gravity_consistency: float  # 1.0 when the accelerometer resolves to a constant

    def __len__(self) -> int:
        return len(self.poses)

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


def _read_odometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, poses, rgb_intrinsics_per_frame) from odometry.csv."""
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    intrinsics: list[tuple[float, float, float, float]] = []

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        column = {name: i for i, name in enumerate(header)}
        for row in reader:
            if not row or not row[column["timestamp"]].strip():
                continue
            timestamps.append(float(row[column["timestamp"]]))

            translation = np.array(
                [float(row[column[axis]]) for axis in ("x", "y", "z")]
            )
            # Stray writes the quaternion as (qx, qy, qz, qw).
            qx, qy, qz, qw = (
                float(row[column[key]]) for key in ("qx", "qy", "qz", "qw")
            )
            pose = np.eye(4)
            pose[:3, :3] = _quaternion_to_matrix(qx, qy, qz, qw)
            pose[:3, 3] = translation
            poses.append(pose)

            intrinsics.append(
                tuple(float(row[column[key]]) for key in ("fx", "fy", "cx", "cy"))
            )

    return (
        np.asarray(timestamps),
        np.asarray(poses),
        np.asarray(intrinsics),
    )


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )


def load_capture(root: str | Path) -> CaptureBundle:
    """Parse a Stray Scanner capture directory."""
    root = Path(root)
    odometry_path = root / "odometry.csv"
    if not odometry_path.exists():
        raise FileNotFoundError(f"no odometry.csv in {root}; not a Stray capture")

    timestamps, poses, rgb_intrinsics = _read_odometry(odometry_path)
    depth_dir = root / "depth"
    has_depth = depth_dir.is_dir() and any(depth_dir.glob("*.png"))
    if not has_depth:
        raise FileNotFoundError(
            f"no depth frames in {depth_dir}; use the no-LiDAR path instead"
        )

    probe = cv2.imread(str(sorted(depth_dir.glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
    depth_height, depth_width = probe.shape[:2]

    # Intrinsics in odometry.csv describe the full-resolution RGB frame; the
    # depth raster is a uniform downscale of it, so the intrinsics scale by the
    # same factor.  Median over frames rejects the occasional ARKit outlier.
    rgb_width = _video_width(root / "rgb.mp4")
    scale = depth_width / rgb_width
    fx, fy, cx, cy = np.median(rgb_intrinsics, axis=0) * scale
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    fps = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 30.0
    gravity_up, consistency = _imu_gravity(root / "imu.csv", timestamps, poses)

    return CaptureBundle(
        root=root,
        name=root.name,
        intrinsics=intrinsics,
        depth_size=(depth_width, depth_height),
        timestamps=timestamps,
        poses=poses,
        has_depth=True,
        fps=float(fps),
        gravity_up=gravity_up,
        gravity_consistency=consistency,
    )


def _imu_gravity(
    path: Path, timestamps: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, float]:
    """Recover the world up axis by averaging accelerometer samples.

    A handheld accelerometer reads gravity plus walking acceleration.  Rotated
    into the world frame by each sample's pose, the gravity term is constant
    and the walking term is zero-mean, so the average converges on gravity.
    Returns the unit up axis and a consistency score in [0, 1]; a low score
    means the poses or the device mapping are wrong, not that gravity moved.
    """
    if not path.exists():
        return np.array([0.0, 1.0, 0.0]), 0.0

    samples: list[tuple[float, float, float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row or not row[0].strip():
                continue
            samples.append(tuple(float(value) for value in row[:4]))
    if not samples:
        return np.array([0.0, 1.0, 0.0]), 0.0

    data = np.asarray(samples)
    nearest = np.clip(np.searchsorted(timestamps, data[:, 0]), 0, len(poses) - 1)
    rotations = poses[nearest][:, :3, :3]
    in_camera = data[:, 1:] @ DEVICE_TO_CAMERA.T
    in_world = np.einsum("nij,nj->ni", rotations, in_camera)

    mean = in_world.mean(axis=0)
    up = mean / np.linalg.norm(mean)
    unit = in_world / np.maximum(
        np.linalg.norm(in_world, axis=1, keepdims=True), 1e-9
    )
    return up, float(np.abs(unit @ up).mean())


def _video_width(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    capture.release()
    return width


def iter_frames(
    bundle: CaptureBundle,
    indices: list[int] | np.ndarray | None = None,
    min_confidence: int = CONFIDENCE_MEDIUM,
    max_depth: float = 8.0,
):
    """Yield `Frame`s for `indices` (default: all), in ascending index order.

    The RGB track is decoded sequentially rather than by seeking -- seeking a
    long-GOP H.264 file per frame is orders of magnitude slower, and callers
    always want an ordered subset.
    """
    if indices is None:
        indices = np.arange(len(bundle))
    wanted = sorted(int(i) for i in indices)
    if not wanted:
        return
    wanted_set = set(wanted)
    last = wanted[-1]

    width, height = bundle.depth_size
    capture = cv2.VideoCapture(str(bundle.root / "rgb.mp4"))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {bundle.root / 'rgb.mp4'}")

    try:
        for index in range(last + 1):
            ok, bgr = capture.read()
            if not ok:
                break
            if index not in wanted_set:
                continue

            color = cv2.cvtColor(
                cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            depth_raw = cv2.imread(
                str(bundle.root / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED
            )
            confidence = cv2.imread(
                str(bundle.root / "confidence" / f"{index:06d}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if depth_raw is None:
                continue
            if confidence is None:
                confidence = np.full(depth_raw.shape, CONFIDENCE_HIGH, np.uint8)

            depth = depth_raw.astype(np.float32) / DEPTH_SCALE
            depth[confidence < min_confidence] = 0.0
            depth[depth > max_depth] = 0.0

            yield Frame(
                index=index,
                timestamp=float(bundle.timestamps[index]),
                pose=bundle.poses[index],
                color=color,
                depth=depth,
                confidence=confidence,
            )
    finally:
        capture.release()


def open3d_intrinsics(bundle: CaptureBundle):
    """`bundle.intrinsics` as an Open3D camera model."""
    import open3d as o3d

    width, height = bundle.depth_size
    fx, fy = bundle.intrinsics[0, 0], bundle.intrinsics[1, 1]
    cx, cy = bundle.intrinsics[0, 2], bundle.intrinsics[1, 2]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
