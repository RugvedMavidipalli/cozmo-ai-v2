from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Retained for raw ARKit adapters that need it. Stray Scanner's exported
# odometry.csv is already in OpenCV camera axes and this matrix must not be
# applied to it again.
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# Measured against recordings-2: CSV poses are camera-to-world transforms in
# OpenCV camera axes. The pipeline intentionally uses them without a flip.
STRAY_ODOMETRY_CONVENTION = "camera_to_world_opencv_csv_no_arkit_to_cv_flip"

# ARKit's three depth-confidence levels, from least to most trustworthy.
CONFIDENCE_LOW = 0
CONFIDENCE_MEDIUM = 1
CONFIDENCE_HIGH = 2

# Depth PNGs store distances as whole millimetres; dividing by this
# converts them to metres.
DEPTH_SCALE = 1000.0

# Rotates a raw accelerometer reading from the phone's physical orientation
# into the same axes the camera uses.
DEVICE_TO_CAMERA = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Frame:
    """One moment in time from the capture, with its color image, depth
    image, and camera position all lined up together.

    A raw capture stores color video, depth data, and camera tracking
    separately, each on its own schedule. A `Frame` is what you get after
    picking one instant and pulling together everything that was recorded
    at (or very close to) that same moment, so the color pixels, the depth
    values, and the camera pose all describe the same snapshot in time.

    Attributes:
        index: This frame's position in the capture, counting from 0.
        timestamp: When this frame was captured, in seconds since the
            start of the recording.
        pose: The camera's position and orientation at this moment, as a
            4x4 camera-to-world transform matrix, using OpenCV's
            coordinate convention.
        color: The color image, as an HxWx3 array of RGB values (0-255),
            resized down to match the depth image's resolution.
        depth: The depth image, as an HxW array of distances in metres. A
            value of 0 means no valid depth was measured for that pixel.
        confidence: How much to trust each depth pixel, as an HxW array of
            ARKit's own confidence levels: 0, 1, or 2, for low, medium, or
            high.
        color_full: The color image at its original, full video
            resolution, instead of resized to depth resolution. This is
            `None` unless it was specifically requested, since keeping
            every frame at full resolution uses a lot more memory.
    """

    index: int
    timestamp: float
    pose: np.ndarray
    color: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    color_full: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray:
        """Where the camera was, in world coordinates, when this frame was
        captured."""
        return self.pose[:3, 3]


@dataclass
class FrameAssociation:
    """Association evidence for one successfully decoded video frame."""

    video_index: int
    sidecar_index: int | None
    pts_s: float | None
    sidecar_timestamp_s: float | None
    delta_s: float | None
    method: str

    def to_dict(self) -> dict:
        return {
            "video_index": self.video_index,
            "sidecar_index": self.sidecar_index,
            "pts_s": self.pts_s,
            "sidecar_timestamp_s": self.sidecar_timestamp_s,
            "delta_s": self.delta_s,
            "method": self.method,
        }


@dataclass
class VideoAvailability:
    """Deterministic availability evidence for a sequential video walk.

    The pose/timestamp table defines the sidecar index space. OpenCV's
    reported frame count is useful evidence, but successful sequential
    decodes are the availability decision used by the frame contract. A
    failed read ends the walk; no later sidecar index is ever shifted onto an
    earlier decoded frame.
    """

    expected_frame_count: int
    reported_frame_count: int | None = None
    decoded_frame_count: int = 0
    missing_indices: tuple[int, ...] = ()
    terminal_decode_missing: bool = False
    decode_complete: bool = False
    association_mode: str = "index"
    pts_tolerance_s: float | None = None
    sidecar_timestamps: np.ndarray | None = field(default=None, repr=False, compare=False)
    associations: list[FrameAssociation] = field(default_factory=list)
    _video_pts_origin_s: float | None = field(default=None, repr=False, compare=False)
    _last_pts_s: float | None = field(default=None, repr=False, compare=False)
    _pts_usable: bool | None = field(default=None, repr=False, compare=False)
    _associated_video_indices: set[int] = field(default_factory=set, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.expected_frame_count < 0:
            raise ValueError("expected_frame_count must be non-negative")
        if self.association_mode not in {"index", "pts"}:
            raise ValueError(
                f"association_mode must be 'index' or 'pts', got {self.association_mode!r}"
            )
        if self.sidecar_timestamps is not None:
            timestamps = np.asarray(self.sidecar_timestamps, dtype=np.float64).reshape(-1)
            if (
                len(timestamps) != self.expected_frame_count
                or not np.isfinite(timestamps).all()
                or (len(timestamps) > 1 and np.any(np.diff(timestamps) < 0))
            ):
                self.sidecar_timestamps = None
            else:
                self.sidecar_timestamps = timestamps
                if self.pts_tolerance_s is None and len(timestamps) > 1:
                    intervals = np.diff(timestamps)
                    positive = intervals[intervals > 0]
                    if len(positive):
                        self.pts_tolerance_s = max(float(np.median(positive) * 0.75), 0.05)
        if self.association_mode == "pts" and self.sidecar_timestamps is None:
            self._pts_usable = False
        if self.pts_tolerance_s is not None and self.pts_tolerance_s < 0:
            raise ValueError("pts_tolerance_s must be non-negative")

    def associate(self, video_index: int, pts_ms: float | None) -> FrameAssociation:
        """Associate a decoded video frame to a stable sidecar index.

        OpenCV exposes presentation time through ``CAP_PROP_POS_MSEC`` after
        each successful read. Both clocks are normalised to their first
        sample, so absolute timestamp epochs do not matter. A PTS match must
        be monotonic and one-to-one; otherwise the frame is recorded as
        unmatched rather than shifting a neighbouring sidecar index.
        """
        for association in self.associations:
            if association.video_index == video_index:
                return association

        raw_pts = float(pts_ms) if pts_ms is not None else float("nan")
        pts_s = raw_pts / 1000.0 if np.isfinite(raw_pts) else None
        sidecar_index: int | None = None
        sidecar_timestamp_s: float | None = None
        delta_s: float | None = None
        method = "index"

        timestamps = self.sidecar_timestamps
        if self.association_mode == "pts":
            if pts_s is None:
                self._pts_usable = False
            elif self._pts_usable is not False:
                if self._last_pts_s is not None and pts_s <= self._last_pts_s:
                    # Some codecs expose a constant or non-monotonic
                    # CAP_PROP_POS_MSEC. Use the safe identity mapping for
                    # the complete stream instead of dropping or shifting
                    # every frame after the first bad timestamp.
                    self._pts_usable = False
                else:
                    self._pts_usable = True
                self._last_pts_s = pts_s
        if (
            self.association_mode == "pts"
            and self._pts_usable is not False
            and timestamps is not None
            and pts_s is not None
            and len(timestamps)
        ):
            if self._video_pts_origin_s is None:
                self._video_pts_origin_s = pts_s
            relative_pts = pts_s - self._video_pts_origin_s
            relative_sidecar = timestamps - timestamps[0]
            candidate = int(np.searchsorted(relative_sidecar, relative_pts, side="left"))
            candidates = [candidate]
            if candidate > 0:
                candidates.append(candidate - 1)
            candidate = min(
                (value for value in candidates if 0 <= value < len(relative_sidecar)),
                key=lambda value: abs(float(relative_sidecar[value] - relative_pts)),
                default=-1,
            )
            if candidate >= 0:
                delta_s = abs(float(relative_sidecar[candidate] - relative_pts))
                tolerance = self.pts_tolerance_s
                previous = {
                    association.sidecar_index
                    for association in self.associations
                    if association.sidecar_index is not None
                }
                if (
                    (tolerance is None or delta_s <= tolerance)
                    and candidate not in previous
                    and (not previous or candidate > max(previous))
                ):
                    sidecar_index = candidate
                    sidecar_timestamp_s = float(relative_sidecar[candidate])
                    method = "pts_nearest"
                else:
                    method = "pts_unmatched"

        if sidecar_index is None and method == "index":
            if video_index < self.expected_frame_count:
                sidecar_index = video_index
            method = "index_fallback" if self.association_mode == "pts" else "index"
        elif self.association_mode == "pts" and self._pts_usable is False:
            if video_index < self.expected_frame_count:
                sidecar_index = video_index
            method = "index_fallback"

        association = FrameAssociation(
            video_index=video_index,
            sidecar_index=sidecar_index,
            pts_s=(
                None
                if pts_s is None or self._video_pts_origin_s is None
                else float(pts_s - self._video_pts_origin_s)
            ),
            sidecar_timestamp_s=sidecar_timestamp_s,
            delta_s=delta_s,
            method=method,
        )
        if video_index not in self._associated_video_indices:
            self.associations.append(association)
            self._associated_video_indices.add(video_index)
        return association

    @property
    def sidecar_frame_count(self) -> int:
        """The sidecar count used as the expected frame-index space."""
        return self.expected_frame_count

    @property
    def reported_shortfall(self) -> int:
        """How many reported video frames failed sequential decode."""
        if self.reported_frame_count is None:
            return 0
        return max(self.reported_frame_count - self.decoded_frame_count, 0)

    def to_dict(self) -> dict:
        """Return JSON-compatible availability evidence."""
        return {
            "expected_frame_count": self.expected_frame_count,
            "sidecar_frame_count": self.sidecar_frame_count,
            "reported_frame_count": self.reported_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "missing_indices": list(self.missing_indices),
            "reported_shortfall": self.reported_shortfall,
            "terminal_decode_missing": self.terminal_decode_missing,
            "decode_complete": self.decode_complete,
            "association_mode": self.association_mode,
            "pts_source": (
                "opencv_cap_pos_msec"
                if any(association.pts_s is not None for association in self.associations)
                else None
            ),
            "pts_status": (
                "used"
                if self._pts_usable is True
                else "not_evaluated"
                if self._pts_usable is None and self.association_mode == "pts"
                else "index_fallback"
                if self.association_mode == "pts"
                else "not_requested"
            ),
            "pts_tolerance_s": self.pts_tolerance_s,
            "associations": [association.to_dict() for association in self.associations],
        }


@dataclass
class CaptureBundle:
    """Everything the rest of the pipeline needs to know about one capture,
    plus enough information to go back and re-read its individual frames on
    demand.

    This doesn't hold the actual color and depth images -- those stay on
    disk and get loaded frame by frame through `iter_frames` -- but it does
    hold the camera's tracked path through the room and its optical
    properties, which are needed constantly throughout the rest of the
    pipeline.

    Attributes:
        root: The capture's directory on disk. It's expected to contain
            `odometry.csv`, `rgb.mp4`, a `depth/` folder, a `confidence/`
            folder, and (optionally) an `imu.csv` file.
        name: The capture directory's own name, used as a label when
            printing progress or naming output files.
        intrinsics: The camera's pinhole intrinsics -- the focal length
            and optical center (fx, fy, cx, cy) packed into a 3x3 matrix
            -- scaled to match the resolution given by `depth_size`.
        depth_size: The `(width, height)`, in pixels, of the depth and
            confidence images.
        timestamps: When each frame was captured, in seconds, as an (N,)
            array.
        poses: The camera's position and orientation at each frame, as an
            (N, 4, 4) array of camera-to-world transform matrices, using
            OpenCV's coordinate convention.
        has_depth: `True` when raw LiDAR depth is present. It remains
            `False` for a video-only capture driven by a precomputed dense
            artifact, so uncertainty reporting can widen its intervals.
        fps: The capture's effective frame rate, in frames per second.
        gravity_up: A unit vector, in world coordinates, pointing away
            from the floor -- the pipeline's best guess at "up" for this
            capture.
        gravity_consistency: How well the phone's accelerometer readings
            agree with each other on a single up direction, from 0 (no
            agreement at all) to 1 (perfect agreement). A low value can be
            a sign that the capture involved a lot of shaking or unusual
            device motion.
        pose_convention: Provenance for the input poses. Stray Scanner CSV
            poses are camera-to-world transforms in OpenCV camera axes and
            are intentionally used without an ARKit-to-OpenCV flip.
    """

    root: Path
    name: str
    intrinsics: np.ndarray
    depth_size: tuple[int, int]
    timestamps: np.ndarray
    poses: np.ndarray
    has_depth: bool
    fps: float
    gravity_up: np.ndarray
    gravity_consistency: float
    # The original RGB calibration is retained alongside the depth-sized
    # matrix.  Stage 4 produces depth at native RGB resolution, so consumers
    # must not try to reuse ``intrinsics`` without scaling it back up.
    rgb_size: tuple[int, int] | None = None
    rgb_intrinsics: np.ndarray | None = None
    pose_source: str = "arkit"
    pose_path: str | None = None
    pose_convention: str = STRAY_ODOMETRY_CONVENTION

    def __len__(self) -> int:
        """How many frames are in this capture."""
        return len(self.poses)

    @property
    def duration(self) -> float:
        """How long the capture lasted, in seconds. Returns 0.0 if there
        are fewer than two timestamps to measure a span between."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def intrinsics_for_size(self, width: int, height: int) -> np.ndarray:
        """Return pinhole intrinsics scaled to ``(width, height)``.

        ``rgb_intrinsics`` is the source calibration and is measured at
        ``rgb_size``.  Keeping the scaling here makes the resolution contract
        explicit for both raw LiDAR and full-resolution dense depth.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size ({width}x{height})")

        if self.rgb_intrinsics is not None and self.rgb_size is not None:
            source_width, source_height = self.rgb_size
            base = self.rgb_intrinsics
        else:
            source_width, source_height = self.depth_size
            base = self.intrinsics

        if source_width <= 0 or source_height <= 0:
            raise ValueError("capture has no valid calibration image size")
        sx = width / source_width
        sy = height / source_height
        scaled = np.asarray(base, dtype=np.float64).copy()
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] *= sx
        scaled[1, 2] *= sy
        return scaled


def _read_odometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the camera's tracked path and per-frame lens settings out of a
    Stray Scanner `odometry.csv` file.

    Each row in this file represents one moment during the capture: when
    it happened, where the camera was and how it was oriented, and what
    the camera's lens settings were at that instant.

    Args:
        path: Path to a Stray Scanner `odometry.csv` file.

    Returns:
        A tuple of `(timestamps, poses, intrinsics)`. `timestamps` is an
        (N,) array of capture times, in seconds. `poses` is an (N, 4, 4)
        array of camera-to-world transform matrices, using OpenCV's
        coordinate convention. `intrinsics` is an (N, 4) array of
        per-frame `(fx, fy, cx, cy)` lens values, measured at `rgb.mp4`'s
        native resolution. Rows with a blank timestamp are skipped
        entirely.
    """
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    intrinsics: list[tuple[float, float, float, float]] = []

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = [column.strip() for column in next(reader)]
        # Look columns up by name rather than by fixed position, so this
        # doesn't break if the CSV's column order ever changes.
        column = {name: i for i, name in enumerate(header)}
        for row in reader:
            if not row or not row[column["timestamp"]].strip():
                continue
            timestamps.append(float(row[column["timestamp"]]))

            translation = np.array(
                [float(row[column[axis]]) for axis in ("x", "y", "z")]
            )
            qx, qy, qz, qw = (
                float(row[column[key]]) for key in ("qx", "qy", "qz", "qw")
            )
            # Combine the rotation (stored as a quaternion) and the
            # translation into one 4x4 camera-to-world transform matrix.
            pose = np.eye(4)
            pose[:3, :3] = _quaternion_to_matrix(qx, qy, qz, qw)
            pose[:3, 3] = translation
            poses.append(pose)

            intrinsics.append(
                tuple(float(row[column[key]]) for key in ("fx", "fy", "cx", "cy"))
            )

    pose_array = np.asarray(poses)
    _validate_camera_to_world_poses(pose_array, path)
    return np.asarray(timestamps), pose_array, np.asarray(intrinsics)


def _validate_camera_to_world_poses(poses: np.ndarray, source: Path) -> None:
    """Structurally validate expected C2W transforms without guessing a flip."""

    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"{source} contains no valid 4x4 camera-to-world poses")
    if not np.isfinite(poses).all() or not np.allclose(
        poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6
    ):
        raise ValueError(f"{source} contains invalid homogeneous camera-to-world poses")
    rotations = poses[:, :3, :3]
    orthogonality = np.einsum("nji,njk->nik", rotations, rotations)
    if not np.allclose(orthogonality, np.eye(3), atol=1e-4) or np.any(
        np.linalg.det(rotations) <= 0.0
    ):
        raise ValueError(f"{source} contains invalid camera rotations; no convention flip was inferred")


def load_odometry_pose_priors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamped Stray Scanner/ARKit camera-to-world pose priors.

    The returned transforms are already in the pipeline's measured OpenCV
    camera convention. Callers must not apply ``ARKIT_TO_CV`` again: that
    conversion is for a different raw ARKit convention and would corrupt
    Stray Scanner's exported odometry.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ARKit pose-prior file not found: {path}")
    timestamps, poses, _intrinsics = _read_odometry(path)
    return timestamps, poses


def _nearest_timestamp_indices(timestamps: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Choose the nearest timestamp, preferring the earlier sample on ties."""

    timestamps = np.asarray(timestamps, dtype=float)
    queries = np.asarray(queries, dtype=float)
    if timestamps.ndim != 1 or len(timestamps) == 0 or np.any(np.diff(timestamps) < 0):
        raise ValueError("pose timestamps must be a non-empty sorted array")
    right = np.clip(np.searchsorted(timestamps, queries, side="left"), 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_right = np.abs(timestamps[right] - queries) < np.abs(queries - timestamps[left])
    return np.where(choose_right, right, left)


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Converts a quaternion -- a compact, four-number way of representing
    a 3D rotation -- into an ordinary 3x3 rotation matrix.

    Args:
        qx: The quaternion's x component.
        qy: The quaternion's y component.
        qz: The quaternion's z component.
        qw: The quaternion's w (scalar) component.

    Returns:
        The equivalent 3x3 rotation matrix. If the quaternion is exactly
        zero, which shouldn't normally happen, this returns the identity
        matrix (no rotation at all) instead of dividing by zero.
    """
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


def load_capture(
    root: str | Path,
    *,
    pose_source: str = "auto",
    slam_poses_path: str | Path | None = None,
    dense_depth_dir: str | Path | None = None,
) -> CaptureBundle:
    """Reads a Stray Scanner capture directory from disk and builds a
    `CaptureBundle` describing it.

    This is normally the very first thing that happens in the pipeline: it
    checks that the expected files are present, works out the camera's
    lens settings and the depth image's resolution, and recovers the up
    direction from the phone's motion sensors, all before any of the
    actual frame processing starts.

    Args:
        root: The capture directory. A normal Stray Scanner capture contains
            `odometry.csv`, a non-empty `depth/` directory, and `rgb.mp4`.
            With `dense_depth_dir`, a precomputed Stage 4 raster may replace
            the raw depth directory. `pose_source="slam"` selects an offline
            SLAM pose table and falls back to ARKit when available.

    Returns:
        A populated `CaptureBundle`. `has_depth` indicates whether raw
        LiDAR is present; a dense-only bundle is marked video-only.

    Raises:
        FileNotFoundError: `root` has no usable pose table, calibration,
            video, or raw/dense depth size.
    """
    root = Path(root)
    odometry_path = root / "odometry.csv"
    slam_path = _find_slam_poses(root, slam_poses_path)
    if pose_source not in {"auto", "arkit", "slam"}:
        raise ValueError(f"pose_source must be 'auto', 'arkit', or 'slam', got {pose_source!r}")
    selected_pose_source = "arkit"
    if pose_source == "slam" or (pose_source == "auto" and not odometry_path.exists()):
        if slam_path is not None:
            try:
                return _load_slam_capture(root, slam_path, dense_depth_dir)
            except (OSError, ValueError) as exc:
                if not odometry_path.exists():
                    raise FileNotFoundError(f"could not load SLAM poses from {slam_path}: {exc}") from exc
                selected_pose_source = "arkit_fallback"
        if pose_source == "slam":
            # Explicit SLAM selection may still use ARKit as a documented
            # fallback when a capture contains it, rather than silently
            # inventing identity poses.
            if odometry_path.exists():
                selected_pose_source = "arkit_fallback"
            else:
                raise FileNotFoundError(f"no SLAM pose table found for {root}")
    if not odometry_path.exists():
        raise FileNotFoundError(f"no odometry.csv in {root}; not a Stray capture")

    timestamps, poses, rgb_intrinsics = _read_odometry(odometry_path)
    depth_dir = root / "depth"
    has_depth = depth_dir.is_dir() and any(depth_dir.glob("*.png"))
    if not has_depth:
        dense_dir = Path(dense_depth_dir) if dense_depth_dir is not None else root / "dense_depth"
        if dense_dir is not None and dense_dir.is_dir() and not any(dense_dir.glob("*.png")) and (dense_dir / "dense_depth").is_dir():
            dense_dir = dense_dir / "dense_depth"
        dense_size = _dense_probe(dense_dir) if dense_dir is not None and dense_dir.is_dir() else None
        if dense_size is None:
            raise FileNotFoundError(
                f"no depth frames in {depth_dir}; use the no-LiDAR path instead"
            )
        # A precomputed, QC-gated Stage 4 artifact is enough to ingest a
        # capture with no raw sensor raster.  ``has_depth`` remains false so
        # uncertainty reporting still identifies this as video-only.
        depth_height, depth_width = dense_size[1], dense_size[0]
    else:
        # Peek at one depth PNG just to learn its resolution -- the depth
        # images are usually a lower resolution than the color video.
        probe = cv2.imread(str(sorted(depth_dir.glob("*.png"))[0]), cv2.IMREAD_UNCHANGED)
        depth_height, depth_width = probe.shape[:2]

    # The per-frame intrinsics from odometry.csv are given at the color
    # video's native resolution, so they need to be rescaled to match the
    # depth image's (usually smaller) resolution.
    rgb_width, rgb_height = _video_size(root / "rgb.mp4")
    rgb_fx, rgb_fy, rgb_cx, rgb_cy = np.median(rgb_intrinsics, axis=0)
    rgb_matrix = np.array(
        [[rgb_fx, 0.0, rgb_cx], [0.0, rgb_fy, rgb_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale_x = depth_width / rgb_width
    scale_y = depth_height / rgb_height
    depth_matrix = rgb_matrix.copy()
    depth_matrix[0, 0] *= scale_x
    depth_matrix[1, 1] *= scale_y
    depth_matrix[0, 2] *= scale_x
    depth_matrix[1, 2] *= scale_y

    # Fall back to a reasonable default frame rate when there's only one
    # timestamp to work with, since there's no time span to divide by.
    fps = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 30.0
    gravity_up, consistency = _imu_gravity(root / "imu.csv", timestamps, poses)

    return CaptureBundle(
        root=root,
        name=root.name,
        intrinsics=depth_matrix,
        depth_size=(depth_width, depth_height),
        timestamps=timestamps,
        poses=poses,
        has_depth=has_depth,
        fps=float(fps),
        gravity_up=gravity_up,
        gravity_consistency=consistency,
        rgb_size=(int(rgb_width), int(rgb_height)),
        rgb_intrinsics=rgb_matrix,
        pose_source=selected_pose_source,
        pose_path=str(odometry_path),
    )


def _find_slam_poses(root: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"SLAM pose table does not exist: {path}")
        return path
    for candidate in (
        root / "slam_poses.csv",
        root / "slam_poses.json",
        root / "slam_poses.npy",
        root / "slam_poses.npz",
        root / "poses.csv",
        root / "poses.json",
        root / "poses.npy",
        root / "poses.npz",
        root / "slam" / "poses.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _read_slam_poses(path: Path, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Read common offline SLAM pose exports as camera-to-world matrices."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        poses = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        timestamps = np.arange(len(poses), dtype=np.float64) / fps
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((key for key in ("poses", "pose", "trajectory") if key in archive), None)
            if key is None:
                raise ValueError(f"{path} has no poses/trajectory array")
            poses = np.asarray(archive[key], dtype=np.float64)
            time_key = next((key for key in ("timestamps", "times", "timestamp") if key in archive), None)
            timestamps = (
                np.asarray(archive[time_key], dtype=np.float64)
                if time_key is not None
                else np.arange(len(poses), dtype=np.float64) / fps
            )
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            raw_poses = payload.get("poses", payload.get("trajectory"))
            raw_times = payload.get("timestamps", payload.get("times"))
        else:
            raw_poses, raw_times = payload, None
        if raw_poses is None:
            raise ValueError(f"{path} has no poses/trajectory field")
        poses = np.asarray(raw_poses, dtype=np.float64)
        timestamps = (
            np.asarray(raw_times, dtype=np.float64)
            if raw_times is not None
            else np.arange(len(poses), dtype=np.float64) / fps
        )
    else:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [{str(k).strip().lower(): value.strip() for k, value in row.items() if k} for row in reader]
        if not rows:
            raise ValueError(f"SLAM pose table is empty: {path}")
        index_key = next((key for key in ("index", "frame", "frame_index", "frame_id") if key in rows[0]), None)
        if index_key is not None:
            try:
                rows.sort(key=lambda row: int(row[index_key]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"SLAM frame indices are not integers in {path}") from exc
        poses_list: list[np.ndarray] = []
        timestamps_list: list[float] = []
        for position, row in enumerate(rows):
            poses_list.append(_pose_from_mapping(row, path))
            timestamp_value = row.get("timestamp", row.get("time", row.get("t")))
            timestamps_list.append(float(timestamp_value) if timestamp_value else position / fps)
        poses = np.asarray(poses_list, dtype=np.float64)
        timestamps = np.asarray(timestamps_list, dtype=np.float64)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"SLAM poses must have shape (N, 4, 4), got {poses.shape}")
    if len(poses) == 0 or len(timestamps) != len(poses):
        raise ValueError(f"SLAM pose/timestamp count mismatch in {path}")
    if not np.isfinite(poses).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"SLAM pose table contains non-finite values: {path}")
    return timestamps, poses


def _pose_from_mapping(row: dict[str, str], path: Path) -> np.ndarray:
    def value(*names: str) -> float | None:
        for name in names:
            raw = row.get(name)
            if raw not in (None, ""):
                return float(raw)
        return None

    matrix = np.eye(4, dtype=np.float64)
    matrix_names = [f"m{r}{c}" for r in range(4) for c in range(4)]
    if all(name in row and row[name] != "" for name in matrix_names):
        return np.asarray([float(row[name]) for name in matrix_names], dtype=np.float64).reshape(4, 4)
    translation = [value(axis, f"t{axis}") for axis in ("x", "y", "z")]
    quaternion = [value(f"q{axis}") for axis in ("x", "y", "z")] + [value("qw", "w")]
    if any(item is None for item in translation + quaternion):
        raise ValueError(f"SLAM row in {path} has neither a 4x4 matrix nor x/y/z/qx/qy/qz/qw fields")
    matrix[:3, :3] = _quaternion_to_matrix(*[float(item) for item in quaternion])
    matrix[:3, 3] = np.asarray([float(item) for item in translation])
    return matrix


def _dense_probe(path: Path) -> tuple[int, int] | None:
    candidates = (
        sorted(path.glob("*.png"))
        + sorted(path.glob("*.npy"))
        + sorted(path.glob("*.npz"))
    )
    if not candidates:
        return None
    candidate = candidates[0]
    if candidate.suffix.lower() == ".npy":
        shape = np.load(candidate, mmap_mode="r", allow_pickle=False).shape
    elif candidate.suffix.lower() == ".npz":
        with np.load(candidate, allow_pickle=False) as archive:
            shape = archive[archive.files[0]].shape if archive.files else ()
    else:
        raster = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
        shape = raster.shape if raster is not None else ()
    if len(shape) != 2:
        raise ValueError(f"dense depth probe must be a single-channel raster: {candidate}")
    return int(shape[1]), int(shape[0])


def _read_intrinsics_sidecar(root: Path, rgb_size: tuple[int, int], pose_path: Path) -> np.ndarray:
    """Read calibration for a video/SLAM capture without ARKit odometry."""
    candidates = [root / "camera_matrix.csv", root / "intrinsics.yaml", root / "intrinsics.json"]
    candidates.extend([pose_path.parent / "intrinsics.yaml", pose_path.parent / "intrinsics.json"])
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            matrix = np.loadtxt(path, delimiter=",")
            if matrix.shape != (3, 3):
                raise ValueError(f"expected a 3x3 matrix in {path}, got {matrix.shape}")
            return np.asarray(matrix, dtype=np.float64)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
        else:
            import yaml
            payload = yaml.safe_load(path.read_text())
        calibration = payload.get("calibration") if isinstance(payload, dict) else None
        if calibration is not None and len(calibration) == 4:
            fx, fy, cx, cy = (float(value) for value in calibration)
            source_width = int(payload.get("width", rgb_size[0]))
            source_height = int(payload.get("height", rgb_size[1]))
            matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            matrix[0, [0, 2]] *= rgb_size[0] / source_width
            matrix[1, [1, 2]] *= rgb_size[1] / source_height
            return matrix
    raise FileNotFoundError(
        f"no camera calibration found for SLAM capture {root}; expected camera_matrix.csv or intrinsics.yaml"
    )


def _load_slam_capture(
    root: Path,
    pose_path: Path,
    dense_depth_dir: str | Path | None,
) -> CaptureBundle:
    rgb_width, rgb_height = _video_size(root / "rgb.mp4")
    fps = 30.0
    capture = cv2.VideoCapture(str(root / "rgb.mp4"))
    if capture.isOpened() and capture.get(cv2.CAP_PROP_FPS) > 0:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    timestamps, poses = _read_slam_poses(pose_path, fps)
    rgb_matrix = _read_intrinsics_sidecar(root, (int(rgb_width), int(rgb_height)), pose_path)

    raw_depth_dir = root / "depth"
    raw_depth = _dense_probe(raw_depth_dir) if raw_depth_dir.is_dir() else None
    dense_dir = Path(dense_depth_dir) if dense_depth_dir is not None else root / "dense_depth"
    if dense_dir.is_dir() and not any(dense_dir.glob("*.png")) and (dense_dir / "dense_depth").is_dir():
        dense_dir = dense_dir / "dense_depth"
    dense_depth = _dense_probe(dense_dir) if dense_dir.is_dir() else None
    depth_size = raw_depth or dense_depth or (int(rgb_width), int(rgb_height))
    depth_matrix = rgb_matrix.copy()
    depth_matrix[0, [0, 2]] *= depth_size[0] / rgb_width
    depth_matrix[1, [1, 2]] *= depth_size[1] / rgb_height

    gravity_up, consistency = _imu_gravity(root / "imu.csv", timestamps, poses) if (root / "imu.csv").exists() else (
        np.array([0.0, 1.0, 0.0]), 0.0
    )
    return CaptureBundle(
        root=root,
        name=root.name,
        intrinsics=depth_matrix,
        depth_size=depth_size,
        timestamps=timestamps,
        poses=poses,
        has_depth=raw_depth is not None,
        fps=fps,
        gravity_up=gravity_up,
        gravity_consistency=consistency,
        rgb_size=(int(rgb_width), int(rgb_height)),
        rgb_intrinsics=rgb_matrix,
        pose_source="slam",
        pose_path=str(pose_path),
    )


def _imu_gravity(
    path: Path, timestamps: np.ndarray, poses: np.ndarray
) -> tuple[np.ndarray, float]:
    """Figures out which direction is "up" in the world by averaging the
    phone's accelerometer readings over the whole capture.

    When a phone is roughly level, gravity mostly shows up as a steady
    pull along one axis of the accelerometer. Averaging many samples over
    the capture washes out the noise caused by the phone's own movement,
    leaving a decent estimate of which way is actually down -- and up is
    just the opposite of that.

    Args:
        path: Path to the capture's `imu.csv` file. If this file doesn't
            exist, this function falls back to assuming world +Y is up,
            and reports zero consistency to signal that the answer is
            only a guess.
        timestamps: Each frame's capture time, in seconds, as an (N,)
            array.
        poses: Each frame's camera-to-world pose, as an (N, 4, 4) array,
            indexed the same way as `timestamps`.

    Returns:
        A tuple of `(up, consistency)`. `up` is a unit vector, in world
        coordinates, pointing away from the floor. `consistency` is how
        closely the individual accelerometer samples agree with that
        average direction, from 0 (no agreement) to 1 (perfect
        agreement).
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
    # Match each accelerometer sample to the genuinely nearest frame pose,
    # rather than always taking the future pose returned by searchsorted.
    nearest = _nearest_timestamp_indices(timestamps, data[:, 0])
    rotations = poses[nearest][:, :3, :3]
    # Accelerometer readings come in the phone's own physical orientation,
    # not the camera's, so rotate them into camera space first.
    in_camera = data[:, 1:] @ DEVICE_TO_CAMERA.T
    in_world = np.einsum("nij,nj->ni", rotations, in_camera)

    mean = in_world.mean(axis=0)
    up = mean / np.linalg.norm(mean)
    unit = in_world / np.maximum(
        np.linalg.norm(in_world, axis=1, keepdims=True), 1e-9
    )
    # `np.abs` here because a sample pointing the "wrong way" (e.g.
    # opposite the average) still agrees with the overall direction, just
    # with a flipped sign, so it shouldn't count against consistency.
    return up, float(np.abs(unit @ up).mean())


def _video_size(path: Path) -> tuple[float, float]:
    """Opens a video file just long enough to read off its native frame
    width, in pixels.

    Args:
        path: Path to a video file (normally `rgb.mp4`).

    Returns:
        The video's `(width, height)`, in pixels.

    Raises:
        FileNotFoundError: `path` couldn't be opened as a video.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    if width <= 0 or height <= 0:
        raise FileNotFoundError(f"video {path} reports invalid dimensions")
    return width, height


def _video_width(path: Path) -> float:
    """Backward-compatible width-only wrapper used by older callers."""
    return _video_size(path)[0]


def iter_raw_frames(
    root: Path,
    indices: list[int] | np.ndarray | None = None,
    availability: VideoAvailability | None = None,
):
    """Walks a capture's colour video alongside its per-frame depth and
    confidence images, handing back whatever came off disk without
    interpreting any of it.

    This is the one place that knows how a Stray Scanner capture is laid
    out on disk -- that colour lives in a single video which can only be
    read forwards from the start, while depth and confidence are numbered
    files beside it. Callers that want metres, filtering, or resizing
    build that on top; see `iter_frames`.

    Args:
        root: The capture directory.
        indices: Which frame numbers to yield; if `None`, every frame in
            the video is yielded.
        availability: Optional mutable record to populate with the reported
            video count, successful decode count, and terminal sidecar gaps.
            When supplied, the walk drains the video after the last requested
            frame so short videos are detected even for strided requests.

    Yields:
        Tuples of `(index, bgr, depth_raw, confidence)`, in ascending
        index order. `bgr` is the raw colour frame at the video's own
        resolution. `depth_raw` and `confidence` are `None` when that
        frame has no matching file on disk, so callers can decide for
        themselves whether that is worth skipping over.

    Raises:
        FileNotFoundError: `root` has no readable `rgb.mp4`.
    """
    wanted_set: set[int] | None = None
    last: int | None = None
    if indices is not None:
        wanted = sorted(int(i) for i in indices)
        if not wanted:
            if availability is None:
                return
            wanted_set = set()
            last = -1
        else:
            wanted_set = set(wanted)
            last = wanted[-1]
        # PTS can associate a video position with a sidecar index that is not
        # numerically identical. Decode the complete stream in that mode so
        # the requested sidecar index cannot be missed by an index cutoff.
        if availability is not None and availability.association_mode == "pts":
            last = None

    video = cv2.VideoCapture(str(root / "rgb.mp4"))
    if not video.isOpened():
        raise FileNotFoundError(f"cannot open {root / 'rgb.mp4'}")

    try:
        if availability is not None:
            reported = video.get(cv2.CAP_PROP_FRAME_COUNT)
            if np.isfinite(reported) and reported > 0:
                availability.reported_frame_count = int(round(reported))

        index = 0
        # Stop as soon as the last requested frame is behind us, rather
        # than decoding the rest of the video for nothing.
        while last is None or index <= last:
            ok, bgr = video.read()
            if not ok:
                break
            sidecar_index = index
            association = None
            if availability is not None:
                association = availability.associate(
                    index, video.get(cv2.CAP_PROP_POS_MSEC)
                )
                sidecar_index = association.sidecar_index
            if sidecar_index is not None and (
                wanted_set is None or sidecar_index in wanted_set
            ):
                yield (
                    sidecar_index,
                    bgr,
                    cv2.imread(
                        str(root / "depth" / f"{sidecar_index:06d}.png"),
                        cv2.IMREAD_UNCHANGED,
                    ),
                    cv2.imread(
                        str(root / "confidence" / f"{sidecar_index:06d}.png"),
                        cv2.IMREAD_UNCHANGED,
                    ),
                )
            index += 1

        # A strided consumer normally stops after its final requested index.
        # Drain only when evidence was requested, so the report still records
        # a terminal decoder shortfall that lies beyond that stride.
        if availability is not None and not availability.decode_complete:
            while True:
                ok, _ = video.read()
                if not ok:
                    break
                availability.associate(index, video.get(cv2.CAP_PROP_POS_MSEC))
                index += 1

            availability.decoded_frame_count = index
            associated = {
                association.sidecar_index
                for association in availability.associations
                if association.sidecar_index is not None
            }
            availability.missing_indices = tuple(
                sidecar_index
                for sidecar_index in range(availability.expected_frame_count)
                if sidecar_index not in associated
            )
            availability.terminal_decode_missing = bool(
                availability.reported_shortfall
                or (
                    availability.expected_frame_count > 0
                    and (
                        not associated
                        or max(associated) < availability.expected_frame_count - 1
                    )
                )
            )
            availability.decode_complete = True
    finally:
        video.release()


def iter_frames(
    bundle: CaptureBundle,
    indices: list[int] | np.ndarray | None = None,
    min_confidence: int = CONFIDENCE_MEDIUM,
    max_depth: float = 8.0,
    include_full_res: bool = False,
):
    """Reads through a capture's video and depth files and yields fully
    assembled `Frame` objects for the requested indices.

    Color frames live in a single video file that has to be read starting
    from the beginning, while depth and confidence images are separate
    files on disk. To avoid re-opening and re-scanning the video for every
    single frame, this walks through the video once, in order, and only
    keeps the frames that were actually asked for.

    Args:
        bundle: The parsed capture to read frames from.
        indices: Which frame indices to yield; if `None`, every frame in
            the capture is yielded.
        min_confidence: The lowest depth-confidence level to keep. Any
            pixel below this gets its depth zeroed out.
        max_depth: The furthest depth value to include, in metres.
            Anything beyond this gets zeroed out too.
        include_full_res: If `True`, also populate `Frame.color_full` with
            the image at its original video resolution, not just the
            smaller depth resolution.

    Yields:
        `Frame` objects for the requested indices, in ascending index
        order. An index with no matching depth PNG on disk is silently
        skipped rather than raising an error.
    """
    if indices is None:
        indices = np.arange(len(bundle))

    width, height = bundle.depth_size
    for index, bgr, depth_raw, confidence in iter_raw_frames(bundle.root, indices):
        if depth_raw is None:
            continue
        if confidence is None:
            # No confidence file for this frame -- treat every pixel
            # as fully trusted rather than dropping the frame.
            confidence = np.full(depth_raw.shape, CONFIDENCE_HIGH, np.uint8)

        color = cv2.cvtColor(
            cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        color_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if include_full_res else None

        depth = depth_raw.astype(np.float32) / DEPTH_SCALE
        # A depth of 0 is this pipeline's way of marking a pixel as
        # invalid, so anything too low-confidence or too far away
        # simply gets zeroed out here rather than removed outright.
        depth[confidence < min_confidence] = 0.0
        depth[depth > max_depth] = 0.0

        yield Frame(
            index=index,
            timestamp=float(bundle.timestamps[index]),
            pose=bundle.poses[index],
            color=color,
            depth=depth,
            confidence=confidence,
            color_full=color_full,
        )


def open3d_intrinsics(bundle: CaptureBundle):
    """Repackages `bundle.intrinsics` into the camera model object that the
    Open3D library expects.

    Args:
        bundle: The parsed capture whose intrinsics should be converted.

    Returns:
        An `o3d.camera.PinholeCameraIntrinsic` built from the bundle's
        image width, height, and lens intrinsics.
    """
    import open3d as o3d

    width, height = bundle.depth_size
    fx, fy = bundle.intrinsics[0, 0], bundle.intrinsics[1, 1]
    cx, cy = bundle.intrinsics[0, 2], bundle.intrinsics[1, 2]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


_INVERSE_ROTATION = {
    cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE,
    cv2.ROTATE_180: cv2.ROTATE_180,
    None: None,
}


def inverse_rotation(rotation: int | None) -> int | None:
    """Looks up the `cv2.ROTATE_*` code that would undo a given rotation --
    in other words, the code that rotates an image back to how it looked
    before `rotation` was applied to it.

    Args:
        rotation: A `cv2.ROTATE_*` code, or `None` for no rotation.

    Returns:
        The rotation code that reverses `rotation`.
    """
    return _INVERSE_ROTATION[rotation]


def display_rotation(pose: np.ndarray, gravity_up: np.ndarray) -> int | None:
    """Works out how a frame needs to be rotated so that "up" in the real
    world points toward the top of the image, the way a person would
    expect to see it.

    A phone can be held in any orientation while recording -- portrait,
    landscape, or even upside down -- and the raw video frames come out
    rotated to match however the phone happened to be held at that
    moment. This figures out, for one specific frame, which one of the
    four 90-degree-multiple rotations would straighten that frame back
    out.

    Args:
        pose: This frame's 4x4 camera-to-world pose.
        gravity_up: The capture's up direction, as a unit vector in world
            coordinates.

    Returns:
        A `cv2.ROTATE_*` code to apply to the raw frame so that world-up
        points toward the top of the image, or `None` if it already does.
    """
    # Project world-up into this frame's own camera axes, then check
    # whether it points more sideways or more up/down within the image to
    # decide which of the four rotations is needed.
    camera_up = pose[:3, :3].T @ gravity_up
    dx, dy = float(camera_up[0]), float(camera_up[1])
    if abs(dx) >= abs(dy):
        return cv2.ROTATE_90_COUNTERCLOCKWISE if dx > 0 else cv2.ROTATE_90_CLOCKWISE
    return None if dy < 0 else cv2.ROTATE_180


def rotate_bbox(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
    rotation: int | None,
) -> tuple[float, float, float, float]:
    """Moves a bounding box's coordinates to match an image that's been
    rotated with one of the `cv2.ROTATE_*` codes.

    This is needed whenever a detection (like a damage bounding box) was
    found on a rotated version of a frame, but needs to be reported in the
    original, unrotated image's coordinates, or vice versa.

    Args:
        bbox: The box as `(x0, y0, x1, y1)`, with `x0 <= x1` and
            `y0 <= y1`, measured in the pre-rotation image.
        width: The width, in pixels, of the pre-rotation image.
        height: The height, in pixels, of the pre-rotation image.
        rotation: A `cv2.ROTATE_*` code (as returned by
            `display_rotation`), or `None` for no rotation.

    Returns:
        The box's `(x0, y0, x1, y1)` coordinates after the rotation has
        been applied.

    Raises:
        ValueError: `rotation` isn't one of the four supported codes.
    """
    x0, y0, x1, y1 = bbox
    if rotation is None:
        return (x0, y0, x1, y1)
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return (height - y1, x0, height - y0, x1)
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (y0, width - x1, y1, width - x0)
    if rotation == cv2.ROTATE_180:
        return (width - x1, height - y1, width - x0, height - y0)
    raise ValueError(f"unsupported rotation code: {rotation}")
