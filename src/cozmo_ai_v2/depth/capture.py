from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..detect import InputDetectionError, InputKind, detect_input
from ..pipeline.ingest import VideoAvailability, iter_raw_frames

DEPTH_SCALE_MM_TO_M = 1000.0


class LidarCaptureError(ValueError):
    pass


@dataclass(frozen=True)
class LidarCaptureInput:
    root: Path
    video_path: Path
    camera_matrix_path: Path
    depth_dir: Path
    confidence_dir: Path
    sidecar_timestamps: np.ndarray | None = None


@dataclass(frozen=True)
class CaptureFrame:
    index: int
    color: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray


def require_lidar_capture(path: Path) -> LidarCaptureInput:
    try:
        detected = detect_input(path)
    except InputDetectionError as exc:
        raise LidarCaptureError(str(exc)) from exc

    if detected.kind is not InputKind.STRAY_SCANNER:
        raise LidarCaptureError(
            f"'{path}' is not a Stray Scanner dataset - LiDAR depth fusion requires real LiDAR data"
        )

    depth_dir = path / "depth"
    confidence_dir = path / "confidence"

    if not depth_dir.is_dir() or not any(depth_dir.glob("*.png")):
        raise LidarCaptureError(f"'{path}' has no depth frames in {depth_dir}")
    if not confidence_dir.is_dir() or not any(confidence_dir.glob("*.png")):
        raise LidarCaptureError(f"'{path}' has no confidence frames in {confidence_dir}")

    return LidarCaptureInput(
        root=path,
        video_path=detected.video_path,
        camera_matrix_path=detected.camera_matrix_path,
        depth_dir=depth_dir,
        confidence_dir=confidence_dir,
        sidecar_timestamps=_read_sidecar_timestamps(path / "odometry.csv"),
    )


def _read_sidecar_timestamps(path: Path) -> np.ndarray | None:
    """Read optional odometry timestamps for PTS-aware Stage 4 decoding."""
    if not path.exists():
        return None
    timestamps: list[float] = []
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("timestamp")
                if value not in (None, ""):
                    timestamps.append(float(value))
    except (OSError, TypeError, ValueError, csv.Error):
        # Stage 4 can still use deterministic index pairing when the
        # optional odometry clock is absent or malformed.
        return None
    return np.asarray(timestamps, dtype=np.float64) if timestamps else None


def _sidecar_frame_count(capture: LidarCaptureInput) -> int:
    """Return the highest contiguous sidecar index space we can audit."""
    indices = []
    for directory in (capture.depth_dir, capture.confidence_dir):
        for path in directory.glob("*.png"):
            try:
                indices.append(int(path.stem))
            except ValueError:
                continue
    if capture.sidecar_timestamps is not None:
        indices.append(len(capture.sidecar_timestamps) - 1)
    return max(indices, default=-1) + 1


def iter_capture_frames(
    capture: LidarCaptureInput,
    indices: list[int] | np.ndarray | None = None,
    availability: VideoAvailability | None = None,
) -> Iterator[CaptureFrame]:
    """Yields a capture's frames with colour at full video resolution and
    depth in metres, exactly as the sensor recorded them.

    Nothing is filtered or resized here: densification needs the raw LiDAR
    samples to anchor against, and the full-resolution colour to guide the
    fusion. Contrast `pipeline.ingest.iter_frames`, which resizes colour
    down to depth resolution and zeroes out low-confidence samples for the
    reconstruction path.

    Args:
        capture: The validated capture to read from.
        indices: Which frame numbers to yield; if `None`, every frame.
        availability: Optional record for PTS/decode-count evidence. When
            supplied, the underlying video walk is drained after selection.

    Yields:
        `CaptureFrame`s in ascending index order. A frame missing either
        its depth or its confidence image is skipped rather than reported,
        since there is nothing to anchor against without both.

    Raises:
        LidarCaptureError: the capture's video could not be opened.
    """
    try:
        for index, bgr, depth_raw, confidence in iter_raw_frames(
            capture.root, indices, availability=availability
        ):
            if depth_raw is None or confidence is None:
                continue
            yield CaptureFrame(
                index=index,
                color=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                depth_m=depth_raw.astype(np.float32) / DEPTH_SCALE_MM_TO_M,
                confidence=confidence,
            )
    except FileNotFoundError as exc:
        raise LidarCaptureError(str(exc)) from exc
