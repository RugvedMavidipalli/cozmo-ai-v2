from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..detect import InputDetectionError, InputKind, detect_input

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
    )


def iter_capture_frames(
    capture: LidarCaptureInput,
    indices: list[int] | np.ndarray | None = None,
) -> Iterator[CaptureFrame]:
    video = cv2.VideoCapture(str(capture.video_path))
    if not video.isOpened():
        raise LidarCaptureError(f"Could not open video file: {capture.video_path}")

    try:
        wanted = None if indices is None else sorted(int(i) for i in indices)
        wanted_set = None if wanted is None else set(wanted)
        last = None if wanted is None else (wanted[-1] if wanted else -1)

        index = 0
        while last is None or index <= last:
            ok, bgr = video.read()
            if not ok:
                break
            if wanted_set is None or index in wanted_set:
                depth_path = capture.depth_dir / f"{index:06d}.png"
                confidence_path = capture.confidence_dir / f"{index:06d}.png"
                depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
                if depth_raw is not None and confidence is not None:
                    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    depth_m = depth_raw.astype(np.float32) / DEPTH_SCALE_MM_TO_M
                    yield CaptureFrame(index=index, color=color, depth_m=depth_m, confidence=confidence)
            index += 1
    finally:
        video.release()
