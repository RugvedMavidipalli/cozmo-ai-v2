from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


class InputKind(Enum):
    STRAY_SCANNER = auto()
    PLAIN_VIDEO = auto()


@dataclass(frozen=True)
class DetectedInput:
    kind: InputKind
    video_path: Path
    camera_matrix_path: Path | None


class InputDetectionError(ValueError):
    pass


def detect_input(path: Path) -> DetectedInput:
    if not path.exists():
        raise InputDetectionError(f"Input path does not exist: {path}")

    if path.is_dir():
        video_path = path / "rgb.mp4"
        camera_matrix_path = path / "camera_matrix.csv"
        has_video = video_path.is_file()
        has_matrix = camera_matrix_path.is_file()
        if has_video and has_matrix:
            return DetectedInput(InputKind.STRAY_SCANNER, video_path, camera_matrix_path)
        if has_video and not has_matrix:
            raise InputDetectionError(
                f"'{path}' looks like a Stray Scanner dataset folder but is missing camera_matrix.csv"
            )
        if has_matrix and not has_video:
            raise InputDetectionError(
                f"'{path}' looks like a Stray Scanner dataset folder but is missing rgb.mp4"
            )
        raise InputDetectionError(
            f"'{path}' is a directory but does not contain rgb.mp4 and camera_matrix.csv "
            "- not a recognized Stray Scanner dataset"
        )

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return DetectedInput(InputKind.PLAIN_VIDEO, path, None)
        raise InputDetectionError(
            f"'{path}' is not a recognized video file (expected one of {sorted(VIDEO_EXTENSIONS)})"
        )

    raise InputDetectionError(f"'{path}' is neither a directory nor a regular file")
