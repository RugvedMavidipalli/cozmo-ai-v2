from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int


class VideoProbeError(ValueError):
    pass


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoProbeError(f"Could not open video file: {path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            # Some codecs don't report dimensions until a frame is decoded.
            ok, frame = capture.read()
            if not ok:
                raise VideoProbeError(f"Video file reports no readable frames: {path}")
            height, width = frame.shape[:2]

        if width <= 0 or height <= 0:
            raise VideoProbeError(f"Video file reports invalid dimensions ({width}x{height}): {path}")

        return VideoInfo(width=width, height=height)
    finally:
        capture.release()
