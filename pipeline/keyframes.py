"""Keyframe selection for the damage pass.

Reconstruction wants frames that overlap; damage analysis wants the opposite --
few frames, sharp, and each showing something the others do not.  Every frame
sent costs an API call, so the selection is a coverage problem: pick the
sharpest frame from each distinct viewpoint and stop.
"""

from __future__ import annotations

import cv2
import numpy as np

from .ingest import CaptureBundle, iter_frames


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian -- low on motion blur, high on crisp detail."""
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_damage_keyframes(
    bundle: CaptureBundle,
    poses: np.ndarray | None = None,
    max_frames: int = 40,
    position_step: float = 0.6,
    rotation_step_degrees: float = 25.0,
    sharpness_window: int = 7,
) -> list[int]:
    """Sharp frames covering distinct viewpoints, capped at `max_frames`.

    Viewpoints are binned by position and heading, then the sharpest frame in
    each bin is scored; the highest-scoring bins fill the budget.  Blur is
    rejected within a bin rather than globally, so a dim room still contributes
    its best available frame instead of being crowded out by a bright one.
    """
    pose_table = bundle.poses if poses is None else poses
    positions = pose_table[:, :3, 3]
    headings = pose_table[:, :3, 2]

    bins: dict[tuple, list[int]] = {}
    angular_bin = np.radians(rotation_step_degrees)
    for index in range(len(pose_table)):
        cell = tuple(np.round(positions[index] / position_step).astype(int))
        yaw = np.arctan2(headings[index][0], headings[index][2])
        bins.setdefault(cell + (int(yaw / angular_bin),), []).append(index)

    candidates = [_middle(members) for members in bins.values()]
    candidates.sort()
    if not candidates:
        return []

    scores = _sharpness_for(bundle, candidates, sharpness_window)
    ranked = sorted(candidates, key=lambda i: -scores.get(i, 0.0))
    return sorted(ranked[:max_frames])


def _middle(members: list[int]) -> int:
    return members[len(members) // 2]


def _sharpness_for(
    bundle: CaptureBundle, candidates: list[int], window: int
) -> dict[int, float]:
    """Best sharpness within a few frames of each candidate.

    Searching a small neighbourhood costs one extra decode per candidate and
    routinely rescues a viewpoint whose centre frame happened to catch a
    camera shake.
    """
    wanted: dict[int, list[int]] = {}
    for candidate in candidates:
        for offset in range(-(window // 2), window // 2 + 1):
            index = candidate + offset
            if 0 <= index < len(bundle):
                wanted.setdefault(index, []).append(candidate)

    best: dict[int, float] = {}
    chosen: dict[int, int] = {}
    for frame in iter_frames(bundle, sorted(wanted), min_confidence=0):
        value = sharpness(frame.color)
        for candidate in wanted[frame.index]:
            if value > best.get(candidate, -1.0):
                best[candidate] = value
                chosen[candidate] = frame.index
    return {chosen[c]: v for c, v in best.items() if c in chosen}
