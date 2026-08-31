from __future__ import annotations

import cv2
import numpy as np

from .ingest import CaptureBundle, iter_frames


def sharpness(image: np.ndarray) -> float:
    """Estimates how sharp (as opposed to blurry) an image is.

    This works by converting the image to greyscale and then measuring
    how much the brightness jumps around from pixel to pixel, using
    something called a Laplacian -- a standard tool for spotting edges.
    A sharp, crisp image is full of sudden brightness changes at every
    edge and piece of fine detail, while a blurry, motion-smeared image
    is much smoother, with barely any sudden changes anywhere. So how
    much that "jumpiness" varies across the whole image turns out to be
    a simple, reliable stand-in for how in-focus the picture actually
    is.

    Args:
        image: An RGB image array of shape (H, W, 3), or an already
            greyscale image of shape (H, W).

    Returns:
        The variance of the image's Laplacian -- a single number that's
        higher for sharper images and lower for blurrier ones.
    """
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
    """Picks a batch of frames to send to the damage-detection model,
    aiming for good coverage of the space and the sharpest image
    available at each spot.

    Sending every single frame in a capture to the model would be slow
    and expensive, and most neighbouring frames look almost identical
    anyway, so this instead groups frames by roughly where the camera
    was standing and which way it was facing -- like sorting photos into
    "the ones taken from this corner of this room, looking that way".
    From each group, it picks whichever nearby frame is the sharpest,
    since a blurry frame is much less useful for spotting damage. If
    there are still more candidate frames than the budget allows, only
    the sharpest-looking groups make the final cut.

    Args:
        bundle: The parsed capture; supplies the frame count and, when
            `poses` is not given, the camera-to-world poses too.
        poses: Camera-to-world poses to use instead of `bundle.poses`, if
            given.
        max_frames: The most keyframes this function is allowed to
            return.
        position_step: How far the camera has to move, in metres, along
            each world axis, before it counts as a new viewpoint group.
        rotation_step_degrees: How far the camera has to turn, in
            degrees, before it counts as a new viewpoint group.
        sharpness_window: How many frames around each group's
            representative frame to check when looking for the sharpest
            actual frame nearby.

    Returns:
        Frame indices in ascending order, at most `max_frames` of them,
        one per selected viewpoint.
    """
    pose_table = bundle.poses if poses is None else poses
    positions = pose_table[:, :3, 3]
    headings = pose_table[:, :3, 2]

    bins: dict[tuple, list[int]] = {}
    angular_bin = np.radians(rotation_step_degrees)
    for index in range(len(pose_table)):
        # Round each frame's position and heading down onto a coarse
        # grid, so frames taken from roughly the same spot, facing
        # roughly the same way, land in the same group.
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
    """Picks the middle frame index out of one viewpoint group's sorted
    list of frame indices, as a simple, cheap representative for that
    group.

    Args:
        members: The frame indices assigned to one (position, heading)
            group, already in ascending order.

    Returns:
        The middle index from the list.
    """
    return members[len(members) // 2]


def _sharpness_for(
    bundle: CaptureBundle, candidates: list[int], window: int
) -> dict[int, float]:
    """For each candidate frame, finds the sharpest actual frame within a
    small neighbourhood around it.

    The frame `_middle` picked to represent a viewpoint group is just a
    convenient stand-in, not necessarily the sharpest frame near that
    viewpoint -- so this looks a few frames to either side of it and
    keeps whichever one actually scores best on `sharpness`.

    Args:
        bundle: The parsed capture; supplies frame decoding through
            `iter_frames`.
        candidates: The representative frame index for each viewpoint
            group.
        window: How many frames wide the neighbourhood to search around
            each candidate should be, in total.

    Returns:
        A dictionary mapping the actual sharpest frame index found near
        each candidate to its sharpness score. A candidate contributes no
        entry at all if none of the frames in its neighbourhood could be
        decoded.
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
            # Keep whichever frame near this candidate has scored highest
            # on sharpness so far.
            if value > best.get(candidate, -1.0):
                best[candidate] = value
                chosen[candidate] = frame.index
    return {chosen[c]: v for c, v in best.items() if c in chosen}
