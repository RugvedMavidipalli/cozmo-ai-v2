"""Faithful SceneCAD density preparation for optional RoomFormer inference.

RoomFormer is trained on a 256-square top-down count-density image.  This
module keeps that preprocessing separate from the optional RoomFormer runtime
so the geometry pipeline can validate the handoff without importing Torch or
the external repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoomFormerDensity:
    """A SceneCAD-compatible density image and its plan-space transform."""

    image: np.ndarray
    minimum: np.ndarray
    span: float
    grid_size: int


def scenecad_density_from_plan(
    points: np.ndarray,
    *,
    grid_size: int = 256,
    padding_fraction: float = 0.05,
) -> RoomFormerDensity:
    """Project plan-space points with the official SceneCAD normalization.

    The original RoomFormer SceneCAD preprocessing uses a square extent, adds
    five percent padding on every side, bins per-cell point counts, and divides
    by the maximum count.  It neither log-scales the counts nor flips the image.
    """

    plan = np.asarray(points, dtype=np.float64)
    if plan.ndim != 2 or plan.shape[1] != 2:
        raise ValueError("RoomFormer plan points must have shape (N, 2)")
    plan = plan[np.isfinite(plan).all(axis=1)]
    if not len(plan):
        raise ValueError("RoomFormer density requires at least one finite plan point")
    if grid_size < 2:
        raise ValueError(f"grid_size must be at least 2, got {grid_size}")
    if not np.isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError(f"padding_fraction must be finite and non-negative, got {padding_fraction}")

    minimum = plan.min(axis=0)
    maximum = plan.max(axis=0)
    base_span = float(np.max(maximum - minimum))
    if not np.isfinite(base_span) or base_span <= 0:
        raise ValueError("RoomFormer plan points must span a non-zero horizontal extent")

    minimum = (maximum + minimum) / 2.0 - base_span / 2.0
    padding = base_span * float(padding_fraction)
    minimum -= padding
    span = base_span + 2.0 * padding

    pixels = plan_to_roomformer_pixels(plan, minimum, span, grid_size)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    density[unique[:, 1], unique[:, 0]] = counts
    density /= float(density.max())
    image = np.rint(density * 255.0).astype(np.uint8)
    return RoomFormerDensity(image=image, minimum=minimum, span=span, grid_size=grid_size)


def plan_to_roomformer_pixels(
    points: np.ndarray,
    minimum: np.ndarray,
    span: float,
    grid_size: int,
) -> np.ndarray:
    """Map plan coordinates to the same rounded, clipped RoomFormer pixels."""

    plan = np.asarray(points, dtype=np.float64)
    if plan.shape[-1] != 2:
        raise ValueError("RoomFormer plan coordinates must end in two values")
    origin = np.asarray(minimum, dtype=np.float64).reshape(2)
    if not np.isfinite(plan).all() or not np.isfinite(origin).all():
        raise ValueError("RoomFormer plan coordinates must be finite")
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"RoomFormer span must be finite and positive, got {span}")
    if grid_size < 2:
        raise ValueError(f"grid_size must be at least 2, got {grid_size}")
    pixels = np.rint((plan - origin) / float(span) * grid_size).astype(np.int32)
    return np.clip(pixels, 0, grid_size - 1)
