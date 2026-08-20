"""Shared geometric primitives: gravity recovery and plane algebra.

The up axis is *measured*, never assumed.  A capture whose world frame is
mislabelled produces a floor plan of a wall, silently and confidently, so the
recovery here reports a quality score that the caller can gate on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GravityEstimate:
    up: np.ndarray  # unit vector, world frame, pointing away from the floor
    floor_height: float  # signed distance along `up` of the floor plane
    ceiling_height: float | None  # None when no ceiling was observed
    inlier_fraction: float  # share of points on the two horizontal slabs

    @property
    def room_height(self) -> float | None:
        if self.ceiling_height is None:
            return None
        return self.ceiling_height - self.floor_height


def estimate_gravity(
    points: np.ndarray,
    hint: np.ndarray,
    normals: np.ndarray | None = None,
    cone_degrees: float = 25.0,
) -> GravityEstimate:
    """Recover the up axis and floor/ceiling heights from a reconstruction.

    `hint` is the accelerometer-derived up axis, which is unambiguous about
    *direction* but noisy by a degree or two.  A degree of tilt smears wall
    footprints by over a centimetre across a storey, which the 2 cm wall gate
    cannot absorb, so the hint is refined against the floor and ceiling
    normals: only normals within `cone_degrees` of the hint are kept (this is
    what rejects the walls that swamp a naive dominant-direction fit), and
    their mean becomes the up axis.
    """
    up = hint / np.linalg.norm(hint)
    if normals is not None and len(normals):
        up = _refine_up(normals, up, cone_degrees)

    heights = points @ up
    floor, ceiling, inliers = _floor_and_ceiling(heights)
    return GravityEstimate(
        up=up,
        floor_height=floor,
        ceiling_height=ceiling,
        inlier_fraction=inliers,
    )


def _refine_up(
    normals: np.ndarray, hint: np.ndarray, cone_degrees: float
) -> np.ndarray:
    """Average the horizontal-surface normals that agree with `hint`.

    Floor normals point up and ceiling normals point down, so both are folded
    onto the hint's hemisphere before averaging.
    """
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    alignment = unit @ hint
    threshold = np.cos(np.radians(cone_degrees))
    selected = unit[np.abs(alignment) > threshold]
    if len(selected) < 100:
        return hint
    folded = selected * np.sign(selected @ hint)[:, None]
    refined = folded.mean(axis=0)
    return refined / np.linalg.norm(refined)


def _floor_and_ceiling(
    heights: np.ndarray, bin_size: float = 0.02
) -> tuple[float, float | None, float]:
    bins = max(16, int(np.ptp(heights) / bin_size))
    counts, edges = np.histogram(heights, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    strong = counts > 0.25 * counts.max()
    if not strong.any():
        return float(np.min(heights)), None, 0.0
    indices = np.flatnonzero(strong)

    floor = float(centers[indices[0]])
    ceiling = float(centers[indices[-1]])
    if ceiling - floor < 1.8:  # too short to be a storey: no ceiling seen
        ceiling_value = None
        inliers = counts[indices[0]] / counts.sum()
    else:
        ceiling_value = ceiling
        near_floor = np.abs(heights - floor) < 0.1
        near_ceiling = np.abs(heights - ceiling) < 0.1
        inliers = float((near_floor | near_ceiling).mean())
    return floor, ceiling_value, float(inliers)


def plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through `points`, as a unit normal and offset.

    The plane is `normal . x = offset`.  Uses the smallest-eigenvalue direction
    of the centred scatter matrix (total least squares), which -- unlike a
    height-over-XY regression -- stays well conditioned for vertical walls.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, float(normal @ centroid)


def point_plane_distance(
    points: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    return points @ normal - offset
