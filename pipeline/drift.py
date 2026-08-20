"""Measure trajectory drift by how far a surface moves between visits.

The tempting metric -- RMS scatter of points about a fitted wall -- mostly
measures depth-sensor noise, and a plane fit averages that noise down by
sqrt(N).  It is nearly blind to drift, because drift displaces a whole visit's
points coherently: the fit lands between the visits and the scatter barely
changes.

What actually breaks the 2 cm budget is that coherent displacement.  So the
metric here groups a wall's supporting points by *when* they were observed and
reports the spread of the per-visit plane offsets.  That number is the drift
contribution to wall position, it is directly comparable against the gate, and
it is what a pose-graph correction is supposed to reduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import CaptureBundle, iter_frames
from .planes import HorizontalFrame, WallSegment


@dataclass
class WallVisit:
    wall_index: int
    visit_count: int
    offsets: np.ndarray  # per-visit fitted offset along the wall normal
    times: np.ndarray  # mean observation time of each visit
    point_counts: np.ndarray

    @property
    def spread(self) -> float:
        """Peak-to-peak disagreement between visits, in metres."""
        return float(np.ptp(self.offsets)) if len(self.offsets) > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.offsets)) if len(self.offsets) > 1 else 0.0


@dataclass
class DriftMeasurement:
    walls_examined: int
    revisited_walls: int
    median_spread: float
    p90_spread: float
    max_spread: float
    per_wall: list[WallVisit]

    def summary(self) -> str:
        return (
            f"{self.revisited_walls}/{self.walls_examined} walls revisited; "
            f"median {self.median_spread * 1000:.1f} mm, "
            f"p90 {self.p90_spread * 1000:.1f} mm, "
            f"max {self.max_spread * 1000:.1f} mm"
        )


def _sample_provenance(
    bundle: CaptureBundle,
    indices: np.ndarray,
    poses: np.ndarray | None,
    stride: int,
    min_confidence: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared back-projection loop: world points, their camera origins, and times.

    Origin is the camera position each point was observed from -- one extra
    array costs nothing extra to compute (the pose is already read to place
    the point) and is what lets `planes.filter_occluded_walls` ask "did the
    ray from here to this point have to pass through solid wall".
    """
    pose_table = bundle.poses if poses is None else poses
    intrinsics = bundle.intrinsics
    point_chunks: list[np.ndarray] = []
    origin_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []

    for frame in iter_frames(
        bundle, indices, min_confidence=min_confidence, max_depth=max_depth
    ):
        depth = frame.depth[::stride, ::stride]
        valid = depth > 0
        if not valid.any():
            continue
        vs, us = np.nonzero(valid)
        z = depth[valid]
        camera_points = np.stack(
            [
                (us * stride - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (vs * stride - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ],
            axis=1,
        )
        pose = pose_table[frame.index]
        world = camera_points @ pose[:3, :3].T + pose[:3, 3]
        point_chunks.append(world)
        origin_chunks.append(np.broadcast_to(pose[:3, 3], world.shape))
        time_chunks.append(np.full(len(world), frame.timestamp))

    if not point_chunks:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    return (
        np.vstack(point_chunks),
        np.vstack(origin_chunks),
        np.concatenate(time_chunks),
    )


def sample_world_points(
    bundle: CaptureBundle,
    indices: np.ndarray,
    poses: np.ndarray | None = None,
    stride: int = 3,
    min_confidence: int = 1,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World points from `indices`, plus the observation time of each point.

    Unlike TSDF output these points keep their provenance, which is what makes
    a per-visit analysis possible at all.
    """
    points, _origins, times = _sample_provenance(
        bundle, indices, poses, stride, min_confidence, max_depth
    )
    return points, times


def sample_world_points_with_origin(
    bundle: CaptureBundle,
    indices: np.ndarray,
    poses: np.ndarray | None = None,
    stride: int = 3,
    min_confidence: int = 1,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`sample_world_points`, plus each point's camera origin. See `_sample_provenance`."""
    return _sample_provenance(
        bundle, indices, poses, stride, min_confidence, max_depth
    )


def refit_wall_offsets(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    points: np.ndarray,
    times: np.ndarray,
    band: float = 0.04,
    visit_gap: float = 3.0,
    min_points_per_visit: int = 250,
    corner_margin: float = 0.15,
) -> int:
    """Re-place each wall at the median of its per-visit offsets.

    A pooled fit puts the wall at the point-weighted centre of all its
    observations, so the visit that lingered longest -- not the one that
    measured best -- decides where the wall is.  Drift between visits then
    becomes position error in proportion to how unevenly they were sampled.
    Taking the median *per visit* first, and the median of those, gives every
    visit one vote: with three or more visits an outlying pass (a grazing
    sweep, a motion-blurred turn) stops moving the wall at all.

    Mutates offsets and shifts endpoints along the wall normal; orientation is
    left alone, because a single visit rarely constrains angle and the
    Manhattan snap already owns it.  Returns how many walls were refitted.
    """
    plan = frame.to_plan(points)
    refitted = 0

    for wall in walls:
        distance = plan @ wall.normal - wall.offset
        along = (plan - wall.start) @ wall.direction
        near = (
            (np.abs(distance) < band)
            & (along > corner_margin)
            & (along < wall.length - corner_margin)
        )
        if near.sum() < min_points_per_visit:
            continue

        wall_times = times[near]
        offsets_raw = (plan[near] @ wall.normal)
        order = np.argsort(wall_times)
        wall_times, offsets_raw = wall_times[order], offsets_raw[order]

        boundaries = np.flatnonzero(np.diff(wall_times) > visit_gap)
        starts = np.concatenate([[0], boundaries + 1])
        ends = np.concatenate([boundaries + 1, [len(wall_times)]])

        visit_offsets = [
            float(np.median(offsets_raw[s:e]))
            for s, e in zip(starts, ends)
            if e - s >= min_points_per_visit
        ]
        if not visit_offsets:
            continue

        new_offset = float(np.median(visit_offsets))
        shift = new_offset - wall.offset
        if abs(shift) > band:  # re-association failed; don't teleport the wall
            continue
        wall.offset = new_offset
        wall.start = wall.start + wall.normal * shift
        wall.end = wall.end + wall.normal * shift
        refitted += 1
    return refitted


def measure_drift(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    points: np.ndarray,
    times: np.ndarray,
    band: float = 0.04,
    visit_gap: float = 3.0,
    min_points_per_visit: int = 250,
    min_length: float = 1.5,
    corner_margin: float = 0.15,
) -> DriftMeasurement:
    """Per-visit plane offsets for every wall seen more than once.

    A "visit" is a run of observations separated from the next by more than
    `visit_gap` seconds -- the operator sweeping past a wall, leaving, and
    coming back later.  Comparing visits isolates drift accumulated in between.

    Two association guards keep the estimator honest about what is now known
    to sit near a wall.  The band stays inside the clutter standoff:
    suppressed parallel surfaces (trim, cabinet fronts) start at ~6 cm by
    construction of `merge_collinear`, and a wider band drifts onto that slab
    the moment a refit shifts the wall a centimetre toward it.  And
    `corner_margin` excludes the junction zones at both ends, where the
    perpendicular wall's own points fall inside any band.  Visit offsets are
    medians, not means, for the same reason: a contaminated tail should not
    move the estimate.
    """
    plan = frame.to_plan(points)
    visits: list[WallVisit] = []
    examined = 0

    for wall in walls:
        if wall.length < min_length:
            continue
        examined += 1

        distance = np.abs(plan @ wall.normal - wall.offset)
        along = (plan - wall.start) @ wall.direction
        nearby = (
            (distance < band)
            & (along > corner_margin)
            & (along < wall.length - corner_margin)
        )
        if nearby.sum() < min_points_per_visit * 2:
            continue

        wall_times = times[nearby]
        wall_plan = plan[nearby]
        order = np.argsort(wall_times)
        wall_times, wall_plan = wall_times[order], wall_plan[order]

        boundaries = np.flatnonzero(np.diff(wall_times) > visit_gap)
        starts = np.concatenate([[0], boundaries + 1])
        ends = np.concatenate([boundaries + 1, [len(wall_times)]])

        offsets: list[float] = []
        visit_times: list[float] = []
        counts: list[int] = []
        for start, end in zip(starts, ends):
            if end - start < min_points_per_visit:
                continue
            # Refit only the offset, holding orientation fixed: a single visit
            # may see too short a span to constrain the angle, and the shared
            # orientation is what makes the offsets comparable.
            offsets.append(float(np.median(wall_plan[start:end] @ wall.normal)))
            visit_times.append(float(wall_times[start:end].mean()))
            counts.append(int(end - start))

        if len(offsets) < 2:
            continue
        visits.append(
            WallVisit(
                wall_index=wall.index,
                visit_count=len(offsets),
                offsets=np.asarray(offsets),
                times=np.asarray(visit_times),
                point_counts=np.asarray(counts),
            )
        )

    spreads = np.asarray([visit.spread for visit in visits])
    return DriftMeasurement(
        walls_examined=examined,
        revisited_walls=len(visits),
        median_spread=float(np.median(spreads)) if len(spreads) else 0.0,
        p90_spread=float(np.percentile(spreads, 90)) if len(spreads) else 0.0,
        max_spread=float(spreads.max()) if len(spreads) else 0.0,
        per_wall=visits,
    )
