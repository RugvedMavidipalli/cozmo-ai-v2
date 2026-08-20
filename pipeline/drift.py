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
    pose_table = bundle.poses if poses is None else poses
    intrinsics = bundle.intrinsics
    chunks: list[np.ndarray] = []
    times: list[np.ndarray] = []

    for frame in iter_frames(
        bundle, indices, min_confidence=min_confidence, max_depth=max_depth
    ):
        depth = frame.depth[::stride, ::stride]
        valid = depth > 0
        if not valid.any():
            continue
        vs, us = np.nonzero(valid)
        z = depth[valid]
        points = np.stack(
            [
                (us * stride - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (vs * stride - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ],
            axis=1,
        )
        pose = pose_table[frame.index]
        chunks.append(points @ pose[:3, :3].T + pose[:3, 3])
        times.append(np.full(len(points), frame.timestamp))

    if not chunks:
        return np.empty((0, 3)), np.empty(0)
    return np.vstack(chunks), np.concatenate(times)


def measure_drift(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    points: np.ndarray,
    times: np.ndarray,
    band: float = 0.06,
    visit_gap: float = 3.0,
    min_points_per_visit: int = 250,
    min_length: float = 1.5,
) -> DriftMeasurement:
    """Per-visit plane offsets for every wall seen more than once.

    A "visit" is a run of observations separated from the next by more than
    `visit_gap` seconds -- the operator sweeping past a wall, leaving, and
    coming back later.  Comparing visits isolates drift accumulated in between.
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
            (distance < band) & (along > -0.1) & (along < wall.length + 0.1)
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
            offsets.append(float((wall_plan[start:end] @ wall.normal).mean()))
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
