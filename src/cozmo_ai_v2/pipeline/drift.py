from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ingest import CaptureBundle, iter_frames
from .planes import HorizontalFrame, WallSegment


@dataclass
class WallVisit:
    """How far a single wall's surface appeared to sit from its fitted
    plane, broken down visit by visit.

    A "visit" here means one continuous stretch of time the camera spent
    near this wall -- if the walkthrough passes the same wall twice, at
    different points during the capture, that shows up as two separate
    visits. Comparing the wall's apparent position across those visits is
    the whole point of this class: it's what lets `measure_drift` tell
    real tracking drift apart from ordinary sensor noise.

    Attributes:
        wall_index: Which wall (as an index into the full `WallSegment`
            list) these visits belong to.
        visit_count: How many separate visits had enough points to be
            counted at all.
        offsets: A `(visit_count,)` array giving each visit's median
            distance from the wall's fitted plane, measured along the
            wall's normal direction, in metres.
        times: A `(visit_count,)` array giving the average capture
            timestamp of each visit, in seconds.
        point_counts: A `(visit_count,)` array giving how many points
            supported each visit's offset estimate.
    """

    wall_index: int
    visit_count: int
    offsets: np.ndarray
    times: np.ndarray
    point_counts: np.ndarray

    @property
    def spread(self) -> float:
        """How far apart the two most disagreeing visits' offsets are,
        in metres -- the simplest way to summarise how much this wall's
        apparent position disagreed between visits."""
        return float(np.ptp(self.offsets)) if len(self.offsets) > 1 else 0.0

    @property
    def std(self) -> float:
        """The standard deviation of this wall's per-visit offsets, in
        metres -- another way to summarise how much the visits
        disagreed, less swayed by a single outlier visit than
        `spread`."""
        return float(np.std(self.offsets)) if len(self.offsets) > 1 else 0.0


@dataclass
class DriftMeasurement:
    """A summary of how much the tracked camera path seems to have
    drifted, based on comparing repeat visits to the same walls.

    Attributes:
        walls_examined: How many walls were long enough to be considered
            at all.
        revisited_walls: How many of those walls were actually seen on
            two or more separate visits with enough points to compare.
        median_spread: The median `WallVisit.spread` across every
            revisited wall, in metres.
        p90_spread: The 90th-percentile `WallVisit.spread` across every
            revisited wall, in metres -- a rougher, worst-case-leaning
            figure than the median.
        max_spread: The single largest `WallVisit.spread` seen on any
            revisited wall, in metres.
        per_wall: One `WallVisit` per revisited wall, kept around for
            detailed reporting.
    """

    walls_examined: int
    revisited_walls: int
    median_spread: float
    p90_spread: float
    max_spread: float
    per_wall: list[WallVisit]

    def summary(self) -> str:
        """Builds a short, human-readable one-line summary of this
        measurement, suitable for printing straight to the console."""
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
    """Turns a set of frames' depth images into 3D world points, while
    also remembering where the camera was and when each point was seen.

    This is the shared "back-projection" step behind both
    `sample_world_points` and `sample_world_points_with_origin`: it reads
    each frame's depth image, converts every valid depth pixel into an
    actual 3D point using the camera's intrinsics, and moves that point
    into world coordinates using the camera's pose for that frame. Along
    with the point itself, it also records the camera position it was
    seen from and the frame's capture timestamp, since later stages (like
    `measure_drift`) need to know not just where a point is, but which
    camera position and which moment in time produced it.

    Args:
        bundle: The parsed capture; supplies the camera intrinsics and,
            when `poses` is not given, the camera-to-world poses too.
        indices: Which frame indices to back-project.
        poses: Camera-to-world poses to use instead of `bundle.poses`, if
            given; must be indexed exactly the same way as
            `bundle.poses`.
        stride: How many pixels to skip between sampled pixels, along
            both image axes -- a larger stride produces fewer, sparser
            points per frame.
        min_confidence: The lowest ARKit depth-confidence level to keep,
            passed straight through to `iter_frames`.
        max_depth: The furthest depth value to keep, in metres, passed
            straight through to `iter_frames`.

    Returns:
        A tuple of `(points, origins, times)`: an (N, 3) array of
        world-space points, an (N, 3) array giving the camera position
        each point was observed from, and an (N,) array giving the
        capture timestamp of the frame each point came from. All three
        come back empty if no frame contributed any valid depth.
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
        # Turn each valid depth pixel into a 3D point in the camera's own
        # coordinates, using the focal length and centre pixel from the
        # intrinsics matrix.
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
    """Samples a set of frames into 3D world points, plus the timestamp
    each point was observed at.

    Args:
        bundle: The parsed capture.
        indices: Which frame indices to sample.
        poses: Camera-to-world poses to use instead of `bundle.poses`, if
            given.
        stride: How many pixels to skip between sampled pixels, along
            both axes.
        min_confidence: The lowest ARKit depth-confidence level to keep.
        max_depth: The furthest depth value to keep, in metres.

    Returns:
        A tuple of `(points, times)`: an (N, 3) array of world points
        and an (N,) array of their capture timestamps.
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
    """The same sampling as `sample_world_points`, but also keeps track
    of the camera position each point was seen from.

    Some callers don't need the camera origin, but others -- like the
    occlusion checks in `planes.py` -- do; this variant returns it
    alongside the points and times so it doesn't have to be recomputed.

    Args:
        bundle: The parsed capture.
        indices: Which frame indices to sample.
        poses: Camera-to-world poses to use instead of `bundle.poses`, if
            given.
        stride: How many pixels to skip between sampled pixels, along
            both axes.
        min_confidence: The lowest ARKit depth-confidence level to keep.
        max_depth: The furthest depth value to keep, in metres.

    Returns:
        A tuple of `(points, origins, times)`, as described in
        `_sample_provenance`.
    """
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
    """Moves each wall to sit at the median of its own per-visit
    positions, instead of wherever its original single fit placed it.

    A wall seen on several separate visits during the walkthrough can end
    up with a slightly different estimated position each time, because of
    small amounts of camera drift between visits. Rather than trusting
    any one visit on its own, this groups a wall's nearby points by which
    visit they came from, finds each visit's own median offset from the
    wall, and then re-places the wall at the median of those per-visit
    numbers -- which is more robust to any single visit being an outlier
    than simply averaging every point together would be.

    Args:
        walls: The fitted wall segments to refit; any wall that gets
            moved has its `offset`, `start`, and `end` updated in place.
        frame: The building's horizontal reference frame, used to
            flatten points into plan coordinates.
        points: An (N, 3) array of world-space points, pooled across the
            whole capture.
        times: An (N,) array giving the capture timestamp of each point,
            used to split the points near each wall into separate
            visits.
        band: How far a point can be from the wall's current plane, in
            metres, and still be attributed to that wall.
        visit_gap: The minimum time gap, in seconds, between two
            consecutive points near a wall for them to be treated as
            separate visits.
        min_points_per_visit: The fewest points a visit can have and
            still have its median offset counted, and also the fewest
            points a wall needs nearby in total to be refitted at all.
        corner_margin: How far from each end of the wall, in metres, to
            ignore points -- this keeps corner clutter from throwing off
            the fit.

    Returns:
        How many walls actually had their offset shifted.
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

        # Split this wall's nearby points into separate visits wherever
        # there's a gap in time bigger than `visit_gap` -- a big gap
        # means the camera moved away and came back later, rather than
        # this all being one continuous look at the wall.
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
        if abs(shift) > band:
            # A shift larger than the band itself would move the wall
            # further than the points used to support that shift were
            # ever allowed to be from it in the first place -- treat that
            # as untrustworthy and leave the wall where it was.
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
    """Measures how much the tracked camera path seems to have drifted,
    by comparing how far apart a wall's apparent position lands across
    separate visits.

    It might seem like the simplest way to check for drift would be to
    just look at how scattered all the points near a wall are -- but that
    scatter is mostly just ordinary sensor noise, not evidence of drift.
    Sensor noise is present even within a single, short visit to a wall,
    and it mostly averages out once enough points are pooled together, so
    it doesn't actually say much about whether the camera's tracked
    position has quietly wandered away from the truth over time. Drift is
    a different kind of problem: it's a real, gradually accumulating
    shift in where the camera *thinks* it is, which means the same
    physical wall can end up looking like it's in a slightly different
    place on a later visit than it did on an earlier one, even though the
    wall itself never moved an inch. Comparing a wall's position across
    separate visits -- rather than just looking at how spread out its
    points are within a single visit -- is what actually isolates that
    effect.

    This walks through every wall long enough to bother with, groups its
    nearby points into visits (splitting on gaps in time, the same way
    `refit_wall_offsets` does), and for every wall seen on two or more
    qualifying visits, records how far apart those visits' median offsets
    landed.

    Args:
        walls: The fitted wall segments to examine (this function only
            reads them, it doesn't change them).
        frame: The building's horizontal reference frame, used to
            flatten points into plan coordinates.
        points: An (N, 3) array of world-space points, typically
            produced by `sample_world_points`.
        times: An (N,) array giving the capture timestamp of each point.
        band: How far a point can be from a wall's plane, in metres, and
            still be attributed to that wall.
        visit_gap: The minimum time gap, in seconds, between consecutive
            points for them to be split into separate visits.
        min_points_per_visit: The fewest points a run of points needs to
            count as a qualifying visit at all.
        min_length: The shortest a wall can be, in metres, and still be
            examined.
        corner_margin: How far from each end of the wall, in metres, to
            ignore points, so corner clutter doesn't throw off the
            comparison.

    Returns:
        A `DriftMeasurement` summarising `WallVisit.spread` across every
        wall that had two or more qualifying visits to compare.
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

        # Same visit-splitting idea as `refit_wall_offsets`: a time gap
        # bigger than `visit_gap` means the camera left and later came
        # back, so treat what's on either side of that gap as separate
        # visits.
        boundaries = np.flatnonzero(np.diff(wall_times) > visit_gap)
        starts = np.concatenate([[0], boundaries + 1])
        ends = np.concatenate([boundaries + 1, [len(wall_times)]])

        offsets: list[float] = []
        visit_times: list[float] = []
        counts: list[int] = []
        for start, end in zip(starts, ends):
            if end - start < min_points_per_visit:
                continue
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
