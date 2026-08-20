"""Extract named, measurable surfaces from a reconstruction.

Walls are recovered as lines in the gravity-aligned horizontal projection
rather than as planes in 3D.  A vertical surface has one free orientation and
one offset once gravity is known, so fitting it in 2D removes two degrees of
freedom that 3D RANSAC would otherwise have to estimate from noise -- and it
pools every point across the wall's full height into a single fit, which is
what makes a 2 cm tolerance reachable from a 256x192 depth sensor.

Wall extent comes from intersecting neighbouring wall lines, never from the
spread of observed points: furniture, doorways and grazing-incidence dropout
all truncate the observed span, while the corner where two wall planes meet is
where a tape measure would be placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import GravityEstimate


@dataclass
class HorizontalFrame:
    """The building's dominant horizontal axes, in world coordinates."""

    up: np.ndarray
    right: np.ndarray  # first Manhattan axis
    forward: np.ndarray  # second Manhattan axis, right-handed with up
    yaw: float  # rotation from world X, radians
    manhattan_fraction: float  # share of wall area on the two dominant axes

    def to_plan(self, points: np.ndarray) -> np.ndarray:
        """World points -> 2D plan coordinates (x along `right`, y along `forward`)."""
        return np.stack([points @ self.right, points @ self.forward], axis=-1)

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.up

    def to_world(self, plan: np.ndarray, height: float | np.ndarray) -> np.ndarray:
        """Plan coordinates plus a height back to world points."""
        plan = np.atleast_2d(plan)
        heights = np.broadcast_to(np.asarray(height, float), plan.shape[0])
        return (
            plan[:, :1] * self.right
            + plan[:, 1:2] * self.forward
            + heights[:, None] * self.up
        )


@dataclass
class WallSegment:
    """One planar vertical surface, expressed in plan coordinates."""

    index: int
    normal: np.ndarray  # 2D unit normal in plan space
    offset: float  # normal . x = offset
    start: np.ndarray  # 2D endpoint
    end: np.ndarray  # 2D endpoint
    inlier_count: int
    residual_rms: float  # metres, spread of inliers about the fitted line
    observed_span: tuple[float, float]  # along-wall extent actually seen
    # Height extent of the supporting points. NOT a measure of how tall the
    # surface is: `wall_band_mask` clips every candidate to the same band
    # before RANSAC runs, so this saturates at the band limits for real walls
    # and low furniture alike. Do not filter on it.
    height_range: tuple[float, float]
    room_id: int | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        delta = self.end - self.start
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 1e-9 else np.array([1.0, 0.0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of plan-space points from this wall's line."""
        return points @ self.normal - self.offset

    def along(self, points: np.ndarray) -> np.ndarray:
        """Coordinate of plan-space points along the wall, 0 at `start`."""
        return (points - self.start) @ self.direction

    @property
    def inferred_fraction(self) -> float:
        """Share of the wall's length that was never directly observed.

        Spans behind furniture or beyond a grazing view are reconstructed from
        the wall plane and its corners; the assignment requires them to be
        reported as inferred rather than passed off as measured.
        """
        if self.length < 1e-6:
            return 0.0
        seen = max(0.0, self.observed_span[1] - self.observed_span[0])
        return float(np.clip(1.0 - seen / self.length, 0.0, 1.0))


def estimate_horizontal_frame(
    normals: np.ndarray, up: np.ndarray, weights: np.ndarray | None = None
) -> HorizontalFrame:
    """Recover the building's yaw from wall normals.

    Wall normals in a rectilinear building cluster at 90-degree spacings, so
    their doubled-then-doubled angle (4*theta) collapses all four onto one
    direction whose circular mean is the yaw.  This is robust to walls being
    unevenly represented, which a histogram peak is not.
    """
    horizontal = normals - np.outer(normals @ up, up)
    magnitude = np.linalg.norm(horizontal, axis=1)
    vertical_enough = magnitude > 0.85  # normal lies within ~32 deg of horizontal
    horizontal = horizontal[vertical_enough] / magnitude[vertical_enough, None]
    if weights is not None:
        weights = weights[vertical_enough]

    right, forward = _orthonormal_basis(up)
    angles = np.arctan2(horizontal @ forward, horizontal @ right)
    resultant = np.exp(4j * angles)
    if weights is not None:
        resultant = resultant * weights
    mean = resultant.mean()
    yaw = float(np.angle(mean) / 4.0)

    axis_a = np.cos(yaw) * right + np.sin(yaw) * forward
    axis_b = np.cross(up, axis_a)
    return HorizontalFrame(
        up=up,
        right=axis_a / np.linalg.norm(axis_a),
        forward=axis_b / np.linalg.norm(axis_b),
        yaw=yaw,
        manhattan_fraction=float(np.abs(mean)),
    )


def _orthonormal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    right = seed - up * (up @ seed)
    right /= np.linalg.norm(right)
    return right, np.cross(up, right)


def wall_band_mask(
    points: np.ndarray,
    normals: np.ndarray,
    gravity: GravityEstimate,
    up: np.ndarray,
    margin: float = 0.35,
) -> np.ndarray:
    """Points belonging to vertical surfaces, clear of floor and ceiling.

    The margins matter: baseboards and crown moulding sit at the extremes and
    are not the wall plane, while furniture tops present horizontal normals
    that the normal test removes.
    """
    heights = points @ up
    ceiling = (
        gravity.ceiling_height
        if gravity.ceiling_height is not None
        else heights.max()
    )
    in_band = (heights > gravity.floor_height + margin) & (heights < ceiling - margin)
    vertical = np.abs(normals @ up) < 0.35  # normal within ~70 deg of horizontal
    return in_band & vertical


def extract_walls(
    plan_points: np.ndarray,
    heights: np.ndarray,
    inlier_threshold: float = 0.03,
    min_inliers: int = 400,
    min_length: float = 0.4,
    max_walls: int = 80,
    angle_tolerance_degrees: float = 8.0,
    point_spacing: float = 0.02,
    min_coverage: float = 0.06,
) -> list[WallSegment]:
    """Sequential RANSAC for wall lines in plan space.

    Each accepted line is refined by total least squares on its inliers, which
    is what converts a coarse RANSAC hypothesis into a metric surface: the
    consensus set selects *which* points belong to the wall, and the refit uses
    all of them to place it.

    Runs are then gated on `min_coverage`: the share of the run's own surface
    area that was actually observed, given the reconstruction's point spacing.
    `min_inliers` only constrains the whole RANSAC consensus set, which is then
    split into contiguous runs -- so without this a line supported by one real
    wall also emits the 30-point slivers that happened to fall on the same
    infinite line metres away.  Coverage is the right test because it is
    scale-free: on recordings-1 real walls score 8-89% while the slivers score
    1-3%, a gap no absolute point count spans across capture densities.
    """
    band_height = float(np.ptp(heights)) if len(heights) else 1.0
    remaining = np.arange(len(plan_points))
    walls: list[WallSegment] = []
    rng = np.random.default_rng(0)

    while len(remaining) >= min_inliers and len(walls) < max_walls:
        subset = plan_points[remaining]
        normal, offset, inlier_mask = _ransac_line(
            subset, inlier_threshold, rng, angle_tolerance_degrees
        )
        if inlier_mask.sum() < min_inliers:
            break

        inlier_points = subset[inlier_mask]
        normal, offset = _refit_line(inlier_points)
        residual = inlier_points @ normal - offset
        direction = np.array([-normal[1], normal[0]])
        projection = inlier_points @ direction

        # A single RANSAC line spans every collinear wall in the building --
        # opposite sides of a corridor share an offset.  Split into runs that
        # are actually contiguous before accepting anything.
        for lo, hi, count in _contiguous_runs(np.sort(projection), gap=0.35):
            run_length = hi - lo
            if run_length < min_length:
                continue
            expected = (run_length / point_spacing) * (band_height / point_spacing)
            if count < min_coverage * expected:
                continue
            base = normal * offset
            walls.append(
                WallSegment(
                    index=len(walls),
                    normal=normal,
                    offset=float(offset),
                    start=base + direction * lo,
                    end=base + direction * hi,
                    inlier_count=int(count),
                    residual_rms=float(np.sqrt((residual**2).mean())),
                    observed_span=(0.0, float(hi - lo)),
                    height_range=(
                        float(heights[remaining][inlier_mask].min()),
                        float(heights[remaining][inlier_mask].max()),
                    ),
                )
            )
        remaining = remaining[~inlier_mask]

    walls.sort(key=lambda wall: -wall.length)
    for position, wall in enumerate(walls):
        wall.index = position
    return walls


def _ransac_line(
    points: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    angle_tolerance_degrees: float,
    iterations: int = 300,
) -> tuple[np.ndarray, float, np.ndarray]:
    best_count = 0
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    count = len(points)

    for _ in range(iterations):
        a, b = rng.choice(count, size=2, replace=False)
        delta = points[b] - points[a]
        length = np.linalg.norm(delta)
        if length < 0.3:  # too short a baseline to define an orientation
            continue
        direction = delta / length
        normal = np.array([-direction[1], direction[0]])
        offset = normal @ points[a]
        inliers = np.abs(points @ normal - offset) < threshold
        hits = int(inliers.sum())
        if hits > best_count:
            best_count, best = hits, (normal, float(offset), inliers)

    if best is None:
        return np.array([1.0, 0.0]), 0.0, np.zeros(count, bool)
    return best


def _refit_line(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return normal, float(normal @ centroid)


def _contiguous_runs(sorted_values: np.ndarray, gap: float):
    """Split sorted 1D positions into runs separated by more than `gap`."""
    if len(sorted_values) == 0:
        return
    breaks = np.flatnonzero(np.diff(sorted_values) > gap)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(sorted_values) - 1]])
    for start, end in zip(starts, ends):
        if end > start:
            yield float(sorted_values[start]), float(sorted_values[end]), end - start + 1


def merge_collinear(
    walls: list[WallSegment],
    offset_tolerance: float = 0.06,
    parallel_tolerance: float = 0.30,
    min_overlap: float = 0.30,
    angle_tolerance_degrees: float = 15.0,
    gap_tolerance: float = 0.4,
) -> list[WallSegment]:
    """Resolve competing near-coplanar surfaces into one wall each.

    Sequential RANSAC produces two distinct kinds of duplicate, and they need
    opposite treatment:

    * **Fragments of one surface** (within `offset_tolerance`): a wall split by
      a doorway, or the same wall seen on two visits that drift apart by a
      couple of centimetres.  These are *merged* -- both are evidence of the
      same plane and of its extent.
    * **Parallel clutter** (within `parallel_tolerance`, overlapping along the
      run): once the wall's points are consumed, RANSAC keeps finding planes a
      few centimetres in front of it -- door reveals, trim, cabinet and
      bookcase fronts.  On recordings-1 a single wall spawned five such planes
      within 23 cm.  These are *suppressed*, not merged: averaging them would
      drag the wall off its true position, and extending the wall to their
      span would credit it with a run that furniture, not masonry, occupies.

    Because sequential RANSAC takes the largest consensus set first, the
    dominant plane in such a family is the wall; the rest are what stands in
    front of it, and they belong to the occlusion machinery instead.

    Normal *direction* is compared throughout, not just orientation, so the two
    faces of a partition are never combined -- they are different surfaces
    bounding different rooms, and their separation is the shared-wall thickness
    the assignment scores.
    """
    cosine_limit = np.cos(np.radians(angle_tolerance_degrees))
    # Strongest first: support, then length. The winner of each family keeps
    # its own geometry, so it must be the best-evidenced plane, not merely the
    # longest fragment.
    ordered = sorted(walls, key=lambda w: (-w.inlier_count, -w.length))
    kept: list[WallSegment] = []

    for wall in ordered:
        for target in kept:
            if target.normal @ wall.normal < cosine_limit:
                continue  # different direction, or the opposite face

            # Separation is measured as the candidate's greatest distance from
            # the target's line, not as a difference of offsets.  Offsets are
            # only comparable between exactly parallel lines: within the 15
            # degrees this tolerance allows, two segments can share an offset
            # at their midpoints and still diverge by half a metre at their
            # ends.  Endpoint distance is what "the same surface" actually
            # means, and it subsumes the parallel case.
            separation = max(
                abs(float(target.project(wall.start))),
                abs(float(target.project(wall.end))),
            )
            if separation > parallel_tolerance:
                continue

            span = [target.along(wall.start), target.along(wall.end)]
            lo, hi = min(span), max(span)
            overlap = max(0.0, min(hi, target.length) - max(lo, 0.0))

            if separation <= offset_tolerance:
                if lo > target.length + gap_tolerance or hi < -gap_tolerance:
                    continue
                _absorb(target, wall, lo, hi)
                break
            if overlap >= min_overlap:
                # The tag lands on the wall that survives, so it must say
                # something true about *that* wall: something parallel stood
                # in front of it, which is also why part of it is occluded.
                target.tags.append("clutter-in-front")
                break  # clutter in front of `target`: drop it
        else:
            kept.append(wall)

    kept.sort(key=lambda w: -w.length)
    for position, wall in enumerate(kept):
        wall.index = position
        wall.tags = sorted(set(wall.tags))
    return kept


def _absorb(target: WallSegment, other: WallSegment, lo: float, hi: float) -> None:
    """Extend `target` to cover `other`, weighting the fit by support."""
    total = target.inlier_count + other.inlier_count
    target.offset = (
        target.offset * target.inlier_count + other.offset * other.inlier_count
    ) / max(total, 1)
    target.residual_rms = (
        target.residual_rms * target.inlier_count
        + other.residual_rms * other.inlier_count
    ) / max(total, 1)

    direction = target.direction
    base = target.normal * target.offset
    origin = base + direction * (direction @ target.start)
    new_lo, new_hi = min(0.0, lo), max(target.length, hi)
    target.start = origin + direction * new_lo
    target.end = origin + direction * new_hi

    # Observed span is tracked in the merged frame so `inferred_fraction`
    # still reports how much of the combined wall was actually seen.
    seen = (
        target.observed_span[1]
        - target.observed_span[0]
        + other.observed_span[1]
        - other.observed_span[0]
    )
    target.observed_span = (0.0, min(seen, new_hi - new_lo))
    target.inlier_count = total
    target.height_range = (
        min(target.height_range[0], other.height_range[0]),
        max(target.height_range[1], other.height_range[1]),
    )
    target.tags = sorted(set(target.tags) | set(other.tags))


def _ray_crosses_wall(
    camera: np.ndarray, target: np.ndarray, wall: WallSegment, margin_fraction: float
) -> np.ndarray:
    """Boolean mask: does camera[i]->target[i] cross `wall`'s solid span strictly between them?

    Vectorised over many rays against one wall -- this is the inner loop of
    `filter_occluded_walls`, tested once per (candidate wall, blocker) pair
    rather than once per point, which is what keeps that function fast.

    Deliberately not built on `_segment_intersection`: that helper compares
    two whole `WallSegment`s with a angle-based parallel cutoff meant for
    finding corners, whereas this compares many short rays against one wall
    with a magnitude-based cutoff and a margin expressed as a fraction of the
    wall's own length -- different enough parametrisations that sharing code
    would obscure both.
    """
    ray = target - camera
    wall_vector = wall.end - wall.start
    denominator = ray[:, 0] * wall_vector[1] - ray[:, 1] * wall_vector[0]
    parallel = np.abs(denominator) < 1e-9
    safe_denominator = np.where(parallel, 1.0, denominator)

    diff = wall.start - camera
    t = (diff[:, 0] * wall_vector[1] - diff[:, 1] * wall_vector[0]) / safe_denominator
    s = (diff[:, 0] * ray[:, 1] - diff[:, 1] * ray[:, 0]) / safe_denominator

    return (
        ~parallel
        & (t > 0.02) & (t < 0.98)  # strictly between camera and the point
        & (s > margin_fraction) & (s < 1 - margin_fraction)  # within wall's solid span
    )


def filter_occluded_walls(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    points: np.ndarray,
    origins: np.ndarray,
    band: float = 0.06,
    corner_margin: float = 0.15,
    min_points: int = 200,
    min_blocker_length: float = 1.0,
    occlusion_fraction: float = 0.6,
) -> tuple[list[WallSegment], int]:
    """Drop candidate walls whose points require passing through a solid wall.

    `merge_collinear` handles surfaces that sit too close to the camera --
    furniture standing in front of a wall, occluding it.  This handles the
    opposite geometry: a candidate plane sitting too far away, whose points
    could only have been produced by light that passed through an already
    better-supported wall to get there.  That is physically impossible for a
    real, independently-measurable surface.

    Two distinct cases were confirmed on recordings-1, traced by hand rather
    than assumed:

    * A candidate 15 cm and 9 cm behind an accepted wall, with an exactly
      *opposite* normal -- too thin to be a second room's wall, and the wrong
      separation to be that wall's own far face properly observed (interior
      partitions run 9-15 cm, which is what these are: a sliver of the SAME
      wall's far face, grazed through a doorway edge from the near room, that
      RANSAC fitted as if it were its own freestanding wall). `merge_collinear`
      does not catch this -- opposite-facing near-parallel planes are its
      signature for two rooms sharing a partition, which is correct in
      general, just not here.
    * A candidate blocked by a wall at a completely different angle (0 degree
      normal alignment, i.e. perpendicular) across most of its points -- no
      thin-partition explanation available, so it is a stray fragment (noise,
      multipath, or a glimpse of a room not yet separated by `rooms.py`, which
      runs after this).

    Both signatures are physically impossible for a freestanding wall and were
    removed identically; the code does not need to (and does not try to)
    distinguish them.

    A candidate is tested only against *stronger* walls (`min_blocker_length`
    keeps stub fragments from acting as blockers), so a real wall is never at
    risk from a weaker impostor, and the check is majority-vote per wall
    (`occlusion_fraction`) rather than any-single-ray, so one grazing ray
    through a genuine doorway gap can't condemn an otherwise-solid wall.
    """
    plan = frame.to_plan(points)
    camera = frame.to_plan(origins)
    ordered = sorted(walls, key=lambda w: -w.inlier_count)
    kept: list[WallSegment] = []
    dropped = 0

    for wall in ordered:
        blockers = [
            other
            for other in walls
            if other is not wall
            and other.inlier_count > wall.inlier_count
            and other.length >= min_blocker_length
        ]
        if not blockers:
            kept.append(wall)
            continue

        distance = plan @ wall.normal - wall.offset
        along = (plan - wall.start) @ wall.direction
        near = (
            (np.abs(distance) < band)
            & (along > corner_margin)
            & (along < wall.length - corner_margin)
        )
        if near.sum() < min_points:
            kept.append(wall)  # too little evidence either way: fail open
            continue

        near_camera, near_target = camera[near], plan[near]
        blocked = np.zeros(near.sum(), bool)
        for blocker in blockers:
            if blocked.all():
                break
            margin_fraction = min(corner_margin / max(blocker.length, 1e-6), 0.45)
            blocked |= _ray_crosses_wall(
                near_camera, near_target, blocker, margin_fraction
            )

        if float(blocked.mean()) >= occlusion_fraction:
            dropped += 1
            continue
        kept.append(wall)

    kept.sort(key=lambda w: -w.length)
    for position, wall in enumerate(kept):
        wall.index = position
    return kept, dropped


def _segment_intersection(
    a: WallSegment, b: WallSegment
) -> tuple[np.ndarray, float, float] | None:
    """Intersection of the two walls' infinite lines, with along-coordinates.

    Returns (point, u_a, u_b) where u_x is the along-wall coordinate of the
    point on wall x (0 at start), or None for near-parallel lines.
    """
    da, db = a.direction, b.direction
    denominator = da[0] * db[1] - da[1] * db[0]
    if abs(denominator) < 0.15:  # < ~9 deg apart: no meaningful corner
        return None
    delta = b.start - a.start
    u_a = (delta[0] * db[1] - delta[1] * db[0]) / denominator
    point = a.start + da * u_a
    u_b = float((point - b.start) @ db)
    return point, float(u_a), u_b


def resolve_crossings(
    walls: list[WallSegment],
    interior_margin: float = 0.15,
    max_trim: float = 0.45,
) -> list[WallSegment]:
    """Enforce the one physical constraint the fitter cannot see: walls do not
    pass through each other.

    An interior-interior intersection has exactly two causes, distinguishable
    by geometry:

    * **T-junction overshoot** -- both surfaces are real, but one wall's
      extent ran a few decimetres past the junction (its collinear inliers
      continue on the far side, in the next room).  The overhang past the
      crossing is short: trim it back to the junction.
    * **Clutter cutting a wall** -- a stair rail, counter edge or furniture
      diagonal fitted as a wall slices through a genuine wall near its middle,
      typically at a shallow angle.  Neither overhang is short, so trimming
      cannot fix it; the surface with weaker support is not a wall.  Drop it.

    Deletion cascades correctly on families of mutually-crossing clutter
    because survivors are re-checked against every remaining wall.
    """
    walls = list(walls)
    changed = True
    while changed:
        changed = False
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                a, b = walls[i], walls[j]
                hit = _segment_intersection(a, b)
                if hit is None:
                    continue
                point, u_a, u_b = hit
                interior_a = interior_margin < u_a < a.length - interior_margin
                interior_b = interior_margin < u_b < b.length - interior_margin
                if not (interior_a and interior_b):
                    continue

                overhang_a = min(u_a, a.length - u_a)
                overhang_b = min(u_b, b.length - u_b)
                if min(overhang_a, overhang_b) <= max_trim:
                    victim = a if overhang_a <= overhang_b else b
                    u = u_a if victim is a else u_b
                    if u < victim.length - u:
                        victim.start = point.copy()
                        victim.observed_span = (
                            max(0.0, victim.observed_span[0] - u),
                            max(0.0, victim.observed_span[1] - u),
                        )
                    else:
                        victim.end = point.copy()
                        victim.observed_span = (
                            min(victim.observed_span[0], victim.length),
                            min(victim.observed_span[1], victim.length),
                        )
                    victim.tags.append("trimmed-at-junction")
                else:
                    weaker = a if a.inlier_count < b.inlier_count else b
                    walls.remove(weaker)
                changed = True
                break
            if changed:
                break

    # Indices are deliberately left alone: they are identities that per-wall
    # drift measurements and surface grids are keyed by, not positions.
    for wall in walls:
        wall.tags = sorted(set(wall.tags))
    return walls


def snap_corners(
    walls: list[WallSegment],
    max_extension: float = 0.45,
    max_trim: float = 0.30,
) -> int:
    """Move wall endpoints onto the intersections of their wall lines.

    An extent taken from observed points stops where the last inlier fell --
    short of the corner when the scan never reached it, past the corner when
    collinear surface continued beyond.  The corner, where two fitted lines
    intersect, is where a tape measure is hooked; only after this step do the
    emitted lengths mean what the wall-length interval model already assumes
    (a difference of two plane intersections).

    Each endpoint adopts the candidate corner needing the smallest adjustment,
    provided the intersection also lies on or near the partner wall -- a line
    crossing three metres away is geometry, not a corner.  Endpoints with no
    candidate keep their observed extent, and remain covered by the
    occupancy/inferred machinery.  Returns the number of endpoints snapped.
    """
    proposals: list[tuple[float, int, str, np.ndarray]] = []
    for i, wall in enumerate(walls):
        for other in walls:
            if other is wall:
                continue
            hit = _segment_intersection(wall, other)
            if hit is None:
                continue
            point, u_self, u_other = hit
            # The corner must lie on (or just beyond) the partner too.
            if not (-max_extension <= u_other <= other.length + max_extension):
                continue
            for end_name, adjustment in (
                ("start", -u_self),
                ("end", u_self - wall.length),
            ):
                # Positive adjustment extends the wall, negative trims it.
                if -max_trim <= adjustment <= max_extension:
                    proposals.append((abs(adjustment), i, end_name, point))

    proposals.sort(key=lambda entry: entry[0])
    taken: set[tuple[int, str]] = set()
    snapped = 0
    for _, index, end_name, point in proposals:
        key = (index, end_name)
        if key in taken:
            continue
        wall = walls[index]
        remaining = (
            float(np.linalg.norm(wall.end - point))
            if end_name == "start"
            else float(np.linalg.norm(point - wall.start))
        )
        if remaining < 0.3:  # would collapse the wall onto its neighbour
            continue
        taken.add(key)
        if end_name == "start":
            shift = float(wall.along(point[None])[0])
            wall.start = point.copy()
            wall.observed_span = (
                max(0.0, wall.observed_span[0] - shift),
                max(0.0, wall.observed_span[1] - shift),
            )
        else:
            wall.end = point.copy()
            wall.observed_span = (
                min(wall.observed_span[0], wall.length),
                min(wall.observed_span[1], wall.length),
            )
        wall.tags = sorted(set(wall.tags) | {f"corner-{end_name}"})
        snapped += 1
    return snapped


def snap_to_frame(
    walls: list[WallSegment], frame: HorizontalFrame, tolerance_degrees: float = 6.0
) -> list[WallSegment]:
    """Rotate near-axis walls onto the building frame.

    Regularisation, not cosmetics: a wall fitted from a partly occluded, noisy
    band can sit a degree or two off true, which displaces its far end by more
    than the error budget allows.  Walls further than `tolerance_degrees` from
    an axis are left alone -- genuinely angled walls exist and forcing them
    would be worse than leaving them.
    """
    for wall in walls:
        angle = np.arctan2(wall.normal[1], wall.normal[0])
        snapped = np.round(angle / (np.pi / 2)) * (np.pi / 2)
        if abs(np.degrees(angle - snapped)) > tolerance_degrees:
            wall.tags.append("off-axis")
            continue
        normal = np.array([np.cos(snapped), np.sin(snapped)])
        # Keep the wall where its points are: re-derive the offset from the
        # midpoint so snapping rotates the line without translating it.
        offset = float(normal @ wall.midpoint)
        direction = np.array([-normal[1], normal[0]])
        half = 0.5 * wall.length
        centre = normal * offset + direction * (direction @ wall.midpoint)
        wall.normal, wall.offset = normal, offset
        wall.start, wall.end = centre - direction * half, centre + direction * half
    return walls
