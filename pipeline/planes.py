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
