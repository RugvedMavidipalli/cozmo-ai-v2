from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .geometry import GravityEstimate

WallCoordinateConvention = Literal["finished_face", "centerline", "room_side"]
FINISHED_FACE: WallCoordinateConvention = "finished_face"


@dataclass(frozen=True)
class ProjectedWallLine:
    """A gravity-aligned 2D line obtained from a 3D vertical plane.

    The source plane is represented by ``normal_3d · world = offset_3d``.
    For a genuinely vertical plane, its horizontal component is already the
    same finished-face line used by :class:`WallSegment`.  Small tilt is
    reported in ``tilt_degrees`` rather than silently changing the frame.
    """

    normal: np.ndarray
    offset: float
    tilt_degrees: float
    confidence: float
    coordinate_convention: WallCoordinateConvention = FINISHED_FACE

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (2,) or not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("projected wall line normal must be finite and non-zero")
        normal = normal / norm
        canonical = _canonical_normal(normal)
        offset = float(self.offset)
        if not np.isfinite(offset):
            raise ValueError("projected wall line offset must be finite")
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported wall coordinate convention")
        confidence = float(self.confidence)
        confidence = confidence if np.isfinite(confidence) else 0.0
        if not np.allclose(canonical, normal):
            offset = -offset
        object.__setattr__(self, "normal", canonical)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "tilt_degrees", float(self.tilt_degrees))
        object.__setattr__(
            self, "confidence", float(np.clip(confidence, 0.0, 1.0))
        )


@dataclass
class HorizontalFrame:
    """The building's own sense of "flat and square", worked out from the walls.

    Most buildings are built out of straight walls that meet at right
    angles, even if the whole building is rotated at some odd angle relative
    to true north or to the raw coordinate system the capture happened to
    use. This class captures that rotation once, as two horizontal
    directions (`right` and `forward`) that line up with the building's own
    walls, plus the `up` direction against gravity. Once we have this frame,
    it's much easier to work with the building in its own natural
    "floor-plan" coordinates instead of the arbitrary coordinates the raw
    3D points came in.

    Attributes:
        up: A world-space unit vector pointing straight up, against gravity.
        right: A world-space unit vector along the first wall direction of
            the building (one of its two dominant, perpendicular wall
            directions).
        forward: A world-space unit vector along the other wall direction,
            perpendicular to `right`, forming a right-handed frame together
            with `up`.
        yaw: How far `right` is rotated away from the world's raw X axis, in
            radians. This is just another way of describing the same
            rotation as `right`/`forward`.
        manhattan_fraction: A rough score, from 0 to 1, of how strongly the
            building's walls actually agree on a single pair of
            perpendicular directions. A value close to 1 means the walls are
            consistently square with each other (a typical rectilinear
            building); a lower value means the walls point in a wider mix of
            directions, so the recovered frame is less trustworthy.
    """

    up: np.ndarray
    right: np.ndarray
    forward: np.ndarray
    yaw: float
    manhattan_fraction: float

    def to_plan(self, points: np.ndarray) -> np.ndarray:
        """Converts world-space points into flat, 2D floor-plan coordinates.

        This drops the height and keeps only the position along the
        building's own `right` and `forward` directions, which is exactly
        what a floor plan needs: a top-down view where the walls line up
        with the grid.

        Args:
            points: World-space points, as an (N, 3) array (or a single
                point, as a length-3 array).

        Returns:
            The same points, as (N, 2) plan-space coordinates, with x
            measured along `right` and y measured along `forward`.
        """
        return np.stack([points @ self.right, points @ self.forward], axis=-1)

    def height(self, points: np.ndarray) -> np.ndarray:
        """Works out how high up each point is, measured along `up`.

        Args:
            points: World-space points, as an (N, 3) array (or a single
                point, as a length-3 array).

        Returns:
            The height of each point above (positive) or below (negative)
            the frame's origin, as an (N,) array.
        """
        return points @ self.up

    def to_world(self, plan: np.ndarray, height: float | np.ndarray) -> np.ndarray:
        """Turns flat floor-plan coordinates plus a height back into a real 3D point.

        This is the reverse of `to_plan` combined with `height`: given a 2D
        position on the floor plan and how high up it should be, it
        reconstructs the actual point in world-space coordinates.

        Args:
            plan: Plan-space coordinates, as an (N, 2) array (or a single
                point, as a length-2 array).
            height: How high up each point should be along `up`. Can be one
                number shared by every point, or an (N,) array giving a
                different height per point.

        Returns:
            The reconstructed points, as an (N, 3) array of world-space
            coordinates.
        """
        plan = np.atleast_2d(plan)
        heights = np.broadcast_to(np.asarray(height, float), plan.shape[0])
        return (
            plan[:, :1] * self.right
            + plan[:, 1:2] * self.forward
            + heights[:, None] * self.up
        )


def _derive_wall_confidence(inlier_count: int, residual_rms: float) -> float:
    """Turn support and line residual into a conservative [0, 1] score."""
    support = 1.0 - np.exp(-max(int(inlier_count), 0) / 100.0)
    residual = (
        np.exp(-max(float(residual_rms), 0.0) / 0.05)
        if np.isfinite(residual_rms)
        else 0.0
    )
    return float(np.clip(support * residual, 0.0, 1.0))


def vertical_plane_to_line(
    normal: np.ndarray,
    offset: float,
    frame: HorizontalFrame,
    *,
    max_tilt_degrees: float = 20.0,
    confidence: float = 1.0,
    coordinate_convention: WallCoordinateConvention = FINISHED_FACE,
) -> ProjectedWallLine:
    """Project a world-space vertical plane into the building plan frame.

    The plane follows the explicit finished-face convention used throughout
    the reconstruction: ``normal · world = offset`` describes the measured
    visible wall face. Its component along gravity is removed, and the
    remaining normal is converted to ``(right, forward)`` coordinates. A
    plane with more than ``max_tilt_degrees`` from vertical is rejected rather
    than being forced into a wall line.
    """
    normal = np.asarray(normal, dtype=float).reshape(-1)
    if normal.shape != (3,):
        raise ValueError("plane normal must have shape (3,)")
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("plane normal must be finite and non-zero")
    offset = float(offset)
    if not np.isfinite(offset):
        raise ValueError("plane offset must be finite")
    if (
        not np.isfinite(max_tilt_degrees)
        or max_tilt_degrees < 0
        or max_tilt_degrees >= 90
    ):
        raise ValueError("max_tilt_degrees must be in [0, 90)")
    up = np.asarray(frame.up, dtype=float).reshape(-1)
    up_norm = float(np.linalg.norm(up))
    if up.shape != (3,) or not np.isfinite(up_norm) or up_norm <= 1e-9:
        raise ValueError("frame.up must be a finite non-zero vector")
    up = up / up_norm
    normal = normal / norm
    horizontal = normal - float(normal @ up) * up
    horizontal_norm = float(np.linalg.norm(horizontal))
    tilt = float(np.degrees(np.arcsin(np.clip(abs(normal @ up), 0.0, 1.0))))
    if horizontal_norm <= 1e-9 or tilt > max_tilt_degrees:
        raise ValueError(
            f"plane is not vertical enough for a wall line (tilt={tilt:.2f}°)"
        )

    plan_normal = np.array(
        [normal @ frame.right, normal @ frame.forward], dtype=float
    )
    plan_normal /= float(np.linalg.norm(plan_normal))
    canonical_plan_normal = _canonical_normal(plan_normal)
    # A slightly tilted plane is represented at the gravity-zero plan slice;
    # tilt remains visible in the returned quality metadata.
    projected_offset = offset / horizontal_norm
    if not np.allclose(canonical_plan_normal, plan_normal):
        projected_offset = -projected_offset
    verticality = float(np.cos(np.radians(tilt)))
    return ProjectedWallLine(
        normal=canonical_plan_normal,
        offset=float(projected_offset),
        tilt_degrees=tilt,
        confidence=float(confidence) * verticality,
        coordinate_convention=coordinate_convention,
    )


# Descriptive alias for callers that use "project" for all frame transforms.
project_vertical_plane = vertical_plane_to_line


@dataclass
class WallSegment:
    """One flat, vertical wall surface, described in the building's own floor-plan coordinates.

    A wall is really just a straight line in the floor plan (an infinite
    line, described by `normal` and `offset`) with a finite start and end
    point marking where the actual wall stops. `normal` is a 2D unit vector
    pointing away from the wall's face, and `offset` is how far that line
    sits from the origin, so that any point `x` on the line satisfies
    `normal . x = offset`. Alongside the geometry, a `WallSegment` also
    carries some bookkeeping about how confidently it was detected, and
    labels attached by later processing steps.

    Attributes:
        index: This wall's position in whatever list currently holds it.
            This gets reassigned whenever walls are re-sorted or filtered,
            so it shouldn't be treated as a permanent ID.
        normal: A 2D unit vector, in plan space, pointing perpendicular to
            the wall's face.
        offset: How far the wall's line sits from the plan-space origin,
            measured along `normal`.
        start: The plan-space coordinate of one end of the wall.
        end: The plan-space coordinate of the other end of the wall.
        inlier_count: How many of the original 3D points were judged to
            actually lie on this wall (its RANSAC "votes"). Higher usually
            means more confidence in the wall.
        residual_rms: How far, on average, the supporting points strayed
            from a perfectly flat line, in metres. Lower means a cleaner,
            flatter wall.
        observed_span: The `(lo, hi)` stretch, in metres along the wall,
            that was actually seen by the sensor, as opposed to space
            between `start` and `end` that was only inferred.
        height_range: The `(min, max)` height of the points that support
            this wall.
        room_id: Which room this wall was assigned to by a later stage, or
            `None` if it hasn't been assigned yet.
        name: A human-readable label (like `"wall_3"`) assigned by a later
            pipeline stage, or `None` if it hasn't been named yet.
        tags: Free-form text labels this module and later stages attach to
            record notable things about the wall, such as `"clutter-in-front"`
            or `"trimmed-at-junction"`.
        confidence: Fit confidence in [0, 1], derived from support and
            residual when omitted by a caller.
        fit_quality: Quality of the fitted line, kept separate from later
            topology acceptance decisions.
        coordinate_convention: Explicit line convention. The integrated
            pipeline uses `"finished_face"`; it never treats a measured
            surface as a centerline implicitly.
        provenance: Source label for audit and vectorizer metadata.
        snap_status: `"unsnapped"`, `"snapped"`, or a rejection reason.
        snap_residual: RMS distance in metres from the pre-snap line to the
            accepted snapped line, or zero when no snap was attempted.
    """

    index: int
    normal: np.ndarray
    offset: float
    start: np.ndarray
    end: np.ndarray
    inlier_count: int
    residual_rms: float
    observed_span: tuple[float, float]
    height_range: tuple[float, float]
    room_id: int | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    # Off-axis detections are retained for diagnostics, but quarantined from
    # room topology, polygonization, and metric wall output.
    quarantined: bool = False
    confidence: float = -1.0
    fit_quality: float = -1.0
    coordinate_convention: WallCoordinateConvention = FINISHED_FACE
    provenance: str = "point-cloud"
    snap_status: str = "unsnapped"
    snap_residual: float = 0.0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (2,):
            raise ValueError("wall normal must be a non-zero 2D vector")
        invalid_normal = not np.isfinite(norm) or norm <= 1e-9
        if invalid_normal:
            normal = np.array([1.0, 0.0])
            norm = 1.0
        normal /= norm
        canonical = _canonical_normal(normal)
        if not np.allclose(canonical, normal):
            self.offset = -float(self.offset)
        self.normal = canonical
        self.start = np.asarray(self.start, dtype=float).reshape(-1)
        self.end = np.asarray(self.end, dtype=float).reshape(-1)
        if self.start.shape != (2,) or self.end.shape != (2,):
            raise ValueError("wall endpoints must be 2D vectors")
        self.tags = list(self.tags)
        self.inlier_count = max(int(self.inlier_count), 0)
        self.residual_rms = (
            float(self.residual_rms)
            if np.isfinite(self.residual_rms)
            else float("inf")
        )
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported wall coordinate convention")
        if not np.isfinite(self.confidence) or self.confidence < 0:
            self.confidence = _derive_wall_confidence(
                self.inlier_count, self.residual_rms
            )
        else:
            self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if not np.isfinite(self.fit_quality) or self.fit_quality < 0:
            self.fit_quality = self.confidence
        else:
            self.fit_quality = float(np.clip(self.fit_quality, 0.0, 1.0))
        self.provenance = str(self.provenance)
        self.snap_status = str(self.snap_status)
        self.snap_residual = (
            float(self.snap_residual)
            if np.isfinite(self.snap_residual) and self.snap_residual >= 0
            else 0.0
        )
        if invalid_normal:
            self.quarantined = True
            self.tags.append("invalid-geometry")
        if not np.isfinite(self.offset):
            self.offset = 0.0
            self.quarantined = True
            self.tags.append("invalid-geometry")
        if not np.isfinite(self.start).all() or not np.isfinite(self.end).all():
            self.quarantined = True
            self.tags.append("invalid-geometry")
        elif np.linalg.norm(self.end - self.start) <= 1e-9:
            self.quarantined = True
            self.tags.append("degenerate")

    @property
    def off_axis(self) -> bool:
        """Whether this wall was quarantined as a non-Manhattan line."""
        return "off-axis" in self.tags

    @property
    def topology_eligible(self) -> bool:
        """Whether later graph stages may use this candidate as a wall."""
        return not self.quarantined and self.length > 1e-6

    @property
    def quality(self) -> float:
        """Short alias for the fitted-line quality score."""
        return self.fit_quality

    @property
    def plane_confidence(self) -> float:
        """Alias exposing the confidence of the supporting wall plane."""
        return self.confidence

    @property
    def length(self) -> float:
        """The straight-line distance between `start` and `end`, in metres."""
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """A unit vector pointing from `start` toward `end`.

        If the two endpoints happen to be the same point (a degenerate,
        zero-length wall), this just returns `[1, 0]` rather than dividing
        by zero.
        """
        delta = self.end - self.start
        norm = np.linalg.norm(delta)
        return delta / norm if norm > 1e-9 else np.array([1.0, 0.0])

    @property
    def midpoint(self) -> np.ndarray:
        """The plan-space point halfway between `start` and `end`."""
        return 0.5 * (self.start + self.end)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Measures how far a set of points sits from this wall's line, and on which side.

        Args:
            points: Plan-space points to measure, as an (N, 2) array (or a
                single point, as a length-2 array).

        Returns:
            The signed distance of each point from the wall's line, as an
            (N,) array. The distance is positive on the side the wall's
            `normal` points toward, and negative on the other side.
        """
        return points @ self.normal - self.offset

    def along(self, points: np.ndarray) -> np.ndarray:
        """Measures how far along the wall each point sits, starting from `start`.

        Args:
            points: Plan-space points to measure, as an (N, 2) array (or a
                single point, as a length-2 array).

        Returns:
            The distance of each point along the wall's `direction`,
            measured from `start`, as an (N,) array. This can be negative
            (before `start`) or larger than `length` (past `end`).
        """
        return (points - self.start) @ self.direction

    @property
    def inferred_fraction(self) -> float:
        """How much of this wall's length was never actually seen by the sensor.

        A wall's `start` and `end` sometimes extend past what the sensor
        directly observed, filling in a gap the pipeline is fairly
        confident about (for example, a short stretch hidden behind
        furniture). This property reports how much of the wall's total
        length falls into that "filled in" category, as opposed to being
        directly observed.

        Returns:
            A fraction between 0 and 1, where 0 means the whole wall was
            directly observed and 1 means none of it was. Returns 0.0 for a
            wall with essentially zero length.
        """
        if self.length < 1e-6:
            return 0.0
        seen = max(0.0, self.observed_span[1] - self.observed_span[0])
        return float(np.clip(1.0 - seen / self.length, 0.0, 1.0))


def estimate_horizontal_frame(
    normals: np.ndarray, up: np.ndarray, weights: np.ndarray | None = None
) -> HorizontalFrame:
    """Works out which way the building is "facing", based on the directions its walls point.

    Most rooms are built from walls that meet at right angles, even if the
    whole room is rotated at some arbitrary angle in the raw capture data.
    This function looks at a big pile of surface normals (arrows pointing
    straight out of whatever surface each point belongs to), keeps only the
    ones that point roughly sideways rather than up or down (since those are
    the ones that could belong to a wall), and then finds the single
    rotation that best explains all of them as pointing along one of two
    perpendicular directions.

    The tricky part is that a wall facing north and a wall facing south (or
    east and west) both belong to the same building orientation -- a wall's
    normal could point either way. To average these directions without them
    cancelling each other out, this multiplies every angle by 4 before
    averaging (which lines up all four possible right-angle directions on
    top of each other), then divides the result back down by 4 at the end.

    Args:
        normals: World-space unit normals, one per candidate wall point or
            patch, as an (N, 3) array.
        up: A world-space unit vector pointing straight up, against
            gravity.
        weights: An optional per-normal weight, as an (N,) array, letting
            some normals count for more than others. If omitted, every
            normal counts equally.

    Returns:
        A `HorizontalFrame` describing the building's recovered
        orientation, with `right`/`forward` set to the two dominant,
        perpendicular wall directions.
    """
    # Keep only normals that point mostly sideways -- these are the ones
    # that could plausibly belong to a vertical wall, as opposed to a floor
    # or ceiling.
    normals = np.asarray(normals, dtype=float)
    up = np.asarray(up, dtype=float)
    up_norm = float(np.linalg.norm(up))
    up = (
        up / up_norm
        if up.shape == (3,) and np.isfinite(up_norm) and up_norm > 1e-9
        else np.array([0.0, 0.0, 1.0])
    )
    if normals.ndim != 2 or normals.shape[1] != 3:
        normals = np.empty((0, 3), dtype=float)
    horizontal = normals - np.outer(normals @ up, up)
    magnitude = np.linalg.norm(horizontal, axis=1)
    vertical_enough = magnitude > 0.85
    horizontal = horizontal[vertical_enough] / magnitude[vertical_enough, None]
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) == len(normals):
            weights = weights[vertical_enough]
        else:
            weights = None

    right, forward = _orthonormal_basis(up)
    if not len(horizontal):
        return HorizontalFrame(
            up=up,
            right=right,
            forward=forward,
            yaw=0.0,
            manhattan_fraction=0.0,
        )

    angles = np.arctan2(horizontal @ forward, horizontal @ right)
    # Multiplying each angle by 4 folds the four right-angle directions
    # (0, 90, 180, 270 degrees) on top of each other, so a plain average
    # of the resulting angles doesn't cancel out to zero.
    if weights is not None:
        weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
        if len(weights) != len(angles) or weights.sum() <= 0:
            weights = None
    resultant = np.exp(4j * angles)
    mean = (
        np.average(resultant, weights=weights)
        if weights is not None
        else resultant.mean()
    )
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
    """Picks some arbitrary pair of directions perpendicular to `up`, to use as a starting point.

    This doesn't need to line up with the building in any particular way --
    it just needs to be a valid, perpendicular pair of horizontal
    directions that `estimate_horizontal_frame` can then rotate to match
    the building's actual walls.

    Args:
        up: The world-space unit vector to build the perpendicular
            directions against.

    Returns:
        A `(right, forward)` pair of unit vectors, perpendicular to each
        other and to `up`.
    """
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
    """Picks out the points that most likely belong to a wall, rather than a floor or ceiling.

    This does two simple checks on each point. First, is it at a height
    that's clearly between the floor and the ceiling, with some margin left
    out at each end (so floor clutter and ceiling fixtures don't sneak in)?
    Second, does its surface normal point mostly sideways rather than
    straight up or down, the way a wall's surface would? Only points that
    pass both checks are kept, since those are the ones actually useful for
    finding walls.

    Args:
        points: World-space points, as an (N, 3) array.
        normals: World-space unit normals, one per point, as an (N, 3)
            array.
        gravity: A `GravityEstimate` supplying the floor height and,
            optionally, the ceiling height.
        up: A world-space unit vector pointing straight up, against
            gravity.
        margin: How much height, in metres, to leave out at both the floor
            and ceiling ends of the band, to avoid catching floor or
            ceiling clutter.

    Returns:
        A boolean mask, as an (N,) array, that is true wherever a point
        both sits in the wall-height band and has a wall-like (mostly
        sideways-pointing) normal.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    up = np.asarray(up, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals must have the same shape as points")
    up_norm = float(np.linalg.norm(up))
    if up.shape != (3,) or not np.isfinite(up_norm) or up_norm <= 1e-9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up = up / up_norm
    heights = points @ up
    # Never use the observed point-cloud extent as a ceiling.  Without an
    # observed ceiling, the configured upper wall-band bound is the only
    # deterministic safe limit available.
    if gravity.ceiling_observed and gravity.ceiling_height is not None:
        upper = gravity.ceiling_height - margin
    else:
        upper = gravity.floor_height + 1.9
    in_band = (heights > gravity.floor_height + margin) & (heights < upper)
    vertical = np.abs(normals @ up) < 0.35
    return in_band & vertical


def _axis_deviation_degrees(normal: np.ndarray) -> float:
    """Return distance to the nearest Manhattan normal, in degrees."""
    angle = float(np.degrees(np.arctan2(normal[1], normal[0])))
    nearest = round(angle / 90.0) * 90.0
    return abs(angle - nearest)


def _nearest_manhattan_normal(normal: np.ndarray) -> np.ndarray:
    """Canonicalize a normal to the nearest one of the two plan axes."""
    angle = float(np.arctan2(normal[1], normal[0]))
    quarter = round(angle / (np.pi / 2.0)) % 4
    result = np.array(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarter],
        dtype=float,
    )
    return _canonical_normal(result)


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    """Choose one sign for a line normal so geometry ordering is stable."""
    result = np.asarray(normal, dtype=float)
    result = result / max(float(np.linalg.norm(result)), 1e-12)
    if result[0] < -1e-9 or (abs(result[0]) <= 1e-9 and result[1] < 0):
        result = -result
    return result


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
    quarantine_off_axis: bool = True,
) -> list[WallSegment]:
    """Finds straight wall lines in a cloud of plan-space points, one wall at a time.

    This works through the points repeatedly, each time finding the single
    strongest straight line left in whatever points haven't been claimed by
    an earlier wall yet, removing those points, and moving on -- this
    general strategy is called "sequential RANSAC". Once a line is found,
    its points get refit more precisely (RANSAC's own line is just a rough
    first guess), and then the along-the-line spread of those points is
    broken up into separate runs wherever there's a gap, since one straight
    line in plan space might really correspond to two separate walls with a
    doorway or hallway between them. Each run is only kept as a real wall if
    it's long enough and if the points found actually cover a healthy
    fraction of what you'd expect to see if the whole run were a solid,
    fully-scanned wall -- a sparse handful of points is more likely noise
    than an actual wall.

    Args:
        plan_points: Plan-space points to search, as an (N, 2) array,
            typically already restricted to wall-like points via
            `wall_band_mask`.
        heights: The world-space height of each `plan_points` row, as an
            (N,) array.
        inlier_threshold: How close a point needs to be to a candidate
            line, in metres, to count as supporting it.
        min_inliers: The fewest points a candidate line needs behind it
            before it's worth pursuing further.
        min_length: The shortest along-line run, in metres, that's still
            kept as a wall.
        max_walls: A hard cap on how many walls this function will return,
            just to keep runaway cases in check.
        angle_tolerance_degrees: Passed straight through to `_ransac_line`
            (currently unused there).
        point_spacing: The typical distance, in metres, between
            neighbouring points in the reconstruction. This is used to
            estimate how many points a fully-observed run of a given size
            "should" have.
        min_coverage: The smallest fraction of a run's expected point count
            that must actually be present for that run to be trusted as a
            real wall, rather than dropped as too sparse.

    Returns:
        The detected wall segments, sorted from longest to shortest, with
        `index` reassigned to match that order.
    """
    plan_points = np.asarray(plan_points, dtype=float)
    heights = np.asarray(heights, dtype=float).reshape(-1)
    if plan_points.ndim == 1 and plan_points.size == 0:
        plan_points = plan_points.reshape(0, 2)
    if plan_points.ndim != 2 or plan_points.shape[1] != 2:
        raise ValueError("plan_points must have shape (N, 2)")
    if len(plan_points) != len(heights):
        raise ValueError("plan_points and heights must contain the same number of rows")
    if plan_points.size == 0:
        return []
    finite = np.isfinite(plan_points).all(axis=1) & np.isfinite(heights)
    plan_points, heights = plan_points[finite], heights[finite]
    # RANSAC has a seeded generator, but its tie behaviour still depends on
    # input order.  Canonical ordering makes the complete wall result stable
    # when frames/voxels are presented in a different order.
    order = np.lexsort((heights, plan_points[:, 1], plan_points[:, 0]))
    plan_points, heights = plan_points[order], heights[order]
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

        # RANSAC's line is just a rough guess from two sample points, so
        # refit it properly now using every point it found.
        inlier_points = subset[inlier_mask]
        normal, offset = _refit_line(inlier_points)
        axis_deviation = _axis_deviation_degrees(normal)
        is_off_axis = axis_deviation > angle_tolerance_degrees
        if not is_off_axis:
            # Manhattan fitting is exact once a candidate is within the
            # angular gate: fit the offset against the quantized normal.
            normal = _nearest_manhattan_normal(normal)
            offset = float(np.median(inlier_points @ normal))
        residual = inlier_points @ normal - offset
        direction = np.array([-normal[1], normal[0]])
        projection = inlier_points @ direction

        # One straight line can cover more than one real wall (for example,
        # two wall segments either side of a doorway), so split it into
        # separate runs wherever the points along it have a gap.
        for lo, hi, count in _contiguous_runs(np.sort(projection), gap=0.35):
            run_length = hi - lo
            if run_length < min_length:
                continue
            expected = (run_length / point_spacing) * (band_height / point_spacing)
            if count < min_coverage * expected:
                continue
            base = normal * offset
            wall = WallSegment(
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
                    quarantined=bool(quarantine_off_axis and is_off_axis),
                    confidence=_derive_wall_confidence(int(count), float(np.sqrt((residual**2).mean()))),
                    fit_quality=_derive_wall_confidence(int(count), float(np.sqrt((residual**2).mean()))),
                    provenance="wall-band point cloud",
                )
            if is_off_axis:
                wall.tags.append("off-axis")
            walls.append(wall)
        remaining = remaining[~inlier_mask]

    walls.sort(key=_wall_output_rank)
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
    """Guesses a line by repeatedly trying two random points, and keeps whichever guess fits best.

    This is the classic RANSAC ("random sample consensus") approach: pick
    two points at random, draw the line through them, count how many of the
    other points land close enough to that line to plausibly be on the same
    wall, and remember whichever random guess collected the most support.
    Trying many random pairs like this is a cheap, effective way to find a
    good line even when a lot of the points are unrelated noise or belong
    to other walls.

    Args:
        points: Plan-space points to fit a line to, as an (N, 2) array.
        threshold: How close a point needs to be to a candidate line, in
            metres, to count as supporting it.
        rng: A seeded random number generator used to pick the random pairs
            of points, so results are reproducible.
        angle_tolerance_degrees: Accepted for consistency with other
            functions in this module, but not currently used to filter
            candidates here.
        iterations: How many random pairs of points to try before giving
            up and returning the best one found.

    Returns:
        A `(normal, offset, inlier_mask)` tuple for whichever candidate
        line collected the most support. If no valid line could be formed
        at all (for example, every random pair happened to be too close
        together), this returns an arbitrary normal, a zero offset, and a
        mask that is false everywhere.
    """
    best_count = 0
    best: tuple[np.ndarray, float, np.ndarray] | None = None
    count = len(points)
    if count < 2:
        return np.array([1.0, 0.0]), 0.0, np.zeros(count, bool)

    for _ in range(iterations):
        a, b = rng.choice(count, size=2, replace=False)
        delta = points[b] - points[a]
        length = np.linalg.norm(delta)
        if length < 0.3:
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
    """Fits the best straight line through a set of points, minimising how far off each point is.

    Unlike the rough 2-point guess RANSAC uses, this looks at every point
    at once and finds the line that keeps all of them, on average, as close
    as possible (measuring closeness perpendicular to the line, not just
    vertically) -- the standard "total least squares" approach for line
    fitting.

    Args:
        points: Plan-space points to fit a line through, as an (N, 2)
            array, typically the set of points RANSAC already found to
            agree on roughly the same line.

    Returns:
        A `(normal, offset)` pair: the fitted line's 2D unit normal and its
        signed offset, such that every point `x` on the line satisfies
        `normal . x = offset`.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.array([1.0, 0.0]), 0.0
    if len(points) == 1:
        return np.array([1.0, 0.0]), float(points[0, 0])
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = _canonical_normal(vh[-1])
    return normal, float(normal @ centroid)


def _contiguous_runs(sorted_values: np.ndarray, gap: float):
    """Breaks a sorted list of 1D positions into separate groups wherever there's a big enough gap.

    This is used to split the points along one detected wall line into
    separate physical wall segments -- for example, if there's a doorway or
    a hallway partway along the line, the points on either side of that gap
    should end up as two separate walls rather than one long one.

    Args:
        sorted_values: A 1D array of positions, already sorted from
            smallest to largest.
        gap: How much space between two consecutive values is enough to
            treat them as belonging to different runs.

    Yields:
        A `(lo, hi, count)` tuple for each run: the lowest and highest
        position in the run, and how many values fall inside it. A "run"
        made of just a single value is skipped, since it can't support a
        useful wall segment.
    """
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
    """Cleans up duplicate and near-duplicate walls left over from wall detection.

    Because `extract_walls` looks for lines one at a time, it sometimes
    finds the same real wall twice -- for instance, a slightly noisy scan
    can produce two almost-identical, almost-overlapping line fits for what
    is actually a single flat wall. Those really should become one wall.
    But there's a second, very different situation that looks superficially
    similar: a wall with something sitting a little in front of it, like a
    baseboard, a cabinet, or a radiator. That clutter creates its own
    roughly-parallel surface a few centimetres in front of the real wall,
    and it should NOT be merged into the wall -- it's a different, separate
    surface, and merging it in would corrupt the wall's true position.

    This function tells the two cases apart using distance: two walls that
    are extremely close together (within `offset_tolerance`) are treated as
    the same real surface seen twice, and get merged into one, combining
    their evidence. Two walls that are further apart, but still fairly
    close (within `parallel_tolerance`) and that overlap along their
    length, are treated as the clutter-in-front case -- rather than being
    merged in and dragging the wall's fitted position off to one side, the
    weaker (clutter) surface is dropped entirely, and the stronger wall
    behind it is simply tagged `"clutter-in-front"` as a note that
    something was found sitting in front of it.

    Args:
        walls: Candidate wall segments to clean up, typically straight from
            `extract_walls`, in any order.
        offset_tolerance: How close two walls need to be, in metres, to be
            treated as two readings of the very same surface and merged
            together.
        parallel_tolerance: How close two walls need to be, in metres, to
            be considered related at all (either as duplicates or as the
            clutter-in-front case). Anything further apart than this is
            treated as unrelated and left alone.
        min_overlap: How much the two walls need to overlap along their
            length, in metres, for a wall in the `parallel_tolerance` range
            (but not close enough to merge) to be tagged as clutter rather
            than simply ignored as unrelated.
        angle_tolerance_degrees: How close to parallel two walls' normals
            need to be, in degrees, before they're even compared against
            each other at all.
        gap_tolerance: Some extra slack, in metres, allowed along the
            wall's length when deciding whether a fragment being merged in
            actually overlaps the target wall's existing span, so two
            fragments with a small real gap between them can still merge.

    Returns:
        The cleaned-up wall segments, with duplicates merged away, sorted
        longest-first, `index` reassigned to match, and each wall's `tags`
        sorted and de-duplicated.
    """
    cosine_limit = np.cos(np.radians(angle_tolerance_degrees))
    # Process the most strongly-supported, longest walls first, so that
    # weaker duplicate fragments get folded into a solid "target" wall
    # rather than the other way around.
    # Work on copies so calling this routine twice with a different input
    # ordering cannot observe the first call's absorbed offsets/counts.
    ordered = sorted((deepcopy(wall) for wall in walls), key=_wall_rank)
    kept: list[WallSegment] = []

    for wall in ordered:
        if wall.quarantined:
            # Keep the diagnostic object out of the topology candidates.  A
            # non-Manhattan line must never win a duplicate merge and move a
            # valid wall off its building axis.
            kept.append(wall)
            continue
        for target in kept:
            if target.quarantined:
                continue
            if abs(float(target.normal @ wall.normal)) < cosine_limit:
                continue

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
                # Close enough to be the same real wall seen twice --
                # combine the two into one, unless the fragment doesn't
                # actually sit anywhere near the target's existing span.
                if lo > target.length + gap_tolerance or hi < -gap_tolerance:
                    continue
                _absorb(target, wall, lo, hi)
                break
            if overlap >= min_overlap:
                # Further away but still overlapping -- likely clutter
                # sitting in front of the real wall, not the wall itself.
                # Tag it and keep the two surfaces separate.
                target.tags.append("clutter-in-front")
                break
        else:
            kept.append(wall)

    kept.sort(key=_wall_output_rank)
    for position, wall in enumerate(kept):
        wall.index = position
        wall.tags = sorted(set(wall.tags))
    return kept


def _wall_geometry_key(wall: WallSegment) -> tuple[float, ...]:
    """Canonical geometry key used for deterministic wall ordering."""
    start = tuple(np.round(np.asarray(wall.start, dtype=float), 8))
    end = tuple(np.round(np.asarray(wall.end, dtype=float), 8))
    normal = tuple(np.round(_canonical_normal(wall.normal), 8))
    return (*normal, round(float(wall.offset), 8), *start, *end)


def _wall_rank(wall: WallSegment) -> tuple:
    return (-int(wall.inlier_count), -round(wall.length, 8), round(wall.residual_rms, 8), _wall_geometry_key(wall))


def _wall_output_rank(wall: WallSegment) -> tuple:
    return (-round(wall.length, 8), -int(wall.inlier_count), _wall_geometry_key(wall))


def _absorb(target: WallSegment, other: WallSegment, lo: float, hi: float) -> None:
    """Folds one wall's evidence into another, treating them as two readings of the same surface.

    This combines the two walls' fitted positions as a weighted average
    (giving more say to whichever one had more supporting points), stretches
    `target`'s endpoints out to cover `other`'s extent as well, and merges
    the bookkeeping (how much was actually observed, the height range, and
    the tags) from both. `target` is updated in place; `other` is expected
    to be thrown away by the caller right after this runs.

    Args:
        target: The wall being extended and updated in place -- normally
            the one with more supporting evidence of the two.
        other: The wall being folded into `target` and then discarded.
        lo: The nearer endpoint of `other`, expressed in `target`'s own
            along-the-wall coordinates (via `target.along(...)`).
        hi: The farther endpoint of `other`, in the same coordinates.
    """
    old_length = target.length
    old_observed = target.observed_span
    other_start = float(target.along(other.start[None])[0])
    other_end = float(target.along(other.end[None])[0])
    other_lo, other_hi = sorted((other_start, other_end))

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
    new_lo, new_hi = min(0.0, lo), max(old_length, hi)
    target.start = origin + direction * new_lo
    target.end = origin + direction * new_hi

    observed_intervals = [
        (old_observed[0], old_observed[1]),
        (other_lo + other.observed_span[0], other_lo + other.observed_span[1]),
    ]
    observed_lo = min(interval[0] for interval in observed_intervals) - new_lo
    observed_hi = max(interval[1] for interval in observed_intervals) - new_lo
    target.observed_span = (
        float(np.clip(observed_lo, 0.0, new_hi - new_lo)),
        float(np.clip(observed_hi, 0.0, new_hi - new_lo)),
    )
    target.inlier_count = total
    target.height_range = (
        min(target.height_range[0], other.height_range[0]),
        max(target.height_range[1], other.height_range[1]),
    )
    target.tags = sorted(set(target.tags) | set(other.tags))


def _ray_crosses_wall(
    camera: np.ndarray, target: np.ndarray, wall: WallSegment, margin_fraction: float
) -> np.ndarray:
    """Checks, for many camera-to-point rays at once, whether each one passes straight through a wall.

    This is the core geometric test behind `filter_occluded_walls`: if the
    straight line from where the camera was standing to a point it
    supposedly measured has to pass through another, solid wall along the
    way, that measurement doesn't make physical sense -- you can't see
    through a wall -- so it's a sign the point (and the "wall" it's
    supporting) might actually be a stray reflection or an error. This
    checks many rays against one candidate wall at once, for speed.

    Under the hood, each ray and the wall's own segment are described as
    parametric lines, and the function solves for where they'd cross: `t`
    is how far along the ray (from camera to point) that crossing happens,
    and `s` is how far along the wall's own segment it happens. A crossing
    only counts as a real obstruction if it happens strictly between the
    camera and the point (not before the camera or beyond the point) and
    strictly within the solid part of the wall itself (not off past one of
    its ends).

    Args:
        camera: The plan-space starting point of each ray (the camera
            position each measurement was taken from), as an (N, 2) array.
        target: The plan-space end point of each ray, aligned with
            `camera`, as an (N, 2) array.
        wall: The wall segment being tested as a possible obstruction.
        margin_fraction: How much of `wall`'s length, as a fraction, to
            exclude from both ends of its span when deciding whether a
            crossing counts -- this avoids treating a ray that just grazes
            past the very corner of the wall as a real obstruction.

    Returns:
        A boolean mask, as an (N,) array, that is true wherever the ray
        from `camera[i]` to `target[i]` passes through the solid,
        margin-trimmed part of `wall`, strictly between the two ends of the
        ray.
    """
    ray = target - camera
    wall_vector = wall.end - wall.start
    denominator = ray[:, 0] * wall_vector[1] - ray[:, 1] * wall_vector[0]
    # A zero (or near-zero) denominator means the ray and the wall are
    # parallel and never cross; swap in a safe placeholder so the division
    # below doesn't blow up, and mask those rays out explicitly afterward.
    parallel = np.abs(denominator) < 1e-9
    safe_denominator = np.where(parallel, 1.0, denominator)

    diff = wall.start - camera
    t = (diff[:, 0] * wall_vector[1] - diff[:, 1] * wall_vector[0]) / safe_denominator
    s = (diff[:, 0] * ray[:, 1] - diff[:, 1] * ray[:, 0]) / safe_denominator

    return (
        ~parallel
        & (t > 0.02) & (t < 0.98)
        & (s > margin_fraction) & (s < 1 - margin_fraction)
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
    """Throws out candidate walls that could only have been seen by looking straight through another wall.

    Occasionally, a "wall" gets detected in a spot that doesn't actually
    make physical sense once you consider where the camera was standing
    when it captured the points that support it -- for instance, a
    reflection in a mirror or a glass door can look, geometrically, like a
    surface sitting in the next room, on the far side of a real wall. This
    function catches that by checking, for each candidate wall, whether the
    straight-line path from the camera to the points supporting it had to
    pass through some other, sturdier wall along the way. If most of a
    wall's points fail that check, the wall is dropped as almost certainly
    an artifact rather than a real surface.

    To keep this efficient and conservative, a candidate wall is only ever
    tested against other walls that are more strongly supported and long
    enough to plausibly be a real blocking surface (its "blockers") -- it's
    never tested against weaker or shorter walls, since a flimsy scrap
    of a wall isn't good evidence that a stronger wall must be fake. And a
    wall is only dropped if a clear majority of its near points turn out to
    be blocked, not just a handful, since a few odd points crossing a wall
    can happen by chance even for a perfectly real wall.

    Args:
        walls: Candidate wall segments to check, typically already passed
            through `merge_collinear`.
        frame: The `HorizontalFrame` used to project the world-space
            `points` and `origins` into plan space.
        points: World-space points that supported wall fitting, as an
            (M, 3) array.
        origins: The world-space camera position that each row of `points`
            was observed from, as an (M, 3) array, aligned with `points`.
        band: How close a point needs to be to a wall's line, in metres, to
            count as "on" that wall for this test.
        corner_margin: How much of a candidate wall's span, in metres, to
            exclude from both ends when picking which of its points to
            test, so points right at a corner (which are more prone to
            false crossings) don't skew the result.
        min_points: The fewest nearby points a wall needs before this test
            is even run on it; a wall with too few nearby points is kept
            without being checked, since there isn't enough evidence either
            way.
        min_blocker_length: How long another wall needs to be, in metres,
            before it's considered a plausible obstruction at all.
        occlusion_fraction: What fraction of a candidate's nearby points
            need to be found blocked before the candidate is dropped
            entirely.

    Returns:
        A `(kept, dropped)` pair: the surviving wall segments, sorted
        longest-first with `index` reassigned, and a count of how many
        walls were removed for being occluded.
    """
    plan = frame.to_plan(points)
    camera = frame.to_plan(origins)
    ordered = sorted((w for w in walls if not w.quarantined), key=_wall_rank)
    dropped = sum(1 for wall in walls if wall.quarantined)
    kept: list[WallSegment] = []

    for wall in ordered:
        # Only stronger, longer walls are trusted enough to act as
        # "blockers" -- a candidate is never thrown out on the say-so of a
        # weaker or shorter one.
        blockers = [
            other
            for other in walls
            if other is not wall
            and not other.quarantined
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
            kept.append(wall)
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

    kept.sort(key=_wall_output_rank)
    for position, wall in enumerate(kept):
        wall.index = position
    return kept, dropped


def _segment_intersection(
    a: WallSegment, b: WallSegment
) -> tuple[np.ndarray, float, float] | None:
    """Finds where two walls' lines would cross, if extended forever in both directions.

    This treats each wall as an infinite line (ignoring where it actually
    starts and ends) and solves for the one point where the two lines meet.
    It also reports how far along each wall's own direction that crossing
    point falls, measured from that wall's `start` -- which lets the caller
    figure out whether the crossing actually happens within the wall's real,
    finite extent, or off somewhere past one of its ends.

    Args:
        a: The first wall segment; only the line it defines is used, not
            its actual `start`/`end` extent.
        b: The second wall segment, likewise treated as an infinite line.

    Returns:
        A `(point, u_a, u_b)` tuple if the two lines meet at a clear,
        well-defined angle: `point` is the crossing point in plan space,
        and `u_a`/`u_b` are how far along each wall's own direction that
        point falls (0 at that wall's `start`). Returns `None` when the two
        lines are close enough to parallel that a crossing point can't be
        reliably pinned down.
    """
    da, db = a.direction, b.direction
    denominator = da[0] * db[1] - da[1] * db[0]
    if abs(denominator) < 0.15:
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
    """Cleans up cases where two walls appear to cut through each other partway along their length.

    Real walls don't pass through each other's middle -- if the detected
    geometry shows that happening, something in the earlier detection
    stages went slightly wrong, and this function decides how to fix it.
    There are two different situations that can produce this kind of
    crossing, and they're handled differently:

    The first is a small overshoot at a T-junction, where one wall (say, an
    interior partition) is supposed to stop right where it meets another
    wall, but the detected segment runs a little too far past that meeting
    point. In that case, the fix is simply to trim the overshooting wall
    back to the crossing point -- nothing is thrown away, the wall's extent
    is just corrected.

    The second is a case where a shorter, weaker wall genuinely cuts across
    the middle of a longer, better-supported one -- which usually means the
    shorter one is spurious (some kind of detection error) rather than a
    real wall. When the overlap is too large to be explained as a small
    T-junction overshoot, this function assumes that's what's happening and
    simply removes the weaker of the two walls instead of trimming it.

    Args:
        walls: Wall segments to check, pairwise, for this kind of interior
            crossing. A shallow copy of the list is worked on internally;
            surviving `WallSegment` objects may be mutated (trimmed) in
            place.
        interior_margin: How close to either end of a wall, in metres, a
            crossing can be and still be treated as a normal corner rather
            than a true mid-span crossing that needs fixing.
        max_trim: The largest overshoot, in metres, past the crossing point
            that's still treated as a simple T-junction overshoot to trim.
            Beyond this distance, the crossing is treated as the
            weaker-wall case instead, and that wall is dropped.

    Returns:
        The surviving wall segments (the same objects, some of them
        trimmed), with each wall's `tags` deduplicated and sorted. The
        order and length of the returned list may differ from `walls`,
        since walls can be removed.
    """
    walls = [wall for wall in walls if not wall.quarantined]
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

                # How far each wall runs past the crossing point on its
                # shorter side -- a small overhang looks like a T-junction
                # that just needs trimming; a big one looks like a wall
                # cutting straight across another.
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
                    # Too much overlap to be a simple overshoot -- treat
                    # the weaker-supported wall as spurious and drop it.
                    weaker = a if a.inlier_count < b.inlier_count else b
                    walls.remove(weaker)
                changed = True
                break
            if changed:
                break

    for wall in walls:
        wall.tags = sorted(set(wall.tags))
    return walls


def snap_corners(
    walls: list[WallSegment],
    max_extension: float = 0.45,
    max_trim: float = 0.30,
) -> int:
    """Compatibility wrapper around the global wall-graph node solver.

    New pipeline code must call :func:`wall_graph.solve_wall_graph` directly.
    This name remains for callers of the Phase 1 API, but no longer performs
    independent endpoint proposals: every shared corner is solved jointly
    from the incident wall lines.
    """
    from .wall_graph import solve_wall_graph

    # Keep graph clustering local and separate from the legacy extension
    # budget.  The old wrapper used the larger value as a global node
    # tolerance, which could silently turn a large gap into a corner.
    extension = max(float(max_extension), float(max_trim))
    graph = solve_wall_graph(
        walls,
        node_tolerance=0.12,
        min_length=1e-9,
        min_confidence=0.0,
        max_endpoint_extension=extension,
    )
    solved_by_index = {wall.index: wall for wall in graph.candidates}
    for wall in walls:
        solved = solved_by_index.get(wall.index)
        if solved is None:
            continue
        wall.start = solved.start.copy()
        wall.end = solved.end.copy()
        wall.tags = list(solved.tags)
        wall.quarantined = solved.quarantined
        wall.snap_status = solved.snap_status
    return graph.snapped_endpoint_count


def snap_to_frame(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    tolerance_degrees: float = 8.0,
    min_confidence: float = 0.25,
    quarantine_weak: bool = True,
) -> list[WallSegment]:
    """Straightens out walls that are already close to square with the building, so they're exactly square.

    Real-world detection is noisy, so a wall that's actually meant to be
    perfectly perpendicular to its neighbours might come out of the earlier
    stages a couple of degrees off. This function looks at each wall's
    normal direction and, if it's already close enough to one of the
    building's own axes (the `right`/`forward` directions from
    `estimate_horizontal_frame`), rounds it exactly onto that axis. A wall
    that's too far from any axis to be a confident match is left exactly as
    it was, but gets tagged `"off-axis"` so later stages know it wasn't
    straightened.

    Args:
        walls: Wall segments to straighten; mutated in place for any wall
            close enough to an axis to be snapped.
        frame: The `HorizontalFrame` whose `right`/`forward` axes define
            the plan-space directions a wall's normal gets rounded toward.
        tolerance_degrees: The largest angle, in degrees, a wall's normal
            may be away from the nearest axis-aligned direction and still
            be snapped onto it.
        min_confidence: Minimum fit confidence required before a candidate is
            allowed to snap. Weak candidates are retained for diagnostics but
            are not forced onto a Manhattan axis.
        quarantine_weak: If true, weak unsnapped candidates are excluded from
            graph topology while remaining in the returned candidate list.

    Returns:
        The same `walls` list, mutated in place and also returned, so
        calls can be chained together.
    """
    if not np.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    for wall in walls:
        if wall.quarantined:
            continue
        if wall.confidence < min_confidence:
            wall.tags.append("low-confidence")
            wall.snap_status = "rejected-low-confidence"
            if quarantine_weak:
                wall.quarantined = True
            continue
        old_endpoints = np.vstack([wall.start, wall.end])
        angle = np.arctan2(wall.normal[1], wall.normal[0])
        quarter = round(angle / (np.pi / 2)) % 4
        snapped = quarter * (np.pi / 2)
        if abs(np.degrees(angle - snapped)) > tolerance_degrees:
            wall.tags.append("off-axis")
            wall.snap_status = "rejected-off-axis"
            wall.quarantined = True
            continue
        normal = _nearest_manhattan_normal(wall.normal)
        offset = float(normal @ wall.midpoint)
        direction = np.array([-normal[1], normal[0]])
        half = 0.5 * wall.length
        centre = normal * offset + direction * (direction @ wall.midpoint)
        wall.normal, wall.offset = normal, offset
        wall.start, wall.end = centre - direction * half, centre + direction * half
        wall.snap_residual = float(
            np.sqrt(np.mean((old_endpoints @ normal - offset) ** 2))
        )
        wall.snap_status = "snapped"
    return walls
