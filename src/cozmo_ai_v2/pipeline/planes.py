from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .geometry import GravityEstimate

if TYPE_CHECKING:
    from .geometry_diagnostics import GeometryDiagnostics


@dataclass
class TLSPlane:
    """A structured plane record exchanged with the TLS geometry stages.

    The measurement stage deliberately depends on this small contract rather
    than on a raster or on a particular plane-fitting implementation.  A
    plane is represented by ``normal . x = offset`` in world coordinates;
    optional inlier points and provenance fields carry the evidence needed to
    make honest confidence and tolerance estimates.  ``start``/``end`` are
    optional finite extents for wall planes and are expressed in world
    coordinates when present.

    The fields after ``role`` are intentionally permissive.  Stage 6 can add
    richer provenance without making Stage 9 depend on that branch landing at
    the same time.
    """

    id: str | int
    normal: np.ndarray
    offset: float
    role: str = "wall"
    inlier_points: np.ndarray | None = None
    inlier_count: int = 0
    residual_rms: float = 0.0
    support_density: float | None = None
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    observed: bool = True
    pose_provenance: str = "unknown"
    depth_provenance: str = "unknown"
    calibration_status: str = "uncalibrated"
    room_id: int | None = None
    tags: list[str] = field(default_factory=list)
    height_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float).reshape(-1)
        if normal.shape != (3,):
            raise ValueError("TLS plane normal must have shape (3,)")
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("TLS plane normal must be finite and non-zero")
        self.normal = normal / norm
        self.offset = float(self.offset)
        if not np.isfinite(self.offset):
            raise ValueError("TLS plane offset must be finite")
        if self.inlier_points is not None:
            points = np.asarray(self.inlier_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("TLS plane inlier_points must have shape (N, 3)")
            self.inlier_points = points[np.isfinite(points).all(axis=1)]
            self.inlier_count = max(int(self.inlier_count), len(self.inlier_points))
        self.inlier_count = max(int(self.inlier_count), 0)
        self.residual_rms = max(float(self.residual_rms), 0.0)
        if self.support_density is not None:
            density = float(self.support_density)
            self.support_density = density if np.isfinite(density) and density >= 0 else None
        if self.start is not None:
            self.start = np.asarray(self.start, dtype=float).reshape(-1)
            if self.start.shape != (3,):
                raise ValueError("TLS plane start must have shape (3,)")
        if self.end is not None:
            self.end = np.asarray(self.end, dtype=float).reshape(-1)
            if self.end.shape != (3,):
                raise ValueError("TLS plane end must have shape (3,)")
        self.tags = list(self.tags)
        if self.height_range is not None:
            heights = np.asarray(self.height_range, dtype=float).reshape(-1)
            if len(heights) < 2 or not np.isfinite(heights[:2]).all():
                self.height_range = None
            else:
                self.height_range = (float(heights[0]), float(heights[1]))

    @classmethod
    def from_any(cls, value: object, *, default_id: str | int = "plane") -> "TLSPlane":
        """Coerce a mapping or Stage 6-like object into the contract."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            get = value.get
        else:
            get = lambda key, default=None: getattr(value, key, default)
        normal = get("normal", get("unit_normal"))
        offset = get("offset", get("d", get("distance")))
        if normal is None:
            coefficients = get("coefficients", get("plane_equation"))
            if coefficients is not None:
                coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
                if coefficients.shape == (4,):
                    normal = coefficients[:3]
                    # The conventional TLS equation is ax+by+cz+d=0.
                    offset = -float(coefficients[3])
        if normal is None or offset is None:
            raise ValueError("TLS plane requires normal and offset")
        inliers = get("inlier_points", get("inliers"))
        if isinstance(inliers, (int, float, np.integer, np.floating)):
            inlier_count = int(inliers)
            inliers = None
        else:
            inlier_count = int(get("inlier_count", get("support_points", 0)) or 0)
        provenance = get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        calibration = get("calibration_status", get("calibration", provenance.get("calibration", "uncalibrated")))
        if isinstance(calibration, bool):
            calibration = "calibrated" if calibration else "uncalibrated"
        return cls(
            id=get("id", get("plane_id", default_id)),
            normal=normal,
            offset=offset,
            role=str(get("role", get("kind", "wall"))),
            inlier_points=inliers,
            inlier_count=inlier_count,
            residual_rms=float(get("residual_rms", get("residual_rms_m", get("rms", 0.0))) or 0.0),
            support_density=get("support_density", get("density")),
            start=get("start", get("endpoint_a")),
            end=get("end", get("endpoint_b")),
            observed=bool(get("observed", get("is_observed", True))),
            pose_provenance=str(get("pose_provenance", get("pose_source", provenance.get("pose", "unknown")))),
            depth_provenance=str(get("depth_provenance", get("depth_source", provenance.get("depth", "unknown")))),
            calibration_status=str(calibration),
            room_id=get("room_id"),
            tags=list(get("tags", []) or []),
            height_range=get("height_range", get("vertical_extent")),
        )


@dataclass
class TLSPlaneModel:
    """Container contract for structured TLS planes and their intersections."""

    planes: list[TLSPlane] = field(default_factory=list)
    intersections: list[object] = field(default_factory=list)
    floor_plane: TLSPlane | None = None
    ceiling_planes: list[TLSPlane] = field(default_factory=list)
    pose_provenance: str = "unknown"
    depth_provenance: str = "unknown"
    calibration_status: str = "uncalibrated"

    def __post_init__(self) -> None:
        plane_values = (
            [self.planes]
            if isinstance(self.planes, Mapping) or hasattr(self.planes, "normal")
            else list(self.planes or [])
        )
        self.planes = [TLSPlane.from_any(plane, default_id=i) for i, plane in enumerate(plane_values)]
        if self.floor_plane is not None:
            self.floor_plane = TLSPlane.from_any(self.floor_plane, default_id="floor")
        ceiling_values = (
            [self.ceiling_planes]
            if isinstance(self.ceiling_planes, Mapping) or hasattr(self.ceiling_planes, "normal")
            else list(self.ceiling_planes or [])
        )
        self.ceiling_planes = [
            TLSPlane.from_any(plane, default_id=f"ceiling_{i}")
            for i, plane in enumerate(ceiling_values)
        ]

    @property
    def wall_planes(self) -> list[TLSPlane]:
        return [
            plane
            for plane in self.planes
            if plane.role.lower() in {"wall", "wall_face", "vertical"}
        ]


# Names used by early geometry prototypes and by the forthcoming Stage 6
# branch.  Keeping these aliases costs nothing and lets the measurement layer
# consume either spelling during the transition.
Plane3D = TLSPlane
StructuredTLSPlane = TLSPlane
TLSModel = TLSPlaneModel


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
    # A wall can be the 2D compatibility view of a fitted 3D structural
    # plane.  These fields are optional so the older wall-only API remains
    # source compatible.
    structural_plane_id: int | None = None
    inlier_indices: np.ndarray | None = None

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
        if self.inlier_indices is not None:
            indices = np.asarray(self.inlier_indices, dtype=int).reshape(-1)
            self.inlier_indices = np.unique(indices)
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


class PlaneClassification(str, Enum):
    """Semantic classes assigned to fitted metric planes.

    ``CLUTTER`` deliberately covers both planes with an intermediate
    orientation and horizontal surfaces that are neither the floor nor a
    plausible ceiling.  They remain in the output for diagnostics, but are
    quarantined from room and wall topology.
    """

    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    CLUTTER = "clutter"


PlaneType = PlaneClassification


@dataclass
class StructuralPlane:
    """A metric 3D plane fitted to a stable set of source points.

    The plane equation is ``normal . x = offset``.  ``inlier_indices`` are
    indices into the point array passed to :func:`extract_structural_planes`,
    rather than indices in a sorted or downsampled working array.  This is
    important: downstream diagnostics can trace every metric surface back to
    the original fused-cloud points.

    ``extents`` is the axis-aligned world-space size.  ``bounds_min`` and
    ``bounds_max`` preserve the corresponding corners, while
    ``in_plane_extents`` stores the two tangent-coordinate ranges used to
    reconstruct the observed footprint.  Keeping both forms makes the model
    useful to numerical callers and straightforward to serialize.
    """

    id: int
    classification: PlaneClassification | str
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    extents: np.ndarray
    inlier_indices: np.ndarray
    residual_rms: float
    residual_mean_abs: float
    residual_median: float
    residual_max: float
    point_density: float
    confidence: float
    # Acceptance evidence is kept with every retained candidate.  The
    # distance threshold is both the RANSAC support threshold and the strict
    # residual target; the adaptive threshold records the robust quality
    # allowance used for reporting noisy support.
    candidate_threshold: float = 0.0
    support_threshold: int = 0
    residual_threshold: float = 0.0
    adaptive_residual_threshold: float = 0.0
    quality_status: str = "high_confidence"
    low_confidence: bool = False
    rejection_reasons: tuple[str, ...] = ()
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None
    in_plane_extents: np.ndarray | None = None
    wall_vertical_extent: tuple[float, float] | None = None
    ceiling_observed: bool = False
    ceiling_confidence: float = 0.0
    quarantined: bool = False
    tags: list[str] = field(default_factory=list)
    # The horizontal along-wall range in a caller's HorizontalFrame.  It is
    # optional because a structural plane can be used without a plan frame.
    _horizontal_tangent: np.ndarray | None = field(default=None, repr=False)
    _horizontal_extent: tuple[float, float] | None = field(default=None, repr=False)
    _up: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        classification = self.classification
        if isinstance(classification, PlaneClassification):
            classification = classification.value
        classification = str(classification).lower()
        if classification not in {item.value for item in PlaneClassification}:
            classification = PlaneClassification.CLUTTER.value
        self.classification = classification

        normal = np.asarray(self.normal, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(normal))
        if normal.shape != (3,) or not np.isfinite(norm) or norm <= 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
            norm = 1.0
            self.quarantined = True
            self.tags.append("invalid-geometry")
        self.normal = normal / norm

        centroid = np.asarray(self.centroid, dtype=float).reshape(-1)
        if centroid.shape != (3,) or not np.isfinite(centroid).all():
            centroid = np.zeros(3, dtype=float)
            self.quarantined = True
            self.tags.append("invalid-geometry")
        self.centroid = centroid

        valid_offset = bool(np.isfinite(self.offset))
        self.offset = float(self.offset) if valid_offset else 0.0
        if not valid_offset:
            self.quarantined = True
            self.tags.append("invalid-geometry")

        self.extents = _finite_vector(self.extents, 3)
        if self.bounds_min is None:
            self.bounds_min = self.centroid - 0.5 * self.extents
        else:
            self.bounds_min = _finite_vector(self.bounds_min, 3)
        if self.bounds_max is None:
            self.bounds_max = self.centroid + 0.5 * self.extents
        else:
            self.bounds_max = _finite_vector(self.bounds_max, 3)
        self.extents = np.maximum(self.bounds_max - self.bounds_min, 0.0)

        indices = np.asarray(self.inlier_indices, dtype=int).reshape(-1)
        self.inlier_indices = np.unique(indices[indices >= 0])
        self.tags = sorted(set(self.tags))
        self.residual_rms = _finite_nonnegative(self.residual_rms)
        self.residual_mean_abs = _finite_nonnegative(self.residual_mean_abs)
        self.residual_median = _finite_nonnegative(self.residual_median)
        self.residual_max = _finite_nonnegative(self.residual_max)
        self.point_density = _finite_nonnegative(self.point_density)
        self.confidence = float(
            np.clip(self.confidence, 0.0, 1.0)
            if np.isfinite(self.confidence)
            else 0.0
        )
        self.candidate_threshold = _finite_nonnegative(self.candidate_threshold)
        self.support_threshold = max(0, int(self.support_threshold))
        self.residual_threshold = _finite_nonnegative(self.residual_threshold)
        self.adaptive_residual_threshold = _finite_nonnegative(
            self.adaptive_residual_threshold
        )
        self.quality_status = str(self.quality_status or "unknown")
        self.low_confidence = bool(
            self.low_confidence or self.quality_status == "low_confidence"
        )
        self.rejection_reasons = tuple(
            sorted({str(reason) for reason in self.rejection_reasons if str(reason)})
        )
        self.ceiling_confidence = float(
            np.clip(self.ceiling_confidence, 0.0, 1.0)
            if np.isfinite(self.ceiling_confidence)
            else 0.0
        )
        if self.classification == PlaneClassification.CLUTTER.value:
            self.quarantined = True
            if "off-orientation" in self.tags:
                self.rejection_reasons = tuple(
                    sorted(
                        set(self.rejection_reasons)
                        | {"orientation_not_horizontal_or_vertical"}
                    )
                )
            elif not self.rejection_reasons:
                self.rejection_reasons = ("not_floor_or_ceiling_height",)
        if self.classification == PlaneClassification.CEILING.value:
            self.ceiling_observed = True
        if self.wall_vertical_extent is not None:
            vertical = np.asarray(self.wall_vertical_extent, dtype=float).reshape(-1)
            if vertical.shape == (2,) and np.isfinite(vertical).all():
                self.wall_vertical_extent = (float(vertical.min()), float(vertical.max()))
            else:
                self.wall_vertical_extent = None
        if self.in_plane_extents is not None:
            in_plane = np.asarray(self.in_plane_extents, dtype=float)
            if in_plane.shape == (2, 2) and np.isfinite(in_plane).all():
                self.in_plane_extents = in_plane
            else:
                self.in_plane_extents = None

    @property
    def kind(self) -> str:
        """Alias for callers that use ``kind`` for semantic plane type."""
        return str(self.classification)

    @property
    def orientation(self) -> str:
        """Return ``horizontal``, ``vertical``, or ``other``."""
        if "horizontal" in self.tags:
            return "horizontal"
        if "vertical" in self.tags:
            return "vertical"
        return "other"

    @property
    def support_count(self) -> int:
        return int(len(self.inlier_indices))

    @property
    def inlier_count(self) -> int:
        return self.support_count

    @property
    def support(self) -> int:
        return self.support_count

    @property
    def support_indices(self) -> np.ndarray:
        """Alias for the source identities supporting this plane."""
        return self.inlier_indices

    @property
    def inlier_ids(self) -> np.ndarray:
        return self.inlier_indices

    @property
    def plane_type(self) -> str:
        return str(self.classification)

    @property
    def type(self) -> str:
        return str(self.classification)

    @property
    def label(self) -> str:
        return str(self.classification)

    @property
    def inliers(self) -> np.ndarray:
        return self.inlier_indices

    @property
    def residual_statistics(self) -> dict[str, float]:
        return {
            "rms": float(self.residual_rms),
            "mean_abs": float(self.residual_mean_abs),
            "median": float(self.residual_median),
            "max": float(self.residual_max),
        }

    @property
    def area(self) -> float:
        """Observed in-plane bounding-box area in square metres."""
        if self.in_plane_extents is None:
            return 0.0
        spans = np.maximum(self.in_plane_extents[:, 1] - self.in_plane_extents[:, 0], 0.0)
        return float(spans[0] * spans[1])

    @property
    def vertical_extent(self) -> tuple[float, float] | None:
        """Alias exposing a wall's height range along the recovered up axis."""
        return self.wall_vertical_extent

    @property
    def extent_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.bounds_min), np.asarray(self.bounds_max)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.extent_bounds

    @property
    def equation(self) -> tuple[np.ndarray, float]:
        """Return the normalized ``(normal, offset)`` plane equation."""
        return self.normal, float(self.offset)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Evaluate signed perpendicular residuals for arbitrary points."""
        values = np.asarray(points, dtype=float)
        return values @ self.normal - self.offset

    def residuals(self, points: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_distance(points))

    @property
    def is_kept(self) -> bool:
        return not self.quarantined

    def to_wall_segment(
        self,
        frame: "HorizontalFrame",
        index: int | None = None,
        points: np.ndarray | None = None,
    ) -> WallSegment | None:
        """Convert this vertical 3D plane to the existing 2D wall model.

        A vertical plane has a line as its floor-plan projection.  The
        projected normal is normalized independently, while the observed
        along-line range is obtained from the fitted plane footprint.  If
        ``points`` is supplied, the range is recomputed from the retained
        source inlier identities; otherwise the stored metric extent is used.
        No Manhattan snapping occurs here, so slanted walls are represented
        faithfully and the established ``snap_to_frame`` policy can decide
        later whether to quarantine them from room topology.
        """
        if self.classification != PlaneClassification.WALL.value:
            return None
        up = np.asarray(frame.up, dtype=float)
        up_norm = float(np.linalg.norm(up))
        up = up / up_norm if up.shape == (3,) and up_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        horizontal_normal = np.array(
            [self.normal @ frame.right, self.normal @ frame.forward], dtype=float
        )
        horizontal_norm = float(np.linalg.norm(horizontal_normal))
        if not np.isfinite(horizontal_norm) or horizontal_norm <= 1e-9:
            return None
        horizontal_normal /= horizontal_norm
        direction = np.array([-horizontal_normal[1], horizontal_normal[0]])
        centre_plan = np.asarray(frame.to_plan(self.centroid), dtype=float).reshape(2)

        along = None
        heights = None
        if points is not None and len(self.inlier_indices):
            source = np.asarray(points, dtype=float)
            valid = self.inlier_indices[self.inlier_indices < len(source)]
            source = source[valid]
            if len(source):
                plan = frame.to_plan(source)
                along = (plan - centre_plan) @ direction
                heights = source @ up
        if along is None or not len(along):
            if self._horizontal_extent is not None:
                lo, hi = self._horizontal_extent
            elif self.in_plane_extents is not None:
                # The major in-plane tangent is normally the horizontal
                # direction for a wall.  This fallback is only for planes
                # constructed manually rather than by the detector.
                lo, hi = tuple(self.in_plane_extents[0])
            else:
                lo, hi = (-0.5 * float(self.extents.max()), 0.5 * float(self.extents.max()))
        else:
            lo, hi = float(np.min(along)), float(np.max(along))
        if hi - lo <= 1e-9:
            return None
        height_range = self.wall_vertical_extent
        if heights is not None and len(heights):
            height_range = (float(np.min(heights)), float(np.max(heights)))
        wall = WallSegment(
            index=self.id if index is None else int(index),
            normal=horizontal_normal,
            offset=float(horizontal_normal @ centre_plan),
            start=centre_plan + direction * lo,
            end=centre_plan + direction * hi,
            inlier_count=self.support_count,
            residual_rms=self.residual_rms,
            observed_span=(0.0, float(hi - lo)),
            height_range=height_range or (0.0, 0.0),
            quarantined=self.quarantined,
            structural_plane_id=self.id,
            inlier_indices=self.inlier_indices.copy(),
        )
        wall.tags = sorted(set(wall.tags) | set(self.tags) - {"horizontal", "vertical"})
        return wall

    # Readable compatibility aliases used by downstream integrations.
    to_wall = to_wall_segment
    to_wall_line = to_wall_segment
    as_wall_segment = to_wall_segment

    def to_dict(self, include_inliers: bool = True) -> dict:
        """Return a JSON-ready metric representation of this plane."""
        extents = {
            "min": np.asarray(self.bounds_min).tolist(),
            "max": np.asarray(self.bounds_max).tolist(),
            "size": np.asarray(self.extents).tolist(),
            "in_plane": (
                np.asarray(self.in_plane_extents).tolist()
                if self.in_plane_extents is not None
                else []
            ),
            "area_m2": round(self.area, 6),
        }
        document = {
            "id": int(self.id),
            "classification": self.classification,
            "kind": self.classification,
            "orientation": self.orientation,
            "normal": self.normal.tolist(),
            "offset": round(float(self.offset), 6),
            "centroid": self.centroid.tolist(),
            "extents": extents,
            "support_points": self.support_count,
            "inlier_count": self.support_count,
            "residual_rms": round(float(self.residual_rms), 6),
            "residual_rms_mm": round(float(self.residual_rms * 1000.0), 3),
            "residual_mean_abs": round(float(self.residual_mean_abs), 6),
            "residual_median": round(float(self.residual_median), 6),
            "residual_max": round(float(self.residual_max), 6),
            "point_density": round(float(self.point_density), 6),
            "density": round(float(self.point_density), 6),
            "confidence": round(float(self.confidence), 6),
            "candidate_threshold": round(float(self.candidate_threshold), 6),
            "support_threshold": int(self.support_threshold),
            "residual_threshold": round(float(self.residual_threshold), 6),
            "adaptive_residual_threshold": round(
                float(self.adaptive_residual_threshold), 6
            ),
            "quality_status": self.quality_status,
            "low_confidence": bool(self.low_confidence),
            "rejection_reasons": list(self.rejection_reasons),
            "residual_statistics": {
                key: round(value, 6)
                for key, value in self.residual_statistics.items()
            },
            "wall_vertical_extent": (
                list(self.wall_vertical_extent)
                if self.wall_vertical_extent is not None
                else None
            ),
            "vertical_extent": (
                list(self.wall_vertical_extent)
                if self.wall_vertical_extent is not None
                else None
            ),
            "ceiling_observed": bool(self.ceiling_observed),
            "ceiling_confidence": round(float(self.ceiling_confidence), 6),
            "quarantined": bool(self.quarantined),
            "tags": list(self.tags),
        }
        if include_inliers:
            document["inlier_indices"] = self.inlier_indices.tolist()
        return document


# Common names for callers that prefer a shorter model name.
Plane3D = StructuralPlane
MetricPlane = StructuralPlane
Plane = StructuralPlane
PlaneModel = StructuralPlane


def _finite_vector(value: np.ndarray, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        return np.zeros(size, dtype=float)
    return result


def _finite_nonnegative(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) and value >= 0.0 else 0.0


def fit_plane_tls(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a 3D plane by total least squares using an SVD.

    The returned unit normal and offset minimize the perpendicular point-to-
    plane residual over all finite input points.  The sign is canonicalized
    by the largest-magnitude normal component, making this helper stable
    across repeated runs and independent of the SVD sign convention.

    A plane needs three non-collinear points.  Degenerate input returns a
    safe horizontal plane rather than raising, which keeps batch extraction
    conservative when a voxel or frame contains too little geometry.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if len(values) == 1:
        return np.array([0.0, 0.0, 1.0]), float(values[0, 2])
    centroid = values.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(values - centroid, full_matrices=False)
    if len(singular_values) < 3 or singular_values[1] <= 1e-9:
        # There is no unique plane for a point or a line.  The fallback is
        # finite and useful to callers that only need a safe equation; the
        # structural detector rejects this case as degenerate.
        return np.array([0.0, 0.0, 1.0]), float(centroid[2])
    normal = _canonical_plane_normal(vh[-1])
    return normal, float(normal @ centroid)


# Explicit name for callers who want to distinguish 3D TLS from the older
# 2D wall-line refit helper below.
refit_plane_tls = fit_plane_tls


def _canonical_plane_normal(normal: np.ndarray) -> np.ndarray:
    """Choose one deterministic sign for an unoriented 3D plane normal."""
    result = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    result = result / norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result = -result
    return result


def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return a stable plane hypothesis from three points, if non-collinear."""
    a, b, c = np.asarray(points, dtype=float)
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    normal = _canonical_plane_normal(normal)
    return normal, float(normal @ a)


def _plane_basis(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two deterministic tangent axes and the SVD normal."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    tangent_a = _canonical_plane_normal(vh[0])
    # The second singular vector is already orthogonal to the first/normal;
    # canonicalizing its sign keeps serialized extents stable.
    tangent_b = _canonical_plane_normal(vh[1])
    fitted_normal = _canonical_plane_normal(vh[-1])
    # SVD can choose the opposite normal sign from the caller's fit.  The
    # tangent span is unaffected, but returning the fitted normal here makes
    # all downstream residuals use exactly the same TLS equation.
    if fitted_normal @ normal < 0.0:
        fitted_normal = -fitted_normal
    return tangent_a, tangent_b, fitted_normal


def _classify_plane(
    normal: np.ndarray,
    height: float,
    floor_height: float,
    up: np.ndarray,
    horizontal_cosine: float,
    vertical_sine: float,
    min_room_height: float,
    is_floor_candidate: bool,
) -> tuple[str, bool, list[str]]:
    """Classify an unoriented plane without imposing a Manhattan grid."""
    alignment = abs(float(normal @ up))
    tags: list[str] = []
    if alignment >= horizontal_cosine:
        tags.append("horizontal")
        if is_floor_candidate:
            return PlaneClassification.FLOOR.value, False, tags
        if height - floor_height >= min_room_height:
            return PlaneClassification.CEILING.value, False, tags
        tags.append("off-orientation")
        return PlaneClassification.CLUTTER.value, True, tags
    if alignment <= vertical_sine:
        tags.append("vertical")
        return PlaneClassification.WALL.value, False, tags
    tags.append("off-orientation")
    return PlaneClassification.CLUTTER.value, True, tags


def extract_structural_planes(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    up: np.ndarray | None = None,
    *,
    floor_height: float | None = None,
    inlier_threshold: float = 0.03,
    min_inliers: int = 30,
    max_planes: int = 80,
    ransac_iterations: int = 400,
    seed: int = 0,
    horizontal_tolerance_degrees: float = 25.0,
    vertical_tolerance_degrees: float = 25.0,
    min_room_height: float = 1.8,
    min_plane_span: float = 1e-3,
    ransac_sample_size: int = 10000,
    quarantine: bool = True,
    gravity_up: np.ndarray | None = None,
    random_seed: int | None = None,
) -> list[StructuralPlane]:
    """Extract major metric planes with deterministic seeded RANSAC.

    Candidate planes are proposed from non-collinear triples, grown by a
    distance-supported region, and refit after every growth pass with 3D
    total least squares.  The RANSAC hypothesis search may use a bounded,
    deterministic working sample for large fused clouds, but final support
    and inlier identities are always evaluated against every source point.

    Horizontal planes are classified relative to ``floor_height`` (normally
    the robust result from :func:`estimate_gravity`): the lowest coherent
    plane is the floor, every coherent horizontal plane at least
    ``min_room_height`` above it is retained as a ceiling, and intermediate
    horizontal surfaces are quarantined as clutter.  This intentionally
    allows more than one ceiling plane and accepts modestly sloped/vaulted
    surfaces within the horizontal orientation tolerance.  Vertical planes
    are retained without Manhattan snapping; the existing 2D wall stage can
    decide later whether an individual wall is suitable for room topology.

    Args:
        points: Source world-space points, shape ``(N, 3)``.
        normals: Optional source surface normals, aligned with ``points``.
            They are used for plane quality diagnostics, not as a hard
            geometric gate, so noisy or missing normals cannot erase a real
            surface.
        up: Recovered world-space up direction. Defaults to +Z.
        floor_height: Existing robust floor offset. If omitted, the lowest
            extracted horizontal candidate supplies the semantic reference.
        inlier_threshold: Perpendicular distance in metres for support.
        min_inliers: Minimum support for a candidate plane.
        max_planes: Maximum number of sequentially extracted candidates.
        ransac_iterations: Number of seeded hypotheses per iteration.
        seed: Seed for the local NumPy generator; no global RNG is touched.
        horizontal_tolerance_degrees: Maximum tilt from up for horizontal
            classification.
        vertical_tolerance_degrees: Maximum tilt from a vertical wall.
        min_room_height: Minimum floor-to-ceiling separation.
        min_plane_span: Reject exact/near-collinear supports.
        ransac_sample_size: Maximum points used to score hypotheses; final
            support is always evaluated on the full remaining set.
        quarantine: Whether clutter/off-orientation planes remain in the
            returned list marked ``quarantined``. If false they are omitted.
        gravity_up: Compatibility alias for ``up``.
        random_seed: Compatibility alias for ``seed``.

    Returns:
        A deterministic list of :class:`StructuralPlane` instances with
        source inlier identities, TLS metrics, and semantic classification.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim == 1 and values.size == 0:
        values = values.reshape(0, 3)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if up is None and gravity_up is not None:
        up = gravity_up
    if random_seed is not None:
        seed = int(random_seed)
    requested_max_planes = int(max_planes)
    if requested_max_planes <= 0:
        return []
    source_indices = np.arange(len(values), dtype=int)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    source_indices = source_indices[finite]
    if not len(values):
        return []

    normals_array = None
    if normals is not None:
        candidate_normals = np.asarray(normals, dtype=float)
        if candidate_normals.shape == (len(points), 3):
            normals_array = candidate_normals[finite]

    # Sorting makes the same point set yield the same sampled hypotheses even
    # when frames or voxels arrive in a different order.  Original indices
    # are carried separately so identity preservation still refers to the
    # caller's point array.
    order = np.lexsort((source_indices, values[:, 2], values[:, 1], values[:, 0]))
    values = values[order]
    source_indices = source_indices[order]
    if normals_array is not None:
        normals_array = normals_array[order]

    up_value = np.asarray(up if up is not None else [0.0, 0.0, 1.0], dtype=float).reshape(-1)
    up_norm = float(np.linalg.norm(up_value))
    up_value = (
        up_value / up_norm
        if up_value.shape == (3,) and np.isfinite(up_norm) and up_norm > 1e-9
        else np.array([0.0, 0.0, 1.0])
    )
    threshold = max(float(inlier_threshold), 1e-6)
    minimum = max(3, int(min_inliers))
    iterations = max(1, int(ransac_iterations))
    remaining = np.arange(len(values), dtype=int)
    rng = np.random.default_rng(seed)
    candidates: list[dict] = []

    while len(remaining) >= minimum and len(candidates) < requested_max_planes:
        subset = values[remaining]
        if len(subset) < 3:
            break
        pool_count = min(len(subset), max(3, int(ransac_sample_size)))
        if pool_count < len(subset):
            pool = np.unique(np.linspace(0, len(subset) - 1, pool_count).round().astype(int))
        else:
            pool = np.arange(len(subset), dtype=int)
        pool_points = subset[pool]
        best: tuple[tuple, np.ndarray, float] | None = None
        for _ in range(iterations):
            sample = rng.choice(len(pool_points), size=3, replace=False)
            hypothesis = _plane_from_three(pool_points[sample])
            if hypothesis is None:
                continue
            normal, offset = hypothesis
            distances = np.abs(pool_points @ normal - offset)
            mask = distances <= threshold
            count = int(mask.sum())
            if not count:
                continue
            residual = float(np.sqrt(np.mean(distances[mask] ** 2)))
            key = (
                -count,
                residual,
                tuple(np.round(normal, 12)),
                round(float(offset), 12),
            )
            if best is None or key < best[0]:
                best = (key, normal, float(offset))
        if best is None:
            break

        normal, offset = best[1], best[2]
        support = np.abs(subset @ normal - offset) <= threshold
        # Region growing and TLS refitting alternate until the support set is
        # stable.  The final pass always uses the full remaining cloud.
        for _ in range(5):
            if int(support.sum()) < 3:
                break
            fitted_normal, fitted_offset = fit_plane_tls(subset[support])
            distances = np.abs(subset @ fitted_normal - fitted_offset)
            grown = distances <= threshold
            normal, offset = fitted_normal, fitted_offset
            if np.array_equal(grown, support):
                support = grown
                break
            support = grown
        if int(support.sum()) < minimum:
            # The best hypothesis can be an accidental small patch once the
            # initial plane is refit.  Remove it and stop rather than emit a
            # misleading candidate from the same remaining data.
            break
        inlier_points = subset[support]
        normal, offset = fit_plane_tls(inlier_points)
        residuals = np.abs(inlier_points @ normal - offset)
        _, singular_values, _ = np.linalg.svd(
            inlier_points - inlier_points.mean(axis=0), full_matrices=False
        )
        if len(singular_values) < 3 or singular_values[1] <= float(min_plane_span):
            # Collinear support is not a plane.  Consume the points only if
            # there is another valid hypothesis left; otherwise terminate.
            remaining = remaining[~support]
            continue
        candidates.append(
            {
                "points": inlier_points.copy(),
                "indices": source_indices[remaining[support]].copy(),
                "normal": normal,
                "offset": float(offset),
                "residuals": residuals,
                "normal_support": normals_array[remaining[support]].copy()
                if normals_array is not None
                else None,
            }
        )
        remaining = remaining[~support]

    if not candidates:
        return []

    horizontal_cosine = float(np.cos(np.radians(max(0.0, horizontal_tolerance_degrees))))
    vertical_sine = float(np.sin(np.radians(max(0.0, vertical_tolerance_degrees))))
    candidate_heights = [float(candidate["points"].mean(axis=0) @ up_value) for candidate in candidates]
    finite_floor = floor_height is not None and np.isfinite(float(floor_height))
    floor_reference = float(floor_height) if finite_floor else min(candidate_heights)
    horizontal_indices = [
        index
        for index, candidate in enumerate(candidates)
        if abs(float(candidate["normal"] @ up_value)) >= horizontal_cosine
    ]
    if horizontal_indices:
        if finite_floor:
            floor_candidates = sorted(
                horizontal_indices,
                key=lambda index: (
                    abs(candidate_heights[index] - floor_reference),
                    candidate_heights[index],
                    -len(candidates[index]["indices"]),
                ),
            )
            floor_index = floor_candidates[0]
            # A supplied floor estimate is intentionally authoritative only
            # when a candidate is plausibly close; otherwise use the lowest
            # horizontal support instead of labelling a countertop a floor.
            if abs(candidate_heights[floor_index] - floor_reference) > 0.5:
                floor_index = min(
                    horizontal_indices,
                    key=lambda index: (candidate_heights[index], -len(candidates[index]["indices"])),
                )
        else:
            floor_index = min(
                horizontal_indices,
                key=lambda index: (candidate_heights[index], -len(candidates[index]["indices"])),
            )
        floor_reference = candidate_heights[floor_index]
    else:
        floor_index = None

    models: list[StructuralPlane] = []
    for candidate_index, candidate in enumerate(candidates):
        plane_points = candidate["points"]
        centroid = plane_points.mean(axis=0)
        normal = np.asarray(candidate["normal"], dtype=float)
        height = float(centroid @ up_value)
        is_floor_candidate = candidate_index == floor_index
        classification, is_quarantined, tags = _classify_plane(
            normal,
            height,
            floor_reference,
            up_value,
            horizontal_cosine,
            vertical_sine,
            float(min_room_height),
            is_floor_candidate,
        )
        # Give floor/ceiling normals physically meaningful signs.  Walls are
        # left with a deterministic sign only; unlike the old wall path,
        # their orientation is not forced to a Manhattan axis.
        if classification == PlaneClassification.FLOOR.value and normal @ up_value < 0:
            normal = -normal
        elif classification == PlaneClassification.CEILING.value and normal @ up_value > 0:
            normal = -normal
        else:
            normal = _canonical_plane_normal(normal)
        offset = float(normal @ centroid)
        residuals = np.abs(plane_points @ normal - offset)
        bounds_min = plane_points.min(axis=0)
        bounds_max = plane_points.max(axis=0)
        tangent_a, tangent_b, _ = _plane_basis(plane_points, normal)
        tangent_coordinates = np.column_stack(
            ((plane_points - centroid) @ tangent_a, (plane_points - centroid) @ tangent_b)
        )
        in_plane_extents = np.column_stack(
            (tangent_coordinates.min(axis=0), tangent_coordinates.max(axis=0))
        )
        spans = in_plane_extents[:, 1] - in_plane_extents[:, 0]
        area = float(max(spans[0], 0.0) * max(spans[1], 0.0))
        density = float(len(plane_points) / area) if area > 1e-9 else 0.0
        robust_mad = float(
            np.median(np.abs(residuals - np.median(residuals)))
            if len(residuals)
            else 0.0
        )
        adaptive_threshold = float(
            np.clip(max(threshold, 2.5 * max(0.005, 1.4826 * robust_mad) + 0.01),
                    threshold, max(3.0 * threshold, 0.12))
        )
        support_score = 1.0 - np.exp(-len(plane_points) / max(float(minimum), 1.0))
        residual_score = float(np.exp(-float(np.sqrt(np.mean(residuals**2))) / threshold))
        density_score = density / (density + 25.0) if density > 0 else 0.0
        extent_score = float(np.clip(np.min(spans) / 0.5, 0.0, 1.0))
        normal_score = 1.0
        if candidate["normal_support"] is not None and len(candidate["normal_support"]):
            source_normals = candidate["normal_support"]
            lengths = np.linalg.norm(source_normals, axis=1)
            valid_normals = np.isfinite(source_normals).all(axis=1) & (lengths > 1e-9)
            if valid_normals.any():
                source_normals = source_normals[valid_normals]
                source_normals = source_normals / lengths[valid_normals, None]
                normal_score = float(np.mean(np.abs(source_normals @ normal)))
        confidence = float(
            np.clip(
                0.40 * support_score
                + 0.30 * residual_score
                + 0.15 * density_score
                + 0.15 * extent_score * normal_score,
                0.0,
                1.0,
            )
        )
        if is_quarantined:
            rejection_reasons = (
                ("orientation_not_horizontal_or_vertical",)
                if "off-orientation" in tags
                else ("not_floor_or_ceiling_height",)
            )
            quality_status = "quarantined"
        else:
            rejection_reasons = ()
            quality_status = "high_confidence"
        vertical_values = plane_points @ up_value
        vertical_extent = tuple(
            float(v) for v in (float(vertical_values.min()), float(vertical_values.max()))
        )
        horizontal_tangent = np.cross(up_value, normal)
        tangent_norm = float(np.linalg.norm(horizontal_tangent))
        horizontal_extent = None
        if tangent_norm > 1e-9:
            horizontal_tangent /= tangent_norm
            horizontal_coordinates = (plane_points - centroid) @ horizontal_tangent
            horizontal_extent = (
                float(horizontal_coordinates.min()),
                float(horizontal_coordinates.max()),
            )
        model = StructuralPlane(
            id=candidate_index,
            classification=classification,
            normal=normal,
            offset=offset,
            centroid=centroid,
            extents=bounds_max - bounds_min,
            inlier_indices=np.sort(candidate["indices"]),
            residual_rms=float(np.sqrt(np.mean(residuals**2))),
            residual_mean_abs=float(np.mean(residuals)),
            residual_median=float(np.median(residuals)),
            residual_max=float(np.max(residuals)),
            point_density=density,
            confidence=confidence,
            candidate_threshold=threshold,
            support_threshold=minimum,
            residual_threshold=threshold,
            adaptive_residual_threshold=adaptive_threshold,
            quality_status=quality_status,
            low_confidence=False,
            rejection_reasons=rejection_reasons,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            in_plane_extents=in_plane_extents,
            wall_vertical_extent=vertical_extent if classification == PlaneClassification.WALL.value else None,
            ceiling_observed=classification == PlaneClassification.CEILING.value,
            ceiling_confidence=confidence if classification == PlaneClassification.CEILING.value else 0.0,
            quarantined=bool(is_quarantined and quarantine),
            tags=tags,
            _horizontal_tangent=horizontal_tangent if horizontal_extent is not None else None,
            _horizontal_extent=horizontal_extent,
            _up=up_value.copy(),
        )
        if is_quarantined and not quarantine:
            continue
        models.append(model)

    class_order = {
        PlaneClassification.FLOOR.value: 0,
        PlaneClassification.CEILING.value: 1,
        PlaneClassification.WALL.value: 2,
        PlaneClassification.CLUTTER.value: 3,
    }
    models.sort(
        key=lambda plane: (
            class_order.get(str(plane.classification), 99),
            float(plane.centroid @ up_value),
            -plane.support_count,
            tuple(np.round(plane.centroid, 9)),
        )
    )
    for index, plane in enumerate(models):
        plane.id = index
    return models


def detect_major_planes(*args, **kwargs) -> list[StructuralPlane]:
    """Compatibility alias for :func:`extract_structural_planes`."""
    return extract_structural_planes(*args, **kwargs)


def extract_planes(*args, **kwargs) -> list[StructuralPlane]:
    """Compatibility alias for :func:`extract_structural_planes`."""
    return extract_structural_planes(*args, **kwargs)


def extract_planes_3d(*args, **kwargs) -> list[StructuralPlane]:
    """Explicit 3D alias for :func:`extract_structural_planes`."""
    return extract_structural_planes(*args, **kwargs)


def extract_structural_plane_models(*args, **kwargs) -> list[StructuralPlane]:
    """Descriptive alias for :func:`extract_structural_planes`."""
    return extract_structural_planes(*args, **kwargs)


def detect_planes(*args, **kwargs) -> list[StructuralPlane]:
    """Short compatibility alias for :func:`extract_structural_planes`."""
    return extract_structural_planes(*args, **kwargs)


fit_plane_3d = fit_plane_tls
refit_plane_3d = fit_plane_tls


def planes_to_wall_segments(
    planes: list[StructuralPlane],
    frame: HorizontalFrame,
    points: np.ndarray | None = None,
    include_quarantined: bool = False,
) -> list[WallSegment]:
    """Convert retained wall planes to the established 2D wall segments."""
    result = []
    for plane in planes:
        if plane.classification != PlaneClassification.WALL.value:
            continue
        if plane.quarantined and not include_quarantined:
            continue
        wall = plane.to_wall_segment(frame, index=len(result), points=points)
        if wall is not None:
            result.append(wall)
    return result


planes_to_walls = planes_to_wall_segments


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
    diagnostics: GeometryDiagnostics | None = None,
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
        if diagnostics is not None:
            diagnostics.set_wall_stage("raw", [])
        return []
    finite = np.isfinite(plan_points).all(axis=1) & np.isfinite(heights)
    if diagnostics is not None and int((~finite).sum()):
        diagnostics.record_drop_summary(
            "non-finite-input",
            int((~finite).sum()),
            stage="raw",
            provenance="extract_walls.finite-input-gate",
        )
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
            if diagnostics is not None:
                diagnostics.record_drop_summary(
                    "insufficient-ransac-support",
                    1,
                    stage="raw",
                    provenance="extract_walls.ransac",
                )
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
                if diagnostics is not None:
                    diagnostics.record_drop_summary(
                        "run-too-short",
                        1,
                        stage="raw",
                        provenance="extract_walls.contiguous_runs",
                    )
                continue
            expected = (run_length / point_spacing) * (band_height / point_spacing)
            if count < min_coverage * expected:
                if diagnostics is not None:
                    diagnostics.record_drop_summary(
                        "insufficient-coverage",
                        1,
                        stage="raw",
                        provenance="extract_walls.coverage_gate",
                    )
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
                )
            if is_off_axis:
                wall.tags.append("off-axis")
                if diagnostics is not None:
                    diagnostics.record_wall_event(
                        wall,
                        stage="raw",
                        action="quarantine",
                        reason="off-axis",
                        provenance="extract_walls.manhattan_gate",
                    )
            elif diagnostics is not None:
                diagnostics.record_wall_event(
                    wall,
                    stage="raw",
                    action="accepted",
                    reason="ransac-fit",
                    provenance="extract_walls",
                )
            walls.append(wall)
        remaining = remaining[~inlier_mask]

    walls.sort(key=_wall_output_rank)
    for position, wall in enumerate(walls):
        wall.index = position
    if diagnostics is not None:
        diagnostics.set_wall_stage("raw", walls)
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
    diagnostics: GeometryDiagnostics | None = None,
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
            if diagnostics is not None:
                diagnostics.record_wall_event(
                    wall,
                    stage="quarantine",
                    action="quarantine",
                    reason=_quarantine_reason(wall),
                    provenance="merge_collinear.input",
                )
            kept.append(wall)
            continue
        matched = False
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
                matched = True
                if diagnostics is not None:
                    diagnostics.record_wall_event(
                        wall,
                        stage="merged",
                        action="drop",
                        reason="duplicate-wall",
                        provenance="merge_collinear.duplicate_suppression",
                        related_walls=[target],
                    )
                break
            if overlap >= min_overlap:
                # Further away but still overlapping -- likely clutter
                # sitting in front of the real wall, not the wall itself.
                # Tag it and keep the two surfaces separate.
                target.tags.append("clutter-in-front")
                matched = True
                if diagnostics is not None:
                    diagnostics.record_wall_event(
                        wall,
                        stage="merged",
                        action="drop",
                        reason="clutter-in-front",
                        provenance="merge_collinear.parallel_surface",
                        related_walls=[target],
                    )
                break
        if not matched:
            kept.append(wall)

    kept.sort(key=_wall_output_rank)
    for position, wall in enumerate(kept):
        wall.index = position
        wall.tags = sorted(set(wall.tags))
    if diagnostics is not None:
        diagnostics.set_wall_stage("merged", kept)
    return kept


def _quarantine_reason(wall: WallSegment) -> str:
    """Choose a stable primary reason from diagnostic wall tags."""
    tags = set(getattr(wall, "tags", []))
    for reason in ("off-axis", "invalid-geometry", "degenerate"):
        if reason in tags:
            return reason
    return "quarantined"


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
    diagnostics: GeometryDiagnostics | None = None,
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
    if diagnostics is not None:
        for wall in walls:
            if wall.quarantined:
                diagnostics.record_wall_event(
                    wall,
                    stage="quarantine",
                    action="quarantine",
                    reason=_quarantine_reason(wall),
                    provenance="filter_occluded_walls.input",
                )

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
            if diagnostics is not None:
                diagnostics.record_wall_event(
                    wall,
                    stage="occlusion",
                    action="drop",
                    reason="occlusion-inconsistent",
                    provenance="filter_occluded_walls.ray_blocking",
                    related_walls=blockers,
                )
            continue
        kept.append(wall)

    kept.sort(key=_wall_output_rank)
    for position, wall in enumerate(kept):
        wall.index = position
    if diagnostics is not None:
        diagnostics.set_wall_stage("occlusion", kept)
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
    diagnostics: GeometryDiagnostics | None = None,
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
    for wall in walls:
        if wall.quarantined and diagnostics is not None:
            diagnostics.record_wall_event(
                wall,
                stage="quarantine",
                action="quarantine",
                reason=_quarantine_reason(wall),
                provenance="resolve_crossings.input",
            )
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
                    if diagnostics is not None:
                        diagnostics.record_wall_event(
                            victim,
                            stage="crossing",
                            action="trim",
                            reason="t-junction-overshoot",
                            provenance="resolve_crossings",
                            related_walls=[a if victim is b else b],
                        )
                else:
                    # Too much overlap to be a simple overshoot -- treat
                    # the weaker-supported wall as spurious and drop it.
                    weaker = a if a.inlier_count < b.inlier_count else b
                    walls.remove(weaker)
                    if diagnostics is not None:
                        diagnostics.record_wall_event(
                            weaker,
                            stage="crossing",
                            action="drop",
                            reason="weaker-crossing-wall",
                            provenance="resolve_crossings",
                            related_walls=[a if weaker is b else b],
                        )
                changed = True
                break
            if changed:
                break

    for wall in walls:
        wall.tags = sorted(set(wall.tags))
    if diagnostics is not None:
        diagnostics.set_wall_stage("crossing", walls)
    return walls


def snap_corners(
    walls: list[WallSegment],
    max_extension: float = 0.45,
    max_trim: float = 0.30,
    diagnostics: GeometryDiagnostics | None = None,
) -> int:
    """Nudges wall endpoints so that walls meeting at a corner actually touch at a clean point.

    Detected walls almost never end exactly where they should -- one wall's
    endpoint might stop just short of, or run just past, the neighbouring
    wall it's supposed to meet. This function finds those nearby corners
    (where two walls' lines would cross) and, for each wall endpoint, moves
    it onto whichever nearby corner would require the smallest adjustment,
    as long as that corner is also plausibly close to where the other wall
    actually ends. Every possible endpoint-to-corner adjustment is
    collected first and then applied smallest-adjustment-first, so that
    endpoints compete fairly for the best-fitting corner rather than each
    wall grabbing the first candidate it happens to consider. An endpoint
    with no good nearby corner is simply left where it was.

    Args:
        walls: Wall segments to snap into corners; mutated in place for any
            endpoint that ends up getting moved.
        max_extension: The furthest an endpoint may be pushed outward, in
            metres, to reach a candidate corner.
        max_trim: The furthest an endpoint may be pulled inward, in metres,
            to reach a candidate corner.

    Returns:
        How many endpoints were actually moved onto a snapped corner.
    """
    proposals: list[tuple[float, int, str, np.ndarray]] = []
    active_walls = [wall for wall in walls if not wall.quarantined]
    for i, wall in enumerate(active_walls):
        for other in active_walls:
            if other is wall:
                continue
            hit = _segment_intersection(wall, other)
            if hit is None:
                continue
            point, u_self, u_other = hit
            if not (-max_extension <= u_other <= other.length + max_extension):
                continue
            for end_name, adjustment in (
                ("start", -u_self),
                ("end", u_self - wall.length),
            ):
                if -max_trim <= adjustment <= max_extension:
                    proposals.append((abs(adjustment), i, end_name, point))

    # Apply the smallest adjustments first, so each endpoint gets matched
    # to its best-fitting corner rather than whichever candidate happened
    # to be considered first.
    proposals.sort(key=lambda entry: entry[0])
    taken: set[tuple[int, str]] = set()
    snapped = 0
    for _, index, end_name, point in proposals:
        key = (index, end_name)
        if key in taken:
            continue
        wall = active_walls[index]
        remaining = (
            float(np.linalg.norm(wall.end - point))
            if end_name == "start"
            else float(np.linalg.norm(point - wall.start))
        )
        if remaining < 0.3:
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
        if diagnostics is not None:
            diagnostics.record_wall_event(
                wall,
                stage="crossing",
                action="trim",
                reason="corner-snap",
                provenance="snap_corners.line_intersection",
                extra={"endpoint": end_name},
            )
        snapped += 1
    return snapped


def snap_to_frame(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    tolerance_degrees: float = 8.0,
    diagnostics: GeometryDiagnostics | None = None,
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

    Returns:
        The same `walls` list, mutated in place and also returned, so
        calls can be chained together.
    """
    for wall in walls:
        if wall.quarantined:
            continue
        angle = np.arctan2(wall.normal[1], wall.normal[0])
        quarter = round(angle / (np.pi / 2)) % 4
        snapped = quarter * (np.pi / 2)
        if abs(np.degrees(angle - snapped)) > tolerance_degrees:
            wall.tags.append("off-axis")
            wall.quarantined = True
            if diagnostics is not None:
                diagnostics.record_wall_event(
                    wall,
                    stage="quarantine",
                    action="quarantine",
                    reason="off-axis",
                    provenance="snap_to_frame.manhattan_gate",
                )
            continue
        normal = _nearest_manhattan_normal(wall.normal)
        offset = float(normal @ wall.midpoint)
        direction = np.array([-normal[1], normal[0]])
        half = 0.5 * wall.length
        centre = normal * offset + direction * (direction @ wall.midpoint)
        wall.normal, wall.offset = normal, offset
        wall.start, wall.end = centre - direction * half, centre + direction * half
    return walls
