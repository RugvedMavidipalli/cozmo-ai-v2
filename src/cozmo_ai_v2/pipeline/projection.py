"""NumPy-only top-down projection of wall-height point-cloud evidence.

The reconstruction pipeline keeps its metric coordinates in a
``HorizontalFrame``.  This module is the small, dependency-free boundary
between that 3D geometry and consumers that need a floor-plan raster (the
room/vectorizer stage in particular).  It deliberately does not use an image
or raster library: a cell is just a half-open metric square and accumulation
is done with :func:`numpy.add.at`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .planes import HorizontalFrame

WallBand: TypeAlias = tuple[float, float]
PlanBounds: TypeAlias = tuple[tuple[float, float], tuple[float, float]]

DEFAULT_RESOLUTION = 0.04
DEFAULT_WALL_BAND: WallBand = (0.35, 1.9)
DEFAULT_MARGIN = 0.5
CEILING_CLEARANCE = 0.15


@dataclass(frozen=True)
class DensityMap:
    """A deterministic top-down density map of wall-height points.

    Coordinates use the same plan frame as :class:`HorizontalFrame` and the
    same axis order as ``PlanGrid``: ``counts[x_cell, y_cell]``.  Cell ``(i,
    j)`` covers
    ``[origin_x + i * resolution, origin_x + (i + 1) * resolution)`` by
    ``[origin_y + j * resolution, origin_y + (j + 1) * resolution)``.  The
    lower edge is inclusive and the upper edge is exclusive, so exact cell
    boundaries have one unambiguous owner.

    ``counts`` counts retained source points, not unique points or binary
    occupancy.  ``density`` converts those counts to points per square metre;
    ``observed`` and ``empty`` are the corresponding boolean masks for a
    vectorizer.  The output bounds are the full raster extent.  When an input
    ``clip_bounds`` was supplied, it is retained separately because a
    non-resolution-aligned clip can end inside the final raster cell.

    Attributes:
        resolution: Cell width and height in metres.
        origin: Plan-space coordinate of the lower-left corner of cell
            ``(0, 0)``.
        bounds: Full half-open plan-space extent of the raster as
            ``((min_x, min_y), (max_x, max_y))``.
        clip_bounds: Optional half-open input clipping extent. ``None`` means
            bounds were derived from the finite input cloud with the default
            margin.
        wall_band: Configured height offsets above ``floor_height`` in metres.
        height_bounds: Effective absolute height interval used for cropping,
            including the observed-ceiling clearance when applicable. Both
            ends are strict, matching ``wall_band_mask``/``build_plan_grid``.
        floor_height: Floor height in the world frame, in metres.
        ceiling_height: Observed ceiling height used to cap the wall band, or
            ``None`` when no observed ceiling was supplied.
        counts: Integer point counts with shape ``(columns, rows)``.
        input_count: Number of rows in the supplied point array.
        finite_input_count: Number of rows whose three source coordinates are
            finite.
        invalid_input_count: Number of source rows excluded because a source
            coordinate or derived frame coordinate was non-finite.
        band_count: Number of finite projected points inside the wall-height
            band before plan bounds clipping.
        retained_count: Number of points actually accumulated into ``counts``.
        out_of_bounds_count: Number of wall-band points excluded by the plan
            bounds. Thus ``retained_count == counts.sum()``.
    """

    resolution: float
    origin: np.ndarray
    bounds: PlanBounds
    clip_bounds: PlanBounds | None
    wall_band: WallBand
    height_bounds: tuple[float, float]
    floor_height: float
    ceiling_height: float | None
    counts: np.ndarray
    input_count: int
    finite_input_count: int
    invalid_input_count: int
    band_count: int
    retained_count: int
    out_of_bounds_count: int

    def __post_init__(self) -> None:
        """Normalize arrays and reject malformed metadata at the API edge."""
        resolution = float(self.resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be a finite positive number")

        origin = np.asarray(self.origin, dtype=float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin must be a finite plan-space (2,) coordinate")

        counts = np.asarray(self.counts)
        if counts.ndim != 2:
            raise ValueError("counts must have shape (columns, rows)")
        if not np.issubdtype(counts.dtype, np.integer):
            counts = counts.astype(np.int64)
        if (counts < 0).any():
            raise ValueError("counts cannot be negative")

        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "counts", counts)

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(columns, rows)`` shape of the count raster."""
        return tuple(int(value) for value in self.counts.shape)

    @property
    def cell_area(self) -> float:
        """Area of one cell in square metres."""
        return self.resolution**2

    @property
    def density(self) -> np.ndarray:
        """Point density in points per square metre, preserving empty cells."""
        return self.counts.astype(float) / self.cell_area

    @property
    def observed(self) -> np.ndarray:
        """Boolean mask for cells containing at least one retained point."""
        return self.counts > 0

    @property
    def empty(self) -> np.ndarray:
        """Boolean mask for cells with no retained point observations."""
        return ~self.observed

    @property
    def total_count(self) -> int:
        """Alias for the conservation count represented by ``counts``."""
        return self.retained_count

    def to_cell(self, plan: np.ndarray) -> np.ndarray:
        """Convert plan-space coordinates to unclipped ``(x_cell, y_cell)``.

        No bounds clipping is performed.  A caller can therefore inspect
        out-of-range indices before deciding whether to discard them.
        """
        plan = np.asarray(plan, dtype=float)
        if plan.ndim == 1 and plan.size == 0:
            plan = plan.reshape(0, 2)
        if plan.ndim not in (1, 2) or plan.shape[-1] != 2:
            raise ValueError("plan must have shape (N, 2) or (2,)")
        return np.floor((plan - self.origin) / self.resolution).astype(np.int64)

    def to_plan(self, cell: np.ndarray) -> np.ndarray:
        """Convert cell indices to the plan-space coordinate of each center."""
        cell = np.asarray(cell, dtype=float)
        if cell.ndim == 1 and cell.size == 0:
            cell = cell.reshape(0, 2)
        if cell.ndim not in (1, 2) or cell.shape[-1] != 2:
            raise ValueError("cell must have shape (N, 2) or (2,)")
        return self.origin + (cell + 0.5) * self.resolution


def rasterize_points(
    plan: np.ndarray,
    *,
    origin: np.ndarray,
    shape: tuple[int, int],
    resolution: float,
) -> np.ndarray:
    """Accumulate finite plan-space points into a bounded NumPy raster.

    This is the public low-level operation for vectorizers that already have
    2D plan coordinates.  Points outside the half-open raster extent are
    dropped.  The output is always ``(columns, rows)`` and integer-valued;
    counts are not normalized or clipped.
    """
    plan = _coerce_plan(plan)
    origin = np.asarray(origin, dtype=float)
    shape = _coerce_shape(shape)
    resolution = _validate_resolution(resolution)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite plan-space (2,) coordinate")

    finite = np.isfinite(plan).all(axis=1)
    cells = np.zeros((len(plan), 2), dtype=np.int64)
    if finite.any():
        cells[finite] = np.floor(
            (plan[finite] - origin) / resolution
        ).astype(np.int64)
    inside = finite & (
        (cells[:, 0] >= 0)
        & (cells[:, 1] >= 0)
        & (cells[:, 0] < shape[0])
        & (cells[:, 1] < shape[1])
    )

    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, (cells[inside, 0], cells[inside, 1]), 1)
    return counts


def project_wall_density(
    points: np.ndarray,
    frame: HorizontalFrame,
    floor_height: float,
    ceiling_height: float | None = None,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    wall_band: WallBand = DEFAULT_WALL_BAND,
    bounds: PlanBounds | np.ndarray | None = None,
    margin: float = DEFAULT_MARGIN,
) -> DensityMap:
    """Crop a 3D cloud to wall height and project it into a top-down map.

    Heights are measured with ``frame.height`` and plan coordinates with
    ``frame.to_plan``; no alternate world-to-plan convention is introduced.
    The configured ``wall_band`` is an offset interval above ``floor_height``
    and uses strict lower/upper edges, matching the existing reconstruction
    stage.  If an observed ceiling is supplied, its 0.15 m clearance also
    caps the effective upper edge, as ``build_plan_grid`` historically did.

    With no explicit ``bounds``, the origin is the finite plan-space minimum
    minus ``margin`` and the shape is derived from the finite cloud extent.
    This makes origin and shape independent of point order and gives the
    existing vectorizer its half-metre context.  Explicit bounds are a
    half-open clipping interval ``[min, max)``; points on its upper edge are
    outside.  In either mode raster cells themselves remain half-open.

    Args:
        points: Cleaned world-space points with shape ``(N, 3)``.  An empty
            ``(0, 3)`` array is valid.
        frame: Existing building-aligned coordinate frame.
        floor_height: World-space floor offset in metres. Non-finite values
            use the existing conservative fallback of ``0.0``.
        ceiling_height: Observed ceiling offset, if available. Non-finite
            values are treated as unavailable.
        resolution: Square cell size in metres.
        wall_band: Strict lower and upper height offsets above the floor.
        bounds: Optional ``((min_x, min_y), (max_x, max_y))`` clipping bounds.
        margin: Margin in metres used only when bounds are omitted.

    Returns:
        A :class:`DensityMap` containing counts, density, observed/empty
        masks, and enough metadata for deterministic vectorizer use.
    """
    points = _coerce_points(points)
    resolution = _validate_resolution(resolution)
    wall_band = _validate_wall_band(wall_band)
    floor_height = (
        float(floor_height) if np.isfinite(floor_height) else 0.0
    )
    ceiling = (
        float(ceiling_height)
        if ceiling_height is not None and np.isfinite(ceiling_height)
        else None
    )
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite non-negative number")

    input_count = len(points)
    finite_source = np.isfinite(points).all(axis=1)
    finite_input_count = int(finite_source.sum())

    plan = np.empty((0, 2), dtype=float)
    heights = np.empty(0, dtype=float)
    if finite_input_count:
        source = points[finite_source]
        plan = np.asarray(frame.to_plan(source), dtype=float)
        heights = np.asarray(frame.height(source), dtype=float).reshape(-1)
        if plan.shape != (finite_input_count, 2):
            raise ValueError("frame.to_plan must return shape (N, 2)")
        if heights.shape != (finite_input_count,):
            raise ValueError("frame.height must return shape (N,)")

    projected_finite = np.isfinite(plan).all(axis=1) & np.isfinite(heights)
    invalid_input_count = input_count - int(projected_finite.sum())
    candidate_plan = plan[projected_finite]
    candidate_heights = heights[projected_finite]

    effective_low = floor_height + wall_band[0]
    effective_high = floor_height + wall_band[1]
    if ceiling is not None:
        effective_high = min(effective_high, ceiling - CEILING_CLEARANCE)
    in_band = (candidate_heights > effective_low) & (
        candidate_heights < effective_high
    )
    band_plan = candidate_plan[in_band]
    band_count = len(band_plan)

    lower, upper, clip_bounds = _choose_bounds(
        candidate_plan, bounds=bounds, margin=float(margin)
    )
    shape = _shape_for_bounds(lower, upper, resolution, explicit=bounds is not None)

    if clip_bounds is not None and band_count:
        # Explicit bounds are clipping bounds, rather than merely a hint for
        # the output extent.  Keep their upper edge exclusive even where a
        # non-aligned final cell extends slightly farther than max_x/max_y.
        in_clip = (
            (band_plan[:, 0] >= clip_bounds[0][0])
            & (band_plan[:, 1] >= clip_bounds[0][1])
            & (band_plan[:, 0] < clip_bounds[1][0])
            & (band_plan[:, 1] < clip_bounds[1][1])
        )
        raster_plan = band_plan[in_clip]
    else:
        raster_plan = band_plan

    counts = rasterize_points(
        raster_plan,
        origin=lower,
        shape=shape,
        resolution=resolution,
    )
    retained_count = int(counts.sum())
    out_of_bounds_count = band_count - retained_count
    raster_bounds: PlanBounds = (
        (float(lower[0]), float(lower[1])),
        (
            float(lower[0] + shape[0] * resolution),
            float(lower[1] + shape[1] * resolution),
        ),
    )
    return DensityMap(
        resolution=resolution,
        origin=lower,
        bounds=raster_bounds,
        clip_bounds=clip_bounds,
        wall_band=wall_band,
        height_bounds=(float(effective_low), float(effective_high)),
        floor_height=floor_height,
        ceiling_height=ceiling,
        counts=counts,
        input_count=input_count,
        finite_input_count=finite_input_count,
        invalid_input_count=invalid_input_count,
        band_count=band_count,
        retained_count=retained_count,
        out_of_bounds_count=out_of_bounds_count,
    )


# The longer name is useful at call sites, while this concise spelling makes
# the public operation easy to discover for clients that call it directly.
project_top_down_density = project_wall_density


def _coerce_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points


def _coerce_plan(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=float)
    if plan.ndim == 1 and plan.size == 0:
        plan = plan.reshape(0, 2)
    if plan.ndim != 2 or plan.shape[1] != 2:
        raise ValueError("plan must have shape (N, 2)")
    return plan


def _coerce_shape(shape: tuple[int, int]) -> tuple[int, int]:
    raw = np.asarray(shape)
    if raw.shape != (2,):
        raise ValueError("shape must contain two positive dimensions")
    if not np.isfinite(raw.astype(float)).all():
        raise ValueError("shape dimensions must be finite integers")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("shape dimensions must be integers")
    values = tuple(int(value) for value in raw)
    if any(value <= 0 for value in values):
        raise ValueError("shape must contain two positive dimensions")
    return values


def _validate_resolution(resolution: float) -> float:
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a finite positive number")
    return resolution


def _validate_wall_band(wall_band: WallBand) -> WallBand:
    values = np.asarray(wall_band, dtype=float).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("wall_band must contain two finite offsets")
    if values[0] >= values[1]:
        raise ValueError("wall_band lower offset must be below its upper offset")
    return float(values[0]), float(values[1])


def _choose_bounds(
    plan: np.ndarray,
    *,
    bounds: PlanBounds | np.ndarray | None,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, PlanBounds | None]:
    if bounds is not None:
        clip_bounds = _coerce_bounds(bounds)
        return (
            np.asarray(clip_bounds[0], dtype=float),
            np.asarray(clip_bounds[1], dtype=float),
            clip_bounds,
        )

    if len(plan):
        lower = plan.min(axis=0) - margin
        upper = plan.max(axis=0) + margin
    else:
        lower = np.zeros(2, dtype=float)
        upper = lower.copy()
    return lower, upper, None


def _coerce_bounds(bounds: PlanBounds | np.ndarray) -> PlanBounds:
    values = np.asarray(bounds, dtype=float)
    if values.shape == (4,):
        values = values.reshape(2, 2)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError(
            "bounds must be ((min_x, min_y), (max_x, max_y)) with finite values"
        )
    if (values[1] < values[0]).any():
        raise ValueError("bounds maximum must not be below its minimum")
    return (
        (float(values[0, 0]), float(values[0, 1])),
        (float(values[1, 0]), float(values[1, 1])),
    )


def _shape_for_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    resolution: float,
    *,
    explicit: bool,
) -> tuple[int, int]:
    # An implicit extent gets one extra cell, retaining the established
    # PlanGrid half-metre margin even when a point lies exactly on a derived
    # boundary.  Explicit extents are half-open and use only the cells needed
    # to cover their interval (with one cell for a zero-width interval).
    span = np.maximum(upper - lower, 0.0)
    extra = 0 if explicit else 1
    shape = np.ceil(span / resolution).astype(np.int64) + extra
    shape = np.maximum(shape, 1)
    return int(shape[0]), int(shape[1])
