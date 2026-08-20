"""Per-wall occupancy in surface (UV) coordinates.

One structure carries three jobs that would otherwise each need their own
representation, because all three are questions about *where on this wall*
something is:

  * openings   -- a hole in the observed surface that reaches the floor is a
                  door; one with material below it is a window.
  * occlusion  -- cells with no observation and no line of sight are hidden by
                  furniture; the assignment requires those spans to be
                  reported as inferred rather than measured.
  * damage     -- per-frame damage masks accumulate here, which is what merges
                  sixty observations of one stain into a single region with a
                  real area instead of sixty double-counted ones.

U runs along the wall from its start corner, V runs up from the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .planes import HorizontalFrame, WallSegment


@dataclass
class SurfaceGrid:
    """Occupancy for one wall, in metres-along by metres-up cells."""

    wall_index: int
    resolution: float
    width: float
    height: float
    base_height: float  # world height of V = 0
    hits: np.ndarray  # observations landing on the wall plane
    passthrough: np.ndarray  # rays that went past the plane: an opening
    near: np.ndarray  # observations well in front: furniture

    @property
    def shape(self) -> tuple[int, int]:
        return self.hits.shape

    def to_uv(self, points: np.ndarray, wall: WallSegment, frame: HorizontalFrame):
        plan = frame.to_plan(points)
        u = (plan - wall.start) @ wall.direction
        v = frame.height(points) - self.base_height
        return np.stack([u, v], axis=-1)

    def to_cell(self, uv: np.ndarray) -> np.ndarray:
        return np.floor(uv / self.resolution).astype(int)

    def cell_to_uv(self, cell: np.ndarray) -> np.ndarray:
        return (np.asarray(cell, float) + 0.5) * self.resolution

    @property
    def observed(self) -> np.ndarray:
        return self.hits > 0

    def area_of(self, mask: np.ndarray) -> float:
        return float(mask.sum() * self.resolution**2)


@dataclass
class Opening:
    wall_index: int
    kind: str  # "door" | "window" | "pass-through"
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    confidence: float

    @property
    def width(self) -> float:
        return self.u_range[1] - self.u_range[0]

    @property
    def height(self) -> float:
        return self.v_range[1] - self.v_range[0]

    @property
    def sill_height(self) -> float:
        return self.v_range[0]

    @property
    def header_height(self) -> float:
        return self.v_range[1]


def build_surface_grid(
    wall: WallSegment,
    frame: HorizontalFrame,
    points: np.ndarray,
    floor_height: float,
    ceiling_height: float,
    resolution: float = 0.04,
    plane_band: float = 0.06,
    near_band: float = 0.6,
) -> SurfaceGrid:
    """Accumulate observations around one wall into surface coordinates.

    Three populations are separated by signed distance from the wall plane:
    points *on* it (the wall itself), points well in *front* of it (furniture,
    which occludes rather than contradicts), and the absence of either, which
    is only meaningful once you know a ray passed through.
    """
    height = ceiling_height - floor_height
    columns = max(int(np.ceil(wall.length / resolution)), 1)
    rows = max(int(np.ceil(height / resolution)), 1)

    plan = frame.to_plan(points)
    distance = plan @ wall.normal - wall.offset
    along = (plan - wall.start) @ wall.direction
    up = frame.height(points) - floor_height

    in_span = (along >= 0) & (along < wall.length) & (up >= 0) & (up < height)
    on_plane = in_span & (np.abs(distance) < plane_band)
    in_front = in_span & (np.abs(distance) >= plane_band) & (np.abs(distance) < near_band)

    grid = SurfaceGrid(
        wall_index=wall.index,
        resolution=resolution,
        width=wall.length,
        height=height,
        base_height=floor_height,
        hits=_bin(along[on_plane], up[on_plane], resolution, columns, rows),
        passthrough=np.zeros((columns, rows), np.int32),
        near=_bin(along[in_front], up[in_front], resolution, columns, rows),
    )
    return grid


def _bin(
    u: np.ndarray, v: np.ndarray, resolution: float, columns: int, rows: int
) -> np.ndarray:
    counts = np.zeros((columns, rows), np.int32)
    if len(u) == 0:
        return counts
    cu = np.clip((u / resolution).astype(int), 0, columns - 1)
    cv = np.clip((v / resolution).astype(int), 0, rows - 1)
    np.add.at(counts, (cu, cv), 1)
    return counts


def find_openings(
    grid: SurfaceGrid,
    min_width: float = 0.5,
    min_height: float = 0.55,
    door_floor_tolerance: float = 0.16,
    min_door_height: float = 1.6,
) -> list[Opening]:
    """Holes in an otherwise observed wall.

    An opening must be *surrounded* by observed wall, which is what separates a
    doorway from the far end of a wall the operator simply never scanned.  The
    distinction between door and window is whether the hole reaches the floor.
    """
    observed = grid.observed
    if observed.sum() < 20:
        return []

    # Fill the wall's observed silhouette, then subtract what was seen: the
    # difference is enclosed holes only.
    silhouette = ndimage.binary_closing(observed, np.ones((5, 5)))
    silhouette = ndimage.binary_fill_holes(silhouette)
    holes = silhouette & ~ndimage.binary_dilation(observed, np.ones((3, 3)))
    holes = ndimage.binary_opening(holes, np.ones((3, 3)))

    labels, count = ndimage.label(holes)
    openings: list[Opening] = []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        u_lo, v_lo = cells.min(axis=0) * grid.resolution
        u_hi, v_hi = (cells.max(axis=0) + 1) * grid.resolution
        width, height = u_hi - u_lo, v_hi - v_lo
        if width < min_width or height < min_height:
            continue

        fill = len(cells) / max((width / grid.resolution) * (height / grid.resolution), 1)
        if fill < 0.45:  # ragged: scan dropout, not an opening
            continue

        reaches_floor = v_lo <= door_floor_tolerance
        if reaches_floor and height >= min_door_height:
            kind = "door"
        elif reaches_floor:
            kind = "pass-through"
        else:
            kind = "window"
        openings.append(
            Opening(
                wall_index=grid.wall_index,
                kind=kind,
                u_range=(float(u_lo), float(u_hi)),
                v_range=(float(v_lo), float(v_hi)),
                confidence=float(min(1.0, fill)),
            )
        )
    return openings


def occluded_mask(grid: SurfaceGrid, min_near: int = 3) -> np.ndarray:
    """Cells hidden behind something in front of the wall.

    These are the spans the report must mark inferred: the wall plane and its
    corners give their dimensions, but nothing was ever measured there.
    """
    return (~grid.observed) & (grid.near >= min_near)


def occluded_spans(grid: SurfaceGrid, min_width: float = 0.25) -> list[tuple[float, float]]:
    """Contiguous along-wall runs that are mostly occluded."""
    hidden = occluded_mask(grid)
    if not hidden.any():
        return []
    column_hidden = hidden.mean(axis=1) > 0.3
    spans: list[tuple[float, float]] = []
    labels, count = ndimage.label(column_hidden)
    for label in range(1, count + 1):
        columns = np.flatnonzero(labels == label)
        lo = columns[0] * grid.resolution
        hi = (columns[-1] + 1) * grid.resolution
        if hi - lo >= min_width:
            spans.append((float(lo), float(hi)))
    return spans
