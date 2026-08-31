from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .openings import NormalizedOpening
from .planes import HorizontalFrame, WallSegment


@dataclass
class SurfaceGrid:
    """A grid laid flat over the face of one wall, used to find doors,
    windows, and other gaps in what was actually observed.

    Think of this as graph paper taped directly onto the wall: one axis
    (the columns) runs along the wall's length, and the other (the rows)
    runs up from the floor toward the ceiling. Each cell counts how many
    times a 3D point landed exactly on the wall's surface there, and how
    many times a point landed just in front of it instead -- which
    usually means something like a couch or a bookshelf is blocking the
    view of the wall at that spot, rather than there being an actual hole
    in it.

    Attributes:
        wall_index: Which wall (as an index into the full `WallSegment`
            list) this grid was built for.
        resolution: The size of one grid cell, in metres, along both
            axes.
        width: The wall's length, in metres -- how far the grid extends
            along its "along-wall" axis.
        height: The floor-to-ceiling height, in metres -- how far the
            grid extends along its "up" axis.
        base_height: The world-space height that counts as the bottom
            row of the grid (row 0), i.e. the floor.
        hits: A `(columns, rows)` grid of counts, one per cell, of how
            many observations landed right on the wall's own surface
            there.
        near: A `(columns, rows)` grid of counts, one per cell, of how
            many observations landed just in front of the wall instead
            of on it -- evidence of furniture or clutter blocking that
            spot.
    """

    wall_index: int
    resolution: float
    width: float
    height: float
    base_height: float
    hits: np.ndarray
    near: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """The `(columns, rows)` shape of this grid, read off the `hits`
        array (`hits` and `near` are always the same shape)."""
        return self.hits.shape

    def to_uv(self, points: np.ndarray, wall: WallSegment, frame: HorizontalFrame):
        """Converts 3D world points into flat (U, V) coordinates on this
        wall's own surface.

        This is like unrolling the wall into a flat rectangle: U measures
        distance along the wall starting from one end, and V measures
        height above the floor. Once points are in this flat coordinate
        system, it's easy to work out which grid cell each one belongs
        in.

        Args:
            points: An (N, 3) array of world-space points to project.
            wall: The wall this grid belongs to; supplies the starting
                point and direction used to measure the U axis.
            frame: The building's horizontal reference frame, used to
                flatten points into plan coordinates and measure height
                above the floor.

        Returns:
            An (N, 2) array of `(u, v)` coordinates in metres -- `u` is
            distance along the wall from `wall.start`, and `v` is height
            above `self.base_height`.
        """
        plan = frame.to_plan(points)
        u = (plan - wall.start) @ wall.direction
        v = frame.height(points) - self.base_height
        return np.stack([u, v], axis=-1)

    def to_cell(self, uv: np.ndarray) -> np.ndarray:
        """Converts surface (U, V) coordinates, in metres, into the
        integer grid cell they fall inside.

        Args:
            uv: An array of shape (..., 2) giving surface coordinates, in
                metres.

        Returns:
            An integer array of the same leading shape giving each
            point's (column, row) cell index. These indices are not
            checked against the grid's bounds, so a point outside the
            grid can come back with a negative or out-of-range index.
        """
        return np.floor(uv / self.resolution).astype(int)

    def cell_to_uv(self, cell: np.ndarray) -> np.ndarray:
        """Converts a grid cell index back into the surface coordinate at
        the centre of that cell.

        Args:
            cell: An array of shape (..., 2) giving integer (column, row)
                cell indices.

        Returns:
            The surface `(u, v)` coordinate, in metres, at the centre of
            each given cell.
        """
        return (np.asarray(cell, float) + 0.5) * self.resolution

    @property
    def observed(self) -> np.ndarray:
        """A boolean grid, True in every cell where at least one
        observation landed directly on the wall's own surface."""
        return self.hits > 0

    def area_of(self, mask: np.ndarray) -> float:
        """Turns a boolean cell mask into an actual area, in square
        metres, by counting the True cells and multiplying by the area
        of a single cell."""
        return float(mask.sum() * self.resolution**2)


@dataclass
class _LegacyOpening:
    """A gap found in one wall's observed surface -- most likely a door,
    window, or an open pass-through to another room.

    `find_openings` looks for spots on a wall's `SurfaceGrid` where the
    surrounding wall was clearly seen but a hole in the middle wasn't --
    exactly the pattern you'd expect from a doorway or window cut into an
    otherwise solid wall.

    Attributes:
        wall_index: Which wall (as an index into the full `WallSegment`
            list) this opening was found in.
        kind: What kind of opening this looks like: `"door"`, `"window"`,
            or `"pass-through"` (an opening that reaches the floor but is
            too short to be a normal door, like a wide archway).
        u_range: The `(low, high)` extent of the opening along the wall,
            in metres.
        v_range: The `(low, high)` height extent of the opening, in
            metres, measured above the grid's `base_height` (the floor).
        confidence: How completely the opening's bounding box is filled
            by actual hole cells, from 0 to 1. A lower value usually
            means the opening's edges were only partly captured by the
            sensor.
    """

    wall_index: int
    kind: str
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    confidence: float

    @property
    def width(self) -> float:
        """How wide the opening is along the wall, in metres."""
        return self.u_range[1] - self.u_range[0]

    @property
    def height(self) -> float:
        """How tall the opening is, in metres."""
        return self.v_range[1] - self.v_range[0]

    @property
    def sill_height(self) -> float:
        """How high the opening's bottom edge sits above the floor, in
        metres (0 for an opening that reaches the floor, like a door)."""
        return self.v_range[0]

    @property
    def header_height(self) -> float:
        """How high the opening's top edge sits above the floor, in
        metres."""
        return self.v_range[1]


# Keep the historical import path (`occupancy.Opening`) while making every
# producer use the source-neutral contract.
Opening = NormalizedOpening


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
    """Builds a `SurfaceGrid` for one wall by sorting every nearby 3D
    point into "on the wall" or "in front of the wall".

    Each point close enough to the wall's plane is checked against two
    distance bands: points within `plane_band` of the plane are treated
    as actually landing on the wall's surface, and points a bit further
    out (but still within `near_band`) are treated as landing on
    something in front of it, like furniture. Sorting points this way is
    what lets `find_openings`, later on, tell the difference between
    "there's a hole here because it's a doorway" and "there's a hole here
    because a sofa is blocking the sensor's view of the wall".

    Args:
        wall: The wall segment to build the grid for; supplies the
            wall's plane (`normal`, `offset`), its along-wall axis
            (`start`, `direction`), and its length.
        frame: The building's horizontal reference frame, used to
            flatten points into plan coordinates and measure height
            above the floor.
        points: An (N, 3) array of world-space points to sort into the
            grid.
        floor_height: The world-space height of the floor, in metres;
            this becomes row 0 (V = 0) of the grid.
        ceiling_height: The world-space height of the ceiling, in
            metres; together with `floor_height`, this sets how many
            rows the grid needs.
        resolution: The size of one grid cell, in metres, along both
            axes.
        plane_band: How far a point can be from the wall's plane, in
            metres, and still count as landing directly "on" the wall.
        near_band: How far beyond `plane_band` a point can be, in
            metres, and still count as being "in front of" the wall
            rather than somewhere else entirely.

    Returns:
        A `SurfaceGrid` sized to the wall's length and floor-to-ceiling
        height, with its `hits` and `near` counts filled in from
        `points`.
    """
    height = ceiling_height - floor_height
    columns = max(int(np.ceil(wall.length / resolution)), 1)
    rows = max(int(np.ceil(height / resolution)), 1)

    plan = frame.to_plan(points)
    distance = plan @ wall.normal - wall.offset
    along = (plan - wall.start) @ wall.direction
    up = frame.height(points) - floor_height

    # Only consider points that actually fall within this wall's own
    # footprint -- along its length and between the floor and ceiling --
    # before sorting them into "on the wall" versus "in front of it".
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
        near=_bin(along[in_front], up[in_front], resolution, columns, rows),
    )
    return grid


def _bin(
    u: np.ndarray, v: np.ndarray, resolution: float, columns: int, rows: int
) -> np.ndarray:
    """Counts how many (u, v) surface points fall into each cell of a 2D
    grid -- basically a 2D histogram.

    Args:
        u: Along-wall coordinates, in metres.
        v: Height-above-floor coordinates, in metres.
        resolution: The size of one grid cell, in metres.
        columns: How many along-wall cells the output grid should have.
        rows: How many vertical cells the output grid should have.

    Returns:
        A `(columns, rows)` grid of integer hit counts. A point that
        would otherwise fall outside the grid is clipped to the nearest
        edge cell rather than being dropped.
    """
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
    """Looks for holes in an otherwise-observed wall, and classifies each
    one as a door, window, or pass-through.

    The idea is simple: if the sensor clearly saw the wall all around a
    patch, but never saw anything at that patch itself, that patch is
    probably a real opening rather than just a spot the sensor happened
    to miss. This traces the outline of everywhere the wall was seen at
    all, fills in its interior, and then looks for holes inside that
    filled-in shape. A hole that reaches down to the floor and is tall
    enough counts as a door; one that reaches the floor but is too short
    counts as a pass-through (like a wide archway); anything else counts
    as a window.

    Args:
        grid: The wall's surface occupancy grid to search, built by
            `build_surface_grid`.
        min_width: The smallest along-wall size, in metres, a hole needs
            before it's reported as an opening.
        min_height: The smallest vertical size, in metres, a hole needs
            before it's reported as an opening.
        door_floor_tolerance: How close a hole's bottom edge needs to be
            to the floor, in metres, to still count as reaching it.
        min_door_height: How tall a floor-reaching opening needs to be,
            in metres, to be classified as a "door" instead of a
            "pass-through".

    Returns:
        One `Opening` per qualifying hole found, in no particular order.
    """
    observed = grid.observed
    if observed.sum() < 20:
        return []

    # Close up small gaps and fill in the outline of everywhere the wall
    # was seen, so what's being searched for below is real holes rather
    # than just noise between neighbouring observed cells.
    silhouette = ndimage.binary_closing(observed, np.ones((5, 5)))
    silhouette = ndimage.binary_fill_holes(silhouette)
    holes = silhouette & ~ndimage.binary_dilation(observed, np.ones((3, 3)))
    # A blank region backed by repeated near-plane returns is more likely a
    # sofa/cabinet than a real opening. Keep this gate separate from the
    # silhouette construction so sparse but clean gaps retain the old path.
    blocked = ndimage.binary_dilation(
        occluded_mask(grid, min_near=2), np.ones((3, 3))
    )
    holes &= ~blocked
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

        # How much of the hole's bounding box is actually filled by hole
        # cells, rather than being a ragged or partial shape -- a low
        # fill fraction is a sign this probably isn't a clean, real
        # opening.
        fill = len(cells) / max((width / grid.resolution) * (height / grid.resolution), 1)
        if fill < 0.45:
            continue

        # Require a wall-supported rim around the candidate. This prevents
        # an unobserved edge of the silhouette from becoming a measurement.
        component = labels == label
        rim = ndimage.binary_dilation(component, np.ones((5, 5))) & ~component
        rim_support = float(observed[rim].mean()) if rim.any() else 0.0
        if rim_support < 0.18:
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
                provenance=["geometry"],
                state="measured",
                uncertainty={
                    "u_sigma_m": float(grid.resolution / np.sqrt(12.0)),
                    "v_sigma_m": float(grid.resolution / np.sqrt(12.0)),
                    "rim_support": float(rim_support),
                    "basis": "surface occupancy cell resolution and observed rim support",
                },
                wall_association_confidence=1.0,
                wall_distance_m=0.0,
            )
        )
    return openings


def occluded_mask(grid: SurfaceGrid, min_near: int = 3) -> np.ndarray:
    """Finds cells on a wall that are probably hidden behind something,
    like a piece of furniture, rather than actually being a hole.

    A cell counts as occluded when the sensor never saw the wall's own
    surface there, but did see something else sitting just in front of
    it often enough to be real furniture or clutter rather than noise.

    Args:
        grid: The wall's surface occupancy grid to inspect.
        min_near: The fewest "in front of the wall" observations a cell
            needs before it's trusted as genuinely occluded, rather than
            just under-sampled.

    Returns:
        A boolean grid, True in every cell that has no on-wall
        observation but does have enough in-front-of-wall observations to
        count as blocked.
    """
    return (~grid.observed) & (grid.near >= min_near)


def occluded_spans(grid: SurfaceGrid, min_width: float = 0.25) -> list[tuple[float, float]]:
    """Finds continuous stretches along a wall that are mostly hidden
    behind furniture or other clutter.

    This looks at the wall one vertical column at a time and treats a
    column as "occluded" if more than 30% of its cells are blocked. Runs
    of neighbouring occluded columns are then grouped together and
    reported as a single span, as long as that span is wide enough to be
    worth mentioning.

    Args:
        grid: The wall's surface occupancy grid to inspect.
        min_width: The smallest along-wall size, in metres, a run of
            occluded columns needs before it's reported.

    Returns:
        A list of `(u_low, u_high)` along-wall ranges, in metres, each
        at least `min_width` wide, describing where the wall is mostly
        hidden from view.
    """
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
