from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .planes import HorizontalFrame, WallSegment
from .projection import DensityMap, project_wall_density


@dataclass
class PlanGrid:
    """A top-down grid of the space, used both to find rooms and to find
    doors and windows.

    Picture looking straight down at the building from above and dividing
    the floor into small square cells, like graph paper. For every cell,
    this keeps two counts: how many times a 3D point landed there at
    "wall height" (evidence that a wall runs through that cell), and how
    many times a point landed there at floor height (evidence that
    someone could actually stand there). Room segmentation and opening
    detection both build on this same grid, so they agree with each other
    about where the walls and open floor actually are.

    Attributes:
        resolution: The size of one grid cell, in metres. Smaller values
            give a more detailed grid but take longer to process.
        origin: The plan-space (x, y) coordinate that cell (0, 0) sits at,
            used to convert between real-world coordinates and grid
            cells.
        occupied: A grid of integers, one per cell, counting how many
            wall-height points landed in each cell.
        free: A grid of integers, one per cell, counting how many
            floor-height points (or camera positions -- see
            `build_plan_grid`) landed in each cell.
        wall_density: The typed wall-height projection that produced
            `occupied`, including its metric bounds and observed/empty masks;
            `None` for manually assembled compatibility grids.
    """

    resolution: float
    origin: np.ndarray
    occupied: np.ndarray
    free: np.ndarray
    wall_density: DensityMap | None = None

    @property
    def density(self) -> DensityMap | None:
        """The wall-height density projection used to build ``occupied``.

        Manually constructed ``PlanGrid`` instances may leave this as
        ``None``; grids produced by :func:`build_plan_grid` always carry the
        projection metadata for downstream vectorizers.
        """
        return self.wall_density

    def to_cell(self, plan: np.ndarray) -> np.ndarray:
        """Converts a real-world (x, y) position into the grid cell it
        falls inside.

        Args:
            plan: An array of plan-space (x, y) coordinates, in metres.

        Returns:
            An integer array of the same shape giving each point's
            (column, row) cell index.
        """
        return np.floor((plan - self.origin) / self.resolution).astype(int)

    def to_plan(self, cell: np.ndarray) -> np.ndarray:
        """Converts a grid cell index back into a real-world (x, y)
        position, at the centre of that cell.

        Args:
            cell: An array of integer (column, row) cell indices.

        Returns:
            The plan-space (x, y) coordinate, in metres, at the centre of
            each given cell.
        """
        return self.origin + (np.asarray(cell, float) + 0.5) * self.resolution

    @property
    def cell_area(self) -> float:
        """The area covered by a single grid cell, in square metres. This
        is just the cell size squared, and it's what turns a raw cell
        count into an actual area measurement elsewhere in this file."""
        return self.resolution**2


@dataclass
class Room:
    """One room found by `segment_rooms`, along with everything known
    about its shape and which walls bound it.

    Attributes:
        id: A number identifying this room, assigned in the order rooms
            were found during segmentation.
        name: An automatically generated name like "room_1", "room_2",
            and so on.
        area: The room's floor area, in square metres.
        centroid: The plan-space (x, y) position at the average of all
            the room's floor cells -- roughly its centre.
        floor_height: The world-space height of the floor, in metres.
        ceiling_height: The world-space height of the ceiling, in metres,
            or `None` if no ceiling plane could be found for this
            capture.
        wall_indices: Which walls (as indices into the full `walls` list)
            bound this room.
        neighbours: The ids of other rooms that share a boundary with
            this one.
        polygon: The (N, 2) plan-space vertices that trace the room's
            outline, or `None` if no usable outline could be traced.
    """

    id: int
    name: str
    area: float
    centroid: np.ndarray
    floor_height: float
    ceiling_height: float | None
    wall_indices: list[int] = field(default_factory=list)
    neighbours: list[int] = field(default_factory=list)
    polygon: np.ndarray | None = None

    @property
    def height(self) -> float | None:
        """The floor-to-ceiling height of this room, in metres, or `None`
        if no ceiling plane was found for this capture."""
        if self.ceiling_height is None:
            return None
        return self.ceiling_height - self.floor_height

    @property
    def perimeter(self) -> float:
        """The total length of this room's outline, in metres, found by
        adding up the length of every edge in `polygon`. Returns 0.0 if
        the room has no usable polygon."""
        if self.polygon is None or len(self.polygon) < 2:
            return 0.0
        closed = np.vstack([self.polygon, self.polygon[:1]])
        return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def build_plan_grid(
    points: np.ndarray,
    frame: HorizontalFrame,
    floor_height: float,
    ceiling_height: float | None,
    resolution: float = 0.04,
    wall_band: tuple[float, float] = (0.35, 1.9),
    floor_band: float = 0.12,
    trajectory: np.ndarray | None = None,
) -> PlanGrid:
    """Builds the top-down `PlanGrid` that room finding and opening
    detection both work from.

    Every point in the full 3D reconstruction gets sorted into one of two
    buckets, based on how high above the floor it sits: points in a
    "wall height" band (roughly waist to head height, which avoids
    furniture near the floor and light fixtures near the ceiling) count
    as evidence of a wall, and points close to floor height count as
    evidence of open, walkable floor. Each bucket is then dropped into
    its matching grid cell. Optionally, the camera's own path through the
    space is marked in its exact visited cells as additional direct
    evidence; neighbouring unknown cells are not filled in.

    Args:
        points: The full set of (N, 3) world-space points from the 3D
            reconstruction.
        frame: The building's horizontal reference frame, used to convert
            points into flat (x, y) plan coordinates and to measure how
            high above the floor each point is.
        floor_height: The world-space height of the floor, in metres.
        ceiling_height: The observed world-space height of the ceiling, in
            metres, or `None` when no ceiling plane was fitted.  No height
            inferred from the point-cloud extent is used.
        resolution: The size of one grid cell, in metres.
        wall_band: The `(low, high)` height range above the floor, in
            metres, that counts as wall evidence.
        floor_band: How close to `floor_height`, in metres, a point has
            to be to count as floor.
        trajectory: Optional (M, 3) world-space camera positions to also
            mark as walkable floor.

    Returns:
        A `PlanGrid` sized to cover every point with half a metre of
        margin around the edges, with wall and floor evidence already
        counted up per cell. Its `wall_density` field is the complete typed
        projection metadata passed to the vectorizer.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    floor_height = float(floor_height) if np.isfinite(floor_height) else 0.0
    wall_density = project_wall_density(
        points,
        frame,
        floor_height,
        ceiling_height,
        resolution=resolution,
        wall_band=wall_band,
    )
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return PlanGrid(
            resolution=resolution,
            origin=wall_density.origin,
            occupied=wall_density.counts,
            free=np.zeros((1, 1), dtype=np.int32),
            wall_density=wall_density,
        )
    heights = frame.height(points)
    plan = frame.to_plan(points)

    lower = wall_density.origin
    shape = np.asarray(wall_density.shape, dtype=int)

    floor_mask = np.abs(heights - floor_height) < floor_band

    free = _accumulate(plan[floor_mask], lower, shape, resolution)
    if trajectory is not None:
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim == 1 and trajectory.size == 0:
            trajectory = trajectory.reshape(0, 3)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError("trajectory must have shape (M, 3)")
        trajectory = trajectory[np.isfinite(trajectory).all(axis=1)]
    if trajectory is not None and len(trajectory):
        # The camera position is direct free-space evidence, but cells around
        # it are not.  Mark only the visited cells; dilation here used to
        # grow free space through completely unobserved regions.
        walked = _accumulate(frame.to_plan(trajectory), lower, shape, resolution)
        free = free + (walked > 0).astype(np.int32) * 3

    return PlanGrid(
        resolution=resolution,
        origin=lower,
        occupied=wall_density.counts,
        free=free,
        wall_density=wall_density,
    )


def _accumulate(
    plan: np.ndarray, origin: np.ndarray, shape: np.ndarray, resolution: float
) -> np.ndarray:
    """Counts how many points fall into each cell of a 2D grid.

    This is the basic building block `build_plan_grid` uses for both the
    wall and floor evidence grids: given a scatter of flat (x, y) points,
    it works out which cell each one lands in and tallies up the count.

    Args:
        plan: The (N, 2) plan-space (x, y) points to count up.
        origin: The plan-space coordinate that cell (0, 0) sits at.
        shape: The `(columns, rows)` size of the output grid.
        resolution: The size of one grid cell, in metres.

    Returns:
        A `shape`-sized grid of integer hit counts. Points that fall
        outside the grid's bounds are simply dropped, not counted.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim == 1 and plan.size == 0:
        plan = plan.reshape(0, 2)
    if plan.ndim != 2 or plan.shape[1] != 2:
        raise ValueError("plan must have shape (N, 2)")
    plan = plan[np.isfinite(plan).all(axis=1)]
    cells = np.floor((plan - origin) / resolution).astype(int)
    inside = (
        (cells[:, 0] >= 0)
        & (cells[:, 1] >= 0)
        & (cells[:, 0] < shape[0])
        & (cells[:, 1] < shape[1])
    )
    counts = np.zeros(tuple(shape), np.int32)
    np.add.at(counts, (cells[inside, 0], cells[inside, 1]), 1)
    return counts


def segment_rooms(
    grid: PlanGrid,
    walls: list[WallSegment],
    frame: HorizontalFrame,
    floor_height: float,
    ceiling_height: float | None,
    min_area: float = 1.4,
    wall_threshold: int = 6,
    floor_threshold: int = 3,
) -> list[Room]:
    """Extract rooms as bounded faces of the fitted wall graph.

    Room topology comes from measured wall segments rather than raster
    watershed labels.  This keeps boundaries on the wall geometry, makes
    output ordering deterministic, and never grows a room through cells for
    which no free-space evidence exists.  A bounded face is retained only
    when it contains observed floor evidence; an incomplete capture falls
    back to connected observed floor components without filling unknown
    cells.
    """
    # Occupied points are evidence of a wall, not evidence of walkable floor.
    # The graph itself supplies the face barriers; this mask only validates
    # that a proposed face was actually observed at floor height.
    free = (grid.free >= floor_threshold) & (grid.occupied < wall_threshold)
    polygons = polygonize_wall_graph(walls, min_area=min_area)
    accepted = [
        polygon for polygon in polygons if _has_observed_floor(polygon, grid, free)
    ]

    if not accepted:
        accepted = _observed_floor_polygons(grid, free, min_area)
    if not accepted:
        return []

    rooms: list[Room] = []
    for polygon in accepted:
        centroid = polygon.centroid
        rooms.append(
            Room(
                id=len(rooms),
                name=f"room_{len(rooms) + 1}",
                area=float(polygon.area),
                centroid=np.array([centroid.x, centroid.y], dtype=float),
                floor_height=floor_height,
                ceiling_height=ceiling_height,
                polygon=np.asarray(polygon.exterior.coords)[:-1],
            )
        )

    _assign_graph_walls(rooms, walls)
    _link_graph_neighbours(rooms, walls)
    return rooms


def polygonize_wall_graph(
    walls: list[WallSegment],
    *,
    min_area: float = 1.4,
    node_tolerance: float = 0.08,
) -> list[object]:
    """Return bounded Shapely faces formed by the usable wall graph.

    Wall intersections are noded before polygonization, so crossing wall
    segments become graph vertices.  ``node_tolerance`` snaps tiny endpoint
    gaps at graph nodes; it is deliberately small relative to a wall and is
    not applied across the interior of a wall.
    """
    from shapely.geometry import LineString
    from shapely.ops import polygonize, snap, unary_union

    lines = []
    for wall in walls:
        if wall.quarantined:
            continue
        start = np.asarray(wall.start, dtype=float)
        end = np.asarray(wall.end, dtype=float)
        if (
            start.shape != (2,)
            or end.shape != (2,)
            or not np.isfinite(start).all()
            or not np.isfinite(end).all()
            or np.linalg.norm(end - start) <= 1e-6
        ):
            continue
        lines.append(LineString([start, end]))
    if len(lines) < 3:
        return []
    network = unary_union(lines)
    if node_tolerance > 0:
        network = snap(network, network, node_tolerance)
        network = unary_union(network)
    faces = [face for face in polygonize(network) if face.area >= min_area]
    return sorted(
        faces,
        key=lambda face: (
            round(float(face.centroid.x), 8),
            round(float(face.centroid.y), 8),
            round(float(face.area), 8),
            tuple(
                round(value, 8)
                for point in face.exterior.coords
                for value in point
            ),
        ),
    )


# Private alias retained for callers that used the geometry-stage naming.
_polygonize_wall_graph = polygonize_wall_graph


def _has_observed_floor(polygon, grid: PlanGrid, free: np.ndarray) -> bool:
    from shapely.geometry import Point
    from shapely.prepared import prep

    cells = np.argwhere(free)
    if not len(cells):
        return False
    shape = prep(polygon)
    return any(shape.covers(Point(x, y)) for x, y in grid.to_plan(cells))


def _observed_floor_polygons(
    grid: PlanGrid, free: np.ndarray, min_area: float
) -> list[object]:
    from shapely.geometry import box
    from shapely.ops import unary_union

    labels, count = ndimage.label(free, structure=np.ones((3, 3), dtype=int))
    polygons = []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        if len(cells) < 3:
            continue
        # Union cell footprints instead of taking a convex hull.  A hull can
        # bridge a doorway-sized unknown gap and thereby manufacture free
        # space that was never observed.
        points = grid.to_plan(cells)
        polygon = unary_union(
            [
                box(
                    point[0] - grid.resolution / 2,
                    point[1] - grid.resolution / 2,
                    point[0] + grid.resolution / 2,
                    point[1] + grid.resolution / 2,
                )
                for point in points
            ]
        )
        # Room's public schema carries one exterior ring.  Do not discard
        # an interior ring here: doing so would claim an unobserved hole is
        # free space.  A future schema can represent holes explicitly.
        if (
            polygon.geom_type == "Polygon"
            and not polygon.interiors
            and polygon.area >= min_area
        ):
            polygons.append(polygon)
    return sorted(
        polygons,
        key=lambda face: (round(face.centroid.x, 8), round(face.centroid.y, 8)),
    )


def _assign_graph_walls(rooms: list[Room], walls: list[WallSegment]) -> None:
    from shapely.geometry import LineString

    for wall in walls:
        if wall.quarantined or wall.length <= 1e-6:
            continue
        line = LineString([wall.start, wall.end])
        for room in rooms:
            boundary = LineString(
                np.vstack([room.polygon, room.polygon[:1]])
            )
            shared = line.intersection(boundary).length
            if shared >= min(0.15, 0.35 * wall.length):
                room.wall_indices.append(wall.index)
                if wall.room_id is None:
                    wall.room_id = room.id
    for room in rooms:
        room.wall_indices = sorted(set(room.wall_indices))
        _name_walls(room, walls)


def _link_graph_neighbours(rooms: list[Room], walls: list[WallSegment]) -> None:
    membership: dict[int, list[int]] = {}
    for room in rooms:
        room.neighbours = []
        for wall_index in room.wall_indices:
            membership.setdefault(wall_index, []).append(room.id)
    for ids in membership.values():
        unique = sorted(set(ids))
        for index, first in enumerate(unique):
            for second in unique[index + 1 :]:
                rooms[first].neighbours.append(second)
                rooms[second].neighbours.append(first)
    for room in rooms:
        room.neighbours = sorted(set(room.neighbours))


def _grow_to_walls(
    labels: np.ndarray,
    barrier: np.ndarray,
    max_steps: int = 14,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Optionally grow labels through explicitly observed free cells.

    This compatibility helper is no longer part of room extraction.  When
    used by a caller, ``allowed`` must explicitly identify observed free
    cells.  With no allow-mask, no unassigned cells are eligible, preventing
    the old behaviour of growing through unknown space.

    Args:
        labels: The integer label grid from the watershed step, where 0
            means "not yet assigned to any room".
        barrier: A boolean grid marking wall cells that growth must never
            cross.
        max_steps: The most times to repeat the one-cell growth step.
            Each step can only reach one cell further out, so this also
            caps how big a gap can be filled.

    Returns:
        A copy of `labels` with only explicitly allowed cells filled in by
        growth from their nearest labelled neighbour, up to `max_steps`
        cells away.
    """
    grown = labels.copy()
    free = (~barrier) & (np.asarray(allowed, dtype=bool) if allowed is not None else False)
    cross = ndimage.generate_binary_structure(2, 1)

    for _ in range(max_steps):
        unassigned = free & (grown == 0)
        if not unassigned.any():
            break
        expanded = ndimage.grey_dilation(grown, footprint=cross)
        newly = unassigned & (expanded > 0)
        if not newly.any():
            break
        grown[newly] = expanded[newly]
    return grown


def _rasterise_walls(grid: PlanGrid, walls: list[WallSegment]) -> np.ndarray:
    """Draws each fitted wall as a solid line on the grid, to use as a
    barrier during room segmentation.

    The raw point evidence used to build `grid.occupied` can leave small
    gaps in a wall -- for instance where an open door let the sensor see
    straight through to the next room. Since the wall-fitting stage
    already knows exactly where each wall actually is, drawing those wall
    lines directly onto the grid closes up those gaps and stops rooms
    from incorrectly bleeding into each other through them.

    Args:
        grid: The target grid; supplies the resolution, origin, and
            output shape to draw into.
        walls: The fitted wall segments to draw.

    Returns:
        A boolean grid, `True` wherever a wall line (thickened by one
        cell in every direction) passes through.
    """
    barrier = np.zeros(grid.occupied.shape, bool)
    for wall in walls:
        if wall.quarantined:
            continue
        steps = max(int(wall.length / grid.resolution) * 2, 2)
        samples = wall.start + np.linspace(0, 1, steps)[:, None] * (
            wall.end - wall.start
        )
        cells = grid.to_cell(samples)
        inside = (
            (cells[:, 0] >= 0)
            & (cells[:, 1] >= 0)
            & (cells[:, 0] < barrier.shape[0])
            & (cells[:, 1] < barrier.shape[1])
        )
        barrier[cells[inside, 0], cells[inside, 1]] = True
    return ndimage.binary_dilation(barrier, np.ones((3, 3)))


def _boundary_polygon(
    cells: np.ndarray, grid: PlanGrid, snap_degrees: float = 32.0
) -> np.ndarray | None:
    """Traces the outer edge of a room's cells and turns it into a clean,
    mostly-rectangular polygon.

    Args:
        cells: A boolean grid, `True` wherever this room's floor cells
            are.
        grid: The `PlanGrid` that `cells` was rasterised into, used to
            convert cell positions back to real-world coordinates.
        snap_degrees: How far, in degrees, an edge can be from being
            perfectly horizontal or vertical and still get snapped onto
            that axis. Most real rooms have walls that are meant to be
            straight, so this cleans up the small wobble left over from
            working on a grid.

    Returns:
        The rectified (N, 2) plan-space polygon vertices tracing the
        room's outline (not closed -- the first and last point are not
        repeated), or `None` if no outline could be traced from `cells`
        at all.
    """
    from skimage import measure

    padded = np.pad(cells.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None

    contour = max(contours, key=len) - 1.0
    simplified = measure.approximate_polygon(contour, tolerance=1.5)
    polygon = grid.origin + (simplified + 0.5) * grid.resolution
    if len(polygon) < 4:
        return polygon
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    return _rectify(polygon, snap_degrees)


def _polygon_area(
    polygon: np.ndarray | None, cell_area: float
) -> tuple[float, np.ndarray | None]:
    """Works out the most trustworthy area figure for a room, and cleans
    up its polygon along the way.

    The rectified polygon should give a more accurate area than simply
    counting grid cells, but the rectification process (snapping edges to
    axes and rebuilding corners) can occasionally distort a room's shape
    too much to trust. This checks the polygon's area against the raw
    cell-count area, and only uses the polygon when the two roughly
    agree; otherwise it falls back to the simpler cell count.

    Args:
        polygon: The rectified (N, 2) plan-space polygon from
            `_boundary_polygon`, or `None` if none could be traced.
        cell_area: The room's area from simply counting cells (cell count
            times `grid.cell_area`), used both as a fallback and as a
            sanity check on the polygon.

    Returns:
        A tuple of `(area, polygon)`. `area` comes from the polygon when
        it is a valid shape and its area falls within 60% to 160% of
        `cell_area`; otherwise `area` falls back to `cell_area`.
        `polygon` is the repaired vertex array in the accepted case, or
        passed through unchanged otherwise.
    """
    if polygon is None or len(polygon) < 3:
        return cell_area, polygon

    from shapely.geometry import Polygon

    shape = Polygon(polygon)
    if not shape.is_valid:
        # A self-intersecting or otherwise invalid polygon can often be
        # repaired by buffering it by zero -- a common trick for cleaning
        # up small geometry errors without meaningfully changing the
        # shape.
        shape = shape.buffer(0)
    if shape.is_empty or shape.geom_type != "Polygon":
        return cell_area, polygon

    area = float(shape.area)
    if not 0.6 * cell_area <= area <= 1.6 * cell_area:
        return cell_area, polygon
    return area, np.asarray(shape.exterior.coords)[:-1]


def _rectify(
    polygon: np.ndarray, snap_degrees: float, min_edge: float = 0.25
) -> np.ndarray:
    """Cleans up a room outline traced from a grid into straight,
    axis-aligned edges with sharp corners.

    A polygon traced directly off a grid of square cells tends to look
    stair-stepped rather than straight, even where the real wall is
    perfectly flat. Since most rooms are built from walls that run either
    north-south or east-west, this snaps any edge that's already close to
    one of those directions onto it exactly, merges edges that end up
    pointing the same way, and then rebuilds each corner as the exact
    intersection of the two edges that meet there.

    Args:
        polygon: The raw, simplified (N, 2) plan-space polygon to clean
            up.
        snap_degrees: How far, in degrees, an edge can be from an axis
            and still get snapped onto it.
        min_edge: The shortest edge length, in metres, allowed to survive
            the initial filtering step; shorter edges are treated as
            noise and dropped.

    Returns:
        The rectified (N, 2) polygon, with corners recovered by
        intersecting adjacent edges, or the original `polygon` unchanged
        if too few usable edges survived to rebuild a shape from.
    """
    count = len(polygon)
    edges: list[tuple[np.ndarray, np.ndarray, float]] = []

    for index in range(count):
        start, end = polygon[index], polygon[(index + 1) % count]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < min_edge:
            continue
        direction = delta / length
        angle = np.degrees(np.arctan2(abs(direction[1]), abs(direction[0])))
        if angle < snap_degrees:
            direction = np.array([1.0, 0.0])
        elif angle > 90 - snap_degrees:
            direction = np.array([0.0, 1.0])
        edges.append((direction, 0.5 * (start + end), length))

    if len(edges) < 3:
        return polygon

    # Neighbouring edges that both snapped to the same axis are really
    # one edge that got split by noise -- merge them into a single edge,
    # weighted by length, so the corner rebuild below doesn't create a
    # spurious extra vertex.
    merged: list[tuple[np.ndarray, np.ndarray, float]] = []
    for direction, point, length in edges:
        if merged and abs(float(merged[-1][0] @ direction)) > 0.999:
            previous_direction, previous_point, weight = merged[-1]
            merged[-1] = (
                previous_direction,
                (previous_point * weight + point * length) / (weight + length),
                weight + length,
            )
        else:
            merged.append((direction, point, length))

    if len(merged) > 2 and abs(float(merged[0][0] @ merged[-1][0])) > 0.999:
        # The last edge wrapped back around to point the same way as the
        # first one -- merge those two as well, since the polygon is a
        # closed loop.
        direction, point, weight = merged.pop()
        first_direction, first_point, first_weight = merged[0]
        merged[0] = (
            first_direction,
            (first_point * first_weight + point * weight) / (first_weight + weight),
            first_weight + weight,
        )

    if len(merged) < 3:
        return polygon

    vertices = []
    for index in range(len(merged)):
        a_direction, a_point, _ = merged[index]
        b_direction, b_point, _ = merged[(index + 1) % len(merged)]
        corner = _intersect(a_direction, a_point, b_direction, b_point)
        vertices.append(corner if corner is not None else b_point)
    return np.asarray(vertices)


def _intersect(
    direction_a: np.ndarray,
    point_a: np.ndarray,
    direction_b: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray | None:
    """Finds where two lines cross, with each line given as a direction
    and a point that sits on it.

    Args:
        direction_a: The unit direction vector of the first line.
        point_a: Any point on the first line.
        direction_b: The unit direction vector of the second line.
        point_b: Any point on the second line.

    Returns:
        The (x, y) point where the two lines intersect, or `None` if the
        lines are parallel and never cross.
    """
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) < 1e-9:
        return None
    delta = point_b - point_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return point_a + direction_a * t


def _assign_walls(
    rooms: list[Room], walls: list[WallSegment], grid: PlanGrid, labels: np.ndarray
) -> None:
    """Works out which room each wall belongs to, by checking which
    room's floor sits on either side of it.

    For each wall, this samples a line of points running just off both
    faces of the wall and checks which room's cells those probe points
    land in. Whichever room shows up often enough on a given side is
    considered to be bounded by that wall. A wall can end up belonging to
    more than one room if it separates two of them.

    Args:
        rooms: The rooms to attach walls to; each room's `wall_indices`
            list is updated in place.
        walls: The fitted wall segments to check; each wall's `room_id`
            is set in place, to whichever room claimed it first.
        grid: The rasterised grid, used to convert probe points into cell
            indices.
        labels: The watershed label grid, with rooms encoded as negative
            numbers (see `segment_rooms`).
    """
    for wall in walls:
        steps = max(int(wall.length / grid.resolution), 2)
        samples = wall.start + np.linspace(0.05, 0.95, steps)[:, None] * (
            wall.end - wall.start
        )
        touching: dict[int, int] = {}
        for side in (+1, -1):
            # Nudge the sample points slightly off the wall's own line,
            # to either side, so they land inside the neighbouring room's
            # floor rather than sitting exactly on the wall itself.
            probes = samples + wall.normal * side * 0.12
            cells = grid.to_cell(probes)
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 1] >= 0)
                & (cells[:, 0] < labels.shape[0])
                & (cells[:, 1] < labels.shape[1])
            )
            values = labels[cells[inside, 0], cells[inside, 1]]
            for value in values[values < 0]:
                room_id = -int(value) - 1
                touching[room_id] = touching.get(room_id, 0) + 1

        for room_id, count in touching.items():
            if count < max(3, steps * 0.15):
                continue
            rooms[room_id].wall_indices.append(wall.index)
            if wall.room_id is None:
                wall.room_id = room_id

    for room in rooms:
        _name_walls(room, walls)


def _name_walls(room: Room, walls: list[WallSegment]) -> None:
    """Gives each of a room's walls a compass-direction name, the way a
    property estimator would refer to them (for example,
    "room_1.north_wall").

    Args:
        room: The room whose walls are being named; the room itself
            isn't changed, only the walls it points to.
        walls: The full list of walls, used to look up the actual
            `WallSegment` objects from `room.wall_indices`.
    """
    lookup = {wall.index: wall for wall in walls}
    members = [lookup[i] for i in room.wall_indices if i in lookup]
    used: dict[str, int] = {}

    for wall in sorted(members, key=lambda w: -w.length):
        inward = wall.normal
        if (room.centroid - wall.midpoint) @ inward < 0:
            inward = -inward
        base = f"{room.name}.{_compass(-inward)}_wall"
        count = used.get(base, 0)
        used[base] = count + 1
        wall.name = base if count == 0 else f"{base}_{count + 1}"


def _compass(direction: np.ndarray) -> str:
    """Turns a flat direction vector into a compass-point name, treating
    plan-space +y as north.

    Args:
        direction: A (2,) plan-space direction vector; only its angle
            matters, not its length.

    Returns:
        One of the 8 compass points: "north", "northeast", "east",
        "southeast", "south", "southwest", "west", or "northwest".
    """
    angle = np.degrees(np.arctan2(direction[0], direction[1])) % 360
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return names[int((angle + 22.5) // 45) % 8]


def _link_neighbours(rooms: list[Room], labels: np.ndarray, grid: PlanGrid) -> None:
    """Works out which rooms are next to each other, by checking whether
    their floor areas come close together anywhere.

    Args:
        rooms: The rooms to link; each room's `neighbours` list is reset
            and then filled back in, in place.
        labels: The watershed label grid, with rooms encoded as negative
            numbers (see `segment_rooms`).
        grid: Not used directly here, but kept as an argument so every
            helper in this file shares a consistent signature.
    """
    for room in rooms:
        room.neighbours = []
    for a in rooms:
        for b in rooms:
            if b.id <= a.id:
                continue
            if _shares_boundary(labels, a.id, b.id):
                a.neighbours.append(b.id)
                b.neighbours.append(a.id)


def _shares_boundary(labels: np.ndarray, a: int, b: int, radius: int = 6) -> bool:
    """Checks whether two rooms' floor areas come within a few cells of
    each other -- close enough to count as neighbours.

    Rooms are rarely separated by literally zero gap once walls are
    drawn in, so this grows room `a`'s area outward by a small margin
    first and then checks whether that expanded area touches any of room
    `b`'s cells.

    Args:
        labels: The watershed label grid, with rooms encoded as negative
            numbers (see `segment_rooms`).
        a: The first room's id.
        b: The second room's id.
        radius: How many cells to grow room `a`'s area outward by, in
            every direction, before testing for overlap with room `b`.

    Returns:
        `True` if the two rooms come within `radius` cells of each
        other.
    """
    mask_a = ndimage.binary_dilation(labels == -(a + 1), np.ones((radius, radius)))
    return bool((mask_a & (labels == -(b + 1))).any())


def check_no_overlaps(
    rooms: list[Room], tolerance_fraction: float = 0.01
) -> list[tuple[int, int, float]]:
    """Sanity-checks the segmented rooms by looking for pairs whose
    polygons overlap more than they reasonably should.

    Rooms are supposed to divide up the floor without their outlines
    overlapping each other, but small geometry errors during
    segmentation can occasionally produce polygons that overlap
    slightly. This flags any pair that overlaps by more than a small
    tolerance, so a caller can decide whether the result is trustworthy
    enough to use.

    Args:
        rooms: The rooms to check, each with an optional `polygon`.
            Rooms without a usable polygon are simply skipped.
        tolerance_fraction: The largest allowed overlap area, as a
            fraction of the smaller of the two rooms' own areas, before a
            pair gets reported as a violation.

    Returns:
        A list of `(room_id_a, room_id_b, overlap_fraction)` for every
        pair whose overlap exceeds `tolerance_fraction`. An empty list
        means every pair of rooms is within tolerance.
    """
    from shapely.geometry import Polygon

    polygons: dict[int, Polygon] = {}
    for room in rooms:
        if room.polygon is None or len(room.polygon) < 3:
            continue
        polygon = Polygon(room.polygon)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.area > 0:
            polygons[room.id] = polygon

    violations: list[tuple[int, int, float]] = []
    ids = sorted(polygons)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            poly_a, poly_b = polygons[a], polygons[b]
            overlap = poly_a.intersection(poly_b).area
            if overlap <= 0:
                continue
            smaller = min(poly_a.area, poly_b.area)
            fraction = overlap / smaller if smaller > 0 else 0.0
            if fraction > tolerance_fraction:
                violations.append((a, b, float(fraction)))
    return violations
