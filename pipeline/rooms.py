"""Segment a capture into rooms and build the adjacency graph.

Rooms are derived from the floor's free space, not from the trajectory: a
capture may start in a hallway outside the unit, wander back through a room it
already visited, or cover one room from two directions.  Watershed over the
free-space distance transform cuts at the narrow necks -- doorways -- which is
where a floor plan should be cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .planes import HorizontalFrame, WallSegment


@dataclass
class PlanGrid:
    """Rasterised plan view shared by room finding and opening detection."""

    resolution: float
    origin: np.ndarray  # plan coordinate of cell (0, 0)
    occupied: np.ndarray  # wall evidence
    free: np.ndarray  # observed floor

    def to_cell(self, plan: np.ndarray) -> np.ndarray:
        return np.floor((plan - self.origin) / self.resolution).astype(int)

    def to_plan(self, cell: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(cell, float) + 0.5) * self.resolution

    @property
    def cell_area(self) -> float:
        return self.resolution**2


@dataclass
class Room:
    id: int
    name: str
    area: float  # m^2, from the segmented floor cells
    centroid: np.ndarray  # plan coordinates
    floor_height: float
    ceiling_height: float | None
    wall_indices: list[int] = field(default_factory=list)
    neighbours: list[int] = field(default_factory=list)
    polygon: np.ndarray | None = None  # (N,2) plan-space boundary

    @property
    def height(self) -> float | None:
        if self.ceiling_height is None:
            return None
        return self.ceiling_height - self.floor_height

    @property
    def perimeter(self) -> float:
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
    """Rasterise wall evidence and observed floor into a common grid.

    The operator's own path is counted as floor.  Depth sensors see little of
    the floor when the camera is held level and swept across walls, so whole
    rooms can end up with too little floor evidence to seed -- but anywhere
    the camera physically travelled is certainly walkable floor.
    """
    heights = frame.height(points)
    plan = frame.to_plan(points)

    lower = np.minimum(plan.min(axis=0) - 0.5, plan.min(axis=0))
    upper = plan.max(axis=0) + 0.5
    shape = np.ceil((upper - lower) / resolution).astype(int) + 1

    ceiling = ceiling_height if ceiling_height is not None else heights.max()
    wall_mask = (heights > floor_height + wall_band[0]) & (
        heights < min(floor_height + wall_band[1], ceiling - 0.15)
    )
    floor_mask = np.abs(heights - floor_height) < floor_band

    free = _accumulate(plan[floor_mask], lower, shape, resolution)
    if trajectory is not None and len(trajectory):
        walked = _accumulate(frame.to_plan(trajectory), lower, shape, resolution)
        # A footprint is strong evidence, so it clears the floor threshold on
        # its own; dilating covers the operator's body width.
        free = free + ndimage.grey_dilation(
            (walked > 0).astype(np.int32) * 10, size=(5, 5)
        )

    return PlanGrid(
        resolution=resolution,
        origin=lower,
        occupied=_accumulate(plan[wall_mask], lower, shape, resolution),
        free=free,
    )


def _accumulate(
    plan: np.ndarray, origin: np.ndarray, shape: np.ndarray, resolution: float
) -> np.ndarray:
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
    """Watershed the free space into rooms, then attach walls to each.

    Wall evidence is drawn from the *fitted* wall lines rather than raw points:
    a doorway the operator walked through leaves a gap in the point cloud on
    both sides, and raw evidence alone leaks one room into the next.
    """
    from skimage.segmentation import watershed

    barrier = grid.occupied >= wall_threshold
    barrier |= _rasterise_walls(grid, walls)
    free = (grid.free >= floor_threshold) & ~barrier
    free = ndimage.binary_opening(free, np.ones((3, 3)))
    free = ndimage.binary_closing(free, np.ones((3, 3)))

    distance = ndimage.distance_transform_edt(free) * grid.resolution
    # Seeds are the cores of open areas; the 0.55 m radius is about half a
    # doorway, so a corridor neck never seeds a room of its own.
    seeds, _ = ndimage.label(distance > 0.55)
    if seeds.max() == 0:
        seeds, _ = ndimage.label(free)
    labels = watershed(-distance, seeds, mask=free)
    labels = _grow_to_walls(labels, barrier)

    rooms: list[Room] = []
    for label in range(1, labels.max() + 1):
        cells = labels == label
        cell_area = float(cells.sum() * grid.cell_area)
        if cell_area < min_area:
            continue
        indices = np.argwhere(cells)
        centroid = grid.to_plan(indices.mean(axis=0))
        area, polygon = _polygon_area(_boundary_polygon(cells, grid), cell_area)
        room = Room(
            id=len(rooms),
            name=f"room_{len(rooms) + 1}",
            area=area,
            centroid=centroid,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            polygon=polygon,
        )
        rooms.append(room)
        labels[cells] = -(room.id + 1)  # renumber to the compacted room ids

    _assign_walls(rooms, walls, grid, labels)
    _link_neighbours(rooms, labels, grid)
    return rooms


def _grow_to_walls(
    labels: np.ndarray, barrier: np.ndarray, max_steps: int = 14
) -> np.ndarray:
    """Expand each room across unobserved floor until it meets a wall.

    A room's floor is only partly observed: furniture casts shadows, and the
    sensor sees little floor near the walls when the camera is held level.
    Reporting the observed region as the room's area understates it by
    whatever the furniture covered, which is exactly the bias the 2% floor
    area gate does not tolerate.  Since a room is bounded by its walls, the
    labelled region is dilated into the unlabelled gaps and stopped by
    barriers -- recovering the enclosed floor rather than the visible floor.

    The step limit matters as much as the barrier.  Wall evidence is never a
    closed curve -- doorways, unscanned spans and grazing dropout all leave
    gaps -- so unbounded growth escapes through them and floods a neighbouring
    space.  Capping the reach at ~0.6 m fills furniture shadows and the
    sensor's blind strip along the skirting, which is what the bias actually
    comes from, while keeping an escape confined to a doorway's depth.
    """
    grown = labels.copy()
    free = ~barrier
    cross = ndimage.generate_binary_structure(2, 1)

    for _ in range(max_steps):
        unassigned = free & (grown == 0)
        if not unassigned.any():
            break
        # Dilate every label at once, then keep expansions only where the
        # cell was previously unclaimed: simultaneous growth means competing
        # rooms meet in the middle instead of one flooding the other.
        expanded = ndimage.grey_dilation(grown, footprint=cross)
        newly = unassigned & (expanded > 0)
        if not newly.any():
            break
        grown[newly] = expanded[newly]
    return grown


def _rasterise_walls(grid: PlanGrid, walls: list[WallSegment]) -> np.ndarray:
    """Draw fitted wall lines into the grid as barriers."""
    barrier = np.zeros(grid.occupied.shape, bool)
    for wall in walls:
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
    """Outer boundary of a room's cells as a rectilinear polygon.

    The raw marching-squares contour is a staircase of 4 cm steps, and its
    length is meaningless: a 14 m2 room came out with a 48 m perimeter, which
    would then inflate both the floor-area interval and the mold containment
    barrier that is priced off perimeter.

    Plan coordinates are already expressed in the building's Manhattan frame,
    so a room's edges should lie along the axes.  Edges close to an axis are
    snapped to it and consecutive co-directional edges merged; corners are then
    recovered by intersecting adjacent edge lines.  Genuinely angled walls
    (beyond `snap_degrees`) are left alone.
    """
    from skimage import measure

    padded = np.pad(cells.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None

    contour = max(contours, key=len) - 1.0
    # Tolerance is in cells. `_rectify` discards sub-25 cm edges, which is what
    # actually removes the staircase, so this only needs to be large enough to
    # keep the vertex count manageable -- and a larger value would cut real
    # corners, biasing every room's area low.
    simplified = measure.approximate_polygon(contour, tolerance=1.5)
    # A cell's extent is [i, i+1) in contour coordinates, so its centre is at
    # i + 0.5.  Omitting the half-cell shift offsets every room boundary
    # inward by 2 cm, which the 2% floor-area gate cannot spare.
    polygon = grid.origin + (simplified + 0.5) * grid.resolution
    if len(polygon) < 4:
        return polygon
    if np.allclose(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    return _rectify(polygon, snap_degrees)


def _polygon_area(
    polygon: np.ndarray | None, cell_area: float
) -> tuple[float, np.ndarray | None]:
    """Area and cleaned outline for a room, from its rectified polygon.

    Area and perimeter must come from the same shape or the scope's
    perimeter-priced items (containment barrier) will not match its
    area-priced ones.  Rectification can occasionally produce a self-touching
    outline, so the polygon is validated; if it cannot be repaired, or if it
    disagrees wildly with the rasterised cells, the cell count wins because it
    cannot be wrong by construction.
    """
    if polygon is None or len(polygon) < 3:
        return cell_area, polygon

    from shapely.geometry import Polygon

    shape = Polygon(polygon)
    if not shape.is_valid:
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
    """Snap near-axis edges to the axes and rebuild corners by intersection.

    Short edges are discarded before merging.  A rasterised staircase
    *alternates* horizontal and vertical steps, so merging only consecutive
    co-directional edges would never collapse one -- the steps are individually
    axis-aligned and each interrupts the next.  Dropping every edge below
    `min_edge` removes the steps while keeping real features like an alcove,
    and what survives are the room's actual runs.
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

    # Merge consecutive co-directional edges, weighting each by its length so a
    # long run places the line and a leftover stub barely moves it.
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

    # The first and last edges are also neighbours around the loop.
    if len(merged) > 2 and abs(float(merged[0][0] @ merged[-1][0])) > 0.999:
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
    """Intersection of two lines given as (direction, point-on-line)."""
    denominator = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(denominator) < 1e-9:  # parallel: no corner to recover
        return None
    delta = point_b - point_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return point_a + direction_a * t


def _assign_walls(
    rooms: list[Room], walls: list[WallSegment], grid: PlanGrid, labels: np.ndarray
) -> None:
    """Attach each wall to the rooms whose free space it bounds.

    A wall is sampled just off each face; an interior partition therefore
    lands in two rooms and becomes a shared wall, while an exterior wall lands
    in one.
    """
    for wall in walls:
        steps = max(int(wall.length / grid.resolution), 2)
        samples = wall.start + np.linspace(0.05, 0.95, steps)[:, None] * (
            wall.end - wall.start
        )
        touching: dict[int, int] = {}
        for side in (+1, -1):
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
    """Give each of a room's walls a compass name an estimator would use."""
    lookup = {wall.index: wall for wall in walls}
    for index in room.wall_indices:
        wall = lookup.get(index)
        if wall is None:
            continue
        # Point the normal into the room, then name the wall for the side it
        # sits on: a wall on the room's north side faces south.
        inward = wall.normal
        if (room.centroid - wall.midpoint) @ inward < 0:
            inward = -inward
        compass = _compass(-inward)
        wall.name = f"{room.name}.{compass}_wall"


def _compass(direction: np.ndarray) -> str:
    """Name a plan-space direction, with plan +y treated as north."""
    angle = np.degrees(np.arctan2(direction[0], direction[1])) % 360
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return names[int((angle + 22.5) // 45) % 8]


def _link_neighbours(rooms: list[Room], labels: np.ndarray, grid: PlanGrid) -> None:
    """Connect rooms whose free space is separated by a single barrier."""
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
    mask_a = ndimage.binary_dilation(labels == -(a + 1), np.ones((radius, radius)))
    return bool((mask_a & (labels == -(b + 1))).any())
