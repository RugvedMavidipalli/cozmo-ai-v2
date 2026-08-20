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
) -> PlanGrid:
    """Rasterise wall evidence and observed floor into a common grid."""
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

    grid = PlanGrid(
        resolution=resolution,
        origin=lower,
        occupied=_accumulate(plan[wall_mask], lower, shape, resolution),
        free=_accumulate(plan[floor_mask], lower, shape, resolution),
    )
    return grid


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

    rooms: list[Room] = []
    for label in range(1, labels.max() + 1):
        cells = labels == label
        area = float(cells.sum() * grid.cell_area)
        if area < min_area:
            continue
        indices = np.argwhere(cells)
        centroid = grid.to_plan(indices.mean(axis=0))
        room = Room(
            id=len(rooms),
            name=f"room_{len(rooms) + 1}",
            area=area,
            centroid=centroid,
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            polygon=_boundary_polygon(cells, grid),
        )
        rooms.append(room)
        labels[cells] = -(room.id + 1)  # renumber to the compacted room ids

    _assign_walls(rooms, walls, grid, labels)
    _link_neighbours(rooms, labels, grid)
    return rooms


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


def _boundary_polygon(cells: np.ndarray, grid: PlanGrid) -> np.ndarray | None:
    """Outer boundary of a room's cells, simplified to a polygon."""
    from skimage import measure

    padded = np.pad(cells.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) - 1.0
    simplified = measure.approximate_polygon(contour, tolerance=1.2)
    return grid.origin + simplified * grid.resolution


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
