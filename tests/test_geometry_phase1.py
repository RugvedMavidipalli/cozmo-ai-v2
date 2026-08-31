from __future__ import annotations

import numpy as np

from cozmo_ai_v2.pipeline.geometry import estimate_gravity
from cozmo_ai_v2.pipeline.planes import (
    HorizontalFrame,
    WallSegment,
    merge_collinear,
    snap_to_frame,
    wall_band_mask,
)
from cozmo_ai_v2.pipeline.rooms import (
    PlanGrid,
    _grow_to_walls,
    polygonize_wall_graph,
    segment_rooms,
)


def _wall(index, start, end, normal, support=100):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    normal = np.asarray(normal, dtype=float)
    return WallSegment(
        index=index,
        normal=normal,
        offset=float(normal @ start),
        start=start,
        end=end,
        inlier_count=support,
        residual_rms=0.01,
        observed_span=(0.0, float(np.linalg.norm(end - start))),
        height_range=(0.4, 2.0),
    )


def test_gravity_fits_planes_and_reports_observed_ceiling():
    rng = np.random.default_rng(4)
    floor = rng.normal(0.0, 0.01, 500)
    ceiling = rng.normal(2.42, 0.01, 350)
    wall = rng.uniform(0.1, 2.3, 700)
    heights = np.concatenate([floor, ceiling, wall])
    points = np.column_stack((rng.normal(size=len(heights)), rng.normal(size=len(heights)), heights))
    normals = np.concatenate(
        [
            np.tile([0.0, 0.0, 1.0], (len(floor), 1)),
            np.tile([0.0, 0.0, -1.0], (len(ceiling), 1)),
            np.tile([1.0, 0.0, 0.0], (len(wall), 1)),
        ]
    )

    gravity = estimate_gravity(points, np.array([0.0, 0.0, 1.0]), normals)

    assert abs(gravity.floor_height) < 0.03
    assert abs(gravity.ceiling_height - 2.42) < 0.03
    assert gravity.ceiling_observed is True
    assert gravity.ceiling_confidence > 0.5
    assert gravity.ceiling_inlier_count >= 300


def test_gravity_does_not_promote_wall_extent_to_ceiling():
    rng = np.random.default_rng(5)
    floor = rng.normal(0.0, 0.01, 500)
    wall = rng.uniform(0.1, 3.5, 1400)
    heights = np.concatenate([floor, wall])
    points = np.column_stack((rng.normal(size=len(heights)), rng.normal(size=len(heights)), heights))
    normals = np.concatenate(
        [
            np.tile([0.0, 0.0, 1.0], (len(floor), 1)),
            np.tile([1.0, 0.0, 0.0], (len(wall), 1)),
        ]
    )

    gravity = estimate_gravity(points, np.array([0.0, 0.0, 1.0]), normals)

    assert gravity.ceiling_height is None
    assert gravity.ceiling_observed is False
    assert gravity.ceiling_confidence == 0.0
    mask = wall_band_mask(points, normals, gravity, gravity.up)
    assert not mask[heights > 1.9].any()


def test_off_axis_wall_is_quarantined_from_manhattan_geometry():
    frame = HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )
    wall = _wall(0, [0.0, 0.0], [2.0, 1.0], [1.0, -2.0])

    snap_to_frame([wall], frame)

    assert wall.quarantined is True
    assert "off-axis" in wall.tags
    np.testing.assert_allclose(wall.normal, [1.0 / np.sqrt(5.0), -2.0 / np.sqrt(5.0)])
    assert polygonize_wall_graph([wall], min_area=0.1) == []


def test_duplicate_wall_suppression_is_independent_of_input_order():
    first = _wall(0, [0.0, 0.0], [4.0, 0.0], [0.0, 1.0], 100)
    duplicate = _wall(1, [0.0, 0.015], [4.0, 0.015], [0.0, -1.0], 90)

    a = merge_collinear([first, duplicate])
    b = merge_collinear([duplicate, first])

    assert len(a) == len(b) == 1
    np.testing.assert_allclose(a[0].start, b[0].start)
    np.testing.assert_allclose(a[0].end, b[0].end)
    assert a[0].inlier_count == b[0].inlier_count == 190


def test_label_growth_never_enters_unknown_cells():
    labels = np.zeros((5, 5), dtype=int)
    labels[2, 0] = 1
    barrier = np.zeros_like(labels, dtype=bool)
    allowed = np.zeros_like(labels, dtype=bool)
    allowed[2, 1] = True
    allowed[2, 2] = True

    grown = _grow_to_walls(labels, barrier, max_steps=10, allowed=allowed)

    assert grown[2, 1] == 1
    assert grown[2, 2] == 1
    assert grown[2, 3] == 0
    assert _grow_to_walls(labels, barrier, max_steps=10)[2, 4] == 0


def test_rooms_are_wall_graph_faces_not_raster_flood_regions():
    segments = [
        ([0, 0], [4, 0], [0, 1]),
        ([4, 0], [4, 3], [1, 0]),
        ([4, 3], [0, 3], [0, 1]),
        ([0, 3], [0, 0], [1, 0]),
        ([2, 0], [2, 3], [1, 0]),
    ]
    walls = [_wall(i, a, b, n) for i, (a, b, n) in enumerate(segments)]
    assert len(polygonize_wall_graph(walls)) == 2

    resolution = 0.04
    origin = np.array([-1.0, -1.0])
    shape = (140, 110)
    free = np.zeros(shape, dtype=np.int32)
    for x in range(shape[0]):
        for y in range(shape[1]):
            point = origin + (np.array([x, y], dtype=float) + 0.5) * resolution
            if 0.0 < point[0] < 4.0 and 0.0 < point[1] < 3.0:
                free[x, y] = 3
    grid = PlanGrid(resolution, origin, np.zeros(shape, dtype=np.int32), free)
    frame = HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )

    rooms = segment_rooms(grid, walls, frame, 0.0, 2.4)

    assert [round(room.area, 3) for room in rooms] == [6.0, 6.0]
    assert rooms[0].neighbours == [1]
    assert rooms[1].neighbours == [0]
    assert all(4 in room.wall_indices for room in rooms)
