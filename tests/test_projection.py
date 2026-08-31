from __future__ import annotations

import numpy as np

from cozmo_ai_v2.pipeline.planes import HorizontalFrame
from cozmo_ai_v2.pipeline.projection import (
    DensityMap,
    project_wall_density,
    rasterize_points,
)
from cozmo_ai_v2.pipeline.rooms import build_plan_grid


def _frame() -> HorizontalFrame:
    return HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )


def _points(*xyz: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(xyz, dtype=float).reshape(-1, 3)


def test_wall_band_crop_and_count_semantics() -> None:
    result = project_wall_density(
        _points(
            (0.1, 0.1, 0.34),  # below the configured wall band
            (0.1, 0.1, 0.35),  # strict lower edge
            (0.1, 0.1, 1.0),
            (0.1, 0.1, 1.9),  # strict upper edge
            (0.1, 0.1, 2.0),  # above the configured wall band
        ),
        _frame(),
        floor_height=0.0,
        resolution=0.5,
        wall_band=(0.35, 1.9),
        bounds=((0.0, 0.0), (1.0, 1.0)),
    )

    assert isinstance(result, DensityMap)
    assert result.wall_band == (0.35, 1.9)
    assert result.height_bounds == (0.35, 1.9)
    assert result.band_count == 1
    assert result.retained_count == 1
    assert int(result.counts.sum()) == result.retained_count
    assert result.counts[0, 0] == 1
    assert result.density[0, 0] == 4.0
    assert result.observed[0, 0]
    assert not result.empty[0, 0]


def test_known_cells_accumulate_in_plan_frame_orientation() -> None:
    result = project_wall_density(
        _points(
            (-0.75, -0.75, 1.0),
            (-0.70, -0.70, 1.1),
            (0.25, -0.25, 1.2),
            (0.25, 0.25, 1.3),
        ),
        _frame(),
        floor_height=0.0,
        resolution=0.5,
        bounds=((-1.0, -1.0), (1.0, 1.0)),
    )

    # The first two coordinates share (-1, -1)'s cell.  x is the first
    # index and y the second; this is plan orientation, not image row order.
    assert result.shape == (4, 4)
    assert result.origin.tolist() == [-1.0, -1.0]
    assert result.counts[0, 0] == 2
    assert result.counts[2, 1] == 1
    assert result.counts[2, 2] == 1
    assert result.counts[1, 2] == 0
    assert result.retained_count == 4
    assert result.empty.sum() == 13


def test_exact_boundaries_are_lower_inclusive_and_upper_exclusive() -> None:
    result = project_wall_density(
        _points(
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 1.0),
            (0.5, 0.5, 1.0),
            (1.0, 0.0, 1.0),  # explicit max x: outside
            (0.0, 1.0, 1.0),  # explicit max y: outside
        ),
        _frame(),
        floor_height=0.0,
        resolution=0.5,
        bounds=((0.0, 0.0), (1.0, 1.0)),
    )

    np.testing.assert_array_equal(result.counts, [[1, 0], [1, 1]])
    assert result.band_count == 5
    assert result.retained_count == 3
    assert result.out_of_bounds_count == 2


def test_negative_coordinates_and_outside_points_are_deterministic() -> None:
    cloud = _points(
        (-1.25, -0.25, 1.0),
        (-0.25, -1.25, 1.0),
        (-1.25, -0.25, 1.0),
        (2.0, 0.0, 1.0),
    )
    kwargs = dict(
        frame=_frame(),
        floor_height=0.0,
        resolution=1.0,
        bounds=((-2.0, -2.0), (0.0, 0.0)),
    )
    first = project_wall_density(cloud, **kwargs)
    second = project_wall_density(cloud[::-1], **kwargs)

    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(first.origin, [-2.0, -2.0])
    assert first.counts[0, 1] == 2
    assert first.counts[1, 0] == 1
    assert first.retained_count == 3
    assert first.out_of_bounds_count == 1


def test_nan_inf_and_empty_clouds_are_safe() -> None:
    result = project_wall_density(
        _points(
            (np.nan, 0.0, 1.0),
            (0.0, np.inf, 1.0),
            (0.0, 0.0, 1.0),
        ),
        _frame(),
        floor_height=0.0,
    )
    assert result.input_count == 3
    assert result.finite_input_count == 1
    assert result.invalid_input_count == 2
    assert result.retained_count == 1
    assert np.isfinite(result.origin).all()

    empty = project_wall_density(np.empty((0, 3)), _frame(), 0.0)
    assert empty.shape == (1, 1)
    assert empty.origin.tolist() == [0.0, 0.0]
    assert empty.bounds == ((0.0, 0.0), (0.04, 0.04))
    assert empty.retained_count == 0
    assert not empty.observed.any()
    assert empty.empty.all()


def test_low_level_rasterizer_drops_nonfinite_and_outside_points() -> None:
    counts = rasterize_points(
        np.array(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [np.inf, 0.0],
            ]
        ),
        origin=np.array([-1.0, -1.0]),
        shape=(2, 2),
        resolution=1.0,
    )
    np.testing.assert_array_equal(counts, [[1, 0], [0, 1]])


def test_plan_grid_carries_projection_at_vectorizer_boundary() -> None:
    grid = build_plan_grid(
        _points((0.1, 0.1, 1.0), (0.1, 0.1, 0.0)),
        _frame(),
        floor_height=0.0,
        ceiling_height=None,
        resolution=0.5,
    )

    assert grid.wall_density is not None
    assert grid.density is grid.wall_density
    np.testing.assert_array_equal(grid.occupied, grid.wall_density.counts)
    assert grid.wall_density.retained_count == int(grid.occupied.sum())

