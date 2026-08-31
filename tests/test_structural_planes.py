from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cozmo_ai_v2.pipeline.export import export_plane_metadata
from cozmo_ai_v2.pipeline.planes import (
    HorizontalFrame,
    PlaneClassification,
    extract_structural_planes,
    fit_plane_tls,
)


def _grid_plane(rng, count, z, x_range=(0.0, 4.0), y_range=(0.0, 3.0), noise=0.0):
    x = rng.uniform(*x_range, count)
    y = rng.uniform(*y_range, count)
    if callable(z):
        height = z(x, y)
    else:
        height = np.full(count, z, dtype=float)
    return np.column_stack((x, y, height + rng.normal(0.0, noise, count)))


def test_structural_planes_are_deterministic_and_retain_source_support():
    rng = np.random.default_rng(14)
    floor = _grid_plane(rng, 260, 0.0, noise=0.006)
    wall = np.column_stack(
        (
            rng.normal(4.0, 0.006, 240),
            rng.uniform(0.0, 3.0, 240),
            rng.uniform(0.0, 2.5, 240),
        )
    )
    ceiling = _grid_plane(rng, 220, 2.45, noise=0.006)
    outliers = rng.uniform(-2.0, 7.0, (90, 3))
    points = np.vstack((floor, wall, ceiling, outliers))
    normals = np.vstack(
        (
            np.tile([0.0, 0.0, 1.0], (len(floor), 1)),
            np.tile([1.0, 0.0, 0.0], (len(wall), 1)),
            np.tile([0.0, 0.0, -1.0], (len(ceiling), 1)),
            np.zeros((len(outliers), 3)),
        )
    )

    first = extract_structural_planes(
        points,
        normals,
        up=[0.0, 0.0, 1.0],
        floor_height=0.0,
        min_inliers=40,
        seed=9,
    )
    second = extract_structural_planes(
        points,
        normals,
        up=[0.0, 0.0, 1.0],
        floor_height=0.0,
        min_inliers=40,
        seed=9,
    )

    assert [plane.classification for plane in first] == [
        plane.classification for plane in second
    ]
    assert [plane.inlier_indices.tolist() for plane in first] == [
        plane.inlier_indices.tolist() for plane in second
    ]
    assert {plane.classification for plane in first} >= {
        PlaneClassification.FLOOR.value,
        PlaneClassification.WALL.value,
        PlaneClassification.CEILING.value,
    }
    for plane in first:
        assert plane.support_count == len(plane.inlier_indices)
        assert np.isclose(np.linalg.norm(plane.normal), 1.0)
        assert np.isfinite(plane.offset)
        assert plane.residual_rms < 0.04
        assert plane.point_density > 0.0
        assert plane.confidence > 0.4

    wall_plane = next(
        plane for plane in first if plane.classification == PlaneClassification.WALL.value
    )
    wall_source_count = np.count_nonzero(
        (wall_plane.inlier_indices >= len(floor))
        & (wall_plane.inlier_indices < len(floor) + len(wall))
    )
    assert wall_source_count >= 0.9 * len(wall)
    assert wall_plane.wall_vertical_extent[1] - wall_plane.wall_vertical_extent[0] > 2.0


def test_multiple_horizontal_ceiling_planes_include_a_sloped_surface():
    rng = np.random.default_rng(22)
    floor = _grid_plane(rng, 250, 0.0, noise=0.004)
    flat_ceiling = _grid_plane(rng, 180, 2.35, x_range=(0.0, 2.0), noise=0.005)
    sloped_ceiling = _grid_plane(
        rng,
        190,
        lambda x, _y: 2.65 + 0.12 * x,
        x_range=(2.0, 4.0),
        noise=0.005,
    )
    points = np.vstack((floor, flat_ceiling, sloped_ceiling))
    planes = extract_structural_planes(
        points,
        up=[0.0, 0.0, 1.0],
        floor_height=0.0,
        min_inliers=35,
        seed=2,
    )

    ceilings = [
        plane for plane in planes if plane.classification == PlaneClassification.CEILING.value
    ]
    assert len(ceilings) == 2
    assert all(plane.ceiling_observed for plane in ceilings)
    assert all(plane.ceiling_confidence > 0.4 for plane in ceilings)
    assert sorted(plane.centroid[2] for plane in ceilings)[1] > 2.7
    assert any(abs(plane.normal[2]) < 0.999 for plane in ceilings)


def test_clutter_and_off_orientation_planes_are_quarantined():
    rng = np.random.default_rng(31)
    floor = _grid_plane(rng, 160, 0.0, noise=0.003)
    tabletop = _grid_plane(rng, 140, 0.8, x_range=(0.0, 2.0), noise=0.003)
    # A diagonal roof/fixture plane, intentionally neither horizontal nor
    # vertical relative to the supplied gravity vector.
    diagonal = _grid_plane(
        rng,
        140,
        lambda x, y: 1.2 + 0.8 * (x - 2.5) + 0.15 * y,
        x_range=(2.0, 3.0),
        y_range=(0.0, 2.0),
        noise=0.003,
    )
    planes = extract_structural_planes(
        np.vstack((floor, tabletop, diagonal)),
        up=[0.0, 0.0, 1.0],
        floor_height=0.0,
        min_inliers=30,
        seed=7,
    )

    clutter = [plane for plane in planes if plane.classification == "clutter"]
    assert len(clutter) >= 2
    assert all(plane.quarantined for plane in clutter)
    assert all("off-orientation" in plane.tags for plane in clutter)


def test_tls_and_wall_conversion_are_safe_for_degenerate_and_metric_inputs():
    normal, offset = fit_plane_tls(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    assert np.isfinite(normal).all()
    assert np.isfinite(offset)

    assert extract_structural_planes(np.empty((0, 3)), min_inliers=3) == []
    assert extract_structural_planes(np.zeros((12, 3)), min_inliers=3) == []

    rng = np.random.default_rng(44)
    wall_points = np.column_stack(
        (
            np.full(120, 1.5) + rng.normal(0.0, 0.004, 120),
            rng.uniform(-1.0, 2.0, 120),
            rng.uniform(0.2, 2.4, 120),
        )
    )
    wall = next(
        plane
        for plane in extract_structural_planes(
            wall_points, up=[0.0, 0.0, 1.0], min_inliers=30, seed=3
        )
        if plane.classification == PlaneClassification.WALL.value
    )
    frame = HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )
    line = wall.to_wall_segment(frame, points=wall_points)
    assert line is not None
    assert line.structural_plane_id == wall.id
    assert len(line.inlier_indices) == wall.support_count
    assert line.length > 2.0
    assert line.height_range[1] - line.height_range[0] > 1.5


def test_plane_metadata_export_and_schema_shape(tmp_path):
    rng = np.random.default_rng(55)
    points = np.vstack((_grid_plane(rng, 100, 0.0), _grid_plane(rng, 100, 2.4)))
    planes = extract_structural_planes(points, up=[0, 0, 1], floor_height=0, min_inliers=20)
    metadata_path = export_plane_metadata(planes, tmp_path / "planes.json")
    payload = json.loads(metadata_path.read_text())
    assert len(payload["planes"]) == len(planes)
    assert all("inlier_indices" in document for document in payload["planes"])

    reconstruction = {
        "structural_planes": [plane.to_dict() | {"wall_line": None} for plane in planes],
        "plane_extraction": {
            "algorithm": "seeded_ransac_region_growing_tls_3d",
            "refit": "total_least_squares_svd_perpendicular_residual",
            "plane_count": len(planes),
            "kept_count": sum(not plane.quarantined for plane in planes),
            "quarantined_count": sum(plane.quarantined for plane in planes),
            "floor_plane_ids": [plane.id for plane in planes if plane.classification == "floor"],
            "ceiling_plane_ids": [plane.id for plane in planes if plane.classification == "ceiling"],
            "multiple_ceiling_planes": False,
        },
    }
    # Validate only the newly-added fragment through the same Draft 2020-12
    # validator used by the pipeline; the top-level result has unrelated
    # required stages that this focused test need not fabricate.
    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "result.schema.json").read_text()
    )
    import jsonschema

    full_payload = {
        "capture": {"name": "test", "modality": "lidar", "frame_count": 0},
        "reconstruction": {
            "up_axis": [0.0, 0.0, 1.0],
            "floor_height": 0.0,
            "floor_confidence": 0.8,
            "floor_observed": True,
            "floor_quality_status": "high_confidence",
            "floor_low_confidence": False,
            "floor_support_fraction": 1.0,
            "floor_adaptive_residual_limit_mm": 40.0,
            "floor_inlier_count": 100,
            "floor_residual_rms_mm": 5.0,
            "ceiling_observed": True,
            "ceiling_confidence": 0.8,
            "walls": [],
            **reconstruction,
        },
        "rooms": [],
        "damage": [],
        "scope": {"line_items": []},
        "diagnostics": {},
    }
    jsonschema.Draft202012Validator(schema).validate(full_payload)
