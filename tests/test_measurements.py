from __future__ import annotations

import numpy as np

from cozmo_ai_v2.pipeline.geometry import GravityEstimate
from cozmo_ai_v2.pipeline.measurements import (
    MeasurementContext,
    TLSPlaneModel,
    door_scale_advisory,
    measure_scene,
    validate_reference_scale,
)
from cozmo_ai_v2.pipeline.planes import HorizontalFrame, TLSPlane, WallSegment
from cozmo_ai_v2.pipeline.rooms import Room


FRAME = HorizontalFrame(
    up=np.array([0.0, 0.0, 1.0]),
    right=np.array([1.0, 0.0, 0.0]),
    forward=np.array([0.0, 1.0, 0.0]),
    yaw=0.0,
    manhattan_fraction=1.0,
)


def _wall(index, start, end, normal, *, support=600, residual=0.004):
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
        residual_rms=residual,
        observed_span=(0.0, float(np.linalg.norm(end - start))),
        height_range=(0.2, 2.4),
    )


def _room():
    # The polygon is deliberately wrong: Stage 9 must use the plane graph.
    return Room(
        id=0,
        name="room_1",
        area=999.0,
        centroid=np.array([2.0, 1.5]),
        floor_height=0.0,
        ceiling_height=2.4,
        wall_indices=[0, 1, 2, 3],
        polygon=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]),
    )


def _rectangle_walls():
    # Endpoints run past the corners.  The measured extent must be the
    # intersections at x/y = 0 and 4/3, not the observed endpoint span.
    return [
        _wall(0, [-0.2, 0.0], [4.2, 0.0], [0.0, 1.0]),
        _wall(1, [4.0, -0.2], [4.0, 3.2], [1.0, 0.0]),
        _wall(2, [4.2, 3.0], [-0.2, 3.0], [0.0, 1.0]),
        _wall(3, [0.0, 3.2], [0.0, -0.2], [1.0, 0.0]),
    ]


def test_measurements_use_plane_intersections_and_primary_interior_area():
    gravity = GravityEstimate(
        FRAME.up, 0.0, 2.4, 0.9, True, 0.9, 0.9, 600, 600, 0.004, 0.004
    )
    result = measure_scene(
        _rectangle_walls(),
        [_room()],
        frame=FRAME,
        gravity=gravity,
        context=MeasurementContext(
            pose_provenance="refined", depth_provenance="tls_lidar", calibration_status="calibrated"
        ),
    )

    assert result.walls[0].length.value == 4.0
    assert result.walls[0].geometry_source == "plane_intersections"
    assert result.rooms[0].interior_face_area.value == 12.0
    assert result.rooms[0].interior_face_area.basis.startswith("PRIMARY:")
    assert result.rooms[0].wall_centerline_area.status == "estimated"
    assert "assumed_wall_thickness" in result.rooms[0].wall_centerline_area.flags
    assert result.rooms[0].area_convention["primary"] == "interior_face_area"


def test_thickness_requires_two_observed_opposing_faces():
    walls = _rectangle_walls()
    single_face = measure_scene(walls, [_room()], frame=FRAME)
    assert single_face.walls[0].thickness.value is None
    assert single_face.walls[0].thickness.status == "unmeasured"
    assert "opposing_face_not_observed" in single_face.walls[0].thickness.flags

    paired = walls + [
        _wall(4, [-0.2, 0.2], [4.2, 0.2], [0.0, 1.0]),
        _wall(5, [4.2, -0.2], [4.2, 3.2], [1.0, 0.0]),
        _wall(6, [4.2, 3.2], [-0.2, 3.2], [0.0, 1.0]),
        _wall(7, [0.2, 3.2], [0.2, -0.2], [1.0, 0.0]),
    ]
    measured = measure_scene(paired, [_room()], frame=FRAME)
    assert abs(measured.walls[0].thickness.value - 0.2) < 1e-8
    assert measured.walls[0].thickness.status == "measured"
    assert measured.rooms[0].wall_centerline_area.status == "measured"


def test_height_statistics_measure_a_sloped_ceiling_perpendicular_to_floor():
    gravity = GravityEstimate(FRAME.up, 0.0, None, 0.9, False, 0.0, 0.9, 600, 0, 0.004, None)
    ceiling_normal = np.array([-0.2, 0.0, 1.0])
    ceiling_normal /= np.linalg.norm(ceiling_normal)
    model = TLSPlaneModel(
        planes=[],
        floor_plane=TLSPlane("floor", FRAME.up, 0.0, role="floor", inlier_count=600, residual_rms=0.004),
        ceiling_planes=[
            TLSPlane(
                "ceiling",
                ceiling_normal,
                2.4 / np.linalg.norm(np.array([-0.2, 0.0, 1.0])),
                role="ceiling",
                inlier_count=600,
                residual_rms=0.004,
            )
        ],
    )
    result = measure_scene(
        _rectangle_walls(), [_room()], tls_model=model, frame=FRAME, gravity=gravity
    )
    heights = result.rooms[0].floor_to_ceiling_height
    assert abs(heights.minimum.value - 2.4) < 1e-6
    assert abs(heights.maximum.value - 3.2) < 1e-6
    assert abs(heights.mean.value - 2.8) < 1e-6
    assert "sloped_or_multiple_ceiling" in heights.mean.flags


def test_structured_tls_3d_plane_model_is_accepted_without_phase1_shims():
    starts_ends_normals = [
        ([-0.2, 0.0, 1.0], [4.2, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([4.0, -0.2, 1.0], [4.0, 3.2, 1.0], [1.0, 0.0, 0.0]),
        ([4.2, 3.0, 1.0], [-0.2, 3.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 3.2, 1.0], [0.0, -0.2, 1.0], [1.0, 0.0, 0.0]),
    ]
    planes = [
        TLSPlane(
            f"wall_{index}", normal, float(np.dot(normal, start)),
            start=start, end=end, inlier_count=500, residual_rms=0.005,
            pose_provenance="slam_refined", depth_provenance="tls_lidar",
            calibration_status="validated",
        )
        for index, (start, end, normal) in enumerate(starts_ends_normals)
    ]
    model = TLSPlaneModel(
        planes=planes,
        floor_plane=TLSPlane("floor", [0, 0, 1], 0, role="floor", inlier_count=500, residual_rms=0.005),
        ceiling_planes=[TLSPlane("ceiling", [0, 0, 1], 2.4, role="ceiling", inlier_count=500, residual_rms=0.005)],
    )
    result = measure_scene(
        tls_model=model,
        rooms=[{"id": 0, "centroid": [2, 1.5], "wall_ids": ["wall_0", "wall_1", "wall_2", "wall_3"]}],
        frame=FRAME,
        context=MeasurementContext(pose_provenance="refined", calibration_status="calibrated"),
    )
    assert result.walls["wall_0"].length.value == 4.0
    assert result.rooms[0].interior_face_area.value == 12.0
    assert result.rooms[0].floor_to_ceiling_height.mean.value == 2.4


def test_structured_model_can_supply_plane_intersections_when_extents_are_separate():
    planes = [
        TLSPlane("bottom", [0, 1, 0], 0, inlier_count=500, residual_rms=0.005),
        TLSPlane("right", [1, 0, 0], 4, inlier_count=500, residual_rms=0.005),
        TLSPlane("top", [0, 1, 0], 3, inlier_count=500, residual_rms=0.005),
        TLSPlane("left", [1, 0, 0], 0, inlier_count=500, residual_rms=0.005),
    ]
    corners = {
        ("bottom", "right"): [4, 0],
        ("right", "top"): [4, 3],
        ("top", "left"): [0, 3],
        ("left", "bottom"): [0, 0],
    }
    model = TLSPlaneModel(
        planes=planes,
        intersections=[
            {"planes": list(pair), "point": point} for pair, point in corners.items()
        ],
    )
    result = measure_scene(
        tls_model=model,
        rooms=[{"id": 0, "centroid": [2, 1.5], "wall_ids": ["bottom", "right", "top", "left"]}],
        frame=FRAME,
    )
    assert result.walls["bottom"].length.value == 4.0
    assert result.rooms[0].interior_face_area.value == 12.0


def test_confidence_and_flags_include_tls_and_capture_provenance():
    weak = [_wall(0, [0, 0], [4, 0], [0, 1], support=5, residual=0.12)]
    result = measure_scene(
        weak,
        [],
        frame=FRAME,
        context=MeasurementContext(
            pose_provenance="raw", depth_provenance="estimated_depth", calibration_status="uncalibrated"
        ),
    )
    measurement = result.walls[0].length
    assert measurement.confidence < 0.60
    assert {"high_tls_residual", "weak_support", "pose_uncertain", "depth_uncertain", "uncalibrated", "manual_review"} <= set(measurement.flags)
    assert measurement.to_dict()["evidence"]["tls_residual_rms_m"] == 0.12


def test_known_reference_validation_is_explicit_and_door_is_advisory():
    validation = validate_reference_scale(2.0, 1.0, reference_type="tape")
    assert validation.status == "validated"
    assert validation.scale_factor == 0.5
    assert validation.applied is False
    assert "calibration_not_applied" in validation.flags

    door = door_scale_advisory(0.8)
    assert door.status == "advisory"
    assert door.scale_factor is None
    assert door.applied is False
    assert "never_used_for_calibration" in door.flags

    invalid = validate_reference_scale(None, 1.0, reference_type="marker")
    assert invalid.status == "unmeasured"
    assert invalid.scale_factor is None
