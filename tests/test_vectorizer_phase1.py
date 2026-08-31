from __future__ import annotations

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.occupancy import Opening, deduplicate_openings
from cozmo_ai_v2.pipeline.planes import (
    FINISHED_FACE,
    HorizontalFrame,
    WallSegment,
    snap_to_frame,
    vertical_plane_to_line,
)
from cozmo_ai_v2.pipeline.projection import project_wall_density
from cozmo_ai_v2.pipeline.rooms import (
    PlanGrid,
    _polygon_components,
    polygonize_wall_graph,
    segment_rooms,
)
from cozmo_ai_v2.pipeline.vectorizer import (
    FaceEvidence,
    build_vectorizer_input,
    build_vectorizer_output,
)
from cozmo_ai_v2.pipeline.wall_graph import solve_wall_graph


def _frame() -> HorizontalFrame:
    return HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )


def _wall(
    index: int,
    normal: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    confidence: float = 0.9,
    support: int = 500,
) -> WallSegment:
    normal_array = np.asarray(normal, dtype=float)
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    return WallSegment(
        index=index,
        normal=normal_array,
        offset=float(normal_array @ start_array),
        start=start_array,
        end=end_array,
        inlier_count=support,
        residual_rms=0.01,
        observed_span=(0.0, float(np.linalg.norm(end_array - start_array))),
        height_range=(0.4, 1.9),
        confidence=confidence,
    )


def test_vertical_plane_projection_is_gravity_aligned_and_face_consistent() -> None:
    projected = vertical_plane_to_line([-1.0, 0.0, 0.0], -2.0, _frame())

    np.testing.assert_allclose(projected.normal, [1.0, 0.0])
    assert projected.offset == 2.0
    assert projected.tilt_degrees == 0.0
    assert projected.coordinate_convention == FINISHED_FACE

    with pytest.raises(ValueError, match="not vertical enough"):
        vertical_plane_to_line([0.0, 0.0, 1.0], 1.0, _frame())


def test_low_confidence_wall_is_retained_but_not_forced_onto_axis() -> None:
    angle = np.radians(5.0)
    normal = np.array([np.cos(angle), np.sin(angle)])
    wall = _wall(0, tuple(normal), (0.0, 0.0), (3.0, 1.0), confidence=0.1)
    before = wall.normal.copy()

    snap_to_frame([wall], _frame())

    np.testing.assert_allclose(wall.normal, before)
    assert wall.quarantined
    assert "low-confidence" in wall.tags
    assert wall.snap_status == "rejected-low-confidence"


def test_global_graph_solves_shared_corner_from_both_plane_lines() -> None:
    walls = [
        _wall(0, (0.0, 1.0), (0.0, -0.02), (4.04, -0.02)),
        _wall(1, (1.0, 0.0), (4.03, 0.04), (4.03, 3.0)),
    ]
    graph = solve_wall_graph(walls, node_tolerance=0.10)

    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.kind == "corner"
    np.testing.assert_allclose(node.coordinate, [4.03, -0.02], atol=1e-8)
    solved = {wall.index: wall for wall in graph.walls}
    np.testing.assert_allclose(solved[0].end, node.coordinate)
    np.testing.assert_allclose(solved[1].start, node.coordinate)
    assert graph.snapped_endpoint_count == 2


def test_global_graph_retains_t_and_x_junction_evidence() -> None:
    t_graph = solve_wall_graph(
        [
            _wall(0, (0.0, 1.0), (0.0, 0.0), (4.0, 0.0)),
            _wall(1, (1.0, 0.0), (2.04, -0.04), (2.04, 2.0)),
        ],
        node_tolerance=0.10,
    )
    assert [node.kind for node in t_graph.nodes] == ["t"]
    assert t_graph.nodes[0].incident_walls == (0, 1)

    x_graph = solve_wall_graph(
        [
            _wall(0, (0.0, 1.0), (-2.0, 0.0), (2.0, 0.0)),
            _wall(1, (1.0, 0.0), (0.0, -2.0), (0.0, 2.0)),
        ],
        node_tolerance=0.10,
    )
    assert [node.kind for node in x_graph.nodes] == ["x"]
    assert x_graph.nodes[0].incident_walls == (0, 1)


def test_unintended_crossing_quarantines_weaker_wall_when_x_disabled() -> None:
    graph = solve_wall_graph(
        [
            _wall(0, (0.0, 1.0), (-2.0, 0.0), (2.0, 0.0), support=500),
            _wall(1, (1.0, 0.0), (0.0, -2.0), (0.0, 2.0), support=50),
        ],
        node_tolerance=0.10,
        allow_x_junctions=False,
    )

    assert [wall.index for wall in graph.walls] == [0]
    weak = next(wall for wall in graph.candidates if wall.index == 1)
    assert weak.quarantined
    assert "unintended-crossing" in weak.tags
    assert graph.rejected_crossings == ((0, 1),)


def test_mixed_wall_coordinate_conventions_are_rejected() -> None:
    finished = _wall(0, (1.0, 0.0), (1.0, 0.0), (1.0, 2.0))
    centerline = _wall(1, (0.0, 1.0), (0.0, 1.0), (2.0, 1.0))
    centerline.coordinate_convention = "centerline"

    with pytest.raises(ValueError, match="cannot mix"):
        solve_wall_graph([finished, centerline])


def test_face_validation_rejects_unknown_or_undercovered_graph_faces() -> None:
    walls = [
        _wall(0, (0.0, 1.0), (0.0, 0.0), (4.0, 0.0)),
        _wall(1, (1.0, 0.0), (4.0, 0.0), (4.0, 3.0)),
        _wall(2, (0.0, 1.0), (4.0, 3.0), (0.0, 3.0)),
        _wall(3, (1.0, 0.0), (0.0, 3.0), (0.0, 0.0)),
    ]
    origin = np.array([-1.0, -1.0])
    free = np.zeros((6, 5), dtype=np.int32)
    free[2, 2] = 3
    grid = PlanGrid(1.0, origin, np.zeros_like(free), free)

    assert segment_rooms(
        grid,
        walls,
        _frame(),
        floor_height=0.0,
        ceiling_height=2.4,
        min_area=1.0,
        min_observed_coverage=0.2,
        min_visibility=0.2,
    ) == []
    assert segment_rooms(
        PlanGrid(1.0, origin, np.zeros_like(free), np.zeros_like(free)),
        walls,
        _frame(),
        floor_height=0.0,
        ceiling_height=2.4,
        min_area=1.0,
    ) == []


def test_vectorizer_contract_exposes_map_and_evidence_metadata() -> None:
    density = project_wall_density(
        np.array([[0.25, 0.25, 1.0]]),
        _frame(),
        0.0,
        bounds=((0.0, 0.0), (1.0, 1.0)),
    )
    wall = _wall(0, (1.0, 0.0), (0.0, 0.0), (0.0, 2.0))
    vector_input = build_vectorizer_input(density, [wall])
    graph = solve_wall_graph([wall])
    vector_output = build_vectorizer_output(
        vector_input,
        graph=graph,
        faces=[FaceEvidence(np.array([[0, 0], [1, 0], [1, 1]]), 0.5, 1, 1, 0.8)],
    )
    metadata = vector_output.to_metadata()

    assert vector_input.wall_support is not None
    assert vector_input.observability[6, 6] is True or vector_input.observability.any()
    assert metadata["density"]["count_semantics"]
    assert metadata["density"]["shape"] == [25, 25]
    assert metadata["candidate_segments"][0]["coordinate_convention"] == FINISHED_FACE
    assert metadata["faces"][0]["observed_coverage"] == 1.0
    assert metadata["wall_graph"]["accepted_connections"] == []


def test_opening_duplicate_suppression_preserves_separate_gaps() -> None:
    openings = [
        Opening(0, "door", (1.0, 2.0), (0.0, 2.0), 0.7, evidence_cells=4),
        Opening(0, "door", (1.05, 2.05), (0.0, 2.0), 0.9, evidence_cells=5),
        Opening(0, "door", (3.0, 4.0), (0.0, 2.0), 0.8, evidence_cells=3),
    ]
    deduplicated = deduplicate_openings(openings)

    assert len(deduplicated) == 2
    assert deduplicated[0].u_range == (1.0, 2.05)
    assert deduplicated[0].confidence == 0.9
    assert deduplicated[0].evidence_cells == 9


def test_recordings_2_shaped_gaps_close_three_faces_without_blanket_snap() -> None:
    # Three adjacent 4 m x 3 m rooms have 0.25 m missing at every endpoint.
    # A bounded line-intersection extension supplies only those observed wall
    # continuations; it does not ask polygonization to snap all endpoints.
    walls = [
        _wall(0, (0.0, 1.0), (0.25, 0.0), (11.75, 0.0)),
        _wall(1, (0.0, 1.0), (0.25, 3.0), (11.75, 3.0)),
        _wall(2, (1.0, 0.0), (0.0, 0.25), (0.0, 2.75)),
        _wall(3, (1.0, 0.0), (4.0, 0.25), (4.0, 2.75)),
        _wall(4, (1.0, 0.0), (8.0, 0.25), (8.0, 2.75)),
        _wall(5, (1.0, 0.0), (12.0, 0.25), (12.0, 2.75)),
    ]

    graph = solve_wall_graph(walls, max_endpoint_extension=0.55)
    faces = polygonize_wall_graph(list(graph.walls), min_area=1.0)

    assert len(faces) == 3
    assert graph.diagnostics.before_endpoint_components == 12
    assert graph.diagnostics.after_endpoint_components == 8
    assert graph.diagnostics.accepted_endpoint_extensions == 12
    assert graph.diagnostics.max_endpoint_extension_m == 0.55
    assert all(
        connection.movement_m <= 0.55
        for connection in graph.diagnostics.accepted_connections
    )
    assert any(
        connection.type == "extension"
        for connection in graph.diagnostics.accepted_connections
    )
    metadata = graph.diagnostics.to_solver_metadata()
    assert set(metadata) == {
        "proposed_connections",
        "accepted_connections",
        "rejected_connections",
        "optimization",
    }
    assert metadata["accepted_connections"][0]["endpoint_ids"][0].startswith(
        "wall_"
    )
    assert metadata["accepted_connections"][0]["before_coordinates"] != metadata[
        "accepted_connections"
    ][0]["after_coordinates"]


def test_collinear_door_gap_is_not_fabricated_by_endpoint_closure() -> None:
    walls = [
        _wall(0, (0.0, 1.0), (0.0, 0.0), (2.0, 0.0)),
        _wall(1, (0.0, 1.0), (3.0, 0.0), (5.0, 0.0)),
        _wall(2, (0.0, 1.0), (0.0, 3.0), (5.0, 3.0)),
        _wall(3, (1.0, 0.0), (0.0, 0.0), (0.0, 3.0)),
        _wall(4, (1.0, 0.0), (5.0, 0.0), (5.0, 3.0)),
    ]

    graph = solve_wall_graph(walls, max_endpoint_extension=0.55)
    paired = [
        connection
        for connection in graph.diagnostics.proposed_connections
        if connection.wall_ids == (0, 1)
    ]

    assert paired == []
    assert not any(
        np.isclose(wall.start[1], 0.0) and np.isclose(wall.end[1], 0.0)
        and wall.start[0] < 2.1 and wall.end[0] > 2.9
        for wall in graph.walls
    )


def test_large_node_tolerance_does_not_enable_blanket_extension() -> None:
    walls = [
        _wall(0, (0.0, 1.0), (0.7, 0.0), (4.0, 0.0)),
        _wall(1, (1.0, 0.0), (0.0, 0.7), (0.0, 3.0)),
    ]

    graph = solve_wall_graph(walls, node_tolerance=0.8)

    assert graph.snapped_endpoint_count == 0
    assert graph.diagnostics.accepted_connections == ()
    rejected = graph.diagnostics.rejected_connections
    assert len(rejected) == 1
    assert rejected[0].reason == "outside bounded endpoint extension"
    np.testing.assert_allclose(graph.walls[0].start, [0.7, 0.0])
    np.testing.assert_allclose(graph.walls[1].start, [0.0, 0.7])


def test_fallback_accepts_multiple_observed_components_and_marks_low_confidence() -> None:
    occupied = np.zeros((10, 10), dtype=np.int32)
    free = np.zeros_like(occupied)
    free[1:3, 1:3] = 3
    free[6:9, 6:9] = 3
    grid = PlanGrid(1.0, np.zeros(2), occupied, free)

    rooms = segment_rooms(
        grid,
        [],
        _frame(),
        floor_height=0.0,
        ceiling_height=2.4,
        min_area=1.4,
    )

    assert len(rooms) == 2
    assert all(room.provenance.startswith("fallback") for room in rooms)
    assert all(room.confidence <= 0.35 for room in rooms)


def test_composite_fallback_geometry_is_componentized_without_unknown_bridges() -> None:
    from shapely.geometry import GeometryCollection, MultiPolygon, box

    composite = GeometryCollection(
        [MultiPolygon([box(0, 0, 2, 2), box(4, 0, 6, 2)]), box(8, 0, 10, 2)]
    )
    components = _polygon_components(composite)

    assert len(components) == 3
    assert sorted(round(component.area, 6) for component in components) == [4.0] * 3


def test_geometry_diagnostics_use_canonical_shape_without_solver_duplication() -> None:
    from cozmo_ai_v2.pipeline.cli import _build_geometry_diagnostics

    wall = _wall(4, (1.0, 0.0), (0.0, 0.0), (0.0, 2.0))
    graph = solve_wall_graph([wall])
    grid = PlanGrid(
        0.5,
        np.zeros(2),
        np.zeros((4, 4), dtype=np.int32),
        np.zeros((4, 4), dtype=np.int32),
    )
    diagnostics = _build_geometry_diagnostics(
        wall_stage_counts={
            "raw": 1,
            "merged": 1,
            "occlusion": 1,
            "crossing": 1,
        },
        wall_drop_counts={},
        graph=graph,
        walls=list(graph.walls),
        grid=grid,
        frame=_frame(),
        rooms=[],
    )

    assert diagnostics["diagnostics_version"] == 1
    assert set(diagnostics["wall_stages"]["stage_counts"]) >= {
        "raw",
        "merged",
        "occlusion",
        "crossing",
        "quarantine",
        "post_refinement_internal",
        "exported",
    }
    assert diagnostics["wall_stages"]["stage_counts"]["exported"] == 1
    assert (
        diagnostics["wall_stages"]["stage_counts"]["final"]
        == diagnostics["wall_stages"]["stage_counts"]["post_refinement_internal"]
    )
    endpoint_gaps = diagnostics["endpoint_gaps"]
    assert isinstance(endpoint_gaps["endpoint_count"], int)
    assert set(endpoint_gaps["gap_quantiles_m"]) == {
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    }
    assert "proposed_connections" not in endpoint_gaps
    assert set(diagnostics["grid"]) >= {
        "resolution_m",
        "origin",
        "bounds_plan",
        "shape",
        "transforms",
    }
    assert diagnostics["room_segmentation"]["zero_room_reason"]
    assert diagnostics["zero_room_reasons"]
    assert diagnostics["wall_records"][0]["endpoint_ids"] == [
        "wall_4:start",
        "wall_4:end",
    ]

    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "result.schema.json").read_text()
    )
    payload = {
        "capture": {"name": "synthetic", "modality": "lidar", "frame_count": 0},
        "reconstruction": {
            "up_axis": [0.0, 0.0, 1.0],
            "floor_height": 0.0,
            "ceiling_observed": False,
            "ceiling_confidence": 0.0,
            "walls": [],
        },
        "rooms": [],
        "damage": [],
        "scope": {"line_items": []},
        "diagnostics": {"geometry": diagnostics},
    }
    assert list(Draft202012Validator(schema).iter_errors(payload)) == []
