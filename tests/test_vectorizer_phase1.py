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
from cozmo_ai_v2.pipeline.rooms import PlanGrid, segment_rooms
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
    vector_output = build_vectorizer_output(
        vector_input,
        accepted_segments=[wall],
        faces=[FaceEvidence(np.array([[0, 0], [1, 0], [1, 1]]), 0.5, 1, 1, 0.8)],
    )
    metadata = vector_output.to_metadata()

    assert vector_input.wall_support is not None
    assert vector_input.observability[6, 6] is True or vector_input.observability.any()
    assert metadata["density"]["count_semantics"]
    assert metadata["density"]["shape"] == [25, 25]
    assert metadata["candidate_segments"][0]["coordinate_convention"] == FINISHED_FACE
    assert metadata["faces"][0]["observed_coverage"] == 1.0


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
