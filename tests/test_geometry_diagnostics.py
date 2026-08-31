from __future__ import annotations

import numpy as np

from cozmo_ai_v2.pipeline.geometry_diagnostics import GeometryDiagnostics
from cozmo_ai_v2.pipeline.planes import (
    HorizontalFrame,
    WallSegment,
    merge_collinear,
    snap_to_frame,
)
from cozmo_ai_v2.pipeline.rooms import PlanGrid, segment_rooms


def _frame() -> HorizontalFrame:
    return HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )


def _wall(index: int, start, end, normal, support: int = 100) -> WallSegment:
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


def _square_walls() -> list[WallSegment]:
    return [
        _wall(0, [0, 0], [4, 0], [0, 1]),
        _wall(1, [4, 0], [4, 3], [1, 0]),
        _wall(2, [4, 3], [0, 3], [0, 1]),
        _wall(3, [0, 3], [0, 0], [1, 0]),
    ]


def _floor_grid() -> PlanGrid:
    resolution = 0.04
    origin = np.array([-1.0, -1.0])
    shape = (140, 110)
    free = np.zeros(shape, dtype=np.int32)
    for x in range(shape[0]):
        for y in range(shape[1]):
            point = origin + (np.array([x, y], dtype=float) + 0.5) * resolution
            if 0.0 < point[0] < 4.0 and 0.0 < point[1] < 3.0:
                free[x, y] = 3
    return PlanGrid(resolution, origin, np.zeros(shape, dtype=np.int32), free)


def test_geometry_diagnostics_has_stable_contract_and_endpoint_summary():
    diagnostics = GeometryDiagnostics()
    walls = _square_walls()
    diagnostics.set_wall_stage("raw", walls)
    diagnostics.record_endpoint_gaps(walls)

    payload = diagnostics.to_dict()

    assert payload["diagnostics_version"] == 1
    assert set(payload) == {
        "diagnostics_version",
        "wall_stages",
        "wall_records",
        "endpoint_gaps",
        "polygonization",
        "grid",
        "room_segmentation",
        "zero_room_reasons",
    }
    assert payload["wall_stages"]["stage_counts"]["raw"] == 4
    assert payload["endpoint_gaps"]["endpoint_count"] == 8
    assert payload["endpoint_gaps"]["component_count"] == 4
    assert payload["endpoint_gaps"]["gap_quantiles_m"]["p50"] == 0.0
    # Graph proposal/connection records belong to the separate wall solver
    # workstream and are intentionally absent from this general summary.
    assert "proposed_connections" not in payload["endpoint_gaps"]


def test_segment_rooms_persists_grid_polygon_and_room_diagnostics():
    diagnostics = GeometryDiagnostics()
    rooms = segment_rooms(
        _floor_grid(),
        _square_walls(),
        _frame(),
        floor_height=0.0,
        ceiling_height=2.4,
        diagnostics=diagnostics,
    )

    geometry = diagnostics.to_dict()
    assert len(rooms) == 1
    assert geometry["polygonization"]["candidate_face_count"] == 1
    assert geometry["polygonization"]["accepted_face_count"] == 1
    assert geometry["polygonization"]["faces"][0]["accepted"] is True
    assert geometry["room_segmentation"] == {
        "method": "wall_graph_polygonize",
        "fallback_used": False,
        "fallback_geometry_types": {},
        "room_count": 1,
        "zero_room_reason": None,
    }
    grid = geometry["grid"]
    assert grid["resolution_m"] == 0.04
    assert grid["transforms"]["world_to_plan"] == {
        "right": [1.0, 0.0, 0.0],
        "forward": [0.0, 1.0, 0.0],
    }
    assert grid["occupied_cells"] == 0
    assert grid["free_cells"] > 0
    assert grid["unknown_cells"] > 0
    assert geometry["zero_room_reasons"] == []


def test_zero_room_diagnostics_explain_missing_observation_and_faces():
    diagnostics = GeometryDiagnostics()
    empty = PlanGrid(
        resolution=0.04,
        origin=np.zeros(2),
        occupied=np.zeros((4, 4), dtype=np.int32),
        free=np.zeros((4, 4), dtype=np.int32),
    )

    assert segment_rooms(empty, [], _frame(), 0.0, None, diagnostics=diagnostics) == []

    room_segmentation = diagnostics.to_dict()["room_segmentation"]
    assert room_segmentation["room_count"] == 0
    assert room_segmentation["fallback_used"] is True
    assert "no-bounded-wall-faces" in room_segmentation["zero_room_reason"]
    assert "no-observed-free-cells" in room_segmentation["zero_room_reason"]
    assert diagnostics.to_dict()["zero_room_reasons"]


def test_wall_lifecycle_records_quarantine_drop_and_provenance():
    diagnostics = GeometryDiagnostics()
    duplicate = _wall(1, [0.0, 0.015], [4.0, 0.015], [0.0, -1.0], 90)
    off_axis = _wall(2, [0.0, 2.0], [2.0, 3.0], [1.0, -2.0])
    walls = snap_to_frame(
        [_wall(0, [0.0, 0.0], [4.0, 0.0], [0.0, 1.0]), duplicate, off_axis],
        _frame(),
        diagnostics=diagnostics,
    )
    merge_collinear(walls, diagnostics=diagnostics)

    payload = diagnostics.to_dict()
    wall_stages = payload["wall_stages"]
    assert wall_stages["quarantines_by_reason"]["off-axis"] == 1
    assert wall_stages["drops_by_reason"]["duplicate-wall"] == 1
    events = [event for event in payload["wall_records"] if event.get("action") in {"drop", "quarantine"}]
    assert events
    assert all(event["provenance"] for event in events)
    assert all("wall_id" in event for event in events)
