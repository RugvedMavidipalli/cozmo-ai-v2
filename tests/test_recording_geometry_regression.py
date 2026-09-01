from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from shapely.geometry import Polygon


GOLDEN_PATH = Path(__file__).parent / "data" / "geometry_goldens.yaml"


def _load_goldens() -> dict:
    override = os.environ.get("COZMO_GEOMETRY_GOLDEN")
    path = Path(override) if override else GOLDEN_PATH
    if not path.exists():
        pytest.skip(f"geometry golden file is absent: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _result_path(recording: str) -> Path:
    root = os.environ.get("COZMO_RECORDING_RESULTS")
    if not root:
        pytest.skip("set COZMO_RECORDING_RESULTS to enable recording regression checks")
    path = Path(root) / recording / "result.json"
    if not path.exists():
        pytest.skip(f"recording result is absent: {path}")
    return path


def _area_value(room: dict) -> float:
    area = room.get("area", 0.0)
    if isinstance(area, dict):
        area = area.get("value", 0.0)
    return float(area)


def _max_overlap_fraction(rooms: list[dict]) -> float:
    polygons = []
    for room in rooms:
        ring = room.get("polygon") or []
        if len(ring) < 3:
            continue
        polygon = Polygon(ring)
        if polygon.is_valid and polygon.area > 0:
            polygons.append((room.get("id"), polygon))
    maximum = 0.0
    for index, (_, first) in enumerate(polygons):
        for _, second in polygons[index + 1 :]:
            smaller = min(first.area, second.area)
            if smaller:
                maximum = max(maximum, first.intersection(second).area / smaller)
    return maximum


def _assert_recording_golden(result: dict, annotation: dict) -> None:
    rooms = result.get("rooms", [])
    expected = annotation.get("expected_room_count")
    if expected is not None:
        tolerance = int(annotation.get("room_count_tolerance", 0))
        assert abs(len(rooms) - int(expected)) <= tolerance, (
            f"room count {len(rooms)} is outside annotated range "
            f"[{int(expected) - tolerance}, {int(expected) + tolerance}]"
        )
    if annotation.get("require_room_polygons"):
        assert all(len(room.get("polygon") or []) >= 3 for room in rooms)

    areas = [_area_value(room) for room in rooms]
    total_area = sum(areas)
    if "min_total_area_m2" in annotation:
        assert total_area >= float(annotation["min_total_area_m2"])
    if "max_total_area_m2" in annotation:
        assert total_area <= float(annotation["max_total_area_m2"])

    if "min_wall_assignment_fraction" in annotation:
        walls = result.get("reconstruction", {}).get("walls", [])
        assigned = sum(wall.get("room_id") is not None for wall in walls)
        fraction = assigned / len(walls) if walls else 0.0
        assert fraction >= float(annotation["min_wall_assignment_fraction"])
    if "max_overlap_fraction" in annotation:
        assert _max_overlap_fraction(rooms) <= float(annotation["max_overlap_fraction"])


@pytest.mark.recording_regression
@pytest.mark.parametrize("recording", ["recordings-2", "recordings-1"])
def test_optional_recording_geometry_golden(recording: str):
    """Check externally supplied capture outputs without checking in media.

    This test is opt-in because running a full reconstruction is expensive.
    A missing capture/result skips cleanly; supplying an annotation and a
    result makes topology/area/overlap/wall-assignment regressions actionable.
    """
    if os.environ.get("COZMO_RECORDING_REGRESSION") != "1":
        pytest.skip("set COZMO_RECORDING_REGRESSION=1 to enable recording checks")
    goldens = _load_goldens()
    annotation = goldens.get(recording) or {}
    if not any(
        key in annotation
        for key in (
            "expected_room_count",
            "min_total_area_m2",
            "max_total_area_m2",
            "min_wall_assignment_fraction",
            "max_overlap_fraction",
        )
    ):
        pytest.skip(f"no numeric annotation supplied for {recording}")
    result = json.loads(_result_path(recording).read_text())
    _assert_recording_golden(result, annotation)

