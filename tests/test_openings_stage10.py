from __future__ import annotations

import json

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.openings import NormalizedOpening, fuse_openings
from cozmo_ai_v2.pipeline.planes import HorizontalFrame, WallSegment
from cozmo_ai_v2.pipeline.rgb_openings import (
    GroundingDINOAdapter,
    ModelUnavailable,
    RGBOpeningBox,
    RGBOpeningMask,
    SAM2Adapter,
    _project_detection,
)
from cozmo_ai_v2.pipeline.roomformer import RoomFormerSDTQAdapter


def _wall() -> WallSegment:
    return WallSegment(
        index=3,
        normal=np.array([1.0, 0.0]),
        offset=2.0,
        start=np.array([2.0, -1.0]),
        end=np.array([2.0, 1.0]),
        inlier_count=500,
        residual_rms=0.01,
        observed_span=(0.0, 2.0),
        height_range=(0.0, 2.5),
    )


def test_geometry_and_rgb_observations_fuse_with_provenance():
    geometry = NormalizedOpening(
        3, "door", (0.4, 1.2), (0.0, 2.1), 0.75, provenance=["geometry"], source_frames=[2]
    )
    rgb = NormalizedOpening(
        3, "door", (0.45, 1.18), (0.02, 2.08), 0.80, provenance=["rgb"], source_frames=[8], depth_support=20
    )

    fused = fuse_openings([rgb, geometry])

    assert len(fused) == 1
    assert fused[0].provenance == ["geometry", "rgb", "fused"]
    assert fused[0].observation_count == 2
    assert fused[0].source_frames == [2, 8]
    assert fused[0].state == "measured"


def test_roomformer_adapter_keeps_image_only_hint_unmeasured_and_rejects_furniture():
    adapter = RoomFormerSDTQAdapter(min_confidence=0.25)
    result = adapter.adapt(
        {
            "labels": ["door", "furniture", "window"],
            "boxes": [[100, 200, 300, 900], [400, 400, 700, 900], [0.2, 0.2, 0.4, 0.5]],
            "scores": [0.9, 0.99, 0.8],
        },
        frame_shape=(1000, 1000),
    )

    assert [opening.kind for opening in result] == ["door", "window"]
    assert all(opening.state == "unmeasured" for opening in result)
    assert all(opening.provenance == ["roomformer"] for opening in result)
    assert result[1].image_bbox == (200.0, 200.0, 400.0, 500.0)
    assert adapter.rejections[0]["reason"] == "unknown/furniture label"

    metric = adapter.adapt(
        [{"label": "window", "wall_index": 3, "u_offset": 1.0, "width_m": 0.9,
          "v_offset": 1.1, "height_m": 0.8, "score": 0.9}]
    )[0]
    assert metric.state == "measured"
    assert metric.u_range == (1.0, 1.9)
    assert metric.v_range[0] == 1.1
    assert metric.v_range[1] == pytest.approx(1.9)


def test_model_adapters_are_lazy_and_mockable():
    calls = []
    detector = GroundingDINOAdapter(runner=lambda image, index: calls.append(index) or [
        {"label": "door", "bbox": [1, 2, 5, 8], "score": 0.9}
    ])
    boxes = detector.detect(np.zeros((10, 10, 3), np.uint8), 7)

    assert calls == [7]
    assert boxes[0].kind == "door"

    class Predictor:
        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, box, multimask_output=False):
            return np.ones((1, *self.shape), bool), np.array([0.9]), None

    masks = SAM2Adapter(predictor=Predictor()).refine(
        np.zeros((10, 10, 3), np.uint8), boxes
    )
    assert masks[0].method == "sam2"
    assert masks[0].mask.shape == (10, 10)

    with pytest.raises(ModelUnavailable):
        GroundingDINOAdapter().detect(np.zeros((4, 4, 3), np.uint8))


def test_rgb_projection_requires_depth_and_associates_to_wall_plane():
    image_shape = (100, 100)
    depth = np.zeros(image_shape, dtype=float)
    mask = np.zeros(image_shape, dtype=bool)
    intrinsics = np.array([[15.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    # Points are behind x=2, with y spanning the wall segment and world z
    # spanning a door. Rasterising their calibrated projections simulates a
    # refined SAM2 mask plus valid depth at those pixels.
    for y in np.linspace(-0.55, 0.55, 6):
        for z in np.linspace(0.7, 2.3, 4):
            u = int(round(50 + 15 * 3.0 / z))
            v = int(round(50 + 50 * y / z))
            if 0 <= u < 100 and 0 <= v < 100:
                depth[v, u] = z
                mask[v, u] = True

    frame = HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )
    opening, reason = _project_detection(
        RGBOpeningBox(4, "door", (55, 25, 100, 80), 0.9),
        RGBOpeningMask(mask),
        depth,
        np.eye(4),
        intrinsics,
        frame,
        [_wall()],
        {},
        0.0,
        3.0,
        4,
        0.02,
        0.65,
    )

    assert reason == ""
    assert opening is not None
    assert opening.wall_index == 3
    assert opening.state == "measured"
    assert opening.depth_support >= 4
    assert opening.wall_association_confidence > 0


def test_rgb_projection_rejects_front_of_wall_occluder():
    depth = np.zeros((60, 60), dtype=float)
    mask = np.zeros((60, 60), dtype=bool)
    intrinsics = np.array([[12.0, 0.0, 30.0], [0.0, 40.0, 30.0], [0.0, 0.0, 1.0]])
    for y in np.linspace(-0.4, 0.4, 5):
        for z in np.linspace(0.8, 2.2, 5):
            u = int(round(30 + 12 * 1.5 / z))
            v = int(round(30 + 40 * y / z))
            if 0 <= u < 60 and 0 <= v < 60:
                depth[v, u] = z
                mask[v, u] = True
    frame = HorizontalFrame(
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), 0.0, 1.0,
    )

    opening, reason = _project_detection(
        RGBOpeningBox(0, "window", (30, 10, 60, 55), 0.9),
        RGBOpeningMask(mask), depth, np.eye(4), intrinsics, frame, [_wall()], {},
        0.0, 3.0, 4, 0.02, 0.65,
    )

    assert opening is None
    assert reason == "occluded"


def test_unmeasured_contract_serializes_without_fabricating_size(tmp_path):
    opening = NormalizedOpening(None, "window", None, None, 0.6, provenance=["roomformer"], state="unmeasured")
    payload = opening.to_dict()
    path = tmp_path / "opening.json"
    path.write_text(json.dumps(payload))

    decoded = json.loads(path.read_text())
    assert decoded["state"] == "unmeasured"
    assert decoded["u_range"] is None
    assert decoded["v_range"] is None
