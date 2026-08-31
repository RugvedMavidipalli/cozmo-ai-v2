from __future__ import annotations

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.planes import HorizontalFrame
from cozmo_ai_v2.pipeline.projection import project_wall_density
from cozmo_ai_v2.pipeline.roomformer import (
    RoomFormerAdapter,
    RoomFormerConfig,
    RoomFormerPrediction,
    build_roomformer_tensor,
)
from cozmo_ai_v2.pipeline.vectorizer import build_vectorizer_input, build_vectorizer_output


def _frame() -> HorizontalFrame:
    return HorizontalFrame(
        up=np.array([0.0, 0.0, 1.0]),
        right=np.array([1.0, 0.0, 0.0]),
        forward=np.array([0.0, 1.0, 0.0]),
        yaw=0.0,
        manhattan_fraction=1.0,
    )


def _input(points: np.ndarray | None = None):
    points = (
        np.asarray([[0.05, 0.05, 0.5], [0.55, 0.05, 0.5]], dtype=float)
        if points is None
        else np.asarray(points, dtype=float)
    )
    density = project_wall_density(
        points,
        _frame(),
        0.0,
        bounds=((0.0, 0.0), (1.0, 1.0)),
        resolution=0.5,
    )
    return build_vectorizer_input(density)


def test_roomformer_tensor_preserves_xy_orientation_and_channels() -> None:
    vector_input = _input()
    tensor = build_roomformer_tensor(vector_input)

    assert tensor.shape == (1, 2, 2, 2)
    # The first source point is x-cell 0/y-cell 0; the second is x-cell 1/y-cell 0.
    assert tensor.data[0, 0, 0, 0] > 0
    assert tensor.data[0, 0, 1, 0] > 0
    assert tensor.data[0, 0, 0, 1] == 0
    np.testing.assert_array_equal(tensor.data[0, 1], vector_input.observability)


def test_default_adapter_is_lazy_and_falls_back_without_importing_roomformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []

    def fail_import(name: str):
        imported.append(name)
        raise AssertionError("optional RoomFormer import should be lazy")

    monkeypatch.setattr("cozmo_ai_v2.pipeline.roomformer.importlib.import_module", fail_import)
    proposal = RoomFormerAdapter().predict(_input())

    assert imported == []
    assert not proposal.available
    assert proposal.segments == ()
    assert proposal.model_provenance == "fallback: point-cloud wall graph"
    assert proposal.fallback_reason == "no local RoomFormer checkpoint configured"


def test_mocked_roomformer_prediction_becomes_global_wall_graph_proposal() -> None:
    received = []

    def factory(config: RoomFormerConfig):
        assert config.model_name == "roomformer"

        def backend(tensor: np.ndarray):
            received.append(tensor.copy())
            return {
                "coordinate_space": "normalized",
                "coordinate_convention": "finished_face",
                "confidence": 0.82,
                "model_provenance": "mock-roomformer@unit-test",
                "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "topology": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
            }

        return backend

    proposal = RoomFormerAdapter(backend_factory=factory).predict(_input())

    assert proposal.available
    assert proposal.model_provenance == "mock-roomformer@unit-test"
    assert proposal.confidence == 0.82
    assert len(proposal.segments) == 4
    assert len(proposal.nodes) == 4
    assert proposal.graph.topology == ((0, 1), (1, 2), (2, 3), (3, 0))
    np.testing.assert_allclose(proposal.corners[2], [1.0, 1.0])
    assert received[0].shape == (1, 2, 2, 2)


def test_cell_coordinates_use_raster_boundaries_and_empty_map_is_safe() -> None:
    outputs = []

    def factory(_config):
        def backend(_tensor):
            return {
                "coordinate_space": "cell",
                "corners": [[0, 0], [2, 0], [2, 2]],
                "topology": [[0, 1], [1, 2], [2, 0]],
                "confidence": 0.7,
            }

        return backend

    proposal = RoomFormerAdapter(backend_factory=factory).predict(_input())
    np.testing.assert_allclose(proposal.corners, [[0, 0], [1, 0], [1, 1]])

    def should_not_load(_config):
        outputs.append(True)
        raise AssertionError("empty maps must not invoke the optional backend")

    empty = RoomFormerAdapter(backend_factory=should_not_load).predict(_input(np.empty((0, 3))))
    assert not empty.available
    assert empty.fallback_reason == "empty wall-density map"
    assert outputs == []


def test_prediction_shape_and_convention_are_validated() -> None:
    with pytest.raises(ValueError, match="polygons"):
        RoomFormerPrediction(polygons=(np.zeros((2, 2)),))
    with pytest.raises(ValueError, match="topology"):
        RoomFormerPrediction(
            corners=np.zeros((2, 2)),
            topology=((0, 2),),
        )
    with pytest.raises(ValueError, match="convention"):
        RoomFormerAdapter(
            RoomFormerConfig(coordinate_convention="centerline")
        ).predict(_input())


def test_sd_tq_opening_extension_is_mockable_without_loading_a_second_model() -> None:
    def factory(_config):
        return lambda _tensor: {
            "corners": [[0, 0], [1, 0], [1, 1]],
            "topology": [[0, 1], [1, 2], [2, 0]],
            "confidence": 0.75,
        }

    calls = []

    def opening_predictor(tensor: np.ndarray, prediction: RoomFormerPrediction):
        calls.append((tensor.shape, prediction.confidence))
        return [{
            "wall_index": -1,
            "kind": "door",
            "u_range_m": [0.2, 0.8],
            "v_range_m": [0.0, 2.0],
            "confidence": 0.6,
            "source": "RoomFormer SD-TQ mock",
        }]

    proposal = RoomFormerAdapter(
        backend_factory=factory,
        opening_predictor=opening_predictor,
    ).predict(_input())

    assert proposal.opening_extension == "SD-TQ opening predictor"
    assert len(proposal.opening_predictions) == 1
    assert proposal.opening_predictions[0].wall_index == -1
    assert calls == [((1, 2, 2, 2), 0.75)]
    metadata = proposal.to_metadata()
    assert metadata["tensor"]["layout"] == "(batch, channel, x, y)"
    assert metadata["graph"]["provenance"] == "roomformer"


def test_roomformer_proposal_is_preserved_at_vectorizer_output_boundary() -> None:
    vector_input = _input()
    proposal = RoomFormerAdapter().predict(vector_input)
    output = build_vectorizer_output(vector_input, roomformer=proposal)

    assert output.roomformer is proposal
    assert output.to_metadata()["roomformer"]["available"] is False
    assert output.to_metadata()["roomformer"]["tensor"]["shape"] == [1, 2, 2, 2]
