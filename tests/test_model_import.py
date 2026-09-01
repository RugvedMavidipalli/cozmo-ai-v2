import importlib

import pytest


def test_model_module_imports_without_torch():
    module = importlib.import_module("cozmo_ai_v2.depth.model")
    assert hasattr(module, "Metric3Dv2Model")
    assert hasattr(module, "DepthModel")


def test_constructing_model_without_torch_raises_clear_error():
    from cozmo_ai_v2.depth.model import ModelUnavailableError, Metric3Dv2Model

    with pytest.raises(ModelUnavailableError, match="torch"):
        Metric3Dv2Model()


def test_metric3d_checkpoint_accepts_official_model_state_dict_key():
    from cozmo_ai_v2.depth.model import _checkpoint_state_dict

    weights = {"encoder.weight": object()}

    assert _checkpoint_state_dict({"model_state_dict": weights}) is weights
    assert _checkpoint_state_dict({"state_dict": weights}) is weights
