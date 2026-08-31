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
