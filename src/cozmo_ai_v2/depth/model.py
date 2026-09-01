from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

CANONICAL_FOCAL_LENGTH = 1000.0
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
DEPTH_CLAMP_MAX_M = 300.0

VIT_INPUT_SIZE = (616, 1064)
CONVNEXT_INPUT_SIZE = (544, 1216)


class ModelUnavailableError(RuntimeError):
    pass


def _checkpoint_state_dict(checkpoint):
    """Return model weights from common local Metric3D checkpoint layouts.

    Metric3D v2 releases have used both ``state_dict`` and
    ``model_state_dict`` as their top-level weight key.  Passing the wrapper
    dictionary to ``load_state_dict`` silently leaves the model uninitialised
    when strict loading is disabled, so select the known wrapper explicitly.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict"):
        weights = checkpoint.get(key)
        if isinstance(weights, dict):
            return weights
    return checkpoint


class DepthModel(Protocol):
    def predict(self, rgb: np.ndarray, fx: float) -> np.ndarray: ...


class Metric3Dv2Model:
    """Adapter for YvanYin/Metric3D (torch.hub), implementing its documented
    keep-ratio-resize -> pad -> normalize -> infer -> un-pad -> upsample ->
    de-canonicalize recipe (see hubconf.py's __main__ block upstream)."""

    def __init__(
        self,
        variant: str = "metric3d_vit_small",
        device: str | None = None,
        *,
        weights_path: str | Path | None = None,
        repository: str | Path | None = None,
        model=None,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                "torch is required to run inference; install with 'uv sync --group depth'"
            ) from exc

        self._torch = torch
        self.variant = variant
        self._device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = self._device
        self.weights_path = str(weights_path) if weights_path is not None else None
        self.repository = str(repository) if repository is not None else None
        self._input_size = CONVNEXT_INPUT_SIZE if "convnext" in variant else VIT_INPUT_SIZE
        if model is not None:
            # Dependency injection is useful for offline tests and for
            # applications that manage model lifecycles themselves.
            self._model = model
        else:
            if weights_path is None:
                raise ModelUnavailableError(
                    "torch is installed but Metric3D v2 weights were not supplied; "
                    "download weights separately and pass weights_path (no automatic download is performed)"
                )
            if repository is None or not Path(repository).is_dir():
                raise ModelUnavailableError(
                    "Metric3D repository must be a local checkout when loading offline; "
                    "automatic torch.hub downloads are disabled"
                )
            if not Path(weights_path).is_file():
                raise ModelUnavailableError(f"Metric3D weights file does not exist: {weights_path}")
            self._model = torch.hub.load(str(repository), variant, source="local", pretrain=False)
            checkpoint = torch.load(str(weights_path), map_location="cpu")
            state_dict = _checkpoint_state_dict(checkpoint)
            self._model.load_state_dict(state_dict, strict=False)
        self._model = self._model.to(self._device).eval()

    def predict(self, rgb: np.ndarray, fx: float) -> np.ndarray:
        import cv2

        torch = self._torch
        input_h, input_w = self._input_size
        h, w = rgb.shape[:2]

        scale = min(input_h / h, input_w / w)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        scaled_fx = fx * scale

        rh, rw = resized.shape[:2]
        pad_h = input_h - rh
        pad_w = input_w - rw
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=IMAGENET_MEAN,
        )

        mean = torch.tensor(IMAGENET_MEAN).float()[:, None, None]
        std = torch.tensor(IMAGENET_STD).float()[:, None, None]
        tensor = torch.from_numpy(padded.transpose((2, 0, 1))).float()
        tensor = torch.div((tensor - mean), std)
        tensor = tensor[None, :, :, :].to(self._device)

        with torch.no_grad():
            pred_depth, _confidence, _output_dict = self._model.inference({"input": tensor})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_top : pred_depth.shape[0] - pad_bottom, pad_left : pred_depth.shape[1] - pad_right]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], (h, w), mode="bilinear"
        ).squeeze()

        canonical_to_real_scale = scaled_fx / CANONICAL_FOCAL_LENGTH
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, DEPTH_CLAMP_MAX_M)

        return pred_depth.cpu().numpy().astype(np.float32)
