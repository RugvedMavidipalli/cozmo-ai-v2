from __future__ import annotations

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


class DepthModel(Protocol):
    def predict(self, rgb: np.ndarray, fx: float) -> np.ndarray: ...


class Metric3Dv2Model:
    """Adapter for YvanYin/Metric3D (torch.hub), implementing its documented
    keep-ratio-resize -> pad -> normalize -> infer -> un-pad -> upsample ->
    de-canonicalize recipe (see hubconf.py's __main__ block upstream)."""

    def __init__(self, variant: str = "metric3d_vit_small", device: str | None = None):
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailableError(
                "torch is required to run inference; install with 'uv sync --group depth'"
            ) from exc

        self._torch = torch
        self._device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._input_size = CONVNEXT_INPUT_SIZE if "convnext" in variant else VIT_INPUT_SIZE
        self._model = torch.hub.load("yvanyin/metric3d", variant, pretrain=True)
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
