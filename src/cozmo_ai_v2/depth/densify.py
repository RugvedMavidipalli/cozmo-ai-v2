from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import extract_calibration_4, parse_camera_matrix
from .align import ScaleShiftFit, fit_scale_shift
from .capture import LidarCaptureInput, iter_capture_frames
from .fusion import FusionResult, fuse_local_residual
from .model import DepthModel

DEPTH_SCALE_M_TO_MM = 1000.0


@dataclass(frozen=True)
class DenseDepthResult:
    dense_depth_m: np.ndarray
    fit: ScaleShiftFit
    fusion: FusionResult


def densify_frame(
    model: DepthModel,
    color: np.ndarray,
    fx: float,
    lidar_depth_m: np.ndarray,
    confidence: np.ndarray,
    min_confidence: int = 1,
    max_depth: float = 8.0,
    guide_radius: int = 20,
    guide_eps: float = 100.0,
) -> DenseDepthResult:
    mono_fullres = model.predict(color, fx)

    lidar_h, lidar_w = lidar_depth_m.shape
    mono_at_lidar_res = cv2.resize(mono_fullres, (lidar_w, lidar_h), interpolation=cv2.INTER_AREA)

    fit = fit_scale_shift(mono_at_lidar_res, lidar_depth_m, confidence, min_confidence, max_depth)
    corrected_mono_fullres = fit.scale * mono_fullres + fit.shift

    fusion = fuse_local_residual(
        color, corrected_mono_fullres, lidar_depth_m, confidence,
        min_confidence, max_depth, guide_radius, guide_eps,
    )

    return DenseDepthResult(dense_depth_m=fusion.fused_depth_m, fit=fit, fusion=fusion)


def densify_capture(
    capture: LidarCaptureInput,
    model: DepthModel,
    output_dir: Path,
    indices: list[int] | None = None,
    min_confidence: int = 1,
    max_depth: float = 8.0,
    guide_radius: int = 20,
    guide_eps: float = 100.0,
) -> None:
    matrix = parse_camera_matrix(capture.camera_matrix_path)
    fx = extract_calibration_4(matrix)[0]

    dense_depth_dir = output_dir / "dense_depth"
    dense_depth_dir.mkdir(parents=True, exist_ok=True)

    frame_reports = []
    for frame in iter_capture_frames(capture, indices):
        result = densify_frame(
            model, frame.color, fx, frame.depth_m, frame.confidence,
            min_confidence, max_depth, guide_radius, guide_eps,
        )

        depth_mm = np.clip(result.dense_depth_m * DEPTH_SCALE_M_TO_MM, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(dense_depth_dir / f"{frame.index:06d}.png"), depth_mm)

        frame_reports.append({
            "index": frame.index,
            "scale": result.fit.scale,
            "shift": result.fit.shift,
            "used_pixels": result.fit.used_pixels,
            "rms_residual_m": result.fit.rms_residual_m,
            "covered_fraction": result.fusion.covered_fraction,
        })

    manifest = {
        "capture": str(capture.root),
        "frame_count": len(frame_reports),
        "min_confidence": min_confidence,
        "max_depth": max_depth,
        "frames": frame_reports,
    }
    with (output_dir / "densify_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
