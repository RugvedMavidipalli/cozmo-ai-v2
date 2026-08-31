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
    qc_mask: np.ndarray | None = None


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

    dense_depth = fusion.fused_depth_m.astype(np.float32, copy=False)
    qc_mask = np.isfinite(dense_depth) & (dense_depth > 0) & (dense_depth <= max_depth)
    return DenseDepthResult(
        dense_depth_m=dense_depth,
        fit=fit,
        fusion=fusion,
        qc_mask=qc_mask,
    )


def densify_capture(
    capture: LidarCaptureInput,
    model: DepthModel,
    output_dir: Path,
    indices: list[int] | None = None,
    min_confidence: int = 1,
    max_depth: float = 8.0,
    guide_radius: int = 20,
    guide_eps: float = 100.0,
    min_qc_coverage: float = 0.25,
    max_qc_rms_m: float = 0.25,
) -> None:
    if not 0.0 <= min_qc_coverage <= 1.0:
        raise ValueError(f"min_qc_coverage must be in [0, 1], got {min_qc_coverage}")
    if max_qc_rms_m < 0.0:
        raise ValueError(f"max_qc_rms_m must be non-negative, got {max_qc_rms_m}")
    matrix = parse_camera_matrix(capture.camera_matrix_path)
    fx = extract_calibration_4(matrix)[0]

    dense_depth_dir = output_dir / "dense_depth"
    dense_depth_dir.mkdir(parents=True, exist_ok=True)
    dense_confidence_dir = output_dir / "dense_confidence"
    dense_confidence_dir.mkdir(parents=True, exist_ok=True)
    dense_qc_dir = output_dir / "dense_qc"
    dense_qc_dir.mkdir(parents=True, exist_ok=True)

    frame_reports = []
    for frame in iter_capture_frames(capture, indices):
        try:
            result = densify_frame(
                model, frame.color, fx, frame.depth_m, frame.confidence,
                min_confidence, max_depth, guide_radius, guide_eps,
            )
        except AlignmentError as exc:
            # A single bad frame must not invalidate a whole capture.  Its
            # raw LiDAR can still be consumed by Stage 5, and the explicit
            # rejection is retained in the manifest for auditability.
            frame_reports.append({
                "index": frame.index,
                "status": "rejected",
                "qc_approved": False,
                "qc_reason": str(exc),
            })
            continue

        depth_mm = np.clip(result.dense_depth_m * DEPTH_SCALE_M_TO_MM, 0, 65535).astype(np.uint16)
        depth_path = dense_depth_dir / f"{frame.index:06d}.png"
        confidence_path = dense_confidence_dir / f"{frame.index:06d}.png"
        qc_path = dense_qc_dir / f"{frame.index:06d}.png"
        if not cv2.imwrite(str(depth_path), depth_mm):
            raise OSError(f"could not write dense depth {depth_path}")

        # Confidence is retained at the same full RGB resolution as the
        # dense raster.  It is metadata for downstream weighting; the QC
        # mask controls which model output is safe to integrate.
        confidence_full = cv2.resize(
            frame.confidence, (result.dense_depth_m.shape[1], result.dense_depth_m.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if not cv2.imwrite(str(confidence_path), confidence_full.astype(np.uint8)):
            raise OSError(f"could not write dense confidence {confidence_path}")
        qc_mask = result.qc_mask
        if qc_mask is None:
            qc_mask = np.isfinite(result.dense_depth_m) & (result.dense_depth_m > 0) & (result.dense_depth_m <= max_depth)
        if not cv2.imwrite(str(qc_path), (qc_mask.astype(np.uint8) * 255)):
            raise OSError(f"could not write dense QC mask {qc_path}")

        valid_fraction = float(qc_mask.mean())
        approved = (
            valid_fraction >= min_qc_coverage
            and result.fit.rms_residual_m <= max_qc_rms_m
        )

        frame_reports.append({
            "index": frame.index,
            "status": "qc_approved" if approved else "rejected",
            "qc_approved": approved,
            "qc_reason": "" if approved else (
                f"valid_fraction={valid_fraction:.4f} < {min_qc_coverage:.4f}"
                if valid_fraction < min_qc_coverage
                else f"rms_residual_m={result.fit.rms_residual_m:.4f} > {max_qc_rms_m:.4f}"
            ),
            "depth_path": str(depth_path.relative_to(output_dir)),
            "confidence_path": str(confidence_path.relative_to(output_dir)),
            "qc_mask_path": str(qc_path.relative_to(output_dir)),
            "depth_unit": "mm",
            "depth_resolution": [int(result.dense_depth_m.shape[1]), int(result.dense_depth_m.shape[0])],
            "scale": result.fit.scale,
            "shift": result.fit.shift,
            "used_pixels": result.fit.used_pixels,
            "rms_residual_m": result.fit.rms_residual_m,
            "covered_fraction": result.fusion.covered_fraction,
            "valid_fraction": valid_fraction,
        })

    manifest = {
        "capture": str(capture.root),
        "frame_count": len(frame_reports),
        "min_confidence": min_confidence,
        "max_depth": max_depth,
        "depth_provenance": "metric3d_v2_scale_shift_lidar_residual",
        "qc_policy": {
            "min_qc_coverage": min_qc_coverage,
            "max_qc_rms_m": max_qc_rms_m,
        },
        "frames": frame_reports,
    }
    with (output_dir / "densify_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
