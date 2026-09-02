from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import extract_calibration_4, parse_camera_matrix
from .align import AlignmentError, ScaleShiftFit, fit_scale_shift
from .capture import LidarCaptureInput, _sidecar_frame_count, iter_capture_frames
from ..pipeline.ingest import VideoAvailability
from .fusion import FusionResult, fuse_local_residual
from .model import DepthModel

DEPTH_SCALE_M_TO_MM = 1000.0


@dataclass(frozen=True)
class DenseDepthResult:
    dense_depth_m: np.ndarray
    fit: ScaleShiftFit
    fusion: FusionResult
    qc_mask: np.ndarray | None = None


def _scaled_color(
    color: np.ndarray,
    output_scale: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Return an aspect-preserving RGB raster and its realised xy scale."""
    if not np.isfinite(output_scale) or not 0.0 < output_scale <= 1.0:
        raise ValueError(f"output_scale must be finite and in (0, 1], got {output_scale}")
    height, width = color.shape[:2]
    scaled_width = max(1, round(width * output_scale))
    scaled_height = max(1, round(height * output_scale))
    scale = (scaled_width / width, scaled_height / height)
    if scale == (1.0, 1.0):
        return color, scale
    return cv2.resize(color, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA), scale


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
    output_scale: float = 1.0,
) -> DenseDepthResult:
    output_color, (scale_x, _scale_y) = _scaled_color(color, output_scale)
    mono_fullres = model.predict(output_color, fx * scale_x)

    lidar_h, lidar_w = lidar_depth_m.shape
    mono_at_lidar_res = cv2.resize(mono_fullres, (lidar_w, lidar_h), interpolation=cv2.INTER_AREA)

    fit = fit_scale_shift(mono_at_lidar_res, lidar_depth_m, confidence, min_confidence, max_depth)
    corrected_mono_fullres = fit.scale * mono_fullres + fit.shift

    fusion = fuse_local_residual(
        output_color, corrected_mono_fullres, lidar_depth_m, confidence,
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
    output_scale: float = 1.0,
) -> None:
    if not 0.0 <= min_qc_coverage <= 1.0:
        raise ValueError(f"min_qc_coverage must be in [0, 1], got {min_qc_coverage}")
    if max_qc_rms_m < 0.0:
        raise ValueError(f"max_qc_rms_m must be non-negative, got {max_qc_rms_m}")
    if not np.isfinite(output_scale) or not 0.0 < output_scale <= 1.0:
        raise ValueError(f"output_scale must be finite and in (0, 1], got {output_scale}")
    matrix = parse_camera_matrix(capture.camera_matrix_path)
    fx = extract_calibration_4(matrix)[0]

    available_indices = sorted(
        {
            int(path.stem)
            for path in capture.depth_dir.glob("*.png")
            if path.stem.isdigit()
        }
        & {
            int(path.stem)
            for path in capture.confidence_dir.glob("*.png")
            if path.stem.isdigit()
        }
    )
    selected_indices = (
        available_indices
        if indices is None
        else sorted({int(index) for index in indices if int(index) >= 0})
    )

    dense_depth_dir = output_dir / "dense_depth"
    dense_depth_dir.mkdir(parents=True, exist_ok=True)
    dense_confidence_dir = output_dir / "dense_confidence"
    dense_confidence_dir.mkdir(parents=True, exist_ok=True)
    dense_qc_dir = output_dir / "dense_qc"
    dense_qc_dir.mkdir(parents=True, exist_ok=True)

    frame_reports = []
    expected_frame_count = _sidecar_frame_count(capture)
    availability = VideoAvailability(
        expected_frame_count=expected_frame_count,
        association_mode=("pts" if capture.sidecar_timestamps is not None else "index"),
        sidecar_timestamps=capture.sidecar_timestamps,
    )
    for frame in iter_capture_frames(capture, indices, availability=availability):
        _output_color, rgb_scale = _scaled_color(frame.color, output_scale)
        try:
            result = densify_frame(
                model, frame.color, fx, frame.depth_m, frame.confidence,
                min_confidence, max_depth, guide_radius, guide_eps, output_scale,
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
            "input_depth_unit": "m",
            "output_depth_unit": "mm",
            "registration_alignment": {
                "mono_to_lidar": "INTER_AREA_resize",
                "lidar_to_rgb": "INTER_NEAREST_resize",
                "dense_output": "native_rgb" if rgb_scale == (1.0, 1.0) else "scaled_rgb",
            },
            "source_rgb_resolution": [int(frame.color.shape[1]), int(frame.color.shape[0])],
            "dense_rgb_scale": [float(rgb_scale[0]), float(rgb_scale[1])],
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
        "selected_frame_indices": selected_indices,
        "population": {
            "input_frames": expected_frame_count,
            "selected_frames": len(selected_indices),
            # A raster is written for both QC-approved and QC-rejected model
            # outputs.  The separate QC count keeps the accounting honest.
            "densified_frames": sum("depth_path" in report for report in frame_reports),
            "qc_approved_frames": sum(bool(report.get("qc_approved")) for report in frame_reports),
            "rejected_frames": sum(report.get("status") == "rejected" for report in frame_reports),
            "missing_selected_frames": max(0, len(selected_indices) - len(frame_reports)),
        },
        "video_availability": availability.to_dict(),
        "model": {
            "adapter": type(model).__name__,
            "variant": getattr(model, "variant", None),
            "device": getattr(model, "device", None),
            "weights_path": getattr(model, "weights_path", None),
            "repository": getattr(model, "repository", None),
        },
        "min_confidence": min_confidence,
        "max_depth": max_depth,
        "depth_provenance": "metric3d_v2_scale_shift_lidar_residual",
        "units": {
            "lidar_input": "m",
            "model_canonical_output": "m",
            "dense_raster_output": "mm",
        },
        "registration_alignment": {
            "mono_to_lidar": "INTER_AREA_resize",
            "lidar_to_rgb": "INTER_NEAREST_resize",
            "dense_output": "native_rgb" if output_scale == 1.0 else "scaled_rgb",
            "confidence_to_rgb": "INTER_NEAREST_resize",
        },
        "dense_rgb_scale": [float(output_scale), float(output_scale)],
        "filter_policy": {
            "confidence_threshold": min_confidence,
            "max_depth_m": max_depth,
            "max_depth_inclusive": True,
            "invalid_depth_action": "zero",
        },
        "qc_policy": {
            "min_qc_coverage": min_qc_coverage,
            "max_qc_rms_m": max_qc_rms_m,
        },
        "frames": frame_reports,
    }
    with (output_dir / "densify_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
