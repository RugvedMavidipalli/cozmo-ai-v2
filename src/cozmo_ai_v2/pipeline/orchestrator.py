"""Start-to-finish orchestration for ``cozmo-ai-v2 pipeline``.

The established reconstruction runner deliberately accepts already-prepared
artifacts.  This module owns the user-facing composition around it: detect the
input tier, run the applicable pose/depth preparation, materialize a minimal
RGB-only capture when needed, and invoke the existing stages exactly once.
Every decision is recorded in :mod:`stage_manifest`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..detect import DetectedInput, InputDetectionError, InputKind, detect_input
from ..depth.capture import LidarCaptureError, require_lidar_capture
from ..depth.densify import densify_capture
from ..depth.model import Metric3Dv2Model, ModelUnavailableError
from ..mast3r_slam import Mast3rSlamError, run_rgb_video
from ..video import VideoProbeError, probe_video
from .cli import REPO_ROOT, build_parser as build_reconstruction_parser
from .ingest import _read_slam_poses
from .slam import (
    SlamResultError,
    parse_mast3r_trajectory,
    resample_trajectory,
)
from .stage_manifest import StageManifest


class PipelineOrchestrationError(RuntimeError):
    """Raised when a required start-to-finish stage cannot run safely."""


def _video_timestamps(path: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineOrchestrationError(f"could not open RGB video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count == 0 or width <= 0 or height <= 0:
        raise PipelineOrchestrationError(f"RGB video has no readable frames: {path}")
    return np.arange(count, dtype=np.float64) / fps, fps, (width, height)


def _load_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    """Load a 3x3 matrix or ``[fx, fy, cx, cy]`` calibration sidecar."""
    if path.suffix.lower() == ".csv":
        try:
            matrix = np.loadtxt(path, delimiter=",")
        except (OSError, ValueError) as exc:
            raise PipelineOrchestrationError(f"could not read intrinsics {path}: {exc}") from exc
        if matrix.shape != (3, 3):
            raise PipelineOrchestrationError(
                f"expected a 3x3 camera matrix in {path}, got {matrix.shape}"
            )
        return np.asarray(matrix, dtype=np.float64)
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineOrchestrationError(f"could not read intrinsics {path}: {exc}") from exc
    if isinstance(payload, dict):
        matrix = payload.get("camera_matrix", payload.get("matrix"))
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.shape == (3, 3):
                return matrix
        calibration = payload.get("calibration")
        if isinstance(calibration, (list, tuple)) and len(calibration) == 4:
            fx, fy, cx, cy = (float(value) for value in calibration)
            source_width = int(payload.get("width", width))
            source_height = int(payload.get("height", height))
            matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            matrix[0, [0, 2]] *= width / source_width
            matrix[1, [1, 2]] *= height / source_height
            return matrix
    raise PipelineOrchestrationError(
        f"{path} has neither a 3x3 camera matrix nor calibration [fx, fy, cx, cy]"
    )


def _write_default_intrinsics(path: Path, width: int, height: int) -> np.ndarray:
    """Create an explicit, uncalibrated pinhole prior for RGB-only input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    focal = float(max(width, height))
    matrix = np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.savetxt(path, matrix, delimiter=",", fmt="%.12g")
    return matrix


def _validate_pose_table(timestamps: np.ndarray, poses: np.ndarray, source: Path) -> None:
    if timestamps.ndim != 1 or poses.shape != (len(timestamps), 4, 4) or len(poses) == 0:
        raise PipelineOrchestrationError(
            f"pose table {source} must contain one finite 4x4 pose per timestamp"
        )
    if not np.isfinite(timestamps).all() or not np.isfinite(poses).all():
        raise PipelineOrchestrationError(f"pose table {source} contains non-finite values")
    rotations = poses[:, :3, :3]
    orthogonality = np.einsum("nji,njk->nik", rotations, rotations)
    if not np.allclose(orthogonality, np.eye(3), atol=1e-4) or np.any(np.linalg.det(rotations) <= 0.0):
        raise PipelineOrchestrationError(f"pose table {source} contains invalid rotations")


def _write_pose_json(path: Path, timestamps: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"timestamps": np.asarray(timestamps, dtype=float).tolist(), "poses": np.asarray(poses, dtype=float).tolist()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _link_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise PipelineOrchestrationError(
            f"refusing to overwrite existing staging path: {target}; choose a fresh --out"
        )
    target.symlink_to(source.resolve())


def _dense_paths(args, detected: DetectedInput) -> tuple[Path | None, Path | None]:
    explicit = getattr(args, "dense_depth_dir", None)
    if explicit is not None:
        dense_dir = Path(explicit).expanduser().resolve()
        if dense_dir.is_dir() and not any(dense_dir.glob("*.png")) and (dense_dir / "dense_depth").is_dir():
            dense_dir = dense_dir / "dense_depth"
        manifest = getattr(args, "densify_manifest", None)
        manifest_path = Path(manifest).expanduser().resolve() if manifest else dense_dir.parent / "densify_manifest.json"
        return dense_dir if dense_dir.is_dir() else None, manifest_path if manifest_path.is_file() else None
    if detected.kind is InputKind.STRAY_SCANNER:
        dense_dir = detected.video_path.parent / "dense_depth"
        if dense_dir.is_dir() and any(dense_dir.glob("*.png")):
            manifest = dense_dir.parent / "densify_manifest.json"
            return dense_dir, manifest if manifest.is_file() else None
    return None, None


def _has_approved_dense(manifest_path: Path | None) -> bool:
    if manifest_path is None or not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        bool(frame.get("qc_approved")) or frame.get("status") in {"approved", "accepted", "qc_approved"}
        for frame in payload.get("frames", [])
        if isinstance(frame, dict)
    )


def _model_options(args) -> tuple[str, Path | None, Path | None, str | None]:
    weights = getattr(args, "metric3d_weights", None)
    repository = getattr(args, "metric3d_repository", None)
    variant = getattr(args, "metric3d_variant", "metric3d_vit_small")
    device = getattr(args, "depth_device", None)
    return variant, Path(weights).expanduser() if weights else None, Path(repository).expanduser() if repository else None, device


def _model_or_error(args) -> Metric3Dv2Model:
    variant, weights, repository, device = _model_options(args)
    if weights is None or repository is None:
        raise PipelineOrchestrationError(
            "Metric3D is required for this input tier; pass --metric3d-weights and --metric3d-repository"
        )
    try:
        return Metric3Dv2Model(
            variant=variant,
            weights_path=weights,
            repository=repository,
            device=device,
        )
    except ModelUnavailableError as exc:
        raise PipelineOrchestrationError(str(exc)) from exc


def _generate_monocular_dense(
    video_path: Path,
    matrix: np.ndarray,
    model: Metric3Dv2Model,
    output_dir: Path,
    indices: set[int] | None,
    max_depth: float,
) -> tuple[Path, Path]:
    dense_dir = output_dir / "dense_depth"
    confidence_dir = output_dir / "dense_confidence"
    qc_dir = output_dir / "dense_qc"
    for directory in (dense_dir, confidence_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineOrchestrationError(f"could not open RGB video for depth: {video_path}")
    reports = []
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if indices is not None and index not in indices:
                index += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            predicted = np.asarray(model.predict(rgb, float(matrix[0, 0])), dtype=np.float32)
            if predicted.shape != rgb.shape[:2]:
                raise PipelineOrchestrationError(
                    f"Metric3D returned {predicted.shape} for RGB frame {rgb.shape[:2]}"
                )
            qc = np.isfinite(predicted) & (predicted > 0.0) & (predicted <= max_depth)
            if float(qc.mean()) < 0.25:
                reports.append({"index": index, "status": "rejected", "qc_approved": False, "qc_reason": "valid_fraction below 0.25"})
                index += 1
                continue
            depth_path = dense_dir / f"{index:06d}.png"
            confidence_path = confidence_dir / f"{index:06d}.png"
            qc_path = qc_dir / f"{index:06d}.png"
            if not cv2.imwrite(str(depth_path), np.clip(predicted * 1000.0, 0, 65535).astype(np.uint16)):
                raise PipelineOrchestrationError(f"could not write dense depth {depth_path}")
            if not cv2.imwrite(str(confidence_path), np.full(qc.shape, 2, dtype=np.uint8)):
                raise PipelineOrchestrationError(f"could not write dense confidence {confidence_path}")
            if not cv2.imwrite(str(qc_path), (qc.astype(np.uint8) * 255)):
                raise PipelineOrchestrationError(f"could not write dense QC mask {qc_path}")
            reports.append({
                "index": index,
                "status": "qc_approved",
                "qc_approved": True,
                "qc_reason": "monocular depth validity and range check",
                "depth_path": str(depth_path.relative_to(output_dir)),
                "confidence_path": str(confidence_path.relative_to(output_dir)),
                "qc_mask_path": str(qc_path.relative_to(output_dir)),
                "depth_unit": "mm",
                "source_depth_unit": "m",
                "valid_fraction": float(qc.mean()),
            })
            index += 1
    finally:
        capture.release()
    if not reports or not any(report.get("qc_approved") for report in reports):
        raise PipelineOrchestrationError("Metric3D produced no QC-approved dense frames")
    manifest = {
        "capture": str(video_path),
        "frame_count": len(reports),
        "frames": reports,
        "model": _stage_model(model),
        "depth_provenance": "metric3d_v2_monocular_uncalibrated",
        "units": {"model_output": "m", "dense_raster_output": "mm"},
        "dense_rgb_scale": [1.0, 1.0],
        "filter_policy": {"max_depth_m": max_depth, "invalid_depth_action": "zero"},
        "qc_policy": {"min_qc_coverage": 0.25, "metric3d_scale": "uncalibrated"},
    }
    manifest_path = output_dir / "densify_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return dense_dir, manifest_path


def _stage_model(model) -> dict:
    return {
        "adapter": type(model).__name__,
        "variant": getattr(model, "variant", None),
        "device": getattr(model, "device", None),
        "weights_path": getattr(model, "weights_path", None),
        "repository": getattr(model, "repository", None),
    }


def _prepare_rgb_capture(
    input_path: Path,
    staging: Path,
    timestamps: np.ndarray,
    poses: np.ndarray,
    matrix: np.ndarray,
) -> tuple[Path, Path]:
    staging.mkdir(parents=True, exist_ok=True)
    _link_once(input_path, staging / "rgb.mp4")
    np.savetxt(staging / "camera_matrix.csv", matrix, delimiter=",", fmt="%.12g")
    pose_path = staging / "slam_poses.json"
    _write_pose_json(pose_path, timestamps, poses)
    return staging, pose_path


def _prepare_reconstruction_args(args, capture_root: Path, out_dir: Path, dense_dir: Path | None, dense_manifest: Path | None, pose_path: Path | None):
    parser = build_reconstruction_parser()
    parsed = parser.parse_args(["run", str(capture_root), "--out", str(out_dir)])
    for name in (
        "rules", "cache_dir", "calibration", "model", "stride", "voxel", "sdf_trunc",
        "max_depth", "plane_threshold", "plane_min_inliers", "max_planes", "plane_seed",
        "min_confidence", "depth_source", "frame_association", "pts_tolerance_s", "damage_frames",
        "min_views", "rgb_openings", "grounding_dino_model", "sam2_checkpoint", "sam2_config",
        "rgb_device", "opening_frames", "rgb_box_threshold", "rgb_text_threshold", "rgb_min_confidence",
        "roomformer_predictions", "roomformer_min_confidence", "min_detection_confidence", "coverage",
        "wall_thickness", "reference_type", "reference_observed_m", "reference_known_m", "no_refine",
        "no_loop_closure", "no_damage", "no_sam", "debug_furniture", "furniture_overlays",
        "mast3r_trajectory", "run_mast3r", "mast3r_slam_dir", "mast3r_config", "mast3r_python",
        "mast3r_save_as", "mast3r_no_viz", "mast3r_metrics", "mast3r_max_pose_gap",
    ):
        if hasattr(args, name):
            setattr(parsed, name, getattr(args, name))
    parsed.dense_depth_dir = dense_dir
    parsed.densify_manifest = dense_manifest
    parsed.pose_source = "slam" if pose_path is not None else "auto"
    parsed.slam_poses = pose_path
    parsed.mast3r_trajectory = None
    parsed.run_mast3r = False
    parsed.stage_manifest = getattr(args, "stage_manifest", None)
    return parsed


def run_pipeline(args) -> int:
    """Run every applicable stage from one input path to validated exports."""
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    manifest = StageManifest(input_path, out_dir)
    args.stage_manifest = manifest
    detected = None
    dense_dir = None
    dense_manifest = None
    pose_path = None
    capture_root = input_path
    try:
        with manifest.stage("input_detection", inputs=[input_path], outputs=[manifest.path]):
            detected = detect_input(input_path)
            manifest.set_context(
                input_tier=detected.kind.name.lower(),
                pose={"status": "pending", "convention": "not_selected"},
            )

        with manifest.stage("ingest_qc", inputs=[input_path], outputs=[out_dir / "ingest_qc.json"]):
            probe_video(detected.video_path)
            if detected.kind is InputKind.STRAY_SCANNER:
                raw_depth = detected.video_path.parent / "depth"
                dense_dir, dense_manifest = _dense_paths(args, detected)
                if not (raw_depth.is_dir() and any(raw_depth.glob("*.png"))) and not _has_approved_dense(dense_manifest):
                    raise PipelineOrchestrationError(
                        "Stray Scanner input has neither raw LiDAR depth nor a QC-approved dense artifact"
                    )
                # Parse the native capture once before any model work.  The
                # reconstruction runner parses it again after preparation.
                from .ingest import load_capture

                load_capture(input_path, dense_depth_dir=dense_dir)
            else:
                manifest.set_context(depth_provenance="video-only depth is not prepared")
            (out_dir / "ingest_qc.json").write_text(
                json.dumps({"kind": detected.kind.name.lower(), "video": str(detected.video_path) }, indent=2) + "\n",
                encoding="utf-8",
            )

        if detected.kind is InputKind.PLAIN_VIDEO:
            with manifest.stage("poses", inputs=[detected.video_path], outputs=[out_dir / "pose_provenance.json"], model={"name": "MASt3R-SLAM"}):
                timestamps, fps, (width, height) = _video_timestamps(detected.video_path)
                supplied = getattr(args, "slam_poses", None)
                if supplied is not None:
                    pose_path = Path(supplied).expanduser().resolve()
                    if not pose_path.is_file():
                        raise PipelineOrchestrationError(f"SLAM pose table does not exist: {pose_path}")
                    supplied_timestamps, supplied_poses = _read_slam_poses(pose_path, fps)
                    _validate_pose_table(supplied_timestamps, supplied_poses, pose_path)
                    if len(supplied_poses) != len(timestamps):
                        raise PipelineOrchestrationError(
                            f"SLAM pose count {len(supplied_poses)} does not match video frame count {len(timestamps)}"
                        )
                    manifest.set_context(pose={"source": "precomputed_slam", "path": str(pose_path), "convention": "camera_to_world_opencv"})
                else:
                    checkout = getattr(args, "mast3r_slam_dir", None)
                    if checkout is None:
                        raise PipelineOrchestrationError(
                            "RGB-only input requires --mast3r-slam-dir or --slam-poses"
                        )
                    save_as = getattr(args, "mast3r_save_as", None) or f"cozmo-{out_dir.name}"
                    invocation = run_rgb_video(
                        detected.video_path,
                        Path(checkout),
                        getattr(args, "mast3r_config", "config/base.yaml"),
                        python_executable=getattr(args, "mast3r_python", sys.executable),
                        save_as=save_as,
                        no_viz=getattr(args, "mast3r_no_viz", True),
                    )
                    trajectory_path = Path(invocation.cwd) / "logs" / save_as / f"{detected.video_path.stem}.txt"
                    try:
                        trajectory = parse_mast3r_trajectory(trajectory_path)
                        sampled = resample_trajectory(
                            trajectory,
                            timestamps,
                            max_gap_seconds=getattr(args, "mast3r_max_pose_gap", 1.0),
                        )
                    except SlamResultError as exc:
                        raise PipelineOrchestrationError(f"MASt3R-SLAM trajectory check failed: {exc}") from exc
                    pose_path = out_dir / "preprocessing" / "slam_poses.json"
                    _write_pose_json(pose_path, sampled.timestamps, sampled.poses)
                    provenance = out_dir / "pose_provenance.json"
                    provenance.write_text(json.dumps({
                        "status": "ok",
                        "pose_source": "mast3r_slam_rgb_only",
                        "trajectory_path": str(trajectory_path.resolve()),
                        "resampled_frame_count": len(sampled),
                        "coordinate_convention": sampled.coordinate_convention,
                        "loop_closure": "not_reported",
                    }, indent=2) + "\n", encoding="utf-8")
                    manifest.set_context(pose={"source": "mast3r_slam_rgb_only", "path": str(pose_path), "convention": sampled.coordinate_convention})
                matrix_path = getattr(args, "intrinsics", None)
                if matrix_path is not None:
                    matrix = _load_intrinsics(Path(matrix_path).expanduser().resolve(), width, height)
                    matrix_provenance = "supplied_intrinsics"
                else:
                    matrix = _write_default_intrinsics(out_dir / "preprocessing" / "camera_matrix.csv", width, height)
                    matrix_provenance = "uncalibrated_pinhole_prior"
                manifest.update_last("poses", outputs=[pose_path, out_dir / "pose_provenance.json"], pose={"source": "mast3r_slam_rgb_only" if supplied is None else "precomputed_slam", "path": str(pose_path), "convention": "camera_to_world_opencv", "intrinsics": matrix_provenance})
                capture_root, pose_path = _prepare_rgb_capture(
                    detected.video_path,
                    out_dir / "preprocessing" / "capture",
                    timestamps,
                    np.asarray(json.loads(pose_path.read_text())["poses"], dtype=float) if pose_path.suffix == ".json" else _read_slam_poses(pose_path, fps)[1],
                    matrix,
                )
                # Keep the output staging file as the canonical path from now on.
                pose_path = capture_root / "slam_poses.json"
        else:
            with manifest.stage("poses", inputs=[input_path / "odometry.csv"], outputs=[input_path / "odometry.csv"], pose={"source": "arkit", "convention": "camera_to_world_opencv_csv_no_arkit_to_cv_flip"}):
                if not (input_path / "odometry.csv").is_file():
                    raise PipelineOrchestrationError("Stray Scanner input is missing odometry.csv")
                manifest.set_context(pose={"source": "arkit", "path": str(input_path / "odometry.csv"), "convention": "camera_to_world_opencv_csv_no_arkit_to_cv_flip"})

        depth_unavailable_reason = None
        with manifest.stage("depth", inputs=[detected.video_path], outputs=[dense_manifest or out_dir / "preprocessing" / "densify_manifest.json"]):
            if dense_dir is not None and _has_approved_dense(dense_manifest):
                manifest.set_context(depth_provenance="existing_qc_approved_dense_depth")
            elif detected.kind is InputKind.STRAY_SCANNER and (detected.video_path.parent / "depth").is_dir() and any((detected.video_path.parent / "depth").glob("*.png")):
                weights = getattr(args, "metric3d_weights", None)
                repository = getattr(args, "metric3d_repository", None)
                if weights is None and repository is None:
                    depth_unavailable_reason = "Metric3D paths not supplied; raw LiDAR remains the explicit fallback"
                    manifest.set_context(depth_provenance="raw_lidar")
                else:
                    lidar = require_lidar_capture(input_path)
                    model = _model_or_error(args)
                    dense_output = out_dir / "preprocessing" / "densify"
                    with_manifest = dense_output / "densify_manifest.json"
                    depth_indices = sorted(
                        int(path.stem)
                        for path in (input_path / "depth").glob("*.png")
                        if path.stem.isdigit()
                    )
                    indices = depth_indices[::max(1, int(args.stride))]
                    densify_capture(
                        lidar,
                        model,
                        dense_output,
                        indices=indices,
                        min_confidence=int(args.min_confidence),
                        max_depth=float(args.max_depth),
                        output_scale=float(getattr(args, "depth_output_scale", 1.0)),
                    )
                    dense_dir, dense_manifest = dense_output / "dense_depth", with_manifest
                    manifest.set_context(model=_stage_model(model), depth_provenance="metric3d_v2_scale_shift_lidar_residual")
            else:
                model = _model_or_error(args)
                matrix = np.loadtxt(capture_root / "camera_matrix.csv", delimiter=",")
                timestamps, _fps, _size = _video_timestamps(detected.video_path)
                selected = set(range(0, len(timestamps), max(1, int(args.stride))))
                dense_dir, dense_manifest = _generate_monocular_dense(
                    detected.video_path,
                    matrix,
                    model,
                    out_dir / "preprocessing" / "densify",
                    selected,
                    float(args.max_depth),
                )
                manifest.set_context(model=_stage_model(model), depth_provenance="metric3d_v2_monocular_uncalibrated")
        if depth_unavailable_reason:
            manifest.update_last(
                "depth",
                status="unavailable",
                reason=depth_unavailable_reason,
                outputs=[out_dir / "fusion_manifest.json"],
                depth_provenance="raw_lidar",
            )

        reconstruction_args = _prepare_reconstruction_args(
            args, capture_root, out_dir, dense_dir, dense_manifest, pose_path
        )
        return_code = __import__("cozmo_ai_v2.pipeline.cli", fromlist=["run"]).run(reconstruction_args)
        if return_code != 0:
            raise PipelineOrchestrationError(f"reconstruction/export returned nonzero status {return_code}")
        expected = [
            out_dir / "result.json", out_dir / "floorplan.svg", out_dir / "scene.glb",
            out_dir / "cloud.ply", out_dir / "mesh.ply", out_dir / "planes.json",
            out_dir / "fusion_manifest.json", out_dir / "openings.csv",
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise PipelineOrchestrationError("required export artifacts are missing: " + ", ".join(missing))
        if any(stage["stage"] == "export" for stage in manifest.stages):
            manifest.update_last("export", outputs=expected)
        else:
            manifest.record("export", "completed", outputs=expected, reason="completed")
        manifest.finalize("completed")
        print(f"Wrote start-to-finish pipeline outputs to {out_dir}")
        print(f"Stage manifest: {manifest.path}")
        return 0
    except Exception as exc:
        manifest.finalize("failed", str(exc))
        print(f"error: {exc}; stage manifest: {manifest.path}", file=sys.stderr)
        return 1


__all__ = ["PipelineOrchestrationError", "run_pipeline"]
