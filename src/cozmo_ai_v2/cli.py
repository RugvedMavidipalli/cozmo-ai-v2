from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .camera import CameraMatrixError, extract_calibration_4, parse_camera_matrix
from .depth.align import AlignmentError
from .depth.capture import LidarCaptureError, require_lidar_capture
from .depth.densify import densify_capture
from .depth.model import Metric3Dv2Model, ModelUnavailableError
from .detect import InputDetectionError, InputKind, detect_input
from .intrinsics_writer import write_intrinsics_yaml
from .mast3r_slam import Mast3rSlamError, run_rgb_video
from .manifest import build_manifest, write_manifest
from .pipeline.measurements import validate_reference_scale
from .pipeline.slam import (
    SlamResultError,
    integrate_mast3r_results,
    mast3r_results_dir,
    mast3r_trajectory_path,
    write_pose_failure_manifest,
    write_pose_integration_manifest,
)
from .video import VideoProbeError, probe_video


def run_prepare(input_path: Path, output_dir: Path, config: str) -> int:
    try:
        detected = detect_input(input_path)
    except InputDetectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        video_info = probe_video(detected.video_path)
    except VideoProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    calib_path: Path | None = None

    if detected.kind is InputKind.STRAY_SCANNER:
        try:
            matrix = parse_camera_matrix(detected.camera_matrix_path)
        except CameraMatrixError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        calibration = extract_calibration_4(matrix)
        calib_path = output_dir / "intrinsics.yaml"
        write_intrinsics_yaml(calib_path, video_info.width, video_info.height, calibration)

    manifest = build_manifest(
        detected.video_path.resolve(),
        calib_path.resolve() if calib_path is not None else None,
        config,
    )
    write_manifest(output_dir / "manifest.json", manifest)

    print(f"Wrote {'intrinsics.yaml and ' if calib_path else ''}manifest.json to {output_dir}")
    print(manifest.command)
    return 0


def run_densify(
    input_path: Path,
    output_dir: Path,
    variant: str,
    min_confidence: int,
    max_depth: float,
    guide_radius: int,
    guide_eps: float,
    weights_path: Path | None = None,
    repository: Path | None = None,
    device: str | None = None,
    stride: int = 1,
    output_scale: float = 1.0,
) -> int:
    try:
        capture = require_lidar_capture(input_path)
    except LidarCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        model = Metric3Dv2Model(
            variant=variant, weights_path=weights_path,
            repository=repository, device=device,
        )
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        indices = _densify_indices(capture, stride)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        densify_capture(
            capture, model, output_dir, indices, min_confidence, max_depth,
            guide_radius, guide_eps, output_scale=output_scale,
        )
    except (AlignmentError, LidarCaptureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote dense_depth/ and densify_manifest.json to {output_dir}")
    return 0


def _densify_indices(capture, stride: int) -> list[int] | None:
    """Return deterministic LiDAR frame indices for optional temporal sampling."""
    if stride < 1:
        raise ValueError(f"densify stride must be at least 1, got {stride}")
    if stride == 1:
        return None
    indices: list[int] = []
    for path in capture.depth_dir.glob("*.png"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return sorted(indices)[::stride]


def run_slam(
    input_path: Path,
    mast3r_slam_dir: Path,
    config: str,
    python_executable: str,
    save_as: str | None,
    no_viz: bool,
    pose_priors_path: Path | None = None,
    metrics_path: Path | None = None,
    pose_manifest_path: Path | None = None,
) -> int:
    """Validate and run MASt3R-SLAM for an uncalibrated RGB video."""

    try:
        detected = detect_input(input_path)
    except InputDetectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if detected.kind is not InputKind.PLAIN_VIDEO:
        print(
            "error: the 'run' command only accepts a standalone RGB video; "
            "use 'prepare' for a Stray Scanner dataset",
            file=sys.stderr,
        )
        return 1

    if pose_priors_path is None:
        discovered_priors = detected.video_path.parent / "odometry.csv"
        if discovered_priors.is_file():
            pose_priors_path = discovered_priors

    try:
        probe_video(detected.video_path)
    except VideoProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        invocation = run_rgb_video(
            detected.video_path,
            mast3r_slam_dir,
            config,
            python_executable=python_executable,
            save_as=save_as,
            no_viz=no_viz,
            pose_priors_path=pose_priors_path,
        )
    except Mast3rSlamError as exc:
        failure_manifest = pose_manifest_path
        if failure_manifest is None and mast3r_slam_dir.is_dir():
            failure_manifest = mast3r_results_dir(mast3r_slam_dir, detected.video_path, save_as) / "pose_provenance.json"
        if failure_manifest is not None:
            try:
                write_pose_failure_manifest(
                    failure_manifest,
                    exc,
                    pose_priors_path=pose_priors_path,
                )
            except OSError as manifest_error:
                print(
                    f"error: {exc}; additionally could not write diagnostics to "
                    f"{failure_manifest}: {manifest_error}",
                    file=sys.stderr,
                )
            else:
                print(f"error: {exc}; wrote diagnostics to {failure_manifest}", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    results_dir = mast3r_results_dir(invocation.cwd, detected.video_path, save_as)
    manifest_path = pose_manifest_path or results_dir / "pose_provenance.json"
    try:
        integration = integrate_mast3r_results(
            mast3r_trajectory_path(invocation.cwd, detected.video_path, save_as),
            pose_priors_path=pose_priors_path,
            pose_prior_mode=invocation.pose_prior_mode,
            results_dir=results_dir,
            metrics_path=metrics_path,
        )
        write_pose_integration_manifest(manifest_path, integration)
    except SlamResultError as exc:
        write_pose_failure_manifest(
            manifest_path,
            exc,
            pose_priors_path=pose_priors_path,
            pose_prior_mode=invocation.pose_prior_mode,
        )
        print(f"error: {exc}; wrote diagnostics to {manifest_path}", file=sys.stderr)
        return 1

    print(f"MASt3R-SLAM completed for {detected.video_path}")
    print(f"Results are in {results_dir}")
    print(f"Wrote pose provenance to {manifest_path}")
    print(f"Loop closure: {integration.loop_closure.status}")
    if integration.alignment is not None:
        alignment = integration.alignment
        print(
            f"ARKit alignment: {alignment.method}, {alignment.matched_frames} matches, "
            f"translation RMSE {alignment.translation_rmse_m:.3f} m, "
            f"rotation RMSE {alignment.rotation_rmse_degrees:.2f}°, "
            f"scale divergence {alignment.scale_divergence_fraction:.1%}"
        )
    if integration.fusion_allowed is False:
        print(
            "error: MASt3R-SLAM trajectory failed ARKit divergence gates; "
            "do not fuse this trajectory",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cozmo-ai-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Prepare a Stray Scanner dataset or plain video for MASt3R-SLAM",
    )
    prepare.add_argument("input", type=Path, help="Path to a Stray Scanner dataset folder or a video file")
    prepare.add_argument("--output-dir", type=Path, required=True, help="Directory to write intrinsics.yaml / manifest.json into")
    prepare.add_argument("--config", default="config/base.yaml", help="MASt3R-SLAM config to reference in the emitted command")

    densify = subparsers.add_parser(
        "densify",
        help="Fuse monocular dense depth with LiDAR depth for a Stray Scanner dataset",
    )
    densify.add_argument("input", type=Path, help="Path to a Stray Scanner dataset folder")
    densify.add_argument("--output-dir", type=Path, required=True, help="Directory to write dense_depth/ and densify_manifest.json into")
    densify.add_argument("--variant", default="metric3d_vit_small", help="Metric3D v2 torch.hub variant to use")
    densify.add_argument("--min-confidence", type=int, default=1, help="Minimum Stray Scanner LiDAR confidence to trust (0-2)")
    densify.add_argument("--max-depth", type=float, default=8.0, help="Maximum LiDAR depth (meters) to trust")
    densify.add_argument("--guide-radius", type=int, default=20, help="Guided filter window radius (pixels) for local residual fusion")
    densify.add_argument("--guide-eps", type=float, default=100.0, help="Guided filter regularization epsilon")
    densify.add_argument(
        "--stride", type=int, default=1,
        help="process every Nth LiDAR/RGB frame; use the same stride for downstream fusion",
    )
    densify.add_argument(
        "--output-scale", type=float, default=1.0,
        help=(
            "aspect-preserving RGB/dense-depth scale in (0, 1]; recorded in the "
            "manifest so downstream intrinsics are scaled consistently"
        ),
    )
    densify.add_argument(
        "--weights", type=Path, default=None,
        help="local Metric3D v2 checkpoint; no weights are downloaded automatically",
    )
    densify.add_argument(
        "--metric3d-repository", type=Path, default=None,
        help="local Metric3D checkout used to construct the model architecture",
    )
    densify.add_argument("--device", default=None, help="inference device (cpu/cuda/mps)")

    scale = subparsers.add_parser(
        "validate-scale",
        help="validate an explicit marker, tape, or user-supplied reference scale",
    )
    scale.add_argument(
        "--reference-type", choices=["marker", "tape", "user", "door"], required=True,
    )
    scale.add_argument("--observed-m", type=float, required=True)
    scale.add_argument("--known-m", type=float, required=True)
    scale.add_argument("--tolerance-m", type=float)

    run = subparsers.add_parser(
        "run",
        help="Run MASt3R-SLAM on a standalone RGB video without calibration",
    )
    run.add_argument("input", type=Path, help="Path to an RGB video file")
    run.add_argument(
        "--mast3r-slam-dir",
        type=Path,
        required=True,
        help="Path to an installed MASt3R-SLAM checkout",
    )
    run.add_argument("--config", default="config/base.yaml", help="Config path relative to the MASt3R-SLAM checkout")
    run.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable from the MASt3R-SLAM environment",
    )
    run.add_argument("--save-as", help="Optional subdirectory name under MASt3R-SLAM/logs")
    run.add_argument("--no-viz", action="store_true", help="Run MASt3R-SLAM without its visualization window")
    run.add_argument(
        "--pose-priors",
        type=Path,
        help=(
            "Optional Stray Scanner/ARKit odometry.csv. If upstream MASt3R-SLAM "
            "does not support priors directly, it is used for post-run validation."
        ),
    )
    run.add_argument(
        "--metrics-path",
        type=Path,
        help="Optional MASt3R-SLAM loop-closure metrics JSON sidecar",
    )
    run.add_argument(
        "--pose-manifest",
        type=Path,
        help="Where to write pose provenance (default: MASt3R-SLAM results directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return run_prepare(args.input, args.output_dir, args.config)
    if args.command == "densify":
        return run_densify(
            args.input, args.output_dir, args.variant,
            args.min_confidence, args.max_depth, args.guide_radius, args.guide_eps,
            args.weights, args.metric3d_repository, args.device,
            args.stride, args.output_scale,
        )
    if args.command == "validate-scale":
        import json

        validation = validate_reference_scale(
            args.observed_m,
            args.known_m,
            reference_type=args.reference_type,
            tolerance_m=args.tolerance_m,
        )
        print(json.dumps(validation.to_dict(), indent=2))
        return 0 if validation.status in {"validated", "advisory"} else 1
    if args.command == "run":
        return run_slam(
            args.input,
            args.mast3r_slam_dir,
            args.config,
            args.python_executable,
            args.save_as,
            args.no_viz,
            args.pose_priors,
            args.metrics_path,
            args.pose_manifest,
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
