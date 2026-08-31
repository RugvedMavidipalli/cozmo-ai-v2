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
from .manifest import build_manifest, write_manifest
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
) -> int:
    try:
        capture = require_lidar_capture(input_path)
    except LidarCaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        model = Metric3Dv2Model(variant=variant)
    except ModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        densify_capture(capture, model, output_dir, None, min_confidence, max_depth, guide_radius, guide_eps)
    except (AlignmentError, LidarCaptureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote dense_depth/ and densify_manifest.json to {output_dir}")
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
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
