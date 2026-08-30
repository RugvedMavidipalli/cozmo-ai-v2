from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .camera import CameraMatrixError, extract_calibration_4, parse_camera_matrix
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return run_prepare(args.input, args.output_dir, args.config)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
