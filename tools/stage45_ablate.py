"""Compare Stage 4/5 depth-source and TSDF choices without model execution.

Example:
    uv run python tools/stage45_ablate.py recordings-2 \
        --dense-depth-dir out/recordings-2 \
        --depth-sources auto,dense,raw \
        --tsdf baseline:0.02:0.08,coarse:0.04:0.16 \
        --out out/recordings-2/stage45_ablation.json

The command only reads existing capture/depth artifacts and runs CPU Open3D
TSDF integration. It never downloads or executes a depth model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cozmo_ai_v2.pipeline.diagnostics import TSDFVariant, compare_tsdf_parameters
from cozmo_ai_v2.pipeline.ingest import load_capture


def _parse_tsdf_variants(value: str) -> list[TSDFVariant]:
    variants: list[TSDFVariant] = []
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) not in {2, 3} or not parts[0]:
            raise ValueError(
                f"invalid TSDF variant {item!r}; expected label:voxel[:sdf_trunc]"
            )
        variants.append(
            TSDFVariant(
                label=parts[0],
                voxel_size_m=float(parts[1]),
                sdf_trunc_m=float(parts[2]) if len(parts) == 3 else None,
            )
        )
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Stage 4/5 depth and TSDF choices using existing artifacts"
    )
    parser.add_argument("capture")
    parser.add_argument("--out", default="out/stage45_ablation.json")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument("--max-depth", type=float, default=3.5)
    parser.add_argument("--depth-sources", default="auto")
    parser.add_argument("--frame-association", choices=("pts", "index"), default="pts")
    parser.add_argument("--pts-tolerance-s", type=float, default=None)
    parser.add_argument("--dense-depth-dir", type=Path, default=None)
    parser.add_argument("--densify-manifest", type=Path, default=None)
    parser.add_argument("--pose-source", choices=("auto", "arkit", "slam"), default="auto")
    parser.add_argument("--slam-poses", type=Path, default=None)
    parser.add_argument(
        "--tsdf",
        default="baseline:0.02:0.08,coarse:0.04:0.16",
        help="comma-separated label:voxel[:sdf_trunc] variants",
    )
    args = parser.parse_args(argv)

    if args.stride <= 0:
        parser.error("--stride must be positive")
    try:
        variants = _parse_tsdf_variants(args.tsdf)
        sources = [source.strip() for source in args.depth_sources.split(",") if source.strip()]
        if not sources or any(source not in {"auto", "dense", "raw"} for source in sources):
            raise ValueError("--depth-sources must contain only auto, dense, or raw")
    except ValueError as exc:
        parser.error(str(exc))

    dense_depth_dir = args.dense_depth_dir
    if dense_depth_dir is None and args.densify_manifest is not None:
        dense_depth_dir = args.densify_manifest.parent / "dense_depth"

    bundle = load_capture(
        args.capture,
        pose_source=args.pose_source,
        slam_poses_path=args.slam_poses,
        dense_depth_dir=dense_depth_dir,
    )

    indices = list(range(0, len(bundle), args.stride))
    records: list[dict] = []
    for source in sources:
        source_records = compare_tsdf_parameters(
            bundle,
            variants,
            indices=indices,
            poses=bundle.poses,
            pose_source=bundle.pose_source,
            dense_depth_dir=str(dense_depth_dir) if dense_depth_dir is not None else None,
            densify_manifest=str(args.densify_manifest) if args.densify_manifest is not None else None,
            min_confidence=args.min_confidence,
            max_depth=args.max_depth,
            depth_source=source,
            frame_association=args.frame_association,
            pts_tolerance_s=args.pts_tolerance_s,
        )
        for record in source_records:
            record["label"] = f"{source}/{record['label']}"
            records.append(record)

    output = {
        "capture": str(Path(args.capture)),
        "configuration": {
            "stride": args.stride,
            "min_confidence": args.min_confidence,
            "max_depth_m": args.max_depth,
            "depth_sources": sources,
            "frame_association": args.frame_association,
            "pts_tolerance_s": args.pts_tolerance_s,
            "indices": indices,
        },
        "variants": records,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {output_path} ({len(records)} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
