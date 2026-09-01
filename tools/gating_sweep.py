from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


from cozmo_ai_v2.pipeline.drift import measure_drift, sample_world_points
from cozmo_ai_v2.pipeline.fuse import fuse
from cozmo_ai_v2.pipeline.geometry import estimate_gravity
from cozmo_ai_v2.pipeline.ingest import load_capture
from cozmo_ai_v2.pipeline.planes import (
    estimate_horizontal_frame,
    extract_walls,
    merge_collinear,
    snap_to_frame,
    wall_band_mask,
)


def evaluate(bundle, stride, min_confidence, max_depth):
    reconstruction = fuse(
        bundle,
        np.arange(0, len(bundle), stride),
        voxel_size=0.02,
        min_confidence=min_confidence,
        max_depth=max_depth,
    )
    cloud = reconstruction.cloud
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30)
    )
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if len(points) < 5000:
        return None

    gravity = estimate_gravity(points, hint=bundle.gravity_up, normals=normals)
    frame = estimate_horizontal_frame(normals, gravity.up)
    band = wall_band_mask(points, normals, gravity, gravity.up)
    walls = merge_collinear(
        snap_to_frame(
            extract_walls(frame.to_plan(points[band]), frame.height(points[band])),
            frame,
        )
    )
    sampled, times = sample_world_points(
        bundle,
        np.arange(0, len(bundle), max(stride, 3)),
        min_confidence=min_confidence,
        max_depth=max_depth,
    )
    drift = measure_drift(walls, frame, sampled, times)
    long_walls = [w for w in walls if w.length > 1.5]
    return {
        "min_confidence": min_confidence,
        "max_depth": max_depth,
        "points": len(points),
        "walls": len(long_walls),
        "total_wall_m": round(sum(w.length for w in long_walls), 1),
        "room_height_m": round(gravity.room_height, 4) if gravity.room_height else None,
        "drift_median_mm": round(drift.median_spread * 1000, 2),
        "drift_p90_mm": round(drift.p90_spread * 1000, 2),
        "revisited": drift.revisited_walls,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Does tighter depth gating actually buy accuracy?"
    )
    parser.add_argument("capture")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--out", default="out/gating_sweep.json")
    args = parser.parse_args(argv)

    bundle = load_capture(args.capture)
    settings = [(1, 5.0), (2, 5.0), (1, 3.5), (2, 3.5), (2, 2.5)]

    results = []
    for min_confidence, max_depth in settings:
        print(f"conf>={min_confidence}, max_depth={max_depth} ...", flush=True)
        record = evaluate(bundle, args.stride, min_confidence, max_depth)
        if record:
            results.append(record)

    print("\n" + "=" * 88)
    print("DEPTH GATING SWEEP — accuracy vs coverage")
    print("=" * 88)
    print(
        f"  {'conf':>5} {'maxD':>6} {'drift med':>11} {'p90':>9} "
        f"{'walls':>6} {'wall m':>8} {'points':>10} {'height':>8}"
    )
    print("  " + "-" * 84)
    for record in results:
        height = f"{record['room_height_m']:.3f}" if record["room_height_m"] else "n/a"
        print(
            f"  {record['min_confidence']:>5} {record['max_depth']:>6.1f} "
            f"{record['drift_median_mm']:>9.1f}mm {record['drift_p90_mm']:>7.1f}mm "
            f"{record['walls']:>6} {record['total_wall_m']:>8.1f} "
            f"{record['points']:>10} {height:>8}"
        )

    baseline = results[0]
    print("\nvs current default (conf>=1, 5.0 m):")
    for record in results[1:]:
        drift_change = 100 * (1 - record["drift_median_mm"] / baseline["drift_median_mm"])
        coverage_change = 100 * (record["total_wall_m"] / baseline["total_wall_m"] - 1)
        print(
            f"  conf>={record['min_confidence']}, {record['max_depth']:.1f}m: "
            f"drift {drift_change:+5.1f}%  wall coverage {coverage_change:+5.1f}%"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
