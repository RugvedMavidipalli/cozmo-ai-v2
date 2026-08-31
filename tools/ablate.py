from __future__ import annotations

import argparse
import json
import sys
import time
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
    snap_to_frame,
    wall_band_mask,
)
from cozmo_ai_v2.pipeline.poses import refine_trajectory, select_keyframes


def evaluate(bundle, poses, stride: int, min_confidence: int, label: str) -> dict:
    """Reconstruct with the given poses and measure the resulting geometry."""
    reconstruction = fuse(
        bundle,
        np.arange(0, len(bundle), stride),
        poses=poses,
        voxel_size=0.02,
        min_confidence=min_confidence,
        max_depth=5.0,
    )
    cloud = reconstruction.cloud
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30)
    )
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    gravity = estimate_gravity(points, hint=bundle.gravity_up, normals=normals)
    frame = estimate_horizontal_frame(normals, gravity.up)
    band = wall_band_mask(points, normals, gravity, gravity.up)
    walls = snap_to_frame(
        extract_walls(frame.to_plan(points[band]), frame.height(points[band])), frame
    )

    sampled, times = sample_world_points(
        bundle,
        np.arange(0, len(bundle), max(stride, 3)),
        poses=poses,
        min_confidence=min_confidence,
    )
    drift = measure_drift(walls, frame, sampled, times)
    long_walls = [w for w in walls if w.length > 1.5]
    residuals = np.array([w.residual_rms for w in long_walls]) if long_walls else np.zeros(1)

    return {
        "label": label,
        "points": len(points),
        "room_height_m": round(gravity.room_height, 4) if gravity.room_height else None,
        "walls_over_1_5m": len(long_walls),
        "drift_median_mm": round(drift.median_spread * 1000, 2),
        "drift_p90_mm": round(drift.p90_spread * 1000, 2),
        "drift_max_mm": round(drift.max_spread * 1000, 2),
        "revisited_walls": drift.revisited_walls,
        "residual_median_mm": round(float(np.median(residuals)) * 1000, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the report's ablations and print the error budget."
    )
    parser.add_argument("capture")
    parser.add_argument("--out", default="out/ablations.json")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args(argv)

    bundle = load_capture(args.capture)
    keyframes = select_keyframes(bundle)
    print(f"{len(bundle)} frames, {len(keyframes)} keyframes\n")

    results: list[dict] = []

    print("[1/4] raw ARKit poses")
    results.append(evaluate(bundle, bundle.poses, args.stride, 1, "raw ARKit"))

    for index, (label, loop) in enumerate(
        (("odometry, no loops", False), ("odometry + loop closure", True)), start=2
    ):
        print(f"[{index}/4] {label}")
        start = time.time()
        poses, report = refine_trajectory(
            bundle, keyframes, enable_loop_closure=loop
        )
        record = evaluate(bundle, poses, args.stride, 1, label)
        record["refine_seconds"] = round(time.time() - start, 1)
        record["loop_edges"] = report.loop_edges
        record["loop_candidates"] = report.loop_candidates
        record["mean_correction_cm"] = round(report.mean_correction * 100, 2)
        record["max_correction_cm"] = round(report.max_correction * 100, 2)
        results.append(record)

    print("[4/4] confidence gating off (depth from glass/dark surfaces kept)")
    results.append(
        evaluate(bundle, bundle.poses, args.stride, 0, "raw + ungated depth")
    )

    print("\n" + "=" * 78)
    print("ERROR BUDGET — drift measured as wall revisit spread")
    print("=" * 78)
    header = f"{'variant':<22} {'median':>9} {'p90':>9} {'max':>9} {'walls':>6} {'height':>8}"
    print(header)
    print("-" * 78)
    for record in results:
        height = f"{record['room_height_m']:.3f}" if record["room_height_m"] else "n/a"
        print(
            f"{record['label']:<22} "
            f"{record['drift_median_mm']:>7.1f}mm "
            f"{record['drift_p90_mm']:>7.1f}mm "
            f"{record['drift_max_mm']:>7.1f}mm "
            f"{record['walls_over_1_5m']:>6} "
            f"{height:>8}"
        )

    baseline = results[0]["drift_median_mm"]
    print("\nchange vs raw ARKit:")
    for record in results[1:]:
        delta = 100 * (1 - record["drift_median_mm"] / baseline)
        direction = "better" if delta > 0 else "WORSE"
        print(f"  {record['label']:<22} {abs(delta):5.1f}% {direction}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
