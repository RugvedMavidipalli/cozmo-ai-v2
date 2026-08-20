"""Attribute wall-position error to its physical causes.

The trajectory ablation showed that loop closure barely moves the residual and
that odometry-only reproduces ARKit exactly, which rules out trajectory drift as
the dominant term.  The remaining candidates are properties of the *depth
sensor*, and they are separable because each predicts a different signature:

  range      - a depth scale or offset error puts a wall further away the
               further you stand from it, so the residual trends with range.
  incidence  - at grazing angles the beam footprint smears across the surface
               and multipath grows, so the residual trends with |cos|.
  confidence - if ARKit's own low-confidence returns carry the error, the
               residual separates by confidence level.

This fits each trend on real walls and reports how much of the error each
explains, then tests whether correcting the dominant one actually shrinks the
per-visit spread that the wall gate cares about.

    python tools/depth_bias.py ../recordings-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fuse import fuse  # noqa: E402
from pipeline.geometry import estimate_gravity  # noqa: E402
from pipeline.ingest import iter_frames, load_capture  # noqa: E402
from pipeline.planes import (  # noqa: E402
    estimate_horizontal_frame,
    extract_walls,
    merge_collinear,
    snap_to_frame,
    wall_band_mask,
)


def gather(bundle, indices, stride=4, min_confidence=0, max_depth=5.0):
    """Per-point observations tagged with range, incidence and confidence."""
    intrinsics = bundle.intrinsics
    chunks = []
    for frame in iter_frames(
        bundle, indices, min_confidence=min_confidence, max_depth=max_depth
    ):
        depth = frame.depth[::stride, ::stride]
        confidence = frame.confidence[::stride, ::stride]
        valid = depth > 0
        if not valid.any():
            continue
        vs, us = np.nonzero(valid)
        z = depth[valid]
        camera = np.stack(
            [
                (us * stride - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (vs * stride - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ],
            axis=1,
        )
        rays = camera / np.linalg.norm(camera, axis=1, keepdims=True)
        pose = bundle.poses[frame.index]
        chunks.append(
            {
                "world": camera @ pose[:3, :3].T + pose[:3, 3],
                "ray": rays @ pose[:3, :3].T,
                "range": np.linalg.norm(camera, axis=1),
                "confidence": confidence[valid],
                "time": np.full(valid.sum(), frame.timestamp),
            }
        )
    return {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}


def analyse(walls, frame, data, band=0.06, min_points=3000):
    """Pool per-wall residuals against range, incidence, and confidence."""
    plan = frame.to_plan(data["world"])
    # `to_plan` is a pair of dot products against unit axes through the origin,
    # so it maps direction vectors as well as points.
    ray_plan = frame.to_plan(data["ray"])
    rows = []

    for wall in walls:
        if wall.length < 1.5:
            continue
        signed = plan @ wall.normal - wall.offset
        along = (plan - wall.start) @ wall.direction
        near = (np.abs(signed) < band) & (along > 0) & (along < wall.length)
        if near.sum() < min_points:
            continue

        # Re-centre on this wall's own points so the comparison is within-wall:
        # an absolute offset would just measure where the wall is.
        residual = signed[near] - np.median(signed[near])
        # Sign the residual along the view direction, so "further away" is
        # positive for every wall regardless of which way its normal points.
        view_sign = np.sign(ray_plan[near] @ wall.normal)
        rows.append(
            {
                "residual": residual * view_sign,
                "range": data["range"][near],
                "incidence": np.abs(ray_plan[near] @ wall.normal),
                "confidence": data["confidence"][near],
                "time": data["time"][near],
            }
        )

    return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}, len(rows)


def trend(x, y, bins):
    """Median residual per bin of x, with counts."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 200:
            continue
        out.append((0.5 * (lo + hi), float(np.median(y[mask])), int(mask.sum())))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--out", default="out/depth_bias.json")
    args = parser.parse_args(argv)

    bundle = load_capture(args.capture)
    indices = np.arange(0, len(bundle), args.stride)

    print("reconstructing ...")
    reconstruction = fuse(
        bundle, indices, voxel_size=0.02, min_confidence=1, max_depth=5.0
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
    walls = merge_collinear(
        snap_to_frame(
            extract_walls(frame.to_plan(points[band]), frame.height(points[band])),
            frame,
        )
    )
    print(f"{len(walls)} walls")

    print("gathering tagged observations ...")
    data = gather(bundle, indices, stride=4)
    pooled, wall_count = analyse(walls, frame, data)
    print(f"{len(pooled['residual'])} observations across {wall_count} walls\n")

    report = {"n": int(len(pooled["residual"])), "walls": wall_count}

    print("=" * 72)
    print("RESIDUAL vs RANGE  (a depth scale/offset error trends here)")
    print("=" * 72)
    print(f"  {'range':>8}  {'median residual':>16}  {'n':>9}")
    range_trend = trend(pooled["range"], pooled["residual"], 10)
    for centre, median, count in range_trend:
        print(f"  {centre:>7.2f}m  {median * 1000:>14.2f}mm  {count:>9}")
    if len(range_trend) > 2:
        xs = np.array([t[0] for t in range_trend])
        ys = np.array([t[1] for t in range_trend])
        slope, intercept = np.polyfit(xs, ys, 1)
        print(
            f"\n  linear fit: {slope * 1000:+.2f} mm per metre of range "
            f"(offset {intercept * 1000:+.2f} mm)"
        )
        print(f"  => across the {xs.min():.1f}-{xs.max():.1f} m observed range, "
              f"a wall moves {abs(slope) * (xs.max() - xs.min()) * 1000:.1f} mm")
        report["range_slope_mm_per_m"] = float(slope * 1000)
        report["range_span_mm"] = float(abs(slope) * (xs.max() - xs.min()) * 1000)

    print("\n" + "=" * 72)
    print("RESIDUAL vs INCIDENCE  (|cos| = 1 is face-on, 0 is grazing)")
    print("=" * 72)
    print(f"  {'|cos|':>8}  {'median residual':>16}  {'n':>9}")
    for centre, median, count in trend(pooled["incidence"], pooled["residual"], 8):
        print(f"  {centre:>8.2f}  {median * 1000:>14.2f}mm  {count:>9}")

    print("\n" + "=" * 72)
    print("RESIDUAL vs ARKIT CONFIDENCE")
    print("=" * 72)
    for level in (0, 1, 2):
        mask = pooled["confidence"] == level
        if mask.sum() < 200:
            continue
        residual = pooled["residual"][mask]
        print(
            f"  level {level}: n={mask.sum():>8}  median {np.median(residual) * 1000:>7.2f}mm  "
            f"IQR {np.subtract(*np.percentile(residual, [75, 25])) * 1000:>6.2f}mm"
        )
        report[f"confidence_{level}_iqr_mm"] = float(
            np.subtract(*np.percentile(residual, [75, 25])) * 1000
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
