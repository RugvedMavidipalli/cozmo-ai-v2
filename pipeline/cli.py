"""One command per capture: `python -m pipeline run <capture_dir>`."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import export
from .damage import fusion as damage_fusion
from .damage.masks import refine
from .damage.vlm import DamageAnalyzer
from .drift import measure_drift, sample_world_points
from .fuse import fuse
from .geometry import estimate_gravity
from .ingest import load_capture, iter_frames
from .keyframes import select_damage_keyframes
from .occupancy import build_surface_grid, find_openings, occluded_spans
from .planes import (
    estimate_horizontal_frame,
    extract_walls,
    merge_collinear,
    snap_to_frame,
    wall_band_mask,
)
from .poses import refine_trajectory, select_keyframes
from .rooms import build_plan_grid, segment_rooms
from .scope import ScopeEngine
from .uncertainty import UncertaintyModel

REPO_ROOT = Path(__file__).resolve().parent.parent


class Timings(dict):
    @contextmanager
    def stage(self, name: str, verbose: bool = True):
        if verbose:
            print(f"  {name} ...", end="", flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        self[name] = round(elapsed, 2)
        if verbose:
            print(f" {elapsed:.1f}s")


def run(args: argparse.Namespace) -> int:
    timings = Timings()
    warnings: list[str] = []
    out_dir = Path(args.out or f"out/{Path(args.capture).name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    print(f"Capture: {args.capture}")
    with timings.stage("ingest"):
        bundle = load_capture(args.capture)
    print(
        f"  {len(bundle)} frames, {bundle.duration:.0f}s, "
        f"gravity consistency {bundle.gravity_consistency:.3f}"
    )
    if bundle.gravity_consistency < 0.9:
        warnings.append(
            f"IMU gravity consistency {bundle.gravity_consistency:.2f} is low; "
            "poses or the device mapping may be wrong for this capture source"
        )

    poses = bundle.poses
    drift_report = None
    if not args.no_refine:
        with timings.stage("pose refinement"):
            keyframes = select_keyframes(bundle)
            poses, drift_report = refine_trajectory(
                bundle, keyframes, enable_loop_closure=not args.no_loop_closure
            )
        print(
            f"  {drift_report.keyframe_count} keyframes, "
            f"{drift_report.loop_edges}/{drift_report.loop_candidates} loop edges"
        )

    with timings.stage("fusion"):
        indices = np.arange(0, len(bundle), args.stride)
        reconstruction = fuse(
            bundle, indices, poses=poses, voxel_size=args.voxel,
            min_confidence=args.min_confidence, max_depth=args.max_depth,
        )
    cloud = reconstruction.cloud
    import open3d as o3d

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30)
    )
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    print(f"  {len(points)} points")

    with timings.stage("geometry"):
        gravity = estimate_gravity(points, hint=bundle.gravity_up, normals=normals)
        frame = estimate_horizontal_frame(normals, gravity.up)
        band = wall_band_mask(points, normals, gravity, gravity.up)
        walls = extract_walls(frame.to_plan(points[band]), frame.height(points[band]))
        walls = merge_collinear(snap_to_frame(walls, frame))
    ceiling = gravity.ceiling_height
    print(
        f"  {len(walls)} walls, room height "
        f"{gravity.room_height:.3f} m" if gravity.room_height else "  no ceiling found"
    )
    if gravity.room_height is None:
        warnings.append("no ceiling plane found; heights are unavailable")

    with timings.stage("rooms"):
        grid = build_plan_grid(
            points, frame, gravity.floor_height, ceiling,
            trajectory=poses[:, :3, 3],
        )
        rooms = segment_rooms(grid, walls, frame, gravity.floor_height, ceiling)
    print(f"  {len(rooms)} rooms")

    with timings.stage("surfaces"):
        surface_grids = {}
        openings = []
        for wall in walls:
            if wall.length < 0.6:
                continue
            surface = build_surface_grid(
                wall, frame, points, gravity.floor_height,
                ceiling if ceiling is not None else gravity.floor_height + 2.4,
            )
            surface_grids[wall.index] = surface
            openings.extend(find_openings(surface))
    print(f"  {len(openings)} openings")

    with timings.stage("drift"):
        sampled, times = sample_world_points(
            bundle, np.arange(0, len(bundle), max(args.stride, 3)), poses=poses
        )
        drift = measure_drift(walls, frame, sampled, times)
    print(f"  {drift.summary()}")

    regions = []
    if not args.no_damage:
        with timings.stage("damage"):
            regions = _damage_pass(
                bundle, poses, frame, walls, gravity, ceiling, surface_grids,
                out_dir, args, warnings,
            )
        print(f"  {len(regions)} fused damage regions")

    with timings.stage("scope"):
        engine = ScopeEngine(args.rules)
        wall_lengths = {
            (wall.name or f"wall_{wall.index}"): wall.length for wall in walls
        }
        line_items, concealed = engine.build(regions, rooms, wall_lengths)
    print(f"  {len(line_items)} line items, {len(concealed)} concealed flags")

    uncertainty = UncertaintyModel(
        coverage=args.coverage,
        calibration_path=args.calibration,
        has_depth=bundle.has_depth,
    )
    if not uncertainty.calibrated:
        warnings.append(
            "confidence intervals are uncalibrated: no ground-truth fit was supplied"
        )

    with timings.stage("export"):
        result = _assemble(
            bundle, gravity, frame, walls, openings, rooms, regions, concealed,
            line_items, drift, drift_report, surface_grids, uncertainty,
            timings, warnings, engine,
        )
        export.write_json(result, out_dir / "result.json")
        problems = export.validate(result, REPO_ROOT / "schema" / "result.schema.json")
        if problems:
            warnings.extend(f"schema: {p}" for p in problems[:5])
        try:
            export.render_floorplan(result, out_dir / "floorplan.svg")
        except Exception as exc:
            warnings.append(f"floor plan render failed: {exc}")
        export.export_scene(
            reconstruction.mesh, result["reconstruction"]["walls"],
            out_dir / "scene.glb", gravity.floor_height, ceiling,
        )
        o3d.io.write_point_cloud(str(out_dir / "cloud.ply"), cloud)

    result["diagnostics"]["timings_s"]["total"] = round(time.time() - total_start, 2)
    export.write_json(result, out_dir / "result.json")

    print(f"\nWrote {out_dir}/ in {time.time() - total_start:.1f}s")
    for warning in warnings:
        print(f"  ! {warning}")
    if problems:
        print(f"  ! result.json has {len(problems)} schema problems")
        return 1
    return 0


def _damage_pass(
    bundle, poses, frame, walls, gravity, ceiling, surface_grids,
    out_dir, args, warnings,
) -> list:
    """Detect, mask, project, and fuse damage across keyframes."""
    selected = select_damage_keyframes(bundle, poses, max_frames=args.damage_frames)
    if not selected:
        return []

    analyzer = DamageAnalyzer(model=args.model, cache_dir=args.cache_dir)
    surfaces = damage_fusion.build_surface_refs(
        walls, frame, gravity.floor_height, ceiling
    )
    accumulators = {}
    for index, surface in enumerate(surfaces):
        if surface.kind == "wall" and surface.wall.index in surface_grids:
            accumulators[index] = damage_fusion.DamageAccumulator(
                surface, surface_grids[surface.wall.index]
            )

    errors = 0
    for capture_frame in iter_frames(bundle, selected, min_confidence=1):
        analysis = analyzer.analyze_frame(capture_frame.index, capture_frame.color)
        if analysis.error:
            errors += 1
            continue
        if not analysis.detections:
            continue

        masks = refine(
            capture_frame.color,
            [d.bbox for d in analysis.detections],
            cache_dir=Path(args.cache_dir).parent / "masks",
            prefer_sam=not args.no_sam,
        )
        for detection, mask in zip(analysis.detections, masks):
            world, rays = damage_fusion.project_detection(
                detection, mask.mask, capture_frame.depth,
                poses[capture_frame.index], bundle.intrinsics,
                (1.0, 1.0),
            )
            if len(world) == 0:
                continue
            assignment = damage_fusion.assign_to_surfaces(world, rays, surfaces, frame)
            for surface_index, point_indices in assignment.items():
                accumulator = accumulators.get(surface_index)
                if accumulator is None:
                    continue
                weight = damage_fusion.incidence_weight(
                    rays[point_indices], surfaces[surface_index].normal
                )
                if weight <= 0:
                    continue
                cells = _to_cells(
                    accumulator, surfaces[surface_index], world[point_indices], frame
                )
                accumulator.add(cells, detection, weight, mask.method)

    if errors:
        warnings.append(f"{errors} damage frames failed analysis (see cache/API state)")

    regions = []
    for accumulator in accumulators.values():
        regions.extend(
            damage_fusion.extract_regions(accumulator, min_views=args.min_views)
        )
    return regions


def _to_cells(accumulator, surface, world, frame) -> np.ndarray:
    grid = accumulator.grid
    plan = frame.to_plan(world)
    wall = surface.wall
    u = (plan - wall.start) @ wall.direction
    v = frame.height(world) - grid.base_height
    columns = np.clip((u / grid.resolution).astype(int), 0, grid.shape[0] - 1)
    rows = np.clip((v / grid.resolution).astype(int), 0, grid.shape[1] - 1)
    inside = (u >= 0) & (u < grid.width) & (v >= 0) & (v < grid.height)
    return np.stack([columns[inside], rows[inside]], axis=1)


def _assemble(
    bundle, gravity, frame, walls, openings, rooms, regions, concealed,
    line_items, drift, drift_report, surface_grids, uncertainty,
    timings, warnings, engine,
) -> dict:
    """Build the schema-shaped result document."""
    drift_by_wall = {v.wall_index: v.std for v in drift.per_wall}
    wall_docs = []
    for wall in walls:
        if wall.length < 0.5:
            continue
        spans = (
            occluded_spans(surface_grids[wall.index])
            if wall.index in surface_grids
            else []
        )
        length = uncertainty.wall_length(
            wall.length, wall.residual_rms, wall.inlier_count,
            drift=drift_by_wall.get(wall.index, drift.median_spread),
            inferred_fraction=wall.inferred_fraction,
        )
        wall_docs.append(
            {
                "id": wall.index,
                "name": wall.name or f"wall_{wall.index}",
                "room_id": wall.room_id,
                "start": wall.start.tolist(),
                "end": wall.end.tolist(),
                "normal": wall.normal.tolist(),
                "length": length.to_dict(),
                "height": uncertainty.ceiling_height(
                    (gravity.room_height or 0.0), wall.residual_rms, wall.residual_rms,
                    wall.inlier_count, wall.inlier_count,
                ).to_dict(),
                "inferred_fraction": round(wall.inferred_fraction, 3),
                "occluded_spans": [list(s) for s in spans],
                "residual_rms_mm": round(wall.residual_rms * 1000, 2),
                "support_points": wall.inlier_count,
                "tags": wall.tags,
            }
        )

    wall_names = {wall.index: (wall.name or f"wall_{wall.index}") for wall in walls}
    opening_docs = []
    for opening in openings:
        grid = surface_grids.get(opening.wall_index)
        resolution = grid.resolution if grid else 0.04
        opening_docs.append(
            {
                "wall": wall_names.get(opening.wall_index, f"wall_{opening.wall_index}"),
                "kind": opening.kind,
                "width": uncertainty.opening_width(
                    opening.width, resolution, opening.confidence
                ).to_dict(),
                "height": uncertainty.opening_width(
                    opening.height, resolution, opening.confidence
                ).to_dict(),
                "sill_height": round(opening.sill_height, 3),
                "header_height": round(opening.header_height, 3),
                "u_offset": round(opening.u_range[0], 3),
                "confidence": round(opening.confidence, 3),
            }
        )

    room_docs = []
    adjacency = []
    for room in rooms:
        room_docs.append(
            {
                "id": room.id,
                "name": room.name,
                "area": uncertainty.floor_area(
                    room.area, room.perimeter, drift.median_spread or 0.02
                ).to_dict(),
                "ceiling_height": uncertainty.ceiling_height(
                    room.height or 0.0, 0.01, 0.01, 5000, 5000
                ).to_dict(),
                "perimeter": round(room.perimeter, 3),
                "centroid": room.centroid.tolist(),
                "polygon": room.polygon.tolist() if room.polygon is not None else [],
                "wall_ids": room.wall_indices,
                "neighbours": room.neighbours,
            }
        )
        for neighbour in room.neighbours:
            if neighbour > room.id:
                adjacency.append({"a": room.id, "b": neighbour, "via": None})

    damage_docs = []
    for region in regions:
        damage_docs.append(
            {
                "id": region.id,
                "surface_ref": region.surface_key,
                "room_id": region.room_id,
                "damage_class": region.damage_class,
                "subtype": region.subtype,
                "area": uncertainty.damage_area(
                    region.area, region.view_count, region.mask_method != "box"
                ).to_dict(),
                "water_category": region.water_category,
                "water_class": region.water_class,
                "mold_condition": region.mold_condition,
                "classification_basis": " | ".join(region.evidence[:2]),
                "extent": {
                    "u_range": list(region.bounds_u),
                    "v_range": list(region.bounds_v),
                },
                "description": region.describe(),
                "view_count": region.view_count,
                "confidence": round(region.confidence, 3),
                "contributing_frames": region.contributing_frames,
                "mask_method": region.mask_method,
            }
        )

    return {
        "capture": {
            "name": bundle.name,
            "modality": "lidar" if bundle.has_depth else "video_only",
            "frame_count": len(bundle),
            "duration_s": round(bundle.duration, 2),
            "path_length_m": round(
                float(
                    np.linalg.norm(np.diff(bundle.poses[:, :3, 3], axis=0), axis=1).sum()
                ),
                2,
            ),
        },
        "reconstruction": {
            "up_axis": gravity.up.tolist(),
            "gravity_consistency": round(bundle.gravity_consistency, 4),
            "floor_height": round(gravity.floor_height, 4),
            "ceiling_height": (
                round(gravity.ceiling_height, 4)
                if gravity.ceiling_height is not None
                else None
            ),
            "manhattan_yaw_deg": round(float(np.degrees(frame.yaw)), 3),
            "manhattan_fraction": round(frame.manhattan_fraction, 4),
            "walls": wall_docs,
            "openings": opening_docs,
        },
        "rooms": room_docs,
        "adjacency": adjacency,
        "damage": damage_docs,
        "concealed": [flag.to_dict() for flag in concealed],
        "scope": {
            "line_items": [item.to_dict() for item in line_items],
            "rules_version": engine.rules.get("version"),
        },
        "diagnostics": {
            "timings_s": dict(timings),
            "drift": {
                "median_mm": round(drift.median_spread * 1000, 2),
                "p90_mm": round(drift.p90_spread * 1000, 2),
                "max_mm": round(drift.max_spread * 1000, 2),
                "revisited_walls": drift.revisited_walls,
            },
            "trajectory": drift_report.__dict__ if drift_report else None,
            "calibration": {
                "calibrated": uncertainty.calibrated,
                "scale": uncertainty.scale,
                "coverage_target": uncertainty.coverage,
            },
            "warnings": warnings,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run", help="process one capture end to end")
    runner.add_argument("capture", help="capture directory")
    runner.add_argument("--out", help="output directory (default out/<capture>)")
    runner.add_argument("--rules", default=str(REPO_ROOT / "rules.yaml"))
    runner.add_argument("--cache-dir", default="cache/vlm")
    runner.add_argument("--calibration", default=str(REPO_ROOT / "bench" / "calibration.json"))
    runner.add_argument("--model", default="claude-opus-5")
    runner.add_argument("--stride", type=int, default=4)
    runner.add_argument("--voxel", type=float, default=0.02)
    # 3.5 m is the knee measured by tools/depth_bias.py: ARKit depth is well
    # behaved below it and reads systematically far above it (+11.6 mm at
    # 5.4 m).  tools/gating_sweep.py confirms the trade is nearly free --
    # 8.2% less drift for 0.7% less wall coverage.
    runner.add_argument("--max-depth", type=float, default=3.5)
    runner.add_argument("--min-confidence", type=int, default=1)
    runner.add_argument("--damage-frames", type=int, default=40)
    runner.add_argument("--min-views", type=int, default=2)
    runner.add_argument("--coverage", type=float, default=0.90)
    runner.add_argument("--no-refine", action="store_true", help="use raw ARKit poses")
    runner.add_argument("--no-loop-closure", action="store_true")
    runner.add_argument("--no-damage", action="store_true")
    runner.add_argument("--no-sam", action="store_true", help="use local GrabCut masks")
    runner.set_defaults(func=run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
