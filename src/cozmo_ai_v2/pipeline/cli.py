from __future__ import annotations

import argparse
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import export
from .damage import fusion as damage_fusion
from .damage.masks import refine
from .damage.vlm import FURNITURE_CLASS, DamageAnalyzer
from .drift import measure_drift, refit_wall_offsets, sample_world_points_with_origin
from .frame_contract import build_frame_contract
from .fuse import fuse
from .geometry import estimate_gravity
from .ingest import display_rotation, load_capture, iter_frames
from .keyframes import select_damage_keyframes
from .occupancy import build_surface_grid, find_openings, occluded_spans
from .planes import (
    estimate_horizontal_frame,
    extract_walls,
    filter_occluded_walls,
    merge_collinear,
    resolve_crossings,
    snap_corners,
    snap_to_frame,
    wall_band_mask,
)
from .poses import refine_trajectory, select_keyframes
from .rooms import build_plan_grid, check_no_overlaps, segment_rooms
from .scope import ScopeEngine
from .uncertainty import UncertaintyModel

REPO_ROOT = Path(__file__).resolve().parents[3]


class Timings(dict):
    """Keeps track of how long each stage of a pipeline run took.

    This is just a regular dictionary (stage name -> seconds elapsed)
    with one extra method, `stage()`, that measures a block of code and
    stores the result automatically.
    """

    @contextmanager
    def stage(self, name: str, verbose: bool = True):
        """Times a block of code and stores how long it took under `name`.

        Used like `with timings.stage("ingest"): ...` -- everything
        inside the `with` block gets timed, and the elapsed seconds get
        saved under that name. While it runs, it also prints a short
        progress line to the console (if `verbose` is on), so a slow
        step doesn't look like the program has frozen.

        Args:
            name: What to call this stage when saving its elapsed time.
            verbose: If True, print a progress line while the block runs.
        """
        if verbose:
            print(f"  {name} ...", end="", flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        self[name] = round(elapsed, 2)
        if verbose:
            print(f" {elapsed:.1f}s")


def run(args: argparse.Namespace) -> int:
    """Runs the whole pipeline once, turning a raw capture into a finished
    result: a floor plan, a 3D model, a damage report, and a scope of work.

    This is the main function that everything else in this file supports.
    It moves through eleven stages, one after another, and each stage's
    output becomes the input to the next one:

    1. Ingest -- read the raw capture files.
    2. Pose refinement -- clean up the camera's tracked path.
    3. Fusion -- merge all the depth frames into one 3D point cloud.
    4. Sampling -- take a lighter-weight second pass over the frames,
       keeping track of where each point was seen from.
    5. Geometry -- figure out which way is up, and find the walls.
    6. Wall refinement -- clean up and finalize the wall positions.
    7. Rooms -- split the space into separate rooms.
    8. Surfaces -- work out where the doors and windows are.
    9. Damage -- look for water, fire, and mold damage.
    10. Scope -- turn any damage found into a list of repair line items.
    11. Export -- write out all the result files.

    See `docs/architecture.md` for more detail on each of these.

    Args:
        args: The parsed command-line arguments (flags like `--stride`,
            `--no-damage`, etc. all live on this object).

    Returns:
        0 if everything went well. 1 if the final result fails a schema
        check (a sign something is actually wrong with the output).
    """
    timings = Timings()
    warnings: list[str] = []
    out_dir = Path(args.out or f"out/{Path(args.capture).name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    # Stage 1: ingest
    print(f"Capture: {args.capture}")
    dense_depth_dir = getattr(args, "dense_depth_dir", None)
    densify_manifest = getattr(args, "densify_manifest", None)
    if dense_depth_dir is None and densify_manifest is not None:
        dense_depth_dir = Path(densify_manifest).parent / "dense_depth"
    with timings.stage("ingest"):
        bundle = load_capture(
            args.capture,
            pose_source=getattr(args, "pose_source", "auto"),
            slam_poses_path=getattr(args, "slam_poses", None),
            dense_depth_dir=dense_depth_dir,
        )
    print(
        f"  {len(bundle)} frames, {bundle.duration:.0f}s, "
        f"gravity consistency {bundle.gravity_consistency:.3f}"
    )
    if bundle.gravity_consistency < 0.9:
        warnings.append(
            f"IMU gravity consistency {bundle.gravity_consistency:.2f} is low; "
            "poses or the device mapping may be wrong for this capture source"
        )

    # Stage 2: pose refinement
    poses = bundle.poses
    drift_report = None
    if not args.no_refine and bundle.has_depth:
        with timings.stage("pose refinement"):
            keyframes = select_keyframes(bundle)
            poses, drift_report = refine_trajectory(
                bundle, keyframes, enable_loop_closure=not args.no_loop_closure
            )
        print(
            f"  {drift_report.keyframe_count} keyframes, "
            f"{drift_report.loop_edges}/{drift_report.loop_candidates} loop edges"
        )
    elif not args.no_refine:
        warnings.append(
            "pose refinement skipped: no raw LiDAR depth is available; "
            "using the selected SLAM/ARKit pose table"
        )

    # Stage 3: fusion.  The contract is the single source of truth for
    # depth resolution, QC masks, provenance, and the pose table used by
    # both TSDF integration and the later provenance-aware sampling pass.
    pose_provenance = (
        f"refined_{bundle.pose_source}" if not args.no_refine else bundle.pose_source
    )
    with timings.stage("frame contract"):
        frame_contract = build_frame_contract(
            bundle,
            indices=np.arange(0, len(bundle), args.stride),
            poses=poses,
            pose_source=pose_provenance,
            dense_depth_dir=dense_depth_dir,
            densify_manifest=densify_manifest,
            min_confidence=args.min_confidence,
            max_depth=args.max_depth,
            depth_source=getattr(args, "depth_source", "auto"),
            frame_association=getattr(args, "frame_association", "pts"),
            pts_tolerance_s=getattr(args, "pts_tolerance_s", None),
        )

    with timings.stage("fusion"):
        indices = np.arange(0, len(bundle), args.stride)
        reconstruction = fuse(
            bundle, indices, poses=poses, voxel_size=args.voxel,
            sdf_trunc=getattr(args, "sdf_trunc", None),
            min_confidence=args.min_confidence, max_depth=args.max_depth,
            frame_contract=frame_contract,
            depth_source=getattr(args, "depth_source", "auto"),
            frame_association=getattr(args, "frame_association", "pts"),
            pts_tolerance_s=getattr(args, "pts_tolerance_s", None),
        )
    cloud = reconstruction.cloud
    import open3d as o3d

    contract_report = reconstruction.contract_report or {}
    rejected_count = len(contract_report.get("rejected_frames", []))
    fallback_count = len(contract_report.get("fallback_frames", []))
    availability = contract_report.get("video_availability") or {}
    if rejected_count:
        warnings.append(f"{rejected_count} requested frame(s) rejected by the depth/pose contract")
    if fallback_count:
        print(f"  {fallback_count} frame(s) used raw LiDAR fallback after dense-depth QC")
    if availability.get("terminal_decode_missing"):
        decoded = availability.get("decoded_frame_count", 0)
        expected = availability.get("expected_frame_count", 0)
        reported = availability.get("reported_frame_count")
        reported_note = f", OpenCV reported {reported}" if reported is not None else ""
        decode_end = f"after index {decoded - 1}" if decoded else "before index 0"
        warnings.append(
            f"RGB video decode ended {decode_end}; "
            f"{len(availability.get('missing_indices', []))} terminal sidecar "
            f"frame(s) unavailable (expected {expected}{reported_note})"
        )

    if len(cloud.points):
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30)
        )
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    print(f"  {len(points)} points")

    # Stage 4: sampling
    with timings.stage("sampling"):
        sampled, origins, times = sample_world_points_with_origin(
            bundle, np.arange(0, len(bundle), max(args.stride, 3)), poses=poses,
            max_depth=args.max_depth, frame_contract=frame_contract,
        )

    # Stage 5: geometry
    with timings.stage("geometry"):
        gravity = estimate_gravity(points, hint=bundle.gravity_up, normals=normals)
        frame = estimate_horizontal_frame(normals, gravity.up)
        band = wall_band_mask(points, normals, gravity, gravity.up)
        walls = extract_walls(frame.to_plan(points[band]), frame.height(points[band]))
        walls = merge_collinear(snap_to_frame(walls, frame))
        walls, occluded_out = filter_occluded_walls(walls, frame, sampled, origins)
    ceiling = gravity.ceiling_height
    print(
        f"  {len(walls)} walls ({occluded_out} occlusion-inconsistent removed), "
        f"room height {gravity.room_height:.3f} m" if gravity.room_height
        else "  no ceiling found"
    )
    if gravity.room_height is None:
        warnings.append("no ceiling plane found; heights are unavailable")

    # Stage 6: wall refinement
    with timings.stage("wall refinement"):
        refitted = refit_wall_offsets(walls, frame, sampled, times)
        drift = measure_drift(walls, frame, sampled, times)
        walls = resolve_crossings(walls)
        corners = snap_corners(walls)
    print(
        f"  {refitted} offsets refitted by visit, {len(walls)} walls after "
        f"crossing resolution, {corners} endpoints snapped to corners"
    )
    print(f"  drift: {drift.summary()}")

    # Stage 7: rooms
    with timings.stage("rooms"):
        grid = build_plan_grid(
            points, frame, gravity.floor_height, ceiling,
            trajectory=poses[:, :3, 3],
        )
        rooms = segment_rooms(grid, walls, frame, gravity.floor_height, ceiling)
        overlaps = check_no_overlaps(rooms)
        if overlaps:
            warnings.append(
                f"{len(overlaps)} room-pair(s) overlap beyond tolerance: "
                + ", ".join(
                    f"room_{a}/room_{b} {frac:.1%}" for a, b, frac in overlaps
                )
            )
    print(f"  {len(rooms)} rooms")

    # Stage 8: surfaces.  Surface dimensions are metric geometry, so do not
    # fabricate a wall grid height when the ceiling was not observed.
    with timings.stage("surfaces"):
        surface_grids = {}
        openings = []
        if gravity.ceiling_observed and ceiling is not None:
            for wall in walls:
                if wall.length < 0.6:
                    continue
                surface = build_surface_grid(
                    wall, frame, points, gravity.floor_height, ceiling
                )
                surface_grids[wall.index] = surface
                openings.extend(find_openings(surface))
        else:
            warnings.append("ceiling not observed; wall opening heights are unavailable")
    print(f"  {len(openings)} openings")

    # Stage 9: damage
    regions = []
    overlay_paths = []
    if not args.no_damage:
        with timings.stage("damage"):
            regions, overlay_paths, furniture_count, furniture_overlay_paths = (
                _damage_pass(
                    bundle, poses, frame, walls, gravity, ceiling, surface_grids,
                    out_dir, args, warnings,
                )
            )
        print(f"  {len(regions)} fused damage regions")
        if overlay_paths:
            print(
                f"  {len(overlay_paths)} damage overlay images written to "
                f"{out_dir / 'damage_overlays'}"
            )
        if args.debug_furniture:
            print(
                f"  [debug-furniture] {furniture_count} furniture detections -- "
                "diagnostic only, not written to result.json"
            )
            if furniture_overlay_paths:
                print(
                    f"  [debug-furniture] {len(furniture_overlay_paths)} overlay "
                    f"images written to {out_dir / 'furniture_debug_overlays'}"
                )
            elif furniture_count:
                print(
                    "  [debug-furniture] pass --furniture-overlays to also save "
                    "annotated images"
                )

    # Stage 10: scope
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

    # Stage 11: export
    with timings.stage("export"):
        result = _assemble(
            bundle, gravity, frame, walls, openings, rooms, regions, concealed,
            line_items, drift, drift_report, surface_grids, uncertainty,
            timings, warnings, engine, reconstruction.contract_report,
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
            result["reconstruction"]["walls"],
            out_dir / "scene.glb", gravity.floor_height, ceiling,
            rooms=result["rooms"],
        )
        export.export_scope_csv(result, out_dir)
        export.export_reconstruction(reconstruction, out_dir)

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
) -> tuple[list, list, int, list]:
    """Looks for damage in a set of frames, and combines what it finds
    into regions on the actual walls.

    For each selected frame, this asks a vision-language model whether it
    sees any damage, turns any detections into pixel masks, projects
    those masks into the 3D scene, and adds them as "votes" onto the
    wall they landed on. A patch of damage only ends up in the final
    result once enough separate frames agree it's really there -- one
    frame's guess on its own isn't trusted.

    Args:
        bundle: The parsed capture.
        poses: Camera poses to use for this run.
        frame: The building's horizontal frame.
        walls: The final wall segments from the geometry stage.
        gravity: Floor and ceiling heights.
        ceiling: Ceiling height, or `None` if one wasn't found.
        surface_grids: Each wall's surface grid, keyed by wall index --
            reused here so damage detection lines up with where doors,
            windows, and hidden spans were already found.
        out_dir: Where to write the overlay images that show what was
            detected.
        args: The parsed command-line arguments.
        warnings: A list this function can add warning messages to.

    Returns:
        A tuple of (the damage regions found, paths to the overlay
        images written, how many furniture items were spotted in
        debug mode, and paths to any furniture overlay images).
    """
    # This function is the only place that writes to these two folders,
    # so clear them out first -- otherwise a rerun that finds nothing
    # could leave old images sitting next to a result that says no
    # damage was found.
    for name in ("damage_overlays", "furniture_debug_overlays"):
        shutil.rmtree(out_dir / name, ignore_errors=True)

    selected = select_damage_keyframes(bundle, poses, max_frames=args.damage_frames)
    if not selected:
        return [], [], 0, []

    analyzer = DamageAnalyzer(
        model=args.model, cache_dir=args.cache_dir,
        include_furniture=args.debug_furniture,
    )
    surfaces = damage_fusion.build_surface_refs(
        walls, frame, gravity.floor_height, ceiling
    )
    accumulators = {}
    for index, surface in enumerate(surfaces):
        # Only walls get a place to collect damage votes here -- the
        # floor and ceiling are included in `surfaces`, but there's no
        # matching entry for them, so any damage detected there
        # currently has nowhere to go and is quietly dropped.
        if surface.kind == "wall" and surface.wall.index in surface_grids:
            accumulators[index] = damage_fusion.DamageAccumulator(
                surface, surface_grids[surface.wall.index]
            )

    errors = 0
    low_confidence = 0
    overlay_frames = []
    detections_by_frame = {}
    masks_by_frame = {}
    furniture_frames = []
    furniture_detections_by_frame = {}
    furniture_masks_by_frame = {}
    furniture_count = 0
    rotations_by_frame = {}
    for capture_frame in iter_frames(
        bundle, selected, min_confidence=1, include_full_res=True
    ):
        rotation = display_rotation(poses[capture_frame.index], bundle.gravity_up)
        rotations_by_frame[capture_frame.index] = rotation
        # Send the model the full, high-resolution image (not the smaller
        # depth-camera resolution) since that's what makes it able to see
        # detail clearly. `target_shape` tells it to hand back detection
        # boxes rescaled to the smaller grid that the rest of this
        # function works with.
        analysis = analyzer.analyze_frame(
            capture_frame.index, capture_frame.color_full, rotation=rotation,
            target_shape=capture_frame.color.shape[:2],
        )
        if analysis.error:
            errors += 1
            continue
        detections = [
            d for d in analysis.detections
            if d.confidence >= args.min_detection_confidence
        ]
        low_confidence += len(analysis.detections) - len(detections)
        if not detections:
            continue

        masks = refine(
            capture_frame.color,
            [d.bbox for d in detections],
            cache_dir=Path(args.cache_dir).parent / "masks",
            prefer_sam=not args.no_sam,
        )
        # Furniture detections are only for sanity-checking that the
        # model can actually recognize objects -- keep them completely
        # separate from real damage so they never end up in the result.
        damage_pairs = [
            (d, m) for d, m in zip(detections, masks)
            if d.damage_class != FURNITURE_CLASS
        ]
        furniture_pairs = [
            (d, m) for d, m in zip(detections, masks)
            if d.damage_class == FURNITURE_CLASS
        ]

        if damage_pairs:
            overlay_frames.append(capture_frame)
            detections_by_frame[capture_frame.index] = [d for d, _ in damage_pairs]
            masks_by_frame[capture_frame.index] = [m for _, m in damage_pairs]
        if furniture_pairs:
            furniture_count += len(furniture_pairs)
            furniture_frames.append(capture_frame)
            furniture_detections_by_frame[capture_frame.index] = [
                d for d, _ in furniture_pairs
            ]
            furniture_masks_by_frame[capture_frame.index] = [
                m for _, m in furniture_pairs
            ]

        for detection, mask in damage_pairs:
            world, rays = damage_fusion.project_detection(
                detection, mask.mask, capture_frame.depth,
                poses[capture_frame.index], bundle.intrinsics,
                (1.0, 1.0),
            )
            if len(world) == 0:
                continue
            # Points that don't clearly belong to any surface (like a
            # reflection in a mirror, which projects to the wrong depth)
            # just don't show up here at all -- they're not forced onto
            # the nearest wall.
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
    if low_confidence:
        print(
            f"  {low_confidence} detections dropped below "
            f"--min-detection-confidence {args.min_detection_confidence}"
        )

    regions = []
    for accumulator in accumulators.values():
        regions.extend(
            damage_fusion.extract_regions(accumulator, min_views=args.min_views)
        )

    overlay_paths = []
    if overlay_frames:
        overlay_paths = export.render_damage_overlays(
            overlay_frames, detections_by_frame, masks_by_frame,
            out_dir / "damage_overlays", rotations=rotations_by_frame,
        )
    furniture_overlay_paths = []
    if furniture_frames and args.furniture_overlays:
        furniture_overlay_paths = export.render_damage_overlays(
            furniture_frames, furniture_detections_by_frame, furniture_masks_by_frame,
            out_dir / "furniture_debug_overlays", rotations=rotations_by_frame,
        )
    return regions, overlay_paths, furniture_count, furniture_overlay_paths


def _to_cells(accumulator, surface, world, frame) -> np.ndarray:
    """Works out which grid cell on a wall each 3D point falls into.

    Every wall has its own small grid laid over its surface, used to
    count up how many separate views agree that a spot is damaged. This
    takes a batch of 3D points already known to belong to one wall, and
    converts each one into a (column, row) position on that wall's grid.

    Args:
        accumulator: The `DamageAccumulator` whose grid to use.
        surface: The wall surface these points belong to.
        world: The 3D points, already known to belong to this wall.
        frame: The building's horizontal frame.

    Returns:
        An `(K, 2)` array of `(column, row)` cell positions, one for each
        point that actually lands inside the wall's grid. A point that
        falls outside the grid is simply left out of the result.
    """
    grid = accumulator.grid
    plan = frame.to_plan(world)
    wall = surface.wall
    u = (plan - wall.start) @ wall.direction
    v = frame.height(world) - grid.base_height
    columns = np.clip((u / grid.resolution).astype(int), 0, grid.shape[0] - 1)
    rows = np.clip((v / grid.resolution).astype(int), 0, grid.shape[1] - 1)
    inside = (u >= 0) & (u < grid.width) & (v >= 0) & (v < grid.height)
    return np.stack([columns[inside], rows[inside]], axis=1)


def _infer_adjacency_via(
    room_a, room_b, walls, openings, wall_names, max_distance: float = 1.2
) -> str | None:
    """Tries to find which door connects two neighbouring rooms.

    Two rooms can be marked as neighbours just from their shapes, without
    knowing which door actually connects them. This looks at every
    detected door or pass-through opening and picks whichever one sits
    closest to the midpoint between the two rooms -- as long as it's
    close enough (within `max_distance`) to plausibly be the right one.
    Not every case has a clear answer, so it's fine for this to come back
    empty rather than guess.

    Args:
        room_a: One of the two rooms.
        room_b: The other room.
        walls: All wall segments.
        openings: All detected openings.
        wall_names: A lookup from wall index to its display name.
        max_distance: How far away, in metres, an opening can still be
            and count as a match.

    Returns:
        The connecting wall's name, or `None` if nothing was close enough.
    """
    walls_by_index = {wall.index: wall for wall in walls}
    boundary_midpoint = 0.5 * (room_a.centroid + room_b.centroid)
    best_name = None
    best_distance = max_distance
    for opening in openings:
        if opening.kind not in ("door", "pass-through"):
            continue
        wall = walls_by_index.get(opening.wall_index)
        if wall is None or wall.room_id not in (room_a.id, room_b.id):
            continue
        distance = float(np.linalg.norm(wall.midpoint - boundary_midpoint))
        if distance < best_distance:
            best_distance = distance
            best_name = wall_names.get(opening.wall_index)
    return best_name


def _assemble(
    bundle, gravity, frame, walls, openings, rooms, regions, concealed,
    line_items, drift, drift_report, surface_grids, uncertainty,
    timings, warnings, engine, fusion_report=None,
) -> dict:
    """Puts everything the pipeline produced together into one dictionary,
    ready to be saved as `result.json`.

    This function doesn't calculate anything new -- it just takes all the
    pieces that earlier stages already worked out (walls, rooms, damage,
    scope items, and so on) and arranges them into the exact shape the
    output file is supposed to have, adding a confidence interval to
    every measurement along the way.

    Args:
        bundle: The parsed capture.
        gravity: Floor/ceiling heights and the up axis.
        frame: The building's horizontal frame.
        walls: The final wall segments.
        openings: The detected doors and windows.
        rooms: The segmented rooms.
        regions: The fused damage regions.
        concealed: The concealed-damage flags.
        line_items: The scope-of-work line items.
        drift: The drift measurement.
        drift_report: The pose-refinement report, or `None` if pose
            refinement was skipped.
        surface_grids: Each wall's surface grid.
        uncertainty: The uncertainty model used to build confidence
            intervals for this run.
        timings: How long each stage took.
        warnings: Any warnings collected while running.
        engine: The scope engine that was used.

    Returns:
        The full result, shaped exactly as `result.json` expects it.
    """
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
                "height": (
                    uncertainty.ceiling_height(
                        gravity.room_height, wall.residual_rms, wall.residual_rms,
                        wall.inlier_count, wall.inlier_count,
                    ).to_dict()
                    if gravity.ceiling_observed and gravity.room_height is not None
                    else None
                ),
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

    rooms_by_id = {room.id: room for room in rooms}
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
                "ceiling_height": (
                    uncertainty.ceiling_height(
                        room.height, 0.01, 0.01, 5000, 5000
                    ).to_dict()
                    if room.height is not None and gravity.ceiling_observed
                    else None
                ),
                "perimeter": round(room.perimeter, 3),
                "centroid": room.centroid.tolist(),
                "polygon": room.polygon.tolist() if room.polygon is not None else [],
                "wall_ids": room.wall_indices,
                "neighbours": room.neighbours,
            }
        )
        for neighbour in room.neighbours:
            if neighbour > room.id:
                via = _infer_adjacency_via(
                    room, rooms_by_id[neighbour], walls, openings, wall_names
                )
                adjacency.append({"a": room.id, "b": neighbour, "via": via})

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
            "ceiling_observed": bool(gravity.ceiling_observed),
            "ceiling_confidence": round(float(gravity.ceiling_confidence), 4),
            "floor_confidence": round(float(gravity.floor_confidence), 4),
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
            "fusion": fusion_report or {},
            "warnings": warnings,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """The command-line entry point: reads the arguments the user typed
    and runs the pipeline with them.

    This is what actually gets called when you run
    `python -m pipeline run <capture>`. It also loads the `.env` file
    first, which is what makes the Anthropic API key available for the
    damage-detection stage.

    Args:
        argv: The arguments to parse, or `None` to read them from the
            command line as usual.

    Returns:
        The exit code from `run()`.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

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
    runner.add_argument(
        "--sdf-trunc", type=float, default=None,
        help="TSDF truncation distance in metres (defaults to 4 * --voxel)",
    )
    runner.add_argument("--max-depth", type=float, default=3.5)
    runner.add_argument("--min-confidence", type=int, default=1)
    runner.add_argument(
        "--depth-source", choices=("auto", "dense", "raw"), default="auto",
        help="use QC-approved dense depth, raw LiDAR, or auto dense-then-raw fallback",
    )
    runner.add_argument(
        "--frame-association", choices=("pts", "index"), default="pts",
        help="associate decoded RGB frames to sidecars by PTS (default) or identity index",
    )
    runner.add_argument(
        "--pts-tolerance-s", type=float, default=None,
        help="maximum sidecar/video timestamp distance for PTS association; defaults from sidecar cadence",
    )
    runner.add_argument(
        "--dense-depth-dir", type=Path, default=None,
        help="Stage 4 dense_depth directory (must have a QC-approved manifest); "
             "defaults to <capture>/dense_depth when present",
    )
    runner.add_argument(
        "--densify-manifest", type=Path, default=None,
        help="Stage 4 densify_manifest.json; defaults beside --dense-depth-dir",
    )
    runner.add_argument(
        "--pose-source", choices=("auto", "arkit", "slam"), default="auto",
        help="select ARKit odometry for LiDAR captures or SLAM poses for video inputs",
    )
    runner.add_argument(
        "--slam-poses", type=Path, default=None,
        help="SLAM pose table (CSV/NPY/NPZ/JSON) for a video input",
    )
    runner.add_argument("--damage-frames", type=int, default=40)
    runner.add_argument("--min-views", type=int, default=2)
    runner.add_argument(
        "--min-detection-confidence", type=float, default=0.0,
        help="drop VLM detections (any class, including furniture) below "
             "this confidence before masking/fusion; e.g. 0.6",
    )
    runner.add_argument("--coverage", type=float, default=0.90)
    runner.add_argument("--no-refine", action="store_true", help="use the selected raw SLAM/ARKit poses")
    runner.add_argument("--no-loop-closure", action="store_true")
    runner.add_argument("--no-damage", action="store_true")
    runner.add_argument("--no-sam", action="store_true", help="use local GrabCut masks")
    runner.add_argument(
        "--debug-furniture", action="store_true",
        help="diagnostic: also ask the VLM to tag furniture, to confirm it is "
             "resolving objects in the frame; never enters result.json/scope",
    )
    runner.add_argument(
        "--furniture-overlays", action="store_true",
        help="with --debug-furniture, also write annotated furniture_debug_overlays/ "
             "images (off by default -- --debug-furniture alone only prints counts)",
    )
    runner.set_defaults(func=run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
