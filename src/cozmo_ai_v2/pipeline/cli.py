from __future__ import annotations

import argparse
from collections import Counter
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
from .fuse import fuse
from .geometry import estimate_gravity
from .ingest import display_rotation, load_capture, iter_frames
from .keyframes import select_damage_keyframes
from .occupancy import (
    build_surface_grid,
    deduplicate_openings,
    find_openings,
    occluded_spans,
)
from .planes import (
    estimate_horizontal_frame,
    extract_walls,
    filter_occluded_walls,
    merge_collinear,
    snap_to_frame,
    wall_band_mask,
)
from .poses import refine_trajectory, select_keyframes
from .rooms import (
    build_plan_grid,
    check_no_overlaps,
    polygonize_wall_graph,
    segment_rooms,
)
from .roomformer import RoomFormerAdapter
from .scope import ScopeEngine
from .uncertainty import UncertaintyModel
from .vectorizer import (
    AdjacencyEvidence,
    FaceEvidence,
    OpeningEvidence,
    build_vectorizer_input,
    build_vectorizer_output,
)
from .wall_graph import solve_wall_graph

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

    # Stage 2: pose refinement
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

    # Stage 3: fusion
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

    # Stage 4: sampling
    with timings.stage("sampling"):
        sampled, origins, times = sample_world_points_with_origin(
            bundle, np.arange(0, len(bundle), max(args.stride, 3)), poses=poses,
            max_depth=args.max_depth,
        )

    # Stage 5: geometry
    wall_stage_counts = {}
    wall_drop_counts = {}
    with timings.stage("geometry"):
        gravity = estimate_gravity(points, hint=bundle.gravity_up, normals=normals)
        frame = estimate_horizontal_frame(normals, gravity.up)
        band = wall_band_mask(points, normals, gravity, gravity.up)
        walls = extract_walls(frame.to_plan(points[band]), frame.height(points[band]))
        wall_stage_counts["raw"] = len(walls)
        walls = merge_collinear(snap_to_frame(walls, frame))
        wall_stage_counts["merged"] = len(walls)
        walls, occluded_out = filter_occluded_walls(walls, frame, sampled, origins)
        wall_stage_counts["occlusion"] = len(walls)
        wall_drop_counts["occlusion"] = occluded_out
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
        wall_stage_counts["graph_input"] = len(walls)
        wall_graph = solve_wall_graph(walls)
        walls = list(wall_graph.walls)
        wall_stage_counts["crossing"] = len(wall_graph.candidates)
        wall_stage_counts["graph_output"] = len(walls)
        wall_stage_counts["persisted"] = len(walls)
        corners = wall_graph.snapped_endpoint_count
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
        vectorizer_input = build_vectorizer_input(
            grid.wall_density,
            wall_graph.candidates,
            wall_graph.nodes,
        )
        # RoomFormer is an optional proposal source at the vectorizer
        # boundary.  The default adapter has no checkpoint configured, so it
        # records a deterministic fallback without importing a model or
        # running GPU inference; an application can inject a local adapter
        # implementation without changing the result contract.
        roomformer_proposal = RoomFormerAdapter().propose(vectorizer_input)
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
        openings = deduplicate_openings(openings)
    print(f"  {len(openings)} openings")

    geometry_diagnostics = _build_geometry_diagnostics(
        wall_stage_counts=wall_stage_counts,
        wall_drop_counts=wall_drop_counts,
        graph=wall_graph,
        walls=walls,
        grid=grid,
        frame=frame,
        rooms=rooms,
    )

    vectorizer_output = build_vectorizer_output(
        vectorizer_input,
        graph=wall_graph,
        faces=(
            FaceEvidence(
                polygon=room.polygon,
                area=room.area,
                observed_coverage=room.observed_coverage,
                visibility=room.visibility,
                confidence=room.confidence,
                provenance=room.provenance,
            )
            for room in rooms
            if room.polygon is not None
        ),
        openings=(
            OpeningEvidence(
                wall_index=opening.wall_index,
                kind=opening.kind,
                u_range=opening.u_range,
                v_range=opening.v_range,
                confidence=opening.confidence,
                source=opening.provenance,
            )
            for opening in openings
        ),
        adjacency=(
            AdjacencyEvidence(
                room_a=room.id,
                room_b=neighbour,
                via=None,
                confidence=min(room.confidence, rooms[neighbour].confidence),
            )
            for room in rooms
            for neighbour in room.neighbours
            if neighbour > room.id
        ),
        roomformer=roomformer_proposal,
    )

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
            vectorizer_output,
            geometry_diagnostics,
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
            result["reconstruction"]["walls"],
            out_dir / "scene.glb", gravity.floor_height, ceiling,
            rooms=result["rooms"],
        )
        export.export_scope_csv(result, out_dir)
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


def _build_geometry_diagnostics(
    *,
    wall_stage_counts: dict[str, int],
    wall_drop_counts: dict[str, int],
    graph,
    walls,
    grid,
    frame,
    rooms,
) -> dict:
    """Assemble canonical, non-model geometry explainability metadata."""
    candidate_walls = list(graph.candidates)
    quarantine_reasons = Counter()
    trim_reasons = Counter()
    for wall in candidate_walls:
        reasons = set(wall.tags)
        if wall.quarantined and not reasons:
            reasons.add(wall.snap_status or "quarantined")
        for tag in sorted(reasons):
            if "trim" in tag:
                trim_reasons[tag] += 1
        if wall.quarantined:
            for reason in sorted(reasons):
                if reason in {
                    "low-confidence",
                    "off-axis",
                    "too-short",
                    "unintended-crossing",
                    "rejected-crossing",
                    "extension-out-of-bounds",
                    "invalid-geometry",
                    "degenerate",
                } or reason.startswith("rejected"):
                    quarantine_reasons[reason] += 1

    exported_wall_ids = {
        wall.index for wall in graph.walls if wall.length >= 0.5
    }
    wall_records = [
        {
            "wall_id": f"wall_{wall.index}",
            "wall_index": wall.index,
            "endpoint_ids": [
                f"wall_{wall.index}:start",
                f"wall_{wall.index}:end",
            ],
            "stage": "quarantine" if wall.quarantined else "final",
            "action": "quarantine" if wall.quarantined else "persist",
            "exported": wall.index in exported_wall_ids,
            "exported_wall_index": wall.index if wall.index in exported_wall_ids else None,
            "reason": (
                " | ".join(sorted(set(wall.tags)))
                if wall.tags
                else "accepted"
            ),
            "provenance": wall.provenance,
            "start": wall.start.tolist(),
            "end": wall.end.tolist(),
            "normal": wall.normal.tolist(),
            "length_m": round(wall.length, 6),
            "tags": sorted(set(wall.tags)),
            "related_wall_ids": [],
            "related_wall_indices": [],
        }
        for wall in candidate_walls
    ]
    graph_diagnostics = graph.diagnostics
    endpoint_gaps = {
        "endpoint_count": graph_diagnostics.after_endpoint_count,
        "gap_quantiles_m": _gap_quantiles(
            graph_diagnostics.after_nearest_endpoint_gap_m
        ),
        "component_count": graph_diagnostics.after_endpoint_components,
        "junction_counts": dict(
            sorted(Counter(node.kind for node in graph.nodes).items())
        ),
        "node_tolerance_m": graph_diagnostics.node_tolerance_m,
        "before": {
            "endpoint_count": graph_diagnostics.before_endpoint_count,
            "gap_quantiles_m": _gap_quantiles(
                graph_diagnostics.before_nearest_endpoint_gap_m
            ),
            "component_count": graph_diagnostics.before_endpoint_components,
            "incidence_count": graph_diagnostics.before_endpoint_incidence_count,
        },
        "after": {
            "incidence_count": graph_diagnostics.after_endpoint_incidence_count,
        },
    }

    candidate_faces = polygonize_wall_graph(walls)
    accepted_faces = [
        room
        for room in rooms
        if room.polygon is not None and not room.provenance.startswith("fallback")
    ]
    from .rooms import _face_observation_metrics

    free_for_rooms = (
        (np.asarray(grid.free) >= 3) & (np.asarray(grid.occupied) < 6)
    )
    accepted_face_geometries = []
    from shapely.geometry import Polygon

    for room in accepted_faces:
        if room.polygon is not None and len(room.polygon) >= 3:
            accepted_face_geometries.append(Polygon(room.polygon))
    rejected_faces_by_reason = Counter()
    face_records = []
    for face_index, face in enumerate(candidate_faces):
        coverage, visibility = _face_observation_metrics(
            face, grid, free_for_rooms
        )
        accepted = any(
            face.symmetric_difference(accepted_face).area <= 1e-6
            for accepted_face in accepted_face_geometries
        )
        if not accepted:
            if coverage < 0.10:
                rejected_faces_by_reason["observed_coverage"] += 1
            if visibility < 0.10:
                rejected_faces_by_reason["visibility"] += 1
            if coverage >= 0.10 and visibility >= 0.10:
                rejected_faces_by_reason["not_persisted"] += 1
        face_records.append(
            {
                "face_id": f"face_{face_index}",
                "polygon": [
                    [round(float(value), 6) for value in point]
                    for point in np.asarray(face.exterior.coords)[:-1]
                ],
                "area_m2": round(float(face.area), 6),
                "observed_coverage": round(coverage, 4),
                "visibility": round(visibility, 4),
                "accepted": accepted,
                "reason": None if accepted else "observed coverage/visibility",
                "provenance": (
                    "validated wall graph"
                    if accepted
                    else "candidate wall graph"
                ),
            }
        )
    polygonization = {
        "candidate_face_count": len(candidate_faces),
        "accepted_face_count": len(accepted_faces),
        "rejected_faces_by_reason": dict(sorted(rejected_faces_by_reason.items())),
        "geometry_types": dict(
            sorted(
                Counter(face.geom_type for face in candidate_faces).items()
            )
        ),
        "faces": face_records,
    }

    occupied_mask = np.asarray(grid.occupied) > 0
    free = (np.asarray(grid.free) > 0) & ~occupied_mask
    observed_free_evidence = np.asarray(grid.free) > 0
    occupied_evidence = occupied_mask
    observed = observed_free_evidence | occupied_evidence
    from scipy import ndimage

    labels, free_component_count = ndimage.label(
        free, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    del labels
    grid_metadata = {
        "resolution_m": grid.resolution,
        "origin": grid.origin.tolist(),
        "bounds_plan": {
            "min": grid.origin.tolist(),
            "max": (
                grid.origin + np.asarray(grid.occupied.shape) * grid.resolution
            ).tolist(),
        },
        "shape": list(grid.occupied.shape),
        "transforms": {
            "world_to_plan": {
                "right": frame.right.tolist(),
                "forward": frame.forward.tolist(),
            },
            "world_height_axis": frame.up.tolist(),
            "yaw_rad": float(frame.yaw),
        },
        "occupied_cells": int((np.asarray(grid.occupied) > 0).sum()),
        "free_cells": int(free.sum()),
        "unknown_cells": int((~observed).sum()),
        "free_component_count": int(free_component_count),
        "occupied_evidence_cells": int(occupied_evidence.sum()),
        "free_evidence_cells": int(observed_free_evidence.sum()),
    }

    fallback_used = any(room.provenance.startswith("fallback") for room in rooms)
    fallback_geometry_types = {}
    if fallback_used:
        from .rooms import _observed_floor_polygons

        fallback_polygons = _observed_floor_polygons(
            grid, free_for_rooms, min_area=1.4
        )
        fallback_geometry_types = dict(
            sorted(Counter(polygon.geom_type for polygon in fallback_polygons).items())
        )
    zero_room_reasons = []
    if not rooms:
        if not candidate_faces:
            zero_room_reasons.append("no candidate wall-graph faces")
        elif not accepted_faces:
            zero_room_reasons.append(
                "candidate wall-graph faces rejected by observed coverage/visibility"
            )
        if not free_component_count:
            zero_room_reasons.append("no observed floor components")
        else:
            zero_room_reasons.append(
                "observed floor components are below the configured area or contain unknown gaps"
            )

    return {
        "diagnostics_version": 1,
        "wall_stages": {
            "stage_counts": {
                "raw": int(wall_stage_counts.get("raw", 0)),
                "merged": int(
                    wall_stage_counts.get(
                        "merged", wall_stage_counts.get("merge", 0)
                    )
                ),
                "occlusion": int(wall_stage_counts.get("occlusion", 0)),
                "crossing": int(
                    wall_stage_counts.get(
                        "crossing", wall_stage_counts.get("graph_input", 0)
                    )
                ),
                "quarantine": int(
                    sum(1 for wall in candidate_walls if wall.quarantined)
                ),
                # The graph's post-refinement population is the internal
                # result (recordings-2: 33); export applies the existing
                # 0.5 m reporting gate (recordings-2: 31).
                "post_refinement_internal": int(len(graph.walls)),
                "exported": int(len(exported_wall_ids)),
                # Compatibility alias for older consumers. New diagnostics
                # must use post_refinement_internal instead of final alone.
                "final": int(len(graph.walls)),
            },
            "drops_by_reason": dict(wall_drop_counts),
            "quarantines_by_reason": dict(sorted(quarantine_reasons.items())),
            "trims_by_reason": dict(sorted(trim_reasons.items())),
        },
        "wall_records": wall_records,
        "endpoint_gaps": endpoint_gaps,
        "polygonization": polygonization,
        "grid": grid_metadata,
        "room_segmentation": {
            "method": (
                "wall_graph_faces"
                if accepted_faces
                else "observed_floor_components" if fallback_used else "none"
            ),
            "fallback_used": fallback_used,
            "fallback_geometry_types": fallback_geometry_types,
            "room_count": len(rooms),
            "zero_room_reason": zero_room_reasons[0] if zero_room_reasons else None,
        },
        "zero_room_reasons": zero_room_reasons,
    }


def _gap_quantiles(metrics: dict[str, float | None]) -> dict[str, float | None]:
    """Map internal endpoint metric names to the published diagnostics keys."""
    return {
        "p50": metrics.get("median"),
        "p75": metrics.get("p75"),
        "p90": metrics.get("p90"),
        "p95": metrics.get("p95"),
        "p99": metrics.get("p99"),
        "max": metrics.get("max"),
    }


def _assemble(
    bundle, gravity, frame, walls, openings, rooms, regions, concealed,
    line_items, drift, drift_report, surface_grids, uncertainty,
    vectorizer_output,
    geometry_diagnostics,
    timings, warnings, engine,
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
        vectorizer_output: Explicit density, observability, wall candidate,
            junction, opening, adjacency, and validated-face evidence.
        geometry_diagnostics: Canonical geometry-stage diagnostics. The
            export wall-length gate appends its own drop records here.
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
                "confidence": round(wall.confidence, 4),
                "fit_quality": round(wall.fit_quality, 4),
                "coordinate_convention": wall.coordinate_convention,
                "snap_status": wall.snap_status,
                "snap_residual_mm": round(wall.snap_residual * 1000, 2),
                "provenance": wall.provenance,
                "tags": wall.tags,
            }
        )

    # Keep the graph's internal post-refinement count distinct from the
    # documents exported below.  The existing 0.5 m gate is an export policy,
    # not a graph-solve decision, so each filtered wall gets an auditable
    # lifecycle record instead of disappearing from diagnostics.
    if geometry_diagnostics is not None:
        stages = geometry_diagnostics.setdefault("wall_stages", {}).setdefault(
            "stage_counts", {}
        )
        internal_count = len(walls)
        stages["post_refinement_internal"] = internal_count
        stages["final"] = internal_count  # deprecated compatibility alias
        stages["exported"] = len(wall_docs)
        records = geometry_diagnostics.setdefault("wall_records", [])
        drops = geometry_diagnostics.setdefault("wall_stages", {}).setdefault(
            "drops_by_reason", {}
        )
        known_export_drops = {
            record.get("wall_index")
            for record in records
            if record.get("stage") == "export"
        }
        for wall in walls:
            if wall.length >= 0.5 or wall.index in known_export_drops:
                continue
            drops["below_export_min_length"] = (
                int(drops.get("below_export_min_length", 0)) + 1
            )
            records.append(
                {
                    "wall_id": f"wall_{wall.index}",
                    "wall_index": wall.index,
                    "endpoint_ids": [
                        f"wall_{wall.index}:start",
                        f"wall_{wall.index}:end",
                    ],
                    "stage": "export",
                    "action": "drop",
                    "reason": "below_export_min_length",
                    "provenance": "cli._assemble.wall_export_gate",
                    "threshold_m": 0.5,
                    "start": wall.start.tolist(),
                    "end": wall.end.tolist(),
                    "normal": wall.normal.tolist(),
                    "length_m": round(wall.length, 6),
                    "tags": sorted(set(wall.tags)),
                    "related_wall_ids": [],
                    "related_wall_indices": [],
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
                "evidence_cells": opening.evidence_cells,
                "provenance": opening.provenance,
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
                "observed_coverage": round(room.observed_coverage, 4),
                "visibility": round(room.visibility, 4),
                "confidence": round(room.confidence, 4),
                "provenance": room.provenance,
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
            "wall_coordinate_convention": "finished_face",
            "walls": wall_docs,
            "openings": opening_docs,
            "vectorization": vectorizer_output.to_metadata(),
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
            "geometry": geometry_diagnostics,
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
    runner.add_argument("--max-depth", type=float, default=3.5)
    runner.add_argument("--min-confidence", type=int, default=1)
    runner.add_argument("--damage-frames", type=int, default=40)
    runner.add_argument("--min-views", type=int, default=2)
    runner.add_argument(
        "--min-detection-confidence", type=float, default=0.0,
        help="drop VLM detections (any class, including furniture) below "
             "this confidence before masking/fusion; e.g. 0.6",
    )
    runner.add_argument("--coverage", type=float, default=0.90)
    runner.add_argument("--no-refine", action="store_true", help="use raw ARKit poses")
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
