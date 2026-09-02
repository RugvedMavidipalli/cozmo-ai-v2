from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..mast3r_slam import Mast3rSlamError, run_rgb_video
from . import export
from .damage import fusion as damage_fusion
from .damage.masks import refine
from .damage.vlm import FURNITURE_CLASS, DamageAnalyzer
from .drift import measure_drift, refit_wall_offsets, sample_world_points_with_origin
from .frame_contract import build_frame_contract
from .fuse import fuse
from .geometry import estimate_gravity
from .geometry_diagnostics import GeometryDiagnostics
from .ingest import display_rotation, load_capture, iter_frames
from .keyframes import select_damage_keyframes
from .measurements import (
    MeasurementContext,
    ScaleValidation,
    measure_scene,
    validate_reference_scale,
)
from .occupancy import build_surface_grid, find_openings, occluded_spans
from .openings import fuse_openings
from .planes import (
    PlaneClassification,
    StructuralPlane,
    extract_structural_planes,
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
from .slam import (
    SlamResultError,
    integrate_mast3r_results,
    mast3r_results_dir,
    mast3r_trajectory_path,
    write_pose_failure_manifest,
    write_pose_integration_manifest,
)
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


def _launch_mast3r_for_capture(
    capture_root: Path,
    mast3r_slam_dir: Path,
    config: str,
    python_executable: str,
    save_as: str | None,
    no_viz: bool,
) -> tuple[Path, Path, str]:
    """Run MASt3R-SLAM on a Stray capture's RGB stream.

    Stray odometry is deliberately retained as a measured metric prior and
    post-run validation reference. It does not suppress RGB SLAM: upstream
    receives it only when that checkout advertises a compatible prior option.
    """

    video_path = capture_root / "rgb.mp4"
    pose_priors_path = capture_root / "odometry.csv"
    invocation = run_rgb_video(
        video_path,
        mast3r_slam_dir,
        config,
        python_executable=python_executable,
        save_as=save_as,
        no_viz=no_viz,
        pose_priors_path=pose_priors_path,
    )
    results_dir = mast3r_results_dir(invocation.cwd, video_path, save_as)
    return (
        mast3r_trajectory_path(invocation.cwd, video_path, save_as),
        results_dir,
        invocation.pose_prior_mode,
    )


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
    9. Measurements -- derive metric quantities from TLS plane geometry and
       intersections, with explicit area/thickness conventions.
    10. Damage -- look for water, fire, and mold damage.
    11. Scope -- turn any damage found into a list of repair line items.
    12. Export -- write out all the result files.

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
    geometry_diagnostics = GeometryDiagnostics()
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

    mast3r_trajectory = args.mast3r_trajectory
    mast3r_results_path: Path | None = None
    mast3r_pose_prior_mode = "post_alignment"
    if args.run_mast3r:
        if args.mast3r_slam_dir is None:
            print(
                "error: --run-mast3r requires --mast3r-slam-dir",
                file=sys.stderr,
            )
            return 1
        mast3r_manifest = out_dir / "mast3r_pose_provenance.json"
        try:
            with timings.stage("MASt3R-SLAM RGB tracking"):
                (
                    mast3r_trajectory,
                    mast3r_results_path,
                    mast3r_pose_prior_mode,
                ) = _launch_mast3r_for_capture(
                    bundle.root,
                    args.mast3r_slam_dir,
                    args.mast3r_config,
                    args.mast3r_python,
                    args.mast3r_save_as,
                    args.mast3r_no_viz,
                )
        except Mast3rSlamError as exc:
            write_pose_failure_manifest(
                mast3r_manifest,
                exc,
                pose_priors_path=bundle.root / "odometry.csv",
                pose_prior_mode="not_started",
            )
            print(f"error: {exc}; wrote diagnostics to {mast3r_manifest}", file=sys.stderr)
            return 1
        print(
            "  MASt3R-SLAM requested with ARKit as a prior/validation reference; "
            "skipping ARKit pose refinement"
        )

    # Stage 2: pose refinement
    poses = bundle.poses
    drift_report = None
    if not args.no_refine and not args.run_mast3r and bundle.has_depth:
        with timings.stage("pose refinement"):
            keyframes = select_keyframes(bundle)
            poses, drift_report = refine_trajectory(
                bundle, keyframes, enable_loop_closure=not args.no_loop_closure
            )
        print(
            f"  {drift_report.keyframe_count} keyframes, "
            f"{drift_report.loop_edges}/{drift_report.loop_candidates} loop edges"
        )
        refinement_status = "rejected; raw ARKit retained" if drift_report.rejected else "accepted"
        print(
            f"  refinement {refinement_status}: objective "
            f"{drift_report.objective_before:.6f} -> {drift_report.objective_after:.6f}, "
            f"loop gap {drift_report.loop_closure_gap_before:.3f} m -> "
            f"{drift_report.candidate_loop_closure_gap_after:.3f} m"
        )
        if drift_report.rejected:
            warnings.extend(f"pose refinement rejected: {reason}" for reason in drift_report.rejection_reasons)
    elif not args.no_refine and not args.run_mast3r:
        warnings.append(
            "pose refinement skipped: no raw LiDAR depth is available; "
            "using the selected SLAM/ARKit pose table"
        )

    # A completed MASt3R-SLAM trajectory takes precedence over the local
    # ARKit refinement only after metric alignment and divergence gates pass.
    # Interpolation fills capture timestamps; it is not another registration
    # pass. This must happen before the frame contract is built so fusion and
    # provenance observe the selected pose table.
    mast3r_integration = None
    if mast3r_trajectory:
        mast3r_manifest = out_dir / "mast3r_pose_provenance.json"
        try:
            with timings.stage("MASt3R-SLAM trajectory validation"):
                mast3r_integration = integrate_mast3r_results(
                    mast3r_trajectory,
                    pose_priors_path=bundle.root / "odometry.csv",
                    pose_prior_mode=mast3r_pose_prior_mode,
                    results_dir=mast3r_results_path or Path(mast3r_trajectory).parent,
                    metrics_path=args.mast3r_metrics,
                    target_timestamps=bundle.timestamps,
                    interpolation_max_gap_seconds=args.mast3r_max_pose_gap,
                )
            write_pose_integration_manifest(mast3r_manifest, mast3r_integration)
        except SlamResultError as exc:
            write_pose_failure_manifest(
                mast3r_manifest,
                exc,
                pose_priors_path=bundle.root / "odometry.csv",
                pose_prior_mode=mast3r_pose_prior_mode,
            )
            print(f"error: {exc}; wrote diagnostics to {mast3r_manifest}", file=sys.stderr)
            return 1
        if not mast3r_integration.fusion_allowed:
            print(
                "error: MASt3R-SLAM trajectory failed ARKit divergence gates; "
                f"see {mast3r_manifest} before fusion",
                file=sys.stderr,
            )
            return 1
        poses = mast3r_integration.trajectory.poses
        alignment = mast3r_integration.alignment
        print(
            f"  MASt3R-SLAM {alignment.method} alignment: "
            f"{alignment.matched_frames} matches, "
            f"translation RMSE {alignment.translation_rmse_m:.3f} m, "
            f"rotation RMSE {alignment.rotation_rmse_degrees:.2f}°, "
            f"scale divergence {alignment.scale_divergence_fraction:.1%}"
        )

    # Stage 3: fusion.  The contract is the single source of truth for
    # depth resolution, QC masks, provenance, and the pose table used by
    # both TSDF integration and the later provenance-aware sampling pass.
    pose_provenance = (
        mast3r_integration.pose_source
        if mast3r_integration is not None
        else (
            f"refined_{bundle.pose_source}"
            if drift_report is not None and not drift_report.rejected
            else bundle.pose_source
        )
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
        structural_planes = extract_structural_planes(
            points,
            normals=normals,
            up=gravity.up,
            floor_height=gravity.floor_height,
            inlier_threshold=getattr(args, "plane_threshold", 0.03),
            min_inliers=getattr(args, "plane_min_inliers", 30),
            max_planes=getattr(args, "max_planes", 80),
            seed=getattr(args, "plane_seed", 0),
        )
        band = wall_band_mask(points, normals, gravity, gravity.up)
        walls = extract_walls(
            frame.to_plan(points[band]),
            frame.height(points[band]),
            diagnostics=geometry_diagnostics,
        )
        walls = snap_to_frame(walls, frame, diagnostics=geometry_diagnostics)
        walls = merge_collinear(walls, diagnostics=geometry_diagnostics)
        _attach_structural_plane_ids(walls, structural_planes, frame)
        if not walls:
            # The established 2D fitter remains authoritative whenever it
            # has enough wall-band support. The retained 3D wall support is
            # also a safe fallback for sparse captures.
            walls = [
                wall
                for plane in structural_planes
                if plane.classification == PlaneClassification.WALL.value
                and not plane.quarantined
                for wall in [plane.to_wall_segment(frame, points=points)]
                if wall is not None
            ]
        geometry_diagnostics.set_wall_stage("quarantine", walls)
        walls, occluded_out = filter_occluded_walls(
            walls,
            frame,
            sampled,
            origins,
            diagnostics=geometry_diagnostics,
        )
    ceiling = gravity.ceiling_height
    print(
        f"  {len(walls)} walls ({occluded_out} occlusion-inconsistent removed), "
        f"room height {gravity.room_height:.3f} m" if gravity.room_height
        else "  no ceiling found"
    )
    if gravity.room_height is None:
        warnings.append("no ceiling plane found; heights are unavailable")
        if gravity.ceiling_fit is not None and gravity.ceiling_fit.rejection_reasons:
            warnings.append(
                "ceiling fit not accepted: "
                + ", ".join(gravity.ceiling_fit.rejection_reasons)
            )
    if gravity.floor_low_confidence:
        warnings.append(
            "floor plane is low confidence: "
            f"{gravity.floor_residual_rms * 1000.0:.1f} mm RMS over "
            f"{gravity.floor_inlier_count} points "
            f"(adaptive limit {gravity.floor_adaptive_residual_limit * 1000.0:.1f} mm)"
        )
    elif not gravity.floor_observed:
        reasons = (
            ", ".join(gravity.floor_fit.rejection_reasons)
            if gravity.floor_fit is not None
            else "support_below_threshold"
        )
        warnings.append(f"floor plane was not observed: {reasons}")

    # Stage 6: wall refinement
    with timings.stage("wall refinement"):
        refitted = refit_wall_offsets(walls, frame, sampled, times)
        drift = measure_drift(walls, frame, sampled, times)
        walls = resolve_crossings(walls, diagnostics=geometry_diagnostics)
        corners = snap_corners(walls, diagnostics=geometry_diagnostics)
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
        geometry_diagnostics.set_wall_stage("crossing", walls)
        geometry_diagnostics.set_wall_stage("post_refinement_internal", walls)
        geometry_diagnostics.record_endpoint_gaps(walls)
        rooms = segment_rooms(
            grid,
            walls,
            frame,
            gravity.floor_height,
            ceiling,
            diagnostics=geometry_diagnostics,
        )
        overlaps = check_no_overlaps(rooms)
        if overlaps:
            warnings.append(
                f"{len(overlaps)} room-pair(s) overlap beyond tolerance: "
                + ", ".join(
                    f"room_{a}/room_{b} {frac:.1%}" for a, b, frac in overlaps
                )
            )
    print(f"  {len(rooms)} rooms")
    if not rooms:
        reason = geometry_diagnostics.room_segmentation.get("zero_room_reason")
        print(f"  room extraction reason: {reason or 'not reported'}")

    # Stage 8: surfaces/openings. Geometry remains the default source. RGB
    # and RoomFormer evidence is optional and must earn a wall association and
    # valid depth before it can enter the same normalized opening contract.
    with timings.stage("surfaces"):
        surface_grids = {}
        geometry_openings = []
        if gravity.ceiling_observed and ceiling is not None:
            for wall in walls:
                if wall.length < 0.6:
                    continue
                surface = build_surface_grid(
                    wall, frame, points, gravity.floor_height, ceiling
                )
                surface_grids[wall.index] = surface
                geometry_openings.extend(find_openings(surface))
        else:
            warnings.append("ceiling not observed; wall opening heights are unavailable")

        opening_rejections = []
        roomformer_openings = []
        predictions_path = getattr(args, "roomformer_predictions", None)
        if predictions_path:
            try:
                from .roomformer import RoomFormerSDTQAdapter

                predictions = json.loads(Path(predictions_path).read_text())
                roomformer_adapter = RoomFormerSDTQAdapter(
                    min_confidence=getattr(args, "roomformer_min_confidence", 0.25)
                )
                roomformer_openings = roomformer_adapter.adapt(
                    predictions, walls=walls
                )
                opening_rejections.extend(roomformer_adapter.rejections)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"RoomFormer opening predictions unavailable: {exc}")

        rgb_openings = []
        if getattr(args, "rgb_openings", False):
            with timings.stage("rgb openings"):
                if not bundle.has_depth:
                    warnings.append("RGB opening detection skipped: calibrated depth is unavailable")
                elif not getattr(args, "grounding_dino_model", None) or not getattr(args, "sam2_checkpoint", None) or not getattr(args, "sam2_config", None):
                    warnings.append(
                        "RGB opening detection skipped: provide local Grounding DINO, SAM2 checkpoint, and SAM2 config paths"
                    )
                else:
                    try:
                        from .rgb_openings import (
                            GroundingDINOAdapter,
                            ModelUnavailable,
                            SAM2Adapter,
                            detect_rgb_openings_with_diagnostics,
                        )

                        detector = GroundingDINOAdapter(
                            getattr(args, "grounding_dino_model"),
                            box_threshold=getattr(args, "rgb_box_threshold", 0.30),
                            text_threshold=getattr(args, "rgb_text_threshold", 0.25),
                            device=getattr(args, "rgb_device", "cuda"),
                        )
                        refiner = SAM2Adapter(
                            getattr(args, "sam2_checkpoint"),
                            model_cfg=getattr(args, "sam2_config"),
                            device=getattr(args, "rgb_device", "cuda"),
                        )
                        rgb_result = detect_rgb_openings_with_diagnostics(
                            bundle,
                            poses,
                            frame,
                            walls,
                            detector=detector,
                            refiner=refiner,
                            surface_grids=surface_grids,
                            floor_height=gravity.floor_height,
                            ceiling_height=ceiling,
                            max_frames=getattr(args, "opening_frames", 40),
                            min_detection_confidence=getattr(args, "rgb_min_confidence", 0.35),
                        )
                        rgb_openings = rgb_result.openings
                        opening_rejections.extend(
                            item.to_dict() if hasattr(item, "to_dict") else item
                            for item in rgb_result.rejected
                        )
                    except ModelUnavailable as exc:
                        warnings.append(f"RGB opening detection unavailable: {exc}")
                    except (OSError, ValueError, RuntimeError) as exc:
                        warnings.append(f"RGB opening detection failed: {exc}")

        openings = fuse_openings(geometry_openings + roomformer_openings + rgb_openings)
    print(
        f"  {len(openings)} openings "
        f"({len(geometry_openings)} geometry, {len(rgb_openings)} RGB, "
        f"{len(roomformer_openings)} RoomFormer)"
    )

    uncertainty = UncertaintyModel(
        coverage=args.coverage,
        calibration_path=args.calibration,
        has_depth=bundle.has_depth,
    )
    if not uncertainty.calibrated:
        warnings.append(
            "confidence intervals are uncalibrated: no ground-truth fit was supplied"
        )

    reference_validation: ScaleValidation | None = None
    reference_observed = getattr(args, "reference_observed_m", None)
    reference_known = getattr(args, "reference_known_m", None)
    if reference_observed is not None or reference_known is not None:
        reference_validation = validate_reference_scale(
            reference_observed,
            reference_known,
            reference_type=getattr(args, "reference_type", "user"),
        )
        if reference_validation.status == "advisory":
            warnings.append("door reference is advisory only and was not used to calibrate scale")
        elif reference_validation.status == "validated":
            warnings.append(
                "known reference scale was validated but not applied; explicitly apply the returned factor"
            )
        else:
            warnings.append("known reference scale could not be validated")

    # Stage 9: measurements.  This pass consumes structured plane geometry,
    # not the room raster or a locally reconstructed wall graph.  Its primary
    # room area is the observed interior wall-face area; centerline and outer
    # areas are explicit offsets.  Phase 1 rooms remain compatible but are
    # explicitly unmeasured until a bounded face is supplied.
    with timings.stage("measurements"):
        measurement_context = MeasurementContext.from_uncertainty(
            uncertainty,
            has_depth=bundle.has_depth,
            pose_provenance="refined" if drift_report is not None else "raw",
            default_wall_thickness_m=getattr(args, "wall_thickness", 0.15),
            pose_uncertainty_m=drift.median_spread,
        )
        scene_measurements = measure_scene(
            walls,
            rooms,
            frame=frame,
            gravity=gravity,
            context=measurement_context,
            default_wall_thickness_m=measurement_context.default_wall_thickness_m,
        )
    measured_walls = sum(1 for value in scene_measurements.walls.values() if value.length.value is not None)
    measured_rooms = sum(
        1 for value in scene_measurements.rooms.values()
        if value.interior_face_area.value is not None
    )
    print(f"  {measured_walls} wall lengths, {measured_rooms} primary room areas")

    # Stage 10: damage
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

    # Stage 11: scope
    with timings.stage("scope"):
        engine = ScopeEngine(args.rules)
        wall_lengths = {
            (wall.name or f"wall_{wall.index}"): (
                scene_measurements.walls[wall.index].length.value
                if wall.index in scene_measurements.walls
                and scene_measurements.walls[wall.index].length.value is not None
                else wall.length
            )
            for wall in walls
        }
        line_items, concealed = engine.build(regions, rooms, wall_lengths)
    print(f"  {len(line_items)} line items, {len(concealed)} concealed flags")

    # Stage 12: export
    with timings.stage("export"):
        result = _assemble(
            bundle, gravity, frame, walls, openings, rooms, regions, concealed,
            line_items, drift, drift_report, surface_grids, uncertainty,
            timings,
            warnings,
            engine,
            fusion_report=reconstruction.contract_report,
            opening_rejections=opening_rejections,
            geometry_diagnostics=geometry_diagnostics,
            scene_measurements=scene_measurements,
            reference_validation=reference_validation,
            structural_planes=structural_planes,
            plane_threshold=getattr(args, "plane_threshold", 0.03),
            plane_min_inliers=getattr(args, "plane_min_inliers", 30),
        )
        if mast3r_integration is not None:
            alignment = mast3r_integration.alignment
            mast3r_diagnostics = {
                "pose_source": mast3r_integration.pose_source,
                "pose_provenance_path": str(out_dir / "mast3r_pose_provenance.json"),
                "fusion_allowed": mast3r_integration.fusion_allowed,
                "loop_closure": {
                    "status": mast3r_integration.loop_closure.status,
                    "candidate_count": mast3r_integration.loop_closure.candidate_count,
                    "accepted_count": mast3r_integration.loop_closure.accepted_count,
                },
            }
            if alignment is not None:
                mast3r_diagnostics["alignment"] = {
                    "method": alignment.method,
                    "translation_rmse_m": alignment.translation_rmse_m,
                    "rotation_rmse_degrees": alignment.rotation_rmse_degrees,
                    "scale_divergence_fraction": alignment.scale_divergence_fraction,
                    "timestamp_offset_seconds": alignment.timestamp_offset_seconds,
                }
            result["diagnostics"]["mast3r_slam"] = mast3r_diagnostics
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
            structural_planes=result["reconstruction"]["structural_planes"],
        )
        export.export_plane_metadata(
            structural_planes, out_dir / "planes.json",
            frame=frame,
        )
        export.export_scope_csv(result, out_dir)
        export.export_reconstruction(reconstruction, out_dir)
        export.export_openings_csv(result, out_dir)

    result["diagnostics"]["timings_s"]["total"] = round(time.time() - total_start, 2)
    export.write_json(result, out_dir / "result.json")

    print(
        "  geometry stages: "
        + ", ".join(
            f"{stage}={count}"
            for stage, count in geometry_diagnostics.stage_counts.items()
        )
    )
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
        if opening.wall_index is None:
            continue
        wall = walls_by_index.get(opening.wall_index)
        if wall is None or wall.room_id not in (room_a.id, room_b.id):
            continue
        distance = float(np.linalg.norm(wall.midpoint - boundary_midpoint))
        if distance < best_distance:
            best_distance = distance
            best_name = wall_names.get(opening.wall_index)
    return best_name


def _attach_structural_plane_ids(
    walls: list,
    planes: list[StructuralPlane],
    frame,
    max_offset: float = 0.20,
) -> None:
    """Link legacy 2D wall fits to their metric 3D plane when possible.

    The 2D wall fitter remains the source of truth for wall topology and its
    established corner/occlusion cleanup.  This small association preserves
    the structural plane identity and source support on that compatible
    representation without replacing any of those deterministic operations.
    """
    references = []
    for plane in planes:
        if plane.classification != PlaneClassification.WALL.value:
            continue
        line = plane.to_wall_segment(frame)
        if line is not None:
            references.append((plane, line))
    for wall in walls:
        candidates = []
        for plane, line in references:
            alignment = abs(float(wall.normal @ line.normal))
            offset_error = abs(float(wall.offset - line.offset))
            if alignment < np.cos(np.radians(20.0)) or offset_error > max_offset:
                continue
            candidates.append(
                (
                    1.0 - alignment + offset_error,
                    -plane.support_count,
                    plane.id,
                    plane,
                )
            )
        if not candidates:
            continue
        plane = min(candidates, key=lambda item: item[:3])[-1]
        wall.structural_plane_id = plane.id
        wall.inlier_indices = plane.inlier_indices.copy()


def _assemble(
    bundle, gravity, frame, walls, openings, rooms, regions, concealed,
    line_items, drift, drift_report, surface_grids, uncertainty,
    timings, warnings, engine, fusion_report=None, opening_rejections=None,
    geometry_diagnostics=None,
    scene_measurements=None,
    reference_validation=None,
    structural_planes=None,
    plane_threshold=0.03,
    plane_min_inliers=30,
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
        scene_measurements: Stage 9 plane-geometry measurements, or `None`
            for callers that still use the Phase 1 assembly signature.
        reference_validation: Optional explicit known-reference scale check.

    Returns:
        The full result, shaped exactly as `result.json` expects it.
    """
    structural_planes = list(structural_planes or [])
    drift_by_wall = {v.wall_index: v.std for v in drift.per_wall}
    wall_docs = []
    exported_walls = (
        geometry_diagnostics.record_export_filter(walls)
        if geometry_diagnostics is not None
        else [wall for wall in walls if wall.length >= 0.5]
    )
    for wall in exported_walls:
        spans = (
            occluded_spans(surface_grids[wall.index])
            if wall.index in surface_grids
            else []
        )
        measured_wall = (
            scene_measurements.walls.get(wall.index)
            if scene_measurements is not None
            else None
        )
        if measured_wall is not None:
            length_doc = measured_wall.length.to_dict()
            vertical_extent_doc = measured_wall.inlier_vertical_extent.to_dict()
            thickness_doc = measured_wall.thickness.to_dict()
            geometry_source = measured_wall.geometry_source
            opposing_face_id = measured_wall.opposing_face_id
        else:
            length = uncertainty.wall_length(
                wall.length, wall.residual_rms, wall.inlier_count,
                drift=drift_by_wall.get(wall.index, drift.median_spread),
                inferred_fraction=wall.inferred_fraction,
            )
            length_doc = length.to_dict()
            vertical_extent_doc = None
            thickness_doc = None
            geometry_source = "phase1_wall_segment"
            opposing_face_id = None
        wall_docs.append(
            {
                "id": wall.index,
                "name": wall.name or f"wall_{wall.index}",
                "room_id": wall.room_id,
                "start": wall.start.tolist(),
                "end": wall.end.tolist(),
                "normal": wall.normal.tolist(),
                "length": length_doc,
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
                "structural_plane_id": wall.structural_plane_id,
                "tags": wall.tags,
                "inlier_vertical_extent": vertical_extent_doc,
                "thickness": thickness_doc,
                "wall_thickness": thickness_doc,
                "geometry_source": geometry_source,
                "opposing_face_id": opposing_face_id,
            }
        )

    structural_plane_docs = []
    for plane in structural_planes:
        document = plane.to_dict()
        wall_line = plane.to_wall_segment(frame)
        document["wall_line"] = (
            {
                "start": wall_line.start.tolist(),
                "end": wall_line.end.tolist(),
                "normal": wall_line.normal.tolist(),
                "offset": round(float(wall_line.offset), 6),
                "vertical_extent": list(wall_line.height_range),
            }
            if wall_line is not None
            else None
        )
        structural_plane_docs.append(document)
    ceiling_plane_ids = [
        int(plane.id)
        for plane in structural_planes
        if plane.classification == PlaneClassification.CEILING.value
        and not plane.quarantined
    ]
    floor_plane_ids = [
        int(plane.id)
        for plane in structural_planes
        if plane.classification == PlaneClassification.FLOOR.value
        and not plane.quarantined
    ]

    wall_names = {wall.index: (wall.name or f"wall_{wall.index}") for wall in walls}
    opening_docs = []
    for opening in openings:
        grid = surface_grids.get(opening.wall_index)
        resolution = grid.resolution if grid else 0.04
        width = (
            uncertainty.opening_width(
                opening.width,
                resolution,
                opening.confidence,
                _opening_sigma(opening, "u_sigma_m"),
            ).to_dict()
            if opening.width is not None
            else None
        )
        height = (
            uncertainty.opening_width(
                opening.height,
                resolution,
                opening.confidence,
                _opening_sigma(opening, "v_sigma_m"),
            ).to_dict()
            if opening.height is not None
            else None
        )
        opening_docs.append(
            {
                "wall": (
                    wall_names.get(opening.wall_index, f"wall_{opening.wall_index}")
                    if opening.wall_index is not None
                    else None
                ),
                "kind": opening.kind,
                "width": width,
                "height": height,
                "sill_height": round(opening.sill_height, 3) if opening.sill_height is not None else None,
                "header_height": round(opening.header_height, 3) if opening.header_height is not None else None,
                "u_offset": round(opening.u_range[0], 3) if opening.u_range is not None else None,
                "u_range": [round(v, 3) for v in opening.u_range] if opening.u_range is not None else None,
                "v_range": [round(v, 3) for v in opening.v_range] if opening.v_range is not None else None,
                "confidence": round(opening.confidence, 3),
                "source": opening.source,
                "state": opening.state,
                "measurement_state": opening.state,
                "provenance": opening.provenance,
                "uncertainty": opening.uncertainty,
                "wall_association": (
                    {
                        "wall_index": opening.wall_index,
                        "confidence": round(opening.wall_association_confidence, 3),
                        "distance_m": (
                            round(opening.wall_distance_m, 4)
                            if opening.wall_distance_m is not None
                            else None
                        ),
                    }
                    if opening.wall_index is not None
                    else None
                ),
                "source_frames": opening.source_frames,
                "observation_count": opening.observation_count,
                "depth_support": opening.depth_support,
                "mask_method": opening.mask_method,
            }
        )

    rooms_by_id = {room.id: room for room in rooms}
    room_docs = []
    adjacency = []
    for room in rooms:
        measured_room = (
            scene_measurements.rooms.get(room.id)
            if scene_measurements is not None
            else None
        )
        primary_area = (
            measured_room.interior_face_area.to_dict()
            if measured_room is not None
            else uncertainty.floor_area(
                room.area, room.perimeter, drift.median_spread or 0.02
            ).to_dict()
        )
        room_areas = (
            {
                "interior_face": measured_room.interior_face_area.to_dict(),
                "wall_centerline": measured_room.wall_centerline_area.to_dict(),
                "outer_footprint": measured_room.outer_footprint_area.to_dict(),
            }
            if measured_room is not None
            else None
        )
        height_statistics = (
            measured_room.floor_to_ceiling_height.to_dict()
            if measured_room is not None
            else None
        )
        legacy_ceiling_height = (
            measured_room.floor_to_ceiling_height.mean.to_dict()
            if measured_room is not None
            and measured_room.floor_to_ceiling_height.mean.value is not None
            else (
                uncertainty.ceiling_height(
                    room.height, 0.01, 0.01, 5000, 5000
                ).to_dict()
                if room.height is not None and gravity.ceiling_observed
                else None
            )
        )
        measured_polygon = (
            measured_room.boundary
            if measured_room is not None and measured_room.boundary
            else (room.polygon.tolist() if room.polygon is not None else [])
        )
        measured_perimeter = (
            float(
                np.linalg.norm(
                    np.diff(
                        np.vstack([np.asarray(measured_polygon), measured_polygon[:1]]),
                        axis=0,
                    ),
                    axis=1,
                ).sum()
            )
            if len(measured_polygon) >= 2
            else room.perimeter
        )
        room_docs.append(
            {
                "id": room.id,
                "name": room.name,
                "area": primary_area,
                "ceiling_height": legacy_ceiling_height,
                "perimeter": round(measured_perimeter, 3),
                "centroid": room.centroid.tolist(),
                "polygon": measured_polygon,
                "wall_ids": room.wall_indices,
                "neighbours": room.neighbours,
                "areas": room_areas,
                "interior_face_area": (
                    measured_room.interior_face_area.to_dict()
                    if measured_room is not None else primary_area
                ),
                "wall_centerline_area": (
                    measured_room.wall_centerline_area.to_dict()
                    if measured_room is not None else None
                ),
                "outer_footprint_area": (
                    measured_room.outer_footprint_area.to_dict()
                    if measured_room is not None else None
                ),
                "floor_to_ceiling_height": height_statistics,
                "area_convention": (
                    measured_room.area_convention if measured_room is not None else None
                ),
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
            "pose_convention": bundle.pose_convention,
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
            "ceiling_inlier_count": int(gravity.ceiling_inlier_count),
            "ceiling_residual_rms_mm": (
                round(float(gravity.ceiling_residual_rms * 1000.0), 3)
                if gravity.ceiling_residual_rms is not None
                else None
            ),
            "floor_confidence": round(float(gravity.floor_confidence), 4),
            "floor_observed": bool(gravity.floor_observed),
            "floor_quality_status": gravity.floor_quality_status,
            "floor_low_confidence": bool(gravity.floor_low_confidence),
            "floor_support_fraction": round(float(gravity.floor_support_fraction), 4),
            "floor_adaptive_residual_limit_mm": round(
                float(gravity.floor_adaptive_residual_limit * 1000.0), 3
            ),
            "floor_inlier_count": int(gravity.floor_inlier_count),
            "floor_residual_rms_mm": round(float(gravity.floor_residual_rms * 1000.0), 3),
            "floor_fit": (
                gravity.floor_fit.to_dict() if gravity.floor_fit is not None else None
            ),
            "ceiling_fit": (
                gravity.ceiling_fit.to_dict()
                if gravity.ceiling_fit is not None
                else None
            ),
            "manhattan_yaw_deg": round(float(np.degrees(frame.yaw)), 3),
            "manhattan_fraction": round(frame.manhattan_fraction, 4),
            "walls": wall_docs,
            "structural_planes": structural_plane_docs,
            "plane_extraction": {
                "algorithm": "seeded_ransac_region_growing_tls_3d",
                "refit": "total_least_squares_svd_perpendicular_residual",
                "candidate_threshold": float(plane_threshold),
                "support_threshold": int(plane_min_inliers),
                "residual_threshold": float(plane_threshold),
                "plane_count": len(structural_plane_docs),
                "kept_count": sum(not plane.quarantined for plane in structural_planes),
                "quarantined_count": sum(plane.quarantined for plane in structural_planes),
                "rejection_reasons": {
                    reason: sum(reason in plane.rejection_reasons for plane in structural_planes)
                    for reason in sorted(
                        {
                            reason
                            for plane in structural_planes
                            for reason in plane.rejection_reasons
                        }
                    )
                },
                "floor_plane_ids": floor_plane_ids,
                "ceiling_plane_ids": ceiling_plane_ids,
                "multiple_ceiling_planes": len(ceiling_plane_ids) > 1,
            },
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
            "trajectory": drift_report.diagnostics() if drift_report else None,
            "calibration": {
                "calibrated": uncertainty.calibrated,
                "scale": uncertainty.scale,
                "coverage_target": uncertainty.coverage,
            },
            "fusion": fusion_report or {},
            "opening_rejections": list(opening_rejections or []),
            "scale_validation": (
                reference_validation.to_dict() if reference_validation is not None else None
            ),
            "measurement_conventions": (
                next(iter(scene_measurements.rooms.values())).area_convention
                if scene_measurements is not None and scene_measurements.rooms
                else None
            ),
            "warnings": warnings,
            "geometry": (
                geometry_diagnostics.to_dict()
                if geometry_diagnostics is not None
                else None
            ),
        },
    }


def _opening_sigma(opening, key: str) -> float:
    """Read a numeric source-specific sigma without trusting free-form metadata."""
    try:
        value = opening.uncertainty.get(key, 0.0)
        return max(0.0, float(value))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _run_validate_scale(args: argparse.Namespace) -> int:
    """CLI entry point for an explicit, non-silent reference check."""
    validation = validate_reference_scale(
        args.observed_m,
        args.known_m,
        reference_type=args.reference_type,
        tolerance_m=args.tolerance_m,
    )
    import json

    print(json.dumps(validation.to_dict(), indent=2))
    return 0 if validation.status in {"validated", "advisory"} else 1


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
    runner.add_argument(
        "--plane-threshold", type=float, default=0.03,
        help="3D structural-plane inlier threshold in metres",
    )
    runner.add_argument(
        "--plane-min-inliers", type=int, default=30,
        help="minimum support for a 3D structural plane",
    )
    runner.add_argument(
        "--max-planes", type=int, default=80,
        help="maximum sequential 3D structural planes to extract",
    )
    runner.add_argument(
        "--plane-seed", type=int, default=0,
        help="deterministic seed for structural-plane RANSAC",
    )
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
        "--rgb-openings", action="store_true",
        help="enable optional local Grounding DINO + SAM2 door/window detection",
    )
    runner.add_argument(
        "--grounding-dino-model",
        help="local Grounding DINO checkpoint directory (no model download is attempted)",
    )
    runner.add_argument(
        "--sam2-checkpoint",
        help="local SAM2 checkpoint path (no model download is attempted)",
    )
    runner.add_argument(
        "--sam2-config",
        help="local SAM2 model config path",
    )
    runner.add_argument(
        "--rgb-device", default="cuda",
        help="device for explicitly enabled RGB models (default: cuda)",
    )
    runner.add_argument("--opening-frames", type=int, default=40)
    runner.add_argument("--rgb-box-threshold", type=float, default=0.30)
    runner.add_argument("--rgb-text-threshold", type=float, default=0.25)
    runner.add_argument("--rgb-min-confidence", type=float, default=0.35)
    runner.add_argument(
        "--roomformer-predictions",
        help="JSON file containing precomputed RoomFormer SD-TQ predictions",
    )
    runner.add_argument("--roomformer-min-confidence", type=float, default=0.25)
    runner.add_argument(
        "--min-detection-confidence", type=float, default=0.0,
        help="drop VLM detections (any class, including furniture) below "
             "this confidence before masking/fusion; e.g. 0.6",
    )
    runner.add_argument("--coverage", type=float, default=0.90)
    runner.add_argument(
        "--wall-thickness", type=float, default=0.15,
        help="explicit default wall thickness (metres) for centerline/outer areas when opposing faces are not observed",
    )
    runner.add_argument(
        "--reference-type", choices=["marker", "tape", "user", "door"], default="user",
        help="known-reference type used with --reference-observed-m/--reference-known-m",
    )
    runner.add_argument("--reference-observed-m", type=float)
    runner.add_argument("--reference-known-m", type=float)
    runner.add_argument("--no-refine", action="store_true", help="use raw ARKit/SLAM poses")
    runner.add_argument("--no-loop-closure", action="store_true")
    mast3r_input = runner.add_mutually_exclusive_group()
    mast3r_input.add_argument(
        "--mast3r-trajectory",
        type=Path,
        help=(
            "MASt3R-SLAM trajectory file to align to this capture's ARKit priors "
            "and use only if it passes pre-fusion divergence gates"
        ),
    )
    mast3r_input.add_argument(
        "--run-mast3r",
        action="store_true",
        help=(
            "run MASt3R-SLAM on this Stray capture's rgb.mp4 even though "
            "ARKit poses exist; ARKit becomes a prior/validation reference "
            "and pose refinement is skipped"
        ),
    )
    runner.add_argument(
        "--mast3r-slam-dir",
        type=Path,
        help="path to the external MASt3R-SLAM checkout (required by --run-mast3r)",
    )
    runner.add_argument(
        "--mast3r-config",
        default="config/base.yaml",
        help="config path relative to --mast3r-slam-dir",
    )
    runner.add_argument(
        "--mast3r-python",
        default=sys.executable,
        help="Python executable from the MASt3R-SLAM environment",
    )
    runner.add_argument(
        "--mast3r-save-as",
        help="optional MASt3R-SLAM logs subdirectory name",
    )
    runner.add_argument(
        "--mast3r-no-viz",
        action="store_true",
        help="pass --no-viz to MASt3R-SLAM",
    )
    runner.add_argument(
        "--mast3r-metrics",
        type=Path,
        help="optional MASt3R-SLAM loop-closure metrics JSON sidecar",
    )
    runner.add_argument(
        "--mast3r-max-pose-gap",
        type=float,
        default=1.0,
        help="maximum seconds between MASt3R poses that may be interpolated for fusion",
    )
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

    reference = subparsers.add_parser(
        "validate-scale",
        help="validate an explicit marker, tape, or user-supplied reference scale",
    )
    reference.add_argument(
        "--reference-type", choices=["marker", "tape", "user", "door"], required=True,
    )
    reference.add_argument("--observed-m", type=float, required=True)
    reference.add_argument("--known-m", type=float, required=True)
    reference.add_argument("--tolerance-m", type=float)
    reference.set_defaults(func=_run_validate_scale)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
