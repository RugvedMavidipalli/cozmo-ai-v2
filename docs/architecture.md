# Architecture

The user-facing command is `cozmo-ai-v2 pipeline INPUT --out OUT`. It is the
single start-to-finish entry point: it detects the input tier, prepares the
applicable pose/depth artifacts, invokes the established reconstruction
runner, and refuses to report success until the schema and required exports
exist. `stage_manifest.json` is written incrementally and records stage order,
status/reason, duration, inputs/outputs, model identity, pose convention, and
depth provenance. Older `prepare`, `densify`, `run`, and `pipeline run`
commands remain available as lower-level debugging interfaces.

One command, `python -m pipeline run <capture_dir>`, turns a handheld LiDAR
walkthrough into a dimensioned floor plan, a 3D model, a damage assessment, and a
restoration scope of work. This document is the map: what runs in what order, what
every CLI flag changes, and what lands in the output directory. The *how* and *why*
of each stage live in the three track documents this file links to — this one stays
at the level of "what calls what."

## The three tracks, and how they compose

| Track | Question it answers | Full writeup |
|---|---|---|
| A — Metric Reconstruction | Where are the walls, rooms, and openings, and how sure are we? | [`track-a-reconstruction.md`](track-a-reconstruction.md) |
| B — Damage Intelligence | Where is the damage, what kind, and how bad? | [`track-b-damage-intelligence.md`](track-b-damage-intelligence.md) |
| C — Scope Generation | What does fixing it actually require? | [`track-c-scope-generation.md`](track-c-scope-generation.md) |

They're not independent passes — B needs A's walls/rooms to fuse damage onto a named
surface, and C needs B's damage regions to price a scope against. A single run of
`pipeline.cli.run()` executes them in that order, in one process, against one capture.

## Stage-by-stage flow

`pipeline/cli.py::run()` is the entire orchestration. Each stage is timed
(`Timings.stage`, printed live and recorded in `diagnostics.timings_s`) and its
output feeds the next:

1. **ingest** (`ingest.load_capture`) — parses the capture's RGB video, calibration,
   and either ARKit `odometry.csv` or an offline SLAM pose table into a
   `CaptureBundle`; raw LiDAR is optional when a precomputed Stage 4 dense raster is
   supplied. Per-frame gravity is read when available, and missing/invalid pose or
   depth artifacts are reported by the frame contract.
2. **pose refinement** (`poses.refine_trajectory`, skippable with `--no-refine`) —
   corrects ARKit's accumulated drift with a pose graph: sequential edges from ARKit
   itself, loop-closure edges from ICP between spatially-near, temporally-far frames.
3. **frame contract + fusion** (`frame_contract.build_frame_contract`, `fuse.fuse`) —
   consumes QC-approved full-resolution Stage 4 depth with aligned confidence/QC
   masks and correctly scaled intrinsics. A rejected dense frame falls back to its
   same-index raw LiDAR frame. The selected pose table is recorded as ARKit or SLAM
   provenance, and the Open3D TSDF exports both a cloud and mesh.
4. **sampling** — back-projects a wider frame set into provenance-tagged 3D points
   (each carrying which camera observed it), used by both occlusion filtering and
   drift measurement below.
5. **geometry** (`geometry.estimate_gravity`, `planes.extract_walls` and friends) —
   recovers the up axis and floor/ceiling heights, fits the building's dominant
   horizontal frame, and extracts, merges, and occlusion-filters wall segments.
6. **wall refinement** (`drift.refit_wall_offsets`, `planes.resolve_crossings`,
   `planes.snap_corners`) — per-visit offset refitting, drift measurement, crossing
   resolution, and corner snapping, in that specific order (drift has to be measured
   before extents change).
7. **rooms** (`rooms.segment_rooms`) — watershed segmentation over free space, room
   naming, adjacency, and (this session) a self-consistency check
   (`rooms.check_no_overlaps`) that flags any polygon overlap as a warning.
8. **surfaces/openings** (`occupancy.build_surface_grid`, `occupancy.find_openings`) —
   per-wall UV occupancy grids and conservative door/window detection from
   silhouette holes. Optional `--rgb-openings` adds local-only Grounding DINO
   boxes and SAM2 masks; `--roomformer-predictions` adapts precomputed
   RoomFormer SD-TQ hints. All sources share `NormalizedOpening`, and only
   wall-associated masks with valid calibrated depth become metric evidence.
9. **measurements** (`measurements.measure_scene`) — metric wall lengths, room
   interior-face/centerline/outer areas, perpendicular floor-to-ceiling height
   statistics, wall inlier extents, and observed-only thickness. The primary
   area is the interior-face area. It consumes finite extents, plane
   intersections, and bounded faces supplied by the geometry stages; it does
   not close the graph or polygonize raster corners. Missing or low-confidence
   faces are explicitly unmeasured. An explicit marker/tape/user reference can
   be checked with `validate-scale`; door sizes remain advisory and never
   silently calibrate the capture.
10. **damage** (`cli._damage_pass`, skippable with `--no-damage`) — Track B in full:
   keyframe selection, VLM detection, mask refinement, fusion into per-surface
   regions. See track-b-damage-intelligence.md.
11. **scope** (`scope.ScopeEngine.build`) — Track C: `rules.yaml` applied to the
   damage regions from stage 10, producing line items and concealed-damage flags. See
    track-c-scope-generation.md.
12. **export** — writes every output file listed below, then validates the assembled
    result against `schema/result.schema.json` (a validation failure is a warning, not
    a crash, but a non-empty problem list makes the CLI exit non-zero).

## CLI flags

All flags apply to `python -m pipeline run <capture> [flags]`.

| Flag | Default | Effect |
|---|---|---|
| `capture` (positional) | — | Capture directory (Stray Scanner layout) |
| `--out` | `out/<capture-name>` | Output directory |
| `--rules` | `rules.yaml` | Path to the scope rule set — pointing this at an edited copy is how a live "change the flood-cut height and rerun" demo works |
| `--cache-dir` | `cache/vlm` | VLM response cache directory |
| `--calibration` | `bench/calibration.json` | Conformal interval calibration file (see track-a-reconstruction.md — this file doesn't exist until `bench/run.py --fit-calibration` has been run against real ground truth) |
| `--model` | `claude-opus-5` | VLM model for damage detection |
| `--stride` | `4` | Frame stride for the main fusion pass |
| `--voxel` | `0.02` | TSDF voxel size, metres |
| `--sdf-trunc` | `4 * --voxel` | TSDF truncation distance, metres |
| `--max-depth` | `3.5` | Depth cutoff — the knee where ARKit depth starts reading systematically far, measured by `tools/depth_bias.py` |
| `--min-confidence` | `1` | Minimum ARKit depth-confidence value to trust (0=low, 1=medium, 2=high) |
| `--depth-source` | `auto` | Use QC-approved dense depth, raw LiDAR, or dense-then-raw fallback |
| `--frame-association` | `pts` | Associate RGB frames to sidecars by normalized presentation timestamps or identity indices |
| `--damage-frames` | `40` | Max keyframes sent to the VLM for damage detection |
| `--min-views` | `2` | Minimum independent views required to accept a fused damage region |
| `--rgb-openings` | off | Enable optional RGB door/window detection; requires local model paths |
| `--grounding-dino-model` | — | Local Grounding DINO checkpoint directory; no download is attempted |
| `--sam2-checkpoint` / `--sam2-config` | — | Local SAM2 checkpoint/config; no download is attempted |
| `--rgb-device` | `cuda` | Device for explicitly enabled RGB model inference |
| `--opening-frames` | `40` | Maximum sharp, spatially diverse RGB opening frames |
| `--roomformer-predictions` | — | Precomputed RoomFormer SD-TQ JSON; image-only hints remain unmeasured |
| `--min-detection-confidence` | `0.0` | Drop VLM detections (any class, including furniture) below this confidence before masking/fusion |
| `--coverage` | `0.90` | Target confidence-interval coverage |
| `--no-refine` | off | Use raw ARKit poses, skip the pose graph |
| `--no-loop-closure` | off | Keep pose refinement but disable ICP loop-closure edges |
| `--no-damage` | off | Skip Track B and C entirely — geometry only |
| `--no-sam` | off | Force local GrabCut masks, skip the SAM2/Replicate path |
| `--debug-furniture` | off | Diagnostic-only: ask the VLM to also tag named furniture, to sanity-check it's resolving objects in the frame at all — never reaches `result.json` or scope. Prints counts only (see track-b-damage-intelligence.md) |
| `--furniture-overlays` | off | With `--debug-furniture`, also render and write `furniture_debug_overlays/` images (off by default — counts alone are usually enough, and rendering costs a full pass over the detected frames) |

## Output inventory

Everything below lands in `<out>/<capture-name>/` (or `--out`'s value directly):

| File | From | Contents |
|---|---|---|
| `result.json` | export | The full schema-shaped result: capture metadata, reconstruction (walls, openings), plane-derived measurements (explicit area conventions, height statistics, thickness status, tolerance/confidence/evidence/review flags), rooms, adjacency, damage, concealed flags, scope line items, diagnostics |
| `floorplan.svg` | `export.render_floorplan` | Dimensioned 2D floor plan — wall lengths with intervals, openings, occluded spans marked as inferred, damage shaded on its wall |
| `scene.glb` | `export.export_scene` | 3D model: every wall, and (this session) every room's floor and ceiling, as individually named, selectable planes (`room_1.north_wall`, `room_1.floor`, `room_1.ceiling`) — no dense mesh; the raw fused surface lives separately in `cloud.ply` |
| `cloud.ply` | fusion | Raw fused point cloud |
| `mesh.ply` | fusion | Triangle mesh extracted from the same TSDF volume |
| `fusion_manifest.json` | fusion | Deterministic integrated/rejected/fallback frame indices, depth/pose provenance, and video availability evidence |
| `scope_sketch.csv` | `export.export_scope_csv` | Room/wall geometry table (room, area, ceiling height, wall, wall length) — the sketch half of an Xactimate-style import |
| `scope_line_items.csv` | `export.export_scope_csv` | Line-item table (room, surface, action, material, description, quantity, unit, trade, rule_id, source, basis) — the scope half |
| `openings.csv` | `export.export_openings_csv` | Door/window evidence with state, provenance, metric intervals, wall association, and depth support |
| `damage_overlays/frame_NNNNNN.jpg` | `export.render_damage_overlays` | Per-frame images with the fused damage mask, box, and class/confidence label drawn on — saved at native video resolution, correctly oriented |
| `furniture_debug_overlays/frame_NNNNNN.jpg` | same renderer | Only present with `--debug-furniture --furniture-overlays` together; same rendering, diagnostic content |

## Benchmarking and deeper usage

Gate-by-gate scoring (`bench/run.py`), the CSV reference formats for the six
scorers that don't require laser ground truth, and a fuller CLI usage walkthrough
than this document or the README cover are in
[`benchmarking-and-usage.md`](benchmarking-and-usage.md).

## Other reference material in this folder

- [`additional_information.md`](additional_information.md) — design rationale and
  coordinate/data conventions extracted from source-code comments, organized by file.
  This is "why is it built this way," not "how do I use it."
- [`misc.md`](misc.md) — a fresh, file:line-cited gap analysis against the assignment
  spec: what's done, what's partial, what's missing, and (in its Update section) what
  this session fixed.
