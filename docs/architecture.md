# Architecture

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

1. **ingest** (`ingest.load_capture`) — parses `odometry.csv`, `imu.csv`, the depth
   PNGs, and `rgb.mp4` into a `CaptureBundle`: poses, intrinsics, per-frame gravity
   direction. This is also where the pipeline's hardest current limitation lives — it
   requires a Stray-Scanner-shaped capture with real depth and poses, and raises
   immediately if either is missing (see track-a-reconstruction.md's gaps section).
2. **pose refinement** (`poses.refine_trajectory`, skippable with `--no-refine`) —
   corrects ARKit's accumulated drift with a pose graph: sequential edges from ARKit
   itself, loop-closure edges from ICP between spatially-near, temporally-far frames.
3. **fusion** (`fuse.fuse`) — TSDF-integrates posed depth frames into a mesh and point
   cloud.
4. **sampling** — back-projects a wider frame set into provenance-tagged 3D points
   (each carrying which camera observed it), used by both occlusion filtering and
   drift measurement below.
5. **geometry** (`geometry.estimate_gravity`, `planes.extract_walls` and friends) —
   recovers the up axis and floor/ceiling heights, fits the building's dominant
   horizontal frame, and extracts, merges, and occlusion-filters wall segments.
6. **wall refinement** (`drift.refit_wall_offsets`, `wall_graph.solve_wall_graph`) —
   per-visit offset refitting and drift measurement followed by one global,
   plane-constrained solve for shared L/T/X nodes. Endpoint closure is only
   proposed when two wall lines provide a bounded, evidence-supported
   intersection (the default per-wall extension cap is 0.55 m); there is no
   global 0.8 m snap. Weak, off-axis, or unintended-crossing candidates remain
   quarantined for diagnostics rather than being forced into the topology.
7. **rooms/vectorization** (`projection.project_wall_density`,
   `vectorizer.VectorizerInput`/`VectorizerOutput`, `rooms.segment_rooms`) —
   `projection.project_wall_density` first
   crops the cleaned cloud to the configured wall-height band and produces the
   deterministic NumPy top-down density map carried by `rooms.PlanGrid`; the
   vectorizer then uses that wall evidence alongside observed floor cells for
   graph-face validation, room naming, and adjacency. Incomplete graphs use
   only independently observed free-cell components as a low-confidence
   fallback; unknown cells are never filled or bridged. A self-consistency
   check (`rooms.check_no_overlaps`) flags any polygon overlap as a warning.
   The optional `roomformer.RoomFormerAdapter` consumes the same explicit
   `(density, observability)` tensor at this boundary and returns a
   finished-face `WallGraphProposal` with polygons, corners, topology, model
   provenance, and confidence. It is lazy and local-checkpoint-only: the
   default path records a deterministic point-cloud-graph fallback without
   importing RoomFormer or running GPU inference. SD-TQ opening predictions
   have an injected predictor hook and remain separate opening evidence until
   validated by the existing gap-preserving occupancy stage.
8. **surfaces** (`occupancy.build_surface_grid`, `occupancy.find_openings`) — per-wall
   UV occupancy grids and door/window detection from silhouette holes in them.
9. **damage** (`cli._damage_pass`, skippable with `--no-damage`) — Track B in full:
   keyframe selection, VLM detection, mask refinement, fusion into per-surface
   regions. See track-b-damage-intelligence.md.
10. **scope** (`scope.ScopeEngine.build`) — Track C: `rules.yaml` applied to the
    damage regions from stage 9, producing line items and concealed-damage flags. See
    track-c-scope-generation.md.
11. **export** — writes every output file listed below, then validates the assembled
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
| `--max-depth` | `3.5` | Depth cutoff — the knee where ARKit depth starts reading systematically far, measured by `tools/depth_bias.py` |
| `--min-confidence` | `1` | Minimum ARKit depth-confidence value to trust (0=low, 1=medium, 2=high) |
| `--damage-frames` | `40` | Max keyframes sent to the VLM for damage detection |
| `--min-views` | `2` | Minimum independent views required to accept a fused damage region |
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
| `result.json` | export | The full schema-shaped result: capture metadata, reconstruction (walls, openings), rooms, adjacency, damage, concealed flags, scope line items, diagnostics (timings, drift, calibration status, warnings) |
| `floorplan.svg` | `export.render_floorplan` | Dimensioned 2D floor plan — wall lengths with intervals, openings, occluded spans marked as inferred, damage shaded on its wall |
| `scene.glb` | `export.export_scene` | 3D model: every wall, and (this session) every room's floor and ceiling, as individually named, selectable planes (`room_1.north_wall`, `room_1.floor`, `room_1.ceiling`) — no dense mesh; the raw fused surface lives separately in `cloud.ply` |
| `cloud.ply` | fusion | Raw fused point cloud |
| `scope_sketch.csv` | `export.export_scope_csv` | Room/wall geometry table (room, area, ceiling height, wall, wall length) — the sketch half of an Xactimate-style import |
| `scope_line_items.csv` | `export.export_scope_csv` | Line-item table (room, surface, action, material, description, quantity, unit, trade, rule_id, source, basis) — the scope half |
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
