# Cozmo capture-to-scope pipeline

Takes a handheld walkthrough of a damaged property (iPhone Pro LiDAR: RGB video,
depth, ARKit poses, IMU) and produces a dimensioned floor plan, per-surface
damage intelligence, and an estimator-ready scope of work — one command per
capture.

```
capture/ ─► ingest ─► pose refinement ─► TSDF fusion ─► planes ─► rooms ─► floor plan + 3D
             │           (loop closure)                  │                  + openings + CIs
             │                                           │
             └─► keyframes ─► VLM detect ─► SAM masks ─► project to surfaces ─► fuse
                                                                                 │
                                              rules.yaml ─► concealed flags + scope of work
```

## Quick start (under 15 minutes from a clean machine)

Requires Python 3.11 (Open3D has no 3.12+ wheels for Intel macOS).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add ANTHROPIC_API_KEY for the damage pass
python -m pipeline run /path/to/capture --out out/capture_name
```

Everything runs locally except two disclosed hosted calls (see
[Disclosed external services](#disclosed-external-services)). Nothing calls our
infrastructure — the keys are yours, and every response is cached to disk so a
second run is offline and free.

### Output

```
out/<capture>/
  result.json                 # schema/result.schema.json — validated on every run;
                              # TLS-plane measurements + tolerances/confidence
  planes.json                 # retained metric 3D planes, support, and inlier identities
  floorplan.svg                # dimensioned plan: openings, inferred spans, damage overlay
  scene.glb                    # 3D reconstruction — every wall, plus each room's floor
                                # and ceiling, as individually named, selectable planes
  cloud.ply                    # fused point cloud
  mesh.ply                     # triangle mesh extracted from the same TSDF volume
  fusion_manifest.json         # depth/pose provenance, frame decisions, and video availability
  scope_sketch.csv             # room/wall geometry + area/height/thickness measurements
  scope_line_items.csv         # scope-of-work line items (action/material/qty/rule/source)
  damage_overlays/             # per-frame detection box + mask + label, full video
                                # resolution, correctly oriented (present when damage is found)
  furniture_debug_overlays/    # same rendering, for --debug-furniture's diagnostic
                                # detections (only written with --furniture-overlays too)
```

### All CLI flags

`python -m pipeline run <capture_dir> [flags]`

| Flag | Default | Effect |
|---|---|---|
| `capture` (positional) | — | Capture directory |
| `--out` | `out/<capture-name>` | Output directory |
| `--rules` | `rules.yaml` | Scope rule set — point at an edited copy for a live "change this rule and rerun" demo |
| `--cache-dir` | `cache/vlm` | VLM response cache directory |
| `--calibration` | `bench/calibration.json` | Conformal interval calibration file (from `bench/run.py --fit-calibration`) |
| `--model` | `claude-opus-5` | VLM model for damage detection |
| `--stride N` | `4` | Frame stride for the main fusion pass. Lower is slower and denser |
| `--voxel` | `0.02` | TSDF voxel size, metres |
| `--sdf-trunc` | `4 * --voxel` | TSDF truncation distance, metres |
| `--dense-depth-dir` | auto | QC-approved Stage 4 `dense_depth/` directory; raw LiDAR is used per frame when dense data is unavailable |
| `--densify-manifest` | auto | Stage 4 manifest carrying the QC approval and mask paths |
| `--depth-source` | `auto` | Ablation hook: `auto` (QC dense then same-index raw), `dense`, or `raw` |
| `--frame-association` | `pts` | Associate decoded RGB to sidecars by presentation timestamp, or use identity index mapping |
| `--pts-tolerance-s` | derived | Maximum timestamp mismatch accepted by PTS association |
| `--pose-source` | `auto` | ARKit for Stray Scanner captures, SLAM for video captures, with explicit fallback metadata |
| `--slam-poses` | — | Offline SLAM pose table (`CSV`, `JSON`, `NPY`, or `NPZ`) |
| `--max-depth` | `3.5` | Depth cutoff — the knee where ARKit depth starts reading systematically far |
| `--plane-threshold` | `0.03` | 3D structural-plane inlier threshold in metres |
| `--plane-min-inliers` | `30` | Minimum support for a 3D structural plane |
| `--max-planes` | `80` | Maximum sequential structural planes to retain |
| `--plane-seed` | `0` | Seed for deterministic structural-plane RANSAC |
| `--min-confidence 0\|1\|2` | `1` | ARKit depth-confidence floor. The glass/mirror ablation |
| `--damage-frames N` | `40` | Max keyframes sent to the VLM for damage detection |
| `--min-views N` | `2` | Independent views required to accept a damage region |
| `--rgb-openings` | off | Enable local-only Grounding DINO + SAM2 door/window evidence |
| `--grounding-dino-model` | — | Local Grounding DINO checkpoint; model downloads are never attempted |
| `--sam2-checkpoint` / `--sam2-config` | — | Local SAM2 checkpoint/config paths |
| `--opening-frames N` | `40` | Maximum spatially diverse, sharp frames sampled for RGB openings |
| `--roomformer-predictions` | — | Precomputed RoomFormer SD-TQ predictions; image-only hints stay unmeasured |
| `--min-detection-confidence` | `0.0` | Drop VLM detections (any class, including furniture) below this confidence before masking/fusion |
| `--coverage` | `0.90` | Target confidence-interval coverage |
| `--wall-thickness` | `0.15` | Explicit default thickness used only for centerline/outer area offsets when opposing faces are unmeasured |
| `--reference-type`, `--reference-observed-m`, `--reference-known-m` | — | Validate an explicit marker/tape/user reference; the factor is reported but not applied automatically |
| `--no-refine` | off | Use raw ARKit poses. The pose-refinement ablation |
| `--no-loop-closure` | off | Refine sequentially only, without loop edges |
| `--run-mast3r` | off | Run MASt3R-SLAM for a Stray capture even when ARKit poses are present; ARKit is retained as a prior/validation reference and refinement is skipped |
| `--no-damage` | off | Geometry only; skips all API calls |
| `--no-sam` | off | Local GrabCut masks instead of hosted SAM 2 |
| `--debug-furniture` | off | Diagnostic: also ask the VLM to tag named furniture, to sanity-check it's resolving objects in the frame at all — never reaches `result.json` or scope. Prints counts only |
| `--furniture-overlays` | off | With `--debug-furniture`, also write annotated images to `furniture_debug_overlays/` (otherwise only console counts are printed) |

An explicit reference can also be checked without running reconstruction:

```bash
python -m pipeline validate-scale --reference-type tape --observed-m 2.01 --known-m 2.00
```

Door dimensions are reported as advisory only and never calibrate scale.

## Input formats

**Stray Scanner** (primary): `rgb.mp4`, `depth/*.png` (uint16 mm), `confidence/*.png`,
`odometry.csv`, `imu.csv`, `camera_matrix.csv`.

Sensor conventions were **measured, not assumed** — `tools/conv_test.py` and
`tools/grav_test.py` score every plausible convention against the data. Two findings
that silently produce confident, geometrically meaningless output if guessed:

- Poses in `odometry.csv` are *already* camera-to-world in the OpenCV
  convention. On recordings-2, the measured median alignment error is 4.80 cm
  with no flip versus 24.59 cm with an ARKit→OpenCV flip; treating the CSV as
  world-to-camera is worse still (70.41 cm median / 99.44 cm metric). The
  pipeline structurally validates the expected homogeneous camera-to-world
  transforms and records `camera_to_world_opencv_csv_no_arkit_to_cv_flip` as
  provenance instead of inferring or applying a flip.
- The IMU body frame is rotated from the camera frame by `rotZ(-90°)`. Only that
  mapping resolves the accelerometer to a constant world vector (0.992
  consistency); the identity mapping scores 0.59 and yields a 10 m "room height".
- Colour/depth frame IDs remain stable for file lookup, but their poses are
  associated by the decoded RGB presentation timestamp (PTS), normalized to
  the video clock origin and matched to the nearest odometry timestamp. This
  preserves variable-frame-rate timing; a decoder that does not provide an
  advancing PTS is explicitly marked as using the frame-index fallback.

Rerun both scripts before trusting a new capture source.

## Benchmarking

`bench/run.py --result out/<capture>/result.json` alone runs a
ground-truth-free gate: no room-polygon overlaps, and a structurally valid,
symmetric room-adjacency graph. Everything else below is additive.

**Against laser ground truth** (wall length, ceiling height, floor area, door/
window widths, affected-area quantity; plus interval calibration and
incumbent head-to-head):

```bash
cp bench/gt_TEMPLATE.csv bench/gt_myhome.csv    # enter laser measurements
python bench/run.py --result out/rec1/result.json --truth bench/gt_myhome.csv
python bench/run.py ... --incumbent out/polycam/result.json    # head-to-head
python bench/run.py ... --fit-calibration                       # calibrate intervals
```

Prints every accuracy gate, the error distribution, and — importantly — whether
the confidence intervals are *calibrated*: if the system claims ±2 cm at 90%,
roughly 90% of measurements must land inside ±2 cm. Until `--fit-calibration`
has been run against real ground truth, every output is stamped
`"calibrated": false` and the floor plan says so.

**Against a laser-measured total footprint** (multi-room stitched footprint
error):

```bash
python bench/run.py --result out/<capture>/result.json --footprint-reference 45.2
```

**Against damage/scope reference CSVs** — five independent scorers, each
optional, matched by `surface_ref`:

| Gate | Flag | Reference CSV columns |
|---|---|---|
| Damage classification macro F1 | `--damage-class-reference` | `surface_ref,damage_class` |
| Water Category/Class accuracy | `--water-reference` | `surface_ref,water_category,water_class` |
| Damage segmentation IoU | `--iou-reference` | `surface_ref,u_lo,u_hi,v_lo,v_hi` (metres, `result.json`'s `extent` convention) |
| Concealed-flag recall/precision | `--concealed-reference` | `surface_ref` |
| Line-item recall vs. reference scope | `--scope-reference` | `surface_ref,action,material,quantity` |

```bash
python bench/run.py --result out/<capture>/result.json \
  --damage-class-reference ref_classes.csv --iou-reference ref_iou.csv
```

## Design notes

**Walls are fitted in 2D, not 3D.** Once gravity is known a vertical surface has
one free orientation and one offset, so fitting in the horizontal projection
removes two degrees of freedom that 3D RANSAC would estimate from noise, and
pools every point across the wall's full height into one fit.

**Wall length comes from corners, not from observed extent.** Furniture,
doorways and grazing dropout all truncate what the sensor sees; the intersection
of two wall planes is where a tape measure would go. `snap_corners` moves
endpoints onto those intersections (median endpoint-to-corner gap on
recordings-1: 22.5 cm → 0.0 cm), `resolve_crossings` enforces the one physical
constraint the fitter cannot see — walls do not pass through each other — and
each wall's offset is re-placed at the median of its per-visit offsets so the
visit that lingered longest does not decide where the wall is. Result: drift
median 9.9 mm (rec-1) / 10.3 mm (rec-2), inside the 2 cm gate's budget.

**Measurements stay on supplied plane geometry.** Stage 9 consumes finite wall
extents, plane intersections, and bounded room faces supplied by the geometry
stages; it does not close the graph or polygonize raster/occupancy corners.
Observed interior-face area is primary, while centerline/outer areas use
documented thickness offsets. Opposing-face thickness is `unmeasured` unless
both faces are observed; a configured default thickness is marked as an
assumption for derived areas. Every Stage 9 quantity carries TLS
residual/support/provenance evidence, tolerance, confidence, and manual-review
flags. Missing or low-confidence room faces remain explicitly unmeasured.

**Drift is measured by revisit spread, not by point scatter.** RMS scatter about
a fitted wall is mostly depth noise, and a plane fit averages it away — it is
nearly blind to drift. `pipeline/drift.py` instead groups each wall's points by
*when* they were observed and reports the spread between visits.

**The dominant error is the depth sensor, not the trajectory.** Having built the
drift metric, the honest finding is that loop closure buys only 1.8% of it
(21.8 → 21.4 mm on recordings-1) — so further pose engineering would have been
wasted. `tools/depth_bias.py` attributes the rest by physical signature over
1.46 M observations: ARKit depth is well behaved to ~3.4 m then reads
systematically **far** (+4.3 mm at 4.0 m, +11.6 mm at 5.4 m), incidence angle
shows no monotonic trend, and ARKit's own confidence is strongly informative by
spread (32 mm IQR at level 2 vs 58 mm at level 0). Both actionable findings are
exposed as `--max-depth` and `--min-confidence`.

**ARKit odometry is trusted; ICP is only for loop closure.** Building sequential
pose-graph edges from pairwise ICP *lowered* measured drift by 66% while
stretching the 2.99 m storey to 4.48 m — a self-consistent, badly wrong
solution. Sequential edges now come from ARKit weighted ~100× above ICP loop
edges. A refinement is accepted only when both the weighted pose-graph objective
and the independent loop residual improve, the loop gap does not worsen by more
than 5 cm, and no keyframe moves more than 75 cm. Otherwise raw ARKit poses and
the full rejection rationale are retained in diagnostics.

**A wall behind a wall gets filtered by ray occlusion, not proximity.**
`merge_collinear` removes clutter too *close* to the camera (furniture in
front of a wall). `filter_occluded_walls` handles the opposite case — a
candidate too *far* away, whose points could only exist if the observing ray
passed through an already-stronger wall. Traced by hand on recordings-1: one
drop was a 15 cm-behind, opposite-normal sliver — too thin to be a second
room, the wrong geometry to be that wall's own far face properly observed, so
it's the same wall's far face grazed through a doorway edge and fitted as its
own wall; another was blocked by a wall at 90°, with no thin-partition
explanation, i.e. stray noise. 13–15 walls dropped per capture; drift held
steady on one capture and rose on the other — reported as measured, since the
claim is a de-duplicated wall set, not a guaranteed drift win.

**One occupancy grid does three jobs.** Openings, occlusion, and damage fusion
are all questions about *where on this surface*, so they share one UV grid per
wall (`pipeline/occupancy.py`). Damage votes accumulate into that fixed grid, so
sixty observations of one stain produce one region with one area rather than
sixty double-counted ones.

**Reflections are rejected geometrically, not by classifier.** A stain seen in a
mirror back-projects to the reflected scene's depth, far from the mirror plane,
and fails the plane-agreement test. Measured: 47% of true-depth pixels land on a
surface versus 6% at doubled depth and 0% at tripled.

**Quantities are not proportional to detected area.** A 0.3 m² mold patch
produces removal plus a margin, containment sized to the *room*, PPE per
technician, and multiple HEPA passes. Every quantity in `rules.yaml` carries its
IICRC source, and rules that are our own simplification say so — those are the
ones to challenge.

**Every scope number is a config change away.** Changing the flood cut from
30 cm to 60 cm is `water.flood_cut.base_height` in `rules.yaml` and a re-run.

## Disclosed external services

| Service | Used for | Fallback if unavailable |
|---|---|---|
| Anthropic API (`claude-opus-5`) | Per-keyframe damage detection and IICRC classification | Damage pass reports errors; geometry still runs |
| Replicate (`meta/sam-2-large`) | Mask refinement from detection boxes | Local GrabCut, with widened area intervals |

Both are cached on disk by content hash, so reruns and demos need no network.

## Repository layout

```
pipeline/
  ingest.py      Stray parser; measured pose + IMU conventions
  poses.py       keyframing, loop closure, pose-graph optimisation
  fuse.py        TSDF fusion
  geometry.py    gravity recovery, plane algebra
  planes.py      metric 3D planes, wall extraction, Manhattan frame, corner intersection
  rooms.py       watershed room segmentation, adjacency, wall naming
  occupancy.py   per-surface UV grids: openings, occlusion, damage
  openings.py    normalized opening contract and cross-source fusion
  rgb_openings.py lazy Grounding DINO/SAM2 RGB adapter and depth association
  roomformer.py  RoomFormer SD-TQ prediction adapter
  drift.py       revisit-spread drift measurement
  damage/        vlm.py (detection) · masks.py (SAM/GrabCut) · fusion.py (cross-frame)
  scope.py       rules.yaml → line items + concealed-damage flags
  uncertainty.py interval model + conformal calibration
  export.py      result.json, floor plan SVG, GLB, overlays
  cli.py         `python -m pipeline run`
rules.yaml       all restoration logic, with IICRC citations
schema/          result.schema.json (output is validated every run)
bench/           ground-truth entry, gate scoring, calibration fitting
docs/            design rationale, algorithm derivations, and a gap analysis
```

## Known limitations

- Room polygons are rectified from the floor raster, not from the fitted wall
  lines. Synthetic rooms come out ~1.2% low in area; snapping to the walls
  themselves is the principled fix and is not done yet.
- A few near-parallel surfaces at corners survive de-duplication as separate
  walls when their fitted angles differ by more than 15 degrees.
- Loop closure has little leverage on open trajectories — recordings-2 ends
  7.3 m from its start with few revisits, so drift there rests on odometry
  quality alone.
- Occlusion detection over-triggers on recordings-2, marking spans inferred
  that were merely viewed at a grazing angle. Errs toward understating
  confidence, but is noisy.
- Water Class (1–4) is rarely inferable from imagery alone and is usually
  returned null rather than guessed.
- Stage 4 dense depth is consumed only when its manifest entry is
  `qc_approved`; missing or rejected entries fall back to same-index raw
  LiDAR and are recorded in `fusion_manifest.json` and `result.json`.
- The Metric3D adapter is offline by default: supply a local checkpoint and
  local repository explicitly to run densification. It never downloads model
  weights implicitly.
- Damage output requires `ANTHROPIC_API_KEY`; without it the geometry tracks
  run and the scope comes out empty.

## MASt3R-SLAM RGB video

To run an installed [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)
checkout on a standalone RGB MP4 without calibration:

```console
cozmo-ai-v2 run recording.mp4 \
  --mast3r-slam-dir /path/to/MASt3R-SLAM \
  --python /path/to/mast3r-slam-environment/bin/python \
  --no-viz \
  --save-as recording
```

The checkout must contain MASt3R-SLAM, its submodules, model checkpoints, and
GPU-enabled dependencies. The command runs from that checkout so its relative
config and checkpoint paths resolve. Results are written beneath its `logs/`
directory. This project does not bundle those external dependencies.

### Pose provenance and ARKit checks

MASt3R-SLAM's result file is read from `logs/<save-as>/<video-stem>.txt` (or
`logs/<video-stem>.txt`) using its `timestamp x y z qx qy qz qw` format. It is
converted into the pipeline's documented pose contract: a timestamped 4×4
**camera-to-world** transform in OpenCV camera axes (+X right, +Y down, +Z
forward).

For a Stray Scanner capture, retain its `odometry.csv` as a metric ARKit prior:

```console
cozmo-ai-v2 run /path/to/capture/rgb.mp4 \
  --mast3r-slam-dir /path/to/MASt3R-SLAM \
  --python /path/to/mast3r-slam-environment/bin/python \
  --pose-priors /path/to/capture/odometry.csv \
  --no-viz --save-as capture
```

When the input video is the sibling `capture/rgb.mp4`, the launcher discovers
`capture/odometry.csv` automatically; `--pose-priors` is useful when the prior
is stored elsewhere.

Stray Scanner odometry is already camera-to-world in this OpenCV convention;
do **not** apply an ARKit axis flip a second time. The launcher inspects
`main.py` without importing CUDA code. If that upstream checkout advertises a
pose-prior option, it is supplied; otherwise the prior is never passed as an
unknown argument and is used solely for post-run robust alignment.

The resulting `pose_provenance.json` records source paths, the prior mode,
coordinate convention, loop-closure counters when an optional
`mast3r_slam_metrics.json` sidecar is supplied, and diagnostics. It aligns
MASt3R-SLAM to ARKit with SE(3) for metric trajectories or Sim(3) when scale
differs, then gates fusion at: 25 cm translation RMSE, 75 cm maximum
translation error, 15° rotation RMSE, 45° maximum rotation error, and 15%
scale divergence. RGB-only input remains supported; without an ARKit prior the
manifest explicitly marks the trajectory as unaligned and no ARKit fusion gate
is claimed. When MASt3R-SLAM timestamps start at video time zero but Stray
odometry uses an absolute capture clock, the adapter detects and records the
single timestamp-origin offset before matching or interpolation.

For the LiDAR pipeline, `--run-mast3r` launches MASt3R-SLAM directly from a
Stray capture. This deliberately prefers accepted MASt3R poses over the ARKit
refinement path: `odometry.csv` remains a capability-gated upstream prior and
the metric post-run validation reference, not a reason to skip SLAM.

```console
python -m cozmo_ai_v2.pipeline run /path/to/capture \
  --run-mast3r \
  --mast3r-slam-dir /path/to/MASt3R-SLAM \
  --mast3r-python /path/to/mast3r-slam-environment/bin/python \
  --mast3r-no-viz --mast3r-save-as capture
```

This mode skips ARKit pose refinement, runs MASt3R-SLAM, then applies the same
pre-fusion ARKit alignment/divergence gates. A failed launcher, missing result,
or rejected trajectory stops before fusion and writes `mast3r_pose_provenance.json`.

Alternatively, validate and use a completed MASt3R trajectory before the fusion
stage with:

```console
python -m cozmo_ai_v2.pipeline run /path/to/capture \
  --mast3r-trajectory /path/to/MASt3R-SLAM/logs/capture/rgb.txt \
  --mast3r-metrics /path/to/MASt3R-SLAM/logs/capture/mast3r_slam_metrics.json
```

The pipeline interpolates only safely bracketed timestamp gaps (at most one
second by default), rejects trajectories that fail the ARKit gate before any
fusion, and writes `mast3r_pose_provenance.json` into its output directory.
