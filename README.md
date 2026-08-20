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
  result.json      # schema/result.schema.json — validated on every run
  floorplan.svg    # dimensioned plan: openings, inferred spans, damage overlay
  scene.glb        # 3D reconstruction
  cloud.ply        # fused point cloud
```

### Useful flags

| Flag | Effect |
|---|---|
| `--no-refine` | Use raw ARKit poses. The pose-refinement ablation. |
| `--no-loop-closure` | Refine sequentially only, without loop edges. |
| `--no-damage` | Geometry only; skips all API calls. |
| `--no-sam` | Local GrabCut masks instead of hosted SAM 2. |
| `--min-views N` | Independent views required to accept a damage region (default 2). |
| `--stride N` | Frame stride for fusion. Lower is slower and denser. |
| `--min-confidence 0\|1\|2` | ARKit depth-confidence floor. The glass/mirror ablation. |

## Input formats

**Stray Scanner** (primary): `rgb.mp4`, `depth/*.png` (uint16 mm), `confidence/*.png`,
`odometry.csv`, `imu.csv`, `camera_matrix.csv`.

Sensor conventions were **measured, not assumed** — `tools/conv_test.py` and
`tools/grav_test.py` score every plausible convention against the data. Two findings
that silently produce confident, geometrically meaningless output if guessed:

- Poses in `odometry.csv` are *already* camera-to-world in the OpenCV
  convention. Applying the textbook ARKit→OpenCV flip degrades frame alignment
  from 4.2 cm to 25.9 cm.
- The IMU body frame is rotated from the camera frame by `rotZ(-90°)`. Only that
  mapping resolves the accelerometer to a constant world vector (0.992
  consistency); the identity mapping scores 0.59 and yields a 10 m "room height".

Rerun both scripts before trusting a new capture source.

## Benchmarking against laser ground truth

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
edges, and `refine_trajectory` refuses its own output past a 75 cm correction.

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
  planes.py      wall extraction, Manhattan frame, corner intersection
  rooms.py       watershed room segmentation, adjacency, wall naming
  occupancy.py   per-surface UV grids: openings, occlusion, damage
  drift.py       revisit-spread drift measurement
  damage/        vlm.py (detection) · masks.py (SAM/GrabCut) · fusion.py (cross-frame)
  scope.py       rules.yaml → line items + concealed-damage flags
  uncertainty.py interval model + conformal calibration
  export.py      result.json, floor plan SVG, GLB, overlays
  cli.py         `python -m pipeline run`
rules.yaml       all restoration logic, with IICRC citations
schema/          result.schema.json (output is validated every run)
bench/           ground-truth entry, gate scoring, calibration fitting
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
- The no-LiDAR fallback path is not yet wired into the CLI.
- Damage output requires `ANTHROPIC_API_KEY`; without it the geometry tracks
  run and the scope comes out empty.
