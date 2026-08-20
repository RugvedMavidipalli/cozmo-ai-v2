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
of two wall planes is where a tape measure would go.

**Drift is measured by revisit spread, not by point scatter.** RMS scatter about
a fitted wall is mostly depth noise, and a plane fit averages it away — it is
nearly blind to drift. `pipeline/drift.py` instead groups each wall's points by
*when* they were observed and reports the spread between visits. On the sample
capture that is a median of 26 mm, which is the real error budget.

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

- Room polygons follow observed floor rather than snapping to the fitted wall
  lines, so reported areas are biased low where furniture blocked the floor.
- Loop-closure edge acceptance is not yet tuned; `--no-loop-closure` is
  currently the safer default on long captures.
- Water Class (1–4) is rarely inferable from imagery alone and is usually
  returned null rather than guessed.
- The no-LiDAR fallback path is not yet wired into the CLI.
