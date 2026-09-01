# Live defense script

A narration script for the screen-recording walkthrough. Written to be read (or
paraphrased) while you drive the terminal and the output files. Total runtime
~6 minutes at a natural pace — trim the bracketed asides first if you need to
hit 5.

Each block is `[SCREEN: what to show]` followed by what to say. Say it in your
own words; this is a script to work from, not a transcript to recite.

Before recording, read **"If the held-out capture has no LiDAR"** at the
bottom now, not during the recording — it's the one scenario worth having a
real answer ready for.

---

## 0. Setup (do this before you hit record)

```bash
cd cozmo-pipeline
source .venv/bin/activate
python -m pipeline.cli run <capture_dir> --out out/demo
```

Kick this off a few minutes before recording if the capture has real damage —
the VLM pass is the slow part (~10-15 min for the default 40 frames at full
resolution) and you want `out/demo/` already populated with a *fresh, real*
run when you start talking, not a spinner. Geometry alone (`--no-damage`) is
under 3 minutes if you just need the reconstruction on screen fast.

---

## 1. Cold open — what this is (~30s)

**[SCREEN: terminal, empty prompt, or the README's architecture diagram]**

"This is a capture-to-scope pipeline for property damage assessment. Input is
a handheld LiDAR walkthrough — an iPhone Pro running Stray Scanner: RGB video,
depth, ARKit poses, IMU. Output is one command later: a dimensioned floor
plan, a 3D model, per-surface damage detection with area estimates, and an
estimator-ready scope of work — the line items an adjuster or restoration
contractor would actually price. Everything runs locally except two disclosed
hosted calls: Claude for damage vision, and optionally Replicate/SAM2 for mask
refinement — both cached to disk, so a rerun is free and offline."

---

## 2. Architecture overview (~45s)

**[SCREEN: README.md's ASCII diagram, or `docs/architecture.md`]**

"Three tracks, one pipeline. Track A is metric reconstruction: ingest, pose
refinement with loop closure, TSDF fusion, plane fitting, room segmentation —
that's the geometry side, floor plan and 3D model. Track B is damage
intelligence: a vision-language model looks at selected keyframes, proposes
damage regions, SAM or GrabCut refines the mask, and everything gets projected
onto the 3D surfaces and fused across views so the same stain seen from three
angles becomes one region, not three. Track C is scope generation: rule-driven
quantity math — flood cuts, drying equipment counts, containment — cited back
to IICRC S500/S520/S700, not hardcoded numbers."

"One design choice up front: sensor conventions — which axis is up, how the
IMU frame maps to the camera frame — were *measured*, not assumed. There are
tools in the repo, `conv_test.py` and `grav_test.py`, that score every
plausible convention against the actual data and pick the one that fits. I'd
rather derive that than guess and be subtly wrong for the whole session."

---

## 3. Track A live — reconstruction (~90s)

**[SCREEN: the terminal running (or already-finished) `pipeline run`,
scrolling through the stage timings]**

"Here's a real run. Ingest reads the capture — pose count, duration, an IMU
gravity-consistency check that catches a bad capture before you waste ten
minutes on it. Pose refinement is next: sequential edges trust ARKit's own
odometry, loop-closure edges use ICP — only on revisits, not everywhere. That
split matters: I tried ICP on sequential edges too, early on, and it
catastrophically warped the whole trajectory. ARKit's relative pose between
consecutive frames is already good; ICP earns its keep only where there's an
actual loop to close. That failure is in the report as an ablation, not
swept under the rug."

**[SCREEN: `floorplan.svg` open in a browser or image viewer]**

"TSDF fusion builds the point cloud, then walls are extracted with sequential
RANSAC in a gravity-aligned 2D projection — not 3D plane fitting, because a
real wall is one plane in 2D regardless of ceiling height or slight camera
roll, and that's a much better-conditioned problem. Collinear fragments get
merged, occluded spans get marked as *inferred* rather than measured, corners
get snapped. This run found [N] walls, [N] rooms, [N] openings. The plan shows
measured intervals on every wall, not just point estimates — that's the
uncertainty model, physical error propagation plus conformal calibration once
ground truth exists to calibrate against."

**[SCREEN: `scene.glb` open in a viewer]**

"And the 3D export: every wall, plus every room's floor and ceiling, as
individually named, selectable planes — `room_1.north_wall`,
`room_1.floor` — because the assignment asks for identifiable surfaces, not
one fused mesh you'd have to go hunting through."

---

## 4. Track B live — damage detection (~75s)

**[SCREEN: `damage_overlays/` folder, or `result.json`'s `damage` array]**

"For damage, keyframes get sent to Claude at full native resolution — that
matters more than it sounds: I originally sent depth-resolution frames,
256 by 192, and it was enough to fabricate a false positive — a ceiling
light's lighting gradient read as a water stain once. Full resolution and
correct orientation fixed that. The model's prompt argues against itself: for
every region it proposes, it also has to state the most plausible *benign*
explanation and why it's rejecting it — a distractor-considered field. That's
there specifically to catch the lighting-gradient, shadow, reflection class of
false positive before it reaches fusion."

"A single frame's detection is a hypothesis, not a result. Fusion requires at
least two independent views to agree before a region is accepted — that's
`--min-views`, default 2 — plus a reflection check based on plane-agreement
geometry, plus grazing-angle discounting so an edge-on view doesn't get equal
weight to a head-on one. [If your capture has real damage: point at a
`damage_overlays/frame_*.jpg`, describe the box/mask/label, and read off the
class, subtype, and confidence.] [If it doesn't: say so directly — 'this
particular walkthrough is a clean room, so the honest thing to show is that
zero individual hypotheses met the two-view bar, which is the fusion gate
doing its job rather than the model finding nothing to look at' — and
optionally run `--debug-furniture` live to prove the model is genuinely
resolving objects in the frame, independent of damage.]"

---

## 5. Track C live — scope of work (~45s)

**[SCREEN: `scope_line_items.csv` and `scope_sketch.csv`]**

"Every accepted damage region becomes line items through `rules.yaml` — flood
cut height, baseboard removal, drying-equipment counts by affected volume,
containment and PPE for mold, concealed-damage flags when a category-3 water
event implies hidden moisture behind a wall the camera never saw. Every rule
cites its source standard. And it exports in two CSVs shaped for a direct
Xactimate-style import — a room/wall geometry sketch table and a line-item
table with quantity, unit, trade, and the rule that produced it."

---

## 6. Accuracy gates and honesty about calibration (~40s)

**[SCREEN: `bench/run.py --help`, or a gate table from the report]**

"There are fourteen accuracy gates in the assignment. `bench/run.py` scores
against laser ground truth where it exists, and — since I don't have
ground-truth annotations for these captures — I also built no-ground-truth
self-consistency checks: room polygons shouldn't overlap, adjacency inferred
from wall geometry should match adjacency inferred from actual doorway
positions. Confidence intervals are honestly reported as uncalibrated right
now — conformal calibration needs a ground-truth fit, and I'd rather say
'uncalibrated' out loud than print a number that looks calibrated and isn't."

---

## 7. Known limitations — say these before you're asked (~50s)

**[SCREEN: `report/technical.md` §8, or just talk]**

"Three I'd flag myself. One: floor and ceiling damage detection isn't wired
end to end yet — the surfaces exist, but the fusion accumulator is only built
for walls, so a floor or ceiling detection currently has nowhere to
vote and gets silently dropped. Two: the runtime gate is tight on a
single-room capture, because the VLM pass costs the same whether you split it
across five rooms or one. Three — and this is the big one —"

*(see the next section — deliver it here, in the same breath)*

---

## 8. If the held-out capture has no LiDAR

This is the one gap worth a rehearsed, calm answer rather than an improvised
one, because it's flagged as the single highest-risk item in the internal gap
analysis: **one of the two live-defense held-out captures is explicitly
plain video, no depth, no poses.**

Current reality: `ingest.load_capture()` raises immediately if `odometry.csv`
or `depth/*.png` is missing. There is no monocular pose/depth/scale path
wired in. `has_depth=False` is a real downstream concept — `uncertainty.py`
already knows to widen intervals 3x for it, `cli.py` already emits a
`"video_only"` diagnostic tag — but nothing upstream can ever produce that
state today. If you run this pipeline against a video-only capture right now,
it fails at ingest, not gracefully.

**Say this, don't dodge it:**

"If this capture has no depth or poses, the honest answer is that the
no-LiDAR fallback isn't built yet — it's designed, the downstream code already
has hooks for a `has_depth=False` state with 3x-widened uncertainty, but
ingest today hard-requires `odometry.csv` and depth frames. Given more time,
the plan is monocular depth estimation plus a single scale anchor — a known
object or a operator-height assumption — feeding the same downstream pipeline
through that existing hook, rather than a separate code path. I'd rather tell
you that now than have the demo fail on stage."

Then, if there's time, pull up `report/technical.md` §9 (the one-month plan)
and point at where this sits in the sequencing — it's deliberately early,
not an afterthought.

---

## 9. Close (~20s)

"That's the system end to end — one command, three tracks, honest uncertainty,
and a scope of work an estimator could actually price. The gaps are
documented, not hidden: `docs/misc.md` has the full audit, and the report's
failure-modes section has all nine, with what would close each one."

---

## Quick reference: commands you might run live

```bash
# Full pipeline, default settings
python -m pipeline.cli run <capture_dir> --out out/demo

# Geometry only, fast (skip the VLM pass entirely)
python -m pipeline.cli run <capture_dir> --out out/demo --no-damage

# Prove the VLM is resolving the scene, without writing overlay images
python -m pipeline.cli run <capture_dir> --out out/demo --debug-furniture

# Same, but also save the annotated furniture images
python -m pipeline.cli run <capture_dir> --out out/demo --debug-furniture --furniture-overlays

# No-ground-truth gate checks
python bench/run.py --result out/demo/result.json
```
