# Track A: Metric Reconstruction

## Overview

Track A turns a raw LiDAR walkthrough — RGB video, per-frame depth PNGs, IMU samples,
and ARKit camera poses from a Stray Scanner capture — into a dimensioned floor plan: a
list of named walls with lengths and heights, rooms with areas and polygons, door and
window openings with widths and sill/header heights, and a 3D model, all with stated
confidence intervals. It owns the geometry portion of `python -m pipeline run
<capture>`, from `pipeline/ingest.py` (parsing the capture) through `pipeline/rooms.py`
(room segmentation), before Track B ever runs. Every wall length, ceiling height, floor
area, and opening width in `result.json` carries an interval computed by
`pipeline/uncertainty.py`; nothing is reported as a bare number.

## Pipeline stages

`pipeline/cli.py::run()` executes Track A as a fixed sequence of stages. Each is a
distinct file with a narrow job:

**Ingest** (`ingest.py::load_capture`). Parses `odometry.csv` for per-frame
camera-to-world poses and intrinsics (`fx/fy/cx/cy`), reads one depth PNG to get the
depth-raster resolution, and recovers the IMU's up axis (`_imu_gravity`) by rotating
every accelerometer sample into the world frame with that sample's pose and averaging
— the walking-motion component is zero-mean and the gravity component is constant, so
it converges. Two conventions here were *measured*, not assumed: `ARKIT_TO_CV` and
`DEVICE_TO_CAMERA` were chosen by scoring every plausible axis mapping against real
data (see `tools/conv_test.py` and `tools/grav_test.py` — the chosen mapping wins by a
6x margin on nearest-neighbour point-cloud alignment). This matters because a wrong
axis convention doesn't crash the pipeline, it silently produces a floor plan of a
wall. `iter_frames` decodes the RGB track sequentially (seeking a long-GOP H.264 file
per frame is far slower) and can optionally also return each frame at native video
resolution (`include_full_res`) for consumers — namely Track B's VLM call — that need
more than the depth raster's resolution.

**Pose refinement** (`poses.py::refine_trajectory`, unless `--no-refine`). ARKit's
visual-inertial odometry is locally excellent but globally drifting: over a full
walkthrough, that drift is exactly what smears one wall's plane across its several
observations, which is fatal to a 2 cm wall-length gate. The fix is a standard pose
graph, but the design decision that matters is *which* edges carry which kind of
correction. Sequential edges (adjacent keyframes) come straight from ARKit's own
relative pose, weighted heavily (`ODOMETRY_INFORMATION`) — over the short ~0.2 m
between keyframes it's accurate to millimetres. Loop-closure edges — keyframes that are
spatially close but temporally far apart, found by `find_loop_candidates` — come from
point-to-plane ICP instead, weighted 100x lower, because that's the only information
ARKit doesn't already have. Using ICP for the sequential edges too is the tempting
mistake: a small systematic bias in pairwise depth registration compounds once chained
over hundreds of keyframes. This was tried and measured on `recordings-1` — it produced
metre-scale pose corrections and stretched a 2.99 m storey to 4.48 m, while *improving*
local wall agreement (a self-consistent, badly wrong solution). `refine_trajectory`
therefore also refuses any correction beyond `max_total_correction` and silently falls
back to the raw ARKit trajectory rather than ship a warped reconstruction — the
`DriftReport.rejected` flag records when this happens.

**Fusion** (`fuse.py::fuse`). Straightforward Open3D TSDF integration of posed,
depth-truncated frames into a mesh and point cloud. Takes the refined poses (or raw
ARKit poses under `--no-refine`) so this stage benefits directly from the previous one.

**Sampling** (`drift.py::sample_world_points_with_origin`). Back-projects a stride of
frames to world points, but — unlike the TSDF mesh — keeps two things the mesh
discards: each point's *camera origin* (which frame observed it, and from where) and
its *timestamp*. The origin is what lets `filter_occluded_walls` ask "did the ray from
here to this point have to pass through an already-accepted wall," and the timestamp is
what lets `measure_drift` group a wall's observations by *visit* rather than pool them.

**Geometry** (`geometry.py` + `planes.py`). `estimate_gravity` refines the IMU's noisy
up-axis hint against the point cloud's own floor/ceiling normals (only normals within a
cone of the hint are kept and averaged — this is what rejects the wall normals that
would otherwise swamp a naive dominant-direction fit) and finds floor/ceiling heights
from a height histogram. `estimate_horizontal_frame` recovers the building's yaw: wall
normals in a rectilinear building cluster at 90° spacings, so quadrupling their angle
collapses all four onto one direction whose circular mean is the yaw — robust to walls
being unevenly represented, which a plain histogram peak isn't. `extract_walls` then
runs **sequential RANSAC in the 2D gravity-aligned plan projection**, not 3D plane
fitting: once gravity is known, a vertical wall has only one free orientation and one
offset, so fitting it in 2D removes two degrees of freedom 3D RANSAC would otherwise
have to estimate from noise, and it pools every point across the wall's full height
into a single fit — this is what makes a 2 cm tolerance reachable from a 256×192 depth
sensor. Each RANSAC line is refined by total-least-squares on its inliers and gated on
`min_coverage` (the fraction of the run's own area actually observed) rather than raw
point count, because coverage is scale-free — on `recordings-1`, real walls score
8–89% while spurious slivers on the same infinite line score 1–3%, a gap absolute
point counts don't reproduce across capture densities.

`merge_collinear` then resolves two distinct kinds of duplicate RANSAC produces, which
need opposite treatment: near-coplanar fragments of the *same* surface (a wall split by
a doorway) are merged, while parallel clutter a few centimetres in front of the real
wall (door reveals, trim, cabinet fronts — a single wall on `recordings-1` spawned five
such planes within 23 cm) is suppressed, not merged, because averaging them would drag
the wall off its true position. `filter_occluded_walls` handles the opposite failure —
a candidate plane sitting *behind* an already-accepted wall, whose points could only
have arrived by light passing through solid material, which is physically impossible;
it casts rays from each point's camera origin and drops any wall more than
`occlusion_fraction` blocked by a stronger wall. `resolve_crossings` then enforces the
one constraint the fitter can't see on its own — walls don't pass through each other —
distinguishing a short T-junction overshoot (trim it) from real clutter cutting through
a wall's middle (drop the weaker surface). Finally `snap_corners` moves each wall
endpoint onto the intersection of the two fitted lines, because that intersection, not
wherever the last inlier happened to fall, is where a tape measure would actually be
hooked — this is also the step that makes reconstructed extent match what the
wall-length uncertainty model assumes (a difference of two plane intersections).
`WallSegment.inferred_fraction` tracks how much of a wall's final length was never
directly observed (behind furniture, or reconstructed purely from its corners) so
`result.json` can report those spans as inferred rather than measured.

**Rooms** (`rooms.py::segment_rooms`). Rasterises wall evidence and observed floor into
a shared grid, then runs watershed segmentation on the free-space distance transform —
the distance transform's local maxima seed one region per open area, and watershed cuts
at the narrow necks (doorways), which is exactly where a floor plan should be divided.
Floor evidence alone under-covers real rooms (a depth sensor held level sees little
floor, and furniture casts shadows), so `_grow_to_walls` dilates each labelled region
into unobserved gaps, capped at a small step limit so growth fills a furniture shadow
but doesn't escape through a doorway into the next room. Room boundaries are then
rectified from a raw marching-squares staircase into straight, axis-aligned edges
(`_boundary_polygon`/`_rectify`) — the raw contour's perimeter is meaningless (a real
14 m² room came out with a 48 m raw perimeter) and both the floor-area interval and
mold-containment barrier pricing depend on a sane perimeter. Walls are attached to
rooms by sampling just off each face (`_assign_walls`): an interior partition lands in
two rooms and becomes shared, an exterior wall lands in one. `check_no_overlaps`
validates that no two room polygons overlap beyond a small tolerance, and
`_link_neighbours` builds the room-adjacency graph from raster proximity between room
masks.

**Surfaces** (`occupancy.py::build_surface_grid` / `find_openings`, called per wall in
`cli.py`). Bins each wall's supporting points into a UV grid (along-wall by
height-above-floor) separating points on the wall plane from points well in front of it
(furniture). `find_openings` looks for holes *surrounded* by observed wall — that
distinguishes an actual doorway from the far end of a wall the operator simply never
scanned — and classifies door vs. window by whether the hole reaches the floor.
`occupancy.occluded_spans` reports the along-wall runs hidden behind furniture, which
feed directly into the wall's `inferred_fraction` and the CI widening in
`uncertainty.py`.

## The uncertainty model

`uncertainty.py::UncertaintyModel` is two layers stacked. The **physical** layer
propagates what's actually known about each measurement: sensor scatter averaged over
the number of supporting points (`plane_offset_sigma`), plus drift measured between
visits (`drift.py::measure_drift`) added in quadrature, *not* averaged down the way
scatter is — because drift displaces a whole visit's points coherently, and treating a
coherent bias as if it were averageable random error is exactly the mistake that
produces confident, wrong intervals. Each `Interval.basis` string records the actual
numbers behind the width (e.g. `"plane offset sigma 10.1 mm from 30393 points (rms
10.3 mm), drift 9.9 mm, 5% inferred"`) so an interval is auditable, not just a number.
The **conformal** layer (`fit_calibration`) is a scale factor fitted against laser
ground truth — the empirical quantile of normalised errors at the target coverage, so
it needs no distributional assumption and a handful of ground-truth points is enough to
be useful. As of this writing that scale is always `1.0` and every run carries the
warning `"confidence intervals are uncalibrated: no ground-truth fit was supplied"`,
because `bench/calibration.json` has never actually been produced — no ground-truth CSV
has been filled in and run through `bench/run.py --fit-calibration` yet. `has_depth`
also multiplies every interval by `no_lidar_multiplier` (3.0 by default) when depth was
estimated rather than measured. Stage 5 can now consume a precomputed, QC-approved
dense artifact for this path; model execution remains an explicit offline Stage 4 step.

## Known gaps and limitations

**No-LiDAR is an artifact-driven fallback, not an automatic model run.**
`ingest.load_capture()` and `frame_contract.build_frame_contract()` accept a capture with
no raw `depth/*.png` when a QC-approved Stage 4 dense directory, calibration, and pose
table are supplied. A Stray Scanner frame with an unapproved/malformed dense raster falls
back to its same-index raw LiDAR, while a video-only frame without either source is
rejected and reported. The frame contract also records OpenCV's reported and successfully
decoded video counts; a terminal decode shortfall is reported with the missing sidecar
indices, and sidecar indices are never shifted. The Metric3D adapter deliberately requires
an explicit local checkpoint/repository so CI and production ingestion never download
weights implicitly.

**Confidence intervals are structurally correct but never calibrated against real data**
— every run reports the honest "uncalibrated" warning rather than silently claiming a
tighter interval than it can support, but the conformal scale has never actually been
fit because no laser ground-truth CSV has been filled in yet.

**Fixed this session, no longer gaps:** floor and ceiling are now individually named
planes in the 3D export (`export.py::export_scene`, alongside walls); adjacency edges
now attempt to identify the connecting door via `cli.py::_infer_adjacency_via` instead
of always reporting `null`; and `rooms.check_no_overlaps` now runs on every reconstruction
to catch a room-boundary self-intersection before it reaches output.

**Still open, smaller:** no PNG floor plan or rasterised 3D-view image is emitted by the
CLI (only `floorplan.svg` and `scene.glb`) — `tools/view_plan.py` produces one but isn't
called from the pipeline. Door-derived adjacency linking is a distance heuristic (nearest
opening to the boundary midpoint within a tolerance), not a guaranteed-correct match. No
automated regression tests exist for any of this — the RANSAC/pose-graph/room-segmentation
logic has all been validated by hand against real captures, not by a test suite.

## Usage

Track A isn't invoked separately — it's the geometry portion of a single
`python -m pipeline run <capture_dir>` call, and its output lands in the same
`result.json` Track B and C write into. Flags that affect it specifically:

- `--stride` (default 4) — frame stride used for the main fusion pass.
- `--voxel` (default 0.02) — TSDF voxel size in metres.
- `--sdf-trunc` (default `4 * --voxel`) — TSDF truncation distance in metres.
- `tools/stage45_ablate.py` — CPU-only comparison of dense/raw/auto depth sources and
  named TSDF `(voxel, truncation)` variants; it writes each variant's full contract
  provenance without executing a model.
- `--max-depth` (default 3.5) — depth truncation; also the knee measured against real
  ARKit depth-accuracy falloff (see `tools/depth_bias.py`).
- `--min-confidence` (default 1) — minimum ARKit depth-confidence value (0/1/2) kept.
- `--depth-source` — force `auto`, QC-approved `dense`, or raw LiDAR for an ablation.
- `--frame-association` (default `pts`) — associate video to sidecars by normalized
  OpenCV presentation timestamps, with identity-index fallback available for comparison.
- `--no-refine` — skip pose-graph refinement and use raw ARKit poses.
- `--no-loop-closure` — refine sequential edges only, skip ICP loop closure.
- `--coverage` (default 0.90) — the target interval coverage passed to
  `UncertaintyModel`.
- `--calibration` — path to a `bench/calibration.json` produced by
  `bench/run.py --fit-calibration`; defaults to a path that, as of this writing, no run
  has ever produced.

Output: `result.json`'s `reconstruction.walls` (each with `length`/`height` intervals,
`occluded_spans`, `inferred_fraction`, and tags like `"clutter-in-front"` or
`"trimmed-at-junction"`), `reconstruction.openings`, `rooms` (area/polygon/neighbours),
and `adjacency` (room-pair edges with a `via` wall name where one was found) all come
from this track, alongside the files `floorplan.svg`, `scene.glb`, and `cloud.ply`.
