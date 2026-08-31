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
`occlusion_fraction` blocked by a stronger wall. `wall_graph.solve_wall_graph` then
enforces the one constraint the fitter can't see on its own — walls don't pass through
each other — distinguishing a short T-junction overshoot from real clutter cutting
through a wall's middle (the weaker candidate is quarantined). It clusters all L/T/X
intersections and solves each shared node from every incident finished-face plane line
at once; endpoints are assigned that one plane-derived coordinate rather than being
snapped independently. A missing endpoint may move only to an intersection supported
by the other fitted wall line and only within the per-wall extension budget (0.55 m by
default, reduced for weak fits); polygonization's 0.08 m noding tolerance is not a
fallback for large gaps. Collinear endpoints are never joined, so measured door gaps
remain gaps, and interior crossings are accepted only as explicit X evidence or the
weaker candidate is quarantined. Weak/off-axis candidates retain their original
geometry and quality metadata but are not forced into Manhattan topology. The legacy
`planes.snap_corners` name is only a compatibility wrapper around this global solver.
`WallSegment.inferred_fraction` tracks how much of a wall's final length was never
directly observed (behind furniture, or reconstructed purely from its corners) so
`result.json` can report those spans as inferred rather than measured.

**Rooms** (`rooms.py::segment_rooms`). Rasterises wall evidence and observed floor into
a shared grid. Validated wall-graph faces are retained only when observed floor
coverage and non-unknown visibility pass their thresholds. If the graph is fragmented
or incomplete, the fallback unions the explicitly observed free cells within each
4-connected component, handling Polygon/MultiPolygon/GeometryCollection results by
component and rejecting holes or tiny pieces. It never grows through unknown cells or
uses a convex hull that could bridge a door or an unobserved gap. Fallback rooms are
marked low-confidence with their evidence provenance. Walls are attached to rooms by
boundary intersection, `check_no_overlaps` validates the resulting faces, and
`_link_graph_neighbours` builds adjacency from shared wall IDs.

The vectorizer boundary is explicit: `projection.DensityMap` carries retained
wall-band counts, points-per-square-metre density, and observed/empty cells;
`vectorizer.VectorizerInput` adds wall candidates and global junction evidence;
`VectorizerOutput` returns accepted graph segments, validated faces, adjacency,
and opening evidence with provenance/confidence. An optional
`roomformer.RoomFormerAdapter` consumes a deterministic two-channel
`(batch, channel, x, y)` tensor (`wall_density_points_per_m2`, `observability`)
and converts local-checkpoint predictions into finished-face graph proposals.
No checkpoint is downloaded and no model is imported on the default path; an
unavailable or malformed optional model falls back to the point-cloud graph.
Predicted normalized/cell coordinates are mapped through the same raster origin
and resolution, so RoomFormer cannot silently invent a second coordinate frame.
SD-TQ opening predictions are an injected extension point and are retained as
separate evidence until the normal opening/gap validation accepts them.
General wall-stage, endpoint-gap, polygonization, grid, and room-fallback
explainability is exported only under `result.diagnostics.geometry` (version 1).
Solver-only connection proposals and decisions, including stable wall/endpoint IDs,
movement, evidence score, reason, and before/after coordinates, live under
`reconstruction.vectorization.wall_graph` so general diagnostics are not duplicated.
The wall-stage diagnostics distinguish the post-refinement internal graph from the
short-wall-filtered exported documents (`post_refinement_internal` versus
`exported`); each wall record carries its `wall_index` and exported mapping.

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
estimated rather than measured — see Known Gaps below for why that path is currently
unreachable.

## Known gaps and limitations

**No-LiDAR fallback is a hard blocker at ingest, not a soft degradation.** The downstream
concept exists and is wired (`UncertaintyModel(has_depth=False)`, the `"video_only"`
modality tag), but `ingest.load_capture()` raises `FileNotFoundError` the instant
`odometry.csv` or any `depth/*.png` is missing, and `has_depth=True` is hardcoded in the
only `CaptureBundle` constructor — there is no monocular pose, depth, or metric-scale
recovery path at all. Camera intrinsics also come from `odometry.csv`, so "no poses" and
"no intrinsics" are the same missing file, not two separable gaps. This is the highest-risk
open item for Track A: it's an explicitly scored gate, and a capture with no depth or
poses is a stated held-out test case.

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
- `--max-depth` (default 3.5) — depth truncation; also the knee measured against real
  ARKit depth-accuracy falloff (see `tools/depth_bias.py`).
- `--min-confidence` (default 1) — minimum ARKit depth-confidence value (0/1/2) kept.
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
