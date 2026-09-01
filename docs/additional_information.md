# Additional Information

Design rationale, coordinate/data conventions, and "why it's built this way" notes
that used to live as module-level docstrings and inline comments in `pipeline/`.
Moved here so the source files carry only class/method/function documentation.
Organized by file, in the order the comments/docstring appeared in each file.

---

## pipeline/keyframes.py

**Module purpose:**
Keyframe selection for the damage pass. Reconstruction wants frames that
overlap; damage analysis wants the opposite — few frames, sharp, and each
showing something the others do not. Every frame sent costs an API call, so
the selection is a coverage problem: pick the sharpest frame from each
distinct viewpoint and stop.

---

## pipeline/fuse.py

**Module purpose:** Volumetric fusion of posed depth frames into a mesh and
point cloud.

**In `fuse()`, at the `volume.integrate(...)` call:** Open3D integrates with
world-to-camera extrinsics.

---

## pipeline/geometry.py

**Module purpose:** Shared geometric primitives: gravity recovery and plane
algebra. The up axis is *measured*, never assumed. A capture whose world
frame is mislabelled produces a floor plan of a wall, silently and
confidently, so the recovery here reports a quality score that the caller
can gate on.

**`GravityEstimate` fields** (now documented via a class docstring in code,
noted here for the original comment wording):
- `up`: unit vector, world frame, pointing away from the floor
- `floor_height`: signed distance along `up` of the floor plane
- `ceiling_height`: `None` when no ceiling was observed
- `inlier_fraction`: share of points on the two horizontal slabs

**In `_floor_and_ceiling()`, at the `ceiling - floor < 1.8` check:** too short
to be a storey: no ceiling seen.

---

## pipeline/ingest.py

**Module purpose:** Capture ingest: parse a recording directory into a
`CaptureBundle`.

**Coordinate conventions** (previously the module's docstring; several
function docstrings in this file reference "see module docstring" — that now
means this section):

Stray Scanner writes odometry as camera-to-world poses whose rotation already
uses the OpenCV camera convention (camera looks down +Z, +Y down in image
space), in a gravity-aligned world frame. This was established empirically,
not assumed: `tools/conv_test.py` scores every plausible convention by how
well it aligns nearby frames' point clouds, and the identity mapping wins by
6x (4.2 cm median nearest-neighbour vs 25.9 cm for the flipped alternatives).
Poses are therefore passed through unmodified.

The world frame's up axis is likewise measured rather than assumed —
`_imu_gravity()` recovers its direction from the accelerometer and
`geometry.estimate_gravity()` refines it against the floor and ceiling,
because a wrong up axis silently produces a floor plan of a wall.

`Frame.pose` is camera-to-world. Code wanting world-to-camera (Open3D
integration, projection) takes `np.linalg.inv(frame.pose)`.

**`ARKIT_TO_CV` constant:** kept so `tools/conv_test.py` can still score the
alternative it rules out.

**`CONFIDENCE_LOW`/`CONFIDENCE_MEDIUM`/`CONFIDENCE_HIGH` constants:**
ARKit's confidence raster: 0 = low, 1 = medium, 2 = high. Low-confidence
returns are where the sensor gives up — glass, mirrors, dark or wet
surfaces, grazing incidence. Those are exactly the surfaces a damaged room is
made of, so the threshold is a first-class pipeline knob rather than a
constant.

**`DEPTH_SCALE` constant:** Stray Scanner stores depth as uint16 millimetres.

**`DEVICE_TO_CAMERA` constant:** the IMU reports in the device body frame,
which is rotated from the camera raster frame. Of the candidate mappings,
only this one turns the accelerometer into a constant world vector: it
scores 0.99 directional consistency and unit magnitude, versus 0.59-0.87 for
the alternatives (see `tools/grav_test.py`). Anything but the true mapping
leaves the walking motion uncancelled and the mean drops well below 1 g.

**`Frame` fields** (documented via the class docstring in code, reproduced
here for traceability):
- `pose`: 4x4 camera-to-world, OpenCV convention
- `color`: HxWx3 uint8 RGB, at depth resolution
- `depth`: HxW float32 metres, 0 where invalid
- `confidence`: HxW uint8 in {0,1,2}
- `color_full`: HxWx3 uint8 RGB at the native video resolution, set only
  when `iter_frames(..., include_full_res=True)` — consumers that don't need
  per-pixel alignment with depth (e.g. the VLM) get real detail instead of
  the depth raster's resolution; `None` otherwise, to avoid holding a ~56x
  larger frame in memory for every caller that has no use for it.

**`CaptureBundle` fields** (documented via the class docstring in code,
reproduced here for traceability):
- `intrinsics`: 3x3 for the depth-resolution image
- `depth_size`: (width, height)
- `poses`: (N,4,4) camera-to-world, OpenCV convention
- `gravity_up`: unit world vector opposing gravity, from the IMU
- `gravity_consistency`: 1.0 when the accelerometer resolves to a constant

**In `_read_odometry()`, at the quaternion unpacking:** Stray writes the
quaternion as (qx, qy, qz, qw).

**In `load_capture()`, at the intrinsics-scaling block:** intrinsics in
`odometry.csv` describe the full-resolution RGB frame; the depth raster is a
uniform downscale of it, so the intrinsics scale by the same factor. Median
over frames rejects the occasional ARKit outlier.

---

## pipeline/export.py

**Module purpose:** Render the pipeline's results: floor plan, 3D scene,
overlays, JSON. The floor plan is the deliverable an estimator reads, so it
carries the things an estimator needs to trust it — dimensions with their
intervals, openings drawn at their measured widths, occluded spans marked as
inferred rather than quietly drawn as if measured, and damage shaded on the
wall it belongs to.

**`ROOM_FILLS` constant:** distinct, low-saturation room fills so adjacent
rooms read apart without competing with the wall linework or the damage
overlay.

**In `render_floorplan()`'s inner `project()` helper:** SVG's y axis points
down; the plan's points up.

**In `render_floorplan()`, before the room-label loop:** room labels are
claimed first — a room's name and area outrank any single wall dimension for
an estimator scanning the drawing.

**In `render_floorplan()`, before the wall-segment-cutting loop:** the wall
is drawn in segments so openings appear as gaps and occluded spans read as
inferred rather than measured.

**In `render_floorplan()`, before the dimension-labelling loop:** dimensions
are drawn last, so no wall or overlay drawn later paints over a number.
Longest walls are labelled first, so when two labels collide the one that
survives is the more significant measurement.

**In `_dimension()`, before building the length label:** wall lengths are
labelled in centimetres (an estimator's working unit for stud-to-stud runs),
with the unit stamped on every label rather than stated once in the legend —
a cropped or covered legend must not leave a bare number's unit ambiguous.
Room areas/heights stay in m/m² (see `_room_label`), so their own labels
carry that unit instead.

**In `_legend()`, at the `placer.skipped` footer line:** say so rather than
let a reader assume every wall is dimensioned.

---

## pipeline/occupancy.py

**Module purpose:** Per-wall occupancy in surface (UV) coordinates. One
structure carries three jobs that would otherwise each need their own
representation, because all three are questions about *where on this wall*
something is:

- openings — a hole in the observed surface that reaches the floor is a
  door; one with material below it is a window.
- occlusion — cells with no observation and no line of sight are hidden by
  furniture; the assignment requires those spans to be reported as inferred
  rather than measured.
- damage — per-frame damage masks accumulate here, which is what merges
  sixty observations of one stain into a single region with a real area
  instead of sixty double-counted ones.

U runs along the wall from its start corner, V runs up from the floor.

**`SurfaceGrid` fields** (folded into the class docstring in code, reproduced
here for traceability):
- `base_height`: world height of V = 0
- `hits`: observations landing on the wall plane (openings are found as
  silhouette holes in this, see `find_openings`)
- `near`: observations well in front — furniture

*(A `passthrough` field — meant for rays that went past the plane, i.e. an
opening seen directly — was removed as dead code: it was always
zero-initialised and never read or written anywhere. Opening detection has
always actually worked off silhouette holes in `hits` instead, per
`find_openings` below.)*

**`Opening` fields** (documented via class docstring in code, reproduced here
for traceability):
- `kind`: one of "door", "window", or "pass-through"

**In `find_openings()`, at the `binary_closing`/`binary_fill_holes` block:**
fill the wall's observed silhouette, then subtract what was seen: the
difference is enclosed holes only.

**In `find_openings()`, at the `if fill < 0.45` check:** a region this
ragged is scan dropout, not an opening.

---

## pipeline/uncertainty.py

**Module purpose:** Confidence intervals for every measurement, and their
calibration. An interval is only worth printing if it is calibrated: if the
system claims ±2 cm at 90%, then close to 90% of measurements must actually
land inside ±2 cm. Overconfidence on a hard capture is explicitly an
automatic red flag in the assignment, so the model here is built to widen
honestly rather than to look precise.

Two layers:
- A *physical* model that propagates what is known about the measurement —
  depth noise averaged over the supporting points, the drift measured
  between visits, and how much of the span was inferred rather than seen.
- A *conformal* scale factor fitted against laser ground truth, which
  corrects the physical model's optimism without assuming the errors are
  Gaussian. Until ground truth exists the factor is 1.0 and the report says
  so.

**`Z_SCORES` constant:** half-width multiplier for a normal distribution at
the quoted coverage.

**`DEPTH_SIGMA_BASE` / `DEPTH_SIGMA_PER_METRE` constants:** ARKit's depth is
a fused LiDAR + ML estimate; its per-pixel error grows with range. This is a
deliberately conservative envelope, not a datasheet figure. `DEPTH_SIGMA_BASE`
is in metres at close range.

**`Interval` fields** (documented via class docstring in code, reproduced
here for traceability):
- `half_width`: the interval half-width at the quoted `coverage`
- `basis`: a human-readable statement of what evidence produced the interval

**In `plane_offset_sigma()`, at the `floor = 0.002` assignment:** no plane
fit is better than a couple of millimetres.

**In `wall_length()`, at the `sigma = per_plane * 2.0` line:** two corners,
each the intersection of this wall with a neighbour: four plane offsets
contribute, added in quadrature.

**In `wall_length()`, inside the `if inferred_fraction > 0` branch:** an
unobserved span is carried by the plane fit alone; widen in proportion to
how much of the wall was never actually seen.

**In `opening_width()`, at the `sigma = resolution / np.sqrt(12) * 2` line:**
the factor of two accounts for two quantised edges.

**In `damage_area()`, at the `relative += 0.25` line:** a box stands in for
a mask here, so the area is an upper bound.

---

## pipeline/drift.py

**Module purpose:** Measure trajectory drift by how far a surface moves
between visits.

The tempting metric — RMS scatter of points about a fitted wall — mostly
measures depth-sensor noise, and a plane fit averages that noise down by
√N. It is nearly blind to drift, because drift displaces a whole visit's
points coherently: the fit lands between the visits and the scatter barely
changes.

What actually breaks the 2 cm budget is that coherent displacement. So the
metric here groups a wall's supporting points by *when* they were observed
and reports the spread of the per-visit plane offsets. That number is the
drift contribution to wall position, it is directly comparable against the
gate, and it is what a pose-graph correction is supposed to reduce.

**`WallVisit` fields** (documented via class docstring in code, reproduced
here for traceability):
- `offsets`: per-visit fitted offset along the wall normal
- `times`: mean observation time of each visit

**In `refit_wall_offsets()`, at the `if abs(shift) > band` check:**
re-association failed; don't teleport the wall.

**In `measure_drift()`, at the `offsets.append(...)` line inside the visit
loop:** refit only the offset, holding orientation fixed — a single visit
may see too short a span to constrain the angle, and the shared orientation
is what makes the offsets comparable.

*(Note: `DriftMeasurement` in this file has neither a class docstring nor
field comments in the original code — left as-is.)*

---

## pipeline/planes.py

**Module purpose:** Extract named, measurable surfaces from a
reconstruction.

Walls are recovered as lines in the gravity-aligned horizontal projection
rather than as planes in 3D. A vertical surface has one free orientation and
one offset once gravity is known, so fitting it in 2D removes two degrees of
freedom that 3D RANSAC would otherwise have to estimate from noise — and it
pools every point across the wall's full height into a single fit, which is
what makes a 2 cm tolerance reachable from a 256x192 depth sensor.

Wall extent comes from intersecting neighbouring wall lines, never from the
spread of observed points: furniture, doorways and grazing-incidence dropout
all truncate the observed span, while the corner where two wall planes meet
is where a tape measure would be placed.

**`HorizontalFrame` fields** (folded into the class docstring in code,
reproduced here for traceability):
- `right`: first Manhattan axis
- `forward`: second Manhattan axis, right-handed with up
- `yaw`: rotation from world X, radians
- `manhattan_fraction`: share of wall area on the two dominant axes

**`WallSegment` fields** (folded into the class docstring in code, reproduced
here for traceability):
- `normal`: 2D unit normal in plan space
- `offset`: `normal · x = offset`
- `start`/`end`: 2D endpoints
- `residual_rms`: metres, spread of inliers about the fitted line
- `observed_span`: along-wall extent actually seen
- `height_range`: height extent of the supporting points. **Not** a measure
  of how tall the surface is — `wall_band_mask` clips every candidate to the
  same band before RANSAC runs, so this saturates at the band limits for
  real walls and low furniture alike. Do not filter on it.

**In `estimate_horizontal_frame()`, at the `vertical_enough = magnitude >
0.85` line:** normal lies within ~32° of horizontal.

**In `wall_band_mask()`, at the `vertical = np.abs(normals @ up) < 0.35`
line:** normal within ~70° of horizontal.

**In `extract_walls()`, before the contiguous-runs loop:** a single RANSAC
line spans every collinear wall in the building — opposite sides of a
corridor share an offset. Split into runs that are actually contiguous
before accepting anything.

**In `_ransac_line()`, at the `if length < 0.3` check:** too short a
baseline to define an orientation.

**In `merge_collinear()`, before the `ordered = sorted(...)` line:**
strongest first — support, then length. The winner of each family keeps its
own geometry, so it must be the best-evidenced plane, not merely the longest
fragment.

**In `merge_collinear()`, at `continue` after the normal-direction check:**
different direction, or the opposite face.

**In `merge_collinear()`, before the `separation = max(...)` block:**
separation is measured as the candidate's greatest distance from the
target's line, not as a difference of offsets. Offsets are only comparable
between exactly parallel lines: within the 15 degrees this tolerance allows,
two segments can share an offset at their midpoints and still diverge by
half a metre at their ends. Endpoint distance is what "the same surface"
actually means, and it subsumes the parallel case.

**In `merge_collinear()`, before `target.tags.append("clutter-in-front")`:**
the tag lands on the wall that survives, so it must say something true about
*that* wall: something parallel stood in front of it, which is also why part
of it is occluded.

**In `merge_collinear()`, at `break` after the clutter-tag append:** clutter
in front of `target`: drop it.

**In `_absorb()`, before the `seen = (...)` block:** observed span is
tracked in the merged frame so `inferred_fraction` still reports how much of
the combined wall was actually seen.

**In `_ray_crosses_wall()`'s return statement:** `(t > 0.02) & (t < 0.98)`
is strictly between camera and the point; `(s > margin_fraction) & (s < 1 -
margin_fraction)` is within the wall's solid span.

**In `filter_occluded_walls()`, at `kept.append(wall)` when `near.sum() <
min_points`:** too little evidence either way: fail open.

**In `_segment_intersection()`, at the `if abs(denominator) < 0.15` check:**
less than ~9° apart: no meaningful corner.

**In `resolve_crossings()`, before the final tag-cleanup loop:** indices are
deliberately left alone — they are identities that per-wall drift
measurements and surface grids are keyed by, not positions.

**In `snap_corners()`, before the `if not (-max_extension <= u_other...)`
check:** the corner must lie on (or just beyond) the partner too.

**In `snap_corners()`, before the `end_name, adjustment` loop's trim check:**
positive adjustment extends the wall, negative trims it.

**In `snap_corners()`, at `if remaining < 0.3`:** would collapse the wall
onto its neighbour.

**In `snap_to_frame()`, before the `offset = float(normal @ wall.midpoint)`
line:** keep the wall where its points are — re-derive the offset from the
midpoint so snapping rotates the line without translating it.

---

## pipeline/rooms.py

**Module purpose:** Segment a capture into rooms and build the adjacency
graph. Rooms are derived from the floor's free space, not from the
trajectory: a capture may start in a hallway outside the unit, wander back
through a room it already visited, or cover one room from two directions.
Watershed over the free-space distance transform cuts at the narrow necks —
doorways — which is where a floor plan should be cut.

**`PlanGrid` fields** (folded into the class docstring in code, reproduced
here for traceability):
- `origin`: plan coordinate of cell (0, 0)
- `occupied`: wall evidence
- `free`: observed floor

**`Room` fields** (documented via a newly created class docstring in code,
reproduced here for traceability):
- `area`: m², from the segmented floor cells
- `centroid`: plan coordinates
- `polygon`: (N,2) plan-space boundary

**In `build_plan_grid()`, at the `free = free + ndimage.grey_dilation(...)`
line inside the trajectory branch:** a footprint is strong evidence, so it
clears the floor threshold on its own; dilating covers the operator's body
width.

**In `segment_rooms()`, at the `seeds, _ = ndimage.label(distance > 0.55)`
line:** seeds are the cores of open areas; the 0.55 m radius is about half a
doorway, so a corridor neck never seeds a room of its own.

**In `segment_rooms()`, at the `labels[cells] = -(room.id + 1)` line:**
renumber to the compacted room ids.

**In `_grow_to_walls()`, at the `expanded = ndimage.grey_dilation(...)`
line:** dilate every label at once, then keep expansions only where the cell
was previously unclaimed — simultaneous growth means competing rooms meet in
the middle instead of one flooding the other.

**In `_boundary_polygon()`, at the `measure.approximate_polygon(contour,
tolerance=1.5)` line:** tolerance is in cells. `_rectify` discards sub-25 cm
edges, which is what actually removes the staircase, so this only needs to
be large enough to keep the vertex count manageable — and a larger value
would cut real corners, biasing every room's area low.

**In `_boundary_polygon()`, at the `polygon = grid.origin + (simplified +
0.5) * grid.resolution` line:** a cell's extent is [i, i+1) in contour
coordinates, so its centre is at i + 0.5. Omitting the half-cell shift
offsets every room boundary inward by 2 cm, which the 2% floor-area gate
cannot spare.

**In `_rectify()`, at the `merged: list[...] = []` declaration:** merge
consecutive co-directional edges, weighting each by its length so a long run
places the line and a leftover stub barely moves it.

**In `_rectify()`, at the `if len(merged) > 2 and abs(...) > 0.999`
check:** the first and last edges are also neighbours around the loop.

**In `_intersect()`, at the `if abs(denominator) < 1e-9` check:** parallel —
no corner to recover.

**In `_name_walls()`, at the `inward = wall.normal` line inside the sorted
loop:** point the normal into the room, then name the wall for the side it
sits on — a wall on the room's north side faces south.

---

## pipeline/poses.py

**Module purpose:** Trajectory refinement: keyframing, loop closure, and
pose-graph optimisation.

ARKit's VIO is locally excellent and globally drifting. Over a walkthrough
the drift is what smears a wall across its several visits, so the fitted
plane sits between them and every measurement taken from it inherits the
error. Replaying ARKit poses therefore cannot reach a 2 cm wall tolerance;
the trajectory has to be corrected against the sensor log itself.

The correction is a standard pose graph, and the design decision that
matters is which edges carry which claim:
- **Sequential edges come from ARKit**, weighted heavily. Its
  visual-inertial odometry fuses the whole sequence and is accurate to
  millimetres over the ~0.2 m between keyframes.
- **Loop-closure edges come from ICP**, weighted ~100x lower, between frames
  that are spatially near and temporally far. These carry the only
  information ARKit does not already have.

Using ICP for the sequential edges too is the tempting mistake, and it is
measurably worse: a small systematic bias in pairwise depth registration
compounds once chained over hundreds of keyframes. On recordings-1 it drove
metre-scale pose corrections and stretched the 2.99 m storey to 4.48 m,
while *improving* local wall agreement — a self-consistent, badly wrong
solution. `refine_trajectory` therefore also refuses corrections beyond
`max_total_correction` and falls back to the raw trajectory.

Open3D pose-graph conventions used throughout: a node's pose is
camera-to-world, and an edge (i -> j) stores `inv(pose_j) @ pose_i`.

**`ODOMETRY_INFORMATION` / `LOOP_INFORMATION` constants:** relative weights
of the two edge families. ARKit's relative pose between adjacent keyframes
is good to a few millimetres; a loop-closure ICP on this depth resolution is
good to a centimetre or two. Information is inverse variance, so the ratio
is roughly the square of that — odometry holds the trajectory's shape while
loop edges supply the global correction.

**`DriftReport` fields** (folded into the class docstring in code,
reproduced here for traceability):
- `rejected`: True when the optimisation was discarded as implausible

**In `find_loop_candidates()`, at the `directions =
bundle.poses[keyframes][:, :3, 2]` line:** camera looks down +Z in the
OpenCV convention this pipeline uses.

**In `refine_trajectory()`, at the `relative = np.linalg.inv(...)` line in
the sequential-edge loop:** sequential edges come from ARKit, NOT from ICP.
ARKit's visual-inertial odometry fuses the whole sequence and is excellent
over the ~0.2 m between keyframes; a pairwise ICP between two 256x192 depth
frames is not, and its small systematic bias compounds once chained.
Measured on recordings-1: substituting ICP for these edges warps the
reconstruction until the 2.99 m storey reads 4.48 m, with metre-scale pose
corrections, even though local wall agreement improves. ICP's job here is
loop closure, nothing else.

**In `refine_trajectory()`, at the `evaluation =
o3d.pipelines.registration.evaluate_registration(...)` call:** score
ARKit's relative pose rather than re-solving it — this is how well the two
frames already agree, which is the diagnostic worth reporting. Running ICP
here would cost ~250 full solves per capture and its answer would be
discarded anyway.

**In `refine_trajectory()`, at the `displacement =
np.linalg.norm(...)` line in the loop-candidate loop:** a loop edge should
be a correction, not a teleport. ICP between two views of a repetitive
interior can converge a room's width away (corridors and matching doorways
look alike); such an edge is confidently wrong and drags the whole graph
with it.

**In `refine_trajectory()`, at the `graph.edges.append(...)` call for loop
edges:** loop edges are weighted well below the odometry edges, so the
solution stays close to ARKit's trajectory and bends only as much as the
revisits actually demand.

**In `refine_trajectory()`, at the `rejected = float(corrections.max()) >
max_total_correction` line:** a correction this large is not a correction;
it means the graph found a self-consistent but wrong configuration. Falling
back to the raw trajectory is always recoverable — shipping a warped
reconstruction that still looks plausible is not.

**In `_slerp()`, at the `if dot < 0` check:** take the shortest arc.

---

## pipeline/scope.py

**Module purpose:** Turn fused damage regions into a defensible scope of
work. The mapping from detected damage to line items is where domain
knowledge lives, and it is deliberately not a proportional one. A 0.3 m²
mold patch does not produce a 0.3 m² line item: it produces removal of the
patch plus a margin, containment sized to the room, PPE per technician, and
HEPA passes over a much larger area. Water is similar — the flood cut is
driven by the *height* of the waterline, not by the stained area, and
baseboard comes off in whole runs.

Every quantity here traces to a rule in `rules.yaml` and carries its
`rule_id` and `source` into the output, so a reviewer can audit any number
back to the standard it came from — and so changing a rule changes the scope
without touching this file.

**`LineItem` fields** (documented via class docstring in code, reproduced
here for traceability):
- `basis`: how the quantity was derived, in words
- `derived_from`: damage region ids

**In `ScopeEngine`, section divider before `_item()`:** helpers.

**In `ScopeEngine`, section divider before `_flood_cut_height()`:** water.

**In `_water_items()`, immediately before the `if actions["drywall"] ==
"flood_cut" and wall_length:` branch:** wall — the flood cut is driven by
the waterline height, not the area.

**In `ScopeEngine`, section divider before `_mold_items()`:** mold.

**In `_mold_items()`, at the `grown = (region.width_extent + 2 * margin) *
(region.height_extent + 2 * margin)` line:** remediation extends past
visible growth on every side.

**In `_mold_containment()`, at the `barrier_area = room.perimeter * height +
room.area` line:** a poly enclosure is walls plus a ceiling over the work
area; the perimeter comes from the room's own footprint.

**In `ScopeEngine`, section divider before `_fire_items()`:** fire.

**In `ScopeEngine`, section divider before `concealed_flags()`:** concealed.

**In `ScopeEngine`, section divider before `build()`:** entry point.

---

## pipeline/damage/fusion.py

**Module purpose:** Fuse per-frame damage detections onto named surfaces.
Detecting a stain in one frame is the easy part. The hard parts, and what
this module exists for:

- **No double counting.** One stain is seen from dozens of viewpoints.
  Summing per-frame areas would multiply it by the number of views. Instead
  every observation votes into the *surface's* fixed UV grid, and the final
  area is the area of the cells that survived — independent of how many
  times the operator walked past.
- **Rejecting reflections.** A mirror or glass wall shows damage that is not
  on that wall. The depth behind a reflected pixel is the reflected scene's
  depth, so its back-projected point lands far from the mirror's plane and
  is discarded by the plane-agreement test. The same test throws out
  detections that landed on furniture in front of the wall.
- **Splitting across surfaces.** A stain spanning a corner belongs to two
  walls. Because assignment happens per pixel rather than per detection,
  each wall receives only its own portion.

Confidence per cell rises with *independent* agreement: how many separate
views, weighted by how squarely each viewed the surface.

**`MIN_INCIDENCE_COSINE` constant (value 0.26):** grazing views smear a mask
across the surface, so their votes are discounted by the cosine of the
incidence angle and dropped entirely past this limit. The value 0.26
corresponds to ~75 degrees off the surface normal.

**`COMBINED_CLASS_RATIO` constant (value 0.4):** a second class whose fused
weight is at least this fraction of the top class's weight is treated as
genuinely co-present rather than noise (e.g. fire damage from firefighting
water, a standard IICRC combined-loss scenario), and the region is
classified `"combined"` instead of picking one winner.

**`SurfaceRef` fields** (folded into the class docstring in code, reproduced
here for traceability):
- `kind`: one of "wall", "floor", or "ceiling"
- `normal`: 3D world unit normal
- `offset`: defines the plane as `normal . x = offset`

**`DamageRegion` fields** (folded into the class docstring in code,
reproduced here for traceability):
- `area`: m² on the surface

---

## pipeline/damage/masks.py

**Module purpose:** Refine detection boxes into pixel masks. A bounding box
overstates a stain's area — a diagonal tide line fills maybe half its box —
and area is what the scope's quantities are built from, so the box has to
become a mask before anything downstream can use it.

SAM 2 via Replicate is the accurate path; GrabCut is the local fallback,
used whenever the API is unavailable so the pipeline degrades instead of
failing. Which one ran is recorded on the mask, because it changes how much
the area should be trusted.

**`RefinedMask` fields** (documented via class docstring in code, reproduced
here for traceability):
- `mask`: bool, full frame resolution
- `method`: one of "sam2", "grabcut", or "box"
- `area_fraction`: share of the box the mask fills

**In `refine()`, at the bare `pass` inside `except Exception:` around the
`_sam2` call:** fall through to the local path rather than fail the run.

*(Note: `scope.py`'s `ConcealedFlag` has neither a class docstring nor field
comments in the original code — left as-is. `damage/fusion.py:285` keeps a
`#` inside an f-string, `f"{surface.key}#{label}"` — that is the region-id
format, not a comment.)*

---

## pipeline/damage/vlm.py

**Module purpose:** Per-keyframe damage detection and classification with a
vision model.

The model is asked for restoration-domain judgements (IICRC S500 water
Category and Class, fire soot/char/consumed, mold condition) rather than
generic labels, because those are what the downstream scope rules consume.

Two properties of this stage matter more than raw detection quality:
- It must argue against itself. The evaluation scenes deliberately contain
  shadows that look like soot, dry surfaces that look wet, mirrors, and
  stains spanning two walls. The schema therefore requires a
  `distractor_considered` field per detection — a detection that cannot say
  what else it might be is usually the one that is wrong.
- It must be replayable. Every response is cached by image content, so a
  rerun costs nothing and a live demo does not depend on the network.

Single-frame output is deliberately treated as a *hypothesis*. Nothing here
is trusted until `damage.fusion` has confirmed it across views on a real
surface.

**`PROMPT_VERSION` constant:** v4 — bbox is `{x0,y0,x1,y1}`, not a 4-array
(the API rejects `minItems`>1).

**`FURNITURE_CLASS` constant:** not a damage class. Opt-in via
`DamageAnalyzer(include_furniture=True)` — lets a run confirm the model is
actually resolving objects in the frame (a real object-detection signal)
independent of whether it finds damage. Regions tagged "furniture" are
dropped before fusion/scope in `cli.py` and never reach `result.json`, so
this can't leak into the schema-checked output.

**In `RESPONSE_SCHEMA`, at the `"bbox"` property definition:** a 4-element
array would be the natural shape, but the structured-output API rejects
`minItems`/`maxItems` values other than 0 or 1 — there is no schema-level
way to require exactly 4 entries in an array. An object with four required
numeric fields gets the same guarantee (every region really does have four
coordinates) through a mechanism the API actually supports.

**`Detection.bbox` field** (folded into the class docstring in code,
reproduced here for traceability): pixels, `(x0, y0, x1, y1)`.

**In `DamageAnalyzer.__init__()`, at the `max_edge: int = 4096` default:**
4096 is a safety ceiling, not a target — comfortably above any phone/ARKit
video resolution in practice (this device: 1920x1440), so real captures
pass through `_encode` at their native resolution untouched. Anthropic's
vision API accepts far larger images than this; the cap exists only to
bound cost/latency if a future capture source turns out to be unexpectedly
huge.

**In `_cache_path()`, before the `prompt_digest = ...` line:** hashing the
effective prompt text (rather than relying solely on a hand-maintained
`PROMPT_VERSION`) means any edit to `SYSTEM_PROMPT` or
`FURNITURE_PROMPT_ADDENDUM` self-invalidates just the responses whose
wording actually changed — no version bump to remember, and no collateral
invalidation of caches whose prompt text didn't move.

**In `analyze_frame()`, at `except Exception as exc:`:** network, rate
limit, refusal handling below.

**In `_to_analysis()`, before the `if target_shape is not None...` block:**
uniform, rotation-independent rescale from the original (possibly full
native res) frame down to the caller's target pixel grid — same idea as
`ingest.py`'s own depth/rgb intrinsics scaling, since depth is a uniform
downscale of the full-res video.

**In `_to_analysis()`, before the `sent_box = (...)` block:** normalised
[0, 1000] -> pixel coords in the sent (rotated, resized) image, then undo
the resize.

---

## pipeline/cli.py

**Module purpose:** One command per capture: `python -m pipeline run
<capture_dir>`.

**In `run()`, before the `sample_world_points_with_origin(...)` call:**
provenance-tagged points (with the camera origin each was observed from)
are needed by both wall extraction below (occlusion filtering) and wall
refinement further down (per-visit offset refit, drift). Sampling once and
threading the result through both avoids paying for the back-projection
twice.

**In `run()`, before the `refit_wall_offsets(...)` call:** drift is
measured BEFORE extents change — crossing resolution and corner snapping
move endpoints, never offsets, so they cannot alter a wall's true visit
spread; they only shrink the association windows and starve the estimator.

**In `_damage_pass()`, before the overlay-directory cleanup loop:** this
function is the sole owner of these two subdirectories — clear them up
front so a rerun's output never carries forward a stale image from a
previous run's (possibly different) findings. Without this, a run that
finds nothing leaves an old `damage_overlays/*.jpg` sitting next to a
`result.json` that says `damage: []`, which is exactly the kind of
inconsistency that shouldn't be discoverable mid-defense.

**In `_damage_pass()`, before the `furniture_frames = []` declaration:**
"furniture" is a diagnostic-only class (see `damage/vlm.py`) that confirms
the VLM is resolving objects in the frame at all. It is kept entirely
separate from real detections: never fused, never scored, never written
into `result.json` — just counted and rendered to its own overlay dir so a
run can be sanity-checked visually.

**In `_damage_pass()`, before the `rotations_by_frame = {}` declaration:**
frame-level (not detection-level) — shared by the real-damage and
furniture-diagnostic overlay renders below, since both draw on the same
underlying frames.

**In `_damage_pass()`, before the `analyzer.analyze_frame(...)` call:**
detection quality depends on real resolution — the depth-aligned `color[]`
is the depth raster's resolution (256x192 on this device, a ~56x
pixel-count cut from the native video), which the VLM's own `frame_notes`
routinely complained about before this. Send the full native frame instead,
and rescale the returned boxes back down to depth resolution
(`target_shape`) since that's the grid `refine()`/fusion below actually
index into.

**In `main()`, at the `--max-depth` argument default:** 3.5 m is the knee
measured by `tools/depth_bias.py` — ARKit depth is well behaved below it and
reads systematically far above it (+11.6 mm at 5.4 m). `tools/gating_sweep.py`
confirms the trade is nearly free: 8.2% less drift for 0.7% less wall
coverage.

---

## bench/run.py

**Module purpose:** Score pipeline output against laser ground truth and
incumbent scans.

Ground truth is entered once per property as a CSV of laser measurements;
this script matches each row to the pipeline's corresponding measurement,
reports every accuracy gate, and fits the conformal scale that calibrates
the intervals.

Usage:
```
python bench/run.py --result out/rec1/result.json --truth bench/gt_home.csv
python bench/run.py --result ... --truth ... --fit-calibration
```
(See `docs/benchmarking-and-usage.md` for the full flag reference, including
the reference-based gates added later in the project.)

**`GATES` dict:** each gate is `(label, predicate over the error record,
target fraction)`.

**In `_head_to_head()`, at the `margin = 0.002` line:** within 2 mm counts as
a tie.

*(Note: `argparse.ArgumentParser(description=__doc__)` used to source its
`--help` description from this module docstring; now that the docstring is
gone, the description is a literal string at the call site instead —
argparse `description=`/`help=` text is functional runtime output, not
documentation-comments, so it correctly stays in the code rather than
moving here. The same applies to every `tools/*.py` script below.)*

---

## tools/ablate.py

**Module purpose:** Run the report's ablations and print the error budget.

Each ablation toggles exactly one stage and re-measures drift as wall
revisit spread — the metric that actually tracks the wall-length gate,
unlike point scatter which mostly measures sensor noise.

Usage:
```
python tools/ablate.py ../recordings-1 --out out/ablations.json
```

**In `main()`, before the loop-closure ablation loop:** without loop edges
the graph holds only ARKit odometry, so it should reproduce the raw
trajectory — that row is the control proving the optimiser is not moving
anything on its own.

---

## tools/conv_test.py

**Module purpose:** Which pose convention actually aligns consecutive
frames? (This is the empirical scoring script `pipeline/ingest.py`'s
"Coordinate conventions" section refers to.)

**Before the `raw = poses_cv @ np.linalg.inv(ARKIT_TO_CV)` line:** raw ARKit
poses (undoes the flip the pipeline applies).

**Before `def variants(i)`:** candidate conventions — how to map the stored
rotation+translation to camera-to-world in the OpenCV convention.

**In `variants()`'s returned dict, at the `'c2w_flip'` entry:** this is the
current assumption (the one the rest of the pipeline actually uses).

**Before the `from scipy.spatial import cKDTree` line:** for each variant,
transform two nearby frames to world and measure how well they overlap
(median nearest-neighbour distance) — the correct convention should score
small.

---

## tools/depth_bias.py

**Module purpose:** Attribute wall-position error to its physical causes.

The trajectory ablation showed that loop closure barely moves the residual
and that odometry-only reproduces ARKit exactly, which rules out trajectory
drift as the dominant term. The remaining candidates are properties of the
*depth sensor*, and they are separable because each predicts a different
signature:
- **range** — a depth scale or offset error puts a wall further away the
  further you stand from it, so the residual trends with range.
- **incidence** — at grazing angles the beam footprint smears across the
  surface and multipath grows, so the residual trends with `|cos|`.
- **confidence** — if ARKit's own low-confidence returns carry the error,
  the residual separates by confidence level.

This fits each trend on real walls and reports how much of the error each
explains, then tests whether correcting the dominant one actually shrinks
the per-visit spread that the wall gate cares about.

Usage:
```
python tools/depth_bias.py ../recordings-1
```

**In `analyse()`, at the `plan = frame.to_plan(data["world"])` line:**
`to_plan` is a pair of dot products against unit axes through the origin, so
it maps direction vectors as well as points.

**In `analyse()`, at the `residual = signed[near] - np.median(signed[near])`
line:** re-centre on this wall's own points so the comparison is
within-wall — an absolute offset would just measure where the wall is.

**In `analyse()`, at the `view_sign = np.sign(...)` line:** sign the
residual along the view direction, so "further away" is positive for every
wall regardless of which way its normal points.

---

## tools/gating_sweep.py

**Module purpose:** Does tighter depth gating actually buy accuracy?

`depth_bias.py` found two exploitable signatures: ARKit depth reads
systematically far beyond ~3.5 m, and its high-confidence returns are far
tighter than its medium ones. Both suggest discarding data to gain
accuracy — but discarding data also removes support from the plane fits and
shrinks coverage, so the trade has to be measured rather than assumed.

Usage:
```
python tools/gating_sweep.py ../recordings-1
```

---

## tools/grav_test.py

**Module purpose:** This is the empirical scoring script
`pipeline/ingest.py`'s `DEVICE_TO_CAMERA` constant refers to — it recovers
which device-frame-to-camera-frame mapping actually turns the accelerometer
into a constant world vector (see `docs/additional_information.md`'s
`pipeline/ingest.py` section for how that result is used).

**Before the `j = np.clip(...)` line:** nearest pose per IMU sample.

**Before the `world = np.einsum(...)` line:** device-frame accel to world —
try the identity mapping device-to-camera first.

**Before the `cands = {...}` dict:** the iPhone device frame and the OpenCV
camera frame differ by a rotation; try the four 90-degree rotations about Z
plus the standard portrait mapping.

---

## tools/make_report.py

**Module purpose:** Generate the benchmark report from pipeline outputs.

Every number in the report is read from `result.json`, the ablation file,
or the benchmark summary — never transcribed by hand — so re-running the
pipeline regenerates a report that still matches the code.

Usage:
```
python tools/make_report.py --results out/rec1/result.json out/rec2/result.json \
    --ablations out/ablations_rec1.json --out report/benchmark.md
```

**`GATE_TABLE` constant:** the published gates, as `(metric, gate text,
stretch text)`. Populated from a benchmark summary when one is supplied to
`--benchmark`; otherwise every row is marked as needing ground truth.

---

## tools/scope_demo.py

**Module purpose:** Exercise the scope engine, and show a rule change as a
line-item delta.

Track C is testable without a capture: it is a pure function from damage
regions to line items. This builds representative regions, prints the
scope, then re-runs with one rule changed and diffs the result — the
"change the flood cut from 30 cm to 60 cm and show the delta across all
rooms" modification.

Usage:
```
python tools/scope_demo.py
python tools/scope_demo.py --set water.flood_cut.base_height=0.60
```

**In `print_delta()`, before the final `for item in after:` loop:** the
basis string is what makes the change defensible in review.

---

## tools/view_plan.py

**Module purpose:** Render a `result.json` floor plan to PNG for quick
inspection.

The shipped plan is SVG (`pipeline/export.py`); this is the raster view
used while iterating, since it needs no cairo and can be eyeballed
directly.

Usage:
```
python tools/view_plan.py out/rec1/result.json
```
