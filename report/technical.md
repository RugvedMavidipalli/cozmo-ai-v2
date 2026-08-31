# Capture-to-scope: technical report

Handheld walkthrough → dimensioned floor plan, per-surface damage intelligence,
and an estimator-ready scope of work.

> **Status markers.** ✅ = measured on the sample captures. ⏳ = requires laser
> ground truth not yet collected. Nothing here is estimated by hand — every ✅
> number comes from `python -m pipeline run` or `tools/ablate.py`.

---

## 1. Architecture

```
capture ─► ingest ─► pose graph ─► TSDF fusion ─► gravity ─► walls ─► rooms ─► plan + 3D
             │       (ARKit odom                            (2D RANSAC) (watershed)
             │        + ICP loops)                              │
             │                                                  ▼
             └─► keyframes ─► VLM ─► SAM ─► project ─► per-surface UV grid ─► regions
                                                                                  │
                                                       rules.yaml ─► concealed + scope
```

Six modules own one invariant each. `ingest` owns sensor conventions;
`poses` owns the trajectory; `geometry`/`planes` own metric surfaces; `rooms`
owns topology; `occupancy` owns *position on a surface*; `scope` owns domain
logic. Damage never touches geometry code — it projects into the grid that
`occupancy` already built for openings and occlusion. Full stage-by-stage
detail: `docs/architecture.md`.

**Stack.** Python 3.11, Open3D 0.19 (TSDF, ICP, pose graph), OpenCV, SciPy,
scikit-image. Damage detection calls the Anthropic API (`claude-opus-5`) and
Replicate (`meta/sam-2-large`); both are disclosed, both cache to disk by
content hash, and both have local fallbacks. Runs on an Intel i9 with no GPU.

---

## 2. Sensor conventions: measured, not assumed

Two conventions had to be established before any geometry was meaningful, and
**both textbook answers were wrong**. Each produced confident, plausible-looking,
geometrically meaningless output.

| Question | Textbook answer | Measured answer | Cost of the wrong choice |
|---|---|---|---|
| Pose handedness | Apply ARKit→OpenCV flip `diag(1,−1,−1,1)` | Poses are *already* camera-to-world in OpenCV convention | Frame alignment 4.2 cm → **25.9 cm** ✅ |
| IMU→camera rotation | Identity | `rotZ(−90°)` | Gravity consistency 0.99 → **0.59**; storey height read **10 m** ✅ |

`tools/conv_test.py` scores every plausible pose convention by nearest-neighbour
agreement between nearby frames' point clouds. `tools/grav_test.py` scores every
device→camera rotation by whether the accelerometer resolves to a *constant*
world vector once rotated through the poses. It scores 0.992; the runner-up
scores 0.87. Both scripts run on any new capture source, and the pipeline
carries `gravity_consistency` into `result.json` and warns below 0.9 — a
capture from a different app or device announces a convention mismatch rather
than silently producing a floor plan of a wall.

**Gravity is then refined against geometry**, not trusted from the IMU alone:
the accelerometer fixes direction to within a degree or two, and a degree of
tilt displaces a wall footprint by over a centimetre across a storey — more
than the 2 cm budget can absorb. The IMU axis seeds a cone filter over surface
normals, and the floor/ceiling normals inside that cone are averaged. On
recordings-1 the refinement moves the axis 0.04° and recovers a 3.04 m storey
with the camera 1.61 m above the floor. ✅

---

## 3. Reconstruction and the error budget

### 3.0 Resolving competing surfaces

Sequential RANSAC produces two kinds of duplicate needing opposite treatment.
**Fragments of one surface** — a wall split by a doorway, or seen on two visits
a couple of centimetres apart — are merged, since both are evidence of the
same plane and its extent. **Parallel clutter** — door reveals, trim, cabinet
fronts a few centimetres in front, of which one wall on recordings-1 spawned
five within 23 cm — is *suppressed*, not merged: averaging would drag the wall
off its true position. Since RANSAC takes the largest consensus set first, the
dominant plane in such a family is the wall.

Separation is measured as the candidate's greatest distance from the target's
line, not a difference of plane offsets — within a 15° tolerance two segments
can share an offset at their midpoints and diverge by half a metre at their
ends. Runs are also gated on **observed coverage** (share of a run's own
surface area actually seen), which is scale-free: real walls score 8–89%,
spurious slivers on the same infinite line score 1–3%.

| recordings-1 | before | after |
|---|---:|---:|
| walls | 64 | **43** |
| same-facing duplicate pairs | 24 | **0** |
| minimum support density | 60 pts/m | **352 pts/m** |
| drift median | 23.3 mm | **12.7 mm** |

*Caveat:* part of the drift improvement is a selection effect — the
suppressed surfaces were the worst-fitting ones, so removing them improves
the median of what remains as well as the geometry itself. The honest claim
is that the *wall set* is clean, not that sensor accuracy improved 45%.

One field is a documented trap: `WallSegment.height_range` looks like it
should separate walls from low furniture, but `wall_band_mask` clips every
candidate to the same height band before RANSAC runs, so it saturates at the
band limits for walls and countertops alike.

### 3.0a Occlusion filtering: a wall behind a wall

`merge_collinear` removes surfaces too *close* to the camera (furniture in
front of a wall); `filter_occluded_walls` handles the opposite geometry — a
candidate plane too *far* away, whose points could only exist if light passed
through an already-better-supported wall to reach them. It traces the ray
from each supporting point back to its observing camera and drops the wall if
a majority of those rays cross a stronger wall's solid span.

Two drops on recordings-1 were traced by hand rather than trusted on faith,
since both had real support (2,942 and 5,595 pts/m — denser than several kept
walls): one turned out to be a sliver of an accepted wall's own far face,
grazed through a doorway edge and fitted as if freestanding — too thin to be
a real partition, a case `merge_collinear` can't catch since opposite-facing
near-parallel planes are normally *its* signature for a shared wall. The
other was a stray fragment with no such explanation. Full derivation:
`docs/track-a-reconstruction.md`.

| capture | walls before | occlusion-inconsistent removed |
|---|---:|---:|
| recordings-1 (current default config) | 38 | **5** ✅ |
| recordings-2 *(earlier measurement, unverified against current build)* | 52 | 15 |

The claim this stage supports is a cleaner, non-duplicated wall *set* — not a
guaranteed drift win on every capture; recordings-2's original measurement
showed drift *rising* after filtering (10.3 → 13.9 mm), since drift is only
measured over walls that survive to be revisited, and removing genuine
observations can shrink that population unfavourably as easily as favourably.

### 3.0b Precision refinement: visits, crossings, corners

Three refinements close the gap between "a plane was fitted" and "a wall was
measured," each owning one error source: ✅

**Offsets are re-placed at the median of per-visit offsets**, not the pooled
mean — a pooled fit lets the visit that lingered longest, not the one that
measured best, decide where the wall is. This cut median drift 11.8 → 9.9 mm
on recordings-1 and put both captures under the 2 cm wall-gate budget.

**Impossible crossings are resolved.** An interior–interior intersection has
exactly two causes, separable by geometry: a T-junction overshoot (trim the
short overhang back to the junction) and clutter cutting a wall at a shallow
angle (drop the weaker surface — neither overhang is short). recordings-1 had
13 such crossings; it now has 0.

**Endpoints snap to corner intersections** — the corner, not wherever the
last inlier fell, is where a tape measure is hooked, and only after this step
do emitted lengths mean what the interval model assumes. Median
endpoint-to-corner gap went 22.5 cm → 0.0 cm, 41 of 72 endpoints snapped
exactly; the rest are free scan-boundary ends covered by the inferred-span
machinery.

Drift is measured *before* extent edits, since trimming/snapping move
endpoints, never offsets. (An early version of the visit-offset fit appeared
to worsen drift 2.5× from an estimator bug, not a geometry bug — full story
in `docs/track-a-reconstruction.md`.)

### 3.1 Why walls are fitted in 2D

Once gravity is known, a vertical surface has **one** free orientation and
**one** offset. Fitting in the gravity-aligned horizontal projection removes
two degrees of freedom 3D RANSAC would otherwise estimate from noise, and
pools every point across the wall's full height into a single fit. Wall
extent then comes from **intersecting neighbouring wall lines**, never from
observed point spread — furniture, doorways and grazing dropout all truncate
what the sensor sees, while the corner where two planes meet is where a tape
measure goes.

### 3.2 Measuring drift honestly

> Scatter is dominated by depth-sensor noise, and a plane fit averages it
> down by √N. Drift does the opposite: it displaces an entire visit's points
> *coherently*, so the fit lands between the visits and the scatter barely
> moves. **Scatter is nearly blind to the error that actually breaks the gate.**

`pipeline/drift.py` groups each wall's supporting points by *when* they were
observed, splits into visits separated by >3 s, and reports the spread
between per-visit plane offsets — the drift contribution to wall position,
directly comparable against the 2 cm gate.

On raw ARKit poses: **median 21.8 mm, p90 46.1 mm** over 24 revisited walls
(recordings-1) ✅ — above the gate, which is why refinement is not optional.

### 3.3 The pose graph, and a mistake that looked like success

| variant (recordings-1) | drift median | p90 | storey height | max correction |
|---|---:|---:|---:|---:|
| raw ARKit | 21.8 mm | 46.1 mm | 2.990 m | — |
| odometry only, no loops *(control)* | 21.8 mm | 46.1 mm | 2.990 m | 0.0 cm |
| odometry + loop closure | 21.4 mm | 46.5 mm | 2.922 m | 24.5 cm |
| **ICP sequential edges** *(rejected)* | **7.3 mm** | 29.6 mm | **4.482 m** | **324 cm** |

The last row is instructive: building sequential edges from pairwise ICP cut
measured drift 66% — and destroyed the reconstruction, stretching a 2.99 m
storey to 4.48 m. A small systematic bias in registering two 256×192 depth
frames compounds once chained over 253 keyframes; the optimiser found a
self-consistent, badly wrong configuration, and the drift metric *approved*
of it because every wall agreed with itself in the warped space — a metric
that improves while the artefact degrades is worth more than one that only
ever agrees with you.

The fix follows from what each sensor is good at: **sequential edges come
from ARKit** (weighted heavily — accurate to millimetres over the ~0.2 m
between keyframes, better than pairwise ICP at this resolution);
**loop-closure edges come from ICP**, weighted ~100× lower, carrying the only
information ARKit doesn't already have. The control row proves the optimiser
moves nothing on its own (0.0 cm with odometry edges alone). Loop candidates
are gated on view direction, ICP fitness, then displacement — a correction,
not a teleport — and `refine_trajectory` refuses its own output past a 75 cm
correction, falling back to raw ARKit.

### 3.4 Where the error actually comes from

With the optimiser honest, loop closure buys **1.8%**. That's the important
negative result: on these captures, trajectory drift is not the dominant
error term, and further pose engineering would have been wasted effort.

The residual was attributed to the depth sensor instead (`tools/depth_bias.py`).
Depth scale error trends with range, beam smear with incidence, sensor
self-assessment with confidence — separable signatures. Pooling 1.46 M
observations across 30 walls, each re-centred on its own wall: ✅

| range | 0.7 m | 1.5 m | 2.4 m | 3.4 m | 4.0 m | 5.4 m |
|---|---:|---:|---:|---:|---:|---:|
| median residual | −2.2 mm | +0.7 mm | −1.5 mm | −3.3 mm | **+4.3 mm** | **+11.6 mm** |

**Range bias is real and nonlinear** — well-behaved to ~3.4 m, then
systematically far, reaching +11.6 mm at 5.4 m; a linear fit understates it,
the effect is a knee. **Incidence showed no clean trend**, so beam smear is
not a leading term here and damage-fusion's incidence weighting is justified
on mask quality, not depth accuracy. **ARKit's own confidence is informative
by spread**: IQR tightens from 57.8 mm (low confidence) to 32.2 mm (high),
and high-confidence returns are the overwhelming majority of the data, so
tightening the gate costs less coverage than it first appears.

Ungated depth changed drift only 3.3% on recordings-1 ✅ — an undamaged,
well-lit property; the adversarial held-out room (mirrored closet, glass
shower) is where confidence gating should separate sharply.

### 3.5 Gating sweep — acting on the findings

Discarding data to gain accuracy is a trade, not a free win. Measured, not
assumed (`tools/gating_sweep.py`, recordings-1): ✅

| min confidence | max depth | drift median | wall coverage | walls | Δ drift | Δ coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 m *(old default)* | 22.8 mm | 116.6 m | 31 | — | — |
| **1** | **3.5 m** *(new default)* | **20.9 mm** | **115.8 m** | **35** | **+8.2%** | **−0.7%** |
| 2 | 3.5 m | 22.4 mm | 90.1 m | 29 | +1.7% | −22.7% |
| 2 | 2.5 m | 13.5 mm | 57.8 m | 19 | +40.6% | −50.4% |

**`--max-depth 3.5` is nearly free** — 8.2% less drift for 0.7% less
coverage — and lands exactly where §3.4 predicted the bias knee; it's the
default. Tightening confidence to level 2 costs a fifth of the wall coverage
for less gain, so medium confidence stays default on well-lit properties; the
flag exists because the adversarial room is where that should reverse. The
last row is tempting and rejected: 40% less drift but half the wall coverage
— an unreconstructed wall cannot be measured accurately, it simply isn't
there.

---

## 4. Damage: detection, fusion, and not double counting

### 4.1 Single-frame output is a hypothesis, never a result

The VLM is prompted as a restoration estimator and required to argue against
itself: every detection fills a `distractor_considered` field naming the most
plausible benign explanation and why it was rejected. The prompt encodes the
specific discriminations the evaluation targets — soot deposits track airflow
and heat sources while shadows track lighting geometry; reflections move with
the camera; dark stone and polished tile read as wet; scuffs and patina are
not restoration damage. Nothing from a single frame reaches the output;
confirmation is geometric.

Two upstream fixes this pass measurably changed detection quality: frames are
now rotated to the model's natural viewing orientation from measured gravity
(not a hardcoded guess) and sent at full native resolution rather than the
~56×-smaller depth-raster crop previously shared with mask/fusion alignment.
Before this, a ceiling-light lighting artifact was misread as a water stain
when fed to the model sideways; re-run correctly oriented, the same frame
produces "no damage called." Before/after evidence: `docs/track-b-damage-intelligence.md`.

### 4.2 The fusion invariant

Masks back-project through depth into the **surface's** fixed UV grid, where
they vote. Three properties follow:

1. **No double counting.** Final area is the area of cells that survived, not
   a sum over observations. Measured: one patch viewed from 2→12 viewpoints
   produces one region whose area *plateaus* at 6.35 m²; summation would have
   given ~12×. ✅
2. **Reflections are rejected by geometry, not a classifier.** A stain seen in
   a mirror back-projects to the *reflected* scene's depth, far from the
   mirror plane, and fails plane agreement: 47% of true-depth pixels land on
   a surface, versus 6% at doubled depth and 0% at tripled. ✅
3. **Stains spanning a corner split correctly**, because assignment is per
   pixel, not per detection.

Votes are weighted by incidence angle (grazing views discounted, dropped past
75°), and a region must be seen from **≥2 independent viewpoints** — the
single most important parameter in Track B, since view-dependent artefacts
appear in one view and vanish in the next while a real stain persists.
Regions that are genuinely co-dominant between two classes (e.g. water damage
from firefighting a fire) are now classified `"combined"` rather than forced
into one winner, and dispatched through every relevant scope handler.

---

## 5. Scope: quantities that are not proportional to area

> A 0.3 m² mold patch does not produce a 0.3 m² line item.

It produces removal of the patch **plus a 30 cm margin on every side**,
containment sized to the **room** (perimeter × height + ceiling), a HEPA air
scrubber sized to the contained **volume**, PPE **per technician per day**,
and two HEPA passes. Water behaves the same way: the flood cut is driven by
the **height** of the waterline, not the stained area, and baseboard comes
off in **whole runs**. A ceiling surface has no waterline to cut above, so it
gets its own quantity path sized to affected area instead — a real gap this
pass, previously a ceiling region silently produced only an antimicrobial
line item and nothing else.

Every rule lives in `rules.yaml` with its IICRC source (S500 water, S520
mold, S700 fire); rules that are our own simplification say so in a `note`
field. **Live modification is a config edit** — changing the flood cut from
30 cm to 60 cm is one line in `rules.yaml` plus a re-run, no code change.

Concealed damage is a separate rule set (`concealed:`) emitting a
probability, a rule id, and a rationale per flag — insulation behind
saturated drywall (p=0.72), pad and subfloor under wet carpet (p=0.85),
smoke in cavities and HVAC beyond visible soot (p=0.70), among eight rules
total. Probabilities are calibrated judgement, not measurements, and are
labelled as such.

The scope is now also exported as two CSVs (room/wall sketch geometry and
line items) alongside the JSON, in the estimator-convention units `rules.yaml`
already specifies — the structured, Xactimate-workflow-consumable format the
assignment asks for, which previously didn't exist. Full rule content and
export column layouts: `docs/track-c-scope-generation.md`.

---

## 6. Calibrated uncertainty

Every measurement ships an interval, built in two layers.

**Physical.** Plane-offset uncertainty combines fit scatter averaged over
supporting points with measured drift — and **drift is not averaged down**,
because it is coherent within a visit; treating it as averageable random
error is precisely what produces confident, wrong intervals. A wall length is
a difference of two corners, each an intersection of two planes, so four
plane offsets contribute; spans never observed widen the interval in
proportion to how much of the wall was inferred.

**Conformal.** A single scale factor is fitted against laser ground truth
from the empirical quantile of normalised errors — no distributional
assumption, and a few dozen measurements suffice. ⏳ Until that fit exists,
every output carries `"calibrated": false`, the floor plan prints *"intervals
uncalibrated"*, and the run warns. Overconfidence on a hard capture is an
automatic red flag; the honest state is to say the intervals are not yet
earned.

**Graceful degradation.** A video-only capture is designed to multiply every
interval by a measured no-LiDAR factor rather than failing or bluffing, but
the path is not yet wired into the CLI — `ingest.load_capture` currently
hard-fails on a capture with no depth or poses rather than degrading. This is
the single highest-risk open item (§8) since it's an explicitly scored gate
and a stated held-out test case.

---

## 7. Accuracy gates

<!-- GATE_TABLE -->

⏳ Gates require laser ground truth. `bench/run.py` scores every gate,
reports the error distribution, fits the conformal calibration, and runs
head-to-head against an incumbent scan. This pass added scoring code for 6 of
the 7 previously-uncovered gates, plus a room-overlap/adjacency
self-consistency check that needs no ground truth and passes today (5 rooms,
0 overlaps, 0 adjacency errors on recordings-1) — leaving only no-LiDAR
wall-length unscoreable until §6's fallback exists. Reference:
`docs/benchmarking-and-usage.md`.

**Runtime** ✅ — geometry only, recordings-1: ~145 s for 5 rooms (~29 s/room),
comfortably inside the 300 s gate and 90 s stretch. **With the real damage
pass** (default 40 frames, full resolution as of this pass): ~617 s total
(~123 s/room) — still inside the gate but with much less margin, and
structurally risky, not just slow: damage cost scales with `--damage-frames`,
not room count, so a 1–2 room capture carries the same absolute cost across
fewer rooms and could plausibly breach it. Intel i9-9980HK, no GPU.

---

## 8. Known failure modes

1. **Pairwise-ICP odometry warps space** (§3.3). Diagnosed, fixed by trusting
   ARKit for sequential edges, and guarded by a correction-magnitude rejection.
2. **Room polygons follow observed floor.** Furniture shadows and the
   sensor's poor floor coverage near walls bias area low. Mitigated by
   treating the operator's own path as floor evidence and growing rooms to
   their bounding walls; still the largest known source of floor-area error.
3. **Duplicate parallel wall fragments** from doorway interruptions and
   drift. Mitigated by `merge_collinear`, which compares normal *direction*
   so the two faces of a partition never merge.
4. **Occlusion detection over-triggers** on recordings-2, marking spans
   inferred that were merely viewed at a grazing angle. Conservative in the
   right direction (understates confidence) but noisy.
5. **Water Class (1–4) is rarely inferable** from single frames. Returned
   `null` rather than guessed.
6. **No-LiDAR fallback is unwired.** Depth estimation, scale anchoring, and
   widened intervals are designed but not in the CLI — see §6.
7. **Loop closure has limited leverage on open trajectories.** recordings-2
   ends 7.3 m from its start with few revisits; drift there is controlled by
   odometry quality alone.
8. **A sideways or low-resolution frame can fabricate damage.** Observed, not
   hypothetical: a ceiling light's lighting gradient, fed to the model at the
   wrong orientation, was read as a Category 2 water stain and produced real
   scope line items. Fixed (§4.1), but the lesson — VLM damage calls are only
   as trustworthy as the image geometry feeding them — has no general guard
   beyond re-verifying on more captures.
9. **Floor and ceiling damage is currently undetectable end to end.** Both
   surfaces exist (`build_surface_refs`), but the pipeline only allocates a
   fusion accumulator for walls — a detection assigned to the floor or
   ceiling has nowhere to vote and is silently dropped before it can reach
   the (now-fixed) ceiling scope logic in §5. Needs a horizontal-plane
   occupancy grid; not yet built.

---

## 9. With one more month and a team of two

**Weeks 1–2 — close the metric gap.**
Bundle-adjust planes jointly with poses instead of sequentially: make wall
planes first-class variables in the graph so a wall seen on three visits
becomes one constraint rather than three observations averaged afterwards.
Add place recognition (learned descriptors) instead of pose-proximity loop
proposal, which is what limits loop closure on open trajectories. Rectify
room polygons from the wall arrangement rather than floor pixels (failure
mode 2). Build the floor/ceiling occupancy grid so damage fusion can actually
reach those surfaces (failure mode 9) and wire the no-LiDAR fallback in
(failure mode 6) — both are designed, neither is built.

**Weeks 2–3 — damage quality.**
Fine-tune a segmentation head so masks stop depending on a hosted model — SAM2
has never actually run in this project for lack of a configured API token,
every mask to date is the GrabCut fallback. Build the adversarial set the
evaluation implies (mirrors, glass, shadows, wet-look-dry) as a regression
suite; failure mode 8 is exactly what it should catch automatically. Add
material classification (drywall/tile/carpet/wood), which most
concealed-damage rules currently assume rather than observe.

**Weeks 3–4 — estimator trust.**
Native Xactimate ESX export, building on this pass's CSV export as an
intermediate step. A review UI showing, per line item, the rule that fired
and the frames that produced it — the audit trail already exists
(`derived_from`, `rule_id`, `contributing_frames`), it needs a surface.
Calibration per capture modality and per property, not one global scale. A
real automated regression suite: everything this pass was verified by hand,
nothing guards it against a future regression.

**Continuously.** Every self-captured property enters the benchmark with
laser ground truth. The gate table runs in CI. The thing that decays fastest
in a system like this is not accuracy but the *honesty* of its intervals, and
only a growing ground-truth set keeps that in check.
