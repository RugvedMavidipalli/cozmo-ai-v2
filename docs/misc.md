# Gap Analysis vs. Cozmo AI Assignment Spec

A fresh, code-level audit of `cozmo-pipeline` against the assignment slides — deliverables,
Track A/B/C requirements, and all 14 accuracy gates. Every claim below is backed by a
file:line citation; this was verified directly against the current source, not carried
over from earlier discussion. Two things are true at once here: the engineering is real
and unusually honest about its own limits (loop closure ablations, drift analysis,
rejected approaches are all in the code and report) — and several requirements are
either unmet or measured with numbers that don't reproduce from current output. Both
need to be visible before the defense.

---

## Update — Tiers 1-3 of the remediation plan are complete

Everything below this line was the original audit and is left as written (including
counts/tables that have since changed) so the before/after is visible. What's since
landed, real code fixes and additions, all verified against real data and/or synthetic
fixtures with hand-checked expected values:

- Pruned `out/`, room-overlap self-consistency check (`rooms.check_no_overlaps`, wired
  into every run), the ceiling-surface scope bug (§4), and `"combined"` damage
  classification (§3) are all fixed.
- Floor/ceiling are now named 3D planes in `scene.glb` (§2's "3D representation" gap),
  door-derived adjacency `via` linking replaces the hardcoded `null` (§2's stitched-plan
  gap), and an Xactimate-consumable CSV export now exists (§4's "not done" export gap).
- `bench/run.py` gained: a ground-truth-free room/adjacency self-consistency gate (now
  runs on every `--result`, no `--truth` needed), a footprint-error scorer
  (`--footprint-reference <m2>`), and five reference-file scorers — damage
  classification macro F1, water Category/Class accuracy, damage segmentation IoU,
  concealed-flag recall/precision, and line-item recall — each taking a documented CSV
  reference format via a new flag. These are real, tested comparison logic, not stubs;
  they're just waiting on reference data that doesn't exist yet (intentionally not
  created this pass). See §5 below for the updated per-gate table.
- Still open: refreshing `report/benchmark.md`/`technical.md` against current numbers,
  and the no-LiDAR fallback (scoped-down "honest failure" version planned, not started).

### Update 2 — documentation pass, dead-code removal, one new gap found

This repo's documentation moved under `docs/` (this file included), a granular
per-track reference now exists (`docs/architecture.md`,
`docs/track-a-reconstruction.md`, `docs/track-b-damage-intelligence.md`,
`docs/track-c-scope-generation.md`, `docs/benchmarking-and-usage.md`), and the
comment-extraction rule from Update 1 (only class/method/function docstrings survive
in `pipeline/`) was extended to `bench/run.py` and every file in `tools/` — all
verified to still compile and, for `bench/run.py`, to still run correctly afterward.

Dead code removed (verified zero references anywhere in the repo before removal, and
a real pipeline run still produces identical output after): `geometry.py`'s
`plane_from_points`/`point_plane_distance` (superseded by `planes.py`'s own line-fit
logic, never actually called), `fuse.py`'s `backproject` (never called), and
`occupancy.SurfaceGrid.passthrough` (always zero, never read or written — opening
detection has always actually worked off silhouette holes in `hits` instead). Two
genuinely unused imports were also removed from `tools/conv_test.py`.

**One new gap found while writing `docs/track-b-damage-intelligence.md`, not caught
by the original audit:** `build_surface_refs` (`damage/fusion.py`) creates `SurfaceRef`s
for a floor and a ceiling alongside every named wall, but `cli.py::_damage_pass` only
allocates a `DamageAccumulator` for `surface.kind == "wall"`. A detection that
`assign_to_surfaces` correctly assigns to the floor or ceiling has nowhere to vote and
is silently dropped. In practice, **floor and ceiling damage is currently
undetectable end to end**, independent of and prior to the ceiling-scope fix in §4
below (that fix only corrects what happens *if* a ceiling region reaches `scope.py` —
today, none ever will, because none are ever fused in the first place). Fixing this
needs a horizontal-plane equivalent of `occupancy.build_surface_grid` (which is
tightly coupled to a `WallSegment`'s along-wall parameterization) plus wiring it into
`cli.py`'s surface/accumulator setup — not attempted this pass; flagging it as the
most important remaining Track B gap, ahead of the SAM2-never-ran item below it in §3.

---

### Update 3 — two real bugs found preparing for the live defense

**`scene.glb` contained two overlapping reconstructions.** `export_scene`
added the dense fused TSDF mesh as a `"reconstruction"` node *in addition to*
the named wall/floor/ceiling quads, so any viewer showed the clean wall boxes
with an unsegmented mesh sitting inside them — the "two scenes" a GLB viewer
made obvious. Fixed by dropping the mesh from the GLB entirely; the raw fused
surface is still available separately in `cloud.ply`, and `scene.glb` now
holds only the named surfaces the assignment actually asks for (57 MB → 24 KB
for `out/recordings-1`).

**`ANTHROPIC_API_KEY` was never actually loaded from `.env`.** `python-dotenv`
is a declared dependency and the README instructs `cp .env.example .env`, but
no file in `pipeline/`, `bench/`, or `tools/` ever called `load_dotenv()`.
Every real invocation of `python -m pipeline.cli run` silently lost the entire
damage pass — every frame returned `error="ANTHROPIC_API_KEY not set"` — unless
the key happened to already be exported in the calling shell. This is why
`out/recordings-1`'s checked-in `result.json` showed `0` damage regions with a
generic "40 damage frames failed analysis" warning that gave no indication the
key was the problem. Fixed with one `load_dotenv(REPO_ROOT / ".env")` call at
the top of `cli.py::main()`. Verified in a fully clean shell (`env -i`, no
`ANTHROPIC_API_KEY` pre-set) that a real run now completes the damage pass
without that warning, and `out/recordings-1` was regenerated with the fix —
its damage pass now genuinely ran (real API calls, ~572 s for 40 frames) and
genuinely found 0 fused regions rather than silently failing to try.

---

## 1. Deliverables checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | Running code, ≤15 min setup, one command, schema JSON, floor plans/3D/overlays | **Mostly done** — see §1 caveats below (no PNG/rasterized 3D view emitted; floor/ceiling not named planes) |
| 2 | Benchmark report | **Not scored** — every gate row in `report/benchmark.md` says "pending ground truth"; figures that *are* present (walls, runtime) are stale vs. current output |
| 3 | Technical report (≤8 pages) | **Over length, stale** — ~4,263 words (~8.5 pages at this density); does not mention any of this session's four fixes (overlay wiring, auto-orientation, full-res VLM input, confidence filtering); several stale figures |
| 4 | 3 self-captured properties, laser GT, incumbent scans | **Not met** — only 2 captures exist, neither has staged damage, no GT file filled in, no incumbent scans anywhere *(user has explicitly deprioritized this requirement — reported for completeness only)* |
| 5 | 5-minute screen recording | **Missing** — only raw capture footage (`rgb.mp4`) exists anywhere in the repo tree, no screen recording of the pipeline running |

---

## 2. Track A — Metric Reconstruction

| Requirement | Status | Evidence |
|---|---|---|
| Dimensioned 2D floor plan/room | **Done**, with a gap | Wall length/height/area/opening CIs all wired (`cli.py:428-485`, `uncertainty.py:102-159`). **Gap:** CLI writes only `floorplan.svg` + `scene.glb` — no PNG floor plan and no rasterized 3D *view* image ever gets produced. `tools/view_plan.py` makes one but is never called from the pipeline (`grep view_plan pipeline/*.py` → 0 hits). |
| Stitched multi-room plan | **Partial** | Room segmentation + adjacency graph real (`rooms.py:128-184, 469-484`). **Door connections never populated** — `cli.py:498` hardcodes `"via": None` on every adjacency edge, even though the schema and slide both describe adjacency as sharing a wall *or doorway*. **No room-overlap check exists anywhere** in `pipeline/` or `bench/` — the gate explicitly requires "no room overlaps." **Shared-wall thickness is claimed as scored** (`technical.md:451`) but no field carries it in `result.json`. |
| 3D representation, named planes | **Partial** | Walls are individually named GLB nodes (`export.py:426-428`). **Floor and ceiling are not** — `export_scene()`'s own docstring claims all three are emitted, but there are only two `add_geometry` calls in the file, neither is a floor/ceiling quad. The slide requires "every wall, floor **and ceiling**" to be a named plane. |
| Occlusion handling | **Done** | Occluded spans detected, emitted per wall, and widen the CI (`occupancy.py:203-226`, `cli.py:423-447`, `uncertainty.py:115-118`). Walls behind furniture are reconstructed via corner intersection, not observed span (`planes.py`). Self-documented known weakness: over-triggers on recordings-2. One dead field: `SurfaceGrid.passthrough` is declared and zeroed but never read or written anywhere else. |
| Confidence intervals, calibrated | **Partial — never calibrated** | The physical + conformal model is real and wired (`uncertainty.py`, `--calibration` flag). **`bench/calibration.json` has never been produced** — no ground truth exists to fit it from, so `calibrated: false` and `scale: 1.0` on every single run, confirmed live in `out/recordings-1/result.json`. |
| No-LiDAR fallback | **Not done — hard blocker at ingest** | `has_depth=False` is a real, wired concept downstream (`uncertainty.py:76,187-189` applies a 3× interval multiplier; `cli.py:531` emits `"video_only"`), but it is **unreachable**: `ingest.load_capture()` raises `FileNotFoundError` the moment `odometry.csv` or `depth/*.png` is missing (`ingest.py:169,174-176`), and `has_depth=True` is hardcoded in the only `CaptureBundle` constructor (`ingest.py:200`). There is no monocular pose/depth/scale path at all. Poses and intrinsics are the same missing file (`fx/fy/cx/cy` come from `odometry.csv`), so "no poses" and "no intrinsics" aren't separable gaps — it's one gap. **This is the single highest-risk item**: it's a scored gate, and one of the two live-defense held-out captures is explicitly plain video with no depth or poses. |

---

## 3. Track B — Damage Intelligence

| Requirement | Status | Evidence |
|---|---|---|
| Detection & segmentation, fused without double counting | **Mechanism done, barely exercised** | Per-surface UV-grid voting (`damage/fusion.py`), reflection rejection via plane-agreement, corner-spanning stains split per pixel, grazing-view discount, `min_views` artefact filter — all real. Genuinely only run against real damage a handful of times this session, and the one real finding it ever produced (a ceiling stain) turned out to be a false positive from the pre-fix orientation bug. |
| Classification: water/fire/mold/**combined**, IICRC attributes | **Partial — "combined" is dead** | IICRC Category/Class/subtype/condition consensus-voting is real (`damage/fusion.py:336-364`). But `"combined"` is in the schema enum (`schema/result.schema.json:179`) with **no producer anywhere**: the VLM's own class enum never offers it (`vlm.py` `DAMAGE_CLASSES`), and `_dominant_class()` always picks one winner. If it ever *did* appear, `scope.py:549-556` only dispatches on `water`/`mold`/`fire` — a combined region would silently produce zero line items. Also: `_dominant_class` defaults to `"water"` when totals are empty — a silent bias worth knowing about. |
| Metric expression per named surface | **Done** | `fusion.py:83-95` emits exactly the slide's phrasing (`"north wall: 4.2 m² Cat 2 water staining, lower 60 cm affected"`). |
| Concealed-damage inference | **Done** | `scope.py:488-533` fires against 8 real rules with probability, rationale, source, and the rule that fired. |
| **SAM 2 segmentation — has never actually run** | **Confirmed** | `masks.py:50` only calls SAM when `REPLICATE_API_TOKEN` is set; it exists only as a placeholder in `.env.example`, never in the real `.env`. `cache/masks/` is empty — zero `.npz` files, which is what a real SAM run would cache. Every mask produced in this project's history is GrabCut or a raw box fallback. Failure is silent (`masks.py:53-54`, bare `except: pass`). This directly affects the honesty of every damage-area measurement. |

---

## 4. Track C — Scope Generation

| Requirement | Status | Evidence |
|---|---|---|
| Quantity math (flood cut, baseboard, drying equipment, containment/PPE) | **Done, well-cited** | All four rules present in `rules.yaml` with real IICRC citations and applied in `scope.py` (e.g. flood-cut base height + snap heights `rules.yaml:28-39` → `scope.py:125-146,194-216`; drying equipment by area+class `rules.yaml:71-88` → `scope.py:256-319`). The 30→60 cm live-modification story is a genuine one-line config edit. |
| Concealed-damage rule set | **Done, substantive** | 8 rules, each with probability (0.55–0.85), rationale, and an IICRC source citation — not thin. |
| **Xactimate-consumable export** | **Not done** | `rules.yaml:204-205` claims units "map cleanly onto an Xactimate-style workflow," but no CSV/XML/ESX/sketch export exists anywhere in `pipeline/` — output is only the `scope.line_items` JSON array. The slide asks for "room geometry sufficient to rebuild the sketch, plus line items keyed to surfaces" as a *consumable* format; the convention is real, the export isn't. |
| **Ceiling damage produces almost no scope — a real bug** | **Confirmed** | `scope.py:157` classifies `is_floor` by surface name; a `"ceiling"`-named surface falls into the *wall* branch, which requires a truthy `wall_length` (`scope.py:194`) keyed by wall names only (`cli.py:181-183`). A ceiling region's lookup returns `None`, so **flood cut, drywall replacement, insulation, and baseboard line items are all silently skipped** for ceiling damage — only antimicrobial treatment survives. This is directly relevant: the one real damage detection this project has ever produced (before being identified as a false positive and fixed) was a ceiling stain. |
| Dead `rules.yaml` keys | Noted | `applies_to_categories`, `coverage_rate`, `extend_to_wall_ends`, `cleaning_rate_m2_per_hour`, `porous_materials` are declared but never read by `scope.py` — two sources of truth for the same gating logic in places (e.g. flood-cut category gating actually happens via `category_actions[...]["drywall"]`, not the declared key). |

---

## 5. Accuracy gates — UPDATED: 13 of 14 now have scoring code; still 1 of 14 measured

*(Table below is exactly as it stood before this pass — see the Update section at the
top for what changed. New state, gate-by-gate:)*

| Gate | Scoring code exists? | Ever measured? |
|---|---|---|
| Wall length error | ✅ `bench/run.py` | ❌ no ground truth |
| Ceiling height error | ✅ | ❌ |
| Floor area error/room | ✅ | ❌ |
| Multi-room stitched footprint error | ✅ **new** — no-overlap + adjacency self-consistency run unconditionally (no truth file needed); footprint-error-vs-laser-total via `--footprint-reference` | ⚠️ self-consistency part passes on real data (0 overlaps, 0 adjacency errors, `out/recordings-1`); error-vs-truth part still needs a laser total |
| Door/window opening widths | ✅ | ❌ |
| Head-to-head vs incumbent | ✅ | ❌ no incumbent data |
| No-LiDAR fallback wall length | ❌ still no code (blocked at ingest, §2 — needs Tier 4) | ❌ |
| Damage classification macro F1 | ✅ **new** — `--damage-class-reference <csv>` | ❌ tested against synthetic fixture only (hand-verified correct), no real annotations |
| Water Category/Class assignment | ✅ **new** — `--water-reference <csv>` | ❌ same |
| Damage segmentation IoU | ✅ **new** — `--iou-reference <csv>` | ❌ same |
| Affected-area quantity per surface | ✅ | ❌ |
| Concealed-damage flags recall/precision | ✅ **new** — `--concealed-reference <csv>` | ❌ same |
| Line-item recall vs reference scope | ✅ **new** — `--scope-reference <csv>` | ❌ same |
| Capture-to-scope runtime | ✅ (`diagnostics.timings_s`) | ⚠️ measured, but the reported number is misleading (below) |

**Now only 1 of 14 gates has zero scoring code** (no-LiDAR wall length — it can't be
scored until a no-LiDAR reconstruction path exists to produce predictions from). The 5
newly-added reference-based scorers in `bench/run.py` are real, tested comparison logic
— each verified against a synthetic fixture built with deliberately mixed hits and
misses, with every output number hand-checked against the fixture's known-correct
answer (documented in the session, not reproduced here) — but every one is still
un-run against real annotations, because no reference-data files exist yet. Each new
CLI flag's docstring/help text states its exact expected CSV column format.

**Runtime gate — the one gate with a number is unreliable:**
- `report/benchmark.md` claims **"92 s per room ... PASS"**, sourced from a run whose
  damage stage was `0.0 s` — i.e. it never included a real VLM pass at all.
- Real cold, full-resolution damage-stage cost measured this session: **~20–30 s per
  frame**, ×40 frames default = ~15-20 min just for damage detection. One real run this
  session hit **367.6 s/room**, over the 300 s gate.
- **Structural risk, not just a stale number:** damage cost scales with
  `--damage-frames` (fixed at 40 by default), not with room count. A 1-2 room capture
  divides that same ~1000 s among fewer rooms — the gate is currently only reliably
  passable on multi-room captures, and could fail outright on a single-room held-out
  capture at the live defense.

---

## 6. Reports are stale relative to the code

- **`report/benchmark.md`**: every gate row says "pending ground truth" (accurate), but
  the numbers that *are* printed don't match current output — 26 walls reported vs. 36
  in the current `result.json`; 99 s vs. 166 s runtime; the geometry-stage timing is off
  by an order of magnitude (373.4 s claimed vs. 24.8 s actual).
- **`report/technical.md`**: ~8.5 pages (over the 8-page cap at this density), and
  **doesn't mention any of the four fixes made this session** — the damage-overlay
  renderer being dead code, gravity-derived auto-orientation replacing a hardcoded
  guess, full-native-resolution VLM input replacing a 256×192 depth-aligned crop, or
  the `--min-detection-confidence` filter. Also has stale figures (claims 6 rooms/88 s
  for recordings-1; current run is 5 rooms/166 s). A `<!-- GATE_TABLE -->` placeholder
  is never filled in.
- **`README.md`**: doesn't mention `--min-detection-confidence` or `--debug-furniture`,
  and its output-tree diagram omits `damage_overlays/`.

---

## 7. Other things not handled well

- **No automated tests.** No `pytest` suite, no `conftest.py`, `pytest` isn't even in
  `requirements.txt`. `tools/conv_test.py`/`grav_test.py` are one-shot ablation scripts,
  not regression tests. Every fix made this session (rotation math, bbox round-trip
  through rescale+rotate, resolution rescale) was verified with a hand-written throwaway
  script, then discarded — nothing guards these against a future regression.
- **`out/` is cluttered with ~10 diagnostic run directories** from this session's
  debugging (`_rotation_check`, `_full_res_check`, `_debug_furniture*`, etc.) sitting
  next to the one real deliverable, `out/recordings-1`. Each carries a ~60 MB
  `cloud.ply` + ~57 MB `scene.glb`. Worth pruning before submission — both for repo size
  and because one of them (`recordings-1_debug_furniture`) still contains the retracted
  false-positive water-stain result from before the orientation fix, which would be
  actively misleading if a reviewer opened it instead of the real output.
- **Stale `bench/calibration.json` reference**: `--calibration` defaults to a path that
  has never been written by anything (see §2).

---

## 8. What was found and fixed this session (for context, not new gaps)

These are resolved, listed here so the technical report's ablation/failure-modes
sections can cite them as real before/after evidence:

1. **Damage overlays were dead code** — `export.render_damage_overlays()` existed but
   was never called from `cli.py`. Fixed: wired in, verified end-to-end with real
   detections.
2. **Orientation was hardcoded, and wrong for most frames** — every frame was rotated
   90° CW unconditionally for VLM input. Replaced with `ingest.display_rotation()`,
   which derives the correct rotation per frame from the capture's own measured gravity
   vector and each frame's pose — a real signal, not a guess. Also fixed the same
   problem in the human-facing overlay renderer, which had never been touched.
3. **VLM input resolution was silently capped at 256×192** (the depth raster's
   resolution) via a shared array, then further capped at `max_edge=1024` inside the
   encoder — a ~56× and then ~2× pixel-count cut from the native 1920×1440 video. Fixed
   by threading a separate full-resolution frame to the VLM call only, rescaling
   detections back to depth-resolution coordinates for fusion, and raising `max_edge` to
   4096. Concretely improved detection quality (verified: went from vague,
   full-frame-spanning boxes with "low resolution" complaints in 67% of `frame_notes`,
   to tight, correctly-labeled, specific-object boxes).
4. **A ceiling-fixture lighting artifact was misread as water damage** — directly caused
   by feeding the model a sideways image (fix #2). Confirmed via direct comparison: the
   same physical frame, correctly oriented, produces "no damage called" with explicit
   reasoning rejecting the water-stain read.
5. **Stale overlay files persisted across runs** — a run finding zero damage left a
   previous run's overlay image in the output directory, contradicting `result.json`.
   Fixed: `_damage_pass()` clears its own output subdirectories at the start of every
   run.
6. **Saved overlay images were depth-resolution (256×192), not video resolution** —
   fixed to upscale to and save at native video resolution.

---

## 9. Priority order for remaining time

1. **No-LiDAR fallback** (§2) — highest risk: it's a scored gate *and* one of the two
   live-defense captures is explicitly plain video. Even a minimal monocular
   depth+pose path that produces honest, wide, clearly-uncalibrated intervals beats the
   current `FileNotFoundError`.
2. **Ground truth** — nothing in §5/§6 can improve without a filled `bench/gt_*.csv`;
   this single artifact unblocks calibration, 3 of the "no code" gates becoming
   measurable via `bench/run.py`, and turns the benchmark report from all-pending into
   real numbers.
3. **Fix the `"combined"` dead schema value and the ceiling-scope bug** (§3/§4) — both
   are small, contained code fixes with an outsized "did you actually test this"
   optics risk if surfaced live during questioning.
4. **Screen recording** — required deliverable, currently entirely absent.
5. **Refresh `report/benchmark.md` and `report/technical.md`** against current code and
   trim the technical report to 8 pages — currently both would visibly contradict a
   live run in front of the graders.
6. **Wire door-derived adjacency, a room-overlap check, and floor/ceiling named
   planes** (§2) — three separable, moderate-effort fixes against explicitly-worded
   gate language.
7. **Prune `out/`** before submission (§7) — five minutes of cleanup, removes a real
   risk of a grader opening the wrong (retracted) result.
