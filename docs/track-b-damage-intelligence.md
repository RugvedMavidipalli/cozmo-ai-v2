# Track B: Damage Intelligence

## 1. Overview

Track B turns a set of selected walkthrough frames into fused, per-surface damage
regions: a named surface (`room_1.north_wall`, `floor`, `ceiling`) carrying a damage
class (water, fire, mold, or a genuine co-occurrence of two), IICRC-style attributes
where inferable (water Category/Class, fire subtype, mold condition), a metric area,
and a confidence built from how many independent views agreed. The pipeline runs this
as one stage inside `python -m pipeline run <capture>` (`cli.py::_damage_pass`), fed by
the geometry Track A has already produced (walls, rooms, per-wall surface grids) and
producing the `damage` and `concealed` sections of `result.json`, plus the fused
regions that Track C turns into a scope of work.

The stage has three parts in series: a vision-language model (Claude) proposes damage
hypotheses per frame (`damage/vlm.py`), each hypothesis is refined from a bounding box
into a pixel mask (`damage/masks.py`), and every mask is back-projected into 3D and
voted into a per-surface grid that only keeps what multiple independent views agree on
(`damage/fusion.py`). Nothing from a single frame is trusted on its own — the fusion
stage exists specifically because a single-frame call is a hypothesis, not a result.

## 2. Keyframe selection

Track A's reconstruction wants overlapping frames — more views of the same wall means
a better plane fit. Track B wants the opposite: as few frames as possible, each showing
something the others don't, because every frame sent to the VLM costs an API call and
real money. `keyframes.select_damage_keyframes` treats this as a coverage problem
rather than a sampling-rate problem: it bins the camera trajectory by position (0.6 m
cells) and heading (25° wedges), so each bin represents one distinct viewpoint, then
picks the single sharpest frame from each bin (via Laplacian variance — `sharpness()` —
searched over a small window around the bin's central frame, since a camera-shake blur
on the exact frame the binning picked shouldn't disqualify an otherwise-good
viewpoint). The sharpest frames across all bins are ranked and the top `max_frames`
(default 40, `--damage-frames`) are kept, restored to chronological order. A dim,
under-visited room still gets its best available frame instead of being crowded out by
a bright, over-visited one, because blur is scored within a bin, not globally.

## 3. VLM detection

### Prompt design

`damage/vlm.py`'s system prompt is written around one central idea: the model is
scored as much on what it correctly declines to flag as on what it finds. The
evaluation scenes are expected to contain deliberate lookalikes — shadows that read as
soot, glare that reads as a wet stain, mirrors, wet-look-but-dry materials like
polished stone — so the prompt spells out five explicit false-positive tests (shadow
vs. soot, reflection vs. stain, wet-look vs. actual moisture, normal ageing vs. damage,
mold vs. ordinary dirt/staining) before asking the model to report anything. The
response schema backs this with a required `distractor_considered` field on every
region: the model must name the most plausible benign explanation and say why it
rejected it. A detection that can't articulate what else it might be is, in practice,
usually the one that's wrong.

Classification is asked for directly in restoration terms, not generic labels, because
that's what Track C's rules consume: water gets an IICRC S500 Category (1 clean / 2
grey / 3 black) only when the frame actually supports it, fire gets a soot/char/consumed
subtype, mold gets a condition level. Category *Class* (1–4, evaporation load) is
explicitly told to return null unless the wet footprint and materials are clearly
visible in a single frame — the prompt is honest that this is usually not inferable
from one image, rather than encouraging the model to guess.

### Orientation and resolution

Two things happen to a frame before it's sent, both added this session and both aimed
at giving the model an image worth reasoning about:

- **Rotation.** The raw sensor frame is in the camera's native orientation, which does
  not necessarily match how a human would view the scene — cached VLM responses used
  to routinely open with "frame appears rotated ~90 degrees" as an aside before the
  model could even get to the damage question. `ingest.display_rotation(pose,
  gravity_up)` computes the correct correction *per frame* from data the pipeline
  already has: it rotates the capture's measured gravity vector into that frame's
  camera space (using the frame's own pose) and reads off which way "up" actually
  points in the raw raster. This replaced an earlier, cruder version of this fix that
  just hardcoded a 90°-clockwise rotation for every frame — that happened to be right
  for the one capture on hand but had no basis for generalizing to a different capture
  or a phone held differently. `vlm.py::_encode` applies whatever rotation
  `display_rotation` returns (or none), and `_to_analysis` unwinds it afterward via
  `ingest.rotate_bbox` with the inverse code, so a detection still lands in the
  original, unrotated frame's coordinates — that's the space `refine()` and fusion
  operate in, and the only reason rotation happens at all is to give the model a better
  look.
- **Resolution.** `capture_frame.color` (the array used everywhere else in the
  pipeline) is downsampled to the depth sensor's resolution — 256×192 on the capture
  device in this project, about a 56x pixel-count cut from the native 1920×1440 video —
  because mask refinement and fusion need color and depth pixel-aligned. There's no
  reason the VLM call needs to share that constraint, and it was silently paying the
  cost anyway: `cli.py::_damage_pass` now requests `ingest.iter_frames(...,
  include_full_res=True)` and sends `capture_frame.color_full` to
  `analyzer.analyze_frame`, passing `target_shape=capture_frame.color.shape[:2]` so the
  returned box gets rescaled back down to depth resolution afterward — the model sees
  full detail, but everything downstream still gets coordinates in the grid it expects.
  A second, independent cap inside `_encode` (`max_edge`, default 4096) exists purely
  as a safety ceiling against a pathologically large future capture device, not as a
  target — it doesn't bind for any current phone resolution.

### Caching and the furniture diagnostic

Every response is cached to disk keyed by a hash of the model, effort level, and the
*actual prompt text sent* (not just a hand-maintained version string) plus the image
bytes — so a prompt edit invalidates exactly the cached responses whose wording
changed, a rerun on unchanged inputs costs nothing, and a live demo doesn't depend on
network access. `DamageAnalyzer(include_furniture=True)` is a diagnostic-only mode
(`--debug-furniture`) that appends an addendum to the prompt asking the model to also
name individual pieces of furniture it can identify with confidence (from a fixed list:
couch, sofa, chair, bed, dresser, nightstand, table, cabinet, bookshelf, desk), one
tightly-boxed region per object. This exists purely to sanity-check that the model is
actually resolving real objects in the frame at all, independent of whether it finds
damage — furniture-tagged regions are filtered out in `cli.py::_damage_pass` before
masking/fusion ever sees them (`d.damage_class != FURNITURE_CLASS`) and, by default,
only their per-frame counts are printed to the console. Passing `--furniture-overlays`
alongside `--debug-furniture` additionally renders them to their own
`furniture_debug_overlays/` directory; either way they never reach `result.json` or
the scope engine. It is not a production damage class.

## 4. Mask refinement

A bounding box systematically overstates a stain's true area — a diagonal tide line
fills maybe half its box — and area is what every downstream quantity is built from, so
`damage/masks.py::refine()` turns each box into a pixel mask before anything else uses
it. The preferred path is SAM 2 via the Replicate API (`_sam2`, cached to `.npz` by
image+box hash); if `REPLICATE_API_TOKEN` isn't set, or the SAM call raises for any
reason, it silently falls through to a local GrabCut fit seeded by the box
(`_grabcut`). GrabCut itself has a floor: if the box is too small to give it any
background to learn a color model from (under 12px either dimension), or if the fitted
foreground comes back under 5% of the box area, the mask degrades to the box itself
rather than something worse than the box. Every `RefinedMask` records which method
actually produced it (`"sam2"`, `"grabcut"`, or `"box"`) via `.method`, and `.trusted`
is `False` exactly when it's a raw box — that flag is meant to feed into how much the
resulting area should be trusted, though see the gap below.

## 5. Fusion

`damage/fusion.py` is where "one frame's opinion" becomes "a region worth reporting."
For each surface, `build_surface_refs` creates a named plane (every wall, plus a
single global `floor` and `ceiling`); a `DamageAccumulator` holds vote buffers over
that surface's UV grid (the same grid `occupancy.build_surface_grid` already built for
occlusion/opening detection on walls). `project_detection` back-projects a mask's
pixels through the frame's depth and pose into world points and unit ray directions;
`assign_to_surfaces` then splits those points among surfaces purely by plane
proximity — a point has to lie close to *and* within the along-wall extent of a
surface, and a point matching nothing is dropped silently rather than forced onto the
nearest plane. That silent drop is the actual mechanism for reflection and furniture
rejection: a mirror shows the reflected scene's depth, not the mirror's own depth, so
its back-projected point lands far from the mirror's real plane and never gets voted
anywhere; the same geometric test discards a detection that landed on furniture
standing in front of a wall. A stain that spans a corner is split correctly because
assignment happens per pixel, not per detection — each wall only receives the portion
of the mask that geometrically belongs to it.

Each vote is weighted by `incidence_weight`: how squarely the view saw the surface
(mean absolute cosine between the ray directions and the surface normal), zeroed out
entirely below `MIN_INCIDENCE_COSINE` (0.26, ≈75° off-normal) since a grazing view
smears a mask across more surface than it actually covers. `extract_regions` then
requires `min_views` independent detections (default 2, `--min-views`) and a minimum
accumulated weight before a cell counts as "supported" — this is the single most
important parameter in the module, because it's what separates a real, persistent
stain from a view-dependent artefact (glare, a moving shadow) that shows up in one
frame and is gone in the next. Supported cells are morphologically closed and opened to
clean up the mask, then connected-component labeled into discrete regions.

### Classification, including "combined"

`_dominant_class` decides a region's damage class from the accumulated per-class
weight in its cells. As of this session it no longer always picks a single winner: if
the second-most-supported class carries at least `COMBINED_CLASS_RATIO` (0.4, i.e. 40%)
of the top class's weight, the region is classified `"combined"` instead — a real,
grounded scenario (fire damage from the water used to extinguish it is a standard
IICRC combined-loss case), not noise from an occasional stray detection. `
_component_classes` identifies which two classes are involved, and
`_combined_attributes` merges their consensus attributes (subtype, water
Category/Class, mold condition each taken from whichever component actually supplies
one; confidence averaged; evidence and contributing frames unioned) rather than
discarding one side's information. `_consensus_attributes` itself is a straightforward
weighted vote per attribute across the detections that agree on the winning class, so a
region's reported Category/Class/subtype reflects what most of the independent views
actually said, not just the single highest-confidence one. `DamageRegion.describe()`
produces the metric phrasing the assignment specifically asks for, e.g. `"north wall:
4.2 m2 Cat 2 water staining, lower 60 cm affected"`, or `"...: 0.9 m2 water+fire
combined, ..."` for a combined region.

## 6. Known gaps and limitations

- **SAM 2 has never actually run in this project.** `refine()` only attempts it when
  `REPLICATE_API_TOKEN` is set in the environment; the project's real `.env` has never
  had one configured (only `.env.example` mentions it as a placeholder), and the
  `cache/masks/` directory — which a real SAM run would populate with `.npz` files —
  has been empty throughout. Every mask produced so far has been GrabCut or a raw box.
  This doesn't invalidate the fusion mechanism, but it means the area-accuracy claims
  this system can currently back up are GrabCut-quality, not SAM-quality, and the
  `.trusted` flag that's meant to communicate "this area is a box upper bound" downstream
  isn't currently threaded into the uncertainty model's damage-area interval — worth
  checking before relying on it.
- **Floor and ceiling surfaces are built but never actually accumulate damage.**
  `build_surface_refs` creates `SurfaceRef`s for a single global floor and ceiling
  alongside the named walls, but `cli.py::_damage_pass` only allocates a
  `DamageAccumulator` for surfaces where `surface.kind == "wall"`. A detection that
  `assign_to_surfaces` correctly assigns to the floor or ceiling has nowhere to vote —
  the accumulator lookup returns `None` and the point is dropped
  (`accumulator = accumulators.get(surface_index); if accumulator is None: continue`).
  In practice this means floor and ceiling damage is currently undetectable end to end,
  independent of the Track C ceiling-scope fix (see `docs/track-c-scope-generation.md`)
  which only fixed what happens *if* a ceiling region exists — building the missing
  floor/ceiling occupancy grids so they can actually accumulate votes is unfinished
  work.
- **The one real damage finding this project produced was a false positive**, caused by
  the pre-fix version of the orientation problem (a ceiling light fixture's lighting
  gradient, read sideways, resembled a tide-line water stain). Re-run correctly
  oriented, the model retracts the call with explicit reasoning. This is genuinely
  useful evidence that the orientation fix mattered, but it also means the fusion
  pipeline's real-world detection accuracy is still unverified against any actual
  damage — everything exercised this session was either synthetic or diagnostic
  (furniture-mode).

## 7. Usage

Track B runs automatically as part of `python -m pipeline run <capture> [flags]`
unless `--no-damage` is passed. Flags that affect it: `--damage-frames` (max keyframes
sent to the VLM, default 40), `--min-views` (independent-agreement threshold before a
region is reported, default 2), `--min-detection-confidence` (drops raw VLM detections
below this confidence before masking/fusion ever sees them, default 0.0 — no
filtering), `--no-sam` (skip the SAM/Replicate attempt and always use GrabCut),
`--model` (which Claude model performs detection, default `claude-opus-5`),
`--cache-dir` (VLM response cache location, default `cache/vlm`; mask cache lives
alongside it at `cache/masks`), `--debug-furniture` (the diagnostic mode described
above), and `--furniture-overlays` (with `--debug-furniture`, also renders and writes
the diagnostic overlay images instead of only printing counts).

Output: fused regions appear in `result.json`'s `damage` array (id, surface_ref,
damage_class, subtype, area with confidence interval, water/mold attributes,
`combined_classes` when applicable, extent, description, contributing frames) and the
`concealed` array (produced by Track C from these regions, see
`docs/track-c-scope-generation.md`). Per-frame overlay images — the detection box, the
tinted mask, and a `class:subtype confidence` label, drawn at full native video
resolution as of this session (previously depth resolution) and rotated to the same
human-natural orientation used for the VLM call — are written to
`out/<capture>/damage_overlays/`; the furniture-diagnostic equivalent goes to
`furniture_debug_overlays/` only when both `--debug-furniture` and
`--furniture-overlays` are set (`--debug-furniture` alone prints per-frame counts to
the console without rendering anything). Both directories are cleared at the start of
every run so a rerun's output can never contain a stale image from a previous run's
different findings.
