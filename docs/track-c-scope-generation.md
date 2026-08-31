# Track C: Scope Generation

## Overview

Track C turns Track B's fused damage regions into two things an estimator can act on:
a structured scope of work (line items with an action, a material, a quantity in a
real unit, a surface reference, and a citation back to the standard that justified the
number) and a list of concealed-damage flags (places the camera cannot see but the
physics of the visible damage implies are also wet, burned, or growing). Both are
produced by `pipeline/scope.py`'s `ScopeEngine`, driven entirely by the data in
`rules.yaml` — no restoration-domain number is hardcoded in the scope engine itself.

The core design decision is that the mapping from damage to line items is **not
proportional to area**. A 0.3 m² mold patch does not produce a 0.3 m² line item — S520
containment and PPE scale with the room, not the patch, so a small patch in an occupied
space still triggers a sealed enclosure and protected workers. Water is similar: a
flood cut is driven by the *height* of the waterline above the floor, not by how many
square metres are stained, and baseboard is removed in whole affected runs, not just
where the stain happens to touch it. Getting this proportional-mapping mistake wrong is
exactly the kind of error a restoration estimator would catch immediately, so the
engine encodes the real domain logic instead.

## The `rules.yaml` design

Every quantity `scope.py` emits traces to a rule in `rules.yaml`, and every rule carries
a `source` string citing the standard it comes from — IICRC S500 (water), S520 (mold),
or S700 (fire/smoke) — plus a `note` field on the handful of rules that are trade
practice or an internal simplification rather than a numeric standard requirement,
flagged as the ones to challenge first. The practical payoff of keeping every number in
YAML instead of the code is that a request like "change the flood-cut extension from
30 cm to 60 cm and show the delta across every room" is a one-line edit to
`water.flood_cut.base_height` and a rerun, not a code change — `ScopeEngine.__init__`
takes a `rules_path` argument specifically so an alternate rules file can be swapped in
without touching `scope.py` at all.

The rule content itself, read directly from the file:

- **Flood cut** (`water.flood_cut`): cut `base_height` (0.30 m) above the highest
  visible waterline, never less than `minimum_height` (0.61 m, i.e. 2 ft — below that,
  removal and patching costs more than just taking the full 2 ft), and if
  `snap_to_standard` is set, rounded up to the nearest `standard_heights` sheet line
  (0.61 / 1.22 / 2.44 m — 2 ft / 4 ft / full sheet), because drywall is sold in sheets
  and estimators cut to them. Cited to IICRC S500 §12.2.5 for the removal requirement
  itself, with the height above the waterline explicitly noted as trade practice, not a
  numeric standard.
- **Baseboard** (`water.baseboard`): the full affected run comes off (`follow_affected_run`),
  not just the stained segment — a partial removal leaves a joint that has to be
  replaced anyway. `replace_if_category: [2, 3]` decides whether the action is a full
  replacement or a detach-and-reset.
- **Drying equipment** (`water.drying_equipment`): air movers sized by
  `per_floor_area` (0.065 units/m², roughly 1 per 15 m²) plus `per_wet_wall_area`
  (0.25 units/m²) with a `minimum` of 1, from S500 §13.4's placement guidance.
  Dehumidifiers sized by room volume × a per-volume rate × a `class_factors` multiplier
  that scales 0.6–1.5× by IICRC water Class (1–4), from S500 §13.5. Monitoring days
  come from `monitoring_days.class_days`, keyed by Class, from S500 §13.7.
- **Mold containment/PPE** (`mold.containment`, `mold.ppe`): triggered once total mold
  area crosses `trigger_area` (1.0 m²), sized `small`/`medium`/`large` by
  `size_thresholds` (1.0 / 9.3 m², i.e. roughly 10/100 sq ft — the long-standing
  EPA/S520 thresholds), each size mapping to a containment type and a PPE tier (N95 up
  to a full-face PAPR suit), cited to S520 §11–12.

## Quantity math implementation

`ScopeEngine` has one method per damage class that turns a single `DamageRegion` into
line items, plus three room-level methods that aggregate across every region in a room
(because equipment counts and containment are properties of the room, not any one
patch):

- `_water_items` — for a floor surface, carpet/pad actions by category (remove for
  Category 3, clean-and-dry for Category 2, dry-in-place for Category 1). For a wall,
  the flood-cut math above produces `drywall_remove`/`drywall_replace`/
  `insulation_remove` sized to `cut height × affected run`, plus a baseboard item sized
  to the full run. **Ceiling surfaces get their own branch** (`is_ceiling`): a ceiling
  has no waterline to cut above, so instead of the height-times-run calculation, the
  same three drywall/insulation line items are sized directly to the region's affected
  *area* — and no baseboard item is produced, since baseboard is wall-only. This branch
  did not exist before this pass: `_water_items` used to require a non-null
  `wall_length` (looked up by wall name) to enter the flood-cut block at all, so a
  ceiling region — which was never in the wall-length lookup table — fell through that
  entire block and produced only the antimicrobial-treatment line item at the very end
  of the function, silently skipping drywall removal, replacement, insulation removal,
  and (correctly, since it's wall-specific) baseboard. The fix widens the entry
  condition to `wall_length or is_ceiling` and branches the area calculation on which
  case applies.
- `_mold_items` — remediation area is visible growth expanded by a `removal_margin`
  (0.30 m) on every side, since remediation always extends past what's actually
  visible; HEPA vacuum quantity is that expanded area times a fixed pass count.
- `_fire_items` — soot on a sound substrate is cleaned and, if `seal_after_cleaning` is
  set, sealed with an odour-blocking primer; char or consumed substrate is removed and
  replaced outright rather than cleaned.
- `_drying_items`, `_mold_containment`, `_fire_odour` — the three room-level methods.
  They take every region in a room, and compute equipment/containment/odour-treatment
  quantities from the room's own area, perimeter, and height rather than from any
  single region's extent — this is where the "small patch still needs full
  containment" logic actually lives (`_mold_containment`'s docstring calls this out
  directly as the clearest case where a proportional area mapping would be wrong).

**Dispatch and the "combined" class.** `build()` walks every region and calls
`_water_items`/`_mold_items`/`_fire_items` (and the room-level methods call
`_drying_items`/`_mold_containment`/`_fire_odour`) based on a new module-level helper,
`_involves(region, damage_class)`, rather than a direct `region.damage_class ==
"water"` equality check. `_involves` returns true either for a region whose own class
matches, or — this is new — for a region whose class is `"combined"` and whose
`combined_classes` tuple (set by Track B's fusion step when two damage classes are
genuinely co-dominant on one surface, e.g. water damage from firefighting a fire)
contains the class being asked about. Every one of the six dispatch sites in
`scope.py` (the three per-region calls in `build()`, plus the three room-level
aggregate functions) was changed from an `elif`/direct-equality chain to independent
`_involves` checks, so a single combined region correctly produces line items from
*every* relevant handler — a water+fire combined region gets both the flood-cut
drywall work and the soot cleaning, not just one or the other.

## Concealed-damage inference

`concealed_flags()` walks every rule in `rules.yaml`'s `concealed` list against every
fused region and fires a `ConcealedFlag` wherever `_matches()` says the rule's `when`
condition holds. A condition can constrain damage class, a minimum water Category or
Class, a minimum mold condition, a fire subtype, which kind of surface it's on
(wall/floor/ceiling, inferred from the surface key), whether the damage sits low enough
on the wall to be adjacent to the floor (`bounds_v[0] <= 0.2` m), or a material hint
that's only checked against floor surfaces. There are eight rules in the current file,
covering: insulation behind wet Category-2+ drywall (W-01, 0.72 probability), pad and
subfloor beneath wet carpet (W-02, 0.85), an adjacent wall cavity from a Class-3
intrusion (W-03, 0.65), the baseboard/sill plate at a wall-floor water junction (W-04,
0.60), growth on the unexposed back face of a substrate with visible Condition-3 mold
(M-01, 0.78), a hidden moisture source implied by any wall mold (M-02, 0.55),
structural charring behind visible char (F-01, 0.58), and smoke residue driven into
cavities and HVAC by any fire damage (F-02, 0.70). Each flag carries the triggering
region's id, the rule's id, its stated probability, a plain-language rationale, and the
IICRC citation — probabilities are explicitly calibrated judgement, not measurements,
and are presented as such.

## Export formats

Every run produces the scope in two forms:

- **`result.json`'s `scope.line_items` array** — the JSON form, one object per
  `LineItem`: `code`, `description`, `action`, `material`, `quantity`, `unit`, `trade`,
  `surface_ref`, `room_id`, `rule_id`, `source`, `basis` (a human-readable sentence
  explaining how the quantity was derived), and `derived_from` (the damage region ids
  that produced it, for auditability).
- **`scope_sketch.csv` and `scope_line_items.csv`**, written by
  `export.export_scope_csv()` — the structured, estimator-convention export that
  `rules.yaml`'s own comments describe the line-item units as designed for
  ("estimator convention so the export maps cleanly onto an Xactimate-style
  workflow"), and which did not exist before this pass; the JSON array alone was not a
  consumable sketch/scope file. `scope_sketch.csv` has one row per wall:
  `room, room_area_m2, ceiling_height_m, wall, wall_length_m,
  wall_length_ci_half_width_m` — sufficient geometry to rebuild the room sketch.
  `scope_line_items.csv` has one row per line item:
  `room, surface_ref, action, material, description, quantity, unit, trade, rule_id,
  source, basis`. Both are written unconditionally at the end of every
  `python -m pipeline run` invocation, alongside `result.json`.

## Known gaps and limitations

- **Five keys in `rules.yaml` are declared but never read by `scope.py`**:
  `applies_to_categories` (on `water.flood_cut`), `coverage_rate` (on
  `water.antimicrobial`), `extend_to_wall_ends` (on `water.baseboard`),
  `cleaning_rate_m2_per_hour` (on `fire`), and the per-category `porous_materials`
  action inside `water.category_actions` (only `drywall`, `carpet`, and `pad` are
  actually read from each category's action set). This creates two sources of truth in
  a couple of places — for instance, flood-cut Category gating actually happens
  through `category_actions[category]["drywall"] == "flood_cut"`, not through the
  declared `applies_to_categories: [2, 3]` list, and both currently agree by
  coincidence rather than by one driving the other. Worth either wiring these in or
  removing them so the YAML doesn't imply behavior the engine doesn't have.
- Everything else specific to Track C flagged in an earlier pass — the ceiling-scoping
  bug and the missing Xactimate-consumable export — is fixed as of this pass; see the
  sections above for what changed. No other Track-C-specific gaps are currently open.

## Usage

Scope generation is not separately invocable — it's the final stage of
`python -m pipeline run <capture_dir>`, run automatically after damage fusion. The one
flag that affects it directly is `--rules <path>`, defaulting to `rules.yaml` at the
repo root — pointing it at a modified copy of the file is exactly the mechanism for a
live "change this rule and rerun" demonstration (e.g. edit `flood_cut.base_height` to
0.60 in a copy, pass `--rules copy.yaml`, and diff the resulting line items against the
default run). Every run also picks up rules unconditionally regardless of whether any
damage was found — `scope.line_items` and `concealed` are simply empty arrays on an
undamaged capture, not omitted. Output: `result.json`'s `scope.line_items` and
`concealed` fields, plus `scope_sketch.csv` and `scope_line_items.csv` in the same
output directory as everything else.
