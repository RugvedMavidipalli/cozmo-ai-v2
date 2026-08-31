# Benchmarking and usage reference

Two things live here: the complete `bench/run.py` gate reference (what each accuracy
gate needs to be scored and how to invoke it), and a CLI usage walkthrough deeper than
the README's 15-minute quick-start.

## Running the pipeline

```
python -m pipeline run <capture_dir> --out out/<name>
```

`<capture_dir>` is a Stray Scanner capture: `odometry.csv`, `imu.csv`, `depth/*.png`,
`confidence/*.png`, `rgb.mp4`. See [`architecture.md`](architecture.md) for the full
flag table and output-file inventory; a few flag combinations worth calling out
specifically:

- **Fast geometry-only iteration**: `--no-damage` skips Track B/C entirely — useful
  when only reconstruction is being changed, since the damage stage (real VLM calls)
  dominates runtime on a capture with real damage.
- **Reproducing without loop closure**: `--no-loop-closure` keeps pose refinement's
  sequential-edge correction but drops ICP loop-closure edges — useful for isolating
  how much of the drift correction loop closure is actually buying (see
  track-a-reconstruction.md's ablation discussion).
- **Sanity-checking VLM object resolution**: `--debug-furniture` combined with a low
  `--min-detection-confidence` (e.g. `0.6`) is a fast way to confirm the model is
  actually looking at the frame — it'll tag named furniture with real confidence
  scores even on a capture with no damage in it, which a zero-detection damage run
  alone can't distinguish from "the VLM call is silently broken." By default this
  only prints per-frame counts to the console; add `--furniture-overlays` to also
  render the annotated images to `furniture_debug_overlays/`.
- **Live rule-change demo**: copy `rules.yaml`, edit e.g. `water.flood_cut.base_height`
  from `0.30` to `0.60`, and pass `--rules <copy>` — the line-item deltas across every
  room are the whole story, no code change involved.

## Benchmarking: `bench/run.py`

```
python bench/run.py --result out/<name>/result.json [flags]
```

`--result` is the only required flag. Everything else is additive — pass only the
flags for the gates you can currently score.

### Always runs: room self-consistency (no reference needed)

Every invocation runs `check_room_consistency`: no room-polygon overlaps beyond a 1%
tolerance, and a structurally valid, symmetric adjacency graph. This is the one gate
that never needed ground truth — it's a property of the pipeline's own output, not a
comparison against a measurement.

### Gate 1-3, 5, 6, 11: laser ground truth (`--truth`)

```
python bench/run.py --result out/<name>/result.json --truth bench/gt_<property>.csv \
  [--incumbent out/<incumbent>/result.json] [--fit-calibration] [--coverage 0.90]
```

`--truth` is a CSV: `kind,name,value,notes` (see `bench/gt_TEMPLATE.csv`). `kind` is
one of `wall_length`, `ceiling_height`, `floor_area`, `opening_width`, `damage_area`;
`name` must match a `surface_ref`/room name the pipeline actually emitted. This scores:
wall length error (gate + stretch), ceiling height error, floor area error, door/window
opening widths, and affected-area quantity — plus, with `--fit-calibration`, writes
`bench/calibration.json` (the conformal scale that turns every reported interval from
"uncalibrated" to real), and with `--incumbent`, a head-to-head win/tie/loss report
against another `result.json` on the same rooms.

### Gate 4: multi-room footprint error (`--footprint-reference`)

```
python bench/run.py --result ... --footprint-reference 45.2
```

A single number: the laser-measured total stitched-footprint area in m². Compared
against the sum of every room's reported area. (The no-overlap and adjacency-validity
parts of this gate need no reference at all — they're covered by the always-on room
self-consistency check above.)

### Gates 8-10, 12-13: damage/scope reference CSVs

Five scorers, each optional, each independent:

| Gate | Flag | Reference CSV columns |
|---|---|---|
| Damage classification macro F1 | `--damage-class-reference` | `surface_ref,damage_class` |
| Water Category/Class accuracy | `--water-reference` | `surface_ref,water_category,water_class` |
| Damage segmentation IoU | `--iou-reference` | `surface_ref,u_lo,u_hi,v_lo,v_hi` (metres, same along-wall/height convention as `result.json`'s `extent.u_range`/`v_range`) |
| Concealed-flag recall/precision | `--concealed-reference` | `surface_ref` (one row per estimator-identified concealed-damage location) |
| Line-item recall vs. reference scope | `--scope-reference` | `surface_ref,action,material,quantity` |

All five match by `surface_ref`; where a surface has multiple fused damage regions,
the pipeline's largest-area region on that surface is what gets compared (the dominant
call is what a human reviewer would actually check). None of these five reference
files exist yet in this repo — the scorers are real, tested comparison logic (see
`docs/misc.md`'s Update section for how they were verified against synthetic fixtures),
just not yet run against real annotations.

### Gate 14: capture-to-scope runtime

Read directly from `result.json`'s `diagnostics.timings_s.total`, divided by room
count. Not a separate `bench/run.py` flag — see `docs/misc.md` for the current
measured numbers and a structural caveat (damage-stage cost scales with
`--damage-frames`, not room count, so it's least favorable on small captures).

### Gate not yet scoreable: no-LiDAR wall length

No scoring code exists for this because no no-LiDAR reconstruction path exists yet to
produce predictions from (`ingest.load_capture` hard-fails on a capture without depth
— see `docs/track-a-reconstruction.md`'s gaps section). Once that path exists, this
reuses the same `--truth` mechanism as gate 1, applied to a `video_only`-modality
result.

### Output

Every gate that ran prints a pass/fail line and is included in the JSON written by
`--out <path>`. The process exit code is non-zero if any run gate failed (stretch
gates and room-consistency's info-only sub-fields don't affect the exit code the same
way — check the printed summary for the authoritative per-gate verdict).
