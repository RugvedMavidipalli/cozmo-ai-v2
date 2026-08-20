# Benchmark report

Generated 2026-08-20 from pipeline output. Every figure is read from `result.json`; none are transcribed.

## Captures

| capture | frames | duration | path | rooms | walls | openings | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| recordings-1 | 3997 | 67 s | 41.6 m | 5 | 64 | 5 | 74 s |
| recordings-2 | 5443 | 91 s | 54.1 m | 5 | 65 | 5 | 115 s |

## Accuracy gates

> **Not yet scored.** Gates require laser ground truth. Enter measurements into `bench/gt_<property>.csv` and run `python bench/run.py --result ... --truth ...`.

| metric | gate | stretch | result |
|---|---|---|---|
| Wall length error | <= 1% or 2 cm on >= 90% of walls | <= 0.5% | pending ground truth |
| Ceiling height error | <= 1.5 cm | <= 1 cm | pending ground truth |
| Floor area error per room | <= 2% | <= 1% | pending ground truth |
| Door / window opening widths | <= 2 cm on >= 85% | <= 1 cm | pending ground truth |
| Affected-area quantity per surface | within +/-10% | +/-5% | pending ground truth |

## Error budget

Drift is measured as **wall revisit spread**: each wall's supporting points are grouped by when they were observed, and the spread between per-visit plane offsets is reported. Point scatter about a fitted wall is dominated by depth noise and is averaged away by the fit, so it is nearly blind to the drift that actually breaks the wall gate.

| capture | median | p90 | max | revisited walls |
|---|---:|---:|---:|---:|
| recordings-1 | 23.3 mm | 41.6 mm | 51.0 mm | 25 |
| recordings-2 | 18.5 mm | 29.3 mm | 43.8 mm | 23 |

### Ablation: trajectory refinement

| variant | drift median | p90 | max | walls > 1.5 m | room height |
|---|---:|---:|---:|---:|---:|
| raw ARKit | 21.8 mm | 46.1 mm | 50.1 mm | 34 | 2.990 m |
| ICP only | 21.8 mm | 46.1 mm | 50.1 mm | 34 | 2.990 m |
| ICP + loop closure | 21.4 mm | 46.5 mm | 47.4 mm | 38 | 2.922 m |
| raw + ungated depth | 21.1 mm | 37.6 mm | 48.3 mm | 40 | 3.064 m |

## Runtime

| stage | mean seconds |
|---|---:|
| total | 94.6 |
| fusion | 33.3 |
| drift | 28.6 |
| geometry | 24.5 |
| surfaces | 4.1 |
| rooms | 0.9 |
| ingest | 0.2 |
| scope | 0.0 |

Capture-to-scope runtime: **19 s per room** (gate <= 300 s, stretch <= 90 s) — PASS. Hardware: Intel Core i9-9980HK, 16 threads, no GPU.

## Damage and scope

> No damage regions in these captures — the sample walkthroughs are undamaged properties. Run against a staged-damage capture with `ANTHROPIC_API_KEY` set to exercise Tracks B and C.
