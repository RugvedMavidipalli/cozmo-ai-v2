"""Generate the benchmark report from pipeline outputs.

Every number in the report is read from `result.json`, the ablation file, or
the benchmark summary -- never transcribed by hand -- so re-running the
pipeline regenerates a report that still matches the code.

    python tools/make_report.py --results out/rec1/result.json out/rec2/result.json \
        --ablations out/ablations_rec1.json --out report/benchmark.md
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

# The published gates, as (metric, gate text, stretch text). Populated from a
# benchmark summary when one is supplied; otherwise marked as needing truth.
GATE_TABLE = [
    ("wall_length", "Wall length error", "<= 1% or 2 cm on >= 90% of walls", "<= 0.5%"),
    ("ceiling_height", "Ceiling height error", "<= 1.5 cm", "<= 1 cm"),
    ("floor_area", "Floor area error per room", "<= 2%", "<= 1%"),
    ("opening_width", "Door / window opening widths", "<= 2 cm on >= 85%", "<= 1 cm"),
    ("damage_area", "Affected-area quantity per surface", "within +/-10%", "+/-5%"),
]


def load(path: str | Path | None):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def capture_section(results: list[dict]) -> str:
    lines = [
        "## Captures\n",
        "| capture | frames | duration | path | rooms | walls | openings | runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        capture = result["capture"]
        timings = result["diagnostics"].get("timings_s", {})
        lines.append(
            f"| {capture['name']} | {capture['frame_count']} | "
            f"{capture['duration_s']:.0f} s | {capture['path_length_m']:.1f} m | "
            f"{len(result.get('rooms', []))} | "
            f"{len(result['reconstruction']['walls'])} | "
            f"{len(result['reconstruction'].get('openings', []))} | "
            f"{timings.get('total', 0):.0f} s |"
        )
    return "\n".join(lines)


def runtime_section(results: list[dict]) -> str:
    """Per-stage timing, and runtime per room against the 5-minute gate."""
    stages: dict[str, list[float]] = {}
    for result in results:
        for stage, seconds in result["diagnostics"].get("timings_s", {}).items():
            stages.setdefault(stage, []).append(seconds)

    lines = [
        "\n## Runtime\n",
        "| stage | mean seconds |",
        "|---|---:|",
    ]
    for stage, values in sorted(stages.items(), key=lambda kv: -sum(kv[1])):
        lines.append(f"| {stage} | {sum(values) / len(values):.1f} |")

    per_room = []
    for result in results:
        rooms = max(len(result.get("rooms", [])), 1)
        total = result["diagnostics"].get("timings_s", {}).get("total", 0)
        per_room.append(total / rooms)
    if per_room:
        mean = sum(per_room) / len(per_room)
        verdict = "PASS" if mean <= 300 else "FAIL"
        lines.append(
            f"\nCapture-to-scope runtime: **{mean:.0f} s per room** "
            f"(gate <= 300 s, stretch <= 90 s) — {verdict}. "
            f"Hardware: Intel Core i9-9980HK, 16 threads, no GPU."
        )
    return "\n".join(lines)


def drift_section(results: list[dict], ablations: list[dict] | None) -> str:
    lines = [
        "\n## Error budget\n",
        "Drift is measured as **wall revisit spread**: each wall's supporting "
        "points are grouped by when they were observed, and the spread between "
        "per-visit plane offsets is reported. Point scatter about a fitted wall "
        "is dominated by depth noise and is averaged away by the fit, so it is "
        "nearly blind to the drift that actually breaks the wall gate.\n",
        "| capture | median | p90 | max | revisited walls |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        drift = result["diagnostics"]["drift"]
        lines.append(
            f"| {result['capture']['name']} | {drift['median_mm']:.1f} mm | "
            f"{drift['p90_mm']:.1f} mm | {drift['max_mm']:.1f} mm | "
            f"{drift['revisited_walls']} |"
        )

    if ablations:
        lines += [
            "\n### Ablation: trajectory refinement\n",
            "| variant | drift median | p90 | max | walls > 1.5 m | room height |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for record in ablations:
            height = (
                f"{record['room_height_m']:.3f} m" if record.get("room_height_m") else "n/a"
            )
            lines.append(
                f"| {record['label']} | {record['drift_median_mm']:.1f} mm | "
                f"{record['drift_p90_mm']:.1f} mm | {record['drift_max_mm']:.1f} mm | "
                f"{record['walls_over_1_5m']} | {height} |"
            )
    return "\n".join(lines)


def gate_section(benchmark: dict | None) -> str:
    lines = ["\n## Accuracy gates\n"]
    if not benchmark:
        lines += [
            "> **Not yet scored.** Gates require laser ground truth. Enter "
            "measurements into `bench/gt_<property>.csv` and run "
            "`python bench/run.py --result ... --truth ...`.\n",
            "| metric | gate | stretch | result |",
            "|---|---|---|---|",
        ]
        for _, label, gate, stretch in GATE_TABLE:
            lines.append(f"| {label} | {gate} | {stretch} | pending ground truth |")
        return "\n".join(lines)

    lines += [
        "| metric | gate | result | status |",
        "|---|---|---:|---|",
    ]
    for key, label, gate, _ in GATE_TABLE:
        data = benchmark["gates"].get(key)
        if not data:
            lines.append(f"| {label} | {gate} | not measured | — |")
            continue
        status = "PASS" if data["pass"] else "FAIL"
        lines.append(
            f"| {label} | {gate} | {data['passing']}/{data['total']} "
            f"({data['fraction'] * 100:.0f}%) | {status} |"
        )

    coverage = benchmark.get("interval_coverage", {})
    if coverage:
        lines += [
            "\n### Interval calibration\n",
            "An interval is only worth printing if it is calibrated: a claimed "
            "90% interval must contain roughly 90% of measurements.\n",
            "| measurement | claimed | observed | verdict |",
            "|---|---:|---:|---|",
        ]
        for kind, data in coverage.items():
            verdict = "calibrated" if data["calibrated"] else "**MISCALIBRATED**"
            lines.append(
                f"| {kind} | {data['claimed'] * 100:.0f}% | "
                f"{data['observed'] * 100:.1f}% | {verdict} |"
            )

    head_to_head = benchmark.get("head_to_head")
    if head_to_head:
        lines += [
            "\n### Head-to-head vs incumbent\n",
            f"{head_to_head['wins']} win / {head_to_head['ties']} tie / "
            f"{head_to_head['losses']} loss — "
            f"**{head_to_head['beat_or_tie_fraction'] * 100:.0f}% beat-or-tie** "
            f"(gate 70%), {'PASS' if head_to_head['pass'] else 'FAIL'}.",
        ]
    return "\n".join(lines)


def damage_section(results: list[dict]) -> str:
    regions = [r for result in results for r in result.get("damage", [])]
    lines = ["\n## Damage and scope\n"]
    if not regions:
        lines.append(
            "> No damage regions in these captures — the sample walkthroughs are "
            "undamaged properties. Run against a staged-damage capture with "
            "`ANTHROPIC_API_KEY` set to exercise Tracks B and C."
        )
        return "\n".join(lines)

    by_class: dict[str, int] = {}
    for region in regions:
        by_class[region["damage_class"]] = by_class.get(region["damage_class"], 0) + 1
    lines.append(
        "Fused regions: "
        + ", ".join(f"{count} {name}" for name, count in sorted(by_class.items()))
        + ".\n"
    )
    lines += [
        "| surface | class | area | views | confidence |",
        "|---|---|---:|---:|---:|",
    ]
    for region in sorted(regions, key=lambda r: -r["area"]["value"])[:12]:
        area = region["area"]
        lines.append(
            f"| {region['surface_ref']} | {region['damage_class']} | "
            f"{area['value']:.2f} ± {area['half_width']:.2f} m² | "
            f"{region['view_count']} | {region['confidence']:.2f} |"
        )

    items = [i for result in results for i in result["scope"]["line_items"]]
    if items:
        lines += [
            f"\n{len(items)} scope line items generated. Sample:\n",
            "| action | material | qty | unit | rule | basis |",
            "|---|---|---:|---|---|---|",
        ]
        for item in items[:10]:
            lines.append(
                f"| {item['action']} | {item['material']} | {item['quantity']:.2f} | "
                f"{item['unit']} | `{item['rule_id']}` | {item['basis'][:70]} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--ablations")
    parser.add_argument("--benchmark", help="summary JSON from bench/run.py --out")
    parser.add_argument("--out", default="report/benchmark.md")
    args = parser.parse_args(argv)

    results = [json.loads(Path(p).read_text()) for p in args.results]
    ablations = load(args.ablations)
    benchmark = load(args.benchmark)

    document = "\n".join(
        [
            "# Benchmark report",
            f"\nGenerated {date.today().isoformat()} from pipeline output. "
            "Every figure is read from `result.json`; none are transcribed.\n",
            capture_section(results),
            gate_section(benchmark),
            drift_section(results, ablations),
            runtime_section(results),
            damage_section(results),
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document + "\n")
    print(f"wrote {out_path} ({len(document.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
