"""Score pipeline output against laser ground truth and incumbent scans.

Ground truth is entered once per property as a CSV of laser measurements; this
script matches each row to the pipeline's corresponding measurement, reports
every accuracy gate, and fits the conformal scale that calibrates the intervals.

Usage:
    python bench/run.py --result out/rec1/result.json --truth bench/gt_home.csv
    python bench/run.py --result ... --truth ... --fit-calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.uncertainty import fit_calibration  # noqa: E402

# Each gate is (label, predicate over the error record, target fraction).
GATES = {
    "wall_length": (
        "Wall length error <= max(1%, 2cm)",
        lambda e: e["abs_error"] <= max(0.01 * e["truth"], 0.02),
        0.90,
    ),
    "wall_length_stretch": (
        "Wall length error <= 0.5%  [stretch]",
        lambda e: e["abs_error"] <= 0.005 * e["truth"],
        0.90,
    ),
    "ceiling_height": (
        "Ceiling height error <= 1.5cm",
        lambda e: e["abs_error"] <= 0.015,
        1.00,
    ),
    "floor_area": ("Floor area error <= 2%", lambda e: e["rel_error"] <= 0.02, 1.00),
    "opening_width": (
        "Door/window width error <= 2cm",
        lambda e: e["abs_error"] <= 0.02,
        0.85,
    ),
    "damage_area": (
        "Affected-area quantity within +/-10%",
        lambda e: e["rel_error"] <= 0.10,
        1.00,
    ),
}


@dataclass
class Comparison:
    kind: str
    name: str
    truth: float
    predicted: float
    half_width: float

    @property
    def abs_error(self) -> float:
        return abs(self.predicted - self.truth)

    @property
    def rel_error(self) -> float:
        return self.abs_error / max(abs(self.truth), 1e-6)

    @property
    def inside_interval(self) -> bool:
        return self.abs_error <= self.half_width

    def as_record(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "truth": self.truth,
            "predicted": self.predicted,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "half_width": self.half_width,
            "inside": self.inside_interval,
        }


def load_truth(path: Path) -> list[dict]:
    """Read a laser ground-truth CSV.

    Columns: kind, name, value  (metres). `kind` is one of wall_length,
    ceiling_height, floor_area, opening_width, damage_area; `name` must match
    the surface_ref / room name the pipeline emits.
    """
    rows: list[dict] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("kind") or row.get("value") in (None, ""):
                continue
            rows.append(
                {
                    "kind": row["kind"].strip(),
                    "name": row["name"].strip(),
                    "value": float(row["value"]),
                }
            )
    return rows


def index_result(result: dict) -> dict[tuple[str, str], tuple[float, float]]:
    """Map (kind, name) -> (value, half_width) for everything measurable."""
    index: dict[tuple[str, str], tuple[float, float]] = {}

    for wall in result["reconstruction"]["walls"]:
        measurement = wall["length"]
        index[("wall_length", wall["name"])] = (
            measurement["value"],
            measurement.get("half_width", 0.0),
        )
    for opening in result["reconstruction"].get("openings", []):
        key = f"{opening['wall']}:{opening['kind']}"
        index[("opening_width", key)] = (
            opening["width"]["value"],
            opening["width"].get("half_width", 0.0),
        )
    for room in result.get("rooms", []):
        index[("floor_area", room["name"])] = (
            room["area"]["value"],
            room["area"].get("half_width", 0.0),
        )
        height = room.get("ceiling_height") or {}
        if height.get("value"):
            index[("ceiling_height", room["name"])] = (
                height["value"],
                height.get("half_width", 0.0),
            )
    for region in result.get("damage", []):
        index[("damage_area", region["surface_ref"])] = (
            region["area"]["value"],
            region["area"].get("half_width", 0.0),
        )
    return index


def compare(result: dict, truth: list[dict]) -> tuple[list[Comparison], list[dict]]:
    index = index_result(result)
    matched: list[Comparison] = []
    missing: list[dict] = []
    for row in truth:
        key = (row["kind"], row["name"])
        if key not in index:
            missing.append(row)
            continue
        value, half_width = index[key]
        matched.append(
            Comparison(row["kind"], row["name"], row["value"], value, half_width)
        )
    return matched, missing


def report(comparisons: list[Comparison], coverage: float = 0.90) -> dict:
    records = [c.as_record() for c in comparisons]
    summary: dict = {"gates": {}, "counts": {}, "interval_coverage": {}}

    for gate, (label, predicate, target) in GATES.items():
        kind = gate.replace("_stretch", "")
        relevant = [r for r in records if r["kind"] == kind]
        if not relevant:
            continue
        passing = sum(1 for r in relevant if predicate(r))
        fraction = passing / len(relevant)
        summary["gates"][gate] = {
            "label": label,
            "passing": passing,
            "total": len(relevant),
            "fraction": round(fraction, 4),
            "target": target,
            "pass": fraction >= target,
        }

    for kind in {r["kind"] for r in records}:
        relevant = [r for r in records if r["kind"] == kind]
        errors = np.array([r["abs_error"] for r in relevant])
        inside = np.mean([r["inside"] for r in relevant])
        summary["counts"][kind] = {
            "n": len(relevant),
            "median_error_mm": round(float(np.median(errors)) * 1000, 2),
            "p90_error_mm": round(float(np.quantile(errors, 0.9)) * 1000, 2),
            "max_error_mm": round(float(errors.max()) * 1000, 2),
        }
        summary["interval_coverage"][kind] = {
            "observed": round(float(inside), 4),
            "claimed": coverage,
            "calibrated": abs(inside - coverage) <= 0.10,
        }
    return summary


def print_report(summary: dict, missing: list[dict], matched: int) -> None:
    print(f"\nMatched {matched} ground-truth measurements")
    if missing:
        print(f"  ! {len(missing)} unmatched (name mismatch or not reconstructed):")
        for row in missing[:6]:
            print(f"      {row['kind']}: {row['name']}")

    print("\nACCURACY GATES")
    print(f"  {'gate':<44} {'result':>12}  {'target':>7}  status")
    for gate, data in summary["gates"].items():
        status = "PASS" if data["pass"] else "FAIL"
        result = f"{data['passing']}/{data['total']} ({data['fraction'] * 100:.0f}%)"
        print(f"  {data['label']:<44} {result:>12}  {data['target'] * 100:>6.0f}%  {status}")

    print("\nERROR DISTRIBUTION")
    for kind, data in summary["counts"].items():
        print(
            f"  {kind:<18} n={data['n']:<4} median {data['median_error_mm']:>7.1f} mm"
            f"   p90 {data['p90_error_mm']:>7.1f} mm   max {data['max_error_mm']:>7.1f} mm"
        )

    print("\nINTERVAL CALIBRATION (claimed vs observed coverage)")
    for kind, data in summary["interval_coverage"].items():
        verdict = "ok" if data["calibrated"] else "MISCALIBRATED"
        print(
            f"  {kind:<18} claimed {data['claimed'] * 100:.0f}%   "
            f"observed {data['observed'] * 100:5.1f}%   {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--incumbent", help="incumbent scan result.json for head-to-head")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="fit the conformal interval scale and write bench/calibration.json",
    )
    parser.add_argument("--out", help="write the summary JSON here")
    args = parser.parse_args(argv)

    result = json.loads(Path(args.result).read_text())
    truth = load_truth(Path(args.truth))
    matched, missing = compare(result, truth)
    if not matched:
        print("No ground-truth rows matched the result. Check the `name` column "
              "against the surface names in result.json.")
        return 1

    summary = report(matched, args.coverage)
    print_report(summary, missing, len(matched))

    if args.incumbent:
        summary["head_to_head"] = _head_to_head(
            matched, json.loads(Path(args.incumbent).read_text()), truth
        )

    if args.fit_calibration:
        calibration = fit_calibration(
            [c.predicted for c in matched],
            [c.truth for c in matched],
            [max(c.half_width, 1e-4) for c in matched],
            coverage=args.coverage,
        )
        target = Path(__file__).parent / "calibration.json"
        target.write_text(json.dumps(calibration, indent=2))
        print(
            f"\nCalibration written to {target}: scale {calibration['scale']:.3f} "
            f"(coverage {calibration['coverage_before'] * 100:.0f}% -> "
            f"{calibration['coverage_after'] * 100:.0f}%)"
        )
        summary["calibration"] = calibration

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))

    failed = [g for g, d in summary["gates"].items() if not d["pass"] and "stretch" not in g]
    return 1 if failed else 0


def _head_to_head(matched: list[Comparison], incumbent: dict, truth: list[dict]) -> dict:
    """Per-dimension win/tie/loss against an incumbent scan of the same rooms."""
    incumbent_index = index_result(incumbent)
    wins = ties = losses = 0
    for comparison in matched:
        key = (comparison.kind, comparison.name)
        if key not in incumbent_index:
            continue
        their_error = abs(incumbent_index[key][0] - comparison.truth)
        margin = 0.002  # within 2 mm counts as a tie
        if comparison.abs_error < their_error - margin:
            wins += 1
        elif comparison.abs_error > their_error + margin:
            losses += 1
        else:
            ties += 1

    total = wins + ties + losses
    fraction = (wins + ties) / total if total else 0.0
    print(
        f"\nHEAD-TO-HEAD vs incumbent: {wins} win / {ties} tie / {losses} loss "
        f"({fraction * 100:.0f}% beat-or-tie, gate 70%)"
    )
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "beat_or_tie_fraction": round(fraction, 4),
        "pass": fraction >= 0.70,
    }


if __name__ == "__main__":
    sys.exit(main())
