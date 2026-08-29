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
    parser = argparse.ArgumentParser(
        description="Score pipeline output against laser ground truth and incumbent scans."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--truth",
        help="laser ground-truth CSV, for the wall/ceiling/floor/opening/"
             "damage-area gates. Omit to run only the gates below that "
             "don't need one.",
    )
    parser.add_argument("--incumbent", help="incumbent scan result.json for head-to-head")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="fit the conformal interval scale and write bench/calibration.json",
    )
    parser.add_argument(
        "--footprint-reference", type=float,
        help="laser-measured total stitched-footprint area (m2)",
    )
    parser.add_argument(
        "--damage-class-reference",
        help="CSV columns: surface_ref,damage_class -- scores classification macro F1",
    )
    parser.add_argument(
        "--water-reference",
        help="CSV columns: surface_ref,water_category,water_class -- scores "
             "IICRC Category/Class assignment accuracy",
    )
    parser.add_argument(
        "--iou-reference",
        help="CSV columns: surface_ref,u_lo,u_hi,v_lo,v_hi -- scores damage segmentation IoU",
    )
    parser.add_argument(
        "--concealed-reference",
        help="CSV column: surface_ref -- scores concealed-damage flag recall/precision",
    )
    parser.add_argument(
        "--scope-reference",
        help="CSV columns: surface_ref,action,material,quantity -- scores "
             "line-item recall against a reference scope",
    )
    parser.add_argument("--out", help="write the summary JSON here")
    args = parser.parse_args(argv)

    result = json.loads(Path(args.result).read_text())
    summary: dict = {"gates": {}, "counts": {}, "interval_coverage": {}}
    any_failure = False

    consistency = check_room_consistency(result)
    summary["room_consistency"] = consistency
    print("\nROOM CONSISTENCY (self-checked, no ground truth needed)")
    print(
        f"  {consistency['room_count']} rooms, {len(consistency['overlaps'])} "
        f"overlap(s), {len(consistency['adjacency_errors'])} adjacency "
        f"error(s)  {'PASS' if consistency['pass'] else 'FAIL'}"
    )
    any_failure = any_failure or not consistency["pass"]

    if args.truth:
        truth = load_truth(Path(args.truth))
        matched, missing = compare(result, truth)
        if not matched:
            print("No ground-truth rows matched the result. Check the `name` "
                  "column against the surface names in result.json.")
            any_failure = True
        else:
            gate_summary = report(matched, args.coverage)
            summary.update(gate_summary)
            print_report(gate_summary, missing, len(matched))

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

            failed_core = [
                g for g, d in gate_summary["gates"].items()
                if not d["pass"] and "stretch" not in g
            ]
            any_failure = any_failure or bool(failed_core)

    if args.footprint_reference is not None:
        data = score_footprint_error(result, args.footprint_reference)
        summary["footprint_error"] = data
        print(f"\nFOOTPRINT ERROR: {data['rel_error'] * 100:.1f}% "
              f"(gate <= 2%)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.damage_class_reference:
        data = score_damage_classification(result, load_reference_csv(args.damage_class_reference))
        summary["damage_classification"] = data
        print(f"\nDAMAGE CLASSIFICATION macro F1: {data['macro_f1']:.3f} "
              f"(gate >= 0.85)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.water_reference:
        data = score_water_category_class(result, load_reference_csv(args.water_reference))
        summary["water_category_class"] = data
        print(f"\nWATER CATEGORY/CLASS accuracy: {data['fraction'] * 100:.1f}% "
              f"(gate >= 70%)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.iou_reference:
        data = score_damage_iou(result, load_reference_csv(args.iou_reference))
        summary["damage_iou"] = data
        print(f"\nDAMAGE SEGMENTATION IoU: {data['fraction'] * 100:.1f}% of "
              f"regions >= 0.5 IoU (gate >= 80%)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.concealed_reference:
        data = score_concealed_flags(result, load_reference_csv(args.concealed_reference))
        summary["concealed_flags"] = data
        print(f"\nCONCEALED-DAMAGE FLAGS: recall {data['recall'] * 100:.1f}%, "
              f"precision {data['precision'] * 100:.1f}% "
              f"(gate >= 60%/50%)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.scope_reference:
        data = score_line_item_recall(result, load_reference_csv(args.scope_reference))
        summary["line_item_recall"] = data
        print(f"\nLINE-ITEM RECALL: {data['fraction'] * 100:.1f}% "
              f"(gate >= 80%)  {'PASS' if data['pass'] else 'FAIL'}")
        any_failure = any_failure or not data["pass"]

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))

    return 1 if any_failure else 0


def check_room_consistency(result: dict) -> dict:
    """No-overlap and adjacency-graph self-consistency -- the parts of the
    "multi-room stitched footprint" gate that are checkable against the
    pipeline's own output, with no ground truth involved. The remaining part
    of that gate, footprint error against a laser-measured total area,
    genuinely needs an external reference and is `score_footprint_error`
    below.
    """
    from shapely.geometry import Polygon

    rooms = result.get("rooms", [])
    polygons: dict[int, Polygon] = {}
    for room in rooms:
        polygon = room.get("polygon")
        if not polygon or len(polygon) < 3:
            continue
        shape = Polygon(polygon)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.area > 0:
            polygons[room["id"]] = shape

    overlaps = []
    ids = sorted(polygons)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            overlap = polygons[a].intersection(polygons[b]).area
            if overlap <= 0:
                continue
            smaller = min(polygons[a].area, polygons[b].area)
            fraction = overlap / smaller if smaller > 0 else 0.0
            if fraction > 0.01:
                overlaps.append({"a": a, "b": b, "overlap_fraction": round(fraction, 4)})

    room_ids = {room["id"] for room in rooms}
    adjacency_errors = []
    for edge in result.get("adjacency", []):
        if edge["a"] not in room_ids or edge["b"] not in room_ids:
            adjacency_errors.append(f"edge references unknown room: {edge}")
    by_id = {room["id"]: room for room in rooms}
    for room in rooms:
        for neighbour in room.get("neighbours", []):
            if neighbour == room["id"]:
                adjacency_errors.append(f"room {room['id']} lists itself as a neighbour")
                continue
            other = by_id.get(neighbour)
            if other is None:
                adjacency_errors.append(
                    f"room {room['id']} neighbours unknown room {neighbour}"
                )
            elif room["id"] not in other.get("neighbours", []):
                adjacency_errors.append(
                    f"asymmetric adjacency: {room['id']} -> {neighbour} but not back"
                )

    passing = not overlaps and not adjacency_errors
    return {
        "room_count": len(rooms),
        "overlaps": overlaps,
        "adjacency_errors": adjacency_errors,
        "pass": passing,
    }


def score_footprint_error(result: dict, reference_area_m2: float) -> dict:
    """Multi-room stitched footprint error against a laser-measured total
    unit area. `reference_area_m2` is that one laser number -- there is no
    per-room reference format needed, since the gate is about the whole
    stitched footprint, not individual rooms.
    """
    predicted = sum(room["area"]["value"] for room in result.get("rooms", []))
    error = abs(predicted - reference_area_m2) / max(reference_area_m2, 1e-6)
    return {
        "predicted_m2": round(predicted, 3),
        "reference_m2": reference_area_m2,
        "rel_error": round(error, 4),
        "gate": 0.02,
        "stretch": 0.01,
        "pass": error <= 0.02,
    }


def load_reference_csv(path: Path) -> list[dict]:
    """Generic CSV loader for the reference files below -- every row becomes
    a dict keyed by column header, values left as strings (each scorer casts
    the columns it needs)."""
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def score_damage_classification(result: dict, reference: list[dict]) -> dict:
    """Macro F1 for water/fire/mold/combined/none, matched by surface_ref.

    Reference CSV columns: surface_ref, damage_class. One row per annotated
    surface. Where a surface has multiple fused regions, the pipeline's
    largest-area region on that surface is what's compared -- the dominant
    call is the one a human reviewer would actually check against.
    """
    predicted_by_surface: dict[str, tuple[str, float]] = {}
    for region in result.get("damage", []):
        key = region["surface_ref"]
        area = region["area"]["value"]
        if key not in predicted_by_surface or area > predicted_by_surface[key][1]:
            predicted_by_surface[key] = (region["damage_class"], area)

    classes = ["water", "fire", "mold", "combined", "none"]
    pairs = [
        (row["damage_class"], predicted_by_surface.get(row["surface_ref"], ("none", 0.0))[0])
        for row in reference
    ]
    f1_scores = []
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "macro_f1": round(macro_f1, 4),
        "n": len(reference),
        "gate": 0.85,
        "stretch": 0.93,
        "pass": macro_f1 >= 0.85,
    }


def score_water_category_class(result: dict, reference: list[dict]) -> dict:
    """Water IICRC Category/Class assignment accuracy, matched by surface_ref.

    Reference CSV columns: surface_ref, water_category (1-3, optional),
    water_class (1-4, optional). A row is "correct" if every reference field
    it supplies matches the pipeline's dominant region on that surface;
    rows with neither field are skipped (nothing to check).
    """
    predicted_by_surface: dict[str, dict] = {}
    for region in result.get("damage", []):
        if region["damage_class"] not in ("water", "combined"):
            continue
        key = region["surface_ref"]
        area = region["area"]["value"]
        if key not in predicted_by_surface or area > predicted_by_surface[key].get("_area", -1):
            predicted_by_surface[key] = {
                "water_category": region.get("water_category"),
                "water_class": region.get("water_class"),
                "_area": area,
            }

    checked = 0
    correct = 0
    for row in reference:
        want_category = row.get("water_category") or None
        want_class = row.get("water_class") or None
        if not want_category and not want_class:
            continue
        checked += 1
        predicted = predicted_by_surface.get(row["surface_ref"], {})
        ok = True
        if want_category and str(predicted.get("water_category")) != str(want_category):
            ok = False
        if want_class and str(predicted.get("water_class")) != str(want_class):
            ok = False
        correct += int(ok)

    fraction = correct / checked if checked else 0.0
    return {
        "correct": correct,
        "checked": checked,
        "fraction": round(fraction, 4),
        "gate": 0.70,
        "stretch": 0.85,
        "pass": fraction >= 0.70,
    }


def score_damage_iou(result: dict, reference: list[dict]) -> dict:
    """Damage-region IoU against a reference extent, matched by surface_ref.

    Reference CSV columns: surface_ref, u_lo, u_hi, v_lo, v_hi (metres, same
    along-wall/height convention as `extent.u_range`/`v_range` in
    result.json). Compared as axis-aligned boxes -- that is the precision
    the schema itself stores extent at, so it is what an apples-to-apples
    IoU can be computed from without re-deriving pixel masks.
    """
    predicted_by_surface: dict[str, tuple[float, float, float, float]] = {}
    for region in result.get("damage", []):
        key = region["surface_ref"]
        extent = region["extent"]
        box = (*extent["u_range"], *extent["v_range"])
        area = region["area"]["value"]
        if key not in predicted_by_surface or area > predicted_by_surface.get(f"_{key}_area", -1):
            predicted_by_surface[key] = box
            predicted_by_surface[f"_{key}_area"] = area

    ious = []
    for row in reference:
        box = predicted_by_surface.get(row["surface_ref"])
        if box is None:
            ious.append(0.0)
            continue
        pu0, pu1, pv0, pv1 = box
        ru0, ru1 = float(row["u_lo"]), float(row["u_hi"])
        rv0, rv1 = float(row["v_lo"]), float(row["v_hi"])
        inter_u = max(0.0, min(pu1, ru1) - max(pu0, ru0))
        inter_v = max(0.0, min(pv1, rv1) - max(pv0, rv0))
        intersection = inter_u * inter_v
        union = (pu1 - pu0) * (pv1 - pv0) + (ru1 - ru0) * (rv1 - rv0) - intersection
        ious.append(intersection / union if union > 0 else 0.0)

    passing = sum(1 for iou in ious if iou >= 0.5)
    fraction = passing / len(ious) if ious else 0.0
    return {
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "passing": passing,
        "total": len(ious),
        "fraction": round(fraction, 4),
        "gate": 0.80,
        "gate_iou": 0.5,
        "stretch_iou": 0.7,
        "pass": fraction >= 0.80,
    }


def score_concealed_flags(result: dict, reference: list[dict]) -> dict:
    """Concealed-damage flag recall/precision, matched by surface_ref.

    Reference CSV columns: surface_ref (one row per estimator-identified
    concealed-damage location). A predicted flag on the same surface_ref
    counts as a match -- rule_id is not required to match, since the
    reference is "damage is concealed here", not "for this exact reason".
    """
    predicted_surfaces = {flag["surface_ref"] for flag in result.get("concealed", [])}
    reference_surfaces = {row["surface_ref"] for row in reference}

    true_positives = predicted_surfaces & reference_surfaces
    recall = len(true_positives) / len(reference_surfaces) if reference_surfaces else 0.0
    precision = len(true_positives) / len(predicted_surfaces) if predicted_surfaces else 0.0
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "reference_count": len(reference_surfaces),
        "predicted_count": len(predicted_surfaces),
        "gate_recall": 0.60,
        "gate_precision": 0.50,
        "stretch_recall": 0.75,
        "stretch_precision": 0.65,
        "pass": recall >= 0.60 and precision >= 0.50,
    }


def score_line_item_recall(result: dict, reference: list[dict]) -> dict:
    """Line-item recall against a reference scope of work.

    Reference CSV columns: surface_ref, action, material, quantity. A
    reference row is "found" if the pipeline emitted a line item on the same
    surface_ref with the same action and material, and quantity within
    +/-10%.
    """
    predicted = result.get("scope", {}).get("line_items", [])
    found = 0
    for row in reference:
        want_quantity = float(row["quantity"])
        for item in predicted:
            if (
                item["surface_ref"] == row["surface_ref"]
                and item["action"] == row["action"]
                and item["material"] == row["material"]
                and abs(item["quantity"] - want_quantity) <= 0.10 * max(want_quantity, 1e-6)
            ):
                found += 1
                break

    fraction = found / len(reference) if reference else 0.0
    return {
        "found": found,
        "total": len(reference),
        "fraction": round(fraction, 4),
        "gate": 0.80,
        "stretch": 0.90,
        "pass": fraction >= 0.80,
    }


def _head_to_head(matched: list[Comparison], incumbent: dict, truth: list[dict]) -> dict:
    """Per-dimension win/tie/loss against an incumbent scan of the same rooms."""
    incumbent_index = index_result(incumbent)
    wins = ties = losses = 0
    for comparison in matched:
        key = (comparison.kind, comparison.name)
        if key not in incumbent_index:
            continue
        their_error = abs(incumbent_index[key][0] - comparison.truth)
        margin = 0.002
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
