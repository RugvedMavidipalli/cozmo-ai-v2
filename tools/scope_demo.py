from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import yaml


from cozmo_ai_v2.pipeline.damage.fusion import DamageRegion
from cozmo_ai_v2.pipeline.rooms import Room
from cozmo_ai_v2.pipeline.scope import ScopeEngine

REPO_ROOT = Path(__file__).resolve().parent.parent


def sample_scene() -> tuple[list[DamageRegion], list[Room], dict[str, float]]:
    """A small but representative damaged property."""
    rooms = [
        Room(
            id=0, name="room_1", area=14.2,
            centroid=np.array([0.0, 0.0]), floor_height=0.0, ceiling_height=2.44,
            polygon=np.array([[0, 0], [4.2, 0], [4.2, 3.38], [0, 3.38]]),
        ),
        Room(
            id=1, name="room_2", area=8.6,
            centroid=np.array([6.0, 0.0]), floor_height=0.0, ceiling_height=2.44,
            polygon=np.array([[5, 0], [8.2, 0], [8.2, 2.7], [5, 2.7]]),
        ),
    ]
    regions = [
        DamageRegion(
            id="r1", surface_key="room_1.north_wall", room_id=0,
            damage_class="water", subtype="staining", area=3.10,
            bounds_u=(0.4, 3.6), bounds_v=(0.0, 0.42),
            view_count=6, confidence=0.86, water_category=2, water_class=2,
            mask_method="sam2",
        ),
        DamageRegion(
            id="r2", surface_key="floor", room_id=0,
            damage_class="water", subtype="saturation", area=6.40,
            bounds_u=(0.0, 4.2), bounds_v=(0.0, 0.0),
            view_count=9, confidence=0.91, water_category=2, water_class=3,
            mask_method="sam2",
        ),
        DamageRegion(
            id="r3", surface_key="room_2.east_wall", room_id=1,
            damage_class="mold", subtype="colonized", area=0.34,
            bounds_u=(1.1, 1.8), bounds_v=(0.0, 0.55),
            view_count=4, confidence=0.79, mold_condition=3,
            mask_method="sam2",
        ),
        DamageRegion(
            id="r4", surface_key="room_2.north_wall", room_id=1,
            damage_class="fire", subtype="soot", area=1.85,
            bounds_u=(0.2, 2.4), bounds_v=(1.1, 2.3),
            view_count=5, confidence=0.83, mask_method="sam2",
        ),
    ]
    wall_lengths = {
        "room_1.north_wall": 4.20,
        "room_2.east_wall": 2.70,
        "room_2.north_wall": 3.20,
    }
    return regions, rooms, wall_lengths


def apply_override(rules: dict, assignment: str) -> dict:
    """Apply a `dotted.path=value` override to a copy of the rules."""
    path, _, raw = assignment.partition("=")
    node = rules
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    node[keys[-1]] = value
    return rules


def build(rules: dict) -> tuple[list, list]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(rules, handle)
        temp = handle.name
    engine = ScopeEngine(temp)
    regions, rooms, wall_lengths = sample_scene()
    items, concealed = engine.build(regions, rooms, wall_lengths)
    Path(temp).unlink(missing_ok=True)
    return items, concealed


def print_scope(items: list, concealed: list) -> None:
    print(f"\n{len(items)} line items\n")
    print(f"  {'action':<18}{'material':<24}{'qty':>9} {'unit':<8}{'rule':<28}surface")
    print("  " + "-" * 104)
    for item in sorted(items, key=lambda i: (i.room_id or 0, i.trade)):
        print(
            f"  {item.action:<18}{item.material[:23]:<24}{item.quantity:>9.2f} "
            f"{item.unit:<8}{item.rule_id[:27]:<28}{item.surface_ref}"
        )

    print(f"\n{len(concealed)} concealed-damage flags\n")
    for flag in sorted(concealed, key=lambda f: -f.probability):
        print(
            f"  [{flag.rule_id}] p={flag.probability:.2f}  {flag.inferred:<34}"
            f"{flag.surface_ref}"
        )
        print(f"          {flag.rationale[:96]}")


def print_delta(before: list, after: list, label: str) -> None:
    """Diff two scopes by (code, surface), the identity an estimator reads."""
    def key(item):
        return (item.code, item.surface_ref)

    left = {key(i): i for i in before}
    right = {key(i): i for i in after}

    print(f"\n{'=' * 96}\nLINE-ITEM DELTA — {label}\n{'=' * 96}")
    print(f"  {'change':<9}{'action':<16}{'material':<22}{'before':>9}{'after':>9}"
          f"{'delta':>9}  surface")
    print("  " + "-" * 94)

    changed = 0
    for identity in sorted(set(left) | set(right)):
        a, b = left.get(identity), right.get(identity)
        before_qty = a.quantity if a else 0.0
        after_qty = b.quantity if b else 0.0
        if abs(after_qty - before_qty) < 1e-6:
            continue
        changed += 1
        item = b or a
        marker = "added" if not a else "removed" if not b else "changed"
        print(
            f"  {marker:<9}{item.action:<16}{item.material[:21]:<22}"
            f"{before_qty:>9.2f}{after_qty:>9.2f}{after_qty - before_qty:>+9.2f}  "
            f"{item.surface_ref}"
        )

    if not changed:
        print(
            "  (no quantities changed — the margin increase was absorbed by\n"
            "   snapping to the same drywall sheet line; see the sweep below)"
        )
    else:
        total_before = sum(i.quantity for i in before if i.unit == "m2")
        total_after = sum(i.quantity for i in after if i.unit == "m2")
        print(
            f"\n  {changed} line items changed; total m2 "
            f"{total_before:.2f} -> {total_after:.2f} "
            f"({total_after - total_before:+.2f} m2)"
        )
    for item in after:
        if item.code == "drywall_remove":
            print(f"\n  basis now reads: {item.basis}")
            break


def print_sweep(rules: dict, margins=(0.0, 0.15, 0.30, 0.45, 0.60, 0.90, 1.20)) -> None:
    """Response of the scope to the flood-cut margin, with snapping on and off.

    The scope is a step function of this parameter, not a linear one, because
    drywall is removed to sheet lines. Showing the steps is a more honest
    defence of the rule than quoting a single delta: it says exactly which
    waterlines are sensitive to the margin and which are not.
    """
    print(f"\n{'=' * 96}\nSENSITIVITY — flood-cut margin vs drywall removed\n{'=' * 96}")
    print(f"  {'margin':>8}  {'snapped to sheet lines':>26}  {'raw (snapping off)':>22}")
    print("  " + "-" * 62)

    for margin in margins:
        snapped = copy.deepcopy(rules)
        snapped["water"]["flood_cut"]["base_height"] = margin
        items_snapped, _ = build(snapped)

        raw = copy.deepcopy(snapped)
        raw["water"]["flood_cut"]["snap_to_standard"] = False
        items_raw, _ = build(raw)

        def drywall(items):
            return sum(i.quantity for i in items if i.code == "drywall_remove")

        print(
            f"  {margin:>7.2f}m  {drywall(items_snapped):>24.2f} m2  "
            f"{drywall(items_raw):>20.2f} m2"
        )

    print(
        "\n  The snapped column is a step function: a margin change only moves the\n"
        "  scope when it pushes the cut past 2 ft / 4 ft / full sheet. That is why\n"
        "  30 -> 60 cm leaves this capture's quantities unchanged."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the scope engine, and show a rule change as a line-item delta."
    )
    parser.add_argument("--rules", default=str(REPO_ROOT / "rules.yaml"))
    parser.add_argument(
        "--set", dest="override",
        help="dotted.path=value override, e.g. water.flood_cut.base_height=0.60",
    )
    args = parser.parse_args(argv)

    rules = yaml.safe_load(Path(args.rules).read_text())
    baseline_items, baseline_concealed = build(copy.deepcopy(rules))

    print("=" * 96)
    print(f"BASELINE — flood cut = visible waterline + "
          f"{rules['water']['flood_cut']['base_height']:.2f} m")
    print("=" * 96)
    print_scope(baseline_items, baseline_concealed)

    override = args.override or "water.flood_cut.base_height=0.60"
    modified = apply_override(copy.deepcopy(rules), override)
    modified_items, _ = build(modified)
    print_delta(baseline_items, modified_items, override)
    print_sweep(rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
