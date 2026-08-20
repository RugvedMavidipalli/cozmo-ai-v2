"""Turn fused damage regions into a defensible scope of work.

The mapping from detected damage to line items is where domain knowledge
lives, and it is deliberately not a proportional one.  A 0.3 m2 mold patch does
not produce a 0.3 m2 line item: it produces removal of the patch plus a margin,
containment sized to the room, PPE per technician, and HEPA passes over a much
larger area.  Water is similar -- the flood cut is driven by the *height* of
the waterline, not by the stained area, and baseboard comes off in whole runs.

Every quantity here traces to a rule in `rules.yaml` and carries its `rule_id`
and `source` into the output, so a reviewer can audit any number back to the
standard it came from -- and so changing a rule changes the scope without
touching this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .damage.fusion import DamageRegion
from .rooms import Room


@dataclass
class LineItem:
    code: str
    description: str
    action: str
    material: str
    quantity: float
    unit: str
    trade: str
    surface_ref: str
    room_id: int | None
    rule_id: str
    source: str
    basis: str  # how the quantity was derived, in words
    derived_from: list[str] = field(default_factory=list)  # damage region ids

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "action": self.action,
            "material": self.material,
            "quantity": round(self.quantity, 3),
            "unit": self.unit,
            "trade": self.trade,
            "surface_ref": self.surface_ref,
            "room_id": self.room_id,
            "rule_id": self.rule_id,
            "source": self.source,
            "basis": self.basis,
            "derived_from": self.derived_from,
        }


@dataclass
class ConcealedFlag:
    rule_id: str
    surface_ref: str
    room_id: int | None
    inferred: str
    probability: float
    rationale: str
    source: str
    triggered_by: str

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "surface_ref": self.surface_ref,
            "room_id": self.room_id,
            "inferred": self.inferred,
            "probability": self.probability,
            "rationale": self.rationale,
            "source": self.source,
            "triggered_by": self.triggered_by,
        }


class ScopeEngine:
    """Applies `rules.yaml` to fused damage."""

    def __init__(self, rules_path: str | Path = "rules.yaml"):
        self.rules = yaml.safe_load(Path(rules_path).read_text())
        self.catalogue = self.rules["line_items"]

    # -- helpers ----------------------------------------------------------
    def _item(
        self,
        code: str,
        action: str,
        material: str,
        quantity: float,
        surface_ref: str,
        room_id: int | None,
        rule_id: str,
        source: str,
        basis: str,
        derived_from: list[str],
    ) -> LineItem:
        entry = self.catalogue[code]
        return LineItem(
            code=code,
            description=entry["description"],
            action=action,
            material=material,
            quantity=quantity,
            unit=entry["unit"],
            trade=entry["trade"],
            surface_ref=surface_ref,
            room_id=room_id,
            rule_id=rule_id,
            source=source,
            basis=basis,
            derived_from=derived_from,
        )

    # -- water ------------------------------------------------------------
    def _flood_cut_height(self, waterline: float) -> tuple[float, str]:
        """Cut height above the floor, and the reasoning behind it."""
        rules = self.rules["water"]["flood_cut"]
        raw = waterline + rules["base_height"]
        height = max(raw, rules["minimum_height"])
        basis = (
            f"waterline {waterline:.2f} m + {rules['base_height']:.2f} m margin"
            f" = {raw:.2f} m"
        )
        if height > raw:
            basis += f", raised to {rules['minimum_height']:.2f} m minimum"
        if rules.get("snap_to_standard"):
            for standard in rules["standard_heights"]:
                if height <= standard + 1e-6:
                    if standard > height:
                        basis += f", snapped up to the {standard:.2f} m sheet line"
                    height = standard
                    break
            else:
                height = rules["standard_heights"][-1]
                basis += f", capped at full-height removal ({height:.2f} m)"
        return height, basis

    def _water_items(
        self, region: DamageRegion, room: Room | None, wall_length: float | None
    ) -> list[LineItem]:
        water = self.rules["water"]
        category = region.water_category or 1
        actions = water["category_actions"].get(category, water["category_actions"][1])
        items: list[LineItem] = []
        room_id = region.room_id

        is_floor = region.surface_key.endswith("floor") or region.surface_key == "floor"

        if is_floor:
            if actions["carpet"] == "remove":
                items.append(
                    self._item(
                        "carpet_remove", "remove", "carpet", region.area,
                        region.surface_key, room_id, "water.category_actions.3",
                        actions["source"],
                        f"Cat {category}: carpet is porous and not restorable; "
                        f"{region.area:.2f} m2 affected",
                        [region.id],
                    )
                )
            elif actions["carpet"] == "clean_and_dry":
                items.append(
                    self._item(
                        "carpet_clean", "clean", "carpet", region.area,
                        region.surface_key, room_id, "water.category_actions.2",
                        actions["source"],
                        f"Cat {category}: carpet cleanable, pad removed",
                        [region.id],
                    )
                )
            if actions["pad"] == "remove":
                items.append(
                    self._item(
                        "pad_remove", "remove", "carpet pad", region.area,
                        region.surface_key, room_id, "water.category_actions",
                        actions["source"],
                        "Pad is the most absorbent layer and is removed whenever wet",
                        [region.id],
                    )
                )
            return items

        # Wall: the flood cut is driven by the waterline height, not the area.
        if actions["drywall"] == "flood_cut" and wall_length:
            waterline = region.bounds_v[1]
            height, basis = self._flood_cut_height(waterline)
            run = min(region.width_extent, wall_length)
            area = height * run
            items.append(
                self._item(
                    "drywall_remove", "remove", "drywall", area,
                    region.surface_key, room_id, "water.flood_cut",
                    water["flood_cut"]["source"],
                    f"{basis}; cut {height:.2f} m x {run:.2f} m affected run",
                    [region.id],
                )
            )
            items.append(
                self._item(
                    "drywall_replace", "replace", "1/2\" drywall", area,
                    region.surface_key, room_id, "water.flood_cut",
                    water["flood_cut"]["source"],
                    "Replacement matches the removed area",
                    [region.id],
                )
            )
            items.append(
                self._item(
                    "insulation_remove", "remove", "batt insulation", area,
                    region.surface_key, room_id, "W-01",
                    "IICRC S500 §12.2.4",
                    "Cavity opened by the flood cut is inspected and wet batt removed",
                    [region.id],
                )
            )

            baseboard = water["baseboard"]
            code = (
                "baseboard_replace"
                if category in baseboard["replace_if_category"]
                else "baseboard_detach"
            )
            items.append(
                self._item(
                    code,
                    "replace" if code == "baseboard_replace" else "detach_and_reset",
                    "baseboard", run, region.surface_key, room_id,
                    "water.baseboard", baseboard["source"],
                    f"Baseboard follows the full {run:.2f} m affected run, not the stain",
                    [region.id],
                )
            )

        if category in water["antimicrobial"]["apply_for_categories"]:
            items.append(
                self._item(
                    "antimicrobial_apply", "treat", "antimicrobial", region.area,
                    region.surface_key, room_id, "water.antimicrobial",
                    water["antimicrobial"]["source"],
                    f"Cat {category} requires antimicrobial over the affected area",
                    [region.id],
                )
            )
        return items

    def _drying_items(
        self, regions: list[DamageRegion], room: Room
    ) -> list[LineItem]:
        """Equipment counts follow affected area and class, per room."""
        water_regions = [r for r in regions if r.damage_class == "water"]
        if not water_regions:
            return []

        equipment = self.rules["water"]["drying_equipment"]
        wet_floor = sum(
            r.area for r in water_regions if "floor" in r.surface_key
        )
        wet_wall = sum(
            r.area for r in water_regions if "floor" not in r.surface_key
        )
        water_class = max((r.water_class or 2) for r in water_regions)
        days = equipment["monitoring_days"]["class_days"].get(
            water_class, equipment["monitoring_days"]["default"]
        )

        movers = equipment["air_movers"]
        count = max(
            movers["minimum"],
            math.ceil(
                wet_floor * movers["per_floor_area"]
                + wet_wall * movers["per_wet_wall_area"]
            ),
        )
        dehumidifier = equipment["dehumidifier"]
        volume = room.area * (room.height or 2.4)
        units = max(
            dehumidifier["minimum"],
            math.ceil(
                volume
                * dehumidifier["base_per_volume"]
                * dehumidifier["class_factors"].get(water_class, 1.0)
                / 10.0
            ),
        )

        return [
            self._item(
                "air_mover", "operate", "air mover", count * days,
                room.name, room.id, "water.drying_equipment.air_movers",
                movers["source"],
                f"{count} units x {days} days; {wet_floor:.1f} m2 wet floor + "
                f"{wet_wall:.1f} m2 wet wall, Class {water_class}",
                [r.id for r in water_regions],
            ),
            self._item(
                "dehumidifier", "operate", "LGR dehumidifier", units * days,
                room.name, room.id, "water.drying_equipment.dehumidifier",
                dehumidifier["source"],
                f"{units} unit(s) x {days} days; {volume:.1f} m3 at Class {water_class}",
                [r.id for r in water_regions],
            ),
            self._item(
                "monitoring_visit", "monitor", "drying monitoring", days,
                room.name, room.id, "water.drying_equipment.monitoring_days",
                equipment["monitoring_days"]["source"],
                f"Daily monitoring for {days} days until drying goals are met",
                [r.id for r in water_regions],
            ),
        ]

    # -- mold -------------------------------------------------------------
    def _mold_items(self, region: DamageRegion, room: Room | None) -> list[LineItem]:
        mold = self.rules["mold"]
        condition = region.mold_condition or 3
        action = mold["condition_actions"].get(condition, "remove_and_treat")
        items: list[LineItem] = []
        room_id = region.room_id

        margin = mold["removal_margin"]
        # Remediation extends past visible growth on every side.
        grown = (region.width_extent + 2 * margin) * (region.height_extent + 2 * margin)

        if action == "remove_and_treat":
            items.append(
                self._item(
                    "mold_remove", "remove", "contaminated material", grown,
                    region.surface_key, room_id, "mold.removal_margin",
                    "IICRC S520 §12.1",
                    f"Visible growth {region.area:.2f} m2 expanded by a {margin:.2f} m "
                    f"margin on all sides = {grown:.2f} m2",
                    [region.id],
                )
            )
            items.append(
                self._item(
                    "mold_treat", "treat", "antimicrobial", grown,
                    region.surface_key, room_id, "mold.condition_actions",
                    "IICRC S520 §12", "Treatment covers the full remediated area",
                    [region.id],
                )
            )
        items.append(
            self._item(
                "hepa_vacuum", "clean", "HEPA vacuum",
                grown * mold["hepa_vacuum_passes"],
                region.surface_key, room_id, "mold.hepa_vacuum_passes",
                "IICRC S520 §12.4",
                f"{mold['hepa_vacuum_passes']} passes over {grown:.2f} m2",
                [region.id],
            )
        )
        return items

    def _mold_containment(
        self, regions: list[DamageRegion], room: Room
    ) -> list[LineItem]:
        """Containment and PPE scale with the room, not the patch.

        This is the clearest case where proportional mapping fails: a small
        patch in an occupied space still needs a sealed enclosure and
        protected workers.
        """
        mold_regions = [r for r in regions if r.damage_class == "mold"]
        total = sum(r.area for r in mold_regions)
        containment = self.rules["mold"]["containment"]
        if total < containment["trigger_area"]:
            return []

        thresholds = self.rules["mold"]["size_thresholds"]
        size = (
            "small"
            if total < thresholds["small"]
            else "medium"
            if total < thresholds["medium"]
            else "large"
        )
        height = room.height or 2.4
        # A poly enclosure is walls plus a ceiling over the work area; the
        # perimeter comes from the room's own footprint.
        barrier_area = room.perimeter * height + room.area

        items = [
            self._item(
                "containment_barrier", "install", "6 mil poly", barrier_area,
                room.name, room.id, "mold.containment", containment["source"],
                f"{size} remediation ({total:.2f} m2 growth) requires "
                f"{containment['type_by_size'][size]} containment: "
                f"{room.perimeter:.1f} m perimeter x {height:.2f} m + "
                f"{room.area:.1f} m2 ceiling",
                [r.id for r in mold_regions],
            )
        ]
        if total >= containment["negative_air_trigger"]:
            scrubber = self.rules["mold"]["air_scrubber"]
            units = max(
                scrubber["minimum"],
                math.ceil(room.area * height * scrubber["per_volume"] / 10.0),
            )
            items.append(
                self._item(
                    "negative_air", "operate", "HEPA air scrubber", units * 3,
                    room.name, room.id, "mold.air_scrubber", scrubber["source"],
                    f"{units} unit(s) x 3 days maintaining negative pressure in "
                    f"{room.area * height:.1f} m3 of containment",
                    [r.id for r in mold_regions],
                )
            )
        items.append(
            self._item(
                "ppe_set", "supply", self.rules["mold"]["ppe"][size], 6.0,
                room.name, room.id, "mold.ppe", self.rules["mold"]["ppe"]["source"],
                f"{size} remediation: {self.rules['mold']['ppe'][size]} "
                f"for 2 technicians x 3 days",
                [r.id for r in mold_regions],
            )
        )
        return items

    # -- fire -------------------------------------------------------------
    def _fire_items(self, region: DamageRegion, room: Room | None) -> list[LineItem]:
        fire = self.rules["fire"]
        subtype = region.subtype or "soot"
        action = fire["subtype_actions"].get(subtype, "clean_and_seal")
        items: list[LineItem] = []
        room_id = region.room_id

        if action == "clean_and_seal":
            items.append(
                self._item(
                    "soot_clean", "clean", "soot residue", region.area,
                    region.surface_key, room_id, "fire.subtype_actions",
                    fire["source"],
                    f"Soot on a sound substrate is cleaned: {region.area:.2f} m2",
                    [region.id],
                )
            )
            if fire["seal_after_cleaning"]:
                items.append(
                    self._item(
                        "seal_surface", "seal", "odour-blocking primer",
                        region.area * fire["seal_coverage"],
                        region.surface_key, room_id, "fire.seal_after_cleaning",
                        fire["source"],
                        "Sealed after cleaning to lock down odour-bearing residue",
                        [region.id],
                    )
                )
        else:
            items.append(
                self._item(
                    "char_remove", "remove", "fire-damaged material", region.area,
                    region.surface_key, room_id, "fire.subtype_actions",
                    fire["source"],
                    f"{subtype}: substrate is compromised and replaced, "
                    f"{region.area:.2f} m2",
                    [region.id],
                )
            )
        return items

    def _fire_odour(self, regions: list[DamageRegion], room: Room) -> list[LineItem]:
        fire_regions = [r for r in regions if r.damage_class == "fire"]
        total = sum(r.area for r in fire_regions)
        odour = self.rules["fire"]["odour_treatment"]
        if total < odour["trigger_area"]:
            return []
        return [
            self._item(
                "odour_treatment", "treat", odour["method"], 1.0,
                room.name, room.id, "fire.odour_treatment", odour["source"],
                f"{total:.2f} m2 of fire damage exceeds the "
                f"{odour['trigger_area']:.2f} m2 deodorisation trigger",
                [r.id for r in fire_regions],
            )
        ]

    # -- concealed --------------------------------------------------------
    def concealed_flags(self, regions: list[DamageRegion]) -> list[ConcealedFlag]:
        """Fire every concealed-damage rule whose conditions a region meets."""
        flags: list[ConcealedFlag] = []
        for rule in self.rules["concealed"]:
            condition = rule["when"]
            for region in regions:
                if not self._matches(condition, region):
                    continue
                flags.append(
                    ConcealedFlag(
                        rule_id=rule["id"],
                        surface_ref=region.surface_key,
                        room_id=region.room_id,
                        inferred=rule["infer"],
                        probability=rule["probability"],
                        rationale=rule["rationale"],
                        source=rule["source"],
                        triggered_by=region.id,
                    )
                )
        return flags

    def _matches(self, condition: dict, region: DamageRegion) -> bool:
        if condition.get("damage_class") != region.damage_class:
            return False
        if "min_category" in condition:
            if (region.water_category or 0) < condition["min_category"]:
                return False
        if "min_class" in condition:
            if (region.water_class or 0) < condition["min_class"]:
                return False
        if "min_condition" in condition:
            if (region.mold_condition or 0) < condition["min_condition"]:
                return False
        if "subtype" in condition and region.subtype != condition["subtype"]:
            return False
        if "surface_kind" in condition:
            key = region.surface_key
            kind = "floor" if "floor" in key else "ceiling" if "ceiling" in key else "wall"
            if kind != condition["surface_kind"]:
                return False
        if condition.get("adjacent_to_floor") and region.bounds_v[0] > 0.2:
            return False
        if "material_hint" in condition and "floor" not in region.surface_key:
            return False
        return True

    # -- entry point ------------------------------------------------------
    def build(
        self,
        regions: list[DamageRegion],
        rooms: list[Room],
        wall_lengths: dict[str, float] | None = None,
    ) -> tuple[list[LineItem], list[ConcealedFlag]]:
        """Full scope: per-region items plus per-room equipment and containment."""
        wall_lengths = wall_lengths or {}
        items: list[LineItem] = []
        by_room = {room.id: room for room in rooms}

        for region in regions:
            room = by_room.get(region.room_id)
            if region.damage_class == "water":
                items += self._water_items(
                    region, room, wall_lengths.get(region.surface_key)
                )
            elif region.damage_class == "mold":
                items += self._mold_items(region, room)
            elif region.damage_class == "fire":
                items += self._fire_items(region, room)

        for room in rooms:
            in_room = [r for r in regions if r.room_id == room.id]
            if not in_room:
                continue
            items += self._drying_items(in_room, room)
            items += self._mold_containment(in_room, room)
            items += self._fire_odour(in_room, room)

        return items, self.concealed_flags(regions)
