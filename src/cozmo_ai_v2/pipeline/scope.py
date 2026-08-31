from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .damage.fusion import DamageRegion
from .rooms import Room


def _involves(region: DamageRegion, damage_class: str) -> bool:
    """True if `region` should be scoped as `damage_class`.

    Args:
        region: The fused damage region to test.
        damage_class: The class being scoped for right now.

    Returns:
        True if `region.damage_class == damage_class`, or
        `region.damage_class == "combined"` and `damage_class` is one of
        `region.combined_classes`.
    """
    if region.damage_class == damage_class:
        return True
    return region.damage_class == "combined" and bool(
        region.combined_classes and damage_class in region.combined_classes
    )


@dataclass
class LineItem:
    """One scope-of-work line item.

    Attributes:
        code: Catalogue key into `rules.yaml`'s `line_items`.
        description: Human-readable line-item description.
        action: Verb describing what's done.
        material: The material or equipment this line item concerns.
        quantity: Quantity in `unit`.
        unit: Unit of measure.
        trade: Responsible trade/skillset.
        surface_ref: Surface or room-name key this item applies to.
        room_id: Room this item belongs to, or `None`.
        rule_id: Dotted path into `rules.yaml` identifying which rule
            produced this item.
        source: Citation for the rule.
        basis: Prose explanation of how the quantity was derived.
        derived_from: Damage region ids this item was produced from.
    """

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
    basis: str
    derived_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-dict form of this line item, for JSON export.

        Returns:
            A dict with every field, `quantity` rounded to 3 decimal places.
        """
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
    """A flagged risk of damage that can't be directly seen, inferred from
    visible evidence.

    Attributes:
        rule_id: Id of the `concealed` rule in `rules.yaml` that fired.
        surface_ref: Surface of the triggering region.
        room_id: Room of the triggering region, or `None`.
        inferred: Description of the concealed condition being flagged
            (the rule's `infer` text).
        probability: The rule's stated likelihood this concealed condition
            is actually present, in `[0, 1]`.
        rationale: The rule's stated reasoning, for a human reviewer.
        source: Citation for the rule.
        triggered_by: Id of the `DamageRegion` that matched the rule's
            condition.
    """

    rule_id: str
    surface_ref: str
    room_id: int | None
    inferred: str
    probability: float
    rationale: str
    source: str
    triggered_by: str

    def to_dict(self) -> dict:
        """Plain-dict form of this flag, for JSON export.

        Returns:
            A dict with every field.
        """
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
    """Turns fused damage regions into a concrete list of repair line items, following the rules in `rules.yaml`.

    None of the pricing or repair logic is hard-coded in this class --
    it just reads whichever action `rules.yaml` specifies for the damage
    it's given and applies it. That means the rules can be updated
    without touching this code.
    """

    def __init__(self, rules_path: str | Path = "rules.yaml"):
        """Load and parse the rules file.

        Args:
            rules_path: Path to the YAML rules file.
        """
        self.rules = yaml.safe_load(Path(rules_path).read_text())
        self.catalogue = self.rules["line_items"]

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
        """Build a `LineItem`, filling `description`/`unit`/`trade` from the
        catalogue entry for `code`.

        Args:
            code: Catalogue key into `self.catalogue`.
            action: Verb describing the work performed.
            material: Material or equipment involved.
            quantity: Quantity in the catalogue entry's unit.
            surface_ref: Surface or room-name key the item applies to.
            room_id: Room the item belongs to, or `None`.
            rule_id: Dotted path identifying the rule that produced this.
            source: Citation for the rule.
            basis: Prose explanation of how `quantity` was derived.
            derived_from: Damage region ids the item was produced from.

        Returns:
            A fully-populated `LineItem`.
        """
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

    def _flood_cut_height(self, waterline: float) -> tuple[float, str]:
        """Cut height above the floor, and the reasoning behind it.

        Args:
            waterline: Height of the visible waterline above the floor,
                metres.

        Returns:
            `(height, basis)`: the drywall cut height in metres (waterline
            plus a safety margin, raised to a configured minimum, and
            optionally snapped up to the nearest standard sheet line), and
            a prose trail of which adjustments applied.
        """
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
        """Builds the water-damage repair items for one region, using different logic for each kind of surface.

        A wet floor, a wet wall, and a wet ceiling don't get repaired the
        same way, so this first checks what kind of surface the region
        sits on. A floor gets its carpet and pad addressed directly, since
        water pools there and soaks straight down. A wall instead gets a
        "flood cut": rather than replacing the whole wall, only the
        drywall up to some height above where the water visibly reached is
        removed and replaced, since damage from wicking rarely climbs much
        higher than that. A ceiling has no equivalent waterline to measure
        from -- water falling from above just soaks whatever area it
        touches -- so a ceiling is scoped by its damaged area instead of a
        cut height. Whichever of those applies, an antimicrobial treatment
        is also added on top whenever the water's contamination category
        calls for it, regardless of which surface it's on.

        Args:
            region: A water (or water-half-of-combined) damage region.
            room: The region's room, if resolved.
            wall_length: Full length of the wall this region sits on,
                metres; `None` for floor/ceiling regions or if unknown.

        Returns:
            Zero or more `LineItem`s: carpet/pad handling for a floor;
            drywall removal, replacement, insulation removal, and baseboard
            handling for a wall or ceiling flood cut; antimicrobial
            treatment for a qualifying category, on any surface kind.
        """
        water = self.rules["water"]
        category = region.water_category or 1
        actions = water["category_actions"].get(category, water["category_actions"][1])
        items: list[LineItem] = []
        room_id = region.room_id

        is_floor = region.surface_key.endswith("floor") or region.surface_key == "floor"
        is_ceiling = region.surface_key.endswith("ceiling") or region.surface_key == "ceiling"

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

        if actions["drywall"] == "flood_cut" and (wall_length or is_ceiling):
            if is_ceiling:
                area = region.area
                basis = (
                    f"ceiling water damage scoped by affected area (a ceiling has "
                    f"no waterline to cut above): {area:.2f} m2"
                )
                rule_id = "water.category_actions.ceiling"
            else:
                waterline = region.bounds_v[1]
                height, basis = self._flood_cut_height(waterline)
                run = min(region.width_extent, wall_length)
                area = height * run
                basis += f"; cut {height:.2f} m x {run:.2f} m affected run"
                rule_id = "water.flood_cut"

            items.append(
                self._item(
                    "drywall_remove", "remove", "drywall", area,
                    region.surface_key, room_id, rule_id,
                    water["flood_cut"]["source"], basis,
                    [region.id],
                )
            )
            items.append(
                self._item(
                    "drywall_replace", "replace", "1/2\" drywall", area,
                    region.surface_key, room_id, rule_id,
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

            if not is_ceiling:
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
        """Equipment counts follow affected area and class, for the whole room.

        Args:
            regions: All damage regions in one room (mixed classes;
                filtered to water-involving ones internally).
            room: The room these regions belong to.

        Returns:
            Empty list if the room has no water damage. Otherwise three
            items: air movers, a dehumidifier, and daily monitoring visits.
        """
        water_regions = [r for r in regions if _involves(r, "water")]
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

    def _mold_items(self, region: DamageRegion, room: Room | None) -> list[LineItem]:
        """Build mold-remediation line items for one region.

        Args:
            region: A mold (or mold-half-of-combined) damage region.
            room: The region's room, if resolved; not used here.

        Returns:
            Zero to three items: material removal and antimicrobial
            treatment (both skipped when the condition's action doesn't call
            for removal), and always a HEPA vacuum pass over the grown
            (margin-expanded) area.
        """
        mold = self.rules["mold"]
        condition = region.mold_condition or 3
        action = mold["condition_actions"].get(condition, "remove_and_treat")
        items: list[LineItem] = []
        room_id = region.room_id

        margin = mold["removal_margin"]
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
        """Builds the containment barrier, air scrubber, and PPE items for a room with mold, sized to the whole room rather than to the mold patch itself.

        Most repair quantities scale with how much damage there is -- more
        damaged area means more material to remove or treat. Containment
        doesn't work that way. Remediating mold safely means sealing off
        the entire room with plastic sheeting so spores can't spread while
        the work happens, and that seal has to cover the room's real walls
        and ceiling whether the visible mold patch is tiny or covers most
        of a wall. So the size of the plastic barrier below is calculated
        from the room's own perimeter, floor area, and height -- not from
        how much of the room the mold actually covers. The total mold area
        still matters for one thing: it decides how serious a containment
        setup is called for (the `size` category below), which in turn
        picks the containment type and PPE level.

        Args:
            regions: All damage regions in one room (mixed classes; filtered
                to mold-involving ones internally).
            room: The room these regions belong to; supplies the
                perimeter/area/height the barrier and air-scrubber sizing
                are based on.

        Returns:
            Empty list if the room's total mold area is under the
            containment trigger threshold. Otherwise a containment barrier
            item; a negative-air (HEPA scrubber) item if the larger
            negative-air trigger is also met; and a PPE supply item sized to
            two technicians over three days.
        """
        mold_regions = [r for r in regions if _involves(r, "mold")]
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
        # Barrier covers the room's own perimeter walls (perimeter x height)
        # plus its ceiling (area) -- a full room-scale enclosure, independent
        # of how much of that room the visible mold actually covers.
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

    def _fire_items(self, region: DamageRegion, room: Room | None) -> list[LineItem]:
        """Build fire-damage line items for one region.

        Dispatches by subtype action: cleanable soot is cleaned (and
        optionally sealed against residual odour), while char or consumed
        material -- where the substrate itself is compromised, not just
        coated -- is removed and replaced outright rather than cleaned.

        Args:
            region: A fire (or fire-half-of-combined) damage region.
            room: The region's room, if resolved (not directly used here,
                accepted for a uniform per-class item-builder signature).

        Returns:
            One or two items: either soot cleaning plus an optional sealant
            coat, or fire-damaged material removal.
        """
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
        """Per-room deodorisation, triggered once total fire damage clears a threshold.

        Args:
            regions: All damage regions in one room (mixed classes; filtered
                to fire-involving ones internally).
            room: The room these regions belong to.

        Returns:
            Empty list if the room's total fire-damaged area is under the
            odour-treatment trigger. Otherwise a single whole-room
            deodorisation item (`quantity=1.0` -- it's a per-room treatment,
            not an area-scaled one).
        """
        fire_regions = [r for r in regions if _involves(r, "fire")]
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

    def concealed_flags(self, regions: list[DamageRegion]) -> list[ConcealedFlag]:
        """Checks every damage region against every concealed-damage rule, and flags the ones that match.

        A concealed-damage rule describes a hidden risk that's likely
        given what's visible -- for example, contaminated water damage on
        a wall often means the insulation behind it is contaminated too,
        even though nobody can see that directly. Every region is checked
        against every rule, not just the rules that match that region's
        own damage class: a rule written for water damage will simply fail
        its first check for a fire-damage region, so nothing extra is
        needed here to keep the classes from mixing. A single region can
        also end up triggering more than one rule, if it happens to
        satisfy several.

        Args:
            regions: All fused damage regions across the whole property.

        Returns:
            One `ConcealedFlag` per (rule, region) pair where the region
            satisfies the rule's `when` condition -- possibly several per
            region, or none.
        """
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
        """Checks whether one region satisfies every condition listed in a concealed-damage rule.

        A rule's `when` block in `rules.yaml` can list several conditions
        at once, like a minimum water category or being close to the
        floor. Each condition checked below is optional: if a rule simply
        doesn't mention it, this skips checking it rather than treating
        its absence as a failure. But any condition a rule does list has
        to be satisfied, or the whole match fails. This lets `rules.yaml`
        describe narrow, specific combinations -- like "Category 3 water,
        next to the floor" -- without this function needing a separate
        case written for every possible rule.

        Args:
            condition: One rule's `when` block from `rules.yaml` --
                `damage_class` is required, every other key optional.
            region: The candidate region to test.

        Returns:
            True only if EVERY condition key present in `condition` is
            satisfied by `region`.
        """
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

    def build(
        self,
        regions: list[DamageRegion],
        rooms: list[Room],
        wall_lengths: dict[str, float] | None = None,
    ) -> tuple[list[LineItem], list[ConcealedFlag]]:
        """Builds the full scope of work: line items for each region, plus per-room items like equipment and containment.

        This runs in two passes. First, every region gets its own
        class-specific items -- water, mold, or fire, via `_involves`, so
        a region with combined damage gets items from both classes.
        Second, every room that has any damage at all gets a batch of
        per-room items computed once from all of that room's regions
        together (drying equipment, mold containment, fire odour). Those
        are handled as a separate pass, rather than per region, because
        they don't scale per region -- computing them region-by-region
        would double-count things like how many air movers a room needs.

        Args:
            regions: All fused damage regions across the property.
            rooms: All rooms, used both to resolve each region's room by id
                and to iterate for the per-room aggregate pass.
            wall_lengths: Map from wall surface key to that wall's full
                length, metres -- passed through to `_water_items` for
                flood-cut sizing. Regions on a wall missing from this map
                get no flood-cut items (see `_water_items`'s
                `wall_length or is_ceiling` guard).

        Returns:
            `(items, concealed_flags)`: every generated line item, and every
            concealed-damage flag from `concealed_flags`.
        """
        wall_lengths = wall_lengths or {}
        items: list[LineItem] = []
        by_room = {room.id: room for room in rooms}

        for region in regions:
            room = by_room.get(region.room_id)
            if _involves(region, "water"):
                items += self._water_items(
                    region, room, wall_lengths.get(region.surface_key)
                )
            if _involves(region, "mold"):
                items += self._mold_items(region, room)
            if _involves(region, "fire"):
                items += self._fire_items(region, room)

        for room in rooms:
            in_room = [r for r in regions if r.room_id == room.id]
            if not in_room:
                continue
            items += self._drying_items(in_room, room)
            items += self._mold_containment(in_room, room)
            items += self._fire_odour(in_room, room)

        return items, self.concealed_flags(regions)
