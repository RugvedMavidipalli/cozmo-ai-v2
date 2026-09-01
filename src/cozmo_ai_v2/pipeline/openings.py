"""A source-neutral contract for door and window evidence.

The reconstruction has three deliberately different ways to discover an
opening: a hole in the depth occupancy grid, an RGB detector/segmenter, and a
RoomFormer prediction.  Keeping those inputs in one small contract makes it
possible to combine them without allowing a weak image-only guess to become a
metric measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable, Mapping

import numpy as np

OPENING_KINDS = ("door", "window", "pass-through")
OPENING_PROVENANCE = ("geometry", "rgb", "roomformer", "fused")
OPENING_STATES = ("measured", "unmeasured", "occluded")


def normalize_opening_kind(value: object) -> str | None:
    """Return a supported opening kind, rejecting furniture/unknown labels."""
    if value is None:
        return None
    text = str(value).strip().lower().strip(".,;:").replace("_", "-")
    text = "-".join(text.split())
    if text in {"door", "doorway", "entry", "entrance"}:
        return "door"
    if text in {"window", "windows"}:
        return "window"
    if text in {"pass-through", "passthrough", "arch", "archway"}:
        return "pass-through"
    return None


def _range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = (value.get("low", value.get("min")), value.get("high", value.get("max")))
    try:
        values = tuple(float(v) for v in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 2 or not all(isfinite(v) for v in values):
        return None
    lo, hi = sorted(values)
    return (lo, hi) if hi > lo else None


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = (value.get("x0"), value.get("y0"), value.get("x1"), value.get("y1"))
    try:
        values = tuple(float(v) for v in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(isfinite(v) for v in values):
        return None
    x0, x1 = sorted((values[0], values[2]))
    y0, y1 = sorted((values[1], values[3]))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


@dataclass
class NormalizedOpening:
    """Opening evidence shared by geometry, RGB, and RoomFormer adapters.

    ``u_range`` and ``v_range`` are metric coordinates in the associated
    wall's 2D segment frame.  They are intentionally optional: a RoomFormer
    image prediction can be useful as an unmeasured hint, but it must not be
    exported as a width/height measurement until depth and a wall association
    support it.
    """

    wall_index: int | None
    kind: str
    u_range: tuple[float, float] | None
    v_range: tuple[float, float] | None
    confidence: float
    provenance: list[str] = field(default_factory=list)
    state: str = "measured"
    uncertainty: dict[str, float | str] = field(default_factory=dict)
    wall_association_confidence: float = 0.0
    wall_distance_m: float | None = None
    source_frames: list[int] = field(default_factory=list)
    observation_count: int = 1
    depth_support: int = 0
    mask_method: str | None = None
    image_bbox: tuple[float, float, float, float] | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        kind = normalize_opening_kind(self.kind)
        if kind is None:
            raise ValueError(f"unsupported opening kind: {self.kind!r}")
        self.kind = kind
        try:
            self.wall_index = int(self.wall_index) if self.wall_index is not None else None
        except (TypeError, ValueError):
            self.wall_index = None
        self.u_range = _range(self.u_range)
        self.v_range = _range(self.v_range)
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        self.confidence = float(np.clip(confidence if isfinite(confidence) else 0.0, 0.0, 1.0))
        try:
            association_confidence = float(self.wall_association_confidence)
        except (TypeError, ValueError):
            association_confidence = 0.0
        self.wall_association_confidence = float(np.clip(association_confidence, 0.0, 1.0))
        if self.wall_distance_m is not None:
            try:
                distance = float(self.wall_distance_m)
            except (TypeError, ValueError):
                distance = float("nan")
            self.wall_distance_m = distance if isfinite(distance) else None
        self.state = str(self.state).strip().lower()
        if self.state not in OPENING_STATES:
            self.state = "unmeasured"
        if self.state == "measured" and (self.wall_index is None or self.u_range is None or self.v_range is None):
            self.state = "unmeasured"
        self.provenance = _provenance(self.provenance)
        frames: list[int] = []
        for value in self.source_frames:
            try:
                frames.append(int(value))
            except (TypeError, ValueError):
                continue
        self.source_frames = sorted(set(frames))
        self.observation_count = max(1, int(self.observation_count))
        self.depth_support = max(0, int(self.depth_support))
        self.image_bbox = _bbox(self.image_bbox)

    @property
    def measurement_state(self) -> str:
        """Compatibility spelling for consumers that prefer a descriptive name."""
        return self.state

    @property
    def source(self) -> str:
        """Primary provenance label, with ``fused`` taking precedence."""
        return "fused" if "fused" in self.provenance else self.provenance[0]

    @property
    def width(self) -> float | None:
        return None if self.u_range is None else self.u_range[1] - self.u_range[0]

    @property
    def height(self) -> float | None:
        return None if self.v_range is None else self.v_range[1] - self.v_range[0]

    @property
    def sill_height(self) -> float | None:
        return None if self.v_range is None else self.v_range[0]

    @property
    def header_height(self) -> float | None:
        return None if self.v_range is None else self.v_range[1]

    def to_dict(self) -> dict:
        """Return a JSON-friendly evidence record for diagnostics/tests."""
        return {
            "wall_index": self.wall_index,
            "kind": self.kind,
            "u_range": list(self.u_range) if self.u_range is not None else None,
            "v_range": list(self.v_range) if self.v_range is not None else None,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "provenance": list(self.provenance),
            "state": self.state,
            "measurement_state": self.state,
            "uncertainty": dict(self.uncertainty),
            "wall_association_confidence": round(self.wall_association_confidence, 4),
            "wall_distance_m": self.wall_distance_m,
            "source_frames": list(self.source_frames),
            "observation_count": self.observation_count,
            "depth_support": self.depth_support,
            "mask_method": self.mask_method,
            "image_bbox": list(self.image_bbox) if self.image_bbox is not None else None,
            "rejection_reason": self.rejection_reason,
        }


# The old occupancy module imported ``Opening`` from its own namespace.  Keep
# that spelling available while all producers now construct this contract.
Opening = NormalizedOpening


def _provenance(values: Iterable[object]) -> list[str]:
    if isinstance(values, str):
        values = [values]
    normalized: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text in OPENING_PROVENANCE and text not in normalized:
            normalized.append(text)
    return normalized or ["geometry"]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 1e-9 else 0.0


def _compatible(a: NormalizedOpening, b: NormalizedOpening, centre_tolerance: float) -> bool:
    if a.wall_index != b.wall_index or a.kind != b.kind:
        return False
    if a.u_range is None or b.u_range is None or a.v_range is None or b.v_range is None:
        return False
    u_centres = abs((a.u_range[0] + a.u_range[1]) - (b.u_range[0] + b.u_range[1])) / 2
    v_centres = abs((a.v_range[0] + a.v_range[1]) - (b.v_range[0] + b.v_range[1])) / 2
    return (u_centres <= centre_tolerance and v_centres <= centre_tolerance) or (
        _overlap(a.u_range, b.u_range) >= 0.15 and _overlap(a.v_range, b.v_range) >= 0.15
    )


def _weighted_range(
    values: list[tuple[float, float]], weights: list[float]
) -> tuple[float, float]:
    weight = np.asarray(weights, dtype=float)
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        weight = np.ones(len(values), dtype=float)
    array = np.asarray(values, dtype=float)
    return (float(np.average(array[:, 0], weights=weight)), float(np.average(array[:, 1], weights=weight)))


def _fuse_group(group: list[NormalizedOpening]) -> NormalizedOpening:
    first = group[0]
    measured = [item for item in group if item.u_range is not None and item.v_range is not None]
    weights = [max(item.confidence, 0.05) for item in measured]
    u_range = _weighted_range([item.u_range for item in measured], weights)  # type: ignore[list-item]
    v_range = _weighted_range([item.v_range for item in measured], weights)  # type: ignore[list-item]
    centres_u = np.asarray([(r[0] + r[1]) / 2 for r in (item.u_range for item in measured)], float)
    centres_v = np.asarray([(r[0] + r[1]) / 2 for r in (item.v_range for item in measured)], float)
    spread_u = float(np.std(centres_u)) if len(centres_u) > 1 else 0.0
    spread_v = float(np.std(centres_v)) if len(centres_v) > 1 else 0.0
    uncertainty: dict[str, float | str] = {}
    for item in measured:
        for key, value in item.uncertainty.items():
            if isinstance(value, (int, float)) and key not in uncertainty:
                uncertainty[key] = float(value)
    uncertainty["basis"] = "fused geometry/RGB/RoomFormer-supported observations"
    uncertainty["u_spread_m"] = spread_u
    uncertainty["v_spread_m"] = spread_v
    sources = _provenance(source for item in group for source in item.provenance)
    if len(group) > 1 or len(sources) > 1:
        if "fused" not in sources:
            sources.append("fused")
    confidence = 1.0 - float(np.prod([1.0 - item.confidence for item in group]))
    confidence = float(np.clip(confidence, max(item.confidence for item in group), 1.0))
    return NormalizedOpening(
        wall_index=first.wall_index,
        kind=first.kind,
        u_range=u_range,
        v_range=v_range,
        confidence=confidence,
        provenance=sources,
        state="measured",
        uncertainty=uncertainty,
        wall_association_confidence=max(item.wall_association_confidence for item in group),
        wall_distance_m=min(
            (item.wall_distance_m for item in group if item.wall_distance_m is not None),
            default=None,
        ),
        source_frames=[frame for item in group for frame in item.source_frames],
        observation_count=sum(item.observation_count for item in group),
        depth_support=sum(item.depth_support for item in group),
        mask_method="fused" if len({item.mask_method for item in group}) > 1 else first.mask_method,
    )


def fuse_openings(
    openings: Iterable[NormalizedOpening], centre_tolerance: float = 0.35
) -> list[NormalizedOpening]:
    """Fuse repeated observations while retaining unsupported image hints.

    Only metric, wall-associated observations enter a fusion group.  A
    RoomFormer hint without depth remains ``unmeasured`` and is kept as such
    so it is visible to a reviewer without being silently promoted to a size.
    """
    measured: list[NormalizedOpening] = []
    hints: list[NormalizedOpening] = []
    for opening in openings:
        if opening.state == "measured" and opening.wall_index is not None and opening.u_range and opening.v_range:
            measured.append(opening)
        else:
            hints.append(opening)

    groups: list[list[NormalizedOpening]] = []
    for opening in sorted(measured, key=lambda item: (item.wall_index or -1, item.kind, item.u_range or (0.0, 0.0))):
        for group in groups:
            if _compatible(group[0], opening, centre_tolerance):
                group.append(opening)
                break
        else:
            groups.append([opening])
    fused = [_fuse_group(group) for group in groups]
    # Do not duplicate an unmeasured hint which was corroborated by a metric
    # result on the same wall and in the same class.
    for hint in hints:
        if not any(
            hint.kind == item.kind
            and hint.wall_index == item.wall_index
            and hint.u_range is not None
            and item.u_range is not None
            and _overlap(hint.u_range, item.u_range) >= 0.15
            for item in fused
        ):
            fused.append(hint)
    return sorted(fused, key=lambda item: (item.wall_index is None, item.wall_index or -1, item.kind, item.u_range or (0.0, 0.0)))


__all__ = [
    "OPENING_KINDS",
    "OPENING_PROVENANCE",
    "OPENING_STATES",
    "NormalizedOpening",
    "Opening",
    "fuse_openings",
    "normalize_opening_kind",
]
