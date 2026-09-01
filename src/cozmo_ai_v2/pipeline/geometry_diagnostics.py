"""Structured, JSON-safe diagnostics for the deterministic geometry stages.

The geometry pipeline deliberately keeps this module separate from the wall
solver.  It records what the existing stages did, rather than making any
topology decisions of its own.  That keeps diagnostics useful when a solver
implementation is swapped in by another workstream.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

import numpy as np


DIAGNOSTICS_VERSION = 1


def wall_id(wall: Any) -> str | None:
    """Return the best available stable display identifier for a wall."""
    if wall is None:
        return None
    existing = getattr(wall, "_geometry_diagnostic_id", None)
    if existing:
        return str(existing)
    name = getattr(wall, "name", None)
    if name:
        identifier = str(name)
        try:
            setattr(wall, "_geometry_diagnostic_id", identifier)
        except Exception:
            pass
        return identifier
    # ``WallSegment.index`` is deliberately re-assigned after each geometry
    # stage.  Keep a deterministic source id on the object instead of using
    # that transient list position in lifecycle records.  Dynamic attributes
    # preserve the id through the existing deepcopy used by merge_collinear.
    pieces: list[str] = []
    for name in ("normal", "offset", "start", "end"):
        value = getattr(wall, name, None)
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            array = np.asarray([], dtype=float)
        pieces.append(",".join(
            "nan" if not np.isfinite(item) else f"{float(item):.8f}"
            for item in array
        ))
    digest = sha1("|".join(pieces).encode("ascii", errors="replace")).hexdigest()[:12]
    identifier = f"wall_{digest}"
    try:
        setattr(wall, "_geometry_diagnostic_id", identifier)
    except Exception:
        # A solver-owned immutable wall-like object can still be diagnosed;
        # it just receives the deterministic id for this call.
        pass
    return identifier


def _point(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (2,) or not np.isfinite(array).all():
        return None
    return [round(float(item), 8) for item in array]


def _vector(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not len(array) or not np.isfinite(array).all():
        return None
    return [round(float(item), 8) for item in array]


def _wall_snapshot(wall: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "wall_id": wall_id(wall),
        "wall_index": int(getattr(wall, "index", -1)),
    }
    if snapshot["wall_id"] is not None:
        snapshot["endpoint_ids"] = [
            f"{snapshot['wall_id']}:start",
            f"{snapshot['wall_id']}:end",
        ]
    for name in ("start", "end", "normal"):
        value = _point(getattr(wall, name, None))
        if value is not None:
            snapshot[name] = value
    length = getattr(wall, "length", None)
    try:
        finite_length = length is not None and np.isfinite(length)
    except (TypeError, ValueError):
        finite_length = False
    if finite_length:
        snapshot["length_m"] = round(float(length), 8)
    tags = getattr(wall, "tags", None)
    if tags:
        snapshot["tags"] = sorted({str(tag) for tag in tags})
    return snapshot


@dataclass
class GeometryDiagnostics:
    """Mutable collector for additive pipeline diagnostics.

    All fields intentionally use plain dictionaries/lists so the final value
    can be embedded in ``result.json`` without introducing a public runtime
    dependency on this collector class.
    """

    stage_counts: dict[str, int] = field(
        default_factory=lambda: {
            "raw": 0,
            "merged": 0,
            "occlusion": 0,
            "crossing": 0,
            "quarantine": 0,
            "post_refinement_internal": 0,
            "exported": 0,
            # Compatibility alias for diagnostics produced by the first
            # Phase 1 implementation.  New callers should use the explicit
            # post_refinement_internal/exported names above.
            "final": 0,
        }
    )
    wall_records: list[dict[str, Any]] = field(default_factory=list)
    drops_by_reason: Counter[str] = field(default_factory=Counter)
    quarantines_by_reason: Counter[str] = field(default_factory=Counter)
    trims_by_reason: Counter[str] = field(default_factory=Counter)
    endpoint_gaps: dict[str, Any] = field(default_factory=dict)
    polygonization: dict[str, Any] = field(
        default_factory=lambda: {
            "candidate_face_count": 0,
            "accepted_face_count": 0,
            "rejected_faces_by_reason": {},
            "geometry_types": {},
            "faces": [],
        }
    )
    grid: dict[str, Any] = field(default_factory=dict)
    room_segmentation: dict[str, Any] = field(
        default_factory=lambda: {
            "method": "wall_graph_polygonize",
            "fallback_used": False,
            "fallback_geometry_types": {},
            "room_count": 0,
            "zero_room_reason": None,
        }
    )
    zero_room_reasons: list[str] = field(default_factory=list)
    _quarantine_keys: set[tuple[Any, ...]] = field(default_factory=set, repr=False)
    _pending_polygon_indices: list[int] = field(default_factory=list, repr=False)

    def set_wall_stage(self, stage: str, walls: list[Any]) -> None:
        """Persist the number of wall objects present after one stage.

        ``final`` is retained as a backward-compatible key and means the
        post-refinement internal wall list.  The separately named
        ``exported`` stage is the public result list after the export length
        gate, so the two counts may legitimately differ.
        """
        if stage not in self.stage_counts:
            raise ValueError(f"unknown wall diagnostic stage: {stage}")
        count = int(len(walls))
        if stage == "quarantine":
            count = int(sum(bool(getattr(wall, "quarantined", False)) for wall in walls))
        self.stage_counts[stage] = count
        if stage == "post_refinement_internal":
            self.stage_counts["final"] = count
        elif stage == "final":
            self.stage_counts["post_refinement_internal"] = count

    def record_export_filter(
        self, walls: list[Any], *, min_length_m: float = 0.5
    ) -> list[Any]:
        """Record the public wall export gate and return exported walls.

        The result schema intentionally omits very short wall fragments, but
        those fragments remain useful when explaining why internal and
        exported counts differ.  This records each omission with its stable
        diagnostic wall id and the exact threshold used by ``_assemble``.
        """
        threshold = float(min_length_m)
        if not np.isfinite(threshold) or threshold < 0:
            threshold = 0.5
        exported: list[Any] = []
        for wall in walls:
            try:
                length = float(getattr(wall, "length", np.nan))
            except (TypeError, ValueError):
                length = float("nan")
            if not np.isfinite(length) or length < threshold:
                self.record_wall_event(
                    wall,
                    stage="export",
                    action="drop",
                    reason="below_export_min_length",
                    provenance="cli._assemble.wall_export_gate",
                    extra={"threshold_m": round(threshold, 6)},
                )
                continue
            exported.append(wall)
        self.set_wall_stage("exported", exported)
        return exported

    def record_wall_event(
        self,
        wall: Any | None,
        *,
        stage: str,
        action: str,
        reason: str,
        provenance: str,
        related_walls: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a per-wall lifecycle event and update reason counters."""
        record: dict[str, Any] = {
            "stage": str(stage),
            "action": str(action),
            "reason": str(reason),
            "provenance": str(provenance),
        }
        if wall is None:
            record["wall_id"] = None
            record["wall_index"] = None
        else:
            record.update(_wall_snapshot(wall))
        if related_walls:
            record["related_wall_ids"] = [wall_id(item) for item in related_walls]
            record["related_wall_indices"] = [
                int(getattr(item, "index", -1)) for item in related_walls
            ]
        if extra:
            record.update(extra)
        self.wall_records.append(record)

        action = action.lower()
        if action in {"drop", "dropped", "reject", "rejected"}:
            self.drops_by_reason[str(reason)] += 1
        elif action in {"quarantine", "quarantined"}:
            snapshot = _wall_snapshot(wall) if wall is not None else {}
            geometry = tuple(
                tuple(snapshot.get(name, []))
                for name in ("start", "end", "normal")
            )
            key = (str(reason), geometry)
            if key in self._quarantine_keys:
                self.wall_records.pop()
                return
            self._quarantine_keys.add(key)
            self.quarantines_by_reason[str(reason)] += 1
        elif action in {"trim", "trimmed", "adjust", "adjusted"}:
            self.trims_by_reason[str(reason)] += 1

    def record_drop_summary(
        self,
        reason: str,
        count: int,
        *,
        stage: str,
        provenance: str,
    ) -> None:
        """Record a count of candidate drops that never formed a wall object."""
        count = int(count)
        if count <= 0:
            return
        self.drops_by_reason[str(reason)] += count
        self.wall_records.append(
            {
                "wall_id": None,
                "wall_index": None,
                "stage": str(stage),
                "action": "drop",
                "reason": str(reason),
                "provenance": str(provenance),
                "count": count,
            }
        )

    def record_endpoint_gaps(
        self, walls: list[Any], *, node_tolerance: float = 0.08
    ) -> None:
        """Describe endpoint proximity without solving or changing the graph."""
        active = [
            wall
            for wall in walls
            if not getattr(wall, "quarantined", False)
            and np.isfinite(getattr(wall, "length", np.nan))
            and float(getattr(wall, "length", 0.0)) > 1e-6
        ]
        endpoints: list[tuple[int, str, np.ndarray]] = []
        for wall in active:
            for endpoint in ("start", "end"):
                point = np.asarray(getattr(wall, endpoint), dtype=float).reshape(-1)
                if point.shape == (2,) and np.isfinite(point).all():
                    endpoints.append((int(getattr(wall, "index", -1)), endpoint, point))

        nearest: list[float] = []
        for index, (_, _, point) in enumerate(endpoints):
            distances = [
                float(np.linalg.norm(point - other_point))
                for other_index, (_, _, other_point) in enumerate(endpoints)
                if other_index != index
                and endpoints[other_index][0] != endpoints[index][0]
            ]
            if distances:
                nearest.append(min(distances))

        if nearest:
            values = np.asarray(nearest, dtype=float)
            quantiles = {
                key: round(float(np.quantile(values, fraction)), 6)
                for key, fraction in (
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                    ("p95", 0.95),
                    ("p99", 0.99),
                    ("max", 1.00),
                )
            }
        else:
            quantiles = {
                key: None for key in ("p50", "p75", "p90", "p95", "p99", "max")
            }

        parent = list(range(len(endpoints)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first, second = find(first), find(second)
            if first != second:
                parent[second] = first

        for first, (_, _, point) in enumerate(endpoints):
            for second in range(first + 1, len(endpoints)):
                if endpoints[first][0] == endpoints[second][0]:
                    continue
                if (
                    np.linalg.norm(point - endpoints[second][2])
                    <= node_tolerance
                ):
                    union(first, second)
        components = {find(index) for index in range(len(endpoints))}

        junction_counts = Counter()
        for root in components:
            incident = {
                endpoints[index][0]
                for index in range(len(endpoints))
                if find(index) == root
            }
            if len(incident) <= 1:
                junction_counts["dangling"] += 1
            elif len(incident) == 2:
                junction_counts["corner"] += 1
            else:
                junction_counts["multi"] += 1

        self.endpoint_gaps = {
            "endpoint_count": len(endpoints),
            "gap_quantiles_m": quantiles,
            "component_count": len(components),
            "junction_counts": dict(sorted(junction_counts.items())),
            "node_tolerance_m": round(float(node_tolerance), 6),
        }

    def record_polygon_candidate(
        self,
        polygon: Any,
        *,
        accepted: bool,
        reason: str | None = None,
        source: str = "wall_graph_polygonize",
    ) -> int:
        """Record one polygonization face and its validation outcome."""
        geometry_type = str(getattr(polygon, "geom_type", type(polygon).__name__))
        types = self.polygonization.setdefault("geometry_types", {})
        types[geometry_type] = int(types.get(geometry_type, 0)) + 1
        if accepted:
            self.polygonization["accepted_face_count"] += 1
        elif reason:
            reasons = self.polygonization.setdefault("rejected_faces_by_reason", {})
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        face: dict[str, Any] = {
            "geometry_type": geometry_type,
            "accepted": bool(accepted),
            "source": str(source),
        }
        if reason:
            face["reason"] = str(reason)
        area = getattr(polygon, "area", None)
        if area is not None and np.isfinite(area):
            face["area_m2"] = round(float(area), 8)
        centroid = getattr(polygon, "centroid", None)
        if centroid is not None:
            point = _point([getattr(centroid, "x", np.nan), getattr(centroid, "y", np.nan)])
            if point is not None:
                face["centroid"] = point
        self.polygonization.setdefault("faces", []).append(face)
        face_index = len(self.polygonization["faces"]) - 1
        if not accepted and reason is None:
            self._pending_polygon_indices.append(face_index)
        return face_index

    def record_polygon_face_decision(
        self, *, accepted: bool, reason: str | None = None
    ) -> None:
        """Resolve the next valid polygon face after floor validation."""
        if not self._pending_polygon_indices:
            return
        index = self._pending_polygon_indices.pop(0)
        face = self.polygonization["faces"][index]
        if face.get("accepted") or face.get("reason"):
            return
        face["accepted"] = bool(accepted)
        if accepted:
            self.polygonization["accepted_face_count"] += 1
        elif reason:
            face["reason"] = str(reason)
            reasons = self.polygonization.setdefault("rejected_faces_by_reason", {})
            reasons[reason] = int(reasons.get(reason, 0)) + 1

    def record_fallback_geometry(self, geometry_type: str) -> None:
        """Count geometry types considered by observed-floor fallback."""
        types = self.room_segmentation.setdefault("fallback_geometry_types", {})
        types[geometry_type] = int(types.get(geometry_type, 0)) + 1

    def record_fallback_rejection(self, reason: str) -> None:
        """Record a fallback rejection in the shared face-reason table."""
        reasons = self.polygonization.setdefault("rejected_faces_by_reason", {})
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def record_plan_grid(
        self,
        grid: Any,
        frame: Any,
        *,
        occupied_mask: np.ndarray | None = None,
        free_mask: np.ndarray | None = None,
    ) -> None:
        """Persist grid geometry, transforms, and disjoint segmentation masks."""
        shape = tuple(int(item) for item in np.asarray(grid.occupied).shape)
        origin = np.asarray(grid.origin, dtype=float).reshape(-1)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            origin = np.zeros(2, dtype=float)
        resolution = float(grid.resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            resolution = 0.04
        occupied = (
            np.asarray(occupied_mask, dtype=bool)
            if occupied_mask is not None
            else np.asarray(grid.occupied) > 0
        )
        observed_free = (
            np.asarray(free_mask, dtype=bool)
            if free_mask is not None
            else np.asarray(grid.free) > 0
        )
        occupied = occupied.reshape(shape)
        observed_free = observed_free.reshape(shape) & ~occupied
        unknown = ~(occupied | observed_free)
        free_labels = np.zeros(shape, dtype=np.uint8)
        free_labels[observed_free] = 1
        from scipy import ndimage

        _, free_component_count = ndimage.label(
            free_labels, structure=np.ones((3, 3), dtype=int)
        )
        upper = origin + np.asarray(shape, dtype=float) * resolution
        self.grid = {
            "resolution_m": round(resolution, 6),
            "origin": [round(float(item), 8) for item in origin],
            "shape": list(shape),
            "bounds_plan": {
                "min": [round(float(item), 8) for item in origin],
                "max": [round(float(item), 8) for item in upper],
            },
            "transforms": {
                "world_to_plan": {
                    "right": _vector(getattr(frame, "right", [])),
                    "forward": _vector(getattr(frame, "forward", [])),
                },
                "world_height_axis": _vector(getattr(frame, "up", [])),
                "yaw_rad": round(float(frame.yaw), 8),
            },
            "occupied_cells": int(occupied.sum()),
            "free_cells": int(observed_free.sum()),
            "unknown_cells": int(unknown.sum()),
            "free_component_count": int(free_component_count),
            "occupied_evidence_cells": int((np.asarray(grid.occupied) > 0).sum()),
            "free_evidence_cells": int((np.asarray(grid.free) > 0).sum()),
        }

    def record_room_segmentation(
        self,
        *,
        room_count: int,
        fallback_used: bool,
        fallback_geometry_types: dict[str, int] | None = None,
        zero_room_reason: str | None = None,
        method: str = "wall_graph_polygonize",
    ) -> None:
        """Persist room extraction method and a human-readable zero reason."""
        self.room_segmentation = {
            "method": str(method),
            "fallback_used": bool(fallback_used),
            "fallback_geometry_types": dict(sorted((fallback_geometry_types or {}).items())),
            "room_count": int(room_count),
            "zero_room_reason": zero_room_reason,
        }
        self.zero_room_reasons = []
        if zero_room_reason:
            self.zero_room_reasons = [
                item.strip() for item in str(zero_room_reason).split(";") if item.strip()
            ]

    def to_dict(self) -> dict[str, Any]:
        """Return the additive, JSON-safe geometry diagnostics contract."""
        return {
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "wall_stages": {
                "stage_counts": dict(self.stage_counts),
                "drops_by_reason": dict(sorted(self.drops_by_reason.items())),
                "quarantines_by_reason": dict(sorted(self.quarantines_by_reason.items())),
                "trims_by_reason": dict(sorted(self.trims_by_reason.items())),
            },
            "wall_records": list(self.wall_records),
            "endpoint_gaps": dict(self.endpoint_gaps),
            "polygonization": dict(self.polygonization),
            "grid": dict(self.grid),
            "room_segmentation": dict(self.room_segmentation),
            "zero_room_reasons": list(self.zero_room_reasons),
        }
