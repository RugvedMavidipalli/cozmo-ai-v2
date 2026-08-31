"""Global, plane-constrained wall graph solving.

Wall fitting produces independent finite segments.  This module turns their
shared line intersections into one deterministic node set before any room
faces are polygonized.  A node is solved from all incident finished-face line
equations at once; endpoints are never snapped independently to a nearby
endpoint estimate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .planes import WallSegment

JunctionKind = Literal["corner", "t", "x"]
ConnectionType = Literal["L", "T", "X", "extension"]


@dataclass(frozen=True)
class WallGraphConnection:
    """One evidence-backed wall-line connection decision."""

    wall_ids: tuple[int, int]
    endpoint_ids: tuple[str | None, str | None]
    movement_m: float
    movement_by_wall_m: tuple[float, float]
    type: ConnectionType
    evidence_score: float
    confidence: float
    decision: str
    reason: str
    before_coordinates: np.ndarray
    after_coordinates: np.ndarray

    def __post_init__(self) -> None:
        if len(self.wall_ids) != 2 or len(self.endpoint_ids) != 2:
            raise ValueError("wall graph connections require two wall endpoints")
        before = np.asarray(self.before_coordinates, dtype=float)
        after = np.asarray(self.after_coordinates, dtype=float)
        if before.shape != (2, 2) or after.shape != (2, 2):
            raise ValueError("connection coordinates must have shape (2, 2)")
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("connection coordinates must be finite")
        if self.type not in ("L", "T", "X", "extension"):
            raise ValueError("unsupported wall graph connection type")
        movement = float(self.movement_m)
        movement_by_wall = tuple(float(value) for value in self.movement_by_wall_m)
        if len(movement_by_wall) != 2 or not np.isfinite((movement, *movement_by_wall)).all():
            raise ValueError("connection movement must be finite")
        object.__setattr__(self, "before_coordinates", before)
        object.__setattr__(self, "after_coordinates", after)
        object.__setattr__(self, "movement_m", max(movement, 0.0))
        object.__setattr__(self, "movement_by_wall_m", movement_by_wall)
        object.__setattr__(
            self, "evidence_score", float(np.clip(self.evidence_score, 0.0, 1.0))
        )
        object.__setattr__(
            self, "confidence", float(np.clip(self.confidence, 0.0, 1.0))
        )

    def to_metadata(self) -> dict:
        """Return a stable JSON-ready connection decision."""
        return {
            "wall_ids": list(self.wall_ids),
            "endpoint_ids": list(self.endpoint_ids),
            "movement_m": round(self.movement_m, 6),
            "movement_by_wall_m": [
                round(value, 6) for value in self.movement_by_wall_m
            ],
            "type": self.type,
            "evidence_score": round(self.evidence_score, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "reason": self.reason,
            "before_coordinates": self.before_coordinates.tolist(),
            "after_coordinates": self.after_coordinates.tolist(),
        }


@dataclass(frozen=True)
class WallGraphNode:
    """One shared wall-line intersection in the solved graph."""

    id: int
    coordinate: np.ndarray
    kind: JunctionKind
    incident_walls: tuple[int, ...]
    confidence: float
    provenance: str = "wall-plane intersection"

    def __post_init__(self) -> None:
        coordinate = np.asarray(self.coordinate, dtype=float)
        if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
            raise ValueError("wall graph node coordinate must be finite (2,)")
        if self.kind not in ("corner", "t", "x"):
            raise ValueError("unsupported wall graph junction kind")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(
            self, "confidence", float(np.clip(self.confidence, 0.0, 1.0))
        )


@dataclass(frozen=True)
class WallGraphDiagnostics:
    """Before/after counts and endpoint-gap metrics for graph attribution."""

    before_wall_count: int
    before_endpoint_count: int
    before_endpoint_components: int
    before_endpoint_incidence_count: int
    before_nearest_endpoint_gap_m: dict[str, float | None]
    after_wall_count: int
    after_endpoint_count: int
    after_endpoint_components: int
    after_endpoint_incidence_count: int
    after_nearest_endpoint_gap_m: dict[str, float | None]
    proposed_endpoint_extensions: int
    accepted_endpoint_extensions: int
    quarantined_wall_count: int
    rejected_crossing_count: int
    node_tolerance_m: float
    max_endpoint_extension_m: float
    proposed_connections: tuple[WallGraphConnection, ...] = ()
    accepted_connections: tuple[WallGraphConnection, ...] = ()
    rejected_connections: tuple[WallGraphConnection, ...] = ()

    def to_solver_metadata(self) -> dict:
        """Return graph-only metadata for ``reconstruction.vectorization``.

        General endpoint-gap/count summaries remain available as typed
        attributes for the result diagnostics assembler, but are not emitted
        here: the canonical result owns those under ``diagnostics.geometry``.
        """
        return {
            "proposed_connections": [
                connection.to_metadata() for connection in self.proposed_connections
            ],
            "accepted_connections": [
                connection.to_metadata() for connection in self.accepted_connections
            ],
            "rejected_connections": [
                connection.to_metadata() for connection in self.rejected_connections
            ],
            "optimization": {
                "proposed_endpoint_extensions": self.proposed_endpoint_extensions,
                "accepted_endpoint_extensions": self.accepted_endpoint_extensions,
                "rejected_crossing_count": self.rejected_crossing_count,
                "max_endpoint_extension_m": self.max_endpoint_extension_m,
            },
        }

    # Backwards-compatible spelling for callers that used the original
    # diagnostics object before the result namespace was finalized.
    def to_metadata(self) -> dict:
        """Return graph-only JSON metadata."""
        return self.to_solver_metadata()

    def to_geometry_metadata(self) -> dict:
        """Return the general endpoint summary for ``diagnostics.geometry``."""
        return {
            "before": {
                "wall_count": self.before_wall_count,
                "endpoint_count": self.before_endpoint_count,
                "endpoint_components": self.before_endpoint_components,
                "endpoint_incidence_count": self.before_endpoint_incidence_count,
                "nearest_endpoint_gap_m": self.before_nearest_endpoint_gap_m,
            },
            "after": {
                "wall_count": self.after_wall_count,
                "endpoint_count": self.after_endpoint_count,
                "endpoint_components": self.after_endpoint_components,
                "endpoint_incidence_count": self.after_endpoint_incidence_count,
                "nearest_endpoint_gap_m": self.after_nearest_endpoint_gap_m,
            },
            "quarantined_wall_count": self.quarantined_wall_count,
        }


@dataclass(frozen=True)
class WallGraph:
    """Result of the global wall-node solve.

    ``candidates`` includes weak/quarantined segments for diagnostics;
    ``walls`` contains only minimum-length, topology-eligible segments used
    by polygonization.  Wall indices are preserved as identities, so existing
    sampled-point and surface-grid associations remain valid.
    """

    walls: tuple[WallSegment, ...]
    candidates: tuple[WallSegment, ...]
    nodes: tuple[WallGraphNode, ...]
    rejected_crossings: tuple[tuple[int, int], ...]
    snapped_endpoint_count: int
    diagnostics: WallGraphDiagnostics | None = None

    @property
    def junction_count(self) -> int:
        """Number of explicit corner/T/X junctions in the solved graph."""
        return len(self.nodes)


@dataclass(frozen=True)
class _Intersection:
    first: int
    second: int
    point: np.ndarray
    along_first: float
    along_second: float
    kind: JunctionKind


def solve_wall_graph(
    walls: list[WallSegment],
    *,
    node_tolerance: float = 0.12,
    min_length: float = 0.4,
    min_confidence: float = 0.25,
    allow_x_junctions: bool = True,
    max_endpoint_extension: float = 0.55,
) -> WallGraph:
    """Solve shared wall nodes and return validated topology candidates.

    All pairwise intersections of active, same-convention wall lines are
    collected first and clustered globally.  Each cluster is solved from all
    incident line equations, which gives a single plane-derived coordinate to
    every L, T, or X junction.  Endpoint changes are then applied in one
    pass.  A true interior crossing is represented as an X node when enabled;
    otherwise the weaker wall is quarantined rather than silently allowing an
    unintended crossing into room topology.

    Args:
        walls: Candidate finished-face wall segments. Inputs are deep-copied;
            caller-owned objects are not mutated.
        node_tolerance: Maximum plan-space distance for associating an
            intersection with a wall endpoint and for clustering node
            estimates.
        min_length: Minimum accepted wall length in metres. Short candidates
            remain in ``candidates`` tagged and quarantined.
        min_confidence: Minimum confidence for topology use. Weak candidates
            remain available for diagnostics but are not allowed to move a
            stronger wall or create a node.
        allow_x_junctions: Whether interior/interior intersections become
            explicit X nodes. If false, the lower-confidence participant is
            quarantined as an unintended crossing.
        max_endpoint_extension: Maximum evidence-supported extension, in
            metres, of any one wall endpoint. This is intentionally separate
            from ``node_tolerance``: the latter only clusters nearby line
            intersections and never enables blanket snapping. The effective
            per-wall limit is reduced for lower-confidence fits.

    Returns:
        A :class:`WallGraph` with deterministic nodes, accepted walls,
        diagnostic candidates, and rejected crossing pairs.
    """
    if not np.isfinite(node_tolerance) or node_tolerance <= 0:
        raise ValueError("node_tolerance must be finite and positive")
    if not np.isfinite(min_length) or min_length <= 0:
        raise ValueError("min_length must be finite and positive")
    if not np.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not np.isfinite(max_endpoint_extension) or max_endpoint_extension <= 0:
        raise ValueError("max_endpoint_extension must be finite and positive")

    candidates = [deepcopy(wall) for wall in walls]
    before_walls = [
        wall
        for wall in candidates
        if not wall.quarantined and _valid_wall_geometry(wall)
    ]
    before_metrics = _endpoint_metrics(before_walls, node_tolerance)
    conventions = {
        wall.coordinate_convention
        for wall in candidates
        if not wall.quarantined
    }
    if len(conventions) > 1:
        raise ValueError(
            "wall graph cannot mix finished-face and centerline conventions"
        )

    active: list[WallSegment] = []
    for wall in candidates:
        if wall.quarantined:
            continue
        if wall.confidence < min_confidence:
            wall.tags = sorted(set(wall.tags) | {"low-confidence"})
            wall.quarantined = True
            wall.snap_status = "rejected-low-confidence"
            continue
        if wall.length < min_length:
            wall.tags = sorted(set(wall.tags) | {"too-short"})
            wall.quarantined = True
            wall.snap_status = "rejected-too-short"
            continue
        active.append(wall)

    active.sort(key=_wall_sort_key)
    intersections: list[_Intersection] = []
    rejected_crossings: list[tuple[int, int]] = []
    connection_records: list[WallGraphConnection] = []
    for first_index, first in enumerate(active):
        for second in active[first_index + 1 :]:
            hit = _line_intersection(first, second)
            if hit is None:
                continue
            point, along_first, along_second = hit
            first_limit = _extension_limit(first, max_endpoint_extension)
            second_limit = _extension_limit(second, max_endpoint_extension)
            first_inside = -first_limit <= along_first <= first.length + first_limit
            second_inside = -second_limit <= along_second <= second.length + second_limit
            if not (first_inside and second_inside):
                connection_records.append(
                    _make_connection_record(
                        first,
                        second,
                        point,
                        along_first,
                        along_second,
                        None,
                        node_tolerance=node_tolerance,
                        max_endpoint_extension=max_endpoint_extension,
                        decision="rejected",
                        reason="outside bounded endpoint extension",
                    )
                )
                continue
            first_endpoint = _near_endpoint(along_first, first.length, first_limit)
            second_endpoint = _near_endpoint(along_second, second.length, second_limit)
            if not first_endpoint and not second_endpoint:
                if allow_x_junctions:
                    kind: JunctionKind = "x"
                else:
                    weaker = first if _wall_sort_key(first) > _wall_sort_key(second) else second
                    weaker.tags = sorted(set(weaker.tags) | {"unintended-crossing"})
                    weaker.quarantined = True
                    weaker.snap_status = "rejected-crossing"
                    rejected_crossings.append(tuple(sorted((first.index, second.index))))
                    connection_records.append(
                        _make_connection_record(
                            first,
                            second,
                            point,
                            along_first,
                            along_second,
                            None,
                            node_tolerance=node_tolerance,
                            max_endpoint_extension=max_endpoint_extension,
                            decision="rejected",
                            reason="unintended interior crossing with X junctions disabled",
                        )
                    )
                    continue
            elif first_endpoint and second_endpoint:
                kind = "corner"
            else:
                kind = "t"
            intersections.append(
                _Intersection(
                    first=first.index,
                    second=second.index,
                    point=point,
                    along_first=float(along_first),
                    along_second=float(along_second),
                    kind=kind,
                )
            )

    active = [wall for wall in active if not wall.quarantined]
    by_index = {wall.index: wall for wall in active}
    original_endpoints = {
        index: (wall.start.copy(), wall.end.copy())
        for index, wall in by_index.items()
    }
    intersections = [
        hit
        for hit in intersections
        if hit.first in by_index and hit.second in by_index
    ]
    clusters = _cluster_intersections(intersections, node_tolerance)
    nodes: list[WallGraphNode] = []
    endpoint_assignments: dict[tuple[int, str], tuple[int, float]] = {}
    cluster_coordinates: dict[int, np.ndarray] = {}
    for node_id, cluster in enumerate(clusters):
        incident = sorted({index for hit in cluster for index in (hit.first, hit.second)})
        coordinate = _solve_node_coordinate(cluster, by_index)
        cluster_coordinates[node_id] = coordinate.copy()
        kind = _node_kind(cluster)
        confidence = min(by_index[index].confidence for index in incident)
        nodes.append(
            WallGraphNode(
                id=node_id,
                coordinate=coordinate,
                kind=kind,
                incident_walls=tuple(incident),
                confidence=confidence,
            )
        )
        for hit in cluster:
            for index, along in (
                (hit.first, hit.along_first),
                (hit.second, hit.along_second),
            ):
                wall = by_index[index]
                endpoint = _endpoint_name(
                    along,
                    wall.length,
                    _extension_limit(wall, max_endpoint_extension),
                )
                if endpoint is None:
                    continue
                key = (index, endpoint)
                distance = abs(along if endpoint == "start" else wall.length - along)
                prior = endpoint_assignments.get(key)
                if prior is None or distance < prior[1]:
                    endpoint_assignments[key] = (node_id, distance)

    node_by_id = {node.id: node for node in nodes}
    snapped_endpoint_count = 0
    endpoint_decisions: dict[tuple[int, str], bool] = {}
    for (index, endpoint), (node_id, _) in endpoint_assignments.items():
        wall = by_index[index]
        coordinate = node_by_id[node_id].coordinate.copy()
        current = wall.start if endpoint == "start" else wall.end
        extension_limit = _extension_limit(wall, max_endpoint_extension)
        if np.linalg.norm(current - coordinate) > extension_limit + 1e-9:
            # A noisy multi-line least-squares node can move farther than the
            # pairwise evidence that proposed it. Do not turn that into an
            # unconstrained bridge through unknown space.
            wall.tags = sorted(set(wall.tags) | {"extension-out-of-bounds"})
            wall.snap_status = "rejected-extension"
            endpoint_decisions[(index, endpoint)] = False
            continue
        endpoint_decisions[(index, endpoint)] = True
        if np.linalg.norm(current - coordinate) > 1e-9:
            snapped_endpoint_count += 1
            if endpoint == "start":
                wall.start = coordinate
            else:
                wall.end = coordinate
            wall.tags = sorted(set(wall.tags) | {"global-junction"})
            wall.tags = sorted(set(wall.tags) | {"evidence-supported-extension"})
            wall.snap_status = "graph-extended"
            wall.provenance = _append_provenance(
                wall.provenance, "global plane-intersection endpoint closure"
            )
            wall.snap_residual = max(
                wall.snap_residual, float(np.linalg.norm(current - coordinate))
            )

    accepted_connections: list[WallGraphConnection] = []
    rejected_connection_records: list[WallGraphConnection] = list(connection_records)
    cluster_rejected_connections: list[WallGraphConnection] = []
    for node_id, cluster in enumerate(clusters):
        coordinate = cluster_coordinates[node_id]
        for hit in cluster:
            first = by_index[hit.first]
            second = by_index[hit.second]
            endpoint_status = []
            for index, along in (
                (hit.first, hit.along_first),
                (hit.second, hit.along_second),
            ):
                endpoint = _endpoint_name(
                    along,
                    by_index[index].length,
                    _extension_limit(by_index[index], max_endpoint_extension),
                )
                if endpoint is not None:
                    assigned = endpoint_assignments.get((index, endpoint))
                    endpoint_status.append(
                        endpoint_decisions.get((index, endpoint), False)
                        and assigned is not None
                        and assigned[0] == node_id
                    )
            accepted = all(endpoint_status) if endpoint_status else True
            record = _make_connection_record(
                first,
                second,
                hit.point,
                hit.along_first,
                hit.along_second,
                coordinate,
                node_tolerance=node_tolerance,
                max_endpoint_extension=max_endpoint_extension,
                decision="accepted" if accepted else "rejected",
                reason=(
                    "evidence-supported endpoint/intersection"
                    if accepted
                    else "global node adjustment exceeded bounded endpoint evidence"
                ),
                before_coordinates=(
                    original_endpoints[hit.first], original_endpoints[hit.second]
                ),
            )
            if accepted:
                accepted_connections.append(record)
            else:
                cluster_rejected_connections.append(record)
                rejected_connection_records.append(record)

    final_walls = []
    for wall in active:
        if wall.length < min_length:
            wall.tags = sorted(set(wall.tags) | {"too-short"})
            wall.quarantined = True
            wall.snap_status = "rejected-too-short"
        else:
            final_walls.append(wall)
    final_indices = {wall.index for wall in final_walls}
    nodes = [
        WallGraphNode(
            id=node.id,
            coordinate=node.coordinate,
            kind=node.kind,
            incident_walls=tuple(index for index in node.incident_walls if index in final_indices),
            confidence=node.confidence,
            provenance=node.provenance,
        )
        for node in nodes
        if any(index in final_indices for index in node.incident_walls)
    ]
    nodes.sort(key=lambda node: (node.coordinate[0], node.coordinate[1], node.kind, node.id))
    # IDs are metadata, not wall identities. Reassign them after deterministic
    # coordinate sorting so output ordering is stable across input order.
    nodes = [
        WallGraphNode(
            id=index,
            coordinate=node.coordinate,
            kind=node.kind,
            incident_walls=node.incident_walls,
            confidence=node.confidence,
            provenance=node.provenance,
        )
        for index, node in enumerate(nodes)
    ]
    candidates.sort(key=_wall_sort_key)
    final_walls.sort(key=_wall_sort_key)
    after_metrics = _endpoint_metrics(final_walls, node_tolerance)
    proposed_extensions = sum(
        1
        for cluster in clusters
        for hit in cluster
        for index, along in (
            (hit.first, hit.along_first),
            (hit.second, hit.along_second),
        )
        if _endpoint_name(
            along,
            by_index[index].length if index in by_index else 0.0,
            _extension_limit(by_index[index], max_endpoint_extension)
            if index in by_index
            else 0.0,
        )
        is not None
    )
    diagnostics = WallGraphDiagnostics(
        before_wall_count=len(candidates),
        before_endpoint_count=before_metrics["endpoint_count"],
        before_endpoint_components=before_metrics["endpoint_components"],
        before_endpoint_incidence_count=before_metrics["endpoint_incidence_count"],
        before_nearest_endpoint_gap_m=before_metrics["nearest_endpoint_gap_m"],
        after_wall_count=len(final_walls),
        after_endpoint_count=after_metrics["endpoint_count"],
        after_endpoint_components=after_metrics["endpoint_components"],
        after_endpoint_incidence_count=after_metrics["endpoint_incidence_count"],
        after_nearest_endpoint_gap_m=after_metrics["nearest_endpoint_gap_m"],
        proposed_endpoint_extensions=proposed_extensions,
        accepted_endpoint_extensions=snapped_endpoint_count,
        quarantined_wall_count=sum(wall.quarantined for wall in candidates),
        rejected_crossing_count=len(set(rejected_crossings)),
        node_tolerance_m=float(node_tolerance),
        max_endpoint_extension_m=float(max_endpoint_extension),
        proposed_connections=tuple(
            sorted(
                [*connection_records, *accepted_connections, *cluster_rejected_connections],
                key=_connection_sort_key,
            )
        ),
        accepted_connections=tuple(sorted(accepted_connections, key=_connection_sort_key)),
        rejected_connections=tuple(
            sorted(rejected_connection_records, key=_connection_sort_key)
        ),
    )
    return WallGraph(
        walls=tuple(final_walls),
        candidates=tuple(candidates),
        nodes=tuple(nodes),
        rejected_crossings=tuple(sorted(set(rejected_crossings))),
        snapped_endpoint_count=snapped_endpoint_count,
        diagnostics=diagnostics,
    )


def _wall_sort_key(wall: WallSegment) -> tuple:
    return (
        -int(wall.inlier_count),
        -round(wall.length, 8),
        round(wall.residual_rms, 8),
        int(wall.index),
        tuple(np.round(wall.start, 8)),
        tuple(np.round(wall.end, 8)),
    )


def _line_intersection(
    first: WallSegment, second: WallSegment
) -> tuple[np.ndarray, float, float] | None:
    first_direction = first.direction
    second_direction = second.direction
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(denominator) < 1e-8:
        return None
    delta = second.start - first.start
    along_first = (
        delta[0] * second_direction[1] - delta[1] * second_direction[0]
    ) / denominator
    point = first.start + first_direction * along_first
    along_second = float((point - second.start) @ second_direction)
    return point, float(along_first), along_second


def _near_endpoint(along: float, length: float, tolerance: float) -> bool:
    return min(abs(along), abs(length - along)) <= tolerance


def _endpoint_name(along: float, length: float, tolerance: float) -> str | None:
    if abs(along) <= tolerance and abs(length - along) <= tolerance:
        return "start" if along <= length / 2 else "end"
    if abs(along) <= tolerance:
        return "start"
    if abs(length - along) <= tolerance:
        return "end"
    return None


def _cluster_intersections(
    intersections: list[_Intersection], tolerance: float
) -> list[list[_Intersection]]:
    if not intersections:
        return []
    parent = list(range(len(intersections)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    ordered = sorted(
        range(len(intersections)),
        key=lambda index: (
            tuple(np.round(intersections[index].point, 8)),
            intersections[index].first,
            intersections[index].second,
        ),
    )
    for position, first_index in enumerate(ordered):
        for second_index in ordered[position + 1 :]:
            if np.linalg.norm(
                intersections[first_index].point - intersections[second_index].point
            ) <= tolerance:
                union(first_index, second_index)
    groups: dict[int, list[_Intersection]] = {}
    for index in ordered:
        groups.setdefault(find(index), []).append(intersections[index])
    return sorted(
        groups.values(),
        key=lambda group: (
            round(float(np.mean([hit.point[0] for hit in group])), 8),
            round(float(np.mean([hit.point[1] for hit in group])), 8),
        ),
    )


def _solve_node_coordinate(
    cluster: list[_Intersection], walls: dict[int, WallSegment]
) -> np.ndarray:
    incident = sorted({index for hit in cluster for index in (hit.first, hit.second)})
    matrix = np.vstack([walls[index].normal for index in incident])
    values = np.asarray([walls[index].offset for index in incident], dtype=float)
    weights = np.asarray(
        [max(walls[index].confidence, 0.05) for index in incident], dtype=float
    )
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_values = values * np.sqrt(weights)
    if np.linalg.matrix_rank(weighted_matrix) >= 2:
        coordinate, _, _, _ = np.linalg.lstsq(
            weighted_matrix, weighted_values, rcond=None
        )
        if np.isfinite(coordinate).all():
            return coordinate
    return np.mean(np.vstack([hit.point for hit in cluster]), axis=0)


def _node_kind(cluster: list[_Intersection]) -> JunctionKind:
    if any(hit.kind == "x" for hit in cluster):
        return "x"
    if any(hit.kind == "t" for hit in cluster):
        return "t"
    return "corner"


def _make_connection_record(
    first: WallSegment,
    second: WallSegment,
    point: np.ndarray,
    along_first: float,
    along_second: float,
    coordinate: np.ndarray | None,
    *,
    node_tolerance: float,
    max_endpoint_extension: float,
    decision: str,
    reason: str,
    before_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
) -> WallGraphConnection:
    """Build an auditable connection record from one pairwise line hit."""
    first_endpoint = _endpoint_name(
        along_first,
        first.length,
        _extension_limit(first, max_endpoint_extension),
    )
    second_endpoint = _endpoint_name(
        along_second,
        second.length,
        _extension_limit(second, max_endpoint_extension),
    )
    if reason == "outside bounded endpoint extension":
        first_endpoint = first_endpoint or _nearest_endpoint_name(
            along_first, first.length
        )
        second_endpoint = second_endpoint or _nearest_endpoint_name(
            along_second, second.length
        )
    if first_endpoint is None and second_endpoint is None:
        connection_type: ConnectionType = "X"
    elif first_endpoint is not None and second_endpoint is not None:
        connection_type = "L"
    else:
        connection_type = "T"

    if before_coordinates is None:
        first_start, first_end = first.start, first.end
        second_start, second_end = second.start, second.end
    else:
        first_start, first_end = before_coordinates[0]
        second_start, second_end = before_coordinates[1]
    before_first = (
        first_start if first_endpoint == "start"
        else first_end if first_endpoint == "end"
        else point
    )
    before_second = (
        second_start if second_endpoint == "start"
        else second_end if second_endpoint == "end"
        else point
    )
    after_point = np.asarray(point if coordinate is None else coordinate, dtype=float)
    before = np.vstack([before_first, before_second]).astype(float)
    after = np.vstack([after_point, after_point]).astype(float)
    movements = tuple(
        float(np.linalg.norm(after[index] - before[index])) for index in range(2)
    )
    if (
        (first_endpoint is not None or second_endpoint is not None)
        and max(movements) > node_tolerance
    ):
        connection_type = "extension"
    confidence = float(min(first.confidence, second.confidence))
    fit_evidence = float(min(first.fit_quality, second.fit_quality))
    movement_evidence = float(
        np.exp(-max(movements) / max(max_endpoint_extension, 1e-9))
    )
    evidence_score = float(np.clip(fit_evidence * movement_evidence, 0.0, 1.0))
    return WallGraphConnection(
        wall_ids=(first.index, second.index),
        endpoint_ids=(
            _stable_endpoint_id(first.index, first_endpoint),
            _stable_endpoint_id(second.index, second_endpoint),
        ),
        movement_m=max(movements),
        movement_by_wall_m=movements,
        type=connection_type,
        evidence_score=evidence_score,
        confidence=confidence,
        decision=decision,
        reason=reason,
        before_coordinates=before,
        after_coordinates=after,
    )


def _stable_endpoint_id(wall_id: int, endpoint: str | None) -> str | None:
    """Return the result-stable endpoint identity used across diagnostics."""
    if endpoint is None:
        return None
    return f"wall_{wall_id}:{endpoint}"


def _connection_sort_key(connection: WallGraphConnection) -> tuple:
    """Keep connection diagnostics deterministic across input ordering."""
    return (
        tuple(connection.wall_ids),
        tuple(value or "" for value in connection.endpoint_ids),
        connection.type,
        connection.decision,
        tuple(np.round(connection.after_coordinates[0], 8)),
        tuple(np.round(connection.after_coordinates[1], 8)),
    )


def _valid_wall_geometry(wall: WallSegment) -> bool:
    """Return whether a candidate can contribute endpoint diagnostics."""
    return (
        np.isfinite(wall.start).all()
        and np.isfinite(wall.end).all()
        and wall.length > 1e-9
    )


def _nearest_endpoint_name(along: float, length: float) -> str:
    """Name the endpoint that an out-of-bounds intersection would extend."""
    return "start" if abs(along) <= abs(length - along) else "end"


def _extension_limit(wall: WallSegment, maximum: float) -> float:
    """Bound extension by the candidate's fit confidence and global cap."""
    quality = float(np.clip(min(wall.confidence, wall.fit_quality), 0.0, 1.0))
    # Even an accepted low-confidence candidate gets a bounded proposal, but
    # it cannot consume the full closure budget reserved for a strong fit.
    return float(maximum * (0.5 + 0.5 * quality))


def _append_provenance(existing: str, addition: str) -> str:
    """Append a deterministic evidence source without duplicating it."""
    values = [value for value in (str(existing), addition) if value]
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return "; ".join(result)


def _endpoint_metrics(
    walls: list[WallSegment], tolerance: float
) -> dict[str, int | dict[str, float | None]]:
    """Summarize endpoint gaps/components without inventing connections."""
    endpoints: list[tuple[int, np.ndarray]] = []
    for wall in walls:
        if not _valid_wall_geometry(wall):
            continue
        endpoints.extend(((wall.index, wall.start), (wall.index, wall.end)))
    if len(endpoints) < 2:
        return {
            "endpoint_count": len(endpoints),
            "endpoint_components": len(endpoints),
            "endpoint_incidence_count": 0,
            "nearest_endpoint_gap_m": {
                "median": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
            },
        }

    gaps: list[float] = []
    incidence = 0
    parent = list(range(len(endpoints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(endpoints)):
        nearest = float("inf")
        first_wall, first_point = endpoints[first]
        for second in range(len(endpoints)):
            if first == second or endpoints[second][0] == first_wall:
                continue
            distance = float(np.linalg.norm(first_point - endpoints[second][1]))
            nearest = min(nearest, distance)
            if second > first and distance <= tolerance:
                incidence += 1
                union(first, second)
        if np.isfinite(nearest):
            gaps.append(nearest)
    roots = {find(index) for index in range(len(endpoints))}
    gap_array = np.asarray(gaps, dtype=float)
    nearest_metadata = {
        "median": float(np.median(gap_array)) if len(gap_array) else None,
        "p75": float(np.percentile(gap_array, 75)) if len(gap_array) else None,
        "p90": float(np.percentile(gap_array, 90)) if len(gap_array) else None,
        "p95": float(np.percentile(gap_array, 95)) if len(gap_array) else None,
        "p99": float(np.percentile(gap_array, 99)) if len(gap_array) else None,
        "max": float(np.max(gap_array)) if len(gap_array) else None,
    }
    return {
        "endpoint_count": len(endpoints),
        "endpoint_components": len(roots),
        "endpoint_incidence_count": incidence,
        "nearest_endpoint_gap_m": nearest_metadata,
    }
