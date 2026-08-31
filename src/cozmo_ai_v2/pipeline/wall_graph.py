"""Global, plane-constrained wall graph solving.

Wall fitting produces independent finite segments.  This module turns their
shared line intersections into one deterministic node set before any room
faces are polygonized.  A node is solved from all incident finished-face line
equations at once; endpoints are never snapped independently to a nearby
endpoint estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from copy import deepcopy

import numpy as np

from .planes import FINISHED_FACE, WallSegment

JunctionKind = Literal["corner", "t", "x"]


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

    candidates = [deepcopy(wall) for wall in walls]
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
    for first_index, first in enumerate(active):
        for second in active[first_index + 1 :]:
            hit = _line_intersection(first, second)
            if hit is None:
                continue
            point, along_first, along_second = hit
            first_inside = -node_tolerance <= along_first <= first.length + node_tolerance
            second_inside = -node_tolerance <= along_second <= second.length + node_tolerance
            if not (first_inside and second_inside):
                continue
            first_endpoint = _near_endpoint(along_first, first.length, node_tolerance)
            second_endpoint = _near_endpoint(along_second, second.length, node_tolerance)
            if not first_endpoint and not second_endpoint:
                if allow_x_junctions:
                    kind: JunctionKind = "x"
                else:
                    weaker = first if _wall_sort_key(first) > _wall_sort_key(second) else second
                    weaker.tags = sorted(set(weaker.tags) | {"unintended-crossing"})
                    weaker.quarantined = True
                    weaker.snap_status = "rejected-crossing"
                    rejected_crossings.append(tuple(sorted((first.index, second.index))))
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
    intersections = [
        hit
        for hit in intersections
        if hit.first in by_index and hit.second in by_index
    ]
    clusters = _cluster_intersections(intersections, node_tolerance)
    nodes: list[WallGraphNode] = []
    endpoint_assignments: dict[tuple[int, str], tuple[int, float]] = {}
    for node_id, cluster in enumerate(clusters):
        incident = sorted({index for hit in cluster for index in (hit.first, hit.second)})
        coordinate = _solve_node_coordinate(cluster, by_index)
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
                endpoint = _endpoint_name(along, wall.length, node_tolerance)
                if endpoint is None:
                    continue
                key = (index, endpoint)
                distance = abs(along if endpoint == "start" else wall.length - along)
                prior = endpoint_assignments.get(key)
                if prior is None or distance < prior[1]:
                    endpoint_assignments[key] = (node_id, distance)

    node_by_id = {node.id: node for node in nodes}
    snapped_endpoint_count = 0
    for (index, endpoint), (node_id, _) in endpoint_assignments.items():
        wall = by_index[index]
        coordinate = node_by_id[node_id].coordinate.copy()
        current = wall.start if endpoint == "start" else wall.end
        if np.linalg.norm(current - coordinate) > 1e-9:
            snapped_endpoint_count += 1
            if endpoint == "start":
                wall.start = coordinate
            else:
                wall.end = coordinate
            wall.tags = sorted(set(wall.tags) | {"global-junction"})
            wall.snap_status = "graph-solved"

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
    return WallGraph(
        walls=tuple(final_walls),
        candidates=tuple(candidates),
        nodes=tuple(nodes),
        rejected_crossings=tuple(sorted(set(rejected_crossings))),
        snapped_endpoint_count=snapped_endpoint_count,
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
