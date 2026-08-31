"""Typed input/output contracts for Phase 1 floor-plan vectorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .planes import FINISHED_FACE, WallSegment
from .projection import DensityMap
from .wall_graph import WallGraph, WallGraphDiagnostics, WallGraphNode

if TYPE_CHECKING:
    from .roomformer import RoomFormerProposal


@dataclass(frozen=True)
class OpeningEvidence:
    """Opening evidence passed from surface occupancy into vectorization."""

    wall_index: int
    kind: str
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    confidence: float
    source: str = "surface occupancy"


@dataclass(frozen=True)
class AdjacencyEvidence:
    """Evidence for an edge between two vectorized room faces."""

    room_a: int
    room_b: int
    via: str | None
    confidence: float
    source: str = "validated wall graph"


@dataclass(frozen=True)
class FaceEvidence:
    """A graph face that passed floor observation/visibility validation."""

    polygon: np.ndarray
    area: float
    observed_coverage: float
    visibility: float
    confidence: float
    provenance: str = "validated wall graph + observed floor"

    def __post_init__(self) -> None:
        polygon = np.asarray(self.polygon, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("face polygon must have shape (N, 2)")
        object.__setattr__(self, "polygon", polygon)
        object.__setattr__(self, "area", float(self.area))
        for name in ("observed_coverage", "visibility", "confidence"):
            object.__setattr__(
                self,
                name,
                float(np.clip(getattr(self, name), 0.0, 1.0)),
            )


@dataclass(frozen=True)
class VectorizerInput:
    """Explicit data contract consumed by the Phase 1 vectorizer.

    ``wall_support`` is the integer wall-point support map and
    ``observability`` is its boolean counterpart.  Candidate segments and
    junction evidence are kept alongside the map so downstream stages cannot
    accidentally use a density raster without knowing which wall fits and
    coordinate convention produced it.
    """

    density: DensityMap
    wall_support: np.ndarray
    observability: np.ndarray
    candidate_segments: tuple[WallSegment, ...] = ()
    junction_evidence: tuple[WallGraphNode, ...] = ()
    coordinate_convention: str = FINISHED_FACE

    def __post_init__(self) -> None:
        wall_support = np.asarray(self.wall_support)
        observability = np.asarray(self.observability, dtype=bool)
        if wall_support.shape != self.density.shape:
            raise ValueError("wall_support must match density shape")
        if observability.shape != self.density.shape:
            raise ValueError("observability must match density shape")
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported wall coordinate convention")
        object.__setattr__(self, "wall_support", wall_support)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(self, "candidate_segments", tuple(self.candidate_segments))
        object.__setattr__(self, "junction_evidence", tuple(self.junction_evidence))

    @property
    def confidence(self) -> float:
        """Conservative confidence summary for the vectorizer input."""
        if not self.candidate_segments:
            return 0.0 if self.density.retained_count == 0 else 1.0
        return float(min(segment.confidence for segment in self.candidate_segments))


@dataclass(frozen=True)
class VectorizerOutput:
    """Stable Phase 1 vectorizer output and auditable evidence metadata."""

    vector_input: VectorizerInput
    accepted_segments: tuple[WallSegment, ...]
    faces: tuple[FaceEvidence, ...] = ()
    openings: tuple[OpeningEvidence, ...] = ()
    adjacency: tuple[AdjacencyEvidence, ...] = ()
    junctions: tuple[WallGraphNode, ...] = ()
    rejected_crossings: tuple[tuple[int, int], ...] = ()
    confidence: float = 0.0
    roomformer: RoomFormerProposal | None = None
    graph_diagnostics: WallGraphDiagnostics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_segments", tuple(self.accepted_segments))
        object.__setattr__(self, "faces", tuple(self.faces))
        object.__setattr__(self, "openings", tuple(self.openings))
        object.__setattr__(self, "adjacency", tuple(self.adjacency))
        object.__setattr__(self, "junctions", tuple(self.junctions))
        object.__setattr__(
            self,
            "rejected_crossings",
            tuple(tuple(pair) for pair in self.rejected_crossings),
        )
        object.__setattr__(
            self,
            "confidence",
            float(np.clip(self.confidence, 0.0, 1.0)),
        )

    def to_metadata(self) -> dict:
        """Return JSON-ready map, support, candidate, and evidence metadata."""
        density = self.vector_input.density
        accepted_ids = {wall.index for wall in self.accepted_segments}
        return {
            "coordinate_convention": self.vector_input.coordinate_convention,
            "confidence": round(self.confidence, 4),
            "density": {
                "resolution_m": density.resolution,
                "origin": density.origin.tolist(),
                "bounds": [list(bound) for bound in density.bounds],
                "clip_bounds": (
                    [list(bound) for bound in density.clip_bounds]
                    if density.clip_bounds is not None
                    else None
                ),
                "shape": list(density.shape),
                "wall_band_m_above_floor": list(density.wall_band),
                "effective_height_bounds_m": list(density.height_bounds),
                "floor_height_m": density.floor_height,
                "ceiling_height_m": density.ceiling_height,
                "count_semantics": "retained finite wall-band points per cell",
                "density_semantics": "retained points per square metre",
                "counts": density.counts.tolist(),
                "observed": density.observed.tolist(),
                "empty": density.empty.tolist(),
                "input_count": density.input_count,
                "finite_input_count": density.finite_input_count,
                "invalid_input_count": density.invalid_input_count,
                "band_count": density.band_count,
                "retained_count": density.retained_count,
                "out_of_bounds_count": density.out_of_bounds_count,
                "observed_cell_count": int(density.observed.sum()),
                "empty_cell_count": int(density.empty.sum()),
            },
            "candidate_segments": [
                _segment_metadata(wall, wall.index in accepted_ids)
                for wall in self.vector_input.candidate_segments
            ],
            "junctions": [_junction_metadata(node) for node in self.junctions],
            "wall_graph": (
                self.graph_diagnostics.to_solver_metadata()
                if self.graph_diagnostics is not None
                else None
            ),
            "rejected_crossings": [list(pair) for pair in self.rejected_crossings],
            "openings": [
                {
                    "wall_id": evidence.wall_index,
                    "kind": evidence.kind,
                    "u_range_m": list(evidence.u_range),
                    "v_range_m": list(evidence.v_range),
                    "confidence": round(evidence.confidence, 4),
                    "source": evidence.source,
                }
                for evidence in self.openings
            ],
            "adjacency": [
                {
                    "room_a": evidence.room_a,
                    "room_b": evidence.room_b,
                    "via": evidence.via,
                    "confidence": round(evidence.confidence, 4),
                    "source": evidence.source,
                }
                for evidence in self.adjacency
            ],
            "faces": [
                {
                    "polygon": face.polygon.tolist(),
                    "area_m2": face.area,
                    "observed_coverage": round(face.observed_coverage, 4),
                    "visibility": round(face.visibility, 4),
                    "confidence": round(face.confidence, 4),
                    "provenance": face.provenance,
                }
                for face in self.faces
            ],
            "roomformer": self.roomformer.to_metadata()
            if self.roomformer is not None
            else None,
        }

def build_vectorizer_input(
    density: DensityMap,
    candidate_segments: Iterable[WallSegment] = (),
    junction_evidence: Iterable[WallGraphNode] = (),
) -> VectorizerInput:
    """Construct the explicit map-plus-geometry vectorizer input contract."""
    return VectorizerInput(
        density=density,
        wall_support=density.counts,
        observability=density.observed,
        candidate_segments=tuple(candidate_segments),
        junction_evidence=tuple(junction_evidence),
        coordinate_convention=FINISHED_FACE,
    )


def build_vectorizer_output(
    vector_input: VectorizerInput,
    *,
    graph: WallGraph | None = None,
    accepted_segments: Iterable[WallSegment] | None = None,
    faces: Iterable[FaceEvidence] = (),
    openings: Iterable[OpeningEvidence] = (),
    adjacency: Iterable[AdjacencyEvidence] = (),
    roomformer: RoomFormerProposal | None = None,
) -> VectorizerOutput:
    """Assemble graph, face, opening, and adjacency evidence into one output."""
    if graph is not None:
        accepted = tuple(graph.walls)
        junctions = tuple(graph.nodes)
    else:
        accepted = tuple(accepted_segments or ())
        junctions = vector_input.junction_evidence
    face_items = tuple(faces)
    opening_items = tuple(openings)
    adjacency_items = tuple(adjacency)
    accepted_confidence = (
        min(wall.confidence for wall in accepted) if accepted else vector_input.confidence
    )
    face_confidence = [face.confidence for face in face_items]
    all_confidence = [accepted_confidence, *face_confidence]
    return VectorizerOutput(
        vector_input=vector_input,
        accepted_segments=accepted,
        faces=face_items,
        openings=opening_items,
        adjacency=adjacency_items,
        junctions=junctions,
        rejected_crossings=(graph.rejected_crossings if graph is not None else ()),
        confidence=float(min(all_confidence)) if all_confidence else 0.0,
        roomformer=roomformer,
        graph_diagnostics=(graph.diagnostics if graph is not None else None),
    )


def _segment_metadata(wall: WallSegment, accepted: bool) -> dict:
    return {
        "id": wall.index,
        "start": wall.start.tolist(),
        "end": wall.end.tolist(),
        "normal": wall.normal.tolist(),
        "length_m": round(wall.length, 6),
        "support_points": wall.inlier_count,
        "residual_rms_m": (
            round(wall.residual_rms, 6)
            if np.isfinite(wall.residual_rms)
            else None
        ),
        "confidence": round(wall.confidence, 4),
        "fit_quality": round(wall.fit_quality, 4),
        "accepted": accepted,
        "quarantined": wall.quarantined,
        "coordinate_convention": wall.coordinate_convention,
        "provenance": wall.provenance,
        "snap_status": wall.snap_status,
        "snap_residual_m": round(wall.snap_residual, 6),
        "tags": sorted(set(wall.tags)),
    }


def _junction_metadata(node: WallGraphNode) -> dict:
    return {
        "id": node.id,
        "coordinate": node.coordinate.tolist(),
        "kind": node.kind,
        "incident_walls": list(node.incident_walls),
        "confidence": round(node.confidence, 4),
        "provenance": node.provenance,
    }
