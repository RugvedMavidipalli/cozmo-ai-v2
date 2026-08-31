"""Optional RoomFormer contract for the Phase 1 vectorizer.

RoomFormer is deliberately an adapter, not a dependency of the integrated
pipeline.  The adapter turns the existing ``(density, observability)`` map
contract into a small, documented tensor and turns model predictions back
into finished-face wall-graph proposals.  Importing this module never imports
RoomFormer, PyTorch, or a GPU runtime; the default adapter also never looks
for a model unless a local checkpoint is explicitly configured.

The model-facing tensor uses ``(batch, channel, x, y)`` order.  This is
intentional: the project's NumPy rasters use ``counts[x, y]`` and retaining
that order avoids a silent vertical flip or transpose at the optional model
boundary.  Channel 0 is wall-point density in points/m² and channel 1 is the
0/1 observability mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np

from .planes import FINISHED_FACE, WallSegment
from .vectorizer import OpeningEvidence, VectorizerInput
from .wall_graph import WallGraphNode

RoomFormerCoordinateSpace = Literal["plan", "cell", "normalized"]
TENSOR_CHANNELS = ("wall_density_points_per_m2", "observability")

__all__ = [
    "RoomFormerCoordinateSpace",
    "TENSOR_CHANNELS",
    "RoomFormerConfig",
    "RoomFormerTensor",
    "RoomFormerPrediction",
    "WallGraphProposal",
    "RoomFormerProposal",
    "RoomFormerAdapter",
    "build_roomformer_tensor",
]


@dataclass(frozen=True)
class RoomFormerConfig:
    """Local-only configuration for an optional RoomFormer backend.

    ``checkpoint`` must point to an existing local file.  No download or
    registry lookup is performed by this adapter.  With no checkpoint and no
    injected ``backend_factory``, prediction deterministically falls back to
    the point-cloud wall graph.
    """

    checkpoint: str | Path | None = None
    device: str = "cpu"
    model_name: str = "roomformer"
    coordinate_convention: str = FINISHED_FACE

    def __post_init__(self) -> None:
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported RoomFormer coordinate convention")
        if not str(self.device):
            raise ValueError("RoomFormer device must be non-empty")
        if not str(self.model_name):
            raise ValueError("RoomFormer model_name must be non-empty")


@dataclass(frozen=True)
class RoomFormerTensor:
    """Validated tensor and geometry metadata supplied to RoomFormer."""

    data: np.ndarray
    resolution: float
    origin: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]
    coordinate_convention: str = FINISHED_FACE
    channels: tuple[str, str] = TENSOR_CHANNELS

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float32)
        origin = np.asarray(self.origin, dtype=float)
        bounds = np.asarray(self.bounds, dtype=float)
        if data.ndim != 4 or data.shape[0] != 1 or data.shape[1] != 2:
            raise ValueError("RoomFormer tensor must have shape (1, 2, x, y)")
        if not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("RoomFormer tensor must be finite and non-negative")
        if min(data.shape[2:]) <= 0:
            raise ValueError("RoomFormer tensor spatial shape must be positive")
        if not np.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("RoomFormer tensor resolution must be positive")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("RoomFormer tensor origin must have shape (2,)")
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("RoomFormer tensor bounds must have shape (2, 2)")
        if (bounds[1] <= bounds[0]).any():
            raise ValueError("RoomFormer tensor bounds must have positive extent")
        expected_upper = origin + np.asarray(data.shape[2:], dtype=float) * float(self.resolution)
        if not np.allclose(bounds[0], origin) or not np.allclose(bounds[1], expected_upper):
            raise ValueError("RoomFormer tensor bounds do not match origin, shape, and resolution")
        if len(self.channels) != 2 or tuple(self.channels) != TENSOR_CHANNELS:
            raise ValueError("RoomFormer tensor channels do not match the contract")
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported RoomFormer coordinate convention")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "bounds", (tuple(bounds[0]), tuple(bounds[1])))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Tensor shape, ``(batch, channel, x, y)``."""
        return tuple(int(value) for value in self.data.shape)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Raster shape in the existing ``(x, y)`` convention."""
        return self.shape[2:]


def build_roomformer_tensor(vector_input: VectorizerInput) -> RoomFormerTensor:
    """Build the deterministic density/observability tensor contract.

    The map is not resized, padded, normalized, or transposed.  A RoomFormer
    implementation that needs a different model input size must perform that
    operation inside its own backend and return coordinates in the declared
    coordinate space.  This keeps metric geometry and observability lossless
    at the pipeline boundary.
    """
    density = vector_input.density
    if vector_input.coordinate_convention != FINISHED_FACE:
        raise ValueError(
            "RoomFormer adapter currently consumes finished-face vectorizer maps"
        )
    support = np.asarray(vector_input.wall_support)
    observed = np.asarray(vector_input.observability, dtype=bool)
    if support.shape != density.shape or observed.shape != density.shape:
        raise ValueError("vectorizer support and observability must match density shape")
    density_channel = np.asarray(density.density, dtype=np.float32)
    if not np.isfinite(density_channel).all() or (density_channel < 0).any():
        raise ValueError("vectorizer density must be finite and non-negative")
    data = np.stack(
        [density_channel, observed.astype(np.float32, copy=False)], axis=0
    )[None, ...]
    return RoomFormerTensor(
        data=data,
        resolution=density.resolution,
        origin=density.origin,
        bounds=density.bounds,
        coordinate_convention=vector_input.coordinate_convention,
    )


@dataclass(frozen=True)
class RoomFormerPrediction:
    """Normalized, validated model prediction before graph conversion.

    Coordinates are allowed in plan metres, continuous raster-cell
    coordinates, or normalized raster coordinates.  Normalized coordinates
    use ``[0, 1]`` over the full raster extent; cell coordinates use the
    raster boundary convention where ``(0, 0)`` is the lower-left boundary
    and ``(shape_x, shape_y)`` is the upper boundary.
    """

    polygons: tuple[np.ndarray, ...] = ()
    corners: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    topology: tuple[tuple[int, int], ...] = ()
    coordinate_space: RoomFormerCoordinateSpace = "plan"
    coordinate_convention: str = FINISHED_FACE
    confidence: float = 0.0
    model_provenance: str = "roomformer"
    opening_predictions: tuple[OpeningEvidence, ...] = ()

    def __post_init__(self) -> None:
        polygons = tuple(np.asarray(polygon, dtype=float) for polygon in self.polygons)
        for polygon in polygons:
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError("RoomFormer polygons must each have shape (N>=3, 2)")
            if not np.isfinite(polygon).all():
                raise ValueError("RoomFormer polygon coordinates must be finite")
        corners = np.asarray(self.corners, dtype=float)
        if corners.ndim != 2 or corners.shape[1] != 2:
            raise ValueError("RoomFormer corners must have shape (N, 2)")
        if not np.isfinite(corners).all():
            raise ValueError("RoomFormer corner coordinates must be finite")
        if self.coordinate_space not in ("plan", "cell", "normalized"):
            raise ValueError("unsupported RoomFormer coordinate space")
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported RoomFormer coordinate convention")
        topology = tuple(_coerce_edge(edge) for edge in self.topology)
        for first, second in topology:
            if first >= len(corners) or second >= len(corners):
                raise ValueError("RoomFormer topology references an unknown corner")
            if first == second:
                raise ValueError("RoomFormer topology cannot contain self-edges")
        confidence = float(self.confidence)
        confidence = confidence if np.isfinite(confidence) else 0.0
        object.__setattr__(self, "polygons", polygons)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "confidence", float(np.clip(confidence, 0.0, 1.0)))
        object.__setattr__(self, "model_provenance", str(self.model_provenance))
        object.__setattr__(self, "opening_predictions", tuple(self.opening_predictions))


@dataclass(frozen=True)
class WallGraphProposal:
    """RoomFormer's global wall-graph proposal in finished-face coordinates."""

    segments: tuple[WallSegment, ...]
    nodes: tuple[WallGraphNode, ...]
    topology: tuple[tuple[int, int], ...]
    polygons: tuple[np.ndarray, ...] = ()
    coordinate_convention: str = FINISHED_FACE
    confidence: float = 0.0
    provenance: str = "roomformer"

    def __post_init__(self) -> None:
        if self.coordinate_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported wall-graph proposal convention")
        segments = tuple(self.segments)
        nodes = tuple(self.nodes)
        topology = tuple(_coerce_edge(edge) for edge in self.topology)
        if any(
            segment.coordinate_convention != self.coordinate_convention
            for segment in segments
        ):
            raise ValueError("wall-graph proposal segments use mixed conventions")
        for polygon in self.polygons:
            polygon = np.asarray(polygon, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2:
                raise ValueError("proposal polygons must have shape (N, 2)")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "polygons", tuple(np.asarray(p, dtype=float) for p in self.polygons))
        confidence = float(self.confidence)
        object.__setattr__(
            self, "confidence", float(np.clip(confidence if np.isfinite(confidence) else 0.0, 0, 1))
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-ready graph proposal metadata."""
        return {
            "coordinate_convention": self.coordinate_convention,
            "confidence": round(self.confidence, 4),
            "provenance": self.provenance,
            "topology": [list(edge) for edge in self.topology],
            "corners": [node.coordinate.tolist() for node in self.nodes],
            "junctions": [
                {
                    "id": node.id,
                    "coordinate": node.coordinate.tolist(),
                    "kind": node.kind,
                    "incident_walls": list(node.incident_walls),
                    "confidence": round(node.confidence, 4),
                }
                for node in self.nodes
            ],
            "segments": [_proposal_segment_metadata(segment) for segment in self.segments],
            "polygons": [polygon.tolist() for polygon in self.polygons],
        }


@dataclass(frozen=True)
class RoomFormerProposal:
    """Adapter result, including deterministic fallback and SD-TQ hook state."""

    available: bool
    tensor_shape: tuple[int, int, int, int]
    tensor_convention: str
    graph: WallGraphProposal
    polygons: tuple[np.ndarray, ...] = ()
    corners: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    topology: tuple[tuple[int, int], ...] = ()
    opening_predictions: tuple[OpeningEvidence, ...] = ()
    confidence: float = 0.0
    model_provenance: str = "fallback: point-cloud wall graph"
    fallback_reason: str | None = None
    opening_extension: str = "SD-TQ opening predictions not configured"

    def __post_init__(self) -> None:
        if len(self.tensor_shape) != 4 or self.tensor_shape[0:2] != (1, 2):
            raise ValueError("RoomFormer proposal tensor shape must be (1, 2, x, y)")
        if min(self.tensor_shape[2:]) <= 0:
            raise ValueError("RoomFormer proposal tensor spatial shape must be positive")
        if self.tensor_convention not in ("finished_face", "centerline", "room_side"):
            raise ValueError("unsupported RoomFormer proposal convention")
        if self.graph.coordinate_convention != self.tensor_convention:
            raise ValueError("RoomFormer graph and tensor conventions do not match")
        corners = np.asarray(self.corners, dtype=float)
        if corners.ndim != 2 or corners.shape[1] != 2 or not np.isfinite(corners).all():
            raise ValueError("RoomFormer proposal corners must have shape (N, 2)")
        polygons = tuple(np.asarray(polygon, dtype=float) for polygon in self.polygons)
        if any(
            polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all()
            for polygon in polygons
        ):
            raise ValueError("RoomFormer proposal polygons must have shape (N, 2)")
        object.__setattr__(self, "tensor_shape", tuple(int(value) for value in self.tensor_shape))
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "polygons", polygons)
        object.__setattr__(
            self,
            "topology",
            tuple(_coerce_edge(edge) for edge in self.topology),
        )
        object.__setattr__(self, "opening_predictions", tuple(self.opening_predictions))
        confidence = float(self.confidence)
        object.__setattr__(
            self,
            "confidence",
            float(np.clip(confidence if np.isfinite(confidence) else 0.0, 0, 1)),
        )

    @property
    def segments(self) -> tuple[WallSegment, ...]:
        """Convenience access to the graph-format candidate segments."""
        return self.graph.segments

    @property
    def nodes(self) -> tuple[WallGraphNode, ...]:
        """Convenience access to graph-format junction proposals."""
        return self.graph.nodes

    def to_metadata(self) -> dict[str, Any]:
        """Return model provenance, tensor contract, and proposal metadata."""
        return {
            "available": self.available,
            "model_provenance": self.model_provenance,
            "confidence": round(self.confidence, 4),
            "fallback_reason": self.fallback_reason,
            "tensor": {
                "shape": list(self.tensor_shape),
                "layout": "(batch, channel, x, y)",
                "channels": list(TENSOR_CHANNELS),
                "coordinate_convention": self.tensor_convention,
            },
            "graph": self.graph.to_metadata(),
            "opening_extension": self.opening_extension,
            "opening_predictions": [
                {
                    "wall_id": opening.wall_index,
                    "kind": opening.kind,
                    "u_range_m": list(opening.u_range),
                    "v_range_m": list(opening.v_range),
                    "confidence": round(opening.confidence, 4),
                    "source": opening.source,
                }
                for opening in self.opening_predictions
            ],
        }


BackendFactory = Callable[[RoomFormerConfig], Any]
OpeningPredictor = Callable[[np.ndarray, RoomFormerPrediction], Iterable[Any]]


class RoomFormerAdapter:
    """Lazy optional RoomFormer backend with a deterministic safe fallback.

    The default path has no checkpoint and returns an unavailable proposal
    without importing any optional package.  Tests and an application-owned
    deployment may inject ``backend_factory``; that factory is called only on
    the first non-empty prediction.  A backend is expected to be callable or
    expose ``predict(tensor)`` and return a mapping or
    :class:`RoomFormerPrediction`.
    """

    def __init__(
        self,
        config: RoomFormerConfig | None = None,
        *,
        backend_factory: BackendFactory | None = None,
        opening_predictor: OpeningPredictor | None = None,
    ) -> None:
        self.config = config or RoomFormerConfig()
        self.backend_factory = backend_factory
        self.opening_predictor = opening_predictor
        self._backend: Any = None
        self._backend_loaded = False
        self._load_reason: str | None = None

    def predict(self, vector_input: VectorizerInput) -> RoomFormerProposal:
        """Predict a graph proposal, or return the stable fallback proposal."""
        if self.config.coordinate_convention != vector_input.coordinate_convention:
            raise ValueError(
                "RoomFormer configuration convention does not match vectorizer input"
            )
        tensor = build_roomformer_tensor(vector_input)
        if vector_input.density.retained_count == 0:
            return self._fallback(tensor, "empty wall-density map")
        backend = self._load_backend()
        if backend is None:
            return self._fallback(tensor, self._load_reason or "RoomFormer unavailable")
        try:
            raw = (
                backend(tensor.data)
                if callable(backend)
                else backend.predict(tensor.data)
            )
            prediction = _coerce_prediction(raw, self.config)
            return self._convert_prediction(tensor, prediction)
        except (TypeError, ValueError, KeyError) as exc:
            # A malformed optional model output must not corrupt the metric
            # graph or make the integrated CLI unusable.
            return self._fallback(tensor, f"invalid RoomFormer prediction: {exc}")
        except Exception as exc:  # pragma: no cover - backend-specific safety net
            return self._fallback(tensor, f"RoomFormer inference unavailable: {type(exc).__name__}")

    # ``propose`` is the name used by vectorizer integrations that keep model
    # proposals separate from accepted graph geometry.
    propose = predict

    def _load_backend(self) -> Any | None:
        if self._backend_loaded:
            return self._backend
        self._backend_loaded = True
        if self.backend_factory is not None:
            try:
                self._backend = self.backend_factory(self.config)
                self._load_reason = None if self._backend is not None else "backend factory returned no model"
            except (ImportError, ModuleNotFoundError):
                self._load_reason = "optional RoomFormer backend is not installed"
            except Exception:
                self._load_reason = "local RoomFormer backend could not be loaded"
            return self._backend

        checkpoint = self.config.checkpoint
        if checkpoint is None:
            self._load_reason = "no local RoomFormer checkpoint configured"
            return None
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            self._load_reason = "configured RoomFormer checkpoint is not a local file"
            return None
        try:
            # The optional module owns model construction.  We pass a local
            # path only; this adapter intentionally has no download branch.
            module = importlib.import_module("roomformer")
            factory = getattr(module, "load_model", None)
            if factory is None:
                self._load_reason = "installed RoomFormer module has no load_model hook"
                return None
            self._backend = factory(
                checkpoint=str(checkpoint_path), device=self.config.device
            )
        except (ImportError, ModuleNotFoundError):
            self._load_reason = "optional RoomFormer backend is not installed"
        except Exception:
            self._load_reason = "local RoomFormer backend could not be loaded"
        return self._backend

    def _fallback(self, tensor: RoomFormerTensor, reason: str) -> RoomFormerProposal:
        graph = WallGraphProposal(
            segments=(),
            nodes=(),
            topology=(),
            coordinate_convention=self.config.coordinate_convention,
            provenance="fallback: point-cloud wall graph",
        )
        return RoomFormerProposal(
            available=False,
            tensor_shape=tensor.shape,
            tensor_convention=tensor.coordinate_convention,
            graph=graph,
            confidence=0.0,
            model_provenance="fallback: point-cloud wall graph",
            fallback_reason=reason,
            opening_extension=(
                "SD-TQ opening predictions not configured"
                if self.opening_predictor is None
                else "SD-TQ opening predictor unavailable during fallback"
            ),
        )

    def _convert_prediction(
        self, tensor: RoomFormerTensor, prediction: RoomFormerPrediction
    ) -> RoomFormerProposal:
        if prediction.coordinate_convention != tensor.coordinate_convention:
            raise ValueError("RoomFormer prediction convention does not match tensor")
        polygons = tuple(_to_plan(polygon, prediction.coordinate_space, tensor) for polygon in prediction.polygons)
        corners = _to_plan(prediction.corners, prediction.coordinate_space, tensor)
        topology = prediction.topology
        if len(topology) == 0 and polygons:
            corners, topology = _polygon_graph(polygons)
        if len(topology) and len(corners) == 0:
            raise ValueError("RoomFormer topology requires corners")
        segments, segment_edges = _segments_from_graph(corners, topology, prediction)
        nodes = _nodes_from_graph(corners, segment_edges, segments, prediction)
        graph = WallGraphProposal(
            segments=tuple(segments),
            nodes=tuple(nodes),
            topology=topology,
            polygons=polygons,
            coordinate_convention=prediction.coordinate_convention,
            confidence=prediction.confidence,
            provenance=prediction.model_provenance,
        )
        openings = tuple(prediction.opening_predictions)
        extension = "SD-TQ opening predictions not configured"
        if self.opening_predictor is not None:
            raw_openings = self.opening_predictor(tensor.data, prediction)
            openings = tuple(_coerce_opening(item) for item in raw_openings)
            extension = "SD-TQ opening predictor"
        return RoomFormerProposal(
            available=True,
            tensor_shape=tensor.shape,
            tensor_convention=tensor.coordinate_convention,
            graph=graph,
            polygons=polygons,
            corners=corners,
            topology=topology,
            opening_predictions=openings,
            confidence=prediction.confidence,
            model_provenance=prediction.model_provenance,
            opening_extension=extension,
        )


def _coerce_prediction(raw: Any, config: RoomFormerConfig) -> RoomFormerPrediction:
    if isinstance(raw, RoomFormerPrediction):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("RoomFormer backend must return a mapping or RoomFormerPrediction")
    return RoomFormerPrediction(
        polygons=_coerce_polygons(raw.get("polygons", ())),
        corners=np.asarray(raw.get("corners", np.empty((0, 2))), dtype=float),
        topology=tuple(raw.get("topology", ())),
        coordinate_space=raw.get("coordinate_space", raw.get("coordinate_system", "plan")),
        coordinate_convention=raw.get("coordinate_convention", config.coordinate_convention),
        confidence=raw.get("confidence", 0.0),
        model_provenance=raw.get("model_provenance", config.model_name),
        opening_predictions=tuple(_coerce_opening(item) for item in raw.get("openings", ())),
    )


def _coerce_polygons(polygons: Any) -> tuple[np.ndarray, ...]:
    if polygons is None:
        return ()
    array = np.asarray(polygons, dtype=object)
    if isinstance(polygons, np.ndarray) and polygons.ndim == 3:
        return tuple(np.asarray(polygon, dtype=float) for polygon in polygons)
    if array.ndim == 2 and array.shape[1] == 2:
        return (np.asarray(polygons, dtype=float),)
    return tuple(np.asarray(polygon, dtype=float) for polygon in polygons)


def _coerce_edge(edge: Any) -> tuple[int, int]:
    values = tuple(edge)
    if len(values) != 2:
        raise ValueError("RoomFormer topology edges must contain two indices")
    first, second = int(values[0]), int(values[1])
    if first < 0 or second < 0:
        raise ValueError("RoomFormer topology indices must be non-negative")
    return first, second


def _to_plan(
    coordinates: np.ndarray,
    coordinate_space: RoomFormerCoordinateSpace,
    tensor: RoomFormerTensor,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("RoomFormer coordinates must have shape (N, 2)")
    if not np.isfinite(coordinates).all():
        raise ValueError("RoomFormer coordinates must be finite")
    if coordinate_space == "plan":
        return coordinates.copy()
    shape = np.asarray(tensor.spatial_shape, dtype=float)
    if coordinate_space == "normalized":
        if (coordinates < 0).any() or (coordinates > 1).any():
            raise ValueError("normalized RoomFormer coordinates must be in [0, 1]")
        cell = coordinates * shape
    elif coordinate_space == "cell":
        if (coordinates < 0).any() or (coordinates > shape).any():
            raise ValueError("cell RoomFormer coordinates must lie within map bounds")
        cell = coordinates
    else:  # pragma: no cover - guarded by RoomFormerPrediction
        raise ValueError("unsupported RoomFormer coordinate space")
    return tensor.origin + cell * tensor.resolution


def _polygon_graph(
    polygons: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    corners: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    for polygon in polygons:
        points = _without_repeated_closure(polygon)
        start = len(corners)
        corners.extend(points)
        edges.extend(
            (start + index, start + ((index + 1) % len(points)))
            for index in range(len(points))
        )
    return np.asarray(corners, dtype=float).reshape(-1, 2), tuple(edges)


def _without_repeated_closure(polygon: np.ndarray) -> np.ndarray:
    if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1], atol=1e-9):
        return polygon[:-1]
    return polygon


def _segments_from_graph(
    corners: np.ndarray,
    topology: tuple[tuple[int, int], ...],
    prediction: RoomFormerPrediction,
) -> tuple[list[WallSegment], list[tuple[int, int]]]:
    segments: list[WallSegment] = []
    segment_edges: list[tuple[int, int]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for edge_number, (first, second) in enumerate(topology):
        start = corners[first]
        end = corners[second]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        key = tuple(sorted((tuple(np.round(start, 8)), tuple(np.round(end, 8)))))
        if key in seen:
            continue
        seen.add(key)
        direction = delta / length
        normal = np.array([-direction[1], direction[0]], dtype=float)
        segments.append(
            WallSegment(
                index=-1 - edge_number,
                normal=normal,
                offset=float(normal @ start),
                start=start.copy(),
                end=end.copy(),
                inlier_count=0,
                residual_rms=0.0,
                observed_span=(0.0, length),
                height_range=(0.0, 0.0),
                confidence=prediction.confidence,
                fit_quality=prediction.confidence,
                coordinate_convention=prediction.coordinate_convention,
                provenance=prediction.model_provenance,
                snap_status="roomformer-proposal",
            )
        )
        segment_edges.append((first, second))
    return segments, segment_edges


def _nodes_from_graph(
    corners: np.ndarray,
    topology: Iterable[tuple[int, int]],
    segments: list[WallSegment],
    prediction: RoomFormerPrediction,
) -> list[WallGraphNode]:
    incident: dict[int, list[int]] = {index: [] for index in range(len(corners))}
    for segment, (first, second) in zip(segments, topology):
        incident[first].append(segment.index)
        incident[second].append(segment.index)
    nodes = []
    for index, coordinate in enumerate(corners):
        walls = tuple(sorted(incident[index]))
        if not walls:
            continue
        kind = "corner" if len(walls) <= 2 else "t" if len(walls) == 3 else "x"
        nodes.append(
            WallGraphNode(
                id=index,
                coordinate=coordinate,
                kind=kind,
                incident_walls=walls,
                confidence=prediction.confidence,
                provenance=prediction.model_provenance,
            )
        )
    return nodes


def _coerce_opening(raw: Any) -> OpeningEvidence:
    if isinstance(raw, OpeningEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("RoomFormer opening predictions must be mappings")
    u_range = tuple(float(value) for value in raw.get("u_range", raw.get("u_range_m", (0, 0))))
    v_range = tuple(float(value) for value in raw.get("v_range", raw.get("v_range_m", (0, 0))))
    if len(u_range) != 2 or len(v_range) != 2 or not np.isfinite((*u_range, *v_range)).all():
        raise ValueError("RoomFormer opening ranges must contain four finite values")
    return OpeningEvidence(
        wall_index=int(raw.get("wall_index", raw.get("wall_id", raw.get("segment_index", -1)))),
        kind=str(raw.get("kind", "door")),
        u_range=(u_range[0], u_range[1]),
        v_range=(v_range[0], v_range[1]),
        confidence=float(raw.get("confidence", 0.0)),
        source=str(raw.get("source", "RoomFormer SD-TQ")),
    )


def _proposal_segment_metadata(segment: WallSegment) -> dict[str, Any]:
    return {
        "id": segment.index,
        "start": segment.start.tolist(),
        "end": segment.end.tolist(),
        "normal": segment.normal.tolist(),
        "length_m": round(segment.length, 6),
        "confidence": round(segment.confidence, 4),
        "coordinate_convention": segment.coordinate_convention,
        "provenance": segment.provenance,
    }
