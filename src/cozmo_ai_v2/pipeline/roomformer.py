"""Adapter for RoomFormer SD-TQ opening predictions.

RoomFormer is intentionally not imported here.  This module consumes an
already-produced prediction JSON/object, which lets Stage 8 use RoomFormer
when a separate GPU job is available while keeping the reconstruction and
tests CPU-safe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .openings import NormalizedOpening, _bbox, _range, normalize_opening_kind


def _items(predictions: Any) -> list[Any]:
    if isinstance(predictions, Mapping):
        for key in ("openings", "predictions", "instances", "detections", "objects"):
            if key in predictions:
                return _items(predictions[key])
        # Common tensor-export shape: {labels: [...], boxes: [...], scores: [...]}
        labels = predictions.get("labels", predictions.get("classes"))
        boxes = predictions.get("boxes", predictions.get("bboxes"))
        scores = predictions.get("scores", predictions.get("confidences"))
        if labels is not None and boxes is not None:
            labels = list(labels)
            boxes = list(boxes)
            scores = list(scores) if scores is not None else [1.0] * len(labels)
            return [
                {"label": label, "bbox": box, "score": score}
                for label, box, score in zip(labels, boxes, scores)
            ]
        return [predictions]
    if isinstance(predictions, np.ndarray):
        return list(predictions)
    if isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes)):
        return list(predictions)
    return []


def _value(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


class RoomFormerSDTQAdapter:
    """Normalize SD-TQ predictions without pretending 2D hints are metric."""

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.rejections: list[dict[str, Any]] = []

    def adapt(
        self,
        predictions: Any,
        *,
        frame_shape: tuple[int, int] | None = None,
        walls: Sequence[Any] | None = None,
    ) -> list[NormalizedOpening]:
        """Return normalized opening hints from an SD-TQ result.

        A prediction is metric only when it contains a wall association and
        metric ``u_range``/``v_range`` (or their equivalent offset/size
        fields).  Pixel boxes alone are exported as ``unmeasured`` hints.
        Unsupported labels, furniture, malformed boxes, and low scores are
        retained in ``rejections`` for diagnostics and never enter fusion.
        """
        self.rejections = []
        wall_names = {
            str(getattr(wall, "name", "")): int(getattr(wall, "index"))
            for wall in (walls or [])
            if getattr(wall, "name", None) is not None
        }
        result: list[NormalizedOpening] = []
        for index, raw in enumerate(_items(predictions)):
            if not isinstance(raw, Mapping):
                self._reject(index, "prediction is not an object")
                continue
            kind = normalize_opening_kind(_value(raw, "kind", "label", "category", "class", "type"))
            score = _score(raw)
            if kind is None:
                self._reject(index, "unknown/furniture label")
                continue
            if score < self.min_confidence:
                self._reject(index, "confidence below threshold")
                continue
            bbox = _bbox(_value(raw, "bbox", "box", "bounding_box"))
            if bbox is not None and frame_shape is not None:
                bbox = _bbox_to_pixels(
                    bbox, frame_shape, _value(raw, "bbox_format", "box_format")
                )

            wall_index = _wall_index(raw, wall_names)
            u_range = _metric_range(raw, "u")
            v_range = _metric_range(raw, "v")
            state = "measured" if wall_index is not None and u_range and v_range else "unmeasured"
            result.append(
                NormalizedOpening(
                    wall_index=wall_index,
                    kind=kind,
                    u_range=u_range,
                    v_range=v_range,
                    confidence=score,
                    provenance=["roomformer"],
                    state=state,
                    uncertainty={
                        "basis": "RoomFormer SD-TQ prediction; metric bounds require calibrated wall association",
                    },
                    wall_association_confidence=_association_confidence(raw, wall_index),
                    source_frames=_ints(_value(raw, "frame_index", "frame", "source_frame")),
                    image_bbox=bbox,
                )
            )
        return result

    normalize = adapt

    def _reject(self, index: int, reason: str) -> None:
        self.rejections.append({"index": index, "state": "occluded" if "occlud" in reason else "unmeasured", "reason": reason, "provenance": ["roomformer"]})


def _score(item: Mapping[str, Any]) -> float:
    value = _value(item, "confidence", "score", "probability")
    try:
        return float(np.clip(float(value) if value is not None else 0.0, 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _association_confidence(item: Mapping[str, Any], wall_index: int | None) -> float:
    value = _value(item, "wall_confidence", "association_confidence")
    try:
        return float(np.clip(float(value) if value is not None else (1.0 if wall_index is not None else 0.0), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        values = [value]
    try:
        return [int(v) for v in values]
    except (TypeError, ValueError):
        return []


def _wall_index(item: Mapping[str, Any], wall_names: Mapping[str, int]) -> int | None:
    value = _value(item, "wall_index", "wall_id", "wall")
    if isinstance(value, str):
        if value in wall_names:
            return wall_names[value]
        try:
            value = int(value)
        except ValueError:
            return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metric_range(item: Mapping[str, Any], axis: str) -> tuple[float, float] | None:
    direct = _value(item, f"{axis}_range", f"{axis}_bounds", f"{axis}Range")
    parsed = _range(direct)
    if parsed is not None:
        return parsed
    offset = _value(item, f"{axis}_offset", f"{axis}0", f"{axis}_min")
    size = _value(
        item,
        f"{axis}_size",
        f"{axis}_width" if axis == "u" else f"{axis}_height",
        "width_m" if axis == "u" else "height_m",
    )
    if offset is None or size is None:
        return None
    try:
        start, length = float(offset), float(size)
        return _range((start, start + length))
    except (TypeError, ValueError):
        return None


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
    format_name: object = None,
) -> tuple[float, float, float, float]:
    height, width = shape
    format_text = str(format_name or "").lower()
    if format_text in {"pixel", "pixels", "xyxy"}:
        return bbox
    if format_text in {"normalized", "normalized_01", "01"} or (
        not format_text and max(bbox) <= 1.0 and min(bbox) >= 0.0
    ):
        return (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
    if format_text in {"sd-tq", "normalized_1000", "1000"} or (
        not format_text and max(bbox) <= 1000.0 and min(bbox) >= 0.0
    ):
        return (
            bbox[0] * width / 1000.0,
            bbox[1] * height / 1000.0,
            bbox[2] * width / 1000.0,
            bbox[3] * height / 1000.0,
        )
    return bbox


def adapt_roomformer_predictions(predictions: Any, **kwargs: Any) -> list[NormalizedOpening]:
    """Functional convenience wrapper around :class:`RoomFormerSDTQAdapter`."""
    return RoomFormerSDTQAdapter().adapt(predictions, **kwargs)


__all__ = ["RoomFormerSDTQAdapter", "adapt_roomformer_predictions"]
