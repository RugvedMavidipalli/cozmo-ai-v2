"""Optional RGB door/window detection with lazy Grounding DINO and SAM2.

The base package has no vision-model dependency.  The adapters only import
``torch``, ``transformers`` or ``sam2`` when explicitly asked to load a local
model, and all model paths use local-only loading.  Tests can inject tiny
detector/refiner doubles and exercise the depth, wall association, and fusion
logic without a GPU or model weights.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .ingest import iter_frames
from .openings import NormalizedOpening, _bbox, fuse_openings, normalize_opening_kind
from .planes import HorizontalFrame, WallSegment


class OpeningBoxDetector(Protocol):
    def detect(self, image: np.ndarray, frame_index: int = 0) -> Sequence[Any]: ...


class OpeningMaskRefiner(Protocol):
    def refine(self, image: np.ndarray, detections: Sequence[Any]) -> Sequence[Any]: ...


class ModelUnavailable(RuntimeError):
    """Raised when an explicitly requested local model cannot be loaded."""


@dataclass(frozen=True)
class RGBOpeningConfig:
    """Runtime settings; paths are required for model-backed inference."""

    grounding_dino_model: str | None = None
    sam2_checkpoint: str | None = None
    sam2_config: str | None = None
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    min_detection_confidence: float = 0.35
    max_frames: int = 40
    min_depth_points: int = 8
    min_support_fraction: float = 0.02
    occlusion_fraction: float = 0.65


@dataclass(frozen=True)
class RGBOpeningBox:
    frame_index: int
    kind: str | None
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class RGBOpeningMask:
    mask: np.ndarray
    method: str = "sam2"
    confidence: float = 1.0


@dataclass
class RGBOpeningResult:
    openings: list[NormalizedOpening] = field(default_factory=list)
    rejected: list[NormalizedOpening | dict[str, Any]] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)


class GroundingDINOAdapter:
    """Grounding DINO adapter with no import or weight loading at import time.

    A ``runner`` can be injected for CPU tests.  The built-in path targets the
    Hugging Face Grounding DINO API and uses ``local_files_only=True`` so a
    typo or missing checkpoint can never trigger a download.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        prompt: str = "door. window.",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cuda",
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = str(model_path) if model_path is not None else None
        self.prompt = prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device
        self._runner = runner
        self._processor = None
        self._model = None

    def detect(self, image: np.ndarray, frame_index: int = 0) -> list[RGBOpeningBox]:
        raw = self._runner(image, frame_index) if self._runner is not None else self._infer(image)
        return _coerce_boxes(raw, image.shape[:2], frame_index)

    def _infer(self, image: np.ndarray) -> Any:
        if not self.model_path:
            raise ModelUnavailable(
                "Grounding DINO is disabled: provide --grounding-dino-model "
                "pointing to a local checkpoint"
            )
        if not Path(self.model_path).exists():
            raise ModelUnavailable(f"Grounding DINO checkpoint does not exist: {self.model_path}")
        try:
            import torch  # noqa: PLC0415 - optional, lazy dependency
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415
        except ImportError as exc:
            raise ModelUnavailable(
                "Grounding DINO requires optional torch/transformers dependencies"
            ) from exc
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_path, local_files_only=True
            ).to(self.device)
            self._model.eval()
        inputs = self._processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.shape[:2]],
        )[0]


class SAM2Adapter:
    """SAM2 mask refiner with local-only, lazy checkpoint loading."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        model_cfg: str | None = None,
        device: str = "cuda",
        predictor: Any | None = None,
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.model_cfg = model_cfg
        self.device = device
        self._predictor = predictor

    def refine(
        self, image: np.ndarray, detections: Sequence[RGBOpeningBox]
    ) -> list[RGBOpeningMask]:
        if self._predictor is None:
            self._predictor = self._load_predictor()
        self._predictor.set_image(image)
        masks: list[RGBOpeningMask] = []
        for detection in detections:
            result = self._predictor.predict(
                box=np.asarray(detection.bbox, dtype=np.float32), multimask_output=False
            )
            mask, score = _first_mask(result)
            masks.append(RGBOpeningMask(mask=mask, method="sam2", confidence=score))
        return masks

    def _load_predictor(self) -> Any:
        if not self.checkpoint or not self.model_cfg:
            raise ModelUnavailable(
                "SAM2 is disabled: provide --sam2-checkpoint and --sam2-config "
                "pointing to local files"
            )
        if not Path(self.checkpoint).exists():
            raise ModelUnavailable(f"SAM2 checkpoint does not exist: {self.checkpoint}")
        if not Path(self.model_cfg).exists():
            raise ModelUnavailable(f"SAM2 model config does not exist: {self.model_cfg}")
        try:
            from sam2.build_sam import build_sam2  # noqa: PLC0415 - optional, lazy dependency
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415
        except ImportError as exc:
            raise ModelUnavailable("SAM2 requires the optional sam2 package") from exc
        model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        return SAM2ImagePredictor(model)


def _first_mask(result: Any) -> tuple[np.ndarray, float]:
    if isinstance(result, tuple):
        masks = result[0]
        scores = result[1] if len(result) > 1 else [1.0]
    else:
        masks, scores = result, [1.0]
    mask_array = np.asarray(masks)
    if mask_array.ndim == 3:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError("SAM2 returned a mask with unexpected shape")
    score_array = np.asarray(scores).reshape(-1)
    score = float(score_array[0]) if len(score_array) else 1.0
    return mask_array.astype(bool), float(np.clip(score, 0.0, 1.0))


def _coerce_boxes(raw: Any, shape: tuple[int, int], frame_index: int) -> list[RGBOpeningBox]:
    if isinstance(raw, Mapping):
        if "boxes" in raw:
            boxes = raw["boxes"]
            labels = raw.get("labels", raw.get("phrases", [None] * len(boxes)))
            scores = raw.get("scores", [1.0] * len(boxes))
            raw = [
                {"bbox": box, "label": label, "confidence": score}
                for box, label, score in zip(boxes, labels, scores)
            ]
        elif "detections" in raw:
            raw = raw["detections"]
        else:
            raw = [raw]
    if isinstance(raw, np.ndarray):
        raw = list(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    result: list[RGBOpeningBox] = []
    for item in raw:
        if isinstance(item, RGBOpeningBox):
            candidate = item
        else:
            if isinstance(item, Mapping):
                box_value = item.get("bbox", item.get("box", item.get("bounding_box")))
                label = item.get("kind", item.get("label", item.get("phrase", item.get("category"))))
                score_value = item.get("confidence", item.get("score", 0.0))
            else:
                box_value = getattr(item, "bbox", getattr(item, "box", None))
                label = getattr(item, "kind", getattr(item, "label", None))
                score_value = getattr(item, "confidence", getattr(item, "score", 0.0))
            box = _bbox(box_value)
            if box is None:
                continue
            if max(box) <= 1.0 and min(box) >= 0.0:
                height, width = shape
                box = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
            try:
                score = float(np.clip(float(score_value), 0.0, 1.0))
            except (TypeError, ValueError):
                score = 0.0
            candidate = RGBOpeningBox(frame_index, normalize_opening_kind(label), box, score)
        result.append(candidate)
    return result


def select_opening_keyframes(bundle: Any, poses: np.ndarray | None = None, max_frames: int = 40) -> list[int]:
    """Use the existing viewpoint/sharpness selector for opening RGB frames."""
    from .keyframes import select_damage_keyframes  # lazy avoids an import cycle at module load

    return select_damage_keyframes(bundle, poses, max_frames=max_frames)


def detect_rgb_openings_with_diagnostics(
    bundle: Any,
    poses: np.ndarray,
    frame: HorizontalFrame,
    walls: Sequence[WallSegment],
    *,
    detector: OpeningBoxDetector,
    refiner: OpeningMaskRefiner,
    surface_grids: Mapping[int, Any] | None = None,
    floor_height: float = 0.0,
    ceiling_height: float | None = None,
    selected_frames: Sequence[int] | None = None,
    max_frames: int = 40,
    min_detection_confidence: float = 0.35,
    min_depth_points: int = 8,
    min_support_fraction: float = 0.02,
    occlusion_fraction: float = 0.65,
) -> RGBOpeningResult:
    """Detect, segment, back-project, associate, and fuse RGB evidence.

    A mask is only promoted to a metric opening when it has enough valid
    calibrated depth rays crossing a fitted wall plane and landing inside the
    finite 2D wall segment.  Furniture/unknown labels, box-only masks,
    shallow front-of-wall occluders, and unsupported candidates are returned
    as diagnostic rejections and never enter ``openings``.
    """
    selected = list(selected_frames) if selected_frames is not None else select_opening_keyframes(bundle, poses, max_frames)
    result = RGBOpeningResult(frame_indices=sorted(set(int(i) for i in selected)))
    grids = surface_grids or {}
    for capture_frame in iter_frames(bundle, selected, min_confidence=0, include_full_res=True):
        image = capture_frame.color_full if capture_frame.color_full is not None else capture_frame.color
        all_detections = _call_detector(detector, image, capture_frame.index)
        detections = []
        for detection in all_detections:
            if detection.kind not in {"door", "window"}:
                result.rejected.append(
                    {
                        "frame_index": detection.frame_index,
                        "state": "unmeasured",
                        "reason": "unknown/furniture label",
                        "provenance": ["rgb"],
                    }
                )
            elif detection.confidence < min_detection_confidence:
                result.rejected.append(
                    _rejected_detection(
                        detection, "confidence below threshold", state="unmeasured"
                    )
                )
            else:
                detections.append(detection)
        if not detections:
            continue
        masks = _call_refiner(refiner, image, detections)
        for detection, mask in zip(detections, masks):
            candidate, reason = _project_detection(
                detection,
                mask,
                capture_frame.depth,
                poses[capture_frame.index],
                bundle.intrinsics,
                frame,
                walls,
                grids,
                floor_height,
                ceiling_height,
                min_depth_points,
                min_support_fraction,
                occlusion_fraction,
            )
            if candidate is None:
                result.rejected.append(
                    _rejected_detection(detection, reason, state="occluded" if reason == "occluded" else "unmeasured")
                )
            else:
                result.openings.append(candidate)
    result.openings = fuse_openings(result.openings)
    return result


def detect_rgb_openings(*args: Any, **kwargs: Any) -> list[NormalizedOpening]:
    """Compatibility wrapper returning only accepted RGB opening evidence."""
    return detect_rgb_openings_with_diagnostics(*args, **kwargs).openings


def _call_detector(detector: OpeningBoxDetector | Callable[..., Any], image: np.ndarray, frame_index: int) -> list[RGBOpeningBox]:
    if hasattr(detector, "detect"):
        raw = detector.detect(image, frame_index)
    else:
        try:
            raw = detector(image, frame_index)  # type: ignore[misc]
        except TypeError:
            raw = detector(image)  # type: ignore[misc]
    return _coerce_boxes(raw, image.shape[:2], frame_index)


def _call_refiner(refiner: OpeningMaskRefiner | Callable[..., Any], image: np.ndarray, detections: Sequence[RGBOpeningBox]) -> list[RGBOpeningMask]:
    if hasattr(refiner, "refine"):
        raw = refiner.refine(image, detections)
    else:
        raw = refiner(image, detections)  # type: ignore[misc]
    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        raw = [raw]
    masks: list[RGBOpeningMask] = []
    for item in raw or []:
        if isinstance(item, RGBOpeningMask):
            masks.append(item)
        elif isinstance(item, Mapping):
            masks.append(RGBOpeningMask(np.asarray(item.get("mask"), dtype=bool), str(item.get("method", "sam2")), float(item.get("confidence", 1.0))))
        elif hasattr(item, "mask"):
            masks.append(
                RGBOpeningMask(
                    np.asarray(item.mask, dtype=bool),
                    str(getattr(item, "method", "sam2")),
                    float(getattr(item, "confidence", 1.0)),
                )
            )
        else:
            masks.append(RGBOpeningMask(np.asarray(item, dtype=bool)))
    return masks


def _project_detection(
    detection: RGBOpeningBox,
    mask: RGBOpeningMask,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    frame: HorizontalFrame,
    walls: Sequence[WallSegment],
    grids: Mapping[int, Any],
    floor_height: float,
    ceiling_height: float | None,
    min_depth_points: int,
    min_support_fraction: float,
    occlusion_fraction: float,
) -> tuple[NormalizedOpening | None, str]:
    if mask.method.lower() == "box":
        return None, "box-only mask"
    mask_array = np.asarray(mask.mask, dtype=bool)
    if mask_array.ndim != 2:
        return None, "mask has invalid shape"
    mask_height, mask_width = mask_array.shape
    box = detection.bbox
    if (mask_width, mask_height) != (depth.shape[1], depth.shape[0]):
        scale_x = depth.shape[1] / max(mask_width, 1)
        scale_y = depth.shape[0] / max(mask_height, 1)
        box = (
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y,
        )
    else:
        box = tuple(float(value) for value in box)
    if mask_array.shape != depth.shape:
        mask_array = cv2.resize(mask_array.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    box_mask = np.zeros_like(mask_array)
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = max(0, x0), min(depth.shape[1], x1)
    y0, y1 = max(0, y0), min(depth.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None, "invalid box"
    box_mask[y0:y1, x0:x1] = True
    mask_array &= box_mask
    valid = mask_array & np.isfinite(depth) & (depth > 0)
    total_mask = int(mask_array.sum())
    depth_count = int(valid.sum())
    if total_mask == 0 or depth_count < min_depth_points or depth_count / total_mask < min_support_fraction:
        return None, "insufficient valid depth"

    vs, us = np.nonzero(valid)
    z = depth[valid].astype(float)
    fx, fy, cx, cy = (float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2]))
    if min(abs(fx), abs(fy)) < 1e-9 or not np.isfinite([fx, fy, cx, cy]).all():
        return None, "invalid camera calibration"
    camera_points = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None, "invalid calibrated pose"
    world = camera_points @ pose[:3, :3].T + pose[:3, 3]
    camera_world = np.broadcast_to(pose[:3, 3], world.shape)
    camera_plan = frame.to_plan(camera_world)
    target_plan = frame.to_plan(world)
    best: tuple[WallSegment, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    strongest_occlusion = 0.0
    for wall in walls:
        camera_signed = camera_plan @ wall.normal - wall.offset
        target_signed = target_plan @ wall.normal - wall.offset
        target_along = (target_plan - wall.start) @ wall.direction
        target_height = frame.height(world) - floor_height
        front_of_wall = (
            (target_signed * camera_signed > 0)
            & (np.abs(target_signed) > 0.06)
            & (np.abs(target_signed) < 0.6)
            & (target_along >= 0)
            & (target_along <= wall.length)
            & (target_height >= 0)
            & (ceiling_height is None or target_height <= ceiling_height - floor_height)
        )
        strongest_occlusion = max(strongest_occlusion, float(front_of_wall.mean()))
        denominator = target_plan @ wall.normal - camera_plan @ wall.normal
        valid_denominator = np.abs(denominator) > 1e-8
        t = np.zeros(len(world), dtype=float)
        t[valid_denominator] = (wall.offset - camera_plan[valid_denominator] @ wall.normal) / denominator[valid_denominator]
        crossing = camera_plan + t[:, None] * (target_plan - camera_plan)
        u = (crossing - wall.start) @ wall.direction
        v = frame.height(camera_world + t[:, None] * (world - camera_world)) - floor_height
        in_segment = valid_denominator & (t > 0.02) & (t < 1.05) & (u >= 0) & (u <= wall.length)
        if ceiling_height is not None:
            in_segment &= (v >= 0) & (v <= ceiling_height - floor_height)
        else:
            in_segment &= v >= 0
        count = int(in_segment.sum())
        if count == 0:
            continue
        association = count / max(depth_count, 1)
        item = (
            wall,
            u[in_segment],
            v[in_segment],
            target_plan[in_segment] @ wall.normal - wall.offset,
            camera_plan[in_segment] @ wall.normal - wall.offset,
            association,
        )
        if best is None or count > len(best[1]):
            best = item
    if strongest_occlusion >= occlusion_fraction:
        return None, "occluded"
    if best is None or best[5] < min_support_fraction:
        return None, "no supported wall association"

    wall, u, v, signed_depth, camera_signed_depth, association = best
    grid = grids.get(wall.index)
    if grid is not None:
        cells = grid.to_cell(np.column_stack([u, v]))
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < grid.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < grid.shape[1])
        if inside.any():
            cell_u, cell_v = cells[inside, 0], cells[inside, 1]
            front = (np.abs(signed_depth[inside]) > 0.06) & (
                signed_depth[inside] * camera_signed_depth[inside] > 0
            )
            # The explicit depth-side test below is authoritative; a grid's
            # near counts are only used as a secondary furniture hint.
            near = (grid.near[cell_u, cell_v] > grid.hits[cell_u, cell_v]) & (
                signed_depth[inside] * camera_signed_depth[inside] > 0
            )
            if float(np.mean(front | near)) >= occlusion_fraction:
                return None, "occluded"

    u_range = _robust_bounds(u)
    v_range = _robust_bounds(v)
    if u_range is None or v_range is None or u_range[1] - u_range[0] < 0.20 or v_range[1] - v_range[0] < 0.20:
        return None, "unsupported opening bounds"
    sigma_u = max(0.02, wall.residual_rms + (u_range[1] - u_range[0]) / np.sqrt(max(depth_count, 1)))
    sigma_v = max(0.02, wall.residual_rms + (v_range[1] - v_range[0]) / np.sqrt(max(depth_count, 1)))
    confidence = float(np.clip(
        0.35 * detection.confidence
        + 0.20 * np.clip(mask.confidence, 0.0, 1.0)
        + 0.25 * min(1.0, depth_count / 100.0)
        + 0.20 * association,
        0.0,
        1.0,
    ))
    return NormalizedOpening(
        wall_index=wall.index,
        kind=detection.kind or "window",
        u_range=u_range,
        v_range=v_range,
        confidence=confidence,
        provenance=["rgb"],
        state="measured",
        uncertainty={
            "u_sigma_m": float(sigma_u),
            "v_sigma_m": float(sigma_v),
            "basis": "SAM2 mask, calibrated depth rays, wall-plane residual, and repeatable support",
        },
        wall_association_confidence=float(np.clip(association, 0.0, 1.0)),
        wall_distance_m=float(np.median(np.abs(signed_depth))),
        source_frames=[detection.frame_index],
        depth_support=depth_count,
        mask_method=mask.method,
        image_bbox=detection.bbox,
    ), ""


def _robust_bounds(values: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 1:
        return None
    if len(values) < 5:
        return float(values.min()), float(values.max())
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def _rejected_detection(detection: RGBOpeningBox, reason: str, state: str) -> NormalizedOpening:
    return NormalizedOpening(
        wall_index=None,
        kind=detection.kind or "window",
        u_range=None,
        v_range=None,
        confidence=detection.confidence,
        provenance=["rgb"],
        state=state,
        uncertainty={"basis": reason},
        source_frames=[detection.frame_index],
        image_bbox=detection.bbox,
        rejection_reason=reason,
    )


__all__ = [
    "GroundingDINOAdapter",
    "ModelUnavailable",
    "OpeningBoxDetector",
    "OpeningMaskRefiner",
    "RGBOpeningBox",
    "RGBOpeningConfig",
    "RGBOpeningMask",
    "RGBOpeningResult",
    "SAM2Adapter",
    "detect_rgb_openings",
    "detect_rgb_openings_with_diagnostics",
    "select_opening_keyframes",
]
