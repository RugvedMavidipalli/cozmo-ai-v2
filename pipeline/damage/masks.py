"""Refine detection boxes into pixel masks.

A bounding box overstates a stain's area -- a diagonal tide line fills maybe
half its box -- and area is what the scope's quantities are built from, so the
box has to become a mask before anything downstream can use it.

SAM 2 via Replicate is the accurate path; GrabCut is the local fallback, used
whenever the API is unavailable so the pipeline degrades instead of failing.
Which one ran is recorded on the mask, because it changes how much the area
should be trusted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RefinedMask:
    mask: np.ndarray  # bool, full frame resolution
    method: str  # "sam2" | "grabcut" | "box"
    area_fraction: float  # share of the box the mask fills

    @property
    def trusted(self) -> bool:
        return self.method != "box"


def refine(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    cache_dir: str | Path = "cache/masks",
    prefer_sam: bool = True,
) -> list[RefinedMask]:
    """Mask each box, preferring SAM 2 and falling back to GrabCut."""
    if not boxes:
        return []

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if prefer_sam and os.environ.get("REPLICATE_API_TOKEN"):
        try:
            return _sam2(image, boxes, cache_dir)
        except Exception:
            pass  # fall through to the local path rather than fail the run
    return [_grabcut(image, box) for box in boxes]


def _sam2(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    cache_dir: Path,
) -> list[RefinedMask]:
    import replicate

    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise RuntimeError("encode failed")
    encoded = base64.standard_b64encode(buffer.tobytes()).decode()

    digest = hashlib.sha256(
        (encoded + json.dumps(boxes, sort_keys=True)).encode()
    ).hexdigest()[:24]
    cache_path = cache_dir / f"{digest}.npz"
    if cache_path.exists():
        stored = np.load(cache_path)
        return [
            RefinedMask(stored[f"m{i}"].astype(bool), "sam2", float(stored[f"f{i}"]))
            for i in range(len(boxes))
        ]

    output = replicate.run(
        "meta/sam-2-large",
        input={
            "image": f"data:image/jpeg;base64,{encoded}",
            "box_prompts": json.dumps([list(map(float, box)) for box in boxes]),
        },
    )
    masks = _decode_masks(output, image.shape[:2], len(boxes))

    payload: dict[str, np.ndarray] = {}
    results: list[RefinedMask] = []
    for index, (mask, box) in enumerate(zip(masks, boxes)):
        fraction = _fill_fraction(mask, box)
        payload[f"m{index}"] = mask
        payload[f"f{index}"] = np.asarray(fraction)
        results.append(RefinedMask(mask, "sam2", fraction))
    np.savez_compressed(cache_path, **payload)
    return results


def _decode_masks(output, shape: tuple[int, int], count: int) -> list[np.ndarray]:
    """Normalise Replicate's several possible return shapes into bool arrays."""
    import urllib.request

    items = output if isinstance(output, list) else [output]
    masks: list[np.ndarray] = []
    for item in items[:count]:
        if hasattr(item, "read"):
            data = item.read()
        elif isinstance(item, str) and item.startswith("http"):
            with urllib.request.urlopen(item, timeout=60) as response:
                data = response.read()
        else:
            continue
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            continue
        if decoded.shape != shape:
            decoded = cv2.resize(decoded, (shape[1], shape[0]), cv2.INTER_NEAREST)
        masks.append(decoded > 127)
    if len(masks) != count:
        raise RuntimeError("SAM returned an unexpected number of masks")
    return masks


def _grabcut(
    image: np.ndarray, box: tuple[float, float, float, float]
) -> RefinedMask:
    """Local fallback: GrabCut seeded by the detection box.

    Boxes too small for GrabCut to have any background to learn from are kept
    as boxes, and marked as such so their area is treated as an upper bound.
    """
    height, width = image.shape[:2]
    x0 = int(np.clip(box[0], 0, width - 2))
    y0 = int(np.clip(box[1], 0, height - 2))
    x1 = int(np.clip(box[2], x0 + 1, width - 1))
    y1 = int(np.clip(box[3], y0 + 1, height - 1))

    if (x1 - x0) < 12 or (y1 - y0) < 12:
        mask = np.zeros((height, width), bool)
        mask[y0:y1, x0:x1] = True
        return RefinedMask(mask, "box", 1.0)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels = np.zeros((height, width), np.uint8)
    try:
        cv2.grabCut(
            bgr,
            labels,
            (x0, y0, x1 - x0, y1 - y0),
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        mask = np.zeros((height, width), bool)
        mask[y0:y1, x0:x1] = True
        return RefinedMask(mask, "box", 1.0)

    mask = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    if mask.sum() < 0.05 * (x1 - x0) * (y1 - y0):
        mask = np.zeros((height, width), bool)
        mask[y0:y1, x0:x1] = True
        return RefinedMask(mask, "box", 1.0)
    return RefinedMask(mask, "grabcut", _fill_fraction(mask, box))


def _fill_fraction(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    return float(mask.sum() / area)
