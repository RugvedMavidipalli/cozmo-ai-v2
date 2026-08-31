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
    """A detection box refined into a pixel mask.

    Attributes:
        mask: Bool array, `(height, width)` at full frame resolution.
        method: How the mask was produced: `"sam2"`, `"grabcut"`, or `"box"`.
        area_fraction: Share of the box's area the mask fills.
    """

    mask: np.ndarray
    method: str
    area_fraction: float

    @property
    def trusted(self) -> bool:
        """Whether this mask is a real segmentation rather than just the box."""
        return self.method != "box"


def refine(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    cache_dir: str | Path = "cache/masks",
    prefer_sam: bool = True,
) -> list[RefinedMask]:
    """Turns each rough detection box into a precise pixel mask of just the damaged area.

    A bounding box always includes some background around the actual
    damage, which would throw off any area measurement made from it. This
    tries SAM 2 first (a general-purpose segmentation model, called through
    the Replicate API), since it's usually much better at finding the true
    outline of the damage. That requires a configured API token and a
    network call, though, so if one isn't available -- or the call fails
    for any reason -- this quietly falls back to GrabCut instead, a classic
    image-processing algorithm that runs locally and does a rougher, but
    still useful, job.

    Args:
        image: RGB frame the boxes were detected in.
        boxes: Detection boxes as `(x0, y0, x1, y1)` in `image`'s pixel
            coordinates.
        cache_dir: Directory for the SAM 2 response cache.
        prefer_sam: When True, try SAM 2 first if a Replicate API token is
            configured; otherwise go straight to GrabCut.

    Returns:
        One `RefinedMask` per box, in the same order as `boxes`.
    """
    if not boxes:
        return []

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if prefer_sam and os.environ.get("REPLICATE_API_TOKEN"):
        try:
            return _sam2(image, boxes, cache_dir)
        except Exception:
            pass
    return [_grabcut(image, box) for box in boxes]


def _sam2(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    cache_dir: Path,
) -> list[RefinedMask]:
    """Segment every box in one frame via SAM 2 on Replicate, with caching.

    Args:
        image: RGB frame the boxes were detected in.
        boxes: Detection boxes as `(x0, y0, x1, y1)` in `image`'s pixel
            coordinates.
        cache_dir: Directory for the on-disk `.npz` response cache.

    Returns:
        One `RefinedMask` per box (method `"sam2"`), in the same order as
        `boxes`.

    Raises:
        RuntimeError: If encoding fails or the decoded mask count doesn't
            match the number of boxes sent.
    """
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
    """Normalise Replicate's several possible return shapes into bool arrays.

    Args:
        output: Whatever `replicate.run` returned -- a single item or a list
            of them, each either file-like or an HTTP URL string.
        shape: Expected `(height, width)` for each decoded mask.
        count: Expected number of masks.

    Returns:
        `count` bool mask arrays, each `(height, width)`.

    Raises:
        RuntimeError: If fewer than `count` items decoded successfully.
    """
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
    """Segments one box locally using OpenCV's GrabCut, seeded by the box itself.

    GrabCut starts by assuming everything inside the box is probably the
    damage and everything outside is background, then iterates to refine
    that guess into a tighter outline. It doesn't work well on very small
    or low-contrast boxes, though, so this falls back to just using the box
    itself as the mask whenever the box is too small to bother segmenting,
    GrabCut raises an error, or the result it produces looks implausibly
    small compared to the box it came from -- all signs that the
    segmentation isn't trustworthy.

    Args:
        image: RGB frame the box was detected in.
        box: Detection box as `(x0, y0, x1, y1)` in `image`'s pixel
            coordinates.

    Returns:
        A `RefinedMask`: `method="grabcut"` on success, or `method="box"`
        if GrabCut couldn't be trusted.
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
    """Share of the box's pixel area that `mask` actually covers.

    Args:
        mask: Bool mask, same resolution as the frame the box came from.
        box: The box the mask was derived from, as `(x0, y0, x1, y1)`.

    Returns:
        `mask.sum() / box_area`.
    """
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    return float(mask.sum() / area)
