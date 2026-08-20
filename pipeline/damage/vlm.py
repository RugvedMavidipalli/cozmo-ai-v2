"""Per-keyframe damage detection and classification with a vision model.

The model is asked for restoration-domain judgements (IICRC S500 water Category
and Class, fire soot/char/consumed, mold condition) rather than generic labels,
because those are what the downstream scope rules consume.

Two properties of this stage matter more than raw detection quality:

* It must argue against itself.  The evaluation scenes deliberately contain
  shadows that look like soot, dry surfaces that look wet, mirrors, and stains
  spanning two walls.  The schema therefore requires a `distractor_considered`
  field per detection -- a detection that cannot say what else it might be is
  usually the one that is wrong.
* It must be replayable.  Every response is cached by image content, so a rerun
  costs nothing and a live demo does not depend on the network.

Single-frame output is deliberately treated as a *hypothesis*.  Nothing here is
trusted until `damage.fusion` has confirmed it across views on a real surface.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "claude-opus-5"
PROMPT_VERSION = "v3"

DAMAGE_CLASSES = ("water", "fire", "mold", "none")

SYSTEM_PROMPT = """\
You are a restoration estimator with IICRC WRT and AMRT certification, \
inspecting still frames from a property walkthrough.

Report only damage you can actually see in THIS frame. You are being evaluated \
as much on what you correctly decline to flag as on what you find: the frames \
you are shown deliberately include lookalikes.

Before reporting any region, rule out the common false positives:
- SHADOW vs SOOT. Shadows have geometry consistent with the light sources and \
occluding objects in frame, sharp or uniformly soft edges, and no deposition \
gradient. Soot is a deposit: it is heaviest near its source, feathers along \
airflow paths, darkens the tops of surfaces and the area above heat sources, \
and does not track the room's lighting geometry.
- REFLECTION / GLARE vs STAIN. A reflection moves with the camera and shows \
scene content; a wet gloss shows a specular highlight with a diffuse dark \
halo. A mirror or glass surface reflects a whole scene and is not damage.
- WET-LOOK-BUT-DRY vs ACTUAL MOISTURE. Dark stone, polished tile, granite and \
some paints read as wet. Real water damage usually shows a tide line, edge \
staining, blistering, cupping, or a boundary that ignores material seams.
- NORMAL AGEING vs DAMAGE. Scuffs, patina, old paint, dirt and rust stains \
around fixtures are not restoration damage.
- MOLD vs DIRT / STAINING. Mold has three-dimensional texture, colonised \
growth patterns, and follows moisture paths, typically in corners, behind \
fixtures, and at wall-floor junctions.

Classification rules:
- water: assign IICRC S500 Category 1 (clean source), 2 (grey, significant \
contamination) or 3 (black, grossly contaminated: sewage, ground surface \
water, wind-driven rain) ONLY where the frame supports it; otherwise null. \
Class 1 to 4 describes evaporation load and is rarely inferable from a single \
frame -- return null unless the wet footprint and materials are clearly visible.
- fire: subtype soot, char, or consumed. Char means the substrate itself is \
burned; consumed means material is missing.
- mold: subtype surface_growth or colonized. Condition 1 (normal fungal \
ecology), 2 (settled spores / light surface), 3 (actual visible growth).

Give every region a bounding box in normalized 0-1000 coordinates as \
[x0, y0, x1, y1] with the origin at the top-left of the image.

Be conservative. An empty list is the correct answer for an undamaged frame."""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["regions", "frame_notes"],
    "properties": {
        "frame_notes": {
            "type": "string",
            "description": "Brief note on lighting, visible surfaces, and anything that made the frame hard to judge.",
        },
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "damage_class",
                    "subtype",
                    "bbox",
                    "confidence",
                    "surface_hint",
                    "evidence",
                    "distractor_considered",
                ],
                "properties": {
                    "damage_class": {"type": "string", "enum": list(DAMAGE_CLASSES)},
                    "subtype": {
                        "type": ["string", "null"],
                        "description": "soot|char|consumed for fire; surface_growth|colonized for mold; staining|saturation for water.",
                    },
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number"},
                    "surface_hint": {
                        "type": ["string", "null"],
                        "description": "wall|floor|ceiling|door|window|cabinet|other",
                    },
                    "water_category": {"type": ["integer", "null"]},
                    "water_class": {"type": ["integer", "null"]},
                    "mold_condition": {"type": ["integer", "null"]},
                    "severity": {"type": ["string", "null"]},
                    "evidence": {
                        "type": "string",
                        "description": "What in the image supports this call.",
                    },
                    "distractor_considered": {
                        "type": "string",
                        "description": "The most plausible benign explanation, and why it was rejected.",
                    },
                },
            },
        },
    },
}


@dataclass
class Detection:
    """One damage hypothesis in one frame."""

    frame_index: int
    damage_class: str
    subtype: str | None
    bbox: tuple[float, float, float, float]  # pixels, (x0, y0, x1, y1)
    confidence: float
    surface_hint: str | None = None
    water_category: int | None = None
    water_class: int | None = None
    mold_condition: int | None = None
    severity: str | None = None
    evidence: str = ""
    distractor_considered: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameAnalysis:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    notes: str = ""
    cached: bool = False
    error: str | None = None


class DamageAnalyzer:
    """Vision-model damage detection with on-disk response caching."""

    def __init__(
        self,
        cache_dir: str | Path = "cache/vlm",
        model: str = DEFAULT_MODEL,
        max_edge: int = 1024,
        effort: str = "high",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_edge = max_edge
        self.effort = effort
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _encode(self, image: np.ndarray) -> tuple[str, np.ndarray]:
        """Downscale for the API and return base64 JPEG plus the sent image."""
        height, width = image.shape[:2]
        scale = min(1.0, self.max_edge / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise RuntimeError("failed to encode frame")
        return base64.standard_b64encode(buffer.tobytes()).decode(), image

    def _cache_path(self, payload: str) -> Path:
        digest = hashlib.sha256(
            f"{PROMPT_VERSION}|{self.model}|{self.effort}|{payload}".encode()
        ).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def analyze_frame(self, frame_index: int, image: np.ndarray) -> FrameAnalysis:
        """Analyse one RGB frame, using the cache when the image is unchanged."""
        encoded, sent = self._encode(image)
        cache_path = self._cache_path(encoded)

        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return self._to_analysis(frame_index, raw, sent.shape, cached=True)

        if not self.available():
            return FrameAnalysis(
                frame_index=frame_index,
                error="ANTHROPIC_API_KEY not set and no cached response",
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Inspect this walkthrough frame for restoration "
                                    "damage. Rule out the lookalikes before reporting."
                                ),
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:  # network, rate limit, refusal handling below
            return FrameAnalysis(frame_index=frame_index, error=str(exc))

        if response.stop_reason == "refusal":
            return FrameAnalysis(frame_index=frame_index, error="model refusal")

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return FrameAnalysis(
                frame_index=frame_index, error="unparseable response"
            )

        cache_path.write_text(json.dumps(raw))
        return self._to_analysis(frame_index, raw, sent.shape, cached=False)

    def _to_analysis(
        self, frame_index: int, raw: dict, shape: tuple, cached: bool
    ) -> FrameAnalysis:
        """Convert normalised boxes back to pixels in the analysed image."""
        height, width = shape[:2]
        detections: list[Detection] = []
        for region in raw.get("regions", []):
            if region.get("damage_class") in (None, "none"):
                continue
            box = region.get("bbox") or [0, 0, 0, 0]
            x0, y0, x1, y1 = (float(v) for v in box)
            detections.append(
                Detection(
                    frame_index=frame_index,
                    damage_class=region["damage_class"],
                    subtype=region.get("subtype"),
                    bbox=(
                        x0 / 1000.0 * width,
                        y0 / 1000.0 * height,
                        x1 / 1000.0 * width,
                        y1 / 1000.0 * height,
                    ),
                    confidence=float(region.get("confidence", 0.5)),
                    surface_hint=region.get("surface_hint"),
                    water_category=region.get("water_category"),
                    water_class=region.get("water_class"),
                    mold_condition=region.get("mold_condition"),
                    severity=region.get("severity"),
                    evidence=region.get("evidence", ""),
                    distractor_considered=region.get("distractor_considered", ""),
                )
            )
        return FrameAnalysis(
            frame_index=frame_index,
            detections=detections,
            notes=raw.get("frame_notes", ""),
            cached=cached,
        )
