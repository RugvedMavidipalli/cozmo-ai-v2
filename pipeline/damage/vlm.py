from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import ingest

DEFAULT_MODEL = "claude-opus-5"
# Bump to invalidate the response cache after a prompt/schema change.
PROMPT_VERSION = "v4"

DAMAGE_CLASSES = ("water", "fire", "mold", "none")

FURNITURE_CLASS = "furniture"
FURNITURE_TYPES = (
    "couch", "sofa", "chair", "bed", "dresser", "nightstand", "table",
    "cabinet", "bookshelf", "desk",
)
FURNITURE_PROMPT_ADDENDUM = f"""

Diagnostic mode: in addition to the damage rules above, also report each \
individual piece of furniture you can name specifically -- one region per \
object, tightly boxed around that object alone (never one box spanning \
several items or most of the frame). Only use one of these exact type names, \
in the subtype field: {", ".join(FURNITURE_TYPES)}. If an object does not \
clearly match one of these names, skip it rather than guessing. Use \
damage_class "furniture", water_category/water_class/mold_condition null, \
and evidence describing what you see. This is NOT a damage judgement and \
does not relax the damage rules above -- it exists only to confirm you are \
resolving specific objects in the frame, tightly enough to be segmented."""

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

Give every region a bounding box as {x0, y0, x1, y1} in normalized 0-1000 \
coordinates, origin at the top-left of the image.

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
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x0", "y0", "x1", "y1"],
                        "properties": {
                            "x0": {"type": "number"},
                            "y0": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                        },
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
    """One damage hypothesis in one frame.

    Attributes:
        frame_index: Index of the frame this detection came from.
        damage_class: One of `DAMAGE_CLASSES`, or `FURNITURE_CLASS` when
            furniture diagnostics are enabled.
        subtype: Damage-class-specific subtype, or a furniture type name
            from `FURNITURE_TYPES`. `None` if the model gave no subtype.
        bbox: Pixel-space box `(x0, y0, x1, y1)` in the original frame's
            coordinates.
        confidence: Model-reported confidence in `[0, 1]`.
        surface_hint: Model's guess at the kind of surface this sits on, or
            `None`.
        water_category: IICRC S500 Category (1/2/3), or `None`.
        water_class: IICRC S500 Class (1-4), or `None`.
        mold_condition: IICRC S520 Condition (1/2/3), or `None`.
        severity: Free-form severity note from the model, or `None`.
        evidence: What in the image supports this call.
        distractor_considered: The most plausible benign explanation the
            model considered and rejected.
    """

    frame_index: int
    damage_class: str
    subtype: str | None
    bbox: tuple[float, float, float, float]
    confidence: float
    surface_hint: str | None = None
    water_category: int | None = None
    water_class: int | None = None
    mold_condition: int | None = None
    severity: str | None = None
    evidence: str = ""
    distractor_considered: str = ""

    def to_dict(self) -> dict:
        """Plain-dict form of this detection, for JSON export.

        Returns:
            A dict with exactly this dataclass's fields.
        """
        return asdict(self)


@dataclass
class FrameAnalysis:
    """The result of analysing one frame -- either detections, or an error.

    Attributes:
        frame_index: Index of the analysed frame.
        detections: Damage (and, if enabled, furniture) regions found in the
            frame. Empty when nothing was reported.
        notes: The model's free-form notes on the frame.
        cached: True when this result came from the on-disk cache rather
            than a live API call.
        error: `None` on success, otherwise a short reason the call could
            not produce detections.
    """

    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    notes: str = ""
    cached: bool = False
    error: str | None = None


class DamageAnalyzer:
    """Asks a vision-language model to look at one frame and report any damage it sees.

    Every call to the model costs time and money, so each response is
    cached on disk, keyed by the exact image and prompt that produced it.
    That means re-running the pipeline on the same capture doesn't pay to
    re-analyse a frame it has already seen.
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache/vlm",
        model: str = DEFAULT_MODEL,
        max_edge: int = 4096,
        effort: str = "high",
        include_furniture: bool = False,
    ):
        """Configure the analyzer and prepare its cache directory.

        Args:
            cache_dir: Directory for cached model responses. Created if it
                doesn't exist.
            model: Anthropic model id to call.
            max_edge: Max long-edge pixel size for the image sent to the
                model; larger images are downscaled.
            effort: Reasoning/output effort level passed to the API.
            include_furniture: When True, also ask the model to report
                individually-boxed furniture items.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_edge = max_edge
        self.effort = effort
        self.include_furniture = include_furniture
        self._system_prompt = SYSTEM_PROMPT + (
            FURNITURE_PROMPT_ADDENDUM if include_furniture else ""
        )
        self._schema = self._build_schema(include_furniture)
        self._client = None

    @staticmethod
    def _build_schema(include_furniture: bool) -> dict:
        """Return the JSON response schema, extended for furniture if asked.

        Args:
            include_furniture: When True, add `FURNITURE_CLASS` to the
                allowed `damage_class` values in a copy of `RESPONSE_SCHEMA`.

        Returns:
            The JSON schema dict to pass as the API's structured-output
            format.
        """
        if not include_furniture:
            return RESPONSE_SCHEMA
        schema = copy.deepcopy(RESPONSE_SCHEMA)
        classes = list(DAMAGE_CLASSES) + [FURNITURE_CLASS]
        schema["properties"]["regions"]["items"]["properties"]["damage_class"][
            "enum"
        ] = classes
        return schema

    @property
    def client(self):
        """Lazily-constructed `anthropic.Anthropic` client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> bool:
        """Whether a live API call is possible (an API key is configured)."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _encode(
        self, image: np.ndarray, rotation: int | None
    ) -> tuple[str, np.ndarray, float, tuple[int, int]]:
        """Prepares one frame to send to the model: rotates it upright, shrinks it if needed, and encodes it as a JPEG.

        A phone's camera sensor records frames sideways or upside-down
        relative to however the phone was actually held, so the image is
        rotated first to whatever way a person would naturally view the
        scene -- the model reads a right-side-up photo far more reliably.
        After that, an oversized image is downscaled, since sending more
        pixels than the model's limit doesn't help it see more detail, it
        just costs more.

        Args:
            image: RGB frame in its raw sensor orientation, at whatever
                resolution the caller wants analysed.
            rotation: A `cv2.ROTATE_*` code, or `None` for no rotation.

        Returns:
            `(encoded, sent, scale, original_shape)`:
            - `encoded`: base64-encoded JPEG bytes of the image actually
              sent to the model.
            - `sent`: the rotated-and-resized image array itself.
            - `scale`: the downscale factor applied after rotation.
            - `original_shape`: `(height, width)` of `image` before rotation.
        """
        original_shape = image.shape[:2]
        rotated = image if rotation is None else cv2.rotate(image, rotation)
        height, width = rotated.shape[:2]
        scale = min(1.0, self.max_edge / max(height, width))
        sent = rotated
        if scale < 1.0:
            sent = cv2.resize(
                rotated,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", cv2.cvtColor(sent, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not ok:
            raise RuntimeError("failed to encode frame")
        return (
            base64.standard_b64encode(buffer.tobytes()).decode(),
            sent,
            scale,
            original_shape,
        )

    def _cache_path(self, payload: str) -> Path:
        """Deterministic cache file path for one (image, prompt) pair.

        Args:
            payload: The base64-encoded image bytes, from `_encode`.

        Returns:
            Path to the (possibly not-yet-existing) cache JSON file.
        """
        prompt_digest = hashlib.sha256(self._system_prompt.encode()).hexdigest()[:12]
        digest = hashlib.sha256(
            f"{PROMPT_VERSION}|{self.model}|{self.effort}|"
            f"prompt={prompt_digest}|{payload}".encode()
        ).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def analyze_frame(
        self,
        frame_index: int,
        image: np.ndarray,
        rotation: int | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> FrameAnalysis:
        """Analyse one RGB frame, using the cache when the image is unchanged.

        Args:
            frame_index: Index to stamp onto the returned analysis and every
                detection in it.
            image: RGB frame to analyse, ideally at full native resolution.
            rotation: `cv2.ROTATE_*` code (or `None`) to view the frame
                human-naturally; forwarded to `_encode`.
            target_shape: `(height, width)` of the pixel grid detections
                should be rescaled into, if different from `image`'s own
                shape. `None` leaves boxes in `image`'s native resolution.

        Returns:
            A `FrameAnalysis` with either populated `detections`, or `error`
            set if the call could not produce a result.
        """
        encoded, sent, scale, original_shape = self._encode(image, rotation)
        cache_path = self._cache_path(encoded)

        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return self._to_analysis(
                frame_index, raw, sent.shape, scale, original_shape, rotation,
                target_shape, cached=True,
            )

        if not self.available():
            return FrameAnalysis(
                frame_index=frame_index,
                error="ANTHROPIC_API_KEY not set and no cached response",
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=self._system_prompt,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": self._schema},
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
        except Exception as exc:
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
        return self._to_analysis(
            frame_index, raw, sent.shape, scale, original_shape, rotation,
            target_shape, cached=False,
        )

    def _to_analysis(
        self,
        frame_index: int,
        raw: dict,
        shape: tuple,
        scale: float,
        original_shape: tuple[int, int],
        rotation: int | None,
        target_shape: tuple[int, int] | None,
        cached: bool,
    ) -> FrameAnalysis:
        """Turns the model's reported boxes back into real pixel coordinates.

        The model sees a rotated, resized version of the frame, and reports
        each box as a normalized 0-1000 coordinate on THAT image -- not on
        the original one. Getting back a box that actually lines up with the
        original frame takes two separate undo steps: first scale the box
        back up to the size the image was before it got shrunk, then rotate
        it back to the frame's original, unrotated orientation. If the
        caller also asked for boxes on a different pixel grid than the
        original frame (for example, to match a lower-resolution depth
        image), a final rescale is applied on top of that.

        Args:
            frame_index: Index to stamp onto the returned analysis and every
                detection in it.
            raw: Parsed JSON response (fresh or from cache), matching
                `RESPONSE_SCHEMA`.
            shape: Shape of the image actually sent to the model.
            scale: The downscale factor `_encode` applied after rotation.
            original_shape: `(height, width)` of the frame before rotation.
            rotation: The `cv2.ROTATE_*` code (or `None`) that was applied
                before sending.
            target_shape: `(height, width)` to rescale final boxes into, if
                different from `original_shape`; `None` to skip that step.
            cached: Whether `raw` came from the on-disk cache.

        Returns:
            A `FrameAnalysis` with one `Detection` per non-"none" region,
            each `bbox` in `target_shape` pixel coordinates (or
            `original_shape`'s, if `target_shape` was omitted).
        """
        sent_height, sent_width = shape[:2]
        original_height, original_width = original_shape
        rotated_width, rotated_height = (
            (original_height, original_width)
            if rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else (original_width, original_height)
        )
        if target_shape is not None and tuple(target_shape) != original_shape:
            target_height, target_width = target_shape
            rescale_x = target_width / original_width
            rescale_y = target_height / original_height
        else:
            rescale_x = rescale_y = 1.0
        detections: list[Detection] = []
        for region in raw.get("regions", []):
            if region.get("damage_class") in (None, "none"):
                continue
            box = region.get("bbox") or {}
            nx0, ny0, nx1, ny1 = (
                float(box.get(key, 0.0)) for key in ("x0", "y0", "x1", "y1")
            )
            sent_box = (
                nx0 / 1000.0 * sent_width / scale,
                ny0 / 1000.0 * sent_height / scale,
                nx1 / 1000.0 * sent_width / scale,
                ny1 / 1000.0 * sent_height / scale,
            )
            original_box = ingest.rotate_bbox(
                sent_box, rotated_width, rotated_height,
                ingest.inverse_rotation(rotation),
            )
            final_box = (
                original_box[0] * rescale_x,
                original_box[1] * rescale_y,
                original_box[2] * rescale_x,
                original_box[3] * rescale_y,
            )
            detections.append(
                Detection(
                    frame_index=frame_index,
                    damage_class=region["damage_class"],
                    subtype=region.get("subtype"),
                    bbox=final_box,
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
