"""Fuse per-frame damage detections onto named surfaces.

Detecting a stain in one frame is the easy part.  The hard parts, and what
this module exists for:

* **No double counting.** One stain is seen from dozens of viewpoints.  Summing
  per-frame areas would multiply it by the number of views.  Instead every
  observation votes into the *surface's* fixed UV grid, and the final area is
  the area of the cells that survived -- independent of how many times the
  operator walked past.

* **Rejecting reflections.** A mirror or glass wall shows damage that is not
  on that wall.  The depth behind a reflected pixel is the reflected scene's
  depth, so its back-projected point lands far from the mirror's plane and is
  discarded by the plane-agreement test.  The same test throws out detections
  that landed on furniture in front of the wall.

* **Splitting across surfaces.** A stain spanning a corner belongs to two
  walls.  Because assignment happens per pixel rather than per detection, each
  wall receives only its own portion.

Confidence per cell rises with *independent* agreement: how many separate
views, weighted by how squarely each viewed the surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..occupancy import SurfaceGrid
from ..planes import HorizontalFrame, WallSegment
from .vlm import Detection

# Grazing views smear a mask across the surface, so their votes are discounted
# by the cosine of the incidence angle and dropped entirely past this limit.
MIN_INCIDENCE_COSINE = 0.26  # ~75 degrees off the surface normal


@dataclass
class SurfaceRef:
    """A named surface damage can be attached to."""

    key: str
    kind: str  # "wall" | "floor" | "ceiling"
    normal: np.ndarray  # 3D world unit normal
    offset: float  # normal . x = offset
    wall: WallSegment | None = None
    room_id: int | None = None


@dataclass
class DamageRegion:
    """A fused damage region on one surface."""

    id: str
    surface_key: str
    room_id: int | None
    damage_class: str
    subtype: str | None
    area: float  # m^2 on the surface
    bounds_u: tuple[float, float]
    bounds_v: tuple[float, float]
    view_count: int
    confidence: float
    water_category: int | None = None
    water_class: int | None = None
    mold_condition: int | None = None
    contributing_frames: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    mask_method: str = "unknown"

    @property
    def height_extent(self) -> float:
        return self.bounds_v[1] - self.bounds_v[0]

    @property
    def width_extent(self) -> float:
        return self.bounds_u[1] - self.bounds_u[0]

    def describe(self) -> str:
        """The metric phrasing the assignment asks for."""
        band = (
            f"lower {self.bounds_v[1]:.2f} m affected"
            if self.bounds_v[0] < 0.15
            else f"between {self.bounds_v[0]:.2f} and {self.bounds_v[1]:.2f} m"
        )
        label = self.damage_class
        if self.damage_class == "water" and self.water_category:
            label = f"Cat {self.water_category} water"
        elif self.subtype:
            label = f"{self.damage_class} ({self.subtype})"
        return f"{self.surface_key}: {self.area:.2f} m2 {label}, {band}"


class DamageAccumulator:
    """Vote buffers for one surface, in that surface's UV grid."""

    def __init__(self, surface: SurfaceRef, grid: SurfaceGrid):
        self.surface = surface
        self.grid = grid
        shape = grid.shape
        self.weight = np.zeros(shape, np.float32)
        self.views = np.zeros(shape, np.int32)
        self.class_weight: dict[str, np.ndarray] = {}
        self.frames: dict[tuple[int, int], set[int]] = {}
        self.attributes: list[tuple[float, Detection]] = []
        self.mask_methods: set[str] = set()

    def add(
        self,
        cells: np.ndarray,
        detection: Detection,
        weight: float,
        mask_method: str,
    ) -> None:
        if len(cells) == 0:
            return
        columns, rows = cells[:, 0], cells[:, 1]
        np.add.at(self.weight, (columns, rows), weight)
        np.add.at(self.views, (columns, rows), 1)
        bucket = self.class_weight.setdefault(
            detection.damage_class, np.zeros(self.weight.shape, np.float32)
        )
        np.add.at(bucket, (columns, rows), weight * detection.confidence)
        self.attributes.append((weight, detection))
        self.mask_methods.add(mask_method)


def build_surface_refs(
    walls: list[WallSegment],
    frame: HorizontalFrame,
    floor_height: float,
    ceiling_height: float | None,
    room_ids: list[int] | None = None,
) -> list[SurfaceRef]:
    """Every wall, plus the floor and ceiling, as attachable named planes."""
    refs: list[SurfaceRef] = []
    for wall in walls:
        normal3d = wall.normal[0] * frame.right + wall.normal[1] * frame.forward
        refs.append(
            SurfaceRef(
                key=wall.name or f"wall_{wall.index}",
                kind="wall",
                normal=normal3d,
                offset=wall.offset,
                wall=wall,
                room_id=wall.room_id,
            )
        )
    refs.append(
        SurfaceRef(key="floor", kind="floor", normal=frame.up, offset=floor_height)
    )
    if ceiling_height is not None:
        refs.append(
            SurfaceRef(
                key="ceiling", kind="ceiling", normal=frame.up, offset=ceiling_height
            )
        )
    return refs


def project_detection(
    detection: Detection,
    mask: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    scale: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a frame-resolution mask to world points.

    Returns the world points and the unit ray directions that produced them;
    the rays are what let the caller measure incidence against a surface.
    """
    mask_height, mask_width = mask.shape
    depth_height, depth_width = depth.shape
    if (mask_height, mask_width) != (depth_height, depth_width):
        import cv2

        mask = (
            cv2.resize(
                mask.astype(np.uint8),
                (depth_width, depth_height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

    valid = mask & (depth > 0)
    if not valid.any():
        return np.empty((0, 3)), np.empty((0, 3))

    vs, us = np.nonzero(valid)
    z = depth[valid]
    camera = np.stack(
        [
            (us - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (vs - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=1,
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    rays = camera / np.maximum(np.linalg.norm(camera, axis=1, keepdims=True), 1e-9)
    return world, rays @ pose[:3, :3].T


def assign_to_surfaces(
    points: np.ndarray,
    rays: np.ndarray,
    surfaces: list[SurfaceRef],
    frame: HorizontalFrame,
    tolerance: float = 0.12,
) -> dict[int, np.ndarray]:
    """Split projected points among surfaces by plane proximity.

    A point must lie close to a surface *and* within its extent.  Points
    matching nothing are dropped -- that is the reflection and furniture
    rejection, and it is deliberate that they vanish silently rather than
    being forced onto the nearest plane.
    """
    if len(points) == 0:
        return {}

    plan = frame.to_plan(points)
    heights = frame.height(points)
    best_index = np.full(len(points), -1)
    best_distance = np.full(len(points), np.inf)

    for index, surface in enumerate(surfaces):
        if surface.kind == "wall":
            wall = surface.wall
            distance = np.abs(plan @ wall.normal - wall.offset)
            along = (plan - wall.start) @ wall.direction
            inside = (along > -0.1) & (along < wall.length + 0.1)
        else:
            distance = np.abs(heights - surface.offset)
            inside = np.ones(len(points), bool)

        candidate = inside & (distance < tolerance) & (distance < best_distance)
        best_index[candidate] = index
        best_distance[candidate] = distance[candidate]

    return {
        index: np.flatnonzero(best_index == index)
        for index in range(len(surfaces))
        if (best_index == index).any()
    }


def incidence_weight(rays: np.ndarray, normal: np.ndarray) -> float:
    """How squarely a view saw a surface, as a vote weight in [0, 1]."""
    if len(rays) == 0:
        return 0.0
    cosine = float(np.abs(rays @ normal).mean())
    return 0.0 if cosine < MIN_INCIDENCE_COSINE else cosine


def extract_regions(
    accumulator: DamageAccumulator,
    min_views: int = 2,
    min_weight: float = 0.6,
    min_area: float = 0.02,
    close_radius: int = 2,
) -> list[DamageRegion]:
    """Turn a surface's votes into discrete regions.

    `min_views` is the single most important parameter here: requiring
    agreement from independent viewpoints is what removes view-dependent
    artefacts -- glare, shadow, a reflection -- which appear in one view and
    vanish in the next, while a real stain persists.
    """
    grid = accumulator.grid
    surface = accumulator.surface
    supported = (accumulator.views >= min_views) & (accumulator.weight >= min_weight)
    if not supported.any():
        return []

    structure = np.ones((close_radius * 2 + 1,) * 2)
    supported = ndimage.binary_closing(supported, structure)
    supported = ndimage.binary_opening(supported, np.ones((2, 2)))

    labels, count = ndimage.label(supported)
    regions: list[DamageRegion] = []
    for label in range(1, count + 1):
        cells = labels == label
        area = grid.area_of(cells)
        if area < min_area:
            continue

        indices = np.argwhere(cells)
        u_lo, v_lo = indices.min(axis=0) * grid.resolution
        u_hi, v_hi = (indices.max(axis=0) + 1) * grid.resolution

        damage_class = _dominant_class(accumulator, cells)
        attributes = _consensus_attributes(accumulator, damage_class)
        views = int(accumulator.views[cells].max())
        mean_weight = float(accumulator.weight[cells].mean())

        regions.append(
            DamageRegion(
                id=f"{surface.key}#{label}",
                surface_key=surface.key,
                room_id=surface.room_id,
                damage_class=damage_class,
                subtype=attributes.get("subtype"),
                area=area,
                bounds_u=(float(u_lo), float(u_hi)),
                bounds_v=(float(v_lo), float(v_hi)),
                view_count=views,
                confidence=_confidence(views, mean_weight, attributes["confidence"]),
                water_category=attributes.get("water_category"),
                water_class=attributes.get("water_class"),
                mold_condition=attributes.get("mold_condition"),
                contributing_frames=attributes["frames"],
                evidence=attributes["evidence"],
                mask_method=",".join(sorted(accumulator.mask_methods)) or "unknown",
            )
        )
    return regions


def _dominant_class(accumulator: DamageAccumulator, cells: np.ndarray) -> str:
    totals = {
        name: float(buffer[cells].sum())
        for name, buffer in accumulator.class_weight.items()
    }
    if not totals:
        return "water"
    return max(totals, key=totals.get)


def _consensus_attributes(accumulator: DamageAccumulator, damage_class: str) -> dict:
    """Weighted vote over the categorical judgements of contributing frames."""
    relevant = [
        (weight, detection)
        for weight, detection in accumulator.attributes
        if detection.damage_class == damage_class
    ]
    if not relevant:
        relevant = accumulator.attributes

    def vote(attribute: str):
        tally: dict = {}
        for weight, detection in relevant:
            value = getattr(detection, attribute, None)
            if value is None:
                continue
            tally[value] = tally.get(value, 0.0) + weight * detection.confidence
        return max(tally, key=tally.get) if tally else None

    confidences = [d.confidence for _, d in relevant] or [0.5]
    return {
        "subtype": vote("subtype"),
        "water_category": vote("water_category"),
        "water_class": vote("water_class"),
        "mold_condition": vote("mold_condition"),
        "confidence": float(np.mean(confidences)),
        "frames": sorted({d.frame_index for _, d in relevant}),
        "evidence": [d.evidence for _, d in relevant if d.evidence][:4],
    }


def _confidence(views: int, weight: float, model_confidence: float) -> float:
    """Combine independent agreement with the model's own confidence.

    Agreement saturates: the fifth view of a stain adds far less than the
    second, so the view term is a saturating curve rather than a count.
    """
    agreement = 1.0 - np.exp(-0.55 * max(views - 1, 0))
    geometry = float(np.clip(weight / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * agreement + 0.25 * geometry + 0.25 * model_confidence, 0, 1))
