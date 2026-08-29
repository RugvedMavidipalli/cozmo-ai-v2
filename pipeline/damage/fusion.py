from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..occupancy import SurfaceGrid
from ..planes import HorizontalFrame, WallSegment
from .vlm import Detection

# Minimum |cos(angle)| between view ray and surface normal for a view's
# votes to count at all.
MIN_INCIDENCE_COSINE = 0.26


@dataclass
class SurfaceRef:
    """A named surface damage can be attached to.

    Attributes:
        key: Unique surface identifier used throughout fusion/scope output.
        kind: One of `"wall"`, `"floor"`, or `"ceiling"`.
        normal: 3D world-space unit normal of the surface's plane.
        offset: Plane offset such that a world point `x` lies on the surface
            when `normal . x == offset`.
        wall: The originating `WallSegment`, for `kind == "wall"` surfaces
            only. `None` for floor/ceiling.
        room_id: Room this surface belongs to, or `None` if unassigned.
    """

    key: str
    kind: str
    normal: np.ndarray
    offset: float
    wall: WallSegment | None = None
    room_id: int | None = None


@dataclass
class DamageRegion:
    """A fused damage region on one surface.

    Attributes:
        id: Unique region id, `"{surface_key}#{connected-component label}"`.
        surface_key: Key of the `SurfaceRef` this region was fused onto.
        room_id: Room the surface belongs to, or `None`.
        damage_class: `"water"`, `"fire"`, `"mold"`, or `"combined"` when two
            classes' votes were close enough to call it ambiguous.
        subtype: Consensus subtype across contributing detections, or `None`.
        area: Region area in m^2, on the surface's own UV grid.
        bounds_u: `(min, max)` extent along the surface's U axis, metres.
        bounds_v: `(min, max)` extent along the surface's V axis, metres.
        view_count: Max number of independent views that voted for any
            single cell in the region.
        confidence: Combined confidence blending view agreement, vote
            weight, and the model's own reported confidence.
        water_category: Consensus IICRC S500 Category, or `None`.
        water_class: Consensus IICRC S500 Class, or `None`.
        mold_condition: Consensus IICRC S520 Condition, or `None`.
        combined_classes: The two classes behind a `"combined"` call, or
            `None` for a single-class region.
        contributing_frames: Frame indices whose detections voted into this
            region.
        evidence: Up to 4 evidence strings quoted from contributing
            detections.
        mask_method: Comma-joined set of mask-refinement methods used by
            contributing detections, or `"unknown"`.
    """

    id: str
    surface_key: str
    room_id: int | None
    damage_class: str
    subtype: str | None
    area: float
    bounds_u: tuple[float, float]
    bounds_v: tuple[float, float]
    view_count: int
    confidence: float
    water_category: int | None = None
    water_class: int | None = None
    mold_condition: int | None = None
    combined_classes: tuple[str, str] | None = None
    contributing_frames: list[int] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    mask_method: str = "unknown"

    @property
    def height_extent(self) -> float:
        """Region's extent along the V axis (height for a wall), metres."""
        return self.bounds_v[1] - self.bounds_v[0]

    @property
    def width_extent(self) -> float:
        """Region's extent along the U axis (along-wall for a wall), metres."""
        return self.bounds_u[1] - self.bounds_u[0]

    def describe(self) -> str:
        """A one-line human summary of the region.

        Returns:
            Surface, area, damage label, and the vertical band affected.
        """
        band = (
            f"lower {self.bounds_v[1]:.2f} m affected"
            if self.bounds_v[0] < 0.15
            else f"between {self.bounds_v[0]:.2f} and {self.bounds_v[1]:.2f} m"
        )
        label = self.damage_class
        if self.damage_class == "combined" and self.combined_classes:
            label = f"{self.combined_classes[0]}+{self.combined_classes[1]} combined"
        elif self.damage_class == "water" and self.water_category:
            label = f"Cat {self.water_category} water"
        elif self.subtype:
            label = f"{self.damage_class} ({self.subtype})"
        return f"{self.surface_key}: {self.area:.2f} m2 {label}, {band}"


class DamageAccumulator:
    """Collects damage "votes" for one surface, across every frame that saw it.

    Each frame that spots damage on this surface contributes a weighted
    vote to whichever grid cells its mask covers. Building up votes this
    way, instead of trusting any single frame on its own, is what lets
    `extract_regions` later tell real damage -- seen consistently from
    several different views -- apart from a one-off false detection.
    """

    def __init__(self, surface: SurfaceRef, grid: SurfaceGrid):
        """Allocate empty per-cell vote buffers for `surface`.

        Args:
            surface: The surface this accumulator collects votes for.
            grid: The surface's UV grid, defining cell resolution and shape.
        """
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
        """Record one detection's votes into this surface's grid cells.

        Args:
            cells: `(N, 2)` array of `(column, row)` grid-cell indices this
                detection's mask projected into on this surface.
            detection: The source `Detection`.
            weight: This view's incidence weight, applied to every cell in
                `cells`.
            mask_method: The mask-refinement method used for this
                detection's mask.
        """
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
    """Every wall, plus the floor and ceiling, as attachable named planes.

    Args:
        walls: Fitted wall segments to expose as wall `SurfaceRef`s.
        frame: The gravity-aligned `HorizontalFrame`.
        floor_height: World height of the floor plane.
        ceiling_height: World height of the ceiling plane, or `None` if no
            ceiling was fitted.
        room_ids: Unused; accepted for a stable call signature.

    Returns:
        One `SurfaceRef` per wall (in `walls` order), followed by `"floor"`,
        then `"ceiling"` if `ceiling_height` was given.
    """
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

    Args:
        detection: The detection this mask belongs to; unused here, kept
            for a uniform per-detection call signature.
        mask: Bool mask from `masks.refine`, resized to match `depth` if
            its resolution differs.
        depth: Depth raster (metres, 0 = invalid) in the frame this
            detection came from.
        pose: 4x4 camera-to-world transform for this frame.
        intrinsics: 3x3 camera intrinsics matrix for `depth`'s resolution.
        scale: Unused; accepted for a stable call signature.

    Returns:
        `(world, rays)`: `(N, 3)` world-space points, and `(N, 3)` unit
        view-ray directions in world space, one per valid pixel.
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
    """Figures out which wall, floor, or ceiling each 3D point actually belongs to.

    A point only counts as belonging to a surface if it's both close to
    that surface's plane and within its bounds -- not just "closer to this
    surface than to any other." That distinction matters because some
    points genuinely don't belong to any real surface: a mirror or a pane
    of glass, for instance, can make the depth camera report a point
    sitting well behind the actual wall, with nothing at the true wall
    plane to match it. Snapping a point like that onto the nearest wall
    anyway would silently record damage in the wrong place, so a point
    that doesn't clearly match any surface is simply left out of the
    result rather than forced onto whichever one happens to be closest.

    Args:
        points: `(N, 3)` world-space points to assign.
        rays: `(N, 3)` unit view-ray directions for each point; unused
            here, accepted so this function's signature mirrors
            `project_detection`'s outputs.
        surfaces: Candidate surfaces (walls, floor, ceiling) to assign
            points to.
        frame: The gravity-aligned frame used to convert points to plan
            and height coordinates.
        tolerance: Max perpendicular distance, metres, from a surface's
            plane for a point to be considered on it.

    Returns:
        Map from surface index (into `surfaces`) to the array of `points`
        indices assigned to it. A surface with no assigned points has no
        key in the dict.
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
    """How squarely a view saw a surface, as a vote weight in [0, 1].

    A camera looking straight at a wall gets a much more reliable read on
    it than one glancing across it at a sharp angle, where a small change
    in the wall's real position can shift a lot in the image. Weighting
    each view's vote by how head-on it was keeps grazing views from
    carrying as much influence as a clean, direct view.

    Args:
        rays: `(N, 3)` unit view-ray directions for the points that hit
            this surface.
        normal: The surface's unit normal.

    Returns:
        The mean `|cos(angle)|` across all rays, zeroed outright below
        `MIN_INCIDENCE_COSINE`. `0.0` for no rays.
    """
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
    """Turns a surface's accumulated votes into a handful of distinct damage patches.

    A grid cell only counts as real damage once enough separate views
    agree on it (`min_views` and `min_weight`) -- a single frame's guess
    isn't trusted on its own, since one detection could easily be a false
    positive. The cells that do clear that bar are cleaned up a little
    (small gaps closed, stray lone cells removed) and then grouped into
    separate connected blobs. Each blob big enough to be worth reporting
    becomes one `DamageRegion`.

    Args:
        accumulator: Accumulated votes for one surface, from repeated
            `DamageAccumulator.add` calls across every frame that saw it.
        min_views: Minimum independent-view count a cell needs to be
            considered supported evidence, alongside `min_weight`.
        min_weight: Minimum accumulated incidence-weighted vote a cell
            needs, alongside `min_views`.
        min_area: Minimum connected-component area, m^2, for a region to
            be kept.
        close_radius: Radius, in grid cells, of the structuring element
            used for morphological closing before labeling.

    Returns:
        One `DamageRegion` per connected component of supported cells that
        clears `min_area`. Empty list if no cell is supported at all.
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
        combined_classes = None
        if damage_class == "combined":
            combined_classes = _component_classes(accumulator, cells)
            attributes = _combined_attributes(accumulator, combined_classes)
        else:
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
                combined_classes=combined_classes,
                contributing_frames=attributes["frames"],
                evidence=attributes["evidence"],
                mask_method=",".join(sorted(accumulator.mask_methods)) or "unknown",
            )
        )
    return regions


def _combined_attributes(
    accumulator: DamageAccumulator, classes: tuple[str, str] | None
) -> dict:
    """Merge consensus attributes across a "combined" region's two classes.

    Args:
        accumulator: The surface's vote accumulator.
        classes: The two classes to merge, from `_component_classes`; `None`
            falls back to the single class of the first recorded vote.

    Returns:
        A merged attribute dict, same shape as `_consensus_attributes`'s
        return: scalar fields prefer the primary class's value, falling
        back to the secondary's; frames are the union of both; evidence is
        concatenated and capped at 4 entries.
    """
    if classes is None:
        return _consensus_attributes(accumulator, accumulator.attributes[0][1].damage_class)
    primary = _consensus_attributes(accumulator, classes[0])
    secondary = _consensus_attributes(accumulator, classes[1])
    return {
        "subtype": primary.get("subtype") or secondary.get("subtype"),
        "water_category": primary.get("water_category") or secondary.get("water_category"),
        "water_class": primary.get("water_class") or secondary.get("water_class"),
        "mold_condition": primary.get("mold_condition") or secondary.get("mold_condition"),
        "confidence": float(np.mean([primary["confidence"], secondary["confidence"]])),
        "frames": sorted(set(primary["frames"]) | set(secondary["frames"])),
        "evidence": (primary["evidence"] + secondary["evidence"])[:4],
    }


# Ratio the second-ranked class's fused weight must reach, relative to the
# top class's, before the region is called "combined" instead.
COMBINED_CLASS_RATIO = 0.4


def _dominant_class(accumulator: DamageAccumulator, cells: np.ndarray) -> str:
    """Pick the winning damage class for a region, or call it "combined".

    Args:
        accumulator: The surface's vote accumulator.
        cells: Bool mask, same shape as the accumulator's grid, selecting
            the connected-component cells that make up this region.

    Returns:
        The single class with the highest fused weight in `cells`, unless a
        second class's weight comes within `COMBINED_CLASS_RATIO` of the
        top one, in which case `"combined"` is returned instead. Falls back
        to `"water"` if no class recorded any weight for these cells.
    """
    totals = _class_totals(accumulator, cells)
    if not totals:
        return "water"
    ranked = sorted(totals, key=totals.get, reverse=True)
    if len(ranked) >= 2 and totals[ranked[1]] >= COMBINED_CLASS_RATIO * totals[ranked[0]]:
        return "combined"
    return ranked[0]


def _class_totals(accumulator: DamageAccumulator, cells: np.ndarray) -> dict[str, float]:
    """Sum each damage class's fused weight within `cells`.

    Args:
        accumulator: The surface's vote accumulator.
        cells: Bool mask selecting the region's cells.

    Returns:
        Map from damage class name to its total weight in `cells`; a class
        never voted on this surface is absent, not zero.
    """
    return {
        name: float(buffer[cells].sum())
        for name, buffer in accumulator.class_weight.items()
    }


def _component_classes(
    accumulator: DamageAccumulator, cells: np.ndarray
) -> tuple[str, str] | None:
    """The two classes behind a "combined" call, ranked by fused weight.

    Args:
        accumulator: The surface's vote accumulator.
        cells: Bool mask selecting the region's cells.

    Returns:
        `(top_class, second_class)` by descending weight, or `None` if
        fewer than two classes have any weight in `cells`.
    """
    totals = _class_totals(accumulator, cells)
    ranked = sorted(totals, key=totals.get, reverse=True)
    if len(ranked) < 2:
        return None
    return ranked[0], ranked[1]


def _consensus_attributes(accumulator: DamageAccumulator, damage_class: str) -> dict:
    """Weighted vote over the categorical judgements of contributing frames.

    Args:
        accumulator: The surface's vote accumulator; `accumulator.attributes`
            holds every `(weight, Detection)` pair recorded via `add`,
            across the whole surface.
        damage_class: The class to restrict votes to.

    Returns:
        A dict of consensus values: `subtype`, `water_category`,
        `water_class`, `mold_condition` each picked by weighted plurality
        vote among detections of `damage_class`; `confidence` is the mean
        of those detections' confidences; `frames` is the sorted set of
        contributing frame indices; `evidence` is up to 4 evidence strings.
        Falls back to voting over every recorded detection on the surface
        if none matched `damage_class`.
    """
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

    Args:
        views: Max independent-view count backing the region.
        weight: Mean accumulated vote weight across the region's cells.
        model_confidence: The consensus attribute confidence from
            `_consensus_attributes`/`_combined_attributes`.

    Returns:
        A blended confidence in `[0, 1]`: 50% independent-view agreement
        (a saturating curve), 25% geometric weight, 25% the model's own
        reported confidence.
    """
    agreement = 1.0 - np.exp(-0.55 * max(views - 1, 0))
    geometry = float(np.clip(weight / 3.0, 0.0, 1.0))
    return float(np.clip(0.5 * agreement + 0.25 * geometry + 0.25 * model_confidence, 0, 1))
