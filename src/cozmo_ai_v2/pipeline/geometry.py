from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GravityEstimate:
    """The result of figuring out which way is "up" in a reconstruction, and
    where the floor and ceiling sit.

    A raw point cloud has no built-in sense of up, down, floor, or ceiling
    -- it's just a cloud of 3D coordinates. This dataclass holds the answer
    to that question for one reconstruction: which direction is up, and how
    far along that direction the floor and ceiling are.

    Attributes:
        up: A unit vector, in world coordinates, that points away from the
            floor and toward the ceiling.
        floor_height: How far along `up` the floor sits, measured as a
            signed distance from the world origin. It's a signed number
            rather than a plain height because it depends on where the
            world's origin happens to be.
        ceiling_height: The same kind of measurement for the ceiling, or
            `None` if no ceiling could be found -- for example, in an
            outdoor capture, or one where the ceiling was never in view.
        inlier_fraction: What share of all the points fall close to either
            the floor or the ceiling. A low value is a sign that the
            floor/ceiling heights found here might not be very trustworthy.
        floor_observed: Whether any supported floor candidate was observed;
            this is independent of the floor quality threshold.
        floor_quality_status: ``high_confidence`` or ``low_confidence`` for
            a supported candidate, with the residual/support evidence kept
            in the remaining floor fields.
    """

    up: np.ndarray
    floor_height: float
    ceiling_height: float | None
    inlier_fraction: float
    # These fields are deliberately separate from ``ceiling_height``.  A
    # caller must be able to distinguish an observed plane from a fallback
    # height inferred from the point-cloud extent.
    ceiling_observed: bool = True
    ceiling_confidence: float = 0.0
    floor_confidence: float = 0.0
    floor_observed: bool = False
    floor_quality_status: str = "unknown"
    floor_low_confidence: bool = False
    floor_support_fraction: float = 0.0
    floor_adaptive_residual_limit: float = 0.0
    floor_inlier_count: int = 0
    ceiling_inlier_count: int = 0
    floor_residual_rms: float = 0.0
    ceiling_residual_rms: float | None = None

    def __post_init__(self) -> None:
        up = np.asarray(self.up, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(up))
        self.up = (
            up / norm
            if up.shape == (3,) and np.isfinite(norm) and norm > 1e-9
            else np.array([0.0, 0.0, 1.0])
        )
        if not np.isfinite(self.floor_height):
            self.floor_height = 0.0
        if self.ceiling_height is not None and not np.isfinite(self.ceiling_height):
            self.ceiling_height = None
        if self.ceiling_height is None:
            self.ceiling_observed = False
            self.ceiling_confidence = 0.0
            self.ceiling_inlier_count = 0
            self.ceiling_residual_rms = None
        self.ceiling_confidence = float(np.clip(self.ceiling_confidence, 0.0, 1.0))
        self.floor_confidence = float(
            np.clip(self.floor_confidence, 0.0, 1.0)
            if np.isfinite(self.floor_confidence)
            else 0.0
        )
        self.floor_support_fraction = float(
            np.clip(self.floor_support_fraction, 0.0, 1.0)
            if np.isfinite(self.floor_support_fraction)
            else 0.0
        )
        self.floor_adaptive_residual_limit = float(
            max(0.0, self.floor_adaptive_residual_limit)
            if np.isfinite(self.floor_adaptive_residual_limit)
            else 0.0
        )
        self.floor_quality_status = str(self.floor_quality_status or "unknown")
        self.floor_low_confidence = bool(
            self.floor_low_confidence
            or self.floor_quality_status == "low_confidence"
        )
        self.inlier_fraction = float(
            np.clip(self.inlier_fraction, 0.0, 1.0)
            if np.isfinite(self.inlier_fraction)
            else 0.0
        )

    @property
    def room_height(self) -> float | None:
        """The floor-to-ceiling distance, in metres, or `None` if no ceiling
        height was found for this reconstruction."""
        if self.ceiling_height is None or not self.ceiling_observed:
            return None
        return self.ceiling_height - self.floor_height


def estimate_gravity(
    points: np.ndarray,
    hint: np.ndarray,
    normals: np.ndarray | None = None,
    cone_degrees: float = 25.0,
) -> GravityEstimate:
    """Works out which direction is "up" for this reconstruction, and finds
    the floor and ceiling heights along that direction.

    The starting point is a rough guess at up -- usually taken from the
    phone's own motion sensors -- which gets refined using the actual
    surfaces in the point cloud. Flat surfaces that are roughly horizontal,
    like floors, ceilings, and countertops, should all point in the same
    direction if the rough guess was close, so averaging them together
    gives a more accurate answer. Once up is known, the floor and ceiling
    heights are found by looking for the two most crowded "slices" of the
    room along that direction, since most points in a typical indoor scan
    end up belonging to one of those two surfaces.

    Args:
        points: The reconstructed 3D points, as an (N, 3) array in world
            coordinates.
        hint: A rough starting guess for the up direction, as a unit
            vector. This is typically `CaptureBundle.gravity_up`, which
            comes from the phone's accelerometer.
        normals: The surface normal at each point, as an (N, 3) array,
            used to refine `hint`. If this is `None` or empty, `hint` is
            used as-is without any refinement.
        cone_degrees: How far, in degrees, a surface normal is allowed to
            lean away from `hint` and still be counted as "roughly
            horizontal" when refining the up direction.

    Returns:
        A `GravityEstimate` holding the refined up direction along with the
        floor and ceiling heights found along it.
    """
    hint = np.asarray(hint, dtype=float).reshape(-1)
    hint_norm = float(np.linalg.norm(hint))
    up = (
        hint / hint_norm
        if hint.shape == (3,) and np.isfinite(hint_norm) and hint_norm > 1e-9
        else np.array([0.0, 0.0, 1.0])
    )
    normals_array = None
    if normals is not None:
        candidate_normals = np.asarray(normals, dtype=float)
        if candidate_normals.ndim == 2 and candidate_normals.shape[1] == 3:
            normals_array = candidate_normals
    if normals_array is not None and len(normals_array):
        up = _refine_up(normals_array, up, cone_degrees)

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    heights = points @ up
    normal_alignment = None
    if normals_array is not None and len(normals_array) == len(points):
        normal_unit = normals_array.copy()
        normal_unit /= np.maximum(
            np.linalg.norm(normal_unit, axis=1, keepdims=True), 1e-9
        )
        normal_alignment = np.abs(normal_unit @ up)

    fit = _fit_horizontal_planes(heights, normal_alignment=normal_alignment)
    return GravityEstimate(
        up=up,
        floor_height=fit.floor.height or 0.0,
        ceiling_height=fit.ceiling.height,
        inlier_fraction=fit.inlier_fraction,
        ceiling_observed=fit.ceiling.observed,
        ceiling_confidence=fit.ceiling.confidence,
        floor_confidence=fit.floor.confidence,
        floor_observed=fit.floor.observed,
        floor_quality_status=fit.floor.quality_status,
        floor_low_confidence=fit.floor.low_confidence,
        floor_support_fraction=fit.floor.support_fraction,
        floor_adaptive_residual_limit=fit.floor.adaptive_residual_limit,
        floor_inlier_count=fit.floor.inlier_count,
        ceiling_inlier_count=fit.ceiling.inlier_count,
        floor_residual_rms=fit.floor.residual_rms,
        ceiling_residual_rms=(
            fit.ceiling.residual_rms if fit.ceiling.observed else None
        ),
    )


def _refine_up(
    normals: np.ndarray, hint: np.ndarray, cone_degrees: float
) -> np.ndarray:
    """Improves a rough guess at "up" by averaging the surface normals that
    are roughly horizontal.

    A surface normal is a vector that points straight out from a surface --
    for a floor or ceiling, that vector points close to straight up or
    straight down. This keeps only the normals that point close enough to
    the rough guess (or its exact opposite, since a ceiling's normal points
    down while a floor's points up), flips the downward-pointing ones
    around so everything faces the same way, and averages what's left to
    get a cleaner estimate.

    Args:
        normals: The surface normal at each point, as an (N, 3) array.
        hint: The rough starting guess for the up direction, as a unit
            vector.
        cone_degrees: How far, in degrees, a normal is allowed to lean
            away from `hint` (in either direction) and still be kept.

    Returns:
        A refined unit vector pointing up, or `hint` unchanged if fewer
        than 100 normals survived the filter -- that's treated as too
        little evidence to trust a refined answer.
    """
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    alignment = unit @ hint
    threshold = np.cos(np.radians(cone_degrees))
    selected = unit[np.abs(alignment) > threshold]
    if len(selected) < 100:
        return hint
    folded = selected * np.sign(selected @ hint)[:, None]
    refined = folded.mean(axis=0)
    return refined / np.linalg.norm(refined)


@dataclass(frozen=True)
class PlaneFit:
    """A robust one-dimensional fit of a horizontal plane.

    ``observed`` means the point cloud contains a supported candidate; it is
    intentionally separate from ``quality_status``.  A floor just beyond
    the 40 mm high-confidence target is therefore still reported with its
    height, support, residual, adaptive limit, and ``low_confidence=True``.
    """

    height: float | None
    inlier_count: int
    residual_rms: float
    confidence: float
    observed: bool
    quality_status: str = "unknown"
    low_confidence: bool = False
    support_fraction: float = 0.0
    adaptive_residual_limit: float = 0.0

    @property
    def status(self) -> str:
        """Readable alias for the quality status in reports."""
        return self.quality_status


@dataclass(frozen=True)
class HorizontalPlaneFits:
    floor: PlaneFit
    ceiling: PlaneFit
    inlier_fraction: float


def _floor_and_ceiling(
    heights: np.ndarray, bin_size: float = 0.02
) -> tuple[float, float | None, float]:
    """Backward-compatible robust floor/ceiling estimate.

    Histogram bins are used only to seed a plane fit.  In particular, this
    function never uses a high percentile or the maximum observed height as
    a ceiling: a ceiling is returned only when a coherent plane cluster is
    supported by the data.
    """
    fit = _fit_horizontal_planes(heights, bin_size=bin_size)
    return fit.floor.height or 0.0, fit.ceiling.height, fit.inlier_fraction


def _fit_horizontal_planes(
    heights: np.ndarray,
    *,
    normal_alignment: np.ndarray | None = None,
    bin_size: float = 0.02,
    min_room_height: float = 1.8,
    plane_band: float = 0.08,
) -> HorizontalPlaneFits:
    """Fit floor and ceiling offsets from robust horizontal-plane evidence.

    Height histograms provide deterministic candidate seeds.  Each seed is
    refit by a median/MAD pass so a few bad LiDAR returns cannot move the
    plane.  A candidate ceiling must have enough coherent support; a wall or
    a single high outlier therefore cannot become a fabricated ceiling.

    When normals are available, candidates preferentially use points whose
    normals are close to vertical.  The fallback remains conservative for
    captures with incomplete normals rather than inventing a ceiling.
    """
    raw = np.asarray(heights, dtype=float).reshape(-1)
    finite = np.isfinite(raw)
    values = raw[finite]
    if not len(values):
        empty = PlaneFit(0.0, 0, 0.0, 0.0, False)
        return HorizontalPlaneFits(empty, PlaneFit(None, 0, 0.0, 0.0, False), 0.0)

    alignment = None
    if normal_alignment is not None:
        raw_alignment = np.asarray(normal_alignment, dtype=float).reshape(-1)
        if len(raw_alignment) == len(raw):
            alignment = raw_alignment[finite]
            alignment = np.where(np.isfinite(alignment), alignment, 0.0)
            alignment = np.clip(alignment, 0.0, 1.0)

    low, high = float(values.min()), float(values.max())
    spread = max(high - low, bin_size)
    bins = max(16, int(np.ceil(spread / bin_size)))
    counts, edges = np.histogram(values, bins=bins, range=(low, high + 1e-9))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak_indices = _height_peaks(counts)
    if not peak_indices:
        # The floor is a useful conservative fallback even with a sparse
        # scan.  This path is never used to infer a ceiling.
        floor_seed = float(np.quantile(values, 0.02))
        peak_indices = [int(np.argmin(np.abs(centers - floor_seed)))]

    candidates: list[tuple[float, int, float, float, float]] = []
    for index in peak_indices:
        seed = float(centers[index])
        near = np.abs(values - seed) <= plane_band
        if alignment is not None:
            horizontal = alignment >= np.cos(np.radians(30.0))
            horizontal_count = int((near & horizontal).sum())
            if horizontal_count >= max(8, int(0.5 * near.sum())):
                near &= horizontal
        if not near.any():
            continue

        fitted, _, residual = _robust_location(values[near], plane_band)
        residuals = np.abs(values - fitted)
        inlier_mask = residuals <= max(plane_band, 3.0 * max(residual, 0.005))
        if alignment is not None:
            horizontal = alignment >= np.cos(np.radians(30.0))
            if int((inlier_mask & horizontal).sum()) >= max(8, int(0.5 * inlier_mask.sum())):
                inlier_mask &= horizontal
        count = int(inlier_mask.sum())
        rms = (
            float(np.sqrt(np.mean((values[inlier_mask] - fitted) ** 2)))
            if count
            else residual
        )
        signed_deviations = values[inlier_mask] - fitted
        candidate_mad = (
            float(np.median(np.abs(signed_deviations - np.median(signed_deviations))))
            if count
            else 0.0
        )
        robust_sigma = max(0.005, 1.4826 * candidate_mad)
        wide = np.abs(values - fitted) <= min(0.4, max(4 * plane_band, 0.2))
        local_background = max(int(wide.sum()) - count, 1)
        prominence = count / local_background
        normal_quality = 1.0
        if alignment is not None and count:
            normal_quality = float(np.mean(alignment[inlier_mask]))
        candidates.append((fitted, count, rms, prominence * normal_quality, robust_sigma))

    if not candidates:
        floor = PlaneFit(float(np.quantile(values, 0.02)), 0, 0.0, 0.0, False)
        return HorizontalPlaneFits(floor, PlaneFit(None, 0, 0.0, 0.0, False), 0.0)

    floor_candidate = min(candidates, key=lambda item: (item[0], -item[1], item[2]))
    floor_minimum_support = max(20, int(np.ceil(0.03 * len(values))))
    floor = _plane_fit_from_candidate(
        floor_candidate,
        minimum_support=floor_minimum_support,
        require_quality=True,
        total_count=len(values),
    )

    ceiling_candidates = [
        candidate
        for candidate in candidates
        if candidate[0] - floor_candidate[0] >= min_room_height
    ]
    minimum_support = max(20, int(np.ceil(0.03 * len(values))))
    ceiling_candidate = None
    if ceiling_candidates:
        candidate = sorted(
            ceiling_candidates,
            key=lambda item: (-item[1], item[2], -item[3], -item[0]),
        )[0]
        if (
            candidate[1] >= minimum_support
            and candidate[2] <= 0.04
            and candidate[3] >= 0.35
        ):
            ceiling_candidate = candidate

    ceiling = (
        _plane_fit_from_candidate(
            ceiling_candidate,
            minimum_support=minimum_support,
            total_count=len(values),
        )
        if ceiling_candidate is not None
        else PlaneFit(None, 0, 0.0, 0.0, False)
    )
    near_floor = np.abs(values - floor_candidate[0]) <= 0.1
    near_ceiling = (
        np.abs(values - ceiling_candidate[0]) <= 0.1
        if ceiling_candidate is not None
        else np.zeros(len(values), dtype=bool)
    )
    return HorizontalPlaneFits(
        floor,
        ceiling,
        float((near_floor | near_ceiling).mean()),
    )


def _height_peaks(counts: np.ndarray) -> list[int]:
    """Return local histogram maxima in deterministic index order."""
    counts = np.asarray(counts, dtype=int)
    if not len(counts) or counts.max() <= 0:
        return []
    threshold = max(5, int(np.ceil(0.02 * counts.sum())))
    peaks: list[int] = []
    for index, count in enumerate(counts):
        left = counts[index - 1] if index else -1
        right = counts[index + 1] if index + 1 < len(counts) else -1
        if count >= threshold and count >= left and count >= right:
            if count > left or count > right:
                peaks.append(index)
    return peaks


def _robust_location(values: np.ndarray, band: float) -> tuple[float, int, float]:
    """Return a median/MAD location and residual for one plane cluster."""
    values = np.asarray(values, dtype=float)
    centre = float(np.median(values))
    for _ in range(3):
        deviations = np.abs(values - centre)
        mad = float(np.median(deviations))
        cutoff = min(band, max(0.01, 3.0 * 1.4826 * mad))
        keep = deviations <= cutoff
        if not keep.any():
            break
        centre = float(np.median(values[keep]))
    residuals = values[np.abs(values - centre) <= band] - centre
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    return centre, int(len(residuals)), rms


def _plane_fit_from_candidate(
    candidate: tuple[float, int, float, float, float],
    minimum_support: int,
    require_quality: bool = False,
    total_count: int | None = None,
) -> PlaneFit:
    height, count, residual, quality = candidate[:4]
    robust_sigma = candidate[4] if len(candidate) > 4 else max(0.005, residual)
    support_score = min(1.0, count / max(minimum_support, 1))
    residual_score = float(np.exp(-residual / 0.04))
    # Quality is allowed to influence confidence but cannot make an
    # otherwise well-supported plane disappear from the output.
    quality_score = min(1.0, max(0.0, quality))
    base_confidence = float(
        np.clip(0.45 * support_score + 0.35 * residual_score + 0.20 * quality_score, 0.0, 1.0)
    )
    # A 40 mm RMS floor is a useful high-confidence target, not a cliff at
    # which a physically supported plane disappears.  The adaptive bound is
    # driven by robust residual scale and is capped to avoid accepting a
    # broad wall/outlier population as a floor.  Support still affects both
    # the confidence and whether the candidate is considered reportable.
    adaptive_limit = float(np.clip(max(0.04, 2.5 * robust_sigma + 0.01), 0.04, 0.12))
    has_support = count >= max(int(minimum_support), 1)
    has_quality = quality >= 0.35
    adaptive_ok = residual <= adaptive_limit
    strict_ok = residual <= 0.04
    observed = bool(count > 0)
    if strict_ok and has_support and has_quality:
        quality_status = "high_confidence"
        low_confidence = False
        confidence = base_confidence
    elif (not require_quality or (has_support and has_quality)) and adaptive_ok:
        quality_status = "low_confidence"
        low_confidence = True
        # Keep the quality signal useful while clearly below the strict
        # target; callers can make their own acceptance decision.
        confidence = float(0.65 * base_confidence)
    else:
        quality_status = "low_confidence"
        low_confidence = True
        confidence = float(0.35 * base_confidence)
    support_fraction = float(
        np.clip(
            count / max(int(total_count or minimum_support), 1),
            0.0,
            1.0,
        )
    )
    return PlaneFit(
        float(height),
        int(count),
        float(residual),
        confidence,
        observed,
        quality_status,
        low_confidence,
        support_fraction,
        adaptive_limit,
    )
