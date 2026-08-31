"""Metric measurements derived from the structured TLS plane model.

This module is intentionally independent of the raster room grid.  It
consumes finite wall extents, plane intersections, and bounded room faces
provided by the geometry stages.  Stage 9 does not close a graph, polygonize
wall lines, or infer a face from raster corners.  The small adapters at the
top of the module accept both Phase 1 ``WallSegment`` objects and the
structured 3D plane records used by the next geometry stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .geometry import GravityEstimate
from .planes import HorizontalFrame, TLSPlane, TLSPlaneModel


_MISSING = object()
_Z_SCORES = {0.68: 1.0, 0.80: 1.282, 0.90: 1.645, 0.95: 1.96}


def _field(value: object, *names: str, default: Any = _MISSING) -> Any:
    """Read a field from a mapping or an object without coupling to Stage 6."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    return None


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _normalise_status(value: object) -> str:
    status = str(value or "unknown").strip().lower().replace(" ", "_")
    return status or "unknown"


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        status = _normalise_status(value)
        if status in {"false", "0", "no", "none", "unknown", "unobserved", "missing"}:
            return False
        if status in {"true", "1", "yes", "observed", "measured", "validated"}:
            return True
    if value is None:
        return default
    return bool(value)


def _as_items(value: object | None) -> list[object]:
    """Treat one plane record and a collection of records uniformly."""
    if value is None:
        return []
    if isinstance(value, Mapping) or hasattr(value, "normal"):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    return list(value)


@dataclass
class MeasurementContext:
    """Global evidence settings applied to every metric in one result."""

    coverage: float = 0.90
    calibration_status: str = "uncalibrated"
    calibration_scale: float = 1.0
    pose_provenance: str = "unknown"
    depth_provenance: str = "tls_lidar"
    has_depth: bool = True
    default_wall_thickness_m: float = 0.15
    pose_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        self.coverage = float(np.clip(self.coverage, 0.5, 0.999))
        self.calibration_status = _normalise_status(self.calibration_status)
        self.calibration_scale = max(_finite_float(self.calibration_scale, 1.0), 0.01)
        self.pose_provenance = _normalise_status(self.pose_provenance)
        self.depth_provenance = _normalise_status(self.depth_provenance)
        if not self.has_depth and self.depth_provenance in {"tls_lidar", "lidar"}:
            self.depth_provenance = "estimated_depth"
        self.default_wall_thickness_m = max(
            _finite_float(self.default_wall_thickness_m, 0.15), 0.0
        )
        self.pose_uncertainty_m = max(_finite_float(self.pose_uncertainty_m), 0.0)

    @classmethod
    def from_uncertainty(
        cls,
        uncertainty: object | None,
        *,
        has_depth: bool = True,
        pose_provenance: str = "unknown",
        default_wall_thickness_m: float = 0.15,
        pose_uncertainty_m: float = 0.0,
    ) -> "MeasurementContext":
        """Bridge the existing interval model into this measurement layer."""
        if uncertainty is None:
            return cls(
                has_depth=has_depth,
                pose_provenance=pose_provenance,
                depth_provenance="tls_lidar" if has_depth else "estimated_depth",
                default_wall_thickness_m=default_wall_thickness_m,
                pose_uncertainty_m=pose_uncertainty_m,
            )
        return cls(
            coverage=_finite_float(getattr(uncertainty, "coverage", 0.90), 0.90),
            calibration_status=(
                "calibrated" if bool(getattr(uncertainty, "calibrated", False))
                else "uncalibrated"
            ),
            calibration_scale=_finite_float(getattr(uncertainty, "scale", 1.0), 1.0),
            pose_provenance=pose_provenance,
            depth_provenance="tls_lidar" if has_depth else "estimated_depth",
            has_depth=has_depth,
            default_wall_thickness_m=default_wall_thickness_m,
            pose_uncertainty_m=pose_uncertainty_m,
        )


@dataclass
class MeasurementEvidence:
    """Evidence attached to one measurement for auditability."""

    tls_residual_rms_m: float = 0.0
    support_points: int = 0
    support_density_per_m2: float | None = None
    pose_provenance: str = "unknown"
    depth_provenance: str = "unknown"
    calibration_status: str = "uncalibrated"
    pose_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        self.tls_residual_rms_m = max(_finite_float(self.tls_residual_rms_m), 0.0)
        self.support_points = max(int(self.support_points), 0)
        if self.support_density_per_m2 is not None:
            density = _finite_float(self.support_density_per_m2, -1.0)
            self.support_density_per_m2 = density if density >= 0 else None
        self.pose_provenance = _normalise_status(self.pose_provenance)
        self.depth_provenance = _normalise_status(self.depth_provenance)
        self.calibration_status = _normalise_status(self.calibration_status)
        self.pose_uncertainty_m = max(_finite_float(self.pose_uncertainty_m), 0.0)

    def to_dict(self) -> dict:
        return {
            "tls_residual_rms_m": round(self.tls_residual_rms_m, 6),
            "support_points": self.support_points,
            "support_density_per_m2": (
                round(self.support_density_per_m2, 3)
                if self.support_density_per_m2 is not None
                else None
            ),
            "pose_provenance": self.pose_provenance,
            "depth_provenance": self.depth_provenance,
            "calibration_status": self.calibration_status,
            "pose_uncertainty_m": round(self.pose_uncertainty_m, 6),
        }


@dataclass
class Measurement:
    """A metric value with tolerance, evidence confidence, and review flags."""

    value: float | None
    tolerance: float | None
    confidence: float
    status: str = "measured"
    basis: str = ""
    flags: list[str] = field(default_factory=list)
    evidence: MeasurementEvidence = field(default_factory=MeasurementEvidence)
    coverage: float = 0.90

    def __post_init__(self) -> None:
        if self.value is not None:
            value = _finite_float(self.value, np.nan)
            self.value = value if np.isfinite(value) else None
        if self.tolerance is not None:
            tolerance = _finite_float(self.tolerance, np.nan)
            self.tolerance = max(tolerance, 0.0) if np.isfinite(tolerance) else None
        self.confidence = float(np.clip(_finite_float(self.confidence), 0.0, 1.0))
        self.coverage = float(np.clip(_finite_float(self.coverage, 0.90), 0.5, 0.999))
        self.status = _normalise_status(self.status)
        self.flags = sorted(set(str(flag) for flag in self.flags if flag))
        if self.value is None:
            self.status = "unmeasured" if self.status == "measured" else self.status
            self.tolerance = None
            self.confidence = 0.0

    @classmethod
    def unmeasured(
        cls,
        basis: str,
        *,
        flags: Iterable[str] = (),
        evidence: MeasurementEvidence | None = None,
    ) -> "Measurement":
        return cls(
            value=None,
            tolerance=None,
            confidence=0.0,
            status="unmeasured",
            basis=basis,
            flags=[*flags, "manual_review"],
            evidence=evidence or MeasurementEvidence(),
        )

    @property
    def half_width(self) -> float | None:
        """Compatibility spelling used by the existing result and benchmark."""
        return self.tolerance

    def to_dict(self) -> dict:
        value = round(self.value, 6) if self.value is not None else None
        tolerance = round(self.tolerance, 6) if self.tolerance is not None else None
        return {
            "value": value,
            "tolerance": tolerance,
            "half_width": tolerance,
            "ci_low": round(self.value - self.tolerance, 6)
            if self.value is not None and self.tolerance is not None
            else None,
            "ci_high": round(self.value + self.tolerance, 6)
            if self.value is not None and self.tolerance is not None
            else None,
            "coverage": self.coverage,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "basis": self.basis,
            "flags": self.flags,
            "evidence": self.evidence.to_dict(),
        }

@dataclass
class WallMeasurement:
    wall_id: str | int
    length: Measurement
    inlier_vertical_extent: Measurement
    thickness: Measurement
    geometry_source: str
    opposing_face_id: str | int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.wall_id,
            "length": self.length.to_dict(),
            "inlier_vertical_extent": self.inlier_vertical_extent.to_dict(),
            "thickness": self.thickness.to_dict(),
            "wall_thickness": self.thickness.to_dict(),
            "geometry_source": self.geometry_source,
            "opposing_face_id": self.opposing_face_id,
        }


@dataclass
class HeightStatistics:
    minimum: Measurement
    mean: Measurement
    maximum: Measurement
    primary: str = "mean"

    def to_dict(self) -> dict:
        return {
            "min": self.minimum.to_dict(),
            "mean": self.mean.to_dict(),
            "max": self.maximum.to_dict(),
            "primary": self.primary,
            "convention": "perpendicular to the observed floor plane",
        }


@dataclass
class RoomMeasurement:
    room_id: str | int
    interior_face_area: Measurement
    wall_centerline_area: Measurement
    outer_footprint_area: Measurement
    floor_to_ceiling_height: HeightStatistics
    wall_thicknesses: dict[str, Measurement]
    boundary: list[list[float]]
    area_convention: dict[str, Any]

    @property
    def primary_area(self) -> Measurement:
        return self.interior_face_area

    def to_dict(self) -> dict:
        return {
            "room_id": self.room_id,
            "interior_face_area": self.interior_face_area.to_dict(),
            "wall_centerline_area": self.wall_centerline_area.to_dict(),
            "outer_footprint_area": self.outer_footprint_area.to_dict(),
            "floor_to_ceiling_height": self.floor_to_ceiling_height.to_dict(),
            "wall_thicknesses": {
                str(key): value.to_dict() for key, value in self.wall_thicknesses.items()
            },
            "boundary": self.boundary,
            "area_convention": self.area_convention,
        }


@dataclass
class SceneMeasurements:
    walls: dict[str | int, WallMeasurement]
    rooms: dict[str | int, RoomMeasurement]
    scale_validation: "ScaleValidation | None" = None


@dataclass
class _WallRecord:
    id: str | int
    normal: np.ndarray
    offset: float
    start: np.ndarray | None
    end: np.ndarray | None
    source: object
    evidence: MeasurementEvidence
    observed: bool = True
    inferred_fraction: float = 0.0
    tags: list[str] = field(default_factory=list)
    geometry_source: str = "unmeasured"
    intersection_points: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        if self.start is not None and self.end is not None:
            delta = self.end - self.start
            norm = float(np.linalg.norm(delta))
            if norm > 1e-9:
                return delta / norm
        return np.array([-self.normal[1], self.normal[0]])

    def along(self, point: np.ndarray) -> float:
        origin = self.start if self.start is not None else self.normal * self.offset
        return float((np.asarray(point) - origin) @ self.direction)


def _evidence_for(
    value: object,
    context: MeasurementContext,
    *,
    length_m: float | None = None,
    area_m2: float | None = None,
) -> MeasurementEvidence:
    provenance = _field(value, "provenance", default={})
    if not isinstance(provenance, Mapping):
        provenance = {}
    residual = _field(value, "residual_rms", "residual_rms_m", "rms", default=0.0)
    support = _field(
        value, "inlier_count", "support_points", "support_count", "point_count", default=0
    )
    density = _field(value, "support_density", "support_density_per_m2", "density")
    if density is None and length_m and area_m2:
        density = _finite_float(support) / max(area_m2, 1e-6)
    pose = _field(
        value, "pose_provenance", "pose_source", default=provenance.get("pose", context.pose_provenance)
    )
    depth = _field(
        value, "depth_provenance", "depth_source", default=provenance.get("depth", context.depth_provenance)
    )
    calibration = _field(
        value,
        "calibration_status",
        "calibration",
        default=provenance.get("calibration", context.calibration_status),
    )
    if isinstance(calibration, bool):
        calibration = "calibrated" if calibration else "uncalibrated"
    pose_uncertainty = _field(
        value, "pose_uncertainty_m", "drift_m", default=provenance.get("pose_uncertainty_m", context.pose_uncertainty_m)
    )
    return MeasurementEvidence(
        tls_residual_rms_m=_finite_float(residual),
        support_points=max(int(_finite_float(support)), 0),
        support_density_per_m2=(
            _finite_float(density, -1.0) if density is not None else None
        ),
        pose_provenance=pose,
        depth_provenance=depth,
        calibration_status=calibration,
        pose_uncertainty_m=_finite_float(pose_uncertainty),
    )


def _evidence_score(evidence: MeasurementEvidence) -> float:
    support_score = min(1.0, np.sqrt(max(evidence.support_points, 0)) / np.sqrt(500.0))
    if evidence.support_density_per_m2 is None:
        density_score = 0.65
    else:
        density_score = min(1.0, evidence.support_density_per_m2 / 80.0)
    residual_score = exp(-evidence.tls_residual_rms_m / 0.045)

    pose = evidence.pose_provenance
    pose_score = 0.95 if any(x in pose for x in ("refined", "optimized", "slam")) else 0.70
    if pose in {"unknown", "none", ""}:
        pose_score = 0.55
    depth = evidence.depth_provenance
    depth_score = 0.95 if any(x in depth for x in ("lidar", "tls", "laser", "measured")) else 0.50
    if depth in {"unknown", "none", ""}:
        depth_score = 0.60
    calibration_score = 1.0 if evidence.calibration_status in {"calibrated", "validated"} else 0.78
    pose_uncertainty_score = exp(-evidence.pose_uncertainty_m / 0.05)
    return float(
        np.clip(
            0.28 * support_score
            + 0.18 * density_score
            + 0.24 * residual_score
            + 0.09 * pose_score
            + 0.03 * pose_uncertainty_score
            + 0.10 * depth_score
            + 0.08 * calibration_score,
            0.0,
            1.0,
        )
    )


def _measurement_flags(
    evidence: MeasurementEvidence,
    confidence: float,
    context: MeasurementContext,
    *,
    inferred: bool = False,
    assumed: bool = False,
) -> list[str]:
    flags: list[str] = []
    if evidence.support_points < 100:
        flags.append("weak_support")
    if evidence.support_density_per_m2 is not None and evidence.support_density_per_m2 < 20:
        flags.append("weak_support_density")
    if evidence.tls_residual_rms_m > 0.04:
        flags.append("high_tls_residual")
    if evidence.pose_uncertainty_m > 0.05:
        flags.append("high_pose_uncertainty")
    if evidence.pose_provenance in {"unknown", "raw", "unrefined", "none"}:
        flags.append("pose_uncertain")
    if evidence.depth_provenance in {"unknown", "estimated_depth", "video_only", "monocular"}:
        flags.append("depth_uncertain")
    if evidence.calibration_status not in {"calibrated", "validated"}:
        flags.append("uncalibrated")
    if inferred:
        flags.append("inferred_geometry")
    if assumed:
        flags.append("assumed_wall_thickness")
    if confidence < 0.60:
        flags.append("manual_review")
    return sorted(set(flags))


def _make_measurement(
    value: float,
    evidence: MeasurementEvidence,
    context: MeasurementContext,
    *,
    sigma_m: float | None = None,
    basis: str,
    status: str = "measured",
    inferred: bool = False,
    assumed: bool = False,
    extra_flags: Iterable[str] = (),
) -> Measurement:
    confidence = _evidence_score(evidence)
    residual_sigma = evidence.tls_residual_rms_m / max(np.sqrt(evidence.support_points), 1.0)
    sigma = max(_finite_float(sigma_m, residual_sigma), 0.002)
    sigma = float(np.hypot(sigma, evidence.pose_uncertainty_m))
    if evidence.support_density_per_m2 is not None and evidence.support_density_per_m2 < 80:
        sigma *= 1.0 + min(1.5, (80.0 - evidence.support_density_per_m2) / 80.0)
    if evidence.pose_provenance in {"unknown", "raw", "unrefined", "none"}:
        sigma *= 1.25
    if evidence.depth_provenance in {"unknown", "estimated_depth", "video_only", "monocular"}:
        sigma *= 3.0
    sigma *= context.calibration_scale
    if evidence.calibration_status not in {"calibrated", "validated"}:
        sigma *= 1.10
    z = _Z_SCORES.get(round(context.coverage, 2), 1.645)
    measurement = Measurement(
        value=float(value),
        tolerance=float(z * sigma),
        confidence=confidence,
        status=status,
        basis=basis,
        flags=[
            *_measurement_flags(
                evidence, confidence, context, inferred=inferred, assumed=assumed
            ),
            *extra_flags,
        ],
        evidence=evidence,
        coverage=context.coverage,
    )
    return measurement


def _unmeasured(
    basis: str,
    evidence: MeasurementEvidence,
    *flags: str,
    context: MeasurementContext | None = None,
) -> Measurement:
    return Measurement(
        value=None,
        tolerance=None,
        confidence=0.0,
        status="unmeasured",
        basis=basis,
        flags=[*flags, "manual_review"],
        evidence=evidence,
        coverage=context.coverage if context is not None else 0.90,
    )


def _coerce_wall_records(
    walls: Sequence[object],
    frame: HorizontalFrame | None,
    context: MeasurementContext,
    explicit_intersections: Mapping[str | int, Sequence[object]] | None = None,
) -> list[_WallRecord]:
    records: list[_WallRecord] = []
    for index, source in enumerate(walls):
        wall_id = _field(source, "id", "index", "plane_id", default=index)
        normal_raw = np.asarray(_field(source, "normal", "unit_normal"), dtype=float).reshape(-1)
        start_raw = _field(source, "start", "endpoint_a")
        end_raw = _field(source, "end", "endpoint_b")
        inlier_points = _field(source, "inlier_points", "inliers")
        intersection_points = list((explicit_intersections or {}).get(wall_id, []))
        source_intersections = _field(
            source,
            "intersection_points",
            "intersections",
            "corner_intersections",
            "endpoint_intersections",
            default=[],
        )
        if isinstance(source_intersections, Mapping):
            if _field(source_intersections, "point", "position", "xyz", default=None) is not None:
                source_intersections = [source_intersections]
            else:
                source_intersections = source_intersections.get(wall_id, [])
        intersection_points.extend(list(source_intersections or []))

        if normal_raw.shape == (3,):
            if frame is None:
                continue
            normal = np.array([normal_raw @ frame.right, normal_raw @ frame.forward])
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-9:
                continue
            normal /= normal_norm
            offset = _finite_float(_field(source, "offset", "d", "distance"))
            # A near-vertical TLS plane has negligible up component.  For a
            # slightly tilted fitted wall, place its plan line at the floor
            # height rather than silently treating it as a horizontal plane.
            up_component = normal_raw @ frame.up
            if abs(up_component) > 1e-4:
                floor_height = _finite_float(_field(source, "floor_height"), 0.0)
                offset = (offset - up_component * floor_height) / normal_norm
        elif normal_raw.shape == (2,):
            normal_norm = float(np.linalg.norm(normal_raw))
            if normal_norm <= 1e-9:
                continue
            normal = normal_raw / normal_norm
            offset = _finite_float(_field(source, "offset", "d", "distance")) / normal_norm
        else:
            continue

        def project_endpoint(endpoint: object) -> np.ndarray | None:
            if endpoint is None:
                return None
            array = np.asarray(endpoint, dtype=float).reshape(-1)
            if array.shape == (2,):
                return array
            if array.shape == (3,) and frame is not None:
                return np.asarray(frame.to_plan(array), dtype=float).reshape(2)
            return None

        start = project_endpoint(start_raw)
        end = project_endpoint(end_raw)
        projected_intersections = [
            point
            for point in (project_endpoint(value) for value in intersection_points)
            if point is not None
        ]
        geometry_source = "unmeasured"
        if len(projected_intersections) >= 2:
            # The closure stage owns the intersection graph.  Stage 9 only
            # consumes the finite points it supplied; it never intersects
            # the fitted wall lines itself.
            tangent = np.array([-normal[1], normal[0]])
            projected_intersections.sort(key=lambda point: float(point @ tangent))
            start = projected_intersections[0]
            end = projected_intersections[-1]
            geometry_source = "supplied_plane_intersections"
        elif start is not None and end is not None and np.linalg.norm(end - start) > 1e-8:
            # Phase 1 already produced finite plane-segment endpoints.  Keep
            # this compatibility path until the structured extent contract
            # lands, but mark it as endpoint geometry for uncertainty/review.
            geometry_source = "phase1_plane_endpoints"
        else:
            start = None
            end = None
        length = (
            float(np.linalg.norm(end - start))
            if start is not None and end is not None
            else 0.0
        )
        height_range = _field(source, "height_range", "vertical_extent")
        vertical_extent = 0.0
        if height_range is not None:
            values = np.asarray(height_range, dtype=float).reshape(-1)
            if len(values) >= 2 and np.isfinite(values[:2]).all():
                vertical_extent = abs(float(values[1] - values[0]))
        if not vertical_extent and inlier_points is not None and frame is not None:
            points = np.asarray(inlier_points, dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points):
                vertical_extent = float(np.ptp(points @ frame.up))
        area = max(length * vertical_extent, 1e-6)
        evidence = _evidence_for(source, context, length_m=length, area_m2=area)
        # Phase 1 WallSegment stores the vertical extent but not an explicit
        # density.  Derive it from observed inliers so confidence remains
        # evidence-based rather than a generic constant.
        if evidence.support_density_per_m2 is None and length > 1e-6 and vertical_extent > 1e-6:
            evidence.support_density_per_m2 = evidence.support_points / area
        records.append(
            _WallRecord(
                id=wall_id,
                normal=normal,
                offset=offset,
                start=start,
                end=end,
                source=source,
                evidence=evidence,
                observed=_as_bool(_field(source, "observed", "is_observed", default=True), True),
                inferred_fraction=float(
                    np.clip(_finite_float(_field(source, "inferred_fraction", default=0.0)), 0.0, 1.0)
                ),
                tags=list(_field(source, "tags", default=[]) or []),
                geometry_source=geometry_source,
                intersection_points=projected_intersections,
            )
        )
    return records


def _room_boundary(
    room: object,
    frame: HorizontalFrame | None,
    *,
    allow_phase1_polygon: bool = False,
) -> tuple[np.ndarray | None, str]:
    """Read a bounded face supplied by the geometry/closure stages.

    A room centroid, raster polygon, or scalar area is not enough evidence to
    create a Stage 9 area.  The legacy ``polygon`` field is opt-in because
    Phase 1 used it for both graph and raster fallbacks; a structured caller
    should provide ``boundary`` (or an equivalent bounded-face field).
    """
    boundary_document = _field(
        room,
        "boundary",
        "boundary_polygon",
        "room_boundary",
        "interior_boundary",
        "bounded_face",
        "face",
        "geometry_boundary",
        "geometry",
        "boundary_vertices",
        "face_vertices",
        "vertices",
        default=None,
    )
    document_status = (
        _field(boundary_document, "status", "geometry_status", default=None)
        if isinstance(boundary_document, Mapping)
        else None
    )
    status = _normalise_status(
        _field(
            room,
            "geometry_status",
            "boundary_status",
            "polygon_status",
            "status",
            default=document_status or "unknown",
        )
    )
    if status in {"unmeasured", "invalid", "rejected", "unknown", "low_confidence"}:
        # An absent status is represented as unknown.  A supplied boundary is
        # still allowed for Phase 1 compatibility below, unless the producer
        # explicitly says the geometry is unavailable.
        explicit_status = _field(
            room,
            "geometry_status",
            "boundary_status",
            "polygon_status",
            "status",
            default=None,
        )
        if explicit_status is not None:
            return None, "unmeasured"

    source = str(
        _field(
            room,
            "geometry_source",
            "boundary_source",
            "polygon_source",
            default=(
                _field(
                    boundary_document,
                    "geometry_source",
                    "boundary_source",
                    "source",
                    default="",
                )
                if isinstance(boundary_document, Mapping)
                else ""
            ),
        )
        or ""
    ).strip().lower().replace(" ", "_")
    if any(token in source for token in ("raster", "occupancy", "grid", "flood", "watershed")):
        return None, "raster_rejected"

    confidence_raw = _field(
        room,
        "geometry_confidence",
        "boundary_confidence",
        "polygon_confidence",
        "confidence",
        default=(
            _field(boundary_document, "confidence", "geometry_confidence", default=None)
            if isinstance(boundary_document, Mapping)
            else None
        ),
    )
    if isinstance(confidence_raw, Mapping):
        confidence_raw = _field(confidence_raw, "confidence", "value", default=None)
    if confidence_raw is not None and _finite_float(confidence_raw, 0.0) < 0.50:
        return None, "low_confidence"

    boundary = boundary_document
    legacy_polygon = False
    if boundary is None and allow_phase1_polygon:
        boundary = _field(room, "polygon", default=None)
        legacy_polygon = boundary is not None
    if isinstance(boundary, Mapping):
        boundary = _field(
            boundary,
            "vertices",
            "points",
            "coordinates",
            "polygon",
            "boundary",
            default=None,
        )
    if isinstance(boundary, (list, tuple)) and boundary and all(
        isinstance(item, Mapping) for item in boundary
    ):
        points = [
            _field(item, "point", "position", "xyz", "vertex", default=None)
            for item in boundary
        ]
        if all(point is not None for point in points):
            boundary = points
    if boundary is None:
        return None, "unmeasured"
    try:
        values = np.asarray(boundary, dtype=float)
    except (TypeError, ValueError):
        return None, "invalid"
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] not in (2, 3):
        return None, "invalid"
    if values.shape[1] == 3:
        if frame is None:
            return None, "invalid"
        values = frame.to_plan(values)
    if not np.isfinite(values).all() or _area(values) <= 1e-8:
        return None, "invalid"
    # Keep the source explicit for provenance, while retaining a stable name
    # for Phase 1 callers whose polygon predates the structured contract.
    return values, source or (
        "phase1_room_polygon_compatibility" if legacy_polygon else "supplied_bounded_face"
    )


def _signed_area(polygon: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )


def _area(polygon: np.ndarray) -> float:
    return abs(_signed_area(polygon))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Small deterministic point-in-face test for selecting TLS samples."""
    x, y = float(point[0]), float(point[1])
    inside = False
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_at_y = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_at_y:
                inside = not inside
    return inside


def _line_for_edge(a: np.ndarray, b: np.ndarray, outward_distance: float) -> tuple[np.ndarray, float]:
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return np.array([1.0, 0.0]), 0.0
    direction /= length
    outward = np.array([direction[1], -direction[0]])
    return outward, float(outward @ a + outward_distance)


def _offset_polygon(polygon: np.ndarray, distances: Sequence[float]) -> np.ndarray:
    """Offset every face line and intersect adjacent offset lines.

    This is a mitered plane construction, not a raster dilation.  It keeps
    the centerline and outside-footprint conventions explicit at corners.
    """
    polygon = np.asarray(polygon, dtype=float)
    if _signed_area(polygon) < 0:
        polygon = polygon[::-1]
        distances = list(distances)[::-1]
    lines = [
        _line_for_edge(polygon[i], polygon[(i + 1) % len(polygon)], float(distances[i]))
        for i in range(len(polygon))
    ]
    vertices: list[np.ndarray] = []
    for index in range(len(lines)):
        n_prev, d_prev = lines[index - 1]
        n_next, d_next = lines[index]
        matrix = np.vstack([n_prev, n_next])
        if abs(float(np.linalg.det(matrix))) <= 1e-8:
            vertices.append(polygon[index] + (n_prev + n_next) * 0.5 * float(distances[index]))
            continue
        vertices.append(np.linalg.solve(matrix, np.array([d_prev, d_next])))
    return np.asarray(vertices)


def _matching_wall_for_edge(
    a: np.ndarray, b: np.ndarray, records: Sequence[_WallRecord]
) -> _WallRecord | None:
    midpoint = 0.5 * (a + b)
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length <= 1e-9:
        return None
    direction = edge / length
    candidates = []
    for record in records:
        parallel = abs(float(direction @ record.direction))
        distance = abs(float(midpoint @ record.normal - record.offset))
        if parallel > 0.96 and distance < 0.15:
            candidates.append((distance, -record.length, record))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def _thickness_pairs(
    records: Sequence[_WallRecord],
    *,
    min_thickness_m: float = 0.025,
    max_thickness_m: float = 0.80,
) -> dict[str | int, tuple[str | int, float, MeasurementEvidence]]:
    pairs: dict[str | int, tuple[str | int, float, MeasurementEvidence]] = {}
    for index, first in enumerate(records):
        if (
            not first.observed
            or first.inferred_fraction > 0.15
            or "inferred" in first.tags
            or first.length <= 0
        ):
            continue
        for second in records[index + 1 :]:
            if (
                not second.observed
                or second.inferred_fraction > 0.15
                or "inferred" in second.tags
            ):
                continue
            if abs(float(first.normal @ second.normal)) < 0.98:
                continue
            separation = abs(first.offset - second.offset)
            if not min_thickness_m <= separation <= max_thickness_m:
                continue
            first_range = sorted([first.along(second.start), first.along(second.end)])
            overlap = min(first.length, first_range[1]) - max(0.0, first_range[0])
            if overlap < max(0.25, 0.25 * min(first.length, second.length)):
                continue
            combined_points = first.evidence.support_points + second.evidence.support_points
            combined_residual = float(
                np.hypot(first.evidence.tls_residual_rms_m, second.evidence.tls_residual_rms_m)
                / np.sqrt(2.0)
            )
            evidence = MeasurementEvidence(
                tls_residual_rms_m=combined_residual,
                support_points=combined_points,
                support_density_per_m2=float(
                    np.mean(
                        [
                            value
                            for value in (
                                first.evidence.support_density_per_m2,
                                second.evidence.support_density_per_m2,
                            )
                            if value is not None
                        ]
                    )
                )
                if first.evidence.support_density_per_m2 is not None
                and second.evidence.support_density_per_m2 is not None
                else None,
                pose_provenance=first.evidence.pose_provenance,
                depth_provenance=first.evidence.depth_provenance,
                calibration_status=first.evidence.calibration_status,
            )
            existing_first = pairs.get(first.id)
            existing_second = pairs.get(second.id)
            candidate_first = (second.id, separation, evidence)
            candidate_second = (first.id, separation, evidence)
            if existing_first is None or separation < existing_first[1]:
                pairs[first.id] = candidate_first
            if existing_second is None or separation < existing_second[1]:
                pairs[second.id] = candidate_second
    return pairs


def _model_parts(
    tls_model: object | None,
    walls: Sequence[object] | None,
    floor_plane: object | None,
    ceiling_planes: Sequence[object] | None,
) -> tuple[list[object], object | None, list[object], list[object], str, str, str]:
    if tls_model is not None:
        if walls is None:
            walls = _field(tls_model, "wall_planes", "wall_faces", "walls")
            if walls is None:
                planes = list(_field(tls_model, "planes", default=[]) or [])
                walls = [
                    plane
                    for plane in planes
                    if str(_field(plane, "role", "kind", default="wall")).lower()
                    in {"wall", "wall_face", "vertical"}
                ]
        if floor_plane is None:
            floor_plane = _field(tls_model, "floor_plane", "floor", "floor_planes")
            if isinstance(floor_plane, Sequence) and not isinstance(floor_plane, (str, bytes)):
                floor_plane = floor_plane[0] if floor_plane else None
        if ceiling_planes is None:
            ceiling_planes = _field(
                tls_model, "ceiling_planes", "ceiling_patches", "ceilings"
            )
            if ceiling_planes is None:
                planes = list(_field(tls_model, "planes", default=[]) or [])
                ceiling_planes = [
                    plane
                    for plane in planes
                    if str(_field(plane, "role", "kind", default="")).lower()
                    in {"ceiling", "ceiling_face", "roof"}
                ]
    walls = _as_items(walls)
    ceiling_planes = _as_items(ceiling_planes)
    intersections = _as_items(
        _field(tls_model, "intersections", "plane_intersections", default=[])
    )
    return (
        walls,
        floor_plane,
        ceiling_planes,
        intersections,
        str(_field(tls_model, "pose_provenance", "pose_source", default="unknown")),
        str(_field(tls_model, "depth_provenance", "depth_source", default="unknown")),
        str(_field(tls_model, "calibration_status", "calibration", default="uncalibrated")),
    )


def _index_intersections(
    intersections: Sequence[object], frame: HorizontalFrame | None
) -> dict[str | int, list[object]]:
    """Index explicit Stage 6 plane intersections by participating plane id."""
    indexed: dict[str | int, list[object]] = {}
    for intersection in intersections:
        point = _field(intersection, "point", "position", "xyz")
        plane_ids = _field(intersection, "planes", "plane_ids")
        if plane_ids is None:
            plane_ids = [
                _field(intersection, "plane_a", "plane1", "a"),
                _field(intersection, "plane_b", "plane2", "b"),
            ]
        if point is None and isinstance(intersection, (tuple, list)) and len(intersection) == 3:
            plane_ids = intersection[:2]
            point = intersection[2]
        if point is None or plane_ids is None or len(plane_ids) < 2:
            continue
        array = np.asarray(point, dtype=float).reshape(-1)
        if array.shape not in {(2,), (3,)} or not np.isfinite(array).all():
            continue
        for plane_id in plane_ids[:2]:
            if plane_id is not None:
                indexed.setdefault(plane_id, []).append(array)
    return indexed


def _make_scalar_plane(
    normal: np.ndarray, offset: float, source: object, role: str
) -> TLSPlane:
    return TLSPlane(
        id=role,
        normal=normal,
        offset=offset,
        role=role,
        inlier_count=int(_field(source, f"{role}_inlier_count", default=0) or 0),
        residual_rms=_finite_float(_field(source, f"{role}_residual_rms", default=0.0)),
        observed=bool(_field(source, f"{role}_observed", default=True)),
    )


def _planes_from_gravity(
    gravity: GravityEstimate | None,
    frame: HorizontalFrame | None,
    floor_plane: object | None,
    ceiling_planes: Sequence[object],
    context: MeasurementContext,
) -> tuple[object | None, list[object]]:
    if gravity is None or frame is None:
        return floor_plane, list(ceiling_planes)
    if floor_plane is None:
        floor_plane = TLSPlane(
            id="floor",
            normal=frame.up,
            offset=gravity.floor_height,
            role="floor",
            inlier_count=gravity.floor_inlier_count,
            residual_rms=gravity.floor_residual_rms,
            # Phase 1 exposes a valid scalar floor even when older callers do
            # not populate its optional support counters.  The ceiling has a
            # separate observed bit; the floor is the reference surface.
            observed=True,
            pose_provenance=context.pose_provenance,
            depth_provenance=context.depth_provenance,
            calibration_status=context.calibration_status,
        )
    if not ceiling_planes and gravity.ceiling_observed and gravity.ceiling_height is not None:
        ceiling_planes = [
            TLSPlane(
                id="ceiling",
                normal=frame.up,
                offset=gravity.ceiling_height,
                role="ceiling",
                inlier_count=gravity.ceiling_inlier_count,
                residual_rms=gravity.ceiling_residual_rms or 0.0,
                observed=True,
                pose_provenance=context.pose_provenance,
                depth_provenance=context.depth_provenance,
                calibration_status=context.calibration_status,
            )
        ]
    return floor_plane, list(ceiling_planes)


def _plane_height_at(
    floor: object,
    ceiling: object,
    plan_point: np.ndarray,
    frame: HorizontalFrame,
) -> float | None:
    floor_normal = np.asarray(_field(floor, "normal", "unit_normal"), dtype=float).reshape(-1)
    ceiling_normal = np.asarray(_field(ceiling, "normal", "unit_normal"), dtype=float).reshape(-1)
    if floor_normal.shape != (3,) or ceiling_normal.shape != (3,):
        return None
    floor_normal /= max(float(np.linalg.norm(floor_normal)), 1e-9)
    ceiling_normal /= max(float(np.linalg.norm(ceiling_normal)), 1e-9)
    floor_offset = _finite_float(_field(floor, "offset", "d", "distance"), np.nan)
    ceiling_offset = _finite_float(_field(ceiling, "offset", "d", "distance"), np.nan)
    if not np.isfinite([floor_offset, ceiling_offset]).all():
        return None
    point_at_zero = np.asarray(frame.to_world(plan_point, 0.0), dtype=float).reshape(3)
    floor_up_component = float(floor_normal @ frame.up)
    if abs(floor_up_component) <= 0.1:
        return None
    floor_height = (floor_offset - float(floor_normal @ point_at_zero)) / floor_up_component
    floor_point = point_at_zero + floor_height * frame.up
    ceiling_up_component = float(ceiling_normal @ frame.up)
    if abs(ceiling_up_component) <= 0.1:
        return None
    height = (ceiling_offset - float(ceiling_normal @ floor_point)) / ceiling_up_component
    return float(height) if np.isfinite(height) and height > 0 else None


def _height_statistics(
    room: object,
    polygon: np.ndarray | None,
    floor_plane: object | None,
    ceilings: Sequence[object],
    frame: HorizontalFrame | None,
    context: MeasurementContext,
) -> HeightStatistics:
    empty = _unmeasured(
        "observed floor/ceiling plane pair unavailable",
        MeasurementEvidence(),
        "ceiling_not_observed",
        context=context,
    )
    if (
        polygon is None
        or floor_plane is None
        or not _as_bool(_field(floor_plane, "observed", "is_observed", default=True), True)
        or not ceilings
        or frame is None
    ):
        return HeightStatistics(empty, empty, empty)
    centroid = polygon.mean(axis=0)
    samples = [centroid, *list(polygon)]
    values: list[float] = []
    evidence_values: list[MeasurementEvidence] = []
    for ceiling in ceilings:
        if not _as_bool(_field(ceiling, "observed", "is_observed", default=True), True):
            continue
        points = _field(ceiling, "inlier_points", "inliers")
        candidate_points = None
        if points is not None and not isinstance(points, (int, float, np.integer, np.floating)):
            array = np.asarray(points, dtype=float)
            if array.ndim == 2 and array.shape[1] == 3 and len(array):
                plan_points = frame.to_plan(array)
                candidate_points = [
                    plan for plan in plan_points if _point_in_polygon(plan, polygon)
                ]
                # A dense TLS ceiling can contain hundreds of thousands of
                # inliers.  A deterministic sample is sufficient for the
                # min/mean/max height statistics and keeps Stage 9 bounded.
                if candidate_points is not None and len(candidate_points) > 5000:
                    indices = np.linspace(0, len(candidate_points) - 1, 5000, dtype=int)
                    candidate_points = [candidate_points[index] for index in indices]
        ceiling_evidence = _evidence_for(ceiling, context)
        had_value = False
        for point in candidate_points or samples:
            value = _plane_height_at(floor_plane, ceiling, np.asarray(point), frame)
            if value is not None:
                values.append(value)
                had_value = True
        if had_value:
            evidence_values.append(ceiling_evidence)
    if not values:
        return HeightStatistics(empty, empty, empty)
    evidence = _combine_evidence(evidence_values)
    floor_evidence = _evidence_for(floor_plane, context)
    evidence = _combine_evidence([evidence, floor_evidence])
    spread = float(np.std(values)) if len(values) > 1 else 0.0
    mean = float(np.mean(values))
    common_sigma = max(spread / max(np.sqrt(len(values)), 1.0), 0.002)
    minimum = _make_measurement(
        min(values), evidence, context,
        sigma_m=common_sigma + spread * 0.25,
        basis=f"perpendicular floor-to-ceiling TLS heights at {len(values)} observed samples",
        extra_flags=("sloped_or_multiple_ceiling",) if spread > 0.01 else (),
    )
    average = _make_measurement(
        mean, evidence, context,
        sigma_m=common_sigma,
        basis=f"mean perpendicular floor-to-ceiling height from {len(values)} TLS samples",
        extra_flags=("sloped_or_multiple_ceiling",) if spread > 0.01 else (),
    )
    maximum = _make_measurement(
        max(values), evidence, context,
        sigma_m=common_sigma + spread * 0.25,
        basis=f"perpendicular floor-to-ceiling TLS heights at {len(values)} observed samples",
        extra_flags=("sloped_or_multiple_ceiling",) if spread > 0.01 else (),
    )
    return HeightStatistics(minimum, average, maximum)


def _combine_evidence(values: Sequence[MeasurementEvidence]) -> MeasurementEvidence:
    values = [value for value in values if value is not None]
    if not values:
        return MeasurementEvidence()
    densities = [value.support_density_per_m2 for value in values if value.support_density_per_m2 is not None]
    return MeasurementEvidence(
        tls_residual_rms_m=float(np.sqrt(np.mean([value.tls_residual_rms_m**2 for value in values]))),
        support_points=sum(value.support_points for value in values),
        support_density_per_m2=float(np.mean(densities)) if densities else None,
        pose_provenance=values[0].pose_provenance,
        depth_provenance=values[0].depth_provenance,
        calibration_status=values[0].calibration_status,
        pose_uncertainty_m=float(max(value.pose_uncertainty_m for value in values)),
    )


def measure_scene(
    walls: Sequence[object] | None = None,
    rooms: Sequence[object] | None = None,
    *,
    tls_model: TLSPlaneModel | Mapping[str, Any] | object | None = None,
    frame: HorizontalFrame | None = None,
    gravity: GravityEstimate | None = None,
    floor_plane: object | None = None,
    ceiling_planes: Sequence[object] | None = None,
    context: MeasurementContext | None = None,
    default_wall_thickness_m: float = 0.15,
) -> SceneMeasurements:
    """Measure a TLS scene using supplied plane geometry and explicit offsets.

    ``walls``/``rooms`` are retained as positional compatibility inputs for
    Phase 1.  A Stage 6 caller may instead pass ``tls_model`` containing
    ``wall_planes``, ``floor_plane`` and ``ceiling_planes``.  Stage 9 does
    not reconstruct intersections or polygonize wall lines.  If the geometry
    stages do not supply a bounded room face, the area is explicitly
    ``unmeasured``.
    """
    context = context or MeasurementContext(default_wall_thickness_m=default_wall_thickness_m)
    if context.default_wall_thickness_m != default_wall_thickness_m and default_wall_thickness_m != 0.15:
        context.default_wall_thickness_m = max(float(default_wall_thickness_m), 0.0)
    (
        wall_values,
        floor_plane,
        ceiling_values,
        explicit_intersections,
        model_pose,
        model_depth,
        model_calibration,
    ) = _model_parts(
        tls_model, walls, floor_plane, ceiling_planes
    )
    if rooms is None and tls_model is not None:
        rooms = _as_items(
            _field(
                tls_model,
                "rooms",
                "room_faces",
                "bounded_faces",
                "room_boundaries",
                default=[],
            )
        )
    if tls_model is not None:
        if context.pose_provenance == "unknown" and model_pose != "unknown":
            context.pose_provenance = model_pose
        if context.depth_provenance == "unknown" and model_depth != "unknown":
            context.depth_provenance = model_depth
        if context.calibration_status == "uncalibrated" and model_calibration != "uncalibrated":
            context.calibration_status = model_calibration
    floor_plane, ceiling_values = _planes_from_gravity(
        gravity, frame, floor_plane, ceiling_values, context
    )
    records = _coerce_wall_records(
        wall_values,
        frame,
        context,
        _index_intersections(explicit_intersections, frame),
    )
    pairs = _thickness_pairs(records)

    wall_results: dict[str | int, WallMeasurement] = {}
    for record in records:
        if record.start is not None and record.end is not None and record.length > 1e-8:
            from_intersections = record.geometry_source == "supplied_plane_intersections"
            length_basis = (
                "distance between supplied finite intersections of observed TLS wall planes"
                if from_intersections
                else "distance between Phase 1 observed TLS wall endpoints; structured intersection extent pending"
            )
            length_measurement = _make_measurement(
                record.length,
                record.evidence,
                context,
                sigma_m=max(record.evidence.tls_residual_rms_m, 0.002)
                * (1.0 if from_intersections else 1.75),
                basis=length_basis,
                inferred=not from_intersections,
                extra_flags=("open_ended_geometry",) if not from_intersections else (),
            )
        else:
            length_measurement = _unmeasured(
                "finite wall extent/intersections not supplied by TLS geometry stage",
                record.evidence,
                "wall_length_unmeasured",
                "plane_extent_missing",
                context=context,
            )
        height_range = _field(record.source, "height_range", "vertical_extent")
        extent = None
        if height_range is not None:
            values = np.asarray(height_range, dtype=float).reshape(-1)
            if len(values) >= 2 and np.isfinite(values[:2]).all():
                extent = abs(float(values[1] - values[0]))
        if extent is None or extent <= 0:
            inlier_points = _field(record.source, "inlier_points", "inliers")
            if frame is not None and inlier_points is not None and not isinstance(inlier_points, (int, float)):
                points = np.asarray(inlier_points, dtype=float)
                if points.ndim == 2 and points.shape[1] == 3 and len(points):
                    extent = float(np.ptp(points @ frame.up))
        vertical_measurement = (
            _make_measurement(
                extent,
                record.evidence,
                context,
                sigma_m=max(record.evidence.tls_residual_rms_m, 0.002),
                basis="observed inlier vertical extent of the TLS wall plane",
            )
            if extent is not None and extent > 0
            else _unmeasured(
                "wall inlier vertical extent unavailable from TLS support",
                record.evidence,
                "vertical_extent_unmeasured",
                context=context,
            )
        )
        opposing_face_id = None
        if record.id in pairs:
            opposing_face_id, thickness_value, thickness_evidence = pairs[record.id]
            thickness_measurement = _make_measurement(
                thickness_value,
                thickness_evidence,
                context,
                sigma_m=max(thickness_evidence.tls_residual_rms_m, 0.002),
                basis="separation of two actually observed opposing TLS wall faces",
            )
        else:
            thickness_measurement = _unmeasured(
                "opposing observed TLS wall face not found",
                record.evidence,
                "opposing_face_not_observed",
                "wall_thickness_unmeasured",
            )
        wall_results[record.id] = WallMeasurement(
            wall_id=record.id,
            length=length_measurement,
            inlier_vertical_extent=vertical_measurement,
            thickness=thickness_measurement,
            geometry_source=record.geometry_source,
            opposing_face_id=opposing_face_id,
        )

    room_results: dict[str | int, RoomMeasurement] = {}
    for room_index, room in enumerate(rooms or []):
        room_id = _field(room, "id", "room_id", default=room_index)
        room_wall_ids = _field(room, "wall_indices", "wall_ids", default=None)
        if room_wall_ids is not None:
            selected = [record for record in records if record.id in set(room_wall_ids)]
        else:
            selected = records
        polygon, boundary_source = _room_boundary(room, frame)
        if polygon is None:
            empty = _unmeasured(
                "bounded room face not supplied by TLS geometry/closure stage",
                _evidence_for(room, context),
                "room_boundary_unmeasured",
                "geometry_not_supplied",
                context=context,
            )
            empty_height = HeightStatistics(empty, empty, empty)
            room_results[room_id] = RoomMeasurement(
                room_id,
                empty,
                empty,
                empty,
                empty_height,
                {},
                [],
                _area_convention(context),
            )
            continue
        edge_records = [
            _matching_wall_for_edge(polygon[i], polygon[(i + 1) % len(polygon)], selected)
            for i in range(len(polygon))
        ]
        edge_records = [record for record in edge_records if record is not None]
        boundary_evidence = _combine_evidence(
            [_evidence_for(room, context), *[record.evidence for record in edge_records]]
        )
        perimeter = float(
            np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).sum()
        )
        edge_sigma = max(boundary_evidence.tls_residual_rms_m, 0.002)
        interior = _make_measurement(
            _area(polygon),
            boundary_evidence,
            context,
            sigma_m=max(perimeter * edge_sigma / 2.0, 0.002),
            basis=f"PRIMARY: supplied bounded interior TLS wall-face area ({boundary_source})",
        )
        distances: list[float] = []
        thickness_docs: dict[str, Measurement] = {}
        assumed = False
        for record in edge_records:
            wall_result = wall_results.get(record.id)
            if wall_result is not None:
                thickness_docs[str(record.id)] = wall_result.thickness
            if wall_result is not None and wall_result.thickness.value is not None:
                distances.append(wall_result.thickness.value / 2.0)
            else:
                distances.append(context.default_wall_thickness_m / 2.0)
                assumed = True
        if len(distances) != len(polygon):
            distances = [context.default_wall_thickness_m / 2.0] * len(polygon)
            assumed = True
        centreline_polygon = _offset_polygon(polygon, distances)
        outer_polygon = _offset_polygon(polygon, [distance * 2.0 for distance in distances])
        edge_thickness_measurements = [
            wall_results[record.id].thickness
            for record in edge_records
            if record.id in wall_results
        ]
        thickness_sigma = (
            float(np.mean([measurement.tolerance or 0.0 for measurement in edge_thickness_measurements]))
            if edge_thickness_measurements
            else 0.0
        )
        centerline = _make_measurement(
            _area(centreline_polygon),
            boundary_evidence,
            context,
            sigma_m=max(perimeter * edge_sigma / 2.0 + perimeter * thickness_sigma / 2.0, 0.002),
            basis="interior face offset outward by one-half of each wall thickness",
            status="estimated" if assumed else "measured",
            assumed=assumed,
        )
        outer = _make_measurement(
            _area(outer_polygon),
            boundary_evidence,
            context,
            sigma_m=max(perimeter * edge_sigma / 2.0 + perimeter * thickness_sigma, 0.002),
            basis="interior face offset outward by one full wall thickness",
            status="estimated" if assumed else "measured",
            assumed=assumed,
        )
        room_results[room_id] = RoomMeasurement(
            room_id=room_id,
            interior_face_area=interior,
            wall_centerline_area=centerline,
            outer_footprint_area=outer,
            floor_to_ceiling_height=_height_statistics(
                room, polygon, floor_plane, ceiling_values, frame, context
            ),
            wall_thicknesses=thickness_docs,
            boundary=polygon.tolist(),
            area_convention=_area_convention(context),
        )
    return SceneMeasurements(walls=wall_results, rooms=room_results)


def _area_convention(context: MeasurementContext) -> dict[str, Any]:
    return {
        "primary": "interior_face_area",
        "interior_face_area": "area bounded by observed interior wall-face plane intersections",
        "wall_centerline_area": "offset each interior face outward by measured thickness/2; assumed default when thickness is unmeasured",
        "outer_footprint_area": "offset each interior face outward by measured thickness; assumed default when thickness is unmeasured",
        "default_wall_thickness_m": context.default_wall_thickness_m,
        "thickness_measurement_rule": "report measured only when two opposing faces are both observed and overlap",
        "geometry_rule": "consume only supplied plane intersections and bounded faces; no Stage 9 graph closure or raster corners",
    }


@dataclass
class ScaleValidation:
    """Result of an explicit known-reference scale check."""

    reference_type: str
    observed_length_m: float | None
    known_length_m: float | None
    scale_factor: float | None
    tolerance: float | None
    confidence: float
    status: str
    flags: list[str] = field(default_factory=list)
    basis: str = ""
    applied: bool = False

    def to_dict(self) -> dict:
        return {
            "reference_type": self.reference_type,
            "observed_length_m": self.observed_length_m,
            "known_length_m": self.known_length_m,
            "recommended_scale_factor": self.scale_factor,
            "tolerance": self.tolerance,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "flags": sorted(set(self.flags)),
            "basis": self.basis,
            "applied": self.applied,
        }


def validate_reference_scale(
    observed_length_m: float | None = None,
    known_length_m: float | None = None,
    *,
    reference_type: str = "user",
    tolerance_m: float | None = None,
    apply: bool = False,
    reference: object | None = None,
) -> ScaleValidation:
    """Validate a marker/tape/user reference without silently calibrating.

    ``scale_factor`` is a recommendation.  It affects no measurements unless
    the caller explicitly opts into applying it.  A door reference is always
    advisory because door sizes vary and must never silently calibrate a TLS
    scene.
    """
    if reference is not None:
        observed_length_m = _field(
            reference,
            "observed_length_m",
            "observed_m",
            "measured_length_m",
            default=observed_length_m,
        )
        known_length_m = _field(
            reference,
            "known_length_m",
            "known_m",
            "reference_length_m",
            default=known_length_m,
        )
        reference_type = _field(
            reference,
            "reference_type",
            "kind",
            "type",
            default=reference_type,
        )
    kind = _normalise_status(reference_type).replace("-", "_")
    if kind in {"user_supplied", "user_reference", "known_reference"}:
        kind = "user"
    observed = _finite_float(observed_length_m, np.nan)
    known = _finite_float(known_length_m, np.nan)
    if kind == "door":
        return ScaleValidation(
            kind,
            observed if np.isfinite(observed) else None,
            known if np.isfinite(known) else None,
            None,
            None,
            0.20,
            "advisory",
            ["advisory_only", "never_used_for_calibration", "manual_review"],
            "door dimensions are a variable heuristic, not a known reference",
            False,
        )
    if not np.isfinite([observed, known]).all() or observed <= 0 or known <= 0:
        return ScaleValidation(
            kind,
            observed if np.isfinite(observed) else None,
            known if np.isfinite(known) else None,
            None,
            None,
            0.0,
            "unmeasured",
            ["invalid_reference", "manual_review"],
            "positive observed and known lengths are required",
            False,
        )
    factor = known / observed
    tolerance = max(_finite_float(tolerance_m, 0.01 * known), 0.001)
    confidence = float(np.clip(1.0 - abs(observed * factor - known) / max(tolerance, 1e-9), 0.0, 1.0))
    return ScaleValidation(
        kind,
        observed,
        known,
        factor,
        tolerance,
        max(confidence, 0.90),
        "validated",
        ["calibration_not_applied"] if not apply else [],
        f"explicit {kind} reference: known / observed = {factor:.6f}",
        bool(apply),
    )


def door_scale_advisory(observed_length_m: float | None, assumed_door_width_m: float = 0.90) -> ScaleValidation:
    """Return a visible door heuristic result that cannot calibrate anything."""
    return validate_reference_scale(
        observed_length_m,
        assumed_door_width_m,
        reference_type="door",
        apply=False,
    )


# Concise aliases for callers that use the names from the Stage 9 design note.
measure_geometry = measure_scene
build_measurements = measure_scene


def validate_scale(
    observed_length_m: float | None = None,
    known_length_m: float | None = None,
    *,
    reference_type: str = "user",
    kind: str | None = None,
    observed_m: float | None = None,
    known_m: float | None = None,
    tolerance_m: float | None = None,
    apply: bool = False,
    reference: object | None = None,
) -> ScaleValidation:
    """Compatibility spelling for the explicit reference validation API."""
    if observed_length_m is None:
        observed_length_m = observed_m
    if known_length_m is None:
        known_length_m = known_m
    return validate_reference_scale(
        observed_length_m,
        known_length_m,
        reference_type=kind or reference_type,
        tolerance_m=tolerance_m,
        apply=apply,
        reference=reference,
    )


compute_measurements = measure_scene
