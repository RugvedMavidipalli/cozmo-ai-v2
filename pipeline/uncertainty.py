"""Confidence intervals for every measurement, and their calibration.

An interval is only worth printing if it is calibrated: if the system claims
+/-2 cm at 90%, then close to 90% of measurements must actually land inside
+/-2 cm.  Overconfidence on a hard capture is explicitly an automatic red flag
in the assignment, so the model here is built to widen honestly rather than to
look precise.

Two layers:

* A *physical* model that propagates what is known about the measurement --
  depth noise averaged over the supporting points, the drift measured between
  visits, and how much of the span was inferred rather than seen.
* A *conformal* scale factor fitted against laser ground truth, which corrects
  the physical model's optimism without assuming the errors are Gaussian.
  Until ground truth exists the factor is 1.0 and the report says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Half-width multiplier for a normal distribution at the quoted coverage.
Z_SCORES = {0.68: 1.0, 0.80: 1.282, 0.90: 1.645, 0.95: 1.96}

# ARKit's depth is a fused LiDAR + ML estimate; its per-pixel error grows with
# range. This is a deliberately conservative envelope, not a datasheet figure.
DEPTH_SIGMA_BASE = 0.010  # metres at close range
DEPTH_SIGMA_PER_METRE = 0.008


@dataclass
class Interval:
    value: float
    half_width: float
    coverage: float
    basis: str

    @property
    def low(self) -> float:
        return self.value - self.half_width

    @property
    def high(self) -> float:
        return self.value + self.half_width

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "half_width": round(self.half_width, 4),
            "coverage": self.coverage,
            "basis": self.basis,
        }


class UncertaintyModel:
    """Turns supporting evidence into an interval, then calibrates it."""

    def __init__(
        self,
        coverage: float = 0.90,
        calibration_path: str | Path | None = None,
        has_depth: bool = True,
    ):
        self.coverage = coverage
        self.z = Z_SCORES.get(coverage, 1.645)
        self.has_depth = has_depth
        self.scale = 1.0
        self.calibrated = False
        self.no_lidar_multiplier = 3.0
        if calibration_path and Path(calibration_path).exists():
            stored = json.loads(Path(calibration_path).read_text())
            self.scale = float(stored.get("scale", 1.0))
            self.calibrated = True
            self.no_lidar_multiplier = float(
                stored.get("no_lidar_multiplier", self.no_lidar_multiplier)
            )

    def _sensor_sigma(self, distance: float) -> float:
        return DEPTH_SIGMA_BASE + DEPTH_SIGMA_PER_METRE * max(distance, 0.0)

    def plane_offset_sigma(
        self, residual_rms: float, point_count: int, drift: float = 0.0
    ) -> float:
        """Uncertainty in where a fitted plane sits.

        Scatter averages down with the number of supporting points; drift does
        not, because it displaces a whole visit coherently.  Treating drift as
        an averageable random error is the mistake that produces confident,
        wrong intervals.
        """
        averaged = residual_rms / max(np.sqrt(max(point_count, 1)), 1.0)
        floor = 0.002  # no plane fit is better than a couple of millimetres
        return float(np.sqrt(max(averaged, floor) ** 2 + drift**2))

    def wall_length(
        self,
        length: float,
        residual_rms: float,
        point_count: int,
        drift: float = 0.0,
        inferred_fraction: float = 0.0,
    ) -> Interval:
        """A wall length is a difference of two corners, each from two planes."""
        per_plane = self.plane_offset_sigma(residual_rms, point_count, drift)
        # Two corners, each the intersection of this wall with a neighbour:
        # four plane offsets contribute, added in quadrature.
        sigma = per_plane * 2.0
        if inferred_fraction > 0:
            # An unobserved span is carried by the plane fit alone; widen in
            # proportion to how much of the wall was never actually seen.
            sigma *= 1.0 + 1.5 * inferred_fraction
        half = self.z * sigma * self.scale * self._modality()
        basis = (
            f"plane offset sigma {per_plane * 1000:.1f} mm from "
            f"{point_count} points (rms {residual_rms * 1000:.1f} mm), "
            f"drift {drift * 1000:.1f} mm"
        )
        if inferred_fraction > 0:
            basis += f", {inferred_fraction * 100:.0f}% inferred"
        return Interval(length, half, self.coverage, basis)

    def ceiling_height(
        self,
        height: float,
        floor_rms: float,
        ceiling_rms: float,
        floor_points: int,
        ceiling_points: int,
    ) -> Interval:
        """Floor and ceiling are large, densely sampled planes: this is tight."""
        floor_sigma = self.plane_offset_sigma(floor_rms, floor_points)
        ceiling_sigma = self.plane_offset_sigma(ceiling_rms, ceiling_points)
        sigma = float(np.hypot(floor_sigma, ceiling_sigma))
        half = self.z * sigma * self.scale * self._modality()
        return Interval(
            height,
            half,
            self.coverage,
            f"floor sigma {floor_sigma * 1000:.1f} mm, "
            f"ceiling sigma {ceiling_sigma * 1000:.1f} mm",
        )

    def floor_area(self, area: float, perimeter: float, edge_sigma: float) -> Interval:
        """Area error is dominated by how well the boundary is placed."""
        sigma = perimeter * edge_sigma / 2.0
        half = self.z * sigma * self.scale * self._modality()
        return Interval(
            area,
            half,
            self.coverage,
            f"{perimeter:.1f} m perimeter at {edge_sigma * 1000:.1f} mm edge sigma",
        )

    def opening_width(self, width: float, resolution: float, confidence: float) -> Interval:
        """Opening edges are quantised by the surface grid and softened by reveals."""
        sigma = resolution / np.sqrt(12) * 2  # two quantised edges
        sigma = float(np.hypot(sigma, 0.01 * (1.0 - confidence) * 3))
        half = self.z * sigma * self.scale * self._modality()
        return Interval(
            width,
            half,
            self.coverage,
            f"grid {resolution * 1000:.0f} mm, edge confidence {confidence:.2f}",
        )

    def damage_area(self, area: float, view_count: int, mask_trusted: bool) -> Interval:
        """Area of a fused region: boundary placement plus mask quality."""
        relative = 0.12 / max(np.sqrt(max(view_count, 1)), 1.0)
        if not mask_trusted:
            relative += 0.25  # a box stands in for a mask: an upper bound
        half = area * relative * self.z * self.scale
        return Interval(
            area,
            half,
            self.coverage,
            f"{view_count} independent views, "
            f"{'mask' if mask_trusted else 'box fallback'}",
        )

    def _modality(self) -> float:
        """Widen everything when depth was estimated rather than measured."""
        return 1.0 if self.has_depth else self.no_lidar_multiplier


def fit_calibration(
    predictions: list[float],
    truths: list[float],
    half_widths: list[float],
    coverage: float = 0.90,
) -> dict:
    """Find the scale that makes stated intervals match observed coverage.

    Conformal rather than parametric: the multiplier is read off the empirical
    quantile of normalised errors, so it needs no distributional assumption and
    a handful of ground-truth measurements is enough to be useful.
    """
    errors = np.abs(np.asarray(predictions) - np.asarray(truths))
    widths = np.maximum(np.asarray(half_widths), 1e-9)
    normalised = errors / widths
    scale = float(np.quantile(normalised, coverage)) if len(normalised) else 1.0

    return {
        "scale": scale,
        "coverage_target": coverage,
        "coverage_before": float((normalised <= 1.0).mean()) if len(normalised) else 0.0,
        "coverage_after": float((normalised <= scale).mean()) if len(normalised) else 0.0,
        "sample_count": len(normalised),
        "median_error_mm": float(np.median(errors) * 1000) if len(errors) else 0.0,
        "p90_error_mm": float(np.quantile(errors, 0.9) * 1000) if len(errors) else 0.0,
    }
