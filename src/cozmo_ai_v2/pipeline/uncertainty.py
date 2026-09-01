from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Z-scores that convert a target coverage probability (for example, "90%
# of the time the truth should fall inside this interval") into how many
# standard deviations wide the interval needs to be.
Z_SCORES = {0.68: 1.0, 0.80: 1.282, 0.90: 1.645, 0.95: 1.96}

# Baseline LiDAR depth noise, in metres, plus how much extra noise to add
# per metre of distance from the sensor -- depth sensors get noisier the
# further away they're measuring.
DEPTH_SIGMA_BASE = 0.010
DEPTH_SIGMA_PER_METRE = 0.008


@dataclass
class Interval:
    """A single measurement together with a confidence interval around
    it -- a range the true value is expected to fall inside, most of the
    time.

    Attributes:
        value: The best-guess measurement itself (for example, a wall's
            length, in metres).
        half_width: How far the interval extends on either side of
            `value`, in the same units as `value`. The full interval
            runs from `value - half_width` to `value + half_width`.
        coverage: The target probability that the true value actually
            falls inside this interval (for example, 0.90 means "90% of
            the time").
        basis: A short, human-readable note explaining where
            `half_width` came from, useful for debugging or for showing
            in a report.
    """

    value: float
    half_width: float
    coverage: float
    basis: str

    @property
    def low(self) -> float:
        """The lower edge of the interval: `value - half_width`."""
        return self.value - self.half_width

    @property
    def high(self) -> float:
        """The upper edge of the interval: `value + half_width`."""
        return self.value + self.half_width

    def to_dict(self) -> dict:
        """Turns this interval into a plain dictionary, ready to be
        written into the result file, with every number rounded to 4
        decimal places.

        Returns:
            A dictionary with `value`, `ci_low`, `ci_high`, `half_width`,
            `coverage`, and `basis` keys.
        """
        return {
            "value": round(self.value, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "half_width": round(self.half_width, 4),
            "coverage": self.coverage,
            "basis": self.basis,
        }


class UncertaintyModel:
    """Turns physical evidence -- sensor noise, how many points supported
    a measurement, how much drift was detected -- into a confidence
    interval, and calibrates that interval against real-world truth when
    it can.

    The raw, "physical" uncertainty estimate this model starts from is
    only ever an approximation: it comes from a simplified model of how
    noisy the sensor is and how measurements get combined, and real-world
    error doesn't always behave quite that neatly. To correct for that, a
    single learned `scale` factor -- fit by `fit_calibration` against
    measurements where the ground truth is actually known -- widens (or
    narrows) every interval this model produces, so the stated confidence
    level ends up matching reality more closely. Before any calibration
    data is available, `scale` defaults to 1.0, meaning the raw physical
    estimate is used as-is.
    """

    def __init__(
        self,
        coverage: float = 0.90,
        calibration_path: str | Path | None = None,
        has_depth: bool = True,
    ):
        """Creates an uncertainty model, loading a previously saved
        calibration if one is available.

        Args:
            coverage: The target coverage probability for every interval
                this model produces (for example, 0.90 for "90% of the
                time"). Looked up in `Z_SCORES`.
            calibration_path: Path to a JSON file previously written out
                by `fit_calibration`. If the file exists, its values
                replace the defaults below and `self.calibrated` is set
                to True; if not, the model falls back to its
                uncalibrated defaults.
            has_depth: Whether this capture actually has real LiDAR
                depth data. When False -- meaning depth had to be
                estimated some other way -- every interval this model
                produces gets widened by `no_lidar_multiplier`, since
                estimated depth is much less trustworthy than measured
                depth.
        """
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
        """Estimates how noisy the LiDAR depth reading is expected to be
        at a given distance from the sensor, in metres.

        Args:
            distance: How far away, in metres, the thing being measured
                is from the sensor.

        Returns:
            The expected noise, as a standard deviation (sigma), in
            metres.
        """
        return DEPTH_SIGMA_BASE + DEPTH_SIGMA_PER_METRE * max(distance, 0.0)

    def plane_offset_sigma(
        self, residual_rms: float, point_count: int, drift: float = 0.0
    ) -> float:
        """Estimates how uncertain the position of a fitted plane (like a
        wall, floor, or ceiling) really is, by combining two very
        different kinds of error.

        The first kind is ordinary sensor noise: individual depth
        readings scatter randomly around the true surface, but that
        scatter averages down as more points are used -- fitting a plane
        from thousands of points pins its position down far more tightly
        than any single point could on its own, which is why this term
        shrinks as `point_count` grows. Drift works completely
        differently and can't be treated the same way: it isn't random
        scatter around the truth, it's a specific, coherent shift in
        where the whole wall appears to sit, caused by the camera's
        tracked position wandering over time. Because every point sampled
        from a drifted wall gets dragged in roughly the same direction
        rather than scattering randomly, adding more points does nothing
        to average drift away -- so it has to be added in as its own
        separate term instead of being folded into the noise
        calculation. The two terms are then combined the way independent
        sources of error normally are, by adding their squares and
        taking the square root.

        Args:
            residual_rms: The RMS residual of the plane fit, in metres --
                roughly, how much the supporting points scatter around
                the fitted plane.
            point_count: How many points supported the fit.
            drift: The coherent per-visit offset spread measured for this
                plane, in metres (see `measure_drift`).

        Returns:
            The combined uncertainty, as a standard deviation (sigma), in
            metres, from the averaged-noise term and the drift term
            together.
        """
        averaged = residual_rms / max(np.sqrt(max(point_count, 1)), 1.0)
        floor = 0.002
        return float(np.sqrt(max(averaged, floor) ** 2 + drift**2))

    def wall_length(
        self,
        length: float,
        residual_rms: float,
        point_count: int,
        drift: float = 0.0,
        inferred_fraction: float = 0.0,
    ) -> Interval:
        """Builds a confidence interval for a wall's length.

        A wall's length is the distance between two corners, and each
        corner in turn comes from where two different wall planes meet --
        so the uncertainty in a single plane's position (from
        `plane_offset_sigma`) ends up counting twice over. If part of the
        wall's extent had to be inferred rather than actually observed
        (for example, because a section was hidden behind furniture),
        the interval is widened further to reflect that extra
        guesswork.

        Args:
            length: The measured wall length, in metres.
            residual_rms: The RMS residual of the wall's plane fit, in
                metres.
            point_count: How many points supported the fit.
            drift: The coherent per-visit offset spread measured for
                this wall, in metres.
            inferred_fraction: What fraction of the wall's extent was
                inferred rather than directly observed.

        Returns:
            An `Interval` around `length` at `self.coverage`.
        """
        per_plane = self.plane_offset_sigma(residual_rms, point_count, drift)
        sigma = per_plane * 2.0
        if inferred_fraction > 0:
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
        """Builds a confidence interval for the floor-to-ceiling height.

        Floor and ceiling planes are usually large and covered by a
        great many points, so each one's own position is already known
        quite precisely; combining their two independent uncertainties
        (again by adding squares and taking the square root) is what
        makes the resulting height interval especially tight compared to
        most other measurements in this file.

        Args:
            height: The measured floor-to-ceiling height, in metres.
            floor_rms: The RMS residual of the floor plane fit, in
                metres.
            ceiling_rms: The RMS residual of the ceiling plane fit, in
                metres.
            floor_points: How many points supported the floor fit.
            ceiling_points: How many points supported the ceiling fit.

        Returns:
            An `Interval` around `height` at `self.coverage`.
        """
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
        """Builds a confidence interval for a room's floor area.

        Most of the error in a measured area comes from uncertainty in
        exactly where the room's boundary sits, not from anything inside
        the room -- a longer perimeter simply gives more edge for small
        position errors to add up along, which is why this scales with
        `perimeter` rather than with the area itself.

        Args:
            area: The measured floor area, in square metres.
            perimeter: The room's perimeter, in metres.
            edge_sigma: The uncertainty in the boundary's position, per
                metre of edge, in metres.

        Returns:
            An `Interval` around `area` at `self.coverage`.
        """
        sigma = perimeter * edge_sigma / 2.0
        half = self.z * sigma * self.scale * self._modality()
        return Interval(
            area,
            half,
            self.coverage,
            f"{perimeter:.1f} m perimeter at {edge_sigma * 1000:.1f} mm edge sigma",
        )

    def opening_width(self, width: float, resolution: float, confidence: float) -> Interval:
        """Builds a confidence interval for a door or window's width.

        An opening's edges are only known down to the resolution of the
        surface grid they were measured on, and the `confidence` score
        (from `Opening.confidence`) captures how cleanly the opening's
        edges actually came through -- a soft or partially blocked edge
        (for example, from door trim) adds extra uncertainty on top of
        the basic grid resolution.

        Args:
            width: The measured opening width, in metres.
            resolution: The surface grid's cell size, in metres.
            confidence: The opening's `confidence` score, from 0 to 1.

        Returns:
            An `Interval` around `width` at `self.coverage`.
        """
        sigma = resolution / np.sqrt(12) * 2
        sigma = float(np.hypot(sigma, 0.01 * (1.0 - confidence) * 3))
        half = self.z * sigma * self.scale * self._modality()
        return Interval(
            width,
            half,
            self.coverage,
            f"grid {resolution * 1000:.0f} mm, edge confidence {confidence:.2f}",
        )

    def damage_area(self, area: float, view_count: int, mask_trusted: bool) -> Interval:
        """Builds a confidence interval for a fused damage region's area.

        Unlike the other interval methods here, this one works in
        relative terms -- as a percentage of the area itself -- rather
        than an absolute distance, since damage regions vary hugely in
        size and shape. Being seen from more independent camera views
        makes the estimate more trustworthy, the same averaging-down
        effect described in `plane_offset_sigma`. If the region's outline
        came from a rough bounding box instead of a real segmentation
        mask, a flat extra margin is added on top, since a box is a much
        cruder approximation of the region's true shape.

        Args:
            area: The measured damage region area, in square metres.
            view_count: How many independent camera views the fused
                region was observed from.
            mask_trusted: Whether the region's shape came from a real
                segmentation mask rather than a bounding-box fallback.

        Returns:
            An `Interval` around `area` at `self.coverage`, using a
            relative rather than absolute error model.
        """
        relative = 0.12 / max(np.sqrt(max(view_count, 1)), 1.0)
        if not mask_trusted:
            relative += 0.25
        half = area * relative * self.z * self.scale
        return Interval(
            area,
            half,
            self.coverage,
            f"{view_count} independent views, "
            f"{'mask' if mask_trusted else 'box fallback'}",
        )

    def _modality(self) -> float:
        """Returns how much extra to widen every interval by, to account
        for depth having been estimated rather than actually measured
        with LiDAR."""
        return 1.0 if self.has_depth else self.no_lidar_multiplier


def fit_calibration(
    predictions: list[float],
    truths: list[float],
    half_widths: list[float],
    coverage: float = 0.90,
) -> dict:
    """Works out the single `scale` factor that makes this model's
    stated confidence intervals actually match reality.

    This is meant to be run separately, against a set of measurements
    where the true value is already known (for example, from
    hand-measuring a test property). For each measurement, it compares
    the model's original error against the interval half-width the model
    originally reported, then finds the scale factor that would have
    needed to be applied so that `coverage` fraction of those
    measurements actually fell inside their interval. That scale factor
    is what `UncertaintyModel` then applies to every future interval it
    produces.

    Args:
        predictions: The model's point estimates for each measurement.
        truths: The corresponding, actually-known ground-truth values.
        half_widths: The (uncalibrated, `scale=1.0`) interval half-width
            the model originally reported for each measurement.
        coverage: The target coverage probability to calibrate to.

    Returns:
        A dictionary with `scale`, `coverage_target`, `coverage_before`,
        `coverage_after`, `sample_count`, `median_error_mm`, and
        `p90_error_mm` keys.
    """
    errors = np.abs(np.asarray(predictions) - np.asarray(truths))
    widths = np.maximum(np.asarray(half_widths), 1e-9)
    normalised = errors / widths
    # The scale factor is simply the value at the target-coverage
    # percentile of "how many interval-widths off was each guess" -- for
    # example, if 90% of guesses were within 1.3x their stated interval,
    # scaling every future interval by 1.3 fixes that.
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
