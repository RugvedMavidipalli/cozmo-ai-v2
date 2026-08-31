"""MASt3R-SLAM result ingestion and ARKit trajectory sanity checks.

The pipeline's pose contract is an ``(N, 4, 4)`` sequence of camera-to-world
transforms in OpenCV camera axes: +X right, +Y down, +Z forward.  The released
MASt3R-SLAM evaluator writes timestamped camera-to-world poses in that same
shape as ``timestamp x y z qx qy qz qw``.  This module deliberately keeps its
work to parsing, alignment, and diagnostics; it does not run inference or add
another registration pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ingest import load_odometry_pose_priors


POSE_CONVENTION = "camera_to_world_opencv_x_right_y_down_z_forward"


class SlamResultError(ValueError):
    """Raised when a MASt3R-SLAM output cannot be safely consumed."""


@dataclass(frozen=True)
class PoseTrajectory:
    """Timestamped camera-to-world poses that meet the pipeline contract."""

    timestamps: np.ndarray
    poses: np.ndarray
    source: str
    coordinate_convention: str = POSE_CONVENTION

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        poses = np.asarray(self.poses, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise SlamResultError("trajectory must contain at least one timestamp")
        if poses.shape != (len(timestamps), 4, 4):
            raise SlamResultError(
                "trajectory poses must have shape "
                f"({len(timestamps)}, 4, 4), got {poses.shape}"
            )
        if not np.isfinite(timestamps).all() or not np.isfinite(poses).all():
            raise SlamResultError("trajectory contains non-finite values")
        if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise SlamResultError("trajectory poses must be homogeneous transforms")
        rotations = poses[:, :3, :3]
        orthogonality = np.einsum("nji,njk->nik", rotations, rotations)
        if not np.allclose(orthogonality, np.eye(3), atol=1e-4) or np.any(
            np.linalg.det(rotations) <= 0.0
        ):
            raise SlamResultError("trajectory poses must contain proper rotation matrices")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "poses", poses)

    def __len__(self) -> int:
        return len(self.timestamps)


@dataclass(frozen=True)
class AlignmentThresholds:
    """Pre-fusion gates for disagreement with a metric ARKit trajectory."""

    translation_rmse_m: float = 0.25
    translation_max_m: float = 0.75
    rotation_rmse_degrees: float = 15.0
    rotation_max_degrees: float = 45.0
    scale_divergence_fraction: float = 0.15


@dataclass(frozen=True)
class TrajectoryAlignment:
    """Robust global SE(3)/Sim(3) alignment and its pre-fusion verdict."""

    method: str
    transform: np.ndarray
    scale: float
    timestamp_offset_seconds: float
    matched_frames: int
    inlier_frames: int
    translation_rmse_m: float
    translation_max_m: float
    rotation_rmse_degrees: float
    rotation_max_degrees: float
    scale_divergence_fraction: float
    thresholds: AlignmentThresholds
    fusion_allowed: bool
    failure_diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopClosureMetrics:
    """Loop-closure observability reported by an upstream sidecar, if any."""

    status: str
    candidate_count: int | None = None
    accepted_count: int | None = None
    residual: float | None = None
    source_path: Path | None = None
    diagnostic: str | None = None


@dataclass(frozen=True)
class PoseIntegration:
    """Result supplied to fusion and persisted as pose provenance."""

    trajectory: PoseTrajectory
    pose_source: str
    trajectory_path: Path
    pose_prior_path: Path | None
    pose_prior_mode: str
    alignment: TrajectoryAlignment | None
    loop_closure: LoopClosureMetrics
    fusion_allowed: bool | None
    diagnostics: tuple[str, ...] = ()


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = float(np.linalg.norm([qx, qy, qz, qw]))
    if norm == 0.0:
        raise SlamResultError("trajectory contains a zero-norm quaternion")
    qx, qy, qz, qw = (qx / norm, qy / norm, qz / norm, qw / norm)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def parse_mast3r_trajectory(path: str | Path) -> PoseTrajectory:
    """Parse MASt3R-SLAM's ``timestamp tx ty tz qx qy qz qw`` trajectory."""

    path = Path(path)
    if not path.is_file():
        raise SlamResultError(f"MASt3R-SLAM trajectory was not found: {path}")

    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SlamResultError(f"Could not read MASt3R-SLAM trajectory {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 8:
            raise SlamResultError(
                f"{path}:{line_number}: expected 8 trajectory fields, got {len(fields)}"
            )
        try:
            timestamp, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in fields)
        except ValueError as exc:
            raise SlamResultError(f"{path}:{line_number}: trajectory fields must be numbers") from exc
        if not np.isfinite([timestamp, tx, ty, tz, qx, qy, qz, qw]).all():
            raise SlamResultError(f"{path}:{line_number}: trajectory contains non-finite values")
        pose = np.eye(4)
        pose[:3, :3] = _quaternion_to_matrix(qx, qy, qz, qw)
        pose[:3, 3] = [tx, ty, tz]
        timestamps.append(timestamp)
        poses.append(pose)

    if not poses:
        raise SlamResultError(f"MASt3R-SLAM trajectory contains no poses: {path}")
    return PoseTrajectory(np.asarray(timestamps), np.stack(poses), "mast3r_slam")


def load_arkit_pose_priors(path: str | Path) -> PoseTrajectory:
    """Load Stray Scanner/ARKit odometry as pipeline-contract pose priors."""

    path = Path(path)
    odometry_path = path / "odometry.csv" if path.is_dir() else path
    try:
        timestamps, poses = load_odometry_pose_priors(odometry_path)
    except (OSError, KeyError, ValueError) as exc:
        raise SlamResultError(f"Could not load ARKit pose priors from {odometry_path}: {exc}") from exc
    if len(timestamps) == 0:
        raise SlamResultError(f"ARKit pose-prior file contains no poses: {odometry_path}")
    return PoseTrajectory(timestamps, poses, "stray_scanner_arkit_odometry")


def mast3r_results_dir(checkout: str | Path, video_path: str | Path, save_as: str | None) -> Path:
    """Return the MASt3R-SLAM ``logs`` directory for one invocation."""

    results_dir = Path(checkout).expanduser().resolve() / "logs"
    if save_as:
        results_dir /= save_as
    return results_dir


def mast3r_trajectory_path(checkout: str | Path, video_path: str | Path, save_as: str | None) -> Path:
    """Return the upstream evaluator's expected trajectory output path."""

    return mast3r_results_dir(checkout, video_path, save_as) / f"{Path(video_path).stem}.txt"


def parse_loop_closure_metrics(
    results_dir: str | Path,
    metrics_path: str | Path | None = None,
) -> LoopClosureMetrics:
    """Read an optional ``mast3r_slam_metrics.json`` loop-closure sidecar.

    Upstream MASt3R-SLAM currently writes trajectory/reconstruction files but
    no machine-readable loop-closure counters. Absence is therefore explicit
    diagnostic state, not a claim that no loop closure happened.
    """

    selected = Path(metrics_path) if metrics_path is not None else Path(results_dir) / "mast3r_slam_metrics.json"
    if not selected.is_file():
        return LoopClosureMetrics(
            "not_reported",
            source_path=selected,
            diagnostic=(
                "No MASt3R-SLAM loop-closure metrics sidecar was found; "
                "this upstream version does not expose loop counters."
            ),
        )
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return LoopClosureMetrics(
            "invalid",
            source_path=selected,
            diagnostic=f"Could not parse loop-closure metrics: {exc}",
        )
    if not isinstance(raw, dict):
        return LoopClosureMetrics("invalid", source_path=selected, diagnostic="metrics root must be an object")
    loop = raw.get("loop_closure", raw)
    if not isinstance(loop, dict):
        return LoopClosureMetrics("invalid", source_path=selected, diagnostic="loop_closure must be an object")

    def integer(*names: str) -> int | None:
        value = next((loop[name] for name in names if name in loop), None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def number(*names: str) -> float | None:
        value = next((loop[name] for name in names if name in loop), None)
        return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None

    candidates = integer("candidate_count", "loop_candidates")
    accepted = integer("accepted_count", "loop_edges")
    status = loop.get("status") if isinstance(loop.get("status"), str) else None
    if status is None:
        status = "detected" if accepted and accepted > 0 else "none"
    return LoopClosureMetrics(status, candidates, accepted, number("residual", "rmse"), selected)


def _matched_indices(
    trajectory_timestamps: np.ndarray,
    prior_timestamps: np.ndarray,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(prior_timestamps)
    ordered_timestamps = prior_timestamps[order]

    def match(offset_seconds: float) -> tuple[np.ndarray, np.ndarray, float]:
        trajectory_indices: list[int] = []
        prior_indices: list[int] = []
        errors: list[float] = []
        used_priors: set[int] = set()
        for trajectory_index, timestamp in enumerate(trajectory_timestamps):
            adjusted_timestamp = timestamp + offset_seconds
            insertion = int(np.searchsorted(ordered_timestamps, adjusted_timestamp))
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(order)]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda index: abs(ordered_timestamps[index] - adjusted_timestamp))
            prior_index = int(order[nearest])
            error = abs(prior_timestamps[prior_index] - adjusted_timestamp)
            if error <= tolerance_seconds and prior_index not in used_priors:
                trajectory_indices.append(trajectory_index)
                prior_indices.append(prior_index)
                errors.append(float(error))
                used_priors.add(prior_index)
        median_error = float(np.median(errors)) if errors else float("inf")
        return np.asarray(trajectory_indices), np.asarray(prior_indices), median_error

    offsets = {
        0.0,
        float(prior_timestamps[0] - trajectory_timestamps[0]),
        float(prior_timestamps[-1] - trajectory_timestamps[-1]),
    }
    candidates = [(match(offset), offset) for offset in offsets]
    (trajectory_indices, prior_indices, _), offset = max(
        candidates,
        key=lambda item: (len(item[0][0]), -item[0][2], -abs(item[1])),
    )
    return trajectory_indices, prior_indices, offset


def _fit_similarity(source: np.ndarray, target: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise SlamResultError("at least three timestamp-matched poses are required for alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance < 1e-10:
        raise SlamResultError("trajectory positions are degenerate; cannot estimate alignment")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular * np.diag(correction)) / variance) if estimate_scale else 1.0
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _robust_similarity(source: np.ndarray, target: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    scale, rotation, translation = _fit_similarity(source, target, estimate_scale)
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    cutoff = max(0.03, median + 3.0 * 1.4826 * mad)
    inliers = residuals <= cutoff
    if inliers.sum() >= 3 and inliers.sum() < len(source):
        scale, rotation, translation = _fit_similarity(source[inliers], target[inliers], estimate_scale)
    else:
        inliers = np.ones(len(source), dtype=bool)
    return scale, rotation, translation, inliers


def _rotation_angles_degrees(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,njk->nik", np.transpose(reference, (0, 2, 1)), estimate)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def align_trajectory_to_arkit(
    trajectory: PoseTrajectory,
    arkit_priors: PoseTrajectory,
    *,
    timestamp_tolerance_seconds: float = 1.0 / 15.0,
    alignment_mode: str = "auto",
    thresholds: AlignmentThresholds = AlignmentThresholds(),
) -> tuple[PoseTrajectory, TrajectoryAlignment]:
    """Robustly align MASt3R poses to metric ARKit poses before fusion.

    ``auto`` selects SE(3) when the robust Sim(3) scale is within 2% of
    metric scale; otherwise it retains Sim(3) so scale divergence is visible
    and can gate fusion. No pairwise or multi-pass registration is performed.
    """

    if alignment_mode not in {"auto", "se3", "sim3"}:
        raise SlamResultError("alignment_mode must be one of: auto, se3, sim3")
    trajectory_indices, prior_indices, timestamp_offset = _matched_indices(
        trajectory.timestamps, arkit_priors.timestamps, timestamp_tolerance_seconds
    )
    if len(trajectory_indices) < 3:
        raise SlamResultError(
            "fewer than three timestamp-matched MASt3R-SLAM and ARKit poses; "
            "cannot perform a safe trajectory comparison"
        )
    source = trajectory.poses[trajectory_indices, :3, 3]
    target = arkit_priors.poses[prior_indices, :3, 3]
    sim_scale, _, _, _ = _robust_similarity(source, target, True)
    method = alignment_mode
    if method == "auto":
        method = "se3" if abs(sim_scale - 1.0) <= 0.02 else "sim3"
    scale, rotation, translation, inliers = _robust_similarity(source, target, method == "sim3")
    aligned_poses = trajectory.poses.copy()
    aligned_poses[:, :3, :3] = np.einsum("ij,njk->nik", rotation, aligned_poses[:, :3, :3])
    aligned_poses[:, :3, 3] = (scale * (rotation @ aligned_poses[:, :3, 3].T)).T + translation
    residuals = np.linalg.norm(
        aligned_poses[trajectory_indices, :3, 3] - arkit_priors.poses[prior_indices, :3, 3], axis=1
    )
    angles = _rotation_angles_degrees(
        arkit_priors.poses[prior_indices, :3, :3], aligned_poses[trajectory_indices, :3, :3]
    )
    translation_rmse = float(np.sqrt(np.mean(residuals**2)))
    rotation_rmse = float(np.sqrt(np.mean(angles**2)))
    scale_divergence = abs(sim_scale - 1.0)
    failures: list[str] = []
    if translation_rmse > thresholds.translation_rmse_m:
        failures.append(f"translation RMSE {translation_rmse:.3f} m exceeds {thresholds.translation_rmse_m:.3f} m")
    if float(residuals.max()) > thresholds.translation_max_m:
        failures.append(f"translation maximum {residuals.max():.3f} m exceeds {thresholds.translation_max_m:.3f} m")
    if rotation_rmse > thresholds.rotation_rmse_degrees:
        failures.append(f"rotation RMSE {rotation_rmse:.2f}° exceeds {thresholds.rotation_rmse_degrees:.2f}°")
    if float(angles.max()) > thresholds.rotation_max_degrees:
        failures.append(f"rotation maximum {angles.max():.2f}° exceeds {thresholds.rotation_max_degrees:.2f}°")
    if scale_divergence > thresholds.scale_divergence_fraction:
        failures.append(
            f"scale divergence {scale_divergence:.1%} exceeds {thresholds.scale_divergence_fraction:.1%}"
        )
    alignment = TrajectoryAlignment(
        method,
        _similarity_matrix(scale, rotation, translation),
        scale,
        timestamp_offset,
        len(trajectory_indices),
        int(inliers.sum()),
        translation_rmse,
        float(residuals.max()),
        rotation_rmse,
        float(angles.max()),
        scale_divergence,
        thresholds,
        not failures,
        tuple(failures),
    )
    return PoseTrajectory(
        trajectory.timestamps + timestamp_offset, aligned_poses, trajectory.source
    ), alignment


def _similarity_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    axis = int(np.argmax(np.diag(rotation)))
    if axis == 0:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return np.array([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
    if axis == 1:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return np.array([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
    scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return np.array([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale])


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(first @ second)
    if cosine < 0.0:
        second = -second
        cosine = -cosine
    if cosine > 0.9995:
        return (first + alpha * (second - first)) / np.linalg.norm(first + alpha * (second - first))
    theta = np.arccos(np.clip(cosine, -1.0, 1.0))
    return (np.sin((1.0 - alpha) * theta) * first + np.sin(alpha * theta) * second) / np.sin(theta)


def resample_trajectory(
    trajectory: PoseTrajectory,
    target_timestamps: np.ndarray,
    *,
    max_gap_seconds: float = 1.0,
) -> PoseTrajectory:
    """Interpolate a sparse MASt3R keyframe trajectory onto capture frames.

    This is timestamp interpolation only, not another registration step. It
    refuses extrapolation or large unobserved gaps so a partial SLAM trajectory
    cannot silently replace ARKit poses during fusion.
    """

    target_timestamps = np.asarray(target_timestamps, dtype=float)
    if target_timestamps.ndim != 1 or len(target_timestamps) == 0:
        raise SlamResultError("target timestamps must be a non-empty one-dimensional array")
    if not np.isfinite(target_timestamps).all() or np.any(np.diff(target_timestamps) < 0.0):
        raise SlamResultError("target timestamps must be finite and non-decreasing")
    order = np.argsort(trajectory.timestamps)
    timestamps = trajectory.timestamps[order]
    poses = trajectory.poses[order]
    if np.any(np.diff(timestamps) <= 0.0):
        raise SlamResultError("MASt3R-SLAM timestamps must be strictly increasing for resampling")
    if target_timestamps[0] < timestamps[0] or target_timestamps[-1] > timestamps[-1]:
        raise SlamResultError("MASt3R-SLAM trajectory does not span all capture timestamps")
    resampled = np.empty((len(target_timestamps), 4, 4), dtype=float)
    for index, timestamp in enumerate(target_timestamps):
        upper = int(np.searchsorted(timestamps, timestamp, side="left"))
        if upper < len(timestamps) and np.isclose(timestamps[upper], timestamp, atol=1e-8):
            resampled[index] = poses[upper]
            continue
        lower = upper - 1
        gap = timestamps[upper] - timestamps[lower]
        if gap > max_gap_seconds:
            raise SlamResultError(
                f"MASt3R-SLAM pose gap {gap:.3f}s exceeds safe interpolation limit {max_gap_seconds:.3f}s"
            )
        alpha = (timestamp - timestamps[lower]) / gap
        first = _matrix_to_quaternion(poses[lower, :3, :3])
        second = _matrix_to_quaternion(poses[upper, :3, :3])
        resampled[index] = np.eye(4)
        resampled[index, :3, :3] = _quaternion_to_matrix(*_slerp(first, second, float(alpha)))
        resampled[index, :3, 3] = (1.0 - alpha) * poses[lower, :3, 3] + alpha * poses[upper, :3, 3]
    return PoseTrajectory(target_timestamps, resampled, trajectory.source)


def integrate_mast3r_results(
    trajectory_path: str | Path,
    *,
    pose_priors_path: str | Path | None = None,
    pose_prior_mode: str = "not_requested",
    results_dir: str | Path | None = None,
    metrics_path: str | Path | None = None,
    target_timestamps: np.ndarray | None = None,
    interpolation_max_gap_seconds: float = 1.0,
    thresholds: AlignmentThresholds = AlignmentThresholds(),
) -> PoseIntegration:
    """Parse MASt3R results, optionally align them to ARKit, and diagnose them."""

    trajectory_path = Path(trajectory_path)
    trajectory = parse_mast3r_trajectory(trajectory_path)
    resolved_results_dir = Path(results_dir) if results_dir is not None else trajectory_path.parent
    loop_closure = parse_loop_closure_metrics(resolved_results_dir, metrics_path)
    diagnostics: list[str] = []
    if loop_closure.diagnostic:
        diagnostics.append(loop_closure.diagnostic)
    if pose_priors_path is None:
        diagnostics.append("No ARKit/Stray pose prior supplied; RGB-only trajectory remains in MASt3R-SLAM world coordinates.")
        return PoseIntegration(
            trajectory,
            "mast3r_slam_rgb_only",
            trajectory_path.resolve(),
            None,
            pose_prior_mode,
            None,
            loop_closure,
            None,
            tuple(diagnostics),
        )

    prior_path = Path(pose_priors_path).expanduser().resolve()
    priors = load_arkit_pose_priors(prior_path)
    aligned, alignment = align_trajectory_to_arkit(trajectory, priors, thresholds=thresholds)
    if target_timestamps is not None:
        aligned = resample_trajectory(
            aligned, target_timestamps, max_gap_seconds=interpolation_max_gap_seconds
        )
    diagnostics.extend(alignment.failure_diagnostics)
    if pose_prior_mode == "post_alignment":
        diagnostics.append("Upstream MASt3R-SLAM did not consume ARKit priors; they were used for post-run validation only.")
    return PoseIntegration(
        aligned,
        "mast3r_slam_aligned_to_arkit",
        trajectory_path.resolve(),
        prior_path,
        pose_prior_mode,
        alignment,
        loop_closure,
        alignment.fusion_allowed,
        tuple(diagnostics),
    )


def write_pose_integration_manifest(path: str | Path, integration: PoseIntegration) -> Path:
    """Persist pose provenance, loop metrics, alignment limits, and diagnostics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = integration.alignment
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "ok",
        "pose_source": integration.pose_source,
        "coordinate_convention": integration.trajectory.coordinate_convention,
        "trajectory": {
            "path": str(integration.trajectory_path),
            "frame_count": len(integration.trajectory),
            "timestamp_start": float(integration.trajectory.timestamps[0]),
            "timestamp_end": float(integration.trajectory.timestamps[-1]),
        },
        "pose_prior": {
            "path": str(integration.pose_prior_path) if integration.pose_prior_path else None,
            "mode": integration.pose_prior_mode,
        },
        "loop_closure": {
            "status": integration.loop_closure.status,
            "candidate_count": integration.loop_closure.candidate_count,
            "accepted_count": integration.loop_closure.accepted_count,
            "residual": integration.loop_closure.residual,
            "source_path": str(integration.loop_closure.source_path) if integration.loop_closure.source_path else None,
            "diagnostic": integration.loop_closure.diagnostic,
        },
        "fusion_allowed": integration.fusion_allowed,
        "failure_diagnostics": list(integration.diagnostics),
    }
    if alignment is not None:
        payload["alignment"] = {
            "method": alignment.method,
            "transform": alignment.transform.tolist(),
            "scale": alignment.scale,
            "timestamp_offset_seconds": alignment.timestamp_offset_seconds,
            "matched_frames": alignment.matched_frames,
            "inlier_frames": alignment.inlier_frames,
            "translation_rmse_m": alignment.translation_rmse_m,
            "translation_max_m": alignment.translation_max_m,
            "rotation_rmse_degrees": alignment.rotation_rmse_degrees,
            "rotation_max_degrees": alignment.rotation_max_degrees,
            "scale_divergence_fraction": alignment.scale_divergence_fraction,
            "thresholds": alignment.thresholds.__dict__,
            "fusion_allowed": alignment.fusion_allowed,
            "failure_diagnostics": list(alignment.failure_diagnostics),
        }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def write_pose_failure_manifest(
    path: str | Path,
    error: Exception | str,
    *,
    pose_priors_path: str | Path | None = None,
    pose_prior_mode: str = "not_requested",
) -> Path:
    """Record a result-ingestion failure in the same provenance location."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "failed",
        "coordinate_convention": POSE_CONVENTION,
        "pose_prior": {
            "path": str(Path(pose_priors_path).expanduser().resolve()) if pose_priors_path else None,
            "mode": pose_prior_mode,
        },
        "failure_diagnostics": [str(error)],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path
