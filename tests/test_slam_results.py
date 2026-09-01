from __future__ import annotations

import json

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.slam import (
    AlignmentThresholds,
    PoseTrajectory,
    SlamResultError,
    align_trajectory_to_arkit,
    integrate_mast3r_results,
    parse_loop_closure_metrics,
    parse_mast3r_trajectory,
    resample_trajectory,
    write_pose_integration_manifest,
)


def _trajectory(positions, rotations=None, source="test"):
    positions = np.asarray(positions, dtype=float)
    poses = np.tile(np.eye(4), (len(positions), 1, 1))
    poses[:, :3, 3] = positions
    if rotations is not None:
        poses[:, :3, :3] = rotations
    return PoseTrajectory(np.arange(len(positions), dtype=float), poses, source)


def _write_odometry(path, trajectory):
    lines = [
        "timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy,distortion_center_x,distortion_center_y"
    ]
    for index, (timestamp, pose) in enumerate(zip(trajectory.timestamps, trajectory.poses)):
        x, y, z = pose[:3, 3]
        lines.append(f"{timestamp},{index:06d},{x},{y},{z},0,0,0,1,500,500,320,240,,")
    path.write_text("\n".join(lines) + "\n")


def test_parse_mast3r_trajectory_uses_pipeline_camera_to_world_contract(tmp_path):
    trajectory_path = tmp_path / "recording.txt"
    trajectory_path.write_text("1.5 1 2 3 0 0 0 1\n")

    trajectory = parse_mast3r_trajectory(trajectory_path)

    assert trajectory.coordinate_convention == "camera_to_world_opencv_x_right_y_down_z_forward"
    assert trajectory.timestamps.tolist() == [1.5]
    np.testing.assert_allclose(trajectory.poses[0, :3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(trajectory.poses[0, :3, :3], np.eye(3))


def test_parse_mast3r_trajectory_rejects_malformed_rows(tmp_path):
    trajectory_path = tmp_path / "recording.txt"
    trajectory_path.write_text("1 2 3\n")

    with pytest.raises(SlamResultError, match="expected 8 trajectory fields"):
        parse_mast3r_trajectory(trajectory_path)


def test_alignment_uses_sim3_for_scale_divergence_and_reports_gate():
    prior_positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0], [3, 2, 0]],
        dtype=float,
    )
    prior = _trajectory(prior_positions, source="arkit")
    angle = np.radians(25.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    scale = 1.25
    translation = np.array([4.0, -2.0, 0.5])
    slam_positions = np.array([rotation.T @ (point - translation) / scale for point in prior_positions])
    slam_rotations = np.tile(rotation.T, (len(slam_positions), 1, 1))
    slam = _trajectory(slam_positions, slam_rotations, source="mast3r_slam")

    aligned, report = align_trajectory_to_arkit(slam, prior)

    assert report.method == "sim3"
    assert report.scale == pytest.approx(scale)
    assert report.translation_rmse_m < 1e-8
    assert report.rotation_rmse_degrees < 1e-5
    assert report.scale_divergence_fraction == pytest.approx(0.25)
    assert report.fusion_allowed is False
    assert any("scale divergence" in diagnostic for diagnostic in report.failure_diagnostics)
    np.testing.assert_allclose(aligned.poses[:, :3, 3], prior_positions, atol=1e-8)


def test_alignment_uses_se3_when_trajectory_is_metric():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    prior = _trajectory(positions, source="arkit")
    slam = _trajectory(positions + [3, -1, 0.5], source="mast3r_slam")

    _aligned, report = align_trajectory_to_arkit(slam, prior)

    assert report.method == "se3"
    assert report.scale == pytest.approx(1.0)
    assert report.fusion_allowed is True


def test_alignment_normalizes_video_relative_timestamps_to_arkit_timebase():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    prior = PoseTrajectory(
        np.array([100.0, 101.0, 102.0, 103.0]), _trajectory(positions).poses, "arkit"
    )
    slam = _trajectory(positions, source="mast3r_slam")

    aligned, report = align_trajectory_to_arkit(slam, prior)

    assert report.timestamp_offset_seconds == pytest.approx(100.0)
    np.testing.assert_allclose(aligned.timestamps, prior.timestamps)


def test_resample_trajectory_fills_capture_timestamps_without_registration():
    sparse = PoseTrajectory(
        np.array([0.0, 0.5, 1.0]),
        _trajectory([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).poses,
        "mast3r_slam",
    )

    dense = resample_trajectory(sparse, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    np.testing.assert_allclose(dense.poses[:, :3, 3], [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0]])


def test_resample_trajectory_rejects_unobserved_gaps():
    sparse = PoseTrajectory(
        np.array([0.0, 2.0]),
        _trajectory([[0, 0, 0], [2, 0, 0]]).poses,
        "mast3r_slam",
    )

    with pytest.raises(SlamResultError, match="gap"):
        resample_trajectory(sparse, np.array([0.0, 1.0, 2.0]), max_gap_seconds=0.5)


def test_integrate_results_records_arkit_provenance_and_loop_metrics(tmp_path):
    trajectory_path = tmp_path / "recording.txt"
    trajectory_path.write_text(
        "\n".join(f"{i} {i} 0 0 0 0 0 1" for i in range(4)) + "\n"
    )
    prior_path = tmp_path / "odometry.csv"
    _write_odometry(prior_path, _trajectory([[i, 0, 0] for i in range(4)], source="arkit"))
    metrics_path = tmp_path / "mast3r_slam_metrics.json"
    metrics_path.write_text(json.dumps({"loop_closure": {"candidate_count": 4, "accepted_count": 2, "residual": 0.03}}))

    integration = integrate_mast3r_results(
        trajectory_path,
        pose_priors_path=prior_path,
        pose_prior_mode="post_alignment",
        results_dir=tmp_path,
    )
    manifest_path = write_pose_integration_manifest(tmp_path / "pose_provenance.json", integration)
    manifest = json.loads(manifest_path.read_text())

    assert integration.pose_source == "mast3r_slam_aligned_to_arkit"
    assert integration.fusion_allowed is True
    assert integration.loop_closure.status == "detected"
    assert manifest["pose_prior"]["mode"] == "post_alignment"
    assert manifest["loop_closure"]["accepted_count"] == 2
    assert manifest["alignment"]["thresholds"]["translation_rmse_m"] == 0.25


def test_missing_loop_metrics_are_explicitly_not_reported(tmp_path):
    metrics = parse_loop_closure_metrics(tmp_path)

    assert metrics.status == "not_reported"
    assert metrics.diagnostic is not None
