from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cozmo_ai_v2.mast3r_slam import Mast3rSlamInvocation
from cozmo_ai_v2.pipeline.cli import _launch_mast3r_for_capture, main


def test_pipeline_cli_accepts_mast3r_trajectory_before_fusion(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.run", fake_run)
    trajectory = tmp_path / "rgb.txt"
    metrics = tmp_path / "mast3r_slam_metrics.json"

    exit_code = main(
        [
            "run",
            str(tmp_path / "capture"),
            "--mast3r-trajectory",
            str(trajectory),
            "--mast3r-metrics",
            str(metrics),
            "--mast3r-max-pose-gap",
            "0.5",
        ]
    )

    assert exit_code == 0
    assert captured["mast3r_trajectory"] == trajectory
    assert captured["mast3r_metrics"] == metrics
    assert captured["mast3r_max_pose_gap"] == 0.5


def test_pipeline_cli_can_run_mast3r_for_a_capture_with_arkit_poses(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.run", fake_run)
    checkout = tmp_path / "MASt3R-SLAM"

    exit_code = main(
        [
            "run",
            str(tmp_path / "capture"),
            "--run-mast3r",
            "--mast3r-slam-dir",
            str(checkout),
            "--mast3r-config",
            "config/fast.yaml",
            "--mast3r-python",
            "mast3r-python",
            "--mast3r-save-as",
            "capture-run",
            "--mast3r-no-viz",
        ]
    )

    assert exit_code == 0
    assert captured["run_mast3r"] is True
    assert captured["mast3r_slam_dir"] == checkout
    assert captured["mast3r_config"] == "config/fast.yaml"
    assert captured["mast3r_python"] == "mast3r-python"
    assert captured["mast3r_save_as"] == "capture-run"
    assert captured["mast3r_no_viz"] is True
    assert captured["mast3r_trajectory"] is None


def test_capture_mast3r_launch_uses_arkit_pose_prior_and_results_path(monkeypatch, tmp_path):
    capture = tmp_path / "capture"
    checkout = tmp_path / "MASt3R-SLAM"
    received = {}

    def fake_run_rgb_video(video_path, mast3r_slam_dir, config, **kwargs):
        received.update(
            video_path=video_path,
            mast3r_slam_dir=mast3r_slam_dir,
            config=config,
            **kwargs,
        )
        return Mast3rSlamInvocation(
            ("mast3r-python", "main.py"),
            checkout,
            0,
            pose_prior_mode="post_alignment",
        )

    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.run_rgb_video", fake_run_rgb_video)
    trajectory, results_dir, prior_mode = _launch_mast3r_for_capture(
        capture,
        checkout,
        "config/base.yaml",
        "mast3r-python",
        "capture-run",
        True,
    )

    assert received == {
        "video_path": capture / "rgb.mp4",
        "mast3r_slam_dir": checkout,
        "config": "config/base.yaml",
        "python_executable": "mast3r-python",
        "save_as": "capture-run",
        "no_viz": True,
        "pose_priors_path": capture / "odometry.csv",
    }
    assert trajectory == checkout / "logs" / "capture-run" / "rgb.txt"
    assert results_dir == checkout / "logs" / "capture-run"
    assert prior_mode == "post_alignment"


def test_run_mast3r_skips_arkit_refinement_before_trajectory_validation(monkeypatch, tmp_path):
    capture = tmp_path / "capture"
    out_dir = tmp_path / "out"

    class Bundle:
        root = capture
        timestamps = np.array([0.0, 0.1])
        poses = np.tile(np.eye(4), (2, 1, 1))
        gravity_consistency = 1.0

        def __len__(self):
            return 2

        @property
        def duration(self):
            return 0.1

    launched = {}

    def fake_launch(*args):
        launched["args"] = args
        return tmp_path / "rgb.txt", tmp_path / "logs", "post_alignment"

    def must_not_refine(*_args, **_kwargs):
        raise AssertionError("ARKit refinement must be skipped when --run-mast3r is set")

    monkeypatch.setattr(
        "cozmo_ai_v2.pipeline.cli.load_capture", lambda _path, **_kwargs: Bundle()
    )
    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli._launch_mast3r_for_capture", fake_launch)
    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.refine_trajectory", must_not_refine)
    monkeypatch.setattr(
        "cozmo_ai_v2.pipeline.cli.integrate_mast3r_results",
        lambda *_args, **_kwargs: SimpleNamespace(fusion_allowed=False),
    )
    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.write_pose_integration_manifest", lambda *_args: None)

    exit_code = main(
        [
            "run",
            str(capture),
            "--out",
            str(out_dir),
            "--run-mast3r",
            "--mast3r-slam-dir",
            str(tmp_path / "MASt3R-SLAM"),
        ]
    )

    assert exit_code == 1  # The mocked divergence gate declines fusion.
    assert launched["args"][0] == capture
