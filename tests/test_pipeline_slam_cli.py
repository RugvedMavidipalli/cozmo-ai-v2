from pathlib import Path

from cozmo_ai_v2.pipeline.cli import main


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
