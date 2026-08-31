from __future__ import annotations

import subprocess

import pytest

from cozmo_ai_v2.mast3r_slam import (
    Mast3rSlamError,
    build_rgb_video_command,
    run_rgb_video,
)


def test_build_rgb_video_command_omits_calibration(synthetic_video):
    command = build_rgb_video_command(
        synthetic_video,
        "config/base.yaml",
        python_executable="mast3r-python",
        save_as="office",
        no_viz=True,
    )

    assert command == [
        "mast3r-python",
        "main.py",
        "--dataset",
        str(synthetic_video.resolve()),
        "--config",
        "config/base.yaml",
        "--save-as",
        "office",
        "--no-viz",
    ]
    assert "--calib" not in command


def test_run_rgb_video_invokes_checkout_entrypoint(synthetic_video, tmp_path):
    mast3r_slam_dir = tmp_path / "MASt3R-SLAM"
    mast3r_slam_dir.mkdir()
    (mast3r_slam_dir / "main.py").write_text("# test entry point\n")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    invocation = run_rgb_video(
        synthetic_video,
        mast3r_slam_dir,
        python_executable="mast3r-python",
        no_viz=True,
        run_process=fake_run,
    )

    assert invocation.returncode == 0
    assert invocation.cwd == mast3r_slam_dir.resolve()
    assert calls == [
        (
            [
                "mast3r-python",
                "main.py",
                "--dataset",
                str(synthetic_video.resolve()),
                "--config",
                "config/base.yaml",
                "--no-viz",
            ],
            {"cwd": mast3r_slam_dir.resolve(), "check": False},
        )
    ]


def test_run_rgb_video_reports_a_missing_checkout(tmp_path, synthetic_video):
    with pytest.raises(Mast3rSlamError, match="directory does not exist"):
        run_rgb_video(synthetic_video, tmp_path / "missing")


def test_run_rgb_video_reports_external_failure(synthetic_video, tmp_path):
    mast3r_slam_dir = tmp_path / "MASt3R-SLAM"
    mast3r_slam_dir.mkdir()
    (mast3r_slam_dir / "main.py").write_text("# test entry point\n")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 42)

    with pytest.raises(Mast3rSlamError, match="status 42"):
        run_rgb_video(synthetic_video, mast3r_slam_dir, run_process=fake_run)
