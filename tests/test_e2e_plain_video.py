import json

from cozmo_ai_v2.cli import run_prepare, run_slam
from cozmo_ai_v2.mast3r_slam import Mast3rSlamInvocation


def test_prepare_plain_video(synthetic_video, tmp_path):
    output_dir = tmp_path / "out"

    exit_code = run_prepare(synthetic_video, output_dir, "config/base.yaml")

    assert exit_code == 0
    assert not (output_dir / "intrinsics.yaml").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset_path"].endswith("test.mp4")
    assert manifest["calib_path"] is None
    assert "--calib" not in manifest["command"]


def test_run_plain_video_starts_mast3r_slam_without_calibration(synthetic_video, tmp_path, monkeypatch):
    mast3r_slam_dir = tmp_path / "MASt3R-SLAM"
    mast3r_slam_dir.mkdir()
    (mast3r_slam_dir / "main.py").write_text("# test entry point\n")
    received = {}

    def fake_run_rgb_video(video_path, checkout, config, **kwargs):
        received.update(video_path=video_path, checkout=checkout, config=config, **kwargs)
        return Mast3rSlamInvocation(("mast3r-python", "main.py"), checkout, 0)

    monkeypatch.setattr("cozmo_ai_v2.cli.run_rgb_video", fake_run_rgb_video)

    exit_code = run_slam(
        synthetic_video,
        mast3r_slam_dir,
        "config/base.yaml",
        "mast3r-python",
        "test-run",
        True,
    )

    assert exit_code == 0
    assert received == {
        "video_path": synthetic_video,
        "checkout": mast3r_slam_dir,
        "config": "config/base.yaml",
        "python_executable": "mast3r-python",
        "save_as": "test-run",
        "no_viz": True,
    }
