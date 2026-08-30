import json

from cozmo_ai_v2.cli import run_prepare


def test_prepare_plain_video(synthetic_video, tmp_path):
    output_dir = tmp_path / "out"

    exit_code = run_prepare(synthetic_video, output_dir, "config/base.yaml")

    assert exit_code == 0
    assert not (output_dir / "intrinsics.yaml").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset_path"].endswith("test.mp4")
    assert manifest["calib_path"] is None
    assert "--calib" not in manifest["command"]
