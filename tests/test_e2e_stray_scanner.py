import json

import yaml

from cozmo_ai_v2.cli import run_prepare


def test_prepare_stray_scanner_dataset(stray_scanner_dataset, tmp_path):
    output_dir = tmp_path / "out"

    exit_code = run_prepare(stray_scanner_dataset, output_dir, "config/base.yaml")

    assert exit_code == 0

    intrinsics = yaml.safe_load((output_dir / "intrinsics.yaml").read_text())
    assert intrinsics["width"] == 64
    assert intrinsics["height"] == 48
    assert intrinsics["calibration"] == [517.3, 516.5, 318.6, 255.3]

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["dataset_path"].endswith("rgb.mp4")
    assert manifest["calib_path"].endswith("intrinsics.yaml")
    assert "--calib" in manifest["command"]
