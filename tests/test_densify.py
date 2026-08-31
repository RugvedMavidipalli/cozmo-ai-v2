import json

import cv2
import numpy as np

from cozmo_ai_v2.cli import run_densify
from cozmo_ai_v2.depth.capture import require_lidar_capture
from cozmo_ai_v2.depth.densify import densify_capture

VIDEO_WIDTH = 64
VIDEO_HEIGHT = 48


class FakeDepthModel:
    """Satisfies the DepthModel protocol. Returns a deliberately mis-shifted
    version of the known ground truth, one frame per call, in call order."""

    def __init__(self, true_depths_m, bias=-0.3):
        self._true_depths_m = true_depths_m
        self._bias = bias
        self._call_index = 0

    def predict(self, rgb, fx):
        true = self._true_depths_m[self._call_index]
        self._call_index += 1
        return (true + self._bias).astype(np.float32)


def _full_res_ramp(depth_mm_low_res):
    start_mm = float(depth_mm_low_res[0, 0])
    end_mm = float(depth_mm_low_res[0, -1])
    ramp_mm = np.linspace(start_mm, end_mm, VIDEO_WIDTH, dtype=np.float32)
    return (np.tile(ramp_mm, (VIDEO_HEIGHT, 1)) / 1000.0).astype(np.float32)


def test_densify_capture_end_to_end(lidar_stray_scanner_dataset, tmp_path):
    capture = require_lidar_capture(lidar_stray_scanner_dataset)

    depth0 = cv2.imread(str(lidar_stray_scanner_dataset / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth1 = cv2.imread(str(lidar_stray_scanner_dataset / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED)
    true_depths = [_full_res_ramp(depth0), _full_res_ramp(depth1)]

    model = FakeDepthModel(true_depths, bias=-0.3)
    output_dir = tmp_path / "out"

    densify_capture(capture, model, output_dir)

    manifest = json.loads((output_dir / "densify_manifest.json").read_text())
    assert manifest["frame_count"] == 2
    assert len(manifest["frames"]) == 2
    for report in manifest["frames"]:
        assert report["scale"] > 0

    for index, true_depth in enumerate(true_depths):
        dense_png = cv2.imread(str(output_dir / "dense_depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        assert dense_png.shape == (VIDEO_HEIGHT, VIDEO_WIDTH)
        dense_m = dense_png.astype(np.float32) / 1000.0
        assert np.abs(dense_m - true_depth).mean() < 0.05


def test_cli_densify_reports_missing_torch(lidar_stray_scanner_dataset, tmp_path, capsys):
    exit_code = run_densify(
        lidar_stray_scanner_dataset, tmp_path / "out", "metric3d_vit_small",
        min_confidence=1, max_depth=8.0, guide_radius=20, guide_eps=100.0,
    )

    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()
