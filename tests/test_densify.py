import json

import cv2
import numpy as np

from cozmo_ai_v2.cli import _densify_indices, run_densify
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


def test_densify_capture_end_to_end(stray_capture, tmp_path):
    capture = require_lidar_capture(stray_capture)

    depth0 = cv2.imread(str(stray_capture / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth1 = cv2.imread(str(stray_capture / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED)
    true_depths = [_full_res_ramp(depth0), _full_res_ramp(depth1)]

    model = FakeDepthModel(true_depths, bias=-0.3)
    output_dir = tmp_path / "out"

    densify_capture(capture, model, output_dir)

    manifest = json.loads((output_dir / "densify_manifest.json").read_text())
    assert manifest["frame_count"] == 2
    assert manifest["units"] == {
        "lidar_input": "m",
        "model_canonical_output": "m",
        "dense_raster_output": "mm",
    }
    assert manifest["filter_policy"]["confidence_threshold"] == 1
    assert manifest["filter_policy"]["max_depth_m"] == 8.0
    assert manifest["registration_alignment"]["dense_output"] == "native_rgb"
    assert manifest["model"]["adapter"] == "FakeDepthModel"
    assert manifest["video_availability"]["expected_frame_count"] == 2
    assert manifest["video_availability"]["decoded_frame_count"] == 2
    assert manifest["video_availability"]["pts_status"] == "used"
    assert manifest["population"] == {
        "input_frames": 2,
        "selected_frames": 2,
        "densified_frames": 2,
        "qc_approved_frames": 2,
        "rejected_frames": 0,
        "missing_selected_frames": 0,
    }
    assert len(manifest["frames"]) == 2
    for report in manifest["frames"]:
        assert report["scale"] > 0
        assert report["qc_approved"] is True
        assert report["depth_path"].startswith("dense_depth/")
        assert report["confidence_path"].startswith("dense_confidence/")
        assert report["qc_mask_path"].startswith("dense_qc/")

    for index, true_depth in enumerate(true_depths):
        dense_png = cv2.imread(str(output_dir / "dense_depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        assert dense_png.shape == (VIDEO_HEIGHT, VIDEO_WIDTH)
        dense_m = dense_png.astype(np.float32) / 1000.0
        assert np.abs(dense_m - true_depth).mean() < 0.05
        assert (output_dir / "dense_confidence" / f"{index:06d}.png").exists()
        assert (output_dir / "dense_qc" / f"{index:06d}.png").exists()


def test_densify_capture_records_alignment_rejection_and_continues(stray_capture, tmp_path):
    capture = require_lidar_capture(stray_capture)

    depth0 = cv2.imread(str(stray_capture / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth1 = cv2.imread(str(stray_capture / "depth" / "000001.png"), cv2.IMREAD_UNCHANGED)
    # The second prediction is anti-correlated with its LiDAR depth, causing
    # the robust scale/shift fit to reject it with a negative scale.
    model = FakeDepthModel(
        [_full_res_ramp(depth0), np.fliplr(_full_res_ramp(depth1))],
        bias=0.0,
    )
    output_dir = tmp_path / "out"

    densify_capture(capture, model, output_dir)

    manifest = json.loads((output_dir / "densify_manifest.json").read_text())
    assert manifest["frame_count"] == 2
    assert manifest["frames"][0]["status"] == "qc_approved"
    rejected = manifest["frames"][1]
    assert rejected["index"] == 1
    assert rejected["status"] == "rejected"
    assert rejected["qc_approved"] is False
    assert "outside the plausible range" in rejected["qc_reason"]
    assert not (output_dir / "dense_depth" / "000001.png").exists()
    assert manifest["population"]["selected_frames"] == 2
    assert manifest["population"]["densified_frames"] == 1
    assert manifest["population"]["qc_approved_frames"] == 1
    assert manifest["population"]["rejected_frames"] == 1


def test_densify_capture_can_emit_manifest_declared_scaled_rgb(stray_capture, tmp_path):
    capture = require_lidar_capture(stray_capture)
    depth = cv2.imread(str(stray_capture / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    model = FakeDepthModel([cv2.resize(_full_res_ramp(depth), (32, 24), interpolation=cv2.INTER_AREA)])
    output_dir = tmp_path / "out"

    densify_capture(capture, model, output_dir, indices=[0], output_scale=0.5)

    manifest = json.loads((output_dir / "densify_manifest.json").read_text())
    assert manifest["dense_rgb_scale"] == [0.5, 0.5]
    report = manifest["frames"][0]
    assert report["registration_alignment"]["dense_output"] == "scaled_rgb"
    assert report["source_rgb_resolution"] == [VIDEO_WIDTH, VIDEO_HEIGHT]
    assert report["depth_resolution"] == [32, 24]
    dense = cv2.imread(str(output_dir / "dense_depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert dense.shape == (24, 32)


def test_cli_densify_reports_missing_torch(lidar_stray_scanner_dataset, tmp_path, capsys):
    exit_code = run_densify(
        lidar_stray_scanner_dataset, tmp_path / "out", "metric3d_vit_small",
        min_confidence=1, max_depth=8.0, guide_radius=20, guide_eps=100.0,
    )

    assert exit_code == 1
    assert "error" in capsys.readouterr().err.lower()


def test_densify_indices_sample_real_depth_frame_ids(stray_capture):
    capture = require_lidar_capture(stray_capture)

    assert _densify_indices(capture, 1) is None
    assert _densify_indices(capture, 2) == [0]
