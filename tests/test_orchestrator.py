from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cozmo_ai_v2.cli import build_parser
from cozmo_ai_v2.pipeline.orchestrator import run_pipeline
from cozmo_ai_v2.pipeline.stage_manifest import StageManifest


def _write_pose_table(path: Path, count: int) -> None:
    poses = []
    for index in range(count):
        pose = np.eye(4)
        pose[0, 3] = index * 0.01
        poses.append(pose.tolist())
    path.write_text(json.dumps({"timestamps": (np.arange(count) / 30.0).tolist(), "poses": poses}))


def _write_dense_artifact(root: Path, count: int, width: int = 64, height: int = 48) -> tuple[Path, Path]:
    dense_root = root / "densify"
    dense = dense_root / "dense_depth"
    confidence = dense_root / "dense_confidence"
    qc = dense_root / "dense_qc"
    for directory in (dense, confidence, qc):
        directory.mkdir(parents=True)
    frames = []
    for index in range(0, count, 2):
        cv2.imwrite(str(dense / f"{index:06d}.png"), np.full((height, width), 2000, np.uint16))
        cv2.imwrite(str(confidence / f"{index:06d}.png"), np.full((height, width), 2, np.uint8))
        cv2.imwrite(str(qc / f"{index:06d}.png"), np.full((height, width), 255, np.uint8))
        frames.append({
            "index": index,
            "status": "qc_approved",
            "qc_approved": True,
            "depth_path": f"dense_depth/{index:06d}.png",
            "confidence_path": f"dense_confidence/{index:06d}.png",
            "qc_mask_path": f"dense_qc/{index:06d}.png",
            "depth_unit": "mm",
        })
    manifest = dense_root / "densify_manifest.json"
    manifest.write_text(json.dumps({
        "frames": frames,
        "depth_provenance": "metric3d_v2_monocular_uncalibrated",
        "dense_rgb_scale": [1.0, 1.0],
    }))
    return dense, manifest


def test_stage_manifest_records_order_and_terminal_status(tmp_path):
    manifest = StageManifest(tmp_path / "input.mp4", tmp_path / "out")
    with manifest.stage("input_detection"):
        pass
    with manifest.stage("ingest_qc"):
        pass
    manifest.unavailable("depth", "model path was not supplied")
    manifest.finalize("completed")

    payload = json.loads((tmp_path / "out" / "stage_manifest.json").read_text())
    assert [item["stage"] for item in payload["stages"]] == [
        "input_detection", "ingest_qc", "depth"
    ]
    assert [item["status"] for item in payload["stages"]] == [
        "completed", "completed", "unavailable"
    ]
    assert payload["status"] == "completed"


def test_one_command_hands_off_rgb_pose_and_depth_artifacts(
    synthetic_video, tmp_path, monkeypatch
):
    pose_path = tmp_path / "poses.json"
    _write_pose_table(pose_path, 2)
    dense_dir, dense_manifest = _write_dense_artifact(tmp_path, 2)
    calls = {}

    def fake_model(_args):
        return object()

    def fake_depth(video, matrix, model, output_dir, indices, max_depth):
        calls["depth"] = (video, matrix, model, output_dir, indices, max_depth)
        return dense_dir, dense_manifest

    def fake_run(reconstruction_args):
        calls["reconstruction"] = reconstruction_args
        output = Path(reconstruction_args.out)
        for name in ("result.json", "floorplan.svg", "scene.glb", "cloud.ply", "mesh.ply", "planes.json", "fusion_manifest.json", "openings.csv"):
            (output / name).write_text("artifact")
        return 0

    monkeypatch.setattr("cozmo_ai_v2.pipeline.orchestrator._model_or_error", fake_model)
    monkeypatch.setattr("cozmo_ai_v2.pipeline.orchestrator._generate_monocular_dense", fake_depth)
    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.run", fake_run)

    args = build_parser().parse_args([
        "pipeline", str(synthetic_video), "--out", str(tmp_path / "out"),
        "--slam-poses", str(pose_path), "--stride", "2",
    ])
    assert run_pipeline(args) == 0
    reconstruction = calls["reconstruction"]
    assert reconstruction.pose_source == "slam"
    assert reconstruction.dense_depth_dir == dense_dir
    assert reconstruction.densify_manifest == dense_manifest
    assert reconstruction.slam_poses == tmp_path / "out" / "preprocessing" / "capture" / "slam_poses.json"
    assert calls["depth"][4] == {0}

    manifest = json.loads((tmp_path / "out" / "stage_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert all(item["status"] in {"completed", "unavailable"} for item in manifest["stages"])
    assert [item["stage"] for item in manifest["stages"]][:4] == [
        "input_detection", "ingest_qc", "poses", "depth"
    ]


def test_required_rgb_depth_failure_is_nonzero_and_recorded(synthetic_video, tmp_path, monkeypatch):
    pose_path = tmp_path / "poses.json"
    _write_pose_table(pose_path, 2)

    def fail_model(_args):
        raise RuntimeError("Metric3D weights unavailable")

    monkeypatch.setattr("cozmo_ai_v2.pipeline.orchestrator._model_or_error", fail_model)
    args = build_parser().parse_args([
        "pipeline", str(synthetic_video), "--out", str(tmp_path / "out"),
        "--slam-poses", str(pose_path),
    ])

    assert run_pipeline(args) == 1
    manifest = json.loads((tmp_path / "out" / "stage_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "Metric3D weights unavailable" in manifest["failure_reason"]
    assert any(item["stage"] == "depth" and item["status"] == "failed" for item in manifest["stages"])


def test_explicit_rgb_dense_artifact_is_handed_off_without_model(
    synthetic_video, tmp_path, monkeypatch
):
    pose_path = tmp_path / "poses.json"
    _write_pose_table(pose_path, 2)
    dense_dir, dense_manifest = _write_dense_artifact(tmp_path, 2)
    calls = {}

    def fail_model(_args):
        raise AssertionError("an approved explicit dense artifact must not load Metric3D")

    def fake_run(reconstruction_args):
        calls["reconstruction"] = reconstruction_args
        output = Path(reconstruction_args.out)
        for name in (
            "result.json", "floorplan.svg", "scene.glb", "cloud.ply", "mesh.ply",
            "planes.json", "fusion_manifest.json", "openings.csv",
        ):
            (output / name).write_text("artifact")
        return 0

    monkeypatch.setattr("cozmo_ai_v2.pipeline.orchestrator._model_or_error", fail_model)
    monkeypatch.setattr("cozmo_ai_v2.pipeline.cli.run", fake_run)
    args = build_parser().parse_args([
        "pipeline", str(synthetic_video), "--out", str(tmp_path / "out"),
        "--slam-poses", str(pose_path), "--dense-depth-dir", str(dense_dir),
        "--densify-manifest", str(dense_manifest),
    ])

    assert run_pipeline(args) == 0
    assert calls["reconstruction"].dense_depth_dir == dense_dir
    manifest = json.loads((tmp_path / "out" / "stage_manifest.json").read_text())
    assert manifest["status"] == "completed"
