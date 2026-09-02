import json

import cv2
import numpy as np
import pytest

from cozmo_ai_v2.pipeline import export
from cozmo_ai_v2.pipeline.diagnostics import TSDFVariant, compare_tsdf_parameters
from cozmo_ai_v2.pipeline.frame_contract import build_frame_contract
from cozmo_ai_v2.pipeline.fuse import fuse
from cozmo_ai_v2.pipeline.ingest import load_capture


def _write_dense_artifacts(
    output_dir, indices=(0, 1), approved=(True, True), shape=(48, 64), rgb_scale=None,
):
    dense_dir = output_dir / "dense_depth"
    confidence_dir = output_dir / "dense_confidence"
    qc_dir = output_dir / "dense_qc"
    dense_dir.mkdir(parents=True)
    confidence_dir.mkdir()
    qc_dir.mkdir()
    reports = []
    for index, is_approved in zip(indices, approved):
        depth = np.full(shape, 2_000 + index * 500, dtype=np.uint16)
        cv2.imwrite(str(dense_dir / f"{index:06d}.png"), depth)
        cv2.imwrite(
            str(confidence_dir / f"{index:06d}.png"),
            np.full(depth.shape, 2, dtype=np.uint8),
        )
        cv2.imwrite(
            str(qc_dir / f"{index:06d}.png"),
            np.full(depth.shape, 255, dtype=np.uint8),
        )
        reports.append(
            {
                "index": index,
                "status": "qc_approved" if is_approved else "rejected",
                "qc_approved": is_approved,
                "qc_reason": "synthetic rejection" if not is_approved else "",
                "depth_path": f"dense_depth/{index:06d}.png",
                "confidence_path": f"dense_confidence/{index:06d}.png",
                "qc_mask_path": f"dense_qc/{index:06d}.png",
                "depth_unit": "mm",
            }
        )
    manifest = output_dir / "densify_manifest.json"
    payload = {"frames": reports}
    if rgb_scale is not None:
        payload["dense_rgb_scale"] = list(rgb_scale)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return dense_dir, manifest


def test_contract_uses_qc_dense_depth_and_rgb_intrinsics(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4")
    bundle = load_capture(stray_capture)

    contract = build_frame_contract(
        bundle,
        indices=[1, 0, 1],
        poses=bundle.poses,
        pose_source="arkit",
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        max_depth=4.0,
    )
    frames = list(contract.iter_frames())

    assert [frame.index for frame in frames] == [0, 1]
    assert all(frame.depth.shape == (48, 64) for frame in frames)
    assert all(frame.color.shape == (48, 64, 3) for frame in frames)
    assert all(frame.provenance.depth_source == "metric3d_v2_scale_shift_lidar_residual" for frame in frames)
    assert all(frame.provenance.source_depth_unit == "mm" for frame in frames)
    assert all(frame.provenance.contract_depth_unit == "m" for frame in frames)
    assert all(frame.provenance.registration == "dense_native_rgb_exact_shape" for frame in frames)
    np.testing.assert_allclose(frames[0].intrinsics, bundle.rgb_intrinsics)
    report = contract.report()
    assert report["integrated_indices"] == []
    assert report["depth_unit"] == "m"
    assert report["confidence_threshold"] == 1
    assert report["max_depth_m"] == 4.0
    assert report["contract_parameters"]["max_depth_m"] == 4.0
    assert report["contract_parameters"]["min_confidence"] == 1
    assert report["registration_alignment"]["dense_depth"]["shape_mismatch"] == "reject_unless_manifest_declared_scale"
    assert report["frame_provenance"][0]["depth_resolution"] == [64, 48]
    assert report["population"] == {
        "input_frames": 2,
        "selected_frames": 2,
        "densified_frames": 2,
        "qc_approved_dense_frames": 2,
        "fused_frames": 0,
        "fused_dense_frames": 0,
        "fused_raw_frames": 0,
        "rejected_frames": 0,
        "dense_qc_rejected_frames": 0,
        "rejected_frames_total": 0,
        "fallback_frames": 0,
    }


def test_contract_uses_manifest_declared_scaled_dense_rgb(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(
        tmp_path / "stage4", indices=(0,), approved=(True,), shape=(24, 32), rgb_scale=(0.5, 0.5),
    )
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(
        bundle, indices=[0], dense_depth_dir=dense_dir, densify_manifest=manifest, max_depth=4.0,
    )

    frame = next(contract.iter_frames())
    assert frame.depth.shape == (24, 32)
    assert frame.color.shape == (24, 32, 3)
    assert frame.provenance.registration == "dense_scaled_rgb_area_resize"
    np.testing.assert_allclose(frame.intrinsics, bundle.intrinsics_for_size(32, 24))


def test_contract_falls_back_per_index_to_raw_lidar(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4", approved=(False, True))
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(
        bundle,
        poses=bundle.poses,
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        max_depth=4.0,
    )

    frames = list(contract.iter_frames())

    assert [frame.index for frame in frames] == [0, 1]
    assert frames[0].provenance.depth_source == "raw_lidar"
    assert frames[0].depth.shape == (12, 16)
    assert frames[1].provenance.depth_source == "metric3d_v2_scale_shift_lidar_residual"
    report = contract.report()
    assert report["fallback_frames"][0]["index"] == 0
    assert report["rejected_frames"] == []
    assert [item["index"] for item in report["frame_provenance"]] == [0, 1]


def test_contract_can_force_raw_source_for_ablation(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4")
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(
        bundle,
        indices=[0],
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        depth_source="raw",
    )

    frame = next(contract.iter_frames())

    assert frame.index == 0
    assert frame.provenance.depth_source == "raw_lidar"
    assert contract.report()["depth_source_mode"] == "raw"
    assert contract.report()["contract_parameters"]["depth_source_policy"] == "raw"


def test_dense_confidence_threshold_is_part_of_qc_policy(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4")
    cv2.imwrite(
        str(dense_dir.parent / "dense_confidence" / "000000.png"),
        np.zeros((48, 64), dtype=np.uint8),
    )
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(
        bundle,
        indices=[0],
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        min_confidence=2,
    )

    frame = next(contract.iter_frames())

    assert frame.provenance.depth_source == "raw_lidar"
    assert contract.report()["contract_parameters"]["min_confidence"] == 2
    assert contract.report()["fallback_frames"][0]["index"] == 0


def test_contract_records_default_3_5m_range_filter(stray_capture):
    depth_path = stray_capture / "depth" / "000000.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth[0, 0] = 4000
    cv2.imwrite(str(depth_path), depth)
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(bundle, indices=[0])

    frame = next(contract.iter_frames())
    report = contract.report()

    assert frame.depth[0, 0] == 0.0
    assert report["contract_parameters"]["max_depth_m"] == 3.5
    assert report["contract_parameters"]["max_depth_inclusive"] is True


def test_contract_reports_one_frame_short_video_without_shifting_sidecars(
    stray_capture,
):
    video_path = stray_capture / "rgb.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"avc1"), 30, (64, 48)
    )
    if not writer.isOpened():
        writer.release()
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
        )
    assert writer.isOpened()
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    bundle = load_capture(stray_capture)
    contract = build_frame_contract(bundle, indices=[0], max_depth=4.0)
    frames = list(contract.iter_frames())

    # The only decoded frame remains sidecar index 0. In particular, its
    # depth is not silently replaced with frame 1's 2.3--2.7 m raster.
    assert [frame.index for frame in frames] == [0]
    assert frames[0].depth.max() == pytest.approx(2.2, abs=1e-3)

    availability = contract.report()["video_availability"]
    assert availability["expected_frame_count"] == 2
    assert availability["sidecar_frame_count"] == 2
    assert availability["reported_frame_count"] is not None
    assert availability["decoded_frame_count"] == 1
    assert availability["missing_indices"] == [1]
    assert availability["terminal_decode_missing"] is True
    assert availability["decode_complete"] is True
    assert availability["association_mode"] == "pts"
    assert availability["associations"][0]["sidecar_index"] == 0


def test_contract_rejects_dense_without_qc_when_raw_is_missing(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4", indices=(0,), approved=(True,))
    bundle = load_capture(stray_capture)
    for path in (stray_capture / "depth").glob("*.png"):
        path.unlink()
    for path in (stray_capture / "confidence").glob("*.png"):
        path.unlink()
    (dense_dir / "000000.png").unlink()

    contract = build_frame_contract(
        bundle,
        poses=bundle.poses,
        indices=[0],
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
    )

    assert list(contract.iter_frames()) == []
    rejection = contract.report()["rejected_frames"][0]
    assert rejection["index"] == 0
    assert "dense frame rejected" in rejection["reason"]


def test_slam_pose_selection_is_explicit_and_deterministic(stray_capture):
    pose_path = stray_capture / "slam_poses.csv"
    pose_path.write_text(
        "timestamp,x,y,z,qx,qy,qz,qw\n"
        "10.0,10.0,0,0,0,0,0,1\n"
        "10.1,11.0,0,0,0,0,0,1\n"
    )

    bundle = load_capture(stray_capture, pose_source="slam")

    assert bundle.pose_source == "slam"
    assert bundle.pose_path == str(pose_path)
    np.testing.assert_allclose(bundle.poses[:, 0, 3], [10.0, 11.0])
    assert load_capture(stray_capture).pose_source == "arkit"


def test_precomputed_dense_depth_can_drive_capture_without_raw_lidar(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4", indices=(0, 1))
    for path in (stray_capture / "depth").glob("*.png"):
        path.unlink()
    for path in (stray_capture / "confidence").glob("*.png"):
        path.unlink()

    bundle = load_capture(stray_capture, dense_depth_dir=dense_dir)
    contract = build_frame_contract(
        bundle,
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        max_depth=4.0,
    )

    frames = list(contract.iter_frames())

    assert bundle.has_depth is False
    assert [frame.index for frame in frames] == [0, 1]
    assert all(frame.provenance.depth_source != "raw_lidar" for frame in frames)


def test_fuse_extracts_and_exports_cloud_and_mesh(stray_capture, tmp_path):
    dense_dir, manifest = _write_dense_artifacts(tmp_path / "stage4")
    bundle = load_capture(stray_capture)
    contract = build_frame_contract(
        bundle,
        indices=[0, 1],
        poses=bundle.poses,
        pose_source="arkit",
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        max_depth=4.0,
    )

    reconstruction = fuse(
        bundle,
        indices=np.array([0, 1]),
        frame_contract=contract,
        voxel_size=0.1,
        sdf_trunc=0.2,
        max_depth=4.0,
    )
    paths = export.export_reconstruction(reconstruction, tmp_path / "out")

    assert reconstruction.frame_count == 2
    assert reconstruction.frame_indices == (0, 1)
    assert len(np.asarray(reconstruction.cloud.points)) > 0
    assert paths["cloud"].exists()
    assert paths["mesh"].exists()
    fusion_manifest = json.loads(paths["manifest"].read_text())
    assert fusion_manifest["contract"]["pose_source"] == "arkit"
    assert fusion_manifest["contract"]["depth_sources"] == [
        "metric3d_v2_scale_shift_lidar_residual"
    ]
    assert fusion_manifest["contract"]["tsdf_parameters"] == {
        "voxel_size_m": 0.1,
        "sdf_trunc_m": 0.2,
        "sdf_trunc_explicit": True,
        "depth_scale": 1.0,
        "depth_unit": "m",
        "depth_trunc_m": 4.0,
        "color_type": "RGB8",
        "extrinsic_convention": "inverse_c2w_opencv_to_world_to_camera",
    }


def test_tsdf_variant_comparison_persists_effective_parameters(stray_capture):
    bundle = load_capture(stray_capture)

    records = compare_tsdf_parameters(
        bundle,
        [
            TSDFVariant("baseline", 0.1, 0.2),
            TSDFVariant("coarse", 0.2, 0.4),
        ],
        indices=[0],
        max_depth=3.5,
    )

    assert [record["label"] for record in records] == ["baseline", "coarse"]
    assert [record["tsdf_parameters"]["voxel_size_m"] for record in records] == [0.1, 0.2]
    assert [record["tsdf_parameters"]["sdf_trunc_m"] for record in records] == [0.2, 0.4]
    assert all(record["contract"]["contract_parameters"]["max_depth_m"] == 3.5 for record in records)
    assert all(record["contract"]["video_availability"]["association_mode"] == "pts" for record in records)
