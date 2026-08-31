"""Synthetic pose-contract checks that do not need a capture or GPU."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cozmo_ai_v2.pipeline.fuse as fuse_module
from cozmo_ai_v2.pipeline.damage.fusion import project_detection
from cozmo_ai_v2.pipeline.ingest import (
    CaptureBundle,
    STRAY_ODOMETRY_CONVENTION,
)


def _bundle(poses: np.ndarray) -> CaptureBundle:
    return CaptureBundle(
        root=Path("synthetic"),
        name="synthetic",
        intrinsics=np.eye(3),
        depth_size=(1, 1),
        timestamps=np.arange(len(poses), dtype=float),
        poses=poses,
        has_depth=True,
        fps=30.0,
        gravity_up=np.array([0.0, 1.0, 0.0]),
        gravity_consistency=1.0,
    )


def test_c2w_opencv_projection_preserves_forward_axis_without_arkit_flip():
    pose = np.eye(4)
    pose[:3, 3] = [2.0, -3.0, 1.0]
    mask = np.ones((1, 1), dtype=bool)
    depth = np.array([[2.0]], dtype=np.float32)

    world, rays = project_detection(None, mask, depth, pose, np.eye(3), (1.0, 1.0))

    assert STRAY_ODOMETRY_CONVENTION == "camera_to_world_opencv_csv_no_arkit_to_cv_flip"
    np.testing.assert_allclose(world, [[2.0, -3.0, 3.0]])
    np.testing.assert_allclose(rays, [[0.0, 0.0, 1.0]])


def test_tsdf_fusion_uses_inverse_of_frame_contract_c2w_pose(monkeypatch):
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[2, :3, 3] = [1.0, 2.0, 3.0]
    bundle = _bundle(poses)
    frame = SimpleNamespace(
        index=1,
        timestamp=2.0,
        pose=poses[2],
        color=np.zeros((1, 1, 3), dtype=np.uint8),
        depth=np.ones((1, 1), dtype=np.float32),
        intrinsics=np.eye(3),
    )
    frame_contract = SimpleNamespace(
        max_depth=3.5,
        integrated_indices=set(),
        iter_frames=lambda _indices: iter([frame]),
        report=lambda: {},
    )

    class FakeMesh:
        def compute_vertex_normals(self):
            pass

    class FakeVolume:
        def __init__(self):
            self.extrinsics = []

        def integrate(self, _rgbd, _intrinsics, extrinsic):
            self.extrinsics.append(extrinsic)

        def extract_triangle_mesh(self):
            return FakeMesh()

        def extract_point_cloud(self):
            return object()

    volume = FakeVolume()
    fake_open3d = SimpleNamespace(
        pipelines=SimpleNamespace(
            integration=SimpleNamespace(
                ScalableTSDFVolume=lambda **_kwargs: volume,
                TSDFVolumeColorType=SimpleNamespace(RGB8="RGB8"),
            )
        ),
        geometry=SimpleNamespace(
            RGBDImage=SimpleNamespace(
                create_from_color_and_depth=lambda _color, _depth, **_kwargs: object()
            ),
            Image=lambda image: image,
        ),
        camera=SimpleNamespace(PinholeCameraIntrinsic=lambda *_args: object()),
    )
    monkeypatch.setattr(fuse_module, "o3d", fake_open3d)

    result = fuse_module.fuse(bundle, poses=poses, frame_contract=frame_contract)

    assert result.frame_count == 1
    np.testing.assert_allclose(volume.extrinsics, [np.linalg.inv(poses[2])])
