"""Volumetric fusion of posed depth frames into a mesh and point cloud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .ingest import CaptureBundle, iter_frames, open3d_intrinsics


@dataclass
class Reconstruction:
    mesh: o3d.geometry.TriangleMesh
    cloud: o3d.geometry.PointCloud
    frame_count: int


def fuse(
    bundle: CaptureBundle,
    indices: list[int] | np.ndarray | None = None,
    poses: np.ndarray | None = None,
    voxel_size: float = 0.02,
    sdf_trunc: float | None = None,
    min_confidence: int = 1,
    max_depth: float = 3.5,
) -> Reconstruction:
    """TSDF-fuse `indices` of `bundle` into a mesh.

    `poses` overrides the bundle's poses (indexed the same way), which is how
    refined trajectories are fed back in without touching the parsed capture.
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc if sdf_trunc is not None else voxel_size * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsics = open3d_intrinsics(bundle)
    pose_table = bundle.poses if poses is None else poses

    count = 0
    for frame in iter_frames(
        bundle, indices, min_confidence=min_confidence, max_depth=max_depth
    ):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color)),
            o3d.geometry.Image(frame.depth),
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        # Open3D integrates with world-to-camera extrinsics.
        volume.integrate(rgbd, intrinsics, np.linalg.inv(pose_table[frame.index]))
        count += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    return Reconstruction(mesh=mesh, cloud=cloud, frame_count=count)


def backproject(
    frame,
    intrinsics: np.ndarray,
    pose: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a frame's depth to 3D points, plus their pixel colors.

    Returns points in world coordinates when `pose` is given, camera
    coordinates otherwise.  Invalid (zero) depth is dropped.
    """
    depth = frame.depth[::stride, ::stride]
    color = frame.color[::stride, ::stride]
    height, width = depth.shape

    us, vs = np.meshgrid(
        np.arange(width) * stride, np.arange(height) * stride
    )
    valid = depth > 0
    z = depth[valid]
    x = (us[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (vs[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack([x, y, z], axis=1)

    if pose is not None:
        points = points @ pose[:3, :3].T + pose[:3, 3]
    return points, color[valid].astype(np.float32) / 255.0
