from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .ingest import CaptureBundle, iter_frames, open3d_intrinsics


@dataclass
class Reconstruction:
    """The result of merging a set of depth frames into one combined 3D
    model.

    This uses a technique called TSDF fusion (short for "truncated signed
    distance function"), which works by laying an invisible 3D grid over
    the space and, for every depth frame, updating each grid cell with how
    far it is from the nearest observed surface and which side of that
    surface it's on. Averaging this across many frames smooths out the
    noise that any single depth frame would have on its own, and a normal
    mesh and point cloud can then be pulled out of the grid afterward.

    Attributes:
        mesh: A triangle mesh built from the fused volume, with the normal
            direction at each vertex already worked out.
        cloud: A point cloud built from the same fused volume.
        frame_count: How many frames actually got folded into the volume.
    """

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
    """Merges a set of a capture's depth frames into one mesh and point
    cloud, using TSDF fusion.

    Each frame only shows one view of the room; this combines all of them
    into a single, cleaned-up 3D reconstruction by integrating them one at
    a time into a shared voxel grid (see `Reconstruction`'s docstring for
    more on how that works).

    Args:
        bundle: The parsed capture to fuse frames from.
        indices: Which frame indices to fuse; if `None`, every frame in
            the capture is used.
        poses: Camera-to-world poses to use instead of `bundle.poses`,
            indexed the same way as the capture's own poses. This is how a
            caller plugs in corrected poses from pose refinement instead
            of the raw, unrefined ones.
        voxel_size: The edge length of each voxel (3D grid cell) in the
            fusion volume, in metres. Smaller values give a more detailed
            result but take longer to run and use more memory.
        sdf_trunc: How far, in metres, the signed distance function is
            allowed to range before it gets clamped. Defaults to
            `voxel_size * 4` if not given.
        min_confidence: The lowest depth-confidence level to keep.
        max_depth: The furthest depth value to include, in metres.

    Returns:
        A `Reconstruction` holding the fused mesh, the fused point cloud,
        and how many frames actually made it into the volume.
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
        # depth_scale=1.0 because `frame.depth` is already stored in
        # metres, not in raw sensor units that would need converting.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color)),
            o3d.geometry.Image(frame.depth),
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        # TSDF fusion wants the world-to-camera transform for each frame,
        # which is why the stored camera-to-world pose gets inverted here.
        volume.integrate(rgbd, intrinsics, np.linalg.inv(pose_table[frame.index]))
        count += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    return Reconstruction(mesh=mesh, cloud=cloud, frame_count=count)
