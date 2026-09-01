from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .frame_contract import FrameContract, build_frame_contract
from .ingest import CaptureBundle


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
    frame_indices: tuple[int, ...] = ()
    contract_report: dict | None = None


def fuse(
    bundle: CaptureBundle,
    indices: list[int] | np.ndarray | None = None,
    poses: np.ndarray | None = None,
    voxel_size: float = 0.02,
    sdf_trunc: float | None = None,
    min_confidence: int = 1,
    max_depth: float = 3.5,
    frame_contract: FrameContract | None = None,
    dense_depth_dir: str | None = None,
    densify_manifest: str | None = None,
    pose_source: str | None = None,
    depth_source: str = "auto",
    frame_association: str = "pts",
    pts_tolerance_s: float | None = None,
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
        depth_source: ``auto`` uses QC-approved dense depth and falls back to
            same-index raw LiDAR; ``dense`` and ``raw`` force an ablation
            source.
        frame_association: ``pts`` associates decoded frames to sidecars by
            presentation timestamp; ``index`` selects the legacy identity
            mapping.
        pts_tolerance_s: Optional maximum timestamp distance for PTS matching.

    Returns:
        A `Reconstruction` holding the fused mesh, the fused point cloud,
        and how many frames actually made it into the volume.
    """
    if voxel_size <= 0 or not np.isfinite(voxel_size):
        raise ValueError(f"voxel_size must be a positive finite number, got {voxel_size}")
    if sdf_trunc is not None and (sdf_trunc <= 0 or not np.isfinite(sdf_trunc)):
        raise ValueError(f"sdf_trunc must be a positive finite number, got {sdf_trunc}")
    if frame_contract is None:
        frame_contract = build_frame_contract(
            bundle,
            indices=indices,
            poses=poses,
            pose_source=pose_source,
            dense_depth_dir=dense_depth_dir,
            densify_manifest=densify_manifest,
            min_confidence=min_confidence,
            max_depth=max_depth,
            depth_source=depth_source,
            frame_association=frame_association,
            pts_tolerance_s=pts_tolerance_s,
        )

    effective_max_depth = (
        frame_contract.max_depth if frame_contract is not None else max_depth
    )
    effective_sdf_trunc = sdf_trunc if sdf_trunc is not None else voxel_size * 4
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=effective_sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    count = 0
    integrated_indices: list[int] = []
    resolutions: set[tuple[int, int]] = set()
    for frame in frame_contract.iter_frames(indices):
        resolutions.add((int(frame.depth.shape[1]), int(frame.depth.shape[0])))
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            frame.depth.shape[1], frame.depth.shape[0],
            frame.intrinsics[0, 0], frame.intrinsics[1, 1],
            frame.intrinsics[0, 2], frame.intrinsics[1, 2],
        )
        # depth_scale=1.0 because `frame.depth` is already stored in
        # metres, not in raw sensor units that would need converting.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(frame.color)),
            o3d.geometry.Image(np.ascontiguousarray(frame.depth)),
            depth_scale=1.0,
            depth_trunc=effective_max_depth,
            convert_rgb_to_intensity=False,
        )
        # TSDF fusion wants the world-to-camera transform for each frame,
        # which is why the stored camera-to-world pose gets inverted here.
        volume.integrate(rgbd, intrinsics, np.linalg.inv(frame.pose))
        count += 1
        integrated_indices.append(frame.index)
        frame_contract.integrated_indices.add(frame.index)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud = volume.extract_point_cloud()
    report = frame_contract.report()
    report["frame_count"] = count
    report["resolutions"] = [list(size) for size in sorted(resolutions)]
    report["tsdf_parameters"] = {
        "voxel_size_m": float(voxel_size),
        "sdf_trunc_m": float(effective_sdf_trunc),
        "sdf_trunc_explicit": sdf_trunc is not None,
        "depth_scale": 1.0,
        "depth_unit": "m",
        "depth_trunc_m": float(effective_max_depth),
        "color_type": "RGB8",
        "extrinsic_convention": "inverse_c2w_opencv_to_world_to_camera",
    }
    return Reconstruction(
        mesh=mesh,
        cloud=cloud,
        frame_count=count,
        frame_indices=tuple(integrated_indices),
        contract_report=report,
    )
