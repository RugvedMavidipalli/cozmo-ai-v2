"""Trajectory refinement: keyframing, loop closure, and pose-graph optimisation.

ARKit's VIO is locally excellent and globally drifting.  Over a walkthrough the
drift is what smears a wall across its several visits, so the fitted plane sits
between them and every measurement taken from it inherits the error.  Replaying
ARKit poses therefore cannot reach a 2 cm wall tolerance; the trajectory has to
be corrected against the sensor log itself.

The correction is a standard pose graph, but the parts that matter here are the
edges: sequential edges refined by ICP, plus loop-closure edges between frames
that are *spatially* near and *temporally* far.  Those are the only edges
carrying information ARKit does not already have, and they are what pull the
end of the walk back onto the start.

Open3D pose-graph conventions used throughout: a node's pose is camera-to-world,
and an edge (i -> j) stores `inv(pose_j) @ pose_i`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .ingest import CaptureBundle, iter_frames


@dataclass
class DriftReport:
    """Per-stage trajectory error, for the report's error budget."""

    keyframe_count: int
    loop_edges: int
    loop_candidates: int
    sequential_rmse: float
    loop_closure_gap_before: float
    loop_closure_gap_after: float
    mean_correction: float
    max_correction: float
    stages: dict[str, float] = field(default_factory=dict)


def select_keyframes(
    bundle: CaptureBundle,
    translation_step: float = 0.18,
    rotation_step_degrees: float = 12.0,
    max_gap: int = 60,
) -> np.ndarray:
    """Frames spaced by motion rather than by time.

    A time-strided selection over-samples where the operator paused and
    under-samples where they swung the camera through a doorway -- exactly
    where registration needs support.
    """
    positions = bundle.poses[:, :3, 3]
    rotations = bundle.poses[:, :3, :3]
    rotation_threshold = np.radians(rotation_step_degrees)

    selected = [0]
    anchor = 0
    for index in range(1, len(bundle)):
        translated = np.linalg.norm(positions[index] - positions[anchor])
        relative = rotations[anchor].T @ rotations[index]
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        if (
            translated > translation_step
            or angle > rotation_threshold
            or index - anchor >= max_gap
        ):
            selected.append(index)
            anchor = index
    return np.asarray(selected)


def _keyframe_clouds(
    bundle: CaptureBundle,
    keyframes: np.ndarray,
    voxel_size: float,
    min_confidence: int,
    max_depth: float,
) -> dict[int, o3d.geometry.PointCloud]:
    """Per-keyframe clouds in their own camera frame, downsampled for ICP."""
    intrinsics = bundle.intrinsics
    clouds: dict[int, o3d.geometry.PointCloud] = {}
    for frame in iter_frames(
        bundle, keyframes, min_confidence=min_confidence, max_depth=max_depth
    ):
        valid = frame.depth > 0
        if valid.sum() < 500:
            continue
        vs, us = np.nonzero(valid)
        z = frame.depth[valid]
        points = np.stack(
            [
                (us - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (vs - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ],
            axis=1,
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(voxel_size)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
        )
        clouds[frame.index] = cloud
    return clouds


def _pairwise_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Point-to-plane ICP, coarse then fine, with an edge information matrix.

    Point-to-plane is the right metric indoors: it lets flat surfaces slide
    over each other instead of forcing spurious point correspondences, which
    is what keeps a featureless wall from dragging the solution sideways.

    The returned information matrix is what makes the pose graph solvable.  It
    encodes how strongly each edge constrains each degree of freedom -- derived
    from the actual correspondences -- so a sparse loop edge cannot outvote a
    dense sequential one.  Weighting every edge equally instead makes the
    optimiser diverge (metre-scale corrections on this capture).
    """
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold * 3,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    information = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, result.transformation
        )
    )
    return (
        result.transformation,
        float(result.fitness),
        float(result.inlier_rmse),
        information,
    )


def find_loop_candidates(
    bundle: CaptureBundle,
    keyframes: np.ndarray,
    max_distance: float = 2.0,
    min_time_gap: float = 4.0,
    max_view_angle_degrees: float = 70.0,
    max_candidates: int = 220,
) -> list[tuple[int, int]]:
    """Keyframe pairs that plausibly see the same surfaces on separate visits.

    Gating on view direction as well as position matters: two poses a metre
    apart facing opposite walls of a corridor share no geometry, and an ICP
    attempt between them is both wasted work and a source of false edges.
    """
    positions = bundle.poses[keyframes][:, :3, 3]
    # Camera looks down +Z in the OpenCV convention this pipeline uses.
    directions = bundle.poses[keyframes][:, :3, 2]
    times = bundle.timestamps[keyframes]
    cosine_limit = np.cos(np.radians(max_view_angle_degrees))

    scored: list[tuple[float, int, int]] = []
    for i in range(len(keyframes)):
        separation = np.linalg.norm(positions - positions[i], axis=1)
        near = np.flatnonzero(
            (separation < max_distance)
            & (np.abs(times - times[i]) > min_time_gap)
            & (np.arange(len(keyframes)) > i)
        )
        for j in near:
            if directions[i] @ directions[j] < cosine_limit:
                continue
            scored.append((separation[j], i, j))

    scored.sort()
    return [(i, j) for _, i, j in scored[:max_candidates]]


def refine_trajectory(
    bundle: CaptureBundle,
    keyframes: np.ndarray | None = None,
    voxel_size: float = 0.05,
    min_confidence: int = 1,
    max_depth: float = 5.0,
    loop_fitness_threshold: float = 0.5,
    max_loop_correction: float = 0.5,
    enable_loop_closure: bool = True,
) -> tuple[np.ndarray, DriftReport]:
    """Return refined camera-to-world poses for every frame, plus a drift report.

    Keyframes are optimised; intermediate frames inherit their correction by
    interpolation, so the returned array is indexed exactly like
    `bundle.poses` and downstream code needs no special cases.
    """
    if keyframes is None:
        keyframes = select_keyframes(bundle)
    clouds = _keyframe_clouds(bundle, keyframes, voxel_size, min_confidence, max_depth)
    keyframes = np.asarray([k for k in keyframes if k in clouds])

    graph = o3d.pipelines.registration.PoseGraph()
    for keyframe in keyframes:
        graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(bundle.poses[keyframe].copy())
        )

    threshold = voxel_size * 1.5
    sequential_errors: list[float] = []
    for position in range(len(keyframes) - 1):
        source_id, target_id = position, position + 1
        source, target = keyframes[source_id], keyframes[target_id]
        init = np.linalg.inv(bundle.poses[target]) @ bundle.poses[source]
        transformation, _, rmse, information = _pairwise_icp(
            clouds[source], clouds[target], init, threshold
        )
        sequential_errors.append(rmse)
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id, transformation, information, uncertain=False
            )
        )

    candidates = (
        find_loop_candidates(bundle, keyframes) if enable_loop_closure else []
    )
    accepted = 0
    for source_id, target_id in candidates:
        source, target = keyframes[source_id], keyframes[target_id]
        init = np.linalg.inv(bundle.poses[target]) @ bundle.poses[source]
        transformation, fitness, _, information = _pairwise_icp(
            clouds[source], clouds[target], init, threshold
        )
        if fitness < loop_fitness_threshold:
            continue
        # A loop edge should be a correction, not a teleport.  ICP between two
        # views of a repetitive interior can converge a room's width away
        # (corridors and matching doorways look alike); such an edge is
        # confidently wrong and drags the whole graph with it.
        displacement = np.linalg.norm(
            (transformation @ np.linalg.inv(init))[:3, 3]
        )
        if displacement > max_loop_correction:
            continue
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id, transformation, information, uncertain=True
            )
        )
        accepted += 1

    o3d.pipelines.registration.global_optimization(
        graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=threshold,
            edge_prune_threshold=0.25,
            reference_node=0,
        ),
    )

    refined_keyframe_poses = np.asarray([node.pose for node in graph.nodes])
    poses = _propagate(bundle.poses, keyframes, refined_keyframe_poses)

    corrections = np.linalg.norm(
        refined_keyframe_poses[:, :3, 3] - bundle.poses[keyframes][:, :3, 3], axis=1
    )
    report = DriftReport(
        keyframe_count=len(keyframes),
        loop_edges=accepted,
        loop_candidates=len(candidates),
        sequential_rmse=float(np.mean(sequential_errors)) if sequential_errors else 0.0,
        loop_closure_gap_before=float(
            np.linalg.norm(bundle.poses[-1][:3, 3] - bundle.poses[0][:3, 3])
        ),
        loop_closure_gap_after=float(
            np.linalg.norm(poses[-1][:3, 3] - poses[0][:3, 3])
        ),
        mean_correction=float(corrections.mean()),
        max_correction=float(corrections.max()),
    )
    return poses, report


def _propagate(
    original: np.ndarray, keyframes: np.ndarray, refined: np.ndarray
) -> np.ndarray:
    """Carry keyframe corrections to every frame.

    The correction is applied in the world frame and interpolated between
    bracketing keyframes -- rotations via a shortest-arc quaternion slerp,
    translations linearly -- so that intermediate frames keep ARKit's locally
    accurate relative motion while inheriting the global fix.
    """
    corrections = refined @ np.linalg.inv(original[keyframes])
    quaternions = np.asarray([_matrix_to_quaternion(c[:3, :3]) for c in corrections])
    translations = corrections[:, :3, 3]

    poses = original.copy()
    for position in range(len(keyframes)):
        start = keyframes[position]
        end = keyframes[position + 1] if position + 1 < len(keyframes) else len(original)
        span = max(end - start, 1)
        for index in range(start, min(end, len(original))):
            if position + 1 < len(keyframes):
                blend = (index - start) / span
                quaternion = _slerp(
                    quaternions[position], quaternions[position + 1], blend
                )
                translation = (
                    translations[position] * (1 - blend)
                    + translations[position + 1] * blend
                )
            else:
                quaternion, translation = quaternions[position], translations[position]
            correction = np.eye(4)
            correction[:3, :3] = _quaternion_to_matrix(quaternion)
            correction[:3, 3] = translation
            poses[index] = correction @ original[index]
    return poses


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x, y, z = 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (
                matrix[0, 2] + matrix[2, 0]
            ) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x, y, z = (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (
                matrix[1, 2] + matrix[2, 1]
            ) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x, y, z = (matrix[0, 2] + matrix[2, 0]) / scale, (
                matrix[1, 2] + matrix[2, 1]
            ) / scale, 0.25 * scale
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(a @ b)
    if dot < 0:  # take the shortest arc
        b, dot = -b, -dot
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    return (
        np.sin((1 - t) * theta) / sin_theta * a + np.sin(t * theta) / sin_theta * b
    )
