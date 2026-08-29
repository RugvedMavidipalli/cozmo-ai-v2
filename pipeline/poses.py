from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .ingest import CaptureBundle, iter_frames

ODOMETRY_INFORMATION = 1000.0
LOOP_INFORMATION = 10.0


@dataclass
class DriftReport:
    """A summary of what happened during pose refinement.

    This gets used to report how much the camera trajectory was
    corrected, and how much that correction can be trusted.

    Attributes:
        keyframe_count: How many keyframes were actually used to refine
            the trajectory.
        loop_edges: How many "revisit" connections were found and
            accepted -- pairs of keyframes that saw the same part of the
            room on two different passes.
        loop_candidates: How many possible revisit connections were
            considered before filtering down to `loop_edges`.
        rejected: True if the correction looked physically wrong and was
            thrown out, so the original ARKit poses were kept instead.
        sequential_rmse: How well ARKit's own frame-to-frame estimate
            matched the depth data, on average. This is only used for
            reporting, not for deciding anything.
        loop_closure_gap_before: How far apart the start and end of the
            walkthrough were, in metres, before correction.
        loop_closure_gap_after: The same distance, after correction.
        mean_correction: On average, how far each keyframe moved from its
            original position, in metres.
        max_correction: The largest single move any keyframe made, in
            metres.
        stages: An optional place for a caller to attach extra timing
            info; this module doesn't fill it in.
    """

    keyframe_count: int
    loop_edges: int
    loop_candidates: int
    rejected: bool
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
    """Chooses which frames to treat as keyframes, based on how much the
    camera actually moved rather than how much time passed.

    Picking keyframes by time alone would waste frames on moments where
    the camera sat still, and skip past moments where it moved quickly --
    like sweeping through a doorway -- which is exactly where good
    coverage matters most.

    Args:
        bundle: The parsed capture.
        translation_step: How far the camera has to move, in metres,
            before the next frame counts as a new keyframe.
        rotation_step_degrees: How far the camera has to turn, in
            degrees, before the next frame counts as a new keyframe.
        max_gap: The most frames allowed to pass without a new keyframe,
            even if the camera barely moved.

    Returns:
        A list of frame numbers, in order, always starting with frame 0.
    """
    positions = bundle.poses[:, :3, 3]
    rotations = bundle.poses[:, :3, :3]
    rotation_threshold = np.radians(rotation_step_degrees)

    selected = [0]
    anchor = 0
    for index in range(1, len(bundle)):
        translated = np.linalg.norm(positions[index] - positions[anchor])
        # How far the camera rotated since the last keyframe, worked out
        # from the two frames' rotation matrices.
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
    """Turns each keyframe's depth image into a 3D point cloud, ready for
    ICP alignment.

    Args:
        bundle: The parsed capture.
        keyframes: Which frame numbers to build clouds for.
        voxel_size: How finely to downsample each cloud, in metres.
        min_confidence: The lowest depth-confidence level to keep.
        max_depth: The furthest depth value to keep, in metres.

    Returns:
        A dictionary mapping each frame number to its point cloud, in
        that frame's own camera coordinates. A keyframe with too little
        valid depth data is skipped, so it may be missing from the
        result.
    """
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
        # Turn each valid depth pixel into an actual 3D point, using the
        # camera's focal length and centre (fx, fy, cx, cy).
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
    """Aligns two point clouds using ICP, first with a loose tolerance and
    then a tighter one.

    ICP (Iterative Closest Point) repeatedly matches up nearby points
    between the two clouds and solves for the rotation and shift that
    brings them closer together, until it stops improving. This uses the
    "point-to-plane" version, which matches each point to the other
    cloud's local surface instead of to one exact point -- that works
    much better on flat, mostly featureless walls.

    Args:
        source: The cloud to move.
        target: The cloud to align it against.
        init: A starting guess for the transform, which ICP refines.
        threshold: How close two points need to be to count as a match,
            in metres, during the fine second pass. The first, looser
            pass uses three times this distance.

    Returns:
        A tuple of (the final transform, how much of the cloud matched
        up, the average matching error in metres, and a matrix
        describing how well-constrained the result is). The last value
        is only for reporting -- it isn't used to decide how much to
        trust this result elsewhere.
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
    """Finds pairs of keyframes that probably show the same part of the
    room, seen on two different passes through it.

    Args:
        bundle: The parsed capture.
        keyframes: Which frame numbers to consider. The pairs returned
            are positions within this list, not raw frame numbers.
        max_distance: How close together, in metres, two keyframes need
            to be to count as a possible revisit.
        min_time_gap: How much time, in seconds, has to separate the two
            keyframes -- this rules out frames that are just next to each
            other in the normal sequence.
        max_view_angle_degrees: How different the two frames' camera
            directions can be and still plausibly be looking at the same
            thing.
        max_candidates: The most pairs to return.

    Returns:
        Up to `max_candidates` pairs of keyframe positions, closest
        pairs first.
    """
    positions = bundle.poses[keyframes][:, :3, 3]
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
            # Skip pairs whose cameras are facing too differently to be
            # looking at the same surfaces (e.g. opposite walls of a hall).
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
    max_total_correction: float = 0.75,
    enable_loop_closure: bool = True,
) -> tuple[np.ndarray, DriftReport]:
    """Corrects the camera trajectory using a pose graph, and reports how
    much it changed.

    The correction combines two kinds of information. Moves from one
    keyframe to the very next one come straight from ARKit's own
    tracking, which is already accurate over such a short distance.
    Longer-range corrections come from matching up keyframes that
    revisit the same spot later in the walkthrough, using ICP -- this is
    the only way to catch drift that built up gradually over the whole
    capture. If the end result looks physically implausible (some
    keyframe moved further than it reasonably should), the whole
    correction is thrown out and the original ARKit poses are returned
    instead.

    Args:
        bundle: The parsed capture to correct.
        keyframes: Which frames to use; worked out automatically with
            `select_keyframes` if not given.
        voxel_size: How finely to downsample points for ICP, in metres.
        min_confidence: The lowest depth-confidence level to keep.
        max_depth: The furthest depth value to keep, in metres.
        loop_fitness_threshold: How well two revisited keyframes need to
            match before that connection is trusted.
        max_loop_correction: The largest single correction, in metres, a
            revisit connection is allowed to make.
        max_total_correction: The largest correction, in metres, allowed
            anywhere before the whole result is thrown out as
            unreliable.
        enable_loop_closure: If False, revisit connections are skipped
            entirely, and only frame-to-frame corrections are used.

    Returns:
        A tuple of (corrected poses for every frame, a report describing
        what happened). The poses are in the same order and shape as the
        input.
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
        # Frame-to-frame moves come straight from ARKit, not from ICP --
        # ARKit is already accurate over this short a distance.
        relative = np.linalg.inv(bundle.poses[target]) @ bundle.poses[source]
        evaluation = o3d.pipelines.registration.evaluate_registration(
            clouds[source], clouds[target], threshold, relative
        )
        sequential_errors.append(float(evaluation.inlier_rmse))
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                relative,
                np.eye(6) * ODOMETRY_INFORMATION,
                uncertain=False,
            )
        )

    candidates = (
        find_loop_candidates(bundle, keyframes) if enable_loop_closure else []
    )
    accepted = 0
    for source_id, target_id in candidates:
        source, target = keyframes[source_id], keyframes[target_id]
        # Revisit connections DO come from ICP -- they carry information
        # ARKit alone doesn't have, since ARKit can't tell it's seeing a
        # familiar spot again.
        init = np.linalg.inv(bundle.poses[target]) @ bundle.poses[source]
        transformation, fitness, _, information = _pairwise_icp(
            clouds[source], clouds[target], init, threshold
        )
        if fitness < loop_fitness_threshold:
            continue
        displacement = np.linalg.norm(
            (transformation @ np.linalg.inv(init))[:3, 3]
        )
        if displacement > max_loop_correction:
            continue
        graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id,
                target_id,
                transformation,
                np.eye(6) * LOOP_INFORMATION,
                uncertain=True,
            )
        )
        accepted += 1

    # Solve for the poses that best satisfy every connection at once,
    # rather than applying each correction one at a time.
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
    corrections = np.linalg.norm(
        refined_keyframe_poses[:, :3, 3] - bundle.poses[keyframes][:, :3, 3], axis=1
    )

    # If any single keyframe moved further than seems physically
    # reasonable, don't trust any of this -- fall back to the raw poses.
    rejected = float(corrections.max()) > max_total_correction
    if rejected:
        poses = bundle.poses.copy()
        refined_keyframe_poses = bundle.poses[keyframes]
    else:
        poses = _propagate(bundle.poses, keyframes, refined_keyframe_poses)
    report = DriftReport(
        keyframe_count=len(keyframes),
        loop_edges=accepted,
        loop_candidates=len(candidates),
        rejected=rejected,
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
    """Spreads each keyframe's correction out to the ordinary frames
    around it.

    Only keyframes get directly corrected by the pose graph, so every
    other frame needs to inherit a version of that correction too. This
    blends smoothly between two keyframes' corrections for the frames
    in between, so nearby frames don't suddenly jump at each keyframe
    boundary.

    Args:
        original: The original, uncorrected poses for every frame.
        keyframes: Which frame numbers were corrected.
        refined: The corrected poses for those keyframes, in the same
            order.

    Returns:
        Corrected poses for every frame. Frames between two keyframes
        get a smooth blend of the two; frames after the last keyframe
        simply keep that keyframe's correction unchanged.
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
    """Converts a rotation matrix into a quaternion -- a different way of
    representing the same rotation, using four numbers instead of nine.

    There are a few equivalent formulas for this conversion, and some of
    them involve dividing by a number that can get close to zero for
    certain rotations, which would blow up the result. This picks
    whichever formula stays safely away from that problem for the given
    matrix.

    Args:
        matrix: A 3x3 rotation matrix.

    Returns:
        The equivalent quaternion, as `[w, x, y, z]`.
    """
    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        # The usual formula would divide by a number close to zero here,
        # so use whichever diagonal entry is largest instead.
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
    """Converts a quaternion back into a 3x3 rotation matrix.

    Args:
        q: A quaternion, as `[w, x, y, z]`.

    Returns:
        The equivalent 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Smoothly blends between two quaternions (rotations), moving along
    the shortest path between them.

    A plain straight-line blend between two rotations would speed up and
    slow down unevenly; this instead moves at a constant rate along the
    curve connecting them. When the two rotations are nearly identical,
    the usual formula becomes unstable, so this falls back to a simple
    straight-line blend in that case, since the difference is
    unnoticeable that close together.

    Args:
        a: The starting quaternion.
        b: The ending quaternion.
        t: How far to blend between them, from 0 (all `a`) to 1 (all
            `b`).

    Returns:
        The blended quaternion.
    """
    dot = float(a @ b)
    if dot < 0:
        # `b` and `-b` represent the same rotation; flip the sign so the
        # blend takes the shorter path between the two.
        b, dot = -b, -dot
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1, 1))
    sin_theta = np.sin(theta)
    return (
        np.sin((1 - t) * theta) / sin_theta * a + np.sin(t * theta) / sin_theta * b
    )
