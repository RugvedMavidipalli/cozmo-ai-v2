from types import SimpleNamespace

import numpy as np

from cozmo_ai_v2.pipeline.cli import _assemble


def test_assemble_serializes_explicit_plane_thresholds_without_cli_args():
    class Bundle(SimpleNamespace):
        def __len__(self):
            return len(self.poses)

    bundle = Bundle(
        name="fixture",
        has_depth=True,
        pose_convention="camera_to_world_opencv",
        poses=np.eye(4, dtype=float)[None, ...],
        gravity_consistency=1.0,
        duration=0.0,
    )
    gravity = SimpleNamespace(
        up=np.array([0.0, 0.0, 1.0]),
        floor_height=0.0,
        ceiling_height=None,
        room_height=None,
        ceiling_observed=False,
        ceiling_confidence=0.0,
        ceiling_inlier_count=0,
        ceiling_residual_rms=None,
        floor_confidence=0.0,
        floor_observed=False,
        floor_quality_status="not_observed",
        floor_low_confidence=False,
        floor_support_fraction=0.0,
        floor_adaptive_residual_limit=0.04,
        floor_inlier_count=0,
        floor_residual_rms=0.0,
        floor_fit=None,
        ceiling_fit=None,
    )
    frame = SimpleNamespace(yaw=0.0, manhattan_fraction=1.0)
    drift = SimpleNamespace(
        per_wall=[],
        median_spread=0.0,
        p90_spread=0.0,
        max_spread=0.0,
        revisited_walls=0,
    )
    uncertainty = SimpleNamespace(calibrated=False, scale=1.0, coverage=0.9)
    engine = SimpleNamespace(rules={"version": "fixture"})

    result = _assemble(
        bundle,
        gravity,
        frame,
        [],
        [],
        [],
        [],
        [],
        [],
        drift,
        None,
        {},
        uncertainty,
        {},
        [],
        engine,
        structural_planes=[],
        plane_threshold=0.08,
        plane_min_inliers=17,
    )

    assert result["reconstruction"]["plane_extraction"] == {
        "algorithm": "seeded_ransac_region_growing_tls_3d",
        "refit": "total_least_squares_svd_perpendicular_residual",
        "candidate_threshold": 0.08,
        "support_threshold": 17,
        "residual_threshold": 0.08,
        "plane_count": 0,
        "kept_count": 0,
        "quarantined_count": 0,
        "rejection_reasons": {},
        "floor_plane_ids": [],
        "ceiling_plane_ids": [],
        "multiple_ceiling_planes": False,
    }
