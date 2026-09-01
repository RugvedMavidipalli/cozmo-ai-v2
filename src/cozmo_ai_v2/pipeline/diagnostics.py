"""CPU-safe Stage 4/5 ablation helpers.

The comparison path deliberately consumes already-produced dense artifacts;
it never constructs or executes a depth model. Each variant gets a fresh
frame contract and TSDF volume so frame decisions and effective parameters
remain reproducible and do not leak across variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame_contract import build_frame_contract
from .fuse import fuse
from .ingest import CaptureBundle


@dataclass(frozen=True)
class TSDFVariant:
    """One named TSDF parameter point for an ablation comparison."""

    label: str
    voxel_size_m: float
    sdf_trunc_m: float | None = None


def compare_tsdf_parameters(
    bundle: CaptureBundle,
    variants: list[TSDFVariant],
    *,
    indices: list[int] | np.ndarray | None = None,
    poses: np.ndarray | None = None,
    pose_source: str | None = None,
    dense_depth_dir: str | None = None,
    densify_manifest: str | None = None,
    min_confidence: int = 1,
    max_depth: float = 3.5,
    depth_source: str = "auto",
    frame_association: str = "pts",
    pts_tolerance_s: float | None = None,
) -> list[dict]:
    """Run CPU TSDF variants and return comparable provenance records.

    The returned records include the full contract report, point/mesh counts,
    and effective TSDF settings. They can be written directly as an
    ablation JSON artifact. ``depth_source`` can be forced to ``dense`` or
    ``raw`` to compare the two sources without silently falling back.
    """
    if not variants:
        raise ValueError("at least one TSDF variant is required")

    records: list[dict] = []
    for variant in variants:
        contract = build_frame_contract(
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
        reconstruction = fuse(
            bundle,
            indices=indices,
            poses=poses,
            voxel_size=variant.voxel_size_m,
            sdf_trunc=variant.sdf_trunc_m,
            min_confidence=min_confidence,
            max_depth=max_depth,
            frame_contract=contract,
        )
        report = reconstruction.contract_report or {}
        records.append(
            {
                "label": variant.label,
                "frame_count": reconstruction.frame_count,
                "frame_indices": list(reconstruction.frame_indices),
                "point_count": len(reconstruction.cloud.points),
                "mesh_vertex_count": len(reconstruction.mesh.vertices),
                "contract": report,
                "tsdf_parameters": report.get("tsdf_parameters", {}),
            }
        )
    return records
