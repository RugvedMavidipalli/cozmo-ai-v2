"""The depth/pose contract shared by Stage 4 and reconstruction stages.

Stage 4 writes model-assisted depth as an artifact.  This module is the
trust boundary for consuming that artifact: a manifest entry must explicitly
be QC-approved, the raster must match the RGB frame, and a valid pose must be
available.  A bad dense frame falls back to the corresponding raw LiDAR frame
when possible; it is never silently substituted with a neighbouring index.

The contract is lazy.  Building it only reads the densification manifest;
video and rasters are read once, in deterministic frame-index order, as the
consumer iterates them.  That keeps the full-resolution path usable on real
captures without retaining every image in memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .ingest import CaptureBundle, VideoAvailability, iter_raw_frames

DEPTH_SCALE_MM_TO_M = 1000.0
CONTRACT_DEPTH_UNIT = "m"
DENSE_DEPTH_SOURCE = "metric3d_v2_scale_shift_lidar_residual"
RAW_LIDAR_SOURCE = "raw_lidar"


class FrameContractError(ValueError):
    """Raised for an invalid frame-contract or densification artifact."""


@dataclass(frozen=True)
class FrameProvenance:
    """Machine-readable origin of the arrays used for one frame."""

    depth_source: str
    depth_path: str | None
    confidence_source: str
    confidence_path: str | None
    qc_source: str
    qc_path: str | None
    pose_source: str
    pose_path: str | None
    source_depth_unit: str
    contract_depth_unit: str
    registration: str
    depth_resolution: tuple[int, int]

    def to_dict(self) -> dict:
        payload = asdict(self)
        # JSON Schema distinguishes Python tuples from JSON arrays even
        # though ``json.dumps`` would serialize both alike.  Keep the public
        # provenance contract schema-valid before validation/export.
        payload["depth_resolution"] = list(self.depth_resolution)
        return payload


@dataclass(frozen=True)
class ReconstructionFrame:
    """A frame with resolution-aligned depth, masks, intrinsics, and pose."""

    index: int
    timestamp: float
    color: np.ndarray  # RGB uint8, same HxW as depth
    depth: np.ndarray  # float32 metres, invalid pixels are zero
    confidence: np.ndarray  # uint8 confidence raster, aligned to depth
    qc_mask: np.ndarray  # bool pixels approved for TSDF integration
    pose: np.ndarray  # camera-to-world, OpenCV convention
    intrinsics: np.ndarray  # scaled to color/depth resolution
    provenance: FrameProvenance

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels that are safe for integration under the contract policy."""
        return (
            self.qc_mask
            & (self.confidence >= 0)
            & np.isfinite(self.depth)
            & (self.depth > 0)
        )

    @property
    def depth_m(self) -> np.ndarray:
        """Compatibility alias used by depth-stage callers."""
        return self.depth


# Public name for callers that refer to the cross-stage object simply as a
# pipeline frame.  ``ReconstructionFrame`` remains descriptive in type hints.
PipelineFrame = ReconstructionFrame


@dataclass(frozen=True)
class FrameRejection:
    index: int
    reason: str
    attempted_depth_source: str


@dataclass(frozen=True)
class _DenseEntry:
    index: int
    qc_approved: bool
    depth_path: Path | None
    confidence_path: Path | None
    qc_path: Path | None
    reason: str = ""
    depth_unit: str = "mm"
    rgb_scale: tuple[float, float] = (1.0, 1.0)
    provenance: str = DENSE_DEPTH_SOURCE


def _normalise_indices(indices: list[int] | np.ndarray | None, count: int) -> tuple[int, ...]:
    values = range(count) if indices is None else (int(value) for value in indices)
    # Sorting and de-duplicating here is part of the on-disk frame contract:
    # output names and provenance must not depend on caller ordering.
    return tuple(sorted({value for value in values if value >= 0}))


def _resolve_artifact_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _has_depth_rasters(path: Path) -> bool:
    return any(any(path.glob(pattern)) for pattern in ("*.png", "*.npy", "*.npz"))


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise FrameContractError(f"{path} contains no arrays")
            array = archive[archive.files[0]]
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FrameContractError(f"cannot read raster {path}")
    array = np.asarray(array)
    if array.ndim != 2:
        raise FrameContractError(f"expected a single-channel raster at {path}, got {array.shape}")
    return array


def _read_depth(path: Path, depth_unit: str) -> np.ndarray:
    raw = _read_array(path)
    if depth_unit.lower() in {"mm", "millimetres", "millimeters"}:
        scale = DEPTH_SCALE_MM_TO_M
    elif depth_unit.lower() in {"m", "metres", "meters"}:
        scale = 1.0
    else:
        raise FrameContractError(f"unsupported depth unit {depth_unit!r} in {path}")
    return raw.astype(np.float32) / scale


def _aligned_mask(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if array.shape == shape:
        return array
    if array.ndim != 2:
        raise FrameContractError(f"{name} must be single-channel, got {array.shape}")
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "approved", "qc_approved"}
    return bool(value)


def _manifest_rgb_scale(payload: dict) -> tuple[float, float]:
    value = payload.get("dense_rgb_scale", (1.0, 1.0))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise FrameContractError("densify manifest dense_rgb_scale must contain [x_scale, y_scale]")
    try:
        scale_x, scale_y = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise FrameContractError("densify manifest dense_rgb_scale must contain numbers") from exc
    if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in (scale_x, scale_y)):
        raise FrameContractError("densify manifest dense_rgb_scale must be finite and in (0, 1]")
    return scale_x, scale_y


def _manifest_entries(manifest_path: Path | None, dense_dir: Path | None) -> dict[int, _DenseEntry]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise FrameContractError(f"cannot read densify manifest {manifest_path}: {exc}") from exc

    manifest_root = manifest_path.parent
    manifest_rgb_scale = _manifest_rgb_scale(payload)
    entries: dict[int, _DenseEntry] = {}
    for raw in payload.get("frames", []):
        try:
            index = int(raw["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameContractError(f"densify manifest has an invalid frame entry: {raw!r}") from exc
        if index in entries:
            raise FrameContractError(f"densify manifest contains duplicate frame index {index}")

        depth_value = raw.get("depth_path") or raw.get("path")
        confidence_value = raw.get("confidence_path")
        qc_value = raw.get("qc_mask_path") or raw.get("qc_path")
        depth_path = _resolve_artifact_path(depth_value, manifest_root)
        confidence_path = _resolve_artifact_path(confidence_value, manifest_root)
        qc_path = _resolve_artifact_path(qc_value, manifest_root)
        if dense_dir is not None:
            depth_path = depth_path or dense_dir / f"{index:06d}.png"
            confidence_path = confidence_path or dense_dir.parent / "dense_confidence" / f"{index:06d}.png"
            qc_path = qc_path or dense_dir.parent / "dense_qc" / f"{index:06d}.png"

        approved = _as_bool(raw.get("qc_approved", raw.get("status") in {"approved", "accepted", "qc_approved"}))
        entries[index] = _DenseEntry(
            index=index,
            qc_approved=approved,
            depth_path=depth_path,
            confidence_path=confidence_path,
            qc_path=qc_path,
            reason=str(raw.get("qc_reason", "")),
            depth_unit=str(raw.get("depth_unit", payload.get("depth_unit", "mm"))),
            rgb_scale=manifest_rgb_scale,
            provenance=str(payload.get("depth_provenance", DENSE_DEPTH_SOURCE)),
        )
    return entries


def discover_dense_artifacts(
    capture_root: str | Path,
    dense_depth_dir: str | Path | None = None,
    densify_manifest: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve explicit or conventional Stage 4 artifact locations."""
    root = Path(capture_root)
    dense_dir = Path(dense_depth_dir) if dense_depth_dir is not None else root / "dense_depth"
    # Accept either the raster directory or the Stage 4 output directory.
    if dense_dir.is_dir() and not _has_depth_rasters(dense_dir) and (dense_dir / "dense_depth").is_dir():
        dense_dir = dense_dir / "dense_depth"
    if not dense_dir.is_dir():
        dense_dir = None

    explicit_manifest = densify_manifest is not None
    manifest = Path(densify_manifest) if explicit_manifest else (
        (dense_dir.parent / "densify_manifest.json") if dense_dir is not None else None
    )
    if manifest is not None and not manifest.exists():
        if explicit_manifest:
            raise FrameContractError(f"densify manifest does not exist: {manifest}")
        manifest = None
    return dense_dir, manifest


@dataclass
class FrameContract:
    """Lazy, deterministic frame source for reconstruction and TSDF fusion."""

    bundle: CaptureBundle
    poses: np.ndarray
    pose_source: str
    pose_path: str | None
    requested_indices: tuple[int, ...]
    dense_depth_dir: Path | None = None
    densify_manifest: Path | None = None
    min_confidence: int = 1
    max_depth: float = 3.5
    depth_source_mode: str = "auto"
    _dense_entries: dict[int, _DenseEntry] = field(default_factory=dict)
    yielded_indices: set[int] = field(default_factory=set)
    integrated_indices: set[int] = field(default_factory=set)
    rejected_frames: dict[int, FrameRejection] = field(default_factory=dict)
    fallback_frames: dict[int, FrameRejection] = field(default_factory=dict)
    depth_sources: dict[int, str] = field(default_factory=dict)
    provenance_by_index: dict[int, FrameProvenance] = field(default_factory=dict)
    video_availability: VideoAvailability | None = None

    def iter_frames(self, indices: list[int] | np.ndarray | None = None) -> Iterator[ReconstructionFrame]:
        requested = _normalise_indices(indices, len(self.poses)) if indices is not None else self.requested_indices
        for index, bgr, raw_depth, raw_confidence in iter_raw_frames(
            self.bundle.root, requested, availability=self.video_availability
        ):
            frame = self._build_frame(index, bgr, raw_depth, raw_confidence)
            if frame is None:
                continue
            if not frame.valid_mask.any():
                self._reject(index, "no valid depth after confidence/QC/max-depth filtering", frame.provenance.depth_source)
                continue
            self.yielded_indices.add(index)
            self.depth_sources[index] = frame.provenance.depth_source
            self.provenance_by_index[index] = frame.provenance
            yield frame

        # A requested index beyond the available video is a deterministic
        # rejection rather than an accidental reindexing of the next frame.
        seen = {index for index in self.yielded_indices} | set(self.rejected_frames)
        for index in requested:
            if index not in seen:
                reason = "frame is not present in the RGB video"
                if self.video_availability is not None and index in self.video_availability.missing_indices:
                    reason += "; terminal decode ended before this sidecar index"
                self._reject(index, reason, "unknown")

    def _reject(self, index: int, reason: str, source: str) -> None:
        self.rejected_frames[index] = FrameRejection(index, reason, source)

    def _build_frame(
        self,
        index: int,
        bgr: np.ndarray,
        raw_depth: np.ndarray | None,
        raw_confidence: np.ndarray | None,
    ) -> ReconstructionFrame | None:
        if index >= len(self.poses) or index >= len(self.bundle.timestamps):
            self._reject(index, "pose/timestamp table has no matching frame", "unknown")
            return None
        pose = np.asarray(self.poses[index], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            self._reject(index, "pose is not a finite 4x4 transform", self.pose_source)
            return None
        if abs(float(np.linalg.det(pose[:3, :3]))) < 1e-8:
            self._reject(index, "pose rotation/linear part is singular", self.pose_source)
            return None

        entry = self._dense_entries.get(index)
        if self.depth_source_mode != "raw" and entry is not None and entry.qc_approved:
            try:
                dense_frame = self._dense_frame(index, bgr, raw_confidence, pose, entry)
                if dense_frame.valid_mask.any():
                    return dense_frame
                raise FrameContractError("dense QC mask contains no usable pixels")
            except (FrameContractError, OSError, ValueError) as exc:
                if self.depth_source_mode == "auto" and raw_depth is not None:
                    self.fallback_frames[index] = FrameRejection(
                        index, f"dense frame rejected ({exc}); raw LiDAR used", entry.provenance
                    )
                else:
                    self._reject(index, f"dense frame rejected: {exc}", entry.provenance)
                    return None
        elif self.depth_source_mode != "raw" and entry is not None:
            reason = entry.reason or "manifest entry is not qc_approved"
            if self.depth_source_mode == "auto" and raw_depth is not None:
                self.fallback_frames[index] = FrameRejection(index, f"{reason}; raw LiDAR used", entry.provenance)
            else:
                self._reject(index, reason, entry.provenance)
                return None
        elif self.depth_source_mode != "raw" and self.dense_depth_dir is not None:
            reason = "dense raster has no QC-approved manifest entry"
            if self.depth_source_mode == "auto" and raw_depth is not None:
                self.fallback_frames[index] = FrameRejection(index, f"{reason}; raw LiDAR used", DENSE_DEPTH_SOURCE)
            else:
                self._reject(index, reason, DENSE_DEPTH_SOURCE)
                return None
        elif self.depth_source_mode == "dense":
            self._reject(index, "dense depth source was requested but no QC-approved entry exists", DENSE_DEPTH_SOURCE)
            return None

        if raw_depth is None:
            reason = (
                "raw LiDAR source was requested but no raw LiDAR depth is available"
                if self.depth_source_mode == "raw"
                else "no QC-approved dense depth and no raw LiDAR depth"
            )
            self._reject(index, reason, RAW_LIDAR_SOURCE)
            return None
        try:
            return self._raw_frame(index, bgr, raw_depth, raw_confidence, pose)
        except (FrameContractError, OSError, ValueError) as exc:
            self._reject(index, str(exc), RAW_LIDAR_SOURCE)
            return None

    def _dense_frame(
        self,
        index: int,
        bgr: np.ndarray,
        raw_confidence: np.ndarray | None,
        pose: np.ndarray,
        entry: _DenseEntry,
    ) -> ReconstructionFrame:
        if entry.depth_path is None or not entry.depth_path.exists():
            raise FrameContractError("manifest has no readable dense depth path")
        depth = _read_depth(entry.depth_path, entry.depth_unit)
        height, width = bgr.shape[:2]
        if depth.shape != (height, width):
            expected_width = round(width * entry.rgb_scale[0])
            expected_height = round(height * entry.rgb_scale[1])
            if (
                entry.rgb_scale == (1.0, 1.0)
                or depth.shape != (expected_height, expected_width)
            ):
                raise FrameContractError(
                    f"dense depth shape {depth.shape} does not match RGB shape {(height, width)} "
                    f"or manifest-declared scaled RGB {(expected_height, expected_width)}"
                )
            bgr = cv2.resize(bgr, (width := depth.shape[1], height := depth.shape[0]), interpolation=cv2.INTER_AREA)
            registration = "dense_scaled_rgb_area_resize"
        else:
            registration = "dense_native_rgb_exact_shape"

        if entry.confidence_path is not None and entry.confidence_path.exists():
            confidence = _aligned_mask(_read_array(entry.confidence_path), depth.shape, "confidence")
            confidence_path = str(entry.confidence_path)
            confidence_source = "dense_confidence"
        elif raw_confidence is not None:
            confidence = _aligned_mask(raw_confidence, depth.shape, "confidence")
            confidence_path = None
            confidence_source = "raw_lidar_resampled"
        else:
            confidence = np.full(depth.shape, 2, dtype=np.uint8)
            confidence_path = None
            confidence_source = "assumed_high"

        if entry.qc_path is not None and entry.qc_path.exists():
            qc = _aligned_mask(_read_array(entry.qc_path), depth.shape, "QC mask") > 0
            qc_source = "densify_qc"
            qc_path = str(entry.qc_path)
        else:
            raise FrameContractError("QC-approved dense depth is missing its QC mask")

        depth = depth.astype(np.float32, copy=False)
        qc &= (
            np.isfinite(depth)
            & (depth > 0)
            & (depth <= self.max_depth)
            & (confidence >= self.min_confidence)
        )
        depth = np.where(qc, depth, 0.0).astype(np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ReconstructionFrame(
            index=index,
            timestamp=float(self.bundle.timestamps[index]),
            color=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth),
            confidence=np.ascontiguousarray(confidence.astype(np.uint8, copy=False)),
            qc_mask=np.ascontiguousarray(qc),
            pose=pose,
            intrinsics=self.bundle.intrinsics_for_size(width, height),
            provenance=FrameProvenance(
                depth_source=entry.provenance,
                depth_path=str(entry.depth_path),
                confidence_source=confidence_source,
                confidence_path=confidence_path,
                qc_source=qc_source,
                qc_path=qc_path,
                pose_source=self.pose_source,
                pose_path=self.pose_path,
                source_depth_unit=entry.depth_unit,
                contract_depth_unit=CONTRACT_DEPTH_UNIT,
                registration=registration,
                depth_resolution=(int(width), int(height)),
            ),
        )

    def _raw_frame(
        self,
        index: int,
        bgr: np.ndarray,
        raw_depth: np.ndarray,
        raw_confidence: np.ndarray | None,
        pose: np.ndarray,
    ) -> ReconstructionFrame:
        if raw_depth.ndim != 2:
            raise FrameContractError(f"raw depth for frame {index} is not single-channel")
        depth = raw_depth.astype(np.float32) / DEPTH_SCALE_MM_TO_M
        if raw_confidence is None:
            confidence = np.full(depth.shape, 2, dtype=np.uint8)
            confidence_source = "assumed_high"
            confidence_path = None
        else:
            if raw_confidence.shape != depth.shape:
                raise FrameContractError(
                    f"raw confidence shape {raw_confidence.shape} does not match depth {depth.shape}"
                )
            confidence = raw_confidence.astype(np.uint8, copy=False)
            confidence_source = "raw_lidar_confidence"
            confidence_path = str(self.bundle.root / "confidence" / f"{index:06d}.png")
        qc = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth)
        valid = qc & (confidence >= self.min_confidence)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        height, width = depth.shape
        rgb = cv2.cvtColor(
            cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        return ReconstructionFrame(
            index=index,
            timestamp=float(self.bundle.timestamps[index]),
            color=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth),
            confidence=np.ascontiguousarray(confidence),
            qc_mask=np.ascontiguousarray(qc),
            pose=pose,
            intrinsics=self.bundle.intrinsics_for_size(width, height),
            provenance=FrameProvenance(
                depth_source=RAW_LIDAR_SOURCE,
                depth_path=str(self.bundle.root / "depth" / f"{index:06d}.png"),
                confidence_source=confidence_source,
                confidence_path=confidence_path,
                qc_source="raw_depth_validity",
                qc_path=None,
                pose_source=self.pose_source,
                pose_path=self.pose_path,
                source_depth_unit="mm",
                contract_depth_unit=CONTRACT_DEPTH_UNIT,
                registration="raw_depth_native_rgb_area_resize",
                depth_resolution=(int(width), int(height)),
            ),
        )

    def report(self) -> dict:
        """Return deterministic provenance and frame-decision metadata."""
        sources = sorted(set(self.depth_sources.values()))
        provenance = list(self.provenance_by_index.values())
        registration = {
            "policy": "dense_requires_native_or_manifest_declared_scaled_rgb_alignment; raw_rgb_area_resize_to_depth",
            "dense_depth": {
                "required_shape": "native_rgb_or_manifest_declared_scaled_rgb",
                "shape_mismatch": "reject_unless_manifest_declared_scale",
                "confidence_alignment": "nearest",
                "qc_alignment": "nearest",
                "scaled_rgb_alignment": "INTER_AREA_resize",
            },
            "raw_lidar": {
                "depth_shape": "native_lidar",
                "rgb_alignment": "INTER_AREA_resize",
                "confidence_alignment": "exact_shape_required",
            },
        }
        return {
            "depth_sources": sources,
            "depth_units": sorted({item.source_depth_unit for item in provenance}),
            "depth_unit": CONTRACT_DEPTH_UNIT,
            "confidence_threshold": self.min_confidence,
            "max_depth_m": self.max_depth,
            "registration_alignment": registration,
            "dense_depth_dir": str(self.dense_depth_dir) if self.dense_depth_dir is not None else None,
            "densify_manifest": str(self.densify_manifest) if self.densify_manifest is not None else None,
            "pose_source": self.pose_source,
            "pose_path": self.pose_path,
            "pose_convention": "camera_to_world_opencv",
            "depth_source_mode": self.depth_source_mode,
            "contract_parameters": {
                "min_confidence": self.min_confidence,
                "max_depth_m": self.max_depth,
                "max_depth_inclusive": True,
                "invalid_depth_action": "zero",
                "depth_output_unit": CONTRACT_DEPTH_UNIT,
                "depth_source_policy": (
                    "qc_approved_dense_else_same_index_raw_lidar"
                    if self.depth_source_mode == "auto"
                    else self.depth_source_mode
                ),
                "registration_alignment": registration,
            },
            "requested_indices": list(self.requested_indices),
            "integrated_indices": sorted(self.integrated_indices),
            "video_availability": (
                self.video_availability.to_dict()
                if self.video_availability is not None
                else None
            ),
            "frame_provenance": [
                {"index": index, **self.provenance_by_index[index].to_dict()}
                for index in sorted(self.provenance_by_index)
            ],
            "rejected_frames": [asdict(self.rejected_frames[i]) for i in sorted(self.rejected_frames)],
            "fallback_frames": [asdict(self.fallback_frames[i]) for i in sorted(self.fallback_frames)],
        }


def build_frame_contract(
    bundle: CaptureBundle,
    *,
    indices: list[int] | np.ndarray | None = None,
    poses: np.ndarray | None = None,
    pose_source: str | None = None,
    pose_path: str | None = None,
    dense_depth_dir: str | Path | None = None,
    densify_manifest: str | Path | None = None,
    min_confidence: int = 1,
    max_depth: float = 3.5,
    depth_source: str = "auto",
    frame_association: str = "pts",
    pts_tolerance_s: float | None = None,
) -> FrameContract:
    """Build the Stage 4→5 contract without reading frame rasters yet."""
    if not 0 <= min_confidence <= 2:
        raise FrameContractError(f"min_confidence must be in [0, 2], got {min_confidence}")
    if max_depth <= 0 or not np.isfinite(max_depth):
        raise FrameContractError(f"max_depth must be a positive finite number, got {max_depth}")
    if depth_source not in {"auto", "dense", "raw"}:
        raise FrameContractError(
            f"depth_source must be 'auto', 'dense', or 'raw', got {depth_source!r}"
        )
    if frame_association not in {"index", "pts"}:
        raise FrameContractError(
            f"frame_association must be 'index' or 'pts', got {frame_association!r}"
        )
    if pts_tolerance_s is not None and (
        pts_tolerance_s < 0 or not np.isfinite(pts_tolerance_s)
    ):
        raise FrameContractError(
            f"pts_tolerance_s must be a non-negative finite number, got {pts_tolerance_s}"
        )
    pose_table = np.asarray(bundle.poses if poses is None else poses, dtype=np.float64)
    if pose_table.ndim != 3 or pose_table.shape[1:] != (4, 4):
        raise FrameContractError(f"poses must have shape (N, 4, 4), got {pose_table.shape}")
    if len(pose_table) != len(bundle.timestamps):
        raise FrameContractError(
            f"pose count {len(pose_table)} does not match timestamp count {len(bundle.timestamps)}"
        )
    if not np.isfinite(pose_table).all():
        raise FrameContractError("pose table contains non-finite values")

    dense_dir, manifest = discover_dense_artifacts(bundle.root, dense_depth_dir, densify_manifest)
    entries = _manifest_entries(manifest, dense_dir)
    source = pose_source or ("refined_" + bundle.pose_source if poses is not None else bundle.pose_source)
    return FrameContract(
        bundle=bundle,
        poses=pose_table,
        pose_source=source,
        pose_path=pose_path or bundle.pose_path,
        requested_indices=_normalise_indices(indices, len(pose_table)),
        dense_depth_dir=dense_dir,
        densify_manifest=manifest,
        min_confidence=min_confidence,
        max_depth=max_depth,
        depth_source_mode=depth_source,
        _dense_entries=entries,
        video_availability=VideoAvailability(
            expected_frame_count=len(pose_table),
            association_mode=frame_association,
            pts_tolerance_s=pts_tolerance_s,
            sidecar_timestamps=np.asarray(bundle.timestamps, dtype=np.float64),
        ),
    )
