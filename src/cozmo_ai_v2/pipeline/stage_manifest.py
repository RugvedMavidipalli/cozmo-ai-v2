"""Durable provenance for a single start-to-finish pipeline invocation.

The reconstruction result is intentionally not the only contract emitted by
the command.  This manifest records the decisions made before and during
reconstruction, including unavailable optional capabilities and the first
failure that stopped a required stage.  It is small enough to write after
every stage, so an interrupted run is still diagnosable.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


STAGE_ORDER = (
    "input_detection",
    "ingest_qc",
    "poses",
    "depth",
    "tsdf_reconstruction",
    "structural_planes",
    "observability",
    "vectorization",
    "openings",
    "measurements",
    "damage",
    "scope",
    "export",
)

STAGE_GROUPS = {
    "ingest": "ingest_qc",
    "pose refinement": "poses",
    "MASt3R-SLAM RGB tracking": "poses",
    "MASt3R-SLAM trajectory validation": "poses",
    "frame contract": "tsdf_reconstruction",
    "fusion": "tsdf_reconstruction",
    "sampling": "observability",
    "geometry": "structural_planes",
    "wall refinement": "vectorization",
    "rooms": "observability",
    "surfaces": "openings",
    "rgb openings": "openings",
    "measurements": "measurements",
    "damage": "damage",
    "scope": "scope",
    "export": "export",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    return [str(Path(value)) for value in values]


class StageManifest:
    """Write-only-on-purpose stage ledger for one pipeline run."""

    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_path = Path(input_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.path = self.output_dir / "stage_manifest.json"
        self.started_at = _now()
        self._started_clock = time.monotonic()
        self.status = "running"
        self.failure_reason: str | None = None
        self.stages: list[dict] = []
        self.context = {
            "model": {"status": "not_applicable"},
            "pose": {"status": "not_recorded"},
            "depth_provenance": "not_recorded",
        }
        self._write()

    @staticmethod
    def group_for(name: str) -> str:
        return STAGE_GROUPS.get(name, name)

    def _write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": self.status,
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "started_at": self.started_at,
            "finished_at": _now() if self.status != "running" else None,
            "runtime_s": round(time.monotonic() - self._started_clock, 3),
            "failure_reason": self.failure_reason,
            "stage_order": list(STAGE_ORDER),
            "stages": self.stages,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def set_context(self, **values) -> None:
        self.context.update(values)
        # Keep provenance attached to the stage that made the decision.  A
        # model or pose is often selected after the stage context is opened;
        # updating only the shared defaults would otherwise leave the durable
        # stage record with stale ``not_recorded`` values.
        for item in reversed(self.stages):
            if item["status"] == "running":
                for key in ("model", "pose", "depth_provenance"):
                    if key in values:
                        item[key] = values[key]
                break
        self._write()

    def record(
        self,
        name: str,
        status: str,
        *,
        reason: str = "",
        inputs=None,
        outputs=None,
        model=None,
        pose=None,
        depth_provenance=None,
        started_at: str | None = None,
        duration_s: float = 0.0,
    ) -> dict:
        if status not in {"running", "completed", "unavailable", "failed"}:
            raise ValueError(f"invalid stage status: {status}")
        item = {
            "stage": self.group_for(name),
            "operation": name,
            "status": status,
            "reason": reason or ("completed" if status == "completed" else ""),
            "started_at": started_at or _now(),
            "finished_at": _now() if status != "running" else None,
            "duration_s": round(float(duration_s), 3),
            "inputs": _paths(inputs if inputs is not None else [self.input_path]),
            "outputs": _paths(outputs if outputs is not None else [self.output_dir]),
            "model": model if model is not None else self.context["model"],
            "pose": pose if pose is not None else self.context["pose"],
            "depth_provenance": (
                depth_provenance
                if depth_provenance is not None
                else self.context["depth_provenance"]
            ),
        }
        self.stages.append(item)
        self._write()
        return item

    def update_last(self, name: str, **values) -> None:
        group = self.group_for(name)
        for item in reversed(self.stages):
            if item["stage"] == group:
                for key, value in values.items():
                    if key in {"inputs", "outputs"}:
                        value = _paths(value)
                    item[key] = value
                self._write()
                return

    def unavailable(self, name: str, reason: str, **kwargs) -> dict:
        return self.record(name, "unavailable", reason=reason, **kwargs)

    @contextmanager
    def stage(self, name: str, **kwargs) -> Iterator[dict]:
        started_at = _now()
        started_clock = time.monotonic()
        group = self.group_for(name)
        item = next(
            (
                candidate
                for candidate in reversed(self.stages)
                if candidate["stage"] == group
                and candidate["status"] in {"completed", "unavailable"}
            ),
            None,
        )
        if item is None:
            item = self.record(name, "running", started_at=started_at, **kwargs)
        else:
            item.update(
                operation=name,
                status="running",
                reason="",
                started_at=started_at,
                finished_at=None,
                duration_s=0.0,
            )
            if "inputs" in kwargs:
                item["inputs"] = _paths(kwargs["inputs"])
            if "outputs" in kwargs:
                item["outputs"] = _paths(kwargs["outputs"])
            self._write()
        try:
            yield item
        except BaseException as exc:
            item.update(
                status="failed",
                reason=str(exc),
                finished_at=_now(),
                duration_s=round(time.monotonic() - started_clock, 3),
            )
            self._write()
            raise
        else:
            item.update(
                status="completed",
                reason="completed",
                finished_at=_now(),
                duration_s=round(time.monotonic() - started_clock, 3),
            )
            self._write()

    def finalize(self, status: str, reason: str | None = None) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError(f"invalid manifest status: {status}")
        self.status = status
        self.failure_reason = reason
        self._write()


__all__ = ["STAGE_ORDER", "StageManifest"]
