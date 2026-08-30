from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Manifest:
    dataset_path: str
    calib_path: str | None
    config: str
    command: str


def build_command(dataset_path: Path, calib_path: Path | None, config: str = "config/base.yaml") -> str:
    parts = ["python", "main.py", "--dataset", str(dataset_path), "--config", config]
    if calib_path is not None:
        parts += ["--calib", str(calib_path)]
    return " ".join(parts)


def build_manifest(dataset_path: Path, calib_path: Path | None, config: str = "config/base.yaml") -> Manifest:
    return Manifest(
        dataset_path=str(dataset_path),
        calib_path=str(calib_path) if calib_path is not None else None,
        config=config,
        command=build_command(dataset_path, calib_path, config),
    )


def write_manifest(output_path: Path, manifest: Manifest) -> None:
    with output_path.open("w") as f:
        json.dump(asdict(manifest), f, indent=2)
        f.write("\n")
