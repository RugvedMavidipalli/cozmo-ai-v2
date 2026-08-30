from __future__ import annotations

from pathlib import Path

import yaml


def write_intrinsics_yaml(
    output_path: Path,
    width: int,
    height: int,
    calibration: list[float],
) -> None:
    if len(calibration) != 4:
        raise ValueError(f"calibration must have exactly 4 elements [fx, fy, cx, cy], got {len(calibration)}")

    data = {
        "width": int(width),
        "height": int(height),
        "calibration": [float(v) for v in calibration],
    }
    with output_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
