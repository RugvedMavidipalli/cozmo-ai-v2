from __future__ import annotations

from pathlib import Path

import numpy as np


class CameraMatrixError(ValueError):
    pass


def parse_camera_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, delimiter=",")
    except (OSError, ValueError) as exc:
        raise CameraMatrixError(f"Could not parse {path} as CSV: {exc}") from exc

    if matrix.shape != (3, 3):
        raise CameraMatrixError(f"Expected a 3x3 matrix in {path}, got shape {matrix.shape}")

    return matrix


def extract_calibration_4(matrix: np.ndarray) -> list[float]:
    return [
        float(matrix[0, 0]),  # fx
        float(matrix[1, 1]),  # fy
        float(matrix[0, 2]),  # cx
        float(matrix[1, 2]),  # cy
    ]
