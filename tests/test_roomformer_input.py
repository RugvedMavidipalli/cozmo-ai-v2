from __future__ import annotations

import numpy as np
import pytest

from cozmo_ai_v2.pipeline.roomformer_input import (
    plan_to_roomformer_pixels,
    scenecad_density_from_plan,
)


def test_scenecad_density_uses_square_padding_and_count_normalization():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    density = scenecad_density_from_plan(points, grid_size=10, padding_fraction=0.05)

    assert density.minimum.tolist() == pytest.approx([-0.05, -0.05])
    assert density.span == pytest.approx(1.1)
    assert density.image[0, 0] == 128  # one count against the two-count maximum
    assert density.image[0, 9] == 255
    assert density.image[9, 0] == 128


def test_roomformer_pixel_mapping_matches_density_transform():
    pixels = plan_to_roomformer_pixels(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([-0.05, -0.05]),
        1.1,
        10,
    )

    assert pixels.tolist() == [[0, 0], [9, 9]]


def test_roomformer_density_rejects_degenerate_input():
    with pytest.raises(ValueError, match="non-zero"):
        scenecad_density_from_plan(np.array([[1.0, 1.0], [1.0, 1.0]]))
