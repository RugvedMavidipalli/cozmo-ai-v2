import numpy as np
import pytest

from cozmo_ai_v2.camera import CameraMatrixError, extract_calibration_4, parse_camera_matrix


def test_parse_valid_matrix(tmp_path):
    matrix = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])
    path = tmp_path / "camera_matrix.csv"
    np.savetxt(path, matrix, delimiter=",")

    parsed = parse_camera_matrix(path)

    assert parsed.shape == (3, 3)
    np.testing.assert_allclose(parsed, matrix)


def test_parse_wrong_shape_raises(tmp_path):
    path = tmp_path / "camera_matrix.csv"
    path.write_text("1.0,2.0\n3.0,4.0\n")

    with pytest.raises(CameraMatrixError):
        parse_camera_matrix(path)


def test_parse_non_numeric_raises(tmp_path):
    path = tmp_path / "camera_matrix.csv"
    path.write_text("a,b,c\nd,e,f\ng,h,i\n")

    with pytest.raises(CameraMatrixError):
        parse_camera_matrix(path)


def test_extract_calibration_4():
    matrix = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])

    calibration = extract_calibration_4(matrix)

    assert calibration == [517.3, 516.5, 318.6, 255.3]
    assert all(isinstance(v, float) for v in calibration)
