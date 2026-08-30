import pytest
import yaml

from cozmo_ai_v2.intrinsics_writer import write_intrinsics_yaml
from cozmo_ai_v2.video import VideoProbeError, probe_video


def test_write_intrinsics_yaml_schema(tmp_path):
    path = tmp_path / "intrinsics.yaml"

    write_intrinsics_yaml(path, width=640, height=480, calibration=[517.3, 516.5, 318.6, 255.3])

    data = yaml.safe_load(path.read_text())
    assert set(data.keys()) == {"width", "height", "calibration"}
    assert data["width"] == 640
    assert data["height"] == 480
    assert data["calibration"] == [517.3, 516.5, 318.6, 255.3]
    assert len(data["calibration"]) == 4


def test_write_intrinsics_yaml_rejects_wrong_length(tmp_path):
    path = tmp_path / "intrinsics.yaml"

    with pytest.raises(ValueError):
        write_intrinsics_yaml(path, width=640, height=480, calibration=[1.0, 2.0, 3.0])


def test_probe_video_reads_synthetic_dimensions(synthetic_video):
    info = probe_video(synthetic_video)

    assert info.width == 64
    assert info.height == 48


def test_probe_video_raises_on_invalid_file(tmp_path):
    path = tmp_path / "not_a_video.mp4"
    path.write_bytes(b"this is not a real video file")

    with pytest.raises(VideoProbeError):
        probe_video(path)
