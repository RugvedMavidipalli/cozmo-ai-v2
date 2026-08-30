import json
from pathlib import Path

from cozmo_ai_v2.manifest import build_command, build_manifest, write_manifest


def test_build_command_with_calib():
    command = build_command(Path("/a/b.mp4"), Path("/a/intrinsics.yaml"), "config/base.yaml")

    assert command == "python main.py --dataset /a/b.mp4 --config config/base.yaml --calib /a/intrinsics.yaml"


def test_build_command_without_calib():
    command = build_command(Path("/a/b.mp4"), None, "config/base.yaml")

    assert command == "python main.py --dataset /a/b.mp4 --config config/base.yaml"


def test_write_manifest_round_trip(tmp_path):
    manifest = build_manifest(Path("/a/b.mp4"), None, "config/base.yaml")
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    data = json.loads(path.read_text())
    assert data["dataset_path"] == "/a/b.mp4"
    assert data["calib_path"] is None
    assert data["config"] == "config/base.yaml"
    assert data["command"] == "python main.py --dataset /a/b.mp4 --config config/base.yaml"
