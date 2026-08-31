from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Callable


class Mast3rSlamError(RuntimeError):
    """Raised when MASt3R-SLAM cannot be invoked."""


@dataclass(frozen=True)
class Mast3rSlamInvocation:
    """The external MASt3R-SLAM process that was started."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int


RunProcess = Callable[..., subprocess.CompletedProcess[object]]


def _resolve_mast3r_slam_dir(path: Path) -> Path:
    root = path.expanduser().resolve()
    entrypoint = root / "main.py"
    if not root.is_dir():
        raise Mast3rSlamError(f"MASt3R-SLAM directory does not exist: {root}")
    if not entrypoint.is_file():
        raise Mast3rSlamError(
            f"MASt3R-SLAM entry point was not found: {entrypoint}. "
            "Pass the root of a MASt3R-SLAM checkout."
        )
    return root


def build_rgb_video_command(
    video_path: Path,
    config: str = "config/base.yaml",
    *,
    python_executable: str | None = None,
    save_as: str | None = None,
    no_viz: bool = False,
) -> list[str]:
    """Build MASt3R-SLAM's uncalibrated MP4 invocation.

    MASt3R-SLAM accepts MP4 input without ``--calib`` and estimates the camera
    rays itself.  Deliberately do not add ``--calib`` here: this path is for a
    standalone RGB video with no camera metadata.
    """

    command = [
        python_executable or sys.executable,
        "main.py",
        "--dataset",
        str(video_path.expanduser().resolve()),
        "--config",
        config,
    ]
    if save_as is not None:
        command.extend(["--save-as", save_as])
    if no_viz:
        command.append("--no-viz")
    return command


def run_rgb_video(
    video_path: Path,
    mast3r_slam_dir: Path,
    config: str = "config/base.yaml",
    *,
    python_executable: str | None = None,
    save_as: str | None = None,
    no_viz: bool = False,
    run_process: RunProcess = subprocess.run,
) -> Mast3rSlamInvocation:
    """Run MASt3R-SLAM on one uncalibrated RGB video.

    The process inherits stdout/stderr so MASt3R-SLAM's progress and failures
    stay visible to the caller.  Its working directory is its checkout because
    its config files and checkpoints use repository-relative paths.
    """

    root = _resolve_mast3r_slam_dir(mast3r_slam_dir)
    command = build_rgb_video_command(
        video_path,
        config,
        python_executable=python_executable,
        save_as=save_as,
        no_viz=no_viz,
    )
    try:
        completed = run_process(command, cwd=root, check=False)
    except OSError as exc:
        raise Mast3rSlamError(f"Could not start MASt3R-SLAM: {exc}") from exc

    invocation = Mast3rSlamInvocation(tuple(command), root, completed.returncode)
    if completed.returncode != 0:
        raise Mast3rSlamError(
            f"MASt3R-SLAM exited with status {completed.returncode}: "
            f"{' '.join(invocation.command)}"
        )
    return invocation
