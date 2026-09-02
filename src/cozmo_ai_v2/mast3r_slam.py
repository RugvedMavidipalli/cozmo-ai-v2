from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable


class Mast3rSlamError(RuntimeError):
    """Raised when MASt3R-SLAM cannot be invoked."""


@dataclass(frozen=True)
class Mast3rSlamCapabilities:
    """Features discovered from an upstream checkout without importing it.

    MASt3R-SLAM normally needs CUDA to import, so capability detection reads
    its CLI source rather than invoking ``main.py --help``. This keeps the
    integration safe to inspect on CPU-only machines.
    """

    supports_pose_priors: bool
    pose_prior_argument: str | None
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mast3rSlamInvocation:
    """The external MASt3R-SLAM process that was started."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    capabilities: Mast3rSlamCapabilities | None = None
    pose_prior_mode: str = "not_requested"


RunProcess = Callable[..., subprocess.CompletedProcess[object]]

_POSE_PRIOR_ARGUMENTS = ("--pose-priors", "--prior-poses", "--poses")


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


def _expected_trajectory_path(root: Path, video_path: Path, save_as: str | None) -> Path:
    results_dir = root / "logs" / save_as if save_as else root / "logs"
    return results_dir / f"{video_path.stem}.txt"


def _partial_trajectory_summary(path: Path) -> str:
    """Describe a partial upstream output without treating it as usable."""
    if not path.is_file():
        return "no trajectory file was produced"
    try:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        return f"trajectory file could not be inspected: {exc}"
    timestamps = []
    for row in rows:
        try:
            if len(row) == 8:
                timestamps.append(float(row[0]))
        except ValueError:
            continue
    if not timestamps:
        return f"trajectory file exists but has no parseable pose rows ({path})"
    return (
        f"partial trajectory {path} has {len(timestamps)} pose row(s) over "
        f"[{min(timestamps):.6f}, {max(timestamps):.6f}]s"
    )


def detect_mast3r_slam_capabilities(path: Path) -> Mast3rSlamCapabilities:
    """Report whether an upstream checkout advertises a pose-prior CLI flag.

    The released upstream MASt3R-SLAM CLI accepts calibration but does not
    currently advertise pose priors. In that case callers retain ARKit/Stray
    poses for post-run trajectory alignment instead of passing an unsupported
    option to MASt3R-SLAM.
    """

    entrypoint = path.expanduser().resolve() / "main.py"
    if not entrypoint.is_file():
        return Mast3rSlamCapabilities(
            False,
            None,
            (f"MASt3R-SLAM entry point not found: {entrypoint}",),
        )
    try:
        source = entrypoint.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Mast3rSlamCapabilities(
            False,
            None,
            (f"Could not inspect MASt3R-SLAM CLI: {exc}",),
        )

    for argument in _POSE_PRIOR_ARGUMENTS:
        if re.search(
            rf"add_argument\(\s*['\"]{re.escape(argument)}['\"]", source
        ):
            return Mast3rSlamCapabilities(True, argument)
    return Mast3rSlamCapabilities(
        False,
        None,
        (
            "The MASt3R-SLAM checkout does not advertise a pose-prior option; "
            "ARKit/Stray poses will be used only for post-run alignment.",
        ),
    )


def build_rgb_video_command(
    video_path: Path,
    config: str = "config/base.yaml",
    *,
    python_executable: str | None = None,
    save_as: str | None = None,
    no_viz: bool = False,
    pose_priors_path: Path | None = None,
    pose_prior_argument: str | None = None,
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
    if pose_priors_path is not None and pose_prior_argument is not None:
        command.extend([pose_prior_argument, str(pose_priors_path.expanduser().resolve())])
    return command


def run_rgb_video(
    video_path: Path,
    mast3r_slam_dir: Path,
    config: str = "config/base.yaml",
    *,
    python_executable: str | None = None,
    save_as: str | None = None,
    no_viz: bool = False,
    pose_priors_path: Path | None = None,
    run_process: RunProcess = subprocess.run,
) -> Mast3rSlamInvocation:
    """Run MASt3R-SLAM on one uncalibrated RGB video.

    The process inherits stdout/stderr so MASt3R-SLAM's progress and failures
    stay visible to the caller.  Its working directory is its checkout because
    its config files and checkpoints use repository-relative paths.
    """

    root = _resolve_mast3r_slam_dir(mast3r_slam_dir)
    capabilities = detect_mast3r_slam_capabilities(root)
    pose_prior_mode = "not_requested"
    if pose_priors_path is not None:
        pose_prior_mode = (
            "upstream" if capabilities.supports_pose_priors else "post_alignment"
        )

    command = build_rgb_video_command(
        video_path,
        config,
        python_executable=python_executable,
        save_as=save_as,
        no_viz=no_viz,
        pose_priors_path=pose_priors_path,
        pose_prior_argument=capabilities.pose_prior_argument,
    )
    try:
        completed = run_process(command, cwd=root, check=False)
    except OSError as exc:
        raise Mast3rSlamError(f"Could not start MASt3R-SLAM: {exc}") from exc

    invocation = Mast3rSlamInvocation(
        tuple(command), root, completed.returncode, capabilities, pose_prior_mode
    )
    if completed.returncode != 0:
        trajectory_path = _expected_trajectory_path(root, video_path, save_as)
        raise Mast3rSlamError(
            f"MASt3R-SLAM exited with status {completed.returncode}; "
            f"{_partial_trajectory_summary(trajectory_path)}; "
            f"expected trajectory: {trajectory_path}; command: "
            f"{' '.join(invocation.command)}"
        )
    return invocation
