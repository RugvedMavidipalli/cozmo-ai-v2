"""Reproduce the Part 4 fix loop on an exact base and the current checkout.

This is deliberately an execution/manifest tool, not a tuning or fixing tool.
It runs the same geometry-only pipeline command from a temporary detached base
worktree and from the current checkout, then runs bench/run.py wherever a
result exists. Raw captures and generated artifacts stay outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


PIPELINE_ARGS = ("--no-damage",)
CONFIG_PATH = Path("report/fix_loop_config.json")
HASH_CHUNK = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest(capture: Path) -> dict:
    entries = []
    for path in sorted(capture.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        relative = path.relative_to(capture).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    canonical = "".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n"
        for item in entries
    ).encode()
    return {
        "path": str(capture),
        "file_count": len(entries),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _environment() -> dict:
    packages = {}
    for name in ("cozmo-ai-v2", "numpy", "open3d", "scipy", "shapely", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        uv = subprocess.run(
            ["uv", "--version"], check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        uv = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv": uv,
        "packages": packages,
    }


def _run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


def _run_side(
    label: str,
    repo: Path,
    capture: Path,
    output_root: Path,
    config_sha256: str,
) -> dict:
    side_dir = output_root / label
    side_dir.mkdir(parents=True, exist_ok=False)
    result_path = side_dir / "result.json"
    pipeline_log = side_dir / "pipeline.log"
    command = [
        "uv", "run", "python", "-m", "cozmo_ai_v2.pipeline", "run",
        str(capture), "--out", str(side_dir), *PIPELINE_ARGS,
    ]
    exit_code = _run_command(command, repo, pipeline_log)
    record = {
        "label": label,
        "repo": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "command": " ".join(command),
        "exit_code": exit_code,
        "result_path": str(result_path),
        "result_present": result_path.exists(),
        "pipeline_log": str(pipeline_log),
        "pipeline_log_sha256": _sha256_file(pipeline_log),
        "config_sha256": config_sha256,
    }
    if result_path.exists():
        benchmark_path = side_dir / "benchmark.json"
        benchmark_log = side_dir / "benchmark.log"
        benchmark_command = [
            "uv", "run", "python", "bench/run.py", "--result",
            str(result_path), "--out", str(benchmark_path),
        ]
        benchmark_exit = _run_command(benchmark_command, repo, benchmark_log)
        record.update(
            {
                "benchmark_command": " ".join(benchmark_command),
                "benchmark_exit_code": benchmark_exit,
                "benchmark_path": str(benchmark_path),
                "benchmark_log": str(benchmark_log),
                "benchmark_log_sha256": _sha256_file(benchmark_log),
            }
        )
        if benchmark_path.exists():
            record["benchmark"] = json.loads(benchmark_path.read_text())
    else:
        record["benchmark_status"] = "not run: pipeline produced no result.json"
    artifacts = []
    for artifact in sorted(side_dir.rglob("*")):
        if not artifact.is_file() or artifact.name == "metrics.json":
            continue
        relative = artifact.relative_to(side_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )
    artifact_canonical = "".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}\n"
        for item in artifacts
    ).encode()
    record["artifacts"] = artifacts
    record["artifact_tree_sha256"] = hashlib.sha256(artifact_canonical).hexdigest()
    (side_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    capture = args.capture.resolve()
    current_repo = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    if not capture.is_dir():
        parser.error(f"capture directory does not exist: {capture}")
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        parser.error(f"output root must be empty: {output_root}")
    config_path = current_repo / CONFIG_PATH
    config_sha256 = _sha256_file(config_path)
    manifest = {
        "schema_version": 1,
        "base_sha": args.base_sha,
        "current_sha": _git(current_repo, "rev-parse", "HEAD"),
        "capture": _input_manifest(capture),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "environment": _environment(),
        "pipeline_arguments": list(PIPELINE_ARGS),
    }

    with tempfile.TemporaryDirectory(prefix="cozmo-fix-loop-") as temporary:
        base_repo = Path(temporary) / "base-worktree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(base_repo), args.base_sha],
            cwd=current_repo, check=True,
        )
        try:
            manifest["before"] = _run_side(
                "before", base_repo, capture, output_root, config_sha256
            )
            manifest["after"] = _run_side(
                "after", current_repo, capture, output_root, config_sha256
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(base_repo)],
                cwd=current_repo, check=True,
            )

    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    return 0 if manifest["after"]["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
