#!/usr/bin/env python3
"""Reproduce the worker17 topology baseline and this branch's after run.

The baseline checkout is worker17's temporary crash-unblocked branch. The
after checkout applies only the diff from the exact main base to --fix-commit;
it never copies worker17 source into this branch. Raw captures and large
outputs remain outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "report" / "room_topology_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(root: Path) -> tuple[int, str]:
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == ".DS_Store" or relative.endswith("/.DS_Store"):
            continue
        rows.append(f"{relative}\t{path.stat().st_size}\t{sha256(path)}\n")
    encoded = "".join(rows).encode()
    return len(rows), hashlib.sha256(encoded).hexdigest()


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return process.returncode


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def artifact_hashes(directory: Path) -> dict[str, str]:
    names = ("result.json", "pipeline.log", "benchmark.json", "benchmark.log")
    return {name: sha256(directory / name) for name in names if (directory / name).exists()}


def metrics(result_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text())
    rooms = result.get("rooms", [])
    walls = result.get("reconstruction", {}).get("walls", [])
    areas = [float(room.get("area", {}).get("value", 0.0)) for room in rooms]
    assigned = sum(wall.get("room_id") is not None for wall in walls)
    return {
        "room_count": len(rooms),
        "total_area_m2": round(sum(areas), 6),
        "room_polygons": all(len(room.get("polygon") or []) >= 3 for room in rooms),
        "wall_assignment_fraction": assigned / len(walls) if walls else 0.0,
        "wall_count": len(walls),
    }


def apply_fix_diff(worktree: Path, base_sha: str, fix_commit: str) -> None:
    diff = subprocess.check_output(
        ["git", "diff", "--binary", base_sha, fix_commit], cwd=REPO
    )
    subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=worktree, input=diff, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--worker17-sha", required=True)
    parser.add_argument("--fix-commit", help="implementation commit; omit with --before-only")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--before-only", action="store_true")
    args = parser.parse_args()
    if bool(args.fix_commit) == args.before_only:
        parser.error("supply exactly one of --fix-commit or --before-only")

    args.output_root.mkdir(parents=True, exist_ok=True)
    captures = [Path(item).resolve() for item in args.capture]
    manifests = {}
    for capture in captures:
        count, tree_sha = input_manifest(capture)
        manifests[capture.name] = {"path": str(capture), "file_count": count, "tree_sha256": tree_sha}

    temporary = Path(tempfile.mkdtemp(prefix="cozmo-room-topology-"))
    before_worktree = temporary / "worker17"
    after_worktree = temporary / "after"
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(before_worktree), args.worker17_sha], cwd=REPO, check=True)
        if not args.before_only:
            subprocess.run(["git", "worktree", "add", "--detach", str(after_worktree), args.worker17_sha], cwd=REPO, check=True)
            apply_fix_diff(after_worktree, args.base_sha, args.fix_commit)

        sides = [("before", before_worktree)]
        if not args.before_only:
            sides.append(("after", after_worktree))
        manifest = {
            "schema_version": 1,
            "base_main_sha": args.base_sha,
            "temporary_unblock_sha": args.worker17_sha,
            "fix_commit": args.fix_commit,
            "config_sha256": sha256(CONFIG),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cwd": str(REPO),
            },
            "captures": manifests,
            "sides": {},
            "reproduction": "uv run python tools/room_topology_fix_loop.py --capture <path> --base-sha 64102d088c754112707d86e59067342395e6b16c --worker17-sha 48d9a646ae4fb1ed979ed6d9dd160f30b59e6721 --fix-commit <implementation-commit> --output-root /tmp/new-room-topology",
        }
        for side, worktree in sides:
            manifest["sides"][side] = {}
            for capture in captures:
                out = args.output_root / side / capture.name
                command = ["uv", "run", "python", "-m", "cozmo_ai_v2.pipeline", "run", str(capture), "--out", str(out), "--no-damage"]
                exit_code = run(command, worktree, out.parent / f"{capture.name}-{side}.log")
                record = {"command": " ".join(command), "exit_code": exit_code, "output": str(out), "input": manifests[capture.name]}
                if (out / "result.json").exists():
                    record["metrics"] = metrics(out / "result.json")
                    record["artifacts"] = artifact_hashes(out)
                    bench = out / "benchmark.json"
                    bench_log = out / "benchmark.log"
                    bench_cmd = ["uv", "run", "python", "bench/run.py", "--result", str(out / "result.json"), "--out", str(bench)]
                    record["benchmark_exit_code"] = run(bench_cmd, worktree, bench_log)
                    record["benchmark_command"] = " ".join(bench_cmd)
                    record["artifacts"] = artifact_hashes(out)
                manifest["sides"][side][capture.name] = record
            if not args.before_only and side == "after":
                result_root = args.output_root / side
                env = os.environ.copy()
                env.update({"COZMO_RECORDING_REGRESSION": "1", "COZMO_RECORDING_RESULTS": str(result_root)})
                topo_cmd = ["uv", "run", "pytest", "-q", "tests/test_recording_geometry_regression.py::test_optional_recording_geometry_golden[recordings-2]"]
                topo_log = args.output_root / "recordings-2-topology.log"
                manifest["topology_command"] = " ".join(topo_cmd)
                manifest["topology_exit_code"] = run(topo_cmd, worktree, topo_log, env)
                manifest["topology_log_sha256"] = sha256(topo_log)
        (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return 0
    finally:
        for worktree in (after_worktree, before_worktree):
            if worktree.exists():
                subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO, check=False)
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
