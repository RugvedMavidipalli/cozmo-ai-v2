# Part 4 fix declaration

Base: `origin/main` = `64102d088c754112707d86e59067342395e6b16c` (clean, refreshed 2026-09-02).

## Selected gate and baseline

The only failing gate with a concrete metric, threshold, available input, and
regenerable command is **pipeline completion**: process exit code `== 0`.
The exact baseline was exit **1 on 2/2 captures**, a normalized shortfall of
`abs(1 - 0) = 1.0`. This ranks ahead of room consistency, which was not
evaluable because no `result.json` was exported; no room-closure failure is
assumed. Laser, footprint, damage, scope, and runtime references are absent;
the complete machine-readable inventory is in `fix_loop_before.json`.

Baseline command (identical flags after the fix):

```text
uv run python -m cozmo_ai_v2.pipeline run /Users/rugved/Desktop/projects/cozmo-ai-v2/recordings-1 --out <output>/recordings-1 --no-damage
```

Base/input/config evidence: base SHA above; recordings-1 tree SHA256
`b86744f1298859e1120c750d675b6d6b3dc21517a3cd2e682d75728b1ca2b375`
(7,998 files, `.DS_Store` excluded); config SHA256
`7993e48b98a1d3062fba3e7737b171bf49f877c090827594fe93a93d013d6768`;
rules `d4aa5729f27cfd17157954f1ba4d61b6d63d828987b80fa0b53e44f28017723d`;
schema `4a2892886952da424086f0d7d9a64ca35cbdf7be2f3c7f599cfa1a241c0ded64`.
Environment: Python 3.11.16, uv 0.12.7, macOS 26.5.1 x86_64, numpy 1.26.4,
Open3D 0.19.0, SciPy 1.17.1, Shapely 2.1.2, jsonschema 4.26.0.

## Hypothesis, evidence, intended fix, prediction

Root cause: `_assemble` references CLI-local `args` while serializing plane
threshold metadata, so every completed reconstruction crashes at export. Both
fresh captures reached export and failed with exactly `NameError: name 'args'
is not defined`; logs and stage metrics are hashed in `fix_loop_before.json`.

Intended fix: pass the already-selected plane threshold/min-inlier values into
`_assemble` and serialize those parameters directly. No metric, threshold,
input, configuration, population, or failure is discarded or changed.

Numeric prediction for the identical after run: **pipeline completion pass
fraction = 2/2 = 1.000** (baseline 0/2 = 0.000), with schema validation then
able to run on both exported results. `tools/fix_loop.py` reproduces the base
SHA in a temporary worktree and the current checkout, preserving manifests,
logs, hashes, and both benchmark outputs.
