# Part 4 fix declaration: room topology

**Base/evaluation.** Exact `origin/main` is
`64102d088c754112707d86e59067342395e6b16c`. Worker17’s pushed
`48d9a646ae4fb1ed979ed6d9dd160f30b59e6721` is a temporary, explicitly
identified unblock only; its export-crash code is not in this branch. Inputs
are `recordings-1` (7,998 files, tree SHA
`b86744f1298859e1120c750d675b6d6b3dc21517a3cd2e682d75728b1ca2b375`) and
`recordings-2` (10,890 files, tree SHA
`cd6e3e054c417a2e6b48093b032b8fd2e276c1d93ba63caa58c5e2e759b7a851`).
Environment and artifact hashes are in `report/room_topology_before.json`.

**Selected worst gate.** `recordings_2_topology_golden`: expected room count
3 +/- 1 (2..4), valid polygons, total area 10..200 m2, wall assignment >=
50%, overlap <=10%. Ranking is maximum normalized deficit relative to each
minimum/maximum/target; undefined metrics and missing references are excluded.
The temporary-unblock result is **0 rooms**, exit 1, a room-count shortfall
of 2, normalized deficit **1.000**. Completion is 2/2 and room consistency
passes only vacuously; laser GT, damage/scope references are absent.

**Hypothesis/evidence.** The current room stage has 0 bounded faces from the
fragmented finite wall graph: 66 endpoints in 56 components, p50 gap 0.403204
m and p90 gap 1.322834 m. Its fallback rejects non-simple observed-floor
components rather than claiming unknown holes. The exact worker17 result,
logs, and topology command are recorded in the before manifest.

**Intended fix.** Add topology-only continuity for same-line wall fragments
when the intervening band has observed floor evidence, using a robust
per-capture gap envelope (median + 1.5 MAD). Preserve original wall
measurements and unknown floor holes; do not change the gate, population,
inputs, config, or worker17/PR6 implementation.

**Prediction.** On the identical worker17-before versus worker17-plus-this
fix run, topology pass fraction will be **1.000** (predicted room count: 2).
