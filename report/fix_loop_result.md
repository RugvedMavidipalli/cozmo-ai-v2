# Part 4 fix-loop result

The declaration selected pipeline completion (`exit code == 0`) as the only
numerically rankable failing gate on exact `origin/main` `64102d088c754112707d86e59067342395e6b16c`.
Fresh runs failed `2/2` with the same export `NameError`; no result existed to
score room closure or any accuracy gate. The fix passes the selected gate on
both identical after runs: `2/2`, absolute movement `+1.0` in pass fraction;
relative movement is undefined from a zero baseline. The prediction was
`1.000`; observed `1.000`, prediction error `0.000`.

The smallest root-cause fix passes the selected plane threshold/min-inlier
values into `_assemble`; it does not alter the metric, threshold, input, or
population. Both after `result.json` files pass the pipeline’s JSON-schema and
fusion-provenance checks. `bench/run.py` exits `0` with zero room overlaps and
zero adjacency errors, but reports zero rooms, so that self-check is vacuous.

The available annotated `recordings-2` topology regression remains **FAIL**:
the after result has `0` rooms against the required `2–4` (shortfall `2` from
the lower bound), with the log identifying no bounded wall faces/usable floor
components. That is outside this export-crash fix and is not silently hidden.
No laser GT, footprint scalar, damage/scope references, or runtime reference
is present, so those gates remain honestly unevaluable.

Reproduce both sides and regenerate all machine-readable outputs with:

```text
uv run python tools/fix_loop.py --capture /path/to/recordings-1 --base-sha 64102d088c754112707d86e59067342395e6b16c --output-root /tmp/new-paired-run
```

Captured outputs, logs, manifests, and hashes are under `/private/tmp/cozmo-fix-loop-18`;
the tracked inventory is `fix_loop_after.json`. Raw captures and large meshes
are intentionally not committed; provide the Stray Scanner capture directory
with `odometry.csv`, `imu.csv`, `camera_matrix.csv`, `rgb.mp4`, `depth/*.png`,
and `confidence/*.png` at the input path.
