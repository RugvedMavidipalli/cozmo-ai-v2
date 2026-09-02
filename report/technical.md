# Capture-to-scope: final technical report

**Evidence cutoff:** 2026-09-02 · **source commit:** `086e742c`
([`origin/main`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/tree/086e742c64edf132152bcd26b352c350561b2165))

This report separates **implemented design**, **unit/synthetic-test evidence**,
**measured local outputs**, **design targets**, and **unvalidated integrations**.
Every number is either in a cited repository/validation artifact or is a
parameter in the pinned source. Local result files are untracked and their
hashes are recorded in the [evidence register](evidence_register.md); the
separate VM outputs are identified by validation-root path and command file.
Neither is release acceptance evidence without the stated qualification.

The finalized GPU validation is a separately pinned PR-8 run, not a claim
about the current all-frame job: its source SHA, command files, environment,
and output paths are recorded in [R13].

## 1. Architecture and runtime contracts

The executable path is an ordered dataflow:

```text
capture → ingest → pose refinement → depth/pose contract → fusion
        → provenance sampling → gravity/2-D walls → wall refinement
        → room faces → openings/surfaces → measurements → damage → scope → export
```

This order is the `run()` implementation, not a product aspiration: ingest
constructs a capture bundle; fusion emits the point cloud; geometry fits walls;
room extraction consumes walls and observed floor; later stages consume those
contracts and export `result.json`, geometry files, CSVs, and overlays
[R1][R7]. The result schema requires capture, reconstruction, rooms, damage,
scope, and diagnostics fields [R7].

The invariant boundaries are deliberate. Ingest owns pose axes, units,
intrinsics, timestamps, and optional IMU. The frame contract records the depth
source, units, confidence/max-range policy, frame decisions, and video
availability. Geometry uses gravity-aligned plan coordinates; its wall path is
sequential RANSAC → Manhattan/off-axis handling → collinear cleanup → ray
occlusion filtering → crossing/corner refinement. Room topology is formed from
the wall graph and is retained only when observed-floor evidence exists
[R3][R5]. Damage is accumulated on wall surface grids, and scope is downstream
of the resulting regions; a damage call never changes the fitted geometry
[R1][R11].

## 2. Tier design and device/input matrix

The tiers below describe code paths and required artifacts, not supported-device
marketing. No named phone, camera, GPU, or accuracy certification is asserted
by the repository.

| Tier | Input contract implemented in code | Runtime status |
|---|---|---|
| A · raw LiDAR capture | A directory is detected as Stray Scanner only when it contains `rgb.mp4` and `camera_matrix.csv`; end-to-end loading additionally requires `odometry.csv` and either raw `depth/*.png` or a dense artifact. Confidence and IMU are optional. Raw depth is converted from millimetres to metres; poses are camera-to-world in OpenCV axes. | Unit-tested detection and loader contracts; no end-to-end device validation is claimed [R2][R3]. |
| B · dense-depth substitution | `load_capture` accepts a QC-probed dense-depth directory when raw depth is absent and deliberately marks `has_depth=False`. The frame contract requires a QC-approved dense entry at native RGB shape; invalid dense frames may fall back to the same-index raw LiDAR frame. | Implemented contract; PR-8 GPU E2E measured on a full-duration stride-60 subset: 91 samples, 69 fused frames, 33 walls, 11 openings, 0 rooms. This is not all-frame or device acceptance [R3][R13]. |
| C · plain video / offline pose | A standalone `.mp4`, `.mov`, `.avi`, or `.mkv` is detected as `PLAIN_VIDEO`. The runner exposes `--slam-poses`, dense-depth, manifest, and pose-source options, but `pipeline run` still consumes a capture directory and requires a pose/depth/intrinsics contract. | Detection is unit-tested. The full-video MASt3R-SLAM run failed the Sim(3) safety gate; it was not fused or loop-closed. Standalone-video reconstruction remains unvalidated [R2][R3][R13]. |
| Reject | Missing required files, an unsupported extension, an unrelated directory, or absent raw/dense depth produces an explicit detection/load error. | Unit-tested for detection; this is an input failure, not graceful reconstruction [R2][R3]. |

The frame association default is PTS, with an explicit index mode. Video
decoding records terminal sidecar gaps rather than shifting later depth or pose
indices [R3]. This makes a malformed or truncated capture visible in the
diagnostics, but does not repair missing frames.

## 3. Drift handling and quantitative error budget

Drift is measured as the spread of per-visit median wall offsets. A visit is a
continuous observation run; a gap greater than the implemented 3-second
threshold separates visits. This measures coherent trajectory displacement,
not just within-visit point scatter [R4]. Pose refinement uses the raw CSV
trajectory as its prior. ICP supplies loop-closure evidence; the candidate is
accepted only if the weighted graph objective and independent loop residual
improve, the loop gap does not worsen by more than 0.05 m, and the maximum
correction is within the 0.75 m bound. Otherwise raw poses are retained
[R4].

The engineering failure that motivated the current design is recorded in the
historical commit: chained pairwise-ICP sequential edges reported 7.3 mm drift
while stretching a 2.99 m storey to 4.48 m with metre-scale corrections. That is
a commit record of the failed experiment, not a new run or a quality result
[R8].

The reportable interval model is explicit. For a plane fit with residual RMS
`r`, support count `N`, and coherent revisit spread `d`:

```text
σ_plane = sqrt(max(r / sqrt(N), 0.002 m)^2 + d^2)
wall half-width = z · 2σ_plane · (1 + 1.5·inferred_fraction) · scale · modality
```

At the default 0.90 coverage target, the implementation uses `z = 1.645`.
The no-LiDAR modality multiplier defaults to 3.0; a supplied calibration file
can replace it. These are model parameters, not measured coverage claims
[R6]. Drift is kept outside the `sqrt(N)` term because the code models it as
coherent across a visit; inferred extent widens the corner-derived length
interval [R6].

## 4. Calibration analysis and evidence status

Calibration is a separate, unearned state. `fit_calibration` requires paired
predictions, known truths, and uncalibrated half-widths; it sets `scale` to the
target empirical quantile of normalized errors. With no supplied truth fit, the
implementation default is `calibrated:false`, `scale:1.0`, and
`coverage_target:0.90`; no laser reference set is checked in, so no accuracy or
interval-coverage gate is scored [R6][R7].

The PR-8 E2E output likewise reports uncalibrated confidence intervals. Its
GPU measurements are throughput/resource observations, not accuracy or
interval-coverage evidence [R6][R13].

No measured accuracy number is promoted here from the untracked baseline or
historical sensor JSON files: their producing build/configuration is not
embedded. They remain listed, hashed, and explicitly excluded from acceptance
claims in the evidence register. The only calibration state asserted for this
report is that implementation default [R6].

## 5. Engineering fix-loop: from plausible output to diagnosable output

The fix loop has three evidence-backed steps:

1. **Pose failure.** The historical pairwise-ICP experiment made the drift
   metric look better while corrupting scale. The implementation moved ARKit
   to sequential edges, restricted ICP to revisits, and added acceptance
   tests for objective/residual/gap regressions and no-loop fallback [R4][R8].
2. **Geometry ambiguity.** Wall extraction now records lifecycle counts and
   reasons, endpoint-gap quantiles, polygonization decisions, grid transforms,
   room fallback, and zero-room reasons. Synthetic tests verify a bounded
   square produces one wall-graph room and an empty grid reports
   `no-bounded-wall-faces` and `no-observed-free-cells` [R5].
3. **Recordings-2 topology failure.** The required failure is present in the
   untracked zero-room artifact: the same 5,443-frame/90.74-second capture
   yields 0 rooms and 31 walls, with no geometry diagnostics. The old output
   cannot say whether wall loss, endpoint connectivity, polygonization, or
   floor evidence was decisive [R10].

   A separately pinned PR-8 GPU run on a 91-sample full-duration temporal
   subset produced a schema-valid result with 69 fused frames, 33 walls, 11
   measured openings from 13 RGB observations, and 0 closed rooms. This
   establishes a traceable integration result for that subset, not an
   improvement over the local artifacts or a room-closure fix [R13].

The first real PR-6-labeled run is more specific: it records 1 room from
`observed_floor_components`, `fallback_used:true`, and zero candidate/accepted
wall-graph faces. Rerun-1 and closure-1 also record one fallback room and zero
graph faces. A later closure-10 artifact, paired with a same-directory
`fusion_manifest`, records the raw-LiDAR/`min_confidence=1`/`max_depth=3.5`/
refined-ARKit/PTS configuration and 2 accepted graph faces/2 rooms; it also
records a terminal RGB/sidecar shortfall and no ceiling. The labeled outputs
have no producing build SHA, the runs are not identical evaluation conditions,
and PR-6 is still **draft and unmerged**. The correct conclusion is diagnostic
coverage plus variable unvalidated trials—not closure or success [R10].

## 6. Known failure modes and limitations

- **Metric accuracy is unvalidated.** There is no checked-in laser truth,
  calibrated result, device qualification, or standards-compliance result.
  GPU validation below is an integration/resource result, not accuracy
  evidence. Benchmark gates that require references remain unscored [R6][R7].
- **Recordings-2 topology remains an acceptance risk.** Zero rooms are directly
  observed in one run; the unmerged closure trials are variable and must not be
  presented as a fix. A final acceptance run needs a pinned build, exact
  invocation, clean output, and truth/inspection criteria [R10].
- **Incomplete vertical evidence propagates.** The unvalidated recordings-2
  outputs and closure-10 trial warn that no ceiling plane was found, so heights
  and wall-opening heights are unavailable [R10].
- **Standalone video is a contract, not a validated product path.** Detection
  accepts a plain video, but reconstruction still needs poses, intrinsics, and
  raw or QC-approved dense depth. The finalized full-video MASt3R-SLAM run
  processed 30-fps RGB at 2.756 FPS for 16m46s, but failed the ARKit Sim(3)
  safety gate: 27 matches/26 inliers, translation RMSE 0.794 m, rotation RMSE
  158.38°, and scale 2.60345. It was not fused or loop-closed [R2][R3][R13].
- **GPU validation is bounded.** The finalized VM was Ubuntu 22.04 with an
  NVIDIA L4 (23,034 MiB), driver 580.126.20, CUDA 12.4, Python 3.10.12, 28
  vCPUs, and 121 GiB RAM. Metric3D covered the full 90.7 s capture only at
  stride 60: 91 samples, 69 QC-approved and 22 rejected; its runtime was
  24:08.48, with 986 MiB peak GPU and 34.3 GiB peak RSS. The PR-8 E2E stage
  ran in 88.8 s with 2,812 MiB peak GPU and 2.7 GiB peak RSS; its schema-valid
  output had 247,929 points, a 250,685-vertex/456,132-triangle mesh, and 22
  depth rejections. The incomplete all-frame stride-1 job is excluded [R13].
- **External floorplan smoke is non-validating.** RoomFormer loaded cleanly
  and emitted one image-space polygon from an old 69-frame cloud, but the
  ad-hoc runner did not meet the official SceneCAD density contract and the
  cloud had no bounded rooms. PR-3 separately failed export with
  `NameError: args is not defined` at `pipeline/cli.py:1459`; neither is a
  valid floorplan result [R13].
- **Damage model performance is unmeasured.** The code names a default VLM
  model and attempts hosted SAM 2 only with a token, falling back to local
  GrabCut on failure; those names and branches are implementation facts, not
  accuracy evidence. Current damage accumulation allocates vote grids only for
  walls, so floor/ceiling detections have no accumulation target [R11].
- **Capture integrity can still limit output.** Closure-10 reports RGB decode
  ending at index 5441 with three terminal sidecars unavailable out of an
  expected 5443; diagnostics expose this, but do not infer missing frames
  [R10].

The next evidence needed is narrow: run the pinned closure candidate and the
plain-video/dense path with recorded environments, collect laser references,
score the benchmark gates, and resolve the wall-only damage limitation. Until
then, the honest deliverable is a traceable reconstruction pipeline with
explicit uncertainty and failure diagnostics—not a certified measurement or
standards-compliant estimator.

## Sources

All repository links below point to commit `086e742c64edf132152bcd26b352c350561b2165`
and were validated 2026-09-02. The full claim-to-evidence ledger and artifact
hashes are in [`report/evidence_register.md`](evidence_register.md).

[R1] [`pipeline/cli.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/cli.py#L125-L148), [`docs/architecture.md`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/docs/architecture.md#L22-L131).

[R2] [`detect.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/detect.py#L7-L57), [`test_detect.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/tests/test_detect.py#L6-L59).

[R3] [`ingest.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/ingest.py#L11-L31), [`frame_contract.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/frame_contract.py#L482-L605).

[R4] [`poses.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/poses.py#L111-L157), [`drift.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/drift.py#L11-L95), [`test_pose_refinement_acceptance.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/tests/test_pose_refinement_acceptance.py#L4-L83).

[R5] [`planes.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/planes.py#L662-L846), [`rooms.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/rooms.py#L275-L410), [`test_geometry_diagnostics.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/tests/test_geometry_diagnostics.py#L64-L190).

[R6] [`uncertainty.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/uncertainty.py#L9-L18), [`uncertainty.py` model](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/uncertainty.py#L75-L232), [`fit_calibration`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/uncertainty.py#L382-L430), [`benchmarking-and-usage.md`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/docs/benchmarking-and-usage.md#L45-L117).

[R7] [`result.schema.json`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/schema/result.schema.json#L1-L70).

[R8] Historical primary commits [`e325b718`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/commit/e325b71834d0baac9168af13b9427486a8d0cbc9) and [`d814ed34`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/commit/d814ed3407bc0b7e37a9f6a9e78429c6fc68d517). The associated untracked legacy JSON hashes (`ablations_rec1.json` `e1e729889a8c20dc2344d549c94a52dbe64f14ad8fcbbe92605bf4581fba58d2`, `gating_sweep.json` `2506cad3d41a654477bf34c06a406c4dbf948ecdfa43f77c66feb363747e7b12`, `depth_bias.json` `fb7083645c0eeef2bcb795249fc589614dd8ead28f56c8b49f41620d1c715c62`) are ledger-only because their producer config is not embedded.

[R10] Local untracked zero-room artifact SHA-256 `69dab1895678e76b58a1595272bcbcffec3ed91be52192592ce433eeaa153903`; first PR-6 real-run artifact SHA-256 `d462e5bb5446c18dbb6103cc2576f81f79eba084c75c1b380e749be5ce4fa834`; rerun-1 SHA-256 `86e3e94e1acaca1587a6e29f7db6a6cde2f7b548f87fe8119e92fc84d36ab04f`; closure-1 SHA-256 `d45d20655136b532515937925f65bf86e07efc4dd7cbcc85cc63fc57a8884f1c`; closure-10 SHA-256 `7fcac6b78a8d040800ea039cf571bca0fd828a46c023fa652070aa2e183b5437`. The closure manifests are same-directory config artifacts with SHA-256 `9479360235dc5ef2bdbf29bd5e9e8fe03495baa140596991de2ae051e8d400a9`; no build SHA is embedded. [Draft PR #6](https://github.com/RugvedMavidipalli/cozmo-ai-v2/pull/6) remains unmerged.

[R11] [`damage/fusion.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/damage/fusion.py#L348-L430), [`cli.py` damage pass](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/cli.py#L718-L779), [`vlm.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/damage/vlm.py#L16-L20), [`masks.py`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/blob/086e742c64edf132152bcd26b352c350561b2165/src/cozmo_ai_v2/pipeline/damage/masks.py#L34-L74).

[R13] Finalized GPU-VM validation packet (accessed 2026-09-02): validation
root `/home/ubuntu/cozmo-validation-20260901`, main source
[`086e742c`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/commit/086e742c64edf132152bcd26b352c350561b2165),
PR-8 source
[`2074694e`](https://github.com/RugvedMavidipalli/cozmo-ai-v2/commit/2074694e3152cbd31c58825d676699d8dbf065fc),
dataset and environment manifest, and exact command/output paths recorded
under that root. Metric3D command/output:
`stages/metric3d_full_duration_stride60_halfres_pr8/{command.txt,...}`;
PR-8 E2E command/output:
`stages/pipeline_pr8_dense_rgb_openings_fixed/{command.txt,...}`;
MASt3R output:
`external/MASt3R-SLAM/logs/validation_full_30fps_main/rgb.{txt,ply}`;
RoomFormer smoke command/output:
`stages/roomformer_scenecad_on_dense_cloud/{command.txt,...}`. Dependency
versions were pinned to [MASt3R-SLAM
`e6f4e3d`](https://github.com/rmurai0610/MASt3R-SLAM/commit/e6f4e3d474fad0e11f561482012be864ba8c3f17),
[Metric3D `eb5b6f`](https://github.com/YvanYin/Metric3D/commit/eb5b6fac0dc155e4e52f576e304fbf11655ff339),
[GroundingDINO `856dde`](https://github.com/IDEA-Research/GroundingDINO/commit/856dde20aee659246248e20734ef9ba5214f5e44),
[SAM2 `2b90b9`](https://github.com/facebookresearch/sam2/commit/2b90b9f5ceec907a1c18123530e92e794ad901a4),
and [RoomFormer `e88a7e`](https://github.com/ywyue/RoomFormer/commit/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1).
MASt3R failed the safety gate; RoomFormer was smoke-only; PR-3 detached
`12ddb6b2` failed export; PR-6 was excluded. The current all-frame Metric3D
stride-1 job was incomplete at cutoff and is excluded.
