# cozmo-ai-v2

cozmo-ai-v2 is a Python capture-processing pipeline for turning a calibrated
handheld walkthrough into inspectable reconstruction and scope artifacts. The
merged path is strongest for Stray Scanner captures that contain RGB video,
per-frame LiDAR depth, and ARKit/Stray odometry. It can also consume
QC-approved dense-depth artifacts and can launch or ingest MASt3R-SLAM poses.

The core stages are:

1. ingest RGB, depth, calibration, timestamps, and poses;
2. optionally refine the ARKit trajectory with pose-graph/loop-closure checks;
3. fuse depth with Open3D TSDF into a point cloud and mesh;
4. estimate gravity, floor/ceiling, structural planes, walls, corners, and
   rooms;
5. detect and fuse openings from geometry, optional RoomFormer predictions, or
   optional local Grounding DINO + SAM2 models;
6. derive plane-geometry measurements with explicit status and intervals;
7. optionally call the damage VLM, fuse detections onto wall surfaces, and
   apply rules.yaml to produce scope line items; and
8. export result.json, a floor plan, named GLB surfaces, point-cloud/mesh
   files, CSVs, and provenance diagnostics.

This README describes origin/main at the time it was audited. It does not claim
production accuracy, a minimum device qualification, a guaranteed runtime, or
a completed external-model run. See Known limitations and Validation status.

## Current capabilities and boundaries

| Area | Current status |
|---|---|
| Stray Scanner RGB + LiDAR + ARKit reconstruction | Implemented in src/cozmo_ai_v2; covered by unit/integration tests and the merged CLI. |
| TSDF fusion, structural 3D planes, walls, rooms, geometry openings, measurements, exports | Implemented; result.json is checked against schema/result.schema.json on each pipeline run. |
| Dense-depth handoff | Implemented as an explicit Metric3D v2 artifact plus QC manifest. auto uses an approved dense frame and falls back to the same-index raw LiDAR frame. |
| Standalone RGB MASt3R-SLAM launch | Implemented as cozmo-ai-v2 run; it writes into the external checkout's logs/ tree and does not itself produce result.json. |
| Damage intelligence | Optional. It uses Anthropic for selected keyframes when ANTHROPIC_API_KEY is available, then uses Replicate SAM 2 when REPLICATE_API_TOKEN is available or local GrabCut otherwise. |
| Local RGB opening models | Optional and local-only when explicitly enabled. The adapters require a local Hugging Face-compatible Grounding DINO directory, an installed sam2 package, a local SAM2 checkpoint, and a package-local SAM2 config. |
| RoomFormer | Adapter-only. This repository reads precomputed RoomFormer SD-TQ JSON; it does not run RoomFormer or load its checkpoints. Pixel-only hints remain unmeasured. |
| Accuracy and performance | Not qualified here. No unsupported historical accuracy or runtime figures are part of this guide. Intervals remain uncalibrated until calibration is fit against real ground truth. |
| RGB-only full reconstruction | Not a merged one-command capability. Standalone RGB tracking is available; full reconstruction needs a pose table, calibration, and usable depth artifact. |

The --no-damage mode is the recommended path for geometry work, privacy-
sensitive runs, and CPU-only smoke testing. It skips both damage detection and
scope generation; it does not skip reconstruction or export.

## Prerequisites

### Python and platform

The package metadata declares requires-python = ">=3.10". The repository does
not declare a supported operating-system matrix or a RAM/disk minimum. Use a
platform on which the pinned dependencies install successfully and size the
machine for the capture, image resolution, TSDF voxel size, and any optional
model. The default TSDF voxel is 0.02 metres; smaller voxels increase work and
memory use.

The core dependency set includes:

- open3d==0.19.0 and numpy<2;
- SciPy, OpenCV contrib headless, Shapely, Trimesh, scikit-image, SVGWrite,
  Matplotlib, PyYAML, jsonschema, and python-dotenv.

Open3D is used for TSDF integration, pose refinement, and point-cloud/mesh
extraction. The pipeline does not open an Open3D viewer. The project uses
opencv-contrib-python-headless; video decoding therefore depends on the codecs
available to that OpenCV build. ffmpeg is not invoked by this repository and is
optional for inspecting or transcoding a problematic video. Git is needed to
clone this repository and the external model checkouts.

Windows support is not declared by this project. For external MASt3R-SLAM and
SAM2, follow the upstream platform guidance, including WSL where their
documentation recommends it.

### CPU and GPU expectations

The core Stray/LiDAR geometry path is written to run with the base dependencies
and does not require an NVIDIA GPU. There is no current CPU performance or RAM
qualification. The optional stages have different requirements:

- `uv sync --group depth` installs PyTorch and TorchVision. Metric3D can be
  selected with `--device cpu`, `cuda`, or `mps`; model inference is optional
  and no device benchmark is promised here.
- MASt3R-SLAM is an external GPU-oriented checkout. Its official setup asks for
  a matching PyTorch/CUDA installation and checkpoint files.
- --rgb-openings defaults to --rgb-device cuda because it loads Grounding DINO
  and SAM2. Passing --rgb-device cpu is accepted by this project, but
  external-model CPU behavior is not validated here.
- If an upstream component compiles CUDA extensions, check nvcc --version and
  set CUDA_HOME as required by that component. This is an upstream requirement,
  not a core cozmo-ai-v2 environment variable.

## Installation

Run these commands from the repository root.

### Recommended: uv

uv reads pyproject.toml and uv.lock; dependency metadata is not maintained in
a separate requirements file.

~~~bash
uv sync --no-dev                         # core dependencies only
uv run python -m cozmo_ai_v2.pipeline --help
~~~

Install optional groups as needed:

~~~bash
uv sync --group dev                         # pytest
uv sync --group depth                       # torch, torchvision
uv sync --group damage                      # anthropic, replicate
uv sync --group dev --group depth --group damage
~~~

The optional depth group does not install Metric3D source, weights, or an
external MASt3R-SLAM checkout. The damage group installs client libraries;
credentials and network access are still separate choices. Grounding DINO's
Hugging Face adapter and SAM2 are not project dependency groups: install them
in a compatible environment only when using --rgb-openings (see Optional
external model setup).

With current uv behavior, plain `uv sync` also installs the project's default
development group. Use `uv sync --no-dev` when you want only the core package.

### Standard venv + pip

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate                    # POSIX shells
python -m pip install --upgrade pip
python -m pip install -e .
~~~

On Windows PowerShell, activate with .venv\Scripts\Activate.ps1. The project
only requires Python 3.10 or newer; python3.11 above is an example, not a
project pin.

Add optional dependencies explicitly:

~~~bash
python -m pip install 'pytest>=8.0'
python -m pip install 'torch>=2.0' torchvision
python -m pip install anthropic replicate
~~~

Do not install optional groups just to run --no-damage geometry. Keep the
environment that runs core reconstruction separate from incompatible upstream
model environments when that is easier to maintain.

### Secrets and local configuration

The reconstruction CLI (`uv run python -m cozmo_ai_v2.pipeline`) loads `.env`
from the repository root. The top-level helper commands do not load this file
for you. The tracked [.env.example](.env.example) contains placeholders. The
real `.env` is ignored by Git, as are out/, cache/, .venv/, model artifacts,
and PLY files. Create `.env` locally only when needed:

~~~dotenv
ANTHROPIC_API_KEY=replace-with-your-key
REPLICATE_API_TOKEN=replace-with-your-token
~~~

Alternatively export a variable in the shell that launches the pipeline. Do not
put real credentials in README examples, capture folders, output files, issue
comments, or commits. Restrict the local file if appropriate:

~~~bash
chmod 600 .env
~~~

ANTHROPIC_API_KEY is needed for an uncached Track B call. Without it, the
damage analysis records per-frame errors and the run continues with geometry
and an empty damage/scope result. REPLICATE_API_TOKEN is optional: the default
damage mask path tries the hosted meta/sam-2-large model and falls back to
local GrabCut on missing credentials or failure. --no-sam always selects local
GrabCut for damage masks. Cached VLM responses live under cache/vlm by default
and mask responses under the sibling cache/masks directory; treat both as
sensitive because they can contain derived or encoded capture content.

## Input tiers and device matrix

The full reconstruction command consumes a directory containing rgb.mp4. The
top-level prepare and densify commands additionally recognize a Stray Scanner
folder only when it contains camera_matrix.csv.

| Tier | Runner/device expectation | Minimum input for the relevant command | Pose and calibration contract | Depth and scale behavior | Status |
|---|---|---|---|---|---|
| Stray / LiDAR + ARKit | `pipeline run` uses base dependencies on CPU; `densify` accepts `cpu`, `cuda`, or `mps`; optional MASt3R uses its external environment. | rgb.mp4, odometry.csv, and non-empty depth/*.png for pipeline run; camera_matrix.csv and non-empty confidence/*.png are also required by cozmo-ai-v2 densify. imu.csv is optional. | odometry.csv contains timestamp,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy columns. Poses are camera-to-world in the measured OpenCV convention; no second ARKit-to-OpenCV flip is applied. | Raw depth PNGs are interpreted as unsigned millimetres and converted to metres. Missing confidence for the core run is treated as high confidence; the densifier requires confidence frames. | Full geometry path; optional refinement, densification, MASt3R pose validation, damage, and openings. |
| Dense capture / handoff | Consuming an approved artifact needs only the core pipeline; producing it uses `densify --device cpu|cuda|mps`. | rgb.mp4, a pose/calibration source, and a Stage 4 output containing dense_depth/, dense_confidence/, dense_qc/, and densify_manifest.json. Raw depth/ may also be present. | With ARKit, use odometry.csv. With SLAM, use --pose-source slam --slam-poses ... and camera_matrix.csv, intrinsics.yaml, or intrinsics.json as accepted by the loader. SLAM tables are camera-to-world 4x4 poses or x/y/z/quaternion rows. | A dense frame is usable only when its manifest entry is qc_approved and its QC mask, depth unit, and RGB scale are valid. auto falls back to the same index of raw LiDAR; dense rejects rather than silently falling back. Densifier output is millimetres and records scale/shift and registration in the manifest. | Implemented handoff. Dense-only full Track B is not qualified: the current damage keyframe path reads raw ingest frames, while geometry/TSDF consume the frame contract. |
| RGB-only tracking | `cozmo-ai-v2 run` delegates to the external MASt3R-SLAM environment, whose upstream setup is GPU/CUDA-oriented; CPU behavior is not validated here. | A standalone video file such as recording.mp4 for cozmo-ai-v2 run, plus an installed MASt3R-SLAM checkout. | Standalone MASt3R-SLAM is intentionally launched without --calib; its trajectory remains in MASt3R-SLAM coordinates unless a Stray odometry.csv prior is supplied for post-run alignment. | Tracking alone has no metric scale guarantee. The top-level command writes the external trajectory and pose_provenance.json; it does not write the project's result.json. | Implemented tracking adapter; full RGB-only reconstruction is not a merged one-command path. |

Raw Stray odometry uses the convention recorded in code as
camera_to_world_opencv_csv_no_arkit_to_cv_flip. MASt3R trajectory ingestion
expects camera_to_world_opencv_x_right_y_down_z_forward, with +X right, +Y
down, and +Z forward. The pipeline validates homogeneous transforms and proper
rotations; it does not infer a coordinate flip from bad data.

### Stray Scanner folder layout

~~~text
capture/
├── rgb.mp4
├── odometry.csv                 # required for the normal full run
├── camera_matrix.csv            # 3x3 CSV; required by prepare/densify
├── imu.csv                      # optional for gravity consistency
├── depth/000000.png             # uint16 millimetres; required for raw fusion
└── confidence/000000.png        # 0, 1, or 2; optional for core run, required by densify
~~~

Depth and confidence filenames are six-digit frame indices. RGB is decoded in
order. The default --frame-association pts matches decoded RGB presentation
timestamps to the odometry clock when usable and records the decision in the
manifest; --frame-association index selects identity index mapping. Missing
terminal video frames are rejected and reported rather than shifted onto an
earlier frame.

### Pose tables for a SLAM directory

When --pose-source slam is selected, the loader accepts CSV, JSON, NPY, or NPZ
pose tables. A CSV row may contain a full m00...m33 matrix or
x,y,z,qx,qy,qz,qw, with an optional timestamp/time/t. JSON may be a list or an
object containing poses/trajectory and optional timestamps/times; NPY uses
frame-rate timestamps; NPZ looks for poses/pose/trajectory and optional
timestamp arrays. The file can be passed with --slam-poses; otherwise the
loader checks conventional names inside the capture directory.

## Command reference

All commands below are copy/paste-ready from the repository root after
installation. Replace angle-bracket placeholders with paths that exist on the
machine. Do not commit the resulting captures, outputs, caches, checkpoints,
or credentials.

### Inspect help

~~~bash
uv run cozmo-ai-v2 --help
uv run cozmo-ai-v2 prepare --help
uv run cozmo-ai-v2 densify --help
uv run cozmo-ai-v2 validate-scale --help
uv run cozmo-ai-v2 run --help
uv run python -m cozmo_ai_v2.pipeline --help
uv run python -m cozmo_ai_v2.pipeline run --help
uv run python -m cozmo_ai_v2.pipeline validate-scale --help
uv run python bench/run.py --help
uv run python tools/make_report.py --help
uv run python tools/view_plan.py --help
~~~

The installed script is `cozmo-ai-v2`. The reconstruction module is
`cozmo_ai_v2.pipeline`; use the current `src/cozmo_ai_v2` package paths shown
above rather than pre-package-layout imports.

### Prepare inputs for external MASt3R-SLAM

prepare validates the input video. For a Stray folder it also converts the 3x3
camera_matrix.csv into an intrinsics.yaml containing the video width, height,
and [fx, fy, cx, cy]. It writes a manifest.json with the exact external
command it would use.

~~~bash
uv run cozmo-ai-v2 prepare <capture-dir> \
  --output-dir <prep-dir> \
  --config config/base.yaml
~~~

For a standalone video, use the file as input; no project calibration is
invented:

~~~bash
uv run cozmo-ai-v2 prepare <recording.mp4> \
  --output-dir <prep-dir> \
  --config config/base.yaml
~~~

Read <prep-dir>/manifest.json before running the printed command. prepare does
not run MASt3R-SLAM and does not download anything.

### Run standalone MASt3R-SLAM

The top-level run command is the project's adapter around an external checkout.
It requires <MAST3R_ROOT>/main.py, runs with that directory as its working
directory, passes the config relative to that checkout, and expects the
upstream trajectory at logs/<save-as>/<video-stem>.txt (or
logs/<video-stem>.txt without --save-as).

~~~bash
uv run cozmo-ai-v2 run <recording.mp4> \
  --mast3r-slam-dir <MAST3R_ROOT> \
  --python <MAST3R_PYTHON> \
  --config config/base.yaml \
  --no-viz \
  --save-as <run-name>
~~~

When a sibling Stray odometry.csv exists, the adapter discovers it; pass
--pose-priors <capture-dir>/odometry.csv explicitly when it is elsewhere.
--metrics-path can point to an optional mast3r_slam_metrics.json sidecar. The
adapter inspects main.py before launch. If that checkout does not advertise a
supported pose-prior flag, the prior is not passed as an unknown upstream
argument and is used only for post-run alignment/validation.

To use a completed MASt3R trajectory in a Stray reconstruction, do not use the
standalone command's output as if it were a project result. Pass the trajectory
to the reconstruction command instead:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name> \
  --mast3r-trajectory <MAST3R_ROOT>/logs/<run-name>/rgb.txt \
  --mast3r-metrics <MAST3R_ROOT>/logs/<run-name>/mast3r_slam_metrics.json
~~~

With a Stray prior, the integration uses robust SE(3) or Sim(3) alignment and
blocks fusion when the default gates fail: translation RMSE over 0.25 m,
translation maximum over 0.75 m, rotation RMSE over 15 degrees, rotation
maximum over 45 degrees, or scale divergence over 15%. See
mast3r_pose_provenance.json for the actual verdict. These are safety gates,
not measured accuracy claims.

### Densify a Stray capture with Metric3D v2

densify is an optional Stage 4 operation. It requires a Stray folder with
rgb.mp4, camera_matrix.csv, raw depth/*.png, and raw confidence/*.png. It
loads a local Metric3D checkout and local checkpoint; it never invokes
torch.hub downloads.

~~~bash
uv run cozmo-ai-v2 densify <capture-dir> \
  --output-dir <dense-output> \
  --metric3d-repository <METRIC3D_ROOT> \
  --weights <METRIC3D_CHECKPOINT> \
  --variant metric3d_vit_small \
  --device cuda
~~~

Useful alternatives are --device cpu or mps, --stride N for deterministic
temporal sampling, --output-scale S in (0, 1], --min-confidence 0|1|2,
--max-depth M, --guide-radius R, and --guide-eps E. The current default
variant is metric3d_vit_small; the upstream repository also exposes
metric3d_vit_large, metric3d_vit_giant2, metric3d_convnext_tiny, and
metric3d_convnext_large. Compatibility of a particular checkpoint with a
variant remains the operator's responsibility.

Use the generated artifact explicitly in the next stage:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name> \
  --dense-depth-dir <dense-output>/dense_depth \
  --densify-manifest <dense-output>/densify_manifest.json \
  --depth-source auto \
  --no-damage
~~~

The manifest is authoritative. An unapproved or malformed dense frame is
rejected; in auto, the exact same index can use raw LiDAR when present.

### Run reconstruction and geometry

CPU-only geometry smoke/operation:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name> \
  --no-damage
~~~

The full command enables Track B and C. It may make hosted calls unless all
responses are already cached:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name>
~~~

For deterministic geometry comparisons, keep the same input and record the
effective --stride, --voxel, --sdf-trunc, --max-depth,
--min-confidence, plane thresholds, pose source, and depth source in the run
notes. Set --plane-seed explicitly when comparing structural-plane results; its
default is 0.

Useful supported examples:

~~~bash
# Raw-depth ablation; do not use dense artifacts even if they are present.
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name>-raw \
  --depth-source raw \
  --no-damage

# CPU-local mask fallback for a damage run; Anthropic is still required for
# uncached VLM detections.
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name>-grabcut \
  --no-sam

# Use a supplied SLAM pose table and calibrated dense artifact in a capture
# directory with rgb.mp4, but no ARKit odometry.
uv run python -m cozmo_ai_v2.pipeline run <slam-capture-dir> \
  --out out/<capture-name>-slam \
  --pose-source slam \
  --slam-poses <POSE_TABLE> \
  --dense-depth-dir <dense-output>/dense_depth \
  --densify-manifest <dense-output>/densify_manifest.json \
  --depth-source dense \
  --no-damage
~~~

The last example exercises a supported artifact contract; it is not a claim
that a dense-only, RGB-only production capture has been device-qualified.

### Optional MASt3R launch from a Stray run

The merged pipeline can launch MASt3R-SLAM for capture/rgb.mp4 while retaining
the Stray odometry as a prior/validation reference. This is an external GPU
operation and is not a CPU smoke test:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name>-mast3r \
  --run-mast3r \
  --mast3r-slam-dir <MAST3R_ROOT> \
  --mast3r-python <MAST3R_PYTHON> \
  --mast3r-config config/base.yaml \
  --mast3r-save-as <run-name> \
  --mast3r-no-viz \
  --no-damage
~~~

The trajectory must pass the pre-fusion checks before it is used. Failures
write mast3r_pose_provenance.json and stop before fusion.

### Validate scale references

Scale validation is advisory/diagnostic. It does not silently rescale a run.
The door type is advisory because a detected door is not an automatic metric
reference.

~~~bash
uv run cozmo-ai-v2 validate-scale \
  --reference-type tape \
  --observed-m 2.01 \
  --known-m 2.00
~~~

The same flags are available under python -m cozmo_ai_v2.pipeline
validate-scale. A pipeline run can record the check with --reference-type,
--reference-observed-m, and --reference-known-m; the returned factor is
reported, not automatically applied.

## Optional external model setup

These integrations intentionally keep model downloads and external
environments out of the base package. Follow each upstream project's current
license, dependency, and checkpoint instructions; the paths below describe
only what this repository's adapters actually consume.

### MASt3R-SLAM

Follow the official
[MASt3R-SLAM README](https://github.com/rmurai0610/MASt3R-SLAM) from its own
environment. Its documented setup uses a recursive checkout, a Python 3.11
environment, a matching PyTorch/CUDA installation, and these package installs:

~~~bash
git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git <MAST3R_ROOT>
cd <MAST3R_ROOT>
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
# Install the PyTorch/CUDA combination required by the upstream README.
python -m pip install -e thirdparty/mast3r
python -m pip install -e thirdparty/in3d
python -m pip install --no-build-isolation -e .
~~~

The checkout used with the default config should contain:

~~~text
<MAST3R_ROOT>/
├── main.py
├── config/base.yaml
└── checkpoints/
    ├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    ├── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
    └── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
~~~

Those filenames and the upstream install layout are documented by the upstream
project; this repository does not ship or verify those files. The adapter
requires only the checkout root, an existing main.py, and a config string
relative to that root. It launches main.py --dataset <video> --config <config>
with the checkout as the working directory. For a standalone video it
deliberately does not pass --calib. The result parser requires eight
whitespace-separated fields per trajectory row:
timestamp x y z qx qy qz qw.

### Metric3D v2

Follow the official
[Metric3D repository](https://github.com/YvanYin/Metric3D) for source,
dependency, and checkpoint acquisition. For example, create a separate local
checkout and follow its current variant-specific dependency instructions:

~~~bash
git clone https://github.com/YvanYin/Metric3D.git <METRIC3D_ROOT>
~~~

The current adapter calls the local
checkout through PyTorch Hub with source="local" and pretrain=False, then
loads the file passed to --weights. Therefore provide both:

~~~text
--metric3d-repository <directory containing hubconf.py and Metric3D source>
--weights <local checkpoint file>
~~~

The default variant is metric3d_vit_small. The adapter recognizes the common
local checkpoint wrappers state_dict and model_state_dict; it does not silently
accept a missing file, download weights, or choose a checkpoint for you.

### RoomFormer SD-TQ

Follow the official
[RoomFormer repository](https://github.com/ywyue/RoomFormer) if you need to
generate predictions. RoomFormer is not imported by this project and its
environment/checkpoint requirements are not folded into pyproject.toml. Export
a JSON file and pass it as:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name> \
  --roomformer-predictions <ROOMFORMER_PREDICTIONS.json> \
  --no-damage
~~~

The adapter accepts common containers such as openings, predictions, instances,
detections, or objects, and recognizes door/window labels. For a metric opening,
a prediction needs a wall association plus metric u_range/v_range (or
equivalent offset/size fields). Pixel-only boxes are kept as state:
"unmeasured"; they do not create metric dimensions or cut the floor plan.
Rejections are recorded under diagnostics.opening_rejections.

### Grounding DINO + SAM2 for RGB openings

This is a distinct local-only opening path, enabled only by --rgb-openings. The
current Grounding DINO adapter does not import the original Grounding DINO
repository: it uses the Hugging Face transformers API and calls
from_pretrained(..., local_files_only=True). Install a compatible torch,
torchvision, and transformers environment, then provide a complete local
Hugging Face-compatible model directory:

~~~text
--grounding-dino-model <LOCAL_GROUNDING_DINO_MODEL_DIRECTORY>
~~~

The directory must be loadable by both `AutoProcessor.from_pretrained` and
`AutoModelForZeroShotObjectDetection.from_pretrained`; it is a complete local
Hugging Face export (configuration, processor metadata, and model weights), not
just a Python checkout. Use the official
[Grounding DINO repository](https://github.com/IDEA-Research/GroundingDINO)
and its linked Hugging Face documentation to obtain compatible code and model
files. No model identifier, version, or download URL is assumed by this
project. The adapter's prompt is door. window.; thresholds are controlled by
--rgb-box-threshold and --rgb-text-threshold.

Install the official [SAM2 repository](https://github.com/facebookresearch/sam2)
in the environment used for this command and acquire a local checkpoint using
its current instructions. The adapter requires:

~~~text
--sam2-checkpoint <LOCAL_SAM2_CHECKPOINT>
--sam2-config <CONFIG_FILE_UNDER_THE_INSTALLED_SAM2_PACKAGE>
~~~

The config is passed through SAM2's Hydra loader after the adapter verifies that
it is under the imported sam2 package. For a standard SAM2 checkout, the
official examples use a package-relative path such as
configs/sam2.1/sam2.1_hiera_l.yaml; pass the actual existing file from the
installed package rather than an unrelated copy. A portable way to discover
the package root is:

~~~bash
SAM2_PACKAGE="$(uv run python -c 'import sam2; from pathlib import Path; print(Path(sam2.__file__).resolve().parent)')"
printf '%s\n' "$SAM2_PACKAGE"
~~~

Then run the opening stage with local paths:

~~~bash
uv run python -m cozmo_ai_v2.pipeline run <capture-dir> \
  --out out/<capture-name>-rgb-openings \
  --rgb-openings \
  --grounding-dino-model <LOCAL_GROUNDING_DINO_MODEL_DIRECTORY> \
  --sam2-checkpoint <LOCAL_SAM2_CHECKPOINT> \
  --sam2-config "$SAM2_PACKAGE/configs/sam2.1/sam2.1_hiera_l.yaml" \
  --rgb-device cuda \
  --no-damage
~~~

This path requires calibrated depth because accepted RGB boxes/masks must be
associated with a wall and supported by depth. Missing paths or model packages
are reported as warnings and do not cause an implicit download. --no-sam does
not disable this RGB-opening SAM2 adapter; it controls the separate Track B
damage-mask path.

## Configuration and flags

The authoritative interface is always live --help. The following tables are
derived from src/cozmo_ai_v2/pipeline/cli.py and the installed parser at the
base commit. Flags not given a special default by the parser have the default
shown as —.

### pipeline run

Invoke as uv run python -m cozmo_ai_v2.pipeline run <capture> [options].

| Option | Default | Meaning |
|---|---:|---|
| --out | out/<capture> | Output directory. |
| --rules | repository rules.yaml | Scope rules YAML. |
| --cache-dir | cache/vlm | VLM response cache; mask cache is derived beside it. |
| --calibration | repository bench/calibration.json | Optional fitted uncertainty calibration. |
| --model | claude-opus-5 | Anthropic damage model string. |
| --stride | 4 | Main fusion frame stride. |
| --voxel | 0.02 | TSDF voxel edge in metres. |
| --sdf-trunc | 4 * --voxel | TSDF truncation in metres. |
| --max-depth | 3.5 | Inclusive depth cutoff in metres. |
| --plane-threshold | 0.03 | Structural-plane inlier/residual threshold in metres. |
| --plane-min-inliers | 30 | Minimum structural-plane support. |
| --max-planes | 80 | Maximum sequential structural planes. |
| --plane-seed | 0 | Structural-plane RANSAC seed. |
| --min-confidence | 1 | LiDAR confidence floor: 0, 1, or 2. |
| --depth-source {auto,dense,raw} | auto | Approved dense then same-index raw fallback; or force one source. |
| --frame-association {pts,index} | pts | PTS association or identity index mapping. |
| --pts-tolerance-s | derived | Maximum PTS/sidecar timestamp distance. |
| --dense-depth-dir | auto | Stage 4 dense-depth directory; defaults to <capture>/dense_depth when present. |
| --densify-manifest | auto | Stage 4 manifest; defaults beside the discovered dense directory. |
| --pose-source {auto,arkit,slam} | auto | Choose ARKit odometry or an offline SLAM pose table. |
| --slam-poses | — | CSV/NPY/NPZ/JSON SLAM pose table. |
| --damage-frames | 40 | Maximum selected VLM keyframes. |
| --min-views | 2 | Independent views required for a fused damage region. |
| --rgb-openings | off | Enable local Grounding DINO + SAM2 opening evidence. |
| --grounding-dino-model | — | Local Grounding DINO model directory. |
| --sam2-checkpoint | — | Local SAM2 checkpoint. |
| --sam2-config | — | Local config under the installed SAM2 package. |
| --rgb-device | cuda | Device for explicitly enabled RGB opening models. |
| --opening-frames | 40 | Maximum RGB opening frames. |
| --rgb-box-threshold | 0.30 | Grounding DINO box threshold. |
| --rgb-text-threshold | 0.25 | Grounding DINO text threshold. |
| --rgb-min-confidence | 0.35 | Minimum RGB opening detection confidence. |
| --roomformer-predictions | — | Precomputed RoomFormer SD-TQ JSON. |
| --roomformer-min-confidence | 0.25 | RoomFormer prediction threshold. |
| --min-detection-confidence | 0.0 | Drop VLM detections below this confidence. |
| --coverage | 0.90 | Target interval coverage. |
| --wall-thickness | 0.15 | Default thickness for derived centerline/outer areas when opposing faces are unmeasured. |
| --reference-type {marker,tape,user,door} | user | Type for an optional explicit scale check. |
| --reference-observed-m | — | Observed reference length in metres. |
| --reference-known-m | — | Known reference length in metres. |
| --no-refine | off | Use raw ARKit/SLAM poses. |
| --no-loop-closure | off | Keep refinement but disable ICP loop-closure edges. |
| --mast3r-trajectory | — | Completed trajectory to validate and use; mutually exclusive with --run-mast3r. |
| --run-mast3r | off | Launch MASt3R-SLAM for this Stray capture; requires --mast3r-slam-dir. |
| --mast3r-slam-dir | — | External MASt3R-SLAM checkout. |
| --mast3r-config | config/base.yaml | Config path relative to the external checkout. |
| --mast3r-python | current interpreter | Python executable from the MASt3R environment. |
| --mast3r-save-as | — | External MASt3R log subdirectory. |
| --mast3r-no-viz | off | Pass --no-viz upstream. |
| --mast3r-metrics | — | Optional loop-closure metrics JSON sidecar. |
| --mast3r-max-pose-gap | 1.0 | Maximum interpolation gap in seconds. |
| --no-damage | off | Skip Track B and C. |
| --no-sam | off | Force local GrabCut masks for Track B. |
| --debug-furniture | off | Ask the VLM for diagnostic furniture detections; never exports them as damage. |
| --furniture-overlays | off | With --debug-furniture, write annotated furniture images. |

--mast3r-trajectory and --run-mast3r are a mutually exclusive group. The
other MASt3R options are accepted for either a completed trajectory workflow
or the launch workflow as applicable.

### Top-level helper commands

~~~text
cozmo-ai-v2 prepare INPUT --output-dir DIR [--config CONFIG]
cozmo-ai-v2 densify INPUT --output-dir DIR [densification options]
cozmo-ai-v2 validate-scale --reference-type {marker,tape,user,door} \
  --observed-m OBSERVED --known-m KNOWN [--tolerance-m TOLERANCE]
cozmo-ai-v2 run INPUT --mast3r-slam-dir DIR [--config CONFIG] \
  [--python PYTHON] [--save-as NAME] [--no-viz] [--pose-priors FILE] \
  [--metrics-path FILE] [--pose-manifest FILE]
~~~

For exact helper defaults and all densification flags, use the help commands;
in particular, densify defaults to metric3d_vit_small, confidence 1, maximum
depth 8.0, guide radius 20, guide epsilon 100.0, stride 1, output scale 1.0,
and no implicit weights/repository/device.

## Output tree and schema

### Reconstruction output

~~~text
out/<capture-name>/
├── result.json
├── floorplan.svg
├── scene.glb
├── planes.json
├── cloud.ply
├── mesh.ply
├── fusion_manifest.json
├── openings.csv
├── scope_sketch.csv
├── scope_line_items.csv
├── damage_overlays/                 # only when damage detections are rendered
├── furniture_debug_overlays/        # only with both furniture debug flags
└── mast3r_pose_provenance.json      # only when MASt3R is supplied/launched
~~~

The exporter writes these files directly under --out; there is no additional
hidden result directory.

- result.json is canonical and follows the published schema. Its top-level keys
  are capture, reconstruction, rooms, adjacency, damage, concealed, scope, and
  diagnostics. Measurements carry value, interval bounds, coverage, confidence,
  status, basis, flags, and evidence. A null value with status: "unmeasured" is
  an explicit result, not a zero. diagnostics.warnings is part of the output,
  and schema problems make the CLI return non-zero.
- floorplan.svg shows room polygons, named walls, dimension labels, opening
  colors, dashed occluded/inferred spans, and wall-associated damage. The
  legend warns when intervals are uncalibrated; labels omitted for visual
  overlap remain in result.json.
- scene.glb contains individually named wall quads and room floor/ceiling
  surfaces for selection in a GLB viewer. It intentionally does not contain
  the dense TSDF mesh; use mesh.ply/cloud.ply for that reconstruction. When
  no ceiling is observed, the GLB uses a display-only fallback top while
  result.json keeps ceiling measurements unavailable.
- planes.json contains retained structural 3D planes, classifications,
  support/inlier identities, residuals, quality, quarantine reasons, and a
  compatible wall line where available.
- cloud.ply is the Open3D fused point cloud. mesh.ply is the triangle mesh
  extracted from the same TSDF volume.
- fusion_manifest.json records integrated, rejected, and fallback frame
  indices, depth source/units, confidence and max-depth policy, resolution
  registration, pose provenance, PTS availability, and TSDF parameters.
- openings.csv is the normalized door/window evidence table. It includes
  measured versus unmeasured state, provenance, intervals, wall association,
  source frames, depth support, and mask method.
- scope_sketch.csv contains room/wall geometry and measurement status;
  scope_line_items.csv contains surface-keyed action, material, description,
  quantity, unit, trade, rule, source, and basis fields. The scope is derived
  from rules.yaml; it is not a contractor sign-off.
- damage_overlays/ contains native-video-resolution annotated JPEGs for frames
  with accepted damage detections. furniture_debug_overlays/ is diagnostic
  only and is written only when both furniture flags are set.
- mast3r_pose_provenance.json records the external trajectory, alignment,
  optional loop metrics, safety gates, and failure diagnostics.

### Dense-depth output

~~~text
<dense-output>/
├── dense_depth/000000.png       # uint16 millimetres
├── dense_confidence/000000.png
├── dense_qc/000000.png          # nonzero pixels are QC-eligible
└── densify_manifest.json
~~~

The manifest records per-frame scale/shift, model/device strings, depth units,
resolution/scale registration, alignment residual, coverage, QC approval, and
rejections. Downstream code must not consume a dense raster without its
qc_approved manifest entry and QC mask.

## Benchmarking and report rendering

The benchmark script never invents ground truth. With only --result, it runs
the self-consistency check for room overlap and adjacency. Add only references
you actually possess:

~~~bash
# Always-available self-consistency check.
uv run python bench/run.py --result out/<capture-name>/result.json

# Laser/reference CSV. Start from the repository template and fill it locally.
cp bench/gt_TEMPLATE.csv bench/gt_<capture-name>.csv
uv run python bench/run.py \
  --result out/<capture-name>/result.json \
  --truth bench/gt_<capture-name>.csv \
  --coverage 0.90
~~~

Optional benchmark flags are --incumbent, --fit-calibration,
--footprint-reference M2, --damage-class-reference CSV, --water-reference CSV,
--iou-reference CSV, --concealed-reference CSV, --scope-reference CSV, and
--out SUMMARY.json. The reference CSV formats are printed by
`uv run python bench/run.py --help` and the implementation in `bench/run.py`;
do not label scorer thresholds as achieved accuracy.

--fit-calibration writes bench/calibration.json from the supplied truth rows.
Only use that generated file when the reference data and calibration procedure
are documented for the run. Without it, diagnostics.calibration is
uncalibrated.

Render a Markdown benchmark report from existing results:

~~~bash
uv run python tools/make_report.py \
  --results out/<capture-name>/result.json \
  --out report/<capture-name>-benchmark.md
~~~

Add --benchmark SUMMARY.json and/or --ablations ABLATIONS.json when those files
were actually generated. For a quick PNG inspection of a result:

~~~bash
uv run python tools/view_plan.py \
  out/<capture-name>/result.json \
  --out out/<capture-name>/plan_view.png
~~~

These report/render helpers consume existing artifacts; they do not run the
capture pipeline or improve a result.

## Architecture and repository layout

~~~text
src/cozmo_ai_v2/
├── cli.py                    # prepare, densify, validate-scale, standalone MASt3R
├── camera.py, video.py       # calibration parsing and OpenCV video probing
├── detect.py                 # Stray-folder versus standalone-video detection
├── manifest.py               # external MASt3R preparation manifest
├── mast3r_slam.py            # external launcher/capability inspection
├── depth/
│   ├── capture.py            # raw Stray depth contract
│   ├── model.py              # local Metric3D v2 adapter
│   ├── densify.py            # scale/shift, residual fusion, QC artifacts
│   └── align.py, fusion.py   # dense/LiDAR alignment helpers
└── pipeline/
    ├── cli.py                # merged reconstruction orchestration
    ├── ingest.py             # RGB, LiDAR, odometry, IMU, SLAM pose loading
    ├── frame_contract.py     # approved dense/raw depth and frame provenance
    ├── poses.py, slam.py     # ARKit refinement and MASt3R pose integration
    ├── fuse.py               # Open3D TSDF
    ├── geometry.py, planes.py, geometry_diagnostics.py
    ├── rooms.py              # room segmentation and adjacency
    ├── occupancy.py, openings.py, rgb_openings.py, roomformer.py
    ├── damage/               # VLM, masks, and cross-frame surface fusion
    ├── measurements.py, uncertainty.py
    ├── scope.py              # rules.yaml -> concealed flags and line items
    └── export.py              # JSON, SVG, GLB, PLY, CSV, overlays
bench/run.py                  # reference scoring and interval fitting
tools/make_report.py          # Markdown report from result artifacts
tools/view_plan.py            # PNG floor-plan inspection helper
schema/result.schema.json     # result contract
rules.yaml                   # versioned restoration rules and citations
tests/                        # pytest suite; configured testpaths
docs/                          # deeper architecture, track, and benchmark notes
~~~

The merged pipeline order is ingest -> optional pose refinement -> frame
contract/TSDF fusion -> sampling -> gravity/planes/walls -> wall refinement ->
rooms -> openings -> measurements -> optional damage -> scope -> export/schema
validation. See [docs/architecture.md](docs/architecture.md) for stage
relationships and the track documents for rationale; use source and tests as
the authority when prose conflicts with code.

## Reproducibility workflow

1. Start from a clean checkout and record the base commit (git rev-parse HEAD),
   Python version, OS, and dependency lock state (uv.lock).
2. Keep raw capture data, external checkouts, checkpoints, .env, out/, and
   caches outside the commit. Record their content hashes or operator-managed
   identifiers separately if the data policy permits.
3. Run --no-damage first to isolate ingest, pose, depth, geometry, and export.
   Inspect diagnostics.warnings, fusion_manifest.json, planes.json, and the
   schema result before enabling remote damage calls.
4. For dense depth, preserve densify_manifest.json with the output and pass it
   explicitly. Do not replace rejected frames by neighboring frames.
5. For MASt3R, preserve the external config/checkpoint identifiers and
   mast3r_pose_provenance.json. A missing metrics sidecar is recorded as
   not_reported, not as zero loop closures.
6. For benchmark work, create a filled local truth CSV, run bench/run.py, save
   its summary, and fit calibration only from the declared reference set.
7. Render the Markdown/PNG reports from the exact result.json under review.
   Compare git diff --check, tests, and schema output before sharing.

The code fixes the random structural-plane seed through --plane-seed and
records frame/depth/pose decisions. External model versions, checkpoints,
device drivers, and video codec behavior remain part of the reproducibility
record and are not guessed by this repository.

## Troubleshooting

### No module named ..., Open3D import errors, or Python mismatch

Confirm the interpreter and install mode from the repository root:

~~~bash
uv run python --version
uv run python -c 'import open3d, cozmo_ai_v2; print(open3d.__version__)'
~~~

The project requires Python 3.10+. Use `uv sync` or install the package with
`python -m pip install -e .`; confirm that commands import the current
`cozmo_ai_v2` package.

### Metric3D says weights/repository are unavailable

Install the depth group, point --metric3d-repository at a real local Metric3D
checkout, and point --weights at an existing checkpoint file. The adapter
explicitly disables automatic hub downloads. Check the variant and checkpoint
match before retrying.

### MASt3R-SLAM cannot start or has no trajectory

--mast3r-slam-dir must be the checkout root containing main.py, not its config/
or checkpoints/ directory. Confirm the external Python can import its own
dependencies, that the configured checkpoint files exist under the paths
expected by that checkout, and that the config is relative to the checkout.
The adapter expects logs/<save-as>/<video-stem>.txt with eight numeric fields per
row. A failed launch writes a failure manifest when a manifest path can be
determined.

### CUDA/OOM or optional model failures

Start with --no-damage and omit --rgb-openings. For Metric3D, use a smaller
--output-scale, a larger --stride, or an explicitly selected device. For RGB
openings, check --rgb-device, the upstream PyTorch/CUDA pairing, and
CUDA_HOME/nvcc only if the upstream installer requires them. The project does
not convert an OOM into a qualified CPU fallback for an external model.

### Missing frames, bad PTS association, or sidecar mismatch

Check that RGB, depth, and confidence filenames use the same six-digit indices
and that the odometry timestamps are sorted. Inspect
diagnostics.fusion.video_availability, frame_provenance, rejected_frames, and
fallback_frames. If the codec exposes unusable timestamps, the contract records
index_fallback; use --frame-association index only when identity mapping is known
to be correct. A missing terminal video frame is not shifted onto another
sidecar.

### Pose validation or wrong-looking geometry

Stray odometry must be camera-to-world in OpenCV axes. Do not apply the
ARKIT_TO_CV flip to the exported Stray CSV a second time. Check homogeneous
rows, proper rotation matrices, sorted timestamps, and gravity_consistency.
For MASt3R, inspect mast3r_pose_provenance.json; alignment/gate failures are
intended to stop fusion. Use --no-refine only as a diagnostic comparison, not
as evidence that raw poses are correct.

### Scale looks wrong

Raw Stray depth is converted from millimetres. Dense depth is accepted only
through its manifest-declared unit and scale. Check fusion_manifest.json and
diagnostics.fusion.tsdf_parameters. Use validate-scale to report a marker, tape,
user, or advisory door comparison; it does not silently apply the factor. For
MASt3R without a Stray prior, report the output as unaligned and unscaled rather
than treating upstream coordinates as metric.

### Zero rooms or no walls

Confirm that the run integrated frames (integrated_indices), that valid depth
survives --min-confidence/--max-depth, and that cloud.ply is non-empty. Inspect
diagnostics.geometry.room_segmentation, zero_room_reasons, wall stage counts,
and diagnostics.warnings. Sparse or grazing captures can lack the support needed
for a room polygon. Lowering thresholds may produce a different hypothesis, not
a validated result.

### Missing ceiling or missing opening heights

Ceiling fitting is quality-gated. A missing or rejected ceiling is retained as
ceiling_observed: false with diagnostics; wall opening heights and
floor-to-ceiling measurements can be unavailable. The GLB display fallback is
not a measurement. Improve coverage/ceiling visibility or treat the affected
quantities as unmeasured.

### Grounding DINO, SAM2, or RoomFormer inputs are ignored

--rgb-openings needs all three local paths: Grounding DINO model directory,
SAM2 checkpoint, and SAM2 config under the installed sam2 package. The
Grounding DINO adapter is local-files-only. RoomFormer JSON needs a recognized
door/window label; a pixel-only prediction is deliberately unmeasured. Read
diagnostics.opening_rejections and the warning text rather than assuming a
model ran.

### Schema failure

Validate the exact output with the repository schema and inspect the first
warnings:

~~~bash
uv run python - <<'PY'
import json
from cozmo_ai_v2.pipeline import export

result = json.load(open("out/<capture-name>/result.json"))
problems = export.validate(result, "schema/result.schema.json")
print("valid" if not problems else "\n".join(problems))
PY
~~~

Also check that custom dense manifests use supported depth units (mm or m),
QC masks, declared RGB scale, and non-duplicate frame indices. A schema failure
makes the pipeline return non-zero; it should not be hidden by opening the SVG
or CSV instead.

### API credentials or unexpectedly empty damage/scope

The pipeline loads only the repository-root .env when invoked through
python -m cozmo_ai_v2.pipeline. Check without printing the secret:

~~~bash
uv run python -c 'import os; print(bool(os.getenv("ANTHROPIC_API_KEY")), bool(os.getenv("REPLICATE_API_TOKEN")))'
~~~

No Anthropic key means uncached damage frames return errors and the geometry run
continues; no Replicate token means local GrabCut is used. Review
diagnostics.warnings, cache/vlm, cache/masks, and the requested flags. Use
--no-damage to guarantee a geometry-only run with no damage API call.

## Known limitations

- No accuracy, device, memory, or throughput number is qualified by this
  README. The benchmark gates are scorer definitions; they become evidence only
  when run against documented reference data.
- Confidence intervals are uncalibrated until bench/run.py --fit-calibration
  is run against appropriate ground truth. A target coverage such as 0.90 is not
  an observed coverage result.
- The core damage accumulator is wall-backed. Damage assigned to floor or
  ceiling surfaces is not currently accumulated into end-to-end Track B regions.
- Dense-only bundles are consumable by the frame contract and geometry/TSDF
  stages, but the current damage keyframe path reads raw ingest depth frames;
  do not present dense-only damage output as validated.
- Standalone RGB MASt3R-SLAM tracking is implemented, but the merged main does
  not expose a one-command plain-video-to-result.json workflow. A full run
  needs compatible pose/calibration/depth artifacts in a capture directory.
- RoomFormer predictions are imported as hints. Pixel-only or unassociated
  predictions remain unmeasured and cannot change metric geometry.
- Local RGB opening adapters depend on external package/model compatibility and
  have no bundled weights. Their live model execution is not smoke-tested by
  the base test suite.
- A missing ceiling makes height-dependent measurements and opening heights
  unavailable. A grazing or sparse capture can produce zero rooms, rejected
  planes, inferred spans, or wide intervals; read diagnostics before using a
  quantity operationally.
- Door dimensions are advisory and never silently calibrate scale. A supplied
  default wall thickness is marked as an assumption when opposing faces are
  unmeasured.
- Cached remote responses, overlays, manifests, and output diagnostics may
  expose sensitive image-derived information or source paths. Treat output as
  private until scrubbed.

## Validation status

This checklist records what was audited against the merged base. It deliberately
separates implemented code, test coverage, smoke checks, and unvalidated
external systems.

### Command audit

The following commands were run from the repository root on the audited
checkout; every parser/help command exited 0.

| Status | Command or group | Evidence |
|---|---|---|
| [x] | `uv run cozmo-ai-v2 --help`, `prepare --help`, `densify --help`, `validate-scale --help`, `run --help` | Top-level installed CLI help. |
| [x] | `uv run python -m cozmo_ai_v2.pipeline --help`, `run --help`, `validate-scale --help` | Reconstruction module parser help. |
| [x] | `uv run python bench/run.py --help` | Benchmark parser help. |
| [x] | `uv run python tools/make_report.py --help`, `tools/view_plan.py --help` | Report and floor-plan helper help. |
| [x] | `uv run cozmo-ai-v2 validate-scale --reference-type tape --observed-m 2.01 --known-m 2.00` | File-free scale smoke; returned `validated`, factor not applied. |
| [x] | `uv run python -m cozmo_ai_v2.pipeline validate-scale --reference-type tape --observed-m 2.01 --known-m 2.00` | Same safe smoke through the reconstruction module. |
| [ ] | `prepare`, `densify`, reconstruction `run`, benchmark scoring, report rendering | Not run against a real capture/result because this checkout contains no capture, weights, credentials, or filled ground truth. |
| [ ] | MASt3R-SLAM, Metric3D, RoomFormer, Grounding DINO, and SAM2 live inference | Not validated by the base smoke audit; each requires an external checkout, package, model, or checkpoint. |

The test suite covers Stray ingest, PTS/frame contracts, dense artifacts,
geometry, schema/export, poses, and adapter behavior. It does not qualify real
model weights, device support, accuracy, or performance against property ground
truth.

Run the complete local checks yourself before publishing a result:

~~~bash
uv run pytest
uv run python -m compileall -q src tests bench tools
git diff --check
~~~

The command audit for this README is intentionally help/smoke based; commands
that would require private captures, credentials, external checkouts, or
weights are documented but not represented as successful local runs.

## Unmerged orchestration note

An orchestration implementation is present on worker17's pushed branch but is
not part of origin/main at this audit. Its proposed command is
cozmo-ai-v2 pipeline ...; it is not a mainline command and must not be used as
the compatibility contract for this README until the worker's PR lands and the
command is re-audited. The merged commands above remain the supported operator
interface.

## Privacy and security

Treat a capture as sensitive: rgb.mp4, depth, odometry, IMU, calibration,
overlays, caches, manifests, and result.json can reveal a property, its layout,
device motion, and local source paths. Keep them outside version control and use
the existing ignore rules as a safeguard, not as a data retention policy.

The geometry, fusion, plane, room, measurement, and export stages are local.
When Track B has no cached response, selected keyframes are encoded and sent to
Anthropic using the configured model string. When Replicate masking is enabled,
encoded frame images and boxes are sent to the hosted meta/sam-2-large model.
Local Grounding DINO and SAM2 opening inference keeps model inference local but
still writes image-derived outputs. Review the relevant provider policies and
your organization's retention rules before enabling remote stages. Use
--no-damage, --no-sam, local-only opening models, and scrubbed output paths when
those controls fit the workflow.

Never commit .env, API tokens, checkpoint files, private capture paths, raw
media, or generated PLY/overlay artifacts. Report only sanitized diagnostics
and schema-valid outputs whose provenance is safe to share.

## Further reading

- [docs/architecture.md](docs/architecture.md) — stage order, output inventory,
  and implementation map.
- [Stray Scanner capture protocol](docs/stray-scanner-capture-protocol.md) —
  detailed capture/acquisition requirements and handoff checks.
- [Printable capture protocol](docs/stray-scanner-capture-protocol.pdf) —
  one-page print reference.
- [docs/benchmarking-and-usage.md](docs/benchmarking-and-usage.md) — detailed
  benchmark scorer formats and usage notes.
- [docs/track-a-reconstruction.md](docs/track-a-reconstruction.md) — metric
  reconstruction design.
- [docs/track-b-damage-intelligence.md](docs/track-b-damage-intelligence.md) —
  damage detection and fusion design.
- [docs/track-c-scope-generation.md](docs/track-c-scope-generation.md) —
  rule-driven scope generation.
- [schema/result.schema.json](schema/result.schema.json) — machine-readable
  result contract.
- [rules.yaml](rules.yaml) — versioned scope rules and citations.
