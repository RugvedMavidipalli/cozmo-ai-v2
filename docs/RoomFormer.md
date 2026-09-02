# RoomFormer SD-TQ: external adapter guide

> **Important boundary.** The `cozmo-ai-v2` pipeline is adapter-only for
> RoomFormer. It does not load a RoomFormer checkpoint and it does not run
> RoomFormer. Run RoomFormer SD-TQ in an external environment, produce
> precomputed JSON, and pass that JSON to the pipeline with
> `--roomformer-predictions`. If no prediction JSON is supplied, room
> extraction remains deterministic wall-graph polygonization with a fallback
> to connected observed-floor components. A pixel-only hint is **unmeasured**
> until it has been mapped to a pipeline wall in metric coordinates and its
> association/evidence has been validated.

This guide separates three things that are easy to conflate:

1. Room polygons produced by the external RoomFormer model.
2. The optional `scripts/run_roomformer_overlay.py` helper, which runs the
   checked-out upstream model as a separate process and writes a review
   overlay.
3. Opening evidence consumed by this repository's
   `RoomFormerSDTQAdapter`.

The current helper does not convert RoomFormer polygons into pipeline rooms or
opening records. The current adapter does not contain a conversion/export
command for upstream outputs. That conversion is a manual boundary.

## What the pipeline does

Room segmentation is independent of RoomFormer. `segment_rooms` first
polygonizes the fitted metric wall graph and keeps faces that contain observed
floor evidence. If no usable face is accepted, it falls back to connected
observed-floor components without filling unknown cells. The result is
recorded in `result.json` under
`diagnostics.geometry.room_segmentation`.

The `--roomformer-predictions` option is read during the surfaces/openings
stage. It accepts a JSON array, one object, or an object containing one of
`openings`, `predictions`, `instances`, `detections`, or `objects`. Each
opening must have a recognized `label`/`kind`/`category`/`class`/`type` such
as `door`, `window`, or `pass-through`; `score`, `confidence`, and
`probability` are recognized confidence keys. Unknown labels, furniture,
malformed records, and records below `--roomformer-min-confidence` (default
`0.25`) are rejected and reported in
`result.json.diagnostics.opening_rejections`.

The adapter's metric contract is:

```json
{
  "openings": [
    {
      "label": "door",
      "score": 0.90,
      "wall_index": 3,
      "u_range": [1.0, 1.9],
      "v_range": [0.0, 2.1]
    }
  ]
}
```

Here `u_range` is metres along the associated pipeline wall, measured from
that wall's start; `v_range` is metres above the pipeline floor. The equivalent
metric fields `u_offset` + `u_size`/`u_width`/`width_m` and `v_offset` +
`v_size`/`v_height`/`height_m` are also accepted. `wall_index` can be an
integer or a wall name present in the current pipeline result. The adapter
marks a record measured only when the wall association and both metric ranges
are present. It does not prove that manually supplied metric values are
physically correct; validate the mapping against the wall geometry and depth
evidence before treating it as a measurement.

There is no standalone RoomFormer JSON Schema file in this repository. The
object above is the complete handoff shape for the adapter: the JSON file may
live at any caller-chosen path, and that path must be supplied explicitly with
`--roomformer-predictions`. The pipeline writes its own final artifact to
`<FINAL_PIPELINE_OUT>/result.json`; it never discovers or imports a RoomFormer
output by filename.

A pixel-only record may carry an image `bbox`, but it must not put pixel or
normalized coordinates in `u_range` or `v_range`:

```json
{
  "openings": [
    {"label": "window", "score": 0.80, "bbox": [100, 200, 300, 900]}
  ]
}
```

That record remains `state: "unmeasured"`; it does not create width/height
dimensions, cut the floor plan, or become metric merely because a confidence
score is present. The pipeline currently does not perform pixel-to-metric
conversion for RoomFormer records.

## External upstream revision and prerequisites

The upstream project is [ywyue/RoomFormer](https://github.com/ywyue/RoomFormer).
This repository's existing validation record pins the external source to
commit [`e88a7e3a81e384e15ea5bdc02d893267a2b6cac1`](https://github.com/ywyue/RoomFormer/commit/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1),
not to a moving `main` branch. The pinned commit is the source used by the
commands below.

The upstream README says its code was tested on Linux with Python 3.8,
PyTorch 1.9.0, and CUDA 11.1. It compiles CUDA extensions for deformable
attention and differentiable rasterization, so the reproducible path below
assumes a Linux NVIDIA/CUDA machine. CPU execution is not a supported or
validated RoomFormer path in this project. The `cozmo-ai-v2` core geometry
path can run without an NVIDIA GPU, but that does not make external
RoomFormer CPU inference supported.

Clone the exact revision in a separate checkout:

```bash
ROOMFORMER_ROOT=/path/to/RoomFormer
git clone https://github.com/ywyue/RoomFormer.git "$ROOMFORMER_ROOT"
git -C "$ROOMFORMER_ROOT" checkout --detach e88a7e3a81e384e15ea5bdc02d893267a2b6cac1
```

Follow the upstream environment and extension setup at that revision:

```bash
ROOMFORMER_ROOT=/path/to/RoomFormer
conda create -n roomformer python=3.8
conda activate roomformer
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
cd "$ROOMFORMER_ROOT"
pip install -r requirements.txt
cd models/ops
sh make.sh
cd ../../diff_ras
python setup.py build develop
```

Those are the upstream README commands. They describe the upstream evaluator,
not a dependency group installed by this repository. This repository itself
requires Python `>=3.10`; do not install it into the upstream Python 3.8
environment. To run the repository helper, use a Python `>=3.10` environment
with this project's dependencies and a compatible PyTorch/CUDA build, with
the pinned RoomFormer extensions built against that same PyTorch. There is no
project command that reconciles the upstream Python 3.8/PyTorch 1.9 pins with
the project's Python environment automatically.

## Checkpoint acquisition and integrity

Download and extract the checkpoints from the official
[RoomFormer checkpoint archive](https://polybox.ethz.ch/index.php/s/vlBo66X0NTrcsTC),
as directed by the upstream README. For the SceneCAD room-layout model used
by this repository's overlay helper, the upstream filename is
`checkpoints/roomformer_scenecad.pth`:

```bash
ROOMFORMER_ROOT=/path/to/RoomFormer
sha256sum "$ROOMFORMER_ROOT/checkpoints/roomformer_scenecad.pth"
```

The project's validation record reports this SHA-256 for that file:

```text
b0604af4e3e37bf5530484c7e6cc57a5568118eb2247d7842f1aa833ff43d13e
```

The upstream README provides the archive link but does not publish a
per-file hash. Treat a different hash as an unverified checkpoint and record
the source, filename, and hash before using it. Do not infer a direct download
URL from the share link or commit checkpoint files to this repository.

The upstream semantically-rich SD-TQ evaluation script uses
`checkpoints/roomformer_stru3d_semantic_rich.pth`, `--semantic_classes=19`,
and Structured3D data. The official script is
[`tools/eval_stru3d_sem_rich.sh`](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/tools/eval_stru3d_sem_rich.sh).
The project has no recorded hash for that checkpoint and no converter from
its semantic output to this adapter's opening JSON.

## Upstream reference evaluation (not a project-capture handoff)

The official SceneCAD evaluation entrypoint is:

```bash
ROOMFORMER_ROOT=/path/to/RoomFormer
cd "$ROOMFORMER_ROOT"
./tools/eval_scenecad.sh
```

At the pinned revision this reads the processed COCO-format dataset under
`data/scenecad`, uses `checkpoints/roomformer_scenecad.pth`, and writes
evaluation artifacts below `checkpoints/eval_scenecad/`, including
`results.txt` and PNG visualizations. It is useful for reproducing the
upstream evaluator, but it does not accept `<BASE_PIPELINE_OUT>` and it does
not emit `ROOMFORMER_PREDICTIONS.json`.

The official SD-TQ/semantically-rich Structured3D entrypoint is:

```bash
ROOMFORMER_ROOT=/path/to/RoomFormer
cd "$ROOMFORMER_ROOT"
./tools/eval_stru3d_sem_rich.sh
```

It uses `data/stru3d`, `checkpoints/roomformer_stru3d_semantic_rich.pth`, and
`--semantic_classes=19`; its output is also PNG visualizations plus
`results.txt`, not this project's opening JSON. The upstream README documents
this semantically-rich path for Structured3D. A project capture therefore
requires the separate density-map/helper path below, followed by the manual
JSON boundary if opening evidence is to be imported.

## External inference for a project capture

The project helper is the only checked-in command that prepares this
repository's density input and runs the checked-out upstream model. First
produce a normal pipeline result without RoomFormer. Use a fresh output
directory and disable damage calls when preparing a geometry/model handoff:

```bash
COZMO_ROOT=/path/to/cozmo-ai-v2
CAPTURE_DIR=/path/to/capture
BASE_PIPELINE_OUT=/path/to/base-pipeline-out
cd "$COZMO_ROOT"
uv sync --group depth
uv run python -m cozmo_ai_v2.pipeline run "$CAPTURE_DIR" \
  --out "$BASE_PIPELINE_OUT" \
  --no-damage
```

The helper requires these exact files in `<BASE_PIPELINE_OUT>`:

```text
<BASE_PIPELINE_OUT>/result.json
<BASE_PIPELINE_OUT>/cloud.ply
```

The project helper must use extensions built against the same PyTorch that
will run it. If `uv sync --group depth` created the usual project `.venv`,
build the two upstream extensions with that interpreter before running the
helper:

```bash
COZMO_ROOT=/path/to/cozmo-ai-v2
ROOMFORMER_ROOT=/path/to/RoomFormer
ROOMFORMER_PYTHON="$COZMO_ROOT/.venv/bin/python"
cd "$ROOMFORMER_ROOT/models/ops"
"$ROOMFORMER_PYTHON" setup.py build install
cd "$ROOMFORMER_ROOT/diff_ras"
"$ROOMFORMER_PYTHON" setup.py build develop
```

These are the same upstream build entrypoints run with the project-compatible
interpreter. The official upstream Python 3.8 environment and its compiled
extensions must not be mixed with the project's Python `>=3.10` runtime.

It reads the Manhattan projection axes from
`result.json.diagnostics.geometry.grid.transforms.world_to_plan.right` and
`.forward`, projects `cloud.ply` into those plan axes, and applies the
SceneCAD preprocessing used by the pinned upstream code:

- finite plan points only;
- one square extent using the larger horizontal span, centered on the point
  bounds, with 5% padding on every side;
- rounded coordinates `(plan - minimum) / span * 256`, clipped to `[0, 255]`;
- per-pixel point counts divided by the maximum count; and
- an 8-bit 256 x 256 grayscale PNG, with no log scaling and no vertical flip.

Run the external model with the project helper from an environment that has
the project package dependencies and the pinned RoomFormer extensions:

```bash
COZMO_ROOT=/path/to/cozmo-ai-v2
ROOMFORMER_ROOT=/path/to/RoomFormer
BASE_PIPELINE_OUT=/path/to/base-pipeline-out
ROOMFORMER_OUT=/path/to/roomformer-out
cd "$COZMO_ROOT"
PYTHONPATH="$COZMO_ROOT/src" \
  uv run python scripts/run_roomformer_overlay.py \
  --input-dir "$BASE_PIPELINE_OUT" \
  --output-dir "$ROOMFORMER_OUT" \
  --roomformer-repository "$ROOMFORMER_ROOT" \
  --checkpoint "$ROOMFORMER_ROOT/checkpoints/roomformer_scenecad.pth" \
  --device cuda \
  --grid-size 256 \
  --display-scale 6
```

This command runs external model code and loads the caller-supplied
checkpoint; it is not invoked by `cozmo-ai-v2 pipeline`. The `--grid-size 256`
value is the trained RoomFormer input size. The helper's
`--device cuda` default and the extension setup above are why GPU/CUDA is
required for this path.

The helper writes these files under `<ROOMFORMER_OUT>`:

```text
density_scenecad_contract.png       # exact density image sent to the model
roomformer_overlay_dimensions.png  # image-space model hypothesis + pipeline walls
roomformer_wall_dimensions.csv     # pipeline wall measurements, not model output
roomformer_overlay_metadata.json   # run metadata and model polygon hypothesis
```

`roomformer_overlay_metadata.json` contains `official_repository`,
`checkpoint`, an `input` object with `cloud`, `point_count`, `projection`,
`grid_size`, `minimum_plan`, `span_m`, and `preprocessing`, a `load_state`
object, `polygon_count`, `polygons`, `wall_count`, and the relative
`wall_dimensions_csv` path. Each item in `polygons` contains
`polygon_index`, `corner_count`, `mean_corner_confidence`,
`normalized_corners` in `[0, 1]`, `pixel_corners` in the 256-pixel image, and
`area_px2`.

That metadata JSON is an overlay report, not the JSON consumed by
`--roomformer-predictions`. The official upstream evaluator likewise writes
PNG visualizations and `results.txt` (for example, the exact
`tools/eval_scenecad.sh` command uses the processed SceneCAD COCO dataset); it
does not write this repository's opening JSON. The helper's default
`roomformer_scenecad.pth` is also the non-semantic SceneCAD room-layout model:
it does not produce door/window SD-TQ records.

## Manual handoff into this pipeline

There is no checked-in command that converts either upstream `results.txt`,
upstream PNGs, or `roomformer_overlay_metadata.json` into
`ROOMFORMER_PREDICTIONS.json`. If an external SD-TQ run provides door/window
evidence, manually map each candidate to the current pipeline wall and write
a JSON file following the adapter contract above. Use pixel fields only as
unmeasured evidence; only write `wall_index`, `u_range`, and `v_range` after
metric mapping and evidence review.

The exact manual boundary is therefore: external RoomFormer output and
reviewed metric mapping on one side; a caller-created
`ROOMFORMER_PREDICTIONS.json` matching the adapter contract on the other. The
pipeline has no default prediction path and does not infer this conversion.

Then pass that precomputed file explicitly on a fresh pipeline run:

```bash
COZMO_ROOT=/path/to/cozmo-ai-v2
CAPTURE_DIR=/path/to/capture
FINAL_PIPELINE_OUT=/path/to/final-pipeline-out
ROOMFORMER_PREDICTIONS=/path/to/ROOMFORMER_PREDICTIONS.json
cd "$COZMO_ROOT"
uv run python -m cozmo_ai_v2.pipeline run "$CAPTURE_DIR" \
  --out "$FINAL_PIPELINE_OUT" \
  --roomformer-predictions "$ROOMFORMER_PREDICTIONS" \
  --roomformer-min-confidence 0.25 \
  --no-damage
```

The resulting `result.json` contains normalized opening records under
`reconstruction.openings`. Measured records include metric width/height
derived from `u_range`/`v_range`; pixel-only records remain
`state: "unmeasured"` with `source`/`provenance` identifying `roomformer`.
Inspect `diagnostics.opening_rejections` and the `diagnostics.opening_stage`
RoomFormer counts after the run.

## Troubleshooting

### The overlay is shifted, mirrored, or rotated

Compare `density_scenecad_contract.png` with the point cloud and the orange
pipeline wall traces in `roomformer_overlay_dimensions.png`. Confirm that the
same `<BASE_PIPELINE_OUT>/cloud.ply` and `result.json` were used, and inspect
the stored `right` and `forward` vectors under
`diagnostics.geometry.grid.transforms.world_to_plan`. Do not add a vertical
flip, use image coordinates as metres, or apply a second independent
normalization. The contract is the square, 5%-padded, count/max-normalized
256-pixel map described above.

### A JSON opening is ignored or remains unmeasured

Check that the file path passed to `--roomformer-predictions` exists and is
valid JSON, that the top-level container/key and label are recognized, and
that its score is at least `--roomformer-min-confidence`. For a measured
record, confirm that `wall_index` or wall name matches the current result and
that `u_range`/`v_range` are finite metres in the wall/floor frame. Values in
`[0, 1]` or `[0, 255]` are still interpreted as metric values if placed in
those fields; the adapter has no pixel-to-metric conversion. Put such values
in `bbox` instead and leave the record unmeasured until mapping is validated.

### RoomFormer produces a poor polygon

First review `density_scenecad_contract.png`, `load_state`, point count, and
the recorded preprocessing metadata. A poor green polygon can indicate
domain shift from the upstream SceneCAD training distribution, incomplete or
unrepresentative capture coverage, or a coordinate transform mistake. The
pipeline does not turn a poor model polygon into metric walls or room
boundaries. Treat it as an image-space hypothesis and use the deterministic
wall graph/fallback result for pipeline rooms; validate any opening mapping
against observed geometry and depth before adding metric ranges.

### No rooms are reported

RoomFormer JSON does not control room segmentation. Inspect
`result.json.diagnostics.geometry.room_segmentation` for `fallback_used` and
`zero_room_reason`, and inspect the wall graph and observed-floor evidence.
Adding a RoomFormer overlay or an opening JSON cannot make a missing wall
face into a pipeline room.

## Official references

- [RoomFormer README at the pinned revision](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/README.md)
- [RoomFormer SceneCAD preprocessing](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/data_preprocess/scenecad/scenecad_utils.py)
- [RoomFormer preprocessing notes](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/data_preprocess/README.md)
- [RoomFormer evaluation entrypoint](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/eval.py)
- [RoomFormer SceneCAD evaluator command](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/tools/eval_scenecad.sh)
- [RoomFormer SD-TQ/semantic-rich evaluator command](https://github.com/ywyue/RoomFormer/blob/e88a7e3a81e384e15ea5bdc02d893267a2b6cac1/tools/eval_stru3d_sem_rich.sh)
- [Project checkpoint/hash validation record](../report/evidence_register.md)
