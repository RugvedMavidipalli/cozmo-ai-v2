#!/usr/bin/env python3
"""Run the official RoomFormer model with a faithful SceneCAD density input.

The model output is always labelled as an image-space hypothesis.  Pipeline
wall dimensions come only from the reconstruction result and are drawn in a
separate colour plus a complete legend; RoomFormer never creates dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import torch
from shapely.geometry import Polygon

from cozmo_ai_v2.pipeline.roomformer_input import (
    plan_to_roomformer_pixels,
    scenecad_density_from_plan,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roomformer-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--display-scale", type=int, default=6)
    return parser.parse_args()


def _load_result(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"pipeline result does not exist: {path}")
    return json.loads(path.read_text())


def _plan_points(input_dir: Path, result: dict[str, Any]) -> np.ndarray:
    cloud_path = input_dir / "cloud.ply"
    points = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    if not len(points):
        raise RuntimeError(f"pipeline cloud is empty: {cloud_path}")
    transform = result["diagnostics"]["geometry"]["grid"]["transforms"]["world_to_plan"]
    right = np.asarray(transform["right"], dtype=np.float64)
    forward = np.asarray(transform["forward"], dtype=np.float64)
    return np.column_stack((points @ right, points @ forward))


def _load_model(repository: Path, checkpoint_path: Path, device: str):
    for candidate in (repository, repository / "diff_ras", repository / "models" / "ops"):
        value = str(candidate.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    from models import build_model  # noqa: PLC0415 - external optional runtime

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    args.device = device
    args.masked_attn = getattr(args, "masked_attn", False)
    args.aux_loss = False
    args.semantic_classes = -1
    model = build_model(args, train=False).to(device).eval()
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    return model, list(missing), list(unexpected)


def _predict_polygons(model: Any, image: np.ndarray, device: str) -> list[dict[str, Any]]:
    with torch.no_grad():
        outputs = model([torch.from_numpy(image[None, ...]).float().div(255.0).to(device)])
    probabilities = torch.sigmoid(outputs["pred_logits"][0]).cpu().numpy()
    coordinates = outputs["pred_coords"][0].cpu().numpy()
    polygons: list[dict[str, Any]] = []
    for index, (scores, normalized) in enumerate(zip(probabilities, coordinates)):
        corners = normalized[scores > 0.5]
        if len(corners) < 4:
            continue
        pixels = corners * np.array([image.shape[1] - 1, image.shape[0] - 1], dtype=np.float32)
        shape = Polygon(pixels)
        if not shape.is_valid or shape.area < 100.0:
            continue
        polygons.append({
            "polygon_index": int(index),
            "corner_count": int(len(corners)),
            "mean_corner_confidence": float(scores[scores > 0.5].mean()),
            "normalized_corners": corners.tolist(),
            "pixel_corners": pixels.tolist(),
            "area_px2": float(shape.area),
        })
    return polygons


def _wall_rows(result: dict[str, Any], minimum: np.ndarray, span: float, grid_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, wall in enumerate(result["reconstruction"]["walls"], start=1):
        length = wall.get("length", {})
        start, end = plan_to_roomformer_pixels(
            np.asarray([wall["start"], wall["end"]], dtype=np.float64), minimum, span, grid_size
        )
        rows.append({
            "label": f"W{number:02d}",
            "wall_id": str(wall.get("name", wall.get("id", number))),
            "start_px": start.tolist(),
            "end_px": end.tolist(),
            "length_m": float(length["value"]),
            "tolerance_m": float(length.get("tolerance", length.get("half_width", 0.0))),
            "status": str(length.get("status", "unknown")),
            "flags": list(length.get("flags", [])),
        })
    return rows


def _text_with_background(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale)))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 2, y - height - 2), (x + width + 2, y + baseline + 2), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _render_overlay(image: np.ndarray, polygons: list[dict[str, Any]], walls: list[dict[str, Any]], display_scale: int) -> np.ndarray:
    height, width = image.shape
    map_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (width * display_scale, height * display_scale), interpolation=cv2.INTER_NEAREST)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon["pixel_corners"], dtype=float) * display_scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(map_image, [points], True, (0, 220, 0), max(2, display_scale // 2), cv2.LINE_AA)
    for wall in walls:
        start = tuple(np.asarray(wall["start_px"], dtype=int) * display_scale)
        end = tuple(np.asarray(wall["end_px"], dtype=int) * display_scale)
        cv2.line(map_image, start, end, (0, 150, 255), max(2, display_scale // 2), cv2.LINE_AA)
        midpoint = tuple(((np.asarray(start) + np.asarray(end)) / 2).astype(int))
        _text_with_background(map_image, wall["label"], midpoint, 0.32 * display_scale, (0, 180, 255))

    legend_width = 840
    canvas = np.full((map_image.shape[0], map_image.shape[1] + legend_width, 3), 24, dtype=np.uint8)
    canvas[:, :map_image.shape[1]] = map_image
    x0 = map_image.shape[1] + 28
    _text_with_background(canvas, "RoomFormer overlay", (x0, 54), 0.9, (255, 255, 255))
    _text_with_background(canvas, "green: RoomFormer image-space hypothesis", (x0, 98), 0.48, (0, 220, 0))
    _text_with_background(canvas, "orange: pipeline wall geometry", (x0, 132), 0.48, (0, 180, 255))
    _text_with_background(canvas, "all dimensions are pipeline measurements", (x0, 166), 0.42, (210, 210, 210))
    _text_with_background(canvas, "uncalibrated intervals; not RoomFormer output", (x0, 198), 0.38, (150, 150, 255))
    column_width = 398
    for index, wall in enumerate(walls):
        column = index // 17
        row = index % 17
        x = x0 + column * column_width
        y = 258 + row * 66
        label = f"{wall['label']}  {wall['length_m']:.2f} m +/- {wall['tolerance_m']:.2f}"
        _text_with_background(canvas, label, (x, y), 0.42, (235, 235, 235))
    return canvas


def main() -> int:
    args = _parse_args()
    result = _load_result(args.input_dir)
    points = _plan_points(args.input_dir, result)
    density = scenecad_density_from_plan(points, grid_size=args.grid_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output_dir / "density_scenecad_contract.png"), density.image):
        raise OSError("could not write RoomFormer density image")

    model, missing, unexpected = _load_model(args.roomformer_repository, args.checkpoint, args.device)
    polygons = _predict_polygons(model, density.image, args.device)
    walls = _wall_rows(result, density.minimum, density.span, density.grid_size)
    overlay = _render_overlay(density.image, polygons, walls, args.display_scale)
    if not cv2.imwrite(str(args.output_dir / "roomformer_overlay_dimensions.png"), overlay):
        raise OSError("could not write RoomFormer dimensional overlay")
    with (args.output_dir / "roomformer_wall_dimensions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "wall_id", "length_m", "tolerance_m", "status", "flags"])
        writer.writeheader()
        for wall in walls:
            writer.writerow({**wall, "flags": ";".join(wall["flags"])})
    metadata = {
        "official_repository": str(args.roomformer_repository),
        "checkpoint": str(args.checkpoint),
        "input": {
            "cloud": str(args.input_dir / "cloud.ply"),
            "point_count": int(len(points)),
            "projection": "pipeline Manhattan right/forward plan axes",
            "grid_size": density.grid_size,
            "minimum_plan": density.minimum.tolist(),
            "span_m": density.span,
            "preprocessing": "official SceneCAD count/max normalization with 5% square padding; no log scaling or vertical flip",
        },
        "load_state": {"missing_keys": missing, "unexpected_keys": unexpected},
        "polygon_count": len(polygons),
        "polygons": polygons,
        "wall_count": len(walls),
        "wall_dimensions_csv": "roomformer_wall_dimensions.csv",
        "limitations": "RoomFormer polygons are image-space layout hypotheses. Wall dimensions are drawn only from the pipeline result and remain uncalibrated where flagged.",
    }
    (args.output_dir / "roomformer_overlay_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"points": len(points), "polygons": len(polygons), "walls": len(walls), "missing": len(missing), "unexpected": len(unexpected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
