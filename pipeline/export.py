"""Render the pipeline's results: floor plan, 3D scene, overlays, JSON.

The floor plan is the deliverable an estimator reads, so it carries the things
an estimator needs to trust it -- dimensions with their intervals, openings
drawn at their measured widths, occluded spans marked as inferred rather than
quietly drawn as if measured, and damage shaded on the wall it belongs to.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import svgwrite

PALETTE = {
    "wall": "#1f2933",
    "inferred": "#9aa5b1",
    "opening": "#ffffff",
    "door": "#2f6f4e",
    "window": "#2b6cb0",
    "room": "#f5f7fa",
    "text": "#3e4c59",
    "water": "#2b6cb0",
    "fire": "#c05621",
    "mold": "#2f855a",
}


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default))
    return path


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def validate(payload: dict, schema_path: str | Path) -> list[str]:
    """Validate output against the published schema, returning problems."""
    import jsonschema

    schema = json.loads(Path(schema_path).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in error.path)}: {error.message}"
        for error in validator.iter_errors(payload)
    ]


def render_floorplan(
    result: dict,
    path: str | Path,
    scale: float = 90.0,
    margin: float = 60.0,
    show_damage: bool = True,
) -> Path:
    """Dimensioned 2D floor plan as SVG."""
    walls = result["reconstruction"]["walls"]
    if not walls:
        raise ValueError("no walls to render")

    points = np.array([w["start"] for w in walls] + [w["end"] for w in walls])
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = upper - lower
    width = span[0] * scale + 2 * margin
    height = span[1] * scale + 2 * margin

    drawing = svgwrite.Drawing(str(path), size=(f"{width}px", f"{height}px"))
    drawing.add(drawing.rect((0, 0), (width, height), fill="white"))

    def project(point) -> tuple[float, float]:
        # SVG's y axis points down; the plan's points up.
        x = (point[0] - lower[0]) * scale + margin
        y = height - ((point[1] - lower[1]) * scale + margin)
        return x, y

    for room in result.get("rooms", []):
        polygon = room.get("polygon")
        if polygon and len(polygon) >= 3:
            drawing.add(
                drawing.polygon(
                    [project(p) for p in polygon],
                    fill=PALETTE["room"],
                    stroke="none",
                    opacity=0.75,
                )
            )

    openings_by_wall: dict[str, list[dict]] = {}
    for opening in result["reconstruction"].get("openings", []):
        openings_by_wall.setdefault(opening["wall"], []).append(opening)

    for wall in walls:
        start, end = np.array(wall["start"]), np.array(wall["end"])
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length

        # Draw the wall in segments so openings appear as gaps and occluded
        # spans read as inferred rather than measured.
        cuts: list[tuple[float, float, str]] = []
        for opening in openings_by_wall.get(wall["name"], []):
            u0 = opening.get("u_offset", 0.0)
            u1 = u0 + opening["width"]["value"]
            cuts.append((u0, u1, opening["kind"]))
        for span in wall.get("occluded_spans", []):
            cuts.append((span[0], span[1], "occluded"))
        cuts.sort()

        cursor = 0.0
        for u0, u1, kind in cuts:
            u0, u1 = max(0.0, u0), min(length, u1)
            if u1 <= cursor:
                continue
            if u0 > cursor:
                _line(drawing, project(start + direction * cursor),
                      project(start + direction * u0), PALETTE["wall"], 5)
            colour = {
                "door": PALETTE["door"],
                "window": PALETTE["window"],
                "pass-through": PALETTE["door"],
            }.get(kind, PALETTE["inferred"])
            dash = "6,4" if kind == "occluded" else None
            _line(drawing, project(start + direction * u0),
                  project(start + direction * u1), colour, 5, dash)
            cursor = u1
        if cursor < length:
            _line(drawing, project(start + direction * cursor), project(end),
                  PALETTE["wall"], 5)

        _dimension(drawing, project(start), project(end), wall)

    if show_damage:
        _damage_overlay(drawing, result, walls, project)

    _legend(drawing, height, result)
    drawing.save()
    return Path(path)


def _line(drawing, a, b, colour, stroke_width, dash=None):
    kwargs = {"stroke": colour, "stroke_width": stroke_width, "stroke_linecap": "round"}
    if dash:
        kwargs["stroke_dasharray"] = dash
    drawing.add(drawing.line(a, b, **kwargs))


def _dimension(drawing, a, b, wall) -> None:
    """Label a wall with its length and interval, offset clear of the line."""
    measurement = wall["length"]
    if measurement["value"] < 0.6:
        return
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    angle = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    if angle > 90 or angle < -90:
        angle += 180

    label = f"{measurement['value']:.2f}"
    half = measurement.get("half_width")
    if half:
        label += f" ±{half * 100:.1f}cm"
    if wall.get("inferred_fraction", 0) > 0.15:
        label += " (part inferred)"

    drawing.add(
        drawing.text(
            label,
            insert=mid,
            fill=PALETTE["text"],
            font_size="11px",
            font_family="Helvetica, Arial, sans-serif",
            text_anchor="middle",
            transform=f"rotate({angle:.1f} {mid[0]:.1f} {mid[1]:.1f}) translate(0 -8)",
        )
    )


def _damage_overlay(drawing, result, walls, project) -> None:
    """Shade each damage region along the wall it was fused onto."""
    by_name = {w["name"]: w for w in walls}
    for region in result.get("damage", []):
        wall = by_name.get(region["surface_ref"])
        if wall is None:
            continue
        start, end = np.array(wall["start"]), np.array(wall["end"])
        length = np.linalg.norm(end - start)
        if length < 1e-6:
            continue
        direction = (end - start) / length
        u_range = region.get("extent", {}).get("u_range", [0, length])
        offset = wall["normal"]
        shift = np.array(offset) * 0.09
        _line(
            drawing,
            project(start + direction * u_range[0] + shift),
            project(start + direction * min(u_range[1], length) + shift),
            PALETTE.get(region["damage_class"], "#000000"),
            7,
        )


def _legend(drawing, height, result) -> None:
    entries = [
        ("wall", "wall"),
        ("door", "door"),
        ("window", "window"),
        ("inferred", "inferred / occluded"),
    ]
    damage_classes = {r["damage_class"] for r in result.get("damage", [])}
    entries += [(c, f"{c} damage") for c in sorted(damage_classes)]

    y = 18
    for key, label in entries:
        _line(drawing, (14, y), (40, y), PALETTE[key], 5,
              "6,4" if key == "inferred" else None)
        drawing.add(
            drawing.text(label, insert=(48, y + 4), fill=PALETTE["text"],
                         font_size="11px", font_family="Helvetica, Arial, sans-serif")
        )
        y += 17

    note = result.get("diagnostics", {}).get("calibration", {})
    if not note.get("calibrated", False):
        drawing.add(
            drawing.text(
                "intervals uncalibrated (no ground truth fitted)",
                insert=(14, height - 14), fill="#a04020", font_size="10px",
                font_family="Helvetica, Arial, sans-serif",
            )
        )


def export_scene(
    mesh,
    walls: list[dict],
    path: str | Path,
    floor_height: float,
    ceiling_height: float | None = None,
) -> Path:
    """Write the reconstruction as GLB with each surface a named node.

    The assignment asks for every wall, floor and ceiling to be an identifiable
    named plane, so the fitted surfaces are exported as separate named quads
    alongside the fused mesh -- a viewer can select `room_1.north_wall` rather
    than hunting through one merged soup of triangles.
    """
    import trimesh

    scene = trimesh.Scene()
    if mesh is not None and len(mesh.vertices):
        scene.add_geometry(
            trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_normals=np.asarray(mesh.vertex_normals)
                if len(mesh.vertex_normals)
                else None,
                process=False,
            ),
            node_name="reconstruction",
        )

    top = ceiling_height if ceiling_height is not None else floor_height + 2.4
    for wall in walls:
        quad = _wall_quad(wall, floor_height, top)
        if quad is not None:
            scene.add_geometry(quad, node_name=wall["name"])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))
    return path


def _wall_quad(wall: dict, floor_height: float, ceiling_height: float):
    """A named, selectable plane for one wall, spanning floor to ceiling.

    Plan coordinates are 2D in the gravity-aligned frame; the third axis is
    height, so the quad is the wall's line swept vertically.
    """
    import trimesh

    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
    if np.linalg.norm(end - start) < 1e-6:
        return None
    vertices = np.array(
        [
            [start[0], floor_height, start[1]],
            [end[0], floor_height, end[1]],
            [end[0], ceiling_height, end[1]],
            [start[0], ceiling_height, start[1]],
        ]
    )
    return trimesh.Trimesh(
        vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False
    )


def render_damage_overlays(
    frames, detections_by_frame, masks_by_frame, out_dir: str | Path
) -> list[Path]:
    """Per-frame images with fused damage masks drawn on, for the report."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for frame in frames:
        image = cv2.cvtColor(frame.color.copy(), cv2.COLOR_RGB2BGR)
        for detection, mask in zip(
            detections_by_frame.get(frame.index, []),
            masks_by_frame.get(frame.index, []),
        ):
            colour = {
                "water": (176, 108, 43),
                "fire": (33, 86, 192),
                "mold": (74, 133, 47),
            }.get(detection.damage_class, (0, 0, 255))
            overlay = image.copy()
            overlay[mask.mask] = colour
            image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
            x0, y0, x1, y1 = (int(v) for v in detection.bbox)
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(
                image,
                f"{detection.damage_class} {detection.confidence:.2f}",
                (x0, max(y0 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2,
            )
        target = out_dir / f"frame_{frame.index:06d}.jpg"
        cv2.imwrite(str(target), image)
        written.append(target)
    return written
