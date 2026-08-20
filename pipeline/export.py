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

FONT = "Helvetica, Arial, sans-serif"

# Distinct, low-saturation room fills so adjacent rooms read apart without
# competing with the wall linework or the damage overlay.
ROOM_FILLS = [
    "#cfe3f5", "#f6ddc9", "#d6ecd8", "#f3d3d6", "#e2dcf0", "#fdeec2",
]

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


class _LabelPlacer:
    """Greedy collision avoidance for dimension labels.

    A floor plan is only useful if its numbers can be read.  Where walls
    cluster -- a doorway jamb, a run of cabinetry -- labels land on top of each
    other and every one of them becomes unreadable.  Dropping the later,
    shorter wall's label keeps the rest legible, which is the right trade: the
    numbers are all in `result.json`, but the drawing has to be scannable.
    """

    def __init__(self, padding: float = 2.0):
        self.boxes: list[tuple[float, float, float, float]] = []
        self.padding = padding
        self.skipped = 0

    def place(self, cx: float, cy: float, text: str, size: float) -> bool:
        half_width = 0.29 * size * len(text) / 2 + self.padding
        half_height = size / 2 + self.padding
        box = (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
        for other in self.boxes:
            if (
                box[0] < other[2]
                and box[2] > other[0]
                and box[1] < other[3]
                and box[3] > other[1]
            ):
                self.skipped += 1
                return False
        self.boxes.append(box)
        return True


def render_floorplan(
    result: dict,
    path: str | Path,
    scale: float = 110.0,
    margin: float = 70.0,
    show_damage: bool = True,
    min_label_length: float = 0.9,
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

    rooms = result.get("rooms", [])
    for index, room in enumerate(rooms):
        polygon = room.get("polygon")
        if polygon and len(polygon) >= 3:
            drawing.add(
                drawing.polygon(
                    [project(p) for p in polygon],
                    fill=ROOM_FILLS[index % len(ROOM_FILLS)],
                    stroke="none",
                    opacity=0.55,
                )
            )

    placer = _LabelPlacer()
    # Room labels are claimed first: a room's name and area outrank any single
    # wall dimension for an estimator scanning the drawing.
    for room in rooms:
        _room_label(drawing, room, project, placer)

    room_centres = np.array([r["centroid"] for r in rooms]) if rooms else None

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

    if show_damage:
        _damage_overlay(drawing, result, walls, project)

    # Dimensions last, so no wall or overlay drawn later paints over a number.
    # Longest walls are labelled first, so when two labels collide the one that
    # survives is the more significant measurement.
    for wall in sorted(walls, key=lambda w: -w["length"]["value"]):
        _dimension(drawing, wall, project, placer, min_label_length, room_centres)

    _legend(drawing, height, result, placer)
    drawing.save()
    return Path(path)


def _line(drawing, a, b, colour, stroke_width, dash=None):
    kwargs = {"stroke": colour, "stroke_width": stroke_width, "stroke_linecap": "round"}
    if dash:
        kwargs["stroke_dasharray"] = dash
    drawing.add(drawing.line(a, b, **kwargs))


def _room_label(drawing, room: dict, project, placer: _LabelPlacer) -> None:
    """Room name and area at its centroid."""
    centre = project(room["centroid"])
    name = room["name"]
    area = room["area"]
    if not placer.place(centre[0], centre[1], name, 13):
        return

    drawing.add(
        drawing.text(
            name, insert=centre, fill=PALETTE["text"], font_size="13px",
            font_weight="bold", font_family=FONT, text_anchor="middle",
        )
    )
    detail = f"{area['value']:.2f} m² ±{area.get('half_width', 0):.2f}"
    height = room.get("ceiling_height", {}).get("value")
    if height:
        detail += f" · h {height:.2f} m"
    drawing.add(
        drawing.text(
            detail, insert=(centre[0], centre[1] + 15), fill=PALETTE["text"],
            font_size="10.5px", font_family=FONT, text_anchor="middle",
        )
    )
    placer.place(centre[0], centre[1] + 15, detail, 10.5)


def _dimension(
    drawing, wall: dict, project, placer: _LabelPlacer,
    min_length: float, room_centres,
) -> None:
    """Label a wall with its length and interval, offset clear of the line.

    The label is pushed perpendicular to the wall, on the side facing away from
    the nearest room centre, so it sits in open drawing space instead of on top
    of the wall it describes or inside the room's own label.
    """
    measurement = wall["length"]
    if measurement["value"] < min_length:
        return

    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
    middle = 0.5 * (start + end)
    normal = np.asarray(wall["normal"], dtype=float)
    if room_centres is not None and len(room_centres):
        nearest = room_centres[np.argmin(np.linalg.norm(room_centres - middle, axis=1))]
        if (middle - nearest) @ normal < 0:
            normal = -normal

    anchor = project(middle + normal * 0.16)
    a, b = project(start), project(end)
    angle = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    if angle > 90 or angle < -90:
        angle += 180

    label = f"{measurement['value']:.2f}"
    half = measurement.get("half_width")
    if half:
        label += f" ±{half * 100:.0f}cm"
    if wall.get("inferred_fraction", 0) > 0.15:
        label += " *"

    if not placer.place(anchor[0], anchor[1], label, 11):
        return
    drawing.add(
        drawing.text(
            label, insert=anchor, fill=PALETTE["text"], font_size="11px",
            font_family=FONT, text_anchor="middle",
            transform=f"rotate({angle:.1f} {anchor[0]:.1f} {anchor[1]:.1f})",
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


def _legend(drawing, height, result, placer: _LabelPlacer) -> None:
    entries = [
        ("wall", "wall"),
        ("door", "door"),
        ("window", "window"),
        ("inferred", "inferred / occluded span"),
    ]
    damage_classes = {r["damage_class"] for r in result.get("damage", [])}
    entries += [(c, f"{c} damage") for c in sorted(damage_classes)]

    walls = result["reconstruction"]["walls"]
    total_length = sum(w["length"]["value"] for w in walls)
    total_area = sum(r["area"]["value"] for r in result.get("rooms", []))
    summary_lines = [
        f"total wall length {total_length:.1f} m over {len(walls)} walls",
        f"total floor area {total_area:.1f} m² in {len(result.get('rooms', []))} rooms",
    ]

    y = 20
    drawing.add(
        drawing.rect(
            (8, 8), (280, 40 + 17 * len(entries) + 14 * len(summary_lines)),
            fill="white", fill_opacity=0.88, stroke="#d0d6dd",
        )
    )
    for key, label in entries:
        _line(drawing, (18, y), (44, y), PALETTE[key], 5,
              "6,4" if key == "inferred" else None)
        drawing.add(
            drawing.text(label, insert=(52, y + 4), fill=PALETTE["text"],
                         font_size="11px", font_family=FONT)
        )
        placer.place(120, y, label, 11)
        y += 17

    drawing.add(
        drawing.text(
            "dimensions in metres; ± is a 90% interval.  * = partly inferred",
            insert=(18, y + 4), fill=PALETTE["text"], font_size="10px",
            font_family=FONT,
        )
    )
    placer.place(120, y + 4, "x" * 60, 10)

    y += 20
    for line in summary_lines:
        drawing.add(
            drawing.text(
                line, insert=(18, y + 4), fill=PALETTE["text"],
                font_size="11px", font_weight="bold", font_family=FONT,
            )
        )
        placer.place(120, y + 4, line, 11)
        y += 15

    footer = []
    if placer.skipped:
        # Say so rather than let a reader assume every wall is dimensioned.
        footer.append(
            f"{placer.skipped} dimension labels omitted where they would "
            f"overlap — full values in result.json"
        )
    calibration = result.get("diagnostics", {}).get("calibration", {})
    if not calibration.get("calibrated", False):
        footer.append("intervals uncalibrated (no ground truth fitted)")

    for offset, note in enumerate(reversed(footer)):
        drawing.add(
            drawing.text(
                note, insert=(14, height - 14 - offset * 14), fill="#a04020",
                font_size="10px", font_family=FONT,
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
