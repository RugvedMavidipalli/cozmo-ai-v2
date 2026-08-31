from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import svgwrite

FONT = "Helvetica, Arial, sans-serif"

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
    """Write `payload` as pretty-printed JSON, creating parent directories.

    Args:
        payload: The (possibly nested, possibly containing numpy arrays,
            dataclasses, or objects with a `to_dict` method) structure to
            serialise -- see `_default` for how non-JSON-native types are
            handled.
        path: Destination file path; its parent directory is created if
            missing.

    Returns:
        The `Path` written to, for chaining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default))
    return path


def _default(value):
    """`json.dumps`'s `default` hook: convert one non-JSON-native value.

    Tried in order: numpy arrays become lists, numpy scalars become Python
    scalars, objects with their own `to_dict` (the pipeline's own
    dataclasses that define one, e.g. `LineItem`, `ConcealedFlag`) use that,
    plain dataclasses without a `to_dict` fall back to `dataclasses.asdict`,
    and anything else is stringified as a last resort so serialisation never
    raises on an unexpected type.

    Args:
        value: The object `json.dumps` couldn't serialise natively.

    Returns:
        A JSON-serialisable representation of `value`.
    """
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
    """Validate output against the published schema, returning problems.

    Args:
        payload: The result payload to validate (typically the same dict
            passed to `write_json` for `result.json`).
        schema_path: Path to the JSON Schema file (Draft 2020-12) to
            validate against.

    Returns:
        A list of `"path/to/field: message"` strings, one per validation
        error found; empty if `payload` is fully valid.
    """
    import jsonschema

    schema = json.loads(Path(schema_path).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in error.path)}: {error.message}"
        for error in validator.iter_errors(payload)
    ]


class _LabelPlacer:
    """Keeps dimension labels on the floor plan from overlapping each other.

    A floor plan is only useful if its numbers can actually be read. Where
    walls cluster together -- a doorway jamb, a run of cabinetry -- their
    labels can land right on top of each other and all become unreadable.
    This keeps track of which spots on the drawing already have a label,
    and simply skips drawing a new one if it would collide with one that's
    already there, rather than trying to shuffle it somewhere else. The
    exact numbers are always still available in `result.json`; this is
    only about keeping the drawing itself legible.

    Whichever label asks for a spot first wins it -- a later, losing call
    just gets told no, and doesn't try anywhere else. That means the
    caller controls which labels matter more simply by controlling the
    order it asks in: `render_floorplan` draws walls longest-first, so a
    long wall's label is never bumped by a shorter wall's.
    """

    def __init__(self, padding: float = 2.0):
        """Start with no placed labels.

        Args:
            padding: Extra clearance, in SVG units, added around every
                label's estimated bounding box before collision-testing it
                against previously placed labels.
        """
        self.boxes: list[tuple[float, float, float, float]] = []
        self.padding = padding
        self.skipped = 0

    def place(self, cx: float, cy: float, text: str, size: float) -> bool:
        """Tries to claim space for one label, and remembers it if the space is free.

        There's no way to know exactly how wide a label will render at the
        point this code runs -- an SVG viewer doesn't lay out text until it
        actually draws it. So the label's width is estimated from how many
        characters it has, using a rough average width per character for
        the font this module uses. That estimate is good enough to catch
        real overlaps without needing a proper font-metrics library.

        Args:
            cx: Label's horizontal center, in SVG coordinates.
            cy: Label's vertical center, in SVG coordinates.
            text: The label's text, used only for its length.
            size: Font size, in SVG units (roughly pixels) -- drives both
                the estimated width-per-character and the label's height.

        Returns:
            True and records the label's box if it doesn't overlap any
            previously placed box; False (and increments `self.skipped`,
            without recording anything) if it does.
        """
        half_width = 0.29 * size * len(text) / 2 + self.padding
        half_height = size / 2 + self.padding
        box = (cx - half_width, cy - half_height, cx + half_width, cy + half_height)
        for other in self.boxes:
            # Standard axis-aligned bounding box overlap test: two boxes
            # overlap iff they overlap on BOTH axes simultaneously.
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
    """Draws the finished floor plan, with measurements, as an SVG file.

    The drawing is built up in layers, in this order: room-colored floor
    polygons with their name and area labels, then the walls themselves
    (broken up wherever a door, window, or occluded span interrupts
    them), then an optional damage overlay, then a length label for every
    wall, and finally a legend in the corner. Each layer is drawn on top
    of the ones before it. Room labels and wall length labels share the
    same `_LabelPlacer`, so they compete fairly for space on the page
    instead of a wall label potentially covering up a room name.

    Args:
        result: The full pipeline result dict (same shape as `result.json`)
            -- reads `reconstruction.walls`, `reconstruction.openings`,
            `rooms`, `damage`, and `diagnostics.calibration`.
        path: Output SVG file path.
        scale: Pixels per metre for the drawing.
        margin: Blank border, in pixels, around the plan's bounding box.
        show_damage: Whether to draw the damage overlay (see
            `_damage_overlay`).
        min_label_length: Minimum wall length, metres, for a dimension
            label to be drawn at all -- very short walls (e.g. a jamb stub)
            would need a label too small or cramped to read.

    Returns:
        The `Path` the SVG was written to.

    Raises:
        ValueError: If `result` has no walls to draw.
    """
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
        """Plan-space (x, y) in metres -> SVG pixel coordinates.

        SVG's y-axis points DOWN from the top-left corner, while the plan's
        own y-axis (from `HorizontalFrame`'s plan coordinates upstream)
        points UP, the usual math/CAD convention. `x` is a straight
        scale-and-shift, but `y` also has to be FLIPPED (`height - (...)`)
        or the whole drawing would render upside down relative to how the
        plan's coordinates were computed -- rooms and walls would still be
        internally consistent with each other, but mirrored vertically from
        the geometry a human expects (e.g. north would render at the
        bottom).
        """
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
    for room in rooms:
        _room_label(drawing, room, project, placer)

    room_centres = np.array([r["centroid"] for r in rooms]) if rooms else None

    openings_by_wall: dict[str, list[dict]] = {}
    for opening in result["reconstruction"].get("openings", []):
        openings_by_wall.setdefault(opening["wall"], []).append(opening)

    # Each wall is drawn as a sequence of coloured segments along its own
    # length axis `u` (0 at `start`, `length` at `end`), rather than as one
    # single line -- a solid wall colour would be wrong wherever a door,
    # window, or occlusion breaks it, so the wall is cut into pieces at
    # every opening/occluded span and each piece gets its own colour.
    for wall in walls:
        start, end = np.array(wall["start"]), np.array(wall["end"])
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length

        # Gather every "cut" along this wall -- door/window openings (their
        # u-extent from `u_offset`/`width`) and occluded spans (segments
        # ARKit's view never covered, so the wall there is geometrically
        # inferred rather than observed) -- then walk them left to right.
        cuts: list[tuple[float, float, str]] = []
        for opening in openings_by_wall.get(wall["name"], []):
            u0 = opening.get("u_offset", 0.0)
            u1 = u0 + opening["width"]["value"]
            cuts.append((u0, u1, opening["kind"]))
        for span in wall.get("occluded_spans", []):
            cuts.append((span[0], span[1], "occluded"))
        cuts.sort()  # ascending by u0, so the sweep below proceeds left to right

        # `cursor` tracks how far along the wall has been drawn so far.
        # Between the end of the previous cut and the start of the next one
        # is plain solid wall; the cut's own span gets the opening/occluded
        # colour and dash style instead.
        cursor = 0.0
        for u0, u1, kind in cuts:
            u0, u1 = max(0.0, u0), min(length, u1)
            if u1 <= cursor:
                continue  # this cut is entirely behind the cursor -- already covered
            if u0 > cursor:
                # Solid wall segment from the cursor up to this cut's start.
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
            # Remaining solid wall segment after the last cut, to the end.
            _line(drawing, project(start + direction * cursor), project(end),
                  PALETTE["wall"], 5)

    if show_damage:
        _damage_overlay(drawing, result, walls, project)

    for wall in sorted(walls, key=lambda w: -w["length"]["value"]):
        _dimension(drawing, wall, project, placer, min_label_length, room_centres)

    _legend(drawing, height, result, placer)
    drawing.save()
    return Path(path)


def _line(drawing, a, b, colour, stroke_width, dash=None):
    """Add one straight SVG line segment to `drawing`.

    Args:
        drawing: The `svgwrite.Drawing` to add the line to.
        a: Line start point, in SVG (already-projected) pixel coordinates.
        b: Line end point, in SVG pixel coordinates.
        colour: Stroke colour.
        stroke_width: Stroke width, in SVG units.
        dash: Optional `stroke-dasharray` string (e.g. `"6,4"`) for a dashed
            line -- used for inferred/occluded wall spans; `None` for solid.
    """
    kwargs = {"stroke": colour, "stroke_width": stroke_width, "stroke_linecap": "round"}
    if dash:
        kwargs["stroke_dasharray"] = dash
    drawing.add(drawing.line(a, b, **kwargs))


def _room_label(drawing, room: dict, project, placer: _LabelPlacer) -> None:
    """Room name and area at its centroid.

    Two lines are drawn (name, then a detail line below it) and each is
    independently gated through `placer` -- so it's possible for the name
    to be placed but the detail line skipped (or vice versa) if space is
    tight, rather than an all-or-nothing pair.

    Args:
        drawing: The `svgwrite.Drawing` to add text to.
        room: One room dict from `result["rooms"]`.
        project: The plan-to-SVG-pixel projection function from
            `render_floorplan`.
        placer: Shared label-collision tracker.
    """
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
    ceiling = room.get("ceiling_height")
    height = ceiling.get("value") if isinstance(ceiling, dict) else None
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

    Args:
        drawing: The `svgwrite.Drawing` to add text to.
        wall: One wall dict from `result["reconstruction"]["walls"]`.
        project: The plan-to-SVG-pixel projection function.
        placer: Shared label-collision tracker.
        min_length: Skip labeling walls shorter than this, metres.
        room_centres: `(N, 2)` array of every room's plan centroid (used to
            pick which side of the wall to offset the label toward), or
            `None`/empty if there are no rooms.
    """
    measurement = wall["length"]
    if measurement["value"] < min_length:
        return

    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
    middle = 0.5 * (start + end)
    normal = np.asarray(wall["normal"], dtype=float)
    if room_centres is not None and len(room_centres):
        # Find the room whose centroid is closest to this wall's midpoint --
        # that's almost always the room the wall actually belongs to. If
        # that centroid sits on the POSITIVE side of the wall's normal
        # (i.e. the normal points toward the room's interior), flip the
        # normal so the label offset below goes the other way, into the
        # open exterior/corridor space instead of into the room.
        nearest = room_centres[np.argmin(np.linalg.norm(room_centres - middle, axis=1))]
        if (middle - nearest) @ normal < 0:
            normal = -normal

    anchor = project(middle + normal * 0.16)
    a, b = project(start), project(end)
    # Rotate the label text to run parallel with the wall, but keep it
    # upright/readable: an angle outside (-90, 90) degrees would render the
    # text upside-down, so it's flipped 180 degrees in that case -- the
    # label still runs along the same line, just readable left-to-right.
    angle = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    if angle > 90 or angle < -90:
        angle += 180

    half = measurement.get("half_width")
    label = f"{measurement['value'] * 100:.0f} cm"
    if half:
        label += f" ±{half * 100:.0f} cm"
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
    """Shade each damage region along the wall it was fused onto.

    Only wall-attached regions are drawn (`by_name.get` returns `None`, and
    the region is skipped, for a floor/ceiling `surface_ref` that doesn't
    match any wall's name) -- floor and ceiling damage has no natural
    representation as a line segment on a 2D plan the way a wall region's
    along-wall extent does; those are left to `result.json` and the 3D
    export instead.

    Args:
        drawing: The `svgwrite.Drawing` to add the overlay lines to.
        result: The full result dict; reads `result["damage"]`.
        walls: The plan's wall dicts, used to resolve each region's
            `surface_ref` to a wall's geometry.
        project: The plan-to-SVG-pixel projection function.
    """
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
        # Offset the overlay line slightly off the wall's own centerline
        # (along its normal) so it doesn't visually merge with the wall
        # stroke drawn earlier -- purely a rendering nicety, not a
        # geometric statement about where on the wall's thickness the
        # damage sits.
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
    """Draws the legend box in the corner: color key, summary numbers, and any caveats worth flagging.

    The legend box has a fixed position and size, so it's drawn first at
    the top-left without being checked for overlap with anything else.
    Its individual text rows are still registered with `placer`, though
    (the return value of those calls is deliberately ignored) -- purely
    so that any dimension or room labels drawn afterward get pushed away
    from the legend instead of landing on top of it.

    Args:
        drawing: The `svgwrite.Drawing` to add the legend to.
        height: Full drawing height, in SVG pixels -- footer notes are
            anchored relative to the bottom of the page.
        result: The full result dict; reads `result["damage"]`,
            `result["reconstruction"]["walls"]`, `result["rooms"]`, and
            `result["diagnostics"]["calibration"]`.
        placer: Shared label-collision tracker (see note above).
    """
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
            "wall lengths in cm; room areas in m².  ± is a 90% interval.  * = partly inferred",
            insert=(18, y + 4), fill=PALETTE["text"], font_size="10px",
            font_family=FONT,
        )
    )
    # A placeholder string of the right rough length (not the real caption
    # text) is registered with the placer -- only the reserved SPACE
    # matters here, since this call's return value is never checked.
    placer.place(120, y + 4, "x" * 76, 10)

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
        # This count is only accurate once every placer.place call for the
        # whole drawing has already happened -- render_floorplan calls
        # _legend last, specifically so this reflects the final total.
        footer.append(
            f"{placer.skipped} dimension labels omitted where they would "
            f"overlap — full values in result.json"
        )
    calibration = result.get("diagnostics", {}).get("calibration", {})
    if not calibration.get("calibrated", False):
        # Uncalibrated intervals mean the ± values on dimension labels are
        # rough guesses, not statistically fitted confidence intervals --
        # worth surfacing directly on the drawing since it changes how much
        # to trust those numbers.
        footer.append("intervals uncalibrated (no ground truth fitted)")

    for offset, note in enumerate(reversed(footer)):
        drawing.add(
            drawing.text(
                note, insert=(14, height - 14 - offset * 14), fill="#a04020",
                font_size="10px", font_family=FONT,
            )
        )


def export_scene(
    walls: list[dict],
    path: str | Path,
    floor_height: float,
    ceiling_height: float | None = None,
    rooms: list[dict] | None = None,
) -> Path:
    """Writes the reconstruction as a 3D GLB file, with every wall, floor, and ceiling as its own named surface.

    Rather than exporting one big fused mesh, each wall, floor, and
    ceiling is written as its own small flat rectangle, named after what
    it is (like `room_1.north_wall`). That way, someone opening the file
    in a 3D viewer can click on or isolate a single surface directly,
    instead of hunting for it inside one large blob of triangles. The
    dense point cloud already saved separately as `cloud.ply` is
    deliberately left out here -- including it too would just bury these
    clean, selectable surfaces inside an unsegmented mesh occupying the
    same space.

    Args:
        walls: Wall dicts (same shape as `result["reconstruction"]["walls"]`)
            to export as named quads.
        path: Output `.glb` file path.
        floor_height: World height of the floor plane -- the bottom edge of
            every wall quad, and the height of every room's floor plane.
        ceiling_height: World height of the ceiling plane; if `None`, a
            default 2.4 m (a typical residential ceiling height) above the
            floor is used instead, purely so walls have SOME top edge to
            render to even when no ceiling was fitted.
        rooms: Room dicts (same shape as `result["rooms"]`) to also export
            floor/ceiling planes for, named `"{room}.floor"` /
            `"{room}.ceiling"`.

    Returns:
        The `Path` the GLB scene was written to.
    """
    import trimesh

    scene = trimesh.Scene()
    top = ceiling_height if ceiling_height is not None else floor_height + 2.4
    for wall in walls:
        quad = _wall_quad(wall, floor_height, top)
        if quad is not None:
            scene.add_geometry(quad, node_name=wall["name"])

    for room in rooms or []:
        floor = _room_plane(room, floor_height)
        if floor is not None:
            scene.add_geometry(floor, node_name=f"{room['name']}.floor")
        ceiling = _room_plane(room, top)
        if ceiling is not None:
            scene.add_geometry(ceiling, node_name=f"{room['name']}.ceiling")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))
    return path


def _room_plane(room: dict, height: float):
    """Builds a flat, named floor or ceiling shape for one room, for the 3D export.

    This exists purely so someone can identify and click on a room's
    floor or ceiling in a 3D viewer -- the actual floor-area number used
    everywhere else in the pipeline still comes from `room["area"]`, not
    from this shape. To fill in the room's outline as a solid surface,
    every edge of the outline is connected back to the room's own center
    point, like slices of a pie. That's a simple approach that only
    works correctly because this pipeline's room outlines are simple,
    mostly-rectangular shapes; a room with an unusual, very concave
    outline could need a more careful method to fill in without gaps or
    overlaps.

    Args:
        room: One room dict (same shape as an entry in `result["rooms"]`) --
            reads its `polygon` (plan-space ring) and `centroid`.
        height: World height (along the vertical axis) to place this plane
            at -- `floor_height` for a floor, `top` for a ceiling.

    Returns:
        A `trimesh.Trimesh` triangle fan covering the room's polygon, or
        `None` if the room has no polygon (or a degenerate one, under 3
        points).
    """
    import trimesh

    polygon = room.get("polygon")
    if not polygon or len(polygon) < 3:
        return None
    centre = np.asarray(room["centroid"])
    ring = np.asarray(polygon)
    # One shared center vertex (index 0) plus every outline vertex
    # (indices 1..N); each triangle connects the center to one edge of the
    # outline, like a slice of a pie. This only tiles the shape correctly
    # without gaps because the room outlines this pipeline produces are
    # simple and roughly rectangular.
    vertices = np.vstack([centre[None, :], ring])
    # Plan coordinates are 2D (x, plan-y); lift into 3D by inserting the
    # fixed world height as the middle (vertical) axis -- matching the
    # (x, height, plan-y) convention `_wall_quad` also uses below.
    vertices_3d = np.stack(
        [vertices[:, 0], np.full(len(vertices), height), vertices[:, 1]], axis=1
    )
    count = len(ring)
    faces = np.array(
        [[0, 1 + i, 1 + (i + 1) % count] for i in range(count)]
    )
    return trimesh.Trimesh(vertices=vertices_3d, faces=faces, process=False)


def _wall_quad(wall: dict, floor_height: float, ceiling_height: float):
    """Builds a flat rectangle for one wall, running from the floor up to the ceiling.

    A wall is stored as just a 2D line on the floor plan (a start point
    and an end point); this turns that line into an actual 3D rectangle
    by sweeping it straight up from the floor height to the ceiling
    height, the way a real wall stands.

    Args:
        wall: One wall dict (same shape as an entry in
            `result["reconstruction"]["walls"]`) -- reads its plan-space
            `start`/`end` points.
        floor_height: World height of the quad's bottom edge.
        ceiling_height: World height of the quad's top edge.

    Returns:
        A `trimesh.Trimesh` with 4 vertices (bottom-start, bottom-end,
        top-end, top-start) and 2 triangular faces forming the quad, or
        `None` if the wall's `start`/`end` are coincident (zero length).
    """
    import trimesh

    start, end = np.asarray(wall["start"]), np.asarray(wall["end"])
    if np.linalg.norm(end - start) < 1e-6:
        return None
    # Vertex order goes around the quad: bottom-start -> bottom-end ->
    # top-end -> top-start, so the two triangles (0,1,2) and (0,2,3) share
    # the diagonal from bottom-start to top-end and together cover the
    # whole quad without gaps or overlap.
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
    frames, detections_by_frame, masks_by_frame, out_dir: str | Path,
    rotations: dict | None = None,
) -> list[Path]:
    """Saves one annotated image per frame, with each detected damage mask and box drawn on top, for a human to review.

    These images are meant to be looked at by a person, so they're saved
    at the frame's full native resolution rather than the smaller
    resolution the depth camera captures at. The detections and masks
    themselves were computed at that smaller depth resolution, since
    that's the grid the rest of the pipeline works on, so they're scaled
    up here to match the bigger image before being drawn. If a frame
    doesn't have a full-resolution copy available, the smaller one is
    used instead.

    The optional `rotations` map also reorients each saved image -- and
    the mask/box drawn on it -- to however a person would naturally view
    the scene, the same way frames get rotated before being sent to the
    vision model. Everywhere else in the pipeline, detections stay in the
    camera's original, unrotated orientation; only this human-facing
    image gets turned upright.

    Args:
        frames: Iterable of frame objects (as from `ingest.iter_frames`)
            carrying `.index`, `.color` (depth-resolution RGB), and
            `.color_full` (full native-resolution RGB, or `None`).
        detections_by_frame: Map from frame index to the `Detection`s found
            in it (in depth-resolution coordinates).
        masks_by_frame: Map from frame index to the `RefinedMask`s
            corresponding to those detections, same order and length.
        out_dir: Directory to write per-frame JPEGs into; created if
            missing.
        rotations: Optional map from frame index to a `cv2.ROTATE_*` code
            (or `None`) for human-natural display orientation.

    Returns:
        Paths written, one per frame in `frames`' iteration order, named
        `frame_{index:06d}.jpg`.
    """
    import cv2

    from . import ingest

    rotations = rotations or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for frame in frames:
        rotation = rotations.get(frame.index)
        depth_height, depth_width = frame.color.shape[:2]
        # Prefer the full native-resolution image for the saved render
        # (sharper, more legible to a human) but fall back to the
        # depth-resolution one if a full-res copy wasn't captured for this
        # frame.
        base = frame.color_full if frame.color_full is not None else frame.color
        full_height, full_width = base.shape[:2]
        # Detections/masks were computed in depth-resolution coordinates
        # (that's the grid fusion operates on); these factors scale them up
        # to match whatever resolution is actually being drawn on.
        upscale_x = full_width / depth_width
        upscale_y = full_height / depth_height
        image = cv2.cvtColor(base.copy(), cv2.COLOR_RGB2BGR)  # cv2 drawing needs BGR
        if rotation is not None:
            image = cv2.rotate(image, rotation)
        for detection, mask in zip(
            detections_by_frame.get(frame.index, []),
            masks_by_frame.get(frame.index, []),
        ):
            colour = {
                "water": (176, 108, 43),
                "fire": (33, 86, 192),
                "mold": (74, 133, 47),
            }.get(detection.damage_class, (0, 0, 255))
            mask_arr = mask.mask
            bbox = detection.bbox
            # Order matters: upscale to full resolution FIRST, then rotate --
            # rotating a smaller array and then trying to upscale it against
            # the already-rotated `image` would mismatch axes whenever the
            # rotation is a 90-degree turn (width and height swap).
            if upscale_x != 1.0 or upscale_y != 1.0:
                mask_arr = cv2.resize(
                    mask_arr.astype(np.uint8), (full_width, full_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                bbox = (
                    bbox[0] * upscale_x, bbox[1] * upscale_y,
                    bbox[2] * upscale_x, bbox[3] * upscale_y,
                )
            if rotation is not None:
                mask_arr = cv2.rotate(mask_arr.astype(np.uint8), rotation).astype(bool)
                bbox = ingest.rotate_bbox(bbox, full_width, full_height, rotation)
            # Semi-transparent colour fill over the mask (not the whole box),
            # so the overlay traces the actual segmented shape rather than a
            # solid rectangle.
            overlay = image.copy()
            overlay[mask_arr] = colour
            image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
            x0, y0, x1, y1 = (int(v) for v in bbox)
            cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
            label = detection.damage_class
            if detection.subtype:
                label = f"{label}:{detection.subtype}"
            cv2.putText(
                image,
                f"{label} {detection.confidence:.2f}",
                (x0, max(y0 - 6, 12)),  # clamp so the label never renders off the top edge
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2,
            )
        target = out_dir / f"frame_{frame.index:06d}.jpg"
        cv2.imwrite(str(target), image)
        written.append(target)
    return written


def export_scope_csv(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Writes the floor plan and the scope of work as two CSV files, in the layout that restoration-estimating software expects to import.

    This -- not the JSON result -- is the file an estimator's workflow
    actually reads in. Two separate files are written instead of one,
    because they describe two different kinds of rows: the sketch file
    has one row per wall (its geometry, whether or not it has any
    damage), while the scope file has one row per repair line item
    (which only exist where damage was actually found). Joining those
    into a single table would mean either repeating every wall's geometry
    on each of its line items, or leaving geometry blank on most rows --
    both worse to import than two focused tables linked by room name.

    Args:
        result: The full pipeline result dict; reads `result["rooms"]`,
            `result["reconstruction"]["walls"]`, and
            `result["scope"]["line_items"]`.
        out_dir: Directory to write both CSVs into; created if missing.

    Returns:
        `(sketch_path, scope_path)`: paths to `scope_sketch.csv` (one row
        per wall, with its room's area/ceiling height repeated on each of
        that room's wall rows) and `scope_line_items.csv` (one row per
        scope-of-work line item).
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rooms_by_id = {room["id"]: room for room in result["rooms"]}

    sketch_path = out_dir / "scope_sketch.csv"
    with sketch_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "room", "room_area_m2", "ceiling_height_m", "wall",
                "wall_length_m", "wall_length_ci_half_width_m",
            ]
        )
        for wall in result["reconstruction"]["walls"]:
            room = rooms_by_id.get(wall["room_id"])
            writer.writerow(
                [
                    room["name"] if room else "",
                    f"{room['area']['value']:.3f}" if room else "",
                    f"{room['ceiling_height']['value']:.3f}" if room else "",
                    wall["name"],
                    f"{wall['length']['value']:.3f}",
                    f"{wall['length']['half_width']:.3f}",
                ]
            )

    scope_path = out_dir / "scope_line_items.csv"
    with scope_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "room", "surface_ref", "action", "material", "description",
                "quantity", "unit", "trade", "rule_id", "source", "basis",
            ]
        )
        for item in result["scope"]["line_items"]:
            room = rooms_by_id.get(item["room_id"])
            writer.writerow(
                [
                    room["name"] if room else "",
                    item["surface_ref"],
                    item["action"],
                    item["material"],
                    item["description"],
                    item["quantity"],
                    item["unit"],
                    item["trade"],
                    item["rule_id"],
                    item["source"],
                    item["basis"],
                ]
            )

    return sketch_path, scope_path
