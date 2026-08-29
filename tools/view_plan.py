from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOURS = {
    "door": "#2f6f4e",
    "window": "#2b6cb0",
    "pass-through": "#2f6f4e",
    "occluded": "#9aa5b1",
    "water": "#2b6cb0",
    "fire": "#c05621",
    "mold": "#2f855a",
}


def render(result: dict, out_path: Path, min_label_length: float = 1.2) -> Path:
    figure, axes = plt.subplots(figsize=(15, 15))

    for room in result.get("rooms", []):
        polygon = np.array(room.get("polygon") or [])
        if len(polygon) > 2:
            axes.fill(polygon[:, 0], polygon[:, 1], alpha=0.28, zorder=0)
            axes.text(
                *room["centroid"],
                f"{room['name']}\n{room['area']['value']:.1f} m²",
                ha="center", va="center", fontsize=9, weight="bold", zorder=6,
            )

    openings: dict[str, list[dict]] = {}
    for opening in result["reconstruction"].get("openings", []):
        openings.setdefault(opening["wall"], []).append(opening)

    for wall in result["reconstruction"]["walls"]:
        start, end = np.array(wall["start"]), np.array(wall["end"])
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        direction = (end - start) / length

        cuts = [
            (o["u_offset"], o["u_offset"] + o["width"]["value"], o["kind"])
            for o in openings.get(wall["name"], [])
        ]
        cuts += [(s[0], s[1], "occluded") for s in wall.get("occluded_spans", [])]
        cuts.sort()

        cursor = 0.0
        for u0, u1, kind in cuts:
            u0, u1 = max(0.0, u0), min(length, u1)
            if u1 <= cursor:
                continue
            if u0 > cursor:
                _segment(axes, start + direction * cursor, start + direction * u0,
                         "#1f2933", 3, zorder=3)
            _segment(axes, start + direction * u0, start + direction * u1,
                     COLOURS.get(kind, "#9aa5b1"), 3.5, zorder=4,
                     style="--" if kind == "occluded" else "-")
            cursor = u1
        if cursor < length:
            _segment(axes, start + direction * cursor, end, "#1f2933", 3, zorder=3)

        measurement = wall["length"]
        if measurement["value"] >= min_label_length:
            middle = (start + end) / 2
            angle = np.degrees(np.arctan2(*(end - start)[::-1]))
            if angle > 90 or angle < -90:
                angle += 180
            axes.text(
                *middle,
                f"{measurement['value'] * 100:.0f} cm ±{measurement['half_width'] * 100:.0f} cm",
                fontsize=6.5, ha="center", color="#3e4c59", rotation=angle, zorder=7,
            )

    walls_by_name = {w["name"]: w for w in result["reconstruction"]["walls"]}
    for region in result.get("damage", []):
        wall = walls_by_name.get(region["surface_ref"])
        if wall is None:
            continue
        start, end = np.array(wall["start"]), np.array(wall["end"])
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        direction = (end - start) / length
        shift = np.array(wall["normal"]) * 0.1
        u0, u1 = region["extent"]["u_range"]
        _segment(
            axes,
            start + direction * u0 + shift,
            start + direction * min(u1, length) + shift,
            COLOURS.get(region["damage_class"], "#000000"), 5, zorder=8,
        )

    axes.set_aspect("equal")
    axes.grid(alpha=0.15)
    reconstruction = result["reconstruction"]
    axes.set_title(
        f"{result['capture']['name']} — {len(result.get('rooms', []))} rooms, "
        f"{len(reconstruction['walls'])} walls, "
        f"{len(reconstruction.get('openings', []))} openings, "
        f"{len(result.get('damage', []))} damage regions\n"
        f"green=door  blue=window  dashed=inferred/occluded"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=85)
    return out_path


def _segment(axes, a, b, colour, width, zorder=1, style="-") -> None:
    axes.plot(
        [a[0], b[0]], [a[1], b[1]], style, color=colour, lw=width,
        zorder=zorder, solid_capstyle="butt",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a result.json floor plan to PNG for quick inspection."
    )
    parser.add_argument("result")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    result_path = Path(args.result)
    out_path = Path(args.out) if args.out else result_path.parent / "plan_view.png"
    render(json.loads(result_path.read_text()), out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
