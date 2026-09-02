"""Render the one-page Stray Scanner capture protocol.

The Markdown file is the editable operator source. This script keeps the
print artifact deterministic and uses only reportlab, which is intentionally
not a project runtime dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


PAGE_W, PAGE_H = letter
MARGIN_X = 24
GUTTER = 16
CONTENT_W = PAGE_W - (2 * MARGIN_X)
COL_W = (CONTENT_W - GUTTER) / 2
BOTTOM_Y = 48

NAVY = colors.HexColor("#0B2239")
BLUE = colors.HexColor("#155E75")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#52616B")
PAPER = colors.HexColor("#F7FAFC")
PALE_BLUE = colors.HexColor("#E8F3F7")
PALE_RED = colors.HexColor("#FFF0EE")
RED = colors.HexColor("#B42318")
RULE = colors.HexColor("#D6E0E6")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    return {
        "body": ParagraphStyle(
            "protocol-body", parent=base, fontName="Helvetica", fontSize=7.35,
            leading=8.85, textColor=INK, alignment=TA_LEFT, spaceAfter=0,
        ),
        "body_tight": ParagraphStyle(
            "protocol-body-tight", parent=base, fontName="Helvetica", fontSize=7.0,
            leading=8.35, textColor=INK, alignment=TA_LEFT, spaceAfter=0,
        ),
        "heading": ParagraphStyle(
            "protocol-heading", parent=base, fontName="Helvetica-Bold", fontSize=8.0,
            leading=9.3, textColor=NAVY, alignment=TA_LEFT, spaceAfter=0,
        ),
        "note": ParagraphStyle(
            "protocol-note", parent=base, fontName="Helvetica", fontSize=5.15,
            leading=6.15, textColor=MUTED, alignment=TA_LEFT, spaceAfter=0,
        ),
        "callout": ParagraphStyle(
            "protocol-callout", parent=base, fontName="Helvetica", fontSize=6.55,
            leading=7.9, textColor=INK, alignment=TA_LEFT, spaceAfter=0,
        ),
    }


def draw_paragraph(canvas: Canvas, text: str, x: float, y_top: float, width: float,
                   style: ParagraphStyle) -> float:
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, PAGE_H)
    paragraph.drawOn(canvas, x, y_top - height)
    return height


def draw_section(canvas: Canvas, x: float, y_top: float, width: float, title: str,
                 body: str, style: ParagraphStyle, fill: colors.Color = colors.white) -> float:
    body_paragraph = Paragraph(body, style)
    _, body_h = body_paragraph.wrap(width - 14, PAGE_H)
    title_paragraph = Paragraph(title, styles()["heading"])
    _, title_h = title_paragraph.wrap(width - 14, PAGE_H)
    total_h = title_h + body_h + 15

    canvas.setFillColor(fill)
    canvas.roundRect(x, y_top - total_h, width, total_h, 4, fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.roundRect(x, y_top - total_h, 3, total_h, 1.5, fill=1, stroke=0)
    title_paragraph.drawOn(canvas, x + 9, y_top - title_h - 5)
    body_paragraph.drawOn(canvas, x + 9, y_top - title_h - body_h - 11)
    return total_h + 5


def check(value: str) -> str:
    return f"<font color='#155E75'>[ ]</font> {value}"


def draw_header(canvas: Canvas) -> float:
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 78, PAGE_W, 78, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 15.3)
    canvas.drawString(MARGIN_X, PAGE_H - 27, "STRAY SCANNER -> COZMO-AI-V2")
    canvas.setFont("Helvetica-Bold", 9.1)
    canvas.drawString(MARGIN_X, PAGE_H - 42, "ONE-PAGE CAPTURE PROTOCOL")
    canvas.setFont("Helvetica", 6.3)
    canvas.drawString(MARGIN_X, PAGE_H - 56, "Print it. Follow every step. [OFFICIAL] = app fact; [PROJECT] = capture rule.")
    canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 56, "Checked 2026-09-02")

    canvas.setFillColor(PALE_RED)
    canvas.roundRect(MARGIN_X, PAGE_H - 106, CONTENT_W, 24, 4, fill=1, stroke=0)
    canvas.setFillColor(RED)
    canvas.setFont("Helvetica-Bold", 6.8)
    canvas.drawString(MARGIN_X + 8, PAGE_H - 91, "STOP RULE")
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 6.3)
    canvas.drawString(MARGIN_X + 67, PAGE_H - 91, "If an avoid/restart condition occurs, stop, mark the folder BAD, fix the room, and start a new capture at the marker.")
    canvas.drawString(MARGIN_X + 67, PAGE_H - 99, "Never resume a failed file or join captures: one pipeline run is one continuous coordinate frame.")
    return PAGE_H - 116


def render(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=letter, pageCompression=1)
    canvas.setTitle("Stray Scanner to cozmo-ai-v2 Capture Protocol")
    canvas.setAuthor("cozmo-ai-v2")
    style = styles()
    y_start = draw_header(canvas)

    left_x = MARGIN_X
    right_x = MARGIN_X + COL_W + GUTTER
    left_y = y_start
    right_y = y_start

    left_sections = [
        (
            "1 | INSTALL + DEVICE",
            "<b>[OFFICIAL]</b> Install <b>Stray Scanner</b> (Kenneth Blomqvist, free) from the App Store; grant Camera. Current listing: iOS 18.6+.<br/>"
            "<b>[OFFICIAL]</b> Use LiDAR: iPhone 12 Pro/Pro Max or later Pro, or iPad Pro 11-inch 2nd gen+ / 12.9-inch 4th gen+. No non-Pro substitute. If iPad install is refused, use an eligible iPhone.<br/>"
            "<b>[PROJECT]</b> Leave 10 GB free; charge to 80% or connect power; close apps; Do Not Disturb; Auto-Lock Never; silence calls/notifications.",
            style["body"], PALE_BLUE,
        ),
        (
            "2 | PREPARE THE ROOM",
            "<b>[PROJECT]</b> Turn on even lighting; avoid direct sun, flicker, glare, and dark corners. Open curtains/blinds so jambs show. Cover or flag mirrors/reflective glass; never rely on them as the only wall evidence.<br/>"
            "Open and wedge every door. Remove people/pets. Clear small clutter from wall/floor views; do not move furniture, doors, or curtains after Record.<br/>"
            "Tape a straight, non-reflective measured marker (prefer 2.00 m); write its exact length. Keep it fixed and show both endpoints at start and finish. It validates scale later; it does not auto-calibrate.",
            style["body"], colors.white,
        ),
        (
            "3 | APP SETTINGS + PREFLIGHT",
            "Open <b>Record new session</b>. Tap the fps control until it reads <b>30 fps</b>. <b>[PROJECT]</b> 30 fps is recommended; the app also offers 60/15/5/1. Keep portrait.<br/>"
            "Make a <b>60-frame test</b>: Record, hold still until 60 RGB frames (about 2 s at 30 fps), Stop, open the test, Share -> Save to Files. Confirm it opens and contains <font name='Courier'>rgb.mp4</font>, <font name='Courier'>depth/</font>, <font name='Courier'>confidence/</font>, <font name='Courier'>odometry.csv</font>, <font name='Courier'>imu.csv</font>, <font name='Courier'>camera_matrix.csv</font>. Delete test only after pass.<br/>"
            "<b>[OFFICIAL]</b> Contract: depth PNG = 16-bit millimetres; confidence PNG = 0/1/2; matching six-digit stems index RGB frames; odometry/IMU timestamps seconds, odometry positions metres, IMU accel m/s2 and angular rate rad/s, camera_matrix 3x3 pixel intrinsics.<br/>"
            "<b>[PROJECT]</b> Nonblank odometry timestamp rows = depth PNG count = confidence PNG count; decoded RGB count should match. Any mismatch, unreadable PNG, or terminal video frame loss = FAIL and recapture.",
            style["body_tight"], PALE_BLUE,
        ),
        (
            "4 | HOLD THE DEVICE",
            "<b>[PROJECT]</b> Portrait; screen toward you; rear camera/LiDAR toward scene; top edge up; no roll. Keep the lens about <b>1.4 m above floor</b> (within about 10 cm) for the run.<br/>"
            "Keep level for wall sweeps. Only at floor/ceiling pauses pitch about 15 deg down/up, then return to level. Never turn sideways/upside down.<br/>"
            "Keep surfaces about <b>1-3 m</b> away where possible; never intentionally rely on &gt; <b>3.5 m</b>. Walk closer in large rooms. Keep both windows uncovered.",
            style["body"], colors.white,
        ),
        (
            "5 | EXACT ROUTE",
            "<b>[PROJECT]</b> Use this route exactly.<br/>"
            "<b>1.</b> Start at marker. Tap Record. Hold full marker motionless <b>3 s</b>.<br/>"
            "<b>2.</b> Walk clockwise, wall on left. Advance no more than <b>0.5 m every 2 s</b>; pause <b>2 s at every corner</b>. Keep knee-to-head wall band visible.<br/>"
            "<b>3.</b> Doorway: approach level; hold <b>2 s</b> with both jambs, header, threshold visible; cross without height/orientation change; hold <b>2 s just inside</b>; sweep room clockwise.<br/>"
            "<b>4.</b> Each room: every wall/corner; centre holds <b>2 s level + 2 s down for floor strip + 2 s up for ceiling edge</b>; each window <b>2 s face-on</b> with jambs, sill, header.<br/>"
            "<b>5.</b> Return through doors to within <b>2 m</b> of marker after at least <b>4 s</b>. Hold marker <b>3 s</b>, then Stop. <b>[PROJECT]</b> This revisit enables loop closure.",
            style["body_tight"], colors.white,
        ),
    ]

    right_sections = [
        (
            "6 | DURATION + SPLIT",
            "<b>[PROJECT]</b> Target <b>30-120 s</b> per connected floor/area. Hard operator cap: <b>5 min</b> per continuous capture. This is a project safety rule, not an App Store limit; it limits drift, interruption, and terminal loss.<br/>"
            "If it will exceed 5 min, cross floors, or enter a disconnected area, stop at a marker and start a new capture with its own marker/folder. <b>Never join folders.</b>",
            style["body"], PALE_BLUE,
        ),
        (
            "7 | AVOID / RESTART",
            "<b>[PROJECT]</b> Stop and restart from marker if: portrait/roll changes; whip turn or blur; lens/LiDAR covered; call, lock, app interruption, or Record stops; AR view jumps/loses tracking; room missed; surface &gt;3.5 m; door/furniture/curtain moves; people/pets enter; or mirror/glass is only evidence.<br/>"
            "Capture solid adjacent surfaces and both jambs; flag unavoidable reflective spans for manual measurement. Do not resume a failed file.",
            style["body"], colors.white,
        ),
        (
            "8 | STOP + EXPORT + HANDOFF",
            "<b>[OFFICIAL]</b> After Stop, recording detail -> <b>Share</b> -> Save to Files or AirDrop; keep app ZIP unchanged. Original-folder route: cable -> Finder -> device -> <b>Files</b> -> <b>Stray Scanner</b> -> drag dataset folder (Windows: iTunes). Files route: Browse -> On My iPhone/iPad -> Stray Scanner -> Share/Save.<br/>"
            "<b>[PROJECT]</b> Name only the delivered top folder <font name='Courier'>site_floor_area_YYYYMMDD-HHMM</font>; retain app hash in note. Do not rename children, rotate/transcode video, resize PNGs, reorder files, or rearrange folders. <font name='Courier'>distortion/</font> is optional; all other listed items are required.<br/>"
            "Before transfer, recheck required names and equal counts. ZIP: <font name='Courier'>unzip -t capture.zip</font>, then <font name='Courier'>shasum -a 256 capture.zip</font>; send original ZIP + checksum. Do not unzip/repack unless asked.",
            style["body_tight"], colors.white,
        ),
        (
            "9 | CURRENT-MAIN PROCESS",
            "From repo root, after <font name='Courier'>uv sync</font> (verified 2026-09-02):<br/>"
            "<font name='Courier' size='6.2'>uv run python -m cozmo_ai_v2.pipeline run \"/path/to/CAPTURE\" \\</font><br/>"
            "<font name='Courier' size='6.2'>  --out \"out/CAPTURE\" --no-damage</font><br/>"
            "Expect <font name='Courier'>result.json</font>, <font name='Courier'>floorplan.svg</font>, <font name='Courier'>scene.glb</font>, <font name='Courier'>cloud.ply</font>, <font name='Courier'>mesh.ply</font>, <font name='Courier'>planes.json</font>, <font name='Courier'>fusion_manifest.json</font>, and scope/openings CSVs.<br/>"
            "Recapture for terminal sidecar unavailable, requested frames rejected, low IMU gravity consistency, floor not observed/low confidence, no ceiling, no rooms, or room-overlap warnings. Uncalibrated intervals and reference-not-applied are review/scale notes. Defaults: confidence &gt;=1, depth &lt;=3.5 m; invalid depth is discarded.",
            style["body_tight"], PALE_BLUE,
        ),
        (
            "10 | FINAL 20-SECOND PASS/FAIL",
            check("LiDAR device; 10 GB free; 80%/power; no interruptions") + "<br/>"
            + check("30 fps; portrait/no roll; lens 1.4 m high; 1-3 m surfaces") + "<br/>"
            + check("Measured marker at start/finish; all rooms, doors, corners, windows") + "<br/>"
            + check("Floor strip + ceiling edge; no people/pets/moving objects; revisit") + "<br/>"
            + check("Original folder/ZIP; names/counts match; checksum recorded") + "<br/>"
            + check("Pipeline output has no recapture warning"),
            style["body_tight"], colors.white,
        ),
    ]

    for title, body, body_style, fill in left_sections:
        left_y -= draw_section(canvas, left_x, left_y, COL_W, title, body, body_style, fill)
    for title, body, body_style, fill in right_sections:
        right_y -= draw_section(canvas, right_x, right_y, COL_W, title, body, body_style, fill)

    if min(left_y, right_y) < BOTTOM_Y + 25:
        raise RuntimeError(f"protocol overflow: left={left_y:.1f}, right={right_y:.1f}")

    # Evidence note is deliberately outside the operator cards and in a small,
    # readable footer so it never competes with the literal steps.
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, BOTTOM_Y + 12, PAGE_W - MARGIN_X, BOTTOM_Y + 12)
    note = (
        "SOURCES (checked 2026-09-02): Official [App Store](https://apps.apple.com/us/app/stray-scanner/id1557051662), "
        "[format](https://github.com/strayrobots/scanner/blob/main/docs/format.md), "
        "[export](https://github.com/strayrobots/scanner/blob/main/docs/export.md), "
        "[record source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Controllers/RecordSessionViewController.swift), "
        "[ZIP source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/ShareUtility.swift), "
        "[portrait/capability source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Info.plist), "
        "[Apple LiDAR models](https://support.apple.com/en-sg/121825). Project contracts: "
        "[ingest](../src/cozmo_ai_v2/pipeline/ingest.py), [frame contract](../src/cozmo_ai_v2/pipeline/frame_contract.py), "
        "[poses](../src/cozmo_ai_v2/pipeline/poses.py), [pipeline CLI](../src/cozmo_ai_v2/pipeline/cli.py), "
        "[fixture](../tests/conftest.py)."
    )
    # Reportlab links are HTML-ish; convert the Markdown links to compact blue labels.
    replacements = {
        "[App Store](https://apps.apple.com/us/app/stray-scanner/id1557051662)": "<link href='https://apps.apple.com/us/app/stray-scanner/id1557051662' color='#155E75'>App Store</link>",
        "[format](https://github.com/strayrobots/scanner/blob/main/docs/format.md)": "<link href='https://github.com/strayrobots/scanner/blob/main/docs/format.md' color='#155E75'>format</link>",
        "[export](https://github.com/strayrobots/scanner/blob/main/docs/export.md)": "<link href='https://github.com/strayrobots/scanner/blob/main/docs/export.md' color='#155E75'>export</link>",
        "[record source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Controllers/RecordSessionViewController.swift)": "<link href='https://github.com/strayrobots/scanner/blob/main/StrayScanner/Controllers/RecordSessionViewController.swift' color='#155E75'>record source</link>",
        "[ZIP source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/ShareUtility.swift)": "<link href='https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/ShareUtility.swift' color='#155E75'>ZIP source</link>",
        "[portrait/capability source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Info.plist)": "<link href='https://github.com/strayrobots/scanner/blob/main/StrayScanner/Info.plist' color='#155E75'>portrait/capability source</link>",
        "[Apple LiDAR models](https://support.apple.com/en-sg/121825)": "<link href='https://support.apple.com/en-sg/121825' color='#155E75'>Apple LiDAR models</link>",
        "[ingest](../src/cozmo_ai_v2/pipeline/ingest.py)": "ingest.py",
        "[frame contract](../src/cozmo_ai_v2/pipeline/frame_contract.py)": "frame_contract.py",
        "[poses](../src/cozmo_ai_v2/pipeline/poses.py)": "poses.py",
        "[pipeline CLI](../src/cozmo_ai_v2/pipeline/cli.py)": "pipeline cli.py",
        "[fixture](../tests/conftest.py)": "tests/conftest.py",
    }
    for old, new in replacements.items():
        note = note.replace(old, new)
    draw_paragraph(canvas, note, MARGIN_X, BOTTOM_Y + 7, CONTENT_W, style["note"])

    canvas.showPage()
    canvas.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/stray-scanner-capture-protocol.pdf"),
        help="PDF output path",
    )
    args = parser.parse_args()
    render(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
