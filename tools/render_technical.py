"""Render the Markdown technical report to a compact, paginated PDF.

This is report tooling only; it does not import or execute the application
pipeline. Document dependencies are intentionally supplied by the caller via
``uv run --with`` so they do not become application dependencies.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _inline(node: object) -> str:
    if isinstance(node, NavigableString):
        return escape(str(node))
    if not isinstance(node, Tag):
        return ""
    inner = "".join(_inline(child) for child in node.children)
    if node.name in ("strong", "b"):
        return f"<b>{inner}</b>"
    if node.name in ("em", "i"):
        return f"<i>{inner}</i>"
    if node.name == "code":
        return f'<font name="Courier" size="6.5">{inner}</font>'
    if node.name == "a":
        href = escape(node.get("href", ""), quote=True)
        return f'<a href="{href}" color="#0645ad">{inner}</a>'
    if node.name == "br":
        return "<br/>"
    return inner


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=20,
            spaceAfter=5,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.1,
            leading=13,
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
            textColor=colors.HexColor("#183b56"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=9.0,
            spaceAfter=2.2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.9,
            leading=8.0,
            spaceAfter=1.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportTable",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=6.35,
            leading=7.15,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportTableHead",
            parent=styles["ReportTable"],
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#102a43"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportQuote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.4,
            leading=8.7,
            leftIndent=8,
            borderPadding=3,
            borderColor=colors.HexColor("#9fb3c8"),
            borderWidth=1,
            borderLeft=True,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=8.7,
            leftIndent=10,
            firstLineIndent=-7,
            spaceAfter=1.2,
        )
    )
    return styles


def _story(source: Path):
    soup = BeautifulSoup(
        markdown.markdown(source.read_text(), extensions=["tables", "fenced_code"]),
        "html.parser",
    )
    styles = _styles()
    story = []
    for node in soup.contents:
        if isinstance(node, NavigableString) or not isinstance(node, Tag):
            continue
        if node.name == "h1":
            story.append(Paragraph(_inline(node), styles["ReportTitle"]))
        elif node.name == "h2":
            story.append(Paragraph(_inline(node), styles["ReportH2"]))
        elif node.name == "p":
            story.append(Paragraph(_inline(node), styles["ReportBody"]))
        elif node.name == "blockquote":
            for child in node.find_all("p", recursive=False):
                story.append(Paragraph(_inline(child), styles["ReportQuote"]))
        elif node.name in ("ul", "ol"):
            for index, item in enumerate(node.find_all("li", recursive=False), 1):
                marker = "•" if node.name == "ul" else f"{index}."
                story.append(
                    Paragraph(marker + " " + _inline(item), styles["ReportBullet"])
                )
        elif node.name == "pre":
            story.extend(
                [
                    Preformatted(
                        node.get_text(), styles["ReportSmall"], maxLineLength=110
                    ),
                    Spacer(1, 1.5),
                ]
            )
        elif node.name == "table":
            rows = []
            for row in node.find_all("tr"):
                cells = [
                    Paragraph(
                        _inline(cell),
                        styles[
                            "ReportTableHead" if cell.name == "th" else "ReportTable"
                        ],
                    )
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                if cells:
                    rows.append(cells)
            if rows:
                table = Table(
                    rows,
                    colWidths=[None] * max(len(row) for row in rows),
                    repeatRows=1,
                    hAlign="LEFT",
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef5")),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#829ab1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.extend([table, Spacer(1, 2)])
    return story


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(colors.HexColor("#52606d"))
    canvas.drawString(12 * mm, 7 * mm, "Cozmo AI v2 · evidence cutoff 2026-09-01")
    canvas.drawRightString(A4[0] - 12 * mm, 7 * mm, f"page {doc.page}")
    canvas.restoreState()


def render(source: Path, output: Path, max_pages: int) -> int:
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title="Capture-to-scope: final technical report",
        author="Cozmo AI v2",
    )
    doc.build(_story(source), onFirstPage=_footer, onLaterPages=_footer)
    pages = len(PdfReader(str(output)).pages)
    print(f"wrote {output} ({pages} pages)")
    if pages > max_pages:
        raise SystemExit(f"page limit exceeded: {pages} > {max_pages}")
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("report/technical.md"))
    parser.add_argument("output", nargs="?", type=Path, default=Path("out/technical.pdf"))
    parser.add_argument("--max-pages", type=int, default=6)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.source, args.output, args.max_pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
