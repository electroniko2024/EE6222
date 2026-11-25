"""Build the final English submission PDF without headers or footers."""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "report_revised_en.md"
FINAL_MD = ROOT / "report_final_en.md"
OUTPUT = ROOT / "output/pdf/MENGXIANGYU-G2607496E-final.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT = RIGHT = 50
TOP = BOTTOM = 48
CONTENT_WIDTH = PAGE_WIDTH - LEFT - RIGHT

styles = {
    "body": ParagraphStyle(
        "final_body", fontName="Times-Roman", fontSize=10.7,
        leading=14.6, spaceAfter=8,
    ),
    "title": ParagraphStyle(
        "final_title", fontName="Helvetica-Bold", fontSize=19,
        leading=24, spaceAfter=12, textColor=colors.HexColor("#163c60"),
    ),
    "heading": ParagraphStyle(
        "final_heading", fontName="Helvetica-Bold", fontSize=12,
        leading=16, spaceBefore=7, spaceAfter=9,
        textColor=colors.HexColor("#163c60"),
    ),
    "small": ParagraphStyle(
        "final_small", fontName="Times-Roman", fontSize=9,
        leading=11.5, spaceAfter=7,
    ),
    "equation": ParagraphStyle(
        "final_equation", fontName="Courier", fontSize=9.2,
        leading=14, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8,
    ),
    "code": ParagraphStyle(
        "final_code", fontName="Courier", fontSize=7.9,
        leading=10.6,
    ),
}


def final_markdown(source: str) -> str:
    """Remove the limitations and reproducibility sections from the revised report."""
    limitation_heading = "## 6. Limitations and conclusions"
    reproducibility_heading = "## 7. Reproducibility"
    references_heading = "## References"

    before = source[:source.index(limitation_heading)].rstrip()
    # The final submission title page intentionally contains no identity/date line.
    before_lines = before.splitlines()
    if len(before_lines) >= 4 and before_lines[2].startswith("Assignment 1"):
        before = "\n".join(before_lines[:2] + before_lines[4:]).rstrip()
    limitation_section = source[
        source.index(limitation_heading) + len(limitation_heading):
        source.index(reproducibility_heading)
    ].strip()
    paragraphs = [part.strip() for part in limitation_section.split("\n\n")
                  if part.strip() and part.strip() != "---"]
    conclusion = paragraphs[-1]
    references = source[source.index(references_heading):].strip()
    return "\n\n".join([
        before,
        "## 6. Conclusions",
        conclusion,
        references,
    ]) + "\n"


def inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#1466a5">\1</link>',
        escaped,
    )
    return escaped


def render_table(lines, story):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return
    columns = len(rows[0])
    if columns == 3:
        widths = [105, 108, CONTENT_WIDTH - 213]
    elif columns == 7:
        widths = [102, 67, 29, 74, 73, 73, CONTENT_WIDTH - 418]
    else:
        widths = [CONTENT_WIDTH / columns] * columns
    data = [[Paragraph(inline(cell), styles["small"]) for cell in row]
            for row in rows]
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#60829e")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#60829e")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f8fa")]),
    ]))
    story.extend([table, Spacer(1, 9)])


def render_markdown(markdown: str):
    story = []
    lines = markdown.splitlines()
    index = 0
    equation_prefixes = ("S =", "z =", "S_W =", "S_B =", "S_B w =", "min_(")
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "---":
            if story:
                story.append(PageBreak())
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            render_table(block, story)
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                path = ROOT / match.group(2)
                width, height = ImageReader(str(path)).getSize()
                image_height = CONTENT_WIDTH * height / width
                story.append(Image(str(path), width=CONTENT_WIDTH,
                                   height=image_height))
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["heading"]))
            index += 1
            continue
        if line.startswith(equation_prefixes):
            story.append(Preformatted(line, styles["equation"]))
            index += 1
            continue

        block = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (not next_line or next_line == "---" or next_line.startswith("#")
                    or next_line.startswith("|") or next_line.startswith("![")
                    or next_line.startswith(equation_prefixes)):
                break
            block.append(next_line)
            index += 1
        story.append(Paragraph(inline(" ".join(block)), styles["body"]))
    return story


def append_code_appendix(story):
    source_lines = (ROOT / "experiment.py").read_text(encoding="utf-8").splitlines()
    if len(source_lines) != 221:
        raise ValueError(f"Expected 221 source lines, found {len(source_lines)}")
    for start in range(0, len(source_lines), 45):
        story.append(PageBreak())
        title = "Appendix A. Complete Python experiment"
        if start:
            title += " (continued)"
        story.append(Paragraph(title, styles["heading"]))
        story.append(Paragraph(
            f"experiment.py | Lines {start + 1}-{min(start + 45, len(source_lines))}",
            styles["small"],
        ))
        numbered = "\n".join(
            f"{line_number:03d}  {line}"
            for line_number, line in enumerate(
                source_lines[start:start + 45], start=start + 1
            )
        )
        story.append(Preformatted(numbered, styles["code"]))


def main():
    revised = SOURCE.read_text(encoding="utf-8")
    final = final_markdown(revised)
    FINAL_MD.write_text(final, encoding="utf-8")
    story = render_markdown(final)
    append_code_appendix(story)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title="Dimensionality Reduction for Classification",
        author="MENG XIANGYU (G2607496E)",
    )
    # No onFirstPage/onLaterPages callbacks: the final PDF has no headers or footers.
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
