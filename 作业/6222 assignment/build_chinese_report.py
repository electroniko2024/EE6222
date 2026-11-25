"""Render report_zh.md, the sole Chinese body source, to a readable PDF."""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, PageBreak,
    Spacer, Image, Table, TableStyle, Preformatted)

ROOT = Path(__file__).resolve().parent
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Chinese", str(FONT_DIR / "msyh.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("ChineseBold", str(FONT_DIR / "msyhbd.ttc"), subfontIndex=0))
pdfmetrics.registerFontFamily("Chinese", normal="Chinese", bold="ChineseBold",
    italic="Chinese", boldItalic="ChineseBold")
W = A4[0] - 96
styles = {
    "body": ParagraphStyle("zh_body", fontName="Chinese", fontSize=10.4,
        leading=16.2, wordWrap="CJK", spaceAfter=8),
    "title": ParagraphStyle("zh_title", fontName="ChineseBold", fontSize=19,
        leading=27, wordWrap="CJK", spaceAfter=13,
        textColor=colors.HexColor("#163c60")),
    "h": ParagraphStyle("zh_h", fontName="ChineseBold", fontSize=12.5,
        leading=19, wordWrap="CJK", spaceBefore=6, spaceAfter=8,
        textColor=colors.HexColor("#163c60")),
    "sub": ParagraphStyle("zh_sub", fontName="ChineseBold", fontSize=10.8,
        leading=16.8, wordWrap="CJK", spaceBefore=4, spaceAfter=6),
    "small": ParagraphStyle("zh_small", fontName="Chinese", fontSize=8.7,
        leading=13.2, wordWrap="CJK", spaceAfter=5),
    "code": ParagraphStyle("zh_code", fontName="Courier", fontSize=9.4,
        leading=14.5, spaceAfter=8),
}


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  r'<link href="\2" color="#1466a5">\1</link>', text)
    return text


def table(lines):
    rows = [[x.strip() for x in line.strip().strip("|").split("|")]
            for line in lines if not re.match(r"^\|[\s:|-]+\|$", line)]
    n = len(rows[0])
    widths = ([88, 58, 33, 69, 89, 89, W-426] if n == 7 else
              [95, 110, W-205] if n == 3 else [155, W-155])
    data = [[Paragraph(inline(cell), styles["small"]) for cell in row] for row in rows]
    result = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eff5")),
        ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#60829e")),
        ("LINEBELOW", (0,-1), (-1,-1), .5, colors.HexColor("#60829e")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return result


story = []
pages = (ROOT / "report_zh.md").read_text(encoding="utf-8").split("\n---\n")
for number, page in enumerate(pages):
    if number:
        story.append(PageBreak())
    blocks = re.split(r"\n\s*\n", page.strip())
    for block in blocks:
        if block.startswith("# "):
            story.append(Paragraph(inline(block[2:]), styles["title"]))
        elif block.startswith("## "):
            story.append(Paragraph(inline(block[3:]), styles["h"]))
        elif block.startswith("### "):
            story.append(Paragraph(inline(block[4:]), styles["sub"]))
        elif block.startswith("```"):
            story.append(Preformatted("\n".join(block.splitlines()[1:-1]), styles["code"]))
        elif block.startswith("|"):
            story.extend([table(block.splitlines()), Spacer(1, 7)])
        elif block.startswith("!["):
            path = ROOT / re.search(r"\]\(([^)]+)\)", block).group(1)
            ratio = 3.8/10.4 if "accuracy" in path.name else 2.9/6.6
            image_width = W if "accuracy" in path.name else W*0.78
            story.append(Image(str(path), width=image_width, height=image_width*ratio))
            story.append(Spacer(1, 5))
        else:
            style = "small" if block.startswith("图 ") or block.startswith("[1]") or \
                re.match(r"^\[[2-6]\]", block) or block.startswith("Assignment") else "body"
            story.append(Paragraph(inline(block), styles[style]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#ccd7df"))
    canvas.line(48, 42, A4[0]-48, 42)
    canvas.setFillColor(colors.HexColor("#506575"))
    canvas.setFont("Chinese", 8)
    canvas.drawString(48, 27, "MENG XIANGYU | G2607496E | 中文说明报告")
    canvas.drawRightString(A4[0]-48, 27, f"第 {doc.page} 页")
    canvas.restoreState()


output = ROOT / "output/pdf/MENGXIANGYU-G2607496E-中文说明报告.pdf"
doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48,
    topMargin=42, bottomMargin=55, title="手写数字分类中的降维：中文说明报告",
    author="MENG XIANGYU (G2607496E)")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(output)
