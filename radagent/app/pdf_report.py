"""
RadAgent PDF report builder.

Renders a clinical-style PDF from a pipeline result. Uses ReportLab Platypus
for safe Windows install (no system deps).

Layout:
  - Header strip: RadAgent | request_id | timestamp
  - Patient meta line: image filename + language + assessment
  - Findings table: class | calibrated p | threshold | above | confidence
  - Retrieved evidence: per-finding bulleted list with hyperlinks
  - Grad-CAM image (if available, top finding)
  - VLM Report: structured prose with bracket citations preserved
  - Footer: disclaimer + audit fingerprint

Run-time dependency: pip install reportlab
"""
from __future__ import annotations

import base64
import io
import re
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image as PDFImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


__all__ = ["build_pdf_report_bytes"]


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_RA_BLUE = colors.HexColor("#1e3a5f")
_RA_ACCENT = colors.HexColor("#5b9dff")
_RA_DARK = colors.HexColor("#0f1117")
_RA_GREEN = colors.HexColor("#3ddc97")
_RA_AMBER = colors.HexColor("#f7b955")
_RA_RED = colors.HexColor("#f06970")
_RA_GRAY = colors.HexColor("#9aa1b3")
_RA_LIGHT_GRAY = colors.HexColor("#e6e9f0")


def _make_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "RAtitle", parent=base["Title"],
            fontName="Helvetica-Bold", fontSize=18, leading=22,
            textColor=_RA_BLUE, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "RAsubtitle", parent=base["Normal"],
            fontName="Helvetica", fontSize=9, leading=12,
            textColor=_RA_GRAY, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "RAh2", parent=base["Heading2"],
            fontName="Helvetica-Bold", fontSize=12, leading=15,
            textColor=_RA_BLUE, spaceBefore=10, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "RAh3", parent=base["Heading3"],
            fontName="Helvetica-Bold", fontSize=10, leading=13,
            textColor=_RA_DARK, spaceBefore=6, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "RAbody", parent=base["Normal"],
            fontName="Helvetica", fontSize=9.5, leading=13,
            textColor=_RA_DARK, spaceAfter=4, alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "RAsmall", parent=base["Normal"],
            fontName="Helvetica", fontSize=8, leading=11,
            textColor=_RA_GRAY, spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "RAcode", parent=base["Normal"],
            fontName="Courier", fontSize=8, leading=11,
            textColor=_RA_DARK, leftIndent=8,
        ),
        "evidence_title": ParagraphStyle(
            "RAevtitle", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=9, leading=12,
            textColor=_RA_BLUE,
        ),
        "evidence_body": ParagraphStyle(
            "RAevbody", parent=base["Normal"],
            fontName="Helvetica", fontSize=8.5, leading=11,
            textColor=_RA_DARK, leftIndent=10, spaceAfter=4,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_text(s: str | None) -> str:
    if not s:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _confidence_color(level: str | None) -> colors.Color:
    if not level:
        return _RA_GRAY
    lvl = level.lower()
    if lvl == "high":
        return _RA_GREEN
    if lvl in ("medium", "med"):
        return _RA_AMBER
    return _RA_GRAY


def _build_findings_table(findings: list[dict], styles: dict) -> Table:
    """Findings table with calibrated p, threshold, above flag, confidence band."""
    headers = ["Finding", "Cal. P", "Threshold", "Above", "Confidence"]
    data = [headers]

    for f in findings:
        name = _safe_text(f.get("name", ""))
        cp = f.get("calibrated_probability")
        th = f.get("threshold")
        above = "★" if f.get("above_threshold") else ""
        level = f.get("confidence_level") or "—"

        cp_str = f"{cp:.3f}" if isinstance(cp, (int, float)) else "—"
        th_str = f"{th:.3f}" if isinstance(th, (int, float)) else "—"

        data.append([name, cp_str, th_str, above, _safe_text(level)])

    col_widths = [2.0 * inch, 0.9 * inch, 0.9 * inch, 0.6 * inch, 1.1 * inch]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _RA_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfd")]),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, _RA_BLUE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, _RA_BLUE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, _RA_LIGHT_GRAY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ])

    # Highlight above-threshold rows with a colored background
    for i, f in enumerate(findings, start=1):
        if f.get("above_threshold"):
            style.add("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e8f0ff"))
            style.add("FONTNAME", (0, i), (0, i), "Helvetica-Bold")
        # Confidence cell color
        col = _confidence_color(f.get("confidence_level"))
        style.add("TEXTCOLOR", (4, i), (4, i), col)
        style.add("FONTNAME", (4, i), (4, i), "Helvetica-Bold")

    tbl.setStyle(style)
    return tbl


def _embed_image_b64(png_b64: str, max_w_mm: float = 110, max_h_mm: float = 110) -> PDFImage:
    raw = base64.b64decode(png_b64)
    bio = io.BytesIO(raw)
    img = PDFImage(bio)
    iw, ih = img.imageWidth, img.imageHeight
    scale = min((max_w_mm * mm) / iw, (max_h_mm * mm) / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def _format_vlm_report_with_links(report: str | None, styles: dict) -> list:
    """Convert the VLM report into Platypus paragraphs, preserving citation
    brackets and turning URLs into clickable links."""
    if not report:
        return [Paragraph("<i>VLM report not available.</i>", styles["body"])]

    out = []
    url_re = re.compile(r"(https?://[^\s\]\)]+)")

    # Split by markdown-ish headings for nicer rendering
    lines = report.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(Spacer(1, 4))
            continue

        if stripped.startswith("####"):
            out.append(Paragraph(_safe_text(stripped.lstrip("#").strip()), styles["h3"]))
        elif stripped.startswith("###"):
            out.append(Paragraph(_safe_text(stripped.lstrip("#").strip()), styles["h2"]))
        elif stripped.startswith("##"):
            out.append(Paragraph(_safe_text(stripped.lstrip("#").strip()), styles["h2"]))
        else:
            # escape, then turn URLs into <link> tags
            text = _safe_text(stripped)
            text = url_re.sub(
                lambda m: f'<link href="{m.group(0)}" color="#1e3a5f"><u>{m.group(0)}</u></link>',
                text,
            )
            # bold-ify **...** emphasis (very simple)
            text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
            out.append(Paragraph(text, styles["body"]))
    return out


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def build_pdf_report_bytes(
    *,
    request_id: str,
    image_filename: str | None,
    structured: dict,
    retrieved: dict,
    cams_b64: dict,
    report: str | None,
    vlm_error: str | None,
    language: str,
    timings_ms: dict,
    vllm_enabled: bool,
    vllm_model: str | None,
) -> bytes:
    """Render a complete PDF report and return its bytes."""
    buf = io.BytesIO()
    styles = _make_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"RadAgent Report — {request_id}",
        author="RadAgent",
    )

    story: list = []
    findings = structured.get("findings", []) if isinstance(structured, dict) else []
    above = [f for f in findings if f.get("above_threshold")]
    n_above = len(above)
    overall = structured.get("overall_assessment", "—")

    # ---- Header ----
    story.append(Paragraph("RadAgent — Chest Radiograph Report", styles["title"]))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header_meta = (
        f"Request <b>{_safe_text(request_id)}</b> &nbsp;·&nbsp; "
        f"Generated <b>{ts}</b> &nbsp;·&nbsp; "
        f"Language <b>{_safe_text(language)}</b>"
    )
    story.append(Paragraph(header_meta, styles["subtitle"]))

    # ---- Image meta + assessment strip ----
    fname = _safe_text(image_filename or "uploaded_image")
    is_abnormal = (overall or "").lower() == "abnormal"
    assessment_hex = "#f7b955" if is_abnormal else "#3ddc97"
    strip_text = (
        f"<b>Image:</b> {fname} &nbsp;·&nbsp; "
        f"<b>Overall assessment:</b> "
        f'<font color="{assessment_hex}"><b>{_safe_text(overall).upper()}</b></font> &nbsp;·&nbsp; '
        f"<b>{n_above}</b> finding{'s' if n_above != 1 else ''} above threshold"
    )
    story.append(Paragraph(strip_text, styles["body"]))
    story.append(Spacer(1, 8))

    # ---- Findings table ----
    story.append(Paragraph("Specialist findings", styles["h2"]))
    story.append(_build_findings_table(findings, styles))
    story.append(Spacer(1, 8))

    # ---- Retrieved evidence (only for above-threshold) ----
    if any(retrieved.get(f["name"]) for f in above):
        story.append(Paragraph("Retrieved evidence", styles["h2"]))
        for f in above:
            name = f["name"]
            passages = retrieved.get(name, [])
            if not passages:
                continue
            story.append(Paragraph(f"{_safe_text(name)}", styles["h3"]))
            for i, p in enumerate(passages, start=1):
                title = _safe_text(p.get("title", "?"))
                section = _safe_text(p.get("section", "?"))
                source = _safe_text(p.get("source", "?"))
                score = p.get("score")
                url = p.get("source_url", "")
                excerpt = p.get("text", "")
                if len(excerpt) > 220:
                    excerpt = excerpt[:220] + "…"
                excerpt = _safe_text(excerpt)

                head = f'[{i}] <b>{title} › {section}</b> ({source}'
                if isinstance(score, (int, float)):
                    head += f", score {score:.3f}"
                head += ")"
                if url:
                    head += f' &nbsp;<link href="{url}" color="#1e3a5f"><u>(link)</u></link>'
                story.append(Paragraph(head, styles["evidence_title"]))
                story.append(Paragraph(excerpt, styles["evidence_body"]))
        story.append(Spacer(1, 6))

    # ---- Grad-CAM (top finding only, to keep PDF compact) ----
    if cams_b64 and above:
        top = above[0]["name"]
        cam_b64 = cams_b64.get(top)
        if cam_b64:
            story.append(Paragraph(f"Visual grounding — {_safe_text(top)} (Grad-CAM++)", styles["h2"]))
            story.append(Paragraph(
                "Heatmap overlay shows the pixels the specialist attended to "
                "for this finding. Warm regions (red/yellow) indicate higher attribution.",
                styles["small"],
            ))
            story.append(Spacer(1, 4))
            try:
                story.append(_embed_image_b64(cam_b64, max_w_mm=130, max_h_mm=130))
            except Exception as e:
                story.append(Paragraph(
                    f"<i>(Grad-CAM rendering failed: {_safe_text(str(e))})</i>",
                    styles["small"],
                ))
            story.append(Spacer(1, 6))

    # ---- VLM report ----
    if report:
        story.append(PageBreak())
        story.append(Paragraph("Vision-LLM grounded report", styles["h2"]))
        story.append(Paragraph(
            f"Generated by Qwen2.5-VL-7B-Instruct via vLLM-ROCm on AMD MI300X. "
            f"Every numbered citation [n] above maps to a real retrieved passage with source URL. "
            f"Language: <b>{_safe_text(language)}</b>.",
            styles["small"],
        ))
        story.append(Spacer(1, 6))
        story.extend(_format_vlm_report_with_links(report, styles))
    elif vlm_error:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Vision-LLM report", styles["h2"]))
        story.append(Paragraph(
            f"<i>VLM call failed: {_safe_text(vlm_error)}</i>", styles["body"]
        ))
    else:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Vision-LLM report", styles["h2"]))
        story.append(Paragraph(
            "<i>VLM disabled. Set VLLM_URL on the dashboard server to enable.</i>",
            styles["body"],
        ))

    # ---- Timings ----
    if timings_ms:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Pipeline timings (ms)", styles["h3"]))
        timings_str = " · ".join(
            f"{_safe_text(k)}: <b>{v}</b>" for k, v in timings_ms.items()
        )
        story.append(Paragraph(timings_str, styles["small"]))

    # ---- Footer disclaimer ----
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "<b>Disclaimer:</b> RadAgent is a research prototype (AMD Developer Hackathon 2026). "
        "Not validated for clinical use. Outputs must not be relied upon for diagnosis "
        "or any patient-facing decision. Specialist trained on NIH ChestX-ray14, RAG "
        "corpus from Wikipedia + StatPearls, VLM = Qwen2.5-VL-7B-Instruct.",
        styles["small"],
    ))

    doc.build(story)
    return buf.getvalue()
