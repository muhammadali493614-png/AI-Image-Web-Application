import io
import hashlib
from datetime import datetime

# --- Compatibility shim ---------------------------------------------------
# Some Python/OpenSSL builds reject the 'usedforsecurity' kwarg on hashlib.md5(),
# but ReportLab's internal pdfdoc.py calls md5(usedforsecurity=False), causing:
# TypeError: 'usedforsecurity' is an invalid keyword argument for openssl_md5()
# Patching BEFORE importing reportlab makes it pick up this safe wrapper instead.
_original_md5 = hashlib.md5
def _safe_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return _original_md5(*args, **kwargs)
hashlib.md5 = _safe_md5
# ---------------------------------------------------------------------------

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ==========================================
# SHARED STYLES
# ==========================================
styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    'TitleStyle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor("#0B3D91"),
    alignment=TA_CENTER, spaceAfter=4
)
SUBTITLE_STYLE = ParagraphStyle(
    'SubtitleStyle', parent=styles['Normal'], fontSize=11, textColor=colors.grey,
    alignment=TA_CENTER, spaceAfter=14
)
SECTION_STYLE = ParagraphStyle(
    'SectionStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor("#0B3D91"),
    spaceBefore=14, spaceAfter=8
)
NORMAL_STYLE = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, alignment=TA_LEFT)
ALERT_STYLE = ParagraphStyle(
    'AlertStyle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor("#B00020"),
    alignment=TA_CENTER, spaceBefore=10, spaceAfter=10
)
OK_STYLE = ParagraphStyle(
    'OkStyle', parent=styles['Normal'], fontSize=12, textColor=colors.HexColor("#007A33"),
    alignment=TA_CENTER, spaceBefore=10, spaceAfter=10
)

PPE_ITEMS = ["helmet", "vest", "gloves", "shoes", "glasses"]


# ==========================================
# INTERNAL HELPERS
# ==========================================
def _header(elements, report_title, subtitle):
    elements.append(Paragraph("🛡️ SafeVision AI — PPE Compliance Report", TITLE_STYLE))
    elements.append(Paragraph(subtitle, SUBTITLE_STYLE))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#0B3D91"), thickness=1))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(report_title, SECTION_STYLE))


def _inspector_table(inspector_name, inspector_email, extra_rows=None):
    rows = [["Inspector Name:", inspector_name or "N/A"],
            ["Inspector Email:", inspector_email or "N/A"],
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]]
    if extra_rows:
        rows.extend(extra_rows)

    table = Table(rows, colWidths=[55 * mm, 110 * mm])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#0B3D91")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    return table


def _ppe_status_table(status, mandates):
    header = ["PPE Item", "Mandatory?", "Detected Status"]
    data = [header]

    for item in PPE_ITEMS:
        item_data = status.get(item, {"status": "❌ Missing", "is_missing": True})
        is_mandatory = "Yes" if mandates.get(item, False) else "No"
        data.append([item.capitalize(), is_mandatory, item_data["status"]])

    table = Table(data, colWidths=[55 * mm, 40 * mm, 60 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FA")]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return table


def _alert_paragraph(status):
    if status.get("alert"):
        return Paragraph(f"⚠️ {status.get('alert_message', 'Safety Violation Detected!')}", ALERT_STYLE)
    return Paragraph(f"✓ {status.get('alert_message', 'All Required PPE Compliant')}", OK_STYLE)


def _build_pdf(elements):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ==========================================
# 1) IMAGE / VIDEO UPLOAD DETECTION REPORT
# ==========================================
def generate_detection_report(source_type, filename, status, mandates, inspector_name="N/A", inspector_email="N/A"):
    """Used for /detect route — single image or video upload analysis."""
    elements = []
    _header(elements, "Detection Report", f"Source: {source_type}")

    elements.append(_inspector_table(inspector_name, inspector_email, extra_rows=[["File Name:", filename]]))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("PPE Detection Summary", SECTION_STYLE))
    elements.append(Paragraph(f"Total Persons Detected: <b>{status.get('total_persons', 0)}</b>", NORMAL_STYLE))
    elements.append(Spacer(1, 8))
    elements.append(_ppe_status_table(status, mandates))
    elements.append(_alert_paragraph(status))

    return _build_pdf(elements)


# ==========================================
# 2) LIVE WEBCAM SNAPSHOT REPORT (instant "Download PDF" button)
# ==========================================
def generate_live_snapshot_report(live_status, mandates, inspector_name="N/A", inspector_email="N/A"):
    """Used for /download_pdf_report — snapshot of current live_ppe_status."""
    elements = []
    _header(elements, "Live Snapshot Report", "Source: Live Webcam Feed (Instant Snapshot)")

    elements.append(_inspector_table(inspector_name, inspector_email))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Current PPE Status", SECTION_STYLE))
    elements.append(Paragraph(f"Total Persons Detected: <b>{live_status.get('total_persons', 0)}</b>", NORMAL_STYLE))
    elements.append(Spacer(1, 8))
    elements.append(_ppe_status_table(live_status, mandates))
    elements.append(_alert_paragraph(live_status))

    return _build_pdf(elements)


# ==========================================
# 3) LIVE RECORDING SESSION REPORT (after Start/Stop Recording)
# ==========================================
def generate_live_session_report(session_data, mandates, inspector_name="N/A", inspector_email="N/A"):
    """Used for /stop_recording — full session summary with frame stats."""
    elements = []
    _header(elements, "Live Recording Session Report", "Source: Live Webcam Feed (Recorded Session)")

    aggregate_status = session_data.get("aggregate_status", {})
    duration = session_data.get("duration_seconds", 0)
    total_frames = session_data.get("total_frames", 0)
    violation_frames = session_data.get("violation_frames", 0)
    violation_pct = round((violation_frames / total_frames) * 100, 1) if total_frames else 0

    extra_rows = [
        ["Session Start:", session_data.get("start_time", "N/A")],
        ["Session End:", session_data.get("end_time", "N/A")],
        ["Duration (sec):", f"{duration:.1f}"],
        ["Video File:", session_data.get("video_path", "N/A")],
    ]
    elements.append(_inspector_table(inspector_name, inspector_email, extra_rows=extra_rows))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Frame Analysis", SECTION_STYLE))
    frame_table = Table([
        ["Total Frames", "Violation Frames", "Violation %"],
        [str(total_frames), str(violation_frames), f"{violation_pct}%"]
    ], colWidths=[52 * mm, 52 * mm, 52 * mm])
    frame_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(frame_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Aggregate PPE Compliance (Full Session)", SECTION_STYLE))
    elements.append(Paragraph(
        f"Total Persons Detected (peak): <b>{aggregate_status.get('total_persons', 0)}</b>", NORMAL_STYLE
    ))
    elements.append(Spacer(1, 8))
    elements.append(_ppe_status_table(aggregate_status, mandates))
    elements.append(_alert_paragraph(aggregate_status))

    return _build_pdf(elements)