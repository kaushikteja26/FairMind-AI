"""
FairMind AI - PDF Compliance Report Generator
No paid APIs. Uses reportlab (free).
"""

import json
import os
from datetime import datetime

def generate_pdf_report(audit_results: dict, output_path: str) -> str:
    """
    Generates a professional PDF bias audit report.
    Falls back to text report if reportlab not available.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, black, white, red, green, orange
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        blue = HexColor('#1A73E8')
        dark = HexColor('#1a1a2e')
        light_blue = HexColor('#E8F0FE')
        red_color = HexColor('#D93025')
        green_color = HexColor('#1E8E3E')
        orange_color = HexColor('#E37400')

        title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                      fontSize=22, textColor=white,
                                      backColor=dark, alignment=TA_CENTER,
                                      spaceAfter=0, spaceBefore=0,
                                      leftIndent=-1*cm, rightIndent=-1*cm,
                                      borderPad=20)
        h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                              fontSize=14, textColor=blue, spaceAfter=6)
        h3 = ParagraphStyle('H3', parent=styles['Heading3'],
                              fontSize=11, textColor=dark, spaceAfter=4)
        body = ParagraphStyle('Body', parent=styles['Normal'],
                               fontSize=10, spaceAfter=4, leading=14)
        code = ParagraphStyle('Code', parent=styles['Normal'],
                               fontSize=9, fontName='Courier',
                               backColor=HexColor('#F8F9FA'),
                               spaceAfter=4, leading=13)

        story = []

        # Header
        story.append(Paragraph("🧠 FairMind AI", title_style))
        story.append(Paragraph("Bias Detection &amp; Fairness Audit Report", title_style))
        story.append(Spacer(1, 0.5*cm))

        # Metadata
        score = audit_results.get("severity_score", 0)
        n = audit_results.get("n_samples", 0)
        domain = audit_results.get("domain", "general")
        attrs = ", ".join(audit_results.get("protected_attributes", []))
        now = datetime.now().strftime("%B %d, %Y at %H:%M")

        meta_data = [
            ["Audit Date", now],
            ["Domain", domain.title()],
            ["Dataset Size", f"{n:,} records"],
            ["Protected Attributes", attrs],
            ["Severity Score", f"{score}/100"],
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 11*cm])
        score_color = green_color if score < 15 else orange_color if score < 40 else red_color
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), light_blue),
            ('TEXTCOLOR', (0, 0), (0, -1), dark),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DADCE0')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [white, HexColor('#F8F9FA')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.4*cm))

        # Severity banner
        if score < 15:
            sev_text = "✅ LOW BIAS - Model appears fair"
            sev_bg = HexColor('#E6F4EA')
            sev_color = green_color
        elif score < 40:
            sev_text = "⚠️ MEDIUM BIAS - Investigation recommended"
            sev_bg = HexColor('#FEF7E0')
            sev_color = orange_color
        elif score < 65:
            sev_text = "🔴 HIGH BIAS - Mitigation required before deployment"
            sev_bg = HexColor('#FCE8E6')
            sev_color = red_color
        else:
            sev_text = "🚨 CRITICAL BIAS - Halt deployment immediately"
            sev_bg = HexColor('#FCE8E6')
            sev_color = red_color

        sev_table = Table([[sev_text]], colWidths=[16*cm])
        sev_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), sev_bg),
            ('TEXTCOLOR', (0, 0), (-1, -1), sev_color),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(sev_table)
        story.append(Spacer(1, 0.4*cm))

        # Metrics per attribute
        story.append(Paragraph("Fairness Metrics by Protected Attribute", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=blue))
        story.append(Spacer(1, 0.2*cm))

        for attr, metrics in audit_results.get("metrics", {}).items():
            story.append(Paragraph(f"Attribute: {attr}", h3))

            metric_rows = [["Metric", "Value", "Threshold", "Status"]]

            dpd = metrics.get("demographic_parity_difference")
            if dpd is not None:
                status = "✅ PASS" if dpd < 0.1 else ("⚠️ WARN" if dpd < 0.2 else "🔴 FAIL")
                metric_rows.append(["Demographic Parity Difference", f"{dpd:.4f}", "< 0.1", status])

            di = metrics.get("disparate_impact_ratio")
            if di is not None:
                status = "✅ PASS" if di >= 0.8 else "🔴 FAIL"
                metric_rows.append(["Disparate Impact Ratio", f"{di:.4f}", "≥ 0.8 (80% rule)", status])

            eod = metrics.get("equalized_odds_difference")
            if eod is not None:
                status = "✅ PASS" if eod < 0.1 else ("⚠️ WARN" if eod < 0.2 else "🔴 FAIL")
                metric_rows.append(["Equalized Odds Difference", f"{eod:.4f}", "< 0.1", status])

            fprd = metrics.get("false_positive_rate_difference")
            if fprd is not None:
                status = "✅ PASS" if fprd < 0.1 else "⚠️ WARN"
                metric_rows.append(["False Positive Rate Difference", f"{fprd:.4f}", "< 0.1", status])

            mt = Table(metric_rows, colWidths=[6.5*cm, 3*cm, 4*cm, 2.5*cm])
            mt.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), dark),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DADCE0')),
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [white, HexColor('#F8F9FA')]),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(mt)

            # Group rates
            rates = metrics.get("positive_rates_by_group", {})
            if rates:
                story.append(Spacer(1, 0.2*cm))
                story.append(Paragraph("Positive Outcome Rates by Group:", body))
                rate_rows = [["Group", "Positive Rate"]]
                for g, r in rates.items():
                    rate_rows.append([str(g), f"{r:.1%}"])
                rt = Table(rate_rows, colWidths=[8*cm, 4*cm])
                rt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), light_blue),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DADCE0')),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(rt)
            story.append(Spacer(1, 0.3*cm))

        # Intersectional findings
        inter = audit_results.get("intersectional_findings", [])
        inter_2way = [x for x in inter if x.get("type") == "intersectional_2way"]
        if inter_2way:
            story.append(Paragraph("★ Intersectional Bias Analysis (Unique Feature)", h2))
            story.append(HRFlowable(width="100%", thickness=1, color=blue))
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                "The following shows bias found in COMBINATIONS of protected attributes — "
                "not detectable by standard single-attribute tools.", body))
            story.append(Spacer(1, 0.2*cm))

            inter_rows = [["Intersectional Group", "Positive Rate", "vs Average", "Disparity"]]
            for finding in inter_2way[:10]:
                group = finding.get("group", "")
                rate = finding.get("positive_rate", 0)
                overall = finding.get("overall_rate", 0)
                ratio = finding.get("disparity_ratio", 1.0)
                diff = rate - overall
                direction = f"+{diff:.1%}" if diff >= 0 else f"{diff:.1%}"
                severity_icon = "🔴" if abs(ratio-1) > 0.3 else ("⚠️" if abs(ratio-1) > 0.1 else "✅")
                inter_rows.append([group, f"{rate:.1%}", direction, severity_icon])

            it = Table(inter_rows, colWidths=[7*cm, 3*cm, 3*cm, 3*cm])
            it.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), dark),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#DADCE0')),
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [white, HexColor('#F8F9FA')]),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(it)
            story.append(Spacer(1, 0.4*cm))

        # Plain language explanation
        story.append(Paragraph("Plain-Language Summary", h2))
        story.append(HRFlowable(width="100%", thickness=1, color=blue))
        story.append(Spacer(1, 0.2*cm))
        explanation = audit_results.get("plain_explanation", "")
        for line in explanation.split("\n"):
            if line.strip():
                story.append(Paragraph(line.replace("&", "&amp;"), code))

        # ── Gemini AI Executive Narrative (if Gemini key was provided) ─────
        gemini_narrative = audit_results.get("gemini_narrative")
        if gemini_narrative:
            story.append(Spacer(1, 0.6*cm))
            story.append(Paragraph("AI-Generated Executive Summary", h2))
            story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#4285F4')))
            story.append(Spacer(1, 0.2*cm))
            # Google branding note
            gemini_label = ParagraphStyle(
                'gemini_label',
                parent=styles['Normal'],
                fontSize=8,
                textColor=HexColor('#4285F4'),
                spaceAfter=6,
            )
            story.append(Paragraph(
                "✦ Generated by Google Gemini AI — FairMind AI x Google Cloud",
                gemini_label
            ))
            gemini_style = ParagraphStyle(
                'gemini_body',
                parent=styles['Normal'],
                fontSize=10,
                leading=15,
                textColor=HexColor('#202124'),
                spaceAfter=8,
            )
            for para in gemini_narrative.split("\n\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(para.replace("&", "&amp;"), gemini_style))
                    story.append(Spacer(1, 0.15*cm))

        # Footer
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#DADCE0')))
        story.append(Paragraph(
            f"Generated by FairMind AI | {now} | "
            "This report is intended for compliance documentation and internal review.",
            ParagraphStyle('footer', parent=styles['Normal'], fontSize=8,
                           textColor=HexColor('#5F6368'), alignment=TA_CENTER)
        ))

        doc.build(story)
        return output_path

    except ImportError:
        # Fallback: plain text report
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w") as f:
            f.write("FAIRMIND AI - BIAS AUDIT REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Domain: {audit_results.get('domain', 'N/A')}\n")
            f.write(f"Severity Score: {audit_results.get('severity_score', 0)}/100\n\n")
            f.write(audit_results.get("plain_explanation", ""))
            f.write("\n\nFULL METRICS:\n")
            f.write(json.dumps(audit_results.get("metrics", {}), indent=2))
        return txt_path
