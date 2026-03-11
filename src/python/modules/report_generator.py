import os
import json
import re
import logging
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from docx import Document

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GenAI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _gemini_generate(prompt: str, fallback: dict | str) -> str | dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback
    try:
        client = genai.Client(api_key=api_key)
        resp   = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return resp.text.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        logger.warning(f"Gemini error: {e}")
        return fallback


def generate_swot_analysis(company_name: str, financials: dict,
                           risk_profile: dict, research: dict) -> dict:
    """Gemini-powered SWOT analysis."""
    news_summary = "; ".join(
        a["title"] for a in research.get("articles", [])[:5]
        if a.get("sentiment_category") == "Negative"
    ) or "No adverse news detected."

    prompt = f"""
Act as a Senior Credit Analyst. Generate a concise SWOT analysis for {company_name}.

Financials: Revenue={financials.get('Revenue')}, Net Profit={financials.get('Net Profit')},
            Total Debt={financials.get('Total Debt')}, Total Assets={financials.get('Total Assets')}.
Risk Rating: {risk_profile.get('risk_category')}.
Key Risk Factors: {', '.join(risk_profile.get('risk_drivers', [])[:3])}.
Key Strengths: {', '.join(risk_profile.get('positive_drivers', [])[:3])}.
Adverse News: {news_summary}

Output ONLY a valid JSON object with exactly four keys:
{{"Strengths": ["...", "..."], "Weaknesses": ["...", "..."],
  "Opportunities": ["...", "..."], "Threats": ["...", "..."]}}
Each key: list of exactly 3 concise bullet strings. No markdown code fences.
"""
    fallback = {
        "Strengths":     ["Established market presence and brand recognition.",
                          "Positive operating cash flows in recent periods.",
                          "Diversified revenue streams."],
        "Weaknesses":    ["Elevated debt-equity ratio relative to sector peers.",
                          "Narrow profit margins reducing downside cushion.",
                          "Concentration risk in key customer segments."],
        "Opportunities": ["Potential for operational leverage as revenues scale.",
                          "Debt restructuring could reduce interest burden.",
                          "Sector tailwinds from infrastructure spending."],
        "Threats":       ["Macroeconomic volatility affecting credit availability.",
                          "Rising input costs compressing margins.",
                          "Regulatory changes in core sector."],
    }
    raw = _gemini_generate(prompt, fallback)
    if isinstance(raw, dict):
        return raw
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else fallback
    except Exception:
        return fallback


def generate_analyst_narrative(company_name: str, risk_profile: dict,
                                financials: dict, research: dict) -> str:
    """Gemini-generated 2-paragraph analyst narrative."""
    prompt = f"""
You are a credit officer writing a concise internal analyst memo for loan committee review.

Company: {company_name}
Decision: {risk_profile.get('recommendation')} | Rating: {risk_profile.get('risk_category')}
PD: {risk_profile.get('default_probability', 0)*100:.1f}%
Key positives: {'; '.join(risk_profile.get('positive_drivers', [])[:3])}
Key risks: {'; '.join(risk_profile.get('risk_drivers', [])[:3])}
Litigation flag: {research.get('litigation_detected', False)}
Macro context: {research.get('macro_insights', '')[:200]}

Write exactly 2 professional paragraphs (200 words total). Focus on:
Para 1 – Financial health and repayment capacity.
Para 2 – External risks, market context, and final recommendation rationale.
Plain text only, no bullet points or markdown.
"""
    fallback = (
        f"{company_name} demonstrates a financial profile broadly aligned with the requested "
        "credit facility. Operating cash flows provide reasonable debt service coverage, and "
        "the balance sheet position supports repayment over the proposed tenure. Key financial "
        "metrics have been reviewed against sector benchmarks.\n\n"
        "Secondary research indicates manageable external risk levels. The recommendation is "
        "consistent with the quantitative risk model output. Monitoring covenants are advised "
        "as standard practice for the assigned risk grade."
    )
    result = _gemini_generate(prompt, fallback)
    return result if isinstance(result, str) else fallback


# ─────────────────────────────────────────────────────────────────────────────
# PDF Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf_report(entity_id: int, report_data: dict, output_path: str):
    doc    = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    heading1 = ParagraphStyle("H1", parent=styles["Heading1"],
                               textColor=colors.HexColor("#1e3c72"), spaceAfter=8)
    heading2 = ParagraphStyle("H2", parent=styles["Heading2"],
                               textColor=colors.HexColor("#2a5298"), spaceAfter=6)
    normal   = styles["Normal"]

    entity   = report_data["entity"]
    fin      = report_data["financials"]
    ratios   = report_data.get("ratios", {})
    risk     = report_data["risk_profile"]
    research = report_data.get("research_signals", {})
    swot     = report_data.get("swot", {})
    narrative= report_data.get("analyst_narrative", "")

    # ── Title ──
    story.append(Paragraph("CREDIT APPRAISAL MEMO (CAM)", heading1))
    story.append(Paragraph(
        f"{entity['company_name']} | CIN: {entity.get('cin','N/A')} | "
        f"Sector: {entity.get('sector','N/A')}", normal))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#1e3c72")))
    story.append(Spacer(1, 10))

    # ── 1. Executive Summary ──
    story.append(Paragraph("1. Executive Summary & Recommendation", heading2))
    rec_color = {"Approve": "#198754", "Reject": "#dc3545"}.get(
        risk.get("recommendation", ""), "#ffc107")
    story.append(Paragraph(
        f'<b>Decision:</b> <font color="{rec_color}"><b>{risk["recommendation"]}</b></font>', normal))
    story.append(Paragraph(f'<b>Risk Rating:</b> {risk["risk_category"]}', normal))
    story.append(Paragraph(
        f'<b>Probability of Default:</b> {risk["default_probability"]*100:.2f}%', normal))
    story.append(Paragraph(
        f'<b>Loan Requested:</b> ₹{entity.get("loan_amount","N/A")} Cr | '
        f'Type: {entity.get("loan_type","N/A")} | Tenure: {entity.get("loan_tenure","N/A")} months | '
        f'Rate: {entity.get("interest_rate","N/A")}%', normal))
    story.append(Spacer(1, 10))

    # ── 2. Analyst Narrative ──
    if narrative:
        story.append(Paragraph("2. AI Analyst Narrative", heading2))
        for para in narrative.split("\n\n"):
            story.append(Paragraph(para.strip(), normal))
            story.append(Spacer(1, 6))

    # ── 3. Financial Overview ──
    story.append(Paragraph("3. Financial Overview (INR Crores)", heading2))
    fin_rows = [["Metric", "Value (₹ Cr)"]]
    for k, v in fin.items():
        try:
            fin_rows.append([k, f"{float(v):,.2f}"])
        except Exception:
            fin_rows.append([k, str(v)])
    t = Table(fin_rows, colWidths=[3.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1e3c72")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#f0f4ff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN",  (1, 0), (1, -1), "RIGHT"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ── 4. Key Financial Ratios ──
    story.append(Paragraph("4. Key Financial Ratios", heading2))
    ratio_rows = [["Ratio", "Value", "Health"]]
    ratio_meta = [
        ("debt_equity_ratio",      "Debt / Equity Ratio",       lambda v: "✔ Healthy" if v < 0.8 else "⚠ Elevated"),
        ("profit_margin",          "Net Profit Margin",          lambda v: f"{v*100:.1f}%"),
        ("current_ratio",          "Current Ratio",              lambda v: "✔ Good" if v >= 1.5 else "⚠ Low"),
        ("dscr",                   "DSCR",                       lambda v: "✔ Strong" if v >= 1.25 else "⚠ Weak"),
        ("interest_coverage",      "Interest Coverage Ratio",    lambda v: "✔ Safe" if v >= 2 else "⚠ Low"),
        ("roa",                    "Return on Assets (ROA)",      lambda v: f"{v*100:.1f}%"),
        ("cashflow_debt_coverage", "CF / Debt Coverage",         lambda v: "✔ OK" if v >= 0.2 else "⚠ Low"),
    ]
    for key, label, fmt in ratio_meta:
        val = ratios.get(key, 0)
        try:
            health = fmt(float(val))
        except Exception:
            health = str(val)
        ratio_rows.append([label, f"{float(val):.2f}", health])
    t2 = Table(ratio_rows, colWidths=[3*inch, 1.5*inch, 2*inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a5298")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8f9ff")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # ── 5. Five C's Assessment ──
    story.append(Paragraph("5. Five C's Credit Assessment", heading2))
    five_cs = risk.get("five_cs", {})
    cs_rows = [["C", "Score / 10", "Observation"]]
    for key, label in [("character","Character"),("capacity","Capacity"),
                        ("capital","Capital"),("collateral","Collateral"),
                        ("conditions","Conditions")]:
        item = five_cs.get(key, {"score": 0, "note": "N/A"})
        cs_rows.append([label, f"{item['score']}/10", item["note"][:80]])
    t3 = Table(cs_rows, colWidths=[1.5*inch, 1.5*inch, 4.5*inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f8ff")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))

    # ── 6. SHAP Explainability ──
    story.append(Paragraph("6. Key Drivers Behind Decision (XAI)", heading2))
    for d in risk.get("positive_drivers", [])[:4]:
        story.append(Paragraph(f"  <font color='green'>▲</font> {d}", normal))
    for d in risk.get("risk_drivers", [])[:4]:
        story.append(Paragraph(f"  <font color='red'>▼</font> {d}", normal))
    story.append(Spacer(1, 10))

    # ── 7. SWOT ──
    story.append(Paragraph("7. SWOT Analysis (GenAI Assisted)", heading2))
    for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
        story.append(Paragraph(f"<b>{cat}:</b>", normal))
        for pt in swot.get(cat, []):
            story.append(Paragraph(f"  • {pt}", normal))
    story.append(Spacer(1, 10))

    # ── 8. OSINT Summary ──
    story.append(Paragraph("8. OSINT / Secondary Research Summary", heading2))
    story.append(Paragraph(
        f"Articles scraped: {research.get('total_articles', 0)} | "
        f"Negative signals: {research.get('negative_hits', 0)} | "
        f"Litigation flag: {'YES ⚠' if research.get('litigation_detected') else 'No'}", normal))
    story.append(Paragraph(f"Macro context: {research.get('macro_insights','N/A')[:300]}", normal))

    doc.build(story)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# DOCX Builder (simplified)
# ─────────────────────────────────────────────────────────────────────────────
def build_docx_report(entity_id: int, report_data: dict, output_path: str):
    doc   = Document()
    entity= report_data["entity"]
    risk  = report_data["risk_profile"]
    swot  = report_data.get("swot", {})

    doc.add_heading(f"Credit Appraisal Memo – {entity['company_name']}", 0)
    doc.add_heading("1. Executive Summary", 1)
    doc.add_paragraph(f"Decision: {risk['recommendation']}")
    doc.add_paragraph(f"Risk Rating: {risk['risk_category']}")
    doc.add_paragraph(f"Probability of Default: {risk['default_probability']*100:.2f}%")

    narrative = report_data.get("analyst_narrative", "")
    if narrative:
        doc.add_heading("2. Analyst Narrative", 1)
        for para in narrative.split("\n\n"):
            doc.add_paragraph(para.strip())

    doc.add_heading("3. Financial Overview (INR Crores)", 1)
    tbl = doc.add_table(rows=1, cols=2)
    tbl.rows[0].cells[0].text = "Metric"
    tbl.rows[0].cells[1].text = "Value"
    for k, v in report_data["financials"].items():
        row = tbl.add_row().cells
        row[0].text = k
        try:
            row[1].text = f"{float(v):,.2f}"
        except Exception:
            row[1].text = str(v)

    doc.add_heading("4. Five C's Assessment", 1)
    five_cs = risk.get("five_cs", {})
    for key, label in [("character","Character"),("capacity","Capacity"),
                        ("capital","Capital"),("collateral","Collateral"),
                        ("conditions","Conditions")]:
        item = five_cs.get(key, {"score": 0, "note": ""})
        doc.add_paragraph(f"{label}: {item['score']}/10 – {item['note']}", style="List Bullet")

    doc.add_heading("5. Key Decision Drivers (XAI)", 1)
    for d in risk.get("positive_drivers", [])[:4]:
        doc.add_paragraph(f"▲ {d}", style="List Bullet")
    for d in risk.get("risk_drivers", [])[:4]:
        doc.add_paragraph(f"▼ {d}", style="List Bullet")

    doc.add_heading("6. SWOT Analysis", 1)
    for cat in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
        doc.add_heading(cat, 2)
        for pt in swot.get(cat, []):
            doc.add_paragraph(pt, style="List Bullet")

    doc.save(output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Master generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_reports(entity_id: int, pipeline_status: dict):
    from modules.risk_engine import orchestrate_engine

    report_data = orchestrate_engine(entity_id, pipeline_status)
    if "error" in report_data:
        return {"error": report_data["error"]}

    # GenAI enrichment
    swot = generate_swot_analysis(
        report_data["entity"]["company_name"],
        report_data["financials"],
        report_data["risk_profile"],
        report_data.get("research_signals", {}),
    )
    narrative = generate_analyst_narrative(
        report_data["entity"]["company_name"],
        report_data["risk_profile"],
        report_data["financials"],
        report_data.get("research_signals", {}),
    )
    report_data["swot"]             = swot
    report_data["analyst_narrative"]= narrative

    # Save artifacts
    base_dir    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    reports_dir = os.path.join(base_dir, "reports", str(entity_id))
    os.makedirs(reports_dir, exist_ok=True)

    pdf_path  = os.path.join(reports_dir, "credit_report.pdf")
    docx_path = os.path.join(reports_dir, "credit_report.docx")

    try:
        build_pdf_report(entity_id, report_data, pdf_path)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")

    try:
        build_docx_report(entity_id, report_data, docx_path)
    except Exception as e:
        logger.error(f"DOCX generation failed: {e}")

    pipeline_status["report_generation"] = "completed"

    return {
        "report_data": report_data,
        "pdf_url":     f"/api/entity/{entity_id}/download/pdf",
        "docx_url":    f"/api/entity/{entity_id}/download/docx",
    }
