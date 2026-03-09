from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.pdfgen import canvas
from datetime import datetime
import json

# Colors from image
HEADER_GREEN = colors.Color(0.85, 0.90, 0.82) # Light greenish color
WHITE = colors.white
DARK_TEXT = colors.black
RED_LINE = colors.Color(0.5, 0.1, 0.1)

def draw_header(canvas, doc, title_data=None):
    canvas.saveState()
    
    # "CREDIT APPRAISAL MEMORANDUM"
    canvas.setFont("Times-Roman", 16)
    canvas.drawCentredString(letter[0]/2, letter[1] - 40, "CREDIT APPRAISAL MEMORANDUM")
    
    # Name of the applicant
    if not title_data:
        title_data = "Name of the Applicant"
    
    canvas.setFillColor(colors.orange)
    canvas.setFont("Times-Roman", 14)
    canvas.drawCentredString(letter[0]/2, letter[1] - 60, title_data)
    
    # Logo placement (approx top right)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawRightString(letter[0] - 40, letter[1] - 40, "intec")
    canvas.setFont("Helvetica", 6)
    canvas.drawRightString(letter[0] - 40, letter[1] - 48, "CAPITAL LIMITED")
    
    # Top border line
    canvas.setStrokeColor(colors.lightblue)
    canvas.setLineWidth(0.5)
    canvas.line(40, letter[1] - 70, letter[0] - 40, letter[1] - 70)
    
    canvas.restoreState()

def draw_footer(canvas, doc, rmdetails="Name of RM & Credit Analyst & Date"):
    canvas.saveState()
    
    # Bottom red line
    canvas.setStrokeColor(RED_LINE)
    canvas.setLineWidth(2)
    canvas.line(40, 50, letter[0] - 40, 50)
    
    canvas.setFillColor(colors.black)
    canvas.setFont("Times-Roman", 12)
    canvas.drawString(40, 35, rmdetails)
    
    canvas.setFont("Times-Roman", 10)
    canvas.setFillColor(colors.orange)
    canvas.drawRightString(letter[0] - 80, 35, "TimesNew Roman 12")
    
    canvas.setFillColor(colors.black)
    canvas.setFont("Times-Roman", 12)
    page_num = f"Page {doc.page}"
    canvas.drawRightString(letter[0] - 40, 35, page_num)
    
    canvas.restoreState()


def get_paragraph(text, style_name='BodyText', align=TA_LEFT, font='Times-Roman', size=10, textColor=colors.black):
    styles = getSampleStyleSheet()
    # Simple dynamic style to avoid conflicts
    unique_style = f"{style_name}_{align}_{font}_{size}_{textColor.hexval()}"
    if unique_style not in styles:
        styles.add(ParagraphStyle(
            name=unique_style,
            fontName=font,
            fontSize=size,
            textColor=textColor,
            alignment=align
        ))
    return Paragraph(str(text), styles[unique_style])

def create_table(data, col_widths, style_cmds):
    t = Table(data, colWidths=col_widths)
    # Default styling for all grids
    base_style = [
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Times-Roman'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]
    base_style.extend(style_cmds)
    t.setStyle(TableStyle(base_style))
    return t

def create_cam_pdf(cam_text, output_path, borrower_name="Unknown Organization"):
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=80, bottomMargin=70 
    )
                            
    try:
        report_data = json.loads(cam_text)
    except Exception as e:
        report_data = {"applicant_name": borrower_name}
        
    app_name = report_data.get("applicant_name", borrower_name)
    rm_name = report_data.get("rm_name", "AI Agent")
    analyst_name = report_data.get("credit_analyst_name", "IntelliCredit Core")
    date_str = report_data.get("date", datetime.now().strftime("%d-%b-%Y"))
    footer_text = f"{rm_name} & {analyst_name} & {date_str}"
    
    story = []
    
    # 1. Borrower Overview
    story.append(get_paragraph("<b>1. Borrower Overview</b>", size=12))
    story.append(Spacer(1, 4))
    b_overview = report_data.get("borrower_overview", {})
    t1_data = [
        [get_paragraph("<b>Business Description</b>"), get_paragraph(b_overview.get("description", ""))],
        [get_paragraph("<b>Industry</b>"), get_paragraph(b_overview.get("industry", ""))],
        [get_paragraph("<b>Key Activities</b>"), get_paragraph(b_overview.get("key_activities", ""))]
    ]
    t1 = create_table(t1_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t1)
    story.append(Spacer(1, 15))
    
    # 2. Financial Performance
    story.append(get_paragraph("<b>2. Financial Performance</b>", size=12))
    story.append(Spacer(1, 4))
    f_perf = report_data.get("financial_performance", {})
    t2_data = [
        [get_paragraph("<b>Net Profit</b>"), get_paragraph(str(f_perf.get("net_profit", "")))],
        [get_paragraph("<b>Return on Assets (ROA)</b>"), get_paragraph(str(f_perf.get("roa", "")))],
        [get_paragraph("<b>Net NPA</b>"), get_paragraph(str(f_perf.get("npa", "")))],
        [get_paragraph("<b>Capital Adequacy Ratio</b>"), get_paragraph(str(f_perf.get("capital_adequacy", "")))]
    ]
    t2 = create_table(t2_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t2)
    story.append(Spacer(1, 15))
    
    # 3. Revenue Validation
    story.append(get_paragraph("<b>3. Revenue Validation (GST & Bank)</b>", size=12))
    story.append(Spacer(1, 4))
    r_val = report_data.get("revenue_validation", {})
    t3_data = [
        [get_paragraph("<b>Mismatch Status</b>"), get_paragraph(str(r_val.get("gst_bank_mismatch_status", "")))],
        [get_paragraph("<b>Suspicious Counterparties</b>"), get_paragraph(str(r_val.get("suspicious_counterparties", "")))],
        [get_paragraph("<b>ML Anomalies Detected</b>"), get_paragraph(str(r_val.get("ml_anomalies", "")))]
    ]
    t3 = create_table(t3_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t3)
    story.append(Spacer(1, 15))
    
    # 4. External Intelligence
    story.append(get_paragraph("<b>4. External Intelligence (News & Litigation)</b>", size=12))
    story.append(Spacer(1, 4))
    e_int = report_data.get("external_intelligence", {})
    t4_data = [
        [get_paragraph("<b>News Sentiment</b>"), get_paragraph(e_int.get("news_sentiment", ""))],
        [get_paragraph("<b>Litigation Found</b>"), get_paragraph(str(e_int.get("litigation_found", "")))],
        [get_paragraph("<b>Key Risks Highlighted</b>"), get_paragraph(e_int.get("key_risks", ""))]
    ]
    t4 = create_table(t4_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t4)
    story.append(Spacer(1, 15))
    
    # 5. Risk Assessment (Five Cs)
    story.append(get_paragraph("<b>5. Risk Assessment (The Five Cs)</b>", size=12))
    story.append(Spacer(1, 4))
    five_cs = report_data.get("risk_assessment_five_cs", {})
    t5_data = [
        [get_paragraph("<b>Character</b>"), get_paragraph(str(five_cs.get("character", "")))],
        [get_paragraph("<b>Capacity</b>"), get_paragraph(str(five_cs.get("capacity", "")))],
        [get_paragraph("<b>Capital</b>"), get_paragraph(str(five_cs.get("capital", "")))],
        [get_paragraph("<b>Collateral</b>"), get_paragraph(str(five_cs.get("collateral", "")))],
        [get_paragraph("<b>Conditions</b>"), get_paragraph(str(five_cs.get("conditions", "")))]
    ]
    t5 = create_table(t5_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t5)
    story.append(Spacer(1, 15))
    
    # 6. Credit Recommendation
    story.append(get_paragraph("<b>6. Final Credit Recommendation</b>", size=12))
    story.append(Spacer(1, 4))
    rec = report_data.get("credit_recommendation", {})
    
    # Color-code the decision
    decision_val = str(rec.get("decision", "Review"))
    dec_color = colors.orange
    if "approve" in decision_val.lower(): dec_color = colors.green
    elif "reject" in decision_val.lower(): dec_color = colors.red
    
    t6_data = [
        [get_paragraph("<b>Decision</b>"), get_paragraph(decision_val, textColor=dec_color, size=12)],
        [get_paragraph("<b>Proposed Limit</b>"), get_paragraph(str(rec.get("limit", "")))],
        [get_paragraph("<b>Interest Rate</b>"), get_paragraph(str(rec.get("interest_rate", "")))],
        [get_paragraph("<b>Rationale</b>"), get_paragraph(rec.get("rationale", ""))]
    ]
    t6 = create_table(t6_data, [150, 380], [('BACKGROUND', (0,0), (0,-1), HEADER_GREEN)])
    story.append(t6)

    try:
        def _on_page(canvas, doc):
            draw_header(canvas, doc, app_name)
            draw_footer(canvas, doc, footer_text)
            
        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    except Exception as e:
        print(f"Error building PDF: {e}")
