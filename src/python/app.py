import os
import shutil
from flask import Flask, render_template, request, redirect, flash, send_file, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import from the newly structured modules
from modules.document_processor import process_pdf
from modules.financial_extractor import extract_financials
from modules.gst_bank_analysis import analyze_gst_bank
from modules.news_intelligence import process_news
from modules.risk_engine import compute_risk_score
from modules.cam_generator import generate_cam
from modules.pdf_report_generator import create_cam_pdf

app = Flask(__name__)
app.secret_key = 'supersecretkey'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
app.config['NEWS_FOLDER'] = os.path.join(basedir, 'uploads', 'news')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['NEWS_FOLDER'], exist_ok=True)

@app.route("/", methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        # Clear old uploads
        for folder in [app.config['UPLOAD_FOLDER'], app.config['NEWS_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path) and file_path != app.config['NEWS_FOLDER']:
                        shutil.rmtree(file_path)
                except Exception as e:
                    pass

        annual_report = request.files.get('annual_report')
        gst_csv = request.files.get('gst_csv')
        bank_csv = request.files.get('bank_csv')
        news_files = request.files.getlist('news_files')
        officer_notes = request.form.get('officer_notes', '')

        if not annual_report or not gst_csv or not bank_csv or not annual_report.filename or not gst_csv.filename or not bank_csv.filename:
            flash("Annual Report, GST CSV, and Bank CSV are required.")
            return redirect(request.url)

        report_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(annual_report.filename))
        gst_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gst_csv.filename))
        bank_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(bank_csv.filename))

        annual_report.save(report_path)
        gst_csv.save(gst_path)
        bank_csv.save(bank_path)

        for news in news_files:
            if news and news.filename:
                news.save(os.path.join(app.config['NEWS_FOLDER'], secure_filename(news.filename)))

        try:
            # 1. Process PDF Document
            print("[1/7] Processing PDF Document & chunking text...")
            extracted_text = process_pdf(report_path)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                flash("Error: The uploaded Annual Report PDF appears to be a scanned image or completely empty. Since Tesseract OCR is missing, the AI cannot read the text. Please upload a machine-readable text PDF.")
                return redirect(request.url)
            
            # 1.5 Extract Tables
            print("[2/7] Extracting structured financial tables (Pages 1-10)...")
            from modules.document_processor import extract_financial_tables
            financial_tables = extract_financial_tables(report_path)
            
            # 2. Extract Financials via Groq LLM & Regex fallbacks
            print("[3/7] Searching semantic vectors & Extracting Financials (Groq)...")
            financials = extract_financials(extracted_text, tables_data=financial_tables)
            
            # 3. Analyze GST and Bank mismatch
            print("[4/7] Running IsolationForest ML on GST vs Bank records...")
            gst_bank_results = analyze_gst_bank(gst_path, bank_path)
            
            # 4. Analyze News Intelligence
            print("[5/7] Crawling Web & Scanning for News Intelligence Risk...")
            organization_name = financials.get('Organization Name', 'Unknown Organization')
            news_insights = process_news(app.config['NEWS_FOLDER'], organization_name)
            
            # 5. Compute Risk Score (Five Cs)
            print("[6/7] Computing AI Five-Cs Risk Profile...")
            risk_results = compute_risk_score(financials, gst_bank_results, news_insights, officer_notes)
            
            # 6. Generate CAM Report text
            print("[7/7] Structuring data and generating ReportLab PDF Payload...")
            ca_memo_text = generate_cam(risk_results, financials, gst_bank_results, news_insights, extracted_text)
            
            # 7. Create Downloadable PDF
            cam_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "cam_report.pdf")
            organization_name = financials.get('Organization Name', 'Unknown Organization')
            create_cam_pdf(ca_memo_text, cam_pdf_path, organization_name)
            
            print("--- Analysis Complete! Rendering Dashboard ---")

            return render_template(
                "dashboard.html",
                total_score=risk_results['total_score'],
                decision=risk_results['decision'],
                interest=risk_results['interest_rate'],
                limit=risk_results['limit'],
                explanation=risk_results['explanation'],
                character=risk_results['character_score'],
                capacity=risk_results['capacity_score'],
                capital=risk_results['capital_score'],
                collateral=risk_results['collateral_score'],
                conditions=risk_results['conditions_score'],
                financials=financials,
                gst_bank_results=gst_bank_results,
                news_insights=news_insights,
                organization_name=financials.get('Organization Name', 'Unknown Organization')
            )
            
        except Exception as e:
            flash(f"Error during processing: {str(e)}")
            return redirect(request.url)

    return render_template("dashboard.html")

@app.route("/download_cam")
def download_cam():
    cam_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "cam_report.pdf")
    if os.path.exists(cam_pdf_path):
        return send_file(cam_pdf_path, as_attachment=True, download_name="Credit_Appraisal_Memo.pdf")
    else:
        flash("No CAM generated yet. Please submit your files first.")
        return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)