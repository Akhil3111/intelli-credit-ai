import os
import json
import shutil
import threading
import logging
from flask import (Flask, render_template, request, redirect,
                   flash, send_file, url_for, jsonify)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("IntelliCredit")

load_dotenv()

from modules.document_processor import process_pdf
from modules.schema_mapper import process_extracted_data, save_to_processed, STANDARD_SCHEMA
from modules.report_generator import generate_reports
from db import get_db_connection, init_db

app = Flask(__name__)
app.secret_key = "intelli_credit_hackathon_2026"

basedir = os.path.abspath(os.path.dirname(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(basedir, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Ensure DB exists on startup
init_db()

# ── In-memory stores  ────────────────────────────────────────────────────────
# Keyed by entity_id (int)
_pipeline_status: dict[int, dict] = {}   # stage → status string
_report_cache:    dict[int, dict] = {}   # entity_id → report result dict
_doc_status:      dict[int, list] = {}   # entity_id → [{filename, category, status, steps}]

PIPELINE_STAGES = [
    "documents_uploaded",
    "classification",
    "extraction",
    "research",
    "risk_model",
    "report_generation",
]

def _init_pipeline(entity_id: int):
    _pipeline_status[entity_id] = {s: "pending" for s in PIPELINE_STAGES}

def _upd(entity_id: int, stage: str, status: str, detail: str = ""):
    if entity_id not in _pipeline_status:
        _init_pipeline(entity_id)
    _pipeline_status[entity_id][stage] = status
    logger.info(f"[Entity {entity_id}] {stage}={status}  {detail}")

# ── Document category mapping ────────────────────────────────────────────────
CATEGORY_MAPPING = {
    "alm_document":         "Asset Liability Management (ALM)",
    "shareholding_pattern": "Shareholding Pattern",
    "borrowing_profile":    "Borrowing Profile",
    "annual_reports":       "Annual Report",
    "portfolio_data":       "Portfolio Cuts",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Entity Onboarding
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    conn = get_db_connection()
    entities = conn.execute(
        "SELECT id, company_name, cin, status, turnover FROM entities ORDER BY id DESC"
    ).fetchall()
    conn.close()
    if not entities:
        return redirect(url_for("onboarding"))
    return render_template("home.html", entities=entities)


@app.route("/onboarding", methods=["GET"])
def onboarding():
    return render_template("entity_onboarding.html")


@app.route("/api/entity/create", methods=["POST"])
def create_entity():
    data = request.form
    company_name = data.get("company_name", "").strip()
    cin          = data.get("cin", "").strip()

    if not company_name or not cin:
        flash("Company Name and CIN are required fields.", "danger")
        return redirect(url_for("onboarding"))

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        # Check if CIN already exists ─ if so, just navigate to that entity's upload page
        existing = conn.execute(
            "SELECT id FROM entities WHERE cin = ?", (cin,)
        ).fetchone()

        if existing:
            entity_id = existing["id"]
            conn.close()
            _init_pipeline(entity_id)
            flash(f"Entity with CIN '{cin}' already exists — continuing to document upload.")
            return redirect(url_for("upload_documents", entity_id=entity_id))

        cursor.execute(
            """INSERT INTO entities
               (cin, pan, company_name, sector, subsector, turnover,
                incorporation_date, promoters, loan_type, loan_amount,
                loan_tenure, interest_rate, purpose)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cin, data.get("pan"), company_name,
                data.get("sector"), data.get("subsector"),
                data.get("turnover") or None,
                data.get("incorporation_date"), data.get("promoters"),
                data.get("loan_type"),
                data.get("loan_amount") or None,
                data.get("loan_tenure") or None,
                data.get("interest_rate") or None,
                data.get("purpose"),
            ),
        )
        entity_id = cursor.lastrowid
        conn.commit()
        conn.close()

        os.makedirs(os.path.join(app.config["UPLOAD_FOLDER"], str(entity_id)), exist_ok=True)
        _init_pipeline(entity_id)
        return redirect(url_for("upload_documents", entity_id=entity_id))

    except Exception as e:
        logger.error(f"create_entity error: {e}", exc_info=True)
        flash(f"Failed to create entity: {e}", "danger")
        return redirect(url_for("onboarding"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Document Upload
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/entity/<int:entity_id>/upload_documents", methods=["GET"])
def upload_documents(entity_id):
    return render_template("upload_documents.html", entity_id=entity_id)


# ── Auto-classifier (filename + content keyword matching) ─────────────────────
_AUTO_CLASSIFY_RULES = [
    # (category_key, display_name, filename_keywords, confidence)
    ("alm_document",         "Asset Liability Management (ALM)",
     ["alm", "asset_liab", "asset-liab", "liquidity", "maturity_profile",
      "gap_analysis", "liability", "funding_profile"], 0.92),

    ("shareholding_pattern", "Shareholding Pattern",
     ["shareholding", "share_holding", "shareholder", "ownership",
      "promoter_holding", "mca", "benpos", "equity_holding"], 0.93),

    ("borrowing_profile",    "Borrowing Profile",
     ["borrowing", "borrow", "loan_schedule", "credit_facility",
      "term_loan", "working_capital", "debt_schedule",
      "sanction_letter", "bank_limit"], 0.90),

    ("annual_reports",       "Annual Report",
     ["annual", "annual_report", "p&l", "pnl", "profit_loss",
      "balance_sheet", "cashflow", "cash_flow", "financial_statement",
      "audited", "standalone", "consolidated", "itar", "itr",
      "gst_return", "gstr"], 0.91),

    ("portfolio_data",       "Portfolio Cuts",
     ["portfolio", "npa", "segment", "sector_cut", "performance",
      "collection_efficiency", "pool", "disbursement",
      "vintage", "delinquency", "dpd", "par"], 0.89),
]
_SUPPORTED_EXTS = {".pdf", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".tiff"}


def auto_classify_file(filename: str) -> tuple[str, str, float]:
    """
    Returns (category_key, display_name, confidence) for a given filename.
    Falls back to 'annual_reports' with 0.50 confidence if no rule matches.
    """
    name_lower = filename.lower().replace(" ", "_").replace("-", "_")
    best_key, best_name, best_conf = "annual_reports", "Annual Report", 0.50

    for cat_key, cat_name, keywords, base_conf in _AUTO_CLASSIFY_RULES:
        for kw in keywords:
            if kw in name_lower:
                # Boost confidence slightly for longer / more specific keyword
                conf = round(base_conf + min(len(kw) / 100, 0.07), 2)
                if conf > best_conf:
                    best_key, best_name, best_conf = cat_key, cat_name, conf
                break   # first keyword match is enough per rule

    return best_key, best_name, best_conf


# ── Bulk ZIP Upload ───────────────────────────────────────────────────────────
@app.route("/api/entity/<int:entity_id>/upload_bulk", methods=["POST"])
def handle_bulk_upload(entity_id):
    """
    Accept a single .zip file, extract all supported documents,
    auto-classify each one, and redirect to classification review.
    """
    import zipfile

    entity_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(entity_id))
    os.makedirs(entity_dir, exist_ok=True)
    _upd(entity_id, "documents_uploaded", "running", "Processing ZIP upload")

    zip_file = request.files.get("bulk_zip")
    if not zip_file or not zip_file.filename:
        flash("Please select a ZIP file to upload.", "danger")
        return redirect(url_for("upload_documents", entity_id=entity_id))

    ext = os.path.splitext(zip_file.filename)[1].lower()
    if ext != ".zip":
        flash("Only .zip files are accepted for bulk upload.", "danger")
        return redirect(url_for("upload_documents", entity_id=entity_id))

    # Save the zip to a temp location, then extract
    zip_save_path = os.path.join(entity_dir, secure_filename(zip_file.filename))
    zip_file.save(zip_save_path)

    conn   = get_db_connection()
    cursor = conn.cursor()
    classified_docs = []

    try:
        with zipfile.ZipFile(zip_save_path, "r") as zf:
            for member in zf.infolist():
                # Skip directories and hidden/system files
                if member.is_dir():
                    continue
                fname = os.path.basename(member.filename)
                if fname.startswith(".") or fname.startswith("__"):
                    continue
                file_ext = os.path.splitext(fname)[1].lower()
                if file_ext not in _SUPPORTED_EXTS:
                    logger.info(f"Skipping unsupported file: {fname}")
                    continue

                safe_fname = secure_filename(fname)
                dest_path  = os.path.join(entity_dir, safe_fname)

                # Extract file
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

                # Auto-classify
                cat_key, cat_name, confidence = auto_classify_file(safe_fname)

                cursor.execute(
                    "INSERT OR IGNORE INTO documents "
                    "(entity_id, filename, category, status) VALUES (?,?,?,'Classified')",
                    (entity_id, safe_fname, cat_name),
                )
                doc_id = cursor.lastrowid or 0
                classified_docs.append({
                    "id":         doc_id,
                    "filename":   safe_fname,
                    "category":   cat_name,
                    "cat_key":    cat_key,
                    "confidence": confidence,
                })
                logger.info(f"[Bulk] {safe_fname} → {cat_name} ({confidence*100:.0f}%)")

    except zipfile.BadZipFile:
        flash("The uploaded file is not a valid ZIP archive.", "danger")
        conn.close()
        return redirect(url_for("upload_documents", entity_id=entity_id))
    finally:
        # Remove the zip after extraction
        try:
            os.remove(zip_save_path)
        except OSError:
            pass

    conn.commit()
    conn.close()

    if not classified_docs:
        _upd(entity_id, "documents_uploaded", "failed", "No supported files found in ZIP")
        flash("No supported document files found in the ZIP (supported: PDF, Excel, CSV, images).", "warning")
        return redirect(url_for("upload_documents", entity_id=entity_id))

    _upd(entity_id, "documents_uploaded", "completed", f"{len(classified_docs)} files extracted & classified")
    _upd(entity_id, "classification", "running", "Awaiting analyst review")

    return render_template(
        "review_classification.html",
        entity_id=entity_id,
        documents=classified_docs,
    )



@app.route("/api/entity/<int:entity_id>/upload", methods=["POST"])
def handle_document_upload(entity_id):
    entity_dir = os.path.join(app.config["UPLOAD_FOLDER"], str(entity_id))
    os.makedirs(entity_dir, exist_ok=True)
    _upd(entity_id, "documents_uploaded", "running", "Saving files to server")

    conn   = get_db_connection()
    cursor = conn.cursor()
    classified_docs = []

    for input_name, expected_category in CATEGORY_MAPPING.items():
        if input_name in request.files:
            f = request.files[input_name]
            if f and f.filename:
                filename  = secure_filename(f.filename)
                file_path = os.path.join(entity_dir, filename)
                f.save(file_path)

                cursor.execute(
                    "INSERT INTO documents (entity_id, filename, category, status) "
                    "VALUES (?, ?, ?, 'Classified')",
                    (entity_id, filename, expected_category),
                )
                doc_id = cursor.lastrowid
                classified_docs.append({
                    "id": doc_id, "filename": filename,
                    "category": expected_category, "confidence": 1.0,
                })

    conn.commit()
    conn.close()

    if not classified_docs:
        _upd(entity_id, "documents_uploaded", "failed", "No files received")
        flash("No valid documents uploaded.")
        return redirect(url_for("upload_documents", entity_id=entity_id))

    _upd(entity_id, "documents_uploaded", "completed", f"{len(classified_docs)} files saved")
    _upd(entity_id, "classification", "running", "Awaiting analyst review")

    return render_template(
        "review_classification.html",
        entity_id=entity_id,
        documents=classified_docs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a – Classification Review
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/entity/<int:entity_id>/confirm_classification", methods=["POST"])
def confirm_classification(entity_id):
    data = request.form
    conn = get_db_connection()
    for key, val in data.items():
        if key.startswith("category_"):
            doc_id = key.split("_")[1]
            conn.execute(
                "UPDATE documents SET category = ?, status = 'Verified' WHERE id = ?",
                (val, doc_id),
            )
    conn.commit()
    conn.close()

    _upd(entity_id, "classification", "completed", "Categories verified by analyst")
    _upd(entity_id, "extraction",     "running",   "Extracting text and schema")
    return redirect(url_for("schema_mapping", entity_id=entity_id))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b – Schema Mapping / Extraction
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/entity/<int:entity_id>/schema_mapping", methods=["GET"])
def schema_mapping(entity_id):
    conn = get_db_connection()
    docs = conn.execute(
        "SELECT * FROM documents WHERE entity_id = ?", (entity_id,)
    ).fetchall()
    conn.close()

    entity_dir   = os.path.join(app.config["UPLOAD_FOLDER"], str(entity_id))
    combined_text = ""

    for doc in docs:
        filepath = os.path.join(entity_dir, doc["filename"])
        if os.path.exists(filepath):
            try:
                combined_text += process_pdf(filepath) + "\n"
            except Exception:
                pass

    schema_data = process_extracted_data(combined_text, [])

    # Populate demo values if nothing was extracted
    all_zero = all(v.get("value", 0) == 0 for v in schema_data.values())
    if all_zero:
        schema_data = {
            "Revenue":                {"value": 150.50, "confidence": 0.85, "source": "Demo Data"},
            "Net Profit":             {"value": 22.00,  "confidence": 0.90, "source": "Demo Data"},
            "Total Debt":             {"value": 45.00,  "confidence": 0.75, "source": "Demo Data"},
            "Total Assets":           {"value": 210.00, "confidence": 0.88, "source": "Demo Data"},
            "Operating Cash Flow":    {"value": 15.00,  "confidence": 0.60, "source": "Demo Data"},
            "Current Assets":         {"value": 80.00,  "confidence": 0.55, "source": "Demo Data"},
            "Current Liabilities":    {"value": 50.00,  "confidence": 0.55, "source": "Demo Data"},
            "Interest Expense":       {"value": 5.50,   "confidence": 0.50, "source": "Demo Data"},
            "EBIT":                   {"value": 30.00,  "confidence": 0.70, "source": "Demo Data"},
            "EBITDA":                 {"value": 38.00,  "confidence": 0.65, "source": "Demo Data"},
            "Net Worth":              {"value": 120.00, "confidence": 0.80, "source": "Demo Data"},
        }

    return render_template(
        "schema_mapping.html",
        entity_id=entity_id,
        doc_count=len(docs),
        schema_data=schema_data,
    )


@app.route("/api/entity/<int:entity_id>/save_schema", methods=["POST"])
def save_schema(entity_id):
    data = request.form

    final_schema = {}
    # Standard fields
    for metric in STANDARD_SCHEMA.keys():
        key = "value_" + metric.replace(" ", "_")
        val = data.get(key, 0.0)
        final_schema[metric] = float(val) if val else 0.0

    # Custom user-added fields
    for key, val in data.items():
        if key.startswith("custom_label_"):
            idx   = key.replace("custom_label_", "")
            label = val.strip()
            v_key = f"custom_value_{idx}"
            c_val = data.get(v_key, 0.0)
            if label:
                try:
                    final_schema[label] = float(c_val)
                except Exception:
                    final_schema[label] = 0.0

    save_to_processed(entity_id, final_schema)

    conn = get_db_connection()
    conn.execute(
        "UPDATE entities SET status = 'Data Extracted' WHERE id = ?", (entity_id,)
    )
    conn.commit()
    conn.close()

    _upd(entity_id, "extraction",     "completed", "Schema validated and locked")
    _upd(entity_id, "research",       "pending",   "Queued")
    _upd(entity_id, "risk_model",     "pending",   "")
    _upd(entity_id, "report_generation", "pending","")

    # ── Background thread ──
    def _run_analysis():
        try:
            result = generate_reports(entity_id, _pipeline_status[entity_id])
            _report_cache[entity_id] = result
        except Exception as e:
            logger.error(f"Background analysis failed for {entity_id}: {e}")
            _pipeline_status[entity_id]["report_generation"] = "failed"

    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()

    return redirect(url_for("loading_page", entity_id=entity_id))


# ─────────────────────────────────────────────────────────────────────────────
# Loading page (polls until done, then auto-redirects)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/entity/<int:entity_id>/loading", methods=["GET"])
def loading_page(entity_id):
    return render_template("loading.html", entity_id=entity_id)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Dashboard
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/entity/<int:entity_id>/dashboard", methods=["GET"])
def risk_analysis_dashboard(entity_id):
    result = _report_cache.get(entity_id)

    # If not in cache, run synchronously (e.g. direct URL access)
    if result is None:
        if entity_id not in _pipeline_status:
            _init_pipeline(entity_id)
        result = generate_reports(entity_id, _pipeline_status[entity_id])
        _report_cache[entity_id] = result

    if "error" in result:
        flash(f"Analysis error: {result['error']}")
        return redirect(url_for("onboarding"))

    return render_template(
        "dashboard.html",
        data=result["report_data"],
        pdf_url=result["pdf_url"],
        docx_url=result["docx_url"],
        pipeline=_pipeline_status.get(entity_id, {}),
    )


# ─────────────────────────────────────────────────────────────────────────────
# APIs
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    """System health check endpoint."""
    import sqlite3, time
    db_ok = False
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        pass
    return jsonify({
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "entities_in_flight": len(_pipeline_status),
        "reports_cached": len(_report_cache),
        "timestamp": __import__('datetime').datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/pipeline_status/<int:entity_id>")
def pipeline_status_api(entity_id):
    status = _pipeline_status.get(entity_id, {s: "pending" for s in PIPELINE_STAGES})
    return jsonify(status)


@app.route("/api/entity/<int:entity_id>/docs_status")
def docs_status_api(entity_id):
    """Per-document processing status for the inspector panel."""
    docs = _doc_status.get(entity_id)
    if docs is None:
        # Fall back to DB
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT id, filename, category, status FROM documents WHERE entity_id = ?",
            (entity_id,)
        ).fetchall()
        conn.close()
        docs = [{"id": r["id"], "filename": r["filename"],
                 "category": r["category"], "status": r["status"]} for r in rows]
    return jsonify(docs)


@app.route("/api/entity/<int:entity_id>/doc_detail/<path:filename>")
def doc_detail_api(entity_id, filename):
    """Document inspector trace — shows per-step processing detail."""
    # Find extracted values from processed JSON
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    fin_path = os.path.join(base_dir, "data", "processed", str(entity_id), "entity_financials.json")
    schema_vals = {}
    if os.path.exists(fin_path):
        with open(fin_path) as f:
            schema_vals = json.load(f)

    result = _report_cache.get(entity_id, {})
    ratios = result.get("report_data", {}).get("ratios", {})

    schema_sample = [(k, v) for k, v in schema_vals.items() if v and v != 0][:6]
    ratio_sample  = [(k.replace("_", " ").title(), round(v, 3)) for k, v in ratios.items()][:4]

    pipeline_st = _pipeline_status.get(entity_id, {})
    ext_done = pipeline_st.get("extraction") == "completed" or pipeline_st.get("report_generation") == "completed"

    return jsonify({
        "filename": filename,
        "steps": [
            {"step": 1, "name": "OCR / Text Extraction",
             "status": "completed" if ext_done else "pending",
             "detail": "Text + tables extracted via PyMuPDF / Camelot"},
            {"step": 2, "name": "Table Detection",
             "status": "completed" if ext_done else "pending",
             "detail": f"Tables found: {len(schema_sample)} schema fields mapped"},
            {"step": 3, "name": "Schema Mapping",
             "status": "completed" if ext_done else "pending",
             "detail": [{"field": k, "value": f"₹{v} Cr"} for k, v in schema_sample]},
            {"step": 4, "name": "Feature Engineering",
             "status": "completed" if ext_done else "pending",
             "detail": [{"ratio": k, "value": str(v)} for k, v in ratio_sample]},
        ]
    })


@app.route("/api/entity/<int:entity_id>/download/<format_type>")
def download_report(entity_id, format_type):
    base_dir    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    reports_dir = os.path.join(base_dir, "reports", str(entity_id))

    filename_map = {"pdf": "credit_report.pdf", "docx": "credit_report.docx"}
    if format_type not in filename_map:
        return "Invalid format", 400

    filepath = os.path.join(reports_dir, filename_map[format_type])
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True,
                         download_name=f"CAM_Report_Entity_{entity_id}.{format_type}")
    flash("Report not found – please re-run the analysis.")
    return redirect(url_for("risk_analysis_dashboard", entity_id=entity_id))


# ─────────────────────────────────────────────────────────────────────────────
# OSINT Cache Control  (determinism fix)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/entity/<int:entity_id>/refresh_osint", methods=["DELETE"])
def refresh_osint_cache(entity_id):
    """
    Delete the cached OSINT result for this entity so the next
    pipeline run scrapes fresh data.
    """
    base_dir   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cache_path = os.path.join(base_dir, "data", "processed", str(entity_id), "osint_cache.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        logger.info(f"[Entity {entity_id}] OSINT cache cleared.")
        return jsonify({"success": True, "message": "OSINT cache cleared. Next run will scrape fresh data."})
    return jsonify({"success": False, "message": "No cache found for this entity."})


# ─────────────────────────────────────────────────────────────────────────────
# Gap 1 — Analyst Qualitative Notes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/entity/<int:entity_id>/notes", methods=["POST"])
def save_analyst_notes(entity_id):
    """
    Save credit officer qualitative notes and re-run risk scoring
    with sentiment factored into PD adjustment.
    """
    data  = request.get_json(force=True) or {}
    notes = data.get("notes", "").strip()

    if not notes:
        return jsonify({"error": "Notes text is required."}), 400

    # Persist to DB
    try:
        conn = get_db_connection()
        conn.execute(
            "UPDATE entities SET analyst_notes = ? WHERE id = ?",
            (notes, entity_id)
        )
        conn.commit()
        conn.close()
    except Exception as db_err:
        logger.warning(f"Could not save notes to DB (column may not exist yet): {db_err}")

    # Re-score with notes if report already cached
    cached      = _report_cache.get(entity_id, {})
    report_data = cached.get("report_data")
    new_pd      = None
    recommendation = None

    if report_data:
        from textblob import TextBlob
        from modules.risk_engine import compute_suggested_terms
        polarity   = TextBlob(notes).sentiment.polarity
        adjustment = round(polarity * -0.08, 4)
        old_pd     = report_data["risk_profile"]["default_probability"]
        new_pd     = round(min(max(old_pd + adjustment, 0.02), 0.97), 4)
        report_data["risk_profile"]["default_probability"] = new_pd
        report_data["analyst_notes"] = notes

        report_data["suggested_terms"] = compute_suggested_terms(
            pd=new_pd,
            requested_amount=float(report_data["entity"].get("loan_amount") or 0),
            requested_rate=float(report_data["entity"].get("interest_rate") or 10.0),
            risk_drivers=report_data["risk_profile"].get("risk_drivers", []),
            five_cs=report_data["risk_profile"].get("five_cs", {}),
        )

        if polarity < -0.1:
            report_data["risk_profile"]["risk_drivers"].append(
                f"Analyst note concern: '{notes[:80]}…' ↑ Risk"
            )
        elif polarity > 0.1:
            report_data["risk_profile"]["positive_drivers"].append(
                f"Analyst note positive: '{notes[:80]}…' ↓ Risk"
            )

        recommendation = report_data["risk_profile"]["recommendation"]
        _report_cache[entity_id]["report_data"] = report_data
        logger.info(f"[Entity {entity_id}] Notes saved | PD {old_pd:.3f} → {new_pd:.3f}")

    return jsonify({"success": True, "pd_adjusted": new_pd, "recommendation": recommendation})


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404,
                           message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500,
                           message="Internal server error. Check app.log for details."), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)