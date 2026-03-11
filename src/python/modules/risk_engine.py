import numpy as np
import warnings
import json
import os
import logging
from modules.feature_engineering import calculate_financial_ratios

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Five C's scoring helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_five_cs(ratios: dict, research: dict, entity: dict) -> dict:
    """
    Scores each of the Five C's on a 0-10 scale.
    Returns dict with {score, note} per C.
    """
    results = {}

    # 1. CHARACTER – driven by litigation, news sentiment
    char_score = 8.0
    char_note  = "Clean public record detected."
    if research.get("litigation_detected"):
        char_score -= 2.5
        char_note   = "Litigation / legal risk keywords detected in OSINT."
    neg_fraction = research.get("negative_hits", 0) / max(research.get("total_articles", 1), 1)
    if neg_fraction > 0.4:
        char_score -= 1.5
        char_note  += " High proportion of negative news."
    results["character"] = {"score": round(max(char_score, 0), 1), "note": char_note}

    # 2. CAPACITY – driven by profit margin, DSCR, interest coverage
    pm      = ratios.get("profit_margin", 0)
    dscr    = ratios.get("dscr", 0)
    ic      = ratios.get("interest_coverage", 0)
    cap_base = 5.0
    if pm > 0.10: cap_base += 1.5
    elif pm > 0.05: cap_base += 0.5
    if dscr > 1.5: cap_base += 1.2
    elif dscr > 1.0: cap_base += 0.5
    if ic > 3.0: cap_base += 1.5
    elif ic > 1.5: cap_base += 0.5
    cap_note = (f"Profit margin {pm*100:.1f}%, DSCR {dscr:.2f}x, "
                f"Interest Coverage {ic:.2f}x.")
    results["capacity"] = {"score": round(min(cap_base, 10), 1), "note": cap_note}

    # 3. CAPITAL – driven by D/E ratio, Net Worth, ROA
    de   = ratios.get("debt_equity_ratio", 1.0)
    roa  = ratios.get("roa", 0)
    cap2 = 5.0
    if de < 0.5:  cap2 += 2.0
    elif de < 1.0: cap2 += 1.0
    else:         cap2 -= 1.0
    if roa > 0.08: cap2 += 1.5
    elif roa > 0.03: cap2 += 0.5
    cap2_note = f"D/E ratio {de:.2f}, ROA {roa*100:.1f}%."
    results["capital"] = {"score": round(min(max(cap2, 0), 10), 1), "note": cap2_note}

    # 4. COLLATERAL – driven by asset coverage vs loan
    total_assets = float(entity.get("turnover", 0) or 0) * 2  # heuristic
    loan_amount  = float(entity.get("loan_amount", 0) or 1)
    coverage     = total_assets / max(loan_amount, 1)
    coll_score   = min(5.0 + coverage * 1.5, 10)
    cr_str       = ratios.get("current_ratio", 0)
    if float(cr_str) > 2: coll_score = min(coll_score + 1, 10)
    coll_note = f"Current ratio {cr_str}x, estimated asset coverage vs loan."
    results["collateral"] = {"score": round(coll_score, 1), "note": coll_note}

    # 5. CONDITIONS – driven by sector risk, macro sentiment
    sec_mod    = research.get("sector_risk_modifier", 5)
    cond_score = max(8.0 - (sec_mod / 3), 2.0)
    cond_note  = (research.get("macro_insights", "")[:120]
                  or "Macro environment assessed from sector OSINT signals.")
    results["conditions"] = {"score": round(cond_score, 1), "note": cond_note}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ML Risk Engine (simulated XGBoost with SHAP-style explanations)
# ─────────────────────────────────────────────────────────────────────────────
def run_ml_risk_engine(financial_ratios: dict, research_signals: dict, five_cs: dict) -> dict:
    risk_score = 0.0

    # Feature gate scoring
    de   = financial_ratios.get("debt_equity_ratio", 1.0)
    pm   = financial_ratios.get("profit_margin", 0.0)
    cf   = financial_ratios.get("cashflow_debt_coverage", 0.0)
    ic   = financial_ratios.get("interest_coverage", 0.0)
    dscr = financial_ratios.get("dscr", 0.0)
    roa  = financial_ratios.get("roa", 0.0)
    cr   = financial_ratios.get("current_ratio", 0.0)

    if de   > 1.0:  risk_score += 0.25
    elif de > 0.5:  risk_score += 0.10
    if pm   < 0.03: risk_score += 0.18
    elif pm < 0.08: risk_score += 0.08
    if research_signals.get("triangulation_flag"): risk_score += 0.18
    if research_signals.get("litigation_detected"): risk_score += 0.12
    if cf   < 0.2:  risk_score += 0.08
    if ic   < 1.5:  risk_score += 0.10
    elif ic < 3.0:  risk_score += 0.04
    if dscr < 1.0:  risk_score += 0.10
    if roa  < 0.03: risk_score += 0.06
    if cr   < 1.0:  risk_score += 0.08

    # Sector penalty
    risk_score += research_signals.get("sector_risk_modifier", 5) / 200.0

    default_probability = min(max(risk_score, 0.02), 0.97)

    # Rating
    if default_probability < 0.25:
        category       = "A+ (Very Low Risk)"
        recommendation = "Approve"
    elif default_probability < 0.40:
        category       = "A (Low Risk)"
        recommendation = "Approve"
    elif default_probability < 0.55:
        category       = "B+ (Moderate Risk)"
        recommendation = "Approve with Conditions"
    elif default_probability < 0.70:
        category       = "B (Medium Risk)"
        recommendation = "Review Required / Covenants Needed"
    else:
        category       = "C (High Risk)"
        recommendation = "Reject"

    # SHAP-style explanations -- split into positive drivers and risk drivers
    positive_drivers = []
    risk_drivers     = []

    if de <= 0.5:
        positive_drivers.append(f"Low Debt-to-Equity ratio ({de:.2f}x) → strong capital structure ↓ Risk")
    else:
        risk_drivers.append(f"High Debt-to-Equity ratio ({de:.2f}x) → elevated leverage ↑ Risk")

    if pm >= 0.08:
        positive_drivers.append(f"Strong profit margin ({pm*100:.1f}%) → pricing power ↓ Risk")
    else:
        risk_drivers.append(f"Thin profit margin ({pm*100:.1f}%) → limited downside buffer ↑ Risk")

    if cf >= 0.2:
        positive_drivers.append(f"Healthy cash-flow debt coverage ({cf:.2f}x) → liquidity assured ↓ Risk")
    else:
        risk_drivers.append(f"Low cash-flow debt coverage ({cf:.2f}x) → liquidity concern ↑ Risk")

    if ic >= 3.0:
        positive_drivers.append(f"Strong interest coverage ({ic:.2f}x) → servicing capacity assured ↓ Risk")
    elif ic >= 1.5:
        positive_drivers.append(f"Adequate interest coverage ({ic:.2f}x) ↓ Risk")
    else:
        risk_drivers.append(f"Weak interest coverage ({ic:.2f}x) → debt servicing risk ↑ Risk")

    if dscr >= 1.5:
        positive_drivers.append(f"DSCR {dscr:.2f}x → loan repayment capacity is comfortable ↓ Risk")
    elif dscr < 1.0:
        risk_drivers.append(f"DSCR below 1.0 ({dscr:.2f}x) → repayment capacity concern ↑ Risk")

    if roa >= 0.06:
        positive_drivers.append(f"ROA {roa*100:.1f}% → efficient asset utilisation ↓ Risk")

    if not research_signals.get("triangulation_flag"):
        positive_drivers.append("No major adverse OSINT flags → stable external risk profile ↓ Risk")
    else:
        risk_drivers.append("High negative media / litigation signals detected in OSINT ↑ Risk")

    if research_signals.get("litigation_detected"):
        risk_drivers.append("Potential legal proceedings detected in secondary research ↑ Risk")

    # Combine for display (risk drivers first for drama)
    all_shap = risk_drivers[:4] + positive_drivers[:4]

    return {
        "default_probability": round(default_probability, 4),
        "risk_category":       category,
        "recommendation":      recommendation,
        "top_features_shap":   all_shap[:8],
        "positive_drivers":    positive_drivers,
        "risk_drivers":        risk_drivers,
        "five_cs":             five_cs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gap 5 — Suggested Facility Terms
# ─────────────────────────────────────────────────────────────────────────────
def compute_suggested_terms(pd: float, requested_amount: float, requested_rate: float,
                            risk_drivers: list, five_cs: dict) -> dict:
    """
    Returns AI-suggested loan amount, interest rate, and a specific reason string.
    """
    base_rate = 9.0  # RBI repo rate floor approximation

    # Suggested amount based on PD
    if pd < 0.25:
        suggested_amount = requested_amount
        amount_note = "Full requested amount recommended — strong credit profile."
    elif pd < 0.40:
        suggested_amount = round(requested_amount * 0.90, 2)
        amount_note = "10% haircut recommended — low-moderate risk profile."
    elif pd < 0.55:
        suggested_amount = round(requested_amount * 0.75, 2)
        amount_note = "25% haircut recommended — moderate risk, enhanced monitoring covenants required."
    elif pd < 0.70:
        suggested_amount = round(requested_amount * 0.50, 2)
        amount_note = "50% haircut recommended — elevated risk; collateral enhancement required."
    else:
        suggested_amount = 0.0
        amount_note = "Facility not recommended — probability of default exceeds acceptable threshold."

    # Suggested rate = max(requested, base + risk spread)
    risk_spread = round(pd * 5.0, 2)   # PD=30% → +1.5%, PD=60% → +3.0%
    suggested_rate = round(max(requested_rate, base_rate + risk_spread), 2)
    rate_note = f"Base {base_rate}% + risk spread {risk_spread}% (PD-linked pricing)."

    # Build specific rejection / reduction reason
    if suggested_amount == 0:
        top_driver = risk_drivers[0] if risk_drivers else "High default probability"
        rejection_reason = (f"Rejected: {top_driver}. PD={pd*100:.1f}% exceeds "
                            f"the 70% rejection threshold. Recommend re-evaluation after "
                            f"12 months of improved financials.")
    elif suggested_amount < requested_amount:
        top_driver = risk_drivers[0] if risk_drivers else "Moderate risk signals"
        rejection_reason = (f"Limit reduced to ₹{suggested_amount} Cr ({int(suggested_amount/max(requested_amount,1)*100)}% of request). "
                            f"Primary driver: {top_driver}")
    else:
        rejection_reason = "Full facility approved at requested terms."

    # Lowest Five C score as additional context
    weakest_c = min(five_cs.items(), key=lambda x: x[1].get('score', 10), default=(None, {}))
    if weakest_c[0]:
        rejection_reason += f" Weakest C: {weakest_c[0].title()} ({weakest_c[1].get('score','—')}/10)."

    return {
        "suggested_amount":  suggested_amount,
        "requested_amount":  requested_amount,
        "amount_note":       amount_note,
        "suggested_rate":    suggested_rate,
        "requested_rate":    requested_rate,
        "rate_note":         rate_note,
        "rejection_reason":  rejection_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def orchestrate_engine(entity_id: int, pipeline_status: dict, analyst_notes: str = ""):
    """Main wrapper – runs full risk pipeline and returns report_data dict."""
    from modules.research_agent import execute_precognitive_research
    from db import get_db_connection

    def _upd(stage, status, msg):
        pipeline_status[stage] = status
        logger.info(f"[Entity {entity_id}] {stage}: {status} – {msg}")

    try:
        conn   = get_db_connection()
        entity = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        conn.close()

        if not entity:
            return {"error": "Entity not found"}

        # Convert sqlite3.Row → plain dict so .get() works everywhere
        entity = dict(entity)

        company_name = entity.get("company_name", "")
        sector       = entity.get("sector", "")

        # Load financials
        base_dir     = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        processed_dir= os.path.join(base_dir, "data", "processed", str(entity_id))
        filepath     = os.path.join(processed_dir, "entity_financials.json")

        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                financials = json.load(f)
        else:
            financials = {
                "Revenue": 0, "Net Profit": 0, "Total Debt": 0,
                "Total Assets": 0, "Operating Cash Flow": 0
            }

        ratios = calculate_financial_ratios(financials)

        _upd("research", "running", f"Scraping OSINT for {company_name}")
        research = execute_precognitive_research(company_name, sector)
        _upd("research", "completed",
             f"{len(research.get('articles', []))} signals | litigation={research.get('litigation_detected')}")

        # Gap 2 — India signals
        try:
            from modules.india_signals import run_india_checks
            india_signals = run_india_checks(financials, dict(entity))
        except Exception as ie:
            logger.warning(f"india_signals skipped: {ie}")
            india_signals = {}

        five_cs = compute_five_cs(ratios, research, dict(entity))

        _upd("risk_model", "running", "Evaluating financial vector and Five C's")
        risk_profile = run_ml_risk_engine(ratios, research, five_cs)

        # Gap 1 — Analyst notes sentiment adjustment
        if analyst_notes and analyst_notes.strip():
            from textblob import TextBlob
            polarity = TextBlob(analyst_notes).sentiment.polarity
            adjustment = round(polarity * -0.08, 4)   # negative note → higher PD
            new_pd = min(max(risk_profile["default_probability"] + adjustment, 0.02), 0.97)
            risk_profile["default_probability"] = round(new_pd, 4)
            if polarity < -0.1:
                risk_profile["risk_drivers"].append(
                    f"Analyst note indicates concern: '{analyst_notes[:80]}…' ↑ Risk")
            elif polarity > 0.1:
                risk_profile["positive_drivers"].append(
                    f"Analyst note is positive: '{analyst_notes[:80]}…' ↓ Risk")

        _upd("risk_model", "completed", f"PD={risk_profile['default_probability']} | {risk_profile['risk_category']}")

        # Gap 5 — Suggested facility terms
        suggested_terms = compute_suggested_terms(
            pd=risk_profile["default_probability"],
            requested_amount=float(entity.get("loan_amount") or 0),
            requested_rate=float(entity.get("interest_rate") or 10.0),
            risk_drivers=risk_profile.get("risk_drivers", []),
            five_cs=five_cs,
        )

        _upd("report_generation", "running", "Synthesising CAM report and GenAI SWOT")

        return {
            "entity":           dict(entity),
            "financials":       financials,
            "ratios":           ratios,
            "research_signals": research,
            "risk_profile":     risk_profile,
            "suggested_terms":  suggested_terms,
            "india_signals":    india_signals,
            "analyst_notes":    analyst_notes,
        }

    except Exception as e:
        import traceback
        _upd("report_generation", "failed", f"Engine Crash: {e}")
        logger.error(f"Engine crash for entity {entity_id}:\n{traceback.format_exc()}")
        return {"error": str(e)}
