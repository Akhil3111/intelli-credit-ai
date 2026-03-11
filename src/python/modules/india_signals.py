"""
modules/india_signals.py
Gap 2 — India-specific risk signals:
  - GST vs Reported Revenue divergence detection (circular trading / revenue inflation)
  - GSTR-2A/3B mismatch estimation
  - NPA classification keywords from extracted text
  - India-specific financial flags
"""

import logging
import re

logger = logging.getLogger(__name__)

# Threshold for flagging GST vs reported revenue divergence
GST_DIVERGENCE_THRESHOLD = 0.15   # >15% divergence is flagged
CIRCULAR_TRADING_THRESHOLD = 0.20  # >20% same-party transactions is flagged

# India-specific NPA / stress keywords
_NPA_KEYWORDS = [
    "sub-standard", "doubtful", "loss asset", "npa", "non-performing",
    "sma-1", "sma-2", "sma1", "sma2", "special mention account",
    "wilful default", "write-off", "written off", "restructured",
    "one time settlement", "ots", "nclt", "insolvency",
]

# GSTR-2A/3B mismatch keywords in extracted text
_GSTR_MISMATCH_KW = [
    "gstr-2a", "gstr-3b", "itc mismatch", "input tax credit",
    "2a vs 3b", "itc reversal", "excess itc",
]


def detect_revenue_inflation(gst_turnover: float, reported_revenue: float) -> dict:
    """
    Compare GST-declared turnover vs reported revenue.
    >15% upward divergence in reported revenue = possible inflation flag.
    """
    if gst_turnover <= 0 or reported_revenue <= 0:
        return {"flag": False, "divergence_pct": 0.0,
                "note": "Insufficient data for GST vs Revenue cross-check."}

    divergence = (reported_revenue - gst_turnover) / gst_turnover
    flag = divergence > GST_DIVERGENCE_THRESHOLD

    return {
        "flag": flag,
        "divergence_pct": round(divergence * 100, 2),
        "gst_turnover": gst_turnover,
        "reported_revenue": reported_revenue,
        "note": (
            f"Reported revenue exceeds GST turnover by {divergence*100:.1f}% — "
            "possible revenue inflation. Recommend GSTR-3B reconciliation."
            if flag else
            f"GST vs Revenue divergence {divergence*100:.1f}% — within acceptable range."
        ),
    }


def detect_circular_trading(financials: dict) -> dict:
    """
    Heuristic circular trading detection:
    If Trade Receivables > 90 days of turnover AND Trade Payables closely match,
    it suggests round-tripping.
    """
    revenue = float(financials.get("Revenue", 0) or 0)
    trade_recv = float(financials.get("Trade Receivables", 0) or 0)
    trade_pay  = float(financials.get("Trade Payables", 0) or 0)

    if revenue <= 0:
        return {"flag": False, "note": "Insufficient data for circular trading check."}

    # Receivable days
    recv_days = (trade_recv / revenue) * 365 if revenue > 0 else 0

    # Symmetry ratio — if receivables ≈ payables it may indicate round-tripping
    symmetry = abs(trade_recv - trade_pay) / max(trade_recv, trade_pay, 1)
    circular_flag = recv_days > 90 and symmetry < CIRCULAR_TRADING_THRESHOLD

    return {
        "flag": circular_flag,
        "receivable_days": round(recv_days, 1),
        "symmetry_ratio": round(symmetry, 3),
        "note": (
            f"High receivable days ({recv_days:.0f}) + symmetric payables "
            "(symmetry={:.1f}%) — possible circular trading. Verify party-wise ledger."
            .format(symmetry * 100)
            if circular_flag else
            f"Receivable days: {recv_days:.0f} — No circular trading pattern detected."
        ),
    }


def detect_npa_risk(financials: dict) -> dict:
    """
    Scan extracted financial text/keys for NPA classification signals.
    """
    all_text = " ".join(str(v) for v in financials.values()).lower()
    hits = [kw for kw in _NPA_KEYWORDS if kw in all_text]
    flag = len(hits) > 0
    return {
        "flag": flag,
        "keywords_found": hits[:5],
        "note": (
            f"NPA/stress keywords detected: {', '.join(hits[:3])}. "
            "Review asset classification and provisioning."
            if flag else
            "No NPA classification signals detected in extracted data."
        ),
    }


def detect_gstr_mismatch(financials: dict) -> dict:
    """
    Check if extracted data contains GSTR-2A/3B mismatch signals.
    """
    all_text = " ".join(str(v) for v in financials.values()).lower()
    hits = [kw for kw in _GSTR_MISMATCH_KW if kw in all_text]
    flag = len(hits) > 0
    return {
        "flag": flag,
        "note": (
            "GSTR-2A / GSTR-3B mismatch signals found — ITC claim discrepancy risk."
            if flag else
            "No GSTR-2A/3B mismatch signals in extracted data."
        ),
    }


def run_india_checks(financials: dict, entity_meta: dict) -> dict:
    """
    Master function — runs all India-specific checks and returns a consolidated dict.
    """
    # Extract GST turnover if present (may have been mapped in schema)
    gst_turnover = float(financials.get("GST Turnover", 0) or
                         financials.get("GSTR-3B Turnover", 0) or 0)
    reported_revenue = float(financials.get("Revenue", 0) or
                              financials.get("Net Revenue", 0) or 0)

    rev_inflation = detect_revenue_inflation(gst_turnover, reported_revenue)
    circular      = detect_circular_trading(financials)
    npa           = detect_npa_risk(financials)
    gstr_mismatch = detect_gstr_mismatch(financials)

    # India-specific computed fields
    gst_number = entity_meta.get("gst_number", "")
    gst_valid  = bool(re.match(
        r"^\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]$", gst_number
    )) if gst_number else None

    # Overall India risk score (0-10)
    india_risk_score = 0
    signals = []
    if rev_inflation["flag"]:
        india_risk_score += 3
        signals.append(f"⚠ Revenue inflation: +{rev_inflation['divergence_pct']}% vs GST")
    if circular["flag"]:
        india_risk_score += 3
        signals.append(f"⚠ Circular trading pattern: {circular['receivable_days']} recv days")
    if npa["flag"]:
        india_risk_score += 3
        signals.append(f"⚠ NPA signals: {', '.join(npa['keywords_found'][:2])}")
    if gstr_mismatch["flag"]:
        india_risk_score += 1
        signals.append("⚠ GSTR-2A/3B mismatch detected")
    if not signals:
        signals.append("✔ No India-specific risk signals detected")

    return {
        "revenue_inflation":   rev_inflation,
        "circular_trading":    circular,
        "npa_risk":            npa,
        "gstr_mismatch":       gstr_mismatch,
        "gst_number_valid":    gst_valid,
        "india_risk_score":    min(india_risk_score, 10),
        "signals":             signals,
        "has_any_flag":        india_risk_score > 0,
    }
