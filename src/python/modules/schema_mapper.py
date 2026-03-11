"""
schema_mapper.py  –  Field extraction with 15+ synonyms per metric,
                     unit normalisation (lakhs / millions / billions → Crores),
                     and table-first confidence scoring.
"""

import re
import json
import os

# ── Standard schema ───────────────────────────────────────────────────────────
STANDARD_SCHEMA = {
    # Core financials
    "Revenue":             0.0,
    "Net Profit":          0.0,
    "Total Debt":          0.0,
    "Total Assets":        0.0,
    "Operating Cash Flow": 0.0,
    "Current Assets":      0.0,
    "Current Liabilities": 0.0,
    "Interest Expense":    0.0,
    "EBIT":                0.0,
    "EBITDA":              0.0,
    "Net Worth":           0.0,
    # India-specific fields (Gap 3)
    "GST Turnover":        0.0,   # GSTR-3B declared turnover
    "Tax Paid (ITR)":      0.0,   # Income tax paid per ITR
    "Trade Receivables":   0.0,   # Debtor days cross-check
    "Trade Payables":      0.0,   # Creditor days cross-check
    "Promoter Shareholding %": 0.0,
    "CIBIL / CMR Score":  0.0,
    "Fixed Assets":        0.0,
}

# ── Unit conversion factors (everything → Crores) ─────────────────────────────
# We detect the unit suffix after the number and normalise accordingly
_UNIT_MAP = {
    "billion":  100.0,     # 1 billion INR ≈ 100 Cr
    "billions": 100.0,
    "bn":       100.0,
    "million":  0.1,       # 1 million INR ≈ 0.1 Cr
    "millions": 0.1,
    "mn":       0.1,
    "m":        0.1,
    "lakh":     0.01,      # 1 lakh INR = 0.01 Cr
    "lakhs":    0.01,
    "lac":      0.01,
    "lacs":     0.01,
    "crore":    1.0,
    "crores":   1.0,
    "cr":       1.0,
    "crs":      1.0,
}

# ── Regex patterns (15+ synonyms per field) ───────────────────────────────────
PATTERNS: dict[str, list[str]] = {
    "Revenue": [
        r"(?i)total\s+(?:revenue|income|receipts?)[^0-9\-₹$\(]{0,40}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)revenue\s+from\s+(?:operations?|sales?)[^0-9\-₹$\(]{0,40}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)net\s+(?:sales?|revenue)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:gross|operating)\s+revenue[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)turnover[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)income\s+from\s+(?:services?|products?|operations?)[^0-9\-₹$\(]{0,40}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:consolidated|standalone)\s+(?:revenue|income)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:sales?)\s*&\s*(?:income|revenue)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Net Profit": [
        r"(?i)(?:profit|loss)\s+(?:after\s+tax|for\s+the\s+(?:year|period))[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)net\s+(?:profit|income|earnings?)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)pat\b[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)profit\s+after\s+(?:income\s+)?tax[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)net\s+(?:surplus|deficit)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)total\s+comprehensive\s+income[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:attributable\s+to\s+)?(?:shareholders?|owners?)\s+(?:of\s+the\s+company)?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Total Debt": [
        r"(?i)total\s+(?:borrowings?|debt|indebtedness)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)long[\s\-]term\s+borrowings?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:external|financial)\s+(?:debt|liabilities?)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)outstanding\s+(?:debt|borrowings?|loans?)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)bank\s+(?:loans?|borrowings?|overdraft)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)debentures?\s+and\s+bonds?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)term\s+loans?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Total Assets": [
        r"(?i)total\s+assets?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)balance\s+sheet\s+total[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)total\s+(?:tangible|net)\s+assets?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)total\s+(?:equity\s+and\s+liabilities?)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)gross\s+assets?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Operating Cash Flow": [
        r"(?i)(?:net\s+)?cash\s+(?:generated\s+)?from\s+operating[^0-9\-₹$\(]{0,40}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)operating\s+(?:activities?|cash\s+flow)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)cash\s+flow\s+from\s+operations?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)operating\s+cash\s+(?:flow|generation)[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)net\s+operating\s+activities?[^0-9\-₹$\(]{0,30}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Current Assets": [
        r"(?i)(?:total\s+)?current\s+assets?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:short[\s\-]term|liquid)\s+assets?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)inventories?\s+and\s+(?:receivables?|debtors?)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Current Liabilities": [
        r"(?i)(?:total\s+)?current\s+liabilit(?:y|ies)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)short[\s\-]term\s+(?:liabilit(?:y|ies)|borrowings?)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)payables?\s+and\s+accruals?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Interest Expense": [
        r"(?i)(?:finance|interest)\s+costs?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)interest\s+(?:expense|paid|charges?|on\s+borrowings?)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)borrowing\s+costs?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)bank\s+(?:charges?|interest)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)interest\s+liability[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "EBIT": [
        r"(?i)ebit\b[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)earnings?\s+before\s+interest\s+(?:and\s+tax|&\s+tax)[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)operating\s+(?:income|earnings?|profit)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)profit\s+before\s+(?:interest|finance)\s+(?:and\s+tax)?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:pbit|pbtd)\b[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "EBITDA": [
        r"(?i)ebitda\b[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)earnings?\s+before\s+(?:interest)[,\s]+(?:tax(?:es)?)[,\s]+dep[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)operating\s+(?:ebitda|cashflow|cash\s+flow)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)ebitda\s+margin[^0-9\-₹$\(]{0,10}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)adjusted\s+ebitda[^0-9\-₹$\(]{0,20}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
    "Net Worth": [
        r"(?i)(?:total\s+)?net\s+worth[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)shareholders?[\s'\u2019]?\s*(?:equity|funds?)[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)(?:total\s+)?(?:owners?|stockholders?)\s*equity[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)book\s+value\s+of\s+equity[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)paid[\s\-]up\s+(?:capital\s+and\s+)?reserves?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
        r"(?i)capital\s+and\s+(?:free\s+)?reserves?[^0-9\-₹$\(]{0,25}[\-₹$]?\s*\(?(\d[\d,]*\.?\d*)\)?",
    ],
}

# ── Value normalisation helpers ───────────────────────────────────────────────
_NUM_RE = re.compile(
    r"(?i)[\-₹$Rs\.]*\s*\(?(\d[\d,]*\.?\d*)\)?"
    r"\s*(billion|billions|bn|million|millions|mn|lakh|lakhs|lac|lacs|crore|crores|cr|crs|m\b)?"
)


def _parse_value(raw: str) -> float:
    """Extract numeric value and apply unit conversion → Crores."""
    raw = raw.replace(",", "")
    # negative bracket notation
    negative = raw.strip().startswith("(") and raw.strip().endswith(")")
    raw = raw.strip().strip("()₹$Rs. ")
    try:
        num = float(raw.split()[0])
    except (ValueError, IndexError):
        return 0.0
    # Check for unit word inline
    lower = raw.lower()
    factor = 1.0
    for unit, f in _UNIT_MAP.items():
        if re.search(r'\b' + unit + r'\b', lower):
            factor = f
            break
    result = num * factor
    return -result if negative else result


def _clean_value(val_str: str) -> float:
    return _parse_value(val_str)


# ── Core extraction ───────────────────────────────────────────────────────────

def _detect_unit_context(text: str) -> float:
    """
    Look for a global unit declaration like 'all figures in lakhs / crores'
    and return the corresponding normalisation factor.
    """
    m = re.search(
        r"(?i)(?:amounts?|figures?|values?|all\s+figures?)\s+(?:are\s+)?(?:in|stated\s+in)\s+"
        r"(billion|lakh|lac|million|crore|rupee)",
        text,
    )
    if m:
        return _UNIT_MAP.get(m.group(1).lower(), 1.0)
    return 1.0   # default: already in crores


def process_extracted_data(text_content: str, tables_list: list) -> dict:
    """
    Primary entry point.
    1. Scan structured tables first (higher confidence 0.93)
    2. Fall back to text scanning (0.76)
    3. Apply global unit factor if found
    Returns { field: {value, confidence, source} }
    """
    mapped: dict = {}
    global_factor = _detect_unit_context(text_content)

    # ── 1. Table scanning ──────────────────────────────────────────────────────
    for tidx, table in enumerate(tables_list):
        for row in table:
            row_str = " ".join(str(v) for v in row.values() if v)
            for field, patterns in PATTERNS.items():
                if field in mapped and mapped[field]["confidence"] >= 0.90:
                    continue
                for pat in patterns:
                    m = re.search(pat, row_str)
                    if m:
                        val = _clean_value(m.group(1)) * global_factor
                        if val != 0.0:
                            mapped[field] = {
                                "value":      round(val, 2),
                                "confidence": 0.93,
                                "source":     f"Table #{tidx + 1}",
                            }
                        break

    # ── 2. Text scanning ───────────────────────────────────────────────────────
    for field, patterns in PATTERNS.items():
        if field in mapped and mapped[field]["confidence"] >= 0.90:
            continue
        for pat in patterns:
            m = re.search(pat, text_content)
            if m:
                val = _clean_value(m.group(1)) * global_factor
                if val != 0.0:
                    mapped[field] = {
                        "value":      round(val, 2),
                        "confidence": 0.76,
                        "source":     "Text Extraction",
                    }
                break

    # ── 3. Fill missing with zeros ─────────────────────────────────────────────
    for field in STANDARD_SCHEMA:
        if field not in mapped:
            mapped[field] = {"value": 0.0, "confidence": 0.0, "source": "Not Found"}

    return mapped


def save_to_processed(entity_id: int, finalized_schema: dict):
    base_dir      = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed", str(entity_id))
    os.makedirs(processed_dir, exist_ok=True)
    fp = os.path.join(processed_dir, "entity_financials.json")
    with open(fp, "w") as f:
        json.dump(finalized_schema, f, indent=4)
    return fp
