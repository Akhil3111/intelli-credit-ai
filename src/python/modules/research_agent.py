from duckduckgo_search import DDGS
from textblob import TextBlob
import feedparser
import datetime
import logging
import os
from google import genai

logger = logging.getLogger(__name__)

# ── Gemini Translation ───────────────────────────────────────────────────────
def _translate_to_english(text: str) -> str:
    """Translates non-English text to English using Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return text
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Translate the following news article text directly to professional English. Output ONLY the translated English text, nothing else.\n\nTEXT:\n{text}"
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return resp.text.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

_SECTOR_RISK_MAP = {
    "Manufacturing":   5,
    "Infrastructure":  8,
    "Real Estate":     12,
    "NBFC":            10,
    "IT Services":     2,
    "Healthcare":      3,
    "Retail":          6,
    "Finance":         7,
    "FMCG":            4,
    "Energy":          9,
    "BFSI":            7,
    "Telecom":         6,
    "Logistics":       5,
    "Auto":            6,
    "Agriculture":     7,
}

def _safe_ddgs_search(query: str, max_results: int = 5) -> list:
    """Run a DDGS text search with error handling."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
        return results
    except Exception as e:
        logger.warning(f"DDGS search failed for '{query}': {e}")
        return []


def _safe_rss_titles(url: str, max_items: int = 5) -> list:
    """Fetch Google News RSS and return headline strings."""
    try:
        feed = feedparser.parse(url)
        return [e.title for e in feed.entries[:max_items]]
    except Exception:
        return []


def _is_english(text: str) -> bool:
    """
    Returns True if the text is predominantly English.
    Uses ASCII character ratio — non-Latin scripts (Hindi, Tamil etc.) have low ASCII ratios.
    """
    if not text:
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.80



def analyze_sentiment(text: str) -> dict:
    """TextBlob sentiment classifier."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        category, modifier = "Positive", -5
    elif polarity < -0.1:
        category, modifier = "Negative", 15
    else:
        category, modifier = "Neutral", 0
    return {"polarity": round(polarity, 2), "category": category, "risk_modifier": modifier}


# ─────────────────────────────────────────────────────────────────────────────
# 8-Query India-Specific OSINT
# ─────────────────────────────────────────────────────────────────────────────
_QUERY_TEMPLATES = [
    # (label, template, max_results, tags)
    ("Financial Performance",  "{company} financial results revenue profit outlook India",           4, []),
    ("Litigation & Legal",     "{company} promoter court case litigation NCLT NCLAT India",         4, ["litigation"]),
    ("MCA / ROC Filings",      "{company} MCA ROC filing default penalty Ministry Corporate Affairs",4, ["regulatory"]),
    ("RBI / SEBI Action",      "{company} RBI SEBI show cause notice penalty fine 2024 2025",       3, ["regulatory", "litigation"]),
    ("Rating Agency Signals",  "{company} CRISIL ICRA CARE rating downgrade watch negative",        3, ["rating"]),
    ("NPA / Bank Stress",      "{company} NPA bad loan bank write-off wilful defaulter",            3, ["litigation", "npa"]),
    ("GST / Tax Intelligence", "{company} GST fraud evasion income tax raid DGGI",                  3, ["regulatory", "gst"]),
    ("Sector Macro Headwinds", "{sector} sector outlook headwinds RBI regulation India 2024 2025",  4, ["macro"]),
]

_LITIGATION_KW = [
    "lawsuit", "fraud", "default", "bankruptcy", "litigation", "penalty",
    "violation", "insolvency", "scam", "sebi", "nclt", "nclat", "npa",
    "wilful defaulter", "mca notice", "roc notice", "court", "fir", "raid",
    "ed probe", "cbi", "enforcement directorate", "itat", "itat order",
]
_REGULATORY_KW = [
    "rbi notice", "sebi penalty", "mca filing", "roc default", "gst fraud",
    "dggi", "income tax raid", "it department", "cibil default",
]
_RATING_KW = ["downgrade", "credit watch", "negative outlook", "rating cut", "crisil d", "icra d"]


def execute_precognitive_research(company_name: str, sector: str = "") -> dict:
    """
    8-query India-specific OSINT research with litigation scoring,
    regulatory risk flag, and rating action detection.
    """
    logger.info(f"Pre-Cognitive Research started: {company_name} | sector: {sector}")

    all_raw: dict[str, list] = {}
    for label, template, max_r, _ in _QUERY_TEMPLATES:
        q = template.format(company=company_name, sector=sector or "Indian corporate")
        all_raw[label] = _safe_ddgs_search(q, max_results=max_r)

    # Google News RSS for company
    rss_titles = _safe_rss_titles(
        f"https://news.google.com/rss/search?q={company_name.replace(' ', '+')}+financial+OR+results",
        max_items=5
    )

    total_risk_modifier = 0
    analyzed_news = []
    negative_hits = 0
    litigation_detected = False
    regulatory_risk_flag = False
    rating_action = None
    litigation_score = 0          # 0-10 scale

    # Process entity-specific queries (all except "Sector Macro Headwinds" and RSS)
    entity_queries = [l for l,_,_,tags in _QUERY_TEMPLATES if "macro" not in tags]
    for label in entity_queries:
        for article in all_raw.get(label, []):
            title   = article['title']
            snippet = article['snippet']
            full_text = f"{title}. {snippet}"

            # Translate non-English articles instead of skipping
            if not _is_english(full_text):
                logger.debug(f"Translating non-English article: {title[:60]}")
                title = f"[Translated] {_translate_to_english(title)}"
                snippet = _translate_to_english(snippet)
                full_text = f"{title}. {snippet}"

            lowered = full_text.lower()
            sentiment = analyze_sentiment(full_text)
            total_risk_modifier += sentiment["risk_modifier"]
            if sentiment["category"] == "Negative":
                negative_hits += 1

            # Litigation detection + scoring
            lit_matches = sum(1 for k in _LITIGATION_KW if k in lowered)
            if lit_matches > 0:
                litigation_detected = True
                litigation_score = min(litigation_score + lit_matches * 1.2, 10)

            # Regulatory risk
            reg_matches = sum(1 for k in _REGULATORY_KW if k in lowered)
            if reg_matches > 0:
                regulatory_risk_flag = True

            # Rating action
            if rating_action is None:
                if any(k in lowered for k in _RATING_KW):
                    rating_action = "Downgrade / Negative Watch"
                elif any(k in lowered for k in ["upgrade", "positive outlook", "rating upgrade"]):
                    rating_action = "Upgrade / Positive Watch"

            analyzed_news.append({
                "title":              article["title"],
                "url":                article["url"],
                "snippet":            article["snippet"][:250],
                "sentiment_category": sentiment["category"],
                "polarity":           sentiment["polarity"],
                "query_label":        label,
            })

    # Google News RSS
    for title in rss_titles:
        if not _is_english(title):
            logger.debug(f"Translating non-English RSS title: {title[:60]}")
            title = f"[Translated] {_translate_to_english(title)}"
            
        sent = analyze_sentiment(title)
        analyzed_news.append({
            "title":              title,
            "url":                "",
            "snippet":            "",
            "sentiment_category": sent["category"],
            "polarity":           sent["polarity"],
            "query_label":        "RSS Feed",
        })
        total_risk_modifier += sent["risk_modifier"]
        if sent["category"] == "Negative":
            negative_hits += 1

    # Macro / Sector summary
    macro_results = all_raw.get("Sector Macro Headwinds", [])
    macro_snippets = [f"{r['title']}: {r['snippet'][:120]}" for r in macro_results]
    macro_insights = " | ".join(macro_snippets) if macro_snippets else "No sector-level signals retrieved."

    # Sector base risk modifier
    sector_modifier = _SECTOR_RISK_MAP.get(sector, 5)
    total_risk_modifier += sector_modifier

    # Litigation score adjustment
    if litigation_detected:
        total_risk_modifier += int(litigation_score)

    # Triangulation flag
    is_high_risk = litigation_detected or regulatory_risk_flag or (
        len(analyzed_news) > 0 and (negative_hits / max(len(analyzed_news), 1)) >= 0.5
    )

    return {
        "articles":               analyzed_news,
        "aggregate_risk_modifier": min(total_risk_modifier, 40),
        "triangulation_flag":     is_high_risk,
        "litigation_detected":    litigation_detected,
        "litigation_score":       round(litigation_score, 1),
        "regulatory_risk_flag":   regulatory_risk_flag,
        "rating_action":          rating_action or "Stable",
        "negative_hits":          negative_hits,
        "total_articles":         len(analyzed_news),
        "macro_insights":         macro_insights,
        "sector_risk_modifier":   sector_modifier,
        "timestamp":              datetime.datetime.now().isoformat(),
    }


if __name__ == "__main__":
    res = execute_precognitive_research("Reliance Industries", "Manufacturing")
    print(f"Risk Modifier: {res['aggregate_risk_modifier']} | Litigation: {res['litigation_detected']}")
    print(f"Litigation Score: {res['litigation_score']}/10 | Regulatory: {res['regulatory_risk_flag']}")
    print(f"Rating Action: {res['rating_action']}")
    for a in res["articles"][:5]:
        print(f"  [{a['sentiment_category']:8s}] [{a['query_label']:22s}] {a['title'][:72]}")
