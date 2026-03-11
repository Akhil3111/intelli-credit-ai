import spacy
import os

# Lazy loading variables
_nlp = None
_nlp_loaded = False
_category_docs = {}

def get_nlp():
    global _nlp, _nlp_loaded, _category_docs
    if not _nlp_loaded:
        try:
            _nlp = spacy.load("en_core_web_sm")
            # Precompute category docs for similarity scoring
            for cat, kws in KEYWORDS.items():
                _category_docs[cat] = _nlp(" ".join(kws))
        except (OSError, ImportError):
            _nlp = None
        _nlp_loaded = True
    return _nlp

CATEGORIES = [
    "Annual Reports",
    "ALM", # Asset Liability Management
    "Shareholding Pattern",
    "Borrowing Profile",
    "Portfolio Cuts"
]

KEYWORDS = {
    "Annual Reports": ["annual report", "financial statement", "balance sheet", "income statement", "10-k", "audit"],
    "ALM": ["asset liability", "alm", "maturity profile", "liquidity", "duration gap"],
    "Shareholding Pattern": ["shareholding", "promoter", "stakeholder", "equity distribution", "cap table"],
    "Borrowing Profile": ["borrowing", "loan profile", "debt profile", "facility", "sanction letter", "repayment"],
    "Portfolio Cuts": ["portfolio", "performance", "npa", "delinquency", "vintage", "collection"]
}

def classify_document(filename: str, text_snippet: str = "") -> dict:
    """
    Classifies a document based on its filename and an optional text snippet from the first page.
    Returns: {"category": "Target Category", "confidence": float_score_0_to_1}
    """
    lower_filename = filename.lower()
    lower_text = text_snippet.lower()
    
    scores = {cat: 0.0 for cat in CATEGORIES}
    
    # 1. Filename heuristic scoring (Heavy Weight)
    for category, kws in KEYWORDS.items():
        for kw in kws:
            if kw in lower_filename:
                scores[category] += 0.6  # High confidence if in title
                
    # 2. Text Content snippet scoring (Medium Weight)
    for category, kws in KEYWORDS.items():
        for kw in kws:
            if kw in lower_text:
                scores[category] += 0.2  # Additive points for text presence
                
    # 3. NLP Similarity scoring (if spacy available)
    nlp_model = get_nlp()
    if nlp_model and text_snippet:
        # Suppress warnings for missing word vectors in sm model by not strictly relying on them, 
        # or maybe we just ignore the warning. Spacy will print it, but it's fine.
        doc = nlp_model(text_snippet[:1000]) # Analyze first 1000 chars
        for category in CATEGORIES:
            cat_doc = _category_docs.get(category)
            if cat_doc:
                similarity = doc.similarity(cat_doc)
                scores[category] += (similarity * 0.3)
            
    # Determine winner
    best_category = max(scores, key=scores.get)
    max_score = scores[best_category]
    
    # Normalize confidence to [0, 1] pseudo-probability
    confidence = min(max_score, 0.95) 
    
    if confidence < 0.2:
        return {"category": "Unknown", "confidence": confidence}
        
    return {"category": best_category, "confidence": round(confidence, 2)}

if __name__ == '__main__':
    # Test cases
    print(classify_document("acme_annual_report_2024.pdf"))
    print(classify_document("shareholding_q3_final.csv"))
    print(classify_document("ALM_mismatch_report.xlsx"))
