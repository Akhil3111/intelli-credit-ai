import os
from groq import Groq

def compute_risk_score(financials, gst_bank_results, news_insights, officer_notes=""):
    """
    Implement a credit risk scoring model based on the Five Cs:
    Character (25), Capacity (20), Capital (15), Collateral (15), Conditions (25).
    """
    
    # 1. CHARACTER (Max 25)
    # Based on news sentiment, litigation, and repeated counterparties
    character_score = 25
    if news_insights:
        sentiment = str(news_insights.get("sentiment_score", "")).lower()
        litigation_flag = news_insights.get("litigation_detected", False)
        
        if "negative" in sentiment:
            character_score -= 10
        elif "neutral" in sentiment:
            character_score -= 3
            
        if litigation_flag:
            character_score -= 15
            
    if gst_bank_results and gst_bank_results.get("suspicious_parties"):
        # Penalty for too many repeated unknown counterparties (potential shell companies)
        character_score -= min(5, len(gst_bank_results["suspicious_parties"]))
        
    character_score = max(0, character_score)
    
    # 2. CAPACITY (Max 20)
    # Based on GST-Bank mismatch and Profitability
    capacity_score = 20
    if gst_bank_results:
        high_mismatch = len(gst_bank_results.get("high_mismatch_months", []))
        capacity_score -= (high_mismatch * 2) # -2 for every month with >15% mismatch
        
        anomaly_months = len(gst_bank_results.get("anomaly_months", []))
        if anomaly_months > 2:
            capacity_score -= 10
    
    net_profit = _parse_val(financials.get("Net Profit"))
    if net_profit is not None:
        if net_profit <= 0: capacity_score -= 10
        elif net_profit < 10: capacity_score -= 5 # Arbitrary small threshold logic
    else:
        capacity_score -= 10 # Heavy penalty for lack of profit clarity
        
    capacity_score = max(0, capacity_score)
    
    # 3. CAPITAL (Max 15)
    # Based on Capital Adequacy Ratio (CAR) and Net NPA
    capital_score = 15
    car_val = _parse_val(financials.get("Capital Adequacy Ratio"))
    npa_val = _parse_val(financials.get("Net NPA"))
    
    if car_val is not None:
        if car_val < 9: capital_score -= 10 # Below Basel norm roughly
        elif car_val < 12: capital_score -= 5
    else:
        capital_score -= 5
        
    if npa_val is not None:
        if npa_val > 5: capital_score -= 10 # High NPA
        elif npa_val > 2: capital_score -= 5
    else:
        capital_score -= 5
        
    capital_score = max(0, capital_score)
    
    # 4. COLLATERAL (Max 15)
    # Since collateral is often not explicitly clear in basic annual reports unless specified, 
    # we assign a moderate default or estimate based on ROA and total assets context.
    collateral_score = 15
    roa_val = _parse_val(financials.get("ROA"))
    if roa_val is not None:
        if roa_val < 0.5:
            # Low return on assets implies assets might be heavily encumbered or unproductive
            collateral_score -= 5
    else:
        collateral_score -= 5
        
    # 5. CONDITIONS (Max 25)
    # Industry conditions, anomaly presence
    conditions_score = 25
    if gst_bank_results and gst_bank_results.get("anomaly_months"):
        # Volatility or anomalies in bank statements
        num_anomalies = len(gst_bank_results["anomaly_months"])
        conditions_score -= min(15, num_anomalies * 2)
        
    # Apply Officer Notes dynamically via Gemini
    if officer_notes and officer_notes.strip():
        adjustment = __analyze_officer_notes(officer_notes)
        conditions_score += adjustment
        
    conditions_score = max(0, conditions_score)
    
    # CALCULATE TOTAL
    total_score = character_score + capacity_score + capital_score + collateral_score + conditions_score
    
    # DECISION LOGIC
    if total_score > 70:
        decision = "Approve"
        interest_rate = "8.5% - 10.0%"
    elif total_score >= 50:
        decision = "Review"
        interest_rate = "10.5% - 13.0%"
    else:
        decision = "Reject"
        interest_rate = "N/A"
        
    # Calculate Limit
    net_profit = _parse_val(financials.get("Net Profit"))
    if decision == "Reject":
        limit = "₹0"
    elif net_profit and net_profit > 0:
        multiplier = 4.0 if decision == "Approve" else 2.5
        limit = f"₹{round(net_profit * multiplier, 2)} Cr"
    else:
        limit = "Requires Collateral Assessment"

    # Generate Explainable Decision Rationale
    explanation = __generate_rationale(total_score, decision, limit, financials, gst_bank_results, news_insights, officer_notes)
        
    return {
        "character_score": round(character_score, 2),
        "capacity_score": round(capacity_score, 2),
        "capital_score": round(capital_score, 2),
        "collateral_score": round(collateral_score, 2),
        "conditions_score": round(conditions_score, 2),
        "total_score": round(total_score, 2),
        "decision": decision,
        "interest_rate": interest_rate,
        "limit": limit,
        "explanation": explanation
    }

def _parse_val(val_str):
    if not val_str or str(val_str).upper() == "N/A":
        return None
    try:
        import re
        # Remove non-numeric characters except decimals and minus
        clean_str = re.sub(r'[^\d.-]', '', str(val_str))
        if clean_str:
            return float(clean_str)
        return None
    except:
        return None

def __analyze_officer_notes(notes):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return 0
    client = Groq(api_key=api_key)
    prompt = f"""You are a scoring assistant. Read these Credit Officer qualitative notes: "{notes}"
Based on these notes, output ONLY a single integer representing a risk score adjustment from -20 (severe risk) to +10 (strong positive mitigate). Do not include any text, just the integer.
"""
    try:
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import re
        val = re.sub(r'[^\d-]', '', response.choices[0].message.content)
        return int(val)
    except:
        return 0

def __generate_rationale(score, decision, limit, financials, gst_bank_results, news_insights, notes):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return f"{decision} based on score {score}."
    client = Groq(api_key=api_key)
    
    news_summary = ""
    if news_insights:
        news_summary = f"Sentiment: {news_insights.get('sentiment_score', 'Neutral')}. Keywords: {', '.join(news_insights.get('risk_keywords', []))}."
        if news_insights.get('litigation_detected'):
            news_summary += " Litigation detected."
            
    mismatch = "Yes" if gst_bank_results and gst_bank_results.get('high_mismatch_months') else "No"
    anomalies = len(gst_bank_results.get('anomaly_months', [])) if gst_bank_results else 0
    
    prompt = f"""You are an automated Credit Manager. Explain the loan {decision} in exactly two clear, professional sentences. 
Data: Score is {score}/100. Limit is {limit}. 
Officer Notes provided: "{notes}". 
High GST/Bank Mismatch: {mismatch}. 
Machine Learning Anomalies Detected: {anomalies} months.
News/Litigation Risks: {news_summary}
Make sure to explain WHY it was approved, reviewed, or rejected.
"""
    try:
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"{decision} based on score {score}."
