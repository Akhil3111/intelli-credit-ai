import os
from groq import Groq
import json
import re

def generate_cam(risk_results, financials, gst_bank_results, news_insights, full_text):
    """
    Use Gemini to generate a structured JSON Credit Appraisal Memo for tabular display.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return '{"error": "GROQ_API_KEY missing. AI unable to generate CAM."}'
        
    client = Groq(api_key=api_key)
    model_name = 'llama-3.3-70b-versatile'
    
    prompt = f"""
    You are an expert Credit Analyst. Please extract and infer information to populate the following Credit Appraisal Memorandum.
    Use the provided data (financials, GST analysis, news intelligence, risk scoring, and officer notes) to complete the fields.
    If exact data is missing, make a highly plausible, professional estimate based on the context, or write "Not available in provided docs".
    
    Do NOT include markdown formatting. Return ONLY a valid JSON object matching this exact schema:
    
    {{
      "applicant_name": "Extracted Organization Name",
      "borrower_overview": {{
        "description": "Brief summary of the borrower's business",
        "industry": "Industry sector",
        "key_activities": "Main business activities"
      }},
      "financial_performance": {{
        "net_profit": "Extracted Net Profit",
        "roa": "Return on Assets",
        "npa": "Net NPA",
        "capital_adequacy": "Capital Adequacy Ratio"
      }},
      "revenue_validation": {{
        "gst_bank_mismatch_status": "Summary of mismatch (>15% months)",
        "suspicious_counterparties": "Number/details of suspicious parties",
        "ml_anomalies": "Summary of IsolationForest anomalies"
      }},
      "external_intelligence": {{
        "news_sentiment": "Positive/Neutral/Negative",
        "litigation_found": "Yes/No with brief details",
        "key_risks": "Summary of news risk keywords"
      }},
      "risk_assessment_five_cs": {{
        "character": "Score and brief justification",
        "capacity": "Score and brief justification",
        "capital": "Score and brief justification",
        "collateral": "Score and brief justification",
        "conditions": "Score and brief justification"
      }},
      "credit_recommendation": {{
        "decision": "Approve/Review/Reject",
        "limit": "Proposed Limit",
        "interest_rate": "Proposed Rate",
        "rationale": "The exact AI generated rationale"
      }}
    }}
    
    DATA PROVIDED:
    
    Executive Text Context (Truncated):
    {full_text[:15000]}
    
    Financial Metrics:
    {json.dumps(financials, indent=2)}
    
    GST vs Bank Analysis:
    - High Mismatch Months (>15% variance): {', '.join(gst_bank_results.get('high_mismatch_months', [])) or 'None recorded.'}
    - Suspicious Repeated Counterparties Formally Identified: {len(gst_bank_results.get('suspicious_parties', []))}
    - Bank Statement Anomalous Transactions flagged by ML: {len(gst_bank_results.get('anomalies', []))}
    
    External Intelligence Findings:
    {json.dumps(news_insights, indent=2)}
    
    Total Risk Engine Decision Outputs:
    {json.dumps(risk_results, indent=2)}
    
    Please return ONLY the raw JSON string matching the specified schema.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        # Parse output to ensure strictly JSON
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith('```'):
            result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
        
        # Additional fallback regex if there's text before/after JSON
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            result_text = match.group(0)
            
        data = json.loads(result_text)
        return json.dumps(data)
    except Exception as e:
        print(f"Error communicating with AI CAM generator: {str(e)}")
        # Fallback JSON structure if generation fails
        fallback = {
            "applicant_name": financials.get("Organization Name", "Unknown"),
            "borrower_overview": {"description": "Error generating.", "industry": "N/A", "key_activities": "N/A"},
            "financial_performance": {"net_profit": financials.get("Net Profit", "N/A"), "roa": financials.get("ROA", "N/A"), "npa": "N/A", "capital_adequacy": "N/A"},
            "revenue_validation": {"gst_bank_mismatch_status": "N/A", "suspicious_counterparties": "N/A", "ml_anomalies": "N/A"},
            "external_intelligence": {"news_sentiment": "N/A", "litigation_found": "N/A", "key_risks": "N/A"},
            "risk_assessment_five_cs": {"character": "N/A", "capacity": "N/A", "capital": "N/A", "collateral": "N/A", "conditions": "N/A"},
            "credit_recommendation": {"decision": risk_results.get("decision", "Error"), "limit": "N/A", "interest_rate": "N/A", "rationale": "Quota exceeded"}
        }
        return json.dumps(fallback)
