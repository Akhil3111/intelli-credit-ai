import os
import json
import re
import time
from groq import Groq
from modules.document_processor import split_document_into_chunks

def extract_regex_fallbacks(text):
    """Fallback regex extraction for known financial patterns."""
    data = {}
    
    patterns = {
        "Net Profit": r"(?i)Net Profit.*?([\d,\.]+)",
        "Profit After Tax": r"(?i)Profit After Tax.*?([\d,\.]+)",
        "ROA": r"(?i)Return on Assets.*?([\d,\.]+)",
        "Capital Adequacy Ratio": r"(?i)Capital Adequacy Ratio.*?([\d,\.]+)",
        "Net NPA": r"(?i)Net NPA.*?([\d,\.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = match.group(1)
            
    return data

def extract_financials(text, tables_data=None):
    """
    Use Groq API to analyze the annual report text and extract structured financial metrics.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set it to use the Groq API.")
        
    client = Groq(api_key=api_key)
    
    # Using llama-3.3-70b-versatile for fast and cost-effective extraction
    model_name = 'llama-3.3-70b-versatile'
    
    prompt = """
    You are an expert financial analyst. Please extract the following financial metrics from the provided annual report text.
    If a metric is not found or cannot be determined, return 'N/A'.
    
    Return ONLY a valid JSON object in the EXACT format below, with no additional markdown formatting, comments, or extra text:
    {
      "Organization Name": "...",
      "Net Profit": "...",
      "Profit After Tax": "...",
      "ROA": "...",
      "Capital Adequacy Ratio": "...",
      "Net NPA": "..."
    }
    
    Annual Report Text (truncating to fit context window if extremely long):
    """
    
    try:
        # Get raw combined text for regex
        if text.startswith("[{"):
            parsed_data = json.loads(text)
            combined_text = "\n".join([item['text'] for item in parsed_data])
        else:
             combined_text = text
    except:
        combined_text = text

    aggregated_data = {
        "Organization Name": "Unknown",
        "Net Profit": "N/A",
        "Profit After Tax": "N/A",
        "ROA": "N/A",
        "Capital Adequacy Ratio": "N/A",
        "Net NPA": "N/A"
    }

    # 1. Regex Fallback Baseline
    aggregated_data.update(extract_regex_fallbacks(combined_text))

    # Incorporate tables_data if present
    table_context = ""
    if tables_data:
        table_context = "Extracted Tables from Document:\n"
        for i, df in enumerate(tables_data):
            table_context += f"Table {i+1}:\n{df.head(10).to_csv(index=False)}\n"
    
    # 2. Semantic Document Vector Search for relevant chunks
    from modules.document_indexer import get_indexer
    indexer = get_indexer()
    indexer.build_index(text, chunk_size=4000)
    
    queries = ["Net Profit Revenue", "Return on Assets ROA", "Capital Adequacy Ratio", "Non Performing Assets NPA"]
    relevant_chunks_set = set()
    for q in queries:
        results = indexer.search(q, top_k=1)
        for res in results:
            relevant_chunks_set.add(res)
            
    # Fallback if search returns nothing
    all_chunks = split_document_into_chunks(text, chunk_size=4000)
    relevant_chunks = list(relevant_chunks_set) if relevant_chunks_set else all_chunks[:3]
    chunks_to_process = relevant_chunks[:3]
    
    prompt = """
    You are an expert financial analyst. Extract the following financial metrics from the provided annual report text chunk.
    If a metric is not found or cannot be determined, return 'N/A'.
    
    Return ONLY a valid JSON object in the EXACT format below, with no additional markdown formatting, comments, or extra text:
    {
      "Organization Name": "...",
      "Net Profit": "...",
      "Profit After Tax": "...",
      "ROA": "...",
      "Capital Adequacy Ratio": "...",
      "Net NPA": "..."
    }
    """
    
    for i, chunk_text in enumerate(chunks_to_process):
        try:
            full_prompt = prompt + "\nChunk Text:\n" + chunk_text
            if i == 0 and table_context:
                full_prompt += "\n" + table_context

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith('```'):
                result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)
                
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result_text = match.group(0)
                
            chunk_data = json.loads(result_text)
            
            # Aggregate results
            for key in aggregated_data.keys():
                if aggregated_data[key] in ["N/A", "Unknown"] and key in chunk_data and str(chunk_data[key]).strip() not in ["N/A", "Unknown", ""]:
                    aggregated_data[key] = chunk_data[key]
                    
            if i < len(chunks_to_process) - 1:
                time.sleep(1) # Prevent rate limits
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue

    return aggregated_data
