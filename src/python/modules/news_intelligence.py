import os
import json
import spacy
import re
import feedparser
import requests
from bs4 import BeautifulSoup
from groq import Groq
from duckduckgo_search import DDGS

# Load English NLP model. If missing, print a warning to install.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'.")
    nlp = None

def process_news(news_folder, org_name="Unknown"):
    """
    Analyze uploaded news articles in the given folder.
    Use DuckDuckGo to automatically perform secondary research.
    Use spaCy NLP for NER and keyword detection.
    Use Groq to extract structured risk summaries.
    """
    combined_text = ""
    
    # Check for text files
    if os.path.exists(news_folder):
        for filename in os.listdir(news_folder):
            file_path = os.path.join(news_folder, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_text += f.read() + "\n\n"
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # Start Web Crawling (Secondary Research)
    if org_name and org_name.lower() not in ["unknown", "n/a", ""]:
        print(f"Crawling web for: {org_name}")
        
        # 1. Google News RSS
        try:
            rss_url = f"https://news.google.com/rss/search?q={org_name.replace(' ', '+')}+financial+or+fraud+or+lawsuit"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                combined_text += f"\n[GOOGLE NEWS]\nTitle: {entry.title}\n"
        except Exception as e:
            print(f"Error fetching Google News: {e}")
            
        # 2. DuckDuckGo Search
        try:
            with DDGS() as ddgs:
                queries = [f"{org_name} lawsuit fraud default", f"{org_name} sector headwinds"]
                for query in queries:
                    results = ddgs.text(query, max_results=3)
                    if results:
                        for r in results:
                            combined_text += f"\n[WEB SEARCH '{query}']\nTitle: {r.get('title')}\nSnippet: {r.get('body')}\n"
        except Exception as e:
            print(f"Error during DuckDuckGo search: {e}")

    if not combined_text.strip():
        return __fallback()
        
    # spaCy Feature Extraction
    entities_found = set()
    legal_keywords = ["fraud", "lawsuit", "default", "bankruptcy", "litigation", "investigation", "penalty", "violation", "insolvency", "scam", "defaulted"]
    found_keywords = set()
    
    if nlp:
        doc = nlp(combined_text[:15000]) # avoid huge memory usage, fit within limits
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"]:
                entities_found.add(ent.text)
                
        # Basic keyword scan using token lemmas
        for token in doc:
            if token.lemma_.lower() in legal_keywords:
                found_keywords.add(token.lemma_.lower())
    else:
        # Fallback to simple matching if spaCy not loaded
        lower_text = combined_text.lower()
        for kw in legal_keywords:
            if kw in lower_text:
                found_keywords.add(kw)
                
    # Groq AI Risk Summarization
    groq_summary = __summarize_risks_with_groq(combined_text)
    
    # Structure Output
    is_litigation = groq_summary.get("litigation_detection", "No").lower() == "yes" or len(found_keywords) > 0
    
    return {
        "sentiment_score": groq_summary.get("sentiment_score", "Neutral"),
        "litigation_detected": is_litigation,
        "risk_keywords": list(found_keywords)
    }

def __summarize_risks_with_groq(text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"sentiment_score": "Neutral", "litigation_detection": "No", "risk_indicators": "Groq API Key missing"}
        
    client = Groq(api_key=api_key)
    model_name = 'llama-3.3-70b-versatile'
    
    prompt = """
    You are a corporate risk analyst. Read the following news articles about a company.
    Evaluate the overall tone for corporate risks.
    Provide your answer ONLY as a JSON object with the following keys, and no other text:
    {
      "sentiment_score": "Positive, Neutral, or Negative",
      "litigation_detection": "Yes/No (indicate if there are legal issues mentioned)",
      "risk_indicators": "A brief 2-sentence summary of the main risks found here."
    }
    
    News Articles:
    """
    
    try:
        truncated = text[:15000]
        full_prompt = prompt + "\n" + truncated
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
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
            
        data = json.loads(result_text)
        
        # Ensure keys
        for key in ["sentiment_score", "litigation_detection", "risk_indicators"]:
            if key not in data:
                data[key] = "N/A"
                
        return data
        
    except Exception as e:
        print(f"Error calling Groq in news analysis: {e}")
        return {
            "sentiment_score": "Unknown",
            "litigation_detection": "Unknown",
            "risk_indicators": "Error during AI analysis."
        }

def __fallback():
    return {
        "sentiment_score": "Neutral",
        "litigation_detected": False,
        "risk_keywords": []
    }
