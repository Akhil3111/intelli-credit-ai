import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json
import logging
import camelot
import tabula
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path):
    """
    Extract text from a PDF using PyMuPDF.
    If a page contains no extractable text, fallback to Tesseract OCR.
    """
    try:
        doc = fitz.open(pdf_path)
        extracted_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            
            if text:
                extracted_data.append({
                    "page_number": page_num + 1,
                    "text": text
                })
            else:
                # Fallback to OCR for scanned pages
                try:
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        extracted_data.append({
                            "page_number": page_num + 1,
                            "text": ocr_text
                        })
                except Exception as ocr_e:
                    logger.error(f"OCR skipped/failed for page {page_num + 1}: {ocr_e}")
                
        # Return combined structured text
        return json.dumps(extracted_data)
        
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        raise RuntimeError(f"Failed to process PDF {pdf_path}: {str(e)}")

def split_document_into_chunks(text, chunk_size=4000):
    """
    Split the document structural JSON into manageable string chunks for LLMs.
    """
    try:
        data = json.loads(text)
        combined_text = "\n\n".join([f"--- Page {item['page_number']} ---\n{item['text']}" for item in data])
    except json.JSONDecodeError:
        combined_text = text

    chunks = []
    for i in range(0, len(combined_text), chunk_size):
        chunks.append(combined_text[i:i+chunk_size])
        
    return chunks

def extract_financial_tables(pdf_path):
    """
    Extract numeric financial tables from annual reports.
    Converts tables to structured Pandas DataFrames.
    """
    tables_data = []
    
    # Try Camelot first
    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages='1-10', flavor='stream')
        for i, table in enumerate(camelot_tables):
            if table.df.shape[0] > 1 and table.df.shape[1] > 1:
                tables_data.append(table.df)
    except Exception as e:
        logger.warning(f"Camelot table extraction failed: {e}")
        
    # Fallback to Tabula if needed
    if not tables_data:
        try:
            tabula_tables = tabula.read_pdf(pdf_path, pages='1-10', multiple_tables=True)
            if tabula_tables:
                for df in tabula_tables:
                    if not df.empty:
                        tables_data.append(df)
        except Exception as e:
            logger.warning(f"Tabula table extraction failed: {e}")

    return tables_data
