import os
import json
from datetime import datetime
from db import get_db_connection

def update_stage(entity_id: int, stage_name: str, status: str, details: str = ""):
    """
    Updates or inserts the processing pipeline status for an entity.
    Statuses: 'pending', 'running', 'completed', 'failed'
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    # Check if a record already exists for this stage
    c.execute("SELECT id FROM processing_pipeline WHERE entity_id = ? AND stage_name = ?", (entity_id, stage_name))
    row = c.fetchone()
    
    if row:
        c.execute('''
            UPDATE processing_pipeline 
            SET status = ?, details = ?, timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (status, details, row['id']))
    else:
        c.execute('''
            INSERT INTO processing_pipeline (entity_id, stage_name, status, details) 
            VALUES (?, ?, ?, ?)
        ''', (entity_id, stage_name, status, details))
        
    conn.commit()
    conn.close()

def get_pipeline_status(entity_id: int) -> dict:
    """
    Retrieves the current status of all pipeline stages.
    """
    conn = get_db_connection()
    rows = conn.execute("SELECT stage_name, status FROM processing_pipeline WHERE entity_id = ?", (entity_id,)).fetchall()
    conn.close()
    
    # Default state if no records exist yet
    status_dict = {
        "documents_uploaded": "pending",
        "classification": "pending",
        "extraction": "pending",
        "research": "pending",
        "risk_model": "pending",
        "report_generation": "pending"
    }
    
    for row in rows:
        status_dict[row['stage_name']] = row['status']
        
    return status_dict

def update_doc_stage(document_id: int, entity_id: int, filename: str, current_stage: str, status: str, progress: int = 0, logs: str = ""):
    """
    Updates the document-level processing trace block.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, logs FROM document_processing WHERE document_id = ?", (document_id,))
    row = c.fetchone()
    
    if row:
        # Append logs for traceability
        existing_logs = row['logs'] if row['logs'] else ""
        new_logs = f"{existing_logs}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {logs}".strip()
        
        c.execute('''
            UPDATE document_processing 
            SET current_stage = ?, status = ?, progress_percentage = ?, logs = ?, timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (current_stage, status, progress, new_logs, row['id']))
    else:
        initial_log = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {logs}"
        c.execute('''
            INSERT INTO document_processing (document_id, entity_id, filename, current_stage, status, progress_percentage, logs) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (document_id, entity_id, filename, current_stage, status, progress, initial_log))
        
    conn.commit()
    conn.close()

def get_document_details(document_id: int) -> dict:
    """
    Returns the processing execution trace for a specific document modal.
    """
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM document_processing WHERE document_id = ?", (document_id,)).fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return {}
