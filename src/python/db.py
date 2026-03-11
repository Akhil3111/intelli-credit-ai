import sqlite3
import os

DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data', 'intellicredit.db')

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Entities table
    c.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cin TEXT UNIQUE NOT NULL,
            pan TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT,
            subsector TEXT,
            turnover REAL,
            incorporation_date TEXT,
            promoters TEXT,
            loan_type TEXT,
            loan_amount REAL,
            loan_tenure INTEGER,
            interest_rate REAL,
            purpose TEXT,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    
    # Create Documents table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            category TEXT,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Uploaded',
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        )
    ''')

    # Create Processing Pipeline Tracker table
    c.execute('''
        CREATE TABLE IF NOT EXISTS processing_pipeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            stage_name TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        )
    ''')

    # Create Document Processing table
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            entity_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            current_stage TEXT NOT NULL,
            status TEXT NOT NULL,
            progress_percentage INTEGER DEFAULT 0,
            logs TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
