import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def analyze_gst_bank(gst_file, bank_file):
    """
    Perform reconciliation and analysis between GST returns and Bank Statements.
    - Matches monthly revenue against bank inflow.
    - Flags mismatches > 15%.
    - Detects repeated counterparties.
    - Uses PyOD Isolation Forest to detect anomalous transaction amounts.
    """
    try:
        gst_df = pd.read_csv(gst_file)
        bank_df = pd.read_csv(bank_file)
        
        # Standardize columns
        gst_df.columns = [str(c).lower().strip() for c in gst_df.columns]
        bank_df.columns = [str(c).lower().strip() for c in bank_df.columns]
        
        # Infer column names
        gst_col_date = next((c for c in gst_df.columns if 'date' in c or 'month' in c or 'period' in c), None)
        gst_col_amt = next((c for c in gst_df.columns if 'amount' in c or 'sales' in c or 'revenue' in c or 'value' in c), None)
        
        bank_col_date = next((c for c in bank_df.columns if 'date' in c or 'time' in c), None)
        bank_col_amt = next((c for c in bank_df.columns if 'credit' in c or 'amount' in c or 'inflow' in c or 'deposit' in c), None)
        bank_col_party = next((c for c in bank_df.columns if 'party' in c or 'particular' in c or 'desc' in c or 'name' in c or 'narr' in c), None)
        
        if not gst_col_date or not gst_col_amt or not bank_col_date or not bank_col_amt:
            print(f"Missing required columns in CSVs. Check formatting.")
            print(f"GST columns found: {gst_df.columns.tolist()}")
            print(f"Bank columns found: {bank_df.columns.tolist()}")
            return __fallback_results()
            
        # Date parsing
        gst_df['parsed_date'] = pd.to_datetime(gst_df[gst_col_date], errors='coerce')
        bank_df['parsed_date'] = pd.to_datetime(bank_df[bank_col_date], errors='coerce')
        
        gst_df = gst_df.dropna(subset=['parsed_date'])
        bank_df = bank_df.dropna(subset=['parsed_date'])
        
        # Year-month formatting
        gst_df['month'] = gst_df['parsed_date'].dt.to_period('M')
        bank_df['month'] = bank_df['parsed_date'].dt.to_period('M')
        
        # Clean amounts
        gst_df[gst_col_amt] = pd.to_numeric(gst_df[gst_col_amt].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        bank_df[bank_col_amt] = pd.to_numeric(bank_df[bank_col_amt].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0)
        
        # Aggregation
        monthly_gst = gst_df.groupby('month')[gst_col_amt].sum().reset_index()
        monthly_bank = bank_df.groupby('month')[bank_col_amt].sum().reset_index()
        
        merged = pd.merge(monthly_gst, monthly_bank, on='month', how='outer').fillna(0)
        merged.rename(columns={gst_col_amt: 'gst_revenue', bank_col_amt: 'bank_inflow'}, inplace=True)
        
        # Mismatch calculation
        merged['mismatch_pct'] = (abs(merged['gst_revenue'] - merged['bank_inflow']) / np.maximum(merged['gst_revenue'], merged['bank_inflow']).replace(0, 1)) * 100
        merged['flagged'] = merged['mismatch_pct'] > 15.0
        
        high_mismatch_months = merged[merged['flagged']]['month'].astype(str).tolist()
        
        # Counterparty analysis
        suspicious_parties = []
        if bank_col_party:
            counts = bank_df[bank_col_party].value_counts()
            repeated = counts[counts > 5]
            suspicious_parties = repeated.index.tolist()[:5]
            
        # Scikit-learn Anomaly Detection on Monthly Values
        anomaly_months = []
        if len(merged) >= 3: # Need a few months to train
            # Prepare features: gst_declared, bank_received, mismatch_percent
            features = merged[['gst_revenue', 'bank_inflow', 'mismatch_pct']].fillna(0)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            preds = iso_forest.fit_predict(features)
            
            # preds are 1 for inliers, -1 for anomalies
            merged['is_anomaly'] = preds == -1
            
            anomaly_data = merged[merged['is_anomaly']]
            anomaly_months = anomaly_data['month'].astype(str).tolist()
            
        # Format month for JSON serialization
        merged['month'] = merged['month'].astype(str)
        
        return {
            "mismatch_table": merged.to_dict(orient='records'),
            "high_mismatch_months": high_mismatch_months,
            "suspicious_counterparties": suspicious_parties,
            "anomaly_months": anomaly_months,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in GST Bank Analysis: {e}")
        return __fallback_results()

def __fallback_results():
    return {
        "mismatch_table": [],
        "high_mismatch_months": [],
        "suspicious_counterparties": [],
        "anomaly_months": [],
        "status": "error"
    }
