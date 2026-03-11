def calculate_financial_ratios(financials: dict) -> dict:
    """
    Computes rigorous financial features from extracted schema.
    Protects against divide-by-zero and handles blank entries gracefully.
    """
    revenue           = float(financials.get('Revenue', 0.0) or 0)
    net_profit        = float(financials.get('Net Profit', 0.0) or 0)
    total_debt        = float(financials.get('Total Debt', 0.0) or 0)
    assets            = float(financials.get('Total Assets', 0.0) or 0)
    ocf               = float(financials.get('Operating Cash Flow', 0.0) or 0)
    current_assets    = float(financials.get('Current Assets', 0.0) or 0)
    current_liab      = float(financials.get('Current Liabilities', 0.0) or 0)
    interest_expense  = float(financials.get('Interest Expense', 0.0) or 0)
    ebit              = float(financials.get('EBIT', net_profit) or net_profit)
    net_worth         = float(financials.get('Net Worth', max(assets - total_debt, 0)) or 0)

    # Safe denominators
    revenue_d      = revenue      if revenue      != 0 else 0.01
    assets_d       = assets       if assets       != 0 else 0.01
    total_debt_d   = total_debt   if total_debt   != 0 else 0.01
    current_liab_d = current_liab if current_liab != 0 else 0.01
    interest_d     = interest_expense if interest_expense != 0 else 0.01
    net_worth_d    = net_worth    if net_worth    != 0 else 0.01

    debt_service = total_debt * 0.15  # approximate annual principal + interest heuristic

    ratios = {
        # --- Core ---
        "debt_equity_ratio":       round(total_debt  / net_worth_d,      2),
        "profit_margin":           round(net_profit  / revenue_d,         4),
        "cashflow_debt_coverage":  round(ocf          / total_debt_d,      2),
        "revenue_acceleration":    1.05,   # placeholder (single-year data)
        # --- Extended ---
        "roa":                     round(net_profit  / assets_d,          4),
        "current_ratio":           round(current_assets / current_liab_d, 2),
        "interest_coverage":       round(ebit        / interest_d,         2),
        "dscr":                    round(ocf          / max(debt_service, 0.01), 2),
    }

    return ratios
