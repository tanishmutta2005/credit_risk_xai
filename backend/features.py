"""
features.py — Feature engineering for the credit risk pipeline.

Derived features:
  dti            Debt-to-Income ratio       = existing_debt / annual_income
  lti            Loan-to-Income ratio       = loan_amount   / annual_income
  payment_burden Monthly payment burden     = installment   / (annual_income / 12)
  savings_ratio  Savings buffer ratio       = savings       / annual_income
  income_expense Monthly income minus expenses (absolute surplus)

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived financial ratio columns to *df*.

    All ratios are clipped to [0, 10] to handle division-by-zero edge cases
    and extreme outliers that would distort tree splits / SHAP values.

    Args:
        df: Raw (but imputed) DataFrame containing the base columns.

    Returns:
        New DataFrame with additional feature columns appended.
    """
    df = df.copy()

    # Avoid division by zero — replace 0/NaN incomes with a small epsilon
    eps = 1e-6
    income_safe = df["annual_income"].replace(0, np.nan).fillna(eps)
    monthly_income_safe = income_safe / 12

    # ── Debt-to-Income (DTI) ─────────────────────────────────────────────────
    # High DTI → higher repayment stress → higher default risk
    df["dti"] = (df["existing_debt"] / income_safe).clip(0, 10)

    # ── Loan-to-Income (LTI) ─────────────────────────────────────────────────
    # Large loans relative to income → harder to repay
    df["lti"] = (df["loan_amount"] / income_safe).clip(0, 10)

    # ── Payment Burden ───────────────────────────────────────────────────────
    # Monthly installment as a fraction of monthly income
    df["payment_burden"] = (df["installment"] / monthly_income_safe).clip(0, 5)

    # ── Savings Ratio ────────────────────────────────────────────────────────
    # Savings relative to annual income — protective factor
    df["savings_ratio"] = (df["savings"] / income_safe).clip(0, 10)

    # ── Income-Expense Surplus (absolute, monthly) ───────────────────────────
    # Positive surplus → higher capacity to service debt
    df["income_expense_surplus"] = (
        monthly_income_safe - df["monthly_expenses"].fillna(0)
    ).clip(-50_000, 50_000)

    print(
        "[features] Engineered features: dti, lti, payment_burden, "
        "savings_ratio, income_expense_surplus"
    )
    return df


ENGINEERED_COLS = [
    "dti",
    "lti",
    "payment_burden",
    "savings_ratio",
    "income_expense_surplus",
]

# Columns used downstream by the preprocessor (base + engineered)
ALL_NUMERIC_COLS = [
    "age",
    "employment_length",
    "annual_income",
    "dependents",
    "credit_history_length",
    "previous_defaults",
    "late_payments",
    "credit_utilization",
    "loan_amount",
    "loan_term",
    "interest_rate",
    "installment",
    "existing_debt",
    "monthly_expenses",
    "savings",
] + ENGINEERED_COLS


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import load_data

    df_raw = load_data()
    df_eng = engineer_features(df_raw)
    print(df_eng[ENGINEERED_COLS].describe().round(3))
