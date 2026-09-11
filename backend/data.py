"""
data.py — Data loading, synthetic data generation, and preprocessing.

Responsibilities:
  - load_data(): Try loading a LendingClub CSV; fall back to synthetic generation.
  - generate_synthetic_data(): Create a realistic 5 000-row dataset.
  - build_preprocessor(): Return a sklearn ColumnTransformer (imputation + encoding).
  - preprocess(): Apply the full preprocessing pipeline.

DISCLAIMER: All outputs are for educational/research purposes only.
Do NOT use for real lending decisions.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
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
]

CATEGORICAL_COLS = [
    "employment_status",
    "education",
    "loan_purpose",
]

TARGET_COL = "loan_status"
POSITIVE_LABEL = "Default"  # minority class (1)
NEGATIVE_LABEL = "Fully Paid"  # majority class (0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str | None = None, n_synthetic: int = 5_000) -> pd.DataFrame:
    """
    Load a LendingClub-style CSV from *csv_path*, or fall back to synthetic data.

    Args:
        csv_path:     Path to an existing CSV file. None → synthetic.
        n_synthetic:  Number of synthetic rows to generate when needed.

    Returns:
        Raw DataFrame with the expected column set.
    """
    if csv_path and Path(csv_path).exists():
        print(f"[data] Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        df = _align_columns(df)
        return df

    print(f"[data] CSV not found -- generating {n_synthetic} synthetic rows.")
    return generate_synthetic_data(n=n_synthetic)


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common LendingClub column names to the internal schema.
    Keeps only the columns we need and fills any missing ones with NaN.
    """
    rename_map = {
        "loan_amnt": "loan_amount",
        "term": "loan_term",
        "int_rate": "interest_rate",
        "emp_length": "employment_length",
        "annual_inc": "annual_income",
        "purpose": "loan_purpose",
        "dti": "existing_debt",           # approximate mapping
        "loan_status": "loan_status",
    }
    df = df.rename(columns=rename_map)
    all_cols = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[all_cols].copy()


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 5_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit dataset.

    Class balance: ~25 % Default, ~75 % Fully Paid (realistic imbalance).
    """
    rng = np.random.default_rng(random_state)

    # ── Employment & demographics ────────────────────────────────────────────
    age = rng.integers(20, 70, n).astype(float)
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Part-Time"],
        n,
        p=[0.55, 0.20, 0.10, 0.15],
    )
    employment_length = rng.uniform(0, 35, n)
    employment_length[employment_status == "Unemployed"] = 0.0
    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD", "None"],
        n,
        p=[0.30, 0.35, 0.20, 0.05, 0.10],
    )
    dependents = rng.integers(0, 6, n).astype(float)

    # ── Income & financial position ──────────────────────────────────────────
    # Income correlated with education & employment
    base_income = rng.lognormal(mean=10.8, sigma=0.6, size=n)
    edu_boost = np.where(education == "PhD", 1.5,
                np.where(education == "Master", 1.3,
                np.where(education == "Bachelor", 1.1, 1.0)))
    annual_income = np.clip(base_income * edu_boost, 15_000, 500_000)

    savings = np.clip(rng.lognormal(8, 1.2, n), 0, 200_000)
    monthly_expenses = np.clip(annual_income / 12 * rng.uniform(0.3, 0.8, n), 500, 20_000)
    existing_debt = np.clip(rng.exponential(scale=annual_income * 0.4, size=n), 0, 300_000)

    # ── Credit history ───────────────────────────────────────────────────────
    credit_history_length = rng.uniform(1, 30, n)
    previous_defaults = rng.choice([0, 1, 2, 3], n, p=[0.65, 0.20, 0.10, 0.05])
    late_payments = rng.choice(range(11), n, p=[0.45, 0.20, 0.12, 0.08, 0.05,
                                                  0.04, 0.02, 0.02, 0.01, 0.005, 0.005])
    credit_utilization = np.clip(rng.beta(2, 3, n), 0.0, 1.0)

    # ── Loan details ─────────────────────────────────────────────────────────
    loan_amount = np.clip(rng.lognormal(9.5, 0.7, n), 500, 40_000)
    loan_term = rng.choice([12, 24, 36, 48, 60], n, p=[0.05, 0.10, 0.45, 0.15, 0.25])
    interest_rate = np.clip(rng.normal(12, 5, n), 3, 30)
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "medical", "car", "vacation", "small_business", "other"],
        n,
        p=[0.35, 0.20, 0.15, 0.08, 0.08, 0.05, 0.05, 0.04],
    )
    # Monthly installment approximation
    monthly_rate = interest_rate / 100 / 12
    installment = np.where(
        monthly_rate > 0,
        loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** (-loan_term)),
        loan_amount / loan_term,
    )

    # ── Target variable (Default/Fully Paid) ─────────────────────────────────
    # Logistic function over risk factors → default probability
    # Coefficients scaled to produce realistic ~0.80 ROC-AUC with ~25% default rate
    dti_raw   = existing_debt / (annual_income + 1)
    lti_raw   = loan_amount   / (annual_income + 1)
    pb_raw    = installment   / (annual_income / 12 + 1)
    sr_raw    = savings       / (annual_income + 1)

    risk_score = (
        -2.5                                            # intercept (controls base rate)
        + 2.5  * np.clip(dti_raw, 0, 5)                # DTI  — strong predictor
        + 1.8  * np.clip(lti_raw, 0, 5)                # LTI
        + 1.5  * credit_utilization                     # util — strong predictor
        + 2.0  * np.clip(pb_raw,  0, 3)                # payment burden
        + 1.2  * previous_defaults                      # defaults history
        + 0.8  * late_payments / 10                     # late payments
        - 1.5  * np.clip(sr_raw, 0, 3)                 # savings (protective)
        - 0.8  * (credit_history_length / 30)           # long history (protective)
        - 0.5  * (employment_length / 35)               # stable employment (protective)
        + rng.normal(0, 0.5, n)                         # reduced noise
    )
    default_prob = 1 / (1 + np.exp(-risk_score))
    loan_status = np.where(
        rng.random(n) < default_prob, POSITIVE_LABEL, NEGATIVE_LABEL
    )

    # ── Introduce realistic missingness ──────────────────────────────────────
    def add_missing(arr, rate=0.03):
        arr = arr.copy().astype(object)
        mask = rng.random(n) < rate
        arr[mask] = np.nan
        return arr

    df = pd.DataFrame(
        {
            "age": add_missing(age),
            "employment_status": employment_status,
            "employment_length": add_missing(employment_length),
            "annual_income": add_missing(annual_income),
            "education": education,
            "dependents": add_missing(dependents),
            "credit_history_length": add_missing(credit_history_length),
            "previous_defaults": add_missing(previous_defaults),
            "late_payments": add_missing(late_payments.astype(float)),
            "credit_utilization": add_missing(credit_utilization),
            "loan_amount": add_missing(loan_amount),
            "loan_term": add_missing(loan_term.astype(float)),
            "interest_rate": add_missing(interest_rate),
            "loan_purpose": loan_purpose,
            "installment": add_missing(installment),
            "existing_debt": add_missing(existing_debt),
            "monthly_expenses": add_missing(monthly_expenses),
            "savings": add_missing(savings),
            "loan_status": loan_status,
        }
    )

    # Convert numeric cols back (object→float after missingness injection)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[data] Synthetic dataset: {n} rows | "
          f"Default rate: {(df[TARGET_COL] == POSITIVE_LABEL).mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str] = NUMERIC_COLS,
    categorical_cols: list[str] = CATEGORICAL_COLS,
    scale_numerics: bool = False,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - Numeric:     median imputation, optionally StandardScaler.
      - Categorical: mode imputation (fill "Unknown" for all NaN), OneHotEncoder.

    Args:
        numeric_cols:     List of numeric column names.
        categorical_cols: List of categorical column names.
        scale_numerics:   If True, apply StandardScaler (use for LogisticRegression).

    Returns:
        Unfitted ColumnTransformer.
    """
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numerics:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def encode_target(series: pd.Series) -> np.ndarray:
    """
    Encode loan_status → binary int (Default=1, Fully Paid=0).
    """
    return (series == POSITIVE_LABEL).astype(int).values


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.dtypes)
    print(df[TARGET_COL].value_counts())
