"""
fairness.py — Fairness audit using fairlearn.

Computes per-group fairness metrics for a proxy sensitive attribute:
  - age_group  (≤35 / >35)  ← primary
  - employment_status        ← secondary

Metrics reported:
  - Demographic Parity Difference / Ratio
  - Equalized Odds (FPR / FNR by group)
  - Disparate Impact Ratio

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        false_negative_rate,
        false_positive_rate,
        selection_rate,
    )
    _HAS_FAIRLEARN = True
except ImportError:
    _HAS_FAIRLEARN = False
    print("[fairness] fairlearn not installed; fairness metrics disabled.")


# ---------------------------------------------------------------------------
# Sensitive attribute derivation
# ---------------------------------------------------------------------------

def derive_sensitive_attributes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create proxy sensitive attribute columns from existing features.

    Columns added:
        age_group          ≤35 → 'Young (≤35)'  | >35 → 'Senior (>35)'
        employment_group   from employment_status column
    """
    X = X.copy()

    if "age" in X.columns:
        X["age_group"] = np.where(
            pd.to_numeric(X["age"], errors="coerce") <= 35,
            "Young (<=35)",
            "Senior (>35)",
        )

    if "employment_status" in X.columns:
        X["employment_group"] = X["employment_status"].fillna("Unknown")

    return X


# ---------------------------------------------------------------------------
# Core fairness audit
# ---------------------------------------------------------------------------

def audit_fairness(
    pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    sensitive_col: str = "age_group",
    threshold: float = 0.5,
) -> dict:
    """
    Run a fairness audit on the held-out test set.

    Args:
        pipeline:      Fitted model pipeline.
        X_test:        Test feature DataFrame (with sensitive columns attached).
        y_test:        Ground-truth binary labels.
        sensitive_col: Column name to use as the sensitive attribute.
        threshold:     Classification threshold.

    Returns:
        Dict containing per-group FNR, FPR, selection rate, and summary metrics.
    """
    if not _HAS_FAIRLEARN:
        return {
            "error": "fairlearn not installed",
            "groups": {},
            "summary": {},
        }

    # Derive sensitive attributes if not already present
    X_aug = derive_sensitive_attributes(X_test)
    if sensitive_col not in X_aug.columns:
        return {
            "error": f"Sensitive column '{sensitive_col}' not found.",
            "groups": {},
            "summary": {},
        }

    sensitive = X_aug[sensitive_col]

    # Predicted probabilities → binary predictions
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── MetricFrame ───────────────────────────────────────────────────────────
    mf = MetricFrame(
        metrics={
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
    dp_ratio = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive)
    eq_odds_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)

    # Disparate Impact Ratio: min(selection rate) / max(selection rate)
    sel_rates = mf.by_group["selection_rate"]
    disparate_impact = float(sel_rates.min() / sel_rates.max()) if sel_rates.max() > 0 else 0.0

    # ── Format per-group results ──────────────────────────────────────────────
    groups_dict = {}
    for group, row in mf.by_group.iterrows():
        group_count = int((sensitive == group).sum())
        groups_dict[str(group)] = {
            "false_negative_rate": round(float(row["false_negative_rate"]), 4),
            "false_positive_rate": round(float(row["false_positive_rate"]), 4),
            "selection_rate": round(float(row["selection_rate"]), 4),
            "n": group_count,
        }

    summary = {
        "demographic_parity_difference": round(float(dp_diff), 4),
        "demographic_parity_ratio": round(float(dp_ratio), 4),
        "equalized_odds_difference": round(float(eq_odds_diff), 4),
        "disparate_impact_ratio": round(disparate_impact, 4),
        "sensitive_attribute": sensitive_col,
    }

    print(f"\n[fairness] Audit on '{sensitive_col}':")
    for grp, metrics in groups_dict.items():
        print(f"   {grp:20s} FNR={metrics['false_negative_rate']:.3f}  "
              f"FPR={metrics['false_positive_rate']:.3f}  "
              f"SR={metrics['selection_rate']:.3f}  n={metrics['n']}")
    print(f"   Demographic Parity Diff : {summary['demographic_parity_difference']}")
    print(f"   Disparate Impact Ratio  : {summary['disparate_impact_ratio']}")

    return {"groups": groups_dict, "summary": summary}


# ---------------------------------------------------------------------------
# Multi-attribute audit (both age_group & employment_group)
# ---------------------------------------------------------------------------

def full_fairness_report(
    pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """
    Run fairness audit on multiple sensitive attributes and return combined results.
    """
    X_aug = derive_sensitive_attributes(X_test)
    results = {}

    for attr in ["age_group", "employment_group"]:
        if attr in X_aug.columns:
            results[attr] = audit_fairness(pipeline, X_test, y_test, sensitive_col=attr)

    return results
