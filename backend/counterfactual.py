"""
counterfactual.py — Counterfactual explanation generator.

Primary:  DiCE-ML (genetic algorithm).
Fallback: Greedy perturbation search (no TensorFlow/PyTorch required).

The goal is to find the MINIMAL set of mutable feature changes that flip
a high-risk applicant's prediction from "Default" → "Fully Paid".

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import dice_ml
    from dice_ml import Dice
    _HAS_DICE = True
except ImportError:
    _HAS_DICE = False
    print("[counterfactual] dice-ml not installed; using perturbation fallback.")


# ---------------------------------------------------------------------------
# Feature mutability config
# ---------------------------------------------------------------------------

# Features that cannot / should not be changed in a counterfactual
IMMUTABLE_FEATURES = {
    "age",
    "employment_status",   # usually fixed short-term
    "education",           # usually fixed
    "dependents",          # fixed
    "loan_amount",         # fixed at application time
    "loan_term",           # fixed at application time
    "loan_purpose",        # fixed at application time
    "installment",         # derived from loan_amount/term/rate
}

# Mutable features and their realistic change ranges
# Format: feature → (min_delta, max_delta) or (abs_min, abs_max)
MUTABLE_FEATURES = [
    "credit_utilization",   # can reduce by paying down cards
    "existing_debt",        # can reduce
    "late_payments",        # can improve over time
    "savings",              # can increase
    "monthly_expenses",     # can reduce
    "dti",                  # derived — adjustable via debt / income
    "lti",                  # derived
    "payment_burden",       # derived
    "savings_ratio",        # derived
]


# ---------------------------------------------------------------------------
# DiCE counterfactual
# ---------------------------------------------------------------------------

def dice_counterfactual(
    pipeline,
    X_train: pd.DataFrame,
    x_instance: pd.DataFrame,
    y_train: np.ndarray,
    n_cfs: int = 3,
) -> list[dict]:
    """
    Generate counterfactuals using DiCE-ML (genetic algorithm).

    Args:
        pipeline:    Fitted model pipeline.
        X_train:     Training features (for DiCE data object).
        x_instance:  Single-row DataFrame (the high-risk applicant).
        y_train:     Training labels.
        n_cfs:       Number of counterfactuals to generate.

    Returns:
        List of counterfactual dicts with changed feature values.
    """
    if not _HAS_DICE:
        return []

    # DiCE needs a combined dataframe with target
    target_col = "loan_status"
    df_train = X_train.copy()
    df_train[target_col] = y_train

    # Identify mutable features that actually exist in X_train
    mutable_in_data = [f for f in MUTABLE_FEATURES if f in X_train.columns]
    immutable_in_data = [c for c in X_train.columns if c not in mutable_in_data]

    d = dice_ml.Data(
        dataframe=df_train,
        continuous_features=[
            c for c in mutable_in_data
            if df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ],
        outcome_name=target_col,
    )

    # Wrap sklearn pipeline as DiCE model
    m = dice_ml.Model(model=pipeline, backend="sklearn")
    exp = Dice(d, m, method="genetic")

    try:
        cf_result = exp.generate_counterfactuals(
            query_instances=x_instance,
            total_CFs=n_cfs,
            desired_class=0,           # 0 = "Fully Paid"
            features_to_vary=mutable_in_data,
            random_seed=42,
        )
        cfs = cf_result.cf_examples_list[0].final_cfs_df
        if cfs is None or cfs.empty:
            return []

        results = []
        for _, row in cfs.iterrows():
            changes = {}
            for col in mutable_in_data:
                orig = float(x_instance[col].iloc[0])
                new_val = float(row[col])
                if abs(new_val - orig) > 1e-4:
                    changes[col] = {
                        "original": round(orig, 4),
                        "counterfactual": round(new_val, 4),
                        "change": round(new_val - orig, 4),
                    }
            results.append(changes)
        print(f"[counterfactual] DiCE generated {len(results)} counterfactual(s).")
        return results

    except Exception as e:
        print(f"[counterfactual] DiCE failed ({e}); falling back to perturbation search.")
        return []


# ---------------------------------------------------------------------------
# Perturbation-search fallback counterfactual
# ---------------------------------------------------------------------------

def perturbation_counterfactual(
    pipeline,
    x_instance: pd.DataFrame,
    n_steps: int = 30,
    target_prob: float = 0.35,
) -> dict:
    """
    Greedy perturbation search: iteratively reduce the highest-impact mutable
    features until the predicted default probability drops below *target_prob*.

    This is model-agnostic and requires only predict_proba.

    Args:
        pipeline:    Fitted model pipeline.
        x_instance:  Single-row DataFrame (the high-risk applicant).
        n_steps:     Max greedy steps.
        target_prob: Target probability threshold (below = "low risk").

    Returns:
        Dict of changed features with original/counterfactual values.
    """
    x_cf = x_instance.copy()
    changes: dict = {}

    # Perturbation schedule: for each mutable feature, try reducing it by ~10–20 %
    mutable_in_data = [f for f in MUTABLE_FEATURES if f in x_instance.columns]

    # Step-size fractions to try for each feature per iteration
    reduction_fractions = [0.10, 0.15, 0.20]

    current_prob = pipeline.predict_proba(x_cf)[:, 1][0]
    print(f"[counterfactual] Starting prob: {current_prob:.3f} | target: <{target_prob}")

    for step in range(n_steps):
        if current_prob <= target_prob:
            break

        best_feature, best_delta, best_prob = None, None, current_prob

        # Try reducing each mutable feature and pick the one that drops prob most
        for feat in mutable_in_data:
            val = float(x_cf[feat].iloc[0])
            if val <= 0:
                continue
            for frac in reduction_fractions:
                candidate = x_cf.copy()
                new_val = val * (1.0 - frac)
                candidate[feat] = new_val
                p = pipeline.predict_proba(candidate)[:, 1][0]
                if p < best_prob:
                    best_prob = p
                    best_feature = feat
                    best_delta = new_val

        if best_feature is None:
            break

        # Commit the best change
        orig_val = float(x_cf[best_feature].iloc[0])
        x_cf[best_feature] = best_delta
        current_prob = best_prob

        if best_feature not in changes:
            changes[best_feature] = {
                "original": round(orig_val, 4),
                "counterfactual": round(best_delta, 4),
                "change": round(best_delta - orig_val, 4),
            }
        else:
            changes[best_feature]["counterfactual"] = round(best_delta, 4)
            changes[best_feature]["change"] = round(
                best_delta - changes[best_feature]["original"], 4
            )

    print(
        f"[counterfactual] Perturbation search complete. "
        f"Final prob: {current_prob:.3f} | Changed {len(changes)} feature(s)."
    )
    return changes


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def find_counterfactual(
    pipeline,
    x_instance: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray | None = None,
) -> dict:
    """
    Attempt DiCE counterfactuals first, fall back to perturbation search.

    Returns:
        Dict with:
            'method': 'dice' | 'perturbation',
            'counterfactuals': list[dict] of changed features.
    """
    if _HAS_DICE and y_train is not None:
        cf_list = dice_counterfactual(pipeline, X_train, x_instance, y_train)
        if cf_list:
            return {"method": "dice", "counterfactuals": cf_list}

    # Fallback
    changes = perturbation_counterfactual(pipeline, x_instance)
    return {"method": "perturbation", "counterfactuals": [changes] if changes else []}
