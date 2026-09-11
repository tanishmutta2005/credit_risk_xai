"""
train.py — End-to-end training pipeline CLI.

Run:
    python train.py [--csv path/to/data.csv] [--output-dir models/]

Steps:
  1. Load / generate data
  2. Feature engineering
  3. Train/test split (stratified 80/20)
  4. Build preprocessors (scaled & unscaled)
  5. Train LR, RF, XGB
  6. Evaluate and select best model
  7. Calibrate best model
  8. Global SHAP analysis
  9. Fairness audit
  10. Save artifacts

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Local modules ────────────────────────────────────────────────────────────
from data import (
    CATEGORICAL_COLS,
    build_preprocessor,
    encode_target,
    load_data,
)
from features import ALL_NUMERIC_COLS, ENGINEERED_COLS, engineer_features
from models import (
    calibrate_model,
    evaluate_models,
    load_model,
    save_model,
    select_best_model,
    train_models,
)
from explain import global_shap_analysis
from fairness import full_fairness_report


# ---------------------------------------------------------------------------
# Risk bucketing
# ---------------------------------------------------------------------------

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def risk_category(prob: float) -> str:
    """Map a probability score to Low / Medium / High risk."""
    if prob < LOW_RISK_THRESHOLD:
        return "Low"
    if prob < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str | None = None, output_dir: str = "models") -> dict:
    """
    Execute the full training pipeline and return a summary dict.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_data(csv_path)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Target encoding ───────────────────────────────────────────────────
    y = encode_target(df["loan_status"])
    X = df.drop(columns=["loan_status"])

    # ── 4. Train/test split (stratified 80/20) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    # Further split training into train + validation (for calibration)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(
        f"[train] Splits: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )
    print(f"[train] Default rate — train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    # ── 5. Preprocessors ─────────────────────────────────────────────────────
    # Columns actually present after engineering
    numeric_cols = [c for c in ALL_NUMERIC_COLS if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    pre_scaled = build_preprocessor(numeric_cols, cat_cols, scale_numerics=True)
    pre_unscaled = build_preprocessor(numeric_cols, cat_cols, scale_numerics=False)

    # Fit preprocessors on training data
    pre_scaled.fit(X_train)
    pre_unscaled.fit(X_train)

    # ── 6. Train models ───────────────────────────────────────────────────────
    trained_models = train_models(X_train, y_train, pre_scaled, pre_unscaled)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    results_df = evaluate_models(trained_models, X_test, y_test)
    print(f"\n[train] Evaluation summary:\n{results_df.to_string()}\n")

    # ── 8. Select best ────────────────────────────────────────────────────────
    best_name, best_model = select_best_model(trained_models, results_df)

    # ── 9. Calibrate ──────────────────────────────────────────────────────────
    calibrated_model = calibrate_model(
        best_model, X_train, y_train,
        method="sigmoid",
        output_path=f"{output_dir}/calibration_curve.png",
    )

    # ── 10. Global SHAP ───────────────────────────────────────────────────────
    print("\n[train] Running global SHAP analysis …")
    global_shap_result = global_shap_analysis(
        calibrated_model, X_train, output_dir=output_dir, sample_size=300
    )

    # ── 11. Fairness audit ────────────────────────────────────────────────────
    print("\n[train] Running fairness audit …")
    fairness_results = full_fairness_report(calibrated_model, X_test, y_test)

    # ── 12. Save artifacts ────────────────────────────────────────────────────
    model_path = f"{output_dir}/best_model.pkl"
    save_model(calibrated_model, model_path)

    # Save preprocessors (for API)
    joblib.dump(pre_scaled, f"{output_dir}/preprocessor_scaled.pkl")
    joblib.dump(pre_unscaled, f"{output_dir}/preprocessor_unscaled.pkl")

    # Save column lists (for API)
    column_config = {
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "all_feature_cols": list(X_train.columns),
    }
    with open(f"{output_dir}/column_config.json", "w") as f:
        json.dump(column_config, f, indent=2)

    # Save metrics
    metrics_dict = results_df.to_dict()
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save fairness
    with open(f"{output_dir}/fairness.json", "w") as f:
        json.dump(fairness_results, f, indent=2)

    # Save SHAP feature importance
    if global_shap_result.get("mean_abs_shap"):
        top_shap = dict(
            list(global_shap_result["mean_abs_shap"].items())[:20]
        )
        with open(f"{output_dir}/shap_importance.json", "w") as f:
            json.dump(top_shap, f, indent=2)

    # Save a sample applicant for testing
    sample_applicant = X_test.iloc[[0]].to_dict(orient="records")[0]
    sample_applicant["true_label"] = int(y_test[0])
    with open(f"{output_dir}/sample_applicant.json", "w") as f:
        json.dump(sample_applicant, f, indent=2)

    # Save 200-row training sample for API SHAP explainer (API rebuilds explainer at startup)
    train_sample = X_train.sample(min(200, len(X_train)), random_state=42)
    joblib.dump(train_sample, f"{output_dir}/train_sample.pkl")
    print(f"[train] Training sample saved: {output_dir}/train_sample.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    best_metrics = results_df.loc[best_name].to_dict()
    summary = {
        "best_model": best_name,
        "metrics": best_metrics,
        "fairness": fairness_results,
        "artifacts": {
            "model": model_path,
            "calibration_curve": f"{output_dir}/calibration_curve.png",
            "shap_bar": f"{output_dir}/shap_summary_bar.png",
            "shap_beeswarm": f"{output_dir}/shap_beeswarm.png",
        },
    }

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model  : {best_name}")
    print(f"  ROC-AUC     : {best_metrics['roc_auc']}")
    print(f"  F1-score    : {best_metrics['f1']}")
    print(f"  Brier score : {best_metrics['brier_score']}")
    print(f"  Artifacts   : {output_dir}/")
    print("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk XAI — Training Pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to LendingClub CSV")
    parser.add_argument("--output-dir", type=str, default="models", help="Model output dir")
    args = parser.parse_args()

    summary = run_pipeline(csv_path=args.csv, output_dir=args.output_dir)
    sys.exit(0)
