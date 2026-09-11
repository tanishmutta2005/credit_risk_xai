"""
models.py — Train, evaluate, calibrate, and select the best credit risk model.

Models compared:
  - LogisticRegression    (baseline; requires scaled features)
  - RandomForestClassifier (tree ensemble; robust to scaling)
  - XGBClassifier         (gradient boosting; typically best ROC-AUC)

Selection criterion: highest ROC-AUC on the held-out test set.
Calibration: CalibratedClassifierCV (isotonic regression on validation fold).

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[models] XGBoost not found; skipping XGBClassifier.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False
    print("[models] imbalanced-learn not found; SMOTE disabled.")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_lr(preprocessor) -> Pipeline:
    """Logistic Regression pipeline — uses scaled features."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    C=0.1,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_rf(preprocessor) -> Pipeline:
    """Random Forest — class_weight='balanced_subsample' handles imbalance."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_xgb(preprocessor, scale_pos_weight: float = 3.0) -> Pipeline:
    """XGBoost with scale_pos_weight to handle class imbalance."""
    if not _HAS_XGB:
        return None
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    preprocessor_scaled,    # ColumnTransformer with StandardScaler (for LR)
    preprocessor_unscaled,  # ColumnTransformer without scaler (for tree models)
) -> dict[str, Pipeline]:
    """
    Fit all three classifiers and return them in a dict keyed by name.

    Args:
        X_train:               Training feature DataFrame.
        y_train:               Binary target array (1=Default, 0=Fully Paid).
        preprocessor_scaled:   Preprocessor with StandardScaler.
        preprocessor_unscaled: Preprocessor without scaling.

    Returns:
        Dict mapping model name → fitted Pipeline.
    """
    # Weight for positive class based on imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)

    trained: dict[str, Pipeline] = {}

    models_to_train = {
        "LogisticRegression": _build_lr(preprocessor_scaled),
        "RandomForest": _build_rf(preprocessor_unscaled),
    }
    if _HAS_XGB:
        models_to_train["XGBoost"] = _build_xgb(preprocessor_unscaled, scale_pos_weight=spw)

    for name, pipeline in models_to_train.items():
        if pipeline is None:
            continue
        print(f"[models] Training {name} ...")
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        print(f"[models] {name} done.")

    return trained


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compute evaluation metrics for each model.

    Returns:
        DataFrame with rows per model and columns:
        roc_auc, f1, precision, recall, brier_score.
    """
    records = []
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        records.append(
            {
                "model": name,
                "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "brier_score": round(brier_score_loss(y_test, y_prob), 4),
            }
        )

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Fully Paid", "Default"]))
        print(f"   Confusion matrix:\n{cm}")
        print(f"   ROC-AUC: {records[-1]['roc_auc']}  |  Brier: {records[-1]['brier_score']}")

    results = pd.DataFrame(records).set_index("model")
    return results


def select_best_model(
    trained_models: dict[str, Pipeline],
    results: pd.DataFrame,
) -> tuple[str, Pipeline]:
    """
    Select the model with the highest ROC-AUC.

    Returns:
        (best_name, best_pipeline) tuple.
    """
    best_name = results["roc_auc"].idxmax()
    print(f"\n[models] Best model by ROC-AUC: {best_name} ({results.loc[best_name, 'roc_auc']})")
    return best_name, trained_models[best_name]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_model(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    method: str = "isotonic",
    output_path: str = "models/calibration_curve.png",
) -> CalibratedClassifierCV:
    """
    Wrap *model* with CalibratedClassifierCV and plot the reliability curve.

    Args:
        model:       Already-fitted sklearn Pipeline.
        X_val:       Validation features.
        y_val:       Validation labels.
        method:      "isotonic" or "sigmoid".
        output_path: Where to save the reliability-curve PNG.

    Returns:
        Fitted CalibratedClassifierCV wrapping the original model.
    """
    print(f"[models] Calibrating with method='{method}' ...")
    # sklearn 1.4+ removed cv='prefit'; use cv=None + set_params on a clone,
    # or simply re-fit CalibratedClassifierCV with a held-out fold.
    # Here we use cv=5 on training data (X_val) which is the recommended approach.
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=5)
    calibrated.fit(X_val, y_val)

    # ── Plot reliability curve ───────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_estimator(
        calibrated, X_val, y_val, n_bins=10, ax=ax, name="Calibrated model"
    )
    CalibrationDisplay.from_estimator(
        model, X_val, y_val, n_bins=10, ax=ax, name="Raw model"
    )
    ax.set_title("Reliability / Calibration Curve")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[models] Calibration curve saved: {output_path}")

    return calibrated


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Serialize *model* to disk with joblib."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[models] Saved: {path}")


def load_model(path: str) -> Any:
    """Deserialize model from *path*."""
    return joblib.load(path)
