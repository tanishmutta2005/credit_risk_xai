"""
explain.py — SHAP and LIME explainability.

Functions:
  global_shap_analysis()  → summary + beeswarm plots (saved as PNG)
  local_shap_explain()    → waterfall values for a single applicant (dict)
  lime_explain()          → LIME local feature importance (dict)

DISCLAIMER: For research/education only.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[explain] shap not installed; SHAP explanations disabled.")

try:
    import lime
    import lime.lime_tabular
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False
    print("[explain] lime not installed; LIME explanations disabled.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_inner_clf(pipeline):
    """
    Extract the final estimator from a sklearn Pipeline (or return as-is).
    """
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    return pipeline


def _transform_X(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Apply all preprocessing steps in *pipeline* except the final classifier.
    Works for both sklearn Pipeline and CalibratedClassifierCV.
    """
    # CalibratedClassifierCV wraps the original pipeline
    if hasattr(pipeline, "estimator"):
        return _transform_X(pipeline.estimator, X)

    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        # Pass through all steps except the last (the classifier)
        Xt = X
        for _, step in pipeline.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

    return X.values if hasattr(X, "values") else X


def _get_feature_names(pipeline, X: pd.DataFrame) -> list[str]:
    """
    Extract feature names from the preprocessor step of a pipeline.
    Falls back to X.columns or positional names.
    """
    # Unwrap CalibratedClassifierCV
    if hasattr(pipeline, "estimator"):
        return _get_feature_names(pipeline.estimator, X)

    if hasattr(pipeline, "steps"):
        pre = pipeline.steps[0][1]
        if hasattr(pre, "get_feature_names_out"):
            try:
                return list(pre.get_feature_names_out())
            except Exception:
                pass

    if hasattr(X, "columns"):
        return list(X.columns)
    return [f"f{i}" for i in range(X.shape[1])]


def _get_shap_explainer(pipeline, X_transformed: np.ndarray):
    """
    Choose the right SHAP explainer based on the underlying classifier type.
    """
    clf = _get_inner_clf(pipeline)
    # Unwrap CalibratedClassifierCV
    if hasattr(clf, "estimator"):
        clf = clf.estimator
    if hasattr(clf, "steps"):
        clf = clf.steps[-1][1]

    clf_type = type(clf).__name__

    if clf_type in ("RandomForestClassifier", "GradientBoostingClassifier",
                    "XGBClassifier", "LGBMClassifier", "ExtraTreesClassifier"):
        print(f"[explain] Using TreeExplainer for {clf_type}")
        return shap.TreeExplainer(clf)

    if clf_type == "LogisticRegression":
        print(f"[explain] Using LinearExplainer for {clf_type}")
        return shap.LinearExplainer(clf, X_transformed, feature_perturbation="correlation_dependent")

    print(f"[explain] Using KernelExplainer for {clf_type} (may be slow)")
    predict_fn = lambda x: pipeline.predict_proba(x)[:, 1]
    bg = shap.sample(X_transformed, 100)
    return shap.KernelExplainer(predict_fn, bg)


# ---------------------------------------------------------------------------
# Global SHAP analysis
# ---------------------------------------------------------------------------

def global_shap_analysis(
    pipeline,
    X_train: pd.DataFrame,
    output_dir: str = "models",
    max_display: int = 15,
    sample_size: int = 500,
) -> dict:
    """
    Compute global SHAP values and save summary + beeswarm plots.

    Args:
        pipeline:    Fitted model pipeline (or CalibratedClassifierCV).
        X_train:     Training feature DataFrame.
        output_dir:  Directory to save PNG plots.
        max_display: Number of top features to show in plots.
        sample_size: Subsample of training rows to keep computation tractable.

    Returns:
        Dict with keys: 'shap_values', 'feature_names', 'mean_abs_shap'.
    """
    if not _HAS_SHAP:
        return {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    if len(X_train) > sample_size:
        X_sample = X_train.sample(sample_size, random_state=42)
    else:
        X_sample = X_train

    X_transformed = _transform_X(pipeline, X_sample)
    feature_names = _get_feature_names(pipeline, X_sample)

    # Unwrap calibrated model for SHAP
    inner = pipeline.estimator if hasattr(pipeline, "estimator") else pipeline
    explainer = _get_shap_explainer(inner, X_transformed)

    print("[explain] Computing SHAP values (this may take a moment) …")
    shap_values_raw = explainer(X_transformed)

    # For binary classification, take the positive class column
    if hasattr(shap_values_raw, "values"):
        sv = shap_values_raw.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base_value = (
            shap_values_raw.base_values[:, 1]
            if shap_values_raw.base_values.ndim > 1
            else shap_values_raw.base_values
        )
    else:
        sv = shap_values_raw
        if isinstance(sv, list):
            sv = sv[1]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

    mean_abs = dict(
        zip(feature_names, np.abs(sv).mean(axis=0))
    )
    mean_abs = dict(sorted(mean_abs.items(), key=lambda x: -x[1]))

    # ── Summary bar plot ─────────────────────────────────────────────────────
    expl_obj = shap.Explanation(
        values=sv,
        base_values=np.full(len(sv), float(np.mean(base_value))),
        data=X_transformed,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(expl_obj, max_display=max_display, show=False, ax=ax)
    ax.set_title("Global Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    bar_path = f"{output_dir}/shap_summary_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] SHAP bar plot saved: {bar_path}")

    # ── Beeswarm plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.beeswarm(expl_obj, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Distribution")
    plt.tight_layout()
    beeswarm_path = f"{output_dir}/shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP beeswarm saved: {beeswarm_path}")

    return {
        "shap_values": sv,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "explainer": explainer,
        "base_value": float(np.mean(base_value)),
        "X_transformed": X_transformed,
    }


# ---------------------------------------------------------------------------
# Local SHAP explanation (per applicant)
# ---------------------------------------------------------------------------

def local_shap_explain(
    pipeline,
    x_instance: pd.DataFrame,
    global_result: dict,
    output_dir: str = "models",
) -> dict:
    """
    Compute and visualise per-applicant SHAP values (waterfall plot).

    Args:
        pipeline:      Fitted model pipeline.
        x_instance:    Single-row DataFrame for the applicant.
        global_result: Dict returned by global_shap_analysis() (reuses explainer).
        output_dir:    Where to save the waterfall PNG.

    Returns:
        Dict with 'shap_contributions': list of {feature, value, shap} dicts,
        sorted by |shap| descending.
    """
    if not _HAS_SHAP or not global_result:
        return {}

    explainer = global_result["explainer"]
    feature_names = global_result["feature_names"]
    base_value = global_result["base_value"]

    inner = pipeline.estimator if hasattr(pipeline, "estimator") else pipeline
    X_t = _transform_X(inner, x_instance)

    raw = explainer(X_t)

    if hasattr(raw, "values"):
        sv = raw.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        bv = (
            raw.base_values[:, 1]
            if raw.base_values.ndim > 1
            else raw.base_values
        )
    else:
        sv = raw[1] if isinstance(raw, list) else raw
        bv = np.array([base_value])

    sv_1d = sv[0]
    bv_scalar = float(bv[0]) if hasattr(bv, "__len__") else float(bv)

    # ── Waterfall plot ───────────────────────────────────────────────────────
    expl_obj = shap.Explanation(
        values=sv_1d,
        base_values=bv_scalar,
        data=X_t[0],
        feature_names=feature_names,
    )
    fig, _ = plt.subplots(figsize=(9, 7))
    shap.plots.waterfall(expl_obj, show=False)
    plt.title("SHAP Waterfall — Individual Applicant")
    plt.tight_layout()
    out_path = f"{output_dir}/shap_waterfall.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] SHAP waterfall saved: {out_path}")

    contributions = [
        {
            "feature": fn,
            "shap_value": round(float(sv), 4),
        }
        for fn, sv in zip(feature_names, sv_1d)
    ]
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {"shap_contributions": contributions, "base_value": bv_scalar}


# ---------------------------------------------------------------------------
# LIME local explanation
# ---------------------------------------------------------------------------

def lime_explain(
    pipeline,
    X_train: pd.DataFrame,
    x_instance: pd.DataFrame,
    feature_names: list[str] | None = None,
    num_features: int = 10,
) -> dict:
    """
    Generate a LIME explanation for *x_instance*.

    Args:
        pipeline:      Fitted model pipeline.
        X_train:       Training data (used to build LIME's background distribution).
        x_instance:    Single-row DataFrame for the applicant.
        feature_names: Column names to use (defaults to X_train.columns).
        num_features:  Number of features to include in the explanation.

    Returns:
        Dict with 'lime_contributions': list of (feature_label, weight) tuples.
    """
    if not _HAS_LIME:
        return {}

    if feature_names is None:
        feature_names = list(X_train.columns)

    # LIME needs arrays
    X_arr = X_train.values
    x_arr = x_instance.values[0]

    predict_fn = lambda arr: pipeline.predict_proba(
        pd.DataFrame(arr, columns=X_train.columns)
    )

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_arr,
        feature_names=feature_names,
        class_names=["Fully Paid", "Default"],
        mode="classification",
        random_state=42,
    )

    exp = explainer.explain_instance(
        data_row=x_arr,
        predict_fn=predict_fn,
        num_features=num_features,
    )

    contributions = exp.as_list(label=1)  # label=1 → "Default" class
    print("[explain] LIME explanation computed.")
    return {"lime_contributions": contributions}
