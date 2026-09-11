"""
api.py — FastAPI backend for the Credit Risk XAI system.

Endpoints:
  GET  /health          → service health check
  POST /predict         → full risk assessment (probability, SHAP, counterfactual, fairness)
  GET  /model-info      → loaded model name, metrics, and feature list

The model and artifacts are loaded once at startup from the models/ directory
produced by train.py.

DISCLAIMER: For research/education only. NOT for real lending decisions.
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ── Local modules ────────────────────────────────────────────────────────────
from features import engineer_features
from explain import global_shap_analysis, local_shap_explain, lime_explain
from counterfactual import find_counterfactual
from fairness import audit_fairness, derive_sensitive_attributes
from train import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, risk_category

# ---------------------------------------------------------------------------
# App state container
# ---------------------------------------------------------------------------

class AppState:
    model: Any = None
    metrics: dict = {}
    column_config: dict = {}
    fairness_cache: dict = {}
    shap_importance: dict = {}
    global_shap_result: dict = {}
    X_train_sample: pd.DataFrame | None = None


state = AppState()
MODELS_DIR = os.getenv("MODELS_DIR", "models")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    print("[api] Loading model artifacts …")

    model_path = Path(MODELS_DIR) / "best_model.pkl"
    if not model_path.exists():
        print(f"[api] WARNING: No model at {model_path}. Run train.py first.")
    else:
        state.model = joblib.load(model_path)
        print(f"[api] Model loaded.")

    # Load column config
    cfg_path = Path(MODELS_DIR) / "column_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            state.column_config = json.load(f)

    # Load metrics
    metrics_path = Path(MODELS_DIR) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            state.metrics = json.load(f)

    # Load fairness
    fairness_path = Path(MODELS_DIR) / "fairness.json"
    if fairness_path.exists():
        with open(fairness_path) as f:
            state.fairness_cache = json.load(f)

    # Load SHAP importance
    shap_path = Path(MODELS_DIR) / "shap_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            state.shap_importance = json.load(f)

    # Rebuild SHAP explainer from a saved training sample (for local explanations)
    train_sample_path = Path(MODELS_DIR) / "train_sample.pkl"
    if train_sample_path.exists() and state.model is not None:
        try:
            X_train_sample = joblib.load(train_sample_path)
            print("[api] Rebuilding SHAP explainer from training sample ...")
            state.global_shap_result = global_shap_analysis(
                state.model, X_train_sample,
                output_dir=MODELS_DIR, sample_size=min(200, len(X_train_sample))
            )
            state.X_train_sample = X_train_sample
            print("[api] SHAP explainer ready.")
        except Exception as e:
            print(f"[api] SHAP setup skipped: {e}")

    print("[api] Startup complete.")
    yield
    print("[api] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk XAI API",
    description="Explainable AI Credit Risk Assessment — Research/Education Only",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ApplicantInput(BaseModel):
    # Demographics
    age: float = Field(35, ge=18, le=100, description="Applicant age")
    employment_status: str = Field("Employed", description="Employment status")
    employment_length: float = Field(5.0, ge=0, description="Years employed")
    annual_income: float = Field(60000.0, gt=0, description="Annual income (USD)")
    education: str = Field("Bachelor", description="Highest education level")
    dependents: int = Field(0, ge=0, le=20, description="Number of dependents")

    # Credit history
    credit_history_length: float = Field(8.0, ge=0, description="Years of credit history")
    previous_defaults: int = Field(0, ge=0, description="Number of previous defaults")
    late_payments: int = Field(0, ge=0, description="Number of late payments")
    credit_utilization: float = Field(0.3, ge=0.0, le=1.0, description="Credit utilization ratio")

    # Loan details
    loan_amount: float = Field(10000.0, gt=0, description="Loan amount (USD)")
    loan_term: int = Field(36, description="Loan term (months)")
    interest_rate: float = Field(12.0, gt=0, description="Interest rate (%)")
    loan_purpose: str = Field("debt_consolidation", description="Loan purpose")
    installment: float = Field(332.0, gt=0, description="Monthly installment (USD)")

    # Financial position
    existing_debt: float = Field(15000.0, ge=0, description="Total existing debt (USD)")
    monthly_expenses: float = Field(2000.0, ge=0, description="Monthly expenses (USD)")
    savings: float = Field(5000.0, ge=0, description="Total savings (USD)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "employment_status": "Employed",
                "employment_length": 5.0,
                "annual_income": 60000.0,
                "education": "Bachelor",
                "dependents": 1,
                "credit_history_length": 8.0,
                "previous_defaults": 0,
                "late_payments": 2,
                "credit_utilization": 0.65,
                "loan_amount": 15000.0,
                "loan_term": 36,
                "interest_rate": 14.5,
                "loan_purpose": "debt_consolidation",
                "installment": 520.0,
                "existing_debt": 25000.0,
                "monthly_expenses": 3200.0,
                "savings": 2000.0,
            }
        }


class FactorItem(BaseModel):
    feature: str
    shap_value: float
    direction: str   # "increases_risk" | "decreases_risk"


class CounterfactualChange(BaseModel):
    feature: str
    original: float
    counterfactual: float
    change: float


class FairnessGroup(BaseModel):
    group: str
    false_negative_rate: float
    false_positive_rate: float
    selection_rate: float
    n: int


class PredictionResponse(BaseModel):
    # Core prediction
    default_probability: float
    risk_category: str

    # SHAP factors
    top_risk_factors: list[FactorItem]
    top_positive_factors: list[FactorItem]

    # Counterfactual
    counterfactual_method: str
    counterfactual_changes: list[CounterfactualChange]
    counterfactual_probability: Optional[float]

    # Model info
    model_metrics: dict
    model_name: str

    # Fairness
    fairness_summary: dict
    fairness_by_group: list[FairnessGroup]

    # Disclaimer
    disclaimer: str = (
        "WARNING: This output is for educational/research purposes only. "
        "Do NOT use for real lending decisions."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_from_input(data: ApplicantInput) -> pd.DataFrame:
    """Convert Pydantic input to a single-row DataFrame."""
    row = data.model_dump()
    df = pd.DataFrame([row])
    return df


def _compute_shap_factors(
    pipeline, x_df: pd.DataFrame, global_shap_result: dict
) -> tuple[list[FactorItem], list[FactorItem]]:
    """
    Run local SHAP and split contributions into risk-increasing / risk-decreasing.
    """
    local_result = local_shap_explain(pipeline, x_df, global_shap_result)
    contributions = local_result.get("shap_contributions", [])

    risk_factors = []
    positive_factors = []

    for item in contributions[:15]:  # top 15 by |SHAP|
        feat = item["feature"]
        sv = item["shap_value"]
        if sv > 0:
            risk_factors.append(
                FactorItem(feature=feat, shap_value=round(sv, 4), direction="increases_risk")
            )
        else:
            positive_factors.append(
                FactorItem(feature=feat, shap_value=round(sv, 4), direction="decreases_risk")
            )

    return risk_factors[:5], positive_factors[:5]


def _format_counterfactual(cf_result: dict, pipeline, x_df: pd.DataFrame) -> tuple:
    """Extract and format counterfactual changes."""
    method = cf_result.get("method", "perturbation")
    cf_list = cf_result.get("counterfactuals", [])

    if not cf_list:
        return method, [], None

    # Use first counterfactual
    cf = cf_list[0]
    changes = [
        CounterfactualChange(
            feature=feat,
            original=vals["original"],
            counterfactual=vals["counterfactual"],
            change=vals["change"],
        )
        for feat, vals in cf.items()
    ]

    # Estimate probability under counterfactual
    cf_prob = None
    try:
        x_cf = x_df.copy()
        for feat, vals in cf.items():
            if feat in x_cf.columns:
                x_cf[feat] = vals["counterfactual"]
        cf_prob = round(float(pipeline.predict_proba(x_cf)[:, 1][0]), 4)
    except Exception:
        pass

    return method, changes, cf_prob


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Service health check."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
    }


@app.get("/model-info", tags=["System"])
async def model_info():
    """Return loaded model name, metrics, and feature list."""
    return {
        "column_config": state.column_config,
        "metrics": state.metrics,
        "top_shap_features": list(state.shap_importance.keys())[:10],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: ApplicantInput):
    """
    Full credit risk assessment for a single applicant.

    Returns:
      - default_probability, risk_category
      - top SHAP risk and protective factors
      - counterfactual scenario (how to become low-risk)
      - model metrics and fairness audit summary
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py to generate artifacts first.",
        )

    # ── 1. Build input DataFrame + engineer features ──────────────────────────
    x_raw = _df_from_input(data)
    x_df = engineer_features(x_raw)

    # ── 2. Predict probability ────────────────────────────────────────────────
    try:
        prob = float(state.model.predict_proba(x_df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    cat = risk_category(prob)

    # ── 3. SHAP local explanation ─────────────────────────────────────────────
    risk_factors: list[FactorItem] = []
    positive_factors: list[FactorItem] = []

    if state.global_shap_result:
        try:
            risk_factors, positive_factors = _compute_shap_factors(
                state.model, x_df, state.global_shap_result
            )
        except Exception as e:
            print(f"[api] SHAP local explain failed: {e}")

    # Fallback: use global SHAP importance ranking
    if not risk_factors and state.shap_importance:
        for feat, importance in list(state.shap_importance.items())[:5]:
            risk_factors.append(
                FactorItem(
                    feature=feat,
                    shap_value=round(importance, 4),
                    direction="increases_risk",
                )
            )

    # ── 4. Counterfactual ─────────────────────────────────────────────────────
    cf_method, cf_changes, cf_prob = "perturbation", [], None
    if prob >= LOW_RISK_THRESHOLD:  # only for medium/high risk
        try:
            cf_result = find_counterfactual(
                state.model, x_df,
                state.X_train_sample if state.X_train_sample is not None else x_df,
            )
            cf_method, cf_changes, cf_prob = _format_counterfactual(
                cf_result, state.model, x_df
            )
        except Exception as e:
            print(f"[api] Counterfactual failed: {e}")

    # ── 5. Fairness from cache ────────────────────────────────────────────────
    fairness_by_group: list[FairnessGroup] = []
    fairness_summary: dict = {}

    age_grp_key = "age_group"
    if age_grp_key in state.fairness_cache:
        fc = state.fairness_cache[age_grp_key]
        fairness_summary = fc.get("summary", {})
        for grp, vals in fc.get("groups", {}).items():
            fairness_by_group.append(
                FairnessGroup(
                    group=grp,
                    false_negative_rate=vals["false_negative_rate"],
                    false_positive_rate=vals["false_positive_rate"],
                    selection_rate=vals["selection_rate"],
                    n=vals["n"],
                )
            )

    # ── 6. Best model metrics ─────────────────────────────────────────────────
    # Pick the metrics row with highest roc_auc from saved dict
    model_metrics = {}
    model_name = "Unknown"
    if state.metrics.get("roc_auc"):
        best = max(state.metrics["roc_auc"], key=state.metrics["roc_auc"].get)
        model_name = best
        model_metrics = {k: state.metrics[k][best] for k in state.metrics}

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_category=cat,
        top_risk_factors=risk_factors,
        top_positive_factors=positive_factors,
        counterfactual_method=cf_method,
        counterfactual_changes=cf_changes,
        counterfactual_probability=cf_prob,
        model_metrics=model_metrics,
        model_name=model_name,
        fairness_summary=fairness_summary,
        fairness_by_group=fairness_by_group,
    )


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
