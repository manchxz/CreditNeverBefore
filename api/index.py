#!/usr/bin/env python3
"""
Vercel serverless entrypoint for CreditNeverBefore.

POST /api/predict
Body: JSON object with model features (raw fields from Home Credit / CNB front-end).
Returns: JSON with `prob_default` and `cnb_credit_score` (300–850).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from xgboost import Booster, DMatrix

# Resolve paths relative to this file
HERE = Path(__file__).resolve().parent.parent
PUBLIC_DIR = HERE / "public"

MODEL_PATH = PUBLIC_DIR / "cnb_model.json"
MODEL_JOBLIB_PATH = HERE / "cnb_model.joblib"
PREPROCESSOR_PATH = PUBLIC_DIR / "preprocessor.joblib"

app = Flask(__name__)
# Allow browser calls from the Next.js dev server.
# Using a permissive policy keeps local development friction low.
CORS(app, resources={r"/*": {"origins": "*"}})

# Feature set for `train_cnb_injected.py` (must match exactly)
FEATURES = [
    "AGE",
    "EXT_SOURCE_1",
    "UPI_VELOCITY",
    "BILL_PAY_CONSISTENCY",
    "APP_USAGE_DAYS",
]

# ---------------------------------------------------------------------------
# Feature engineering (must mirror `train_model.py`)
# ---------------------------------------------------------------------------

DOCUMENT_FLAG_COLS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
DOCUMENT_COMPLETENESS_COL = "document_completeness"

DAYS_EMPLOYED_ANOMALY = 365_243
AGE_YEARS_COL = "age_years"
EMPLOYMENT_YEARS_COL = "employment_years"

# Columns fed into the sklearn preprocessor
NUMERIC_FEATURES = [
    AGE_YEARS_COL,
    EMPLOYMENT_YEARS_COL,
    "phone_change_days",
    "registration_days",
    "id_publish_days",
    DOCUMENT_COMPLETENESS_COL,
    "log_income",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    # Synthetic / NTC-style behavioral markers
    "UPI_VELOCITY",
    "BILL_PAY_CONSISTENCY",
    "APP_USAGE_DAYS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

CATEGORICAL_FEATURES = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
]


def _engineer_behavioral_features(df: Any) -> Any:
    """
    Mirror `train_model.py:engineer_behavioral_features` for inference-time inputs.

    Input expected: raw Home Credit fields (e.g. `DAYS_BIRTH`, `DAYS_EMPLOYED`, etc.)
    Output: engineered columns that the stored sklearn preprocessor expects.
    """
    import pandas as pd  # local import keeps cold-start overhead lower

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    out = df.copy()

    # Age (DAYS_BIRTH is negative in Kaggle Home Credit)
    if "DAYS_BIRTH" in out.columns:
        out[AGE_YEARS_COL] = -out["DAYS_BIRTH"].astype(np.float64) / 365.25
    else:
        out[AGE_YEARS_COL] = np.nan

    # Employment: negative days = employed; anomaly -> NaN -> imputed later
    if "DAYS_EMPLOYED" in out.columns:
        de = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan).astype(np.float64)
        employed_mask = de < 0
        out[EMPLOYMENT_YEARS_COL] = np.where(employed_mask, -de / 365.25, np.nan)
    else:
        out[EMPLOYMENT_YEARS_COL] = np.nan

    # Phone / address / ID stability signals (more negative => longer stable history)
    for col, name in [
        ("DAYS_LAST_PHONE_CHANGE", "phone_change_days"),
        ("DAYS_REGISTRATION", "registration_days"),
        ("DAYS_ID_PUBLISH", "id_publish_days"),
    ]:
        out[name] = out[col].astype(np.float64) if col in out.columns else np.nan

    # Document consistency: count of submitted optional docs (FLAG_DOCUMENT_2..21)
    present_docs = [c for c in DOCUMENT_FLAG_COLS if c in out.columns]
    if present_docs:
        out[DOCUMENT_COMPLETENESS_COL] = out[present_docs].fillna(0).sum(axis=1)
    else:
        out[DOCUMENT_COMPLETENESS_COL] = 0.0

    # Income capacity control (log1p on non-negative income)
    if "AMT_INCOME_TOTAL" in out.columns:
        out["log_income"] = np.log1p(out["AMT_INCOME_TOTAL"].clip(lower=0).astype(np.float64))
    else:
        out["log_income"] = np.nan

    # External behavioral scores (when present)
    for i in (1, 2, 3):
        c = f"EXT_SOURCE_{i}"
        if c not in out.columns:
            out[c] = np.nan

    return out


def _inject_synthetic_behavioral_ntc(df: Any) -> Any:
    """
    Mirror `train_model.py:inject_synthetic_behavioral_ntc`.

    During inference, we won't have `TARGET`, so missing synthetic markers are set to NaN.
    """
    import pandas as pd  # local import keeps cold-start overhead lower

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")

    out = df.copy()
    required_cols = {"UPI_VELOCITY", "BILL_PAY_CONSISTENCY", "APP_USAGE_DAYS"}

    if required_cols.issubset(out.columns):
        return out

    if "TARGET" not in out.columns:
        # Cannot inject label-aligned synthetic behavior without TARGET (inference path).
        for col in required_cols - set(out.columns):
            out[col] = np.nan
        return out

    rng = np.random.default_rng(42)

    def _row_inject(target: int) -> tuple[float, float, float]:
        if target == 1:
            upi_velocity = rng.uniform(0.1, 0.5)
            bill_pay_consistency = rng.uniform(0.2, 0.6)
            app_usage_days = rng.integers(1, 15)
        else:
            upi_velocity = rng.uniform(0.6, 1.0)
            bill_pay_consistency = rng.uniform(0.8, 1.0)
            app_usage_days = rng.integers(20, 31)
        return float(upi_velocity), float(bill_pay_consistency), float(app_usage_days)

    upi_vals: list[float] = []
    bill_vals: list[float] = []
    app_vals: list[float] = []
    for t in out["TARGET"].astype(int):
        u, b, a = _row_inject(t)
        upi_vals.append(u)
        bill_vals.append(b)
        app_vals.append(a)

    if "UPI_VELOCITY" not in out.columns:
        out["UPI_VELOCITY"] = upi_vals
    if "BILL_PAY_CONSISTENCY" not in out.columns:
        out["BILL_PAY_CONSISTENCY"] = bill_vals
    if "APP_USAGE_DAYS" not in out.columns:
        out["APP_USAGE_DAYS"] = app_vals

    return out


def _build_feature_matrix(df: Any) -> Any:
    """
    Mirror `train_model.py:build_feature_matrix`.

    Ensures the stored sklearn ColumnTransformer always sees every required column.
    """
    engineered = _engineer_behavioral_features(df)
    engineered = _inject_synthetic_behavioral_ntc(engineered)

    # Ensure all expected numeric columns exist
    for col in NUMERIC_FEATURES:
        if col not in engineered.columns:
            engineered[col] = np.nan

    # Ensure all expected categorical columns exist
    for col in CATEGORICAL_FEATURES:
        if col not in engineered.columns:
            engineered[col] = "Unknown"

    use_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return engineered[use_cols].copy()


def _load_artifacts() -> Dict[str, Any]:
    # Prefer the simpler injected model if present (trained on `FEATURES`).
    if MODEL_JOBLIB_PATH.is_file():
        model = joblib.load(MODEL_JOBLIB_PATH)
        return {"mode": "joblib_injected", "model": model}

    # Fallback: sklearn ColumnTransformer preprocessor + XGBoost booster
    booster = Booster()
    booster.load_model(str(MODEL_PATH))
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return {"mode": "sklearn_pipeline", "booster": booster, "preprocessor": preprocessor}


ARTIFACTS = _load_artifacts()


def _prob_to_score_300_850(prob_default: float) -> int:
    """Map default probability to a 300–850 CNB-style score."""
    p = float(np.clip(prob_default, 0.0, 1.0))
    # 300 = worst, 850 = best
    return int(round(300.0 + (1.0 - p) * (850.0 - 300.0)))


def get_score_metadata(score: int) -> Dict[str, str]:
    # Centralized business rules for score buckets + frontend actions.
    if score >= 750:
        return {"category": "Excellent", "action": "Instant Approval", "apr": "10%-12%"}
    if score >= 650:
        return {"category": "Good", "action": "Probable Approval", "apr": "14%-18%"}
    if score >= 500:
        return {"category": "Average", "action": "Needs Manual Review", "apr": "20%+"}
    return {"category": "Poor", "action": "Reject", "apr": "N/A"}


@app.post("/api/predict")
def predict() -> Any:
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object with feature keys"}), 400

    # Single-row DataFrame from the incoming feature dict
    import pandas as pd

    df = pd.DataFrame([payload])

    if ARTIFACTS["mode"] == "joblib_injected":
        # Ensure all required columns exist; if missing, fill with NaN.
        for col in FEATURES:
            if col not in df.columns:
                if col == "AGE" and "DAYS_BIRTH" in df.columns:
                    # Mirror prepare_cnb_data.py: AGE = DAYS_BIRTH / -365.0
                    df[col] = df["DAYS_BIRTH"].astype(np.float64) / -365.0
                else:
                    df[col] = np.nan

        X = df[FEATURES]
        model = ARTIFACTS["model"]
        # XGBoost classifier should support predict_proba; if not, fall back to raw predict.
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(X)[:, 1]
        else:
            proba_arr = model.predict(X)
        prob_default = float(proba_arr[0])
    else:
        preprocessor = ARTIFACTS["preprocessor"]
        booster: Booster = ARTIFACTS["booster"]

        # IMPORTANT: mirror training feature engineering so the preprocessor columns exist
        engineered_df = _build_feature_matrix(df)
        X = preprocessor.transform(engineered_df)
        dmat = DMatrix(X)
        proba_arr = booster.predict(dmat)
        prob_default = float(proba_arr[0])

    score = _prob_to_score_300_850(prob_default)
    metadata = get_score_metadata(int(score))

    return jsonify(
        {
            "prob_default": prob_default,
            "cnb_credit_score": int(score),
            "category": metadata["category"],
            "action": metadata["action"],
            "apr_estimate": metadata["apr"],
            "version": "1.1.0-injected",
        }
    )


# Vercel looks for `app` by default when using Flask
handler = app


if __name__ == "__main__":
    # Local dev helper so you can run `python api/index.py` and curl the endpoint.
    # Default to 5000 (matches frontend), but allow override when 5000 is taken.
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "5000")), debug=False)