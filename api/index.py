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

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from xgboost import Booster, DMatrix

# Resolve paths relative to this file
HERE = Path(__file__).resolve().parent.parent
MODEL_JSON_PATH = HERE / "cnb_model_injected.json"

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

def _load_artifacts() -> Dict[str, Any]:
    booster = Booster()
    booster.load_model(str(MODEL_JSON_PATH))
    return {"booster": booster}

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

    # Extract features manually to avoid pandas
    age = payload.get("AGE")
    if age is None and "DAYS_BIRTH" in payload:
        try:
            age_val = float(payload.get("DAYS_BIRTH", 0)) / -365.0
        except ValueError:
            age_val = np.nan
    else:
        try:
            age_val = float(age) if age is not None else np.nan
        except ValueError:
            age_val = np.nan

    def _safe_float(val: Any) -> float:
        try:
            return float(val) if val is not None else np.nan
        except ValueError:
            return np.nan

    ext_1 = _safe_float(payload.get("EXT_SOURCE_1"))
    upi = _safe_float(payload.get("UPI_VELOCITY"))
    bill = _safe_float(payload.get("BILL_PAY_CONSISTENCY"))
    app_days = _safe_float(payload.get("APP_USAGE_DAYS"))

    # Create 2D numpy array [1, 5]
    row = np.array([[age_val, ext_1, upi, bill, app_days]], dtype=np.float32)
    
    booster: Booster = ARTIFACTS["booster"]
    dmat = DMatrix(row, feature_names=FEATURES)
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
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "5000")), debug=False)