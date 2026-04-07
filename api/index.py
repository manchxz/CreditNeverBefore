#!/usr/bin/env python3
"""
Vercel serverless entrypoint for CreditNeverBefore.
Pure Python inference version (zero-dependency).

POST /api/predict
Body: JSON object with model features.
Returns: JSON with `prob_default` and `cnb_credit_score` (300–850).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import the generated pure Python model logic
try:
    from .model_logic import score
except ImportError:
    from model_logic import score

app = Flask(__name__)
# Allow browser calls from the Next.js dev server.
CORS(app, resources={r"/*": {"origins": "*"}})

# Feature set (must match the order in train_cnb_injected.py)
FEATURES = [
    "AGE",
    "EXT_SOURCE_1",
    "UPI_VELOCITY",
    "BILL_PAY_CONSISTENCY",
    "APP_USAGE_DAYS",
]

def _prob_to_score_300_850(prob_default: float) -> int:
    """Map default probability to a 300–850 CNB-style score."""
    p = max(0.0, min(1.0, float(prob_default)))
    # 300 = worst, 850 = best
    return int(round(300.0 + (1.0 - p) * (850.0 - 300.0)))

def get_score_metadata(score_val: int) -> Dict[str, str]:
    # Centralized business rules for score buckets + frontend actions.
    if score_val >= 750:
        return {"category": "Excellent", "action": "Instant Approval", "apr": "10%-12%"}
    if score_val >= 650:
        return {"category": "Good", "action": "Probable Approval", "apr": "14%-18%"}
    if score_val >= 500:
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

    def _safe_float(val: Any) -> float:
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    # Extract features in the EXACT order the model expects: 
    # [AGE, EXT_SOURCE_1, UPI_VELOCITY, BILL_PAY_CONSISTENCY, APP_USAGE_DAYS]
    
    age = payload.get("AGE")
    if age is None and "DAYS_BIRTH" in payload:
        age_val = _safe_float(payload.get("DAYS_BIRTH")) / -365.0
    else:
        age_val = _safe_float(age)

    ext_1 = _safe_float(payload.get("EXT_SOURCE_1"))
    upi = _safe_float(payload.get("UPI_VELOCITY"))
    bill = _safe_float(payload.get("BILL_PAY_CONSISTENCY"))
    app_days = _safe_float(payload.get("APP_USAGE_DAYS"))

    # Pure Python input list
    input_data = [age_val, ext_1, upi, bill, app_days]
    
    # Generate the class probabilities using the generated pure Python function
    # m2cgen's score() for XGBClassifier returns [prob_class_0, prob_class_1]
    results = score(input_data)
    prob_default = results[1]

    final_score = _prob_to_score_300_850(prob_default)
    metadata = get_score_metadata(int(final_score))

    return jsonify(
        {
            "prob_default": prob_default,
            "cnb_credit_score": int(final_score),
            "category": metadata["category"],
            "action": metadata["action"],
            "apr_estimate": metadata["apr"],
            "version": "1.2.0-pure-python",
        }
    )

# Vercel looks for `app` by default when using Flask
handler = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "5000")), debug=False)

# Vercel looks for `app` by default when using Flask
handler = app

if __name__ == "__main__":
    # Local dev helper so you can run `python api/index.py` and curl the endpoint.
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "5000")), debug=False)