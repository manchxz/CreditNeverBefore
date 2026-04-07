#!/usr/bin/env python3
"""
Train XGBoost on `data/cnb_training_data.csv` (injected NTC behavioral columns).

Outputs `cnb_model.joblib` for API / Vercel loading. SHAP uses TreeExplainer.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "data" / "cnb_training_data.csv"
MODEL_JSON = HERE / "cnb_model_injected.json"
FEATURES = [
    "AGE",
    "EXT_SOURCE_1",
    "UPI_VELOCITY",
    "BILL_PAY_CONSISTENCY",
    "APP_USAGE_DAYS",
]


def _shap_matrix(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.asarray(raw[1] if len(raw) > 1 else raw[0])
    return np.asarray(raw)


def main() -> None:
    if not DATA_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}. Run prepare_cnb_data.py first."
        )

    df = pd.read_csv(DATA_CSV)
    X = df[FEATURES]
    y = df["TARGET"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos) if pos else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    print("Training the CreditNeverBefore Engine...")
    print(f"scale_pos_weight (from train set): {scale_pos_weight:.4f}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nModel Performance (AUC-ROC): {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model.get_booster().save_model(MODEL_JSON)
    print(f"\nModel saved as {MODEL_JSON}")

    explainer = shap.TreeExplainer(model)
    sample = X_test[: min(500, len(X_test))]
    shap_values = _shap_matrix(explainer, sample.to_numpy())
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    print("\nSHAP mean |value| (top features, on sample):")
    for i in order[: len(FEATURES)]:
        print(f"  {FEATURES[i]}: {mean_abs[i]:.5f}")
    # summary_plot(shap_values, sample)  # use in Jupyter for portfolio charts


if __name__ == "__main__":
    main()
