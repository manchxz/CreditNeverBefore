#!/usr/bin/env python3
"""
CreditNeverBefore (CNB) — behavioral underwriting model trainer.

Trains an XGBoost classifier on Kaggle Home Credit Default Risk `application_train.csv`,
with feature engineering oriented toward behavioral discipline signals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PUBLIC_DIR = PROJECT_ROOT / "public"

DEFAULT_DATA_PATH = DATA_DIR / "application_train.csv"
MODEL_PATH = PUBLIC_DIR / "cnb_model.json"
PREPROCESSOR_PATH = PUBLIC_DIR / "preprocessor.joblib"

# Document flag columns in Home Credit application table (submitted = 1)
DOCUMENT_FLAG_COLS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]

# Kaggle anomaly: positive value means "unemployed" placeholder
DAYS_EMPLOYED_ANOMALY = 365_243

# Lean model settings (keeps JSON model small; tune if needed)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "reg_lambda": 1.5,
    "reg_alpha": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_application_train(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Download Kaggle 'Home Credit Default Risk' "
            "and place application_train.csv under data/."
        )
    df = pd.read_csv(csv_path)
    if "TARGET" not in df.columns:
        raise ValueError("Expected column TARGET in application_train.csv")
    return df


# ---------------------------------------------------------------------------
# Feature engineering — CNB “behavioral discipline” pivot
# ---------------------------------------------------------------------------


def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features that proxy discipline & stability rather than raw income alone.

    - Age, employment tenure (with Home Credit anomaly handling)
    - Phone change recency, registration / ID document stability
    - Document completeness (count of submitted docs)
    - Regional stability ratings; external score behavior proxies
    """
    out = df.copy()

    # Age (DAYS_BIRTH is negative)
    out["age_years"] = -out["DAYS_BIRTH"].astype(np.float64) / 365.25

    # Employment: negative days = employed; anomaly -> NaN → imputed later
    de = out["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan).astype(np.float64)
    employed_mask = de < 0
    out["employment_years"] = np.where(employed_mask, -de / 365.25, np.nan)

    # Phone / address / ID stability (more negative = longer stable history in raw data)
    for col, name in [
        ("DAYS_LAST_PHONE_CHANGE", "phone_change_days"),
        ("DAYS_REGISTRATION", "registration_days"),
        ("DAYS_ID_PUBLISH", "id_publish_days"),
    ]:
        if col in out.columns:
            out[name] = out[col].astype(np.float64)
        else:
            out[name] = np.nan

    # Document consistency: how many optional docs submitted
    present_docs = [c for c in DOCUMENT_FLAG_COLS if c in out.columns]
    if present_docs:
        out["document_completeness"] = out[present_docs].fillna(0).sum(axis=1)
    else:
        out["document_completeness"] = 0.0

    # Income as capacity control (log1p), not the main story
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


def inject_synthetic_behavioral_ntc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject synthetic NTC-style behavioral markers if not already present.

    Mirrors the logic from prepare_cnb_data.py:
    - TARGET == 1 → weaker discipline
    - TARGET == 0 → stronger discipline
    Only runs when TARGET is available (training data).
    """
    required_cols = {"UPI_VELOCITY", "BILL_PAY_CONSISTENCY", "APP_USAGE_DAYS"}
    if required_cols.issubset(df.columns):
        return df
    if "TARGET" not in df.columns:
        # Cannot inject label-aligned synthetic behavior without TARGET (e.g. inference path)
        for col in required_cols - set(df.columns):
            df[col] = np.nan
        return df

    rng = np.random.default_rng(42)

    def _row_inject(target: int) -> Tuple[float, float, float]:
        if target == 1:
            # Add noise so TARGET-conditioned ranges overlap (prevents trivial leakage)
            upi_velocity = rng.uniform(0.1, 0.7)
            bill_pay_consistency = rng.uniform(0.2, 0.7)
            app_usage_days = rng.integers(1, 22)
        else:
            # Add noise so good users can still have low metrics (and vice-versa)
            upi_velocity = rng.uniform(0.4, 1.0)
            bill_pay_consistency = rng.uniform(0.5, 1.0)
            app_usage_days = rng.integers(10, 31)
        return float(upi_velocity), float(bill_pay_consistency), float(app_usage_days)

    upi_vals: List[float] = []
    bill_vals: List[float] = []
    app_vals: List[float] = []
    for t in df["TARGET"].astype(int):
        u, b, a = _row_inject(t)
        upi_vals.append(u)
        bill_vals.append(b)
        app_vals.append(a)

    if "UPI_VELOCITY" not in df.columns:
        df["UPI_VELOCITY"] = upi_vals
    if "BILL_PAY_CONSISTENCY" not in df.columns:
        df["BILL_PAY_CONSISTENCY"] = bill_vals
    if "APP_USAGE_DAYS" not in df.columns:
        df["APP_USAGE_DAYS"] = app_vals

    return df


# Columns fed to the model after engineering (behavior-forward)
NUMERIC_FEATURES: List[str] = [
    "age_years",
    "employment_years",
    "phone_change_days",
    "registration_days",
    "id_publish_days",
    "document_completeness",
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

CATEGORICAL_FEATURES: List[str] = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_behavioral_features(df)
    engineered = inject_synthetic_behavioral_ntc(engineered)

    # Ensure all expected columns exist; fall back to NaN / "Unknown" when absent.
    for col in NUMERIC_FEATURES:
        if col not in engineered.columns:
            engineered[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in engineered.columns:
            engineered[col] = "Unknown"

    use_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = engineered[use_cols].copy()
    return X


def _make_one_hot_encoder() -> OneHotEncoder:
    """sklearn 1.2+ uses sparse_output; older uses sparse. Cap categories for lean OHE."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=20,
        )
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def _shap_matrix_for_risk(explainer: Any, X_t: np.ndarray) -> np.ndarray:
    """
    Binary classifier SHAP can be a list (per class) or a single array (margin toward class 1).
    Return shape (n_samples, n_features) for the default / positive class.
    """
    raw = explainer.shap_values(X_t)
    if isinstance(raw, list):
        if len(raw) == 1:
            return np.asarray(raw[0])
        # Prefer contributions aligned with P(class 1) — default risk
        return np.asarray(raw[1])
    return np.asarray(raw)


def train_xgb_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> Pipeline:
    preprocessor = build_preprocessor()
    clf = XGBClassifier(**XGB_PARAMS)

    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    print(f"Validation ROC-AUC: {auc:.4f}")
    return pipe


def shap_feature_summary(model: Pipeline, X_sample: pd.DataFrame, max_display: int = 12) -> None:
    """Compute SHAP values on a sample for global importance (TreeExplainer)."""
    prep = model.named_steps["prep"]
    X_t = prep.transform(X_sample)
    booster = model.named_steps["model"]
    explainer = shap.TreeExplainer(booster)
    shap_values = _shap_matrix_for_risk(explainer, X_t)
    feature_names = prep.get_feature_names_out()
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:max_display]
    print("\n--- SHAP mean |value| (top features) ---")
    for idx in order:
        print(f"  {feature_names[idx]}: {mean_abs[idx]:.5f}")


def explain_rejection(
    model: Pipeline, row: pd.DataFrame, top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Local explanation: which features pushed toward default (positive SHAP = higher default risk).
    """
    prep = model.named_steps["prep"]
    Xt = prep.transform(row)
    booster = model.named_steps["model"]
    explainer = shap.TreeExplainer(booster)
    mat = _shap_matrix_for_risk(explainer, Xt)
    sv = mat[0]
    names = prep.get_feature_names_out()
    pairs = list(zip(names, sv))
    pairs.sort(key=lambda x: -abs(x[1]))
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Credit score (0–800) & serialization
# ---------------------------------------------------------------------------


def default_probability_to_credit_score(prob_default: float) -> int:
    """Map P(default) to a 0–800 score; lower default risk → higher score."""
    p = float(np.clip(prob_default, 0.0, 1.0))
    return int(round((1.0 - p) * 800))


def save_artifacts(model: Pipeline) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    booster = model.named_steps["model"]
    booster.save_model(str(MODEL_PATH))
    # Fitted preprocessor for inference (same transform as training)
    joblib.dump(model.named_steps["prep"], PREPROCESSOR_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved model JSON: {MODEL_PATH} ({size_mb:.2f} MB)")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")
    if size_mb >= 50:
        print("Warning: model file is >= 50 MB. Reduce n_estimators/max_depth or features.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mock user test
# ---------------------------------------------------------------------------


def mock_user_row() -> Dict[str, Any]:
    """Single-row dict mimicking raw `application_train` fields needed for FE."""
    return {
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_LAST_PHONE_CHANGE": -300,
        "DAYS_REGISTRATION": -5000,
        "DAYS_ID_PUBLISH": -4000,
        "AMT_INCOME_TOTAL": 75000,
        "EXT_SOURCE_1": 0.45,
        "EXT_SOURCE_2": 0.55,
        "EXT_SOURCE_3": 0.40,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "OCCUPATION_TYPE": "Laborers",
        **{f"FLAG_DOCUMENT_{i}": 1 if i % 2 == 0 else 0 for i in range(2, 22)},
    }


def test_mock_user(user_json: Optional[Union[str, Dict[str, Any]]] = None) -> None:
    """
    Load `cnb_model.json` + `preprocessor.joblib` and print score + SHAP drivers.
    Pass a JSON string, or a dict, or None to use the built-in mock profile.
    """
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        print(
            "Expected cnb_model.json and preprocessor.joblib next to this script. Train first.",
            file=sys.stderr,
        )
        return
    clf = XGBClassifier()
    clf.load_model(str(MODEL_PATH))
    pipe = Pipeline([("prep", joblib.load(PREPROCESSOR_PATH)), ("model", clf)])
    if user_json is None:
        run_mock_user_inference(pipe, user_json=None)
    elif isinstance(user_json, dict):
        run_mock_user_inference(pipe, user_json=json.dumps(user_json))
    else:
        run_mock_user_inference(pipe, user_json=user_json)


def run_mock_user_inference(
    model: Pipeline,
    user_json: Optional[str] = None,
) -> None:
    """
    Load optional JSON string (one object) or use built-in mock user.
    Prints predicted P(default), 0–800 score, and top SHAP drivers.
    """
    if user_json:
        raw = json.loads(user_json)
    else:
        raw = mock_user_row()
    df_raw = pd.DataFrame([raw])
    X = build_feature_matrix(df_raw)
    proba = model.predict_proba(X)[0, 1]
    score = default_probability_to_credit_score(proba)
    print("\n--- Mock user inference ---")
    print(f"P(default): {proba:.4f}")
    print(f"CNB credit score (0–800): {score}")
    print("Top local SHAP contributions (higher → more toward default):")
    for name, val in explain_rejection(model, X, top_k=8):
        print(f"  {name}: {val:+.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train CNB XGBoost behavioral risk model")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run test_mock_user (requires saved model artifacts)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to application_train.csv",
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=2000,
        help="Rows to use for SHAP summary (subset for speed)",
    )
    args = parser.parse_args()

    if args.test_only:
        test_mock_user()
        return

    df = load_application_train(args.data)
    y = df["TARGET"].astype(int)
    X = build_feature_matrix(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_xgb_classifier(X_train, y_train, X_val, y_val)

    sample = X_train.sample(n=min(args.shap_sample, len(X_train)), random_state=42)
    shap_feature_summary(model, sample)

    save_artifacts(model)

    # Reload booster from JSON + preprocessor to mirror production inference path
    clf_loaded = XGBClassifier()
    clf_loaded.load_model(str(MODEL_PATH))
    loaded_pipe = Pipeline([("prep", joblib.load(PREPROCESSOR_PATH)), ("model", clf_loaded)])
    run_mock_user_inference(loaded_pipe)


if __name__ == "__main__":
    main()
