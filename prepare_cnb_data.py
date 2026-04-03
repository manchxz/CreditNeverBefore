#!/usr/bin/env python3
"""
Build CreditNeverBefore training CSV: Home Credit base + injected NTC behavioral markers.

Run from repo root or any cwd; resolves paths relative to this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INPUT_CSV = DATA_DIR / "application_train.csv"
OUTPUT_CSV = DATA_DIR / "cnb_training_data.csv"


def inject_behavioral_data(row: pd.Series) -> pd.Series:
    """Synthetic NTC signals aligned with TARGET (demo / research use only)."""
    if row["TARGET"] == 1:
        # Add noise so the TARGET-conditioned ranges overlap (prevents trivial leakage)
        upi_velocity = np.random.uniform(0.1, 0.7)
        bill_pay_consistency = np.random.uniform(0.2, 0.7)
        app_usage_days = np.random.randint(1, 22)
    else:
        # Add noise so good users can still have low metrics, and vice-versa
        upi_velocity = np.random.uniform(0.4, 1.0)
        bill_pay_consistency = np.random.uniform(0.5, 1.0)
        app_usage_days = np.random.randint(10, 31)

    return pd.Series(
        {
            "UPI_VELOCITY": upi_velocity,
            "BILL_PAY_CONSISTENCY": bill_pay_consistency,
            "APP_USAGE_DAYS": app_usage_days,
        }
    )


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    df["AGE"] = df["DAYS_BIRTH"] / -365.0

    np.random.seed(42)
    injected = df.apply(inject_behavioral_data, axis=1)
    df = pd.concat([df, injected], axis=1)

    df["EXT_SOURCE_1"] = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median())

    df_final = df[
        [
            "TARGET",
            "AGE",
            "EXT_SOURCE_1",
            "UPI_VELOCITY",
            "BILL_PAY_CONSISTENCY",
            "APP_USAGE_DAYS",
        ]
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("CNB Dataset Prepared! Rows:", len(df_final))
    print(df_final.head())


if __name__ == "__main__":
    main()
