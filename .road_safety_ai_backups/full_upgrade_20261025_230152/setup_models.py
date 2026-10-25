"""One-time model preparation for Road Safety AI.

Run this only when a model .pkl file is missing or invalid:

    python setup_models.py

This script is intentionally NOT imported by app.py — normal application
startup never trains models, it only loads what this script produces.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "risk_analysis.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "2019 Accidents",
    "2020 Accidents",
    "2021 Accidents",
    "2022 Accidents",
]
TARGET = "2023 Accidents"


def load_training_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    if "State" in df.columns:
        df = df[
            df["State"].astype(str).str.casefold() != "all india"
        ].copy()

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Dataset missing: " + ", ".join(missing))

    X = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )
    y = (
        pd.to_numeric(df[TARGET], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )
    return X, y


def main():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None
        print("XGBoost is not installed; XGBoost will be skipped.")

    X, y = load_training_data()

    # A scaler is optional at prediction time (predict.py falls back to raw
    # features if scaler.pkl is absent), but saving one keeps every model's
    # input pipeline consistent and reproducible.
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, MODELS / "scaler.pkl")
    print(f"Saved: {MODELS / 'scaler.pkl'}")

    models = {
        "accident_prediction.pkl": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "gb_model.pkl": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    if XGBRegressor is not None:
        models["xgb_model.pkl"] = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )

    print(f"Training on {len(X)} state/UT records...")

    for filename, model in models.items():
        print(f"Training {filename} ...")
        model.fit(X, y)
        output = MODELS / filename
        joblib.dump(model, output)
        print(f"Saved: {output}")

    print("\nModel preparation complete.")
    print("Start the app with:")
    print("streamlit run app/app.py")


if __name__ == "__main__":
    main()