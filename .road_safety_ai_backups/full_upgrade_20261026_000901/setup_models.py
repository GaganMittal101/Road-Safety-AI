"""Train the three Road Safety AI regressors on consistent, complete inputs.

Run manually after installing dependencies:
    python setup_models.py
Models train on 2019–2022 counts to estimate the 2023 count. The dataset has
only one target year per region, so CV describes region-level generalization,
not future-year forecasting.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.core import clean_accident_frame, prepare_training_xy

DATA = ROOT / "data" / "processed" / "risk_analysis.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FEATURES = ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
TARGET = "2023 Accidents"
MODEL_CONFIG = {
    "accident_prediction.pkl": {
        "name": "Random Forest",
        "factory": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    },
    "gb_model.pkl": {
        "name": "Gradient Boosting",
        "factory": lambda: GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        ),
    },
}


def load_training_data():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    raw = pd.read_csv(DATA)
    clean = clean_accident_frame(raw, [*FEATURES, TARGET], TARGET)
    X, y, eligible, excluded = prepare_training_xy(clean, FEATURES, TARGET)
    if len(X) < 5:
        raise ValueError(
            f"Only {len(X)} complete independent region records are available; at least five are required."
        )
    return X, y, eligible, excluded


def build_estimators():
    from xgboost import XGBRegressor

    estimators = {
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.90,
            colsample_bytree=0.90, objective="reg:squarederror", random_state=42, n_jobs=2,
        ),
    }
    return estimators


def cross_validation_report(X, y, estimators):
    cv = KFold(n_splits=min(5, max(2, len(X) // 2)), shuffle=True, random_state=42)
    models = {"Mean Baseline": DummyRegressor(strategy="mean"), **estimators}
    scoring = {
        "R2": "r2",
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(lambda a, b: float(np.sqrt(mean_squared_error(a, b))), greater_is_better=False),
    }
    rows = []
    for name, estimator in models.items():
        result = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        rows.append({
            "Model": name,
            "R2_mean": float(np.nanmean(result["test_R2"])),
            "R2_std": float(np.nanstd(result["test_R2"])),
            "MAE_mean": float((-result["test_MAE"]).mean()),
            "MAE_std": float((-result["test_MAE"]).std()),
            "RMSE_mean": float((-result["test_RMSE"]).mean()),
            "RMSE_std": float((-result["test_RMSE"]).std()),
            "MSE_mean": float((-result["test_MSE"]).mean()),
            "rows": int(len(X)),
            "folds": int(min(5, max(2, len(X) // 2))),
        })
    return pd.DataFrame(rows)


def atomic_dump(obj, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    try:
        joblib.dump(obj, tmp_name)
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    X, y, eligible, excluded = load_training_data()
    estimators = build_estimators()
    print(f"Complete training records: {len(X)}")
    if not excluded.empty:
        print("Excluded incomplete records:", ", ".join(excluded["State"].astype(str).tolist()))

    # Calculate CV results before writing any model artifacts.
    metrics = cross_validation_report(X, y, estimators)
    fitted = {}
    for name, model in estimators.items():
        print(f"Training {name} ...")
        model.fit(X, y)
        fitted[name] = model

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_paths = {
        "Random Forest": MODELS / "accident_prediction.pkl",
        "XGBoost": MODELS / "xgb_model.pkl",
        "Gradient Boosting": MODELS / "gb_model.pkl",
    }
    # Back up all existing artifacts before replacing any.
    backup_dir = MODELS / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    existing = [path for path in [*artifact_paths.values(), MODELS / "training_metadata.json"] if path.is_file()]
    for path in existing:
        target = backup_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        print(f"Backup: {target.relative_to(ROOT)}")

    for name, model in fitted.items():
        destination = artifact_paths[name]
        atomic_dump(model, destination)
        print(f"Saved: {destination.relative_to(ROOT)}")

    metadata = {
        "project": "Road Safety AI",
        "created_at_utc": timestamp,
        "features": FEATURES,
        "target": TARGET,
        "target_year": 2023,
        "training_rows": int(len(X)),
        "excluded_incomplete_regions": excluded["State"].astype(str).tolist(),
        "model_names": list(fitted),
        "preprocessing": "raw non-negative accident counts; incomplete feature rows excluded; no scaler",
        "validation_method": "shuffled K-fold cross-validation across complete state/UT records; not a temporal forecast backtest",
        "versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "validation_note": "Metrics are exploratory because the dataset is small and contains one target year per region.",
    }
    (MODELS / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics.to_csv(REPORTS / "model_evaluation.csv", index=False)
    print(f"Saved: {(MODELS / 'training_metadata.json').relative_to(ROOT)}")
    print(f"Saved: {(REPORTS / 'model_evaluation.csv').relative_to(ROOT)}")
    print("Training complete. Start with: python -m streamlit run app/app.py")


if __name__ == "__main__":
    main()
