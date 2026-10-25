"""Road Safety AI prediction utilities.

Normal application startup:
- loads saved .pkl models only;
- never trains models;
- imports SHAP only when an explanation is requested.

Run setup_models.py once to create/update model files.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
TARGET = "2023 Accidents"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_FILES = {
    "Random Forest": "accident_prediction.pkl",
    "XGBoost": "xgb_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
}


@st.cache_resource(show_spinner=False)
def get_model_bundle() -> Dict[str, object]:
    """Load saved models without training."""
    models: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            errors[name] = f"Missing file: {filename}"
            continue

        try:
            model = joblib.load(path)
            if not hasattr(model, "predict"):
                raise ValueError("Loaded object has no predict() method.")
            models[name] = model
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    scaler = None
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            errors["_scaler"] = "scaler.pkl could not be loaded."

    models["_scaler"] = scaler
    models["_errors"] = errors
    return models


def _prepare_input(y2019, y2020, y2021, y2022) -> pd.DataFrame:
    values = (
        pd.to_numeric(
            pd.Series([y2019, y2020, y2021, y2022]),
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .astype(float)
        .values
        .reshape(1, -1)
    )
    return pd.DataFrame(values, columns=FEATURES)


def _apply_scaler(X: pd.DataFrame, scaler):
    if scaler is None:
        return X
    try:
        return pd.DataFrame(
            scaler.transform(X),
            columns=FEATURES,
            index=X.index,
        )
    except Exception:
        return X


def _safe_predict(model, X: pd.DataFrame, scaler=None) -> float:
    if model is None:
        raise ValueError("Model is not loaded.")

    candidates = [X]
    scaled = _apply_scaler(X, scaler)
    if not scaled.equals(X):
        candidates.append(scaled)

    last_error = None
    for candidate in candidates:
        for payload in (candidate, candidate.to_numpy()):
            try:
                value = float(np.asarray(model.predict(payload)).reshape(-1)[0])
                if not np.isfinite(value):
                    raise ValueError("Model returned a non-finite prediction.")
                return max(0.0, value)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Prediction failed: {last_error}")


def predict_all(y2019, y2020, y2021, y2022) -> Tuple[float, float, float]:
    """Return predictions from Random Forest, XGBoost and Gradient Boosting."""
    X = _prepare_input(y2019, y2020, y2021, y2022)
    models = get_model_bundle()
    scaler = models.get("_scaler")

    values = []
    for name in ("Random Forest", "XGBoost", "Gradient Boosting"):
        model = models.get(name)
        if model is None:
            values.append(np.nan)
            continue
        try:
            values.append(float(round(_safe_predict(model, X, scaler))))
        except Exception:
            values.append(np.nan)

    return tuple(values)


def predict_national(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Sum state-level predictions to create All India estimates."""
    if "State" not in df.columns:
        raise KeyError("Dataset must contain a State column.")

    states = df[df["State"].astype(str).str.casefold() != "all india"].copy()
    if states.empty:
        return (np.nan, np.nan, np.nan)

    missing = [col for col in FEATURES if col not in states.columns]
    if missing:
        raise KeyError("Missing prediction features: " + ", ".join(missing))

    X = (
        states[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )

    models = get_model_bundle()
    scaler = models.get("_scaler")
    model_input = X
    if scaler is not None:
        try:
            model_input = pd.DataFrame(
                scaler.transform(X),
                columns=FEATURES,
                index=X.index,
            )
        except Exception:
            model_input = X

    totals = []
    for name in ("Random Forest", "XGBoost", "Gradient Boosting"):
        model = models.get(name)
        if model is None:
            totals.append(np.nan)
            continue
        try:
            predictions = np.asarray(model.predict(model_input), dtype=float)
            predictions = predictions[np.isfinite(predictions)]
            totals.append(float(round(max(0.0, float(predictions.sum())))) if predictions.size else np.nan)
        except Exception:
            totals.append(np.nan)

    return tuple(totals)


def predict_accidents(y2019, y2020, y2021, y2022):
    return predict_all(y2019, y2020, y2021, y2022)[0]


def predict_accidents_xgb(y2019, y2020, y2021, y2022):
    return predict_all(y2019, y2020, y2021, y2022)[1]


def predict_accidents_gb(y2019, y2020, y2021, y2022):
    return predict_all(y2019, y2020, y2021, y2022)[2]


@st.cache_data(show_spinner=False)
def _risk_reference_max() -> float:
    data_path = os.path.join(PROJECT_DIR, "data", "processed", "risk_analysis.csv")
    if not os.path.exists(data_path):
        return 0.0

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        return 0.0

    values = pd.to_numeric(df[TARGET], errors="coerce")
    return float(values.max()) if values.notna().any() else 0.0


def predict_risk(y2019, y2020, y2021, y2022) -> str:
    """Return a relative High/Medium/Low indicator for custom input."""
    prediction = predict_accidents(y2019, y2020, y2021, y2022)
    maximum = _risk_reference_max()

    if not np.isfinite(prediction) or maximum <= 0:
        return "No Data"

    score = float(prediction) / maximum * 100.0
    if score >= 75:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


@st.cache_data(show_spinner=False)
def get_model_metrics() -> pd.DataFrame:
    """Actual 5-fold CV metrics calculated on demand."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, cross_validate

    data_path = os.path.join(PROJECT_DIR, "data", "processed", "risk_analysis.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise KeyError("Dataset missing: " + ", ".join(missing))

    if "State" in df.columns:
        df = df[df["State"].astype(str).str.casefold() != "all india"].copy()

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0).astype(float)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).clip(lower=0).astype(float)

    if len(df) < 2:
        raise ValueError("At least 2 records are required.")

    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

    estimators = {
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    try:
        from xgboost import XGBRegressor
        estimators["XGBoost"] = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )
    except Exception:
        pass

    scoring = {
        "R2": "r2",
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
            greater_is_better=False,
        ),
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
    }

    rows = []
    for name, estimator in estimators.items():
        result = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        rows.append({
            "Model": name,
            "R²": float(result["test_R2"].mean()),
            "R² Std": float(result["test_R2"].std()),
            "MAE": float((-result["test_MAE"]).mean()),
            "MAE Std": float((-result["test_MAE"]).std()),
            "RMSE": float((-result["test_RMSE"]).mean()),
            "RMSE Std": float((-result["test_RMSE"]).std()),
            "MSE": float((-result["test_MSE"]).mean()),
        })

    return pd.DataFrame(rows).set_index("Model")


def explain_prediction(selected_data):
    """Return (figure, contribution dataframe, plain-language summary)."""
    try:
        import shap
        import plotly.graph_objects as go
    except ImportError:
        return None

    try:
        model = get_model_bundle().get("Random Forest")
        if model is None:
            return None

        row = selected_data if isinstance(selected_data, pd.Series) else selected_data.iloc[0]
        X = _prepare_input(
            row.get("2019 Accidents"),
            row.get("2020 Accidents"),
            row.get("2021 Accidents"),
            row.get("2022 Accidents"),
        )

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values).reshape(-1)

        labels = ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
        shap_df = pd.DataFrame({"Feature": labels, "Contribution": values})
        shap_df["Direction"] = np.where(
            shap_df["Contribution"] >= 0,
            "Increases prediction",
            "Decreases prediction",
        )
        shap_df["AbsContribution"] = shap_df["Contribution"].abs()
        shap_df = shap_df.sort_values("Contribution").reset_index(drop=True)

        fig = go.Figure(
            go.Bar(
                x=shap_df["Contribution"],
                y=shap_df["Feature"].str.replace(" Accidents", "", regex=False),
                orientation="h",
                marker_color=[
                    "#F43F5E" if v >= 0 else "#38BDF8"
                    for v in shap_df["Contribution"]
                ],
                text=[f"{v:+.1f}" for v in shap_df["Contribution"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="How historical accident values influenced the prediction",
            xaxis_title="SHAP contribution",
            yaxis_title="Historical year",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
            margin=dict(l=20, r=60, t=60, b=30),
        )

        strongest = shap_df.iloc[shap_df["AbsContribution"].idxmax()]
        direction = "increased" if strongest["Contribution"] >= 0 else "decreased"
        summary = (
            f"{strongest['Feature']} had the strongest influence and {direction} "
            f"the model output by approximately "
            f"{abs(float(strongest['Contribution'])):,.1f} model units."
        )
        return fig, shap_df, summary
    except Exception:
        return None


def load_models():
    bundle = get_model_bundle()
    return {
        "rf": bundle.get("Random Forest"),
        "xgb": bundle.get("XGBoost"),
        "gb": bundle.get("Gradient Boosting"),
        "scaler": bundle.get("_scaler"),
    }