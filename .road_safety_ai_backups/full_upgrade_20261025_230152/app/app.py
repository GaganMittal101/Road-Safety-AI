from __future__ import annotations

import datetime as dt
import html
import io
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "risk_analysis.csv"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
DB_PATH = ROOT_DIR / "database.db"

for path in (APP_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(
    page_title="Road Safety AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
DEFAULTS = {
    "analysis_state": "All India",
    "selected_page": "Dashboard",
    "custom_result": None,
    "clear_db_confirm": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------------
# PREMIUM UI
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
    --bg:#F5E8D8;
    --panel:#FFFCF5;
    --panel2:#F0E1C8;
    --line:rgba(29,42,68,.16);
    --text:#1D2A44;
    --muted:#59677D;
    --blue:#0047AB;
    --sun:#FFD447;
    --green:#15803D;
    --amber:#B45309;
    --red:#C62828;
}

html, body, .stApp, [class*="css"]{
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
.stApp{background:var(--bg);color:var(--text);}
.main .block-container{max-width:1500px;padding-top:1rem;padding-bottom:2rem;}
h1,h2,h3,h4,h5{color:var(--text)!important;letter-spacing:-.025em;}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#EAD9BC,#F5E8D8);
    border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] > div{padding-top:1rem;}

div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea{
    background:#FFFCF5!important;
    color:#1D2A44!important;
    border:1px solid #D4C3A6!important;
    border-radius:14px!important;
}
div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus{
    border-color:var(--blue)!important;
    box-shadow:0 0 0 3px rgba(0,71,171,.14)!important;
}
label{color:#536079!important;font-weight:600!important;}

.stButton > button,.stDownloadButton > button{
    border:0!important;
    border-radius:14px!important;
    color:#FFFFFF!important;
    font-weight:700!important;
    background:linear-gradient(135deg,#0047AB,#1B62C5)!important;
    box-shadow:0 8px 25px rgba(0,71,171,.18)!important;
    min-height:44px;
    transition:transform .12s ease, filter .12s ease;
}
.stButton > button:hover,.stDownloadButton > button:hover{
    filter:brightness(1.06);
    transform:translateY(-1px);
    outline:2px solid #FFD447!important;
    outline-offset:1px;
}

[data-testid="stMetric"]{
    background:linear-gradient(145deg,#FFFCF5,#F4E6D1)!important;
    border:1px solid var(--line)!important;
    border-radius:18px!important;
    padding:1rem!important;
    box-shadow:0 8px 24px rgba(71,53,22,.06);
}
[data-testid="stMetricLabel"]{color:#59677D!important;}
[data-testid="stMetricValue"]{color:#1D2A44!important;font-weight:800!important;}

.card{
    background:linear-gradient(145deg,#FFFCF5,#F1E2CA);
    border:1px solid var(--line);
    border-radius:22px;
    padding:20px;
    box-shadow:0 18px 50px rgba(71,53,22,.09);
}
.explain-card,.scenario-card,.reliability-card{
    background:linear-gradient(145deg,#FFFCF5,#F8EDD9);
    border:1px solid rgba(0,71,171,.16);
    border-radius:18px;
    padding:16px 18px;
    box-shadow:0 10px 30px rgba(71,53,22,.07);
    color:#1D2A44;
}
.scenario-card{border-color:rgba(255,212,71,.9);}
.footer{
    margin-top:42px;padding:18px 0;border-top:1px solid var(--line);
    text-align:center;color:#687184;font-size:.82rem;
}

.status{
    display:flex;align-items:center;gap:10px;padding:15px 17px;
    border-radius:17px;background:rgba(255,252,245,.82);
    border:1px solid var(--line);
}
.status-ready{color:#15803D;}
.status-warn{color:#B45309;}
.status-error{color:#C62828;}
.status-value{margin-left:auto;font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------
ACCIDENT_COLS = [f"{year} Accidents" for year in range(2019, 2024)]
FEATURE_COLS = [f"{year} Accidents" for year in range(2019, 2023)]
TARGET_COL = "2023 Accidents"


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    """Load and validate the project dataset once."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Expected file: data/processed/risk_analysis.csv"
        )

    df = pd.read_csv(DATA_PATH)

    if "State" not in df.columns:
        raise ValueError("Dataset must contain a 'State' column.")

    df["State"] = (
        df["State"]
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown"})
    )

    for col in ACCIDENT_COLS:
        if col not in df.columns:
            raise ValueError(f"Dataset is missing required column: {col}")
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )

    # Always recompute the displayed score from the current data.
    # This prevents stale/hard-coded risk scores from being trusted.
    max_2023 = float(df[TARGET_COL].max())
    if max_2023 > 0:
        df["Risk Score"] = (
            df[TARGET_COL] / max_2023 * 100
        ).clip(0, 100)
    else:
        df["Risk Score"] = 0.0

    df["Risk Category"] = np.select(
        [df["Risk Score"] >= 75, df["Risk Score"] >= 40],
        ["High", "Medium"],
        default="Low",
    )

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def region_names() -> list[str]:
    df = load_processed_data()
    states = sorted(
        {
            str(x).strip()
            for x in df["State"].dropna().tolist()
            if str(x).strip() and str(x).strip().casefold() != "all india"
        },
        key=str.casefold,
    )
    return ["All India"] + states


@st.cache_data(show_spinner=False)
def all_india_record() -> pd.Series:
    """Create a true national aggregate from state/UT rows."""
    df = load_processed_data()
    states = df[
        df["State"].astype(str).str.casefold() != "all india"
    ].copy()

    row = {"State": "All India"}
    for col in ACCIDENT_COLS:
        row[col] = float(states[col].sum())

    # National risk is deliberately not forced into High/Medium/Low.
    # Those categories are designed for state/UT relative risk.
    row["Risk Score"] = np.nan
    row["Risk Category"] = "National Overview"
    return pd.Series(row)


def state_record(df: pd.DataFrame, state: str) -> pd.Series:
    if state == "All India":
        return all_india_record()

    matches = df[df["State"].astype(str) == state]
    if matches.empty:
        return all_india_record()
    return matches.iloc[0]


def safe_num(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def fmt_int(value) -> str:
    x = safe_num(value)
    return "—" if not np.isfinite(x) else f"{int(round(x)):,}"


def fmt_score(value) -> str:
    x = safe_num(value)
    return "—" if not np.isfinite(x) else f"{x:.1f}"


def risk_badge(risk: str) -> str:
    return {
        "High": "🔴 High",
        "Medium": "🟠 Medium",
        "Low": "🟢 Low",
        "National Overview": "🔵 National",
        "No Data": "⚪ No Data",
    }.get(risk, "⚪ No Data")


def recommendations(risk: str) -> list[str]:
    if risk == "High":
        return [
            "Prioritize high-risk corridors and accident-prone locations.",
            "Increase targeted enforcement and speed monitoring.",
            "Review signage, junction design and road-engineering controls.",
            "Strengthen emergency response readiness.",
        ]
    if risk == "Medium":
        return [
            "Increase monitoring on high-incident routes.",
            "Review signage, lighting and engineering controls.",
            "Run targeted road-safety awareness campaigns.",
            "Track trends for early intervention.",
        ]
    if risk == "National Overview":
        return [
            "Use the state/UT ranking to prioritize regional interventions.",
            "Monitor national accident trends and emerging state-level hotspots.",
            "Combine model insights with official road-safety and traffic data.",
        ]
    return [
        "Maintain preventive road-safety measures.",
        "Continue periodic monitoring of accident trends.",
        "Promote road-user awareness and preventive maintenance.",
    ]


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
MODEL_FILES = {
    "Random Forest": "accident_prediction.pkl",
    "XGBoost": "xgb_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
}


@st.cache_resource(show_spinner=False)
def load_saved_models() -> Dict[str, object]:
    """
    Load only saved models.

    IMPORTANT:
    No model is trained here. This keeps normal application startup fast.
    If a model file is missing, Settings shows it as unavailable.
    Use setup_models.py once to create missing model files.
    """
    models: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            errors[name] = f"Missing file: {filename}"
            continue

        try:
            model = joblib.load(path)
            if not hasattr(model, "predict"):
                raise ValueError("Loaded object does not expose predict().")
            models[name] = model
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    scaler = None
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            errors["_scaler"] = "scaler.pkl could not be loaded."

    models["_scaler"] = scaler
    models["_errors"] = errors
    return models


def prepare_features(values: Iterable[float]) -> pd.DataFrame:
    numbers = []
    for value in values:
        x = safe_num(value, 0.0)
        numbers.append(max(0.0, x))
    return pd.DataFrame([numbers], columns=FEATURE_COLS)


def _scaled_input(X: pd.DataFrame, scaler):
    if scaler is None:
        return X
    try:
        return pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS, index=X.index)
    except Exception:
        return X


def safe_model_predict(model, X: pd.DataFrame, scaler=None) -> float:
    if model is None:
        raise ValueError("Model is not loaded.")

    candidates = [X]
    scaled = _scaled_input(X, scaler)
    if not scaled.equals(X):
        candidates.append(scaled)

    last_error = None
    for candidate in candidates:
        for payload in (candidate, candidate.to_numpy()):
            try:
                prediction = float(
                    np.asarray(model.predict(payload)).reshape(-1)[0]
                )
                if not np.isfinite(prediction):
                    raise ValueError("Model returned a non-finite prediction.")
                return max(0.0, prediction)
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Prediction failed: {last_error}")


def predict_all(values: Iterable[float]) -> Dict[str, float]:
    models = load_saved_models()
    scaler = models.get("_scaler")
    X = prepare_features(values)
    output: Dict[str, float] = {}

    for name in MODEL_FILES:
        model = models.get(name)
        if model is None:
            continue
        try:
            output[name] = round(safe_model_predict(model, X, scaler))
        except Exception:
            continue

    if not output:
        raise RuntimeError(
            "No saved prediction model is available. "
            "Open Settings to see model diagnostics."
        )

    return output


def predict_national(df: pd.DataFrame) -> Dict[str, float]:
    """Sum state-level predictions to create an All India prediction."""
    states = df[
        df["State"].astype(str).str.casefold() != "all india"
    ].copy()

    if states.empty:
        return {}

    X = (
        states[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )

    models = load_saved_models()
    # setup_models.py trains these regressors on raw accident counts.
    # Keep inference preprocessing identical to training; applying a scaler
    # only at national-prediction time would change the model's input scale.
    model_input = X
    output: Dict[str, float] = {}

    for name in MODEL_FILES:
        model = models.get(name)
        if model is None:
            continue
        try:
            values = np.asarray(model.predict(model_input), dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                output[name] = round(max(0.0, float(values.sum())))
        except Exception:
            continue

    return output


def state_prediction(df: pd.DataFrame, state: str, record: pd.Series) -> Dict[str, float]:
    if state == "All India":
        return predict_national(df)

    values = [record.get(col, 0) for col in FEATURE_COLS]
    return predict_all(values)


def ensemble_summary(predictions: Dict[str, float]) -> dict:
    values = [
        float(v)
        for v in predictions.values()
        if np.isfinite(safe_num(v))
    ]

    if not values:
        return {
            "average": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "spread": np.nan,
            "agreement": np.nan,
        }

    avg = float(np.mean(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    spread = maximum - minimum

    # Agreement is a simple consistency indicator, NOT a statistical
    # confidence probability.
    agreement = (
        max(0.0, 100.0 - (spread / avg * 100.0))
        if avg > 0
        else 0.0
    )

    return {
        "average": avg,
        "minimum": minimum,
        "maximum": maximum,
        "spread": spread,
        "agreement": agreement,
    }


def state_risk(record: pd.Series, state: str) -> Tuple[float, str]:
    if state == "All India":
        return np.nan, "National Overview"

    score = safe_num(record.get("Risk Score"))
    if not np.isfinite(score):
        return np.nan, "No Data"

    if score >= 75:
        return score, "High"
    if score >= 40:
        return score, "Medium"
    return score, "Low"


def custom_risk(prediction: float, df: pd.DataFrame) -> Tuple[float, str]:
    """
    Relative indicator for custom inputs.

    This is deliberately labelled as a relative risk indicator, because
    custom inputs are user-created scenarios rather than observed state data.
    """
    maximum = float(df[TARGET_COL].max())
    if maximum <= 0:
        return np.nan, "No Data"

    score = float(np.clip(prediction / maximum * 100.0, 0, 100))

    if score >= 75:
        return score, "High"
    if score >= 40:
        return score, "Medium"
    return score, "Low"


# ---------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            year INTEGER NOT NULL,
            predicted_accidents REAL NOT NULL,
            model TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def db_count() -> int:
    try:
        init_db()
        conn = get_db_connection()
        value = int(
            conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        )
        conn.close()
        return value
    except Exception:
        return 0


def log_prediction(state: str, year: int, prediction: float, model: str):
    init_db()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions
        (state, year, predicted_accidents, model, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            state,
            int(year),
            float(prediction),
            model,
            dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def clear_database():
    conn = get_db_connection()
    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.commit()
    conn.close()
    init_db()


# ---------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------
def transparent_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#34425E"),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def trend_chart(record: pd.Series, state: str):
    import plotly.express as px

    years = list(range(2019, 2024))
    values = [safe_num(record.get(f"{year} Accidents"), 0) for year in years]

    data = pd.DataFrame({"Year": years, "Accidents": values})

    fig = px.line(
        data,
        x="Year",
        y="Accidents",
        markers=True,
        title=f"{state} — Historical Accident Trend",
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(hovermode="x unified")
    return transparent_layout(fig)


def ranking_table(df: pd.DataFrame):
    x = df[df["State"].astype(str).str.casefold() != "all india"].copy()

    x["2023 Accidents"] = pd.to_numeric(x["2023 Accidents"], errors="coerce")
    x["Risk Score"] = pd.to_numeric(x["Risk Score"], errors="coerce")

    x = x.sort_values(
        ["Risk Score", "2023 Accidents"],
        ascending=[False, False],
        na_position="last",
    )

    x.insert(0, "Rank", range(1, len(x) + 1))
    return x[["Rank", "State", "2023 Accidents", "Risk Score", "Risk Category"]]


def correlation_chart(df: pd.DataFrame):
    import plotly.express as px

    cols = [c for c in ACCIDENT_COLS if c in df.columns]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=-1,
        zmax=1,
        title="Historical Accident Correlation",
    )
    return transparent_layout(fig)


# ---------------------------------------------------------------------
# SINGLE SEARCH + SELECT CONTROL
# ---------------------------------------------------------------------
def render_region_control(df: pd.DataFrame, page: str) -> Tuple[str, pd.Series]:
    """
    One control only.

    Streamlit selectbox is searchable by typing:
    - type r -> matching state names
    - type raj -> Rajasthan
    - type kar -> Karnataka
    - All India stays at the top
    """
    if page == "Settings":
        return "All India", all_india_record()

    options = region_names()
    current = st.session_state.get("analysis_state", "All India")
    index = options.index(current) if current in options else 0

    state = st.selectbox(
        "📍 Search or select a state",
        options=options,
        index=index,
        key="analysis_state_control",
        help=(
            "Click the box and type to search. "
            "Example: r → Rajasthan, raj → Rajasthan, "
            "kar → Karnataka. All India is always first."
        ),
    )

    st.session_state.analysis_state = state
    st.caption("⌕ Type to search • All India is always at the top")

    return state, state_record(df, state)


# ---------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------
def build_forecast(base_prediction: float, growth_percent: int):
    """Build an illustrative 2024–2035 scenario from observed 2023 counts.

    The first forecast year (2024) applies one full year of the growth rate
    to the 2023 observed baseline; this is a scenario, not a trained forecast.
    """
    years = list(range(2024, 2036))
    base = max(0.0, safe_num(base_prediction, 0.0))
    rate = float(growth_percent) / 100.0
    values = [base * ((1.0 + rate) ** step) for step in range(1, len(years) + 1)]

    return pd.DataFrame(
        {"Year": years, "Projected Accidents": np.rint(values).astype(int)}
    )


def render_forecast(state: str, observed_2023: float):
    import plotly.express as px

    st.title("📈 Scenario Forecast — 2024 to 2035")
    st.caption(
        "An illustrative planning scenario based on the observed 2023 accident count. "
        "It is not a trained future-year prediction."
    )

    baseline = safe_num(observed_2023, 0.0)
    if not np.isfinite(baseline) or baseline < 0:
        st.warning("A valid observed 2023 accident count is required for this scenario.")
        return

    growth_percent = st.slider(
        "Annual growth assumption",
        min_value=-5,
        max_value=5,
        value=0,
        step=1,
        format="%d%%",
        help=(
            "−5% means a 5% annual decrease. "
            "0% means no annual change. "
            "+5% means a 5% annual increase. "
            "This is a scenario, not an official forecast."
        ),
    )

    forecast_df = build_forecast(baseline, growth_percent)
    first_projection = float(forecast_df.iloc[0]["Projected Accidents"])
    final = float(forecast_df.iloc[-1]["Projected Accidents"])
    total = int(forecast_df["Projected Accidents"].sum())
    overall_change = ((final / baseline) - 1) * 100 if baseline > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2023 Observed Base", f"{baseline:,.0f}")
    c2.metric("2035 Scenario", f"{final:,.0f}")
    c3.metric("12-Year Projected Total", f"{total:,.0f}")
    c4.metric("Change vs. 2023", f"{overall_change:+.1f}%")

    if growth_percent > 0:
        interpretation = f"the scenario assumes accidents increase by {growth_percent}% every year."
    elif growth_percent < 0:
        interpretation = f"the scenario assumes accidents decrease by {abs(growth_percent)}% every year."
    else:
        interpretation = "the scenario assumes no annual change from the 2023 observed baseline."

    st.markdown(
        f"""
        <div class="scenario-card">
            <b>Scenario interpretation</b><br><br>
            Starting from the observed 2023 baseline of
            <b>{baseline:,.0f}</b> accidents, {html.escape(interpretation)}
            The 2024 scenario value is <b>{first_projection:,.0f}</b> and the
            2035 scenario value is <b>{final:,.0f}</b>.
            <br><br>
            <span style="color:#59677D">
            ⚠️ This is a mathematical planning scenario, not a validated
            prediction. It does not model future changes in traffic,
            infrastructure, weather, enforcement, or reporting.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart = px.line(
        forecast_df,
        x="Year",
        y="Projected Accidents",
        markers=True,
        title=f"{state} — 2024–2035 Scenario",
    )
    chart.update_traces(line=dict(width=3), marker=dict(size=8))
    chart.update_layout(hovermode="x unified")
    st.plotly_chart(transparent_layout(chart), use_container_width=True)

    with st.expander("ℹ️ How the scenario is calculated"):
        st.code(
            "Projected Year = 2023 Observed Base × (1 + annual growth)^(Year − 2023)",
            language="text",
        )
        st.write(f"Annual assumption: **{growth_percent:+d}%**")
        st.write("Base: observed 2023 count • Horizon: 2024–2035 (12 years)")

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name=f"road_safety_forecast_{state.replace(' ', '_')}_2024_2035.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# RISK MAP
# ---------------------------------------------------------------------
STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Andaman & Nicobar Islands": (11.7401, 92.6586),
    "Andaman and Nicobar Islands": (11.7401, 92.6586),
    "Chandigarh": (30.7333, 76.7794),
    "Dadra & Nagar Haveli*": (20.1809, 73.0169),
    "Dadra and Nagar Haveli and Daman and Diu": (20.1809, 73.0169),
    "Daman & Diu": (20.43, 72.84),
    "Delhi": (28.7041, 77.1025),
    "J & K #": (33.7782, 76.5762),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771),
    "Lakshadweep": (10.5667, 72.6417),
    "Puducherry": (11.9416, 79.8083),
}


def normalize_map_name(name: str) -> str:
    value = str(name).strip().casefold().replace("&", "and")

    aliases = {
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "jammu and kashmir": "j & k #",
        "andaman and nicobar islands": "andaman & nicobar islands",
        "dadra and nagar haveli": "dadra & nagar haveli*",
        "daman and diu": "daman & diu",
    }
    return aliases.get(value, value)


def coords_for(state: str):
    if state in STATE_COORDS:
        return STATE_COORDS[state]

    target = normalize_map_name(state)
    for name, coords in STATE_COORDS.items():
        if normalize_map_name(name) == target:
            return coords

    return None


@st.cache_data(show_spinner=False)
def load_geojson():
    """Load India state boundaries using either historic filename spelling."""
    candidates = (
        ASSETS_DIR / "india_states.geojson",
        ASSETS_DIR / "india_state.geojson",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            geo = json.load(f)
        if not isinstance(geo, dict) or not isinstance(geo.get("features"), list):
            return None
        return geo
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def geo_feature_name(feature) -> str:
    props = feature.get("properties", {})
    for key in ("ST_NM", "STATE", "State", "state", "NAME_1", "name", "NAME"):
        value = props.get(key)
        if value:
            return str(value)
    return ""


def render_risk_map(df: pd.DataFrame):
    import folium
    from folium.plugins import Fullscreen, MiniMap
    from streamlit_folium import st_folium

    st.title("🗺️ India Risk Intelligence Map")
    st.caption(
        "State/UT relative accident-count categories based on supplied 2023 data. "
        "These are not exposure-adjusted crash rates."
    )

    data = df[df["State"].astype(str).str.casefold() != "all india"].copy()
    categories = data["Risk Category"].astype(str)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regions Mapped", len(data))
    c2.metric("🔴 High Risk", int((categories == "High").sum()))
    c3.metric("🟠 Medium Risk", int((categories == "Medium").sum()))
    c4.metric("🟢 Low Risk", int((categories == "Low").sum()))

    risk_colors = {
        "High": "#EF4444",
        "Medium": "#F59E0B",
        "Low": "#10B981",
        "No Data": "#94A3B8",
    }

    # Light, readable base map. Avoids unofficial tile-scraping assumptions
    # and is reliable for a local project demo.
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        min_zoom=4,
        max_zoom=8,
        control_scale=True,
        tiles="CartoDB positron",
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)

    geo = load_geojson()

    lookup = {
        normalize_map_name(row["State"]): str(row.get("Risk Category", "No Data"))
        for _, row in data.iterrows()
    }

    if geo:
        def geo_risk(feature):
            name = geo_feature_name(feature)
            return lookup.get(normalize_map_name(name), "No Data")

        folium.GeoJson(
            geo,
            name="Risk Boundaries",
            style_function=lambda feature: {
                "fillColor": risk_colors.get(geo_risk(feature), "#CBD5E1"),
                "color": "#64748B",
                "weight": 0.8,
                "fillOpacity": 0.30,
            },
            highlight_function=lambda feature: {
                "weight": 2.5,
                "color": "#2563EB",
                "fillOpacity": 0.45,
            },
        ).add_to(m)

    marker_count = 0

    for _, row in data.iterrows():
        state = str(row["State"])
        coords = coords_for(state)

        if coords is None:
            continue

        risk = str(row.get("Risk Category", "No Data"))
        color = risk_colors.get(risk, "#94A3B8")
        score = safe_num(row.get("Risk Score"))
        accidents = safe_num(row.get("2023 Accidents"))

        score_text = "—" if not np.isfinite(score) else f"{score:.1f}/100"
        accident_text = "—" if not np.isfinite(accidents) else f"{int(accidents):,}"

        popup_html = (
            "<div style='min-width:220px;color:#111827;font-family:Arial'>"
            f"<div style='font-size:17px;font-weight:700;margin-bottom:8px'>"
            f"{html.escape(state)}</div>"
            f"<div><b>Risk:</b> <span style='color:{color}'>{html.escape(risk)}</span></div>"
            f"<div><b>Risk Score:</b> {score_text}</div>"
            f"<div><b>2023 Accidents:</b> {accident_text}</div>"
            "</div>"
        )

        folium.CircleMarker(
            location=coords,
            radius=9 if risk == "High" else 8 if risk == "Medium" else 7,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"{state} • {risk}",
            popup=folium.Popup(popup_html, max_width=330),
        ).add_to(m)

        marker_count += 1

    Fullscreen(position="topright", title="Full screen", title_cancel="Exit full screen").add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(m, width=None, height=700, returned_objects=[])

    st.caption(
        f"Mapped markers: {marker_count}/{len(data)} • 🔴 High  •  🟠 Medium  •  🟢 Low"
    )

    if marker_count < len(data):
        st.warning(
            "Some dataset regions do not have a matching coordinate entry. "
            "Their boundary may still appear through GeoJSON."
        )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
def render_shap(record: pd.Series, state: str):
    st.subheader("🧠 Explainable AI — SHAP")
    st.caption(
        "SHAP shows which historical accident values pushed the Random Forest "
        "prediction higher or lower."
    )

    if state == "All India":
        st.info(
            "SHAP is shown for a specific state/UT. "
            "The All India prediction is an aggregation of state-level predictions."
        )
        return

    if not st.button("🔍 Explain Why This Prediction Was Made", key=f"shap_{state}"):
        return

    try:
        from predict import explain_prediction

        with st.spinner("Calculating feature contributions..."):
            result = explain_prediction(record)

        if not result:
            st.warning("SHAP is unavailable. Check the SHAP installation and Random Forest model.")
            return

        fig, shap_df, summary = result

        st.plotly_chart(fig, use_container_width=True)

        strongest = shap_df.iloc[shap_df["AbsContribution"].argmax()]
        direction = "increased" if strongest["Contribution"] >= 0 else "decreased"

        st.success(
            f"**Simple explanation:** "
            f"{strongest['Feature']} had the strongest influence. "
            f"It {direction} the model output by approximately "
            f"{abs(float(strongest['Contribution'])):,.1f} model units."
        )

        with st.expander("ℹ️ How to read SHAP"):
            st.markdown(
                """
**Positive contribution** → pushes the prediction higher.

**Negative contribution** → pushes the prediction lower.

**Larger absolute value** → stronger influence.

SHAP explains the model's behaviour; it does **not** prove that a
historical year caused the accidents.
"""
            )

    except Exception as exc:
        st.error(f"SHAP explanation failed: {exc}")


# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_model_metrics(data_signature: Tuple) -> pd.DataFrame:
    """
    Calculate actual 5-fold CV metrics on demand.

    cross_validate is used rather than computing metrics from
    cross_val_predict, since scikit-learn notes that cross_val_predict
    predictions are primarily diagnostic and can give misleading
    generalization metrics.
    """
    del data_signature

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, cross_validate

    # The All India row duplicates the sum of state/UT observations.
    # Keep only independent state/UT rows during model evaluation.
    df = load_processed_data()
    df = df[df["State"].astype(str).str.casefold() != "all india"].copy()
    if len(df) < 2:
        raise ValueError("At least two state/UT records are required for cross-validation.")

    X = (
        df[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(float)
    )
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).clip(lower=0).astype(float)

    n_splits = min(5, len(df))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False,
        ),
    }

    rows = []

    for name, estimator in estimators.items():
        result = cross_validate(
            estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
        )

        mae_scores = -result["test_MAE"]
        mse_scores = -result["test_MSE"]
        rmse_scores = -result["test_RMSE"]

        rows.append({
            "Model": name,
            "R²": float(result["test_R2"].mean()),
            "R² Std": float(result["test_R2"].std()),
            "MAE": float(mae_scores.mean()),
            "MAE Std": float(mae_scores.std()),
            "RMSE": float(rmse_scores.mean()),
            "RMSE Std": float(rmse_scores.std()),
            "MSE": float(mse_scores.mean()),
        })

    return pd.DataFrame(rows).set_index("Model")


def render_model_evaluation(df: pd.DataFrame):
    import plotly.graph_objects as go

    st.title("📊 Model Performance & Reliability")
    st.caption(
        "Five-fold cross-validation across state/UT records, excluding the duplicate All India aggregate. "
        "This is region-level validation, not a temporal backtest of future years. "
        "Higher R² is better; lower MAE/RMSE/MSE are better."
    )

    signature = tuple(df[FEATURE_COLS + [TARGET_COL]].itertuples(index=False, name=None))

    try:
        metrics = calculate_model_metrics(signature)
    except Exception as exc:
        st.error(f"Model evaluation failed: {exc}")
        return

    if metrics.empty:
        st.warning("No evaluation results are available.")
        return

    best_r2 = metrics["R²"].idxmax()
    best_mae = metrics["MAE"].idxmin()
    best_rmse = metrics["RMSE"].idxmin()

    c1, c2, c3 = st.columns(3)
    c1.metric("Best R²", best_r2, f"{metrics.loc[best_r2, 'R²']:.3f}")
    c2.metric("Lowest MAE", best_mae, f"{metrics.loc[best_mae, 'MAE']:,.0f}")
    c3.metric("Lowest RMSE", best_rmse, f"{metrics.loc[best_rmse, 'RMSE']:,.0f}")

    display = metrics.reset_index().rename(columns={"index": "Model"})

    st.dataframe(
        display.style.format({
            "R²": "{:.3f}",
            "R² Std": "{:.3f}",
            "MAE": "{:,.0f}",
            "MAE Std": "{:,.0f}",
            "RMSE": "{:,.0f}",
            "RMSE Std": "{:,.0f}",
            "MSE": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    a, b = st.columns(2)

    with a:
        fig = go.Figure(
            go.Bar(x=metrics.index, y=metrics["R²"], text=metrics["R²"].round(3), textposition="auto")
        )
        fig.update_layout(title="R² — Higher is Better", yaxis_title="R²")
        st.plotly_chart(transparent_layout(fig), use_container_width=True)

    with b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics.index, y=metrics["MAE"], name="MAE"))
        fig.add_trace(go.Bar(x=metrics.index, y=metrics["RMSE"], name="RMSE"))
        fig.update_layout(title="Error Metrics — Lower is Better", barmode="group", yaxis_title="Error")
        st.plotly_chart(transparent_layout(fig), use_container_width=True)

    st.markdown(
        """
<div class="explain-card">
<b>How to understand these numbers</b><br><br>
<b>R²:</b> how much variation in the target is explained by the model.
Closer to 1 is generally better.<br>
<b>MAE:</b> average absolute prediction error. Lower is better.<br>
<b>RMSE:</b> error metric that penalizes larger mistakes more strongly.
Lower is better.<br>
<b>R² Std / MAE Std / RMSE Std:</b> variation across the five validation
folds. Smaller variation generally means more stable validation results.
</div>
""",
        unsafe_allow_html=True,
    )

    st.warning(
        "Reliability note: these metrics describe performance on this "
        "historical state-level dataset. They are not a guarantee of future "
        "real-world accident counts."
    )


# ---------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------
def build_pdf_bytes(
    state: str,
    record: pd.Series,
    predictions: Dict[str, float],
    risk_score: float,
    risk_level: str,
    forecast_df: pd.DataFrame,
    metrics: Optional[pd.DataFrame],
) -> bytes:
    """Build a self-contained PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "LuxuryTitle", parent=styles["Title"], fontSize=22, leading=26,
        textColor=colors.HexColor("#0047AB"), spaceAfter=8,
    )
    heading = ParagraphStyle(
        "LuxuryHeading", parent=styles["Heading2"], fontSize=14, leading=18,
        textColor=colors.HexColor("#0047AB"), spaceBefore=8, spaceAfter=6,
    )
    body = ParagraphStyle(
        "LuxuryBody", parent=styles["BodyText"], fontSize=9.5, leading=14,
        textColor=colors.HexColor("#26334B"),
    )

    story = [
        Paragraph("Road Safety AI", title),
        Paragraph("AI-Powered Road Accident Prediction & Risk Assessment Report", styles["Heading2"]),
        Paragraph(
            f"Analysis region: <b>{html.escape(state)}</b> | "
            f"Generated: {dt.datetime.now().strftime('%d %B %Y, %H:%M')}",
            body,
        ),
        Spacer(1, 8),
    ]

    story.append(Paragraph("1. Executive Summary", heading))

    risk_text = "National overview" if state == "All India" else f"{risk_level} risk"

    story.append(Paragraph(
        f"This assessment covers <b>{html.escape(state)}</b>. "
        f"The system identifies the region as <b>{risk_text}</b>. "
        "The results are analytical estimates based on historical accident "
        "data and saved machine-learning models.",
        body,
    ))

    story.append(Paragraph("2. Historical Accident Data", heading))

    history = [["Year", "Accidents"]]
    for year in range(2019, 2024):
        history.append([str(year), f"{safe_num(record.get(f'{year} Accidents'), 0):,.0f}"])

    table = Table(history, colWidths=[45 * mm, 60 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0047AB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8C6A7")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFCF5"), colors.HexColor("#F5E8D8")]),
    ]))

    story.extend([table, Spacer(1, 8)])

    story.append(Paragraph("3. 2023 Model Estimates (model target year)", heading))

    prediction_rows = [["Model", "Estimated Accidents"]]
    for name, value in predictions.items():
        prediction_rows.append([name, f"{safe_num(value, 0):,.0f}"])

    summary = ensemble_summary(predictions)
    if np.isfinite(summary["average"]):
        prediction_rows.append(["Ensemble Average", f"{summary['average']:,.0f}"])

    pt = Table(prediction_rows, colWidths=[80 * mm, 55 * mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0047AB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8C6A7")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFCF5"), colors.HexColor("#F5E8D8")]),
    ]))

    story.extend([pt, Spacer(1, 8)])

    story.append(Paragraph("4. 2024–2035 Scenario Projection", heading))
    story.append(Paragraph(
        "This report uses the observed 2023 accident count as its baseline "
        "and assumes 0% annual growth for the 2024–2035 scenario. It is a "
        "mathematical planning scenario, not an official government forecast.",
        body,
    ))

    forecast_rows = [["Year", "Projected Accidents"]]
    for _, row in forecast_df.iterrows():
        forecast_rows.append([str(int(row["Year"])), f"{int(row['Projected Accidents']):,}"])

    ft = Table(forecast_rows, colWidths=[45 * mm, 70 * mm])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFD447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1D2A44")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8C6A7")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    story.extend([ft, Spacer(1, 8)])

    story.append(Paragraph("5. Model Performance", heading))

    if metrics is not None and not metrics.empty:
        metric_rows = [["Model", "R²", "MAE", "RMSE", "MSE"]]
        for name, row in metrics.iterrows():
            metric_rows.append([
                name, f"{row['R²']:.3f}", f"{row['MAE']:,.1f}", f"{row['RMSE']:,.1f}", f"{row['MSE']:,.1f}",
            ])

        mt = Table(metric_rows, colWidths=[45 * mm, 20 * mm, 28 * mm, 28 * mm, 30 * mm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0047AB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8C6A7")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))

        story.extend([mt, Spacer(1, 8)])

    story.append(Paragraph("6. Safety Recommendations", heading))
    for item in recommendations(risk_level):
        story.append(Paragraph(f"• {html.escape(item)}", body))

    story.extend([
        Spacer(1, 8),
        Paragraph("7. Methodology & Limitations", heading),
        Paragraph(
            "The system uses historical state/UT accident counts from "
            "the supplied project dataset. The machine-learning models "
            "use the 2019–2022 accident counts as features and 2023 as the target year. "
            "Risk categories are relative indicators derived "
            "from the supplied state-level data. The application does "
            "not use live traffic, weather, road-condition or real-time "
            "incident feeds.",
            body,
        ),
        Spacer(1, 6),
        Paragraph(
            "<b>Important:</b> AI outputs are estimates and should not "
            "be treated as guaranteed accident counts or as a substitute "
            "for official road-safety decisions.",
            body,
        ),
    ])

    doc.build(story)
    return buffer.getvalue()


def render_reports(
    state: str,
    record: pd.Series,
    predictions: Dict[str, float],
    risk_score: float,
    risk_level: str,
    ensemble_prediction: float,
    df: pd.DataFrame,
):
    st.title("📄 Professional PDF Report")
    st.caption("Generate a transparent analytical report for the selected region.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Region", state)
    c2.metric("2023 Accidents", fmt_int(record.get("2023 Accidents")))
    c3.metric("2023 Model Estimate", fmt_int(ensemble_prediction))
    c4.metric("Risk", risk_badge(risk_level))

    st.markdown(
        """
<div class="explain-card">
<b>Report transparency</b><br>
The report separates historical observations, model estimates and
scenario projections. It also includes model-validation metrics and
limitations so the reader can understand what the system can and cannot
claim.
</div>
""",
        unsafe_allow_html=True,
    )

    signature = tuple(df[FEATURE_COLS + [TARGET_COL]].itertuples(index=False, name=None))

    try:
        metrics = calculate_model_metrics(signature)
    except Exception:
        metrics = None

    forecast_df = build_forecast(safe_num(record.get("2023 Accidents"), 0.0), 0)

    if st.button("📥 Generate Complete PDF Report", use_container_width=True):
        try:
            with st.spinner("Building professional PDF..."):
                pdf_bytes = build_pdf_bytes(
                    state, record, predictions, risk_score, risk_level, forecast_df, metrics
                )

            filename = f"Road_Safety_AI_Report_{state.replace(' ', '_')}.pdf"

            st.success("Professional report generated successfully.")

            st.download_button(
                "⬇️ Download PDF Report",
                pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    st.subheader("📋 Report Contents")
    st.write(
        "✓ Executive summary • ✓ 2019–2023 historical data • "
        "✓ 2023 model estimates • ✓ Ensemble comparison • "
        "✓ Risk assessment • ✓ 2024–2035 scenario • "
        "✓ Model validation metrics • ✓ Recommendations • "
        "✓ Methodology & limitations"
    )


# ---------------------------------------------------------------------
# CUSTOM USER INPUT
# ---------------------------------------------------------------------
def render_custom_prediction(df: pd.DataFrame):
    st.title("✏️ Custom Accident Prediction")
    st.caption(
        "Enter four historical values and compare the saved ML models. "
        "The result changes with your inputs."
    )

    with st.form("custom_prediction_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            y2019 = st.number_input("2019 Accidents", min_value=0.0, max_value=1_000_000.0, value=1000.0, step=50.0)
            y2021 = st.number_input("2021 Accidents", min_value=0.0, max_value=1_000_000.0, value=1300.0, step=50.0)

        with c2:
            y2020 = st.number_input("2020 Accidents", min_value=0.0, max_value=1_000_000.0, value=1200.0, step=50.0)
            y2022 = st.number_input("2022 Accidents", min_value=0.0, max_value=1_000_000.0, value=1500.0, step=50.0)

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

    if submitted:
        try:
            values = [y2019, y2020, y2021, y2022]
            predictions = predict_all(values)
            summary = ensemble_summary(predictions)
            score, risk = custom_risk(summary["average"], df)

            st.session_state.custom_result = {
                "inputs": values,
                "predictions": predictions,
                "summary": summary,
                "score": score,
                "risk": risk,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    result = st.session_state.get("custom_result")

    if not result:
        return

    predictions = result["predictions"]
    summary = result["summary"]

    st.markdown("---")
    st.subheader("Prediction Results")

    cols = st.columns(4)
    model_names = list(MODEL_FILES.keys())

    for col, name in zip(cols[:3], model_names):
        value = predictions.get(name)
        col.metric(name, fmt_int(value))

    cols[3].metric(
        "Risk Indicator",
        risk_badge(result["risk"]),
        delta=(f"{result['score']:.1f}/100" if np.isfinite(result["score"]) else None),
    )

    if np.isfinite(summary["average"]):
        st.markdown(
            f"""
<div class="reliability-card">
<b>🤖 Ensemble estimate:</b> {summary['average']:,.0f} accidents<br><br>
<b>Model range:</b> {summary['minimum']:,.0f} – {summary['maximum']:,.0f}<br>
<b>Model spread:</b> {summary['spread']:,.0f}<br>
<b>Agreement indicator:</b> {summary['agreement']:.1f}%
<br><br>
<span style="color:#59677D">
The agreement indicator measures how close the available models are
to each other. It is not a probability of prediction accuracy.
</span>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("ℹ️ Why does this custom result change?"):
        st.write(
            "The four values you enter become the model features for the 2023 target. "
            "Changing those values changes the input and can change the estimate."
        )

    if st.button("💾 Save Custom Prediction", use_container_width=False):
        log_prediction("Custom Input", 2023, summary["average"], "Ensemble Average")
        st.success("Custom prediction saved to SQLite.")


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
def render_settings(df: pd.DataFrame):
    st.title("⚙️ System Settings")
    st.caption("Essential application health and database controls — no state selector.")

    models = load_saved_models()
    ready = [name for name in MODEL_FILES if models.get(name) is not None]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", len(df))
    c2.metric("Models Ready", f"{len(ready)}/3")
    c3.metric("Prediction Logs", db_count())
    c4.metric("System Status", "Operational" if ready else "Attention")

    st.markdown("---")
    st.subheader("🤖 Model Status")

    errors = models.get("_errors", {})
    cols = st.columns(3)

    for col, name in zip(cols, MODEL_FILES):
        if models.get(name) is not None:
            col.markdown(
                f"""
<div class="status status-ready">
<span>●</span>
<b>{name}</b>
<span class="status-value">Ready</span>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"""
<div class="status status-warn">
<span>●</span>
<b>{name}</b>
<span class="status-value">Unavailable</span>
</div>
""",
                unsafe_allow_html=True,
            )
            if errors.get(name):
                with col.expander("Why?"):
                    st.caption(str(errors[name]))

    st.markdown("---")
    st.subheader("🗄️ Database Management")
    st.write(f"Stored prediction logs: **{db_count()}**")

    if not st.session_state.clear_db_confirm:
        if st.button("🗑️ Clear Prediction Logs", use_container_width=False):
            st.session_state.clear_db_confirm = True
            st.rerun()
    else:
        st.warning("This permanently removes all saved prediction logs.")

        a, b = st.columns(2)

        if a.button("Yes, Clear Database"):
            clear_database()
            st.session_state.clear_db_confirm = False
            st.success("Prediction logs cleared.")
            st.rerun()

        if b.button("Cancel"):
            st.session_state.clear_db_confirm = False
            st.rerun()

    st.markdown("---")
    st.subheader("📁 Data & System Information")

    c1, c2 = st.columns(2)

    with c1:
        st.write("**Dataset**")
        st.code(str(DATA_PATH))
        st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")

    with c2:
        st.write("**Models directory**")
        st.code(str(MODELS_DIR))
        st.write("Normal startup never trains models.")

    st.markdown(
        """
<div class="explain-card">
<b>⚡ Performance design</b><br><br>
• Dataset is cached.<br>
• Saved model objects are cached.<br>
• SHAP is calculated only when requested.<br>
• Cross-validation is calculated only on the Model Performance page.<br>
• PDF generation happens only after the user requests it.<br>
• Missing models are never silently replaced by fake predictions.
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> str:
    try:
        from streamlit_option_menu import option_menu
    except ImportError:
        st.error("Please install streamlit-option-menu.")
        return "Dashboard"

    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center;padding:8px 0 18px">
    <div style="font-size:42px">🚦</div>
    <div style="font-size:20px;font-weight:800;color:#1D2A44">
        Road Safety AI
    </div>
    <div style="color:#687184;font-size:12px">
        Intelligent Analytics
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.caption("Road-safety analytics workspace")

        st.markdown("---")

        page = option_menu(
            menu_title=None,
            options=[
                "Dashboard",
                "Prediction",
                "Forecast",
                "Risk Map",
                "Reports",
                "Model Evaluation",
                "User Input",
                "Settings",
            ],
            icons=[
                "bar-chart-fill",
                "cpu",
                "graph-up-arrow",
                "geo-alt-fill",
                "file-earmark-pdf",
                "bar-chart-line",
                "keyboard",
                "gear",
            ],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#0047AB", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "4px 0",
                    "padding": "11px 15px",
                    "border-radius": "13px",
                    "color": "#536079",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg,rgba(0,71,171,.12),rgba(255,212,71,.38))",
                    "color": "#0047AB",
                    "font-weight": "700",
                },
            },
        )

        st.markdown("---")

        c1, c2 = st.columns(2)
        c1.metric(
            "States / UTs",
            len(df[df["State"].astype(str).str.casefold() != "all india"]),
        )
        c2.metric("2023 Accidents", f"{int(df[TARGET_COL].sum()):,}")

    return page


# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
def main_app():
    try:
        df = load_processed_data()
    except Exception as exc:
        st.error(f"Unable to load project data: {exc}")
        st.stop()

    init_db()

    page = render_sidebar(df)
    st.session_state.selected_page = page

    now = dt.datetime.now()

    st.markdown(
        f"""
<div style="
display:flex;
justify-content:space-between;
align-items:center;
border-bottom:1px solid rgba(29,42,68,.16);
padding-bottom:16px;
margin-bottom:22px;">
    <div>
        <span style="font-size:28px">🚦</span>
        <span style="font-size:26px;font-weight:800;color:#1D2A44">
            Road Safety AI
        </span>
    </div>
    <div style="color:#687184">
        {now.strftime('%B %d, %Y')}
        &nbsp;
        <span style="
        background:#FFD447;
        padding:7px 12px;
        border-radius:20px;
        color:#1D2A44">
            ◷ {now.strftime('%H:%M:%S')}
        </span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if page == "Settings":
        state = "All India"
        record = all_india_record()
        render_settings(df)

        st.markdown(
            "<div class='footer'>🚦 Road Safety AI | Machine Learning Project | © 2026 | by <b style='color:#59677D'>Gagan Mittal</b></div>",
            unsafe_allow_html=True,
        )
        return

    state, record = render_region_control(df, page)

    try:
        if state == "All India":
            predictions = predict_national(df)
        else:
            predictions = state_prediction(df, state, record)
    except Exception as exc:
        st.error(f"Prediction engine error: {exc}")
        predictions = {}

    summary = ensemble_summary(predictions)
    ensemble_prediction = summary["average"]

    risk_score, risk_level = state_risk(record, state)

    # ---------------------------------------------------------------
    # DASHBOARD
    # ---------------------------------------------------------------
    if page == "Dashboard":
        st.title("🚦 AI-Powered Road Safety Dashboard")
        st.caption(f"Decision-support overview for **{state}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Region", state)
        c2.metric("2023 Accidents", fmt_int(record.get("2023 Accidents")))

        if state == "All India":
            state_ut_count = int((df["State"].astype(str).str.casefold() != "all india").sum())
            c3.metric("Coverage", f"{state_ut_count} states / UTs")
            c4.metric("Risk", "National Overview")
        else:
            c3.metric("Risk Score", fmt_score(risk_score))
            c4.metric("Risk Level", risk_badge(risk_level))

        a, b = st.columns([1.8, 1])

        with a:
            st.subheader("📈 Historical Accident Trend")
            st.plotly_chart(trend_chart(record, state), use_container_width=True)

        with b:
            st.subheader("🔮 2023 Model Estimates")

            for name in MODEL_FILES:
                value = predictions.get(name)
                st.metric(name, fmt_int(value))

            if np.isfinite(ensemble_prediction):
                st.metric("Ensemble Average", fmt_int(ensemble_prediction))

        if np.isfinite(summary["spread"]):
            st.markdown(
                f"""
<div class="reliability-card">
<b>Model agreement:</b> {summary['agreement']:.1f}%<br>
<b>Model range:</b> {summary['minimum']:,.0f} – {summary['maximum']:,.0f}<br>
<span style="color:#59677D">
Agreement is a consistency indicator, not a probability of accuracy.
</span>
</div>
""",
                unsafe_allow_html=True,
            )

        a, b = st.columns(2)

        with a:
            st.subheader("🏆 State Risk Ranking")
            st.dataframe(ranking_table(df), use_container_width=True, height=330, hide_index=True)

        with b:
            import plotly.express as px

            x = df[df["State"].astype(str).str.casefold() != "all india"].copy()

            fig = px.scatter(
                x,
                x="2023 Accidents",
                y="Risk Score",
                color="Risk Category",
                hover_name="State",
                size="2023 Accidents",
                size_max=30,
                color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
                title="Risk vs Accident Volume",
            )

            st.plotly_chart(transparent_layout(fig), use_container_width=True)

        st.subheader("📊 Historical Correlation")
        st.plotly_chart(
            correlation_chart(df[df["State"].astype(str).str.casefold() != "all india"]),
            use_container_width=True,
        )

        with st.expander("📂 View Complete Dataset"):
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------------
    elif page == "Prediction":
        st.title("🤖 AI Prediction Engine")
        st.caption(
            f"Estimated 2023 accident count for **{state}** "
            "from historical 2019–2022 accident features."
        )

        c1, c2, c3, c4 = st.columns(4)

        for col, name in zip([c1, c2, c3], MODEL_FILES):
            col.metric(name, fmt_int(predictions.get(name)))

        c4.metric("Ensemble Estimate", fmt_int(ensemble_prediction))

        if np.isfinite(summary["average"]):
            st.markdown(
                f"""
<div class="reliability-card">
<b>📏 Estimated model range:</b>
{summary['minimum']:,.0f} – {summary['maximum']:,.0f}<br><br>
<b>Model spread:</b> {summary['spread']:,.0f} accidents<br>
<b>Agreement indicator:</b> {summary['agreement']:.1f}%<br><br>
<span style="color:#59677D">
This range reflects disagreement among the available models. It is
not a statistical confidence interval.
</span>
</div>
""",
                unsafe_allow_html=True,
            )

        if state != "All India":
            st.subheader("⚠️ Risk Assessment")
            st.metric("Current Relative Risk", risk_badge(risk_level), f"Score {risk_score:.1f}/100")

            with st.expander("Why is this risk level shown?"):
                st.write(
                    "The risk score is calculated from the selected "
                    "state's 2023 accident count relative to the highest "
                    "2023 accident count in the supplied state-level dataset."
                )
        else:
            st.info(
                "All India is shown as a national aggregation. "
                "State-level High/Medium/Low risk classification is not "
                "assigned to the national total."
            )

        st.markdown(
            """
<div class="explain-card">
<b>What does this prediction mean?</b><br><br>
The models use four historical inputs (2019–2022) to estimate the
2023 target-year accident count. The output is an analytical estimate,
not a guaranteed number or a real-time accident warning.
</div>
""",
            unsafe_allow_html=True,
        )

        render_shap(record, state)

        st.subheader("🛡️ Safety Recommendations")

        for item in recommendations(risk_level if state != "All India" else "Low"):
            st.write(f"• {item}")

        if np.isfinite(ensemble_prediction):
            if st.button("💾 Save Ensemble Prediction", use_container_width=False):
                log_prediction(state, 2023, ensemble_prediction, "Ensemble Average")
                st.success("Prediction logged successfully.")

        st.warning(
            "Important: this system supports analysis and planning. "
            "It does not replace official road-safety authorities or "
            "real-time emergency systems."
        )

    # ---------------------------------------------------------------
    # FORECAST
    # ---------------------------------------------------------------
    elif page == "Forecast":
        observed_2023 = safe_num(record.get("2023 Accidents"))
        if not np.isfinite(observed_2023):
            st.warning("A valid observed 2023 accident count is required for the scenario forecast.")
        else:
            render_forecast(state, observed_2023)

    # ---------------------------------------------------------------
    # RISK MAP
    # ---------------------------------------------------------------
    elif page == "Risk Map":
        render_risk_map(df)

    # ---------------------------------------------------------------
    # REPORTS
    # ---------------------------------------------------------------
    elif page == "Reports":
        if not np.isfinite(ensemble_prediction):
            st.warning("A valid prediction is required before generating a report.")
        else:
            render_reports(state, record, predictions, risk_score, risk_level, ensemble_prediction, df)

    # ---------------------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------------------
    elif page == "Model Evaluation":
        render_model_evaluation(df)

    # ---------------------------------------------------------------
    # USER INPUT
    # ---------------------------------------------------------------
    elif page == "User Input":
        render_custom_prediction(df)

    st.markdown(
        """
<div class="footer">
🚦 Road Safety AI &nbsp;|&nbsp; Machine Learning Project
&nbsp;|&nbsp; © 2026 &nbsp;|&nbsp;
by <b style="color:#59677D">Gagan Mittal</b>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------
main_app()
