import os
import pandas as pd
import plotly.express as px
import json
import streamlit as st

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "processed", "risk_analysis.csv")
    return pd.read_csv(path)

def preprocess_data(df):
    # Ensure required columns exist; if some are missing, fill with 0
    required = ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents", "2023 Accidents"]
    for col in required:
        if col not in df.columns:
            df[col] = 0
    df["State"] = df["State"].fillna("Unknown")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Risk Score"] = (df["2023 Accidents"] / df["2023 Accidents"].max()) * 100 if df["2023 Accidents"].max() > 0 else 0

    def risk_cat(score):
        if score >= 75:
            return "High"
        elif score >= 40:
            return "Medium"
        else:
            return "Low"
    df["Risk Category"] = df["Risk Score"].apply(risk_cat)
    return df

def create_trend_chart(selected, state):
    trend = pd.DataFrame({
        "Year": ["2019", "2020", "2021", "2022", "2023"],
        "Accidents": [
            selected["2019 Accidents"].values[0],
            selected["2020 Accidents"].values[0],
            selected["2021 Accidents"].values[0],
            selected["2022 Accidents"].values[0],
            selected["2023 Accidents"].values[0]
        ]
    })
    fig = px.line(trend, x="Year", y="Accidents", markers=True,
                  title=f"{state} Accident Trend",
                  line_shape="spline")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_ranking_table(df):
    ranking = df[["State", "Risk Score", "Risk Category"]].sort_values("Risk Score", ascending=False)
    ranking.index = range(1, len(ranking)+1)
    ranking.index.name = "Rank"
    return ranking

def create_correlation_heatmap(df):
    cols = ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents", "2023 Accidents", "Risk Score"]
    # Ensure columns exist
    existing_cols = [c for c in cols if c in df.columns]
    if len(existing_cols) < 2:
        return px.scatter(title="Not enough data for correlation")
    corr = df[existing_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    title="Correlation Matrix")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@st.cache_data
def get_geo_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "assets", "india_states.geojson")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None