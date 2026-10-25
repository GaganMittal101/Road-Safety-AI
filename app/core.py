from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd


def clean_accident_frame(
    frame: pd.DataFrame,
    accident_columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    """Clean state-level accident data without turning missing values into zero.

    Removes footnote rows and keeps the optional All India source row for
    compatibility; the app calculates its national aggregate from the regional
    rows. Missing data stays NaN so charts and model inputs can distinguish it
    from a genuine zero-accident value.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if "State" not in frame.columns:
        raise ValueError("Dataset must contain a 'State' column.")

    missing = [col for col in accident_columns if col not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing required column(s): " + ", ".join(missing))

    result = frame.copy()
    result["State"] = result["State"].astype("string").str.strip()
    normalized = result["State"].fillna("").str.casefold()
    # The source CSV contains explanatory footnotes as rows; those are not regions.
    is_footnote = result["State"].fillna("").str.startswith(("#", "*"))
    is_invalid_name = normalized.isin(("", "nan", "none", "total"))
    result = result.loc[~(is_footnote | is_invalid_name)].copy()
    result["State"] = result["State"].astype(str).str.strip()

    non_aggregate = result.loc[result["State"].str.casefold() != "all india", "State"]
    duplicated = non_aggregate[non_aggregate.str.casefold().duplicated(keep=False)]
    if not duplicated.empty:
        duplicate_names = sorted(set(duplicated.tolist()), key=str.casefold)
        raise ValueError("Duplicate state/UT rows found: " + ", ".join(duplicate_names))

    for col in accident_columns:
        # Remove presentation separators in case a source includes formatted counts.
        values = result[col].astype("string").str.replace(",", "", regex=False)
        numeric = pd.to_numeric(values, errors="coerce")
        negatives = numeric < 0
        if bool(negatives.fillna(False).any()):
            bad_rows = result.loc[negatives.fillna(False), "State"].astype(str).tolist()
            raise ValueError(f"Negative accident counts in {col}: {', '.join(bad_rows)}")
        result[col] = numeric.astype(float)

    max_target = pd.to_numeric(result[target_column], errors="coerce").max(skipna=True)
    target = pd.to_numeric(result[target_column], errors="coerce")
    if pd.notna(max_target) and float(max_target) > 0:
        result["Risk Score"] = (target / float(max_target) * 100.0).clip(0, 100)
    else:
        result["Risk Score"] = np.nan

    scores = result["Risk Score"]
    categories = pd.Series("No Data", index=result.index, dtype="object")
    observed = scores.notna()
    categories.loc[observed & (scores < 40)] = "Low"
    categories.loc[observed & (scores >= 40) & (scores < 75)] = "Medium"
    categories.loc[observed & (scores >= 75)] = "High"
    result["Risk Category"] = categories
    return result.reset_index(drop=True)



def prepare_risk_scatter_frame(
    frame: pd.DataFrame,
    accident_column: str = "2023 Accidents",
    score_column: str = "Risk Score",
) -> pd.DataFrame:
    """Return finite, non-negative regional points for the risk scatter chart.

    Incomplete points are excluded from this chart only. The source dataframe is
    not mutated; missing rows remain available to the full dataset table.
    """
    required = {"State", accident_column, score_column, "Risk Category"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Scatter data is missing required column(s): " + ", ".join(missing))

    plot_data = frame.loc[
        frame["State"].astype(str).str.strip().str.casefold() != "all india"
    ].copy()
    accidents = pd.to_numeric(plot_data[accident_column], errors="coerce")
    scores = pd.to_numeric(plot_data[score_column], errors="coerce")
    valid = (
        np.isfinite(accidents.to_numpy(dtype=float))
        & np.isfinite(scores.to_numpy(dtype=float))
        & accidents.ge(0).fillna(False).to_numpy(dtype=bool)
    )
    plot_data = plot_data.loc[valid].copy()
    plot_data[accident_column] = accidents.loc[plot_data.index].astype(float)
    plot_data[score_column] = scores.loc[plot_data.index].astype(float)
    plot_data["Risk Category"] = plot_data["Risk Category"].fillna("No Data").astype(str)
    return plot_data


def prepare_training_xy(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return complete, independent regional rows for fitting/evaluation.

    Returns X, y, eligible rows, excluded rows. It excludes All India aggregate
    rows and does not silently convert missing historical values into zero.
    """
    required = ["State", *feature_columns, target_column]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError("Dataset missing training column(s): " + ", ".join(missing))

    regional = frame.loc[frame["State"].astype(str).str.casefold() != "all india"].copy()
    X = regional[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(regional[target_column], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    valid &= (X >= 0).all(axis=1) & (y >= 0)
    eligible = regional.loc[valid].copy()
    excluded = regional.loc[~valid].copy()
    X = X.loc[valid].astype(float).reset_index(drop=True)
    y = y.loc[valid].astype(float).reset_index(drop=True)
    return X, y, eligible, excluded


def normalize_map_name(name: object) -> str:
    """Normalize common India state/UT aliases for joining GeoJSON to data."""
    value = str(name).strip().casefold().replace("&", " and ")
    value = re.sub(r"[*#]", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    aliases = {
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "j and k": "jammu and kashmir",
        "jk": "jammu and kashmir",
        "andaman nicobar islands": "andaman and nicobar islands",
        "dadra and nagar haveli": "dadra and nagar haveli",
        "daman and diu": "daman and diu",
    }
    return aliases.get(value, value)


def build_scenario_forecast(
    observed_baseline: float,
    growth_percent: int,
    start_year: int = 2023,
    horizon: int = 12,
) -> pd.DataFrame:
    """Generate an explicitly illustrative compound-growth scenario.

    The baseline is the observed count in start_year. The first projection is
    start_year + 1 after one full compounding period; this is not a trained
    time-series forecast.
    """
    try:
        base = float(observed_baseline)
        growth = float(growth_percent) / 100.0
        periods = int(horizon)
        start = int(start_year)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Baseline, growth, start year, and horizon must be numeric.") from exc
    if not np.isfinite(base) or base < 0:
        raise ValueError("Observed baseline must be a finite non-negative value.")
    if not np.isfinite(growth) or growth <= -1:
        raise ValueError("Annual growth must be greater than -100%.")
    if periods < 1 or periods > 100:
        raise ValueError("Forecast horizon must be between 1 and 100 years.")
    years = list(range(start + 1, start + periods + 1))
    projected = [int(np.rint(base * ((1.0 + growth) ** step))) for step in range(1, periods + 1)]
    return pd.DataFrame({"Year": years, "Projected Accidents": projected})
