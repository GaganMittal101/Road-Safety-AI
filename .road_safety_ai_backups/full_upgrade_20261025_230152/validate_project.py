from __future__ import annotations

"""Local integrity checks for the Road Safety AI Phase 1 patch.

Run from the project root in the project's virtual environment:
    python validate_project.py

This checks source/config/data/model/map presence and a small pure-function
forecast smoke test. It does not prove predictive accuracy or replace a full
browser-based end-to-end test.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "app.py"
CONFIG = ROOT / ".streamlit" / "config.toml"
DATA = ROOT / "data" / "processed" / "risk_analysis.csv"
MODEL_FILES = (
    ROOT / "models" / "accident_prediction.pkl",
    ROOT / "models" / "xgb_model.pkl",
    ROOT / "models" / "gb_model.pkl",
)
GEOJSON_FILES = (ROOT / "assets" / "india_states.geojson", ROOT / "assets" / "india_state.geojson")
MOJIBAKE_MARKERS = ("ðŸ", "â€”", "â€“", "âš", "â€", "ï¸", "Ãƒ", "Ã¢")
failures: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    state = "PASS" if ok else "FAIL"
    print(f"[{state}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def safe_num(value, default=0.0):
    try:
        import numpy as np
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def check_forecast_function(source: str) -> tuple[bool, str]:
    """Execute only build_forecast from the AST; do not import/run the UI."""
    try:
        import numpy as np
        import pandas as pd
        tree = ast.parse(source, filename=str(APP))
        node = next(
            item for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "build_forecast"
        )
        module = ast.Module(
            body=[ast.Import(names=[ast.alias(name="numpy", asname="np")]),
                  ast.Import(names=[ast.alias(name="pandas", asname="pd")]), node],
            type_ignores=[],
        )
        namespace = {"safe_num": safe_num}
        exec(compile(ast.fix_missing_locations(module), str(APP), "exec"), namespace)
        forecast = namespace["build_forecast"](1000, 0)
        if forecast["Year"].tolist() != list(range(2024, 2036)):
            return False, "forecast years are not 2024–2035"
        if forecast["Projected Accidents"].tolist() != [1000] * 12:
            return False, "zero-growth scenario does not preserve the baseline"
        positive = namespace["build_forecast"](1000, 5)
        if int(positive.iloc[0]["Projected Accidents"]) != 1050:
            return False, "the 2024 value does not apply one year of growth"
        if int(positive.iloc[-1]["Projected Accidents"]) != int(np.rint(1000 * 1.05 ** 12)):
            return False, "the 2035 value does not apply twelve years of growth"
        return True, "zero growth and +5% compounding scenarios"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print("ROAD SAFETY AI — PHASE 1 PROJECT CHECK\n")
    report(APP.is_file(), "app/app.py exists", str(APP))
    report(CONFIG.is_file(), ".streamlit/config.toml exists", str(CONFIG))
    for path in MODEL_FILES:
        report(path.is_file(), f"model file exists: {path.name}", str(path))
    geo = next((path for path in GEOJSON_FILES if path.is_file()), None)
    report(geo is not None, "India state GeoJSON exists", str(geo) if geo else "neither supported filename found")

    if not APP.is_file():
        print("\nCannot continue without app/app.py. Run this script from the project root.")
        return 1

    try:
        source = APP.read_text(encoding="utf-8")
        compile(source, str(APP), "exec")
        report(True, "app/app.py compiles", "syntax check only; this does not run Streamlit")
    except Exception as exc:
        source = ""
        report(False, "app/app.py compiles", f"{type(exc).__name__}: {exc}")

    if source:
        bad = [marker for marker in MOJIBAKE_MARKERS if marker in source]
        report(not bad, "app source has no common mojibake markers", ", ".join(bad) if bad else "UTF-8 source text looks clean")
        report("main_app()" in source.splitlines()[-5:], "application launches directly without login", "expected direct main_app() entrypoint")
        report("login_page()" not in source.splitlines()[-5:], "no login page at entrypoint")
        report('"india_states.geojson"' in source and '"india_state.geojson"' in source,
               "map loader supports both GeoJSON filename variants")
        report("model_input = X" in source,
               "national prediction uses raw features consistent with setup_models.py")
        report('"2024 Model Estimates"' not in source and '"3. 2024 Model Predictions"' not in source,
               "UI/PDF do not mislabel the 2023 model target as a 2024 prediction")
        ok, detail = check_forecast_function(source)
        report(ok, "forecast function smoke test", detail)

    if CONFIG.is_file():
        try:
            config = CONFIG.read_text(encoding="utf-8")
            colors_ok = all(code in config for code in ("#0047AB", "#F5E8D8", "#FFFCF5", "#1D2A44"))
            report(colors_ok, "Streamlit theme contains approved warm-beige/cobalt palette")
            bad = [marker for marker in MOJIBAKE_MARKERS if marker in config]
            report(not bad, "Streamlit config has no common mojibake markers", ", ".join(bad) if bad else "UTF-8 config looks clean")
        except OSError as exc:
            report(False, "Streamlit config readable", str(exc))

    if DATA.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(DATA)
            required = ["State", *[f"{year} Accidents" for year in range(2019, 2024)]]
            missing = [column for column in required if column not in df.columns]
            report(not missing, "dataset schema includes State and 2019–2023 accident columns", f"missing: {missing}" if missing else f"{len(df)} total rows")
            if not missing:
                independent = df[df["State"].astype(str).str.casefold() != "all india"]
                report(len(independent) >= 2, "dataset has independent state/UT rows", f"{len(independent)} state/UT rows; aggregate excluded from CV")
        except Exception as exc:
            report(False, "dataset can be read", f"{type(exc).__name__}: {exc}")
    else:
        report(False, "processed dataset exists", str(DATA))

    if geo:
        try:
            # The project GeoJSON is large; stream json parser still reads it as a whole,
            # consistent with the app's current loader behavior.
            with geo.open("r", encoding="utf-8") as handle:
                geometry = json.load(handle)
            valid = isinstance(geometry, dict) and isinstance(geometry.get("features"), list) and bool(geometry["features"])
            report(valid, "GeoJSON parses and contains features", f"{len(geometry.get('features', []))} features" if valid else "missing/empty features")
        except Exception as exc:
            report(False, "GeoJSON parses and contains features", f"{type(exc).__name__}: {exc}")

    print("\nSummary:", "PASS" if not failures else f"{len(failures)} check(s) need attention")
    if failures:
        print("Review failed checks before the interview demo. This validator does not measure model accuracy or replace testing in a browser.")
        return 1
    print("Static/data/forecast checks passed. Now launch Streamlit and manually test each page, map, prediction, and PDF download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
