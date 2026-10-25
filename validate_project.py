from __future__ import annotations

"""Local validation for Road Safety AI. Run with the project's Python environment."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.core import clean_accident_frame, prepare_training_xy

APP = ROOT / "app" / "app.py"
PREDICT = ROOT / "app" / "predict.py"
CONFIG = ROOT / ".streamlit" / "config.toml"
DATA = ROOT / "data" / "processed" / "risk_analysis.csv"
FEATURES = [f"{year} Accidents" for year in range(2019, 2023)]
TARGET = "2023 Accidents"
MODEL_FILES = {
    "Random Forest": ROOT / "models" / "accident_prediction.pkl",
    "XGBoost": ROOT / "models" / "xgb_model.pkl",
    "Gradient Boosting": ROOT / "models" / "gb_model.pkl",
}
MOJIBAKE = ("ðŸ", "â€”", "â€“", "âš", "â€", "ï¸", "Ãƒ", "Ã¢")
failures = []
warnings = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{ 'PASS' if ok else 'FAIL' }] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def warn(label: str, detail: str) -> None:
    print(f"[WARN] {label} — {detail}")
    warnings.append(label)


def main() -> int:
    print("ROAD SAFETY AI — FULL PROJECT CHECK\n")
    report(APP.is_file(), "app/app.py exists", str(APP))
    report(PREDICT.is_file(), "app/predict.py exists", str(PREDICT))
    report(CONFIG.is_file(), ".streamlit/config.toml exists", str(CONFIG))
    if not APP.is_file() or not DATA.is_file():
        print("Cannot complete validation because required project files are missing.")
        return 1

    sources = []
    for file in (APP, PREDICT, ROOT / "app" / "core.py", ROOT / "setup_models.py", ROOT / "validate_project.py"):
        if file.is_file():
            try:
                text = file.read_text(encoding="utf-8")
                compile(text, str(file), "exec")
                report(True, f"Python syntax: {file.relative_to(ROOT)}")
                if file.name == "validate_project.py":
                    report(True, f"UTF-8 content: {file.relative_to(ROOT)}", "validator contains marker definitions intentionally")
                elif any(marker in text for marker in MOJIBAKE):
                    report(False, f"UTF-8 content: {file.relative_to(ROOT)}", "common mojibake markers found")
                else:
                    report(True, f"UTF-8 content: {file.relative_to(ROOT)}")
                sources.append((file, text))
            except Exception as exc:
                report(False, f"Source check: {file.relative_to(ROOT)}", f"{type(exc).__name__}: {exc}")

    app_text = APP.read_text(encoding="utf-8")
    report(app_text.rstrip().endswith("main_app()"), "app launches dashboard directly")
    report("def login_page()" not in app_text, "legacy login function removed")
    report('ASSETS_DIR / "india_states.geojson"' in app_text and 'ASSETS_DIR / "india_state.geojson"' in app_text,
           "map loader supports both GeoJSON filename variants")

    try:
        config_text = CONFIG.read_text(encoding="utf-8")
        report(all(color in config_text for color in ("#F5E8D8", "#0047AB", "#FFFCF5")),
               "Streamlit theme background and primary colours present")
        report("#FFD447" in app_text, "Sun Yellow accent present in app CSS")
    except OSError as exc:
        report(False, "Streamlit theme readable", str(exc))

    try:
        raw = pd.read_csv(DATA)
        clean = clean_accident_frame(raw, [f"{year} Accidents" for year in range(2019, 2024)], TARGET)
        report(True, "Dataset loads and cleans", f"{len(clean)} rows including any All India aggregate")
        labels = clean["State"].astype(str)
        regional = clean.loc[labels.str.casefold() != "all india"]
        footnotes = regional["State"].astype(str).str.startswith(("#", "*"))
        report(not footnotes.any(), "dataset footnote rows excluded", f"{len(regional)} geographic rows")
        X, y, eligible, excluded = prepare_training_xy(clean, FEATURES, TARGET)
        report(len(X) >= 5, "enough complete model rows", f"{len(X)} rows used; {len(excluded)} incomplete rows excluded")
        if not excluded.empty:
            warn("Regions excluded from model training/inference", ", ".join(excluded["State"].astype(str).tolist()))
        if clean["State"].astype(str).str.casefold().duplicated().any():
            report(False, "state/UT names unique", "duplicate normalized state/UT labels found")
        else:
            report(True, "state/UT names unique")
        if (regional[FEATURES + [TARGET]].select_dtypes(include=[np.number]) < 0).any().any():
            report(False, "accident counts non-negative")
        else:
            report(True, "accident counts non-negative or missing")
    except Exception as exc:
        report(False, "dataset validation", f"{type(exc).__name__}: {exc}")
        clean = None

    geo_candidates = (ROOT / "assets" / "india_states.geojson", ROOT / "assets" / "india_state.geojson")
    geo = next((path for path in geo_candidates if path.is_file()), None)
    report(geo is not None, "India GeoJSON exists", str(geo) if geo else "neither supported filename found")
    if geo:
        try:
            with geo.open("r", encoding="utf-8") as handle:
                geo_data = json.load(handle)
            valid_geo = isinstance(geo_data, dict) and isinstance(geo_data.get("features"), list) and len(geo_data["features"]) > 0
            report(valid_geo, "GeoJSON parses and contains features", f"{len(geo_data.get('features', []))} features" if valid_geo else "invalid feature collection")
        except Exception as exc:
            report(False, "GeoJSON parses", f"{type(exc).__name__}: {exc}")

    sample = pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)
    for name, path in MODEL_FILES.items():
        if not path.is_file():
            report(False, f"model artifact exists: {name}", str(path))
            continue
        try:
            model = joblib.load(path)
            if not hasattr(model, "predict"):
                raise ValueError("artifact has no predict() method")
            count = getattr(model, "n_features_in_", len(FEATURES))
            if int(count) != len(FEATURES):
                raise ValueError(f"expected {len(FEATURES)} features, artifact declares {count}")
            predicted = np.asarray(model.predict(sample), dtype=float).reshape(-1)
            if not predicted.size or not np.isfinite(predicted[0]):
                raise ValueError("prediction is empty or non-finite")
            report(True, f"model smoke test: {name}", f"sample output {predicted[0]:.2f}; this does not validate accuracy")
        except Exception as exc:
            report(False, f"model smoke test: {name}", f"{type(exc).__name__}: {exc}")

    metadata = ROOT / "models" / "training_metadata.json"
    if metadata.is_file():
        try:
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            report(meta.get("target") == TARGET, "training metadata target matches app", str(meta.get("target")))
            warn("Model validity scope", str(meta.get("validation_note", "Metadata exists; read report for assumptions.")))
        except Exception as exc:
            report(False, "training metadata parses", str(exc))
    else:
        warn("Training metadata missing", "Run `python setup_models.py` after installing the upgrade to regenerate models and metadata.")

    print("\nSummary:", "PASS" if not failures else f"{len(failures)} failure(s)", f"and {len(warnings)} warning(s)")
    if failures:
        print("Address FAIL items before claiming the project is fully validated.")
        return 1
    print("Static/data/model smoke checks passed. Browser walkthrough and real-world predictive validity are separate checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
