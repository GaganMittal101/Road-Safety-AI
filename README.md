# 🚦 Road Safety AI

**AI-powered road accident analysis, relative risk assessment, and transparent scenario planning for India.**

Road Safety AI is a Python/Streamlit application that analyses historical state/UT accident counts, estimates the 2023 target value from 2019–2022 inputs, visualises relative accident-count categories, compares regression models, provides SHAP explanations, logs predictions, and generates PDF reports.

> **Scope note:** This is an academic/portfolio decision-support prototype. It is not a live accident alert service, an official government forecasting tool, or a production safety system. Model outputs are estimates, and the 2024–2035 page is an assumed-growth scenario rather than a trained future-year forecast.

## Features

- Dashboard with historical trends, regional ranking, correlations, and model estimates.
- State/UT selection and a Folium risk map with GeoJSON boundaries.
- Regression estimates from Random Forest, XGBoost, and Gradient Boosting.
- Cross-validation reporting with R², MAE, RMSE, MSE, and a simple mean baseline.
- SHAP-based feature contribution explanations for eligible regional inputs.
- Custom-input model comparison, SQLite prediction logs, and PDF/CSV exports.
- System settings with dataset/model status and training metadata.
- GitHub Actions checks for Python syntax and unit tests, including regression coverage for missing-data charts.

## Technology stack

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Plotly, Folium, SHAP, SQLite, Joblib, and ReportLab.

## Run locally (Windows PowerShell)

From the repository root:

```powershell
py -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python setup_models.py
python -m streamlit run app/app.py
```

The application opens at `http://localhost:8501`. The local demo intentionally opens directly without a login screen; **do not expose it to the public internet without adding proper authentication and deployment security**.

If you already have a virtual environment, activate it instead of creating another one. Model retraining backs up existing model artifacts inside `models/backups/` and creates `models/training_metadata.json` plus `reports/model_evaluation.csv`.

## Project structure

```text
app/
  app.py                 Streamlit UI and application orchestration
  core.py                Data cleaning, eligibility, map-name, and scenario helpers
  predict.py             Prediction/SHAP utility functions
assets/                  India state/UT GeoJSON boundary file(s)
data/raw/                Source analysis CSVs
data/processed/          Processed state/UT accident-count data
models/                  Trained model artefacts and training metadata
reports/                 Generated evaluation/report artefacts
setup_models.py          Model training and cross-validation report generation
validate_project.py      Local data/model/app integrity checks
tests/                   Unit tests for data and scenario helpers
```

## Model methodology

The current feature set is the four yearly counts for 2019–2022 and the target column is `2023 Accidents`. Training and evaluation exclude the duplicated All India aggregate, embedded footnote rows, and records with incomplete feature/target values. The application uses raw count features at training and inference time; a legacy `scaler.pkl` is not applied.

The evaluation page compares the three regression algorithms with a `Mean Baseline`. Shuffled cross-validation estimates performance across the available complete regional records; it is **not** a temporal backtest. Read the actual metrics produced in the application and `reports/model_evaluation.csv` rather than quoting an unverified accuracy percentage.

## Data and interpretation limitations

- The bundled processed dataset contains one target year per region for this feature setup. It is insufficient by itself to prove reliable future-year forecasting.
- Regions missing one or more 2019–2022 values are excluded from model training and prediction; the app discloses partial coverage in national aggregation.
- High/Medium/Low categories are based on relative accident counts in the supplied data. They are not exposure-adjusted crash rates because traffic volume, distance travelled, road length, and population are not included.
- The forecast applies a user-selected constant annual growth rate to the observed 2023 baseline. It is a transparent planning scenario, not a fitted time-series model.
- SHAP explains the model's learned associations; it does not establish that a historical feature caused an accident.
- Verify and document the exact source, publication date, and licensing of each raw data file before using the outputs in a public report or operational decision.

## Validate before a demo

```powershell
python -m unittest discover -s tests -v
python validate_project.py
python -m streamlit run app/app.py
```

The validator checks file syntax, UTF-8 text, dataset cleaning, GeoJSON parsing, model artifact loading, and a sample prediction. It does not prove real-world accuracy. Manually check each UI page, risk-map markers/boundaries, model evaluation, custom prediction, SQLite logging, PDF download, and CSV export before an interview/demo.

## License

No license is currently specified in this repository. Add a license only after deciding what rights you want to grant to others.


**Upgrade compatibility:** If an older Phase 1 package was previously applied, the full-upgrade installer backs up and quarantines its obsolete `tests/test_phase1.py` before running the current test suite. This avoids stale assertions that conflict with the current prediction/forecast architecture.
