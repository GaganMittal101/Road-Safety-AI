# Road Safety AI — Full Upgrade Package

This package consolidates the planned upgrade phases for the existing local project. It does not include the user's dataset, model binary files, database, or 22 MB GeoJSON; those remain in `C:\Road-Safety-AI` and are not overwritten.

## Included phases

### Phase 1 — Application correctness
- Direct dashboard entry; no demo login gate or admin badge.
- Warm Beige `#F5E8D8`, Cobalt Blue `#0047AB`, Sun Yellow `#FFD447` theme.
- Repairs data footnotes accidentally interpreted as state/UT rows.
- Preserves missing historical values as missing rather than silently converting them to zero.
- Marks a missing 2023 target as `No Data`, instead of incorrectly categorizing it as Low.
- Supports both `assets/india_states.geojson` and `assets/india_state.geojson`.
- Normalizes common state/UT names consistently for map joins.
- Shows which regions are omitted from national model estimates when historical inputs are incomplete.
- Avoids saving invalid predictions to SQLite.
- Uses explicit UTF-8 source text, replacing the corrupted local app source with the clean packaged copy.

### Phase 2 — Model pipeline and evaluation
- Uses the same four raw count features for training and inference; legacy `scaler.pkl` is ignored.
- Excludes the duplicated All India aggregate, source footnotes, and rows with incomplete model features from training/evaluation.
- Adds a `Mean Baseline` to cross-validation model comparisons.
- Generates `models/training_metadata.json` and `reports/model_evaluation.csv` when models are retrained.
- Labels outputs as 2023 target-year estimates; the 2024–2035 view is an assumed-growth scenario based on the observed 2023 count, not a trained future-year prediction.

### Phase 3 — Professional UX and reporting
- Consistent theme colours in Streamlit and exported PDF report tables.
- Clearer risk category wording: these are relative accident-count indicators, not exposure-adjusted crash rates.
- Missing values create chart gaps rather than misleading zero-value points.
- Settings shows training metadata when available and model readiness is checked with a probe prediction.

### Phase 4 — Quality and reproducibility
- Unit tests cover data cleaning, missing-value treatment, map aliases, scenario math, direct app entry, theme, and model-evaluation guardrails.
- `validate_project.py` checks app syntax, data schema/quality, GeoJSON, model artifact loadability and prediction smoke tests.
- GitHub Actions workflow compiles Python modules and runs unit tests on pushes/pull requests.
- Batch launch/check shortcuts avoid PowerShell execution-policy issues.

## Install on Windows (recommended)

1. Stop Streamlit with `Ctrl+C`.
2. Extract the ZIP. The ZIP may create a nested folder with the same name; that is okay.
3. Open PowerShell in your activated project virtual environment (`(venv)` should be visible).
4. Copy and run this command to locate the installer automatically and install the update, including retraining on the cleaned rows:

   ```powershell
   $installer = Get-ChildItem "$env:USERPROFILE\Downloads" -Recurse -File -Filter "install_project_update.py" | Select-Object -First 1
   if ($null -eq $installer) { throw "Installer not found. Extract the Full Upgrade ZIP first." }
   python $installer.FullName --project-root "C:\Road-Safety-AI" --retrain-models
   ```

   If the ZIP was extracted somewhere other than Downloads, change the search root to that folder. This uses Python instead of running an unsigned `.ps1` script. It does not require changing the machine-wide PowerShell execution policy.

5. The installer creates backups under `.road_safety_ai_backups`, quarantines the obsolete `tests/test_phase1.py` from the previous repair bundle (retaining a backup), runs the current unit tests, optionally retrains the model files in the active Python environment, and runs project validation. It does not change `data/`, `assets/`, `database.db`, or GitHub `main`.
6. Launch the application:

   ```powershell
   Set-Location C:\Road-Safety-AI
   python -m streamlit run app/app.py
   ```

## Validation and troubleshooting

Run these separately if you need to recheck later:

```powershell
Set-Location C:\Road-Safety-AI
python -m unittest discover -s tests -v
python validate_project.py
python -m streamlit run app/app.py
```

The first two commands should complete before you make interview claims. If validation fails, preserve `.road_safety_ai_backups` and share the complete console output. Do not delete backups before checking the failure.

## Important scientific limitations

- The supplied dataset contains one target year (2023) for each region, rather than a long panel of repeated target years. The existing models estimate the 2023 target from the 2019–2022 cross-section; this is **not** enough evidence to claim validated future-year forecasting accuracy.
- Some regional rows have incomplete input history. The updated pipeline excludes those rows from training/evaluation and reports the excluded regions. All-India model predictions sum outputs for the eligible regions only; the app discloses coverage when incomplete regions are omitted.
- High/Medium/Low are relative accident-count categories based on supplied counts, not rates normalized for traffic volume, vehicle-kilometres, road length, population, or exposure.
- The forecast page is a transparent compound-growth scenario, not a trained forecast, government forecast, or guarantee.
- An attractive UI, a passing syntax check, and a passing test suite do not prove real-world predictive validity. Review the evaluation metrics and data source before using the app for decisions.

## File safety and scope

The installer backs up replaced files first. Optional model retraining is enabled by the recommended command and backs up old model artifacts. If tests fail, source/config changes are rolled back where backups exist. GitHub is not modified by this package; changes remain local until you review and commit them yourself.


### Troubleshooting note (updated package)

If a previous Phase 1 package was installed, it may leave behind `tests/test_phase1.py`. Those older tests conflict with the new shared scenario helper and outdated source-code literal checks. The updated installer now backs up and quarantines that obsolete file before running the replacement suite. The old file is retained under `.road_safety_ai_backups` and can be restored if needed.
