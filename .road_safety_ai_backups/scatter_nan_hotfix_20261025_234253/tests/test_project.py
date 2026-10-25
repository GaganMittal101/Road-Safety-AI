from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.core import (
    build_scenario_forecast,
    clean_accident_frame,
    normalize_map_name,
    prepare_training_xy,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"


class DataCleaningTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "State": ["Rajasthan", "Daman & Diu", "All India", "# footnote", "* another footnote"],
            "2019 Accidents": [100, 20, np.nan, np.nan, np.nan],
            "2020 Accidents": [110, np.nan, np.nan, np.nan, np.nan],
            "2021 Accidents": [120, np.nan, np.nan, np.nan, np.nan],
            "2022 Accidents": [130, 0, np.nan, np.nan, np.nan],
            "2023 Accidents": [140, np.nan, np.nan, np.nan, np.nan],
        })

    def test_footnote_rows_removed_but_aggregate_row_kept(self):
        cleaned = clean_accident_frame(self.raw, [f"{y} Accidents" for y in range(2019, 2024)], "2023 Accidents")
        self.assertEqual(cleaned["State"].tolist(), ["Rajasthan", "Daman & Diu", "All India"])

    def test_missing_values_remain_missing_and_are_not_low_risk(self):
        cleaned = clean_accident_frame(self.raw, [f"{y} Accidents" for y in range(2019, 2024)], "2023 Accidents")
        d = cleaned.loc[cleaned["State"] == "Daman & Diu"].iloc[0]
        self.assertTrue(pd.isna(d["2020 Accidents"]))
        self.assertTrue(pd.isna(d["Risk Score"]))
        self.assertEqual(d["Risk Category"], "No Data")

    def test_negative_accident_count_rejected(self):
        bad = self.raw.copy()
        bad.loc[0, "2020 Accidents"] = -1
        with self.assertRaisesRegex(ValueError, "Negative accident counts"):
            clean_accident_frame(bad, [f"{y} Accidents" for y in range(2019, 2024)], "2023 Accidents")

    def test_training_rows_exclude_aggregate_and_incomplete_history(self):
        cleaned = clean_accident_frame(self.raw, [f"{y} Accidents" for y in range(2019, 2024)], "2023 Accidents")
        features = [f"{y} Accidents" for y in range(2019, 2023)]
        X, y, eligible, excluded = prepare_training_xy(cleaned, features, "2023 Accidents")
        self.assertEqual(len(X), 1)
        self.assertEqual(eligible["State"].tolist(), ["Rajasthan"])
        self.assertEqual(excluded["State"].tolist(), ["Daman & Diu"])
        self.assertEqual(float(y.iloc[0]), 140.0)


class GeographyTests(unittest.TestCase):
    def test_common_names_and_aliases_normalize_consistently(self):
        self.assertEqual(normalize_map_name("J & K #"), normalize_map_name("Jammu and Kashmir"))
        self.assertEqual(normalize_map_name("Orissa"), normalize_map_name("Odisha"))
        self.assertEqual(normalize_map_name("Uttaranchal"), normalize_map_name("Uttarakhand"))
        self.assertEqual(normalize_map_name("Andaman & Nicobar Islands"), normalize_map_name("Andaman and Nicobar Islands"))


class ScenarioTests(unittest.TestCase):
    def test_zero_growth_preserves_baseline(self):
        result = build_scenario_forecast(1000, 0)
        self.assertEqual(result["Year"].tolist(), list(range(2024, 2036)))
        self.assertEqual(result["Projected Accidents"].tolist(), [1000] * 12)

    def test_compound_growth_for_first_and_last_year(self):
        result = build_scenario_forecast(1000, 5)
        self.assertEqual(int(result.iloc[0]["Projected Accidents"]), 1050)
        self.assertEqual(int(result.iloc[-1]["Projected Accidents"]), int(np.rint(1000 * 1.05**12)))

    def test_negative_growth(self):
        result = build_scenario_forecast(1000, -5)
        self.assertEqual(int(result.iloc[0]["Projected Accidents"]), 950)

    def test_invalid_baseline_rejected(self):
        with self.assertRaises(ValueError):
            build_scenario_forecast(float("nan"), 0)
        with self.assertRaises(ValueError):
            build_scenario_forecast(-1, 0)


class ApplicationSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        compile(cls.source, str(APP_PATH), "exec")

    def test_app_compiles_and_launches_without_login(self):
        self.assertTrue(self.source.rstrip().endswith("main_app()"))
        self.assertNotIn("def login_page()", self.source)
        self.assertNotIn("if st.session_state.authenticated", self.source)

    def test_streamlit_ui_uses_current_width_argument(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("use_container_width=", source)
        self.assertIn('width="stretch"', source)

    def test_theme_uses_requested_palette(self):
        config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        for color in ("#F5E8D8", "#0047AB", "#FFD447"):
            self.assertIn(color, self.source)
        self.assertIn("#F5E8D8", config)
        self.assertIn("#0047AB", config)

    def test_map_supports_both_geojson_names(self):
        self.assertIn('ASSETS_DIR / "india_states.geojson"', self.source)
        self.assertIn('ASSETS_DIR / "india_state.geojson"', self.source)

    def test_model_eval_includes_mean_baseline_and_excludes_incomplete_rows(self):
        self.assertIn('"Mean Baseline": DummyRegressor(strategy="mean")', self.source)
        self.assertIn("prepare_training_xy(df, FEATURE_COLS, TARGET_COL)", self.source)
        self.assertIn("complete independent state/UT rows", self.source)

    def test_source_does_not_contain_common_mojibake(self):
        markers = ("ðŸ", "â€”", "â€“", "âš", "â€", "ï¸", "Ãƒ", "Ã¢")
        self.assertFalse([m for m in markers if m in self.source])


if __name__ == "__main__":
    unittest.main(verbosity=2)
