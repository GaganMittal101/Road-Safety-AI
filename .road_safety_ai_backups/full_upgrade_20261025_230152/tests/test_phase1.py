from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"


def safe_num(value, default=np.nan):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def get_build_forecast():
    """Load only the pure forecast helper, without importing/running Streamlit UI."""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_forecast"
    )
    module = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="numpy", asname="np")]),
            ast.Import(names=[ast.alias(name="pandas", asname="pd")]),
            function,
        ],
        type_ignores=[],
    )
    namespace = {"safe_num": safe_num}
    exec(compile(ast.fix_missing_locations(module), str(APP_PATH), "exec"), namespace)
    return namespace["build_forecast"]


class PhaseOneSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.build_forecast = staticmethod(get_build_forecast())

    def test_forecast_years_are_2024_through_2035(self):
        result = self.build_forecast(1000, 0)
        self.assertEqual(result["Year"].tolist(), list(range(2024, 2036)))
        self.assertEqual(len(result), 12)

    def test_zero_growth_preserves_baseline(self):
        result = self.build_forecast(1000, 0)
        self.assertEqual(result["Projected Accidents"].tolist(), [1000] * 12)

    def test_positive_growth_applies_one_year_to_2024_and_twelve_years_to_2035(self):
        result = self.build_forecast(1000, 5)
        self.assertEqual(int(result.iloc[0]["Projected Accidents"]), 1050)
        self.assertEqual(int(result.iloc[-1]["Projected Accidents"]), int(np.rint(1000 * 1.05**12)))

    def test_negative_growth_applies_one_year_to_2024(self):
        result = self.build_forecast(1000, -5)
        self.assertEqual(int(result.iloc[0]["Projected Accidents"]), 950)

    def test_geojson_loader_supports_both_repository_filenames(self):
        self.assertIn('"india_states.geojson"', self.source)
        self.assertIn('"india_state.geojson"', self.source)

    def test_model_target_is_labeled_2023_not_2024(self):
        self.assertIn('"🔮 2023 Model Estimates"', self.source)
        self.assertNotIn('"🔮 2024 Model Estimates"', self.source)
        self.assertNotIn('"3. 2024 Model Predictions"', self.source)

    def test_direct_dashboard_entry_without_login_gate(self):
        self.assertNotIn("def login_page()", self.source)
        self.assertTrue(self.source.rstrip().endswith("main_app()"))

    def test_national_inference_uses_raw_features(self):
        self.assertIn("model_input = X", self.source)
        self.assertNotIn("model_input = _scaled_input(X, scaler)", self.source)

    def test_source_has_no_common_mojibake_markers(self):
        markers = ("ðŸ", "â€”", "â€“", "âš", "â€", "ï¸", "Ãƒ", "Ã¢")
        self.assertFalse([marker for marker in markers if marker in self.source])


if __name__ == "__main__":
    unittest.main(verbosity=2)
