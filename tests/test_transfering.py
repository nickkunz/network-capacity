import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluators.transfering import (
    compile_transfer_resampling,
    spec_transfer_delta,
    stat_transfer_tost,
)


class TransferResamplingTests(unittest.TestCase):

    def test_compiles_common_group_metrics_from_aligned_predictions(self) -> None:
        target = np.exp(np.array([0.0, 1.0, 0.0, 1.0], dtype = float))
        data = pd.DataFrame({
            "target": target,
            "domain": ["a", "a", "b", "b"],
        })
        prediction = np.log1p(target)

        result = compile_transfer_resampling(
            predictions = {"Domain (LOGO)": {"model": prediction}},
            data = data,
            target = "target",
            group = "domain",
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["group"].tolist(), ["a", "b"])
        self.assertTrue(np.allclose(result["ei"], 1.0))
        self.assertTrue(np.allclose(result["ci"], 1.0))

    def test_rejects_prediction_vectors_that_do_not_align_with_data(self) -> None:
        data = pd.DataFrame({
            "target": [1.0, 2.0, 3.0, 4.0],
            "domain": ["a", "a", "b", "b"],
        })

        with self.assertRaisesRegex(ValueError, "expected"):
            compile_transfer_resampling(
                predictions = {
                    "Domain (LOGO)": {
                        "model": np.array([0.0, 1.0], dtype = float),
                    },
                },
                data = data,
                target = "target",
                group = "domain",
            )

    def test_transfer_delta_uses_equally_weighted_model_means(self) -> None:
        results = pd.DataFrame({
            "regime": ["Domain (LOGO)"] * 6,
            "model": ["a", "a", "b", "b", "c", "c"],
            "group": ["x", "y"] * 3,
            "ei": [0.0, 1.0, 0.4, 0.6, 0.8, 1.0],
        })

        delta = spec_transfer_delta(
            results = results,
            metric = "ei",
            reference = "Domain (LOGO)",
            scale = 1.0,
            decimals = 2,
        )

        self.assertEqual(delta, 0.2)

    @patch("src.evaluators.transfering.wilcoxon")
    def test_tost_uses_raw_effects_and_full_precision_holm(
        self,
        wilcoxon_mock,
        ) -> None:
        wilcoxon_mock.side_effect = [
            (0.0, 0.0049),
            (0.0, 0.0010),
            (0.0, 0.0251),
            (0.0, 0.0010),
        ]
        models = ["a", "b", "c", "d"]
        reference = np.full(shape = 4, fill_value = 0.5)
        comparison = np.array([0.75, 0.375, 0.5, 0.625], dtype = float)
        results = pd.DataFrame({
            "regime": (
                ["Domain (LOGO)"] * 4
                + ["Discipline (LOGO)"] * 4
                + ["Random (5-Fold)"] * 4
            ),
            "model": models * 3,
            "group": ["domain"] * 12,
            "ei": np.concatenate([reference, comparison, comparison]),
        })

        summary = stat_transfer_tost(
            results = results,
            metric = "ei",
            delta = 0.5,
            reference = "Domain (LOGO)",
            comparisons = ("Discipline (LOGO)", "Random (5-Fold)"),
            decimals = 2,
            index = False,
        )

        self.assertEqual(summary["Rank-biserial r"].tolist(), ["0.50", "0.50"])
        self.assertEqual(summary["Holm-adj. p"].tolist(), ["0.010", "0.025"])
        self.assertEqual(summary["Eq."].tolist(), ["Yes", "Yes"])

    def test_tost_rejects_duplicate_pairing_keys(self) -> None:
        results = pd.DataFrame({
            "regime": ["Domain (LOGO)", "Domain (LOGO)", "Random (5-Fold)"],
            "model": ["model", "model", "model"],
            "group": ["domain", "domain", "domain"],
            "ei": [0.5, 0.5, 0.5],
        })

        with self.assertRaisesRegex(ValueError, "duplicated"):
            stat_transfer_tost(
                results = results,
                metric = "ei",
                delta = 0.1,
                reference = "Domain (LOGO)",
                comparisons = ("Random (5-Fold)",),
                decimals = 2,
                index = False,
            )


if __name__ == "__main__":
    unittest.main()