import unittest
from unittest.mock import patch

import dcor
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluators.decomposing import compile_decomposed_consensus, compile_decomposed_full
from src.evaluators.falsifying import compile_falsified_full, stat_falsified_summary
from src.evaluators.metrics import _distance_corr, _rescaled_spearman_rho, consensus_index, frontier_consensus
from src.evaluators.perturbing import compile_perturbed_full
from src.evaluators.predicting import compile_corpus_full, results_prediction_consensus


class RescaledSpearmanTests(unittest.TestCase):

    def test_reported_rho_is_rescaled_without_changing_ci(self) -> None:
        y_true = np.array(object = [1.0, 2.0, 3.0, 4.0])
        predictions = (
            (y_true, 1.0),
            (y_true[::-1], 0.0),
            (np.array(object = [2.0, 4.0, 1.0, 3.0]), 0.5),
        )
        for y_pred, expected_rho in predictions:
            with self.subTest(y_pred = y_pred):
                metrics = frontier_consensus(y_true = y_true, y_pred = y_pred)
                raw_rho = float(spearmanr(a = y_true, b = y_pred).statistic)
                self.assertAlmostEqual(first = metrics["rho"], second = (raw_rho + 1.0) / 2.0)
                self.assertAlmostEqual(first = metrics["rho"], second = expected_rho)
                self.assertAlmostEqual(
                    first = _rescaled_spearman_rho(y_true = y_true, y_pred = y_pred),
                    second = expected_rho,
                )
                self.assertAlmostEqual(
                    first = metrics["ci"],
                    second = (expected_rho * metrics["rbo"] * metrics["dcr"]) ** (1.0 / 3.0),
                )
                self.assertAlmostEqual(
                    first = metrics["ci"],
                    second = consensus_index(rho = metrics["rho"], rbo = metrics["rbo"], dcr = metrics["dcr"]),
                )

    def test_undefined_rho_preserves_neutral_agreement(self) -> None:
        for values in ([], [1.0], [1.0, 1.0, 1.0]):
            with self.subTest(values = values):
                vector = np.array(object = values, dtype = float)
                self.assertEqual(
                    first = _rescaled_spearman_rho(y_true = vector, y_pred = vector),
                    second = 0.5,
                )
                self.assertEqual(first = frontier_consensus(y_true = vector, y_pred = vector)["ci"], second = 0.0)

    def test_summary_tables_preserve_rho_label_without_transforming_again(self) -> None:
        targets = np.array(object = [1.0, 2.0, 3.0, 4.0])
        results = compile_corpus_full(predictions = {"reverse": targets[::-1]}, y_true = targets)
        prediction_summary = results_prediction_consensus(results = results, print_summary = False)
        self.assertEqual(first = prediction_summary.loc["all", "ρ"], second = "0.00")
        falsified_summary = stat_falsified_summary(
            results = results.assign(condition = "original", Falsification = "original", Method = "original"),
            metrics = ["rho", "rbo", "dcr", "ci"],
            decimals = 2,
        )
        self.assertEqual(first = falsified_summary.loc[("original", "original"), "ρ"], second = 0.0)


class DistanceCorrelationTests(unittest.TestCase):

    def test_constant_vectors_return_zero(self) -> None:
        for size in (3, 10, 25, 64):
            varying = np.linspace(start = 0.0, stop = 1.0, num = size)
            for constant in (0.0, 0.1, 1.23456789):
                fixed = np.full(shape = size, fill_value = constant)
                for y_true, y_pred in ((fixed, varying), (varying, fixed), (fixed, fixed)):
                    with self.subTest(size = size, constant = constant, y_true = y_true, y_pred = y_pred):
                        self.assertEqual(
                            first = _distance_corr(y_true = y_true, y_pred = y_pred),
                            second = 0.0,
                        )

    def test_constant_inputs_skip_distance_estimator(self) -> None:
        varying = np.linspace(start = 0.0, stop = 1.0, num = 10)
        fixed = np.full(shape = 10, fill_value = 0.1)
        with patch(target = "src.evaluators.metrics.dcor.distance_correlation") as estimator:
            self.assertEqual(
                first = _distance_corr(y_true = varying, y_pred = fixed),
                second = 0.0,
            )
            estimator.assert_not_called()

    def test_empty_and_single_value_inputs_return_zero(self) -> None:
        varying = np.array(object = [1.0, 2.0, 3.0])
        for values in ([], [0.1]):
            short = np.array(object = values, dtype = float)
            for y_true, y_pred in ((short, varying), (varying, short), (short, short)):
                with self.subTest(values = values, y_true = y_true, y_pred = y_pred):
                    self.assertEqual(
                        first = _distance_corr(y_true = y_true, y_pred = y_pred),
                        second = 0.0,
                    )

    def test_nonconstant_inputs_preserve_distance_estimator(self) -> None:
        rng = np.random.default_rng(seed = 42)
        for size in (3, 10, 25):
            with self.subTest(size = size):
                y_true = rng.normal(loc = 0.0, scale = 1.0, size = size)
                y_pred = rng.normal(loc = 0.0, scale = 1.0, size = size)
                self.assertAlmostEqual(
                    first = _distance_corr(y_true = y_true, y_pred = y_pred),
                    second = float(dcor.distance_correlation(x = y_true, y = y_pred)),
                    places = 12,
                )

    def test_distance_estimator_failures_propagate(self) -> None:
        y_true = np.array(object = [1.0, 2.0, 3.0])
        y_pred = np.array(object = [1.5, 1.0, 3.0])
        for scorer in (_distance_corr, frontier_consensus):
            with self.subTest(scorer = scorer.__name__):
                with patch(
                    target = "src.evaluators.metrics.dcor.distance_correlation",
                    side_effect = RuntimeError("distance estimator failed"),
                ):
                    with self.assertRaisesRegex(expected_exception = RuntimeError, expected_regex = "distance estimator failed"):
                        scorer(y_true = y_true, y_pred = y_pred)

    def test_nearly_constant_inputs_are_not_treated_as_constant(self) -> None:
        y_true = np.array(object = [1.0, 2.0, 3.0])
        y_pred = np.array(object = [0.1, 0.1, 0.1 + np.finfo(dtype = float).eps])
        with patch(
            target = "src.evaluators.metrics.dcor.distance_correlation", return_value = 0.75,
        ) as estimator:
            self.assertEqual(
                first = _distance_corr(y_true = y_true, y_pred = y_pred),
                second = 0.75,
            )
            estimator.assert_called_once()

    def test_constant_vectors_have_zero_consensus(self) -> None:
        varying = np.linspace(start = 0.0, stop = 1.0, num = 10)
        fixed = np.full(shape = 10, fill_value = 0.1)
        for y_true, y_pred in ((fixed, varying), (varying, fixed), (fixed, fixed)):
            with self.subTest(y_true = y_true, y_pred = y_pred):
                metrics = frontier_consensus(y_true = y_true, y_pred = y_pred)
                self.assertEqual(first = metrics["dcr"], second = 0.0)
                self.assertEqual(first = metrics["ci"], second = 0.0)


class FullCorpusAgreementTests(unittest.TestCase):

    def test_agreement_uses_all_systems(self) -> None:
        targets = np.array([1.0, 4.0, 2.0, 3.0])
        result = compile_corpus_full(predictions = {"model": targets}, y_true = targets)
        self.assertAlmostEqual(result.loc[0, "ci"], 1.0)
        self.assertEqual(result.loc[0, "group"], "all")
        self.assertEqual(result.loc[0, "evaluation"], "full_corpus")

    def test_misaligned_predictions_are_rejected(self) -> None:
        with self.assertRaisesRegex(expected_exception = ValueError, expected_regex = "target shape"):
            compile_corpus_full(
                predictions = {"model": np.array([1.0, 2.0])},
                y_true = np.array([1.0, 2.0, 3.0]),
            )

    @patch("src.evaluators.predicting.frontier_consensus")
    def test_perturbation_agreement_averages_predictions_first(self, scorer) -> None:
        scorer.side_effect = lambda y_true, y_pred: {
            metric: float(np.mean(y_pred) ** 2) for metric in ("rho", "rbo", "dcr", "ci")
        }
        result = compile_perturbed_full(
            results = {
                "baseline": {"model": np.array([0.0, 1.0])},
                "perturbed": [
                    {
                        "model": "model", "pert_type": "invariants", "method": "noise",
                        "intensity": 0.35, "realization": realization,
                        "y_pred": np.array([value, value]),
                    }
                    for realization, value in enumerate((0.2, 0.6))
                ],
            },
            data = pd.DataFrame(data = {"target": [0.0, 1.0]}),
            target = "target",
        )
        self.assertAlmostEqual(result.loc[result["perturbation"] == "invariants", "ci"].iloc[0], 0.16)
        self.assertEqual(scorer.call_count, 2)
        np.testing.assert_allclose(actual = scorer.call_args.kwargs["y_pred"], desired = [0.4, 0.4])

    def test_falsification_scores_against_the_falsified_targets(self) -> None:
        data = pd.DataFrame(data = {"target": [1.0, 3.0, 7.0, 15.0]})
        permuted = data.iloc[::-1].reset_index(drop = True)
        predictions = {"model": np.log1p(data["target"].to_numpy())}
        result = compile_falsified_full(
            results = {"original": predictions, "falsified": {"frozen": {"target remap": predictions}}},
            data_proc = data, data_fals = {"target remap": permuted}, target = "target",
        )
        self.assertAlmostEqual(result.loc[result["condition"] == "original", "ci"].iloc[0], 1.0)
        self.assertAlmostEqual(result.loc[result["condition"] == "falsified", "ci"].iloc[0], 0.0)

    def test_decomposition_compilers_preserve_full_corpus_provenance(self) -> None:
        predictions = pd.DataFrame(data = {
            "model": ["first"] * 3 + ["second"] * 3,
            "specification": ["joint"] * 6,
            "dataset": ["a", "b", "c"] * 2,
            "group": ["all"] * 6,
            "y_true": [1.0, 2.0, 3.0] * 2,
            "y_pred": [1.0, 2.0, 3.0] * 2,
            "evaluation": ["full_corpus"] * 6,
        })
        for result in (
            compile_decomposed_consensus(predictions = predictions),
            compile_decomposed_full(predictions = predictions),
        ):
            self.assertEqual(set(result["evaluation"]), {"full_corpus"})
            np.testing.assert_allclose(actual = result["ci"], desired = 1.0)
        with self.assertRaisesRegex(expected_exception = ValueError, expected_regex = "Full-corpus"):
            compile_decomposed_full(predictions = predictions.drop(columns = "evaluation"))


if __name__ == "__main__":
    unittest.main()