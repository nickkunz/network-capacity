import unittest
from unittest.mock import patch

import dcor
import numpy as np
import pandas as pd

from src.evaluators.decomposing import compile_decomposed_consensus, compile_decomposed_full
from src.evaluators.falsifying import compile_falsified_full
from src.evaluators.metrics import _distance_corr, frontier_consensus
from src.evaluators.perturbing import compile_perturbed_full
from src.evaluators.predicting import compile_corpus_full


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