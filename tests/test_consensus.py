import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluators.decomposing import compile_decomposed_consensus, compile_decomposed_full
from src.evaluators.falsifying import compile_falsified_full
from src.evaluators.perturbing import compile_perturbed_full
from src.evaluators.predicting import compile_full_corpus_agreement


class FullCorpusAgreementTests(unittest.TestCase):

    def test_agreement_uses_all_systems(self) -> None:
        targets = np.array([1.0, 4.0, 2.0, 3.0])
        result = compile_full_corpus_agreement(predictions = {"model": targets}, y_true = targets)
        self.assertAlmostEqual(result.loc[0, "ci"], 1.0)
        self.assertEqual(result.loc[0, "group"], "all")
        self.assertEqual(result.loc[0, "evaluation"], "full_corpus")

    def test_misaligned_predictions_are_rejected(self) -> None:
        with self.assertRaisesRegex(expected_exception = ValueError, expected_regex = "target shape"):
            compile_full_corpus_agreement(
                predictions = {"model": np.array([1.0, 2.0])},
                y_true = np.array([1.0, 2.0, 3.0]),
            )

    @patch("src.evaluators.predicting.consensus_metrics")
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