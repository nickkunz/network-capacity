import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.evaluators.decomposing import (
    _build_nested_capacity_targets,
    _run_single_stage_fold,
    train_decomposed_consensus,
)
from src.evaluators.training import fit_predict_frontier


class FullCorpusConsensusTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame(data = {
            "name": ["one", "two", "three", "four"],
            "domain": ["first", "first", "second", "second"],
            "structure": [1.0, 2.0, 3.0, 4.0],
            "process": [3.0, 1.0, 4.0, 2.0],
            "target": [11.0, 41.0, 6.0, 25.0],
        })
        self.models = {
            "tree": SimpleNamespace(
                estimator_c = DecisionTreeRegressor(random_state = 42),
                estimator_r = DecisionTreeRegressor(random_state = 42),
            ),
        }

    def test_all_specifications_fit_and_predict_the_full_corpus(self) -> None:
        original = self.data.copy(deep = True)
        with patch(
            target = "src.evaluators.decomposing._run_single_stage_fold",
            wraps = _run_single_stage_fold,
        ) as single_stage, patch(
            target = "src.evaluators.decomposing.fit_predict_frontier",
            wraps = fit_predict_frontier,
        ) as two_stage:
            predictions = train_decomposed_consensus(
                data = self.data, models = self.models,
                feat_x = ["structure"], feat_z = ["process"],
                target = "target", n_repeats = 2, random_state = 42, n_jobs = 1,
            )

        expected_specs = ["additive", "interaction", "joint", "invariants", "signatures"]
        self.assertEqual(predictions["specification"].drop_duplicates().tolist(), expected_specs)
        self.assertEqual(set(predictions["evaluation"]), {"full_corpus"})
        self.assertEqual(set(predictions["group"]), {"all"})
        self.assertEqual(len(predictions), len(self.data) * len(expected_specs))
        np.testing.assert_allclose(
            actual = predictions["y_pred"],
            desired = np.tile(A = np.log1p(self.data["target"]), reps = len(expected_specs)),
        )
        self.assertEqual(single_stage.call_count, 6)
        for call in single_stage.call_args_list:
            np.testing.assert_array_equal(actual = call.kwargs["train_idx"], desired = np.arange(len(self.data)))
            np.testing.assert_array_equal(actual = call.kwargs["test_idx"], desired = np.arange(len(self.data)))
        self.assertEqual(
            [call.kwargs["random_state"] for call in single_stage.call_args_list],
            [42, 43, 42, 43, 42, 43],
        )
        self.assertEqual(two_stage.call_count, 2)
        for call in two_stage.call_args_list:
            self.assertEqual(call.kwargs["data"]["name"].tolist(), self.data["name"].tolist())
            self.assertEqual(call.kwargs["n_repeat"], 2)
        self.assertEqual(two_stage.call_args.kwargs["feat_z"], ["process", "structure_x_process"])
        pd.testing.assert_frame_equal(left = self.data, right = original)

    def test_single_stage_predictions_are_averaged_before_compilation(self) -> None:
        def seeded_prediction(**kwargs: object) -> dict:
            return {
                "kept_indices": np.arange(len(self.data)),
                "y_pred": np.full(shape = len(self.data), fill_value = kwargs["random_state"], dtype = float),
            }

        with patch(
            target = "src.evaluators.decomposing._run_single_stage_fold",
            side_effect = seeded_prediction,
        ):
            predictions = train_decomposed_consensus(
                data = self.data, models = self.models,
                feat_x = ["structure"], feat_z = ["process"],
                target = "target", n_repeats = 2, random_state = 42, n_jobs = 1,
            )
        single_stage = predictions.loc[
            predictions["specification"].isin(["joint", "invariants", "signatures"])
        ]
        np.testing.assert_allclose(actual = single_stage["y_pred"], desired = 42.5)


class ResidualAttributionTests(unittest.TestCase):

    def test_nested_capacity_targets_exclude_outer_test_from_training(self) -> None:
        X = pd.DataFrame(data = {"structure": np.arange(6, dtype = float)})
        Z = pd.DataFrame(data = {"process": np.arange(6, dtype = float)})
        y_star = pd.Series(data = np.arange(10, 16, dtype = float))
        groups = np.array(["first", "first", "second", "second", "third", "third"])
        outer_train = np.array([0, 1, 2, 3])
        outer_test = np.array([4, 5])

        def capacity_fold(**kwargs: object) -> dict:
            test_idx = np.asarray(a = kwargs["test_idx"])
            return {
                "kept_indices": test_idx,
                "c_hat": np.zeros(shape = len(test_idx), dtype = float),
                "slack": y_star.iloc[test_idx].to_numpy(),
            }

        with patch(
            target = "src.evaluators.decomposing._run_capacity_fold",
            side_effect = capacity_fold,
        ) as capacity_worker:
            result = _build_nested_capacity_targets(
                train_idx = outer_train,
                test_idx = outer_test,
                X = X,
                Z = Z,
                y_star = y_star,
                feat_x = ["structure"],
                feat_z = ["process"],
                estimator_c = DecisionTreeRegressor(random_state = 42),
                groups = groups,
                random_state = 42,
            )

        self.assertIsNotNone(result)
        slack_nested, c_hat_outer = result
        self.assertEqual(capacity_worker.call_count, 3)
        for call in capacity_worker.call_args_list:
            self.assertTrue(set(call.kwargs["train_idx"]).isdisjoint(outer_test))
        np.testing.assert_allclose(actual = slack_nested, desired = y_star)
        self.assertTrue(np.isnan(c_hat_outer[outer_train]).all())
        np.testing.assert_allclose(actual = c_hat_outer[outer_test], desired = 0.0)


if __name__ == "__main__":
    unittest.main()