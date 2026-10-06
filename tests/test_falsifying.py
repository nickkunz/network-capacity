import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluators.falsifying import (
    train_falsified_agreement,
    train_falsified_transfer,
)


class FalsificationRepeatTests(unittest.TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "z": [1.0, 0.0],
                "target": [1.0, 2.0],
                "domain": ["a", "b"],
            }
        )
        self.models = {
            "model": SimpleNamespace(estimator_c = object(), estimator_r = object())
        }

    @staticmethod
    def _logo_result(*args, **kwargs):
        frontier = pd.DataFrame(
            {
                "group": ["a", "b"],
                "ei": [1.0, 1.0],
                "vr": [0.0, 0.0],
                "mv": [0.0, 0.0],
                "ms": [0.0, 0.0],
            }
        )
        return frontier, np.array([1.0, 2.0])

    @staticmethod
    def _frozen_result(*args, **kwargs):
        frontier, predictions = FalsificationRepeatTests._logo_result()
        return frontier, predictions, None

    def _run_trainer(self, trainer) -> tuple[list, list]:
        with (
            patch("src.evaluators.falsifying.logo_cross_valid", side_effect = self._logo_result) as logo_mock,
            patch("src.evaluators.falsifying.logo_cross_valid_frozen", side_effect = self._frozen_result) as frozen_mock,
        ):
            trainer(
                data_proc = self.data,
                data_fals = {"test": self.data.copy()},
                models = self.models,
                feat_x = ["x"],
                feat_z = ["z"],
                n_repeats = 2,
                n_jobs = 1,
            )
        return logo_mock.call_args_list, frozen_mock.call_args_list

    def test_transfer_inner_cv_runs_once_per_outer_seed(self) -> None:
        logo_calls, frozen_calls = self._run_trainer(train_falsified_transfer)

        self.assertEqual(len(logo_calls), 4)
        self.assertEqual(len(frozen_calls), 2)
        self.assertTrue(all(call.kwargs["n_repeats"] == 1 for call in logo_calls + frozen_calls))

    def test_agreement_inner_cv_runs_once_per_outer_seed(self) -> None:
        logo_calls, frozen_calls = self._run_trainer(train_falsified_agreement)

        self.assertEqual(len(logo_calls), 4)
        self.assertEqual(len(frozen_calls), 2)
        self.assertTrue(all(call.kwargs["n_repeats"] == 1 for call in logo_calls + frozen_calls))


if __name__ == "__main__":
    unittest.main()