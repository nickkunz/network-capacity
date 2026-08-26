import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.helpers import _create_igraph_object
from src.data.loaders.windmill import _process_events_wind
from src.evaluators.metrics import spec_marginal_delta
from src.evaluators.perturbing import find_perturbed_max, stat_perturbed_tost
from src.vectorizers.invariants import BipartiteInvariants, GraphInvariants


class BipartiteInvariantsTests(unittest.TestCase):

    def test_random_walk_fourth_moment_matches_explicit_graph(self) -> None:
        left = ["left_0", "left_1"]
        right = ["right_0", "right_1", "right_2"]
        graph = _create_igraph_object(
            nodes = left + right,
            edges = [(source, target) for source in left for target in right],
        )

        expected = GraphInvariants(graph = graph).spectral()["random_walk_fourth_moment"]
        actual = BipartiteInvariants(m = len(left), n = len(right)).spectral()["random_walk_fourth_moment"]

        self.assertAlmostEqual(actual, 2.0 / graph.vcount())
        self.assertAlmostEqual(actual, expected)


class CreateIgraphObjectTests(unittest.TestCase):

    def test_create_igraph_object_returns_simple_graph(self) -> None:
        graph = _create_igraph_object(
            nodes = ["a", "b"],
            edges = [("a", "b"), ("b", "a"), ("a", "a")],
        )

        self.assertTrue(graph.is_simple())
        self.assertEqual(graph.ecount(), 1)
        self.assertEqual(graph.degree(), [1, 1])


class WindmillEventTests(unittest.TestCase):

    def test_positive_production_below_mean_remains_on(self) -> None:
        production = np.array([[0.1], [0.2], [1.0], [0.2]])

        events = _process_events_wind(data = production, hours = 4)

        self.assertEqual(events["target"].tolist(), [0])


class SpecMarginalDeltaTests(unittest.TestCase):

    def test_spec_marginal_delta_rounds_to_nearest_decimal(self) -> None:
        results = pd.DataFrame(
            {
                "reference": ["baseline", "baseline"],
                "lower_margin": [0.0, 0.084],
                "upper_margin": [0.0, 0.086],
            }
        )

        lower_margin = spec_marginal_delta(
            results = results,
            feat_value = ["lower_margin"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "max",
            decimals = 2,
        )
        upper_margin = spec_marginal_delta(
            results = results,
            feat_value = ["upper_margin"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "max",
            decimals = 2,
        )

        self.assertEqual(lower_margin, 0.08)
        self.assertEqual(upper_margin, 0.09)


class PerturbationStatisticsTests(unittest.TestCase):

    @patch("src.evaluators.perturbing.wilcoxon")
    def test_tost_uses_full_precision_before_holm(self, wilcoxon_mock) -> None:
        wilcoxon_mock.side_effect = [
            (0.0, 0.0049), (0.0, 0.001), (1.0, 0.5),
            (0.0, 0.0251), (0.0, 0.001), (1.0, 0.5),
        ]
        results = pd.DataFrame(
            {
                "model": ["a", "b", "a", "b", "a", "b", "a", "b"],
                "group": ["g1", "g2", "g1", "g2", "g1", "g2", "g1", "g2"],
                "perturbation": ["baseline"] * 4 + ["invariants"] * 4,
                "method": [None] * 4 + ["noise", "noise", "jitter", "jitter"],
                "ei": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            }
        )

        summary = stat_perturbed_tost(
            results = results,
            feat_value = ["ei"],
            feat_group = ["method"],
            decimals = 2,
            index = False,
        )

        self.assertEqual(summary["Holm-adj. p"].tolist(), ["0.01", "0.03"])

    def test_find_perturbed_max_uses_method_severity(self) -> None:
        results = pd.DataFrame(
            {
                "perturbation": ["invariants"] * 6,
                "method": ["subset", "subset", "scaling", "scaling", "noise", "noise"],
                "intensity": [0.65, 0.95, 0.25, 1.75, 0.05, 0.35],
            }
        )

        strongest = find_perturbed_max(results = results)
        selected = strongest.set_index("method")["intensity"].to_dict()

        self.assertEqual(selected, {"subset": 0.65, "scaling": 0.25, "noise": 0.35})