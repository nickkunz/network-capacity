import unittest
from unittest.mock import patch

import igraph as ig
import numpy as np
import pandas as pd

from src.data.helpers import _create_igraph_object
from src.data.loaders.windmill import _process_events_wind
from src.evaluators.config import FEAT_X
from src.evaluators.decomposing import stat_decomposed_summary, stat_decomposed_test
from src.evaluators.falsifying import stat_falsified_test
from src.evaluators.metrics import paired_rank_biserial, spec_marginal_delta
from src.evaluators.perturbing import (
    _iter_perturbation_realizations,
    analytical_perturb,
    compile_perturbed_consensus,
    compile_perturbed_recovery,
    compile_perturbed_transfer,
    feature_perturb,
    find_perturbed_max,
    network_perturb,
    stat_perturbed_tost,
)
from src.vectorizers.invariants import (
    BipartiteInvariants, 
    GraphInvariants
)


class RankBiserialTests(unittest.TestCase):

    def test_uses_nonzero_raw_differences_with_test_positive_sign(self) -> None:
        differences = np.array([3.0, -2.0, 0.0, 2.0])

        actual = paired_rank_biserial(differences = differences)

        self.assertAlmostEqual(actual, 0.5)

    def test_returns_nan_without_nonzero_differences(self) -> None:
        self.assertTrue(np.isnan(paired_rank_biserial(differences = [0.0, np.nan])))

    def test_decomposition_summary_uses_unshifted_specification_difference(self) -> None:
        models = ["a", "b", "c", "d"]
        reference = np.full(shape = 4, fill_value = 0.5)
        tested = np.array([0.75, 0.375, 0.5, 0.625])
        results = pd.DataFrame({
            "model": models * 2,
            "group": ["g"] * 8,
            "specification": ["additive"] * 4 + ["joint"] * 4,
            "ei": np.concatenate([reference, tested]),
        })

        summary = stat_decomposed_test(
            results = results,
            delta = 0.5,
            metric = "ei",
            specs = ("joint",),
            direction = "noninferiority",
            decimals = 2,
            index = False,
        )

        self.assertEqual(summary.loc[0, "Rank-biserial r"], "0.50")
        self.assertEqual(summary.loc[0, "Ablation"], "Joint")

    def test_decomposition_tables_populate_display_labels(self) -> None:
        specs = ("additive", "interaction", "joint", "invariants", "signatures", "interaction_joint")
        labels = ["Original", "Interaction", "Joint", "Invariants", "Signatures", "Interaction Joint"]
        results = pd.DataFrame({
            "model": ["m"] * (2 * len(specs)),
            "group": ["g1", "g2"] * len(specs),
            "specification": [spec for spec in specs for _ in range(2)],
            "ei": [0.5, 0.6] * len(specs),
        })

        summary = stat_decomposed_summary(results = results, metrics = ("ei",))
        self.assertEqual(summary.index.get_level_values("Method").tolist(), labels)

        reversed_summary = stat_decomposed_summary(results = results.iloc[::-1], metrics = ("ei",))
        self.assertEqual(reversed_summary.index.get_level_values("Method").tolist(), labels[::-1])
        ordered_summary = stat_decomposed_summary(
            results = results.iloc[::-1], metrics = ("ei",), spec_order = specs,
        )
        self.assertEqual(ordered_summary.index.get_level_values("Method").tolist(), labels)

        test = stat_decomposed_test(
            results = results, specs = specs[1:], delta = 0.5, index = False,
        )
        self.assertEqual(test["Ablation"].tolist(), labels[1:])

    def test_falsification_summary_uses_falsified_minus_original_sign(self) -> None:
        models = ["a", "b", "c", "d"]
        original = np.full(shape = 4, fill_value = 0.5)
        falsified = np.array([0.75, 0.375, 0.5, 0.625])
        results = pd.DataFrame({
            "model": models * 2,
            "group": ["g"] * 8,
            "Falsification": ["frozen"] * 8,
            "Method": ["target_remap"] * 8,
            "condition": ["original"] * 4 + ["falsified"] * 4,
            "ei": np.concatenate([original, falsified]),
        })

        summary = stat_falsified_test(
            results = results,
            feat_value = ["ei"],
            feat_pairs = ["model", "group"],
            decimals = 2,
            index = False,
        )

        self.assertEqual(summary.loc[0, "Rank-biserial r"], "0.50")


class BipartiteInvariantsTests(unittest.TestCase):

    def test_all_invariants_match_explicit_complete_bipartite_graphs(self) -> None:
        for m, n in ((0, 4), (1, 1), (1, 4), (2, 3), (3, 3), (4, 7)):
            with self.subTest(m = m, n = n):
                graph = ig.Graph.Full_Bipartite(m, n)
                explicit = GraphInvariants(graph = graph).all()
                analytical = BipartiteInvariants(m = m, n = n).all()
                self.assertEqual(list(explicit), FEAT_X)
                self.assertEqual(list(analytical), FEAT_X)
                for key in explicit:
                    self.assertAlmostEqual(explicit[key], analytical[key], places = 12, msg = key)
                if m == 0 or n == 0:
                    self.assertEqual(analytical["k_core_size"], m + n)

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

    def test_analytical_sampling_matches_seeded_explicit_perturbation(self) -> None:
        for m, n in ((1, 4), (2, 3), (4, 7), (7, 4)):
            for intensity in (0.0, 0.20, 0.35, 1.0):
                for random_state in (1, 42, 99):
                    with self.subTest(m = m, n = n, intensity = intensity, random_state = random_state):
                        graph = ig.Graph.Full_Bipartite(m, n)
                        _, partition_types = graph.is_bipartite(return_types = True)
                        expected = network_perturb(
                            graph = graph,
                            method = "sample",
                            intensity = intensity,
                            random_state = random_state,
                        )
                        actual = analytical_perturb(
                            dimensions = (m, n),
                            partition_types = partition_types,
                            method = "uniform_node_sampling",
                            intensity = intensity,
                            random_state = random_state,
                        )
                        self.assertEqual(list(actual), FEAT_X)
                        for key in expected:
                            self.assertAlmostEqual(actual[key], expected[key], places = 12, msg = key)

    def test_graph_free_analytical_sampling_matches_canonical_explicit_graph(self) -> None:
        for m, n in ((1, 4), (2, 3), (7, 4)):
            with self.subTest(m = m, n = n):
                graph = ig.Graph.Full_Bipartite(m, n)
                expected = network_perturb(
                    graph = graph,
                    method = "sample",
                    intensity = 0.35,
                    random_state = 42,
                )
                actual = analytical_perturb(
                    dimensions = (m, n),
                    method = "uniform_node_sampling",
                    intensity = 0.35,
                    random_state = 42,
                )
                for key in expected:
                    self.assertAlmostEqual(actual[key], expected[key], places = 12, msg = key)

    def test_analytical_surrogates_return_canonical_invariant_vector(self) -> None:
        baseline = BipartiteInvariants(m = 2, n = 3).all()
        for method in ("degree_preserving_rewire", "bernoulli_edge_densification"):
            with self.subTest(method = method):
                identity = analytical_perturb(
                    dimensions = (2, 3),
                    method = method,
                    intensity = 0.0,
                )
                actual = analytical_perturb(
                    dimensions = (2, 3),
                    method = method,
                    intensity = 0.35,
                )

                self.assertEqual(list(actual), FEAT_X)
                self.assertEqual(len(actual), 21)
                self.assertTrue(np.isfinite(list(actual.values())).all())
                self.assertEqual(actual["n_nodes"], 5)
                for key in FEAT_X:
                    self.assertAlmostEqual(identity[key], baseline[key], places = 12, msg = key)

        rewired = analytical_perturb(
            dimensions = (2, 3),
            method = "degree_preserving_rewire",
            intensity = 0.35,
        )
        self.assertEqual(rewired["n_nodes"], baseline["n_nodes"])
        self.assertEqual(rewired["n_edges"], baseline["n_edges"])

        densified = analytical_perturb(
            dimensions = (2, 3),
            method = "bernoulli_edge_densification",
            intensity = 0.35,
        )
        self.assertEqual(densified["n_nodes"], baseline["n_nodes"])
        self.assertEqual(densified["n_edges"], 8)


class CreateIgraphObjectTests(unittest.TestCase):

    def test_create_igraph_object_returns_simple_graph(self) -> None:
        graph = _create_igraph_object(
            nodes = ["a", "b"],
            edges = [("a", "b"), ("b", "a"), ("a", "a")],
        )

        self.assertTrue(graph.is_simple())
        self.assertEqual(graph.ecount(), 1)
        self.assertEqual(graph.degree(), [1, 1])

    def test_edgeless_graph_has_zero_spectral_moments(self) -> None:
        invariants = GraphInvariants(graph = _create_igraph_object(
            nodes = ["a", "b"],
            edges = [],
        )).all()

        self.assertEqual(invariants["n_edges"], 0)
        self.assertEqual(invariants["normalized_laplacian_second_moment"], 0.0)
        self.assertEqual(invariants["adjacency_fourth_moment_per_node"], 0.0)


class GraphInvariantsEqualityTests(unittest.TestCase):

    @staticmethod
    def _dense_reference(graph) -> dict:
        adjacency = np.asarray(graph.get_adjacency().data, dtype = float)
        n = adjacency.shape[0]
        degrees = adjacency.sum(axis = 1)
        d_inv_sqrt = np.zeros_like(degrees)
        np.divide(1.0, np.sqrt(degrees), out = d_inv_sqrt, where = degrees > 0)
        a_hat = d_inv_sqrt[:, None] * adjacency * d_inv_sqrt[None, :]
        random_walk = np.zeros_like(adjacency)
        np.divide(adjacency, degrees[:, None], out = random_walk, where = degrees[:, None] > 0)
        laplacian = np.diag((degrees > 0).astype(float)) - a_hat
        return {
            "normalized_laplacian_second_moment": np.trace(np.linalg.matrix_power(laplacian, 2)) / n,
            "normalized_laplacian_third_moment": np.trace(np.linalg.matrix_power(laplacian, 3)) / n,
            "random_walk_triangle_weight": np.trace(np.linalg.matrix_power(random_walk, 3)) / n,
            "random_walk_fourth_moment": np.trace(np.linalg.matrix_power(random_walk, 4)) / n,
            "adjacency_fourth_moment_per_node": np.trace(np.linalg.matrix_power(adjacency, 4)) / n,
        }

    def test_spectral_matches_dense_reference_on_small_graphs(self) -> None:
        graphs = [
            _create_igraph_object(  ## path
                nodes = [f"n{i}" for i in range(5)],
                edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")],
            ),
            _create_igraph_object(  ## triangle with tail
                nodes = [f"n{i}" for i in range(5)],
                edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n0"), ("n2", "n3"), ("n3", "n4")],
            ),
            _create_igraph_object(  ## star
                nodes = [f"n{i}" for i in range(6)],
                edges = [("n0", f"n{i}") for i in range(1, 6)],
            ),
            _create_igraph_object(  ## bipartite cycle
                nodes = [f"n{i}" for i in range(6)],
                edges = [(f"n{i}", f"n{(i + 1) % 6}") for i in range(6)],
            ),
            _create_igraph_object(  ## complete graph
                nodes = [f"n{i}" for i in range(5)],
                edges = [(f"n{i}", f"n{j}") for i in range(5) for j in range(i + 1, 5)],
            ),
            _create_igraph_object(  ## triangle with an isolate
                nodes = [f"n{i}" for i in range(4)],
                edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n0")],
            ),
        ]
        for graph in graphs:
            with self.subTest(n_nodes = graph.vcount(), n_edges = graph.ecount()):
                actual = GraphInvariants(graph = graph).spectral()
                expected = self._dense_reference(graph = graph)
                for key, value in expected.items():
                    self.assertAlmostEqual(actual[key], value, places = 12, msg = key)

    def test_spectral_block_streaming_matches_single_block(self) -> None:
        graph = _create_igraph_object(
            nodes = [f"n{i}" for i in range(8)],
            edges = [
                ("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"),
                ("n4", "n5"), ("n5", "n6"), ("n6", "n7"), ("n7", "n0"),
                ("n0", "n3"), ("n2", "n5"), ("n1", "n4"),
            ],
        )

        single = GraphInvariants(graph = graph).spectral()
        streamed = GraphInvariants(graph = graph).spectral(block_size = 2)

        for key in single:
            self.assertAlmostEqual(single[key], streamed[key], places = 12, msg = key)

    def test_cohesion_matches_diameter_and_eccentricity(self) -> None:
        graphs = [
            _create_igraph_object(  ## tree
                nodes = ["a", "b", "c", "d", "e"],
                edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e")],
            ),
            _create_igraph_object(  ## cycle
                nodes = [f"n{i}" for i in range(6)],
                edges = [(f"n{i}", f"n{(i + 1) % 6}") for i in range(6)],
            ),
            _create_igraph_object(  ## disconnected giant plus edge
                nodes = ["a", "b", "c", "d", "e", "x", "y"],
                edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e"), ("x", "y")],
            ),
        ]
        for graph in graphs:
            giant = graph.components().giant()
            cohesion = GraphInvariants(graph = graph).cohesion()
            self.assertEqual(cohesion["diameter"], float(giant.diameter(directed = False, unconn = False)))
            self.assertEqual(cohesion["radius"], float(min(giant.eccentricity())))


class WindmillEventTests(unittest.TestCase):

    def test_positive_production_below_mean_remains_on(self) -> None:
        production = np.array([[0.1], [0.2], [1.0], [0.2]])

        events = _process_events_wind(data = production, hours = 4)

        self.assertEqual(events["target"].tolist(), [0])


class SpecMarginalDeltaTests(unittest.TestCase):

    def test_spec_marginal_delta_floors_to_decimal(self) -> None:
        results = pd.DataFrame(
            {
                "reference": ["baseline", "baseline"],
                "ei_iqr": [0.0, 0.108803],
                "ci_iqr": [0.0, 0.256123],
            }
        )

        ei_margin = spec_marginal_delta(
            results = results,
            feat_value = ["ei_iqr"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "max",
            decimals = 2,
        )
        ci_margin = spec_marginal_delta(
            results = results,
            feat_value = ["ci_iqr"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "max",
            scale = 0.40,
            decimals = 2,
        )

        self.assertEqual(ei_margin, 0.10)
        self.assertEqual(ci_margin, 0.10)


class PerturbationStatisticsTests(unittest.TestCase):

    def test_realization_iterator_broadcasts_deterministic_datasets(self) -> None:
        data = pd.DataFrame({
            "dataset": ["explicit", "explicit", "analytical"],
            "realization": [0, 1, 0],
            "feature": [1.0, 2.0, 3.0],
        })

        realizations = list(_iter_perturbation_realizations(data))

        self.assertEqual([realization for realization, _ in realizations], [0, 1])
        self.assertEqual(
            [set(frame["dataset"]) for _, frame in realizations],
            [{"explicit", "analytical"}, {"explicit", "analytical"}],
        )
        self.assertEqual(
            [frame.set_index("dataset").loc["analytical", "feature"] for _, frame in realizations],
            [3.0, 3.0],
        )

    @patch("src.evaluators.perturbing.consensus_metrics")
    def test_recovery_compilation_averages_realizations(self, metrics_mock) -> None:
        metrics_mock.side_effect = lambda y_true, y_pred: {
            metric: float(np.mean(y_pred)) for metric in ("rho", "rbo", "dcr", "ci")
        }
        baseline = {
            "track": "frozen", "model": "model", "perturbation": "baseline",
            "method": None, "intensity": None, "realization": -1,
            "y_true": np.array([0.0, 1.0]), "y_pred": np.array([0.0, 1.0]),
            "groups": np.array(["g", "g"]),
        }
        perturbed = [
            {
                **baseline, "perturbation": "invariants", "method": "noise",
                "intensity": 0.1, "realization": realization,
                "y_pred": np.array([value, value]),
            }
            for realization, value in enumerate((0.2, 0.6))
        ]

        compiled = compile_perturbed_recovery({
            "baseline": {"model": baseline},
            "perturbed": perturbed,
        })
        row = compiled.query("perturbation == 'invariants'").iloc[0]

        self.assertEqual(row["n_realizations"], 2)
        self.assertAlmostEqual(row["ci"], 0.4)

    @patch("src.evaluators.perturbing.consensus_metrics")
    def test_consensus_compilation_averages_predictions_before_scoring(self, metrics_mock) -> None:
        metrics_mock.side_effect = lambda y_true, y_pred: {
            metric: float(np.mean(y_pred) ** 2) for metric in ("rho", "rbo", "dcr", "ci")
        }
        perturbed = list()
        for realization, value in enumerate((0.2, 0.6)):
            perturbed.extend([
                {
                    "model": "a", "pert_type": "invariants", "method": "noise",
                    "intensity": 0.1, "realization": realization,
                    "y_pred": np.array([0.0, 1.0]), "n_rows": 2,
                },
                {
                    "model": "b", "pert_type": "invariants", "method": "noise",
                    "intensity": 0.1, "realization": realization,
                    "y_pred": np.array([value, value]), "n_rows": 2,
                },
            ])

        compiled = compile_perturbed_consensus({
            "model_names": ["a", "b"],
            "baseline": {"a": np.array([0.0, 1.0]), "b": np.array([0.0, 1.0])},
            "perturbed": perturbed,
        })
        row = compiled.query("perturbation == 'invariants'").iloc[0]

        self.assertEqual(row["n_realizations"], 2)
        self.assertAlmostEqual(row["ci"], 0.16)
        self.assertEqual(metrics_mock.call_count, 2)
        np.testing.assert_allclose(
            actual = metrics_mock.call_args.kwargs["y_pred"],
            desired = [0.4, 0.4],
        )

    def test_transfer_compilation_averages_realizations_before_pairing(self) -> None:
        def frontier(ei: float) -> pd.DataFrame:
            return pd.DataFrame({
                "group": ["g"],
                "vr": [0.1],
                "mv": [0.2],
                "ms": [0.3],
                "ei": [ei],
            })

        results = {
            "frozen": {
                ("model", "baseline", None, None, -1): frontier(1.0),
                ("model", "invariants", "noise", 0.1, 0): frontier(0.6),
                ("model", "invariants", "noise", 0.1, 1): frontier(0.8),
            },
            "retrain": {
                ("model", "baseline", None, None, -1): frontier(1.0),
                ("model", "invariants", "noise", 0.1, 0): frontier(0.7),
                ("model", "invariants", "noise", 0.1, 1): frontier(0.9),
            },
        }

        compiled, _ = compile_perturbed_transfer(results = results)
        perturbed = compiled.query("perturbation == 'invariants'").sort_values("track")

        self.assertEqual(len(perturbed), 2)
        self.assertEqual(perturbed["n_realizations"].tolist(), [2, 2])
        self.assertEqual(perturbed["ei"].tolist(), [0.7, 0.8])

    @patch("src.evaluators.perturbing.wilcoxon")
    def test_tost_uses_full_precision_before_holm(self, wilcoxon_mock) -> None:
        wilcoxon_mock.side_effect = [
            (0.0, 0.0049), (0.0, 0.001),
            (0.0, 0.0251), (0.0, 0.001),
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

        self.assertEqual(summary["Holm-adj. p"].tolist(), ["0.010", "0.025"])

    def test_tost_rank_biserial_uses_raw_nonzero_differences(self) -> None:
        results = pd.DataFrame(
            {
                "model": ["a", "b", "c", "d"] * 2,
                "group": ["g"] * 8,
                "perturbation": ["baseline"] * 4 + ["invariants"] * 4,
                "method": [None] * 4 + ["noise"] * 4,
                "ei": [0.5, 0.5, 0.5, 0.5, 0.75, 0.375, 0.5, 0.625],
            }
        )

        summary = stat_perturbed_tost(
            results = results,
            feat_value = ["ei"],
            feat_group = ["method"],
            delta = 0.5,
            decimals = 2,
            index = False,
        )

        self.assertEqual(summary.loc[0, "Rank-biserial r"], "0.50")

    def test_find_perturbed_max_uses_method_severity(self) -> None:
        results = pd.DataFrame(
            {
                "perturbation": ["invariants"] * 8,
                "method": ["subset", "subset", "scaling", "scaling", "noise", "noise", "bootstrapping", "bootstrapping"],
                "intensity": [0.65, 0.95, 0.25, 1.75, 0.05, 0.35, 0.05, 0.35],
            }
        )

        strongest = find_perturbed_max(results = results)
        selected = strongest.set_index("method")["intensity"].to_dict()

        self.assertEqual(
            selected,
            {"subset": 0.65, "scaling": 0.25, "noise": 0.35, "bootstrapping": 0.05},
        )

    def test_one_row_noise_uses_supplied_corpus_scale(self) -> None:
        features = pd.DataFrame({"feature": [10.0]})
        scale = pd.Series({"feature": 2.0})

        perturbed = feature_perturb(
            X = features,
            method = "noise",
            noise = 0.5,
            random_state = 42,
            scale = scale,
        )
        expected = 10.0 + np.random.default_rng(42).normal(0, 1.0)

        self.assertAlmostEqual(perturbed.loc[0, "feature"], expected)

    def test_one_row_noise_rejects_missing_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "scale is required"):
            feature_perturb(
                X = pd.DataFrame({"feature": [10.0]}),
                method = "noise",
            )