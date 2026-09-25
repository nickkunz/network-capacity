import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.visualizers import visualize


class VisualizationCacheTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.context = visualize.VisualizationContext(
            data = pd.DataFrame({"target": [1.0, 2.0]}),
            models = {"model_b": object(), "model_a": object()},
            feat_x = ("x_1",),
            feat_z = ("z_1",),
            target = "target",
            n_repeats = 30,
            random_state = 42,
            cache_dir = root / "cache",
            figure_dir = root / "figures",
            submission_dir = root / "submission",
        )
        self.context.cache_dir.mkdir(parents = True, exist_ok = True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _metadata(self) -> dict[str, object]:
        return {
            "n_obs": 2,
            "n_repeats": 30,
            "random_state": 42,
            "model_names": ["model_a", "model_b"],
            "target": "target",
            "feat_x": ["x_1"],
            "feat_z": ["z_1"],
        }

    def _write_consensus_cache(self, payload: dict[str, object]) -> Path:
        path = self.context.cache_dir / "consensus_results.pkl"
        path.write_bytes(pickle.dumps(payload))
        return path

    def test_load_results_cache_accepts_current_producer_contract(self) -> None:
        payload = {
            "metadata": self._metadata(),
            "frontiers": {"model_a": [1.0, 2.0]},
            "results_data": pd.DataFrame({"ci": [0.9]}),
        }
        self._write_consensus_cache(payload = payload)

        actual = visualize.load_results_cache(
            name = "consensus",
            context = self.context,
        )

        self.assertEqual(actual["metadata"], payload["metadata"])
        pd.testing.assert_frame_equal(actual["results_data"], payload["results_data"])

    def test_load_results_cache_reports_missing_producer(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "consensus.ipynb"):
            visualize.load_results_cache(
                name = "consensus",
                context = self.context,
            )

    def test_load_results_cache_rejects_metadata_mismatch(self) -> None:
        metadata = self._metadata()
        metadata["random_state"] = 7
        self._write_consensus_cache(payload = {
            "metadata": metadata,
            "frontiers": {},
            "results_data": pd.DataFrame(),
        })

        with self.assertRaisesRegex(RuntimeError, "random_state=7"):
            visualize.load_results_cache(
                name = "consensus",
                context = self.context,
            )

    def test_load_results_cache_rejects_missing_payload_key(self) -> None:
        self._write_consensus_cache(payload = {
            "metadata": self._metadata(),
            "results_data": pd.DataFrame(),
        })

        with self.assertRaisesRegex(RuntimeError, "frontiers"):
            visualize.load_results_cache(
                name = "consensus",
                context = self.context,
            )

    def test_load_results_cache_rejects_unknown_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown results cache"):
            visualize.load_results_cache(
                name = "unknown",
                context = self.context,
            )

    def test_consensus_figure_rejects_legacy_caches(self) -> None:
        with patch.object(target = visualize, attribute = "load_results_cache", return_value = {}):
            with self.assertRaisesRegex(expected_exception = RuntimeError, expected_regex = "full-corpus.*perturb.ipynb"):
                visualize.generate_consensus_figure(context = self.context, show = False)


class VisualizationInterfaceTests(unittest.TestCase):

    def test_exposes_one_generator_per_figure(self) -> None:
        generators = (
            visualize.generate_conceptual_figure,
            visualize.generate_universality_figure,
            visualize.generate_consensus_figure,
            visualize.generate_stress_test_figure,
        )

        self.assertTrue(all(callable(generator) for generator in generators))


class TransferFigureTests(unittest.TestCase):

    def test_transfer_differences_use_random_minus_domain_logo(self) -> None:
        disciplines = np.array(["positive", "negative", "mixed"])
        domain_prediction = np.array([2.0, 2.0, 2.0])
        random_5fold = np.array([1.0, 3.0, 1.0])
        random_10fold = np.array([1.5, 4.0, 4.0])
        fold_metrics = pd.DataFrame({
            "model": ["test_model", "test_model"],
            "ei": [0.7, 0.9],
        })

        summary = visualize._summarize_transfer_panel(
            y_true_log = np.ones(shape = 3),
            disciplines = disciplines,
            ordered_disciplines = disciplines.tolist(),
            results_dict_domain = {"test_model": domain_prediction},
            results_dict_5fold = {"test_model": random_5fold},
            results_dict_10fold = {"test_model": random_10fold},
            results_data_5fold = fold_metrics,
            results_data_10fold = fold_metrics,
            equivalence_fallback = 0.02,
            feasibility_threshold = 0.0,
        )

        expected = np.stack(arrays = [
            np.cbrt(1.0 / random_5fold),
            np.cbrt(1.0 / random_10fold),
        ]) - np.cbrt(1.0 / domain_prediction)
        for column, quantile in (("q1", 0.25), ("median", 0.5), ("q3", 0.75)):
            with self.subTest(column = column):
                np.testing.assert_allclose(
                    actual = summary["delta_summary"][column],
                    desired = np.quantile(a = expected, q = quantile, axis = 0),
                )
        np.testing.assert_allclose(
            actual = [summary["overall_delta_5fold"], summary["overall_delta_10fold"]],
            desired = np.median(a = expected, axis = 1),
        )
        self.assertEqual(first = summary["n_feasible"], second = 6)
        self.assertEqual(first = summary["n_total"], second = 6)


class ConsensusFigureTests(unittest.TestCase):

    def test_all_four_conditions_use_full_corpus_results(self) -> None:
        pairs = pd.DataFrame({
            "model_i": ["linear_quantile"],
            "model_j": ["forest_quantile"],
            "ci": [0.94],
        })
        decomposed = pd.concat(
            objs = [
                pairs.assign(specification = "additive", ci = 0.61),
                pairs.assign(specification = "joint", ci = 0.52),
            ],
            ignore_index = True,
        ).assign(evaluation = "full_corpus")
        separation = pd.DataFrame({
            "model": ["linear_quantile", "linear_quantile"],
            "specification": ["additive", "joint"],
            "ci": [0.80, 0.55],
        }).assign(evaluation = "full_corpus")
        recovery = pd.DataFrame({
            "model": ["linear_quantile"],
            "track": ["frozen"],
            "perturbation": ["invariants"],
            "method": ["noise"],
            "intensity": [0.35],
            "ci": [0.75],
        }).assign(evaluation = "full_corpus")
        agreement = pd.DataFrame({
            "model": ["linear_quantile"],
            "Falsification": ["frozen"],
            "condition": ["falsified"],
            "ci": [0.40],
        }).assign(evaluation = "full_corpus")

        with tempfile.TemporaryDirectory() as directory, visualize.mpl.rc_context(), patch.object(
            target = visualize.Figure, attribute = "savefig",
        ):
            figure, _ = visualize._render_consensus_figure(
                results_data = pairs,
                results_perturbed_consensus = pairs.assign(
                    perturbation = "invariants", method = "noise", intensity = 0.35, ci = 0.85,
                ),
                results_decomposed_consensus = decomposed,
                results_falsified_consensus = pairs.assign(
                    condition = "falsified", Falsification = "frozen", ci = 0.45,
                ),
                results_original_agreement = pd.DataFrame(data = {
                    "model": ["linear_quantile"], "ci": [0.98], "evaluation": ["full_corpus"],
                }),
                results_decomposed_full_agreement = separation,
                results_perturbed_full_agreement = recovery,
                results_falsified_full_agreement = agreement,
                figure_dir = Path(directory),
                export_pdf_scaled = Mock(return_value = Path(directory) / "3.pdf"),
                n_decimals = 2,
                show = False,
            )
            try:
                heatmaps = {axis.get_title(): axis for axis in figure.axes if axis.images}
                expected = {
                    "1)  Original": 0.94,
                    "2)  Perturbed": 0.85,
                    "3)  Falsified": 0.45,
                    "4)  Ablated": 0.52,
                }
                self.assertEqual(set(heatmaps), set(expected))
                self.assertNotIn(
                    member = "LOGO",
                    container = "\n".join(
                        artist.get_text() for artist in figure.findobj(match = visualize.Text)
                    ),
                )
                for title, value in expected.items():
                    np.testing.assert_allclose(
                        actual = heatmaps[title].images[0].get_array().compressed(),
                        desired = value,
                    )
                np.testing.assert_allclose(
                    actual = figure.axes[0].collections[0].get_offsets(),
                    desired = [[0.98, 0.94]],
                )
                np.testing.assert_allclose(
                    actual = figure.axes[0].collections[3].get_offsets(),
                    desired = [[0.55, 0.52]],
                )
            finally:
                visualize.plt.close(fig = figure)


if __name__ == "__main__":
    unittest.main()
