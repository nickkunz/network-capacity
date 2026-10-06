import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.colors as mcolors
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
            "rho_rescaled": True,
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

    def test_load_results_cache_rejects_unscaled_rho(self) -> None:
        for scale in (None, False):
            with self.subTest(scale = scale):
                metadata = self._metadata()
                if scale is None:
                    metadata.pop("rho_rescaled")
                else:
                    metadata["rho_rescaled"] = scale
                self._write_consensus_cache(payload = {
                    "metadata": metadata,
                    "frontiers": {},
                    "results_data": pd.DataFrame(data = {"rho": [0.89]}),
                })
                with self.assertRaisesRegex(expected_exception = RuntimeError, expected_regex = "rho_rescaled"):
                    visualize.load_results_cache(name = "consensus", context = self.context)

    def test_flat_transfer_cache_requires_rescaled_rho(self) -> None:
        payload = {
            **self._metadata(),
            "results_dict_domain": {},
            "results_dict_5fold": {},
            "results_dict_10fold": {},
            "results_data_domain_consensus": pd.DataFrame(),
            "results_data_5fold": pd.DataFrame(),
            "results_data_10fold": pd.DataFrame(),
        }
        path = self.context.cache_dir / "transfer_results.pkl"
        arguments = {
            "data": self.context.data,
            "models": self.context.models,
            "feat_x": self.context.feat_x,
            "feat_z": self.context.feat_z,
            "target": self.context.target,
            "cache_path": path,
            "n_repeats": self.context.n_repeats,
            "random_state": self.context.random_state,
            "require_cache": True,
        }
        path.write_bytes(pickle.dumps(obj = payload))
        _, source = visualize.load_or_compute_transfer_consensus_results(**arguments)
        self.assertEqual(first = source, second = "cached consensus/transfer results")
        payload.pop("rho_rescaled")
        path.write_bytes(pickle.dumps(obj = payload))
        with self.assertRaisesRegex(expected_exception = FileNotFoundError, expected_regex = "compatible transfer cache"):
            visualize.load_or_compute_transfer_consensus_results(**arguments)

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


class DecompositionFigureTests(unittest.TestCase):

    def test_simple_specifications_use_native_labels(self) -> None:
        results = pd.DataFrame({
            "model": ["model"] * 5,
            "group": ["domain"] * 5,
            "specification": ["additive", "invariants", "signatures", "joint", "interaction"],
            "ei": [0.5, 0.4, 0.3, 0.45, 0.42],
            "ci": [0.5, 0.4, 0.3, 0.45, 0.42],
        })

        figure, axes = visualize.plot_decomposition_moneyshot(
            results = results,
            delta_ei = 0.1,
            delta_ci = 0.1,
            show = False,
        )
        try:
            self.assertTrue(expr = axes[0].collections)
            self.assertEqual(
                first = [label.get_text().split(":")[0] for label in axes[0].get_legend().get_texts()],
                second = ["Invariants", "Signatures"],
            )
        finally:
            visualize.plt.close(fig = figure)

    def test_stress_test_figure_decomposition_labels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cache_dir = root / "notebooks" / "cache"
        if not (cache_dir / "ablate_results.pkl").exists():
            self.skipTest(reason = "ablate_results.pkl cache not found")
        from src.data.builders import load_processed_data
        from src.estimators.factories import load_estimators
        from src.evaluators.config import FEAT_X, FEAT_Z, TARGET
        with tempfile.TemporaryDirectory() as temp_dir:
            context = visualize.VisualizationContext(
                data = load_processed_data(),
                models = load_estimators(random_state = 42),
                feat_x = tuple(FEAT_X),
                feat_z = tuple(FEAT_Z),
                target = TARGET,
                n_repeats = 30,
                random_state = 42,
                cache_dir = cache_dir,
                figure_dir = Path(temp_dir),
            )
            with patch.object(target = visualize, attribute = "export_pdf_scaled", return_value = Path(temp_dir) / "fig4.pdf"):
                figure, _ = visualize.generate_stress_test_figure(context = context, show = False)
                try:
                    simple_axis = next(ax for ax in figure.axes if ax.get_title() == "Simple")
                    legend = simple_axis.get_legend()
                    self.assertIsNotNone(obj = legend)
                    labels = [text.get_text() for text in legend.get_texts()]
                    self.assertEqual(
                        first = labels,
                        second = ["Invariants Only", "Signatures Only"],
                    )
                finally:
                    visualize.plt.close(fig = figure)


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

    def test_transfer_panel_renders_equivalence_band(self) -> None:
        disciplines = np.array(["positive", "negative", "mixed"])
        domain_prediction = np.array([2.0, 2.0, 2.0])
        random_5fold = np.array([1.0, 3.0, 1.0])
        random_10fold = np.array([1.5, 4.0, 4.0])
        fold_metrics = pd.DataFrame({
            "model": ["test_model", "test_model"],
            "ei": [0.74, 0.76],
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

        fig, ax = visualize.plt.subplots()
        try:
            visualize._draw_transfer_panel(
                axis = ax,
                transfer_summary = summary,
                group_ranges = [("test_domain", 0, 2)],
                ordered_disciplines = disciplines.tolist(),
                discipline_domain_map = {discipline: "test_domain" for discipline in disciplines},
                domain_palette = {"test_domain": "#123456"},
                panel_label = "c",
                transfer_ylim = (-0.2, 0.2),
            )
            self.assertEqual(first = len(ax.patches), second = 1)
            span_patch = ax.patches[0]
            self.assertAlmostEqual(first = span_patch.get_y(), second = summary["baseline_iqr"][0])
            self.assertAlmostEqual(first = span_patch.get_height(), second = summary["baseline_iqr"][1] - summary["baseline_iqr"][0])
            self.assertEqual(first = span_patch.get_zorder(), second = 0)
            facecolor_hex = mcolors.to_hex(c = span_patch.get_facecolor()[:3]).upper()
            self.assertEqual(first = facecolor_hex, second = "#E6F4EA")

            ## verify dashed guide lines at margin boundaries
            guide_lines = [
                line for line in ax.lines
                if line.get_linestyle() == "--"
                and any(
                    np.isclose(a = line.get_ydata()[0], b = bound)
                    for bound in summary["baseline_iqr"]
                )
            ]
            self.assertEqual(first = len(guide_lines), second = 2)
            for guide in guide_lines:
                self.assertEqual(first = mcolors.to_hex(c = guide.get_color()).upper(), second = "#8A8A8A")

            ## verify delta tick labels in grey italic with matching panel text size
            delta_texts = {text.get_text(): text for text in ax.texts if text.get_text() in {"+δ", "-δ"}}
            self.assertIn(member = "+δ", container = delta_texts)
            self.assertIn(member = "-δ", container = delta_texts)
            for text_artist in delta_texts.values():
                self.assertEqual(first = mcolors.to_hex(c = text_artist.get_color()).upper(), second = "#8A8A8A")
                self.assertEqual(first = text_artist.get_fontstyle(), second = "italic")
                self.assertEqual(first = text_artist.get_fontsize(), second = visualize.PANEL_TEXT_SIZE)

            ## verify panel lettering normalization preserves italic delta labels and text size
            visualize._apply_panel_lettering(fig = fig)
            for text_artist in delta_texts.values():
                self.assertEqual(first = text_artist.get_fontstyle(), second = "italic")
                self.assertEqual(first = text_artist.get_fontsize(), second = visualize.PANEL_TEXT_SIZE)
        finally:
            visualize.plt.close(fig = fig)


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
