import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib._mathtext as _mathtext
import matplotlib.pyplot as plt

from src.evaluators import operating
from src.evaluators.operating import (
    FEAT_6U,
    GLM_FAMILIES,
    SCT_MODES,
    RESAMPLING_REGIMES,
    _utilization_style,
    compile_utilization_frontier,
    compile_utilization_metrics,
    compile_utilization_domain_cells,
    format_utilization_table,
    load_utilization_windows,
    plot_utilization_frontier,
    stat_utilization_comparisons,
    stat_utilization_superiority,
    stat_utilization_superiority_holm,
    train_utilization,
)
from src.vectorizers.signatures import ProcessSignatures


class UtilizationTests(unittest.TestCase):
    def test_superiority_orients_ci_and_handles_zero_differences(self) -> None:
        anchor = np.linspace(start = 0.5, stop = 0.8, num = 15)
        for metric, comparator in (("mae", anchor + 0.1), ("ci", anchor - 0.1)):
            result = stat_utilization_superiority(
                metric = metric,
                comparator_cells = comparator,
                anchor_cells = anchor,
            )
            self.assertEqual(first = result["Pairs"], second = 15)
            self.assertEqual(first = result["Pairs Favoring Anchor"], second = 15)
            self.assertAlmostEqual(first = result["Median Δ"], second = 0.1)
            self.assertEqual(first = result["Rank-biserial r"], second = 1.0)
            self.assertLess(a = result["One-sided p"], b = 0.05)
            self.assertFalse(expr = any("margin" in key.lower() for key in result))
        zeros = stat_utilization_superiority(
            metric = "mae",
            comparator_cells = anchor + 1e-12,
            anchor_cells = anchor,
        )
        self.assertEqual(first = zeros["One-sided p"], second = 1.0)
        self.assertTrue(expr = np.isnan(zeros["Rank-biserial r"]))

    def test_domain_cells_score_remaps_before_averaging(self) -> None:
        observed = np.linspace(start = 1.0, stop = 5.0, num = 25)
        cells = compile_utilization_domain_cells(
            y_true = observed,
            predictions = {
                family: [observed - 0.5, observed + 0.5] for family in GLM_FAMILIES
            },
            domains = np.repeat(a = np.arange(5), repeats = 5),
        )
        self.assertEqual(first = len(cells["mae"]), second = 15)
        np.testing.assert_allclose(actual = cells["mae"], desired = 0.5)

    def test_superiority_holm_separates_metrics_and_masks_primary_rows(self) -> None:
        frame = pd.DataFrame(
            data = {
                "Metric": ["MAE", "MAE", "CI", "CI"],
                "One-sided p": [0.01, 0.04, 0.02, 0.03],
                "Median Δ": [0.1] * 4,
            }
        )
        complete = stat_utilization_superiority_holm(
            frame = frame,
            family_mask = pd.Series(data = True, index = frame.index),
            suffix = "Complete",
        )
        np.testing.assert_allclose(
            actual = complete["Holm-adj. p (Complete)"],
            desired = [0.02, 0.04, 0.04, 0.04],
        )
        self.assertTrue(expr = complete["Superior (Complete)"].all())
        self.assertFalse(expr = any("margin" in column.lower() for column in complete))
        primary = stat_utilization_superiority_holm(
            frame = frame,
            family_mask = pd.Series(data = [True, False, True, False]),
            suffix = "Primary",
        )
        self.assertTrue(expr = primary.loc[[1, 3], "Sp."].isna().all())

    def test_complete_superiority_has_no_margin_outputs(self) -> None:
        observed = np.linspace(start = 1.0, stop = 5.0, num = 25)
        data = pd.DataFrame(
            data = {"y_mean": observed, "domain": np.repeat(a = np.arange(5), repeats = 5)}
        )
        frames = {fraction: data.copy() for fraction in (0.25, 0.5, 0.75)}
        controls = {
            (regime, family, configuration): [
                observed + (0.1 if configuration == SCT_MODES[1] else 0.5)
            ]
            for regime in RESAMPLING_REGIMES
            for family in GLM_FAMILIES
            for configuration in operating.UTILIZATION_CONTROLS
        }
        windows = {
            (fraction, regime, True): {
                (family, configuration): {
                    "y_pred": controls[(regime, family, configuration)][0]
                }
                for family in GLM_FAMILIES
                for configuration in SCT_MODES
            }
            for fraction in frames
            for regime in RESAMPLING_REGIMES
        }
        result = operating.compile_utilization_superiority(
            data = data,
            controls = controls,
            frames = frames,
            windows = windows,
            tables = {
                name: pd.DataFrame()
                for name in ("ablation", "ablation_contrasts", "temporal_contrasts", "window_contrasts")
            },
        )
        self.assertEqual(first = len(result["superiority"]), second = 160)
        self.assertEqual(first = len(result["support"]), second = 32)
        for name in ("superiority", "ablation_tests", "temporal_tests", "support"):
            self.assertFalse(expr = any("margin" in column.lower() for column in result[name]))
        for column in ("Sp.", "Superior (Complete)"):
            self.assertIn(member = column, container = result["superiority"])

    def test_table_formats_intervals_and_tests_without_mutating_results(self) -> None:
        frame = pd.DataFrame(
            data = {
                "Configuration": ["Anchor", "Alternative"],
                "MAE": [1.234, 2.0],
                "MAE_lo": [1.0, 1.5],
                "MAE_hi": [1.5, 2.5],
                "Holm-adj. p": [np.nan, 1e-8],
                "Sp.": [False, True],
            }
        )
        original = frame.copy(deep = True)
        result = format_utilization_table(
            frame = frame,
            title = "Controls & uncertainty",
            columns = ["Configuration", "MAE [95% CI]", "Holm-adj. p", "Sp."],
            groups = ["Configuration"],
            note = "Positive differences favor the anchor.",
        ).data
        self.assertIn(member = "1.23 [1.00, 1.50]", container = result)
        self.assertIn(member = "MAE [95% CI]", container = result)
        self.assertIn(member = "&lt;0.001", container = result)
        self.assertNotIn(member = "1.00e-08", container = result)
        self.assertIn(member = ">Holm-adj. p</th>", container = result)
        self.assertIn(member = ">Sig.</th>", container = result)
        self.assertIn(member = ">***<", container = result)
        self.assertIn(member = ">Sp.</th>", container = result)
        self.assertNotIn(member = "Superior (primary)", container = result)
        self.assertIn(member = "Controls &amp; uncertainty", container = result)
        self.assertNotIn(member = "MAE_lo", container = result)
        self.assertNotIn(member = ">nan<", container = result)
        pd.testing.assert_frame_equal(left = frame, right = original)

    def test_table_title_case_and_metric_order(self) -> None:
        frame = pd.DataFrame(data = {
            "Estimator": ["Poisson"], "MAE": [0.2], "RMSE": [0.3],
            "GMME": [1.2], "CI": [0.9], "Within 2x (%)": [80.0],
            "Within 3x (%)": [90.0], "Observed IQR": [0.5],
            "Median Δ": [0.1], "Rank-biserial r": [0.8],
            "One-sided p": [0.01], "Holm-adj. p": [0.02],
            "TOST p": [0.01],
        })
        original = frame.copy(deep = True)
        html = format_utilization_table(
            frame = frame, title = "Metrics", groups = ["Estimator"]
        ).data
        for label in (
            "Estimator",
            "Observed IQR",
            "Within 2x (%)",
            "Within 3x (%)",
            "Median Δ",
            "Rank-biserial r",
            "One-sided p",
            "Holm-adj. p",
            "TOST p",
        ):
            self.assertIn(member = f">{label}</th>", container = html)
        positions = [html.index(f">{metric}</th>") for metric in ("CI", "MAE", "RMSE", "GMME")]
        self.assertEqual(first = positions, second = sorted(positions))
        pd.testing.assert_frame_equal(left = frame, right = original)
        wide = pd.DataFrame(
            data = [[0.2, 0.3, 0.9, 1.2]],
            columns = pd.MultiIndex.from_product(iterables = [["MAE", "RMSE", "CI", "GMME"], ["Domain LOGO-CV"]]),
        )
        html = format_utilization_table(frame = wide, title = "Regimes").data
        positions = [html.index(f">{metric} / Domain LOGO-CV</th>") for metric in ("CI", "MAE", "RMSE", "GMME")]
        self.assertEqual(first = positions, second = sorted(positions))
        grouped = pd.DataFrame(data = {
            "Metric": ["MAE", "GMME", "RMSE", "CI"], "Holm-adj. p": [0.01] * 4,
        })
        original = grouped.copy(deep = True)
        html = format_utilization_table(frame = grouped, title = "Tests", groups = ["Metric"]).data
        positions = [html.index(f">{metric}</th>") for metric in ("CI", "MAE", "RMSE", "GMME")]
        self.assertEqual(first = positions, second = sorted(positions))
        pd.testing.assert_frame_equal(left = grouped, right = original)

    def test_complete_table_is_expandable_and_retains_all_rows(self) -> None:
        frame = pd.DataFrame(
            data = {
                "System": [f"system-{index:03d}" for index in range(80)],
                "MAE": np.arange(80, dtype = float),
                "Equivalent": pd.array([True, False] * 40, dtype = "boolean"),
            }
        )
        result = format_utilization_table(
            frame = frame,
            title = "Complete records",
            groups = ["System"],
            details = True,
        ).data
        self.assertIn(
            member = "<details><summary>Complete records (80 rows)", container = result
        )
        for index in range(80):
            self.assertIn(member = f"system-{index:03d}", container = result)
        self.assertIn(member = ">Yes<", container = result)
        self.assertIn(member = ">No<", container = result)

    def test_publication_style_restores_matplotlib_settings(self) -> None:
        original_shrink = _mathtext.SHRINK_FACTOR
        with mpl.rc_context(rc = {"font.family": "serif", "axes.linewidth": 2.0}):
            with self.assertRaises(expected_exception = RuntimeError):
                with _utilization_style():
                    self.assertEqual(
                        first = mpl.rcParams["font.family"], second = ["Arial"]
                    )
                    self.assertEqual(first = mpl.rcParams["axes.linewidth"], second = 0.5)
                    self.assertEqual(first = _mathtext.SHRINK_FACTOR, second = 0.73)
                    raise RuntimeError("render interrupted")
            self.assertEqual(first = mpl.rcParams["font.family"], second = ["serif"])
            self.assertEqual(first = mpl.rcParams["axes.linewidth"], second = 2.0)
        self.assertEqual(first = _mathtext.SHRINK_FACTOR, second = original_shrink)

    def test_frontier_preserves_original_geometry_and_png_preview(self) -> None:
        data = pd.DataFrame(
            data = {
                "discipline": ["Physics", "Biology", "Commerce"],
                "y_mean": [1.0, 3.0, 5.0],
                "evt_atten": [0.5, 1.0, 1.5],
            }
        )
        predictions = {
            (GLM_FAMILIES[-1], mode): {"y_pred": data["y_mean"].to_numpy() + 0.2}
            for mode in SCT_MODES
        }
        resampling = {
            "folds": {
                RESAMPLING_REGIMES[0]: [
                    (
                        np.array(object = [], dtype = int),
                        np.arange(3),
                        None,
                        np.array(object = [6.0, 8.0, 10.0]),
                        [],
                    )
                ]
            },
            "predictions": {RESAMPLING_REGIMES[0]: predictions},
        }
        open_figures = plt.get_fignums()
        with patch.object(
            target = operating,
            attribute = "_render_utilization_figure",
            wraps = operating._render_utilization_figure,
        ) as render:
            image = plot_utilization_frontier(data = data, resampling = resampling)
        figure = render.call_args.kwargs["figure"]
        np.testing.assert_allclose(
            actual = figure.get_size_inches(), desired = [183.0 / 25.4, 3.10]
        )
        np.testing.assert_array_equal(
            actual = figure.axes[0].get_yticks(), desired = [0, 5, 10, 15, 20]
        )
        self.assertEqual(first = image.width, second = 720)
        pixels = plt.imread(fname = BytesIO(image.data), format = "png")
        self.assertGreater(a = pixels.shape[1], b = 4000)
        self.assertGreater(a = float(pixels.std()), b = 0.01)
        self.assertEqual(first = plt.get_fignums(), second = open_figures)

    def test_frontier_requires_exactly_one_prediction_per_system(self) -> None:
        folds = [
            (
                np.array(object = [1]),
                np.array(object = [0]),
                None,
                np.array(object = [2.0]),
                [],
            ),
            (
                np.array(object = [0]),
                np.array(object = [1]),
                None,
                np.array(object = [3.0]),
                [],
            ),
        ]
        np.testing.assert_array_equal(
            actual = compile_utilization_frontier(folds = folds, n_obs = 2),
            desired = [2.0, 3.0],
        )
        for invalid in (folds[:1], folds + folds[:1]):
            with self.assertRaises(expected_exception = ValueError):
                compile_utilization_frontier(folds = invalid, n_obs = 2)

    def test_rate_metrics_use_log1p_and_original_rate_ratios(self) -> None:
        observed = np.log1p(np.array(object = [1.0, 4.0, 10.0]))
        predicted = np.log1p(np.array(object = [2.0, 8.0, 20.0]))
        result = compile_utilization_metrics(y_true = observed, y_pred = predicted)
        self.assertAlmostEqual(first = result["gmme"], second = 2.0)
        self.assertAlmostEqual(
            first = result["mae"], second = float(np.mean(a = np.abs(observed - predicted)))
        )
        self.assertAlmostEqual(
            first = result["rmse"],
            second = float(np.sqrt(np.mean(a = (observed - predicted) ** 2))),
        )

    def test_temporal_inputs_use_early_intervals_only(self) -> None:
        counts = np.arange(1, 13, dtype = float)
        events = pd.DataFrame(data = {"day": np.arange(12), "target": counts})
        signatures = ProcessSignatures(
            data = events, sort_by = ["day"], target = "target"
        ).all()
        data = pd.DataFrame(
            data = [
                {
                    "name": "sample",
                    "domain": "domain",
                    "discipline": "discipline",
                    "target": counts.max(),
                    "r_max": counts.max(),
                    "r_mean": counts.mean(),
                    "utilization": counts.mean() / counts.max(),
                    "n_edges": 9,
                    **signatures,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text(
                data = json.dumps(
                    obj = {"events": events.iloc[::-1].to_dict(orient = "records")}
                ),
                encoding = "utf-8",
            )
            frame = load_utilization_windows(
                data = data, fractions = (0.50,), proc_dir = directory
            )[0.50]
        early = ProcessSignatures(
            data = events.iloc[:6], sort_by = ["day"], target = "target"
        ).all()
        for feature, value in early.items():
            self.assertAlmostEqual(first = frame.loc[0, feature], second = value)
        self.assertEqual(first = frame.loc[0, "n_edges"], second = 9)
        self.assertAlmostEqual(first = frame.loc[0, "y_max"], second = np.log1p(6.0))
        self.assertEqual(first = frame.loc[0, "M"], second = 57.0)
        self.assertEqual(first = frame.loc[0, "r_mean"], second = 9.5)
        self.assertNotIn(member = "target", container = frame.columns)

    def test_paired_bootstrap_averages_remap_metrics_after_scoring(self) -> None:
        observed = np.log1p(np.arange(1, 13, dtype = float))
        anchor = observed + 0.1
        remaps = [observed + 0.6, observed - 0.6]
        indices = np.random.default_rng(seed = 42).integers(low = 0, high = 12, size = (50, 12))
        summaries, contrasts = stat_utilization_comparisons(
            y_true = observed,
            predictions = {SCT_MODES[1]: [anchor], "SCT Frontier Remap": remaps},
            indices = indices,
            n_boot_ci = 10,
        )
        remap = summaries.set_index(keys = "Configuration").loc["SCT Frontier Remap"]
        self.assertAlmostEqual(first = remap["MAE"], second = 0.6)
        self.assertAlmostEqual(first = remap["MAE_lo"], second = 0.6)
        ratios = np.maximum(np.expm1(np.asarray(remaps)), 1e-9) / np.expm1(observed)
        expected = 100 * np.mean(np.abs(np.log(ratios)) <= np.log(3.0))
        self.assertAlmostEqual(first = remap["Within 3x (%)"], second = expected)
        self.assertGreater(a = remap["Within 3x (%)"], b = remap["Within 2x (%)"])
        self.assertAlmostEqual(
            first = contrasts.set_index(keys = "Metric").loc["MAE", "Delta"], second = 0.5
        )
        self.assertNotIn(member = "Wilcoxon_p", container = contrasts.columns)
        self.assertEqual(
            first = int(contrasts["Systems With Lower Anchor Error"].notna().sum()), second = 1
        )
        self.assertTrue(
            expr = np.isfinite(contrasts[["Delta_CI_Lower", "Delta_CI_Upper"]])
            .all()
            .all()
        )

    @patch(target = "src.evaluators.operating.fit_parametric_glm_family")
    @patch(target = "src.evaluators.operating.minimize_scalar")
    @patch(target = "src.evaluators.operating.stats.poisson.ppf")
    @patch(target = "src.evaluators.operating.stats.nbinom.ppf")
    def test_benchmark_caps_anchor_and_retains_actual_fold_fits(
        self, nbinom_mock: MagicMock, poisson_mock: MagicMock,
        scalar_mock: MagicMock, fit_mock: MagicMock
        ) -> None:
        fit_mock.return_value = (4.0, np.zeros(shape = len(FEAT_6U)))
        data = pd.DataFrame(
            data = {
                **{feature: np.arange(1, 7, dtype = float) for feature in FEAT_6U},
                "M": np.arange(1, 7, dtype = float) * 10,
                "L": np.full(shape = 6, fill_value = 10.0),
                "r_mean": np.arange(1, 7, dtype = float),
                "y_mean": np.log1p(np.arange(1, 7, dtype = float)),
                "evt_atten": np.full(shape = 6, fill_value = 0.2),
            }
        )
        folds = list()
        for test_idx in (np.array(object = [0, 1, 2]), np.array(object = [3, 4, 5])):
            train_idx = np.setdiff1d(ar1 = np.arange(6), ar2 = test_idx)
            inner = [
                (
                    np.array(object = [0, 1]),
                    np.array(object = [2]),
                    np.ones(shape = 2),
                    np.ones(shape = 1),
                )
            ]
            folds.append(
                (train_idx, test_idx, np.ones(shape = 3), np.ones(shape = 3), inner)
            )
        summary, predictions = train_utilization(
            df_data = data, folds = folds
        )
        _, uncapped = train_utilization(
            df_data = data, folds = folds, families = GLM_FAMILIES[:1], cap_frontier = False
        )
        anchor = predictions[(GLM_FAMILIES[0], SCT_MODES[1])]
        np.testing.assert_allclose(actual = anchor["r_pred"], desired = np.exp(1.0))
        self.assertEqual(first = len(summary), second = 6)
        self.assertNotIn(member = "Mean_LogLik", container = summary.columns)
        self.assertNotIn(member = "Cov_90_%", container = summary.columns)
        for prediction in predictions.values():
            self.assertEqual(
                first = set(prediction), second = {"r_pred", "y_pred", "fold_fits"}
            )
        nbinom_mock.assert_not_called()
        poisson_mock.assert_not_called()
        scalar_mock.assert_not_called()
        self.assertEqual(first = len(anchor["fold_fits"]), second = 2)
        self.assertTrue(
            expr = np.all(
                uncapped[(GLM_FAMILIES[0], SCT_MODES[1])]["r_pred"] > anchor["r_pred"]
            )
        )


if __name__ == "__main__":
    unittest.main()
