import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig
import numpy as np
import pandas as pd

from src.data.loaders.bitcoin import build_network_bitcoin
from src.data.loaders.epilepsy import _load_events_epilepsy
from src.data.loaders.faers import _iter_reports_faers, _process_events_faers
from src.data.loaders.federal import FederalProcessor
from src.data.loaders.metrla import _process_events_metrla
from src.data.loaders.pemsbay import _process_events_pemsbay
from src.data.builders import load_perturbed_data
from src.data.perturbers import _execute_perturbations, _jitter_count_series


class TrafficEventTests(unittest.TestCase):

    def test_raw_speed_congestion_onsets_are_counted(self) -> None:
        speed = np.array([[5.0, 5.0], [1.0, 5.0], [5.0, 1.0], [1.0, 5.0]])

        for processor in (_process_events_metrla, _process_events_pemsbay):
            events = processor(
                data = speed,
                sample_rate_minutes = 720,
                thres_percentile = 25,
            )

            self.assertEqual(events["target"].tolist(), [1, 2])


class BitcoinNetworkTests(unittest.TestCase):

    def test_network_uses_every_history_row(self) -> None:
        history = pd.DataFrame(
            {
                "src": [1, 3],
                "dst": [2, 4],
                "rating": [-1, 10],
            }
        )

        nodes, edges = build_network_bitcoin(data = history)

        self.assertEqual(set(nodes), {"user_1", "user_2", "user_3", "user_4", "rating_-1", "rating_10"})
        self.assertEqual(len(edges), 2 * len(history))


class EpilepsyEventTests(unittest.TestCase):

    @patch("src.data.loaders.epilepsy._request_with_retry")
    def test_relative_days_use_clock_rollover_and_file_fallback(self, request_mock) -> None:
        request_mock.side_effect = [
            SimpleNamespace(
                text=(
                    "File Name: chb01_01.edf\n"
                    "File Start Time: 23:50:00\n"
                    "Seizure 1 Start Time: 120 seconds\n"
                    "File Name: chb01_02.edf\n"
                    "File Start Time: 00:10:00\n"
                    "Seizure 1 Start Time: 60 seconds\n"
                )
            ),
            SimpleNamespace(
                text=(
                    "File Name: chb24_01.edf\n"
                    "Seizure Start Time: 480 seconds\n"
                    "File Name: chb24_25.edf\n"
                    "Seizure Start Time: 60 seconds\n"
                )
            ),
        ]

        clocked = _load_events_epilepsy(url = "unused", ids = "chb01")
        fallback = _load_events_epilepsy(url = "unused", ids = "chb24")

        self.assertEqual(clocked["day"].tolist(), [0, 1])
        self.assertEqual(fallback["day"].tolist(), [0, 1])


class FaersEventTests(unittest.TestCase):

    def test_events_are_unique_case_receipts(self) -> None:
        reports = pd.DataFrame(
            {
                "report_id": ["A", "A", "B"],
                "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            }
        )

        events = _process_events_faers(data = reports)

        self.assertEqual(events["target"].tolist(), [1, 1])

    @patch("src.data.loaders.faers._request_with_retry")
    def test_incomplete_pagination_is_rejected(self, request_mock) -> None:
        request_mock.return_value = SimpleNamespace(
            json=lambda: {
                "meta": {"results": {"total": 2}},
                "results": [{"safetyreportid": "A"}],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "incomplete FAERS pagination"):
            list(_iter_reports_faers(id = "IMATINIB", url = "unused"))


class FederalEventTests(unittest.TestCase):

    def test_start_dates_are_limited_to_configured_window(self) -> None:
        processor = FederalProcessor(
            url = "unused",
            start_date = "2011-01-01",
            end_date = "2024-12-31",
            keyword = "waterfowl",
        )
        processor.data_raw = pd.DataFrame(
            {
                "Award Amount": [1, 1, 1],
                "Start Date": ["2006-11-01", "2011-09-23", "2025-01-01"],
                "End Date": ["2007-01-01", "2012-01-01", "2025-02-01"],
                "Recipient Name": ["A", "B", "C"],
                "Awarding Agency": ["X", "Y", "Z"],
                "recipient_id": ["1", "2", "3"],
            }
        )

        processor.process_data()

        self.assertEqual(
            processor.data_processed["Start Date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2011-09-23"],
        )


class PerturbationDispatchTests(unittest.TestCase):

    def test_chunked_temporal_jitter_matches_vectorized_draws(self) -> None:
        positions = np.array([0, 2])
        counts = np.array([3, 2])
        seed = 42
        expanded = np.repeat(positions, counts)
        expected_positions = np.rint(
            expanded + np.random.default_rng(seed).normal(0, 0.5, size = len(expanded))
        ).astype(int).clip(0, 2)
        expected = np.bincount(expected_positions, minlength = 3)

        actual = _jitter_count_series(
            positions = positions,
            counts = counts,
            sigma = 0.5,
            lower = 0,
            upper = 2,
            rng = np.random.default_rng(seed),
        )

        np.testing.assert_array_equal(actual, expected)

    def test_chunked_temporal_jitter_conserves_large_counts(self) -> None:
        actual = _jitter_count_series(
            positions = np.array([0, 1]),
            counts = np.array([1_000_000, 2_000_000]),
            sigma = 0.5,
            lower = 0,
            upper = 1,
            rng = np.random.default_rng(42),
            chunk_size = 10_000,
        )

        self.assertEqual(int(actual.sum()), 3_000_000)
        self.assertEqual(len(actual), 2)
        expected_lower = 1_000_000 * 0.841344746 + 2_000_000 * 0.158655254
        self.assertLess(abs(float(actual[0]) - expected_lower), 5_000)

    @patch("src.data.perturbers.GraphInvariants")
    @patch("src.data.perturbers.network_perturb")
    @patch("src.data.perturbers.analytical_perturb")
    def test_large_non_bipartite_graph_uses_explicit_network_perturbation(
        self,
        analytical_perturb_mock,
        network_perturb_mock,
        graph_invariants_mock,
    ) -> None:
        baseline = {"n_nodes": 1_001, "n_edges": 1_001}
        graph_invariants_mock.return_value.all.return_value = baseline
        network_perturb_mock.return_value = baseline
        processor = SimpleNamespace(
            graph = ig.Graph.Ring(1_001),
            invariants = baseline,
            dimensions = None,
            events = None,
        )

        with (
            patch.dict("src.data.perturbers.NETWORK_METHODS", {"rewire": (0.1,)}, clear = True),
            patch.dict("src.data.perturbers.INVARIANT_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.PROCESS_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.SIGNATURE_METHODS", {}, clear = True),
        ):
            result = _execute_perturbations(
                proc = processor,
                name = "test",
                n_realizations = 2,
            )

        analytical_perturb_mock.assert_not_called()
        self.assertEqual(network_perturb_mock.call_count, 2)
        self.assertEqual(len(result["network_perturbed"]["rewire"]), 2)

    @patch("src.data.perturbers.network_perturb")
    @patch("src.data.perturbers.GraphInvariants")
    def test_explicit_network_perturbations_repeat_with_distinct_seeds(
        self,
        graph_invariants_mock,
        network_perturb_mock,
    ) -> None:
        graph_invariants_mock.return_value.all.return_value = {"n_nodes": 5, "n_edges": 5}
        network_perturb_mock.side_effect = lambda *args, random_state, **kwargs: {"seed": random_state}
        processor = SimpleNamespace(
            graph = ig.Graph.Ring(5),
            invariants = None,
            dimensions = None,
            events = None,
        )

        with (
            patch.dict("src.data.perturbers.NETWORK_METHODS", {"rewire": (0.1,)}, clear = True),
            patch.dict("src.data.perturbers.INVARIANT_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.PROCESS_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.SIGNATURE_METHODS", {}, clear = True),
        ):
            result = _execute_perturbations(
                proc = processor,
                name = "test",
                n_realizations = 3,
            )

        records = result["network_perturbed"]["rewire"]
        self.assertEqual([record["realization"] for record in records], [0, 1, 2])
        self.assertEqual([record["invariants"]["seed"] for record in records], [42, 43, 44])

    @patch("src.data.perturbers.analytical_perturb")
    def test_complete_bipartite_graph_uses_analytical_network_perturbation(
        self,
        analytical_perturb_mock,
    ) -> None:
        baseline = {"n_nodes": 5, "n_edges": 6}
        analytical_perturb_mock.return_value = baseline
        processor = SimpleNamespace(
            graph = ig.Graph.Full_Bipartite(2, 3),
            invariants = baseline,
            dimensions = None,
            events = None,
        )

        with (
            patch.dict("src.data.perturbers.NETWORK_METHODS", {"rewire": (0.1,)}, clear = True),
            patch.dict("src.data.perturbers.INVARIANT_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.PROCESS_METHODS", {}, clear = True),
            patch.dict("src.data.perturbers.SIGNATURE_METHODS", {}, clear = True),
        ):
            result = _execute_perturbations(proc = processor, name = "test")

        analytical_perturb_mock.assert_called_once()
        self.assertEqual(
            result["network_perturbed"]["rewire"][0]["invariants"],
            baseline,
        )

    def test_loader_preserves_repeated_realizations(self) -> None:
        payload = {
            "invariants_perturbed": {
                "noise": [
                    {"intensity": 0.1, "realization": 0, "invariants": {"feature": 1.0}},
                    {"intensity": 0.1, "realization": 1, "invariants": {"feature": 2.0}},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "example.json"
            path.write_text(json.dumps(payload), encoding = "utf-8")
            loaded = load_perturbed_data(path_pert = temp_dir)

        frame = loaded["invariants_perturbed"]["noise"][0.1]
        self.assertEqual(frame["realization"].tolist(), [0, 1])
        self.assertEqual(frame["feature"].tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()