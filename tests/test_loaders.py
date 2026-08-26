import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.loaders.bitcoin import build_network_bitcoin
from src.data.loaders.epilepsy import _load_events_epilepsy
from src.data.loaders.faers import _iter_reports_faers, _process_events_faers
from src.data.loaders.federal import FederalProcessor
from src.data.loaders.metrla import _process_events_metrla
from src.data.loaders.pemsbay import _process_events_pemsbay


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


if __name__ == "__main__":
    unittest.main()