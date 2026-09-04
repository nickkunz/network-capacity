import pickle
import tempfile
import unittest
from pathlib import Path

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


class VisualizationInterfaceTests(unittest.TestCase):

    def test_exposes_one_generator_per_figure(self) -> None:
        generators = (
            visualize.generate_conceptual_figure,
            visualize.generate_universality_figure,
            visualize.generate_consensus_figure,
            visualize.generate_stress_test_figure,
        )

        self.assertTrue(all(callable(generator) for generator in generators))


if __name__ == "__main__":
    unittest.main()
