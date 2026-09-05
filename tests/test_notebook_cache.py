import ast
import json
import pickle
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.evaluators.caching import load_notebook_cache


class NotebookCacheTests(unittest.TestCase):
    def test_notebooks_recompute_and_persist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("ablate", "consensus", "transfer", "perturb", "falsify"):
            notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
            cells = notebook["cells"]
            initialization = "".join(cells[5]["source"])
            cache_setup = initialization[initialization.index("from src.evaluators.caching import"):]
            pipeline = []
            for cell in cells[6:]:
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    pipeline.append(source)
                    if "cache_path.write_bytes" in source:
                        break
            for mode in ("missing", "mismatch", "forced"):
                with self.subTest(notebook = name, mode = mode), tempfile.TemporaryDirectory() as directory:
                    namespace = {
                        "root": Path(directory), "data": [1], "data_proc": [1],
                        "data_pert": {}, "data_fals": {},
                        "models": {"model": SimpleNamespace(estimator_c = None, estimator_r = None)},
                        "N_REPEATS": 30, "RANDOM_STATE": 42,
                        "TARGET": "target", "FEAT_X": ["x"], "FEAT_Z": ["z"],
                        "FORCE_RECOMPUTE": False, "pickle": pickle,
                        "combinations": combinations,
                        "pd": SimpleNamespace(DataFrame = Mock(return_value = "compiled")),
                    }
                    exec(cache_setup, namespace)
                    if mode != "missing":
                        payload = {
                            "metadata": dict(namespace["cache_metadata"]),
                            **{key: "stale" for key in namespace["cache_keys"]},
                        }
                        if mode == "mismatch":
                            payload["metadata"]["random_state"] = 0
                        cache_path = namespace["cache_path"]
                        cache_path.parent.mkdir(parents = True, exist_ok = True)
                        cache_path.write_bytes(pickle.dumps(obj = payload))
                    namespace["FORCE_RECOMPUTE"] = mode == "forced"
                    exec(cache_setup, namespace)
                    self.assertIsNone(obj = namespace["cache_payload"])
                    functions = {}
                    for source in pipeline:
                        for node in ast.walk(node = ast.parse(source = source)):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                                if node.id.startswith(("results_dict_", "predicts_dict_")):
                                    namespace[node.id] = {"stale": True}
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                function_name = node.func.id
                                if function_name.startswith(("train_", "compile_")) or function_name in (
                                    "logo_cross_valid", "kfold_cross_valid", "fit_predict_frontier", "consensus_metrics"
                                ):
                                    result = "compiled"
                                    if function_name in ("logo_cross_valid", "kfold_cross_valid"):
                                        result = ({}, [1])
                                    elif function_name == "fit_predict_frontier":
                                        result = {"y_pred": [1]}
                                    elif function_name in (
                                        "compile_decomposed_attribution", "compile_decomposed_separation",
                                        "compile_perturbed_transfer",
                                    ):
                                        result = ("compiled", "predictions")
                                    functions[function_name] = Mock(return_value = result)
                    namespace.update(functions)
                    for source in pipeline:
                        exec(source, namespace)
                    for function_name, function in functions.items():
                        if function_name != "consensus_metrics":
                            self.assertTrue(expr = function.called, msg = function_name)
                    loaded = load_notebook_cache(
                        cache_path = namespace["cache_path"],
                        metadata = namespace["cache_metadata"],
                        required_keys = namespace["cache_keys"],
                    )
                    self.assertIsNotNone(obj = loaded)
                    for key in namespace["cache_keys"]:
                        self.assertEqual(first = loaded[key], second = namespace[key])

    def test_notebooks_reuse_disk_cache_in_fresh_namespace(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("ablate", "consensus", "transfer", "perturb", "falsify"):
            with self.subTest(notebook = name), tempfile.TemporaryDirectory() as directory:
                notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
                cells = notebook["cells"]
                initialization = "".join(cells[5]["source"])
                cache_setup = initialization[initialization.index("from src.evaluators.caching import"):]
                namespace = {
                    "root": Path(directory), "data": [1], "data_proc": [1],
                    "models": {"model": None}, "N_REPEATS": 30, "RANDOM_STATE": 42,
                    "TARGET": "target", "FEAT_X": ["x"], "FEAT_Z": ["z"],
                    "FORCE_RECOMPUTE": False,
                }
                exec(cache_setup, namespace)
                payload = {
                    "metadata": namespace["cache_metadata"],
                    **{key: f"cached {key}" for key in namespace["cache_keys"]},
                }
                cache_path = namespace["cache_path"]
                cache_path.parent.mkdir(parents = True, exist_ok = True)
                original_bytes = pickle.dumps(obj = payload)
                cache_path.write_bytes(original_bytes)
                exec(cache_setup, namespace)
                post_processing = False
                for cell in cells[6:]:
                    source = "".join(cell["source"])
                    if cell["cell_type"] == "markdown":
                        if post_processing:
                            break
                        post_processing = "Post-Processing" in source
                    else:
                        exec(source, namespace)
                for key in namespace["cache_keys"]:
                    self.assertEqual(first = namespace[key], second = payload[key])
                self.assertEqual(first = cache_path.read_bytes(), second = original_bytes)

    def test_cache_validation(self) -> None:
        metadata = {"n_obs": 25, "n_repeats": 30, "random_state": 42}
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "results.pkl"
            arguments = {
                "cache_path": cache_path,
                "metadata": metadata,
                "required_keys": ("results",),
            }
            self.assertIsNone(obj = load_notebook_cache(**arguments))
            payload = {"metadata": metadata, "results": [1, 2]}
            cache_path.write_bytes(pickle.dumps(obj = payload))
            self.assertEqual(first = load_notebook_cache(**arguments), second = payload)
            self.assertIsNone(obj = load_notebook_cache(**arguments, force_recompute = True))
            for invalid in (
                {"metadata": {**metadata, "n_repeats": 10}, "results": [1]},
                {"metadata": metadata},
                {"metadata": metadata, "results": None},
                {"metadata": None, "results": [1]},
                [],
            ):
                with self.subTest(payload = invalid):
                    cache_path.write_bytes(pickle.dumps(obj = invalid))
                    self.assertIsNone(obj = load_notebook_cache(**arguments))
            for damaged in (b"", b"invalid pickle"):
                cache_path.write_bytes(damaged)
                self.assertIsNone(obj = load_notebook_cache(**arguments))