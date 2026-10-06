import ast
import json
import pickle
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from types import CodeType, SimpleNamespace
from unittest.mock import Mock

from src.evaluators.caching import load_notebook_cache


FULL_CORPUS_KEYS = {
    "ablate": "results_decomposed_full_agreement",
    "perturb": "results_perturbed_full_agreement",
    "falsify": "results_falsified_full_agreement",
}


def _cache_setup_code(source: str) -> CodeType:
    tree = ast.parse(source = source)
    start = next(
        index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {"cache_dir", "cache_path"} for target in node.targets)
    )
    tree.body = tree.body[start:]
    return compile(tree, "<notebook cache setup>", "exec")


def _execute_mocked_cell(source: str, namespace: dict) -> None:
    tree = ast.parse(source = source)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "<mocked notebook cell>", "exec"), namespace)


class NotebookCacheTests(unittest.TestCase):
    def test_notebooks_recompute_and_persist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("ablate", "consensus", "transfer", "perturb", "falsify"):
            notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
            cells = notebook["cells"]
            initialization = "".join(cells[5]["source"])
            cache_setup = _cache_setup_code(source = initialization)
            pipeline = []
            for cell in cells[6:]:
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    pipeline.append(source)
                    if "cache_path.write_bytes" in source:
                        break
            for mode in ("missing", "mismatch", "legacy_rho", "raw_rho", "forced"):
                with self.subTest(notebook = name, mode = mode), tempfile.TemporaryDirectory() as directory:
                    namespace = {
                        "root": Path(directory), "data": [1], "data_proc": [1],
                        "data_pert": {}, "data_fals": {},
                        "models": {"model": SimpleNamespace(estimator_c = None, estimator_r = None)},
                        "N_REPEATS": 30, "RANDOM_STATE": 42,
                        "TARGET": "target", "FEAT_X": ["x"], "FEAT_Z": ["z"],
                        "FORCE_RECOMPUTE": False, "pickle": pickle,
                        "load_notebook_cache": load_notebook_cache,
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
                        elif mode == "legacy_rho":
                            payload["metadata"].pop("rho_rescaled")
                        elif mode == "raw_rho":
                            payload["metadata"]["rho_rescaled"] = False
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
                                    "logo_cross_valid", "kfold_cross_valid", "fit_predict_frontier", "frontier_consensus"
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
                        _execute_mocked_cell(source = source, namespace = namespace)
                    for function_name, function in functions.items():
                        if function_name != "frontier_consensus":
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
                cache_setup = _cache_setup_code(source = initialization)
                namespace = {
                    "root": Path(directory), "data": [1], "data_proc": [1],
                    "models": {"model": None}, "N_REPEATS": 30, "RANDOM_STATE": 42,
                    "TARGET": "target", "FEAT_X": ["x"], "FEAT_Z": ["z"],
                    "FORCE_RECOMPUTE": False,
                    "load_notebook_cache": load_notebook_cache,
                    "compile_transfer_resampling": Mock(return_value = "compiled"),
                }
                exec(cache_setup, namespace)
                payload = {
                    "metadata": namespace["cache_metadata"],
                    **{key: f"cached {key}" for key in namespace["cache_keys"]},
                }
                if name in FULL_CORPUS_KEYS:
                    payload[FULL_CORPUS_KEYS[name]] = "cached full-corpus agreement"
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
                        _execute_mocked_cell(source = source, namespace = namespace)
                for key in namespace["cache_keys"]:
                    self.assertEqual(first = namespace[key], second = payload[key])
                self.assertEqual(first = cache_path.read_bytes(), second = original_bytes)

    def test_consensus_upgrade_preserves_resampling_results(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name, full_key in FULL_CORPUS_KEYS.items():
            with self.subTest(notebook = name), tempfile.TemporaryDirectory() as directory:
                notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
                source = next(
                    "".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code" and "cache_path.write_bytes" in "".join(cell["source"])
                )
                namespace = {
                    "root": Path(directory), "data": [1], "data_proc": [1],
                    "data_pert": {}, "data_fals": {}, "models": {},
                    "N_REPEATS": 30, "RANDOM_STATE": 42,
                    "TARGET": "target", "FEAT_X": ["x"], "FEAT_Z": ["z"],
                    "FORCE_RECOMPUTE": False, "pickle": pickle,
                    "load_notebook_cache": load_notebook_cache,
                    "compile_transfer_resampling": Mock(return_value = "compiled"),
                }
                exec(_cache_setup_code(source = "".join(notebook["cells"][5]["source"])), namespace)
                payload = {
                    "metadata": namespace["cache_metadata"],
                    **{key: f"cached {key}" for key in namespace["cache_keys"]},
                }
                namespace.update(payload)
                namespace["cache_payload"] = payload
                functions = {}
                for node in ast.walk(node = ast.parse(source = source)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        function_name = node.func.id
                        if function_name.startswith(("train_", "compile_")):
                            functions[function_name] = Mock(return_value = "full-corpus result")
                namespace.update(functions)
                _execute_mocked_cell(source = source, namespace = namespace)
                upgraded = pickle.loads(namespace["cache_path"].read_bytes())
                self.assertEqual(upgraded[full_key], "full-corpus result")
                for key in namespace["cache_keys"]:
                    if not key.endswith("_consensus"):
                        self.assertEqual(upgraded[key], payload[key])
                for function_name, function in functions.items():
                    if function_name.endswith(("_consensus", "_full")):
                        function.assert_called_once()
                    else:
                        function.assert_not_called()

    def test_cache_validation(self) -> None:
        metadata = {"n_obs": 25, "n_repeats": 30, "random_state": 42, "rho_rescaled": True}
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
                {"metadata": {key: value for key, value in metadata.items() if key != "rho_rescaled"}, "results": [1]},
                {"metadata": {**metadata, "rho_rescaled": False}, "results": [1]},
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