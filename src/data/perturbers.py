## libraries
import re
import os
import sys
import logging
import configparser
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Any, Sequence
from pathlib import Path
from scipy.special import ndtr

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.vectorizers.invariants import GraphInvariants
from src.vectorizers.signatures import ProcessSignatures
from src.data.loaders.federal import FederalProcessor
from src.data.loaders.bitcoin import BitcoinProcessor
from src.data.loaders.amazon import AmazonProcessor
from src.data.loaders.mooc import MoocProcessor
from src.data.loaders.world import WorldBankProcessor
from src.data.loaders.wiki import WikiProcessor
from src.data.loaders.jodie import JodieProcessor
from src.data.loaders.overflow import OverflowProcessor
from src.data.loaders.email import EmailProcessor
from src.data.loaders.celegans import CelegansProcessor
from src.data.loaders.college import CollegeProcessor
from src.data.loaders.idling import IdlingProcessor
from src.data.loaders.windmill import WindmillProcessor
from src.data.loaders.metrla import MetrLaProcessor
from src.data.loaders.pemsbay import PemsBayProcessor
from src.data.loaders.montevideo import MontevideoProcessor
from src.data.loaders.crop import CropProcessor
from src.data.loaders.faers import FaersProcessor
from src.data.loaders.epilepsy import EpilepsyProcessor
from src.data.loaders.gwosc import GwoscProcessor
from src.data.loaders.river import NwisProcessor
from src.data.loaders.auger import AugerProcessor
from src.data.loaders.seismic import SeismicProcessor
from src.data.loaders.rain import RainProcessor
from src.data.loaders.chickenpox import ChickenpoxProcessor
from src.evaluators.perturbing import (
    network_perturb,
    feature_perturb,
    process_perturb,
    analytical_perturb
)
from src.data.helpers import (
    _save_to_json, 
    _extract_counts
)

## logging
logger = logging.getLogger(__name__)

## configs
config = configparser.ConfigParser()
config.read(os.path.join(root, 'conf', 'settings.ini'))

## constants
PATH_ROOT = config['paths']['PATH_ROOT'].strip('"')
PATH_PROC = config['paths']['PATH_PROC'].strip('"')
PATH_PERT = config['paths']['PATH_PERT'].strip('"')

NAME_AMAZON = config['names']['NAME_AMAZON']
NAME_BITCOIN = config['names']['NAME_BITCOIN']
NAME_FEDERAL = config['names']['NAME_FEDERAL']
NAME_MOOC = config['names']['NAME_MOOC']
NAME_WORLD = config['names']['NAME_WORLD']
NAME_WIKI = config['names']['NAME_WIKI']
NAME_JODIE = config['names']['NAME_JODIE']
NAME_OVERFLOW = config['names']['NAME_OVERFLOW']
NAME_EMAIL = config['names']['NAME_EMAIL']
NAME_COLLEGE = config['names']['NAME_COLLEGE']
NAME_CELEGANS = config['names']['NAME_CELEGANS']
NAME_IDLING = config['names']['NAME_IDLING']
NAME_WINDMILL = config['names']['NAME_WINDMILL']
NAME_METRLA = config['names']['NAME_METRLA']
NAME_PEMSBAY = config['names']['NAME_PEMSBAY']
NAME_MONTEVIDEO = config['names']['NAME_MONTEVIDEO']
NAME_CROP = config['names']['NAME_CROP']
NAME_FAERS = config['names']['NAME_FAERS']
NAME_EPILEPSY = config['names']['NAME_EPILEPSY']
NAME_CHICKENPOX = config['names']['NAME_CHICKENPOX']
NAME_GWOSC = config['names']['NAME_GWOSC']
NAME_RIVER = config['names']['NAME_RIVER']
NAME_AUGER = config['names']['NAME_AUGER']
NAME_SEISMIC = config['names']['NAME_SEISMIC']
NAME_RAIN = config['names']['NAME_RAIN']

URL_AMAZON = config['urls']['URL_AMAZON'].strip('"')
URL_FEDERAL = config['urls']['URL_FEDERAL'].strip('"')
URL_MOOC = config['urls']['URL_MOOC'].strip('"')
URL_WORLD_NETWORK = config['urls']['URL_WORLD_NETWORK'].strip('"')
URL_WORLD_METADATA = config['urls']['URL_WORLD_METADATA'].strip('"')
URL_WIKI = config['urls']['URL_WIKI'].strip('"')
URL_OVERFLOW = config['urls']['URL_OVERFLOW'].strip('"')
URL_EMAIL = config['urls']['URL_EMAIL'].strip('"')
URL_COLLEGE = config['urls']['URL_COLLEGE'].strip('"')
URL_CROP = config['urls'].get('URL_CROP', config['urls'].get('URL_CROP_RADER', '')).strip('"')
URL_CROP_SAMPLING = config['urls']['URL_CROP_SAMPLING'].strip('"')
URL_CROP_FIELD = config['urls']['URL_CROP_FIELD'].strip('"')
URL_FAERS = config['urls']['URL_FAERS'].strip('"')
URL_EPILEPSY = config['urls']['URL_EPILEPSY'].strip('"')
URL_CHICKENPOX_EVENTS = config['urls']['URL_CHICKENPOX_EVENTS'].strip('"')
URL_GWOSC = config['urls']['URL_GWOSC'].strip('"')
URL_RIVER_SITE = config['urls']['URL_RIVER_SITE'].strip('"')
URL_RIVER_IV = config['urls']['URL_RIVER_IV'].strip('"')
URL_AUGER_NETWORK = config['urls']['URL_AUGER_NETWORK'].strip('"')
URL_AUGER_EVENTS = config['urls']['URL_AUGER_EVENTS'].strip('"')
URL_SEISMIC_NETWORK = config['urls']['URL_SEISMIC_NETWORK'].strip('"')
URL_SEISMIC_EVENTS = config['urls']['URL_SEISMIC_EVENTS'].strip('"')

NETWORK_METHODS = {
    "rewire":      tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    "densify":     tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    "sample":      tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2))
}
INVARIANT_METHODS = {
    'noise':  tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    'jitter': tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    'subset': tuple(np.round(np.linspace(start = 0.95, stop = 0.65, num = 7), decimals = 2)),
}
PROCESS_METHODS = {
    'scaling':       (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75),
    'smoothing':     (3.00, 5.00, 7.00, 9.00, 11.00, 13.00, 15.00),
    'bootstrapping': tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2))
}
SIGNATURE_METHODS = {
    'noise':  tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    'jitter': tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
    'subset': tuple(np.round(np.linspace(start = 0.95, stop = 0.65, num = 7), decimals = 2)),
}
# TEMPORAL_METHODS = {
#     'aggregation': ('2D', '7D', '14D', '30D', '60D', '90D', '180D'),
#     'jitter':      tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
#     'dropout':     tuple(np.round(np.linspace(start = 0.05, stop = 0.35, num = 7), decimals = 2)),
# }

## helper functions
def _is_fully_connected_bipartite(graph: Any) -> bool:

    """Check if a graph is a fully connected bipartite graph."""

    if graph.vcount() == 0 or graph.ecount() == 0:
        return False
    is_bip, types = graph.is_bipartite(return_types=True)
    if not is_bip or types is None:
        return False
    n1 = sum(types)
    n2 = len(types) - n1
    return graph.ecount() == n1 * n2

def _load_corpus_feature_scales(path_proc: str = PATH_PROC) -> tuple[pd.Series, pd.Series]:

    """Load corpus-wide standard deviations for invariant and signature noise."""

    invariants = list()
    signatures = list()
    for path in sorted((Path(root) / path_proc).glob("*.json")):
        with open(path, "r") as file:
            payload = json.load(file)
        invariants.append(payload.get("invariants", dict()))
        signatures.append(payload.get("signatures", dict()))

    if len(invariants) < 2 or len(signatures) < 2:
        raise ValueError("at least two processed datasets are required for corpus feature scales")

    invariant_scale = pd.DataFrame(invariants).apply(pd.to_numeric, errors = "coerce").std(axis = 0)
    signature_scale = pd.DataFrame(signatures).apply(pd.to_numeric, errors = "coerce").std(axis = 0)
    return invariant_scale.fillna(0.0), signature_scale.fillna(0.0)

def _jitter_count_series(
    positions: np.ndarray,
    counts: np.ndarray,
    sigma: float,
    lower: int,
    upper: int,
    rng: np.random.Generator,
    chunk_size: int = 100_000,
    multinomial_threshold: int = 1_000_000,
    ) -> np.ndarray:

    """Jitter count-weighted positions without expanding to individual events."""

    shifted_counts = np.zeros(upper - lower + 1, dtype = np.int64)
    if sigma <= 0:
        for position, count in zip(positions, counts):
            shifted_counts[int(np.clip(position, lower, upper)) - lower] += max(0, int(count))
        return shifted_counts

    if int(np.sum(np.maximum(counts, 0))) > multinomial_threshold:
        boundaries = np.arange(lower, upper, dtype = float) + 0.5
        for position, count in zip(positions, counts):
            count = max(0, int(count))
            if count == 0:
                continue
            cumulative = ndtr((boundaries - float(position)) / sigma)
            probabilities = np.diff(np.concatenate(([0.0], cumulative, [1.0])))
            probabilities = np.clip(probabilities, 0.0, 1.0)
            probabilities /= probabilities.sum()
            shifted_counts += rng.multinomial(count, probabilities)
        return shifted_counts

    for position, count in zip(positions, counts):
        remaining = max(0, int(count))
        while remaining > 0:
            draw_size = min(remaining, chunk_size)
            shifted = np.rint(
                float(position) + rng.normal(0, sigma, size = draw_size)
            ).astype(np.int64)
            np.clip(shifted, lower, upper, out = shifted)
            shifted_counts += np.bincount(
                shifted - lower,
                minlength = len(shifted_counts),
            )
            remaining -= draw_size
    return shifted_counts

def _network_worker(
    graph: Any,
    job: tuple[str, float, int, int],
    ) -> tuple[str, float, int, dict[str, Any] | None, str | None]:

    """Run one explicit network perturbation."""

    method, intensity, realization, seed = job
    try:
        features = network_perturb(
            graph = graph,
            method = method,
            intensity = intensity,
            random_state = seed,
        )
        return method, intensity, realization, features, None
    except Exception as exc:
        return method, intensity, realization, None, f"{type(exc).__name__}: {exc}"

def _network_worker_batch(
    graph: Any,
    jobs: list[tuple[int, tuple[str, float, int, int]]],
    connection: Any,
    ) -> None:

    """Run a process-local batch and return indexed results."""

    try:
        connection.send([
            (index, _network_worker(graph = graph, job = job))
            for index, job in jobs
        ])
    finally:
        connection.close()

def _resolve_n_jobs(
    n_jobs: int,
    n_tasks: int,
    cpu_count: int | None = None,
    ) -> int:

    """Resolve worker count against available CPUs and pending tasks."""

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or >= 1")
    if n_tasks < 1:
        return 1
    if cpu_count is None:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
    requested = cpu_count if n_jobs == -1 else n_jobs
    return max(1, min(int(requested), int(cpu_count), n_tasks))

def _validate_perturbation_records(
    records: dict[str, list[dict[str, Any]]],
    methods: dict[str, Sequence[Any]],
    realizations: int | dict[str, int],
    name: str,
    channel: str,
    ) -> None:

    """Reject incomplete perturbation output before it can be saved."""

    mismatches = list()
    for method, settings in methods.items():
        count = realizations[method] if isinstance(realizations, dict) else realizations
        expected = len(settings) * count
        actual = len(records.get(method, list()))
        if actual != expected:
            mismatches.append(f"{method}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError(f"Incomplete {channel} perturbations for {name}: {'; '.join(mismatches)}")

def _execute_perturbations(
    proc: Any,
    name: str,
    random_state: int = 42,
    n_realizations: int = 30,
    n_jobs: int = -1,
    ) -> dict[str, Any]:

    """Run network, process, and temporal perturbations for a given processor."""

    ## validate inputs
    if n_realizations < 1:
        raise ValueError("n_realizations must be >= 1")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or >= 1")
    results = dict()

    ## --- network perturbation --- ##
    graph = getattr(proc, 'graph', None)
    pre_inv = getattr(proc, 'invariants', None)
    dimensions = getattr(proc, 'dimensions', None)
    if graph is not None:
        
        ## ensure simple undirected graph (remove multi-edges and self-loops)
        graph_was_simple = graph.is_simple()
        graph.simplify()

        ## use the exact analytical shortcut only when node sampling preserves structure
        network_results: dict[str, list[dict[str, Any]]] = dict()
        complete_bipartite = _is_fully_connected_bipartite(graph)
        bipartite_dimensions = None
        partition_types = None

        if complete_bipartite:
            _, types = graph.is_bipartite(return_types = True)
            partition_types = np.asarray(types, dtype = bool)
            bipartite_dimensions = (
                int(np.count_nonzero(~partition_types)),
                int(np.count_nonzero(partition_types)),
            )
            invariants = dict(pre_inv) if pre_inv is not None else GraphInvariants(graph).all(analytical = True)
            logging.info(f"  Using exact analytical node sampling for {name}")
        else:
            invariants = (
                dict(pre_inv)
                if pre_inv is not None and graph_was_simple
                else GraphInvariants(graph).all(analytical = False)
            )

        jobs = [
            (method, float(intensity), realization, random_state + realization)
            for method, intensities in NETWORK_METHODS.items()
            if not (complete_bipartite and method == "sample")
            for intensity in intensities
            for realization in range(n_realizations)
        ]
        if jobs:
            workers = _resolve_n_jobs(n_jobs = n_jobs, n_tasks = len(jobs))
            logging.info(f"  Using {workers} network worker process(es) for {name}")

            if workers == 1:
                worker_results = [
                    _network_worker(graph = graph, job = job)
                    for job in jobs
                ]
            else:
                start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
                context = mp.get_context(start_method)
                indexed_jobs = list(enumerate(jobs))
                batches = [
                    indexed_jobs[worker_index::workers]
                    for worker_index in range(workers)
                ]
                processes = list()
                connections = list()
                for batch in batches:
                    parent_connection, child_connection = context.Pipe(duplex = False)
                    process = context.Process(
                        target = _network_worker_batch,
                        args = (graph, batch, child_connection),
                    )
                    process.start()
                    child_connection.close()
                    processes.append(process)
                    connections.append(parent_connection)

                indexed_results = list()
                try:
                    for connection in connections:
                        indexed_results.extend(connection.recv())
                except BaseException:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    raise
                finally:
                    for connection in connections:
                        connection.close()
                    for process in processes:
                        process.join()

                failed = [process.pid for process in processes if process.exitcode != 0]
                if failed:
                    raise RuntimeError(f"network worker processes failed: {failed}")
                worker_results = [
                    result
                    for _, result in sorted(indexed_results, key = lambda item: item[0])
                ]

            for method, intensity, realization, features, error in worker_results:
                if error is not None:
                    logging.warning(
                        f"Network {method} @ {intensity:.2f} realization {realization} failed for {name}: {error}"
                    )
                    continue
                network_results.setdefault(method, []).append({
                    'intensity': intensity,
                    'realization': realization,
                    'invariants': features
                })

        if complete_bipartite and "sample" in NETWORK_METHODS:
            for intensity in NETWORK_METHODS["sample"]:
                for realization in range(n_realizations):
                    features = analytical_perturb(
                        dimensions = bipartite_dimensions,
                        partition_types = partition_types,
                        method = "uniform_node_sampling",
                        intensity = float(intensity),
                        random_state = random_state + realization,
                    )
                    network_results.setdefault("sample", []).append({
                        'intensity': float(intensity),
                        'realization': realization,
                        'invariants': features,
                    })
        _validate_perturbation_records(
            records = network_results,
            methods = NETWORK_METHODS,
            realizations = n_realizations,
            name = name,
            channel = "network",
        )
        results['network_perturbed'] = network_results
        total = sum(len(v) for v in network_results.values())
        logging.info(f"  Network perturbation: {total} records")
    elif pre_inv is not None and dimensions is not None:
        m, n = int(dimensions[0]), int(dimensions[1])
        invariants = dict(pre_inv)
        network_results: dict[str, list[dict[str, Any]]] = dict()
        analytical_methods = {
            "rewire": "degree_preserving_rewire",
            "densify": "bernoulli_edge_densification",
            "sample": "uniform_node_sampling",
        }
        unknown = sorted(set(NETWORK_METHODS) - set(analytical_methods))
        if unknown:
            raise ValueError(f"Unsupported graph-free network methods: {unknown}")
        realization_counts = {
            method: n_realizations if method == "sample" else 1
            for method in NETWORK_METHODS
        }
        logging.info(f"  Using analytical perturbation for {name} ({m:,} x {n:,}) [graph-free]")
        for method, intensities in NETWORK_METHODS.items():
            for intensity in intensities:
                for realization in range(realization_counts[method]):
                    features = analytical_perturb(
                        dimensions = (m, n),
                        method = analytical_methods[method],
                        intensity = float(intensity),
                        random_state = random_state + realization,
                    )
                    network_results.setdefault(method, []).append({
                        'intensity': float(intensity),
                        'realization': realization,
                        'invariants': features
                    })
        _validate_perturbation_records(
            records = network_results,
            methods = NETWORK_METHODS,
            realizations = realization_counts,
            name = name,
            channel = "network",
        )
        results['network_perturbed'] = network_results
        total = sum(len(v) for v in network_results.values())
        logging.info(f"  Network perturbation: {total} records")
    else:
        logging.warning(f"  No graph object for {name}, skipping network perturbation.")

    ## --- invariant perturbation --- ##
    if graph is not None or pre_inv is not None:
        invariant_scale, signature_scale = _load_corpus_feature_scales()
        base_inv = invariants if graph is not None else pre_inv
        base_df = pd.DataFrame([base_inv])
        invariant_results: dict[str, list[dict[str, Any]]] = dict()
        for method, params in INVARIANT_METHODS.items():
            for param in params:
                for realization in range(n_realizations):
                    try:
                        perturbed_df = feature_perturb(
                            base_df.copy(),
                            method = method,
                            noise = float(param) if method != 'subset' else 0.05,
                            subset = float(param) if method == 'subset' else 0.8,
                            random_state = random_state + realization,
                            scale = invariant_scale,
                        )
                        row = perturbed_df.iloc[0].to_dict()
                    except Exception as exc:
                        logging.warning(f"Invariant {method} @ {param:.3f} realization {realization} failed for {name}: {exc}")
                        continue
                    invariant_results.setdefault(method, []).append({
                        'intensity': float(param),
                        'realization': realization,
                        'invariants': row
                    })
        _validate_perturbation_records(
            records = invariant_results,
            methods = INVARIANT_METHODS,
            realizations = n_realizations,
            name = name,
            channel = "invariant",
        )
        results['invariants_perturbed'] = invariant_results
        total = sum(len(v) for v in invariant_results.values())
        logging.info(f"  Invariant perturbation: {total} records")

    ## --- process perturbation --- ##
    events = getattr(proc, 'events', None)
    counts = _extract_counts(events)

    if counts is not None and len(counts) > 0:
        process_results: dict[str, list[dict[str, Any]]] = dict()
        for method, params in PROCESS_METHODS.items():
            for param in params:
                realizations = range(n_realizations) if method == 'bootstrapping' else range(1)
                for realization in realizations:
                    try:
                        sigs = process_perturb(
                            counts,
                            method = method,
                            param = float(param),
                            random_state = random_state + realization,
                        )
                    except Exception as exc:
                        logging.warning(f"Process {method} @ {param} realization {realization} failed for {name}: {exc}")
                        continue
                    process_results.setdefault(method, []).append({
                        'intensity': float(param),
                        'realization': realization,
                        'signatures': sigs
                    })
        _validate_perturbation_records(
            records = process_results,
            methods = PROCESS_METHODS,
            realizations = {
                method: n_realizations if method == 'bootstrapping' else 1
                for method in PROCESS_METHODS
            },
            name = name,
            channel = "process",
        )
        results['process_perturbed'] = process_results
        total = sum(len(v) for v in process_results.values())
        logging.info(f"  Process perturbation: {total} records")
    else:
        logging.warning(f"  No count series for {name}, skipping process perturbation.")

    ## --- signature perturbation (z -> z') --- ##
    if counts is not None and len(counts) > 0:
        if "signature_scale" not in locals():
            _, signature_scale = _load_corpus_feature_scales()
        data_temp = pd.DataFrame({"counts": counts, "idx": range(len(counts))})
        base_sigs = ProcessSignatures(data_temp, sort_by = ["idx"], target = "counts").all()
        base_sig_df = pd.DataFrame([base_sigs])
        sig_pert_results: dict[str, list[dict[str, Any]]] = dict()
        for method, params in SIGNATURE_METHODS.items():
            for param in params:
                for realization in range(n_realizations):
                    try:
                        perturbed_df = feature_perturb(
                            base_sig_df.copy(),
                            method = method,
                            noise = float(param) if method != 'subset' else 0.05,
                            subset = float(param) if method == 'subset' else 0.8,
                            random_state = random_state + realization,
                            scale = signature_scale,
                        )
                        row = perturbed_df.iloc[0].to_dict()
                    except Exception as exc:
                        logging.warning(f"Signature {method} @ {param:.3f} realization {realization} failed for {name}: {exc}")
                        continue
                    sig_pert_results.setdefault(method, []).append({
                        'intensity': float(param),
                        'realization': realization,
                        'signatures': row
                    })
        _validate_perturbation_records(
            records = sig_pert_results,
            methods = SIGNATURE_METHODS,
            realizations = n_realizations,
            name = name,
            channel = "signature",
        )
        results['signatures_perturbed'] = sig_pert_results
        total = sum(len(v) for v in sig_pert_results.values())
        logging.info(f"  Signature perturbation: {total} records")

    ## --- temporal aggregation --- ##
    # if events is not None and isinstance(events, pd.DataFrame) and not events.empty:
    #     date_col = next((c for c in ('date', 'datetime', 'timestamp', 'day') if c in events.columns), None)
    #     target_col = next((c for c in ('target', 'count') if c in events.columns), None)

    #     if date_col is not None and target_col is not None:
    #         temporal_results: dict[str, list[dict[str, Any]]] = dict()
    #         data_temp = events[[date_col, target_col]].copy()
    #         is_ordinal = pd.api.types.is_integer_dtype(data_temp[date_col])

    #         if is_ordinal:
    #             data_temp = data_temp.sort_values(date_col).reset_index(drop = True)
    #             day_min = int(data_temp[date_col].min())
    #             day_max = int(data_temp[date_col].max())
    #         else:
    #             data_temp[date_col] = pd.to_datetime(data_temp[date_col])
    #             data_temp = data_temp.set_index(date_col).sort_index()

    #         for method, params in TEMPORAL_METHODS.items():
    #             for param in params:
    #                 realizations = range(1) if method == 'aggregation' else range(n_realizations)
    #                 for realization in realizations:
    #                     try:
    #                         rng = np.random.default_rng(random_state + realization)
    #                         if method == 'aggregation':
    #                             scale = param
    #                             if is_ordinal:
    #                                 scale_days = int(re.match(r'(\d+)', scale).group(1))
    #                                 bin_edges = list(range(day_min, day_max + scale_days, scale_days))
    #                                 if len(bin_edges) < 2:
    #                                     bin_edges = [day_min, day_min + scale_days]
    #                                 labels = bin_edges[:-1]
    #                                 data_temp['_bin'] = pd.cut(
    #                                     data_temp[date_col], bins = bin_edges,
    #                                     right = False, labels = labels, include_lowest = True
    #                                 )
    #                                 agg = data_temp.groupby('_bin', observed = False)[target_col].sum()
    #                                 records = [
    #                                     {'day': int(b), 'target': int(v)}
    #                                     for b, v in agg.items()
    #                                 ]
    #                                 data_temp.drop(columns = '_bin', inplace = True, errors = 'ignore')
    #                             else:
    #                                 resampled = data_temp[target_col].resample(scale).sum()
    #                                 records = [
    #                                     {'date': str(dt.date()), 'target': int(val)}
    #                                     for dt, val in resampled.items()
    #                                 ]
    #                             temporal_results.setdefault(method, []).append({'intensity': scale, 'realization': realization, 'events': records})

    #                         elif method == 'jitter':
    #                             intensity = float(param)
    #                             if is_ordinal:
    #                                 counts_arr = data_temp[target_col].values.clip(0).astype(int)
    #                                 if int(counts_arr.sum()) == 0:
    #                                     continue
    #                                 sigma = intensity * max(day_max - day_min, 1)
    #                                 new_counts = _jitter_count_series(
    #                                     positions = data_temp[date_col].values.astype(int),
    #                                     counts = counts_arr,
    #                                     sigma = sigma,
    #                                     lower = day_min,
    #                                     upper = day_max,
    #                                     rng = rng,
    #                                 )
    #                                 records = [
    #                                     {'day': int(day_min + offset), 'target': int(new_counts[offset])}
    #                                     for offset in np.flatnonzero(new_counts)
    #                                 ]
    #                             else:
    #                                 daily = data_temp[target_col].resample('1D').sum()
    #                                 n_days = len(daily)
    #                                 if n_days == 0:
    #                                     continue
    #                                 counts_arr = daily.values.clip(0).astype(int)
    #                                 if int(counts_arr.sum()) == 0:
    #                                     continue
    #                                 sigma = intensity * max(n_days, 1)
    #                                 new_daily = _jitter_count_series(
    #                                     positions = np.arange(n_days),
    #                                     counts = counts_arr,
    #                                     sigma = sigma,
    #                                     lower = 0,
    #                                     upper = n_days - 1,
    #                                     rng = rng,
    #                                 )
    #                                 records = [
    #                                     {'date': str(daily.index[i].date()), 'target': int(new_daily[i])}
    #                                     for i in range(n_days)
    #                                 ]
    #                             temporal_results.setdefault(method, []).append({'intensity': intensity, 'realization': realization, 'events': records})

    #                         elif method == 'dropout':
    #                             intensity = float(param)
    #                             if is_ordinal:
    #                                 counts_arr = data_temp[target_col].values.clip(0).astype(int)
    #                                 survived = rng.binomial(counts_arr, max(0.0, 1.0 - intensity))
    #                                 records = [
    #                                     {'day': int(data_temp[date_col].iloc[i]), 'target': int(survived[i])}
    #                                     for i in range(len(data_temp))
    #                                 ]
    #                             else:
    #                                 counts_arr = data_temp[target_col].values.clip(0).astype(int)
    #                                 survived = rng.binomial(counts_arr, max(0.0, 1.0 - intensity))
    #                                 dropped = pd.Series(survived.astype(float), index = data_temp.index)
    #                                 resampled = dropped.resample('1D').sum()
    #                                 records = [
    #                                     {'date': str(dt.date()), 'target': int(val)}
    #                                     for dt, val in resampled.items()
    #                                 ]
    #                             temporal_results.setdefault(method, []).append({'intensity': intensity, 'realization': realization, 'events': records})

    #                     except Exception as exc:
    #                         logging.warning(f"Temporal {method} @ {param} realization {realization} failed for {name}: {exc}")
    #                         continue

    #         _validate_perturbation_records(
    #             records = temporal_results,
    #             methods = TEMPORAL_METHODS,
    #             realizations = {
    #                 method: 1 if method == 'aggregation' else n_realizations
    #                 for method in TEMPORAL_METHODS
    #             },
    #             name = name,
    #             channel = "temporal",
    #         )
    #         results['temporal_perturbed'] = temporal_results
    #         total = sum(len(v) for v in temporal_results.values())
    #         logging.info(f"  Temporal perturbations: {total} records")
    #     else:
    #         logging.warning(f"  No date/target columns for {name}, skipping temporal perturbations.")
    # else:
    #     logging.warning(f"  No events for {name}, skipping temporal perturbations.")

    return results

## perturbation pipeline
def json_perturber(
    force: bool = False,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    ):

    ## ensure perturbation directory exists
    os.makedirs(name = PATH_PERT, exist_ok = True)
    included = None if include is None else set(include)
    excluded = set(exclude)

    def should_regenerate(path: str) -> bool:
        name = Path(path).stem
        selected = included is None or name in included
        return selected and name not in excluded and (force or not os.path.exists(path))

    ## --- federal contracts --- ##
    federal_path = os.path.join(PATH_PERT, f"{NAME_FEDERAL}.json")
    if should_regenerate(federal_path):
        if force and os.path.exists(federal_path):
            logging.info(f"Overwriting existing federal perturbations at {federal_path}")
        else:
            logging.info("Perturbing Federal data...")
        proc = FederalProcessor(
            url = URL_FEDERAL,
            start_date = "2011-01-01",
            end_date = "2024-12-31",
            keyword = "waterfowl"
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_FEDERAL)
        _save_to_json(data = data, path = federal_path)
        logging.info(f"Federal perturbations saved to {federal_path}")
    else:
        logging.info(f"Federal perturbations already exist at {federal_path}. Skipping.")

    ## --- mooc students --- ##
    mooc_path = os.path.join(PATH_PERT, f"{NAME_MOOC}.json")
    if should_regenerate(mooc_path):
        if force and os.path.exists(mooc_path):
            logging.info(f"Overwriting existing MOOC perturbations at {mooc_path}")
        else:
            logging.info("Perturbing MOOC data...")
        proc = MoocProcessor(url = URL_MOOC)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_MOOC)
        _save_to_json(data = data, path = mooc_path)
        logging.info(f"MOOC perturbations saved to {mooc_path}")
    else:
        logging.info(f"MOOC perturbations already exist at {mooc_path}. Skipping.")

    ## --- bitcoin trust --- ##
    bitcoin_path = os.path.join(PATH_PERT, f"{NAME_BITCOIN}.json")
    if should_regenerate(bitcoin_path):
        if force and os.path.exists(bitcoin_path):
            logging.info(f"Overwriting existing Bitcoin perturbations at {bitcoin_path}")
        else:
            logging.info("Perturbing Bitcoin data...")
        proc = BitcoinProcessor(root_path = PATH_ROOT, name = NAME_BITCOIN)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_BITCOIN)
        _save_to_json(data = data, path = bitcoin_path)
        logging.info(f"Bitcoin perturbations saved to {bitcoin_path}")
    else:
        logging.info(f"Bitcoin perturbations already exist at {bitcoin_path}. Skipping.")

    ## --- world bank --- ##
    world_path = os.path.join(PATH_PERT, f"{NAME_WORLD}.json")
    if should_regenerate(world_path):
        if force and os.path.exists(world_path):
            logging.info(f"Overwriting existing World Bank perturbations at {world_path}")
        else:
            logging.info("Perturbing World Bank data...")
        proc = WorldBankProcessor(
            url_projects = URL_WORLD_NETWORK,
            url_meta = URL_WORLD_METADATA,
            start_year = "2014",
            end_year = "2024"
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_WORLD)
        _save_to_json(data = data, path = world_path)
        logging.info(f"World Bank perturbations saved to {world_path}")
    else:
        logging.info(f"World Bank perturbations already exist at {world_path}. Skipping.")

    ## --- math wiki --- ##
    wiki_path = os.path.join(PATH_PERT, f"{NAME_WIKI}.json")
    if should_regenerate(wiki_path):
        if force and os.path.exists(wiki_path):
            logging.info(f"Overwriting existing Wiki perturbations at {wiki_path}")
        else:
            logging.info("Perturbing Wiki data...")
        proc = WikiProcessor(url = URL_WIKI, name = "wikivital_mathematics.json")
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_WIKI)
        _save_to_json(data = data, path = wiki_path)
        logging.info(f"Wiki perturbations saved to {wiki_path}")
    else:
        logging.info(f"Wiki perturbations already exist at {wiki_path}. Skipping.")

    ## --- jodie wiki --- ##
    jodie_path = os.path.join(PATH_PERT, f"{NAME_JODIE}.json")
    if should_regenerate(jodie_path):
        if force and os.path.exists(jodie_path):
            logging.info(f"Overwriting existing JODIE perturbations at {jodie_path}")
        else:
            logging.info("Perturbing JODIE data...")
        proc = JodieProcessor(root_path = PATH_ROOT, name = "wikipedia")
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_JODIE)
        _save_to_json(data = data, path = jodie_path)
        logging.info(f"JODIE perturbations saved to {jodie_path}")
    else:
        logging.info(f"JODIE perturbations already exist at {jodie_path}. Skipping.")

    ## --- mathoverflow --- ##
    overflow_path = os.path.join(PATH_PERT, f"{NAME_OVERFLOW}.json")
    if should_regenerate(overflow_path):
        if force and os.path.exists(overflow_path):
            logging.info(f"Overwriting existing MathOverflow perturbations at {overflow_path}")
        else:
            logging.info("Perturbing MathOverflow data...")
        proc = OverflowProcessor(url = URL_OVERFLOW)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_OVERFLOW)
        _save_to_json(data = data, path = overflow_path)
        logging.info(f"MathOverflow perturbations saved to {overflow_path}")
    else:
        logging.info(f"MathOverflow perturbations already exist at {overflow_path}. Skipping.")

    ## --- eu-core email --- ##
    email_path = os.path.join(PATH_PERT, f"{NAME_EMAIL}.json")
    if should_regenerate(email_path):
        if force and os.path.exists(email_path):
            logging.info(f"Overwriting existing EU-Core Email perturbations at {email_path}")
        else:
            logging.info("Perturbing EU-Core Email data...")
        proc = EmailProcessor(url = URL_EMAIL)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_EMAIL)
        _save_to_json(data = data, path = email_path)
        logging.info(f"EU-Core Email perturbations saved to {email_path}")
    else:
        logging.info(f"EU-Core Email perturbations already exist at {email_path}. Skipping.")

    ## --- college --- ##
    college_path = os.path.join(PATH_PERT, f"{NAME_COLLEGE}.json")
    if should_regenerate(college_path):
        if force and os.path.exists(college_path):
            logging.info(f"Overwriting existing UC Irvine College Message perturbations at {college_path}")
        else:
            logging.info("Perturbing UC Irvine College Message data...")
        proc = CollegeProcessor(url = URL_COLLEGE)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_COLLEGE)
        _save_to_json(data = data, path = college_path)
        logging.info(f"UC Irvine College Message perturbations saved to {college_path}")
    else:
        logging.info(f"UC Irvine College Message perturbations already exist at {college_path}. Skipping.")

    ## --- idling --- ##
    idling_path = os.path.join(PATH_PERT, f"{NAME_IDLING}.json")
    if should_regenerate(idling_path):
        if force and os.path.exists(idling_path):
            logging.info(f"Overwriting existing Halifax idling perturbations at {idling_path}")
        else:
            logging.info("Perturbing Halifax idling data...")
        proc = IdlingProcessor(path_events = PATH_ROOT + "idling/")
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_IDLING)
        _save_to_json(data = data, path = idling_path)
        logging.info(f"Halifax idling perturbations saved to {idling_path}")
    else:
        logging.info(f"Halifax idling perturbations already exist at {idling_path}. Skipping.")

    ## --- windmill --- ##
    windmill_path = os.path.join(PATH_PERT, f"{NAME_WINDMILL}.json")
    if should_regenerate(windmill_path):
        if force and os.path.exists(windmill_path):
            logging.info(f"Overwriting existing Windmill perturbations at {windmill_path}")
        else:
            logging.info("Perturbing Windmill data...")
        proc = WindmillProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_WINDMILL))
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_WINDMILL)
        _save_to_json(data = data, path = windmill_path)
        logging.info(f"Windmill perturbations saved to {windmill_path}")
    else:
        logging.info(f"Windmill perturbations already exist at {windmill_path}. Skipping.")

    ## --- metr-la --- ##
    metrla_path = os.path.join(PATH_PERT, f"{NAME_METRLA}.json")
    if should_regenerate(metrla_path):
        if force and os.path.exists(metrla_path):
            logging.info(f"Overwriting existing METR-LA perturbations at {metrla_path}")
        else:
            logging.info("Perturbing METR-LA data...")
        proc = MetrLaProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_METRLA))
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_METRLA)
        _save_to_json(data = data, path = metrla_path)
        logging.info(f"METR-LA perturbations saved to {metrla_path}")
    else:
        logging.info(f"METR-LA perturbations already exist at {metrla_path}. Skipping.")

    ## --- pems-bay --- ##
    pemsbay_path = os.path.join(PATH_PERT, f"{NAME_PEMSBAY}.json")
    if should_regenerate(pemsbay_path):
        if force and os.path.exists(pemsbay_path):
            logging.info(f"Overwriting existing PEMS-BAY perturbations at {pemsbay_path}")
        else:
            logging.info("Perturbing PEMS-BAY data...")
        proc = PemsBayProcessor(raw_data_dir = os.path.join(PATH_ROOT, NAME_PEMSBAY))
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_PEMSBAY)
        _save_to_json(data = data, path = pemsbay_path)
        logging.info(f"PEMS-BAY perturbations saved to {pemsbay_path}")
    else:
        logging.info(f"PEMS-BAY perturbations already exist at {pemsbay_path}. Skipping.")

    ## --- montevideo --- ##
    montevideo_path = os.path.join(PATH_PERT, f"{NAME_MONTEVIDEO}.json")
    if should_regenerate(montevideo_path):
        if force and os.path.exists(montevideo_path):
            logging.info(f"Overwriting existing Montevideo perturbations at {montevideo_path}")
        else:
            logging.info("Perturbing Montevideo data...")
        proc = MontevideoProcessor()
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_MONTEVIDEO)
        _save_to_json(data = data, path = montevideo_path)
        logging.info(f"Montevideo perturbations saved to {montevideo_path}")
    else:
        logging.info(f"Montevideo perturbations already exist at {montevideo_path}. Skipping.")

    ## --- crop pollinator --- ##
    crop_path = os.path.join(PATH_PERT, f"{NAME_CROP}.json")
    if should_regenerate(crop_path):
        if force and os.path.exists(crop_path):
            logging.info(f"Overwriting existing CropPol perturbations at {crop_path}")
        else:
            logging.info("Perturbing CropPol data...")
        proc = CropProcessor(url = URL_CROP)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_CROP)
        _save_to_json(data = data, path = crop_path)
        logging.info(f"CropPol perturbations saved to {crop_path}")
    else:
        logging.info(f"CropPol perturbations already exist at {crop_path}. Skipping.")

    ## --- faers --- ##
    faers_path = os.path.join(PATH_PERT, f"{NAME_FAERS}.json")
    if should_regenerate(faers_path):
        if force and os.path.exists(faers_path):
            logging.info(f"Overwriting existing FAERS perturbations at {faers_path}")
        else:
            logging.info("Perturbing FAERS data...")
        proc = FaersProcessor(id = "IMATINIB", url = URL_FAERS)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_FAERS)
        _save_to_json(data = data, path = faers_path)
        logging.info(f"FAERS perturbations saved to {faers_path}")
    else:
        logging.info(f"FAERS perturbations already exist at {faers_path}. Skipping.")

    ## --- c. elegans --- ##
    celegans_path = os.path.join(PATH_PERT, f"{NAME_CELEGANS}.json")
    if should_regenerate(celegans_path):
        if force and os.path.exists(celegans_path):
            logging.info(f"Overwriting existing C. Elegans perturbations at {celegans_path}")
        else:
            logging.info("Perturbing C. Elegans data...")
        proc = CelegansProcessor()
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_CELEGANS)
        _save_to_json(data = data, path = celegans_path)
        logging.info(f"C. Elegans perturbations saved to {celegans_path}")
    else:
        logging.info(f"C. Elegans perturbations already exist at {celegans_path}. Skipping.")

    ## --- epilepsy --- ##
    epilepsy_path = os.path.join(PATH_PERT, f"{NAME_EPILEPSY}.json")
    if should_regenerate(epilepsy_path):
        if force and os.path.exists(epilepsy_path):
            logging.info(f"Overwriting existing Epilepsy perturbations at {epilepsy_path}")
        else:
            logging.info("Perturbing Epilepsy data...")
        ids = [f'chb{i:02d}' for i in range(1, 25)]
        proc = EpilepsyProcessor(url = URL_EPILEPSY, ids = ids)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_EPILEPSY)
        _save_to_json(data = data, path = epilepsy_path)
        logging.info(f"Epilepsy perturbations saved to {epilepsy_path}")
    else:
        logging.info(f"Epilepsy perturbations already exist at {epilepsy_path}. Skipping.")

    ## --- chickenpox --- ##
    chickenpox_path = os.path.join(PATH_PERT, f"{NAME_CHICKENPOX}.json")
    if should_regenerate(chickenpox_path):
        if force and os.path.exists(chickenpox_path):
            logging.info(f"Overwriting existing Chickenpox perturbations at {chickenpox_path}")
        else:
            logging.info("Perturbing Chickenpox data...")
        proc = ChickenpoxProcessor(
            url = URL_CHICKENPOX_EVENTS,
            name = "hungary_chickenpox.csv"
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_CHICKENPOX)
        _save_to_json(data = data, path = chickenpox_path)
        logging.info(f"Chickenpox perturbations saved to {chickenpox_path}")
    else:
        logging.info(f"Chickenpox perturbations already exist at {chickenpox_path}. Skipping.")

    ## --- gwosc --- ##
    gwosc_path = os.path.join(PATH_PERT, f"{NAME_GWOSC}.json")
    if should_regenerate(gwosc_path):
        if force and os.path.exists(gwosc_path):
            logging.info(f"Overwriting existing GWOSC perturbations at {gwosc_path}")
        else:
            logging.info("Perturbing GWOSC data...")
        proc = GwoscProcessor(url = URL_GWOSC)
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_GWOSC)
        _save_to_json(data = data, path = gwosc_path)
        logging.info(f"GWOSC perturbations saved to {gwosc_path}")
    else:
        logging.info(f"GWOSC perturbations already exist at {gwosc_path}. Skipping.")

    ## --- nwis --- ##
    river_path = os.path.join(PATH_PERT, f"{NAME_RIVER}.json")
    if should_regenerate(river_path):
        if force and os.path.exists(river_path):
            logging.info(f"Overwriting existing NWIS river perturbations at {river_path}")
        else:
            logging.info("Perturbing NWIS river data...")
        params = {
            "format": "rdb",
            "huc": "15010001,15010002,15010005",
            "siteType": "ST",
            "agencyCd": "USGS",
            "siteStatus": "all",
        }
        proc = NwisProcessor(
            url_site = URL_RIVER_SITE,
            url_iv = URL_RIVER_IV,
            params = params,
            start_date = "2014-01-01",
            end_date = "2024-12-31"
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_RIVER)
        _save_to_json(data = data, path = river_path)
        logging.info(f"NWIS river perturbations saved to {river_path}")
    else:
        logging.info(f"NWIS river perturbations already exist at {river_path}. Skipping.")

    ## --- auger --- ##
    auger_path = os.path.join(PATH_PERT, f"{NAME_AUGER}.json")
    if should_regenerate(auger_path):
        if force and os.path.exists(auger_path):
            logging.info(f"Overwriting existing Auger perturbations at {auger_path}")
        else:
            logging.info("Perturbing Auger data...")
        proc = AugerProcessor(
            url_network = URL_AUGER_NETWORK,
            url_events = URL_AUGER_EVENTS
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_AUGER)
        _save_to_json(data = data, path = auger_path)
        logging.info(f"Auger perturbations saved to {auger_path}")
    else:
        logging.info(f"Auger perturbations already exist at {auger_path}. Skipping.")

    ## --- seismic --- ##
    seismic_path = os.path.join(PATH_PERT, f"{NAME_SEISMIC}.json")
    if should_regenerate(seismic_path):
        if force and os.path.exists(seismic_path):
            logging.info(f"Overwriting existing Seismic perturbations at {seismic_path}")
        else:
            logging.info("Perturbing Seismic data...")

        ## define parameters for the seismic data
        params_network = {"level": "station", "format": "xml", "network": "IU"}
        namespace = {"ns": "http://www.fdsn.org/xml/station/1"}
        row_path = ".//ns:Station"
        col_map = {
            "code": ".@code",
            "lat": ".//ns:Latitude",
            "lon": ".//ns:Longitude"
        }
        params_events = {
            "starttime": "2023-01-01",
            "endtime": "2023-12-31"
        }

        proc = SeismicProcessor(
            url_network = URL_SEISMIC_NETWORK,
            params_network = params_network,
            namespace = namespace,
            row_path = row_path,
            col_map = col_map,
            url_events = URL_SEISMIC_EVENTS,
            params_events = params_events
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_SEISMIC)
        _save_to_json(data = data, path = seismic_path)
        logging.info(f"Seismic perturbations saved to {seismic_path}")
    else:
        logging.info(f"Seismic perturbations already exist at {seismic_path}. Skipping.")

    ## --- rain --- ##
    rain_path = os.path.join(PATH_PERT, f"{NAME_RAIN}.json")
    if should_regenerate(rain_path):
        if force and os.path.exists(rain_path):
            logging.info(f"Overwriting existing Rain perturbations at {rain_path}")
        else:
            logging.info("Perturbing Rain data...")
        proc = RainProcessor(
            country = "LA",  ## iso country code for laos
            start_date = "2024-01-01",
            end_date = "2024-03-31"
        )
        proc.run()
        data = _execute_perturbations(proc = proc, name = NAME_RAIN)
        _save_to_json(data = data, path = rain_path)
        logging.info(f"Rain perturbations saved to {rain_path}")
    else:
        logging.info(f"Rain perturbations already exist at {rain_path}. Skipping.")

    ## --- amazon reviews --- ##
    amazon_path = os.path.join(PATH_PERT, f"{NAME_AMAZON}.json")
    if should_regenerate(amazon_path):
        if force and os.path.exists(amazon_path):
            logging.info(f"Overwriting existing Amazon perturbations at {amazon_path}")
        else:
            logging.info("Perturbing Amazon data...")
        proc = AmazonProcessor(root_path = PATH_ROOT, url = URL_AMAZON, name = NAME_AMAZON)
        proc.run()

        ## note: 10m-node graph, a densify worker peaks ~12 gb private memory, cap workers for local ram
        data = _execute_perturbations(proc = proc, name = NAME_AMAZON, n_jobs = 6)

        _save_to_json(data = data, path = amazon_path)
        logging.info(f"Amazon perturbations saved to {amazon_path}")
    else:
        logging.info(f"Amazon perturbations already exist at {amazon_path}. Skipping.")

## primary execution
if __name__ == '__main__':
    json_perturber(force = True)
