## libraries
import dcor
import warnings
import numpy as np
import pandas as pd
from typing import Literal, Sequence
from scipy.stats import ConstantInputWarning, rankdata, spearmanr

## violation rate
def _violation_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ VR fraction of points that violate the frontier. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## violation rate is fraction with positive violation
    return float(np.mean(v > 0.0))

## mean violation magnitude
def _mean_violation(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ MV mean size of violations conditional on violating. """

    ## compute violation: true above frontier
    v = np.maximum(0.0, y_true - y_pred)
    
    ## restrict to positive violations only
    mask = v > 0.0
    if not np.any(mask):
        return 0.0
    return float(v[mask].mean())

## mean slack
def _mean_slack(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    """ MS average slack (how far below the frontier the data lie). """

    ## compute slack: frontier above true
    s = np.maximum(0.0, y_pred - y_true)
    return float(s.mean())

## efficiency index
def _efficiency_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
    ) -> float:

    """ EI efficiency index that combines violation rate, mean violation,
    and mean slack via geometric mean. MV and MS are normalized by the
    mean absolute target magnitude, making the score relative to the
    scale of the evaluated sample. Higher is better. """

    ## reuse helper metrics
    vr = _violation_rate(y_true = y_true, y_pred = y_pred)
    mv = _mean_violation(y_true = y_true, y_pred = y_pred)
    ms = _mean_slack(y_true = y_true, y_pred = y_pred)

    ## normalize violation and slack magnitudes by sample target magnitude
    mean_target = np.mean(np.abs(y_true)) + eps
    mv_norm = mv / mean_target
    ms_norm = ms / mean_target

    ## transform to bounded scores in (0, 1]
    vr_score = np.clip(1.0 - vr, eps, 1.0)
    mv_score = 1.0 / (1.0 + mv_norm)
    ms_score = 1.0 / (1.0 + ms_norm)

    ## geometric mean via log-space
    log_ei = (np.log(vr_score) + np.log(mv_score) + np.log(ms_score)) / 3.0
    return float(np.exp(log_ei))

## joint frontier metrics
def frontier_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> dict:

    """ Compute all frontier metrics and return as a dictionary. """
    
    metrics = {
        key: func(y_true, y_pred)
        for key, func in [
            ("vr", _violation_rate),
            ("mv", _mean_violation),
            ("ms", _mean_slack),
        ]
    }
    metrics["ei"] = _efficiency_index(
        y_true = y_true,
        y_pred = y_pred,
        eps = eps
    )
    return metrics

## spearman rank correlation
def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Global monotone agreement of predicted capacities. """

    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    if y_true.size < 2 or y_pred.size < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = ConstantInputWarning)
        rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0

## distance correlation
def _distance_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """ Non-linear dependence measured by distance correlation. """

    try:
        return float(dcor.distance_correlation(y_true, y_pred))
    except Exception:
        return 0.0

## rank-biased overlap
def _rank_biased_overlap(y_true: np.ndarray, y_pred: np.ndarray, p: float) -> float:

    """ Frontier-focused agreement; emphasizes the top of the ranking. """

    if not 0.0 <= p < 1.0:
        raise ValueError("p must satisfy 0 <= p < 1.")
    
    ## stable descending sort
    ## ties retain ascending index order so the statistic is deterministic
    s = np.argsort(-np.asarray(y_true, dtype = float), kind = "stable")
    t = np.argsort(-np.asarray(y_pred, dtype = float), kind = "stable")
    n = len(s)
    if n == 0:
        return 0.0
    
    score = 0.0
    weight = 1.0
    overlap = 0
    last_agreement = 0.0
    seen_s = set()
    seen_t = set()
    
    for d in range(1, n + 1):
        idx_s = s[d-1]
        idx_t = t[d-1]
        
        if idx_s == idx_t:
            overlap += 1
        else:
            if idx_s in seen_t:
                overlap += 1
            if idx_t in seen_s:
                overlap += 1
        
        seen_s.add(idx_s)
        seen_t.add(idx_t)
        
        agreement = overlap / d
        score += weight * agreement
        last_agreement = agreement
        weight *= p
        
        if weight < 1e-6:
            break

    ## extrapolate the unobserved tail from the last evaluated agreement level
    tail = weight * last_agreement
    return float((1.0 - p) * score + tail)

## consensus index
def consensus_index(
    rho: float,
    rbo: float,
    dcr: float,
    ) -> float:

    """ CI geometric mean over comparable agreement-scale metrics.
    Components are clipped to their theoretical [0, 1] bounds to correct
    floating-point error; exact zeros are retained so CI = 0 when any
    factor is zero. """

    rho_agree = (rho + 1.0) / 2.0
    ci_vals = np.clip(
        np.array([rho_agree, rbo, dcr], dtype = float),
        a_min = 0.0,
        a_max = 1.0,
    )
    return float(np.prod(ci_vals) ** (1.0 / 3.0))


def _consensus_index(
    rho: float,
    rbo: float,
    dcr: float,
    ) -> float:

    """ CI geometric mean over comparable agreement-scale metrics. """

    return consensus_index(
        rho = rho,
        rbo = rbo,
        dcr = dcr,
    )

## joint consensus metrics
def consensus_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.9) -> dict:

    """ Compute all consensus metrics and return as a dictionary. """

    rho = _spearman_rho(y_true, y_pred)
    rbo = _rank_biased_overlap(y_true, y_pred, p = p)
    dcr = _distance_corr(y_true, y_pred)

    metrics = {
        "rho": rho,
        "rbo": rbo,
        "dcr": dcr,
    }
    metrics["ci"] = _consensus_index(
        rho = metrics["rho"],
        rbo = metrics["rbo"],
        dcr = metrics["dcr"],
    )
    return metrics

## empirical margin of equivalent variability
def spec_marginal_delta(
    results: pd.DataFrame,
    feat_value: Sequence[str],
    track: str | Sequence[str] | None = None,
    label_ref: str | None = None,
    value_ref: str | None = None,
    label_pert: str | None = None,
    label_base: str = "baseline",
    method: Literal["mad", "iqr", "max"] = "iqr",
    scale: float = 1.0,  ## full iqr
    decimals: int = 2,
    ) -> float:

    """
    Desc:
        Compute a data-driven margin from the natural variability of reference
        metric values. Delta is anchored entirely to the reference condition and
        is independent of any test contrast.

    Args:
        results: Results table containing reference rows and metric columns.
        feat_value: Metric columns used to derive the margin.
        track: Optional track value or values to restrict before estimating.
        label_ref: Column identifying the reference rows. Defaults to
            label_pert for perturbation compatibility.
        value_ref: Value identifying the reference rows. Defaults to
            label_base for perturbation compatibility.
        label_pert: Perturbation label column used by the default reference.
        label_base: Baseline value used by the default reference.
        method: Dispersion estimator ("mad", "iqr", or "max").
        scale: Multiplier applied to the dispersion estimate.
        decimals: Number of decimal places to floor the resulting margin.

    Returns:
        Scalar empirical margin delta, floored to `decimals` places.

    Raises:
        ValueError: If reference labels are unspecified, required columns are
            missing, fewer than two finite reference values exist, or method is
            unknown.
    """

    data = results.copy()
    feat_value = list(feat_value)
    label_ref = label_ref or label_pert
    value_ref = label_base if value_ref is None else value_ref

    if label_ref is None:
        raise ValueError("label_ref or label_pert must be specified to derive delta")

    required = {label_ref, *feat_value}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if track is not None and "track" in data.columns:
        track_vals = [track] if isinstance(track, str) else list(track)
        data = data.loc[data["track"].isin(track_vals)]

    reference = data.loc[data[label_ref] == value_ref]
    vals = np.concatenate([
        reference[metric].to_numpy(dtype = float) for metric in feat_value
    ])
    vals = vals[np.isfinite(vals)]

    if len(vals) < 2:
        raise ValueError(
            "At least two finite reference values are required to derive delta"
        )

    if method == "mad":
        dispersion = float(np.median(np.abs(vals - np.median(vals))))
    elif method == "iqr":
        dispersion = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    elif method == "max":
        dispersion = float(np.max(vals) - np.min(vals))
    else:
        raise ValueError(f"unknown method: {method}")

    ## floor the scaled dispersion to the specified number of decimal places
    scaled = max(float(scale * dispersion), 1e-6)
    factor = 10 ** int(decimals)
    return float(np.floor(scaled * factor + 1e-12) / factor)

## paired rank-biserial correlation
def paired_rank_biserial(differences: Sequence[float]) -> float:

    """
    Desc:
        Compute rank-biserial correlation from nonzero paired differences.
    Args:
        differences: Raw paired differences oriented as test minus reference.
    Returns:
        Rank-biserial correlation, or NaN when no nonzero differences exist.
    """

    values = np.asarray(differences, dtype = float)
    values = values[np.isfinite(values) & (values != 0.0)]
    if len(values) == 0:
        return np.nan

    ranks = rankdata(np.abs(values), method = "average")
    positive = float(np.sum(ranks[values > 0.0]))
    negative = float(np.sum(ranks[values < 0.0]))
    return (positive - negative) / float(np.sum(ranks))
