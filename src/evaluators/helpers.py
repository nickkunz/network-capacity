## libraries
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Literal, Sequence
from scipy.stats import rankdata

## set floating-point round-off differences to exact zero before rank-based tests
def _clean_differences(diff: ArrayLike, tol: float = 1e-10) -> np.ndarray:
    values = np.array(diff, dtype = float)
    values[np.abs(values) < tol] = 0.0
    return values

## paired rank-biserial correlation
def paired_rank_biserial(diff: Sequence[float]) -> float:

    """
    Desc:
        Compute rank-biserial correlation from nonzero paired differences.

    Args:
        diff: Raw paired differences oriented as test minus reference.
        Only nonzero, finite differences are considered.

    Returns:
        float: Rank-biserial correlation, or NaN when no nonzero differences exist.

    Raises:
        None.
    """
    values = np.asarray(diff, dtype = float)
    values = values[np.isfinite(values) & (values != 0.0)]
    if len(values) == 0:
        return np.nan

    ranks = rankdata(np.abs(values), method = "average")
    positive = float(np.sum(ranks[values > 0.0]))
    negative = float(np.sum(ranks[values < 0.0]))
    return (positive - negative) / float(np.sum(ranks))

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
