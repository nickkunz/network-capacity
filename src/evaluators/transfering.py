## libraries
import numpy as np
import pandas as pd
from typing import Mapping, Sequence, cast
from scipy.stats import wilcoxon
from pandas.io.formats.style import Styler

## modules
from src.vectorizers.scalers import _log_transformer
from src.evaluators.metrics import (
    consensus_metrics,
    frontier_metrics,
    paired_rank_biserial,
    spec_marginal_delta,
)

## constants
from src.evaluators.config import (
    FRONTIER_METRICS,
    CONSENSUS_METRICS
)

## ----------------------------------------------------------------------------
## domain-aligned resampling results
## ----------------------------------------------------------------------------
def compile_transfer_resampling(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    data: pd.DataFrame,
    target: str = "target",
    group: str = "domain",
    ) -> pd.DataFrame:

    """
    Desc:
        Score each resampling regime on common held-out-group units using its
        averaged out-of-fold predictions.

    Args:
        predictions: Mapping from regime labels to model prediction mappings.
            Each prediction vector must align row-wise with data.
        data: Evaluation data containing target and group columns.
        target: Target column name.
        group: Column defining the common evaluation units.

    Returns:
        DataFrame with one row per regime, model, and group containing frontier
        and prediction-consensus metrics.

    Raises:
        ValueError: If required columns are missing or a prediction vector is
            not aligned with data.
    """

    required = {target, group}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    y_true = _log_transformer(data[target]).astype(float).to_numpy()
    group_values = data[group].to_numpy()
    group_names = sorted(pd.Series(data = group_values).dropna().unique())

    rows = list()
    for regime, regime_predictions in predictions.items():
        for model, prediction in regime_predictions.items():
            y_pred = np.asarray(a = prediction, dtype = float)
            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"Predictions for {model!r} under {regime!r} have shape "
                    f"{y_pred.shape}, expected {y_true.shape}"
                )

            for group_name in group_names:
                valid = (
                    (group_values == group_name)
                    & np.isfinite(y_true)
                    & np.isfinite(y_pred)
                )
                if int(np.sum(a = valid)) < 2:
                    continue

                rows.append({
                    "regime": regime,
                    "model": model,
                    "group": group_name,
                    **frontier_metrics(
                        y_true = y_true[valid],
                        y_pred = y_pred[valid],
                    ),
                    **consensus_metrics(
                        y_true = y_true[valid],
                        y_pred = y_pred[valid],
                    ),
                })

    columns = [
        "regime",
        "model",
        "group",
        *FRONTIER_METRICS,
        *CONSENSUS_METRICS,
    ]
    if not rows:
        return pd.DataFrame(columns = columns)
    return pd.DataFrame(data = rows).reindex(columns = columns)


## ----------------------------------------------------------------------------
## transfer equivalence margin
## ----------------------------------------------------------------------------
def spec_transfer_delta(
    results: pd.DataFrame,
    metric: str = "ei",
    reference: str = "Domain (LOGO)",
    label_regime: str = "regime",
    model_col: str = "model",
    scale: float = 1.0,
    decimals: int = 2,
    ) -> float:

    """
    Desc:
        Derive an empirical equivalence margin from the IQR of equally
        weighted model-level means under the reference resampling regime.

    Args:
        results: Domain-aligned output from compile_transfer_resampling.
        metric: Metric column used to derive the margin.
        reference: Reference regime label.
        label_regime: Column containing resampling regime labels.
        model_col: Column identifying fitted models.
        scale: Multiplier applied to the reference IQR.
        decimals: Decimal places used to ceil the margin.

    Returns:
        Empirical equivalence margin on the metric scale.

    Raises:
        ValueError: If required columns or reference rows are absent.
    """

    required = {label_regime, model_col, metric}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    reference_rows = results.loc[results[label_regime] == reference]
    if reference_rows.empty:
        raise ValueError(f"No rows found for regime {reference!r}")

    model_results = (
        reference_rows
        .groupby(by = model_col, observed = True, as_index = False)[metric]
        .mean()
        .assign(**{label_regime: reference})
    )
    return spec_marginal_delta(
        results = model_results,
        feat_value = [metric],
        label_ref = label_regime,
        value_ref = reference,
        method = "iqr",
        scale = scale,
        decimals = decimals,
    )


## ----------------------------------------------------------------------------
## resampling equivalence test
## ----------------------------------------------------------------------------
def stat_transfer_tost(
    results: pd.DataFrame,
    metric: str = "ei",
    delta: float = 0.05,
    reference: str = "Domain (LOGO)",
    comparisons: Sequence[str] | None = None,
    feat_pairs: Sequence[str] = ("model", "group"),
    label_regime: str = "regime",
    decimals: int = 4,
    index: bool = True,
    ) -> pd.DataFrame:

    """
    Desc:
        Test equivalence between a reference resampling regime and comparison
        regimes with paired Wilcoxon two one-sided tests (TOST).

    Args:
        results: Domain-aligned output from compile_transfer_resampling.
        metric: Metric column to test.
        delta: Symmetric equivalence margin on the metric scale.
        reference: Reference regime label.
        comparisons: Ordered comparison regime labels. None uses first
            appearance order after excluding the reference.
        feat_pairs: Columns identifying matched evaluation units.
        label_regime: Column containing resampling regime labels.
        decimals: Display precision for paired differences and effect sizes.
        index: Whether to index the result by regime.

    Returns:
        Display-ready table with paired median shifts, rank-biserial effects,
        TOST p-values, Holm-adjusted p-values, and equivalence decisions.

    Raises:
        ValueError: If inputs are invalid, regimes are absent, or pairing keys
            are duplicated within a tested regime.
    """

    if delta <= 0.0:
        raise ValueError("delta must be greater than zero")

    pair_cols = list(feat_pairs)
    required = {label_regime, metric, *pair_cols}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    comparison_labels = (
        list(comparisons)
        if comparisons is not None
        else results.loc[
            results[label_regime] != reference,
            label_regime,
        ].drop_duplicates().tolist()
    )
    if not comparison_labels:
        raise ValueError("At least one comparison regime is required")
    if reference in comparison_labels:
        raise ValueError("The reference regime cannot also be a comparison")

    tested_labels = [reference, *comparison_labels]
    for regime in tested_labels:
        regime_rows = results.loc[results[label_regime] == regime]
        if regime_rows.empty:
            raise ValueError(f"No rows found for regime {regime!r}")
        if regime_rows.duplicated(subset = pair_cols).any():
            raise ValueError(
                f"Pairing columns {pair_cols} are duplicated within regime "
                f"{regime!r}"
            )

    reference_data = (
        results.loc[results[label_regime] == reference, [*pair_cols, metric]]
        .rename(columns = {metric: "reference_value"})
    )

    rows = list()
    pair_counts = list()
    for regime in comparison_labels:
        comparison_data = results.loc[
            results[label_regime] == regime,
            [*pair_cols, metric],
        ].copy()
        comparison_data["comparison_value"] = comparison_data.pop(metric)
        paired = reference_data.merge(
            right = comparison_data,
            on = pair_cols,
            how = "inner",
            validate = "one_to_one",
        )
        reference_values = paired["reference_value"].to_numpy(dtype = float)
        comparison_values = paired["comparison_value"].to_numpy(dtype = float)
        valid = np.isfinite(reference_values) & np.isfinite(comparison_values)
        differences = comparison_values[valid] - reference_values[valid]
        n_pairs = len(differences)
        pair_counts.append(n_pairs)

        if n_pairs < 2:
            p_tost = np.nan
        else:
            upper_shift = differences - delta
            lower_shift = differences + delta
            if np.all(upper_shift == 0.0):
                p_upper = 1.0
            else:
                p_upper = cast(
                    float,
                    wilcoxon(x = upper_shift, alternative = "less")[1],
                )
            if np.all(lower_shift == 0.0):
                p_lower = 1.0
            else:
                p_lower = cast(
                    float,
                    wilcoxon(x = lower_shift, alternative = "greater")[1],
                )
            p_tost = max(float(p_upper), float(p_lower))

        rows.append({
            "Regime": regime,
            f"Median Δ {metric.upper()}": (
                float(np.median(differences)) if n_pairs else np.nan
            ),
            "Rank-biserial r": paired_rank_biserial(
                differences = differences.tolist(),
            ),
            "TOST p": p_tost,
        })

    summary = pd.DataFrame(data = rows)
    p_values = summary["TOST p"].to_numpy(dtype = float, copy = True)
    valid_p = np.isfinite(p_values)
    holm = np.full(shape = len(p_values), fill_value = np.nan, dtype = float)
    if np.any(valid_p):
        p_valid = p_values[valid_p]
        n_tests = len(p_valid)
        order = np.argsort(p_valid)
        holm_sorted = np.maximum.accumulate(
            p_valid[order] * (n_tests - np.arange(n_tests))
        )
        holm_valid = np.empty(shape = n_tests, dtype = float)
        holm_valid[order] = np.minimum(holm_sorted, 1.0)
        holm[valid_p] = holm_valid

    summary["Holm-adj. p"] = holm
    summary["Sig."] = summary["Holm-adj. p"].map(
        lambda p: (
            "-" if not np.isfinite(p)
            else "***" if p < 0.001
            else "**" if p < 0.01
            else "*" if p < 0.05
            else ""
        )
    )
    summary["Eq."] = summary["Holm-adj. p"].map(
        lambda p: "-" if not np.isfinite(p) else "Yes" if p < 0.05 else "No"
    )

    unique_counts = sorted(set(pair_counts))
    n_display = unique_counts[0] if len(unique_counts) == 1 else "varies"
    metric_label = metric.upper()
    print(
        f"Paired TOST (Wilcoxon Signed-Rank): n = {n_display}, "
        f"δ = {delta}, metric = {metric_label}"
    )
    print(f"H₀: |Δ {metric_label}| ≥ δ")
    print(f"H₁: |Δ {metric_label}| < δ")
    print(
        f"Median Δ {metric_label}: Median of paired differences "
        f"(comparison - {reference})"
    )
    print(
        "Rank-biserial r: Raw paired effect size. Positive values indicate "
        f"comparison > {reference}"
    )
    print("TOST p: max(Upper p, Lower p)")
    print("Holm-adj. p: Holm-Bonferroni adjusted across regime comparisons")
    print("Significance codes reflect Holm-adj. p")
    print("*** p < 0.001, ** p < 0.01, * p < 0.05")

    numeric_cols = [f"Median Δ {metric_label}", "Rank-biserial r"]
    for column in numeric_cols:
        formatted = list()
        for value in summary[column].to_numpy(dtype = float):
            if not np.isfinite(value):
                formatted.append(value)
                continue
            rounded = round(float(value), decimals)
            rounded = 0.0 if rounded == 0.0 else rounded
            formatted.append(f"{rounded:.{decimals}f}")
        summary[column] = formatted
    for column in ["TOST p", "Holm-adj. p"]:
        summary[column] = summary[column].apply(
            lambda value: "-" if not (pd.notna(value) and np.isfinite(float(value))) else "<0.001" if float(value) < 0.001 else f"{float(value):.3f}"
        )

    if index:
        summary = summary.set_index("Regime")
    return summary.astype(object).where(pd.notna(summary), "-")

## ----------------------------------------------------------------------------
## domain transfer result helper
## ----------------------------------------------------------------------------
def compile_domain_transfer(results: dict) -> pd.DataFrame:

    """
    Desc:
        Converts a dictionary of domain-transfer frontiers into a single
        formatted DataFrame, moving the "model" column to the front.

    Args:
        results: Dict mapping model names to their frontier dataframes.

    Returns:
        A concatenated DataFrame with the 'model' column moved to index 0.
    """

    frame = pd.concat(results.values(), ignore_index = True)
    feat = ["model"] + [c for c in frame.columns if c != "model"]
    return frame[feat]


## ----------------------------------------------------------------------------
## domain transfer results
## ----------------------------------------------------------------------------
def results_domain_transfer(
    *results: dict[str, pd.DataFrame],
    keys: Sequence[str] | None = None,
    indicies: tuple[str, str] | None = None,
    n_repeats: int = 30,
    random_state: int = 42,
    decimals: int = 3
    ) -> pd.DataFrame | Styler:

    """
    Desc:
        Compiles one or more dictionaries of domain-transfer results into a
        single DataFrame. With a single dict, concatenates frontiers and moves
        "model" to the front. With multiple dicts and keys, groups each dict
        by model, computes means, and builds a multi-index summary table.

    Args:
        *results: One or more dicts mapping names to DataFrames.
        keys: Optional list of group labels for multi-index rows.
        indicies: Names for the multi-index levels. Defaults to
            ("procedure", "method").
        n_repeats: Optional number of repeats used in CV, for printing only.
        random_state: Optional base random seed used in CV, for printing only.
        decimals: If set, returns a styled table with left-justified index and
            the given decimal precision. If None, returns a plain DataFrame.

    Returns:
        A DataFrame or Styler of compiled domain-transfer results.
    """

    ## single dict: concatenate and reorder columns
    if len(results) == 1 and keys is None:
        frame = pd.concat(results[0].values(), ignore_index = True)
        feat = ["model"] + [c for c in frame.columns if c != "model"]
        result = frame[feat]

    ## multiple dicts with keys: grouped summary table
    else:
        if keys is None:
            keys = [f"Group {i}" for i in range(len(results))]

        ## print summary header
        first_data = next(iter(results[0].values()))
        n_models = first_data["model"].nunique() if "model" in first_data.columns else None
        if n_repeats is None and "repeat" in first_data.columns:
            n_repeats = int(first_data["repeat"].nunique())
        if n_repeats is not None and random_state is not None:
            print(
                f"Cross-Validation: {n_models} models, {n_repeats} repeats "
                f"(seeds {random_state}-{random_state + n_repeats - 1})"
            )
        else:
            print(f"Cross-Validation: {n_models} models")
        print("Across-model aggregation: median of model-level means")
        print(
            "Resampling: LOGO splits are fixed across repeats; "
            "random k-fold splits are reshuffled across repeats"
        )
        print(
            "Weighting: groups, folds, repeats, and models are equally weighted; "
            "results are not observation-weighted"
        )
        blocks = []
        for key, table in zip(keys, results):
            for label, data in table.items():
                if hasattr(data, "data"):
                    data = data.data
                numeric = data.select_dtypes(include = "number").drop(
                    columns = ["iteration", "repeat", "fold", "n_folds_used"],
                    errors = "ignore",
                )
                if "model" in data.columns:
                    model_means = numeric.groupby(data["model"]).mean()
                    summary = model_means.median()
                    ei_q1 = model_means["ei"].quantile(0.25)
                    ei_q3 = model_means["ei"].quantile(0.75)
                else:
                    summary = numeric.median()
                    ei_q1 = numeric["ei"].quantile(0.25)
                    ei_q3 = numeric["ei"].quantile(0.75)

                ## format ei with iqr
                summary = summary.astype(object)
                if decimals is not None:
                    summary["ei"] = f"{summary['ei']:.{decimals}f} [{ei_q1:.{decimals}f}-{ei_q3:.{decimals}f}]"
                else:
                    summary["ei"] = f"{summary['ei']} [{ei_q1}-{ei_q3}]"
                summary.name = (key, label)
                blocks.append(summary)
        result = pd.DataFrame(blocks)
        result.index = pd.MultiIndex.from_tuples(result.index, names = indicies)

        ## reorder columns with ei first
        cols = result.columns.tolist()
        cols.remove("ei")
        result = result[["ei"] + cols]

        ## capitalize core metrics
        rename_map = {
            c: "EI [IQR]" if c == "ei" else c.upper()
            for c in FRONTIER_METRICS
            if c in result.columns
        }
        result = result.rename(columns = rename_map)

    ## optionally return styled output
    if decimals is not None:
        return result.style.set_table_styles([
            {
                "selector": "th.row_heading, th.index_name",
                "props": [("text-align", "left"), ("vertical-align", "top")],
            }
        ]).format(precision = decimals)
    return result