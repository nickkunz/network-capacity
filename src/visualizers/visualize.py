## libraries
import pickle
import logging
import colorsys
import subprocess
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Rectangle
from matplotlib.text import Text

## modules
from src.evaluators.helpers import _spec_marginal_delta
from src.evaluators.metrics import _efficiency_index, frontier_consensus
from src.evaluators.perturbing import find_perturbed_max
from src.evaluators.predicting import compile_corpus_full, compile_prediction_consensus
from src.evaluators.resampling import kfold_cross_valid, logo_cross_valid
from src.evaluators.transfering import compile_domain_transfer
from src.vectorizers.scalers import _log_transformer
from src.visualizers.illustrate import build_capacity_frontier_illustration

## constants
DEFAULT_PARADIGM_ORDER = (
    "linear parametric",
    "non-linear ensemble",
    "neural networks",
)
DEFAULT_MODEL_TO_PARADIGM = {
    "linear_quantile": "linear parametric",
    "linear_convex": "linear parametric",
    "linear_laws": "linear parametric",
    "forest_quantile": "non-linear ensemble",
    "boosted_quantile": "non-linear ensemble",
    "xgboost_quantile": "non-linear ensemble",
    "neural_quantile": "neural networks",
    "neural_expectile": "neural networks",
    "neural_convex": "neural networks",
}
DEFAULT_PARADIGM_PALETTE = {
    "linear parametric": "#3B6E8F",
    "non-linear ensemble": "#7A5A9A",
    "neural networks": "#A3623A",
}
DEFAULT_PARADIGM_MARKERS = {
    "linear parametric": "o",
    "non-linear ensemble": "o",
    "neural networks": "o",
}
DEFAULT_PARADIGM_LABELS = {
    "linear parametric": "LP",
    "non-linear ensemble": "NE",
    "neural networks": "NN",
}
DEFAULT_CONSENSUS_PANELS = {
    "CI": "ci",
    "ρ": "rho",
    "RBO": "rbo",
    "DCR": "dcr",
}
DEFAULT_PERTURBATION_ORDER = (
    "network",
    "invariants",
    "process",
    "signatures",
)
DEFAULT_PERTURBATION_PALETTE = {
    "network": "#2C6E91",
    "invariants": "#2C6E91",
    "process": "#2C6E91",
    "signatures": "#2C6E91",
}
DEFAULT_PERTURBATION_TITLE_COLORS = {
    "network": "#2C6E91",
    "invariants": "#2C6E91",
    "process": "#2C6E91",
    "signatures": "#2C6E91",
}
DEFAULT_METHOD_LINESTYLES = ("-", "--", ":")
DEFAULT_METHOD_SHADE_FACTORS = (0.65, 1.0, 1.22)
UNIVERSALITY_REFERENCE_LINE_WIDTH = 0.75


def build_paradigm_consensus_matrices(
    results: pd.DataFrame,
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    metric_panels: Mapping[str, str] | None = None,
    model_i: str = "model_i",
    model_j: str = "model_j",
    ) -> dict[str, pd.DataFrame]:

    """
    Desc:
        Build paradigm-by-paradigm median consensus matrices from pairwise
        learner comparison results.

    Args:
        results: Pairwise comparison table with model identifier columns and
            metric columns.
        model_to_paradigm: Mapping from model names to paradigm labels.
        paradigm_order: Row and column ordering for the output matrices.
        metric_panels: Mapping from display labels to metric column names.
        model_i: Column name for the first model in each pair.
        model_j: Column name for the second model in each pair.

    Returns:
        Dictionary mapping metric display labels to paradigm-by-paradigm
        median consensus matrices.

    Raises:
        ValueError: If required columns are missing, no metrics are provided,
            or a model lacks a paradigm label.
    """

    if model_to_paradigm is None:
        model_to_paradigm = DEFAULT_MODEL_TO_PARADIGM
    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    if metric_panels is None:
        metric_panels = DEFAULT_CONSENSUS_PANELS

    paradigm_order = list(paradigm_order)
    metric_panels = dict(metric_panels)
    if not metric_panels:
        raise ValueError("metric_panels must contain at least one metric")

    required_columns = {model_i, model_j, *metric_panels.values()}
    missing_columns = sorted(required_columns - set(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    pairs = results.copy()
    pairs["_paradigm_i"] = pairs[model_i].map(model_to_paradigm)
    pairs["_paradigm_j"] = pairs[model_j].map(model_to_paradigm)

    missing_models = sorted(
        set(pairs.loc[pairs["_paradigm_i"].isna(), model_i])
        | set(pairs.loc[pairs["_paradigm_j"].isna(), model_j])
    )
    if missing_models:
        raise ValueError(f"Missing paradigm labels for models: {missing_models}")

    matrices = dict()
    for panel_label, metric in metric_panels.items():
        matrix = pd.DataFrame(
            data = np.nan,
            index = paradigm_order,
            columns = paradigm_order,
        )
        for row_label in paradigm_order:
            for column_label in paradigm_order:
                mask = (
                    (
                        (pairs["_paradigm_i"] == row_label)
                        & (pairs["_paradigm_j"] == column_label)
                    )
                    | (
                        (pairs["_paradigm_i"] == column_label)
                        & (pairs["_paradigm_j"] == row_label)
                    )
                )
                if mask.any():
                    matrix.loc[row_label, column_label] = pairs.loc[mask, metric].median()
        matrices[panel_label] = matrix

    return matrices


def plot_consensus(
    results: pd.DataFrame,
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    metric_panels: Mapping[str, str] | None = None,
    model_i: str = "model_i",
    model_j: str = "model_j",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "RdYlGn",
    decimals: int = 2,
    figsize: tuple[float, float] = (10.5, 3.8),
    title: str = "Learning consensus by estimator paradigm",
    colorbar_label: str = "Median pairwise consensus",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray, dict[str, pd.DataFrame]]:

    """
    Desc:
        Plot one or more paradigm-by-paradigm consensus heatmaps with shared
        color scale and numeric cell annotations.

    Args:
        results: Pairwise comparison table with model identifier columns and
            metric columns.
        model_to_paradigm: Mapping from model names to paradigm labels.
        paradigm_order: Row and column ordering for the output matrices.
        metric_panels: Mapping from panel titles to metric column names.
        model_i: Column name for the first model in each pair.
        model_j: Column name for the second model in each pair.
        vmin: Lower bound for the shared color scale.
        vmax: Upper bound for the shared color scale.
        cmap_name: Matplotlib colormap name.
        decimals: Number of decimals shown in cell annotations.
        figsize: Figure size in inches.
        title: Figure title.
        colorbar_label: Label for the shared colorbar.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure, axes array, and computed matrices.
    """

    matrices = build_paradigm_consensus_matrices(
        results = results,
        model_to_paradigm = model_to_paradigm,
        paradigm_order = paradigm_order,
        metric_panels = metric_panels,
        model_i = model_i,
        model_j = model_j,
    )

    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    paradigm_order = list(paradigm_order)
    norm = Normalize(vmin = vmin, vmax = vmax)
    cmap = plt.get_cmap(cmap_name)

    def text_color(value: float) -> str:
        rgba = cmap(norm(value))
        luminance = rgba[0] * 0.299 + rgba[1] * 0.587 + rgba[2] * 0.114
        return "white" if luminance < 0.55 else "black"

    fig, axes = plt.subplots(
        nrows = 1,
        ncols = len(matrices),
        figsize = figsize,
        constrained_layout = True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    image = None

    for panel_index, (axis, (panel_label, matrix)) in enumerate(zip(axes_array, matrices.items())):
        image = axis.imshow(
            matrix.to_numpy(dtype = float),
            cmap = cmap,
            norm = norm,
            aspect = "equal",
        )
        for row_index, row_label in enumerate(paradigm_order):
            for column_index, column_label in enumerate(paradigm_order):
                value = matrix.loc[row_label, column_label]
                if pd.isna(value):
                    axis.text(
                        column_index, row_index, "NA",
                        ha = "center",
                        va = "center",
                        color = "black",
                        fontsize = 9,
                    )
                else:
                    axis.text(
                        column_index, row_index, f"{value:.{decimals}f}",
                        ha = "center",
                        va = "center",
                        color = text_color(float(value)),
                        fontsize = 9,
                    )
        axis.set_title(panel_label, fontsize = 11)
        axis.set_xticks(range(len(paradigm_order)))
        axis.set_yticks(range(len(paradigm_order)))
        axis.set_xticklabels(
            paradigm_order,
            rotation = 35,
            ha = "right",
            fontsize = 9,
        )
        if panel_index == 0:
            axis.set_yticklabels(paradigm_order, fontsize = 9)
        else:
            axis.set_yticklabels([])
        axis.tick_params(axis = "both", which = "both", length = 0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax = list(axes_array),
            shrink = 0.82,
            pad = 0.02,
        )
        colorbar.set_label(colorbar_label, fontsize = 9)

    fig.suptitle(title, fontsize = 12)
    if show:
        plt.show()
    return fig, axes_array, matrices


def _filter_plot_track(
    results: pd.DataFrame,
    track: str | Sequence[str] | None,
    ) -> pd.DataFrame:

    if track is None or "track" not in results.columns:
        return results.copy()

    track_values = [track] if isinstance(track, str) else list(track)
    return results.loc[results["track"].isin(track_values)].copy()


def _resolve_plot_unit_column(
    results_transfer: pd.DataFrame,
    results_recovery: pd.DataFrame,
    unit_col: str | None,
    ) -> str:

    if unit_col is not None:
        if unit_col not in results_transfer.columns or unit_col not in results_recovery.columns:
            raise ValueError(f"Unit column '{unit_col}' must be present in both result tables")
        return unit_col

    for candidate in ("group", "domain"):
        if candidate in results_transfer.columns and candidate in results_recovery.columns:
            return candidate

    raise ValueError("Could not infer a shared unit column; expected 'group' or 'domain'")


def _sorted_plot_levels(values: pd.Series) -> list[object]:

    levels = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(levels, key = lambda value: float(value))
    except (TypeError, ValueError):
        return sorted(levels, key = lambda value: str(value))


def _normalized_plot_positions(levels: Sequence[object]) -> dict[object, float]:

    if len(levels) == 0:
        return dict()

    numeric = pd.to_numeric(pd.Series(levels), errors = "coerce").to_numpy(dtype = float)
    if np.all(np.isfinite(numeric)):
        span = float(np.max(numeric) - np.min(numeric))
        if span <= 0.0:
            positions = np.zeros(shape = len(levels), dtype = float)
        else:
            positions = (numeric - float(np.min(numeric))) / span
    else:
        positions = np.linspace(start = 0.0, stop = 1.0, num = len(levels))

    return {level: float(position) for level, position in zip(levels, positions)}


def _shade_plot_color(hex_color: str, factor: float) -> str:

    color = hex_color.lstrip("#")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness_new = min(1.0, max(0.0, lightness * factor))
    red_new, green_new, blue_new = colorsys.hls_to_rgb(hue, lightness_new, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        int(red_new * 255),
        int(green_new * 255),
        int(blue_new * 255),
    )


def _paired_plot_deltas(
    results: pd.DataFrame,
    value_col: str,
    delta_col: str,
    track: str | Sequence[str] | None,
    label_pert: str,
    label_base: str,
    unit_col: str,
    ) -> pd.DataFrame:

    data = _filter_plot_track(results = results, track = track)
    required_columns = {label_pert, "method", "intensity", "model", unit_col, value_col}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required perturbation columns: {missing_columns}")

    pair_columns = ["model", unit_col]
    if "track" in data.columns:
        pair_columns = ["track", *pair_columns]

    baseline = data.loc[data[label_pert] == label_base].copy()
    perturbed = data.loc[data[label_pert] != label_base].copy()
    if baseline.empty or perturbed.empty:
        return pd.DataFrame(columns = list(data.columns) + [delta_col])

    baseline_values = (
        baseline
        .groupby(by = pair_columns, observed = True)[value_col]
        .median()
        .reset_index()
        .rename(columns = {value_col: f"{value_col}_baseline"})
    )
    paired = perturbed.merge(
        right = baseline_values,
        on = pair_columns,
        how = "left",
    )
    paired[delta_col] = paired[value_col] - paired[f"{value_col}_baseline"]
    paired = paired.dropna(subset = [delta_col]).copy()
    return paired


def _aggregate_plot_sweep(
    paired: pd.DataFrame,
    delta_col: str,
    label_pert: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    group_columns = [label_pert, "method", "intensity"]
    aggregate = (
        paired
        .groupby(by = group_columns, observed = True)[delta_col]
        .median()
        .reset_index()
        .rename(columns = {delta_col: "delta"})
    )
    interval = (
        paired
        .groupby(by = group_columns, observed = True)[delta_col]
        .agg(
            q1 = lambda series: series.quantile(0.25),
            q3 = lambda series: series.quantile(0.75),
        )
        .reset_index()
    )
    return aggregate, interval


def _symmetric_plot_limit(
    aggregate: pd.DataFrame,
    interval: pd.DataFrame,
    margin: float,
    ) -> float:

    pieces = []
    if "delta" in aggregate.columns:
        pieces.append(pd.to_numeric(aggregate["delta"], errors = "coerce"))
    for column in ("q1", "q3"):
        if column in interval.columns:
            pieces.append(pd.to_numeric(interval[column], errors = "coerce"))

    finite_max = 0.0
    if pieces:
        values = pd.concat(objs = pieces, ignore_index = True).to_numpy(dtype = float)
        finite_values = np.abs(values[np.isfinite(values)])
        if finite_values.size:
            finite_max = float(np.max(finite_values))

    return max(finite_max * 1.12, float(margin) * 2.5, 0.05)


def plot_perturbation_superfigure(
    results_transfer: pd.DataFrame,
    results_recovery: pd.DataFrame,
    results_transfer_max: pd.DataFrame,
    results_recovery_max: pd.DataFrame,
    delta_ei: float,
    delta_ci: float,
    track: str | Sequence[str] | None = "frozen",
    perturbation_order: Sequence[str] | None = None,
    palette: Mapping[str, str] | None = None,
    method_linestyles: Sequence[str] | None = None,
    method_shade_factors: Sequence[float] | None = None,
    unit_col: str | None = None,
    sweep_ylim: tuple[float, float] = (-0.25, 0.25),
    figsize: tuple[float, float] = (14.2, 8.8),
    title: str = "Perturbation Stability Across Intensity and Joint Endpoint Response",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a perturbation superfigure that aligns EI and CI intensity
        sweeps with a max-intensity coordinate fingerprint by perturbation
        family.

    Args:
        results_transfer: Full perturbation transfer table with EI values.
        results_recovery: Full perturbation prediction consensus table with CI
            values.
        results_transfer_max: Maximum-intensity transfer table including
            baseline rows.
        results_recovery_max: Maximum-intensity recovery table including
            baseline rows.
        delta_ei: EI equivalence margin.
        delta_ci: CI equivalence margin.
        track: Track or tracks to plot when a track column is present.
        perturbation_order: Column order for perturbation families.
        palette: Mapping from perturbation family to color.
        method_linestyles: Line styles used for method sweeps.
        method_shade_factors: Lightness factors used for method fingerprints.
        unit_col: Pairing column. If None, inferred from group/domain.
        sweep_ylim: Fixed y-axis limits for the EI and CI sweep rows.
        figsize: Figure size in inches.
        title: Figure title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and 3 x n axes array.

    Raises:
        ValueError: If required columns are missing or no perturbation data are
            available.
    """

    unit_name = _resolve_plot_unit_column(
        results_transfer = results_transfer,
        results_recovery = results_recovery,
        unit_col = unit_col,
    )
    palette = dict(DEFAULT_PERTURBATION_PALETTE if palette is None else palette)
    title_colors = dict(DEFAULT_PERTURBATION_TITLE_COLORS)
    method_linestyles = tuple(DEFAULT_METHOD_LINESTYLES if method_linestyles is None else method_linestyles)
    method_shade_factors = tuple(
        DEFAULT_METHOD_SHADE_FACTORS if method_shade_factors is None else method_shade_factors
    )
    tick_color = "#8E8E8E"

    transfer_paired = _paired_plot_deltas(
        results = results_transfer,
        value_col = "ei",
        delta_col = "delta_ei",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_paired = _paired_plot_deltas(
        results = results_recovery,
        value_col = "ci",
        delta_col = "delta_ci",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    transfer_aggregate, transfer_interval = _aggregate_plot_sweep(
        paired = transfer_paired,
        delta_col = "delta_ei",
        label_pert = "perturbation",
    )
    recovery_aggregate, recovery_interval = _aggregate_plot_sweep(
        paired = recovery_paired,
        delta_col = "delta_ci",
        label_pert = "perturbation",
    )

    transfer_max_paired = _paired_plot_deltas(
        results = results_transfer_max,
        value_col = "ei",
        delta_col = "delta_ei",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_max_paired = _paired_plot_deltas(
        results = results_recovery_max,
        value_col = "ci",
        delta_col = "delta_ci",
        track = track,
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )

    fingerprint_columns = ["perturbation", "method", "model", unit_name]
    fingerprint = (
        transfer_max_paired[fingerprint_columns + ["delta_ei"]]
        .groupby(by = fingerprint_columns, observed = True)["delta_ei"]
        .median()
        .reset_index()
        .merge(
            right = (
                recovery_max_paired[fingerprint_columns + ["delta_ci"]]
                .groupby(by = fingerprint_columns, observed = True)["delta_ci"]
                .median()
                .reset_index()
            ),
            on = fingerprint_columns,
            how = "inner",
        )
    )

    available_perturbations = set(transfer_aggregate["perturbation"].unique())
    available_perturbations |= set(recovery_aggregate["perturbation"].unique())
    available_perturbations |= set(fingerprint["perturbation"].unique())
    if perturbation_order is None:
        ordered = [name for name in DEFAULT_PERTURBATION_ORDER if name in available_perturbations]
        ordered.extend(sorted(available_perturbations - set(ordered)))
        perturbation_order = ordered
    else:
        perturbation_order = [name for name in perturbation_order if name in available_perturbations]
    if not perturbation_order:
        raise ValueError("No perturbation rows are available for plotting")

    row_specs = [
        {
            "label": r"$\Delta$ EI",
            "margin": float(delta_ei),
            "aggregate": transfer_aggregate,
            "interval": transfer_interval,
        },
        {
            "label": r"$\Delta$ CI",
            "margin": float(delta_ci),
            "aggregate": recovery_aggregate,
            "interval": recovery_interval,
        },
    ]
    fig = plt.figure(figsize = figsize)
    outer_grid = fig.add_gridspec(
        nrows = 2,
        ncols = 1,
        height_ratios = [1.34, 1.62],
        hspace = 0.18,
    )
    sweep_grid = outer_grid[0].subgridspec(
        nrows = 2,
        ncols = len(perturbation_order),
        hspace = 0.10,
        wspace = 0.28,
    )
    fingerprint_grid = outer_grid[1].subgridspec(
        nrows = 1,
        ncols = len(perturbation_order),
        wspace = 0.28,
    )
    axes = np.empty(shape = (3, len(perturbation_order)), dtype = object)
    for column_index in range(len(perturbation_order)):
        if column_index == 0:
            axes[0, column_index] = fig.add_subplot(sweep_grid[0, column_index])
            axes[1, column_index] = fig.add_subplot(sweep_grid[1, column_index])
            axes[2, column_index] = fig.add_subplot(fingerprint_grid[0, column_index])
            continue
        axes[0, column_index] = fig.add_subplot(
            sweep_grid[0, column_index],
            sharey = axes[0, 0],
        )
        axes[1, column_index] = fig.add_subplot(
            sweep_grid[1, column_index],
            sharey = axes[1, 0],
        )
        axes[2, column_index] = fig.add_subplot(
            fingerprint_grid[0, column_index],
            sharex = axes[2, 0],
            sharey = axes[2, 0],
        )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top = 0.88, bottom = 0.08, left = 0.07, right = 0.985)
    neutral_color = "#C0C0C0"
    margin_color = "#E6F4EA"
    fingerprint_margin_color = "#CFEAD6"
    margin_edge_color = "#A6A6A6"
    margin_text_color = "#1F5E2E"
    fingerprint_limit = 0.5
    fingerprint_ticks = np.linspace(-0.5, 0.5, 5)
    sweep_ticks = [float(sweep_ylim[0]), 0.0, float(sweep_ylim[1])]

    def _format_method_label(method: object) -> str:
        return str(method).replace("_", " ").title()

    def _compose_delta_ticks(
        base_ticks: Sequence[float],
        delta_ticks: Mapping[float, str],
        ) -> tuple[list[float], list[str]]:

        ordered_ticks = sorted(
            [float(tick) for tick in base_ticks] + [float(tick) for tick in delta_ticks.keys()]
        )
        merged_ticks: list[float] = []
        for tick in ordered_ticks:
            if merged_ticks and np.isclose(tick, merged_ticks[-1], atol = 1e-9, rtol = 0.0):
                continue
            merged_ticks.append(tick)

        labels = []
        for tick in merged_ticks:
            tick_label = None
            for delta_tick, delta_label in delta_ticks.items():
                if np.isclose(tick, float(delta_tick), atol = 1e-9, rtol = 0.0):
                    tick_label = delta_label
                    break
            labels.append(f"{tick:.2f}" if tick_label is None else tick_label)

        return merged_ticks, labels

    def _draw_method_legend(
        axis,
        text_colors: Mapping[str, str],
        loc: str,
        bbox_to_anchor: tuple[float, float],
        handles: Sequence[Line2D] | None = None,
        handlelength: float = 1.9,
        ) -> None:

        legend = axis.legend(
            handles = handles,
            fontsize = 5.9,
            loc = loc,
            bbox_to_anchor = bbox_to_anchor,
            borderaxespad = 0.0,
            framealpha = 0.9,
            edgecolor = "#D0D0D0",
            handlelength = handlelength,
            borderpad = 0.45,
            labelspacing = 0.25,
            handletextpad = 0.45,
        )
        for legend_text in legend.get_texts():
            legend_text.set_color(text_colors.get(legend_text.get_text(), "#444444"))

    for row_index, row_spec in enumerate(row_specs):
        aggregate = row_spec["aggregate"]
        interval = row_spec["interval"]
        margin = row_spec["margin"]
        for column_index, perturbation in enumerate(perturbation_order):
            axis = axes[row_index, column_index]
            color = palette.get(perturbation, "#666666")
            aggregate_panel = aggregate.loc[aggregate["perturbation"] == perturbation].copy()
            interval_panel = interval.loc[interval["perturbation"] == perturbation].copy()

            if aggregate_panel.empty:
                axis.set_visible(False)
                continue

            axis.add_patch(
                Rectangle(
                    xy = (0.0, -float(margin)),
                    width = 1.0,
                    height = float(2.0 * margin),
                    facecolor = margin_color,
                    edgecolor = "none",
                    linewidth = 0.0,
                    zorder = 0,
                )
            )
            sweep_clip = Rectangle(
                xy = (0.0, float(sweep_ylim[0])),
                width = 1.0,
                height = float(sweep_ylim[1] - sweep_ylim[0]),
                transform = axis.transData,
            )
            axis.text(
                x = 0.03,
                y = float(margin) * 0.6,
                s = "Eq.",
                ha = "left",
                va = "center",
                fontsize = 6.8,
                color = margin_text_color,
                fontweight = "semibold",
                zorder = 5.5,
                bbox = {
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "#F3FBF5",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
            )
            axis.hlines(
                y = float(margin),
                xmin = 0.0,
                xmax = 1.0,
                colors = margin_edge_color,
                linestyles = "--",
                linewidths = 0.6,
                zorder = 2.0,
            )
            axis.hlines(
                y = -float(margin),
                xmin = 0.0,
                xmax = 1.0,
                colors = margin_edge_color,
                linestyles = "--",
                linewidths = 0.6,
                zorder = 2.0,
            )
            axis.hlines(
                y = 0.0,
                xmin = 0.0,
                xmax = 1.0,
                colors = neutral_color,
                linestyles = "--",
                linewidths = 0.85,
                zorder = 1,
            )
            axis.axvline(x = 1.0, color = "#000000", lw = 0.6, ls = "-", zorder = 4.5)

            methods = sorted(aggregate_panel["method"].unique())
            method_text_colors = {
                _format_method_label(method = method): color
                for method in methods
            }
            for method_index, method in enumerate(methods):
                method_label = _format_method_label(method = method)
                line_style = method_linestyles[method_index % len(method_linestyles)]
                method_aggregate = aggregate_panel.loc[aggregate_panel["method"] == method].copy()
                method_interval = interval_panel.loc[interval_panel["method"] == method].copy()
                intensities = _sorted_plot_levels(values = method_aggregate["intensity"])
                intensity_lookup = _normalized_plot_positions(levels = intensities)
                method_aggregate["_x"] = method_aggregate["intensity"].map(intensity_lookup)
                method_interval["_x"] = method_interval["intensity"].map(intensity_lookup)
                method_aggregate = method_aggregate.sort_values(by = "_x")
                method_interval = method_interval.sort_values(by = "_x")

                if len(method_interval) > 1:
                    interval_artist = axis.fill_between(
                        method_interval["_x"].to_numpy(dtype = float),
                        method_interval["q1"].to_numpy(dtype = float),
                        method_interval["q3"].to_numpy(dtype = float),
                        color = color,
                        alpha = 0.20,
                        zorder = 2,
                        lw = 0,
                    )
                    interval_artist.set_clip_path(sweep_clip)
                line_artist = axis.plot(
                    method_aggregate["_x"].to_numpy(dtype = float),
                    method_aggregate["delta"].to_numpy(dtype = float),
                    color = color,
                    lw = 1.1,
                    ls = line_style,
                    marker = "o",
                    markersize = 4.4,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    markeredgewidth = 0.0,
                    zorder = 4,
                    label = method_label,
                    solid_capstyle = "butt",
                )[0]
                line_artist.set_clip_path(sweep_clip)

            if row_index == 0:
                title = axis.set_title(
                    label = perturbation.capitalize(),
                    fontsize = 11.2,
                    color = title_colors.get(perturbation, color),
                    fontweight = "semibold",
                    y = 1.0,
                    pad = 0,
                )
                title.set_verticalalignment("center")

            if row_index == 1:
                _draw_method_legend(
                    axis = axis,
                    text_colors = method_text_colors,
                    loc = "center left",
                    bbox_to_anchor = (0.03, 1.06),
                    handlelength = 3.2,
                )

            axis.set_xticks(ticks = np.linspace(start = 0.0, stop = 1.0, num = 7))
            if row_index == 0:
                axis.set_xticklabels(labels = [])
            else:
                axis.set_xticklabels(
                    labels = ["0", "0.17", "0.33", "0.50", "0.67", "0.83", "1\nMax"],
                    fontsize = 6.5,
                    color = "#000000",
                )
                axis.get_xticklabels()[-1].set_fontweight("bold")
            axis.set_xlim(left = 0.0, right = 1.03)
            axis.set_ylim(bottom = float(sweep_ylim[0]), top = float(sweep_ylim[1]))
            sweep_y_ticks, sweep_y_labels = _compose_delta_ticks(
                base_ticks = sweep_ticks,
                delta_ticks = {
                    -float(margin): r"$-\delta$",
                    float(margin): r"$+\delta$",
                },
            )
            axis.set_yticks(ticks = sweep_y_ticks)
            axis.set_yticklabels(labels = sweep_y_labels)
            for tick_label in axis.get_yticklabels():
                if tick_label.get_text() in {r"$-\delta$", r"$+\delta$"}:
                    tick_label.set_color(tick_color)
            if row_index == 0:
                axis.tick_params(axis = "x", bottom = False, color = tick_color)
            else:
                axis.tick_params(
                    axis = "x",
                    bottom = True,
                    length = 2.8,
                    width = 0.6,
                    color = tick_color,
                )
            axis.tick_params(axis = "y", labelsize = 7, color = tick_color)
            axis.spines[["top", "right", "bottom"]].set_visible(False)
            if row_index == 1:
                axis.spines["bottom"].set_visible(True)
                axis.spines["bottom"].set_linewidth(0.6)
                axis.spines["bottom"].set_color("#9A9A9A")
            axis.spines["left"].set_linewidth(0.6)
            axis.spines["left"].set_color("#9A9A9A")
            axis.yaxis.grid(True, lw = 0.35, color = "#E8E8E8", zorder = 0)
            axis.set_axisbelow(True)

        axes[row_index, 0].set_ylabel(ylabel = row_spec["label"], fontsize = 9.2, labelpad = 8)

    for column_index, perturbation in enumerate(perturbation_order):
        axis = axes[2, column_index]
        color = palette.get(perturbation, "#666666")
        panel = fingerprint.loc[fingerprint["perturbation"] == perturbation].copy()
        if panel.empty:
            axis.set_visible(False)
            continue

        axis.add_patch(
            Rectangle(
                xy = (-float(delta_ei), -float(delta_ci)),
                width = float(2.0 * delta_ei),
                height = float(2.0 * delta_ci),
                facecolor = fingerprint_margin_color,
                edgecolor = margin_edge_color,
                linewidth = 0.6,
                linestyle = "--",
                alpha = 0.82,
                zorder = 1.85,
            )
        )
        axis.text(
            x = float(delta_ei) * 0.95,
            y = float(delta_ci) * 0.85,
            s = "Eq.",
            ha = "right",
            va = "top",
            fontsize = 6.9,
            color = margin_text_color,
            fontweight = "semibold",
            zorder = 2.05,
        )
        axis.axhline(y = float(delta_ci), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axhline(y = -float(delta_ci), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axvline(x = float(delta_ei), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axvline(x = -float(delta_ei), color = margin_edge_color, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axhline(y = 0.0, color = "#7A7A7A", lw = 0.6, ls = "-", zorder = 2.1)
        axis.axvline(x = 0.0, color = "#7A7A7A", lw = 0.6, ls = "-", zorder = 2.1)

        methods = sorted(panel["method"].unique())
        fingerprint_handles: list[Line2D] = []
        fingerprint_text_colors: dict[str, str] = {}
        for method_index, method in enumerate(methods):
            method_panel = panel.loc[panel["method"] == method].copy()
            if method_panel.empty:
                continue

            method_label = _format_method_label(method = method)
            method_depth = len(methods) - method_index
            line_style = method_linestyles[method_index % len(method_linestyles)]
            shade = _shade_plot_color(
                hex_color = color,
                factor = method_shade_factors[method_index % len(method_shade_factors)],
            )
            x_values = method_panel["delta_ei"].to_numpy(dtype = float)
            y_values = method_panel["delta_ci"].to_numpy(dtype = float)
            axis.scatter(
                x = x_values,
                y = y_values,
                s = 18,
                color = shade,
                alpha = 0.26,
                edgecolors = "none",
                zorder = 3.0 + (0.1 * method_depth),
            )
            median_x = float(np.nanmedian(x_values))
            median_y = float(np.nanmedian(y_values))
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = shade,
                lw = 1.1,
                ls = line_style,
                alpha = 0.88,
                zorder = 4,
            )
            axis.scatter(
                x = [median_x],
                y = [median_y],
                s = 76,
                color = shade,
                edgecolors = "none",
                linewidths = 0.0,
                zorder = 5.0 + (0.1 * method_depth),
            )
            median_x_label = f"{median_x:+.2f}"
            median_y_label = f"{median_y:+.2f}"
            if median_x_label in {"+0.00", "-0.00"}:
                median_x_label = "0.00"
            if median_y_label in {"+0.00", "-0.00"}:
                median_y_label = "0.00"
            legend_label = (
                f"{method_label}: "
                f"ΔEI = {median_x_label}, ΔCI = {median_y_label}"
            )
            fingerprint_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color = shade,
                    lw = 0.0,
                    ls = "None",
                    marker = "o",
                    markersize = 4.4,
                    markerfacecolor = shade,
                    markeredgecolor = shade,
                    markeredgewidth = 0.0,
                    label = legend_label,
                )
            )
            fingerprint_text_colors[legend_label] = color

        if fingerprint_handles:
            _draw_method_legend(
                axis = axis,
                handles = fingerprint_handles,
                text_colors = fingerprint_text_colors,
                loc = "upper left",
                bbox_to_anchor = (0.03, 1.0),
            )

        axis.set_xlim(left = -fingerprint_limit, right = fingerprint_limit)
        axis.set_ylim(bottom = -fingerprint_limit, top = fingerprint_limit)
        fingerprint_x_ticks, fingerprint_x_labels = _compose_delta_ticks(
            base_ticks = fingerprint_ticks,
            delta_ticks = {
                -float(delta_ei): r"$-\delta$",
                float(delta_ei): r"$+\delta$",
            },
        )
        fingerprint_y_ticks, fingerprint_y_labels = _compose_delta_ticks(
            base_ticks = fingerprint_ticks,
            delta_ticks = {
                -float(delta_ci): r"$-\delta$",
                float(delta_ci): r"$+\delta$",
            },
        )
        axis.set_xticks(ticks = fingerprint_x_ticks)
        axis.set_xticklabels(labels = fingerprint_x_labels)
        axis.set_yticks(ticks = fingerprint_y_ticks)
        axis.set_yticklabels(labels = fingerprint_y_labels)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            if tick_label.get_text() in {r"$-\delta$", r"$+\delta$"}:
                tick_label.set_color(tick_color)
        axis.tick_params(axis = "both", labelsize = 7, color = tick_color)
        axis.grid(False)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_visible(True)
        axis.spines["left"].set_color("#9A9A9A")
        axis.spines["bottom"].set_color("#9A9A9A")
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    axes[2, 0].set_ylabel(ylabel = r"$\Delta$ CI at Maximum Perturbation Intensity", fontsize = 9.2, labelpad = 8)
    x_label_gap = axes[2, 0].yaxis.labelpad / 72.0 / float(fig.get_size_inches()[1])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fingerprint_xaxis_bbox = axes[2, 0].xaxis.get_tightbbox(renderer = renderer)
    sweep_xaxis_bbox = axes[1, 0].xaxis.get_tightbbox(renderer = renderer)
    if fingerprint_xaxis_bbox is None or sweep_xaxis_bbox is None:
        fingerprint_label_y = axes[2, 0].get_position().y0 - x_label_gap
        sweep_label_y = axes[1, 0].get_position().y0 - x_label_gap
    else:
        fingerprint_tick_bottom = fig.transFigure.inverted().transform(
            (0.0, fingerprint_xaxis_bbox.y0)
        )[1]
        sweep_tick_bottom = fig.transFigure.inverted().transform(
            (0.0, sweep_xaxis_bbox.y0)
        )[1]
        fingerprint_label_y = fingerprint_tick_bottom - x_label_gap
        sweep_label_y = sweep_tick_bottom - x_label_gap
    fig.text(
        x = 0.52,
        y = sweep_label_y,
        s = "Normalized Perturbation Intensity",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    fig.text(
        x = 0.52,
        y = fingerprint_label_y,
        s = r"$\Delta$ EI at Maximum Perturbation Intensity",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(["Helvetica", "Arial", "sans-serif"])
        text_artist.set_fontsize(text_artist.get_fontsize() + 2.0)

    if show:
        plt.show()
    return fig, axes


def plot_falsification_fingerprint(
    results_transfer: pd.DataFrame,
    results_agreement: pd.DataFrame,
    method_order: Sequence[str] | None = None,
    track_order: Sequence[str] | None = ("frozen", "retrain"),
    track_colors: Mapping[str, str] | None = None,
    method_col: str = "Method",
    track_col: str = "Falsification",
    condition_col: str = "condition",
    condition_original: str = "original",
    condition_falsified: str = "falsified",
    unit_col: str = "group",
    value_transfer: str = "ei",
    value_agreement: str = "ci",
    delta_ei: float | None = None,
    delta_ci: float | None = None,
    iqr_scale_ei: float = 1.0,
    iqr_scale_ci: float = 1.0,
    iqr_decimals: int = 2,
    delta_label: str | None = "Eq.",
    degradation_label: str | None = "Falsified",
    limit: float = 0.5,
    figsize: tuple[float, float] = (13.8, 4.7),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot falsification paired-delta fingerprints using the same visual
        language as the perturbation fingerprint panels.

    Args:
        results_transfer: Falsified transfer results with original and
            falsified EI values.
        results_agreement: Falsified agreement results with original and
            falsified CI values.
        method_order: Optional order for falsification method panels.
        track_order: Optional order for falsification tracks.
        track_colors: Optional mapping from track to color.
        method_col: Column identifying falsification method.
        track_col: Column identifying falsification track.
        condition_col: Column identifying original versus falsified rows.
        condition_original: Label for original rows.
        condition_falsified: Label for falsified rows.
        unit_col: Pairing unit column.
        value_transfer: EI metric column.
        value_agreement: CI metric column.
        delta_ei: Optional shared EI reference margin. If both delta values are
            None, a single shared margin is derived from the larger original
            EI/CI IQR so both axes use the same delta.
        delta_ci: Optional shared CI reference margin. If one delta is provided
            and the other is None, the provided value is used for both axes.
        iqr_scale_ei: Multiplier applied to the EI IQR margin.
        iqr_scale_ci: Multiplier applied to the CI IQR margin.
        iqr_decimals: Number of decimals used when rounding IQR margins up.
        delta_label: Optional text label drawn near the reference margins.
        degradation_label: Optional text label drawn in the lower-left
            degradation quadrant.
        limit: Symmetric axis limit.
        figsize: Figure size in inches.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing or no paired rows exist.
    """

    if track_colors is None:
        track_colors = {
            "frozen": "#A6363A",
            "retrain": "#C46A1C",
        }
    else:
        track_colors = dict(track_colors)

    pair_columns = [track_col, method_col, "model", unit_col]

    def _iqr_margin(
        results: pd.DataFrame,
        value_col: str,
        scale: float,
        ) -> float:

        required_columns = {condition_col, value_col}
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            raise ValueError(f"Missing required falsification columns: {missing_columns}")

        reference = results.loc[results[condition_col] == condition_original]
        values = pd.to_numeric(reference[value_col], errors = "coerce").to_numpy(dtype = float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ValueError("At least two finite original values are required to derive IQR margins")

        dispersion = float(np.percentile(values, 75) - np.percentile(values, 25))
        return float(round(max(float(scale * dispersion), 1e-6), int(iqr_decimals)))

    def _paired_metric_frame(
        results: pd.DataFrame,
        value_col: str,
        delta_col: str,
        ) -> pd.DataFrame:

        required_columns = set(pair_columns + [condition_col, value_col])
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            raise ValueError(f"Missing required falsification columns: {missing_columns}")

        paired = (
            results
            .loc[:, pair_columns + [condition_col, value_col]]
            .pivot_table(
                index = pair_columns,
                columns = condition_col,
                values = value_col,
                aggfunc = "median",
                observed = True,
            )
        )
        missing_conditions = [
            condition
            for condition in (condition_original, condition_falsified)
            if condition not in paired.columns
        ]
        if missing_conditions:
            raise ValueError(f"Missing falsification conditions: {missing_conditions}")

        paired = paired.dropna(subset = [condition_original, condition_falsified]).reset_index()
        paired[delta_col] = paired[condition_falsified] - paired[condition_original]
        return paired.loc[:, pair_columns + [delta_col]].copy()

    transfer_paired = _paired_metric_frame(
        results = results_transfer,
        value_col = value_transfer,
        delta_col = "delta_ei",
    )
    agreement_paired = _paired_metric_frame(
        results = results_agreement,
        value_col = value_agreement,
        delta_col = "delta_ci",
    )
    fingerprint = transfer_paired.merge(
        right = agreement_paired,
        on = pair_columns,
        how = "inner",
    )
    if fingerprint.empty:
        raise ValueError("No paired falsification rows are available for plotting")

    if delta_ei is None and delta_ci is None:
        delta_ei_iqr = _iqr_margin(
            results = results_transfer,
            value_col = value_transfer,
            scale = iqr_scale_ei,
        )
        delta_ci_iqr = _iqr_margin(
            results = results_agreement,
            value_col = value_agreement,
            scale = iqr_scale_ci,
        )
        delta_shared = max(delta_ei_iqr, delta_ci_iqr)
        delta_ei = delta_shared
        delta_ci = delta_shared
    elif delta_ei is None:
        delta_ei = float(delta_ci)
    elif delta_ci is None:
        delta_ci = float(delta_ei)

    def _normalize_label(value: object) -> str:
        return str(value).replace("_", " ").replace("-", " ").lower()

    available_methods = fingerprint[method_col].dropna().unique().tolist()
    if method_order is None:
        preferred_methods = ("target remap", "random generate", "vector generate")
        ordered_methods = []
        for preferred_method in preferred_methods:
            ordered_methods.extend(
                [
                    method
                    for method in available_methods
                    if _normalize_label(value = method) == preferred_method
                ]
            )
        ordered_methods.extend(
            sorted(
                [method for method in available_methods if method not in ordered_methods],
                key = lambda method: str(method),
            )
        )
    else:
        ordered_methods = [method for method in method_order if method in available_methods]
    if not ordered_methods:
        raise ValueError("No falsification method rows are available for plotting")

    available_tracks = fingerprint[track_col].dropna().unique().tolist()
    if track_order is None:
        ordered_tracks = sorted(available_tracks, key = lambda track: str(track))
    else:
        ordered_tracks = [track for track in track_order if track in available_tracks]
        ordered_tracks.extend(
            sorted(
                [track for track in available_tracks if track not in ordered_tracks],
                key = lambda track: str(track),
            )
        )

    def _format_label(value: object) -> str:
        return str(value).replace("_", " ").title()

    def _format_signed(value: float) -> str:
        label = f"{value:+.2f}"
        return "0.00" if label in {"+0.00", "-0.00"} else label

    def _compose_reference_ticks(
        base_ticks: Sequence[float],
        delta_ticks: Mapping[float, str],
        ) -> tuple[list[float], list[str]]:

        ordered_ticks = sorted(
            [float(tick) for tick in base_ticks] + [float(tick) for tick in delta_ticks.keys()]
        )
        merged_ticks: list[float] = []
        for tick in ordered_ticks:
            if merged_ticks and np.isclose(tick, merged_ticks[-1], atol = 1e-9, rtol = 0.0):
                continue
            merged_ticks.append(tick)

        labels = []
        for tick in merged_ticks:
            tick_label = None
            for delta_tick, delta_label in delta_ticks.items():
                if np.isclose(tick, float(delta_tick), atol = 1e-9, rtol = 0.0):
                    tick_label = delta_label
                    break
            labels.append(f"{tick:.2f}" if tick_label is None else tick_label)

        return merged_ticks, labels

    fig, axes_grid = plt.subplots(
        nrows = 1,
        ncols = len(ordered_methods),
        figsize = figsize,
        sharex = True,
        sharey = True,
        squeeze = False,
    )
    axes = axes_grid.ravel()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top = 0.88, bottom = 0.16, left = 0.07, right = 0.985, wspace = 0.28)

    title_color = "#8F2A2A"
    zero_color = "#7A7A7A"
    spine_color = "#9A9A9A"
    tick_color = "#8E8E8E"
    degradation_face = "#F2CACA"
    ticks = np.linspace(start = -float(limit), stop = float(limit), num = 5)
    x_ticks, x_tick_labels = _compose_reference_ticks(
        base_ticks = ticks,
        delta_ticks = {},
    )
    y_ticks, y_tick_labels = _compose_reference_ticks(
        base_ticks = ticks,
        delta_ticks = {},
    )

    for axis, method in zip(axes, ordered_methods):
        panel = fingerprint.loc[fingerprint[method_col] == method].copy()
        axis.add_patch(
            Rectangle(
                xy = (-float(limit), -float(limit)),
                width = float(limit),
                height = float(limit),
                facecolor = degradation_face,
                edgecolor = "none",
                alpha = 0.22,
                zorder = 1.6,
            )
        )
        if degradation_label is not None and str(degradation_label).strip():
            degradation_label_pad = 0.04
            axis.text(
                x = -float(limit) + degradation_label_pad,
                y = -float(limit) + degradation_label_pad,
                s = str(degradation_label),
                ha = "left",
                va = "bottom",
                fontsize = 8.4,
                color = title_color,
                fontweight = "black",
                alpha = 0.82,
                zorder = 2.0,
            )
        axis.axhline(y = 0.0, color = zero_color, lw = 0.6, ls = "-", zorder = 2.1)
        axis.axvline(x = 0.0, color = zero_color, lw = 0.6, ls = "-", zorder = 2.1)

        legend_handles: list[Line2D] = []
        legend_text_colors: dict[str, str] = {}
        for track_index, track in enumerate(ordered_tracks):
            track_panel = panel.loc[panel[track_col] == track].copy()
            if track_panel.empty:
                continue

            color = track_colors.get(str(track), "#666666")
            track_depth = len(ordered_tracks) - track_index
            x_values = track_panel["delta_ei"].to_numpy(dtype = float)
            y_values = track_panel["delta_ci"].to_numpy(dtype = float)
            axis.scatter(
                x = x_values,
                y = y_values,
                s = 18,
                color = color,
                alpha = 0.26,
                edgecolors = "none",
                zorder = 3.0 + (0.1 * track_depth),
            )
            median_x = float(np.nanmedian(x_values))
            median_y = float(np.nanmedian(y_values))
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = color,
                lw = 1.1,
                alpha = 0.88,
                zorder = 4,
            )
            axis.scatter(
                x = [median_x],
                y = [median_y],
                s = 64,
                color = color,
                edgecolors = "none",
                linewidths = 0.0,
                zorder = 5.0 + (0.1 * track_depth),
            )
            legend_label = (
                f"{_format_label(value = track)}: "
                f"ΔEI = {_format_signed(value = median_x)}, "
                f"ΔCI = {_format_signed(value = median_y)}"
            )
            legend_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color = color,
                    lw = 0.0,
                    ls = "None",
                    marker = "o",
                    markersize = 4.4,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    markeredgewidth = 0.0,
                    label = legend_label,
                )
            )
            legend_text_colors[legend_label] = color

        if legend_handles:
            legend = axis.legend(
                handles = legend_handles,
                fontsize = 5.9,
                loc = "upper left",
                bbox_to_anchor = (0.03, 1.0),
                borderaxespad = 0.0,
                framealpha = 0.9,
                edgecolor = "#D0D0D0",
                handlelength = 1.9,
                borderpad = 0.45,
                labelspacing = 0.25,
                handletextpad = 0.45,
            )
            for legend_text in legend.get_texts():
                legend_text.set_color(legend_text_colors.get(legend_text.get_text(), "#444444"))

        axis.set_title(
            label = _format_label(value = method),
            fontsize = 10.2,
            fontweight = "semibold",
            color = title_color,
            pad = 8,
        )
        axis.set_xlim(left = -float(limit), right = float(limit))
        axis.set_ylim(bottom = -float(limit), top = float(limit))
        axis.set_xticks(ticks = x_ticks)
        axis.set_xticklabels(labels = x_tick_labels)
        axis.set_yticks(ticks = y_ticks)
        axis.set_yticklabels(labels = y_tick_labels)
        axis.tick_params(axis = "both", labelsize = 7, color = tick_color)
        axis.tick_params(axis = "y", labelleft = True)
        axis.grid(False)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_visible(True)
        axis.spines["left"].set_color(spine_color)
        axis.spines["bottom"].set_color(spine_color)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    axes[0].set_ylabel(
        ylabel = r"$\Delta$ CI (Falsified - Original)",
        fontsize = 9.2,
        labelpad = 8,
    )
    fig.text(
        x = 0.52,
        y = 0.08,
        s = r"$\Delta$ EI (Falsified - Original)",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(["Helvetica", "Arial", "sans-serif"])
        text_artist.set_fontsize(text_artist.get_fontsize() + 2.0)

    if show:
        plt.show()
    return fig, axes


def plot_falsification_track_fingerprint(
    results_transfer: pd.DataFrame,
    results_agreement: pd.DataFrame,
    track_order: Sequence[str] | None = ("frozen", "retrain"),
    method_order: Sequence[str] | None = None,
    method_colors: Mapping[str, str] | None = None,
    method_col: str = "Method",
    track_col: str = "Falsification",
    condition_col: str = "condition",
    condition_original: str = "original",
    condition_falsified: str = "falsified",
    unit_col: str = "group",
    value_transfer: str = "ei",
    value_agreement: str = "ci",
    degradation_label: str | None = "Falsified",
    limit: float = 0.5,
    figsize: tuple[float, float] = (9.0, 4.5),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot falsification paired-delta fingerprints with columns separated by
        frozen/retrained evaluation track and legend entries separated by
        falsification technique.

    Args:
        results_transfer: Falsified transfer results with original and
            falsified EI values.
        results_agreement: Falsified agreement results with original and
            falsified CI values.
        track_order: Optional order for falsification track panels.
        method_order: Optional order for falsification methods within panels.
        method_colors: Optional mapping from falsification method to color.
        method_col: Column identifying falsification method.
        track_col: Column identifying falsification track.
        condition_col: Column identifying original versus falsified rows.
        condition_original: Label for original rows.
        condition_falsified: Label for falsified rows.
        unit_col: Pairing unit column.
        value_transfer: EI metric column.
        value_agreement: CI metric column.
        degradation_label: Optional text label drawn in the lower-left
            degradation quadrant.
        limit: Symmetric axis limit.
        figsize: Figure size in inches.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing or no paired rows exist.
    """

    if method_colors is None:
        method_colors = {
            "target remap": "#A6363A",
            "random generate": "#C46A1C",
            "vector generate": "#2C6E91",
        }
    else:
        method_colors = dict(method_colors)

    pair_columns = [track_col, method_col, "model", unit_col]

    def _normalize_label(value: object) -> str:
        return str(value).replace("_", " ").replace("-", " ").lower()

    def _paired_metric_frame(
        results: pd.DataFrame,
        value_col: str,
        delta_col: str,
        ) -> pd.DataFrame:

        required_columns = set(pair_columns + [condition_col, value_col])
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            raise ValueError(f"Missing required falsification columns: {missing_columns}")

        paired = (
            results
            .loc[:, pair_columns + [condition_col, value_col]]
            .pivot_table(
                index = pair_columns,
                columns = condition_col,
                values = value_col,
                aggfunc = "median",
                observed = True,
            )
        )
        missing_conditions = [
            condition
            for condition in (condition_original, condition_falsified)
            if condition not in paired.columns
        ]
        if missing_conditions:
            raise ValueError(f"Missing falsification conditions: {missing_conditions}")

        paired = paired.dropna(subset = [condition_original, condition_falsified]).reset_index()
        paired[delta_col] = paired[condition_falsified] - paired[condition_original]
        return paired.loc[:, pair_columns + [delta_col]].copy()

    transfer_paired = _paired_metric_frame(
        results = results_transfer,
        value_col = value_transfer,
        delta_col = "delta_ei",
    )
    agreement_paired = _paired_metric_frame(
        results = results_agreement,
        value_col = value_agreement,
        delta_col = "delta_ci",
    )
    fingerprint = transfer_paired.merge(
        right = agreement_paired,
        on = pair_columns,
        how = "inner",
    )
    if fingerprint.empty:
        raise ValueError("No paired falsification rows are available for plotting")

    available_methods = fingerprint[method_col].dropna().unique().tolist()
    if method_order is None:
        preferred_methods = ("target remap", "random generate", "vector generate")
        ordered_methods = []
        for preferred_method in preferred_methods:
            ordered_methods.extend(
                [
                    method
                    for method in available_methods
                    if _normalize_label(value = method) == preferred_method
                ]
            )
        ordered_methods.extend(
            sorted(
                [method for method in available_methods if method not in ordered_methods],
                key = lambda method: str(method),
            )
        )
    else:
        ordered_methods = [method for method in method_order if method in available_methods]
    if not ordered_methods:
        raise ValueError("No falsification method rows are available for plotting")

    available_tracks = fingerprint[track_col].dropna().unique().tolist()
    if track_order is None:
        ordered_tracks = sorted(available_tracks, key = lambda track: str(track))
    else:
        ordered_tracks = [track for track in track_order if track in available_tracks]
        ordered_tracks.extend(
            sorted(
                [track for track in available_tracks if track not in ordered_tracks],
                key = lambda track: str(track),
            )
        )
    if not ordered_tracks:
        raise ValueError("No falsification track rows are available for plotting")

    def _format_label(value: object) -> str:
        return str(value).replace("_", " ").title()

    def _format_signed(value: float) -> str:
        label = f"{value:+.2f}"
        return "0.00" if label in {"+0.00", "-0.00"} else label

    def _compose_reference_ticks(
        base_ticks: Sequence[float],
        ) -> tuple[list[float], list[str]]:

        ticks = [float(tick) for tick in base_ticks]
        return ticks, [f"{tick:.2f}" for tick in ticks]

    method_color_lookup = {
        _normalize_label(value = method): color
        for method, color in method_colors.items()
    }
    fallback_colors = ("#A6363A", "#C46A1C", "#2C6E91", "#6A4C93", "#4F6F2A")
    track_title_colors = {
        "frozen": "#8F2A2A",
        "retrain": "#8F2A2A",
    }

    def _method_color(method: object, method_index: int) -> str:
        return method_color_lookup.get(
            _normalize_label(value = method),
            fallback_colors[method_index % len(fallback_colors)],
        )

    fig, axes_grid = plt.subplots(
        nrows = 1,
        ncols = len(ordered_tracks),
        figsize = figsize,
        sharex = True,
        sharey = True,
        squeeze = False,
    )
    axes = axes_grid.ravel()
    fig.patch.set_facecolor("white")
    target_gap_inches = 14.2 * (0.985 - 0.07) * 0.28 / (4.0 + 3.0 * 0.28)
    plot_width_inches = float(figsize[0]) * (0.985 - 0.08)
    panel_width_inches = (plot_width_inches - target_gap_inches * (len(ordered_tracks) - 1)) / len(ordered_tracks)
    panel_wspace = target_gap_inches / panel_width_inches
    fig.subplots_adjust(top = 0.84, bottom = 0.16, left = 0.08, right = 0.985, wspace = panel_wspace)

    title_color = "#8F2A2A"
    zero_color = "#7A7A7A"
    spine_color = "#9A9A9A"
    tick_color = "#8E8E8E"
    degradation_face = "#F2CACA"
    ticks = np.linspace(start = -float(limit), stop = float(limit), num = 5)
    x_ticks, x_tick_labels = _compose_reference_ticks(base_ticks = ticks)
    y_ticks, y_tick_labels = _compose_reference_ticks(base_ticks = ticks)

    for axis, track in zip(axes, ordered_tracks):
        panel = fingerprint.loc[fingerprint[track_col] == track].copy()
        axis.add_patch(
            Rectangle(
                xy = (-float(limit), -float(limit)),
                width = float(limit),
                height = float(limit),
                facecolor = degradation_face,
                edgecolor = "none",
                alpha = 0.22,
                zorder = 1.6,
            )
        )
        if degradation_label is not None and str(degradation_label).strip():
            degradation_label_pad = 0.04
            axis.text(
                x = -float(limit) + degradation_label_pad,
                y = -float(limit) + degradation_label_pad,
                s = str(degradation_label),
                ha = "left",
                va = "bottom",
                fontsize = 8.4,
                color = title_color,
                fontweight = "black",
                alpha = 0.82,
                zorder = 2.0,
            )
        axis.axhline(y = 0.0, color = zero_color, lw = 0.6, ls = "-", zorder = 2.1)
        axis.axvline(x = 0.0, color = zero_color, lw = 0.6, ls = "-", zorder = 2.1)

        legend_handles: list[Line2D] = []
        legend_text_colors: dict[str, str] = {}
        for method_index, method in enumerate(ordered_methods):
            method_panel = panel.loc[panel[method_col] == method].copy()
            if method_panel.empty:
                continue

            color = _method_color(method = method, method_index = method_index)
            method_depth = len(ordered_methods) - method_index
            line_style = DEFAULT_METHOD_LINESTYLES[method_index % len(DEFAULT_METHOD_LINESTYLES)]
            x_values = method_panel["delta_ei"].to_numpy(dtype = float)
            y_values = method_panel["delta_ci"].to_numpy(dtype = float)
            axis.scatter(
                x = x_values,
                y = y_values,
                s = 18,
                color = color,
                alpha = 0.24,
                edgecolors = "none",
                zorder = 3.0 + (0.1 * method_depth),
            )
            median_x = float(np.nanmedian(x_values))
            median_y = float(np.nanmedian(y_values))
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = color,
                lw = 1.1,
                ls = line_style,
                alpha = 0.88,
                zorder = 4,
            )
            axis.scatter(
                x = [median_x],
                y = [median_y],
                s = 64,
                color = color,
                edgecolors = "none",
                linewidths = 0.0,
                zorder = 5.0 + (0.1 * method_depth),
            )
            legend_label = (
                f"{_format_label(value = method)}: "
                f"ΔEI = {_format_signed(value = median_x)}, "
                f"ΔCI = {_format_signed(value = median_y)}"
            )
            legend_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color = color,
                    lw = 0.0,
                    ls = "None",
                    marker = "o",
                    markersize = 4.4,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    markeredgewidth = 0.0,
                    label = legend_label,
                )
            )
            legend_text_colors[legend_label] = color

        if legend_handles:
            legend = axis.legend(
                handles = legend_handles,
                fontsize = 5.9,
                loc = "upper left",
                bbox_to_anchor = (0.03, 1.0),
                borderaxespad = 0.0,
                framealpha = 0.9,
                edgecolor = "#D0D0D0",
                handlelength = 1.9,
                borderpad = 0.45,
                labelspacing = 0.25,
                handletextpad = 0.45,
            )
            for legend_text in legend.get_texts():
                legend_text.set_color(legend_text_colors.get(legend_text.get_text(), "#444444"))

        track_label = _format_label(value = track)
        axis.set_title(
            label = track_label,
            fontsize = 10.2,
            fontweight = "semibold",
            color = track_title_colors.get(_normalize_label(value = track), title_color),
            pad = 8,
        )
        axis.set_xlim(left = -float(limit), right = float(limit))
        axis.set_ylim(bottom = -float(limit), top = float(limit))
        axis.set_xticks(ticks = x_ticks)
        axis.set_xticklabels(labels = x_tick_labels)
        axis.set_yticks(ticks = y_ticks)
        axis.set_yticklabels(labels = y_tick_labels)
        axis.tick_params(axis = "both", labelsize = 7, color = tick_color)
        axis.tick_params(axis = "y", labelleft = True)
        axis.grid(False)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_visible(True)
        axis.spines["left"].set_color(spine_color)
        axis.spines["bottom"].set_color(spine_color)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    axes[0].set_ylabel(
        ylabel = r"$\Delta$ CI (Falsified - Original)",
        fontsize = 9.2,
        labelpad = 8,
    )
    fig.text(
        x = 0.52,
        y = 0.08,
        s = r"$\Delta$ EI (Falsified - Original)",
        ha = "center",
        va = "top",
        fontsize = 9.2,
    )
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(["Helvetica", "Arial", "sans-serif"])
        text_artist.set_fontsize(text_artist.get_fontsize() + 2.0)

    if show:
        plt.show()
    return fig, axes


def plot_decomposition_evidence(
    results: pd.DataFrame,
    noninferiority: pd.DataFrame,
    attribution: pd.DataFrame,
    delta: float,
    specification_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (13.5, 4.8),
    title: str = "Evidence for the additive decomposition",
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a three-panel decomposition summary showing specification-level
        EI distributions, non-inferiority gaps relative to additive, and the
        residual-attribution contrast.

    Args:
        results: Decomposition results with at least specification and ei
            columns.
        noninferiority: Output table from stat_decomposed_test.
        attribution: Output table from stat_decomposed_attribution.
        delta: Non-inferiority margin used in the paired EI tests.
        specification_order: Display order for decomposition specifications.
        figsize: Figure size in inches.
        title: Figure title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing.
    """

    if specification_order is None:
        specification_order = (
            "additive",
            "interaction",
            "interaction_joint",
            "joint",
            "invariants",
            "signatures",
        )

    def format_specification(specification: str) -> str:
        return {
            "interaction_joint": "Interaction-Joint",
        }.get(specification, str(specification).replace("_", " ").title())

    def coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex = False), errors = "coerce")

    required_results = {"specification", "ei"}
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(f"Missing required result columns: {missing_results}")

    ni_frame = noninferiority.reset_index().copy()
    attr_frame = attribution.reset_index().copy()

    required_ni = {"Specification", "Median Δ EI", "Holm-adj. p", "NI."}
    missing_ni = sorted(required_ni - set(ni_frame.columns))
    if missing_ni:
        raise ValueError(f"Missing required non-inferiority columns: {missing_ni}")

    required_attr = {"Property", "Comparison", "Median Δ MAE", "Holm-adj. p", "Diff."}
    missing_attr = sorted(required_attr - set(attr_frame.columns))
    if missing_attr:
        raise ValueError(f"Missing required attribution columns: {missing_attr}")

    summary = (
        results
        .groupby(by = "specification", observed = True)["ei"]
        .agg(
            median = "median",
            q1 = lambda values: values.quantile(0.25),
            q3 = lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary["display"] = summary["specification"].map(format_specification)

    ordered_specs = [spec for spec in specification_order if spec in set(summary["specification"])]
    ordered_specs.extend(
        spec for spec in summary["specification"].tolist() if spec not in ordered_specs
    )
    summary["specification"] = pd.Categorical(
        summary["specification"],
        categories = ordered_specs,
        ordered = True,
    )
    summary = summary.sort_values("specification", ascending = True).reset_index(drop = True)

    ni_frame["Median Δ EI"] = coerce_numeric(ni_frame["Median Δ EI"])
    ni_frame["Holm-adj. p"] = coerce_numeric(ni_frame["Holm-adj. p"])
    ni_frame["display"] = ni_frame["Specification"].astype(str)
    ni_frame["order"] = ni_frame["display"].map(
        {format_specification(spec): idx for idx, spec in enumerate(ordered_specs)}
    )
    ni_frame = ni_frame.sort_values("order").reset_index(drop = True)

    attr_frame["Median Δ MAE"] = coerce_numeric(attr_frame["Median Δ MAE"])
    attr_frame["Holm-adj. p"] = coerce_numeric(attr_frame["Holm-adj. p"])
    attr_row = attr_frame.iloc[0]

    fig = plt.figure(figsize = figsize, constrained_layout = True)
    grid = fig.add_gridspec(nrows = 1, ncols = 3, width_ratios = [1.5, 1.2, 0.9])
    axes = np.array([
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ], dtype = object)

    additive_color = "#4F5D2F"
    comparison_color = "#A3B18A"
    success_color = "#9A6A1B"
    warning_color = "#7F5539"
    neutral_color = "#6c757d"

    y_summary = np.arange(len(summary))[::-1]
    for y_value, (_, row) in zip(y_summary, summary.iterrows()):
        is_additive = str(row["specification"]) == "additive"
        color = additive_color if is_additive else comparison_color
        axes[0].hlines(y = y_value, xmin = row["q1"], xmax = row["q3"], color = color, linewidth = 5, alpha = 0.8)
        axes[0].scatter(row["median"], y_value, s = 90 if is_additive else 70, color = color, edgecolor = "white", linewidth = 1.1, zorder = 3)
    axes[0].set_yticks(y_summary)
    axes[0].set_yticklabels(summary["display"], fontsize = 9)
    axes[0].set_xlabel("EI median with IQR", fontsize = 9)
    axes[0].set_title("Specification Performance", fontsize = 11)
    axes[0].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[0].spines[spine].set_visible(False)
    axes[0].tick_params(axis = "y", length = 0)

    y_ni = np.arange(len(ni_frame))[::-1]
    gap_min = float(np.nanmin(np.r_[ni_frame["Median Δ EI"].to_numpy(dtype = float), 0.0]))
    gap_max = float(np.nanmax(np.r_[ni_frame["Median Δ EI"].to_numpy(dtype = float), delta]))
    gap_pad = max(0.02, 0.08 * (gap_max - gap_min if gap_max > gap_min else 1.0))
    axes[1].axvspan(gap_min - gap_pad, delta, color = "#d8f3dc", alpha = 0.55)
    axes[1].axvline(0.0, color = neutral_color, linewidth = 1.0)
    axes[1].axvline(delta, color = success_color, linewidth = 1.4, linestyle = "--")
    for y_value, (_, row) in zip(y_ni, ni_frame.iterrows()):
        decision = str(row["NI."])
        color = success_color if decision == "Yes" else warning_color if decision == "No" else neutral_color
        axes[1].hlines(y = y_value, xmin = 0.0, xmax = row["Median Δ EI"], color = color, linewidth = 2.5)
        axes[1].scatter(row["Median Δ EI"], y_value, s = 65, color = color, edgecolor = "white", linewidth = 1.0, zorder = 3)
        axes[1].text(
            x = gap_max + gap_pad * 0.15,
            y = y_value,
            s = decision,
            va = "center",
            ha = "left",
            fontsize = 8,
            color = color,
        )
    axes[1].set_xlim(gap_min - gap_pad, gap_max + gap_pad * 1.8)
    axes[1].set_yticks(y_ni)
    axes[1].set_yticklabels(ni_frame["display"], fontsize = 9)
    axes[1].set_xlabel("Median Δ EI", fontsize = 9)
    axes[1].set_title("Non-Inferiority vs Additive", fontsize = 11)
    axes[1].text(delta, len(ni_frame) - 0.2, f"δ = {delta:.2f}", color = success_color, fontsize = 8, ha = "left", va = "bottom")
    axes[1].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[1].spines[spine].set_visible(False)
    axes[1].tick_params(axis = "y", length = 0)

    mae_value = float(attr_row["Median Δ MAE"])
    attr_color = success_color if str(attr_row["Diff."]) == "Yes" else warning_color
    attr_limit = max(abs(mae_value) * 1.35, 0.2)
    axes[2].axvline(0.0, color = neutral_color, linewidth = 1.0)
    axes[2].barh([0], [mae_value], color = attr_color, height = 0.42, alpha = 0.85)
    axes[2].scatter([mae_value], [0], s = 80, color = attr_color, edgecolor = "white", linewidth = 1.0, zorder = 3)
    axes[2].set_xlim(-attr_limit * 0.2, attr_limit)
    axes[2].set_ylim(-0.9, 0.9)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Median Δ MAE", fontsize = 9)
    axes[2].set_title("Residual Attribution", fontsize = 11)
    axes[2].grid(axis = "x", alpha = 0.18, linewidth = 0.8)
    for spine in ["top", "right", "left"]:
        axes[2].spines[spine].set_visible(False)
    axes[2].text(
        x = 0.02,
        y = 0.96,
        s = (
            f"{attr_row['Comparison']}\n"
            f"Diff. = {attr_row['Diff.']}\n"
            f"Holm-adj. p = {attr_row['Holm-adj. p']}"
        ),
        transform = axes[2].transAxes,
        ha = "left",
        va = "top",
        fontsize = 8.5,
        bbox = {"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#d0d7de"},
    )
    axes[2].text(
        x = mae_value,
        y = 0.18,
        s = "favors dynamics" if mae_value >= 0 else "favors topology",
        ha = "center",
        va = "bottom",
        fontsize = 8,
        color = attr_color,
    )

    fig.suptitle(title, fontsize = 13)
    if show:
        plt.show()
    return fig, axes


def plot_decomposition_moneyshot(
    results: pd.DataFrame,
    delta_ei: float,
    delta_ci: float,
    specification_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (14.8, 3.8),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot paired decomposition coordinate fingerprints showing how each
        test specification moves relative to the original log-additive decomposition on
        efficiency and consensus.

    Args:
        results: Decomposition result table with model, group, specification,
            ei, and ci columns.
        predictions: Residual attribution prediction table retained for
            compatibility with previous notebook calls.
        delta_ei: Empirical EI margin used to contextualize paired gaps.
        delta_ci: Empirical CI margin used to contextualize paired gaps.
        specification_order: Display order for alternative specifications.
        figsize: Figure size in inches.
        title: Figure title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing.
    """

    if specification_order is None:
        specification_order = (
            "interaction",
            "interaction_joint",
            "joint",
            "invariants",
            "signatures",
        )

    required_results = {"model", "group", "specification", "ei", "ci"}
    missing_results = sorted(required_results - set(results.columns))
    if missing_results:
        raise ValueError(f"Missing required result columns: {missing_results}")

    spec_family = {
        "interaction": "Complex",
        "interaction_joint": "Complex",
        "joint": "Complex",
        "invariants": "Simple",
        "signatures": "Simple",
    }
    decomposition_orange = "#C46A1C"
    orange_shades = tuple(
        _shade_plot_color(
            hex_color = decomposition_orange,
            factor = factor,
        )
        for factor in DEFAULT_METHOD_SHADE_FACTORS
    )
    spec_colors = {
        "interaction": orange_shades[0],
        "interaction_joint": orange_shades[1],
        "joint": orange_shades[2],
        "invariants": orange_shades[0],
        "signatures": orange_shades[1],
    }
    spec_markers = {
        "interaction": "o",
        "interaction_joint": "o",
        "joint": "o",
        "invariants": "o",
        "signatures": "o",
    }

    additive = (
        results
        .query("specification == 'additive'")
        .set_index(keys = ["model", "group"])[["ei", "ci"]]
        .rename(columns = {"ei": "ei_additive", "ci": "ci_additive"})
    )

    paired_frames = []
    for specification in specification_order:
        alternative = (
            results
            .query("specification == @specification")
            .set_index(keys = ["model", "group"])[["ei", "ci"]]
            .rename(columns = {"ei": "ei_alt", "ci": "ci_alt"})
        )
        gaps = additive.join(other = alternative, how = "inner").dropna()
        if gaps.empty:
            continue
        paired_frames.append(
            gaps.assign(
                specification = specification,
                display = specification.replace("_", " ").title(),
                family = spec_family.get(specification, "Other"),
                delta_ei = lambda frame: frame["ei_alt"] - frame["ei_additive"],
                delta_ci = lambda frame: frame["ci_alt"] - frame["ci_additive"],
            ).reset_index()[["model", "group", "specification", "display", "family", "delta_ei", "delta_ci"]]
        )

    paired = pd.concat(objs = paired_frames, ignore_index = True)
    paired["specification"] = pd.Categorical(
        values = paired["specification"],
        categories = list(specification_order),
        ordered = True,
    )

    category_order = ("Simple", "Complex")

    fig, axes_raw = plt.subplots(
        nrows = 1,
        ncols = len(category_order),
        figsize = figsize,
        sharex = True,
        sharey = True,
        squeeze = False,
        constrained_layout = False,
    )
    axes = axes_raw.ravel().astype(object)
    fig.patch.set_facecolor("white")
    target_gap_inches = 14.2 * (0.985 - 0.07) * 0.28 / (4.0 + 3.0 * 0.28)
    plot_width_inches = float(figsize[0]) * (0.985 - 0.08)
    panel_width_inches = (plot_width_inches - target_gap_inches * (len(category_order) - 1)) / len(category_order)
    panel_wspace = target_gap_inches / panel_width_inches
    fig.subplots_adjust(top = 0.84, bottom = 0.16, left = 0.08, right = 0.985, wspace = panel_wspace)

    neutral_color = "#7A7A7A"
    margin_face = "#FCF7D6"
    margin_edge = "#A6A6A6"
    margin_text_color = "#9A6A1B"
    spine_color = "#9A9A9A"
    tick_color = "#8E8E8E"
    delta_annotation_color = "#8E8E8E"
    category_title_colors = {
        "Complex": decomposition_orange,
        "Simple": decomposition_orange,
    }

    axis_limits = (-0.5, 0.5)
    def _compose_reference_ticks(
        base_ticks: Sequence[float],
        delta_ticks: Mapping[float, str],
        ) -> tuple[list[float], list[str]]:

        ordered_ticks = sorted([float(tick) for tick in base_ticks] + [float(tick) for tick in delta_ticks.keys()])
        merged_ticks: list[float] = []
        for tick in ordered_ticks:
            if merged_ticks and np.isclose(tick, merged_ticks[-1], atol = 1e-9, rtol = 0.0):
                continue
            merged_ticks.append(tick)

        labels: list[str] = []
        for tick in merged_ticks:
            tick_label = None
            for delta_tick, delta_label in delta_ticks.items():
                if np.isclose(tick, float(delta_tick), atol = 1e-9, rtol = 0.0):
                    tick_label = delta_label
                    break
            labels.append(f"{tick:.2f}" if tick_label is None else tick_label)

        return merged_ticks, labels

    axis_ticks = [-0.5, -0.25, 0.0, 0.25, 0.5]
    x_ticks, x_tick_labels = _compose_reference_ticks(
        base_ticks = axis_ticks,
        delta_ticks = {
            float(delta_ei): r"$+\delta$",
        },
    )
    y_ticks, y_tick_labels = _compose_reference_ticks(
        base_ticks = axis_ticks,
        delta_ticks = {
            float(delta_ci): r"$+\delta$",
        },
    )

    for axis, category in zip(axes, category_order):
        category_specs = [
            specification
            for specification in specification_order
            if spec_family.get(specification) == category
        ]

        ni_origin_x = float(axis_limits[0])
        ni_origin_y = float(axis_limits[0])
        ni_width = float(delta_ei) - ni_origin_x
        ni_height = float(delta_ci) - ni_origin_y
        axis.add_patch(
            Rectangle(
                xy = (ni_origin_x, ni_origin_y),
                width = ni_width,
                height = ni_height,
                facecolor = margin_face,
                edgecolor = "none",
                linewidth = 0.0,
                alpha = 0.82,
                zorder = 1.85,
            )
        )
        region_label_pad = 0.03
        axis.text(
            x = ni_origin_x + region_label_pad,
            y = ni_origin_y + region_label_pad,
            s = "Non-Inferior",
            ha = "left",
            va = "bottom",
            fontsize = 6.9,
            color = margin_text_color,
            fontweight = "semibold",
            zorder = 2.05,
        )
        axis.axhline(y = float(delta_ci), color = margin_edge, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axvline(x = float(delta_ei), color = margin_edge, lw = 0.6, ls = "--", zorder = 2.0)
        axis.axhline(y = 0.0, color = neutral_color, lw = 0.6, ls = "-", zorder = 2.1)
        axis.axvline(x = 0.0, color = neutral_color, lw = 0.6, ls = "-", zorder = 2.1)

        legend_handles: list[Line2D] = []
        legend_text_colors: dict[str, str] = {}
        for specification in category_specs:
            spec_points = paired.loc[paired["specification"] == specification].copy()
            if spec_points.empty:
                continue

            color = spec_colors[specification]
            marker = spec_markers[specification]
            x = spec_points["delta_ei"].to_numpy(dtype = float)
            y = spec_points["delta_ci"].to_numpy(dtype = float)
            median_x = float(np.nanmedian(x))
            median_y = float(np.nanmedian(y))
            axis.scatter(
                x = x,
                y = y,
                s = 18,
                color = color,
                alpha = 0.26,
                edgecolors = "none",
                zorder = 3.0,
            )
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = color,
                lw = 1.1,
                alpha = 0.88,
                zorder = 4.0,
            )
            axis.scatter(
                x = [median_x],
                y = [median_y],
                s = 76,
                marker = marker,
                color = color,
                edgecolors = "none",
                linewidths = 0.0,
                zorder = 5.0,
            )

            legend_label = (
                f"{specification.replace('_', ' ').title()}: "
                f"ΔEI = {median_x:+.2f}, "
                f"ΔCI = {median_y:+.2f}"
            )
            legend_handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    color = color,
                    lw = 0.0,
                    ls = "None",
                    marker = marker,
                    markersize = 4.8,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    markeredgewidth = 0.0,
                    label = legend_label,
                )
            )
            legend_text_colors[legend_label] = color

        if legend_handles:
            legend = axis.legend(
                handles = legend_handles,
                fontsize = 6.0,
                loc = "upper left",
                bbox_to_anchor = (0.03, 1.0),
                borderaxespad = 0.0,
                framealpha = 0.9,
                edgecolor = "#D0D0D0",
                handlelength = 1.9,
                borderpad = 0.45,
                labelspacing = 0.25,
                handletextpad = 0.45,
            )
            for legend_text in legend.get_texts():
                legend_text.set_color(legend_text_colors.get(legend_text.get_text(), "#444444"))

        axis.set_title(
            label = category,
            fontsize = 10.2,
            fontweight = "semibold",
            color = category_title_colors.get(category, "#333333"),
            pad = 8,
        )
        axis.set_xlim(*axis_limits)
        axis.set_ylim(*axis_limits)
        axis.set_xticks(ticks = x_ticks)
        axis.set_xticklabels(labels = x_tick_labels)
        axis.set_yticks(ticks = y_ticks)
        axis.set_yticklabels(labels = y_tick_labels)
        axis.tick_params(axis = "both", labelsize = 7, color = tick_color)
        axis.tick_params(axis = "y", labelleft = True)
        for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
            tick_text = tick_label.get_text().lower()
            if ("delta" in tick_text) or ("δ" in tick_text):
                tick_label.set_color(delta_annotation_color)
        axis.grid(False)
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_visible(True)
        axis.spines["left"].set_color(spine_color)
        axis.spines["bottom"].set_color(spine_color)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    current_positions = [axis.get_position() for axis in axes]
    target_gap = 0.035
    panel_width = min(position.width for position in current_positions)
    panel_height = min(position.height for position in current_positions)
    panel_bottom = min(position.y0 for position in current_positions)
    total_width = (len(axes) * panel_width) + ((len(axes) - 1) * target_gap)
    left_start = 0.5 - (0.5 * total_width)
    for axis_index, axis in enumerate(axes):
        axis.set_position([
            left_start + (axis_index * (panel_width + target_gap)),
            panel_bottom,
            panel_width,
            panel_height,
        ])

    x_label_pad_points = 8.0
    x_tick_label_points = 9.0
    x_tick_pad_points = 3.5
    x_label_clearance_points = x_label_pad_points + x_tick_label_points + x_tick_pad_points
    x_label_y = panel_bottom - ((x_label_clearance_points / 72.0) / float(figsize[1]))
    axes[0].set_ylabel("Δ CI (Ablated - Original)", fontsize = 9, labelpad = 8)
    fig.text(
        x = left_start + (0.5 * total_width),
        y = x_label_y,
        s = "Δ EI (Ablated - Original)",
        ha = "center",
        va = "top",
        fontsize = 9,
    )
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(["Helvetica", "Arial", "sans-serif"])
        text_artist.set_fontsize(text_artist.get_fontsize() + 2.0)

    if show:
        plt.show()
    return fig, axes


## ----------------------------------------------------------------------------
## residual handoff money shot
## ----------------------------------------------------------------------------
DEFAULT_DOMAIN_PALETTE = {
    "Earth & Physical Sciences":       "#C07D3A",
    "Life Sciences & Medicine":        "#3A7D55",
    "Technology & Information":        "#2C6E91",
    "Trade & Institutions":            "#B8860B",
    "Transportation & Infrastructure": "#8B3A3A",
}
DEFAULT_DOMAIN_LABELS = {
    "Earth & Physical Sciences":       "Earth\n& Physical",
    "Life Sciences & Medicine":        "Life\nSciences",
    "Technology & Information":        "Technology\n& Information",
    "Trade & Institutions":            "Trade &\nInstitutions",
    "Transportation & Infrastructure": "Transport &\nInfrastructure",
}
DEFAULT_MODEL_LABELS = {
    "linear_quantile":  "Linear Quantile",
    "linear_convex":    "Linear Convex",
    "linear_laws":      "Linear Laws",
    "forest_quantile":  "Forest Quantile",
    "boosted_quantile": "Boosted Quantile",
    "xgboost_quantile": "XGBoost Quantile",
    "neural_quantile":  "Neural Quantile",
    "neural_expectile": "Neural Expectile",
    "neural_convex":    "Neural Convex",
}


def plot_universality_domain_columns(
    data: pd.DataFrame,
    predictions: Mapping[tuple[str, str], np.ndarray],
    target: str,
    regime_key: str = "domain_logo",
    paradigm_order: Sequence[str] | None = None,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    paradigm_palette: Mapping[str, str] | None = None,
    paradigm_markers: Mapping[str, str] | None = None,
    paradigm_labels: Mapping[str, str] | None = None,
    domain_column: str = "domain",
    discipline_column: str = "discipline",
    axis_limits: tuple[float, float] = (0.0, 20.0),
    diagonal_color: str = "#AAAAAA",
    figsize: tuple[float, float] | None = None,
    title: str = "Domain LOGO: Universality by Domain",
    fig: Figure | None = None,
    axes: Sequence[Axes] | None = None,
    add_title: bool = True,
    add_shared_labels: bool = True,
    add_legend: bool = True,
    show: bool = True,
    ei_badge_position: str = "bottom",
    ei_badge_background: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot one scatter panel per domain, collapsing learning paradigms into
        one discipline-median point within each domain panel.

    Args:
        data: Processed data with target, domain, and discipline columns.
        predictions: Mapping from (paradigm, regime_key) to predicted frontier
            arrays aligned to data rows.
        target: Target capacity column in data.
        regime_key: Prediction regime to render.
        paradigm_order: Ordered learning paradigm labels.
        domain_palette: Mapping from domain to display color.
        domain_labels: Mapping from domain to display label.
        paradigm_palette: Deprecated; retained for API compatibility.
        paradigm_markers: Deprecated; retained for API compatibility.
        paradigm_labels: Deprecated; retained for API compatibility.
        domain_column: Domain column name.
        discipline_column: Discipline column name.
        axis_limits: Shared x/y axis limits.
        diagonal_color: Color for the equality diagonal reference line.
        figsize: Optional figure size in inches.
        title: Figure title.
        fig: Optional existing figure for composable rendering.
        axes: Optional existing axes, one per plotted domain.
        add_title: Whether to add the standalone figure title.
        add_shared_labels: Whether to add standalone shared axis labels.
        add_legend: Whether to add the standalone legend.
        show: Whether to call plt.show().
        ei_badge_position: Vertical placement for EI badges; supported values
            are "bottom" and "top".
        ei_badge_background: Whether to draw the rounded white badge behind EI text.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        ValueError: If required columns are missing or predictions are invalid.
    """

    required_columns = {target, domain_column, discipline_column}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    paradigm_order = list(paradigm_order)
    if not paradigm_order:
        raise ValueError("paradigm_order must contain at least one paradigm")
    if ei_badge_position not in {"bottom", "top"}:
        raise ValueError("ei_badge_position must be 'bottom' or 'top'")

    domain_palette = dict(domain_palette or DEFAULT_DOMAIN_PALETTE)
    domain_labels = dict(domain_labels or DEFAULT_DOMAIN_LABELS)

    y_true = np.asarray(_log_transformer(data[target]), dtype = float)
    domain_values = data[domain_column].to_numpy()
    discipline_values = data[discipline_column].to_numpy()
    domain_order = [
        domain_name for domain_name in domain_palette
        if domain_name in set(data[domain_column].dropna())
    ]
    domain_order.extend(
        sorted(
            domain_name for domain_name in data[domain_column].dropna().unique()
            if domain_name not in set(domain_order)
        )
    )
    if not domain_order:
        raise ValueError("No domains available for plotting")

    prediction_arrays: dict[str, np.ndarray] = dict()
    for paradigm in paradigm_order:
        prediction_key = (paradigm, regime_key)
        if prediction_key not in predictions:
            raise ValueError(f"Missing predictions for {prediction_key}")
        prediction_values = np.asarray(predictions[prediction_key], dtype = float)
        if prediction_values.shape[0] != len(data):
            raise ValueError(
                f"Predictions for {prediction_key} have length {prediction_values.shape[0]}, "
                f"expected {len(data)}"
            )
        prediction_arrays[paradigm] = prediction_values
    collapsed_prediction_values = pd.DataFrame(
        data = {paradigm: prediction_arrays[paradigm] for paradigm in paradigm_order}
    ).median(axis = 1, skipna = True).to_numpy(dtype = float)

    if figsize is None:
        figsize = (len(domain_order) * 2.35 + 0.85, 3.15)

    if axes is None:
        fig, axes_raw = plt.subplots(
            nrows = 1,
            ncols = len(domain_order),
            figsize = figsize,
            gridspec_kw = {"wspace": 0.08},
            squeeze = False,
        )
        axes = axes_raw.ravel().astype(object)
        fig.subplots_adjust(top = 0.78, left = 0.075, right = 0.985, bottom = 0.235)
    else:
        axes = np.asarray(list(axes), dtype = object).ravel()
        if len(axes) != len(domain_order):
            raise ValueError(
                f"Expected {len(domain_order)} axes for domain columns, got {len(axes)}"
            )
        if fig is None:
            fig = axes[0].figure

    axis_low, axis_high = axis_limits
    tick_values = np.linspace(start = axis_low, stop = axis_high, num = 5)
    diagonal_values = np.array([axis_low, axis_high], dtype = float)

    for axis_index, (axis, domain_name) in enumerate(zip(axes, domain_order)):
        domain_mask = domain_values == domain_name
        domain_color = domain_palette.get(domain_name, "#777777")
        domain_ei_values = list()
        discipline_order = sorted(
            np.unique(discipline_values[domain_mask]),
            key = lambda discipline_name: np.nanmedian(
                y_true[(discipline_values == discipline_name) & np.isfinite(y_true)]
            ),
        )

        axis.set_xlim(left = axis_low, right = axis_high)
        axis.set_ylim(bottom = axis_low, top = axis_high)
        axis.set_xticks(ticks = tick_values)
        axis.set_yticks(ticks = tick_values)
        axis.fill_between(
            x = diagonal_values,
            y1 = diagonal_values,
            y2 = axis_high,
            color = "#E6F4EA",
            zorder = 0,
            clip_on = True,
            antialiased = False,
        )
        axis.plot(
            diagonal_values,
            diagonal_values,
            color = diagonal_color,
            lw = UNIVERSALITY_REFERENCE_LINE_WIDTH,
            ls = "--",
            zorder = 1,
        )

        domain_fit_x: list[float] = list()
        domain_fit_y: list[float] = list()
        for discipline_name in discipline_order:
            discipline_mask = (discipline_values == discipline_name) & domain_mask
            observed_mask = discipline_mask & np.isfinite(y_true)
            if not observed_mask.any():
                continue

            observed_median = float(np.nanmedian(y_true[observed_mask]))
            frontier_medians = list()
            for paradigm in paradigm_order:
                prediction_values = prediction_arrays[paradigm]
                prediction_mask = discipline_mask & np.isfinite(prediction_values)
                if not prediction_mask.any():
                    continue

                frontier_medians.append(float(np.nanmedian(prediction_values[prediction_mask])))
            if not frontier_medians:
                continue

            frontier_median = float(np.nanmedian(frontier_medians))
            domain_fit_x.append(observed_median)
            domain_fit_y.append(frontier_median)
            axis.plot(
                [observed_median],
                [frontier_median],
                marker = "o",
                markersize = 5.0,
                markerfacecolor = domain_color,
                markeredgecolor = domain_color,
                markeredgewidth = 0.0,
                linewidth = 0,
                zorder = 4,
            )

        valid = domain_mask & np.isfinite(y_true) & np.isfinite(collapsed_prediction_values)
        if int(np.sum(valid)) >= 2:
            domain_ei_values.append(
                _efficiency_index(
                    y_true = y_true[valid],
                    y_pred = collapsed_prediction_values[valid],
                )
            )

        if len(domain_fit_x) >= 2:
            fit_x = np.asarray(domain_fit_x, dtype = float)
            fit_y = np.asarray(domain_fit_y, dtype = float)
            median_offset = float(np.nanmedian(fit_y - fit_x))
            if np.isfinite(median_offset):
                line_low = max(axis_low, axis_low - median_offset)
                line_high = min(axis_high, axis_high - median_offset)
                if line_low < line_high:
                    fit_line_x = np.array([line_low, line_high], dtype = float)
                    fit_line_y = fit_line_x + median_offset
                    axis.plot(
                        fit_line_x,
                        fit_line_y,
                        color = domain_color,
                        lw = UNIVERSALITY_REFERENCE_LINE_WIDTH,
                        ls = "--",
                        alpha = 0.55,
                        zorder = 2,
                    )

        axis.set_title(
            label = domain_labels.get(domain_name, domain_name),
            fontsize = 9.1,
            fontweight = "bold",
            pad = 6,
            color = domain_color,
        )
        axis.set_aspect(aspect = "equal", adjustable = "box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_linewidth(0.45)
        axis.tick_params(labelsize = 6.4, pad = 1.6, length = 2.4)
        if domain_ei_values:
            badge_y = 0.045 if ei_badge_position == "bottom" else 0.955
            badge_va = "bottom" if ei_badge_position == "bottom" else "top"
            badge_bbox = None
            if ei_badge_background:
                badge_bbox = {
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                }
            axis.text(
                x = 0.50,
                y = badge_y,
                s = f"EI = {float(np.median(domain_ei_values)):.2f}",
                transform = axis.transAxes,
                ha = "center",
                va = badge_va,
                fontsize = 6.3,
                fontweight = "bold",
                color = "#111111",
                bbox = badge_bbox,
                zorder = 6,
            )
        if axis_index > 0:
            axis.set_yticklabels(labels = [])

    if add_title:
        fig.suptitle(
            t = title,
            x = 0.50,
            y = 0.94,
            fontsize = 11.5,
            fontweight = "semibold",
            color = "#222222",
        )
    if add_shared_labels:
        fig.text(
            x = 0.50,
            y = 0.115,
            s = "Observed Log-Capacity (x)",
            ha = "center",
            va = "bottom",
            fontsize = 10.5,
            fontweight = "semibold",
            color = "#222222",
        )
        fig.text(
            x = 0.020,
            y = 0.51,
            s = "Estimated Capacity Frontier (y)",
            ha = "center",
            va = "center",
            fontsize = 10.5,
            fontweight = "semibold",
            rotation = 90,
            color = "#222222",
        )

    legend_handles = [
        Line2D(
            xdata = [],
            ydata = [],
            color = "#777777",
            marker = "o",
            linestyle = "None",
            markersize = 5.2,
            markeredgecolor = "none",
            label = "Median across paradigms",
        )
    ]
    if add_legend:
        fig.legend(
            handles = legend_handles,
            loc = "lower center",
            ncol = 4,
            fontsize = 7.0,
            framealpha = 0.92,
            edgecolor = "#D0D0D0",
            bbox_to_anchor = (0.50, 0.010),
        )

    if show:
        plt.show()
    return fig, axes


def load_universality_paradigm_predictions_from_cache(
    data: pd.DataFrame,
    models: Mapping[str, object],
    random_state: int,
    cache_path: str | Path,
    regime_key: str = "domain_logo",
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    ) -> dict[tuple[str, str], np.ndarray]:

    """
    Desc:
        Load cached universality predictions and aggregate base learners by
        learning paradigm for a single cached prediction regime.

    Args:
        data: Processed data aligned to cached prediction rows.
        models: Mapping of model names to estimator objects.
        random_state: Random state expected in the cache metadata.
        cache_path: Path to universality_paradigm_preds.pkl.
        regime_key: Cached prediction regime to aggregate.
        model_to_paradigm: Mapping from model name to paradigm.
        paradigm_order: Ordered learning paradigm labels.

    Returns:
        Mapping from (paradigm, regime_key) to median paradigm predictions.

    Raises:
        FileNotFoundError: If cache_path does not exist.
        RuntimeError: If cache metadata or cache entries are invalid.
    """

    if model_to_paradigm is None:
        model_to_paradigm = DEFAULT_MODEL_TO_PARADIGM
    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    paradigm_order = list(paradigm_order)

    resolved_cache_path = Path(cache_path)
    if not resolved_cache_path.exists():
        raise FileNotFoundError(
            f"Cached predictions not found at {resolved_cache_path}. "
            "Run the universality COMPUTE cell once, then rerun this render cell."
        )

    with open(file = resolved_cache_path, mode = "rb") as file_handle:
        cached = pickle.load(file_handle)

    metadata = cached.get("metadata", cached)
    if metadata.get("n_obs") != len(data) or metadata.get("random_state") != random_state:
        raise RuntimeError(
            "Cached predictions were produced for a different data shape or RANDOM_STATE. "
            "Rerun notebooks/transfer.ipynb through its post-processing cell."
        )

    model_names = list(models.keys())
    prediction_keys = {
        "domain_logo": "predicts_dict_domain",
        "disc_logo": "predicts_dict_disc",
        "kfold_5": "predicts_dict_5fold",
        "kfold_10": "predicts_dict_10fold",
    }
    if "viz_preds_par" in cached:
        raw_predictions = cached["viz_preds_par"]
    else:
        prediction_key = prediction_keys.get(regime_key)
        prediction_map = cached.get(prediction_key, {}) if prediction_key is not None else {}
        raw_predictions = {
            (model_name, regime_key): prediction
            for model_name, prediction in prediction_map.items()
        }
    paradigm_to_models = {paradigm: [] for paradigm in paradigm_order}
    for model_name in model_names:
        paradigm = model_to_paradigm.get(model_name)
        if paradigm is not None and paradigm in paradigm_to_models:
            paradigm_to_models[paradigm].append(model_name)

    missing_keys = [
        (model_name, regime_key)
        for model_name in model_names
        if (model_name, regime_key) not in raw_predictions
    ]
    if missing_keys:
        preview_keys = missing_keys[:5]
        raise RuntimeError(
            f"Cached predictions are missing {len(missing_keys)} model/regime entries; "
            f"examples: {preview_keys}. Rerun notebooks/transfer.ipynb through "
            "its post-processing cell."
        )

    aggregated_predictions: dict[tuple[str, str], np.ndarray] = dict()
    for paradigm, member_models in paradigm_to_models.items():
        if not member_models:
            continue
        stacked_predictions = np.vstack(
            [raw_predictions[(model_name, regime_key)] for model_name in member_models]
        )
        aggregated_predictions[(paradigm, regime_key)] = np.nanmedian(
            stacked_predictions,
            axis = 0,
        )

    return aggregated_predictions


def plot_universality_domain_columns_from_cache(
    data: pd.DataFrame,
    models: Mapping[str, object],
    target: str,
    random_state: int,
    cache_path: str | Path,
    regime_key: str = "domain_logo",
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    **plot_kwargs: object,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Load cached universality predictions, aggregate base learners by
        learning paradigm, and render collapsed domain-column scatter panels.

    Args:
        data: Processed data aligned to cached prediction rows.
        models: Mapping of model names to estimator objects.
        target: Target capacity column in data.
        random_state: Random state expected in the cache metadata.
        cache_path: Path to universality_paradigm_preds.pkl.
        regime_key: Cached prediction regime to render.
        model_to_paradigm: Mapping from model name to paradigm.
        paradigm_order: Ordered learning paradigm labels.
        plot_kwargs: Additional keyword arguments passed to the plotter.

    Returns:
        Tuple containing the figure and axes array.

    Raises:
        FileNotFoundError: If cache_path does not exist.
        RuntimeError: If cache metadata or cache entries are invalid.
    """

    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    paradigm_order = list(paradigm_order)
    aggregated_predictions = load_universality_paradigm_predictions_from_cache(
        data = data,
        models = models,
        random_state = random_state,
        cache_path = cache_path,
        regime_key = regime_key,
        model_to_paradigm = model_to_paradigm,
        paradigm_order = paradigm_order,
    )

    return plot_universality_domain_columns(
        data = data,
        predictions = aggregated_predictions,
        target = target,
        regime_key = regime_key,
        paradigm_order = paradigm_order,
        **plot_kwargs,
    )


def _bootstrap_median_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
    ) -> tuple[float, float, float]:

    """
    Desc: Percentile bootstrap CI for the median.
    Args:
        values: 1D array of finite values.
        n_bootstrap: Number of bootstrap resamples.
        rng: NumPy random generator.
        confidence: Confidence level for the interval.
    Returns:
        Tuple of (median, lower_bound, upper_bound).
    Raises:
        ValueError: If values is empty.
    """

    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("values must contain at least one finite element")

    median = float(np.median(values))
    if values.size < 2:
        return median, median, median

    indices = rng.integers(low = 0, high = values.size, size = (n_bootstrap, values.size))
    medians = np.median(values[indices], axis = 1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(medians, alpha))
    upper = float(np.quantile(medians, 1.0 - alpha))
    return median, lower, upper


def plot_residual_handoff_universality(
    predictions: pd.DataFrame,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    domain_order: Sequence[str] | None = None,
    model_order: Sequence[str] | None = None,
    n_bootstrap: int = 4000,
    random_state: int = 42,
    figsize: tuple[float, float] = (13.5, 11.2),
    title: str = "Process signatures explain what topology leaves behind",
    subtitle: str = (
        "After fitting the capacity stage $C(X)$, the residual slack $s = y^* - \\hat{C}(X)$ is\n"
        "predicted more accurately from process signatures $Z$ than from the same topology features $X$\n"
        "that produced it — universally, across every learning paradigm and every scientific domain."
    ),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Plot a single figure summarizing the residual-handoff universality
        finding: after fitting C(X), the residual slack is better predicted
        from process features Z than from topology features X, robustly
        across models and domains.

    Args:
        predictions: Residual attribution prediction table with columns
            model, residual_features, dataset, group, abs_error.
        domain_palette: Mapping from domain label to color.
        domain_labels: Mapping from domain label to short display label.
        model_labels: Mapping from model identifier to display label.
        domain_order: Ordering of domains in summary panels.
        model_order: Ordering of models in the per-model forest.
        n_bootstrap: Bootstrap resample count for CI estimation.
        random_state: Seed for the bootstrap random generator.
        figsize: Figure size in inches.
        title: Figure title.
        subtitle: Figure subtitle shown beneath the title.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple containing the figure and a flat axes array.

    Raises:
        ValueError: If required columns are missing or only one residual
            feature source is present.
    """

    required = {"model", "residual_features", "dataset", "group", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sources = set(predictions["residual_features"].astype(str).unique())
    if not {"X_to_slack", "Z_to_slack"}.issubset(sources):
        raise ValueError(
            "predictions must include both X_to_slack and Z_to_slack residual_features"
        )

    domain_palette = dict(domain_palette or DEFAULT_DOMAIN_PALETTE)
    domain_labels = dict(domain_labels or DEFAULT_DOMAIN_LABELS)
    model_labels = dict(model_labels or DEFAULT_MODEL_LABELS)

    aggregated = (
        predictions
        .groupby(by = ["model", "dataset", "group", "residual_features"], observed = True)["abs_error"]
        .mean()
        .reset_index()
        .pivot_table(
            index = ["model", "dataset", "group"],
            columns = "residual_features",
            values = "abs_error",
            observed = True,
        )
        .rename_axis(columns = None)
        .reset_index()
        .dropna(subset = ["X_to_slack", "Z_to_slack"])
    )
    aggregated = aggregated.loc[
        (aggregated["X_to_slack"] > 0.0) & (aggregated["Z_to_slack"] > 0.0)
    ].copy()
    aggregated["log_ratio"] = np.log10(aggregated["X_to_slack"] / aggregated["Z_to_slack"])
    aggregated["favors_dynamics"] = aggregated["log_ratio"] > 0.0

    if domain_order is None:
        domain_order = list(
            dict.fromkeys(
                list(domain_palette.keys())
                + aggregated["group"].astype(str).unique().tolist()
            )
        )
        domain_order = [domain for domain in domain_order if domain in set(aggregated["group"])]
    if model_order is None:
        model_order = list(
            dict.fromkeys(
                list(model_labels.keys())
                + aggregated["model"].astype(str).unique().tolist()
            )
        )
        model_order = [model for model in model_order if model in set(aggregated["model"])]

    rng = np.random.default_rng(seed = random_state)

    domain_summary_rows = []
    for domain in domain_order:
        values = aggregated.loc[aggregated["group"] == domain, "log_ratio"].to_numpy(dtype = float)
        if values.size == 0:
            continue
        median, lower, upper = _bootstrap_median_ci(
            values = values,
            n_bootstrap = n_bootstrap,
            rng = rng,
        )
        domain_summary_rows.append({
            "domain": domain,
            "median": median,
            "lower": lower,
            "upper": upper,
            "share_dynamics": float((values > 0.0).mean()),
            "n": int(values.size),
        })
    domain_summary = pd.DataFrame(domain_summary_rows)

    model_summary_rows = []
    for model in model_order:
        values = aggregated.loc[aggregated["model"] == model, "log_ratio"].to_numpy(dtype = float)
        if values.size == 0:
            continue
        median, lower, upper = _bootstrap_median_ci(
            values = values,
            n_bootstrap = n_bootstrap,
            rng = rng,
        )
        model_summary_rows.append({
            "model": model,
            "median": median,
            "lower": lower,
            "upper": upper,
            "share_dynamics": float((values > 0.0).mean()),
            "n": int(values.size),
        })
    model_summary = pd.DataFrame(model_summary_rows)

    overall_median, overall_lower, overall_upper = _bootstrap_median_ci(
        values = aggregated["log_ratio"].to_numpy(dtype = float),
        n_bootstrap = n_bootstrap,
        rng = rng,
    )
    n_total = int(len(aggregated))
    n_dynamics = int(aggregated["favors_dynamics"].sum())
    share_dynamics = n_dynamics / n_total if n_total else float("nan")

    background_color = "#fbfaf6"
    grid_color = "#e3ddd0"
    diagonal_color = "#3a3a3a"
    dynamics_color = "#1f5e3a"
    topology_color = "#a14a2a"
    neutral_color = "#6c757d"
    headline_color = "#1f5e3a"

    fig = plt.figure(figsize = figsize, constrained_layout = False)
    fig.patch.set_facecolor(background_color)

    grid = fig.add_gridspec(
        nrows = 3,
        ncols = 2,
        height_ratios = [0.55, 2.6, 1.05],
        width_ratios = [2.7, 1.0],
        left = 0.07,
        right = 0.97,
        top = 0.80,
        bottom = 0.07,
        hspace = 0.45,
        wspace = 0.22,
    )
    kde_axis = fig.add_subplot(grid[0, 0])
    scatter_axis = fig.add_subplot(grid[1, 0])
    domain_axis = fig.add_subplot(grid[1, 1])
    model_axis = fig.add_subplot(grid[2, :])
    axes = np.array([kde_axis, scatter_axis, domain_axis, model_axis], dtype = object)

    for axis in axes:
        axis.set_facecolor(background_color)

    ## scatter ----------------------------------------------------------------
    x_values = aggregated["X_to_slack"].to_numpy(dtype = float)
    y_values = aggregated["Z_to_slack"].to_numpy(dtype = float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
    log_lower = float(np.log10(np.minimum(x_values[finite_mask], y_values[finite_mask]).min()))
    log_upper = float(np.log10(np.maximum(x_values[finite_mask], y_values[finite_mask]).max()))
    log_pad = max(0.15, 0.08 * (log_upper - log_lower))
    diagonal_lo = log_lower - log_pad
    diagonal_hi = log_upper + log_pad

    diagonal_x = np.linspace(start = diagonal_lo, stop = diagonal_hi, num = 200)
    scatter_axis.fill_between(
        x = 10.0 ** diagonal_x,
        y1 = 10.0 ** diagonal_x,
        y2 = 10.0 ** diagonal_hi,
        color = topology_color,
        alpha = 0.10,
        zorder = 0,
    )
    scatter_axis.fill_between(
        x = 10.0 ** diagonal_x,
        y1 = 10.0 ** diagonal_lo,
        y2 = 10.0 ** diagonal_x,
        color = dynamics_color,
        alpha = 0.12,
        zorder = 0,
    )
    scatter_axis.plot(
        10.0 ** diagonal_x,
        10.0 ** diagonal_x,
        color = diagonal_color,
        linewidth = 1.2,
        linestyle = "--",
        alpha = 0.7,
        zorder = 1,
    )

    for domain in domain_order:
        spec_mask = aggregated["group"] == domain
        if not spec_mask.any():
            continue
        scatter_axis.scatter(
            aggregated.loc[spec_mask, "X_to_slack"],
            aggregated.loc[spec_mask, "Z_to_slack"],
            s = 32,
            color = domain_palette.get(domain, "#555555"),
            edgecolor = "white",
            linewidth = 0.6,
            alpha = 0.88,
            label = domain,
            zorder = 3,
        )

    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.set_xlim(10.0 ** diagonal_lo, 10.0 ** diagonal_hi)
    scatter_axis.set_ylim(10.0 ** diagonal_lo, 10.0 ** diagonal_hi)
    scatter_axis.set_aspect(aspect = "equal", adjustable = "box")
    scatter_axis.set_xlabel(
        "MAE predicting residual from topology  $X \\to s$",
        fontsize = 11,
    )
    scatter_axis.set_ylabel(
        "MAE predicting residual from process  $Z \\to s$",
        fontsize = 11,
    )
    scatter_axis.grid(which = "both", color = grid_color, alpha = 0.55, linewidth = 0.7)
    scatter_axis.tick_params(which = "both", labelsize = 9)
    for spine in ["top", "right"]:
        scatter_axis.spines[spine].set_visible(False)

    scatter_axis.text(
        x = 0.96,
        y = 0.55,
        s = "structure\nwins",
        transform = scatter_axis.transAxes,
        ha = "right",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = topology_color,
        alpha = 0.55,
    )
    scatter_axis.text(
        x = 0.05,
        y = 0.10,
        s = "process\nwins",
        transform = scatter_axis.transAxes,
        ha = "left",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = dynamics_color,
        alpha = 0.65,
    )

    headline = (
        f"{n_dynamics} of {n_total} matched (model × dataset) units favor process\n"
        f"({share_dynamics:.0%}; "
        f"median $\\log_{{10}}\\!\\left(\\mathrm{{MAE}}_X / \\mathrm{{MAE}}_Z\\right)$ = "
        f"{overall_median:+.2f}, 95% CI [{overall_lower:+.2f}, {overall_upper:+.2f}])"
    )
    scatter_axis.text(
        x = 0.03,
        y = 0.97,
        s = headline,
        transform = scatter_axis.transAxes,
        ha = "left",
        va = "top",
        fontsize = 10.5,
        color = headline_color,
        bbox = {
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": headline_color,
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        zorder = 6,
    )

    legend = scatter_axis.legend(
        loc = "lower right",
        bbox_to_anchor = (1.0, 0.18),
        frameon = True,
        facecolor = "white",
        edgecolor = "#d0d7de",
        framealpha = 0.95,
        fontsize = 8.5,
        title = "Domain",
        title_fontsize = 9,
        markerscale = 1.3,
    )
    legend.get_frame().set_linewidth(0.7)

    ## kde strip --------------------------------------------------------------
    log_ratio_values = aggregated["log_ratio"].to_numpy(dtype = float)
    kde_x_max = float(np.max(np.abs(log_ratio_values))) * 1.05 + 0.1
    kde_x_min = -kde_x_max
    grid_x = np.linspace(start = kde_x_min, stop = kde_x_max, num = 512)
    bandwidth = max(
        0.08,
        1.06 * float(np.std(log_ratio_values)) * (log_ratio_values.size ** (-1.0 / 5.0)),
    )
    diff = grid_x[None, :] - log_ratio_values[:, None]
    kernel = np.exp(-0.5 * (diff / bandwidth) ** 2.0) / (bandwidth * np.sqrt(2.0 * np.pi))
    density = kernel.mean(axis = 0)

    kde_axis.fill_between(
        x = grid_x,
        y1 = 0.0,
        y2 = np.where(grid_x <= 0.0, density, 0.0),
        color = topology_color,
        alpha = 0.30,
        linewidth = 0,
    )
    kde_axis.fill_between(
        x = grid_x,
        y1 = 0.0,
        y2 = np.where(grid_x >= 0.0, density, 0.0),
        color = dynamics_color,
        alpha = 0.32,
        linewidth = 0,
    )
    kde_axis.plot(grid_x, density, color = "#2c2c2c", linewidth = 1.2)
    kde_axis.axvline(x = 0.0, color = neutral_color, linewidth = 1.0)
    kde_axis.axvline(
        x = overall_median,
        color = headline_color,
        linewidth = 1.6,
        linestyle = "--",
    )
    ymax = float(density.max()) * 1.18
    kde_axis.set_ylim(0.0, ymax)
    kde_axis.set_xlim(kde_x_min, kde_x_max)
    kde_axis.set_yticks([])
    kde_axis.set_xticks([])
    kde_axis.set_xlabel("")
    for spine in ["top", "right", "left"]:
        kde_axis.spines[spine].set_visible(False)
    kde_axis.spines["bottom"].set_color(neutral_color)
    kde_axis.text(
        x = kde_x_min + 0.04 * (kde_x_max - kde_x_min),
        y = ymax * 0.92,
        s = "structure",
        ha = "left",
        va = "top",
        fontsize = 9.5,
        color = topology_color,
    )
    kde_axis.text(
        x = kde_x_max - 0.04 * (kde_x_max - kde_x_min),
        y = ymax * 0.92,
        s = "process",
        ha = "right",
        va = "top",
        fontsize = 9.5,
        color = dynamics_color,
    )
    structure_share = 1.0 - share_dynamics
    kde_axis.text(
        x = (kde_x_min + 0.0) / 2.0,
        y = ymax * 0.50,
        s = f"{structure_share:.0%}\nof units",
        ha = "center",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = topology_color,
    )
    kde_axis.text(
        x = (kde_x_max + 0.0) / 2.0,
        y = ymax * 0.50,
        s = f"{share_dynamics:.0%}\nof units",
        ha = "center",
        va = "center",
        fontsize = 11,
        fontweight = "bold",
        color = dynamics_color,
    )
    kde_axis.text(
        x = overall_median,
        y = ymax * 1.12,
        s = f"median = {overall_median:+.2f}",
        ha = "center",
        va = "bottom",
        fontsize = 9,
        color = headline_color,
    )
    kde_axis.set_title(
        label = "Where each (model × dataset) unit lands on the residual-advantage axis  $\\log_{10}(\\mathrm{MAE}_X / \\mathrm{MAE}_Z)$",
        fontsize = 10.5,
        loc = "left",
        pad = 18,
    )

    ## per-domain forest ------------------------------------------------------
    n_domains = len(domain_summary)
    y_positions = np.arange(n_domains)[::-1]
    forest_extent = max(
        float(np.abs(domain_summary[["lower", "upper", "median"]].to_numpy()).max()),
        float(np.abs(model_summary[["median"]].to_numpy()).max()) + 0.15,
    )
    forest_x_max = min(forest_extent, 0.6) * 1.15 + 0.05
    forest_x_min = -forest_x_max

    domain_axis.axvspan(
        xmin = forest_x_min,
        xmax = 0.0,
        color = topology_color,
        alpha = 0.06,
        zorder = 0,
    )
    domain_axis.axvspan(
        xmin = 0.0,
        xmax = forest_x_max,
        color = dynamics_color,
        alpha = 0.07,
        zorder = 0,
    )
    domain_axis.axvline(x = 0.0, color = neutral_color, linewidth = 1.0, zorder = 1)

    for y_pos, (_, row) in zip(y_positions, domain_summary.iterrows()):
        color = domain_palette.get(str(row["domain"]), "#555555")
        clipped_lower = max(float(row["lower"]), forest_x_min)
        clipped_upper = min(float(row["upper"]), forest_x_max)
        domain_axis.hlines(
            y = y_pos,
            xmin = clipped_lower,
            xmax = clipped_upper,
            color = color,
            linewidth = 3.2,
            alpha = 0.9,
            zorder = 2,
        )
        if float(row["lower"]) < forest_x_min:
            domain_axis.scatter(forest_x_min + 0.015, y_pos, marker = "<", s = 28, color = color, zorder = 3)
        if float(row["upper"]) > forest_x_max:
            domain_axis.scatter(forest_x_max - 0.015, y_pos, marker = ">", s = 28, color = color, zorder = 3)
        marker_color = dynamics_color if row["median"] > 0.0 else topology_color
        domain_axis.scatter(
            row["median"],
            y_pos,
            s = 110,
            marker = "o",
            color = color,
            edgecolor = marker_color,
            linewidth = 1.6,
            zorder = 4,
        )
        domain_axis.text(
            x = forest_x_max * 0.97,
            y = y_pos,
            s = f"{row['share_dynamics']:.0%}",
            ha = "right",
            va = "center",
            fontsize = 8.5,
            color = "#3a3a3a",
        )

    domain_axis.set_yticks(y_positions)
    domain_axis.set_yticklabels(
        labels = [domain_labels.get(str(domain), str(domain)) for domain in domain_summary["domain"]],
        fontsize = 9,
    )
    domain_axis.set_xlim(forest_x_min, forest_x_max)
    domain_axis.set_ylim(-0.7, n_domains - 0.3)
    domain_axis.set_xlabel("median advantage (95% CI)", fontsize = 9.5)
    domain_axis.set_title("Every domain", fontsize = 11, loc = "left", pad = 6)
    domain_axis.tick_params(axis = "x", labelsize = 9)
    domain_axis.tick_params(axis = "y", length = 0)
    domain_axis.grid(axis = "x", color = grid_color, alpha = 0.55, linewidth = 0.7)
    for spine in ["top", "right", "left"]:
        domain_axis.spines[spine].set_visible(False)

    ## per-model forest -------------------------------------------------------
    n_models = len(model_summary)
    x_positions = np.arange(n_models)
    model_axis.axhspan(ymin = forest_x_min, ymax = 0.0, color = topology_color, alpha = 0.06, zorder = 0)
    model_axis.axhspan(ymin = 0.0, ymax = forest_x_max, color = dynamics_color, alpha = 0.07, zorder = 0)
    model_axis.axhline(y = 0.0, color = neutral_color, linewidth = 1.0, zorder = 1)

    paradigm_color = {
        "linear_quantile":  "#3b6e8f",
        "linear_convex":    "#3b6e8f",
        "linear_laws":      "#3b6e8f",
        "forest_quantile":  "#7a5a9a",
        "boosted_quantile": "#7a5a9a",
        "xgboost_quantile": "#7a5a9a",
        "neural_quantile":  "#a3623a",
        "neural_expectile": "#a3623a",
        "neural_convex":    "#a3623a",
    }

    for x_pos, (_, row) in zip(x_positions, model_summary.iterrows()):
        color = paradigm_color.get(str(row["model"]), "#555555")
        clipped_lower = max(float(row["lower"]), forest_x_min)
        clipped_upper = min(float(row["upper"]), forest_x_max)
        model_axis.vlines(
            x = x_pos,
            ymin = clipped_lower,
            ymax = clipped_upper,
            color = color,
            linewidth = 3.0,
            alpha = 0.9,
            zorder = 2,
        )
        if float(row["lower"]) < forest_x_min:
            model_axis.scatter(x_pos, forest_x_min + 0.02, marker = "v", s = 28, color = color, zorder = 3)
        if float(row["upper"]) > forest_x_max:
            model_axis.scatter(x_pos, forest_x_max - 0.02, marker = "^", s = 28, color = color, zorder = 3)
        marker_edge = dynamics_color if row["median"] > 0.0 else topology_color
        model_axis.scatter(
            x_pos,
            row["median"],
            s = 110,
            marker = "o",
            color = color,
            edgecolor = marker_edge,
            linewidth = 1.6,
            zorder = 4,
        )
        model_axis.text(
            x = x_pos,
            y = forest_x_max * 0.93,
            s = f"{row['share_dynamics']:.0%}",
            ha = "center",
            va = "top",
            fontsize = 8.5,
            color = "#3a3a3a",
        )

    model_axis.set_xticks(x_positions)
    model_axis.set_xticklabels(
        labels = [model_labels.get(str(name), str(name)) for name in model_summary["model"]],
        fontsize = 9,
        rotation = 22,
        ha = "right",
    )
    model_axis.set_ylim(forest_x_min, forest_x_max)
    model_axis.set_xlim(-0.6, n_models - 0.4)
    model_axis.set_ylabel("median advantage\n(95% CI)", fontsize = 9.5)
    model_axis.set_title("Every learning paradigm", fontsize = 11, loc = "left", pad = 6)
    model_axis.tick_params(axis = "y", labelsize = 9)
    model_axis.tick_params(axis = "x", length = 0)
    model_axis.grid(axis = "y", color = grid_color, alpha = 0.55, linewidth = 0.7)
    for spine in ["top", "right"]:
        model_axis.spines[spine].set_visible(False)

    ## title ------------------------------------------------------------------
    fig.text(
        x = 0.07,
        y = 0.965,
        s = title,
        ha = "left",
        va = "top",
        fontsize = 18,
        fontweight = "bold",
        color = "#1a1a1a",
    )
    fig.text(
        x = 0.07,
        y = 0.918,
        s = subtitle,
        ha = "left",
        va = "top",
        fontsize = 10.5,
        color = "#3a3a3a",
    )

    if show:
        plt.show()
    return fig, axes


## ----------------------------------------------------------------------------
## capacity law compass (cover-iconic radial figure)
## ----------------------------------------------------------------------------
def plot_capacity_law_compass(
    predictions: pd.DataFrame,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    domain_order: Sequence[str] | None = None,
    model_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (13.5, 13.5),
    title: str = "The capacity law",
    subtitle: str = (
        "Across every system, every paradigm, every domain — once topology fixes the\n"
        "capacity ceiling $C(X)$, the residual $s = y^* - \\hat{C}(X)$ is recovered by process, not by more topology."
    ),
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Render a radial compass of the residual-attribution law: each
        dataset is a wedge, each learning paradigm a bar within the
        wedge, bar length is the percent reduction in residual MAE
        achieved by process features Z over topology features X.
        Outward green bars = process wins; inward orange bars =
        topology wins.

    Args:
        predictions: Residual attribution prediction frame with
            columns model, residual_features, dataset, group, abs_error.
        domain_palette: Mapping from domain to color.
        domain_labels: Mapping from domain to short display label.
        model_labels: Mapping from model identifier to display label.
        domain_order: Ordering of domains around the compass.
        model_order: Ordering of models within each wedge.
        figsize: Figure size in inches.
        title: Figure title.
        subtitle: Figure subtitle.
        show: Whether to call plt.show() before returning.

    Returns:
        Tuple of (fig, axes) with axes a 1-element ndarray containing
        the polar axis.

    Raises:
        ValueError: If required columns are missing or only one
            residual feature source is present.
    """

    required = {"model", "residual_features", "dataset", "group", "abs_error"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sources = set(predictions["residual_features"].astype(str).unique())
    if not {"X_to_slack", "Z_to_slack"}.issubset(sources):
        raise ValueError(
            "predictions must include both X_to_slack and Z_to_slack residual_features"
        )

    domain_palette = dict(domain_palette or DEFAULT_DOMAIN_PALETTE)
    domain_labels = dict(domain_labels or DEFAULT_DOMAIN_LABELS)
    model_labels = dict(model_labels or DEFAULT_MODEL_LABELS)

    aggregated = (
        predictions
        .groupby(by = ["model", "dataset", "group", "residual_features"], observed = True)["abs_error"]
        .mean()
        .reset_index()
        .pivot_table(
            index = ["model", "dataset", "group"],
            columns = "residual_features",
            values = "abs_error",
            observed = True,
        )
        .rename_axis(columns = None)
        .reset_index()
        .dropna(subset = ["X_to_slack", "Z_to_slack"])
    )
    aggregated = aggregated.loc[
        (aggregated["X_to_slack"] > 0.0) & (aggregated["Z_to_slack"] > 0.0)
    ].copy()
    aggregated["pct_gain"] = 100.0 * (
        1.0 - aggregated["Z_to_slack"] / aggregated["X_to_slack"]
    )
    aggregated["log_ratio"] = np.log10(aggregated["X_to_slack"] / aggregated["Z_to_slack"])

    if domain_order is None:
        domain_order = [domain for domain in domain_palette.keys() if domain in set(aggregated["group"])]
        for domain in aggregated["group"].astype(str).unique():
            if domain not in domain_order:
                domain_order.append(domain)
    if model_order is None:
        model_order = [model for model in model_labels.keys() if model in set(aggregated["model"])]
        for model in aggregated["model"].astype(str).unique():
            if model not in model_order:
                model_order.append(model)

    paradigm_palette = {
        "linear_quantile":  "#3b6e8f",
        "linear_convex":    "#3b6e8f",
        "linear_laws":      "#3b6e8f",
        "forest_quantile":  "#7a5a9a",
        "boosted_quantile": "#7a5a9a",
        "xgboost_quantile": "#7a5a9a",
        "neural_quantile":  "#a3623a",
        "neural_expectile": "#a3623a",
        "neural_convex":    "#a3623a",
    }

    dataset_records: list[tuple[str, str]] = []
    for domain in domain_order:
        domain_subset = (
            aggregated.loc[aggregated["group"] == domain, ["dataset"]]
            .drop_duplicates()
            .sort_values("dataset")
        )
        for dataset in domain_subset["dataset"].tolist():
            dataset_records.append((str(dataset), str(domain)))
    n_datasets = len(dataset_records)
    n_models = len(model_order)

    if n_datasets == 0 or n_models == 0:
        raise ValueError("No (model, dataset) units available for plotting")

    angles_per_dataset = 2.0 * np.pi / n_datasets
    inter_wedge_pad = angles_per_dataset * 0.18
    bar_span = angles_per_dataset - inter_wedge_pad
    bar_width = bar_span / n_models

    background_color = "#fbfaf6"
    dynamics_color = "#1f7547"
    topology_color = "#b35a36"
    baseline_radius = 1.0
    radial_scale = 0.55
    max_clip = 60.0
    domain_band_inner = baseline_radius + radial_scale + 0.05
    domain_band_outer = domain_band_inner + 0.07
    dataset_label_radius = domain_band_outer + 0.10
    domain_label_radius = domain_band_outer + 0.30
    inner_disc_radius = baseline_radius - radial_scale - 0.02

    overall_pct = float(aggregated["pct_gain"].median())
    overall_log = float(aggregated["log_ratio"].median())
    n_total = int(len(aggregated))
    n_dynamics = int((aggregated["pct_gain"] > 0.0).sum())
    share_dynamics = n_dynamics / n_total if n_total else float("nan")

    fig = plt.figure(figsize = figsize, constrained_layout = False)
    fig.patch.set_facecolor(background_color)
    main_axis = fig.add_subplot(projection = "polar")
    main_axis.set_facecolor(background_color)
    main_axis.set_theta_zero_location(loc = "N")
    main_axis.set_theta_direction(direction = -1)

    ## reference rings
    theta_full = np.linspace(start = 0.0, stop = 2.0 * np.pi, num = 720)
    label_angle = 0.0
    for ref_pct in (15.0, 30.0, 45.0):
        ring_radius = baseline_radius + (ref_pct / max_clip) * radial_scale
        main_axis.plot(
            theta_full,
            np.full_like(theta_full, ring_radius),
            color = "#cdcabb",
            linewidth = 0.6,
            linestyle = (0, (2, 3)),
            zorder = 1,
        )
        main_axis.text(
            x = label_angle,
            y = ring_radius,
            s = f"+{int(ref_pct)}%",
            ha = "center",
            va = "center",
            fontsize = 7.5,
            color = "#9b9580",
            zorder = 1.5,
            bbox = {"boxstyle": "round,pad=0.10", "facecolor": background_color, "edgecolor": "none"},
        )
    for ref_pct in (-15.0, -30.0):
        ring_radius = baseline_radius + (ref_pct / max_clip) * radial_scale
        main_axis.plot(
            theta_full,
            np.full_like(theta_full, ring_radius),
            color = "#e3d8c8",
            linewidth = 0.5,
            linestyle = (0, (2, 4)),
            zorder = 1,
        )
    main_axis.plot(
        theta_full,
        np.full_like(theta_full, baseline_radius),
        color = "#3a3a3a",
        linewidth = 1.0,
        zorder = 2,
    )

    ## bars
    for dataset_index, (dataset, domain) in enumerate(dataset_records):
        wedge_start = dataset_index * angles_per_dataset + inter_wedge_pad / 2.0
        for model_index, model in enumerate(model_order):
            row = aggregated.loc[
                (aggregated["dataset"] == dataset) & (aggregated["model"] == model)
            ]
            if row.empty:
                continue
            gain = float(row["pct_gain"].iloc[0])
            gain_clipped = float(np.clip(gain, -max_clip, max_clip))
            radial_height = (gain_clipped / max_clip) * radial_scale
            bar_center = wedge_start + (model_index + 0.5) * bar_width
            face_color = dynamics_color if gain >= 0.0 else topology_color
            paradigm_edge = paradigm_palette.get(model, face_color)
            if radial_height >= 0.0:
                bottom = baseline_radius
                height = radial_height
            else:
                bottom = baseline_radius + radial_height
                height = -radial_height
            main_axis.bar(
                x = bar_center,
                height = height,
                width = bar_width * 0.86,
                bottom = bottom,
                color = face_color,
                edgecolor = paradigm_edge,
                linewidth = 0.7,
                alpha = 0.9,
                zorder = 3,
            )

    ## domain bands and labels
    domain_to_indices: dict[str, list[int]] = {}
    for index, (_, domain) in enumerate(dataset_records):
        domain_to_indices.setdefault(domain, []).append(index)

    for domain, indices in domain_to_indices.items():
        first = min(indices) * angles_per_dataset
        last = (max(indices) + 1) * angles_per_dataset
        theta = np.linspace(start = first + inter_wedge_pad / 4.0, stop = last - inter_wedge_pad / 4.0, num = 96)
        main_axis.fill_between(
            x = theta,
            y1 = domain_band_inner,
            y2 = domain_band_outer,
            color = domain_palette.get(domain, "#888888"),
            alpha = 0.92,
            zorder = 4,
        )
        midpoint = (first + last) / 2.0
        rotation_deg = np.degrees(midpoint) * -1.0
        if 90.0 < (np.degrees(midpoint) % 360.0) < 270.0:
            rotation_deg += 180.0
        main_axis.text(
            x = midpoint,
            y = domain_label_radius,
            s = domain_labels.get(domain, domain).replace("\n", " "),
            ha = "center",
            va = "center",
            fontsize = 11,
            fontweight = "bold",
            color = domain_palette.get(domain, "#444444"),
            rotation = rotation_deg,
            rotation_mode = "anchor",
            zorder = 5,
        )

    ## dataset labels
    for dataset_index, (dataset, _) in enumerate(dataset_records):
        midpoint = (dataset_index + 0.5) * angles_per_dataset
        deg = np.degrees(midpoint) % 360.0
        if 90.0 < deg < 270.0:
            rotation_deg = -deg + 180.0
            ha = "right"
        else:
            rotation_deg = -deg
            ha = "left"
        main_axis.text(
            x = midpoint,
            y = dataset_label_radius,
            s = dataset,
            ha = ha,
            va = "center",
            fontsize = 8.5,
            color = "#2a2a2a",
            rotation = rotation_deg,
            rotation_mode = "anchor",
            zorder = 5,
        )

    ## inner disc with headline
    inner_theta = np.linspace(start = 0.0, stop = 2.0 * np.pi, num = 360)
    main_axis.fill_between(
        x = inner_theta,
        y1 = 0.0,
        y2 = inner_disc_radius,
        color = "white",
        alpha = 0.92,
        zorder = 6,
    )
    main_axis.plot(
        inner_theta,
        np.full_like(inner_theta, inner_disc_radius),
        color = "#3a3a3a",
        linewidth = 0.8,
        zorder = 6.5,
    )
    headline_lines = [
        f"$\\bf{{{n_dynamics}/{n_total}}}$",
        "tests favor process",
        "",
        f"({share_dynamics:.0%} of model × system pairs)",
        "",
        "median residual",
        "error reduction",
        f"$\\bf{{{overall_pct:+.1f}\\%}}$",
    ]
    main_axis.text(
        x = 0.0,
        y = 0.0,
        s = "\n".join(headline_lines),
        ha = "center",
        va = "center",
        fontsize = 11.5,
        color = "#1a1a1a",
        zorder = 7,
    )

    ## axis cosmetics
    main_axis.set_ylim(bottom = 0.0, top = domain_label_radius + 0.18)
    main_axis.set_yticks([])
    main_axis.set_xticks([])
    main_axis.spines["polar"].set_visible(False)
    main_axis.grid(visible = False)

    ## legend strip (bars: green out / orange in)
    legend_axis = fig.add_axes(rect = (0.06, 0.045, 0.36, 0.04))
    legend_axis.set_facecolor(background_color)
    legend_axis.set_xlim(-1.0, 1.0)
    legend_axis.set_ylim(0.0, 1.0)
    legend_axis.axis("off")
    legend_axis.add_patch(Rectangle(xy = (-0.95, 0.30), width = 0.85, height = 0.40, facecolor = topology_color, edgecolor = "none"))
    legend_axis.add_patch(Rectangle(xy = (0.10, 0.30), width = 0.85, height = 0.40, facecolor = dynamics_color, edgecolor = "none"))
    legend_axis.text(x = -0.525, y = 0.92, s = "topology wins", ha = "center", va = "bottom", fontsize = 9, color = topology_color, fontweight = "bold")
    legend_axis.text(x = 0.525, y = 0.92, s = "process wins", ha = "center", va = "bottom", fontsize = 9, color = dynamics_color, fontweight = "bold")
    legend_axis.text(x = -0.525, y = 0.10, s = "bar points inward", ha = "center", va = "top", fontsize = 8, color = "#5a5a5a", style = "italic")
    legend_axis.text(x = 0.525, y = 0.10, s = "bar points outward", ha = "center", va = "top", fontsize = 8, color = "#5a5a5a", style = "italic")

    ## paradigm legend (right side)
    paradigm_legend_axis = fig.add_axes(rect = (0.58, 0.045, 0.36, 0.04))
    paradigm_legend_axis.set_facecolor(background_color)
    paradigm_legend_axis.set_xlim(0.0, 1.0)
    paradigm_legend_axis.set_ylim(0.0, 1.0)
    paradigm_legend_axis.axis("off")
    paradigm_groups = [
        ("Linear", "#3b6e8f", "Linear Quantile · Linear Convex · Linear Laws"),
        ("Tree-based", "#7a5a9a", "Forest · Boosted · XGBoost"),
        ("Neural", "#a3623a", "Neural Quantile · Expectile · Convex"),
    ]
    paradigm_legend_axis.text(x = 0.0, y = 0.92, s = "9 learning paradigms per wedge:", ha = "left", va = "bottom", fontsize = 9, color = "#3a3a3a", fontweight = "bold")
    for index, (name, color, members) in enumerate(paradigm_groups):
        y_pos = 0.55 - index * 0.30
        paradigm_legend_axis.add_patch(Rectangle(xy = (0.0, y_pos), width = 0.04, height = 0.18, facecolor = color, edgecolor = "none"))
        paradigm_legend_axis.text(x = 0.06, y = y_pos + 0.09, s = f"{name}: {members}", ha = "left", va = "center", fontsize = 7.8, color = "#3a3a3a")

    ## title
    fig.text(
        x = 0.50,
        y = 0.975,
        s = title,
        ha = "center",
        va = "top",
        fontsize = 22,
        fontweight = "bold",
        color = "#1a1a1a",
    )
    fig.text(
        x = 0.50,
        y = 0.940,
        s = subtitle,
        ha = "center",
        va = "top",
        fontsize = 11,
        color = "#3a3a3a",
    )

    fig.subplots_adjust(left = 0.04, right = 0.96, top = 0.92, bottom = 0.10)

    if show:
        plt.show()
    return fig, np.array([main_axis], dtype = object)


def _plot_transfer_invariance_legacy(
    results_data_domain: pd.DataFrame,
    results_data_disc: pd.DataFrame,
    results_data_5fold: pd.DataFrame,
    results_data_10fold: pd.DataFrame,
    disc_domain_map: Mapping[str, str],
    *,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    equivalence_margin: float | None = None,
    feasibility_threshold: float = 0.05,
    figsize: tuple[float, float] = (9.6, 13.2),
    title: str = "The capacity law transfers across every scientific domain",
    subtitle: str = (
        "After exiling entire domains and disciplines from training, predictive efficiency "
        "(EI) is statistically indistinguishable from random-split baselines — "
        "held-out generalization is a no-op."
    ),
    background_color: str = "#fbfaf6",
    random_state: int = 42,
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Money-shot visualization of held-out-group transfer invariance. Shows
        that EI under leave-one-domain-out and leave-one-discipline-out
        cross-validation is statistically indistinguishable from random k-fold
        baselines, both distributionally (top strip) and per held-out group
        (bottom forest of paired Delta EI from pooled random baseline).

    Args:
        results_data_domain: DataFrame from logo_cross_valid on 'domain', with
            columns including 'model', 'group', 'ei'.
        results_data_disc: DataFrame from logo_cross_valid on 'discipline'.
        results_data_5fold: DataFrame from kfold_cross_valid (5-fold), with
            one row per model and a 'model' + 'ei' column.
        results_data_10fold: DataFrame from kfold_cross_valid (10-fold).
        disc_domain_map: Mapping from discipline -> domain.
        domain_palette: Optional override for domain -> color.
        domain_labels: Optional override for domain -> short label.
        equivalence_margin: Half-width of the equivalence band on Delta EI. If
            None, derives the margin from the IQR of pooled random-split
            baseline EI values using _spec_marginal_delta.
        feasibility_threshold: Estimators with transfer EI at or below this
            value are treated as feasibility failures (predictions violating
            the capacity bound) and excluded from per-row Delta EI summaries.
            Per-row counts of feasible / total estimators are annotated when
            any failure is present.
        figsize: Figure size in inches.
        title: Headline title.
        subtitle: Subtitle below the title.
        background_color: Figure background color.
        random_state: RNG seed for reproducibility.
        show: If True, calls plt.show().

    Returns:
        Tuple of (figure, axes_array) where axes_array is [strip_axis,
        forest_axis].

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """

    palette = dict(domain_palette) if domain_palette is not None else dict(DEFAULT_DOMAIN_PALETTE)
    labels = dict(domain_labels) if domain_labels is not None else dict(DEFAULT_DOMAIN_LABELS)

    ## validate required columns
    for name, frame, cols in (
        ("results_data_domain", results_data_domain, ("model", "group", "ei")),
        ("results_data_disc",   results_data_disc,   ("model", "group", "ei")),
        ("results_data_5fold",  results_data_5fold,  ("model", "ei")),
        ("results_data_10fold", results_data_10fold, ("model", "ei")),
    ):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    ## pooled random baseline per model (mean of 5-fold and 10-fold)
    base = pd.concat(
        [results_data_5fold[["model", "ei"]], results_data_10fold[["model", "ei"]]],
        ignore_index = True,
    )
    base = base.dropna(subset = ["ei"])
    baseline_per_model = base.groupby("model")["ei"].mean()
    if baseline_per_model.empty:
        raise ValueError("no valid baseline EI rows in 5-fold/10-fold results")

    ## paired Delta EI = transfer EI - pooled random baseline EI for that model
    def _attach_delta(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.dropna(subset = ["ei"]).copy()
        out["baseline_ei"] = out["model"].map(baseline_per_model)
        out = out.dropna(subset = ["baseline_ei"])
        out["delta_ei"] = out["ei"].astype(float) - out["baseline_ei"].astype(float)
        out["feasible"] = out["ei"].astype(float) > float(feasibility_threshold)
        return out

    domain_delta = _attach_delta(results_data_domain)
    disc_delta = _attach_delta(results_data_disc)
    domain_delta["domain"] = domain_delta["group"]
    disc_delta["domain"] = disc_delta["group"].map(disc_domain_map).fillna("Unknown")

    ## ordering: domain order from palette
    domain_order = [d for d in palette if d in domain_delta["domain"].unique()]
    extras = [d for d in domain_delta["domain"].unique() if d not in domain_order]
    domain_order = domain_order + sorted(extras)

    ## domain rows: ordered by palette
    domain_rows = list()
    for dom in domain_order:
        sub = domain_delta[domain_delta["domain"] == dom]
        if len(sub) == 0:
            continue
        feasible_mask = sub["feasible"].to_numpy()
        domain_rows.append({
            "label":       dom,
            "kind":        "domain",
            "domain":      dom,
            "values":      sub.loc[feasible_mask, "delta_ei"].astype(float).to_numpy(),
            "n_total":     int(len(sub)),
            "n_feasible": int(feasible_mask.sum()),
        })

    ## discipline rows: ordered by domain then median Delta EI (feasible only) ascending
    disc_rows: list[dict] = list()
    disc_per_group = disc_delta.groupby("group")
    feasible_disc = disc_delta[disc_delta["feasible"]]
    disc_summary = (
        feasible_disc.groupby("group")["delta_ei"]
        .median()
        .rename("median_delta")
        .reset_index()
    )
    ## include groups with zero feasible models so they still render
    all_groups = pd.DataFrame({"group": sorted(disc_delta["group"].unique())})
    disc_summary = all_groups.merge(disc_summary, on = "group", how = "left")
    disc_summary["median_delta"] = disc_summary["median_delta"].fillna(0.0)
    disc_summary["domain"] = disc_summary["group"].map(disc_domain_map).fillna("Unknown")
    disc_summary["_drank"] = disc_summary["domain"].map(
        {d: i for i, d in enumerate(domain_order)}
    ).fillna(len(domain_order))
    disc_summary = disc_summary.sort_values(
        by = ["_drank", "median_delta"], ascending = [True, False],
    ).reset_index(drop = True)
    for _, row in disc_summary.iterrows():
        sub = disc_per_group.get_group(row["group"])
        feasible_mask = sub["feasible"].to_numpy()
        disc_rows.append({
            "label":       row["group"],
            "kind":        "discipline",
            "domain":      row["domain"],
            "values":      sub.loc[feasible_mask, "delta_ei"].astype(float).to_numpy(),
            "n_total":     int(len(sub)),
            "n_feasible": int(feasible_mask.sum()),
        })

    ## headline statistics (over feasible estimators only)
    rng = np.random.default_rng(seed = random_state)
    all_rows = domain_rows + disc_rows
    nonempty_rows = [r for r in all_rows if r["values"].size > 0]
    all_delta = np.concatenate(
        [r["values"] for r in nonempty_rows]
    ).astype(float) if nonempty_rows else np.array([], dtype = float)
    overall_med = float(np.median(all_delta)) if all_delta.size else float("nan")
    n_groups = len(all_rows)
    n_within = sum(
        1 for r in nonempty_rows
        if abs(float(np.median(r["values"]))) <= equivalence_margin
    )
    ## per-row feasibility totals for the headline
    n_total_estimators = sum(r["n_total"] for r in all_rows)
    n_feasible_estimators = sum(r["n_feasible"] for r in all_rows)

    ## figure scaffolding
    fig = plt.figure(figsize = figsize, facecolor = background_color)
    gs = fig.add_gridspec(
        nrows = 2, ncols = 1,
        height_ratios = [0.22, 0.78],
        hspace = 0.20,
        left = 0.21, right = 0.905,
        top = 0.805, bottom = 0.060,
    )
    strip_axis = fig.add_subplot(gs[0, 0])
    forest_axis = fig.add_subplot(gs[1, 0])
    for ax in (strip_axis, forest_axis):
        ax.set_facecolor(background_color)

    ## ---- top panel: regime distribution overlay ----
    ## LOGO regimes are filtered to feasible estimators only (consistent with
    ## the per-group panel below); random-split regimes show all values.
    def _feasible_ei(frame: pd.DataFrame) -> np.ndarray:
        ei = frame["ei"].dropna().to_numpy()
        return ei[ei > float(feasibility_threshold)]

    regimes = [
        ("Random (10-Fold)",   results_data_10fold["ei"].dropna().to_numpy(), "#9aa0a6", "o"),
        ("Random (5-Fold)",    results_data_5fold["ei"].dropna().to_numpy(),  "#6b7075", "o"),
        ("Held-out domain",   _feasible_ei(results_data_domain),             "#2C6E91", "D"),
        ("Held-out discipline", _feasible_ei(results_data_disc),             "#3A7D55", "D"),
    ]
    strip_jitter = 0.18
    for i, (name, vals, color, marker) in enumerate(regimes):
        if vals.size == 0:
            continue
        y_center = len(regimes) - 1 - i
        ## individual points
        jitter = rng.uniform(low = -strip_jitter, high = strip_jitter, size = vals.size)
        strip_axis.scatter(
            vals, np.full_like(vals, y_center, dtype = float) + jitter,
            s = 12, color = color, alpha = 0.32, edgecolors = "none", zorder = 2,
        )
        ## IQR bar + median tick
        q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        strip_axis.plot(
            [q1, q3], [y_center, y_center],
            color = color, lw = 5.5, alpha = 0.45, solid_capstyle = "round", zorder = 3,
        )
        strip_axis.scatter(
            [med], [y_center], s = 75, color = color, marker = marker,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## strip axis cosmetics
    all_vals = np.concatenate([v for _, v, _, _ in regimes if v.size > 0])
    pad = 0.04
    x_lo_strip = float(np.min(all_vals)) - pad
    x_hi_strip = float(np.max(all_vals)) + pad
    strip_axis.set_xlim(x_lo_strip, x_hi_strip)
    strip_axis.set_ylim(-0.7, len(regimes) - 0.3)
    strip_axis.set_yticks(range(len(regimes)))
    strip_axis.set_yticklabels(
        [name for name, _, _, _ in regimes][::-1], fontsize = 9,
    )
    for ytick, (_, _, color, _) in zip(strip_axis.get_yticklabels(), regimes):
        ytick.set_color(color)
    strip_axis.set_xlabel("Efficiency Index (EI)", fontsize = 9.5, labelpad = 4)
    strip_axis.tick_params(axis = "x", labelsize = 8.5)
    strip_axis.tick_params(axis = "y", left = False, pad = 4)
    strip_axis.spines[["top", "right", "left"]].set_visible(False)
    strip_axis.spines["bottom"].set_linewidth(0.7)
    strip_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    strip_axis.set_axisbelow(True)
    strip_axis.text(
        x = 0.0, y = 1.04,
        s = "A   EI distribution under transfer vs. random-split baselines",
        transform = strip_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## ---- bottom panel: held-out group forest of Delta EI ----
    n_domain = len(domain_rows)
    n_disc = len(disc_rows)
    spacer = 1.2
    ## y positions: domains at top, then spacer, then disciplines below
    y_positions: list[float] = list()
    for i in range(n_domain):
        y_positions.append((n_disc + spacer) + (n_domain - 1 - i))
    domain_y = list(y_positions)
    for i in range(n_disc):
        y_positions.append(n_disc - 1 - i)
    disc_y = y_positions[n_domain:]

    rows = domain_rows + disc_rows

    ## equivalence band
    forest_axis.axvspan(
        -equivalence_margin, equivalence_margin,
        color = "#e8efe9", alpha = 0.7, zorder = 0,
    )
    forest_axis.axvline(0.0, color = "#7a7a7a", lw = 1.0, zorder = 1)
    forest_axis.axvline(
        -equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )
    forest_axis.axvline(
        +equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )

    ## domain / discipline section divider
    divider_y = n_disc + spacer / 2.0 - 0.5
    forest_axis.axhline(divider_y, color = "#d0d0d0", lw = 0.6, zorder = 1)

    ## render rows
    for row, y in zip(rows, y_positions):
        color = palette.get(row["domain"], "#555555")
        vals = row["values"]
        ## feasibility annotation: only show when at least one estimator failed
        if row["n_feasible"] < row["n_total"]:
            forest_axis.text(
                x = 1.005,
                y = y,
                s = f"{row['n_feasible']}/{row['n_total']} feasible",
                transform = forest_axis.get_yaxis_transform(),
                ha = "left", va = "center",
                fontsize = 6.8, color = "#9a8a6a", fontstyle = "italic",
                clip_on = False,
            )
        if vals.size == 0:
            continue
        q1, med, q3 = np.quantile(vals, [0.25, 0.50, 0.75])
        ## faint individual points
        jitter = rng.uniform(low = -0.18, high = 0.18, size = vals.size)
        forest_axis.scatter(
            vals, np.full_like(vals, y, dtype = float) + jitter,
            s = 10, color = color, alpha = 0.25, edgecolors = "none", zorder = 2,
        )
        ## IQR line
        forest_axis.plot(
            [q1, q3], [y, y],
            color = color, lw = 2.0, alpha = 0.55, solid_capstyle = "round", zorder = 3,
        )
        ## median marker (size emphasizes domain rows)
        marker_size = 95 if row["kind"] == "domain" else 48
        forest_axis.scatter(
            [med], [y], s = marker_size, color = color,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## y-tick labels
    all_y = list(y_positions)
    all_labels: list[str] = list()
    for r in rows:
        if r["kind"] == "domain":
            all_labels.append(labels.get(r["label"], r["label"]).replace("\n", " "))
        else:
            all_labels.append(r["label"])
    ## sort by ascending y for matplotlib tick semantics (so labels match positions)
    order = np.argsort(all_y)
    forest_axis.set_yticks([all_y[i] for i in order])
    forest_axis.set_yticklabels([all_labels[i] for i in order], fontsize = 7.6)
    for tick_lbl, idx in zip(forest_axis.get_yticklabels(), order):
        r = rows[idx]
        color = palette.get(r["domain"], "#555555")
        tick_lbl.set_color(color)
        if r["kind"] == "domain":
            tick_lbl.set_fontsize(9.0)
            tick_lbl.set_fontweight("bold")

    ## x-axis
    all_delta_finite = all_delta[np.isfinite(all_delta)]
    x_lim = max(
        equivalence_margin * 2.4,
        float(np.quantile(np.abs(all_delta_finite), 0.99)) + 0.02,
    )
    forest_axis.set_xlim(-x_lim, x_lim)
    forest_axis.set_ylim(-0.8, n_disc + spacer + n_domain - 0.2)
    forest_axis.spines[["top", "right", "left"]].set_visible(False)
    forest_axis.spines["bottom"].set_linewidth(0.7)
    forest_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    forest_axis.set_axisbelow(True)
    forest_axis.tick_params(axis = "y", left = False, pad = 4)
    forest_axis.tick_params(axis = "x", labelsize = 8.5)
    forest_axis.set_xlabel(
        "$\\Delta$EI  =  held-out transfer EI  $-$  pooled random-split baseline EI   (paired per model)",
        fontsize = 9.5, labelpad = 6,
    )

    ## panel B label
    forest_axis.text(
        x = 0.0, y = 1.012,
        s = (
            f"B   Per-group transfer penalty $\\Delta$EI "
            f"(shaded $\\pm${equivalence_margin:.2f} equivalence band)"
        ),
        transform = forest_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## ---- header: title + subtitle + headline statistic plate ----
    fig.text(
        x = 0.50, y = 0.978, s = title,
        ha = "center", va = "top",
        fontsize = 18, fontweight = "bold", color = "#1a1a1a",
    )
    fig.text(
        x = 0.50, y = 0.948, s = subtitle,
        ha = "center", va = "top",
        fontsize = 10, color = "#3a3a3a",
    )

    ## headline statistic plate (centered just below subtitle)
    feasibility_share = (
        n_feasible_estimators / n_total_estimators
        if n_total_estimators > 0 else 0.0
    )
    feasibility_label = f"{feasibility_share:.0%}".replace("%", "\\%")
    stat_text = (
        f"median $\\Delta$EI across feasible transfers: "
        f"$\\bf{{{overall_med:+.3f}}}$   "
        f"(n = {all_delta.size} estimator×group transfers)"
        f"      ·      "
        f"$\\bf{{{n_within}\\,/\\,{n_groups}}}$ held-out-group medians within "
        f"$\\pm${equivalence_margin:.2f} EI of baseline"
        f"\nfeasibility: "
        f"$\\bf{{{n_feasible_estimators}\\,/\\,{n_total_estimators}}}$ "
        f"({feasibility_share:.0%}) estimator×group transfers satisfy the "
        f"capacity bound (EI $>$ {feasibility_threshold:.2f})"
    )
    fig.text(
        x = 0.50, y = 0.895, s = stat_text,
        ha = "center", va = "top",
        fontsize = 9.5, color = "#1a1a1a", linespacing = 1.55,
        bbox = {
            "boxstyle":  "round,pad=0.55",
            "facecolor": "#f1efe6",
            "edgecolor": "#d6d2c3",
            "linewidth": 0.6,
        },
    )

    if show:
        plt.show()
    return fig, np.array([strip_axis, forest_axis], dtype = object)


def plot_transfer_invariance(
    results_data_domain: pd.DataFrame,
    results_data_disc: pd.DataFrame,
    results_data_5fold: pd.DataFrame,
    results_data_10fold: pd.DataFrame,
    disc_domain_map: Mapping[str, str],
    *,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    equivalence_margin: float | None = None,
    feasibility_threshold: float = 0.05,
    figsize: tuple[float, float] = (9.6, 10.2),
    title: str = "The capacity law transfers across every scientific domain",
    subtitle: str = (
        "After exiling entire disciplines from training, predictive efficiency "
        "(EI) is statistically indistinguishable from random-split baselines — "
        "held-out generalization is a no-op."
    ),
    background_color: str = "#fbfaf6",
    random_state: int = 42,
    show: bool = True,
    ) -> tuple[Figure, np.ndarray]:

    """
    Desc:
        Money-shot discipline transfer visualization. Shows per-discipline
        paired Delta EI against each model's pooled random-split baseline on a
        fixed [-1, 1] scale. Domain aggregate rows and the regime strip panel
        are intentionally omitted so the figure focuses only on held-out
        discipline transfer penalties.

    Args:
        results_data_domain: DataFrame from logo_cross_valid on 'domain'. This
            argument is retained for API compatibility but is not drawn.
        results_data_disc: DataFrame from logo_cross_valid on 'discipline',
            with columns including 'model', 'group', and 'ei'.
        results_data_5fold: DataFrame from kfold_cross_valid (5-fold), with
            one row per model and a 'model' + 'ei' column.
        results_data_10fold: DataFrame from kfold_cross_valid (10-fold).
        disc_domain_map: Mapping from discipline -> domain.
        domain_palette: Optional override for domain -> color.
        domain_labels: Optional override for domain -> short label.
        equivalence_margin: Half-width of the equivalence band on Delta EI. If
            None, derives the margin from the IQR of pooled random-split
            baseline EI values using _spec_marginal_delta.
        feasibility_threshold: Estimators with transfer EI at or below this
            value are treated as feasibility failures and excluded from per-row
            Delta EI summaries.
        figsize: Figure size in inches.
        title: Headline title.
        subtitle: Subtitle below the title.
        background_color: Figure background color.
        random_state: RNG seed for reproducibility.
        show: If True, calls plt.show().

    Returns:
        Tuple of (figure, axes_array) where axes_array contains the discipline
        forest axis.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """

    del results_data_domain

    palette = dict(domain_palette) if domain_palette is not None else dict(DEFAULT_DOMAIN_PALETTE)
    labels = dict(domain_labels) if domain_labels is not None else dict(DEFAULT_DOMAIN_LABELS)

    ## validate required columns
    for frame_name, frame, required_columns in (
        ("results_data_disc",   results_data_disc,   ("model", "group", "ei")),
        ("results_data_5fold",  results_data_5fold,  ("model", "ei")),
        ("results_data_10fold", results_data_10fold, ("model", "ei")),
    ):
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{frame_name} missing columns: {missing_columns}")

    ## pooled random baseline per model
    baseline = pd.concat(
        [results_data_5fold[["model", "ei"]], results_data_10fold[["model", "ei"]]],
        ignore_index = True,
    )
    baseline = baseline.dropna(subset = ["ei"])
    baseline_per_model = baseline.groupby("model")["ei"].mean()
    if baseline_per_model.empty:
        raise ValueError("no valid baseline EI rows in 5-fold/10-fold results")

    empirical_equivalence_margin = equivalence_margin is None
    if equivalence_margin is None:
        equivalence_margin = _spec_marginal_delta(
            results = baseline.assign(reference = "baseline"),
            feat_value = ["ei"],
            label_ref = "reference",
            value_ref = "baseline",
            method = "iqr",
            scale = 1.0,
            decimals = 2,
        )
    else:
        equivalence_margin = float(equivalence_margin)
    margin_label = "empirical $\\delta$" if empirical_equivalence_margin else "$\\delta$"

    ## within-model random-split residuals on Delta EI scale (centered at 0)
    baseline_residuals = (
        baseline["ei"].astype(float)
        - baseline["model"].map(baseline_per_model).astype(float)
    ).dropna().to_numpy()
    if baseline_residuals.size >= 2:
        baseline_q1, baseline_q3 = np.quantile(baseline_residuals, [0.25, 0.75])
    else:
        baseline_q1, baseline_q3 = 0.0, 0.0

    ## paired Delta EI = discipline LOGO EI - pooled random baseline EI
    disc_delta = results_data_disc.dropna(subset = ["ei"]).copy()
    disc_delta["baseline_ei"] = disc_delta["model"].map(baseline_per_model)
    disc_delta = disc_delta.dropna(subset = ["baseline_ei"])
    disc_delta["delta_ei"] = disc_delta["ei"].astype(float) - disc_delta["baseline_ei"].astype(float)
    disc_delta["feasible"] = disc_delta["ei"].astype(float) > float(feasibility_threshold)
    disc_delta["domain"] = disc_delta["group"].map(disc_domain_map).fillna("Unknown")
    if disc_delta.empty:
        raise ValueError("no valid discipline transfer rows remain after pairing baselines")

    ## order disciplines by domain and feasible median Delta EI
    domain_order = [domain for domain in palette if domain in disc_delta["domain"].unique()]
    extra_domains = [domain for domain in disc_delta["domain"].unique() if domain not in domain_order]
    domain_order = domain_order + sorted(extra_domains)
    domain_rank = {domain: index for index, domain in enumerate(domain_order)}

    feasible_disc = disc_delta[disc_delta["feasible"]]
    disc_summary = (
        feasible_disc.groupby("group")["delta_ei"]
        .median()
        .rename("median_delta")
        .reset_index()
    )
    all_groups = pd.DataFrame({"group": sorted(disc_delta["group"].unique())})
    disc_summary = all_groups.merge(disc_summary, on = "group", how = "left")
    disc_summary["median_delta"] = disc_summary["median_delta"].fillna(0.0)
    disc_summary["domain"] = disc_summary["group"].map(disc_domain_map).fillna("Unknown")
    disc_summary["_domain_rank"] = disc_summary["domain"].map(domain_rank).fillna(len(domain_order))
    disc_summary = disc_summary.sort_values(
        by = ["_domain_rank", "median_delta"],
        ascending = [True, False],
    ).reset_index(drop = True)

    grouped_disc = dict(tuple(disc_delta.groupby("group")))
    disc_rows: list[dict] = list()
    for _, summary_row in disc_summary.iterrows():
        group_name = summary_row["group"]
        group_frame = grouped_disc[group_name]
        feasible_frame = group_frame[group_frame["feasible"]]
        disc_rows.append({
            "label":      group_name,
            "domain":     summary_row["domain"],
            "values":     feasible_frame["delta_ei"].astype(float).to_numpy(),
            "n_total":    int(len(group_frame)),
            "n_feasible": int(len(feasible_frame)),
        })

    nonempty_rows = [row for row in disc_rows if row["values"].size > 0]
    if not nonempty_rows:
        raise ValueError("no feasible discipline transfer rows remain")

    all_delta = np.concatenate([row["values"] for row in nonempty_rows]).astype(float)
    rng = np.random.default_rng(seed = random_state)
    overall_med = float(np.median(all_delta))
    n_total_estimators = sum(row["n_total"] for row in disc_rows)
    n_feasible_estimators = sum(row["n_feasible"] for row in disc_rows)
    feasibility_share = (
        n_feasible_estimators / n_total_estimators
        if n_total_estimators > 0 else 0.0
    )
    feasibility_label = f"{feasibility_share:.0%}".replace("%", "\\%")

    ## figure scaffolding
    fig = plt.figure(figsize = figsize, facecolor = background_color)
    fig.subplots_adjust(left = 0.21, right = 0.855, top = 0.805, bottom = 0.075)
    forest_axis = fig.add_subplot(111)
    forest_axis.set_facecolor(background_color)

    ## random-split baseline IQR band (within-model residuals on Delta EI scale)
    if baseline_q3 > baseline_q1:
        forest_axis.axvspan(
            baseline_q1, baseline_q3,
            color = "#d9e3ec", alpha = 0.85, zorder = 0,
        )
        forest_axis.axvline(
            baseline_q1, color = "#9fb1c2", lw = 0.6, ls = (0, (2, 3)), zorder = 1,
        )
        forest_axis.axvline(
            baseline_q3, color = "#9fb1c2", lw = 0.6, ls = (0, (2, 3)), zorder = 1,
        )

    ## equivalence band
    forest_axis.axvspan(
        -equivalence_margin, equivalence_margin,
        color = "#e8efe9", alpha = 0.55, zorder = 0,
    )
    forest_axis.axvline(0.0, color = "#7a7a7a", lw = 1.0, zorder = 1)
    forest_axis.axvline(
        -equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )
    forest_axis.axvline(
        +equivalence_margin, color = "#a8b3a9", lw = 0.7, ls = (0, (3, 3)), zorder = 1,
    )

    n_disc = len(disc_rows)
    y_positions = [n_disc - 1 - row_index for row_index in range(n_disc)]

    ## domain separators
    previous_domain = None
    for row_index, row in enumerate(disc_rows):
        if previous_domain is not None and row["domain"] != previous_domain:
            forest_axis.axhline(
                n_disc - row_index - 0.5,
                color = "#d0d0d0", lw = 0.6, zorder = 1,
            )
        previous_domain = row["domain"]

    ## render discipline rows
    for row, y_position in zip(disc_rows, y_positions):
        color = palette.get(row["domain"], "#555555")
        values = row["values"]
        if values.size == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        jitter = rng.uniform(low = -0.18, high = 0.18, size = values.size)
        forest_axis.scatter(
            values, np.full_like(values, y_position, dtype = float) + jitter,
            s = 10, color = color, alpha = 0.25, edgecolors = "none", zorder = 2,
        )
        forest_axis.plot(
            [q1, q3], [y_position, y_position],
            color = color, lw = 2.0, alpha = 0.55, solid_capstyle = "round", zorder = 3,
        )
        forest_axis.scatter(
            [median], [y_position], s = 48, color = color,
            edgecolors = "white", linewidths = 0.9, zorder = 5,
        )

    ## y-axis discipline labels
    tick_order = np.argsort(y_positions)
    forest_axis.set_yticks([y_positions[index] for index in tick_order])
    forest_axis.set_yticklabels([disc_rows[index]["label"] for index in tick_order], fontsize = 7.6)
    for tick_label, row_index in zip(forest_axis.get_yticklabels(), tick_order):
        domain = disc_rows[row_index]["domain"]
        tick_label.set_color(palette.get(domain, "#555555"))

    ## right-margin domain labels
    for domain in domain_order:
        matching_positions = [
            y_positions[row_index]
            for row_index, row in enumerate(disc_rows)
            if row["domain"] == domain
        ]
        if not matching_positions:
            continue
        y_mid = (min(matching_positions) + max(matching_positions)) / 2.0
        forest_axis.text(
            x = 1.025,
            y = y_mid,
            s = labels.get(domain, domain),
            transform = forest_axis.get_yaxis_transform(),
            ha = "left", va = "center",
            fontsize = 7.0, color = palette.get(domain, "#555555"),
            fontweight = "semibold", linespacing = 1.35,
            clip_on = False,
        )

    ## fixed x-axis scale and cosmetics
    forest_axis.set_xlim(-1.0, 1.0)
    forest_axis.set_xticks(np.linspace(-1.0, 1.0, 9))
    forest_axis.set_xticks(np.arange(-1.0, 1.0001, 0.05), minor = True)
    forest_axis.set_ylim(-0.8, n_disc - 0.2)
    forest_axis.spines[["top", "right", "left"]].set_visible(False)
    forest_axis.spines["bottom"].set_linewidth(0.7)
    forest_axis.xaxis.grid(True, lw = 0.4, color = "#EBEBEB", zorder = 0)
    forest_axis.set_axisbelow(True)
    forest_axis.tick_params(axis = "y", left = False, pad = 4)
    forest_axis.tick_params(axis = "x", labelsize = 8.5)
    forest_axis.set_xlabel(
        "$\\Delta$EI  =  discipline LOGO EI  $-$  pooled random-split baseline EI   "
        "(fixed scale: -1 to +1)",
        fontsize = 9.5, labelpad = 6,
    )
    forest_axis.text(
        x = 0.0, y = 1.012,
        s = (
            f"Per-discipline transfer penalty $\\Delta$EI "
            f"(dot = median; bar = IQR; "
            f"blue = random-split IQR [{baseline_q1:+.2f}, {baseline_q3:+.2f}]; "
            f"green {margin_label} = $\\pm${equivalence_margin:.2f})"
        ),
        transform = forest_axis.transAxes,
        ha = "left", va = "bottom",
        fontsize = 10.5, fontweight = "semibold", color = "#1a1a1a",
    )

    ## header
    fig.text(
        x = 0.50, y = 0.978, s = title,
        ha = "center", va = "top",
        fontsize = 18, fontweight = "bold", color = "#1a1a1a",
    )
    fig.text(
        x = 0.50, y = 0.948, s = subtitle,
        ha = "center", va = "top",
        fontsize = 10, color = "#3a3a3a",
    )

    stat_text = (
        f"median $\\Delta$EI across feasible model×discipline transfers: "
        f"$\\bf{{{overall_med:+.3f}}}$   "
        f"(n = {all_delta.size})"
        f"      ·      "
        f"$\\bf{{{feasibility_label}}}$ raw-EI feasible "
        f"({n_feasible_estimators}/{n_total_estimators}; EI > {feasibility_threshold:.2f})"
    )
    fig.text(
        x = 0.50, y = 0.895, s = stat_text,
        ha = "center", va = "top",
        fontsize = 9.5, color = "#1a1a1a", linespacing = 1.55,
        bbox = {
            "boxstyle":  "round,pad=0.55",
            "facecolor": "#f1efe6",
            "edgecolor": "#d6d2c3",
            "linewidth": 0.6,
        },
    )

    if show:
        plt.show()
    return fig, np.array([forest_axis], dtype = object)




## universality superfigure
BG = "#FFFFFF"
TEXT = "#202020"
MUTED = "#666666"
LINE_GREY = "#000000"
LIGHT = LINE_GREY
SPINE = LINE_GREY
LEGEND_EDGE = "#B3B3B3"
DASHED_GUIDE = "#9A9A9A"
OBSERVED = "#000000"
COLORED_LINE_WIDTH = 1.05
DASHED_LINE_WIDTH = 0.75
## fig2 is exported via a ~0.819x downscale to 183 mm; source weights are set
## so axis/tick/divider/baseline render ~0.5 pt and text renders 7 pt / 8 pt.
AXIS_LINE_WIDTH = 0.61
LEGEND_BORDERPAD_A = 0.35
LEGEND_BORDERPAD_BC = 0.45
EQUIVALENCE = "#000000"
EQUIVALENCE_LINE = LINE_GREY
DELTA_EQ_FALLBACK = 0.05
FEASIBILITY_THRESHOLD = 0.05
LOWER_DOMAIN_LABELS = {
    "Earth & Physical Sciences": "Earth & Physical",
    "Life Sciences & Medicine": "Life Sciences",
    "Technology & Information": "Technology & Info",
    "Trade & Institutions": "Trade & Institutions",
    "Transportation & Infrastructure": "Transport & Infra",
}
TOP_DOMAIN_LABELS = {
    "Earth & Physical Sciences": "Earth & Physical Sciences",
    "Life Sciences & Medicine": "Life Sciences & Medicine",
    "Technology & Information": "Technology & Information",
    "Trade & Institutions": "Trade & Institutions",
    "Transportation & Infrastructure": "Transport & Infrastructure",
}
DISCIPLINE_LABELS = {
    "Email Exchanges": "Email\nExchanges",
    "Social Networks": "Social\nNetworks",
    "Interaction Networks": "Interaction\nNetworks",
    "Online Forums": "Online\nForums",
    "Knowledge Graphs": "Knowledge\nGraphs",
    "International Development": "International\nDevelopment",
    "Cryptocurrency": "Crypto-\ncurrency",
}
FONT_FAMILY = ["Arial", "Helvetica", "sans-serif"]
PANEL_TEXT_SIZE = 8.54   ## renders ~7 pt after the ~0.819x export downscale
PANEL_LABELS = {"a", "b", "c"}
PANEL_LABEL_SIZE = 9.76   ## renders ~8 pt after the ~0.819x export downscale
AXIS_LABEL_COLOR = "#000000"
AXIS_LABEL_SIZE = 8.0
AXIS_TICK_LENGTH = 2.4
AXIS_TICK_PAD = 1.6
BASE_ROW_HEIGHT_RATIO = 1.62
TOP_ROW_HEIGHT_RATIO = 1.50
TRANSFER_LAYOUT_SPAN = 1.0
PANEL_A_AXIS_SCALE = 0.90
PANEL_B_HEIGHT_SCALE = 0.92
LOWER_PANELS_VERTICAL_SHIFT = 0.045
PANEL_C_VERTICAL_SHIFT = 0.08
CONSENSUS_YLIM = (-0.14, 1.14)
## domain dividers span only the labelled tick range so their height matches
## the (tick-bounded) left y-axis line: structure ticks span [0, 1]; transfer
## ticks span the full y-limits.
CONSENSUS_DIVIDER_Y_SPAN = (
    (0.0 - CONSENSUS_YLIM[0]) / (CONSENSUS_YLIM[1] - CONSENSUS_YLIM[0]),
    (1.0 - CONSENSUS_YLIM[0]) / (CONSENSUS_YLIM[1] - CONSENSUS_YLIM[0]),
)
TRANSFER_DIVIDER_Y_SPAN = (0.0, 1.0)


def _format_decimal(value: float, decimals: int = 2, signed: bool = False) -> str:

    """
    Desc:
        Format numeric labels while collapsing exact zero to `0`.

    Args:
        value: Numeric value to format.
        decimals: Number of decimal places for non-zero values.
        signed: Whether to include an explicit sign for non-zero values.

    Returns:
        Formatted numeric label.
    """

    if np.isclose(value, 0.0):
        return "0"
    sign = "+" if signed else ""
    return f"{value:{sign}.{decimals}f}"


def _discipline_label(discipline: object) -> str:

    """
    Desc:
        Format long discipline names for compact x-axis display.

    Args:
        discipline: Discipline name to format.

    Returns:
        Display label for the discipline.
    """

    discipline_text = str(discipline)
    return DISCIPLINE_LABELS.get(discipline_text, discipline_text)


def _apply_panel_lettering(fig: Figure) -> None:

    """
    Desc:
        Apply the shared sans-serif family and uniform 8 pt lettering to a
        finished figure.

    Args:
        fig: Figure whose text artists should be normalized.

    Returns:
        None.
    """

    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(FONT_FAMILY)
        text_artist.set_fontstyle("normal")
        if text_artist.get_text() in PANEL_LABELS:
            text_artist.set_fontsize(PANEL_LABEL_SIZE)
            text_artist.set_fontweight("bold")
            continue

        text_artist.set_fontsize(PANEL_TEXT_SIZE)


## ----------------------------------------------------------------------------
## prediction consensus transfer cache
## ----------------------------------------------------------------------------
def _cache_matches(
    cached: Mapping[str, object],
    data: pd.DataFrame,
    models: Mapping[str, object],
    target: str,
    n_repeats: int,
    random_state: int,
    ) -> bool:

    """
    Desc:
        Check whether a cached prediction consensus transfer payload matches the current
        data and evaluation configuration.

    Args:
        cached: Loaded cache payload.
        data: Processed data used by the evaluation.
        models: Mapping of model names to estimator objects.
        target: Target capacity column.
        n_repeats: Number of resampling repeats.
        random_state: Base random state.

    Returns:
        True when cache metadata is compatible, otherwise False.
    """

    expected = {
        "n_obs": len(data),
        "random_state": random_state,
        "n_repeats": n_repeats,
        "model_names": sorted(models.keys()),
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            return False
    if cached.get("target", target) != target:
        return False
    required_payload = {
        "results_dict_domain",
        "results_dict_5fold",
        "results_dict_10fold",
        "results_data_domain_consensus",
        "results_data_5fold",
        "results_data_10fold",
    }
    return required_payload.issubset(set(cached.keys()))


def load_or_compute_transfer_consensus_results(
    data: pd.DataFrame,
    models: Mapping[str, object],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str,
    cache_path: str | Path,
    n_repeats: int,
    random_state: int,
    n_jobs: int = -1,
    force_recompute: bool = False,
    require_cache: bool = False,
    ) -> tuple[dict[str, object], str]:

    """
    Desc:
        Load cached consensus/transfer results or recompute them with the same
        LOGO and k-fold evaluation pipeline used by the universality notebook.

    Args:
        data: Processed data used for evaluation.
        models: Mapping of model names to estimator objects.
        feat_x: Graph invariant feature columns.
        feat_z: Process signature feature columns.
        target: Target capacity column.
        cache_path: Path for the consensus/transfer cache.
        n_repeats: Number of resampling repeats.
        random_state: Base random state.
        n_jobs: Number of parallel jobs passed to resampling helpers.
        force_recompute: Whether to ignore any compatible cache.
        require_cache: Whether to raise instead of computing when the cache is
            missing or incompatible.

    Returns:
        Tuple of cache payload and source label.

    Raises:
        ValueError: If feature lists are empty.
    """

    if not feat_x:
        raise ValueError("feat_x must contain at least one column name")
    if not feat_z:
        raise ValueError("feat_z must contain at least one column name")

    resolved_cache_path = Path(cache_path)
    if not force_recompute and resolved_cache_path.exists():
        with open(file = resolved_cache_path, mode = "rb") as file_handle:
            cached = pickle.load(file_handle)
        metadata = cached.get("metadata", {})
        required_transfer_payload = {
            "predicts_dict_domain",
            "predicts_dict_5fold",
            "predicts_dict_10fold",
            "results_data_consensus",
            "results_data_5fold",
            "results_data_10fold",
        }
        expected_metadata = {
            "n_obs": len(data),
            "n_repeats": n_repeats,
            "random_state": random_state,
            "model_names": sorted(models.keys()),
            "target": target,
            "feat_x": list(feat_x),
            "feat_z": list(feat_z),
        }
        if (
            isinstance(metadata, Mapping)
            and all(metadata.get(key) == value for key, value in expected_metadata.items())
            and required_transfer_payload.issubset(cached)
        ):
            return {
                "results_dict_domain": cached["predicts_dict_domain"],
                "results_dict_5fold": cached["predicts_dict_5fold"],
                "results_dict_10fold": cached["predicts_dict_10fold"],
                "results_data_domain_consensus": cached["results_data_consensus"],
                "results_data_5fold": cached["results_data_5fold"],
                "results_data_10fold": cached["results_data_10fold"],
            }, "cached transfer notebook results"
        if _cache_matches(
            cached = cached,
            data = data,
            models = models,
            target = target,
            n_repeats = n_repeats,
            random_state = random_state,
        ):
            return cached, "cached consensus/transfer results"

    if require_cache:
        raise FileNotFoundError(
            f"A compatible transfer cache was not found at {resolved_cache_path}. "
            "Run notebooks/transfer.ipynb through its post-processing cell, "
            "then rerun notebooks/universality.ipynb."
        )

    results_dict_domain: dict[str, np.ndarray] = dict()
    results_dict_5fold: dict[str, np.ndarray] = dict()
    results_dict_10fold: dict[str, np.ndarray] = dict()
    transfer_5fold_dict: dict[str, pd.DataFrame] = dict()
    transfer_10fold_dict: dict[str, pd.DataFrame] = dict()

    for model_name, model in models.items():
        frontier_domain, y_pred_domain = logo_cross_valid(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            group = "domain",
            n_repeats = n_repeats,
            random_state = random_state,
            n_jobs = n_jobs,
        )
        frontier_domain["model"] = model_name
        results_dict_domain[model_name] = np.asarray(y_pred_domain, dtype = float)

        frontier_5fold, y_pred_5fold = kfold_cross_valid(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            n_splits = 5,
            n_repeats = n_repeats,
            random_state = random_state,
            n_jobs = n_jobs,
        )
        frontier_5fold["model"] = model_name
        results_dict_5fold[model_name] = np.asarray(y_pred_5fold, dtype = float)
        transfer_5fold_dict[model_name] = frontier_5fold

        frontier_10fold, y_pred_10fold = kfold_cross_valid(
            data = data,
            feat_x = feat_x,
            feat_z = feat_z,
            estimator_c = model.estimator_c,
            estimator_r = model.estimator_r,
            target = target,
            n_splits = 10,
            n_repeats = n_repeats,
            random_state = random_state,
            n_jobs = n_jobs,
        )
        frontier_10fold["model"] = model_name
        results_dict_10fold[model_name] = np.asarray(y_pred_10fold, dtype = float)
        transfer_10fold_dict[model_name] = frontier_10fold

    results_data_domain_consensus = compile_prediction_consensus(
        predictions = results_dict_domain,
        data = data,
        target = target,
        group = "domain",
    )
    results_data_5fold = compile_domain_transfer(results = transfer_5fold_dict)
    results_data_10fold = compile_domain_transfer(results = transfer_10fold_dict)

    payload: dict[str, object] = {
        "results_dict_domain": results_dict_domain,
        "results_dict_5fold": results_dict_5fold,
        "results_dict_10fold": results_dict_10fold,
        "results_data_domain_consensus": results_data_domain_consensus,
        "results_data_5fold": results_data_5fold,
        "results_data_10fold": results_data_10fold,
        "n_obs": len(data),
        "n_repeats": n_repeats,
        "random_state": random_state,
        "model_names": sorted(models.keys()),
        "target": target,
        "feat_x": list(feat_x),
        "feat_z": list(feat_z),
    }
    resolved_cache_path.parent.mkdir(parents = True, exist_ok = True)
    with open(file = resolved_cache_path, mode = "wb") as file_handle:
        pickle.dump(obj = payload, file = file_handle)
    return payload, "freshly computed consensus/transfer results"


## ----------------------------------------------------------------------------
## summaries
## ----------------------------------------------------------------------------
def _domain_order(
    data: pd.DataFrame,
    domain_palette: Mapping[str, str],
    domain_column: str,
    ) -> list[str]:

    """
    Desc:
        Resolve domain order from palette order plus any remaining domains.

    Args:
        data: Processed data with a domain column.
        domain_palette: Ordered domain palette.
        domain_column: Domain column name.

    Returns:
        Ordered domain names.

    Raises:
        ValueError: If no domains are available.
    """

    observed_domains = set(data[domain_column].dropna())
    ordered = [domain for domain in domain_palette if domain in observed_domains]
    ordered.extend(
        sorted(
            domain for domain in data[domain_column].dropna().unique()
            if domain not in set(ordered)
        )
    )
    if not ordered:
        raise ValueError("No domains available for plotting")
    return ordered


def _ordered_disciplines(
    y_true_log: np.ndarray,
    domains: np.ndarray,
    disciplines: np.ndarray,
    domain_order: Sequence[str],
    ) -> tuple[list[str], list[tuple[str, int, int]]]:

    """
    Desc:
        Order disciplines within each domain by observed median log-capacity.

    Args:
        y_true_log: Log-transformed observed capacity values.
        domains: Domain labels aligned to y_true_log.
        disciplines: Discipline labels aligned to y_true_log.
        domain_order: Ordered domain names.

    Returns:
        Ordered disciplines and inclusive domain ranges over that order.
    """

    ordered = list()
    ranges = list()
    cursor = 0
    for domain in domain_order:
        domain_mask = domains == domain
        summaries = list()
        for discipline in pd.unique(disciplines[domain_mask]):
            mask = domain_mask & (disciplines == discipline) & np.isfinite(y_true_log)
            if mask.any():
                summaries.append((discipline, float(np.median(y_true_log[mask]))))
        domain_disciplines = [
            discipline for discipline, _ in sorted(summaries, key = lambda item: item[1])
        ]
        if not domain_disciplines:
            continue
        start = cursor
        ordered.extend(domain_disciplines)
        cursor += len(domain_disciplines)
        ranges.append((domain, start, cursor - 1))
    return ordered, ranges


def _scale_jointly(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:

    """
    Desc:
        Scale arrays jointly onto [0, 1] using their shared finite range.

    Args:
        *arrays: Numeric arrays to scale.

    Returns:
        Tuple of scaled arrays.
    """

    finite_blocks = [np.asarray(array, dtype = float)[np.isfinite(array)] for array in arrays]
    finite_blocks = [block for block in finite_blocks if block.size > 0]
    if not finite_blocks:
        return tuple(np.zeros_like(array, dtype = float) for array in arrays)
    values = np.concatenate(finite_blocks)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return tuple(np.zeros_like(array, dtype = float) for array in arrays)
    return tuple((array - low) / (high - low) for array in arrays)


def _summarize_consensus_panel(
    y_true_log: np.ndarray,
    disciplines: np.ndarray,
    discipline_domain_map: Mapping[str, str],
    ordered_disciplines: Sequence[str],
    domain_order: Sequence[str],
    results_dict_domain: Mapping[str, np.ndarray],
    results_data_domain_consensus: pd.DataFrame,
    domain_palette: Mapping[str, str],
    ) -> dict[str, object]:

    """
    Desc:
        Build the prediction consensus panel from domain-LOGO predictions.

    Args:
        y_true_log: Log-transformed observed capacity values.
        disciplines: Discipline labels aligned to y_true_log.
        discipline_domain_map: Mapping from discipline to domain.
        ordered_disciplines: Shared discipline order.
        domain_order: Ordered domain names.
        results_dict_domain: Domain-LOGO predictions by model.
        results_data_domain_consensus: Prediction consensus result table.
        domain_palette: Domain color palette.

    Returns:
        Dictionary of arrays and domain statistics for the consensus panel.

    Raises:
        ValueError: If a prediction vector length differs from y_true_log.
    """

    x_pos = {discipline: index for index, discipline in enumerate(ordered_disciplines)}
    obs_med = np.full(shape = len(ordered_disciplines), fill_value = np.nan)
    pred_med = np.full(shape = len(ordered_disciplines), fill_value = np.nan)
    pred_q1 = np.full(shape = len(ordered_disciplines), fill_value = np.nan)
    pred_q3 = np.full(shape = len(ordered_disciplines), fill_value = np.nan)
    point_colors = list()

    prediction_arrays = dict()
    for model_name, prediction in results_dict_domain.items():
        prediction_array = np.asarray(prediction, dtype = float)
        if prediction_array.shape[0] != y_true_log.shape[0]:
            raise ValueError(
                f"Prediction length for {model_name} is {prediction_array.shape[0]}, "
                f"expected {y_true_log.shape[0]}"
            )
        prediction_arrays[model_name] = prediction_array

    for discipline in ordered_disciplines:
        index = x_pos[discipline]
        mask = (disciplines == discipline) & np.isfinite(y_true_log)
        if mask.any():
            obs_med[index] = np.median(y_true_log[mask])

        model_values = list()
        for prediction in prediction_arrays.values():
            valid = mask & np.isfinite(prediction)
            if valid.any():
                model_values.append(float(np.median(prediction[valid])))
        if model_values:
            pred_med[index] = np.median(model_values)
            pred_q1[index] = np.quantile(model_values, 0.25)
            pred_q3[index] = np.quantile(model_values, 0.75)
        point_colors.append(domain_palette.get(discipline_domain_map.get(discipline, ""), "#888888"))

    obs_scaled, pred_scaled, q1_scaled, q3_scaled = _scale_jointly(
        obs_med,
        pred_med,
        pred_q1,
        pred_q3,
    )

    domain_stats = dict()
    if {"group", "ci"}.issubset(set(results_data_domain_consensus.columns)):
        for domain in domain_order:
            domain_disciplines = [
                discipline for discipline in ordered_disciplines
                if discipline_domain_map.get(discipline) == domain
            ]
            domain_mask = np.isin(disciplines, domain_disciplines) & np.isfinite(y_true_log)
            ei_values = list()
            for prediction in prediction_arrays.values():
                valid = domain_mask & np.isfinite(prediction)
                if int(np.sum(valid)) >= 2:
                    ei_values.append(
                        _efficiency_index(
                            y_true = y_true_log[valid],
                            y_pred = prediction[valid],
                        )
                    )
            domain_rows = results_data_domain_consensus.loc[results_data_domain_consensus["group"] == domain]
            if ei_values and not domain_rows.empty:
                domain_stats[domain] = float(domain_rows["ci"].median())

    return {
        "obs_scaled": obs_scaled,
        "pred_scaled": pred_scaled,
        "q1_scaled": q1_scaled,
        "q3_scaled": q3_scaled,
        "point_colors": point_colors,
        "domain_stats": domain_stats,
    }


def _summarize_transfer_panel(
    y_true_log: np.ndarray,
    disciplines: np.ndarray,
    ordered_disciplines: Sequence[str],
    results_dict_domain: Mapping[str, np.ndarray],
    results_dict_5fold: Mapping[str, np.ndarray],
    results_dict_10fold: Mapping[str, np.ndarray],
    results_data_5fold: pd.DataFrame,
    results_data_10fold: pd.DataFrame,
    equivalence_fallback: float,
    feasibility_threshold: float,
    ) -> dict[str, object]:

    """
    Desc:
        Build paired EI differences as random-split resampling minus domain
        LOGO, using 5-fold and 10-fold random splits as separate comparisons.
        Domain LOGO and k-fold EI values are summarized separately and only
        compared at the downstream delta step.

    Args:
        y_true_log: Log-transformed observed capacity values.
        disciplines: Discipline labels aligned to y_true_log.
        ordered_disciplines: Shared discipline order.
        results_dict_domain: Domain LOGO predicted values by model.
        results_dict_5fold: Random 5-fold predicted values by model.
        results_dict_10fold: Random 10-fold predicted values by model.
        results_data_5fold: Random 5-fold frontier metrics.
        results_data_10fold: Random 10-fold frontier metrics.
        equivalence_fallback: Fallback equivalence margin.
        feasibility_threshold: Minimum raw EI used for feasibility filtering.

    Returns:
        Dictionary of transfer-panel summaries and annotations.

    Raises:
        ValueError: If required transfer inputs are missing or incompatible.
    """

    required_by_table = {
        "results_data_5fold": {"model", "ei"},
        "results_data_10fold": {"model", "ei"},
    }
    for table_name, table in {
        "results_data_5fold": results_data_5fold,
        "results_data_10fold": results_data_10fold,
    }.items():
        missing_columns = sorted(required_by_table[table_name] - set(table.columns))
        if missing_columns:
            raise ValueError(f"{table_name} missing required columns: {missing_columns}")

    shared_models = sorted(
        set(results_dict_domain.keys())
        & set(results_dict_5fold.keys())
        & set(results_dict_10fold.keys())
    )
    if not shared_models:
        raise ValueError("No shared model predictions available for transfer summary")

    expected_n_obs = int(y_true_log.shape[0])
    for table_name, prediction_dict in {
        "results_dict_domain": results_dict_domain,
        "results_dict_5fold": results_dict_5fold,
        "results_dict_10fold": results_dict_10fold,
    }.items():
        for model_name in shared_models:
            prediction_array = np.asarray(prediction_dict[model_name], dtype = float)
            if prediction_array.shape[0] != expected_n_obs:
                raise ValueError(
                    f"{table_name}[{model_name}] has length {prediction_array.shape[0]}, "
                    f"expected {expected_n_obs}"
                )

    fold_iqrs: dict[str, tuple[float, float]] = dict()
    fold_margins: dict[str, float] = dict()
    baseline_residuals: list[np.ndarray] = list()
    for fold_label, fold_table in (("5-fold", results_data_5fold), ("10-fold", results_data_10fold)):
        cleaned = fold_table[["model", "ei"]].dropna(subset = ["ei"])
        per_model_mean = cleaned.groupby("model", observed = True)["ei"].mean()
        residual = (
            cleaned["ei"].astype(float)
            - cleaned["model"].map(per_model_mean).astype(float)
        ).dropna().to_numpy(dtype = float)
        if residual.size:
            baseline_residuals.append(residual)
        iqr = (0.0, 0.0)
        if residual.size >= 2:
            iqr = tuple(np.quantile(residual, [0.25, 0.75]))
        try:
            margin = _spec_marginal_delta(
                results = cleaned.assign(reference = "baseline"),
                feat_value = ["ei"],
                label_ref = "reference",
                value_ref = "baseline",
                method = "iqr",
                scale = 1.0,
                decimals = 2,
            )
        except ValueError:
            margin = equivalence_fallback
        fold_iqrs[fold_label] = iqr
        fold_margins[fold_label] = margin

    domain_records: list[dict[str, object]] = list()
    kfold_records: list[dict[str, object]] = list()
    for model_name in shared_models:
        domain_prediction = np.asarray(results_dict_domain[model_name], dtype = float)
        baseline_predictions = {
            "5-fold": np.asarray(results_dict_5fold[model_name], dtype = float),
            "10-fold": np.asarray(results_dict_10fold[model_name], dtype = float),
        }
        for discipline in ordered_disciplines:
            discipline_mask = disciplines == discipline
            domain_valid = (
                discipline_mask
                & np.isfinite(y_true_log)
                & np.isfinite(domain_prediction)
            )
            if int(np.sum(domain_valid)) < 1:
                continue

            domain_ei = float(
                _efficiency_index(
                    y_true = y_true_log[domain_valid],
                    y_pred = domain_prediction[domain_valid],
                )
            )
            domain_records.append(
                {
                    "model": model_name,
                    "discipline": discipline,
                    "domain_ei": domain_ei,
                    "feasible": domain_ei > feasibility_threshold,
                }
            )

            for baseline_label, baseline_prediction in baseline_predictions.items():
                baseline_valid = (
                    discipline_mask
                    & np.isfinite(y_true_log)
                    & np.isfinite(baseline_prediction)
                )
                if int(np.sum(baseline_valid)) < 1:
                    continue
                baseline_ei = float(
                    _efficiency_index(
                        y_true = y_true_log[baseline_valid],
                        y_pred = baseline_prediction[baseline_valid],
                    )
                )
                kfold_records.append(
                    {
                        "model": model_name,
                        "discipline": discipline,
                        "baseline": baseline_label,
                        "baseline_ei": baseline_ei,
                    }
                )

    if not domain_records or not kfold_records:
        raise ValueError("No valid domain-transfer deltas available for panel c")

    domain_ei_data = pd.DataFrame.from_records(domain_records)
    kfold_ei_data = pd.DataFrame.from_records(kfold_records)
    combined_delta = kfold_ei_data.merge(
        right = domain_ei_data,
        on = ["model", "discipline"],
        how = "inner",
    )
    if combined_delta.empty:
        raise ValueError("No matched domain LOGO and k-fold deltas available for panel c")

    combined_delta["delta_ei"] = (
        combined_delta["baseline_ei"] - combined_delta["domain_ei"]
    )
    combined_for_plot = combined_delta.loc[combined_delta["feasible"]].copy()
    if combined_for_plot.empty:
        combined_for_plot = combined_delta.copy()

    baseline_iqr = (0.0, 0.0)
    if baseline_residuals:
        baseline_residual = np.concatenate(baseline_residuals)
        if baseline_residual.size >= 2:
            baseline_iqr = tuple(np.quantile(baseline_residual, [0.25, 0.75]))
    finite_margins = [
        float(margin)
        for margin in fold_margins.values()
        if np.isfinite(margin)
    ]
    equivalence_margin = max(finite_margins) if finite_margins else equivalence_fallback

    delta_summary = (
        combined_for_plot.groupby("discipline", observed = True)["delta_ei"]
        .agg(
            median = "median",
            q1 = lambda values: values.quantile(0.25),
            q3 = lambda values: values.quantile(0.75),
            n = "count",
        )
        .reindex(ordered_disciplines)
        .reset_index()
    )

    def _baseline_median(label: str) -> float:
        rows = combined_for_plot.loc[combined_for_plot["baseline"] == label, "delta_ei"]
        return float(rows.median()) if not rows.empty else float("nan")

    n_feasible = int(combined_delta["feasible"].sum())
    n_total = int(len(combined_delta))

    return {
        "baseline_iqr_5fold": fold_iqrs["5-fold"],
        "baseline_iqr_10fold": fold_iqrs["10-fold"],
        "baseline_iqr": baseline_iqr,
        "equivalence_margin_5fold": fold_margins["5-fold"],
        "equivalence_margin_10fold": fold_margins["10-fold"],
        "equivalence_margin": equivalence_margin,
        "delta_summary": delta_summary,
        "overall_delta_5fold": _baseline_median("5-fold"),
        "overall_delta_10fold": _baseline_median("10-fold"),
        "n_feasible": n_feasible,
        "n_total": n_total,
    }


## ----------------------------------------------------------------------------
## lower panel rendering
## ----------------------------------------------------------------------------
def _style_lower_axis(axis: Axes) -> None:

    """
    Desc:
        Apply shared visual styling to lower diagnostic axes.

    Args:
        axis: Axis to style.

    Returns:
        None.
    """

    axis.set_facecolor(BG)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(SPINE)
    axis.spines[["left", "bottom"]].set_linewidth(AXIS_LINE_WIDTH)
    axis.tick_params(
        axis = "y",
        color = SPINE,
        labelcolor = AXIS_LABEL_COLOR,
        labelsize = AXIS_LABEL_SIZE,
        left = True,
        right = False,
        length = AXIS_TICK_LENGTH,
        pad = AXIS_TICK_PAD,
    )
    axis.tick_params(axis = "x", color = SPINE, labelcolor = AXIS_LABEL_COLOR, labelsize = AXIS_LABEL_SIZE, length = 0)
    axis.grid(axis = "y", color = LIGHT, lw = 0.55, alpha = 0.95)


def _draw_left_axis_line(axis: Axes) -> None:

    """
    Desc:
        Draw a custom left y-axis line spanning only the labelled tick range
        (no extension beyond the lowest and highest visible ticks).

    Args:
        axis: Axis receiving the left line.

    Returns:
        None.
    """

    axis.spines["left"].set_visible(False)
    y_lo, y_hi = axis.get_ylim()
    ticks = [t for t in axis.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    lo, hi = (min(ticks), max(ticks)) if ticks else (y_lo, y_hi)
    axis.plot(
        [0.0, 0.0],
        [lo, hi],
        transform = axis.get_yaxis_transform(),
        color = SPINE,
        lw = AXIS_LINE_WIDTH,
        alpha = 1.0,
        clip_on = False,
        zorder = 5,
    )


def _draw_consensus_panel(
    axis: Axes,
    transfer_axis: Axes,
    consensus_summary: Mapping[str, object],
    group_ranges: Sequence[tuple[str, int, int]],
    domain_palette: Mapping[str, str],
    domain_labels: Mapping[str, str],
    panel_label: str,
    ) -> None:

    """
    Desc:
        Render the prediction consensus panel.

    Args:
        axis: Consensus axis.
        transfer_axis: Transfer axis sharing domain separators.
        consensus_summary: Summary arrays from `_summarize_consensus_panel`.
        group_ranges: Inclusive domain ranges across ordered disciplines.
        domain_palette: Domain color palette.
        domain_labels: Domain display labels.
        panel_label: Panel letter.

    Returns:
        None.
    """

    def _plot_extended_step(
        x_values: np.ndarray,
        y_values: np.ndarray,
        color: str,
        linewidth: float,
        linestyle: str,
        alpha: float,
        zorder: float,
        left_edge_override: float | None = None,
        ) -> None:

        """
        Desc:
            Draw a step trace that extends half a discipline bin beyond the
            outermost finite medians in each contiguous run.

        Args:
            x_values: Discipline-center x positions for one domain block.
            y_values: Median values aligned to `x_values`.
            color: Step line color.
            linewidth: Step line width.
            linestyle: Step line style.
            alpha: Step line opacity.
            zorder: Matplotlib draw order.
            left_edge_override: Optional left edge for the first contiguous run.

        Returns:
            None.
        """

        finite_indices = np.flatnonzero(np.isfinite(y_values))
        if len(finite_indices) == 0:
            return
        split_indices = np.where(np.diff(finite_indices) > 1)[0] + 1
        for run_indices in np.split(finite_indices, split_indices):
            run_x = x_values[run_indices]
            run_y = y_values[run_indices]
            left_edge = float(run_x[0] - 0.5)
            if left_edge_override is not None and run_indices[0] == 0:
                left_edge = float(left_edge_override)
            midpoint_edges = 0.5 * (run_x[:-1] + run_x[1:])
            right_edge = float(run_x[-1] + 0.5)
            step_x = np.concatenate(
                (
                    np.array([left_edge], dtype = float),
                    midpoint_edges.astype(float, copy = False),
                    np.array([right_edge], dtype = float),
                )
            )
            step_y = np.append(run_y, run_y[-1])
            axis.step(
                x = step_x,
                y = step_y,
                where = "post",
                color = color,
                lw = linewidth,
                ls = linestyle,
                alpha = alpha,
                zorder = zorder,
            )

    obs_scaled = consensus_summary["obs_scaled"]
    pred_scaled = consensus_summary["pred_scaled"]
    q1_scaled = consensus_summary["q1_scaled"]
    q3_scaled = consensus_summary["q3_scaled"]
    point_colors = consensus_summary["point_colors"]
    domain_stats = consensus_summary["domain_stats"]

    if panel_label:
        axis.text(
            x = -0.055,
            y = 1.20,
            s = panel_label,
            transform = axis.transAxes,
            ha = "right",
            va = "center",
            color = TEXT,
            fontsize = 11.0,
            fontweight = "bold",
        )

    for domain, start, end in group_ranges:
        segment = np.arange(start = start, stop = end + 1, dtype = float)
        observed_segment = obs_scaled[start:end + 1]
        predicted_segment = pred_scaled[start:end + 1]
        domain_color = point_colors[start] if start < len(point_colors) else domain_palette.get(domain, "#888888")
        left_edge_override = axis.get_xlim()[0] if start == 0 else None
        _plot_extended_step(
            x_values = segment,
            y_values = observed_segment,
            color = OBSERVED,
            linewidth = 0.75,
            linestyle = "-",
            alpha = 1.0,
            zorder = 2,
            left_edge_override = left_edge_override,
        )
        if np.isfinite(predicted_segment).any():
            _plot_extended_step(
                x_values = segment,
                y_values = predicted_segment,
                color = domain_color,
                linewidth = COLORED_LINE_WIDTH,
                linestyle = "-",
                alpha = 0.72,
                zorder = 3,
                left_edge_override = left_edge_override,
            )

    for index, color in enumerate(point_colors):
        x_coordinate = float(index)
        if np.isfinite(obs_scaled[index]):
            axis.plot(
                [x_coordinate],
                [obs_scaled[index]],
                marker = "o",
                markersize = 5.0,
                markerfacecolor = OBSERVED,
                markeredgecolor = OBSERVED,
                markeredgewidth = 0.0,
                linestyle = "None",
                zorder = 4,
            )
        if np.isfinite(pred_scaled[index]):
            axis.plot(
                [x_coordinate],
                [pred_scaled[index]],
                marker = "o",
                markersize = 5.0,
                markerfacecolor = color,
                markeredgecolor = color,
                markeredgewidth = 0.0,
                linestyle = "None",
                zorder = 5,
            )

    for range_index, (domain, start, end) in enumerate(group_ranges):
        if range_index > 0:
            axis.axvline(
                x = start - 0.5,
                ymin = CONSENSUS_DIVIDER_Y_SPAN[0],
                ymax = CONSENSUS_DIVIDER_Y_SPAN[1],
                color = SPINE,
                lw = AXIS_LINE_WIDTH,
                alpha = 1.0,
                zorder = 1,
            )
            transfer_axis.axvline(
                x = start - 0.5,
                ymin = TRANSFER_DIVIDER_Y_SPAN[0],
                ymax = TRANSFER_DIVIDER_Y_SPAN[1],
                color = SPINE,
                lw = AXIS_LINE_WIDTH,
                alpha = 1.0,
                clip_on = False,
                zorder = 1,
            )
        midpoint = 0.5 * (start + end)
        if domain in domain_stats:
            ci_value = domain_stats[domain]
            axis.text(
                x = midpoint,
                y = 1.0,
                s = f"CI = {_format_decimal(value = ci_value, decimals = 2)}",
                ha = "center",
                va = "bottom",
                color = TEXT,
                fontsize = 6.4,
                fontweight = "bold",
            )

    for reference_level in (0.0,):
        axis.axhline(
            y = reference_level,
            color = EQUIVALENCE,
            lw = AXIS_LINE_WIDTH,
            ls = "-",
            alpha = 1.0,
            zorder = 1,
        )
    axis.set_ylim(bottom = CONSENSUS_YLIM[0], top = CONSENSUS_YLIM[1])
    axis.set_yticks(ticks = [0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(labels = ["0", "0.25", "0.50", "0.75", "1"], color = AXIS_LABEL_COLOR, fontsize = AXIS_LABEL_SIZE)
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    _draw_left_axis_line(axis = axis)
    axis.tick_params(axis = "x", bottom = False, labelbottom = False)
    axis.tick_params(
        axis = "y",
        color = SPINE,
        labelcolor = AXIS_LABEL_COLOR,
        labelsize = AXIS_LABEL_SIZE,
        left = True,
        right = False,
        length = AXIS_TICK_LENGTH,
        pad = AXIS_TICK_PAD,
    )
    axis.grid(visible = False, axis = "y")
    axis.set_ylabel(ylabel = "Normalized Capacity", color = AXIS_LABEL_COLOR, fontsize = AXIS_LABEL_SIZE, labelpad = 5)


def _draw_transfer_panel(
    axis: Axes,
    transfer_summary: Mapping[str, object],
    group_ranges: Sequence[tuple[str, int, int]],
    ordered_disciplines: Sequence[str],
    discipline_domain_map: Mapping[str, str],
    domain_palette: Mapping[str, str],
    panel_label: str,
    transfer_ylim: tuple[float, float],
    ) -> None:

    """
    Desc:
        Render the resampling EI difference panel.

    Args:
        axis: Transfer axis.
        transfer_summary: Summary values from `_summarize_transfer_panel`.
        group_ranges: Inclusive domain ranges across ordered disciplines.
        ordered_disciplines: Shared discipline order.
        discipline_domain_map: Mapping from discipline to domain.
        domain_palette: Domain color palette.
        panel_label: Panel letter.
        transfer_ylim: Y-axis limits for resampling EI differences.

    Returns:
        None.
    """

    x_pos = {discipline: index for index, discipline in enumerate(ordered_disciplines)}
    x = np.arange(start = 0, stop = len(ordered_disciplines))
    delta_summary = transfer_summary["delta_summary"]
    baseline_iqr = transfer_summary["baseline_iqr"]
    equivalence_margin = transfer_summary["equivalence_margin"]
    transfer_tick_candidates = [transfer_ylim[0], -0.1, 0.0, 0.1, transfer_ylim[1]]
    transfer_ticks = []
    for tick in transfer_tick_candidates:
        if transfer_ylim[0] <= tick <= transfer_ylim[1] and not any(
            np.isclose(a = tick, b = existing_tick)
            for existing_tick in transfer_ticks
        ):
            transfer_ticks.append(float(tick))

    if panel_label:
        axis.text(
            x = -0.055,
            y = 1.22,
            s = panel_label,
            transform = axis.transAxes,
            ha = "right",
            va = "center",
            color = TEXT,
            fontsize = 11.0,
            fontweight = "bold",
        )
    if baseline_iqr[1] > baseline_iqr[0]:
        axis.axhspan(
            ymin = baseline_iqr[0],
            ymax = baseline_iqr[1],
            facecolor = "#DCEAF7",
            edgecolor = "none",
            alpha = 0.28,
            zorder = 1,
        )
    for reference_level in transfer_ticks:
        ## keep only reference lines at 0 and 1 (omit intermediate gridlines)
        if not (np.isclose(reference_level, 0.0) or np.isclose(reference_level, 1.0)):
            continue
        axis.axhline(
            y = reference_level,
            color = OBSERVED if reference_level == 0.0 else EQUIVALENCE,
            lw = 0.9 if reference_level == 0.0 else 0.75,
            ls = "-",
            alpha = 0.65 if reference_level == 0.0 else 0.90,
            zorder = 1,
        )

    for _, row in delta_summary.iterrows():
        discipline = row["discipline"]
        if pd.isna(discipline) or discipline not in x_pos:
            continue
        x_coordinate = float(x_pos[discipline])
        color = domain_palette.get(discipline_domain_map.get(discipline, ""), "#888888")
        has_iqr = np.isfinite(row["q1"]) and np.isfinite(row["q3"])
        has_median = np.isfinite(row["median"])
        if has_iqr and has_median:
            axis.plot(
                [x_coordinate, x_coordinate, x_coordinate],
                [float(row["q1"]), float(row["median"]), float(row["q3"])],
                color = color,
                lw = COLORED_LINE_WIDTH,
                marker = "o",
                markersize = 5.0,
                markerfacecolor = color,
                markeredgecolor = color,
                markeredgewidth = 0.0,
                markevery = [1],
                solid_capstyle = "round",
                zorder = 4,
            )
        elif has_iqr:
            axis.plot(
                [x_coordinate, x_coordinate],
                [float(row["q1"]), float(row["q3"])],
                color = color,
                lw = COLORED_LINE_WIDTH,
                solid_capstyle = "round",
                zorder = 3,
            )
        elif has_median:
            axis.plot(
                [x_coordinate],
                [float(row["median"])],
                marker = "o",
                markersize = 5.0,
                markerfacecolor = color,
                markeredgecolor = color,
                markeredgewidth = 0.0,
                linestyle = "None",
                zorder = 4,
            )

    for domain, start, end in group_ranges:
        domain_disciplines = set(ordered_disciplines[start:end + 1])
        domain_rows = delta_summary.loc[
            delta_summary["discipline"].isin(domain_disciplines)
        ].copy()
        finite_medians = pd.to_numeric(
            arg = domain_rows["median"],
            errors = "coerce",
        ).dropna()
        if finite_medians.empty:
            continue

        median_delta = float(finite_medians.median())
        axis.hlines(
            y = median_delta,
            xmin = start - 0.5,
            xmax = end + 0.5,
            color = domain_palette.get(domain, TEXT),
            lw = DASHED_LINE_WIDTH,
            linestyles = "--",
            alpha = 0.95,
            zorder = 2,
        )
        axis.text(
            x = 0.5 * (start + end),
            y = 0.955,
            s = rf"$\Delta$ EI = {_format_decimal(value = median_delta, decimals = 2, signed = True)}",
            transform = axis.get_xaxis_transform(),
            ha = "center",
            va = "top",
            color = TEXT,
            fontsize = 6.2,
            fontweight = "bold",
            clip_on = False,
        )

    axis.set_ylim(bottom = transfer_ylim[0], top = transfer_ylim[1])
    axis.set_yticks(ticks = transfer_ticks)
    axis.set_yticklabels(
        labels = [_format_decimal(value = tick, decimals = 1) for tick in transfer_ticks],
    )
    axis.set_ylabel(ylabel = r"$\Delta$ EI ($\mathit{k}$-Fold - Domain LOGO)", color = AXIS_LABEL_COLOR, fontsize = AXIS_LABEL_SIZE, labelpad = 6)
    axis.set_xticks(ticks = x)
    axis.set_xticklabels(
        labels = [_discipline_label(discipline) for discipline in ordered_disciplines],
        rotation = 90,
        rotation_mode = "anchor",
        ha = "right",
        va = "center",
        fontsize = AXIS_LABEL_SIZE,
        color = AXIS_LABEL_COLOR,
    )
    _style_lower_axis(axis = axis)
    for label, discipline in zip(axis.get_xticklabels(), ordered_disciplines):
        domain = discipline_domain_map.get(discipline, "")
        label.set_color(domain_palette.get(domain, AXIS_LABEL_COLOR))
    axis.grid(visible = False, axis = "y")
    axis.tick_params(
        axis = "x",
        pad = 2.6,
        length = 3.5,
        color = SPINE,
    )
    _draw_left_axis_line(axis = axis)


def _panel_a_legend_handles() -> list[Line2D]:

    """
    Desc:
        Build compact legend handles for the top calibration panel.

    Returns:
        Legend handle list for panel a.
    """

    return [
        Line2D(
            xdata = [0],
            ydata = [0],
            color = "#777777",
            marker = "o",
            linestyle = "None",
            markersize = 5.0,
            markeredgewidth = 0.0,
            label = "Discipline Median",
        ),
        Line2D(
            xdata = [0],
            ydata = [0],
            color = "#777777",
            lw = DASHED_LINE_WIDTH,
            ls = "--",
            alpha = 1.0,
            label = "Domain Median",
        ),
    ]


def _panel_b_legend_handles() -> list[Line2D]:

    """
    Desc:
        Build compact legend handles for the consensus panel.

    Returns:
        Legend handle list for panel b.
    """

    return [
        Line2D(
            xdata = [0],
            ydata = [0],
            color = "#777777",
            lw = COLORED_LINE_WIDTH,
            ls = "-",
            alpha = 1.0,
            marker = "o",
            markersize = 5.0,
            markeredgewidth = 0.0,
            label = "Estimated Median",
        ),
        Line2D(
            xdata = [0],
            ydata = [0],
            color = OBSERVED,
            lw = 1.0,
            ls = "-",
            alpha = 1.0,
            marker = "o",
            markersize = 5.0,
            markeredgewidth = 0.0,
            label = "Observed Median",
        ),
    ]


class _VerticalIntervalLegendHandle:

    """
    Desc:
        Store style metadata for a vertical interval legend glyph.

    Args:
        color: Interval and marker color.
        linewidth: Interval stroke width.
        markersize: Marker size in points.
        label: Legend label text.
    """

    def __init__(
        self,
        color: str,
        linewidth: float,
        markersize: float,
        label: str,
    ) -> None:

        self.color = color
        self.linewidth = linewidth
        self.markersize = markersize
        self._label = label

    def get_label(self) -> str:

        """
        Desc:
            Return the legend label for the custom handle.

        Returns:
            Legend label text.
        """

        return self._label


class _VerticalIntervalLegendHandler(HandlerBase):

    """
    Desc:
        Draw a vertical interval with a centered median marker in legends.
    """

    def legend_artist(
        self,
        legend: object,
        orig_handle: _VerticalIntervalLegendHandle,
        fontsize: float,
        handlebox: object,
    ) -> Line2D:

        """
        Desc:
            Render the custom legend glyph inside the legend handle box.

        Args:
            legend: Parent legend instance.
            orig_handle: Source handle metadata.
            fontsize: Active legend font size.
            handlebox: Legend drawing area.

        Returns:
            Center marker artist used as the representative legend artist.
        """

        del legend, fontsize

        center_x = handlebox.xdescent + (0.5 * handlebox.width)
        center_y = handlebox.ydescent + (0.5 * handlebox.height)
        lower_y = handlebox.ydescent - (0.35 * handlebox.height)
        upper_y = handlebox.ydescent + (1.35 * handlebox.height)
        transform = handlebox.get_transform()

        interval = Line2D(
            xdata = [center_x, center_x],
            ydata = [lower_y, upper_y],
            color = orig_handle.color,
            lw = orig_handle.linewidth,
            solid_capstyle = "round",
            clip_on = False,
            transform = transform,
        )
        median = Line2D(
            xdata = [center_x],
            ydata = [center_y],
            color = orig_handle.color,
            marker = "o",
            markersize = orig_handle.markersize,
            markeredgewidth = 0.0,
            ls = "None",
            clip_on = False,
            transform = transform,
        )
        handlebox.add_artist(interval)
        handlebox.add_artist(median)
        return median


def _panel_c_legend_handles() -> list[object]:

    """
    Desc:
        Build compact legend handles for the transfer panel.

    Returns:
        Legend handle list for panel c.
    """

    return [
        _VerticalIntervalLegendHandle(
            color = "#777777",
            linewidth = COLORED_LINE_WIDTH,
            markersize = 5.0,
            label = "Discipline Median [IQR]",
        ),
        Line2D(
            xdata = [0],
            ydata = [0],
            color = "#777777",
            lw = DASHED_LINE_WIDTH,
            ls = "--",
            alpha = 1.0,
            label = "Domain Median",
        ),
    ]


## ----------------------------------------------------------------------------
## public renderer
## ----------------------------------------------------------------------------
def plot_universality_superfigure(
    data: pd.DataFrame,
    models: Mapping[str, object],
    feat_x: Sequence[str],
    feat_z: Sequence[str],
    target: str,
    random_state: int,
    n_repeats: int,
    universality_cache_path: str | Path,
    transfer_consensus_cache_path: str | Path,
    regime_key: str = "domain_logo",
    model_to_paradigm: Mapping[str, str] | None = None,
    paradigm_order: Sequence[str] | None = None,
    domain_palette: Mapping[str, str] | None = None,
    domain_labels: Mapping[str, str] | None = None,
    lower_domain_labels: Mapping[str, str] | None = None,
    force_recompute: bool = False,
    require_cache: bool = False,
    n_jobs: int = -1,
    axis_limits: tuple[float, float] = (0.0, 20.0),
    transfer_ylim: tuple[float, float] = (-0.2, 0.2),
    figsize: tuple[float, float] = (9.00, 8.40),
    title: str = "Universality of Network Event Capacity",
    show: bool = True,
    ) -> tuple[Figure, dict[str, object], str]:

    """
    Desc:
        Render a composite universality superfigure with a Domain LOGO scatter
        strip above aligned consensus and transfer diagnostics.

    Args:
        data: Processed data containing target, domain, and discipline columns.
        models: Mapping of model names to estimator objects.
        feat_x: Graph invariant feature columns.
        feat_z: Process signature feature columns.
        target: Target capacity column.
        random_state: Base random state used by cached/evaluated results.
        n_repeats: Number of resampling repeats for consensus/transfer results.
        universality_cache_path: Cached paradigm predictions for top scatter.
        transfer_consensus_cache_path: Cache for lower-panel source results.
        regime_key: Prediction regime for the top scatter.
        model_to_paradigm: Mapping from model name to paradigm.
        paradigm_order: Ordered learning paradigm labels.
        domain_palette: Domain color palette.
        domain_labels: Domain labels for top scatter panels.
        lower_domain_labels: Domain labels for lower aligned panels.
        force_recompute: Whether to recompute lower-panel source results.
        require_cache: Whether lower-panel source results must already exist.
        n_jobs: Number of parallel jobs used if recomputation is needed.
        axis_limits: Shared x/y limits for top scatter panels.
        transfer_ylim: Y-axis limits for resampling EI differences.
        figsize: Figure size in inches.
        title: Main figure title.
        show: Whether to call plt.show().

    Returns:
        Tuple of figure, axes dictionary, and consensus/transfer source label.

    Raises:
        ValueError: If required data columns are missing or summaries are empty.
        FileNotFoundError: If cached top-scatter predictions are unavailable.
        RuntimeError: If cached top-scatter predictions are invalid.
    """

    required_columns = {target, "domain", "discipline"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if model_to_paradigm is None:
        model_to_paradigm = DEFAULT_MODEL_TO_PARADIGM
    if paradigm_order is None:
        paradigm_order = DEFAULT_PARADIGM_ORDER
    if domain_palette is None:
        domain_palette = DEFAULT_DOMAIN_PALETTE
    if domain_labels is None:
        domain_labels = TOP_DOMAIN_LABELS
    if lower_domain_labels is None:
        lower_domain_labels = LOWER_DOMAIN_LABELS

    paradigm_order = list(paradigm_order)
    domain_palette = dict(domain_palette)
    domain_labels = dict(domain_labels)
    lower_domain_labels = dict(lower_domain_labels)

    top_predictions = load_universality_paradigm_predictions_from_cache(
        data = data,
        models = models,
        random_state = random_state,
        cache_path = universality_cache_path,
        regime_key = regime_key,
        model_to_paradigm = model_to_paradigm,
        paradigm_order = paradigm_order,
    )
    source_results, source_label = load_or_compute_transfer_consensus_results(
        data = data,
        models = models,
        feat_x = feat_x,
        feat_z = feat_z,
        target = target,
        cache_path = transfer_consensus_cache_path,
        n_repeats = n_repeats,
        random_state = random_state,
        n_jobs = n_jobs,
        force_recompute = force_recompute,
        require_cache = require_cache,
    )

    y_true_log = np.asarray(_log_transformer(data[target]).astype(float), dtype = float)
    domains = data["domain"].to_numpy()
    disciplines = data["discipline"].to_numpy()
    domain_order = _domain_order(
        data = data,
        domain_palette = domain_palette,
        domain_column = "domain",
    )
    discipline_domain_map = (
        data[["discipline", "domain"]]
        .drop_duplicates()
        .set_index("discipline")["domain"]
        .to_dict()
    )
    ordered_disciplines, group_ranges = _ordered_disciplines(
        y_true_log = y_true_log,
        domains = domains,
        disciplines = disciplines,
        domain_order = domain_order,
    )
    if not ordered_disciplines:
        raise ValueError("No disciplines available for plotting")

    consensus_summary = _summarize_consensus_panel(
        y_true_log = y_true_log,
        disciplines = disciplines,
        discipline_domain_map = discipline_domain_map,
        ordered_disciplines = ordered_disciplines,
        domain_order = domain_order,
        results_dict_domain = source_results["results_dict_domain"],
        results_data_domain_consensus = source_results["results_data_domain_consensus"],
        domain_palette = domain_palette,
    )
    transfer_summary = _summarize_transfer_panel(
        y_true_log = y_true_log,
        disciplines = disciplines,
        ordered_disciplines = ordered_disciplines,
        results_dict_domain = source_results["results_dict_domain"],
        results_dict_5fold = source_results["results_dict_5fold"],
        results_dict_10fold = source_results["results_dict_10fold"],
        results_data_5fold = source_results["results_data_5fold"],
        results_data_10fold = source_results["results_data_10fold"],
        equivalence_fallback = DELTA_EQ_FALLBACK,
        feasibility_threshold = FEASIBILITY_THRESHOLD,
    )

    transfer_span = float(transfer_ylim[1] - transfer_ylim[0])
    if transfer_span <= 0.0:
        raise ValueError("transfer_ylim must have a positive span")

    consensus_span = float(CONSENSUS_YLIM[1] - CONSENSUS_YLIM[0])
    consensus_height_ratio = BASE_ROW_HEIGHT_RATIO * (consensus_span / TRANSFER_LAYOUT_SPAN)
    base_height_sum = (2.0 * BASE_ROW_HEIGHT_RATIO) + TOP_ROW_HEIGHT_RATIO
    grid_height_sum = TOP_ROW_HEIGHT_RATIO + consensus_height_ratio + BASE_ROW_HEIGHT_RATIO
    adjusted_figsize = (
        figsize[0],
        figsize[1] * (grid_height_sum / base_height_sum),
    )

    fig = plt.figure(figsize = adjusted_figsize, facecolor = BG)
    grid = fig.add_gridspec(
        nrows = 3,
        ncols = len(domain_order),
        height_ratios = [TOP_ROW_HEIGHT_RATIO, consensus_height_ratio, BASE_ROW_HEIGHT_RATIO],
        hspace = 0.34,
        wspace = 0.09,
        left = 0.086,
        right = 0.986,
        top = 0.95,
        bottom = 0.215,
    )
    domain_axes = np.asarray(
        [fig.add_subplot(grid[0, index]) for index in range(len(domain_order))],
        dtype = object,
    )
    ax_consensus = fig.add_subplot(grid[1, :])
    ax_transfer = fig.add_subplot(grid[2, :], sharex = ax_consensus)
    consensus_position = ax_consensus.get_position()
    scaled_consensus_height = consensus_position.height * PANEL_B_HEIGHT_SCALE
    ax_consensus.set_position(
        pos = [
            consensus_position.x0,
            consensus_position.y1 - scaled_consensus_height,
            consensus_position.width,
            scaled_consensus_height,
        ],
    )
    transfer_position = ax_transfer.get_position()
    ax_transfer.set_position(
        pos = [
            transfer_position.x0,
            transfer_position.y0 + PANEL_C_VERTICAL_SHIFT,
            transfer_position.width,
            transfer_position.height,
        ],
    )
    for axis in (ax_consensus, ax_transfer):
        axis_position = axis.get_position()
        axis.set_position(
            pos = [
                axis_position.x0,
                axis_position.y0 + LOWER_PANELS_VERTICAL_SHIFT,
                axis_position.width,
                axis_position.height,
            ],
        )

    plot_universality_domain_columns(
        data = data,
        predictions = top_predictions,
        target = target,
        regime_key = regime_key,
        paradigm_order = paradigm_order,
        domain_palette = domain_palette,
        domain_labels = domain_labels,
        paradigm_markers = DEFAULT_PARADIGM_MARKERS,
        axis_limits = axis_limits,
        diagonal_color = SPINE,
        fig = fig,
        axes = domain_axes,
        add_title = False,
        add_shared_labels = False,
        add_legend = False,
        show = False,
        ei_badge_position = "top",
        ei_badge_background = False,
    )

    for axis in domain_axes:
        axis.set_aspect(aspect = "equal", adjustable = "box", anchor = "W")
        tick_positions = axis.get_yticks()
        tick_labels = [f"{int(value)}" for value in tick_positions]
        for tick_index, tick_value in enumerate(tick_positions):
            if np.isclose(tick_value, axis_limits[0]):
                tick_labels[tick_index] = ""
        axis.set_yticklabels(
            labels = tick_labels,
        )
        axis.spines[["bottom", "left"]].set_color(SPINE)
        axis.spines[["bottom", "left"]].set_linewidth(AXIS_LINE_WIDTH)
        axis.tick_params(
            axis = "both",
            color = SPINE,
            labelcolor = AXIS_LABEL_COLOR,
            labelsize = AXIS_LABEL_SIZE,
        )
        axis.tick_params(axis = "y", labelleft = True)

    _style_lower_axis(axis = ax_consensus)
    ax_consensus.set_xlim(left = -0.65, right = len(ordered_disciplines) - 0.35)
    _draw_consensus_panel(
        axis = ax_consensus,
        transfer_axis = ax_transfer,
        consensus_summary = consensus_summary,
        group_ranges = group_ranges,
        domain_palette = domain_palette,
        domain_labels = lower_domain_labels,
        panel_label = "",
    )
    _draw_transfer_panel(
        axis = ax_transfer,
        transfer_summary = transfer_summary,
        group_ranges = group_ranges,
        ordered_disciplines = ordered_disciplines,
        discipline_domain_map = discipline_domain_map,
        domain_palette = domain_palette,
        panel_label = "",
        transfer_ylim = transfer_ylim,
    )
    ax_consensus.set_ylabel(ylabel = "")
    ax_transfer.set_ylabel(ylabel = "")

    aligned_domain_x_positions = [ax_consensus.get_position().x0]
    for _, start, _ in group_ranges[1:]:
        divider_x = fig.transFigure.inverted().transform(
            ax_consensus.transData.transform((start - 0.5, CONSENSUS_YLIM[0]))
        )[0]
        aligned_domain_x_positions.append(float(divider_x))
    for axis, aligned_x in zip(domain_axes, aligned_domain_x_positions):
        axis_position = axis.get_position()
        scaled_width = axis_position.width * PANEL_A_AXIS_SCALE
        scaled_height = axis_position.height * PANEL_A_AXIS_SCALE
        axis.set_position(
            pos = [
                aligned_x,
                axis_position.y1 - scaled_height,
                scaled_width,
                scaled_height,
            ],
        )

    ## match tick-mark weight to the (0.5) axis line weight
    for _tick_axis in (*domain_axes, ax_consensus, ax_transfer):
        _tick_axis.tick_params(which = "both", width = AXIS_LINE_WIDTH)

    top_row_position_left = domain_axes[0].get_position()
    top_row_position_right = domain_axes[-1].get_position()
    shared_legend_right_x = ax_consensus.get_position().x1
    fig.text(
        x = 0.5 * (top_row_position_left.x0 + top_row_position_right.x1),
        y = top_row_position_left.y0 - 0.03,
            s = "Observed Log-Max",
        ha = "center",
        va = "center",
        fontsize = AXIS_LABEL_SIZE,
        fontweight = "normal",
        color = AXIS_LABEL_COLOR,
    )

    outer_ylabel_x = domain_axes[0].get_position().x0 - 0.045
    panel_label_x = outer_ylabel_x - 0.006
    panel_label_y = {
        "a": top_row_position_left.y1 + 0.009,
        "b": ax_consensus.get_position().y1 - 0.025,
        "c": ax_transfer.get_position().y1 + 0.0,
    }
    for panel_label, panel_y in panel_label_y.items():
        fig.text(
            x = panel_label_x,
            y = panel_y,
            s = panel_label,
            ha = "center",
            va = "bottom",
            color = TEXT,
            fontsize = 8.0,
            fontweight = "bold",
        )
    for axis, label_text in (
        (domain_axes[0], "Estimated Log-Max"),
        (ax_consensus, "Relative Log-Max"),
        (ax_transfer, r"$\Delta$ EI ($\mathit{k}$-Fold - Domain LOGO)"),
    ):
        axis_position = axis.get_position()
        fig.text(
            x = outer_ylabel_x,
            y = 0.5 * (axis_position.y0 + axis_position.y1),
            s = label_text,
            ha = "center",
            va = "center",
            fontsize = AXIS_LABEL_SIZE,
            fontweight = "normal",
            rotation = 90,
            color = AXIS_LABEL_COLOR,
        )

    panel_a_legend = fig.legend(
        handles = _panel_a_legend_handles(),
        loc = "lower right",
        bbox_to_anchor = (
            shared_legend_right_x,
            top_row_position_left.y0 + 0.005,
        ),
        bbox_transform = fig.transFigure,
        ncol = 3,
        fontsize = PANEL_TEXT_SIZE,
        frameon = True,
        facecolor = BG,
        edgecolor = LEGEND_EDGE,
        handlelength = 1.45,
        columnspacing = 0.75,
        handletextpad = 0.40,
        borderpad = LEGEND_BORDERPAD_A,
        borderaxespad = 0.0,
    )
    panel_b_legend = ax_consensus.legend(
        handles = _panel_b_legend_handles(),
        loc = "lower right",
        bbox_to_anchor = (
            shared_legend_right_x,
            ax_consensus.get_position().y0 + 0.035,
        ),
        bbox_transform = fig.transFigure,
        ncol = 2,
        fontsize = 6.2,
        frameon = True,
        facecolor = BG,
        edgecolor = LEGEND_EDGE,
        handlelength = 1.45,
        columnspacing = 0.75,
        handletextpad = 0.40,
        borderpad = LEGEND_BORDERPAD_BC,
        borderaxespad = 0.0,
    )
    panel_c_legend = ax_transfer.legend(
        handles = _panel_c_legend_handles(),
        handler_map = {
            _VerticalIntervalLegendHandle: _VerticalIntervalLegendHandler(),
        },
        loc = "lower right",
        bbox_to_anchor = (
            shared_legend_right_x,
            ax_transfer.get_position().y0 + 0.010,
        ),
        bbox_transform = fig.transFigure,
        ncol = 3,
        fontsize = 6.2,
        frameon = True,
        facecolor = BG,
        edgecolor = LEGEND_EDGE,
        handlelength = 1.45,
        columnspacing = 0.75,
        handletextpad = 0.40,
        borderpad = LEGEND_BORDERPAD_BC,
        borderaxespad = 0.0,
    )
    for legend in (panel_a_legend, panel_b_legend, panel_c_legend):
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor(LEGEND_EDGE)

    _apply_panel_lettering(fig = fig)

    axes = {
        "domain": domain_axes,
        "consensus": ax_consensus,
        "transfer": ax_transfer,
    }
    if show:
        plt.show()
    return fig, axes, source_label


def _render_consensus_figure(


    results_data: pd.DataFrame,


    results_perturbed_consensus: pd.DataFrame,


    results_decomposed_consensus: pd.DataFrame,


    results_falsified_consensus: pd.DataFrame,


    results_original_agreement: pd.DataFrame,


    results_decomposed_full_agreement: pd.DataFrame,


    results_perturbed_full_agreement: pd.DataFrame,


    results_falsified_full_agreement: pd.DataFrame,


    figure_dir: Path,


    export_pdf_scaled: Callable[..., Path],


    n_decimals: int = 2,


    show: bool = False,


    ) -> tuple[Figure, Path]:


    """


    Desc:


        Generate the cache-backed consensus superfigure.





    Args:


        results_data: Original pairwise consensus results.


        results_perturbed_consensus: Perturbed pairwise consensus results.


        results_decomposed_consensus: Decomposed pairwise consensus results.


        results_falsified_consensus: Falsified pairwise consensus results.


        results_original_agreement: Original full-corpus model-truth results.


        results_decomposed_full_agreement: Decomposition full-corpus model-truth results.


        results_perturbed_full_agreement: Perturbation full-corpus model-truth results.


        results_falsified_full_agreement: Falsification full-corpus model-truth results.


        figure_dir: Project figure output directory.


        export_pdf_scaled: Publication PDF export callback.


        n_decimals: Number of displayed decimal places.


        show: Whether to display the Matplotlib figure.





    Returns:


        Figure and project PDF artifact path.


    """


    for source_name, frame in {
        "original agreement": results_original_agreement,
        "ablation consensus": results_decomposed_consensus,
        "ablation agreement": results_decomposed_full_agreement,
        "perturbation agreement": results_perturbed_full_agreement,
        "falsification agreement": results_falsified_full_agreement,
    }.items():
        if "evaluation" not in frame.columns or not frame["evaluation"].eq("full_corpus").all():
            raise ValueError(f"Figure 3 requires full-corpus {source_name} results")

    FIGURE_DIR = figure_dir


    N_DECIMALS = n_decimals
    ## ci-only consensus superfigure -- producer-cache-backed
    ## arial mathtext patch
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.sf": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "it",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    
    out_dir = FIGURE_DIR
    N_DECIMALS = int(N_DECIMALS)
    
    def _ci_matrix(results: pd.DataFrame) -> pd.DataFrame:
        matrix = build_paradigm_consensus_matrices(
            results = results,
            paradigm_order = DEFAULT_PARADIGM_ORDER,
            metric_panels = {"CI": "ci"},
        )["CI"]
        values = matrix.to_numpy(dtype = float)
        if not np.any(np.isfinite(values)):
            raise RuntimeError("No Finite CI Values Were Available For Plotting.")
        return matrix
    
    
    def _stack_display_label(label: str) -> str:
        return "\n".join(word.title() for word in label.split())
    
    
    results_original_ci = results_data.copy()
    results_perturbed_ci = results_perturbed_consensus.copy()
    results_ablated_ci = results_decomposed_consensus.copy()
    results_falsified_ci = results_falsified_consensus.copy()
    perturbed_source = "Perturb Producer Cache"
    ablated_source = "Ablate Producer Cache"
    falsified_source = "Falsify Producer Cache"
    
    ## condition-specific plotting frames
    results_perturbed_max = find_perturbed_max(
        results = results_perturbed_ci,
    )
    results_perturbed_plot = results_perturbed_max.loc[
        results_perturbed_max["perturbation"] != "baseline"
    ] .copy()
    if results_perturbed_plot.empty:
        raise RuntimeError("No Non-Baseline Perturbed Consensus Rows Were Available For Plotting.")
    
    results_ablated_plot = results_ablated_ci.loc[
        results_ablated_ci["specification"] != "additive"
    ].copy()
    if results_ablated_plot.empty:
        raise RuntimeError("No Non-Additive Ablated Consensus Rows Were Available For Plotting.")
    
    results_falsified_plot = results_falsified_ci.copy()
    if "condition" in results_falsified_plot.columns:
        results_falsified_plot = results_falsified_plot.loc[
            results_falsified_plot["condition"] == "falsified"
        ].copy()
    if "Falsification" in results_falsified_plot.columns:
        frozen_rows = results_falsified_plot["Falsification"] == "frozen"
        if frozen_rows.any():
            results_falsified_plot = results_falsified_plot.loc[frozen_rows].copy()
    if results_falsified_plot.empty:
        raise RuntimeError("No Falsified Consensus Rows Were Available For Plotting.")
    
    condition_matrices = {
        "Original": _ci_matrix(results_original_ci),
        "Perturbed": _ci_matrix(results_perturbed_plot),
        "Falsified": _ci_matrix(results_falsified_plot),
        "Ablated": _ci_matrix(results_ablated_plot),
    }
    
    ## ---- scatter inputs: consensus (y) reuses the plotting frames above ----
    CONDITION_ORDER = list(condition_matrices.keys())
    PANEL_NUMBER = {label: index + 1 for index, label in enumerate(CONDITION_ORDER)}
    CONDITION_COLOR = {
        "Original": "#3366CC",
        "Perturbed": "#7B3FB5",
        "Falsified": "#C81C8E",
        "Ablated": "#566573",
    }
    
    consensus_series = {
        "Original": results_original_ci["ci"],
        "Perturbed": results_perturbed_plot["ci"],
        "Falsified": results_falsified_plot["ci"],
        "Ablated": results_ablated_plot["ci"],
    }
    
    
    def _median_value(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors = "coerce").dropna()
        if values.empty:
            raise RuntimeError("No finite CI values were available for a scatter point.")
        return float(values.median())
    
    
    ## ---- scatter inputs: validity (x) from canonical structural-agreement frames ----
    separation = results_decomposed_full_agreement.copy()
    recovery = results_perturbed_full_agreement.copy()
    agreement = results_falsified_full_agreement.copy()
    
    recovery_use = recovery.copy()
    if "track" in recovery_use.columns:
        recovery_use = recovery_use.loc[recovery_use["track"] == "frozen"].copy()
    recovery_validity_max = find_perturbed_max(results = recovery_use)
    
    agreement_use = agreement.copy()
    if "Falsification" in agreement_use.columns:
        agreement_use = agreement_use.loc[agreement_use["Falsification"] == "frozen"].copy()
    elif "track" in agreement_use.columns:
        agreement_use = agreement_use.loc[agreement_use["track"] == "frozen"].copy()
    
    validity_series = {
        "Original": results_original_agreement["ci"],
        "Perturbed": recovery_validity_max.loc[recovery_validity_max["perturbation"] != "baseline", "ci"],
        "Falsified": agreement_use.loc[agreement_use["condition"] == "falsified", "ci"],
        "Ablated": separation.loc[separation["specification"] != "additive", "ci"],
    }
    
    validity_frames = {
        "Original": results_original_agreement,
        "Perturbed": recovery_validity_max.loc[recovery_validity_max["perturbation"] != "baseline"],
        "Falsified": agreement_use.loc[agreement_use["condition"] == "falsified"],
        "Ablated": separation.loc[separation["specification"] != "additive"],
    }
    
    
    def _paradigm_validity_medians(label: str) -> list[float]:
        """Median Model-Truth CI per paradigm (mirrors the per-paradigm consensus matrices)."""
        frame = validity_frames[label]
        ci_values = pd.to_numeric(frame["ci"], errors = "coerce")
        if "model" in frame.columns:
            paradigm = frame["model"].map(DEFAULT_MODEL_TO_PARADIGM)
            medians = ci_values.groupby(paradigm).median().dropna()
            if not medians.empty:
                return medians.tolist()
        return [float(ci_values.median())]
    
    dot_points = {
        label: {
            "x": _median_value(validity_series[label]),
            "y": _median_value(consensus_series[label]),
            "color": CONDITION_COLOR[label],
        }
        for label in CONDITION_ORDER
    }
    
    ## figure scaffold -- double column (183 mm wide; fonts 5-7 pt)
    plt.close("all")
    FONT_FAMILY = "Arial"
    TEXT_COLOR = "#000000"
    SPINE_COLOR = "#000000"
    
    PANEL_LETTER_FONT_SIZE = 8.44  ## ->8.0 pt final after x0.948 downscale
    SCATTER_LABEL_FONT_SIZE = 8.0
    SCATTER_TICK_FONT_SIZE = 7.0
    LEGEND_FONT_SIZE = 7.0
    DOT_NUMBER_FONT_SIZE = 7.0
    HEATMAP_TITLE_FONT_SIZE = 8.0
    HEATMAP_VALUE_FONT_SIZE = 7.5
    PARADIGM_LABEL_FONT_SIZE = 7.0
    COLORBAR_LABEL_FONT_SIZE = 7.0
    COLORBAR_TICK_FONT_SIZE = 7.0
    SPINE_WIDTH = 0.5
    COLORBAR_PAD = 0.05
    COLORBAR_WIDTH = 0.018
    
    paradigm_order = list(DEFAULT_PARADIGM_ORDER)
    display_paradigm_order = [_stack_display_label(label = label) for label in paradigm_order]
    norm = Normalize(vmin = 0.0, vmax = 1.0)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color = "#F7F7F7")
    
    fig = plt.figure(figsize = (7.2047, 3.612), facecolor = "white")
    outer_grid = fig.add_gridspec(
        nrows = 1,
        ncols = 2,
        width_ratios = [0.56, 0.44],
        wspace = 0.22,
        left = 0.07,
        right = 0.91,
        top = 0.92,
        bottom = 0.16,
    )
    ax_scatter = fig.add_subplot(outer_grid[0, 0])
    inner_grid = outer_grid[0, 1].subgridspec(
        nrows = 2,
        ncols = 2,
        hspace = 0.22,
        wspace = 0.05,
    )
    heatmap_positions = ((0, 0), (0, 1), (1, 0), (1, 1))
    axes_array = np.array([
        fig.add_subplot(inner_grid[row_index, column_index])
        for row_index, column_index in heatmap_positions
    ])
    
    ## ---- panel a: square overview scatter (numbered dots) ----
    ax_scatter.set_facecolor("white")
    dot_number_artists = list()
    for label in CONDITION_ORDER:
        point = dot_points[label]
        ax_scatter.scatter(
            point["x"], point["y"],
            s = 118, facecolor = "#000000", edgecolor = "none",
            linewidth = 0.0, zorder = 6,
        )
        dot_number_artists.append(ax_scatter.text(
            point["x"], point["y"], str(PANEL_NUMBER[label]),
            ha = "center", va = "center_baseline", fontsize = DOT_NUMBER_FONT_SIZE,
            fontweight = "bold", color = "white", zorder = 7,
        ))
    
    ## ---- group bounds use the same pairwise-consensus evaluation protocol ----
    GROUP_CONDITIONS = [
        ("Original", "Perturbed"),
        ("Falsified", "Ablated"),
    ]
    GROUP_BOX_PAD = 0.05
    MINMAX_FONT_SIZE = 8.0
    MINMAX_LABEL_INSET = 0.01   # inward offset of Min/Max from the square corners
    MIN_GREEN_LABEL_DROP = 0.03   # extra downward shift of the green (high-CI) group's "Min"
    minmax_label_artists = []
    for _group in GROUP_CONDITIONS:
        ## extremes that the square encapsulates: y from the panel-b matrix cells
        ## (the "table" values), x from the corresponding validity values
        ## box spans the actual min/max of the per-paradigm medians, the same way on
        ## both axes: x = per-paradigm model-truth CI medians; y = per-paradigm-pair
        ## consensus matrix cells
        _xs = []
        for _g in _group:
            _xs.extend(_paradigm_validity_medians(_g))
        _ys = []
        for _g in _group:
            _cells = condition_matrices[_g].to_numpy(dtype = float)
            _ys.extend(_cells[np.isfinite(_cells)].tolist())
        _x_min, _x_max = min(_xs), max(_xs)
        _y_min, _y_max = min(_ys), max(_ys)
        _group_ci = float(np.median(_ys))
        _group_color = cmap(norm(_group_ci))
        _darken = 0.8
        _group_color = (_group_color[0] * _darken, _group_color[1] * _darken, _group_color[2] * _darken, 1.0)
        _group_fill = (_group_color[0], _group_color[1], _group_color[2], 0.20)
        ax_scatter.add_patch(Rectangle(
            (_x_min, _y_min),
            _x_max - _x_min, _y_max - _y_min,
            facecolor = _group_fill, edgecolor = _group_color, linewidth = 0.6,
            linestyle = (0, (4, 2)), zorder = 4,
        ))
        _minmax_lum = 0.2126 * _group_color[0] + 0.7152 * _group_color[1] + 0.0722 * _group_color[2]
        _minmax_text_darken = 0.55 if _minmax_lum > 0.55 else 0.9
        _minmax_text_color = (
            _group_color[0] * _minmax_text_darken,
            _group_color[1] * _minmax_text_darken,
            _group_color[2] * _minmax_text_darken,
            1.0,
        )
        _min_label_drop = MIN_GREEN_LABEL_DROP if _group_ci > 0.65 else 0.0
        minmax_label_artists.append(ax_scatter.text(
            _x_min + MINMAX_LABEL_INSET, _y_min + MINMAX_LABEL_INSET - 0.01, "Min",
            color = _minmax_text_color, ha = "left", va = "bottom",
            fontsize = MINMAX_FONT_SIZE, zorder = 5,
        ))
        minmax_label_artists.append(ax_scatter.text(
            _x_max - MINMAX_LABEL_INSET, _y_max - MINMAX_LABEL_INSET, "Max",
            color = _minmax_text_color, ha = "right", va = "top",
            fontsize = MINMAX_FONT_SIZE, zorder = 5,
        ))
        ax_scatter.scatter(
            [_x_min, _x_max],
            [_y_min, _y_max],
            s = 40, color = _group_color, edgecolor = "none", zorder = 5,
        )
    
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)
    ax_scatter.set_xticks(np.arange(0.0, 1.01, 0.25))
    ax_scatter.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax_scatter.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax_scatter.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax_scatter.set_xlabel(r"Median Model-Truth CI $(\hat{y}, \, y)$", fontsize = SCATTER_LABEL_FONT_SIZE, color = TEXT_COLOR)
    ax_scatter.set_ylabel(r"Median Pairwise CI $(y_i, \, y_j)$", fontsize = SCATTER_LABEL_FONT_SIZE, color = TEXT_COLOR)
    ## reference gridlines only at 0 and 1 (omit intermediate gridlines)
    for _grid_pos in (0.0, 1.0):
        ax_scatter.axhline(_grid_pos, linestyle = (0, (3, 3)), linewidth = 0.6, color = "#BDBDBD", zorder = 0)
        ax_scatter.axvline(_grid_pos, linestyle = (0, (3, 3)), linewidth = 0.6, color = "#BDBDBD", zorder = 0)
    ax_scatter.set_axisbelow(True)
    for spine in ax_scatter.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_WIDTH)
    ax_scatter.tick_params(color = SPINE_COLOR, labelcolor = TEXT_COLOR, labelsize = SCATTER_TICK_FONT_SIZE, length = 2.5, width = 0.6)
    
    
    ## ---- panel b: per-condition paradigm heatmaps ----
    image = None
    y_label_artists = []
    for panel_index, (axis, (condition_label, matrix)) in enumerate(zip(axes_array, condition_matrices.items())):
        values = matrix.to_numpy(dtype = float)
        image = axis.imshow(
            np.ma.masked_invalid(values),
            cmap = cmap,
            norm = norm,
            aspect = "equal",
            interpolation = "nearest",
        )
        axis.set_facecolor("white")
    
        for row_index, row_label in enumerate(paradigm_order):
            for column_index, column_label in enumerate(paradigm_order):
                value = matrix.loc[row_label, column_label]
                cell_label = "NA" if pd.isna(value) else f"{float(value):.{N_DECIMALS}f}"
                axis.text(
                    x = column_index,
                    y = row_index,
                    s = cell_label,
                    ha = "center",
                    va = "center",
                    fontsize = HEATMAP_VALUE_FONT_SIZE,
                    fontweight = "normal",
                    color = TEXT_COLOR,
                )
    
        axis.set_title(
            label = f"{PANEL_NUMBER[condition_label]})  {condition_label}",
            fontsize = HEATMAP_TITLE_FONT_SIZE,
            fontweight = "semibold",
            color = TEXT_COLOR,
            y = 1.03,
            pad = 4,
            loc = "center",
        )
        axis.set_xticks(ticks = [])
        if panel_index in (2, 3):
            for label_index, display_label in enumerate(display_paradigm_order):
                axis.text(
                    x = label_index,
                    y = len(paradigm_order) + 1.18,
                    s = display_label,
                    transform = axis.transData,
                    rotation = 90,
                    rotation_mode = "default",
                    ha = "center",
                    va = "bottom",
                    multialignment = "left",
                    fontsize = PARADIGM_LABEL_FONT_SIZE,
                    color = TEXT_COLOR,
                    clip_on = False,
                )
        axis.set_yticks(ticks = range(len(paradigm_order)))
        axis.set_yticklabels(labels = [])
        if panel_index in (0, 2):
            y_label_transform = blended_transform_factory(axis.transAxes, axis.transData)
            for row_index, display_label in enumerate(display_paradigm_order):
                y_label_artists.append((axis, axis.text(
                    x = -0.5,
                    y = row_index,
                    s = display_label,
                    transform = y_label_transform,
                    ha = "left",
                    va = "center",
                    multialignment = "left",
                    fontsize = PARADIGM_LABEL_FONT_SIZE,
                    color = TEXT_COLOR,
                    clip_on = False,
                )))
        axis.tick_params(axis = "x", which = "major", length = 0, color = SPINE_COLOR, pad = 0)
        axis.tick_params(axis = "y", which = "major", length = 0, color = SPINE_COLOR, pad = 4)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(SPINE_WIDTH)
    
    fig.canvas.draw()
    
    ## ---- y-labels: left-aligned, widest label hugging the left panels ----
    ## manual text artists honour set_x across draws (tick labels do not)
    _renderer = fig.canvas.get_renderer()
    Y_LABEL_GAP_AXES = 0.14
    for _axis in (axes_array[0], axes_array[2]):
        _axis_labels = [art for (ax, art) in y_label_artists if ax is _axis]
        _width_axes = max(
            art.get_window_extent(renderer = _renderer).transformed(_axis.transAxes.inverted()).width
            for art in _axis_labels
        )
        _x_anchor = -(_width_axes + Y_LABEL_GAP_AXES)
        for art in _axis_labels:
            art.set_x(_x_anchor)
    fig.canvas.draw()
    
    ## ---- equalize horizontal spacing: set the gap between the two matrix columns
    ## ---- equal to the gap between the y-labels and the left column ----
    _inv = fig.transFigure.inverted()
    _matrix_w = axes_array[1].get_position().width
    _label_right = max(
        art.get_window_extent(renderer = _renderer).transformed(_inv).x1
        for (ax, art) in y_label_artists
    )
    _left_col_x0 = axes_array[0].get_position().x0
    _panel_gap_frac = max(_left_col_x0 - _label_right, 0.01)
    _extra_col_gap_frac = 0.02
    for panel_index, (_, column_index) in enumerate(heatmap_positions):
        if column_index == 0:
            continue
        position = axes_array[panel_index].get_position()
        column_left = _left_col_x0 + column_index * (_matrix_w + _panel_gap_frac + _extra_col_gap_frac)
        axes_array[panel_index].set_position([column_left, position.y0, _matrix_w, position.height])
    fig.canvas.draw()
    
    ## ---- match both scatter axes to the median-pairwise-CI colorbar ----
    ## the colorbar spans the full 2x2 block height, so the scatter is made a square
    ## of that exact side length: its height is the block height and its width is the
    ## same physical length, centred in the room left of the matrices' label gutter
    fig_width_in, fig_height_in = fig.get_size_inches()
    block_top_frac = axes_array[0].get_position().y1
    block_bottom_frac = axes_array[2].get_position().y0
    block_height_frac = block_top_frac - block_bottom_frac
    colorbar_length_in = block_height_frac * fig_height_in
    LABEL_GUTTER_FRAC = 0.12
    region_left_frac = 0.06
    region_right_frac = axes_array[0].get_position().x0 - LABEL_GUTTER_FRAC
    region_width_frac = max(region_right_frac - region_left_frac, 0.05)
    square_width_frac = min(colorbar_length_in / fig_width_in, region_width_frac)
    scatter_left_frac = min(region_left_frac + 0.03, region_right_frac - square_width_frac)
    ax_scatter.set_position([
        scatter_left_frac,
        block_bottom_frac,
        square_width_frac,
        block_height_frac,
    ])
    fig.canvas.draw()
    
    if image is not None:
        top_right_position = axes_array[1].get_position()
        bottom_right_position = axes_array[3].get_position()
        colorbar_axis = fig.add_axes([
            top_right_position.x1 + COLORBAR_PAD,
            bottom_right_position.y0,
            COLORBAR_WIDTH,
            top_right_position.y1 - bottom_right_position.y0,
        ])
        colorbar_axis.set_facecolor("white")
        colorbar = fig.colorbar(image, cax = colorbar_axis)
        colorbar.set_ticks([0.0, 0.5, 1.0])
        colorbar.set_ticklabels(["0", "0.5", "1"])
        colorbar.set_label(
            label = "Median Pairwise CI",
            fontsize = COLORBAR_LABEL_FONT_SIZE,
            color = TEXT_COLOR,
            labelpad = 5,
        )
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(
            labelsize = COLORBAR_TICK_FONT_SIZE,
            colors = TEXT_COLOR,
            length = 2.0,
            width = SPINE_WIDTH,
        )
        colorbar.outline.set_edgecolor(SPINE_COLOR)
        colorbar.outline.set_linewidth(SPINE_WIDTH)
    
    ## ---- bold panel letters anchored to the rendered (square) axes ----
    fig.canvas.draw()
    scatter_position = ax_scatter.get_position()
    first_heatmap_position = axes_array[0].get_position()
    fig.text(
        scatter_position.x0 - 0.07, scatter_position.y1 + 0.012,
        "a", fontsize = PANEL_LETTER_FONT_SIZE, fontweight = "bold",
        color = TEXT_COLOR, ha = "left", va = "bottom",
    )
    _b_label_left = min(
        art.get_window_extent(renderer = _renderer).transformed(_inv).x0
        for (ax, art) in y_label_artists
    )
    fig.text(
        _b_label_left - 0.006, first_heatmap_position.y1 + 0.012,
        "b", fontsize = PANEL_LETTER_FONT_SIZE, fontweight = "bold",
        color = TEXT_COLOR, ha = "left", va = "bottom",
    )
    
    ## ---- arial 8pt everywhere (titles/legend, panel letters, ticks, colorbar, labels) ----
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(FONT_FAMILY)
        if text_artist.get_text() in {"a", "b"}:
            text_artist.set_fontsize(PANEL_LETTER_FONT_SIZE)
        else:
            text_artist.set_fontsize(7.38)  ## ->7.0 pt final after x0.948 downscale
    
    fig.canvas.draw()
    
    ## save -- publication standards
    ## vector PDF is the submission file: editable Type-42 (TrueType) embedded fonts,
    ## rasterised heatmaps at >=600 dpi, RGB, white background, tight bbox.
    plt.rcParams["pdf.fonttype"] = 42   # embed editable fonts (not Type 3 / not outlined)
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    
    pdf_path = export_pdf_scaled(fig, 3, target_width_mm = 183.0)
    if show:
        plt.show()
    
    return fig, pdf_path


def _render_stress_test_figure(


    results_perturbed_transfer: pd.DataFrame,


    results_perturbed_recovery: pd.DataFrame,


    results_falsified_transfer: pd.DataFrame,


    results_falsified_agreement: pd.DataFrame,


    results_decomposed_separation: pd.DataFrame,


    figure_dir: Path,


    export_pdf_scaled: Callable[..., Path],


    n_decimals: int = 2,


    show: bool = False,


    ) -> tuple[Figure, Path]:


    """


    Desc:


        Generate the cache-backed stress-test superfigure.





    Args:


        results_perturbed_transfer: Perturbation transfer results.


        results_perturbed_recovery: Perturbation recovery results.


        results_falsified_transfer: Falsification transfer results.


        results_falsified_agreement: Falsification agreement results.


        results_decomposed_separation: Decomposition separation results.


        figure_dir: Project figure output directory.


        export_pdf_scaled: Publication PDF export callback.


        n_decimals: Number of displayed decimal places.


        show: Whether to display the Matplotlib figure.





    Returns:


        Figure and project PDF artifact path.


    """


    FIGURE_DIR = figure_dir


    N_DECIMALS = n_decimals
    ## n3s stress-test data preparation -- producer-cache-backed
    N_DECIMALS_WORK = int(N_DECIMALS)
    
    results_pert_transfer = results_perturbed_transfer.copy()
    results_pert_recovery = results_perturbed_recovery.copy()
    results_fals_transfer = results_falsified_transfer.copy()
    results_fals_agreement = results_falsified_agreement.copy()
    results_decomp = results_decomposed_separation.copy()
    
    results_pert_transfer_max = find_perturbed_max(
        results = results_pert_transfer,
    )
    results_pert_recovery_max = find_perturbed_max(
        results = results_pert_recovery,
    )
    delta_pert_ei = _spec_marginal_delta(
        results = results_pert_transfer,
        feat_value = ["ei"],
        track = "frozen",
        label_ref = "perturbation",
        method = "iqr",
        scale = 1.0,
        decimals = N_DECIMALS_WORK,
    )
    delta_pert_ci = _spec_marginal_delta(
        results = results_pert_recovery,
        feat_value = ["ci"],
        label_pert = "perturbation",
        track = "frozen",
        method = "iqr",
        scale = 1.0,
        decimals = N_DECIMALS_WORK,
    )
    delta_decomp_ei = _spec_marginal_delta(
        results = results_decomp,
        feat_value = ["ei"],
        label_ref = "specification",
        value_ref = "additive",
        method = "iqr",
        scale = 1.0,
        decimals = N_DECIMALS_WORK,
    )
    delta_decomp_ci = _spec_marginal_delta(
        results = results_decomp,
        feat_value = ["ci"],
        label_ref = "specification",
        value_ref = "additive",
        method = "iqr",
        scale = 1.0,
        decimals = N_DECIMALS_WORK,
    )
    
    ## n3s stress-test superfigure render -- publication print layout
    out_dir = FIGURE_DIR
    
    ## print geometry and typography
    PUBLICATION_FONT_FAMILY = "Arial"
    PUBLICATION_FONT_SIZE = 8.0
    PAGE_SIZE_INCHES = (8.5, 9.0)
    DOUBLE_COLUMN_WIDTH_INCHES = 183.0 / 25.4
    MAX_DEPTH_INCHES = 205.0 / 25.4
    ART_WIDTH_INCHES = min(DOUBLE_COLUMN_WIDTH_INCHES, PAGE_SIZE_INCHES[0] - 0.50)
    ART_HEIGHT_INCHES = min(MAX_DEPTH_INCHES, PAGE_SIZE_INCHES[1] - 0.50)
    ART_LEFT = (PAGE_SIZE_INCHES[0] - ART_WIDTH_INCHES) / (2.0 * PAGE_SIZE_INCHES[0])
    ART_RIGHT = 1.0 - ART_LEFT
    ART_BOTTOM = (PAGE_SIZE_INCHES[1] - ART_HEIGHT_INCHES) / (2.0 * PAGE_SIZE_INCHES[1])
    ART_TOP = 1.0 - ART_BOTTOM
    
    mpl.rcParams.update({
        "font.family": PUBLICATION_FONT_FAMILY,
        "font.sans-serif": [PUBLICATION_FONT_FAMILY],
        "font.size": PUBLICATION_FONT_SIZE,
        "axes.labelsize": PUBLICATION_FONT_SIZE,
        "axes.titlesize": PUBLICATION_FONT_SIZE,
        "xtick.labelsize": PUBLICATION_FONT_SIZE,
        "ytick.labelsize": PUBLICATION_FONT_SIZE,
        "legend.fontsize": PUBLICATION_FONT_SIZE,
        "figure.titlesize": PUBLICATION_FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": PUBLICATION_FONT_FAMILY,
        "mathtext.sf": PUBLICATION_FONT_FAMILY,
        "mathtext.it": f"{PUBLICATION_FONT_FAMILY}:italic",
        "mathtext.bf": f"{PUBLICATION_FONT_FAMILY}:bold",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    
    TEXT_COLOR = "#000000"
    SPINE_COLOR = "#8A8A8A"
    DELTA_LABEL_COLOR = SPINE_COLOR
    GRID_COLOR = "#E6E6E6"
    ZERO_COLOR = "#6E6E6E"
    DASHED_GUIDE_COLOR = SPINE_COLOR
    DASHED_GUIDE_LINEWIDTH = 0.6
    PLOT_LIMIT = 0.50
    DOT_MARKERSIZE = 3.0
    DOT_MARKER_AREA = DOT_MARKERSIZE ** 3
    RESIDUAL_DOT_MARKERSIZE = 2.5
    RESIDUAL_DOT_MARKER_AREA = RESIDUAL_DOT_MARKERSIZE ** 3
    FINGERPRINT_TICKS = [-0.50, -0.25, 0.00, 0.25, 0.50]
    SWEEP_TICKS = [-0.25, 0.00, 0.25]
    SWEEP_X_TICKS = [0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0]
    SWEEP_X_LABELS = ["0", "", "", "0.5", "", "", "1"]
    METHOD_LINESTYLES = tuple(DEFAULT_METHOD_LINESTYLES)
    METHOD_SHADE_FACTORS = tuple(DEFAULT_METHOD_SHADE_FACTORS)
    
    
    def _format_label(value: object) -> str:
        return str(value).replace("_", " ").replace("-", " ").title()
    
    
    def _relative_luminance(color: str) -> float:
        red, green, blue = mpl.colors.to_rgb(color)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue
    
    
    def _apply_publication_text(figure: plt.Figure) -> None:
        title_artists = set()
        for axis in figure.axes:
            for attr_name in ("title", "_left_title", "_right_title"):
                title_artist = getattr(axis, attr_name, None)
                if title_artist is not None:
                    title_artists.add(title_artist)
        for text_artist in figure.findobj(match = Text):
            text_artist.set_fontfamily(PUBLICATION_FONT_FAMILY)
            text_artist.set_fontstyle("normal")
            if text_artist.get_text() in {"a", "b", "c", "d"}:
                text_artist.set_fontsize(8.84)  ## ->8.0 pt final after x0.905 downscale
                text_artist.set_fontweight("bold")
                continue
            text_artist.set_fontsize(7.74)  ## ->7.0 pt final after x0.905 downscale
            if text_artist in title_artists:
                text_artist.set_fontweight("bold")
            elif text_artist.get_gid() == PANEL_DELTA_ANNOTATION_GID:
                text_artist.set_fontweight("bold")
            elif text_artist.get_text() in {"+δ", "-δ"}:
                text_artist.set_fontstyle("italic")
                text_artist.set_color(DELTA_LABEL_COLOR)
            elif text_artist.get_fontweight() == "bold":
                text_artist.set_fontweight("normal")
    
    
    def _format_zero_tick(value: float, _position: int | None = None) -> str:
        if np.isclose(value, 0.0):
            return "0"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    
    
    PANEL_DELTA_ANNOTATION_GID = "panel_delta_median_annotation"
    
    
    def _format_panel_delta(value: float) -> str:
        if not np.isfinite(value):
            return "NA"
        rounded_value = float(np.round(value, decimals = 2))
        if np.isclose(rounded_value, 0.0):
            return "0.00"
        return f"{rounded_value:+.2f}"
    
    
    def _annotate_panel_delta_median(axis, panel: pd.DataFrame) -> None:
        for text_artist in list(axis.texts):
            if text_artist.get_gid() == PANEL_DELTA_ANNOTATION_GID:
                text_artist.remove()
    
        required_columns = {"delta_ei", "delta_ci"}
        if panel.empty or not required_columns.issubset(panel.columns):
            return
    
        delta_ei_values = pd.to_numeric(panel["delta_ei"], errors = "coerce").to_numpy(dtype = float)
        delta_ci_values = pd.to_numeric(panel["delta_ci"], errors = "coerce").to_numpy(dtype = float)
        median_delta_ei = float(np.nanmedian(delta_ei_values))
        median_delta_ci = float(np.nanmedian(delta_ci_values))
    
        annotation = axis.text(
            x = 1.035,
            y = 0.035,
            s = (
                rf"$\mathbf{{\Delta}}$ EI = {_format_panel_delta(value = median_delta_ei)}"
                "\n"
                rf"$\mathbf{{\Delta}}$ CI = {_format_panel_delta(value = median_delta_ci)}"
            ),
            transform = axis.transAxes,
            ha = "right",
            va = "bottom",
            multialignment = "left",
            color = TEXT_COLOR,
            fontsize = PUBLICATION_FONT_SIZE,
            fontweight = "bold",
            linespacing = 1.15,
            bbox = dict(
                facecolor = "white",
                edgecolor = "none",
                alpha = 0.72,
                pad = 0,
            ),
            zorder = 20,
        )
        annotation.set_gid(PANEL_DELTA_ANNOTATION_GID)
    
    
    def _format_fingerprint_y_tick(value: float, _position: int | None = None) -> str:
        if np.isclose(np.abs(value), 0.25):
            return ""
        if np.isclose(value, FINGERPRINT_TICKS[0]):
            return ""
        return _format_zero_tick(value = value, _position = _position)
    
    
    def _format_fingerprint_x_tick(value: float, _position: int | None = None) -> str:
        if np.isclose(np.abs(value), 0.25):
            return ""
        return _format_zero_tick(value = value, _position = _position)
    
    
    def _style_delta_axis(
        axis,
        show_left: bool = True,
        show_bottom: bool = True,
        format_x_ticks: bool = False,
        hide_lowest_y_tick_label: bool = False,
    ) -> None:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["left"].set_visible(show_left)
        axis.spines["bottom"].set_visible(show_bottom)
        for spine in axis.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.5)
        axis.tick_params(
            axis = "both",
            labelsize = PUBLICATION_FONT_SIZE,
            length = 2.5,
            width = 0.5,
            color = "#000000",
            labelcolor = TEXT_COLOR,
            pad = 1.5,
        )
        y_formatter = _format_fingerprint_y_tick if hide_lowest_y_tick_label else _format_zero_tick
        axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_formatter))
        if format_x_ticks:
            axis.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_format_fingerprint_x_tick))
        axis.grid(visible = False)
        axis.set_axisbelow(True)
    
    
    def _label_delta_y_lines(axis, delta_value: float, include_negative: bool = True) -> None:
        label_specs = [(float(delta_value), "+δ")]
        if include_negative:
            label_specs.append((-float(delta_value), "-δ"))
        for y_value, label in label_specs:
            axis.text(
                x = -0.035,
                y = y_value,
                s = label,
                transform = axis.get_yaxis_transform(),
                ha = "right",
                va = "center",
                color = DELTA_LABEL_COLOR,
                fontstyle = "italic",
                clip_on = False,
                zorder = 6,
            )
    
    
    def _label_delta_x_lines(axis, delta_value: float, include_negative: bool = True) -> None:
        label_specs = [(float(delta_value), "+δ")]
        if include_negative:
            label_specs.append((-float(delta_value), "-δ"))
        pad_points = axis.xaxis.get_major_ticks()[0].get_pad() if axis.xaxis.get_major_ticks() else 1.5
        label_transform, label_valign, _ = axis.get_xaxis_text1_transform(pad_points = pad_points)
        for x_value, label in label_specs:
            axis.text(
                x = x_value,
                y = 0.0,
                s = label,
                transform = label_transform,
                ha = "center",
                va = label_valign,
                color = DELTA_LABEL_COLOR,
                fontstyle = "italic",
                clip_on = False,
                zorder = 6,
            )
    
    
    def _draw_center_equivalence(axis, delta_x: float, delta_y: float) -> None:
        axis.add_patch(
            Rectangle(
                xy = (-float(delta_x), -float(delta_y)),
                width = float(2.0 * delta_x),
                height = float(2.0 * delta_y),
                facecolor = "#DDEFE3",
                edgecolor = DASHED_GUIDE_COLOR,
                linewidth = DASHED_GUIDE_LINEWIDTH,
                linestyle = "--",
                alpha = 1,
                zorder = 1,
            )
        )
        axis.axhline(y = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        axis.axvline(x = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        axis.axhline(y = float(delta_y), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        axis.axhline(y = -float(delta_y), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        _label_delta_y_lines(axis = axis, delta_value = float(delta_y))
        axis.axvline(x = float(delta_x), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        axis.axvline(x = -float(delta_x), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        _label_delta_x_lines(axis = axis, delta_value = float(delta_x))
    
    
    def _condition_delta_frame(
        results: pd.DataFrame,
        value_col: str,
        delta_col: str,
        pair_columns: list[str],
        condition_col: str = "condition",
        condition_original: str = "original",
        condition_altered: str = "falsified",
        ) -> pd.DataFrame:
    
        required_columns = set(pair_columns + [condition_col, value_col])
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            raise ValueError(f"Missing required condition columns: {missing_columns}")
    
        paired = (
            results.loc[:, pair_columns + [condition_col, value_col]]
            .pivot_table(
                index = pair_columns,
                columns = condition_col,
                values = value_col,
                aggfunc = "median",
                observed = True,
            )
        )
        for condition in (condition_original, condition_altered):
            if condition not in paired.columns:
                raise ValueError(f"Missing condition '{condition}' in {value_col} results")
        paired = paired.dropna(subset = [condition_original, condition_altered]).reset_index()
        paired[delta_col] = paired[condition_altered] - paired[condition_original]
        return paired.loc[:, pair_columns + [delta_col]].copy()
    
    
    def _decomposition_delta_frame(results: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"model", "group", "specification", "ei", "ci"}
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            raise ValueError(f"Missing required decomposition columns: {missing_columns}")
    
        additive = (
            results.loc[results["specification"] == "additive", ["model", "group", "ei", "ci"]]
            .rename(columns = {"ei": "ei_additive", "ci": "ci_additive"})
        )
        alternatives = results.loc[results["specification"] != "additive"].copy()
        paired = alternatives.merge(
            right = additive,
            on = ["model", "group"],
            how = "inner",
        )
        paired["delta_ei"] = paired["ei"] - paired["ei_additive"]
        paired["delta_ci"] = paired["ci"] - paired["ci_additive"]
        return paired.dropna(subset = ["delta_ei", "delta_ci"]).copy()
    
    
    def _draw_point_fingerprint(
        axis,
        panel: pd.DataFrame,
        category_col: str,
        category_order: list[str],
        colors: dict[str, str],
        labels: dict[str, str] | None = None,
        legend_loc: str = "upper left",
        ) -> None:
    
        labels = {} if labels is None else labels
        draw_order = sorted(
            category_order,
            key = lambda category: _relative_luminance(colors.get(category, "#555555")),
            reverse = True,
        )
        handles = []
        for category_index, category in enumerate(draw_order):
            category_panel = panel.loc[panel[category_col] == category].copy()
            if category_panel.empty:
                continue
            color = colors.get(category, "#555555")
            x_values = category_panel["delta_ei"].to_numpy(dtype = float)
            y_values = category_panel["delta_ci"].to_numpy(dtype = float)
            median_x = float(np.nanmedian(x_values))
            median_y = float(np.nanmedian(y_values))
            axis.scatter(
                x = x_values,
                y = y_values,
                s = RESIDUAL_DOT_MARKER_AREA,
                color = color,
                alpha = 0.14,
                edgecolors = "none",
                zorder = 3 + category_index,
            )
            axis.plot(
                [0.0, median_x],
                [0.0, median_y],
                color = color,
                lw = 0.8,
                alpha = 0.90,
                zorder = 5 + category_index,
            )
            axis.scatter(
                [median_x],
                [median_y],
                s = DOT_MARKER_AREA,
                color = color,
                edgecolors = "none",
                zorder = 8 + category_index,
            )
        for category in category_order:
            category_panel = panel.loc[panel[category_col] == category].copy()
            if category_panel.empty:
                continue
            color = colors.get(category, "#555555")
            legend_label = labels.get(category, _format_label(category))
            handles.append(
                Line2D(
                    [0.0],
                    [0.0],
                    marker = "o",
                    linestyle = "None",
                    markersize = DOT_MARKERSIZE,
                    markerfacecolor = color,
                    markeredgecolor = color,
                    label = legend_label,
                )
            )
        if handles:
            legend = axis.legend(
                handles = handles,
                loc = legend_loc,
                bbox_to_anchor = (0.03, 1.0),
                bbox_transform = axis.transAxes,
                frameon = True,
                framealpha = 0.92,
                edgecolor = "#D0D0D0",
                borderaxespad = 0.0,
                borderpad = 0.20,
                labelspacing = 0.18,
                handletextpad = 0.35,
                handlelength = 0.8,
            )
            legend.get_frame().set_linewidth(0.5)
    
    
    ## perturbation summaries
    unit_name = _resolve_plot_unit_column(
        results_transfer = results_pert_transfer,
        results_recovery = results_pert_recovery,
        unit_col = None,
    )
    transfer_paired = _paired_plot_deltas(
        results = results_pert_transfer,
        value_col = "ei",
        delta_col = "delta_ei",
        track = "frozen",
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_paired = _paired_plot_deltas(
        results = results_pert_recovery,
        value_col = "ci",
        delta_col = "delta_ci",
        track = "frozen",
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    transfer_aggregate, transfer_interval = _aggregate_plot_sweep(
        paired = transfer_paired,
        delta_col = "delta_ei",
        label_pert = "perturbation",
    )
    recovery_aggregate, recovery_interval = _aggregate_plot_sweep(
        paired = recovery_paired,
        delta_col = "delta_ci",
        label_pert = "perturbation",
    )
    transfer_max_paired = _paired_plot_deltas(
        results = results_pert_transfer_max,
        value_col = "ei",
        delta_col = "delta_ei",
        track = "frozen",
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    recovery_max_paired = _paired_plot_deltas(
        results = results_pert_recovery_max,
        value_col = "ci",
        delta_col = "delta_ci",
        track = "frozen",
        label_pert = "perturbation",
        label_base = "baseline",
        unit_col = unit_name,
    )
    fingerprint_columns = ["perturbation", "method", "model", unit_name]
    perturbation_fingerprint = (
        transfer_max_paired[fingerprint_columns + ["delta_ei"]]
        .groupby(by = fingerprint_columns, observed = True)["delta_ei"]
        .median()
        .reset_index()
        .merge(
            right = (
                recovery_max_paired[fingerprint_columns + ["delta_ci"]]
                .groupby(by = fingerprint_columns, observed = True)["delta_ci"]
                .median()
                .reset_index()
            ),
            on = fingerprint_columns,
            how = "inner",
        )
    )
    available_perturbations = set(transfer_aggregate["perturbation"].unique())
    available_perturbations |= set(recovery_aggregate["perturbation"].unique())
    available_perturbations |= set(perturbation_fingerprint["perturbation"].unique())
    perturbation_order = [
        name
        for name in DEFAULT_PERTURBATION_ORDER
        if name in available_perturbations
    ]
    perturbation_order.extend(sorted(available_perturbations - set(perturbation_order)))
    if not perturbation_order:
        raise RuntimeError("No perturbation rows are available for plotting.")
    perturbation_palette = dict(DEFAULT_PERTURBATION_PALETTE)
    perturbation_title_colors = dict(DEFAULT_PERTURBATION_TITLE_COLORS)
    
    ## falsification summaries
    fals_method_col = "Method" if "Method" in results_fals_transfer.columns else "method"
    fals_track_col = "Falsification" if "Falsification" in results_fals_transfer.columns else "track"
    fals_unit_col = "group" if "group" in results_fals_transfer.columns else "domain"
    fals_pair_columns = [fals_track_col, fals_method_col, "model", fals_unit_col]
    fals_transfer_delta = _condition_delta_frame(
        results = results_fals_transfer,
        value_col = "ei",
        delta_col = "delta_ei",
        pair_columns = fals_pair_columns,
    )
    fals_agreement_delta = _condition_delta_frame(
        results = results_fals_agreement,
        value_col = "ci",
        delta_col = "delta_ci",
        pair_columns = fals_pair_columns,
    )
    falsification_fingerprint = fals_transfer_delta.merge(
        right = fals_agreement_delta,
        on = fals_pair_columns,
        how = "inner",
    )
    fals_track_order = [
        track
        for track in ("frozen", "retrain")
        if track in set(falsification_fingerprint[fals_track_col])
    ]
    fals_track_order.extend(
        sorted(
            track for track in falsification_fingerprint[fals_track_col].dropna().unique()
            if track not in set(fals_track_order)
        )
    )
    fals_method_order = [
        method
        for method in ("target remap", "random generate", "vector generate")
        if method in set(falsification_fingerprint[fals_method_col])
    ]
    fals_method_order.extend(
        sorted(
            method for method in falsification_fingerprint[fals_method_col].dropna().unique()
            if method not in set(fals_method_order)
        )
    )
    fals_red_palette = ["#5C1717", "#8F2D2D", "#C43A3A"]
    fals_method_colors = {
        method: fals_red_palette[index % len(fals_red_palette)]
        for index, method in enumerate(fals_method_order)
    }
    
    ## decomposition summaries
    decomposition_fingerprint = _decomposition_delta_frame(results = results_decomp)
    decomposition_family_specs = {
        "Simple": ["invariants", "signatures"],
        "Complex": ["joint", "interaction"],
    }
    decomposition_spec_labels = {
        "interaction": "Interaction Terms",
        "joint": "Joint Features",
    }
    decomposition_base_color = "#C46A1C"
    decomposition_specs = [ 
        spec
        for specs in decomposition_family_specs.values()
        for spec in specs
        if spec in set(decomposition_fingerprint["specification"])
    ]
    decomposition_colors = {
        spec: _shade_plot_color(
            hex_color = decomposition_base_color,
            factor = METHOD_SHADE_FACTORS[index % len(METHOD_SHADE_FACTORS)],
        )
        for index, spec in enumerate(decomposition_specs)
    }
    
    ## figure scaffold -- uniform row heights matching panel a geometry
    plt.close("all")
    fig = plt.figure(figsize = PAGE_SIZE_INCHES, facecolor = "white")
    grid = fig.add_gridspec(
        nrows = 4,
        ncols = 4,
        height_ratios = [1.0, 1.0, 1.0, 1.0],
        hspace = 0.48,
        wspace = 0.32,
        left = ART_LEFT,
        right = ART_RIGHT,
        bottom = ART_BOTTOM,
        top = ART_TOP,
    )
    ## no sharey across rows 1-2 so each panel renders its own y-tick labels
    perturbation_axes = np.empty(shape = (3, len(perturbation_order)), dtype = object)
    for column_index in range(len(perturbation_order)):
        perturbation_axes[0, column_index] = fig.add_subplot(grid[0, column_index])
        perturbation_axes[1, column_index] = fig.add_subplot(grid[1, column_index])
        perturbation_axes[2, column_index] = fig.add_subplot(grid[2, column_index])
    
    bottom_axes = [fig.add_subplot(grid[3, column_index]) for column_index in range(4)]
    
    SECOND_ROW_VERTICAL_SHIFT = 0.07
    for axis in perturbation_axes[1, :]:
        axis_position = axis.get_position()
        axis.set_position([
            axis_position.x0,
            axis_position.y0 + SECOND_ROW_VERTICAL_SHIFT,
            axis_position.width,
            axis_position.height,
        ])
    
    PANEL_A_HEIGHT_SCALE = 0.85
    for axes_to_shrink in (
        perturbation_axes[0, :],
        perturbation_axes[1, :],
    ):
        for axis in axes_to_shrink:
            axis_position = axis.get_position()
            new_height = axis_position.height * PANEL_A_HEIGHT_SCALE
            axis.set_position([
                axis_position.x0,
                axis_position.y1 - new_height,
                axis_position.width,
                new_height,
            ])
    
    LOWER_ROWS_VERTICAL_SHIFT = 0.10
    for axes_to_shift in (
        perturbation_axes[2, :],
        bottom_axes,
    ):
        for axis in axes_to_shift:
            axis_position = axis.get_position()
            axis.set_position([
                axis_position.x0,
                axis_position.y0 + LOWER_ROWS_VERTICAL_SHIFT,
                axis_position.width,
                axis_position.height,
            ])
    
    PANEL_C_VERTICAL_SHIFT = 0.01
    for axis in bottom_axes:
        axis_position = axis.get_position()
        axis.set_position([
            axis_position.x0,
            axis_position.y0 + PANEL_C_VERTICAL_SHIFT,
            axis_position.width,
            axis_position.height,
        ])
    
    ## perturbation panels
    PANEL_A_EQUIVALENT_Y_OFFSET = 0.018
    PANEL_A_EQUIVALENT_Y_EI = -float(delta_pert_ei) + PANEL_A_EQUIVALENT_Y_OFFSET
    PANEL_A_EQUIVALENT_Y_CI = -float(delta_pert_ci) + PANEL_A_EQUIVALENT_Y_OFFSET
    row_specs = [
        {
            "label": r"$\Delta$ EI (Perturbed $-$ Original)",
            "margin": float(delta_pert_ei),
            "aggregate": transfer_aggregate,
            "interval": transfer_interval,
            "equivalent_y": PANEL_A_EQUIVALENT_Y_EI,
        },
        {
            "label": r"$\Delta$ CI (Perturbed $-$ Original)",
            "margin": float(delta_pert_ci),
            "aggregate": recovery_aggregate,
            "interval": recovery_interval,
            "equivalent_y": PANEL_A_EQUIVALENT_Y_CI,
        },
    ]
    for row_index, row_spec in enumerate(row_specs):
        for column_index, perturbation in enumerate(perturbation_order):
            axis = perturbation_axes[row_index, column_index]
            color = perturbation_palette.get(perturbation, "#2C6E91")
            aggregate_panel = row_spec["aggregate"].loc[
                row_spec["aggregate"]["perturbation"] == perturbation
            ].copy()
            interval_panel = row_spec["interval"].loc[
                row_spec["interval"]["perturbation"] == perturbation
            ].copy()
            axis.add_patch(
                Rectangle(
                    xy = (0.0, -float(row_spec["margin"])),
                    width = 1.0,
                    height = float(2.0 * row_spec["margin"]),
                    facecolor = "#DDEFE3",
                    edgecolor = "none",
                    alpha = 0.75,
                    zorder = 1,
                )
            )
            axis.axhline(y = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
            axis.plot(
                [0.0, 1.0],
                [float(row_spec["margin"]), float(row_spec["margin"])],
                color = DASHED_GUIDE_COLOR,
                lw = DASHED_GUIDE_LINEWIDTH,
                ls = "--",
                zorder = 2,
            )
            axis.plot(
                [0.0, 1.0],
                [-float(row_spec["margin"]), -float(row_spec["margin"])],
                color = DASHED_GUIDE_COLOR,
                lw = DASHED_GUIDE_LINEWIDTH,
                ls = "--",
                zorder = 2,
            )
            _label_delta_y_lines(axis = axis, delta_value = float(row_spec["margin"]))
            axis.axvline(
                x = 1.0,
                color = "#000000",
                lw = 0.8,
                zorder = 5,
                clip_on = False,
            )
            axis.text(
                x = 0.03,
                y = float(row_spec["equivalent_y"]),
                s = "Equivalent",
                ha = "left",
                va = "bottom",
                color = "#1F5E2E",
                zorder = 4,
            )
            method_plot_specs = []
            for method_index, method in enumerate(sorted(aggregate_panel["method"].dropna().unique())):
                shade = _shade_plot_color(
                    hex_color = color,
                    factor = METHOD_SHADE_FACTORS[method_index % len(METHOD_SHADE_FACTORS)],
                )
                line_style = METHOD_LINESTYLES[method_index % len(METHOD_LINESTYLES)]
                method_plot_specs.append((method, shade, line_style))
            method_draw_order = sorted(
                method_plot_specs,
                key = lambda method_spec: _relative_luminance(method_spec[1]),
                reverse = True,
            )
            for layer_index, (method, shade, line_style) in enumerate(method_draw_order):
                method_aggregate = aggregate_panel.loc[aggregate_panel["method"] == method].copy()
                method_interval = interval_panel.loc[interval_panel["method"] == method].copy()
                if method_aggregate.empty:
                    continue
                intensities = _sorted_plot_levels(values = method_aggregate["intensity"])
                intensity_lookup = _normalized_plot_positions(levels = intensities)
                method_aggregate["_x"] = method_aggregate["intensity"].map(intensity_lookup)
                method_interval["_x"] = method_interval["intensity"].map(intensity_lookup)
                method_aggregate = method_aggregate.sort_values(by = "_x")
                method_interval = method_interval.sort_values(by = "_x")
                if len(method_interval) > 1:
                    axis.fill_between(
                        method_interval["_x"].to_numpy(dtype = float),
                        method_interval["q1"].to_numpy(dtype = float),
                        method_interval["q3"].to_numpy(dtype = float),
                        color = shade,
                        alpha = 0.16,
                        lw = 0.0,
                        zorder = 2,
                    )
                axis.plot(
                    method_aggregate["_x"].to_numpy(dtype = float),
                    method_aggregate["delta"].to_numpy(dtype = float),
                    color = shade,
                    lw = 0.9,
                    ls = line_style,
                    marker = "o",
                    markersize = DOT_MARKERSIZE,
                    markerfacecolor = shade,
                    markeredgewidth = 0.0,
                    zorder = 4 + layer_index,
                    label = _format_label(method),
                )
            axis.set_xlim(left = 0.0, right = 1.0)
            axis.set_ylim(bottom = -0.25, top = 0.25)
            axis.set_yticks(ticks = SWEEP_TICKS)
            axis.set_xticks(ticks = SWEEP_X_TICKS)
            axis.set_xticklabels(labels = SWEEP_X_LABELS)
            if row_index == 0:
                axis.set_title(
                    label = _format_label(perturbation),
                    color = perturbation_title_colors.get(perturbation, color),
                    fontweight = "bold",
                    pad = 1.2,
                )
            else:
                axis.annotate(
                    text = "Max",
                    xy = (1.0, 0.0),
                    xycoords = ("data", "axes fraction"),
                    xytext = (0.0, -15.0),
                    textcoords = "offset points",
                    ha = "center",
                    va = "top",
                    color = TEXT_COLOR,
                    annotation_clip = False,
                )
            if column_index == 0:
                axis.set_ylabel(row_spec["label"], labelpad = 3.0)
            if row_index == 1:
                method_legend_order = sorted(
                    method_plot_specs,
                    key = lambda method_spec: _relative_luminance(method_spec[1]),
                )
                method_legend_handles = [
                    Line2D(
                        [0.0],
                        [0.0],
                        color = shade,
                        lw = 0.9,
                        ls = line_style,
                        marker = "o",
                        markersize = DOT_MARKERSIZE,
                        markerfacecolor = shade,
                        markeredgewidth = 0.0,
                        label = f"{_format_label(method)} [IQR]",
                    )
                    for method, shade, line_style in method_legend_order
                ]
                legend = axis.legend(
                    handles = method_legend_handles,
                    loc = "upper left",
                    bbox_to_anchor = (0.03, 1.0),
                    bbox_transform = axis.transAxes,
                    frameon = True,
                    framealpha = 0.90,
                    edgecolor = "#D0D0D0",
                    borderaxespad = 0.0,
                    borderpad = 0.16,
                    labelspacing = 0.12,
                    handlelength = 1.7,
                    handletextpad = 0.30,
                )
                legend.get_frame().set_linewidth(0.5)
            _style_delta_axis(axis = axis, show_left = True, show_bottom = True)
    
    for column_index, perturbation in enumerate(perturbation_order):
        axis = perturbation_axes[2, column_index]
        color = perturbation_palette.get(perturbation, "#2C6E91")
        panel = perturbation_fingerprint.loc[
            perturbation_fingerprint["perturbation"] == perturbation
        ].copy()
        _draw_center_equivalence(
            axis = axis,
            delta_x = float(delta_pert_ei),
            delta_y = float(delta_pert_ci),
        )
        methods = sorted(panel["method"].dropna().unique())
        method_colors = {
            method: _shade_plot_color(
                hex_color = color,
                factor = METHOD_SHADE_FACTORS[index % len(METHOD_SHADE_FACTORS)],
            )
            for index, method in enumerate(methods)
        }
        _draw_point_fingerprint(
            axis = axis,
            panel = panel,
            category_col = "method",
            category_order = methods,
            colors = method_colors,
        )
        axis.set_xlim(left = -PLOT_LIMIT, right = PLOT_LIMIT)
        axis.set_ylim(bottom = -PLOT_LIMIT, top = PLOT_LIMIT)
        axis.set_xticks(ticks = FINGERPRINT_TICKS)
        axis.set_yticks(ticks = FINGERPRINT_TICKS)
        if column_index == 0:
            axis.set_ylabel(r"$\Delta$ CI at Max Perturbed Intensity", labelpad = 3.0)
        _style_delta_axis(axis = axis, show_left = True, show_bottom = True, format_x_ticks = True, hide_lowest_y_tick_label = True)
        _annotate_panel_delta_median(
            axis = axis,
            panel = panel,
        )
    
    
    ## falsification panels
    for panel_index, track in enumerate(fals_track_order[:2]):
        axis = bottom_axes[panel_index]
        panel = falsification_fingerprint.loc[falsification_fingerprint[fals_track_col] == track].copy()
        axis.add_patch(
            Rectangle(
                xy = (-PLOT_LIMIT, -PLOT_LIMIT),
                width = PLOT_LIMIT,
                height = PLOT_LIMIT,
                facecolor = "#EEC6C6",
                edgecolor = "none",
                alpha = 0.45,
                zorder = 1,
            )
        )
        axis.text(
            x = -0.47,
            y = -0.47,
            s = "Inferior",
            ha = "left",
            va = "bottom",
            color = "#9B2F2F",
            zorder = 4,
        )
        axis.axhline(y = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        axis.axvline(x = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        _draw_point_fingerprint(
            axis = axis,
            panel = panel,
            category_col = fals_method_col,
            category_order = fals_method_order,
            colors = fals_method_colors,
        )
        axis.set_title(
            _format_label(track),
            color = "#9B2F2F",
            fontweight = "bold",
            pad = 6.2,
        )
        axis.set_xlim(left = -PLOT_LIMIT, right = PLOT_LIMIT)
        axis.set_ylim(bottom = -PLOT_LIMIT, top = PLOT_LIMIT)
        axis.set_xticks(ticks = FINGERPRINT_TICKS)
        axis.set_yticks(ticks = FINGERPRINT_TICKS)
        axis.set_xlabel("")
        if panel_index == 0:
            axis.set_ylabel(r"$\Delta$ CI (Test $-$ Original)", labelpad = 3.0)
        _style_delta_axis(axis = axis, show_left = True, show_bottom = True, format_x_ticks = True, hide_lowest_y_tick_label = True)
        _annotate_panel_delta_median(
            axis = axis,
            panel = panel,
        )
    
    
    ## decomposition panels
    for panel_index, (family, specs) in enumerate(decomposition_family_specs.items(), start = 2):
        axis = bottom_axes[panel_index]
        available_specs = [
            spec
            for spec in specs
            if spec in set(decomposition_fingerprint["specification"])
    ]
        panel = decomposition_fingerprint.loc[
            decomposition_fingerprint["specification"].isin(available_specs)
        ].copy()
        axis.add_patch(
            Rectangle(
                xy = (-PLOT_LIMIT, -PLOT_LIMIT),
                width = PLOT_LIMIT + float(delta_decomp_ei),
                height = PLOT_LIMIT + float(delta_decomp_ci),
                facecolor = "#F0E8B0",
                edgecolor = "none",
                alpha = 0.55,
                zorder = 1,
            )
        )
        axis.text(
            x = -0.48,
            y = -0.47,
            s = "Non-Superior",
            ha = "left",
            va = "bottom",
            color = "#B35F00",
            zorder = 4,
        )
        axis.axhline(y = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        axis.axvline(x = 0.0, color = ZERO_COLOR, lw = 0.6, zorder = 2)
        axis.axhline(y = float(delta_decomp_ci), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        _label_delta_y_lines(axis = axis, delta_value = float(delta_decomp_ci), include_negative = False)
        axis.axvline(x = float(delta_decomp_ei), color = DASHED_GUIDE_COLOR, lw = DASHED_GUIDE_LINEWIDTH, ls = "--", zorder = 2)
        _label_delta_x_lines(axis = axis, delta_value = float(delta_decomp_ei), include_negative = False)
        decomposition_shade_factors = (METHOD_SHADE_FACTORS[0], METHOD_SHADE_FACTORS[-1])
        draw_colors = {
            spec: _shade_plot_color(
                hex_color = decomposition_base_color,
                factor = decomposition_shade_factors[index % len(decomposition_shade_factors)],
            )
            for index, spec in enumerate(available_specs)
        }
        _draw_point_fingerprint(
            axis = axis,
            panel = panel,
            category_col = "specification",
            category_order = available_specs,
            colors = draw_colors,
            labels = decomposition_spec_labels,
        )
        axis.set_title(
            family,
            color = "#B35F00",
            fontweight = "bold",
            pad = 6.2,
        )
        axis.set_xlim(left = -PLOT_LIMIT, right = PLOT_LIMIT)
        axis.set_ylim(bottom = -PLOT_LIMIT, top = PLOT_LIMIT)
        axis.set_xticks(ticks = FINGERPRINT_TICKS)
        axis.set_yticks(ticks = FINGERPRINT_TICKS)
        axis.set_xlabel("")
        _style_delta_axis(axis = axis, show_left = True, show_bottom = True, format_x_ticks = True, hide_lowest_y_tick_label = True)
        _annotate_panel_delta_median(
            axis = axis,
            panel = panel,
        )
    
    
    ## panel letters and shared labels
    fig.canvas.draw()
    outer_label_axes = [
        perturbation_axes[0, 0],
        perturbation_axes[1, 0],
        perturbation_axes[2, 0],
        bottom_axes[0],
    ]
    OUTER_Y_LABEL_SHIFT_POINTS = 12.5
    _outer_label_x_fig = min(
        axis.yaxis.label.get_window_extent(renderer = fig.canvas.get_renderer()).x0
        for axis in outer_label_axes
    )
    _outer_label_shift_pixels = fig.dpi * OUTER_Y_LABEL_SHIFT_POINTS / 72.0
    _outer_label_x_fig_frac = fig.transFigure.inverted().transform((_outer_label_x_fig + _outer_label_shift_pixels, 0))[0]
    for axis in outer_label_axes:
        _ax_x0 = axis.get_position().x0
        _ax_width = axis.get_position().width
        axis.yaxis.set_label_coords((_outer_label_x_fig_frac - _ax_x0) / _ax_width, 0.5)
    row_1_position = perturbation_axes[1, 0].get_position()
    row_2_position = perturbation_axes[2, 0].get_position()
    falsification_position = bottom_axes[0].get_position()
    decomposition_position = bottom_axes[2].get_position()
    PANEL_LABEL_X_OFFSET = -0.055
    PANEL_D_LABEL_X_OFFSET = -0.04
    fig.text(
        x = ART_LEFT + PANEL_LABEL_X_OFFSET,
        y = perturbation_axes[0, 0].get_position().y1 + 0.01,
        s = "a",
        ha = "left",
        va = "bottom",
        fontweight = "bold",
    )
    fig.text(
        x = decomposition_position.x0 + PANEL_D_LABEL_X_OFFSET,
        y = falsification_position.y1 + 0.01,
        s = "d",
        ha = "left",
        va = "bottom",
        fontweight = "bold",
    )
    fig.text(
        x = ART_LEFT + PANEL_LABEL_X_OFFSET,
        y = row_2_position.y1 + 0.015,
        s = "b",
        ha = "left",
        va = "bottom",
        fontweight = "bold",
    )
    fig.text(
        x = ART_LEFT + PANEL_LABEL_X_OFFSET,
        y = falsification_position.y1 + 0.01,
        s = "c",
        ha = "left",
        va = "bottom",
        fontweight = "bold",
    )
    fig.text(
        x = 0.5 * (ART_LEFT + ART_RIGHT),
        y = row_1_position.y0 - 0.04,
        s = "Normalized Perturbed Intensity",
        ha = "center",
        va = "top",
    )
    fig.text(
        x = 0.5 * (ART_LEFT + ART_RIGHT),
        y = row_2_position.y0 - 0.022,
        s = r"$\Delta$ EI at Max Perturbed Intensity",
        ha = "center",
        va = "top",
    )
    panel_c_label_y = falsification_position.y0 - 0.022
    fig.text(
        x = 0.5 * (bottom_axes[0].get_position().x0 + bottom_axes[1].get_position().x1),
        y = panel_c_label_y,
        s = r"$\Delta$ EI (Falsified $-$ Original)",
        ha = "center",
        va = "top",
    )
    fig.text(
        x = 0.5 * (bottom_axes[2].get_position().x0 + bottom_axes[3].get_position().x1),
        y = panel_c_label_y,
        s = r"$\Delta$ EI (Ablated $-$ Original)",
        ha = "center",
        va = "top",
    )
    
    
    _apply_publication_text(figure = fig)
    
    pdf_path = export_pdf_scaled(fig, 4, target_width_mm = 183.0)
    if show:
        plt.show()
    
    return fig, pdf_path


@dataclass(frozen = True)
class VisualizationContext:

    """
    Desc:
        Hold the shared data, analysis identity, and output paths for figures.
    """

    data: pd.DataFrame
    models: Mapping[str, object]
    feat_x: tuple[str, ...]
    feat_z: tuple[str, ...]
    target: str
    n_repeats: int
    random_state: int
    cache_dir: Path
    figure_dir: Path
    submission_dir: Path | None = None
    n_decimals: int = 2
    target_width_mm: float = 183.0


CACHE_SPECS = {
    "consensus": {
        "filename": "consensus_results.pkl",
        "notebook": "consensus.ipynb",
        "keys": {"frontiers", "results_data"},
    },
    "transfer": {
        "filename": "transfer_results.pkl",
        "notebook": "transfer.ipynb",
        "keys": {
            "predicts_dict_domain",
            "predicts_dict_disc",
            "predicts_dict_5fold",
            "predicts_dict_10fold",
            "results_data_domain",
            "results_data_disc",
            "results_data_5fold",
            "results_data_10fold",
            "results_data_consensus",
        },
    },
    "perturb": {
        "filename": "perturb_results.pkl",
        "notebook": "perturb.ipynb",
        "keys": {
            "results_perturbed_transfer",
            "results_perturbed_recovery",
            "results_perturbed_consensus",
        },
    },
    "falsify": {
        "filename": "falsify_results.pkl",
        "notebook": "falsify.ipynb",
        "keys": {
            "results_falsified_transfer",
            "results_falsified_agreement",
            "results_falsified_consensus",
        },
    },
    "ablate": {
        "filename": "ablate_results.pkl",
        "notebook": "ablate.ipynb",
        "keys": {
            "results_decomposed_attribution",
            "predictions_decomposed_attribution",
            "results_decomposed_separation",
            "predictions_decomposed_separation",
            "results_decomposed_consensus",
        },
    },
}


def load_results_cache(
    name: str,
    context: VisualizationContext,
    ) -> dict[str, Any]:

    """
    Desc:
        Load and validate one producer notebook cache.

    Args:
        name: Cache name from `CACHE_SPECS`.
        context: Current visualization execution context.

    Returns:
        Validated cache payload.

    Raises:
        ValueError: If the cache name is unknown.
        FileNotFoundError: If the required cache does not exist.
        RuntimeError: If the cache payload or metadata is incompatible.
    """

    if name not in CACHE_SPECS:
        raise ValueError(f"Unknown results cache: {name}")

    spec = CACHE_SPECS[name]
    path = context.cache_dir / str(spec["filename"])
    notebook = str(spec["notebook"])
    guidance = (
        f"Run notebooks/{notebook} through its post-processing cell, "
        "then rerun this figure."
    )
    if not path.exists():
        raise FileNotFoundError(f"Required cache not found: {path}. {guidance}")

    payload = pickle.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid cache payload at {path}: expected a dictionary. {guidance}")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Cache metadata is missing at {path}. {guidance}")

    expected_metadata = {
        "n_obs": len(context.data),
        "n_repeats": context.n_repeats,
        "random_state": context.random_state,
        "model_names": sorted(context.models.keys()),
        "target": context.target,
        "feat_x": list(context.feat_x),
        "feat_z": list(context.feat_z),
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(f"Incompatible cache at {path}: {details}. {guidance}")

    required_keys = set(spec["keys"])
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise RuntimeError(f"Cache at {path} is missing keys {missing_keys}. {guidance}")
    return payload


def export_pdf_scaled(
    fig: Figure,
    index: int,
    output_dir: Path,
    target_width_mm: float = 183.0,
    pad_in: float = 0.1,
    bbox: mpl.transforms.Bbox | None = None,
    ) -> Path:

    """
    Desc:
        Crop and uniformly scale a vector PDF to the publication width.

    Args:
        fig: Figure to export.
        index: Publication figure number.
        output_dir: Publication output directory.
        target_width_mm: Final page width in millimetres.
        pad_in: Tight-bounding-box padding in inches.
        bbox: Optional explicit crop bounding box in inches.

    Returns:
        Path to the scaled publication PDF.
    """

    logging.getLogger("fontTools").setLevel(logging.ERROR)
    output_dir.mkdir(parents = True, exist_ok = True)
    fig.canvas.draw()
    if bbox is None:
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = mpl.transforms.Bbox.from_extents(
            tight.x0 - pad_in,
            tight.y0 - pad_in,
            tight.x1 + pad_in,
            tight.y1 + pad_in,
        )
    width_in = bbox.width
    height_in = bbox.height
    scale = target_width_mm / (width_in * 25.4)

    minimum_line_width = 0.25 / scale
    for artist in fig.findobj():
        if hasattr(artist, "set_linewidths") and hasattr(artist, "get_linewidths"):
            line_widths = np.atleast_1d(artist.get_linewidths()).astype(float)
            thin_lines = (line_widths > 0) & (line_widths < minimum_line_width)
            if line_widths.size and thin_lines.any():
                artist.set_linewidths(np.where(thin_lines, minimum_line_width, line_widths))
        elif hasattr(artist, "set_linewidth") and hasattr(artist, "get_linewidth"):
            line_width = artist.get_linewidth()
            if line_width is not None and 0 < line_width < minimum_line_width:
                artist.set_linewidth(minimum_line_width)
    fig.canvas.draw()

    native_path = output_dir / f".fig{index}_native.pdf"
    final_path = output_dir / f"fig{index}.pdf"
    with mpl.rc_context(rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig.savefig(
            native_path,
            format = "pdf",
            dpi = 300.0 * scale,
            facecolor = fig.get_facecolor(),
            bbox_inches = bbox,
            pad_inches = 0.0,
        )

    page_width = width_in * 72.0 * scale
    page_height = height_in * 72.0 * scale
    subprocess.run(
        [
            "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dAutoRotatePages=/None",
            f"-dDEVICEWIDTHPOINTS={page_width:.4f}",
            f"-dDEVICEHEIGHTPOINTS={page_height:.4f}",
            "-dFIXEDMEDIA",
            "-dColorConversionStrategy=/LeaveColorUnchanged",
            f"-sOutputFile={final_path}",
            "-c", f"<</BeginPage{{{scale:.6f} {scale:.6f} scale}}>> setpagedevice",
            "-f", str(native_path),
        ],
        check = True,
    )
    native_path.unlink(missing_ok = True)
    return final_path


## backwards-compatible alias
export_nature_pdf_scaled = export_pdf_scaled


def _figure_exporter(context: VisualizationContext) -> Callable[..., Path]:

    def exporter(
        fig: Figure,
        index: int,
        target_width_mm: float = context.target_width_mm,
        pad_in: float = 0.1,
        bbox: mpl.transforms.Bbox | None = None,
        ) -> Path:
        return export_pdf_scaled(
            fig = fig,
            index = index,
            output_dir = context.figure_dir,
            target_width_mm = target_width_mm,
            pad_in = pad_in,
            bbox = bbox,
        )

    return exporter


def generate_conceptual_figure(
    context: VisualizationContext,
    show: bool = False,
    ) -> tuple[Figure, Path]:

    """
    Desc:
        Generate and export the conceptual capacity-frontier figure.
    """

    return build_capacity_frontier_illustration(
        figure_dir = context.figure_dir,
        export_pdf_scaled = _figure_exporter(context = context),
    )


def generate_universality_figure(
    context: VisualizationContext,
    show: bool = False,
    ) -> tuple[Figure, dict[str, object], str, Path]:

    """
    Desc:
        Generate and export the cache-backed universality superfigure.
    """

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.sf": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    load_results_cache(name = "transfer", context = context)
    transfer_cache_path = context.cache_dir / "transfer_results.pkl"
    fig, axes, source_label = plot_universality_superfigure(
        data = context.data,
        models = context.models,
        feat_x = context.feat_x,
        feat_z = context.feat_z,
        target = context.target,
        random_state = context.random_state,
        n_repeats = context.n_repeats,
        universality_cache_path = transfer_cache_path,
        transfer_consensus_cache_path = transfer_cache_path,
        force_recompute = False,
        require_cache = True,
        show = False,
    )

    pdf_path = _figure_exporter(context = context)(
        fig = fig,
        index = 2,
        target_width_mm = context.target_width_mm,
    )
    if show:
        plt.show()
    return fig, axes, source_label, pdf_path


def generate_consensus_figure(
    context: VisualizationContext,
    show: bool = False,
    ) -> tuple[Figure, Path]:

    """
    Desc:
        Generate and export the cache-backed consensus superfigure.
    """

    consensus = load_results_cache(name = "consensus", context = context)
    perturb = load_results_cache(name = "perturb", context = context)
    falsify = load_results_cache(name = "falsify", context = context)
    ablate = load_results_cache(name = "ablate", context = context)
    for name, payload, required_key in (
        ("perturb", perturb, "results_perturbed_full_agreement"),
        ("falsify", falsify, "results_falsified_full_agreement"),
        ("ablate", ablate, "results_decomposed_full_agreement"),
    ):
        if required_key not in payload:
            raise RuntimeError(
                f"Figure 3 requires full-corpus consensus results. Run notebooks/{name}.ipynb "
                "through its consensus training and post-processing cells, then rerun this figure."
            )
    original_agreement = compile_corpus_full(
        predictions = consensus["frontiers"],
        y_true = _log_transformer(context.data[context.target]).to_numpy(dtype = float),
    )
    return _render_consensus_figure(
        results_data = consensus["results_data"],
        results_perturbed_consensus = perturb["results_perturbed_consensus"],
        results_decomposed_consensus = ablate["results_decomposed_consensus"],
        results_falsified_consensus = falsify["results_falsified_consensus"],
        results_original_agreement = original_agreement,
        results_decomposed_full_agreement = ablate["results_decomposed_full_agreement"],
        results_perturbed_full_agreement = perturb["results_perturbed_full_agreement"],
        results_falsified_full_agreement = falsify["results_falsified_full_agreement"],
        figure_dir = context.figure_dir,
        export_pdf_scaled = _figure_exporter(context = context),
        n_decimals = context.n_decimals,
        show = show,
    )


def generate_stress_test_figure(
    context: VisualizationContext,
    show: bool = False,
    ) -> tuple[Figure, Path]:

    """
    Desc:
        Generate and export the cache-backed stress-test superfigure.
    """

    perturb = load_results_cache(name = "perturb", context = context)
    falsify = load_results_cache(name = "falsify", context = context)
    ablate = load_results_cache(name = "ablate", context = context)
    return _render_stress_test_figure(
        results_perturbed_transfer = perturb["results_perturbed_transfer"],
        results_perturbed_recovery = perturb["results_perturbed_recovery"],
        results_falsified_transfer = falsify["results_falsified_transfer"],
        results_falsified_agreement = falsify["results_falsified_agreement"],
        results_decomposed_separation = ablate["results_decomposed_separation"],
        figure_dir = context.figure_dir,
        export_pdf_scaled = _figure_exporter(context = context),
        n_decimals = context.n_decimals,
        show = show,
    )
