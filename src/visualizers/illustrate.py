## directives
from __future__ import annotations

## libraries
import logging
import numpy as np
import matplotlib as mpl
import matplotlib._mathtext as _mathtext
import matplotlib.pyplot as plt
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon
from matplotlib.text import Text
from matplotlib.textpath import TextPath

## 
def build_capacity_frontier_illustration(
    figure_dir: Path,
    export_pdf_scaled: Callable[..., Path],
    ) -> tuple[mpl.figure.Figure, Path]:

    """
    Desc:
        Build and export the conceptual network-capacity illustration.

    Args:
        figure_dir: Directory for project figure artifacts.
        export_pdf_scaled: Publication PDF export callback.

    Returns:
        Figure and project PDF artifact path.
    """
    FIGURE_DIR = figure_dir
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    _mathtext.SHRINK_FACTOR = 0.73
    ## n3s conceptual figure: observations, feasible frontier, and substrate
    ## reproducible, vectorizable, notebook-ready
    
    
    ## reproducibility and style
    rng = np.random.default_rng(seed = 7)
    BASE_FIGSIZE_INCHES = (10.24, 7.68)
    TARGET_WIDTH_MM = 183.0
    SCALE = TARGET_WIDTH_MM / (BASE_FIGSIZE_INCHES[0] * 25.4)
    FIGSIZE_INCHES = (BASE_FIGSIZE_INCHES[0] * SCALE, BASE_FIGSIZE_INCHES[1] * SCALE)
    N3S_FONT_FAMILY = "Arial"
    N3S_FONT_SIZE = 3.84  ## tuned: ~7.0 pt final after 183 mm upscale (x1.587, no pad)
    VECTOR_PANEL_NUMBER_FONT_SIZE = max(1, int(round(4 * SCALE)))
    mpl.rcParams.update({
        "font.family": N3S_FONT_FAMILY,
        "font.sans-serif": [N3S_FONT_FAMILY],
        "font.size": N3S_FONT_SIZE,
        "axes.labelsize": N3S_FONT_SIZE,
        "axes.titlesize": N3S_FONT_SIZE,
        "xtick.labelsize": N3S_FONT_SIZE,
        "ytick.labelsize": N3S_FONT_SIZE,
        "legend.fontsize": N3S_FONT_SIZE,
        "figure.titlesize": N3S_FONT_SIZE,
        "axes.linewidth": 0.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": N3S_FONT_FAMILY,
        "mathtext.sf": N3S_FONT_FAMILY,
        "mathtext.it": f"{N3S_FONT_FAMILY}:italic",
        "mathtext.bf": f"{N3S_FONT_FAMILY}:bold",
        "mathtext.default": "it",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    
    ## palette
    PURPLE_DARK = "#5A005A"
    PURPLE = "#800080"
    PURPLE_LIGHT = "#D9B3D9"
    STRUCTURE_NAVY_DARK = "#1F45A8"
    STRUCTURE_NAVY = "#4169E1"
    STRUCTURE_NAVY_LIGHT = "#AFC3FF"
    STRUCTURE_PLANE_FILL = "#EEF3FF"
    STRUCTURE_PLANE_GRID = "#7892DD"
    PROCESS_PINK_DARK = "#B0185A"
    PROCESS_PINK = "#E04F85"
    PROCESS_PINK_LIGHT = "#F5A6C8"
    PROCESS_PLANE_FILL = "#FDE7F1"
    PROCESS_PLANE_GRID = "#EA7EA8"
    GREEN_DARK = "#5A005A"
    GREEN = "#800080"
    GREEN_LIGHT = "#D9B3D9"
    GREEN_GRID = "#A05FA0"
    FRONTIER_MARKER_GOLD = "#FFD84D"
    TEXT_DARK = "#111111"
    TEXT_PURPLE = "#5A005A"
    TEXT_STRUCTURE = "#1F45A8"
    TEXT_PROCESS = "#B0185A"
    TEXT_GREEN = "#5A005A"
    
    ## square layer geometry
    PLANE_HALF_SIZE = 1.18
    PLANE_X_MIN = -PLANE_HALF_SIZE
    PLANE_X_MAX = PLANE_HALF_SIZE
    PLANE_Y_MIN = -PLANE_HALF_SIZE
    PLANE_Y_MAX = PLANE_HALF_SIZE
    SUBSTRATE_Z = 0.00
    STRUCTURAL_Z = 1.05
    FRONTIER_BASE_Z = 2.35
    PROCESS_Z = 3.72
    OBSERVATION_Z = 4.58
    
    ## oblique view: yaw turns the scene sideways; tilt pitches planes toward the reader
    VIEW_YAW_DEGREES = 38.0
    VIEW_SIDE_SCALE = 1.12
    VIEW_TILT_SCALE = 0.36
    VIEW_YAW_RADIANS = np.deg2rad(VIEW_YAW_DEGREES)
    
    
    def project(
        x_coord: float | np.ndarray,
        y_coord: float | np.ndarray,
        z_coord: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        side_coord = (
            np.cos(VIEW_YAW_RADIANS) * x_coord
            - np.sin(VIEW_YAW_RADIANS) * y_coord
        )
        depth_coord = (
            np.sin(VIEW_YAW_RADIANS) * x_coord
            + np.cos(VIEW_YAW_RADIANS) * y_coord
        )
        return VIEW_SIDE_SCALE * side_coord, z_coord + VIEW_TILT_SCALE * depth_coord
    
    
    def plane_display_basis(axis: Axes, z_level: float) -> tuple[np.ndarray, np.ndarray]:
        origin_data = np.array(
            object = project(x_coord = 0.0, y_coord = 0.0, z_coord = z_level),
            dtype = float,
        )
        x_basis_data = np.array(
            object = project(x_coord = 1.0, y_coord = 0.0, z_coord = z_level),
            dtype = float,
        )
        y_basis_data = np.array(
            object = project(x_coord = 0.0, y_coord = 1.0, z_coord = z_level),
            dtype = float,
        )
        origin_display = axis.transData.transform(values = origin_data)
        x_direction = axis.transData.transform(values = x_basis_data) - origin_display
        y_direction = axis.transData.transform(values = y_basis_data) - origin_display
        x_direction = x_direction / float(np.linalg.norm(x = x_direction))
        y_direction = y_direction / float(np.linalg.norm(x = y_direction))
        return x_direction, y_direction
    
    
    def surface_height(
        x_coord: float | np.ndarray,
        y_coord: float | np.ndarray,
    ) -> float | np.ndarray:
        x_scaled = x_coord / PLANE_HALF_SIZE
        y_scaled = y_coord / PLANE_HALF_SIZE
        rear_blend = np.clip(a = (y_scaled - 0.12) / 0.88, a_min = 0.0, a_max = 1.0)
        rear_blend = rear_blend * rear_blend * (3.0 - 2.0 * rear_blend)
        rolling_wave = (
            0.15 * np.sin(np.pi * (0.92 * x_scaled - 0.34 * y_scaled) + 0.28)
            + 0.09 * np.cos(np.pi * (0.38 * x_scaled + 1.02 * y_scaled) - 0.18)
            + 0.05 * np.sin(np.pi * (1.22 * y_scaled + 0.24))
        )
        saddle = 0.075 * (0.72 * x_scaled ** 2 - 0.46 * y_scaled ** 2) - 0.055 * x_scaled * y_scaled
        corner_flare = 0.070 * (x_scaled ** 2) * (y_scaled ** 2)
        rear_wave = 0.045 * np.sin(np.pi * (0.82 * x_scaled + 0.20))
        return FRONTIER_BASE_Z + (1.0 - rear_blend) * (rolling_wave + saddle) + rear_blend * rear_wave + corner_flare
    
    
    def plane_polygon(z_level: float) -> np.ndarray:
        plane_xyz = np.array(
            object = [
                [PLANE_X_MIN, PLANE_Y_MIN, z_level],
                [PLANE_X_MAX, PLANE_Y_MIN, z_level],
                [PLANE_X_MAX, PLANE_Y_MAX, z_level],
                [PLANE_X_MIN, PLANE_Y_MAX, z_level],
            ],
            dtype = float,
        )
        projected_x, projected_y = project(
            x_coord = plane_xyz[:, 0],
            y_coord = plane_xyz[:, 1],
            z_coord = plane_xyz[:, 2],
        )
        return np.column_stack(tup = (projected_x, projected_y))
    
    
    def draw_plane(
        axis: Axes,
        z_level: float,
        edge_color: str,
        fill_color: str,
        fill_alpha: float,
        line_width: float,
        z_order: float,
    ) -> np.ndarray:
        polygon_xy = plane_polygon(z_level = z_level)
        face_color = (
            mpl.colors.to_rgba(c = fill_color, alpha = fill_alpha)
            if fill_alpha > 0.0
            else "none"
        )
        plane_patch = Polygon(
            xy = polygon_xy,
            closed = True,
            fill = True,
            facecolor = face_color,
            edgecolor = edge_color,
            linewidth = line_width,
            alpha = 1.0,
            zorder = z_order,
        )
        axis.add_patch(p = plane_patch)
        return polygon_xy
    
    
    def add_line(
        axis: Axes,
        x_values: np.ndarray | list[float],
        y_values: np.ndarray | list[float],
        color: str,
        line_width: float,
        z_order: float,
        alpha: float = 1.0,
        line_style: str | tuple[float, tuple[float, ...]] = "-",
    ) -> None:
        axis.add_line(
            line = Line2D(
                xdata = x_values,
                ydata = y_values,
                color = color,
                linewidth = line_width,
                linestyle = line_style,
                alpha = alpha,
                zorder = z_order,
            )
        )
    
    
    def draw_plane_grid(
        axis: Axes,
        z_level: float,
        grid_color: str,
        grid_alpha: float,
        line_width: float,
        z_order: float,
        n_steps: int = 6,
        n_cols: int | None = None,
        n_rows: int | None = None,
    ) -> None:
        cell_cols = (n_steps - 1) if n_cols is None else n_cols
        cell_rows = (n_steps - 1) if n_rows is None else n_rows
        x_lines = np.linspace(start = PLANE_X_MIN, stop = PLANE_X_MAX, num = cell_cols + 1)
        y_lines = np.linspace(start = PLANE_Y_MIN, stop = PLANE_Y_MAX, num = cell_rows + 1)
        for x_coord in x_lines:
            grid_x, grid_y = project(
                x_coord = np.full(shape = 2, fill_value = x_coord, dtype = float),
                y_coord = np.array(object = [PLANE_Y_MIN, PLANE_Y_MAX], dtype = float),
                z_coord = z_level,
            )
            add_line(
                axis = axis,
                x_values = grid_x,
                y_values = grid_y,
                color = grid_color,
                line_width = line_width,
                alpha = grid_alpha,
                z_order = z_order,
            )
        for y_coord in y_lines:
            grid_x, grid_y = project(
                x_coord = np.array(object = [PLANE_X_MIN, PLANE_X_MAX], dtype = float),
                y_coord = np.full(shape = 2, fill_value = y_coord, dtype = float),
                z_coord = z_level,
            )
            add_line(
                axis = axis,
                x_values = grid_x,
                y_values = grid_y,
                color = grid_color,
                line_width = line_width,
                alpha = grid_alpha,
                z_order = z_order,
            )
    
    
    def draw_plane_cell_numbers(
        axis: Axes,
        z_level: float,
        text_color: str,
        text_alpha: float,
        z_order: float,
        n_steps: int = 6,
        value_offset: int = 0,
        font_size: float = N3S_FONT_SIZE,
        n_cols: int | None = None,
        n_rows: int | None = None,
        label_count: int | None = None,
    ) -> None:
        cell_cols = (n_steps - 1) if n_cols is None else n_cols
        cell_rows = (n_steps - 1) if n_rows is None else n_rows
        x_lines = np.linspace(start = PLANE_X_MIN, stop = PLANE_X_MAX, num = cell_cols + 1)
        y_lines = np.linspace(start = PLANE_Y_MIN, stop = PLANE_Y_MAX, num = cell_rows + 1)
        x_centers = 0.5 * (x_lines[:-1] + x_lines[1:])
        y_centers = 0.5 * (y_lines[:-1] + y_lines[1:])
        max_label = cell_cols * cell_rows if label_count is None else label_count
        x_direction, y_direction = plane_display_basis(axis = axis, z_level = z_level)
        points_to_pixels = 2.35 * axis.figure.dpi / 72.0
        display_to_data = axis.transData.inverted()
        font_properties = FontProperties(family = N3S_FONT_FAMILY, weight = "regular")
        for row_index, y_coord in enumerate(y_centers[::-1]):
            for column_index, x_coord in enumerate(x_centers):
                seq_val = int(row_index * cell_cols + column_index + 1 + value_offset)
                if seq_val > max_label:
                    continue
                
                label_value = f"{rng.normal(loc=0.0, scale=1.5):.1f}"
                label_x, label_y = project(
                    x_coord = x_coord,
                    y_coord = y_coord,
                    z_coord = z_level,
                )
                label_center_display = axis.transData.transform(values = [label_x, label_y])
                text_path = TextPath(
                    xy = (0.0, 0.0),
                    s = label_value,
                    size = font_size,
                    prop = font_properties,
                )
                text_bounds = text_path.get_extents()
                text_vertices = text_path.vertices.copy()
                text_vertices[:, 0] -= 0.5 * (text_bounds.x0 + text_bounds.x1)
                text_vertices[:, 1] -= 0.5 * (text_bounds.y0 + text_bounds.y1)
                display_vertices = (
                    label_center_display
                    + points_to_pixels * (
                        text_vertices[:, [0]] * x_direction[None, :]
                        + 0.72 * text_vertices[:, [1]] * y_direction[None, :]
                    )
                )
                data_vertices = display_to_data.transform(values = display_vertices)
                label_path = MplPath(vertices = data_vertices, codes = text_path.codes)
                axis.add_patch(
                    p = PathPatch(
                        path = label_path,
                        facecolor = text_color,
                        edgecolor = text_color,
                        linewidth = 0.08,
                        alpha = text_alpha,
                        zorder = z_order,
                    )
                )
    
    
    def draw_wavy_tether(
        axis: Axes,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        color: str,
        line_width: float,
        z_order: float,
        alpha: float,
        phase: float,
        amplitude: float = 0.018,
    ) -> None:
        t_values = np.linspace(start = 0.0, stop = 1.0, num = 80)
        path_xy = (
            (1.0 - t_values)[:, None] * start_xy[None, :]
            + t_values[:, None] * end_xy[None, :]
        )
        direction = end_xy - start_xy
        normal = np.array(object = [-direction[1], direction[0]], dtype = float)
        normal_length = float(np.linalg.norm(x = normal))
        if normal_length > 0.0:
            normal = normal / normal_length
            path_xy = path_xy + amplitude * np.sin(2.0 * np.pi * t_values + phase)[:, None] * normal
        add_line(
            axis = axis,
            x_values = path_xy[:, 0],
            y_values = path_xy[:, 1],
            color = color,
            line_width = line_width,
            line_style = (0.0, (1.0, 2.2)),
            alpha = alpha,
            z_order = z_order,
        )
    
    
    ## figure setup
    fig, ax = plt.subplots(figsize = FIGSIZE_INCHES, facecolor = "white")
    ax.set_aspect(aspect = "equal")
    ax.axis("off")
    ax.set_xlim(left = -2.88, right = 3.25)
    ax.set_ylim(bottom = -0.88, top = 5.98)
    fig.subplots_adjust(left = 0.02, right = 0.98, bottom = 0.03, top = 0.97)
    fig.canvas.draw()
    
    ## substrate graph without any plane underneath
    substrate_points = np.array(
        object = [
            [-1.12, -0.78], [-1.18, -0.31], [-1.02, 0.20], [-1.15, 0.71], [-0.86, 1.03],
            [-0.78, -1.08], [-0.60, -0.61], [-0.76, -0.12], [-0.54, 0.35], [-0.68, 0.83],
            [-0.36, -0.76], [-0.16, -0.43], [-0.33, 0.10], [-0.10, 0.70], [-0.30, 1.14],
            [0.06, -1.03], [0.30, -0.62], [0.10, -0.06], [0.36, 0.39], [0.16, 0.96],
            [0.48, -0.84], [0.73, -0.28], [0.52, 0.20], [0.82, 0.68], [0.56, 1.10],
            [0.94, -1.02], [1.16, -0.54], [0.96, 0.04], [1.12, 0.56], [0.92, 0.86],
            [-0.52, 1.02], [0.34, -1.16],
        ],
        dtype = float,
    )
    substrate_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9),
        (10, 11), (11, 12), (12, 13), (13, 14), (15, 16), (16, 17), (17, 18), (18, 19),
        (20, 21), (21, 22), (22, 23), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11), (7, 12),
        (8, 13), (9, 14), (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 20), (16, 21), (17, 22), (18, 23), (19, 24), (20, 25), (21, 26),
        (22, 27), (23, 28), (24, 29), (1, 7), (3, 9), (6, 12), (8, 14),
        (11, 17), (13, 19), (16, 22), (18, 24), (21, 27), (23, 29),
        (4, 30), (9, 30), (14, 30), (15, 31), (20, 31), (25, 31), (12, 22), (7, 17),
    ]
    substrate_x, substrate_y = project(
        x_coord = substrate_points[:, 0],
        y_coord = substrate_points[:, 1],
        z_coord = SUBSTRATE_Z,
    )
    for source_idx, target_idx in substrate_edges:
        add_line(
            axis = ax,
            x_values = [substrate_x[source_idx], substrate_x[target_idx]],
            y_values = [substrate_y[source_idx], substrate_y[target_idx]],
            color = STRUCTURE_NAVY,
            line_width = 0.72,
            alpha = 0.90,
            z_order = 3.75,
        )
    ax.scatter(
        x = substrate_x,
        y = substrate_y,
        s = 28,
        c = STRUCTURE_NAVY_DARK,
        edgecolors = "white",
        linewidths = 0.28,
        zorder = 4.0,
    )
    
    ## structural abstraction tethers and representation plane
    ## Sparse, evenly spaced structural tethers: choose substrate nodes whose
    ## projected screen-x positions are uniformly spread so the upward dashed
    ## arrows do not visually overlap.
    N_STRUCTURAL_ARROWS = 9
    _tether_targets = np.linspace(substrate_x.min(), substrate_x.max(), N_STRUCTURAL_ARROWS)
    structural_anchor_idx = []
    _tether_used = set()
    for _target_x in _tether_targets:
        for _candidate in np.argsort(np.abs(substrate_x - _target_x)):
            if int(_candidate) not in _tether_used:
                _tether_used.add(int(_candidate))
                structural_anchor_idx.append(int(_candidate))
                break
    structural_anchor_idx = sorted(structural_anchor_idx)
    for anchor_idx in structural_anchor_idx:
        start_xy = np.array(
            object = project(
                x_coord = substrate_points[anchor_idx, 0],
                y_coord = substrate_points[anchor_idx, 1],
                z_coord = SUBSTRATE_Z + 0.05,
            ),
            dtype = float,
        )
        end_xy = np.array(
            object = project(
                x_coord = substrate_points[anchor_idx, 0],
                y_coord = substrate_points[anchor_idx, 1],
                z_coord = STRUCTURAL_Z,
            ),
            dtype = float,
        )
        arrow_patch = FancyArrowPatch(
            posA = tuple(start_xy),
            posB = tuple(end_xy),
            arrowstyle = "-|>",
            mutation_scale = 7.5,
            color = STRUCTURE_NAVY,
            linewidth = 0.95,
            linestyle = (0.0, (1.0, 2.2)),
            alpha = 0.46,
            zorder = 2.4,
        )
        ax.add_patch(p = arrow_patch)
    
    draw_plane(
        axis = ax,
        z_level = STRUCTURAL_Z,
        edge_color = STRUCTURE_NAVY,
        fill_color = STRUCTURE_PLANE_FILL,
        fill_alpha = 0.28,
        line_width = 0.85,
        z_order = 3.2,
    )
    
    N_OBS_PANEL = 21
    
    draw_plane_grid(
        axis = ax,
        z_level = STRUCTURAL_Z,
        grid_color = STRUCTURE_PLANE_GRID,
        grid_alpha = 0.48,
        line_width = 0.42,
        z_order = 3.35,
        n_cols = 21,
        n_rows = N_OBS_PANEL,
    )
    draw_plane_cell_numbers(
        axis = ax,
        z_level = STRUCTURAL_Z,
        text_color = STRUCTURE_NAVY_DARK,
        text_alpha = 0.82,
        z_order = 4.90,
        n_cols = 21,
        n_rows = N_OBS_PANEL,
        value_offset = 0,
        font_size = VECTOR_PANEL_NUMBER_FONT_SIZE * 0.35,
    )
    
    ## learned feasible frontier surface
    surface_x = np.linspace(start = PLANE_X_MIN, stop = PLANE_X_MAX, num = 34)
    surface_y = np.linspace(start = PLANE_Y_MIN, stop = PLANE_Y_MAX, num = 34)
    surface_x_grid, surface_y_grid = np.meshgrid(surface_x, surface_y)
    surface_z_grid = surface_height(x_coord = surface_x_grid, y_coord = surface_y_grid)
    MANIFOLD_LAYER_OFFSET = 0.25
    frontier_arrow_interior = rng.uniform(
        low = [PLANE_X_MIN + 0.18, PLANE_Y_MIN + 0.18],
        high = [PLANE_X_MAX - 0.18, PLANE_Y_MAX - 0.18],
        size = (14, 2),
    )
    frontier_arrow_corner_centers = np.array(
        object = [
            [PLANE_X_MIN + 0.18, PLANE_Y_MIN + 0.18],
            [PLANE_X_MAX - 0.18, PLANE_Y_MIN + 0.18],
            [PLANE_X_MAX - 0.18, PLANE_Y_MAX - 0.18],
            [PLANE_X_MIN + 0.18, PLANE_Y_MAX - 0.18],
        ],
        dtype = float,
    )
    frontier_arrow_corners = np.repeat(a = frontier_arrow_corner_centers, repeats = 2, axis = 0)
    frontier_arrow_corners = frontier_arrow_corners + rng.normal(loc = 0.0, scale = 0.12, size = frontier_arrow_corners.shape)
    frontier_arrow_points = np.vstack(tup = (frontier_arrow_interior, frontier_arrow_corners))
    frontier_arrow_points[:, 0] = np.clip(a = frontier_arrow_points[:, 0], a_min = PLANE_X_MIN + 0.10, a_max = PLANE_X_MAX - 0.10)
    frontier_arrow_points[:, 1] = np.clip(a = frontier_arrow_points[:, 1], a_min = PLANE_Y_MIN + 0.10, a_max = PLANE_Y_MAX - 0.10)
    frontier_arrow_x = frontier_arrow_points[:, 0]
    frontier_arrow_y = frontier_arrow_points[:, 1]
    ## omit the two outermost frontier arrows on each side (4 total): drop the
    ## arrows whose projected screen-x is the most extreme left and right, so the
    ## arrows flanking the capacity-frontier manifold are removed.
    frontier_arrow_screen_x = VIEW_SIDE_SCALE * (
        np.cos(VIEW_YAW_RADIANS) * frontier_arrow_x - np.sin(VIEW_YAW_RADIANS) * frontier_arrow_y
    )
    _frontier_x_sorted = np.argsort(frontier_arrow_screen_x)
    _frontier_omit = set(_frontier_x_sorted[:2].tolist()) | set(_frontier_x_sorted[-2:].tolist())
    frontier_arrow_order = np.array(
        [int(i) for i in np.argsort(frontier_arrow_y) if int(i) not in _frontier_omit]
    )
    for arrow_index in frontier_arrow_order:
        arrow_x = frontier_arrow_x[arrow_index]
        arrow_y = frontier_arrow_y[arrow_index]
        arrow_start = np.array(
            object = project(
                x_coord = arrow_x,
                y_coord = arrow_y,
                z_coord = STRUCTURAL_Z + 0.035,
            ),
            dtype = float,
        )
        arrow_end = np.array(
            object = project(
                x_coord = arrow_x,
                y_coord = arrow_y,
                z_coord = surface_height(x_coord = arrow_x, y_coord = arrow_y),
            ),
            dtype = float,
        )
        frontier_arrow = FancyArrowPatch(
            posA = tuple(arrow_start),
            posB = tuple(arrow_end),
            arrowstyle = "-|>",
            mutation_scale = 5.6,
            color = STRUCTURE_NAVY,
            linewidth = 0.95,
            linestyle = (0.0, (1.0, 2.2)),
            alpha = 0.46,
            zorder = 4.18 + 0.001 * float(arrow_index),
        )
        ax.add_patch(p = frontier_arrow)
    
    
    def build_surface_polygons(z_offset: float) -> list[np.ndarray]:
        surface_polygons = []
        for row_idx in range(surface_x_grid.shape[0] - 1):
            for col_idx in range(surface_x_grid.shape[1] - 1):
                cell_x = np.array(
                    object = [
                        surface_x_grid[row_idx, col_idx],
                        surface_x_grid[row_idx, col_idx + 1],
                        surface_x_grid[row_idx + 1, col_idx + 1],
                        surface_x_grid[row_idx + 1, col_idx],
                    ],
                    dtype = float,
                )
                cell_y = np.array(
                    object = [
                        surface_y_grid[row_idx, col_idx],
                        surface_y_grid[row_idx, col_idx + 1],
                        surface_y_grid[row_idx + 1, col_idx + 1],
                        surface_y_grid[row_idx + 1, col_idx],
                    ],
                    dtype = float,
                )
                cell_z = surface_height(x_coord = cell_x, y_coord = cell_y) + z_offset
                cell_plot_x, cell_plot_y = project(x_coord = cell_x, y_coord = cell_y, z_coord = cell_z)
                surface_polygons.append(np.column_stack(tup = (cell_plot_x, cell_plot_y)))
        return surface_polygons
    
    
    def draw_manifold_boundary(z_offset: float, z_order: float) -> None:
        offset_surface_z_grid = surface_z_grid + z_offset
        for edge_selector in [
            (0, slice(None)),
            (-1, slice(None)),
            (slice(None), 0),
            (slice(None), -1),
        ]:
            boundary_x, boundary_y = project(
                x_coord = surface_x_grid[edge_selector],
                y_coord = surface_y_grid[edge_selector],
                z_coord = offset_surface_z_grid[edge_selector],
            )
            add_line(
                axis = ax,
                x_values = boundary_x,
                y_values = boundary_y,
                color = PURPLE,
                line_width = 0.28,
                alpha = 0.92,
                z_order = z_order,
            )
    
    
    ax.add_collection(
        collection = PolyCollection(
            verts = build_surface_polygons(z_offset = -MANIFOLD_LAYER_OFFSET),
            facecolors = PROCESS_PINK,
            edgecolors = "none",
            alpha = 0.26,
            zorder = 3.55,
        )
    )
    draw_manifold_boundary(
        z_offset = -MANIFOLD_LAYER_OFFSET,
        z_order = 4.25,
    )
    surface_poly = build_surface_polygons(z_offset = 0.0)
    ax.add_collection(
        collection = PolyCollection(
            verts = surface_poly,
            facecolors = GREEN_LIGHT,
            edgecolors = "none",
            alpha = 0.58,
            zorder = 4.0,
        )
    )
    ax.add_collection(
        collection = PolyCollection(
            verts = build_surface_polygons(z_offset = MANIFOLD_LAYER_OFFSET),
            facecolors = PROCESS_PINK,
            edgecolors = "none",
            alpha = 0.26,
            zorder = 3.85,
        )
    )
    draw_manifold_boundary(
        z_offset = MANIFOLD_LAYER_OFFSET,
        z_order = 4.95,
    )
    for row_idx in range(0, surface_x_grid.shape[0], 2):
        mesh_x, mesh_y = project(
            x_coord = surface_x_grid[row_idx, :],
            y_coord = surface_y_grid[row_idx, :],
            z_coord = surface_z_grid[row_idx, :],
        )
        add_line(
            axis = ax,
            x_values = mesh_x,
            y_values = mesh_y,
            color = GREEN_GRID,
            line_width = 0.62,
            alpha = 0.68,
            z_order = 4.5,
        )
    for col_idx in range(0, surface_x_grid.shape[1], 2):
        mesh_x, mesh_y = project(
            x_coord = surface_x_grid[:, col_idx],
            y_coord = surface_y_grid[:, col_idx],
            z_coord = surface_z_grid[:, col_idx],
        )
        add_line(
            axis = ax,
            x_values = mesh_x,
            y_values = mesh_y,
            color = GREEN_GRID,
            line_width = 0.62,
            alpha = 0.68,
            z_order = 4.5,
        )
    for edge_selector in [
        (0, slice(None)),
        (-1, slice(None)),
    ]:
        boundary_x, boundary_y = project(
            x_coord = surface_x_grid[edge_selector],
            y_coord = surface_y_grid[edge_selector],
            z_coord = surface_z_grid[edge_selector],
        )
        add_line(
            axis = ax,
            x_values = boundary_x,
            y_values = boundary_y,
            color = GREEN_DARK,
            line_width = 1.55,
            alpha = 0.96,
            z_order = 5.0,
        )
    for edge_selector in [
        (slice(None), 0),
        (slice(None), -1),
    ]:
        boundary_x, boundary_y = project(
            x_coord = surface_x_grid[edge_selector],
            y_coord = surface_y_grid[edge_selector],
            z_coord = surface_z_grid[edge_selector],
        )
        add_line(
            axis = ax,
            x_values = boundary_x,
            y_values = boundary_y,
            color = GREEN_DARK,
            line_width = 1.55,
            alpha = 0.96,
            z_order = 5.0,
        )
    
    ## process representation plane and observations
    process_anchor_xy = np.array(
        object = [
            [-0.92, -0.86], [-0.92, -0.32], [-0.92, 0.26], [-0.92, 0.82],
            [-0.42, -0.92], [-0.42, -0.38], [-0.42, 0.30], [-0.42, 0.88],
            [0.06, -0.86], [0.06, -0.26], [0.06, 0.34], [0.06, 0.82],
            [0.56, -0.90], [0.56, -0.36], [0.56, 0.28], [0.56, 0.86],
            [0.98, -0.66], [0.98, 0.54],
        ],
        dtype = float,
    )
    process_anchor_x = process_anchor_xy[:, 0] + rng.normal(loc = 0.0, scale = 0.035, size = process_anchor_xy.shape[0])
    process_anchor_y = process_anchor_xy[:, 1] + rng.normal(loc = 0.0, scale = 0.035, size = process_anchor_xy.shape[0])
    process_anchor_x = np.clip(a = process_anchor_x, a_min = PLANE_X_MIN + 0.08, a_max = PLANE_X_MAX - 0.08)
    process_anchor_y = np.clip(a = process_anchor_y, a_min = PLANE_Y_MIN + 0.08, a_max = PLANE_Y_MAX - 0.08)
    process_plane_x, process_plane_y = project(
        x_coord = process_anchor_x,
        y_coord = process_anchor_y,
        z_coord = PROCESS_Z,
    )
    observation_z = OBSERVATION_Z + rng.normal(loc = 0.0, scale = 0.055, size = process_anchor_x.shape)
    observation_x, observation_y = project(
        x_coord = process_anchor_x,
        y_coord = process_anchor_y,
        z_coord = observation_z,
    )
    
    ## Drop each connector perfectly vertically (constant screen-x) from its
    ## observation onto the substrate network: intersect the vertical line
    ## x = observation_x with the projected graph edges and stop at the topmost
    ## crossing, so every grey line ends exactly on a node or an edge of G while
    ## staying vertical (the projection makes screen-x independent of z).
    substrate_edges_screen = [
        (substrate_x[source_idx], substrate_y[source_idx], substrate_x[target_idx], substrate_y[target_idx])
        for source_idx, target_idx in substrate_edges
    ]
    
    
    def vertical_landing_on_network(line_x, top_y):
        crossings = []
        for edge_x0, edge_y0, edge_x1, edge_y1 in substrate_edges_screen:
            if edge_x0 == edge_x1:
                continue
            if (edge_x0 - line_x) * (edge_x1 - line_x) > 0.0:
                continue
            t = (line_x - edge_x0) / (edge_x1 - edge_x0)
            crossings.append(edge_y0 + t * (edge_y1 - edge_y0))
        below = [y for y in crossings if y <= top_y]
        pool = below if below else crossings
        return max(pool) if pool else None
    
    
    process_substrate_x = np.array(observation_x, dtype = float)
    process_substrate_y = np.empty(len(observation_y), dtype = float)
    for obs_idx in range(len(observation_x)):
        landing_y = vertical_landing_on_network(observation_x[obs_idx], observation_y[obs_idx])
        process_substrate_y[obs_idx] = (
            landing_y if landing_y is not None else float(np.min(substrate_y))
        )
    for obs_idx in range(len(process_anchor_x)):
        add_line(
            axis = ax,
            x_values = [observation_x[obs_idx], process_substrate_x[obs_idx]],
            y_values = [observation_y[obs_idx], process_substrate_y[obs_idx]],
            color = "#8A8A8A",
            line_width = 0.5,
            line_style = "solid",
            alpha = 0.6,
            z_order = 1.0,
        )
    
    surface_anchor_idx = np.argsort(process_anchor_x ** 2 + process_anchor_y ** 2)[:5]
    for surface_idx in surface_anchor_idx:
        surface_z = surface_height(
            x_coord = process_anchor_x[surface_idx],
            y_coord = process_anchor_y[surface_idx],
        )
        surface_xy = np.array(
            object = project(
                x_coord = process_anchor_x[surface_idx],
                y_coord = process_anchor_y[surface_idx],
                z_coord = surface_z,
            ),
            dtype = float,
        )
        ax.scatter(
            x = [surface_xy[0]],
            y = [surface_xy[1]],
            s = 50,
            c = FRONTIER_MARKER_GOLD,
            edgecolors = GREEN_DARK,
            linewidths = 1.5,
            zorder = 6.2,
        )
    
    draw_plane(
        axis = ax,
        z_level = PROCESS_Z,
        edge_color = PROCESS_PINK,
        fill_color = PROCESS_PLANE_FILL,
        fill_alpha = 0.30,
        line_width = 0.85,
        z_order = 6.0,
    )
    draw_plane_grid(
        axis = ax,
        z_level = PROCESS_Z,
        grid_color = PROCESS_PLANE_GRID,
        grid_alpha = 0.52,
        line_width = 0.42,
        z_order = 6.15,
        n_cols = 7,
        n_rows = 7,
    )
    draw_plane_cell_numbers(
        axis = ax,
        z_level = PROCESS_Z,
        text_color = PROCESS_PINK_DARK,
        text_alpha = 0.82,
        z_order = 6.32,
        n_cols = 7,
        n_rows = 7,
        value_offset = 0,
        font_size = VECTOR_PANEL_NUMBER_FONT_SIZE * 0.85,
    )
    process_plane_corners = np.array(
        object = [
            [PLANE_X_MAX, PLANE_Y_MIN],
            [PLANE_X_MIN, PLANE_Y_MAX],
        ],
        dtype = float,
    )
    for corner_x, corner_y in process_plane_corners:
        corner_start_x, corner_start_y = project(
            x_coord = corner_x,
            y_coord = corner_y,
            z_coord = PROCESS_Z,
        )
        corner_end_x, corner_end_y = project(
            x_coord = corner_x,
            y_coord = corner_y,
            z_coord = surface_height(x_coord = corner_x, y_coord = corner_y) - MANIFOLD_LAYER_OFFSET,
        )
        add_line(
            axis = ax,
            x_values = [corner_start_x, corner_end_x],
            y_values = [corner_start_y, corner_end_y],
            color = PROCESS_PINK_DARK,
            line_width = 0.95,
            line_style = (0.0, (1.0, 2.2)),
            alpha = 0.46,
            z_order = 6.28,
        )
    for obs_idx in range(len(process_anchor_x)):
        start_xy = np.array(object = [process_plane_x[obs_idx], process_plane_y[obs_idx]], dtype = float)
        end_xy = np.array(object = [observation_x[obs_idx], observation_y[obs_idx]], dtype = float)
        connector_vector = start_xy - end_xy
        connector_length = float(np.linalg.norm(x = connector_vector))
        connector_normal = np.array(object = [-connector_vector[1], connector_vector[0]], dtype = float)
        if connector_length > 0:
            connector_normal = connector_normal / connector_length
        connector_t = np.linspace(start = 0.0, stop = 1.0, num = 72)
        wave_frequency = 1.10 + 0.33 * float(obs_idx % 5)
        wave_phase = 0.83 * float(obs_idx)
        wave_amplitude = (0.115 + 0.018 * float((3 * obs_idx + 2) % 5)) * connector_length
        secondary_amplitude = (0.030 + 0.010 * float(obs_idx % 3)) * connector_length
        wave_envelope = np.sin(np.pi * connector_t)
        wave_offset = wave_envelope * (
            wave_amplitude * np.sin(2.0 * np.pi * wave_frequency * connector_t + wave_phase)
            + secondary_amplitude * np.sin(2.0 * np.pi * (wave_frequency + 1.65) * connector_t - 0.55 * wave_phase)
        )
        connector_xy = (
            end_xy[None, :]
            + connector_t[:, None] * connector_vector[None, :]
            + wave_offset[:, None] * connector_normal[None, :]
        )
        add_line(
            axis = ax,
            x_values = connector_xy[:, 0],
            y_values = connector_xy[:, 1],
            color = PROCESS_PINK_DARK,
            line_width = 1.12,
            line_style = (0.0, (1.0, 2.2)),
            alpha = 0.46,
            z_order = 7.3,
        )
        marker_start_xy = connector_xy[-5]
        marker_end_xy = start_xy
        observation_marker_arrow = FancyArrowPatch(
            posA = tuple(marker_start_xy),
            posB = tuple(marker_end_xy),
            arrowstyle = "-|>",
            mutation_scale = 5.6,
            color = PROCESS_PINK_DARK,
            linewidth = 1.12,
            linestyle = (0.0, (1.0, 2.2)),
            alpha = 0.46,
            zorder = 7.6,
        )
        ax.add_patch(p = observation_marker_arrow)
    ax.scatter(
        x = observation_x,
        y = observation_y,
        s = 28,
        c = PROCESS_PINK_DARK,
        marker = "o",
        edgecolors = "white",
        linewidths = 0.25,
        zorder = 8.2,
    )
    
    ## abstraction arrows
    process_arrow = FancyArrowPatch(
        posA = (-2.02, 4.50),
        posB = (-2.02, 3.00),
        arrowstyle = "simple",
        mutation_scale = 9,
        facecolor = PROCESS_PINK,
        edgecolor = "none",
        alpha = 0.80,
        zorder = 7.0,
    )
    ax.add_patch(p = process_arrow)
    ax.text(
        x = -2.11,
        y = 4.11,
        s = "Dynamics Abstraction",
        ha = "center",
        va = "center",
        fontsize = 8,
        rotation = -90,
        rotation_mode = "anchor",
        color = TEXT_PROCESS, 
    )
    structural_arrow = FancyArrowPatch(
        posA = (-2.02, 0.30),
        posB = (-2.02, 1.80),
        arrowstyle = "simple",
        mutation_scale = 9,
        facecolor = STRUCTURE_NAVY,
        edgecolor = "none",
        alpha = 0.78,
        zorder = 7.0,
    )
    ax.add_patch(p = structural_arrow)
    ax.text(
        x = -2.1,
        y = 0.7,
        s = "Structural Abstraction",
        ha = "center",
        va = "center",
        fontsize = 8,
        rotation = 90,
        rotation_mode = "anchor",
        color = TEXT_STRUCTURE,
    )
    ax.text(
        x = -2.125,
        y = 2.425,
        s = "Target Predictions",
        ha = "center",
        va = "center",
        fontsize = 8,
        rotation = 90,
        color = TEXT_GREEN,
    )
    ## layer labels
    ANN_X_LABEL_POS = 2.30
    
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 4.51,
        s = r"Dynamical Process:" + "\n" + r"$S$",
        ha = "left",
        va = "center",
        fontsize = 11.0,
        linespacing = 1.1,
        color = TEXT_PROCESS,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 4.0785,
        s = r"Signature Mapping:" + "\n" + r"$\psi(S) = z$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_PROCESS,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 3.647,
        s = r"Process Signatures:" + "\n" + r"$z = (z_1, \ldots, z_{7}) \in \mathbb{R}^{7}$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_PROCESS,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 3.06,
        s = r"Standardization:" + "\n" + r"$z \rightarrow z{\prime}$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_PROCESS,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 2.482,
        s = r"Learned Modulation:" + "\n" + r"$f_R(z{\prime})$",
        ha = "left",
        va = "center",
        fontsize = 11.0,
        linespacing = 1.1,
        color = TEXT_GREEN,
    )
    ax.scatter(
        x = [-2.01],
        y = [2.45],
        s = 50,
        c = FRONTIER_MARKER_GOLD,
        edgecolors = GREEN_DARK,
        linewidths = 1.5,
        zorder = 6.2,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 2.232,
        s = r"Capacity Frontier:" + "\n" + r"$F(x{\prime}, z{\prime}) = f_C(x{\prime}) + f_R(z{\prime})$",
        ha = "left",
        va = "center",
        fontsize = 12.0,
        linespacing = 1.3,
        fontfamily = "Arial",
        fontweight = "bold",
        color = TEXT_GREEN,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 1.982,
        s = r"Learned Manifold:" + "\n" + r"$f_C(x{\prime})$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_GREEN,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 1.48,
        s = r"Standardization:" + "\n" + r"$x \rightarrow x{\prime}$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_STRUCTURE,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 0.977,
        s = r"Graph Invariants:" + "\n" + r"$x = (x_1, \ldots, x_{21}) \in \mathbb{R}^{21}$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_STRUCTURE,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 0.4885,
        s = r"Invariant Mapping:" + "\n" + r"$\phi(G) = x$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_STRUCTURE,
    )
    ax.text(
        x = ANN_X_LABEL_POS,
        y = 0.0,
        s = r"Network Substrate:" + "\n" + r"$G = (V, E)$",
        ha = "left",
        va = "center",
        fontsize = 8,
        linespacing = 1.1,
        color = TEXT_STRUCTURE,
    )
    
    ## leader arrows: straight horizontal, small, uniform length, black; from each
    ## layer label to its element (one per row)
    ANN_ARROW_TAIL_X = 2.25
    ANN_ARROW_LENGTH = 0.30
    _layer_arrow_y = [4.51, 4.0785, 3.647, 3.06, 2.232, 1.48, 0.977, 0.4885, 0.00]
    for _ay in _layer_arrow_y:
        ax.annotate(
            "",
            xy = (ANN_ARROW_TAIL_X - ANN_ARROW_LENGTH, _ay),
            xytext = (ANN_ARROW_TAIL_X, _ay),
            arrowprops = dict(
                arrowstyle = "-|>",
                color = "#111111",
                linewidth = 0.6 * SCALE,  ## thinner than the dashed arrows; SCALE pre-applied since annotation arrows skip the normalization loop
                shrinkA = 0.0,
                shrinkB = 0.0,
                mutation_scale = 4,
            ),
            zorder = 8.0,
        )
    
    ## final typography and export
    for text_artist in fig.findobj(match = Text):
        text_artist.set_fontfamily(fontname = N3S_FONT_FAMILY)
        text_artist.set_fontname(fontname = N3S_FONT_FAMILY)
        text_artist.set_fontsize(N3S_FONT_SIZE)
    
    ## manual +/- labels for the offset manifolds
    MANIFOLD_OFFSET_LABEL_FONT_SIZE = 7.68  ## symbolic +/- -> 14 pt at final render (exempt from 7 pt body rule)
    MANIFOLD_OFFSET_LABEL_COLOR = PURPLE_DARK
    MANIFOLD_PLUS_LABEL_X = 1.6
    MANIFOLD_PLUS_LABEL_Y = 2.5
    MANIFOLD_MINUS_LABEL_X = 1.6
    MANIFOLD_MINUS_LABEL_Y = 2.0
    for label_text, label_x, label_y in [
        ("+", MANIFOLD_PLUS_LABEL_X, MANIFOLD_PLUS_LABEL_Y),
        ("-", MANIFOLD_MINUS_LABEL_X, MANIFOLD_MINUS_LABEL_Y),
    ]:
        ax.text(
            x = label_x,
            y = label_y,
            s = label_text,
            ha = "center",
            va = "center",
            fontsize = MANIFOLD_OFFSET_LABEL_FONT_SIZE,
            fontfamily = N3S_FONT_FAMILY,
            fontweight = "bold",
            color = MANIFOLD_OFFSET_LABEL_COLOR,
            alpha = 0.98,
            zorder = 10.0,
        )
    
    ## uniform style scaling for target width
    for line_artist in fig.findobj(match = Line2D):
        line_artist.set_linewidth(line_artist.get_linewidth() * SCALE)
        line_artist.set_markersize(line_artist.get_markersize() * SCALE)
    
    for collection_artist in fig.findobj(match = PathCollection):
        collection_artist.set_sizes(collection_artist.get_sizes() * (SCALE ** 2))
        collection_artist.set_linewidths(collection_artist.get_linewidths() * SCALE)
    
    for collection_artist in fig.findobj(match = LineCollection):
        collection_artist.set_linewidths(collection_artist.get_linewidths() * SCALE)
    
    for collection_artist in fig.findobj(match = PolyCollection):
        collection_artist.set_linewidths(collection_artist.get_linewidths() * SCALE)
    
    for patch_artist in fig.findobj(match = FancyArrowPatch):
        patch_artist.set_linewidth(patch_artist.get_linewidth() * SCALE)
        patch_artist.set_mutation_scale(patch_artist.get_mutation_scale() * SCALE)
    
    for patch_artist in fig.findobj(match = PathPatch):
        patch_artist.set_linewidth(patch_artist.get_linewidth() * SCALE)
    
    for patch_artist in fig.findobj(match = Polygon):
        patch_artist.set_linewidth(patch_artist.get_linewidth() * SCALE)
    
    fig.canvas.draw()
    crop_buffer = BytesIO()
    with mpl.rc_context(rc = {"savefig.bbox": None}):
        fig.savefig(
            fname = crop_buffer,
            format = "png",
            dpi = 300,
            facecolor = fig.get_facecolor(),
        )
    crop_buffer.seek(0)
    rendered_image = PILImage.open(fp = crop_buffer).convert(mode = "RGBA")
    image_array = np.asarray(a = rendered_image)
    background_mask = np.all(image_array[:, :, :3] >= 250, axis = 2)
    content_rows, content_cols = np.where(~background_mask)
    CROP_PAD_PIXELS = 12
    CROP_PAD_PIXELS_VERTICAL = 48  ## extra top/bottom whitespace
    crop_box = (
        max(int(content_cols.min()) - CROP_PAD_PIXELS, 0),
        max(int(content_rows.min()) - CROP_PAD_PIXELS_VERTICAL, 0),
        min(int(content_cols.max()) + CROP_PAD_PIXELS + 1, rendered_image.width),
        min(int(content_rows.max()) + CROP_PAD_PIXELS_VERTICAL + 1, rendered_image.height),
    )
    ## export vector PDF cropped to the same content extent as the raster
    ## (crop_box is in pixels, PIL top-left origin, at dpi 300 -> Bbox in inches)
    _pdf_dpi = 300
    _fig_h_px = rendered_image.height
    ## uniform margin matching figs 2-4 (~2.3 mm at final size, given fig1's x1.59 upscale)
    _pdf_pad_in = 0.057
    _pdf_pad_in_vertical = 0.16  ## extra top/bottom whitespace
    _ink_left = int(content_cols.min()) / _pdf_dpi
    _ink_right = (int(content_cols.max()) + 1) / _pdf_dpi
    _ink_bottom = (_fig_h_px - (int(content_rows.max()) + 1)) / _pdf_dpi
    _ink_top = (_fig_h_px - int(content_rows.min())) / _pdf_dpi
    _pdf_bbox = mpl.transforms.Bbox.from_extents(
        _ink_left - _pdf_pad_in,
        _ink_bottom - _pdf_pad_in_vertical,
        _ink_right + _pdf_pad_in,
        _ink_top + _pdf_pad_in_vertical,
    )
    FIGURE_DIR.mkdir(parents = True, exist_ok = True)
    pdf_path = export_pdf_scaled(fig, 1, target_width_mm = 183.0, bbox = _pdf_bbox)

    return fig, pdf_path
