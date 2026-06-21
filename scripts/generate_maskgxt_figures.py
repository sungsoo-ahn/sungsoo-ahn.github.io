#!/usr/bin/env python3
"""Generate figures for the MaskGXT / AI co-scientist blog post."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import blog_figure_style as bfs


OUTPUT = "assets/img/blog/maskgxt_results_bars.svg"
THEME_PURPLE = "#7a53ec"
BASELINE_GRAY = "#c8d2d7"


# Data redrawn from Tables 1 and 2 of:
# Seong et al., "Human-AI Co-Discovery of a State-of-the-Art Crystal
# Structure Prediction Algorithm" (main paper PDF, 2026-06-21 draft).
MODELS = ["DiffCSP", "FlowMM", "OMatG", "MCFlow", "Crystalite", "MaskGXT"]
MODEL_LABELS = [
    "DiffCSP (2023)",
    "FlowMM (2024)",
    "OMatG (2025)",
    "MCFlow (2026)",
    "Crystalite (2026)",
    "MaskGXT (2026)",
]

TABLE1_FILTERED = {
    "MP-20\nMR (%) ↑": [52.51, 59.98, 63.75, 63.14, 66.05, 67.06],
    "MP-20\nRMSE ↓": [0.0600, 0.0629, 0.0720, 0.0611, 0.0329, 0.0325],
    "MPTS-52\nMR (%) ↑": [14.29, 20.28, 25.15, 26.46, 31.49, 33.34],
    "MPTS-52\nRMSE ↓": [0.1489, 0.1486, 0.1931, 0.1577, 0.0701, 0.0975],
}

TABLE2_METRE = {
    "MP-20\nMETRe (%) ↑": [58.80, 67.00, 66.00, 69.70, 70.45, 74.78],
    "MP-20\ncRMSE ↓": [0.244, 0.210, 0.208, 0.200, 0.178, 0.152],
    "Polymorph split\nMETRe (%) ↑": [53.14, 65.18, 70.50, 70.70, 70.87, 79.06],
    "Polymorph split\ncRMSE ↓": [0.279, 0.226, 0.187, 0.195, 0.174, 0.132],
}


def _format_value(value: float, title: str) -> str:
    if "MR" in title or "METRe" in title:
        return f"{value:.1f}"
    if "cRMSE" in title:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _panel_xlim(values: list[float], title: str) -> tuple[float, float]:
    vmax = max(values)
    if "MR" in title or "METRe" in title:
        return 0.0, max(40.0, np.ceil(vmax / 10.0) * 10.0 + 5.0)
    return 0.0, vmax * 1.34


def _bar_colors() -> list[str]:
    maskgxt_idx = MODELS.index("MaskGXT")
    return [THEME_PURPLE if i == maskgxt_idx else BASELINE_GRAY for i, _ in enumerate(MODELS)]


def _draw_panel(ax, title: str, values: list[float], *, show_models: bool) -> None:
    y = np.arange(len(MODELS))
    colors = _bar_colors()

    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.8, height=0.72)
    ax.set_title(title, loc="left", pad=7, fontsize=11.2, fontweight="semibold")
    ax.set_yticks(y)
    if show_models:
        ax.set_yticklabels(MODEL_LABELS)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(*_panel_xlim(values, title))

    xmin, xmax = ax.get_xlim()
    label_dx = 0.018 * (xmax - xmin)
    for yi, value in zip(y, values):
        ax.text(
            value + label_dx,
            yi,
            _format_value(value, title),
            ha="left",
            va="center",
            fontsize=8.4,
            color=bfs.TEXT,
        )

    ax.xaxis.grid(True, color=bfs.GRID, linewidth=0.8, alpha=0.55)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    bfs.clean_axes(ax, grid=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)


def make_results_bars() -> None:
    bfs.use_blog_style()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12.6, 7.0),
        gridspec_kw={"left": 0.13, "right": 0.99, "top": 0.83, "bottom": 0.115, "wspace": 0.32, "hspace": 0.62},
    )

    for col, (title, values) in enumerate(TABLE1_FILTERED.items()):
        _draw_panel(axes[0, col], title, values, show_models=(col == 0))

    for col, (title, values) in enumerate(TABLE2_METRE.items()):
        _draw_panel(axes[1, col], title, values, show_models=(col == 0))

    fig.text(
        0.13,
        0.965,
        "MaskGXT benchmark results",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=bfs.TEXT,
    )
    legend = [
        Patch(facecolor=THEME_PURPLE, edgecolor="none", label="MaskGXT"),
        Patch(facecolor=BASELINE_GRAY, edgecolor="none", label="baselines"),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.13, 0.02),
        ncol=3,
        frameon=False,
        fontsize=9.4,
        handlelength=1.5,
        columnspacing=1.8,
    )

    bfs.save_svg_png(fig, OUTPUT, transparent=False)
    svg_path = OUTPUT
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


if __name__ == "__main__":
    make_results_bars()
