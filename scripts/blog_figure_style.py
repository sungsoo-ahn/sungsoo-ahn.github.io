"""Shared visual helpers for generated blog figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


TEXT = "#263238"
MUTED = "#6f7f86"
GRID = "#d8e0e4"
SPINE = "#b6c2c8"

BLUE = "#4f7ea8"
BLUE_LIGHT = "#e6eef6"
AMBER = "#d99a24"
AMBER_LIGHT = "#fff4df"
TEAL = "#158574"
TEAL_LIGHT = "#e2f3ef"
RED = "#c85b4a"
RED_LIGHT = "#fae8e4"
GREEN = "#3f9b5f"
GREEN_LIGHT = "#e4f3e8"
VIOLET = "#7d6aa8"
VIOLET_LIGHT = "#eeeaf5"
NEUTRAL = "#aebbc2"


def use_blog_style() -> None:
    """Apply restrained defaults for editorial blog figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def style_axis(ax, xlabel: str = "", ylabel: str = "", title: str = "", *, grid: bool = False) -> None:
    """Use quiet axes with minimal frame weight."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=8, fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(length=3, width=0.8, colors=MUTED)
    if grid:
        ax.grid(color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.02,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
        color=MUTED,
    )


def direct_label(ax, x: float, y: float, text: str, color: str, *, size: float = 9.5, ha: str = "center") -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight="semibold",
        bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 1.5},
        zorder=30,
    )


def save_figure(fig, output_path, *, dpi: int = 220) -> None:
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")
