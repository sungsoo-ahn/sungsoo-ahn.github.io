"""
Generate figures for the adsorption / GCMC / classical DFT blog post.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


TEXT_COLOR = "#263238"
ARROW_COLOR = "#455a64"

BOX_MAIN = "#dce8f4"
EDGE_MAIN = "#5b7fa5"
BOX_HIGHLIGHT = "#fff3e0"
EDGE_HIGHLIGHT = "#e8a030"
BOX_OUTPUT = "#e0f2e9"
EDGE_OUTPUT = "#4caf50"
BOX_CORRECTION = "#ede7f6"
EDGE_CORRECTION = "#7e57c2"

COLOR_ACCEPT = "#4caf50"
COLOR_REJECT = "#d32f2f"
COLOR_FRAMEWORK = "#455a64"
COLOR_METHANE = "#e8a030"
COLOR_DENSITY = "#5b7fa5"


def _clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _arrow(ax, start, end, color=ARROW_COLOR, lw=1.7, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arrow)


def _rounded_box(ax, xy, width, height, label, fc, ec, fontsize=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.7,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontweight="bold",
        zorder=4,
    )
    return box


def _draw_framework_cell(ax):
    ax.add_patch(Rectangle((0.1, 0.18), 0.8, 0.62, facecolor="white", edgecolor=EDGE_MAIN, linewidth=1.7))
    for x, y in [(0.2, 0.25), (0.82, 0.31), (0.2, 0.7), (0.82, 0.68), (0.5, 0.49)]:
        ax.add_patch(Circle((x, y), 0.035, color=COLOR_FRAMEWORK, zorder=2))


def generate_gcmc_moves_figure(output_path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = [
        ("translation", "move an existing molecule"),
        ("rotation", "change orientation"),
        ("insertion", "try a new molecule"),
        ("deletion", "remove a molecule"),
    ]

    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _clean_axis(ax)
        _draw_framework_cell(ax)

        molecules = [(0.38, 0.38), (0.65, 0.58), (0.56, 0.32)]
        for x, y in molecules:
            ax.add_patch(Circle((x, y), 0.035, color=COLOR_METHANE, zorder=3))

        if idx == 0:
            _arrow(ax, (0.38, 0.38), (0.5, 0.52), color=COLOR_ACCEPT)
            ax.add_patch(Circle((0.5, 0.52), 0.035, facecolor="none", edgecolor=COLOR_ACCEPT, linewidth=2.0, zorder=5))
        elif idx == 1:
            ax.plot([0.6, 0.72], [0.58, 0.58], color=COLOR_METHANE, linewidth=4, solid_capstyle="round", zorder=3)
            ax.plot([0.6, 0.7], [0.48, 0.68], color=COLOR_ACCEPT, linewidth=2, zorder=4)
            ax.text(0.72, 0.69, r"$\Omega$", fontsize=12, color=COLOR_ACCEPT, fontweight="bold")
        elif idx == 2:
            ax.add_patch(Circle((0.28, 0.58), 0.035, facecolor="none", edgecolor=COLOR_ACCEPT, linewidth=2.0, zorder=5))
            _arrow(ax, (0.28, 0.88), (0.28, 0.63), color=COLOR_ACCEPT)
        else:
            ax.add_patch(Circle((0.65, 0.58), 0.055, facecolor="none", edgecolor=COLOR_REJECT, linewidth=2.0, zorder=5))
            _arrow(ax, (0.65, 0.58), (0.65, 0.88), color=COLOR_REJECT)

        title, subtitle = panels[idx]
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax.text(0.5, 0.04, subtitle, ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)

    fig.suptitle("GCMC move set in a variable-N state space", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved GCMC moves figure to {output_path}")


def generate_snapshots_to_density_figure(output_path):
    fig = plt.figure(figsize=(11.8, 4.8))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.35, 2.25], wspace=0.24, hspace=0.2)
    rng = np.random.default_rng(8)
    all_points = []

    for idx in range(4):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _clean_axis(ax)
        ax.add_patch(Rectangle((0.08, 0.08), 0.84, 0.84, facecolor="white", edgecolor="#b0bec5", linewidth=1.2))

        n_points = [7, 12, 9, 15][idx]
        cluster_a = rng.normal([0.32, 0.68], [0.08, 0.08], (n_points // 2, 2))
        cluster_b = rng.normal([0.68, 0.34], [0.1, 0.08], (n_points - n_points // 2, 2))
        points = np.clip(np.vstack([cluster_a, cluster_b]), 0.13, 0.87)
        all_points.append(points)

        for x, y in points:
            ax.add_patch(Circle((x, y), 0.022, color=COLOR_METHANE, alpha=0.9))
        ax.text(0.5, 0.98, f"sample {idx + 1}", ha="center", va="top", fontsize=9, color=TEXT_COLOR)

    arrow_axis = fig.add_subplot(grid[:, 2])
    arrow_axis.set_xlim(0, 1)
    arrow_axis.set_ylim(0, 1)
    _clean_axis(arrow_axis)
    _arrow(arrow_axis, (0.1, 0.5), (0.95, 0.5), color=EDGE_MAIN, lw=2.4)

    density_axis = fig.add_subplot(grid[:, 3])
    _clean_axis(density_axis)
    points = np.vstack(all_points)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=45, range=[[0, 1], [0, 1]])
    for _ in range(5):
        hist = (
            hist
            + np.roll(hist, 1, axis=0)
            + np.roll(hist, -1, axis=0)
            + np.roll(hist, 1, axis=1)
            + np.roll(hist, -1, axis=1)
        ) / 5

    density_axis.imshow(hist.T, origin="lower", extent=(0, 1, 0, 1), cmap="YlGnBu")
    density_axis.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=EDGE_MAIN, linewidth=1.8))
    density_axis.text(
        0.5,
        1.04,
        r"coarse-grained density $\rho(\mathbf{r})$",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    density_axis.text(
        0.5,
        -0.08,
        r"integrate grid $\rightarrow$ uptake $\langle N\rangle$",
        ha="center",
        va="top",
        fontsize=10.5,
        color=TEXT_COLOR,
    )

    fig.suptitle("From particle snapshots to a density field", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.99)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved snapshots-to-density figure to {output_path}")


def generate_cdft_fixed_point_figure(output_path):
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.0)
    _clean_axis(ax)

    _rounded_box(ax, (0.65, 2.78), 2.05, 0.78, "current density\n" + r"$\rho^{(n)}$", BOX_MAIN, EDGE_MAIN)
    _rounded_box(
        ax,
        (3.55, 2.78),
        2.25,
        0.78,
        "evaluate correction\n" + r"$\delta F_{exc}/\delta\rho$",
        BOX_CORRECTION,
        EDGE_CORRECTION,
        fontsize=10.5,
    )
    _rounded_box(
        ax,
        (6.6, 2.78),
        2.25,
        0.78,
        "Boltzmann update\nwith many-body term",
        BOX_HIGHLIGHT,
        EDGE_HIGHLIGHT,
        fontsize=10.5,
    )
    _rounded_box(ax, (6.6, 1.15), 2.25, 0.78, "new density\n" + r"$\rho^{(n+1)}$", BOX_MAIN, EDGE_MAIN)
    _rounded_box(ax, (3.55, 1.15), 2.25, 0.78, "converged?", BOX_OUTPUT, EDGE_OUTPUT)

    _arrow(ax, (2.78, 3.17), (3.25, 3.17))
    _arrow(ax, (5.85, 3.17), (6.18, 3.17))
    _arrow(ax, (7.73, 2.70), (7.73, 2.03))
    _arrow(ax, (6.08, 1.54), (5.85, 1.54))
    _arrow(ax, (3.25, 1.54), (2.15, 2.60), color=COLOR_REJECT, rad=0.2)
    ax.text(2.62, 2.10, "No", color=COLOR_REJECT, fontsize=10.5, fontweight="bold")

    _arrow(ax, (5.80, 1.16), (9.0, 0.72), color=COLOR_ACCEPT)
    _rounded_box(ax, (9.35, 0.38), 1.55, 0.68, r"$\rho_{eq}$", "white", EDGE_OUTPUT, fontsize=13)
    ax.text(7.15, 0.98, "Yes", color=COLOR_ACCEPT, fontsize=10.5, fontweight="bold")

    ax.text(
        5.6,
        4.45,
        "same rhythm as quantum-DFT SCF, different object",
        ha="center",
        fontsize=11.5,
        color=TEXT_COLOR,
    )
    fig.suptitle("The cDFT fixed-point loop", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved cDFT fixed-point figure to {output_path}")


if __name__ == "__main__":
    output_dir = Path("assets/img/blog")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_gcmc_moves_figure(output_dir / "adsorption_gcmc_cdft_moves.png")
    generate_snapshots_to_density_figure(output_dir / "adsorption_gcmc_cdft_snapshots_to_density.png")
    generate_cdft_fixed_point_figure(output_dir / "adsorption_gcmc_cdft_fixed_point.png")

    print("Done!")
