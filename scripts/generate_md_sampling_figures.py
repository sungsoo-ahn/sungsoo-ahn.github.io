"""Generate figures for the molecular dynamics enhanced sampling blog post."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


TEXT = "#263238"
BLUE = "#5b7fa5"
BLUE_LIGHT = "#dce8f4"
AMBER = "#e8a030"
AMBER_LIGHT = "#fff3e0"
TEAL = "#1a8a7a"
TEAL_LIGHT = "#e0f2f1"
RED = "#c0503f"
RED_LIGHT = "#fce4ec"
GREEN = "#4caf50"
GREEN_LIGHT = "#e0f2e9"
NEUTRAL = "#b0bec5"

OUTPUT_DIR = Path("assets/img/blog")


def _style_axis(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=12, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=12, color=TEXT)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=8)
    ax.tick_params(colors="#78909c", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL)
    ax.spines["bottom"].set_color(NEUTRAL)


def _double_well(x):
    return 1.15 * (x**2 - 1.0) ** 2 + 0.12 * x


def _arrow(ax, start, end, color=TEXT, lw=1.8, ms=14, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        zorder=5,
    )
    ax.add_patch(patch)


def _box(ax, xy, width, height, text, fc, ec, fontsize=10.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.7,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        zorder=4,
    )


def generate_metastability(output_path: Path):
    """Rare events in a double-well landscape."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.32})
    x = np.linspace(-2.1, 2.1, 600)
    u = _double_well(x)

    ax = axes[0]
    ax.plot(x, u, color=BLUE, linewidth=2.6)
    ax.fill_between(x, 0, u, color=BLUE_LIGHT, alpha=0.45)
    ax.axvspan(-1.45, -0.45, color=BLUE_LIGHT, alpha=0.6)
    ax.axvspan(0.55, 1.55, color=GREEN_LIGHT, alpha=0.65)
    ax.plot([-0.92], [_double_well(np.array([-0.92]))[0] + 0.08], "o", color=BLUE, markersize=9)
    ax.plot([0.93], [_double_well(np.array([0.93]))[0] + 0.08], "o", color=GREEN, markersize=9)
    ax.plot([0.0], [_double_well(np.array([0.0]))[0] + 0.08], "*", color=AMBER, markersize=13)
    ax.annotate(
        "high barrier",
        xy=(0.0, _double_well(np.array([0.0]))[0] + 0.08),
        xytext=(-0.65, 1.95),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
        fontsize=10.5,
        color=TEXT,
        ha="center",
    )
    _arrow(ax, (-0.85, 0.35), (-0.25, 0.95), color=RED, lw=2.0)
    _arrow(ax, (-0.25, 0.95), (-0.78, 0.36), color=RED, lw=2.0)
    ax.text(-1.02, 0.12, "metastable\nstate A", ha="center", va="bottom", fontsize=10, color=BLUE)
    ax.text(1.02, 0.12, "state B", ha="center", va="bottom", fontsize=10, color=GREEN)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="reaction coordinate", ylabel="free energy", title="Unbiased MD gets trapped")

    ax = axes[1]
    ax.plot(x, u, color=BLUE, linewidth=2.0, alpha=0.8, label="original")
    bias = -0.82 * np.exp(-0.5 * ((x - 0.0) / 0.55) ** 2)
    ub = u + bias
    ax.plot(x, ub, color=TEAL, linewidth=2.7, label="biased")
    ax.fill_between(x, ub, u, where=bias < 0, color=TEAL_LIGHT, alpha=0.75)
    path_x = np.linspace(-0.93, 0.95, 70)
    path_y = np.interp(path_x, x, ub) + 0.12 + 0.03 * np.sin(np.linspace(0, 5 * np.pi, 70))
    ax.plot(path_x, path_y, color=RED, linewidth=2.1)
    for i in [12, 28, 44, 58]:
        _arrow(ax, (path_x[i - 2], path_y[i - 2]), (path_x[i + 2], path_y[i + 2]), color=RED, lw=1.7, ms=12)
    ax.text(0.0, 1.65, "bias lowers the\nsampling barrier", ha="center", fontsize=10.5, color=TEAL)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="reaction coordinate", ylabel="", title="Enhanced sampling changes what is easy")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved metastability to {output_path}")


def generate_cv_metadynamics(output_path: Path):
    """Collective variables and metadynamics biasing."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw={"wspace": 0.36})

    ax = axes[0]
    rng = np.random.default_rng(7)
    mean_a = np.array([-1.0, -0.45])
    mean_b = np.array([1.05, 0.55])
    cloud_a = rng.normal(mean_a, [0.32, 0.23], size=(80, 2))
    cloud_b = rng.normal(mean_b, [0.34, 0.25], size=(80, 2))
    ax.scatter(cloud_a[:, 0], cloud_a[:, 1], s=18, color=BLUE, alpha=0.72)
    ax.scatter(cloud_b[:, 0], cloud_b[:, 1], s=18, color=GREEN, alpha=0.72)
    ax.plot([-1.7, 1.7], [-0.95, 0.95], color=AMBER, linewidth=2.2)
    _arrow(ax, (-1.45, -0.82), (1.45, 0.82), color=AMBER, lw=2.0)
    ax.text(0.0, 1.05, r"CV $s(\mathbf{x})$", ha="center", fontsize=11, color=AMBER, fontweight="bold")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="coordinates", ylabel="", title="Compress configurations")

    ax = axes[1]
    s = np.linspace(-2.0, 2.0, 600)
    f = 0.95 * (s**2 - 1.0) ** 2 + 0.08 * s
    ax.plot(s, f, color=BLUE, linewidth=2.5)
    ax.fill_between(s, 0, f, color=BLUE_LIGHT, alpha=0.45)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    _style_axis(ax, xlabel=r"CV $s$", ylabel=r"$F(s)$", title="Estimate a free-energy profile")
    ax.text(0.02, 1.22, r"$F(s)=-\beta^{-1}\log p(s)$", ha="center", fontsize=10.5, color=TEXT)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(s, f, color=BLUE, linewidth=2.0, alpha=0.65, label="free energy")
    centers = np.array([-1.0, -0.72, -0.35, 0.02, 0.38])
    heights = np.linspace(0.22, 0.11, len(centers))
    bias = np.zeros_like(s)
    for c, h in zip(centers, heights):
        g = h * np.exp(-0.5 * ((s - c) / 0.18) ** 2)
        bias += g
        ax.fill_between(s, 0, g, color=AMBER_LIGHT, alpha=0.9)
        ax.plot(s, g, color=AMBER, linewidth=1.2, alpha=0.9)
    effective = f + bias
    ax.plot(s, effective, color=TEAL, linewidth=2.5, label="biased surface")
    ax.text(-0.65, 1.18, "metadynamics deposits\nhistory-dependent hills", ha="center", fontsize=10, color=AMBER)
    _arrow(ax, (-0.65, 1.05), (-0.75, 0.34), color=AMBER, lw=1.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    ax.set_yticks([])
    _style_axis(ax, xlabel=r"CV $s$", ylabel="", title="Bias along the CV")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved cv metadynamics to {output_path}")


def generate_method_map(output_path: Path):
    """Map classical enhanced sampling and ML methods."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.05, 0.70), 0.22, 0.15, "Unbiased MD\nphysical dynamics\nrare transitions", BLUE_LIGHT, BLUE)
    _box(ax, (0.39, 0.70), 0.22, 0.15, "CV-based biasing\numbrella, metadynamics,\nOPES, SMD", AMBER_LIGHT, AMBER)
    _box(ax, (0.73, 0.70), 0.22, 0.15, "Free energies\nreweighting recovers\nunbiased statistics", GREEN_LIGHT, GREEN)

    _box(ax, (0.22, 0.36), 0.25, 0.15, "Learn the CV\nBioEmu-CV:\ntime-lagged slow modes", TEAL_LIGHT, TEAL)
    _box(ax, (0.55, 0.36), 0.25, 0.15, "Learn the path bias\nTPS-DPS:\nCV-free path sampling", RED_LIGHT, RED)

    _box(ax, (0.30, 0.08), 0.40, 0.13, "Path-measure view\nJarzynski, AIS, diffusion models,\ntrajectory objectives", "white", NEUTRAL, fontsize=10.3)

    _arrow(ax, (0.27, 0.775), (0.39, 0.775), color=TEXT)
    _arrow(ax, (0.61, 0.775), (0.73, 0.775), color=TEXT)
    _arrow(ax, (0.34, 0.51), (0.45, 0.70), color=TEAL)
    _arrow(ax, (0.67, 0.51), (0.55, 0.70), color=RED)
    _arrow(ax, (0.43, 0.36), (0.43, 0.21), color=NEUTRAL, lw=1.5)
    _arrow(ax, (0.68, 0.36), (0.60, 0.21), color=NEUTRAL, lw=1.5)
    _arrow(ax, (0.86, 0.70), (0.70, 0.21), color=GREEN, lw=1.5)

    ax.text(0.33, 0.82, "add a bias", ha="center", fontsize=9.5, color=TEXT)
    ax.text(0.67, 0.82, "undo the bias", ha="center", fontsize=9.5, color=TEXT)
    ax.text(0.16, 0.60, "classical problem:\nwhere should the bias act?", ha="center", fontsize=10.0, color=TEXT)
    ax.text(0.50, 0.94, "Enhanced sampling is controlled distribution shift", ha="center", fontsize=14, color=TEXT, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved method map to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_metastability(OUTPUT_DIR / "md_metastability_bias.png")
    generate_cv_metadynamics(OUTPUT_DIR / "md_cv_metadynamics.png")
    generate_method_map(OUTPUT_DIR / "md_sampling_method_map.png")


if __name__ == "__main__":
    main()
