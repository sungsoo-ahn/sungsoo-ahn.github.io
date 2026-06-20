"""Shared visual helpers for generated blog figures.

Generated blog figures are SVG-first.  Matplotlib scripts should save an
editable SVG plus a PNG preview, while sourced internet figures should keep
their original format and provenance.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

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
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "lines.markersize": 4.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": True,
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax, *, grid: bool = False) -> None:
    """Remove nonessential chart furniture."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(length=3, width=0.8, colors=MUTED)
    ax.grid(False)
    ax.margins(x=0.02)
    if grid:
        ax.grid(color=GRID, linewidth=0.8, alpha=0.55)
        ax.set_axisbelow(True)


def style_axis(ax, xlabel: str = "", ylabel: str = "", title: str = "", *, grid: bool = False) -> None:
    """Use quiet axes with minimal frame weight."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=8, fontweight="semibold")
    clean_axes(ax, grid=grid)


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


def label_box(*, alpha: float = 0.88, pad: float = 1.6) -> dict:
    """White backing for labels placed near plotted marks."""
    return {"fc": "white", "ec": "none", "alpha": alpha, "pad": pad}


def direct_label(ax, x: float, y: float, text: str, color: str, *, size: float = 9.5, ha: str = "center"):
    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight="semibold",
        bbox=label_box(alpha=0.84, pad=1.5),
        zorder=30,
    )


def curve_label(ax, x: float, y: float, text: str, color: str, *, size: float = 9.0, ha: str = "left"):
    """Label a curve directly, with enough backing to stay legible over fills."""
    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight="semibold",
        bbox=label_box(alpha=0.9, pad=1.4),
        zorder=35,
    )


def callout_label(
    ax,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    color: str,
    *,
    size: float = 9.5,
    ha: str = "center",
    rad: float = 0.0,
):
    """Place an annotation away from the data and point back to the feature."""
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight="semibold",
        bbox=label_box(alpha=0.92, pad=1.8),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.4,
            "mutation_scale": 12,
            "shrinkA": 4,
            "shrinkB": 5,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=40,
    )


def state_marker(
    ax,
    x: float,
    y: float,
    label: str,
    color: str,
    *,
    label_dx: float = 0.16,
    label_dy: float = 0.18,
    marker_size: float = 86,
):
    """Draw a metastable-state marker with a separate label."""
    ax.scatter([x], [y], s=marker_size * 1.7, color="white", edgecolors="none", zorder=20)
    ax.scatter([x], [y], s=marker_size, color=color, edgecolors="white", linewidths=1.5, zorder=22)
    return ax.text(
        x + label_dx,
        y + label_dy,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="semibold",
        color=color,
        bbox=label_box(alpha=0.9, pad=1.3),
        zorder=36,
    )


def _as_paths(output_path) -> tuple[Path, Path]:
    path = Path(output_path)
    if path.suffix.lower() == ".svg":
        return path, path.with_suffix(".png")
    return path.with_suffix(".svg"), path.with_suffix(".png")


def render_svg_preview(svg_path, png_path, *, width: int | None = None) -> bool:
    """Render an SVG to PNG for inspection with local command-line tools."""
    svg_path = Path(svg_path)
    png_path = Path(png_path)
    if shutil.which("rsvg-convert"):
        cmd = ["rsvg-convert", str(svg_path), "-o", str(png_path)]
        if width is not None:
            cmd.extend(["--width", str(width)])
        subprocess.run(cmd, check=True)
        return True
    if shutil.which("magick"):
        cmd = ["magick", str(svg_path)]
        if width is not None:
            cmd.extend(["-resize", f"{width}x"])
        cmd.append(str(png_path))
        subprocess.run(cmd, check=True)
        return True
    return False


def save_svg_png(fig, output_path, *, dpi: int = 300, transparent: bool = True) -> tuple[Path, Path]:
    """Save an editable SVG and a PNG preview with the same stem."""
    svg_path, png_path = _as_paths(output_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", transparent=transparent)
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print(f"Saved {svg_path} and {png_path}")
    return svg_path, png_path


def save_figure(fig, output_path, *, dpi: int = 300) -> None:
    """Backward-compatible save helper that now emits SVG plus PNG."""
    save_svg_png(fig, output_path, dpi=dpi)
