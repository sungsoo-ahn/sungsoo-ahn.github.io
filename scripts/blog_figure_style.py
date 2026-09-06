"""Shared visual helpers for generated blog figures.

Generated blog figures are SVG-first.  Matplotlib scripts should save an
editable SVG plus a PNG preview, while sourced internet figures should keep
their original format and provenance.

Official palette: docs/palette.md and _data/palette.yml.

Publication sizing and text-audit portions are adapted from alphaXiv's
OpenResearch orx_figstyle.py (revision 95b2d961966c128b27f9d49753e227791a01c909).
Copyright (c) 2026 alphaXiv, MIT. License: .agents/third-party/alphaXiv-MIT.txt.
Local changes retain the site palette, accept custom widths, preserve legacy
exports, and treat audit results as findings for visual review.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from matplotlib.transforms import Bbox


TEXT = "#231533"
MUTED = "#665A75"
GRID = "#E4DDEC"
SPINE = "#C8BBD8"

PURPLE = "#7A53EC"
PURPLE_STRONG = "#5B3A8C"
PURPLE_SOFT = "#F6F2FA"
PURPLE_LIGHT = "#F2EDFF"

# Purple-led semantic palette.  Keep the older color names as public aliases
# because the figure scripts use them for semantic roles such as "blue curve"
# or "red loss"; the hues are tuned to sit cleanly beside the site purple.
BLUE = "#5F6ED8"
BLUE_LIGHT = "#EEF0FF"
AMBER = "#B7791F"
AMBER_LIGHT = "#FFF4DC"
TEAL = "#16877F"
TEAL_LIGHT = "#E4F4F2"
RED = "#C7506B"
RED_LIGHT = "#FCE8EE"
GREEN = "#3C8F63"
GREEN_LIGHT = "#E5F3EA"
VIOLET = PURPLE
VIOLET_DARK = PURPLE_STRONG
VIOLET_LIGHT = PURPLE_LIGHT
ROSE = RED
ROSE_LIGHT = RED_LIGHT
NEUTRAL = "#B9AEC7"

# Starting points, not venue requirements; use the destination's actual width.
COLUMN = 3.25
TEXT_WIDTH = 5.5
WIDE = 6.75


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
    save_kwargs = {"bbox_inches": "tight", "transparent": transparent}
    if not transparent:
        save_kwargs.update({"facecolor": fig.get_facecolor(), "edgecolor": fig.get_edgecolor()})
    fig.savefig(svg_path, format="svg", **save_kwargs)
    fig.savefig(png_path, format="png", dpi=dpi, **save_kwargs)
    plt.close(fig)
    print(f"Saved {svg_path} and {png_path}")
    return svg_path, png_path


def save_figure(fig, output_path, *, dpi: int = 300) -> None:
    """Backward-compatible save helper that now emits SVG plus PNG."""
    save_svg_png(fig, output_path, dpi=dpi)


def use_publication_style() -> None:
    """Use the site palette with text/lines sized for a printed column."""
    use_blog_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "axes.linewidth": 0.6,
            "savefig.bbox": None,
            "savefig.transparent": False,
        }
    )


@dataclass(frozen=True)
class FigureIssue:
    """A geometric finding; visual review decides whether it is a defect."""

    code: str
    message: str


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _hidden_text_ids(fig) -> set[int]:
    """Exclude text artists that exist in the tree but are not drawn."""
    hidden: set[int] = set()
    for ax in fig.axes:
        if not ax.get_visible():
            hidden.update(id(t) for t in ax.findobj(Text))
            continue
        legend = ax.get_legend()
        if legend is not None and not legend.get_visible():
            hidden.update(id(t) for t in legend.findobj(Text))
        for axis in (ax.xaxis, ax.yaxis):
            if not ax.axison or not axis.get_visible():
                hidden.update(id(t) for t in axis.findobj(Text))
                continue
            lo, hi = sorted(axis.get_view_interval())
            tolerance = (hi - lo) * 1e-6
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if not tick.get_visible() or not lo - tolerance <= tick.get_loc() <= hi + tolerance:
                    hidden.update((id(tick.label1), id(tick.label2)))
    for legend in fig.legends:
        if not legend.get_visible():
            hidden.update(id(t) for t in legend.findobj(Text))
    return hidden


def audit_figure(
    fig,
    *,
    target_width_in: float | None = None,
    min_font_pt: float = 7.0,
    ignore: Iterable[Text] = (),
) -> list[FigureIssue]:
    """Find potential small text, clipping, overlap, and target-width drift.

    Inspect a rendered Agg canvas, then restore the original canvas. Shared
    axes, colorbars, diagrams without axes, and custom widths are valid.
    Pass specific Text artists in ``ignore`` for documented intentional
    annotations. Bounding boxes are conservative, especially for rotated text;
    this audit cannot replace visual review or detect text over data marks.
    """
    _positive_finite(min_font_pt, "min_font_pt")
    issues: list[FigureIssue] = []
    if target_width_in is not None:
        _positive_finite(target_width_in, "target_width_in")
        actual_width = float(fig.get_size_inches()[0])
        if not math.isclose(actual_width, target_width_in, rel_tol=0, abs_tol=1e-6):
            issues.append(FigureIssue("width", f"Canvas is {actual_width:g}in; target is {target_width_in:g}in."))
    if not fig.get_visible():
        return issues

    original_canvas = fig.canvas
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        hidden = _hidden_text_ids(fig) | {id(t) for t in ignore}
        boxes: list[tuple[Text, Bbox]] = []
        tolerance = fig.dpi * 0.8 / 72
        for artist in fig.findobj(Text):
            if id(artist) in hidden or not artist.get_visible() or not artist.get_text().strip():
                continue
            # Annotation.get_window_extent includes its arrow. Compare the text
            # itself so a legitimate arrow does not become a text collision.
            box = Text.get_window_extent(artist, renderer=renderer)
            if not all(math.isfinite(float(v)) for v in box.extents) or box.width <= 0 or box.height <= 0:
                continue
            bounds = fig.bbox
            if artist.get_clip_on() and artist.get_clip_box() is not None:
                clip_box = Bbox.intersection(bounds, artist.get_clip_box())
                if clip_box is None or Bbox.intersection(box, clip_box) is None:
                    continue
                bounds = clip_box
            label = repr(artist.get_text()[:60])
            if artist.get_fontsize() < min_font_pt:
                issues.append(FigureIssue("small-text", f"{label} is {artist.get_fontsize():g}pt; target floor is {min_font_pt:g}pt."))
            if (box.x0 < bounds.x0 - tolerance or box.x1 > bounds.x1 + tolerance
                    or box.y0 < bounds.y0 - tolerance or box.y1 > bounds.y1 + tolerance):
                issues.append(FigureIssue("clipped-text", f"{label} extends outside its visible bounds."))
            boxes.append((artist, box))

        for i, (first, first_box) in enumerate(boxes):
            for second, second_box in boxes[i + 1:]:
                overlap = Bbox.intersection(first_box, second_box)
                if overlap is not None and overlap.width > tolerance and overlap.height > tolerance:
                    issues.append(FigureIssue("text-overlap", f"{first.get_text()[:40]!r} overlaps {second.get_text()[:40]!r}."))
    finally:
        fig.set_canvas(original_canvas)
    return issues


def save_publication_figure(
    fig,
    output_path,
    *,
    target_width_in: float,
    dpi: int = 300,
    close: bool = True,
) -> tuple[Path, Path, Path]:
    """Export fixed-size PDF, editable SVG, and PNG without cropping/resizing.

    The declared width must match the canvas. Audit findings are printed for
    review, not raised as style errors. Existing web save helpers are unchanged.
    """
    _positive_finite(target_width_in, "target_width_in")
    _positive_finite(dpi, "dpi")
    if not math.isclose(float(fig.get_size_inches()[0]), target_width_in, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Build the figure at target_width_in before exporting; export does not rescale it.")
    issues = audit_figure(fig, target_width_in=target_width_in)
    stem = Path(output_path)
    if stem.suffix.lower() in {".pdf", ".svg", ".png"}:
        stem = stem.with_suffix("")
    paths = tuple(Path(f"{stem}.{ext}") for ext in ("pdf", "svg", "png"))
    stem.parent.mkdir(parents=True, exist_ok=True)
    layout_engine = fig.get_layout_engine()
    try:
        # Freeze the layout measured above so export backends use the same
        # panel positions. Reset bbox rcParams too: bbox_inches=None alone
        # would still inherit a global savefig.bbox='tight'.
        fig.set_layout_engine("none")
        with mpl.rc_context({"savefig.bbox": None, "pdf.fonttype": 42,
                             "ps.fonttype": 42, "svg.fonttype": "none"}):
            for path in paths:
                fig.savefig(path, format=path.suffix[1:], dpi=dpi, bbox_inches=None,
                            transparent=False, facecolor=fig.get_facecolor(),
                            edgecolor=fig.get_edgecolor())
    finally:
        fig.set_layout_engine(layout_engine)
    if close:
        plt.close(fig)
    for issue in issues:
        print(f"Figure audit [{issue.code}]: {issue.message}", file=sys.stderr)
    print("Saved " + ", ".join(str(path) for path in paths))
    return paths
