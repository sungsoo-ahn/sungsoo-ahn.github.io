"""
Generate figures for the adsorption / GCMC / classical DFT blog post.
"""

from html import escape
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import blog_figure_style as bfs

bfs.use_blog_style()


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


class Svg:
    """Small native-SVG helper for explanatory diagrams."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 18,
        weight: int | str = 500,
        fill: str = TEXT_COLOR,
        anchor: str = "middle",
        lines: list[str] | None = None,
        italic: bool = False,
    ) -> None:
        items = lines if lines is not None else value.split("\n")
        line_height = size * 1.22
        start = y - line_height * (len(items) - 1) / 2
        tspans = []
        for i, item in enumerate(items):
            tspans.append(
                f'<tspan x="{x:.1f}" y="{start + i * line_height:.1f}">{escape(item)}</tspan>'
            )
        style = "font-style:italic;" if italic else ""
        self.add(
            f'<text text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" style="{style}">{"".join(tspans)}</text>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "white",
        stroke: str = "#d8e0e4",
        sw: float = 1.6,
        rx: float = 10,
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" '
            f'opacity="{opacity:.2f}"/>'
        )

    def circle(self, x: float, y: float, r: float, *, fill: str, stroke: str = "white", sw: float = 1.0, opacity: float = 1.0) -> None:
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity:.2f}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = ARROW_COLOR, sw: float = 2.0, dashed: bool = False, marker: str | None = None, opacity: float = 1.0) -> None:
        dash = ' stroke-dasharray="6 6"' if dashed else ""
        marker_attr = f' marker-end="url(#{marker})"' if marker else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw:.1f}" stroke-linecap="round"{dash}{marker_attr} '
            f'opacity="{opacity:.2f}"/>'
        )

    def svg(self) -> str:
        defs = (
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_COLOR}"/></marker>'
            '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{EDGE_MAIN}"/></marker>'
        )
        style = (
            "<style>"
            "text{font-family:Arial,Helvetica,'DejaVu Sans',sans-serif;dominant-baseline:middle}"
            ".caption{fill:#6f7f86;font-size:15px;font-weight:500}"
            "</style>"
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}"><rect width="100%" height="100%" fill="white"/>'
            f"<defs>{defs}</defs>{style}{''.join(self.parts)}</svg>\n"
        )


def _mini_framework(svg: Svg, x: float, y: float, w: float, h: float, *, density: bool = False, samples: int = 7) -> None:
    svg.rect(x, y, w, h, fill="white", stroke="#b6c2c8", sw=1.3, rx=4)
    node_color = "#455a64"
    for px, py in [(0.16, 0.20), (0.82, 0.25), (0.22, 0.72), (0.78, 0.68), (0.50, 0.48)]:
        svg.circle(x + px * w, y + py * h, 4.4, fill=node_color, stroke="white", sw=0.8)
    svg.line(x + 0.16 * w, y + 0.20 * h, x + 0.50 * w, y + 0.48 * h, stroke="#b6c2c8", sw=1.3)
    svg.line(x + 0.82 * w, y + 0.25 * h, x + 0.50 * w, y + 0.48 * h, stroke="#b6c2c8", sw=1.3)
    svg.line(x + 0.22 * w, y + 0.72 * h, x + 0.50 * w, y + 0.48 * h, stroke="#b6c2c8", sw=1.3)
    svg.line(x + 0.78 * w, y + 0.68 * h, x + 0.50 * w, y + 0.48 * h, stroke="#b6c2c8", sw=1.3)

    if density:
        for px, py, r, alpha in [
            (0.36, 0.66, 18, 0.28),
            (0.58, 0.42, 24, 0.34),
            (0.71, 0.32, 13, 0.24),
        ]:
            svg.circle(x + px * w, y + py * h, r, fill=COLOR_DENSITY, stroke="none", sw=0, opacity=alpha)
    else:
        pts = [
            (0.32, 0.64),
            (0.42, 0.58),
            (0.62, 0.40),
            (0.70, 0.33),
            (0.52, 0.52),
            (0.76, 0.63),
            (0.28, 0.30),
            (0.57, 0.73),
            (0.68, 0.53),
            (0.40, 0.38),
        ][:samples]
        for px, py in pts:
            svg.circle(x + px * w, y + py * h, 4.2, fill=COLOR_METHANE, stroke="white", sw=0.7)


def generate_particle_density_overview(output_path):
    """Native SVG overview: GCMC samples particles, cDFT solves density directly."""
    output_path = Path(output_path)
    svg_path = output_path.with_suffix(".svg")
    png_path = output_path.with_suffix(".png")
    svg = Svg(1160, 560)

    svg.text(580, 34, "Particle and density views of adsorption", size=26, weight=700)
    svg.text(580, 66, "same equilibrium problem, different representation", size=16, fill=bfs.MUTED, italic=True)

    # Open-system setup.
    svg.rect(38, 120, 210, 260, fill="#f8fbfc", stroke="#d8e0e4", sw=1.6, rx=12)
    svg.text(143, 145, "Open pore", size=18, weight=700, fill=EDGE_MAIN)
    _mini_framework(svg, 72, 178, 142, 112, samples=6)
    svg.text(143, 320, "reservoir fixes", size=15, fill=bfs.MUTED)
    svg.text(143, 346, "μ, V, T", size=24, weight=700, fill=TEXT_COLOR)

    # Branch arrows.
    svg.line(248, 222, 342, 185, stroke=ARROW_COLOR, sw=2.4, marker="arrow")
    svg.line(248, 278, 342, 375, stroke=ARROW_COLOR, sw=2.4, marker="arrow")

    # GCMC branch.
    svg.rect(342, 120, 300, 160, fill=BOX_MAIN, stroke=EDGE_MAIN, sw=2.0, rx=12)
    svg.text(492, 144, "GCMC particle view", size=18, weight=700, fill=EDGE_MAIN)
    _mini_framework(svg, 372, 172, 74, 58, samples=5)
    _mini_framework(svg, 462, 172, 74, 58, samples=8)
    _mini_framework(svg, 552, 172, 60, 58, samples=4)
    svg.text(492, 250, "average many samples", size=15, fill=bfs.MUTED)

    svg.rect(746, 120, 310, 160, fill="#f8fbfc", stroke="#d8e0e4", sw=1.6, rx=12)
    svg.text(901, 144, "Outputs", size=18, weight=700)
    svg.text(820, 198, "uptake", size=15, fill=bfs.MUTED)
    svg.text(820, 224, "⟨N⟩", size=25, weight=700, fill=EDGE_MAIN)
    _mini_framework(svg, 904, 174, 94, 74, density=True)
    svg.text(951, 262, "density ρ(r)", size=15, fill=bfs.MUTED)
    svg.line(642, 200, 742, 200, stroke=ARROW_COLOR, sw=2.3, marker="arrow")

    # cDFT branch.
    svg.rect(342, 330, 300, 160, fill=BOX_OUTPUT, stroke=EDGE_OUTPUT, sw=2.0, rx=12)
    svg.text(492, 354, "cDFT density view", size=18, weight=700, fill=EDGE_OUTPUT)
    _mini_framework(svg, 374, 386, 88, 68, density=True)
    svg.text(548, 407, "optimize ρ(r)", size=20, weight=700, fill=TEXT_COLOR)
    svg.text(548, 437, "fixed-point solve", size=15, fill=bfs.MUTED)

    svg.rect(746, 330, 310, 160, fill="#fffaf0", stroke=EDGE_HIGHLIGHT, sw=2.0, rx=12)
    svg.text(901, 354, "ML target", size=18, weight=700, fill=EDGE_HIGHLIGHT)
    _mini_framework(svg, 786, 386, 88, 68, density=True)
    svg.text(958, 402, "density field", size=19, weight=700, fill=TEXT_COLOR)
    svg.text(958, 430, "+ uptake", size=19, weight=700, fill=TEXT_COLOR)
    svg.text(958, 458, "learn ρ(r), integrate N", size=14, fill=bfs.MUTED)
    svg.line(642, 410, 742, 410, stroke=ARROW_COLOR, sw=2.3, marker="arrow")

    # Branch comparison.
    svg.text(492, 296, "high fidelity, sampling cost", size=16, fill=EDGE_MAIN)
    svg.text(492, 506, "fast density estimate, approximate functional", size=16, fill=EDGE_OUTPUT)

    svg_path.write_text(svg.svg(), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=1400):
        raise RuntimeError("Could not render SVG preview; install rsvg-convert or ImageMagick.")
    print(f"Saved {svg_path} and {png_path}")


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
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.05))
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
            ax.text(0.72, 0.69, r"$\Omega$", fontsize=13, color=COLOR_ACCEPT, fontweight="bold")
        elif idx == 2:
            ax.add_patch(Circle((0.28, 0.58), 0.035, facecolor="none", edgecolor=COLOR_ACCEPT, linewidth=2.0, zorder=5))
            _arrow(ax, (0.28, 0.88), (0.28, 0.63), color=COLOR_ACCEPT)
        else:
            ax.add_patch(Circle((0.65, 0.58), 0.055, facecolor="none", edgecolor=COLOR_REJECT, linewidth=2.0, zorder=5))
            _arrow(ax, (0.65, 0.58), (0.65, 0.88), color=COLOR_REJECT)

        title, subtitle = panels[idx]
        ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.text(0.5, 0.04, subtitle, ha="center", va="bottom", fontsize=10.5, color=TEXT_COLOR)

    fig.suptitle("GCMC move set in a variable-N state space", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved GCMC moves figure to {output_path}")


def generate_snapshots_to_density_figure(output_path):
    fig = plt.figure(figsize=(9.4, 4.45))
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
        ax.text(0.5, 1.04, f"sample {idx + 1}", ha="center", va="bottom", fontsize=10.5, color=TEXT_COLOR)

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
        fontsize=13.5,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    density_axis.text(
        0.5,
        -0.08,
        r"integrate grid $\rightarrow$ uptake $\langle N\rangle$",
        ha="center",
        va="top",
        fontsize=11.5,
        color=TEXT_COLOR,
    )

    fig.suptitle("From particle snapshots to a density field", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.99)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved snapshots-to-density figure to {output_path}")


def generate_cdft_fixed_point_figure(output_path):
    fig, ax = plt.subplots(figsize=(8.8, 4.25))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.0)
    _clean_axis(ax)

    _rounded_box(ax, (0.65, 2.78), 2.05, 0.78, "current density\n" + r"$\rho^{(n)}$", BOX_MAIN, EDGE_MAIN, fontsize=10.3)
    _rounded_box(
        ax,
        (3.55, 2.78),
        2.25,
        0.78,
        "evaluate correction\n" + r"$\delta F_{exc}/\delta\rho$",
        BOX_CORRECTION,
        EDGE_CORRECTION,
        fontsize=10.3,
    )
    _rounded_box(
        ax,
        (6.6, 2.78),
        2.25,
        0.78,
        "Boltzmann update\nwith many-body term",
        BOX_HIGHLIGHT,
        EDGE_HIGHLIGHT,
        fontsize=10.3,
    )
    _rounded_box(ax, (6.6, 1.15), 2.25, 0.78, "new density\n" + r"$\rho^{(n+1)}$", BOX_MAIN, EDGE_MAIN)
    _rounded_box(ax, (3.55, 1.15), 2.25, 0.78, "converged?", BOX_OUTPUT, EDGE_OUTPUT)

    _arrow(ax, (2.78, 3.17), (3.25, 3.17))
    _arrow(ax, (5.85, 3.17), (6.18, 3.17))
    _arrow(ax, (7.73, 2.70), (7.73, 2.03))
    _arrow(ax, (6.08, 1.54), (5.85, 1.54))
    _arrow(ax, (3.25, 1.54), (2.15, 2.60), color=COLOR_REJECT, rad=0.2)
    ax.text(2.62, 2.10, "No", color=COLOR_REJECT, fontsize=11.5, fontweight="bold")

    _arrow(ax, (5.80, 1.16), (9.0, 0.72), color=COLOR_ACCEPT)
    _rounded_box(ax, (9.35, 0.38), 1.55, 0.68, r"$\rho_{eq}$", "white", EDGE_OUTPUT, fontsize=14)
    ax.text(6.18, 1.34, "Yes", color=COLOR_ACCEPT, fontsize=11.5, fontweight="bold")

    ax.text(
        5.6,
        4.45,
        "same rhythm as quantum-DFT SCF, different object",
        ha="center",
        fontsize=12.5,
        color=TEXT_COLOR,
    )
    fig.suptitle("The cDFT fixed-point loop", fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print(f"Saved cDFT fixed-point figure to {output_path}")


if __name__ == "__main__":
    output_dir = Path("assets/img/blog")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_particle_density_overview(output_dir / "adsorption_gcmc_cdft_particle_density.svg")

    for ext in ("svg", "png"):
        generate_gcmc_moves_figure(output_dir / f"adsorption_gcmc_cdft_moves.{ext}")
        generate_snapshots_to_density_figure(output_dir / f"adsorption_gcmc_cdft_snapshots_to_density.{ext}")
        generate_cdft_fixed_point_figure(output_dir / f"adsorption_gcmc_cdft_fixed_point.{ext}")

    print("Done!")
