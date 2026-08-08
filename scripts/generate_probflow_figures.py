#!/usr/bin/env python3
"""Generate original figures for the ODE/SDE probability-flow blog post.

All figures are original explanatory diagrams saved as editable SVG and PNG
previews. They use rounded geometric forms, restrained solid colors, consistent
strokes, and minimal visual noise. No artwork from the lecture decks or papers
is reproduced.

Conceptual sources
------------------
* Chen et al. (2018), https://arxiv.org/abs/1806.07366
* Anderson (1982), https://doi.org/10.1016/0304-4149(82)90051-5
* Song et al. (2021), https://arxiv.org/abs/2011.13456

License: same as the blog.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import math
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


OUT = Path("assets/img/blog")
WIDTH = 1200


def document(body: str, height: int = 540) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker>
    <marker id="purpleArrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {bfs.TEXT}; }}
      .title {{ font-size: 24px; font-weight: 700; }}
      .label {{ font-size: 19px; font-weight: 600; }}
      .body {{ font-size: 17px; }}
      .small {{ font-size: 15px; fill: {bfs.MUTED}; }}
      .axis {{ stroke: {bfs.SPINE}; stroke-width: 3; stroke-linecap: round; }}
      .arrow {{ fill: none; stroke: {bfs.MUTED}; stroke-width: 3; marker-end: url(#arrow); }}
      .parrow {{ fill: none; stroke: {bfs.PURPLE}; stroke-width: 3; marker-end: url(#purpleArrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x: float, y: float, w: float, h: float, fill: str, stroke: str, radius: int = 25) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, *, purple: bool = False) -> str:
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{"parrow" if purple else "arrow"}"/>'


def particle(x: float, y: float, color: str = bfs.PURPLE, r: int = 9) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="white" stroke-width="3"/>'


def density_path(cx: float, base: float, sigma: float, height: float, color: str, fill: str) -> str:
    points = []
    for k in range(81):
        x = cx - 3 * sigma + 6 * sigma * k / 80
        y = base - height * math.exp(-0.5 * ((x - cx) / sigma) ** 2)
        points.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"M{points[0][0]:.1f},{base:.1f} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points) + f" L{points[-1][0]:.1f},{base:.1f} Z"
    return f'<path d="{area}" fill="{fill}" opacity="0.85"/><polyline points="{line}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>'


def save(stem: str, body: str, height: int = 540) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(document(body, height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=WIDTH):
        raise RuntimeError(f"Could not render {svg_path}")


def transport_continuity() -> None:
    """Illustrate particles, a density, and flux under an ODE flow."""
    b = [text(60, 48, "A velocity field moves particles and transports their density", "title", "start")]
    b.append(box(45, 90, 360, 355, bfs.PURPLE_SOFT, bfs.PURPLE, 30))
    b.append(box(795, 90, 360, 355, bfs.GREEN_LIGHT, bfs.GREEN, 30))
    b.append(text(225, 130, "initial density p(0)", "label"))
    b.append(text(975, 130, "transported density p(t)", "label"))
    b.append(density_path(225, 365, 55, 175, bfs.PURPLE, bfs.PURPLE_LIGHT))
    b.append(density_path(975, 365, 78, 128, bfs.GREEN, bfs.GREEN_LIGHT))
    for dx in [-70, -40, -15, 12, 42, 68]:
        y = 352 - 120 * math.exp(-0.5 * (dx / 55) ** 2)
        b.append(particle(225 + dx, y, bfs.PURPLE))
    for dx in [-105, -65, -25, 20, 63, 102]:
        y = 352 - 85 * math.exp(-0.5 * (dx / 78) ** 2)
        b.append(particle(975 + dx, y, bfs.GREEN))
    # Flow arrows through the center.
    for y, bend in [(175, -35), (245, 0), (315, 35)]:
        b.append(f'<path d="M420,{y} C560,{y+bend} 640,{y-bend} 780,{y}" class="parrow"/>')
    b.append(box(475, 148, 250, 55, "white", bfs.PURPLE, 18))
    b.append(text(600, 183, "dX = v(X,t) dt", "label"))
    b.append(text(600, 345, "flow map", "small"))
    b.append(text(600, 490, "The continuity equation balances local accumulation against the divergence of probability flux p v.", "body"))
    b.append(text(600, 518, "Particles are neither created nor destroyed; their spacing changes the density.", "small"))
    save("probflow_transport_continuity", "\n".join(b), 550)


def ode_sde_paths() -> None:
    """Contrast smooth deterministic characteristics and noisy SDE paths."""
    b = [text(60, 48, "ODE trajectories are smooth characteristics; SDE trajectories receive random kicks", "title", "start")]
    panels = [(45, "ODE", "one path per initial state", bfs.PURPLE_SOFT, bfs.PURPLE), (625, "SDE", "a distribution over paths", bfs.BLUE_LIGHT, bfs.BLUE)]
    for x0, name, sub, fill, stroke in panels:
        b.append(box(x0, 88, 530, 370, fill, stroke, 28))
        b.append(text(x0 + 265, 130, name, "label"))
        b.append(text(x0 + 265, 156, sub, "small"))
        b.append(f'<line x1="{x0+65}" y1="405" x2="{x0+465}" y2="405" class="axis"/>')
        b.append(f'<line x1="{x0+65}" y1="405" x2="{x0+65}" y2="185" class="axis"/>')
        b.append(text(x0 + 470, 430, "time", "small"))
        b.append(text(x0 + 45, 190, "state", "small"))
    # ODE paths.
    for idx, y0 in enumerate([365, 325, 285, 245]):
        color = [bfs.PURPLE, bfs.BLUE, bfs.TEAL, bfs.AMBER][idx]
        b.append(f'<path d="M110,{y0} C230,{y0-15} 365,{y0-80} 510,{y0-105}" fill="none" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        b.append(particle(110, y0, color, 8))
        b.append(particle(510, y0 - 105, color, 8))
    # Reproducible random SDE paths.
    rng = random.Random(17)
    for idx, y0 in enumerate([365, 325, 285, 245]):
        color = [bfs.PURPLE, bfs.BLUE, bfs.TEAL, bfs.AMBER][idx]
        pts = [(690, y0)]
        y = y0
        for k in range(1, 21):
            x = 690 + 20 * k
            y += -4.2 + rng.uniform(-17, 17)
            y = min(395, max(190, y))
            pts.append((x, y))
        b.append('<polyline points="' + " ".join(f"{x},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        b.append(particle(690, y0, color, 8))
        b.append(particle(1090, y, color, 8))
    b.append(text(600, 510, "Both dynamics define a time-indexed density, but only the ODE defines a deterministic flow map.", "body"))
    save("probflow_ode_sde_paths", "\n".join(b), 540)


def marginal_equivalence() -> None:
    """Show forward SDE and two reverse dynamics with shared marginals."""
    b = [text(60, 48, "The reverse SDE and probability-flow ODE share marginals, not trajectories", "title", "start")]
    times = [(170, 0.0, 42, 145), (430, 0.33, 65, 115), (690, 0.67, 88, 92), (990, 1.0, 110, 75)]
    for x, t, sigma, h in times:
        b.append(density_path(x, 270, sigma, h, bfs.PURPLE if t < 0.5 else bfs.BLUE, bfs.PURPLE_LIGHT if t < 0.5 else bfs.BLUE_LIGHT))
        b.append(text(x, 305, f"p{int(round(t*3))}", "small"))
    # Forward SDE upper lane.
    b.append(text(70, 105, "forward SDE", "label", "start"))
    b.append(f'<path d="M155,120 C360,80 770,155 1010,115" fill="none" stroke="{bfs.BLUE}" stroke-width="5" stroke-dasharray="9 7" marker-end="url(#arrow)"/>')
    # Reverse lanes.
    b.append(text(70, 375, "reverse SDE", "label", "start"))
    rng = random.Random(23)
    pts = [(1010, 365)]
    y = 365
    for k in range(1, 30):
        x = 1010 - 27 * k
        y += rng.uniform(-13, 13)
        pts.append((x, y))
    b.append('<polyline points="' + " ".join(f"{x},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{bfs.AMBER}" stroke-width="5" stroke-linejoin="round" marker-end="url(#arrow)"/>')
    b.append(text(70, 455, "probability-flow ODE", "label", "start"))
    b.append(f'<path d="M1010,445 C790,485 430,420 280,450" fill="none" stroke="{bfs.GREEN}" stroke-width="6" marker-end="url(#arrow)"/>')
    b.append(text(600, 515, "At every marked time, all three dynamics have the same density p(t) when the exact score is used.", "body"))
    save("probflow_shared_marginals", "\n".join(b), 545)


def gaussian_probability_flow() -> None:
    """Exact variance-exploding Gaussian example and deterministic scaling map."""
    b = [text(60, 48, "For Gaussian diffusion, probability flow is a deterministic radial scaling", "title", "start")]
    b.append(box(45, 90, 500, 360, bfs.PURPLE_SOFT, bfs.PURPLE, 30))
    b.append(box(655, 90, 500, 360, bfs.GREEN_LIGHT, bfs.GREEN, 30))
    b.append(text(295, 132, "same Gaussian marginals", "label"))
    b.append(density_path(245, 370, 48, 195, bfs.PURPLE, bfs.PURPLE_LIGHT))
    b.append(density_path(345, 370, 92, 105, bfs.BLUE, bfs.BLUE_LIGHT))
    b.append(text(185, 205, "p(0)", "label"))
    b.append(text(430, 275, "p(t)", "label"))
    b.append(text(295, 420, "variance: s0² → s0² + σ²t", "small"))

    b.append(text(905, 132, "probability-flow trajectories", "label"))
    origin_x, origin_y = 905, 300
    b.append(particle(origin_x, origin_y, bfs.TEXT, 12))
    for ang, length, color in [(-2.6, 155, bfs.PURPLE), (-2.0, 128, bfs.BLUE), (-1.25, 150, bfs.TEAL), (-0.45, 145, bfs.AMBER), (0.35, 135, bfs.RED), (0.95, 120, bfs.GREEN)]:
        x1 = origin_x + 42 * math.cos(ang); y1 = origin_y + 42 * math.sin(ang)
        x2 = origin_x + length * math.cos(ang); y2 = origin_y + length * math.sin(ang)
        b.append(particle(x1, y1, color, 8))
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="5" marker-end="url(#arrow)"/>')
    b.append(text(905, 420, "x(t) − m = scale(t) · (x(0) − m)", "small"))
    b.append(text(600, 505, "The ODE expands deviations from the mean just enough to match the variance created by Brownian noise.", "body"))
    save("probflow_gaussian_example", "\n".join(b), 540)


def main() -> None:
    transport_continuity()
    ode_sde_paths()
    marginal_equivalence()
    gaussian_probability_flow()


if __name__ == "__main__":
    main()
