#!/usr/bin/env python3
"""Generate original figures for the discrete flow and generator matching post.

The diagrams are original, source-first SVG artwork with PNG previews. They use
the site's restrained flat-icon visual language; no figure from a lecture deck
or paper is reproduced.

Conceptual sources
------------------
* Campbell et al. (2022), https://proceedings.neurips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html
* Gat et al. (2024), https://openreview.net/forum?id=GTDKo3Sv9p
* Holderrieth et al. (2025), https://arxiv.org/abs/2410.20587

License: same as the blog.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


OUT = Path("assets/img/blog")
WIDTH = 1200


def document(body: str, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker>
    <marker id="purpleArrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {bfs.TEXT}; }}
      .title {{ font-size: 24px; font-weight: 700; }}
      .label {{ font-size: 19px; font-weight: 600; }}
      .body {{ font-size: 17px; }}
      .small {{ font-size: 15px; fill: {bfs.MUTED}; }}
      .arrow {{ fill: none; stroke: {bfs.MUTED}; stroke-width: 3; stroke-linecap: round; marker-end: url(#arrow); }}
      .parrow {{ fill: none; stroke: {bfs.PURPLE}; stroke-width: 4; stroke-linecap: round; marker-end: url(#purpleArrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x: float, y: float, w: float, h: float, fill: str, stroke: str, radius: int = 26) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def state(x: float, y: float, label: str, fill: str, stroke: str, r: int = 48) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="4"/>{text(x, y+7, label, "label")}'


def save(stem: str, body: str, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(document(body, height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=WIDTH):
        raise RuntimeError(f"Could not render {svg_path}")


def ctmc_master_equation() -> None:
    b = [text(55, 48, "A continuous-time chain is specified by rates, not step probabilities", "title", "start")]
    b.append(box(45, 90, 620, 385, bfs.PURPLE_SOFT, bfs.PURPLE, 30))
    b.append(text(355, 130, "three-state generator", "label"))
    b.append(state(175, 285, "MASK", bfs.AMBER_LIGHT, bfs.AMBER, 56))
    b.append(state(465, 205, "A", bfs.GREEN_LIGHT, bfs.GREEN))
    b.append(state(465, 365, "B", bfs.BLUE_LIGHT, bfs.BLUE))
    b.append('<path d="M230,263 C305,225 345,211 408,207" class="parrow"/>')
    b.append('<path d="M229,307 C300,348 345,365 408,365" class="parrow"/>')
    b.append(text(315, 213, "q(A | M)", "small"))
    b.append(text(318, 374, "q(B | M)", "small"))
    b.append(text(355, 440, "exit rate λ(M) = q(A | M) + q(B | M)", "small"))

    b.append(box(710, 90, 445, 385, "white", bfs.SPINE, 30))
    b.append(text(932, 130, "probability balance", "label"))
    rows = [("MASK", "outgoing", bfs.AMBER, 150), ("A", "incoming", bfs.GREEN, 235), ("B", "incoming", bfs.BLUE, 320)]
    for name, kind, color, y in rows:
        b.append(f'<circle cx="770" cy="{y}" r="15" fill="{color}"/>')
        b.append(text(803, y + 6, name, "label", "start"))
        b.append(text(1110, y + 5, kind, "small", "end"))
    b.append('<line x1="765" y1="385" x2="1095" y2="385" stroke="'+bfs.SPINE+'" stroke-width="3" stroke-linecap="round"/>')
    b.append(text(930, 425, "inflow − outflow", "label"))
    b.append(text(600, 525, "The diagonal generator entry is minus the total exit rate, so every generator column sums to zero.", "body"))
    save("discgm_ctmc_master_equation", "\n".join(b), 560)


def conditional_marginalization() -> None:
    b = [text(55, 48, "Simple endpoint-conditioned chains average into the data-generating chain", "title", "start")]
    b.append(box(45, 95, 290, 360, bfs.PURPLE_SOFT, bfs.PURPLE, 28))
    b.append(text(190, 135, "condition on endpoint Z", "label"))
    b.append(state(115, 265, "M", bfs.AMBER_LIGHT, bfs.AMBER, 42))
    b.append(state(265, 205, "A", bfs.GREEN_LIGHT, bfs.GREEN, 40))
    b.append(state(265, 330, "B", bfs.BLUE_LIGHT, bfs.BLUE, 40))
    b.append('<path d="M153,244 C190,220 205,210 220,208" class="parrow"/>')
    b.append('<path d="M153,285 C188,316 202,327 220,328" class="parrow"/>')
    b.append(text(190, 395, "known conditional rates qᶻ", "small"))

    b.append('<path d="M350,275 L470,275" class="arrow"/>')
    b.append(box(475, 180, 250, 190, "white", bfs.SPINE, 28))
    b.append(text(600, 220, "posterior average", "label"))
    b.append(text(600, 265, "q(y | x)", "label"))
    b.append(text(600, 300, "= E[qᶻ(y | x) | X=x]", "small"))
    b.append(text(600, 340, "the endpoint is latent", "small"))

    b.append('<path d="M730,275 L850,275" class="arrow"/>')
    b.append(box(855, 95, 300, 360, bfs.GREEN_LIGHT, bfs.GREEN, 28))
    b.append(text(1005, 135, "learn one marginal chain", "label"))
    b.append(state(930, 265, "x", "white", bfs.PURPLE, 42))
    b.append(state(1080, 205, "y₁", "white", bfs.GREEN, 40))
    b.append(state(1080, 330, "y₂", "white", bfs.BLUE, 40))
    b.append('<path d="M969,244 C1000,222 1020,210 1035,208" class="parrow"/>')
    b.append('<path d="M969,286 C1000,315 1018,326 1035,328" class="parrow"/>')
    b.append(text(1005, 395, "network predicts all outgoing rates", "small"))
    b.append(text(600, 515, "Conditional training never requires evaluating the intractable marginal density; it samples an endpoint and an intermediate state.", "body"))
    save("discgm_conditional_marginalization", "\n".join(b), 555)


def factorized_molecules() -> None:
    b = [text(55, 48, "Single-coordinate jumps make large discrete objects tractable", "title", "start")]
    b.append(box(45, 90, 520, 380, bfs.BLUE_LIGHT, bfs.BLUE, 28))
    b.append(text(305, 130, "sequence: one token changes", "label"))
    tokens = [("M", bfs.AMBER_LIGHT, bfs.AMBER), ("A", bfs.GREEN_LIGHT, bfs.GREEN), ("M", bfs.AMBER_LIGHT, bfs.AMBER), ("G", bfs.PURPLE_LIGHT, bfs.PURPLE), ("M", bfs.AMBER_LIGHT, bfs.AMBER)]
    for i, (label, fill, stroke) in enumerate(tokens):
        x = 85 + i * 90
        b.append(box(x, 190, 66, 66, fill, stroke, 18))
        b.append(text(x + 33, 232, label, "label"))
    b.append('<path d="M298,290 L298,345" class="parrow"/>')
    b.append(box(265, 365, 66, 66, bfs.GREEN_LIGHT, bfs.GREEN, 18))
    b.append(text(298, 407, "L", "label"))
    b.append(text(305, 450, "each rate may still inspect the full sequence", "small"))

    b.append(box(635, 90, 520, 380, bfs.GREEN_LIGHT, bfs.GREEN, 28))
    b.append(text(895, 130, "molecular graph: one categorical edit", "label"))
    nodes = [(755, 225, "C", bfs.PURPLE), (895, 180, "N", bfs.BLUE), (1035, 225, "O", bfs.RED), (895, 350, "C", bfs.GREEN)]
    for x1, y1, label, color in nodes:
        b.append(state(x1, y1, label, "white", color, 38))
    for x1, y1, x2, y2 in [(793,215,855,192),(935,192,997,215),(1012,253,925,330),(865,330,780,253)]:
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{bfs.MUTED}" stroke-width="6" stroke-linecap="round"/>')
    b.append('<circle cx="965" cy="291" r="13" fill="'+bfs.AMBER+'" stroke="white" stroke-width="4"/>')
    b.append(text(985, 300, "bond edit", "small", "start"))
    b.append(text(895, 430, "node and edge rates must respect graph symmetry", "small"))
    b.append(text(600, 520, "The output scales like coordinates × alphabet size, while dependencies enter through the shared context network.", "body"))
    save("discgm_factorized_molecules", "\n".join(b), 555)


def simulation_tradeoffs() -> None:
    b = [text(55, 48, "The same learned rates support different simulation budgets", "title", "start")]
    panels = [(45, "event-driven", "sample the next jump time", bfs.PURPLE_SOFT, bfs.PURPLE), (625, "fixed-step", "approximate a short transition", bfs.GREEN_LIGHT, bfs.GREEN)]
    for x, title, sub, fill, stroke in panels:
        b.append(box(x, 90, 530, 370, fill, stroke, 28))
        b.append(text(x + 265, 132, title, "label"))
        b.append(text(x + 265, 160, sub, "small"))
        b.append(f'<line x1="{x+65}" y1="310" x2="{x+465}" y2="310" stroke="{bfs.SPINE}" stroke-width="4" stroke-linecap="round"/>')
    # Irregular event times.
    for px, label, color in [(125,"M",bfs.AMBER),(240,"A",bfs.GREEN),(435,"B",bfs.BLUE)]:
        b.append(f'<line x1="{px}" y1="285" x2="{px}" y2="335" stroke="{color}" stroke-width="5" stroke-linecap="round"/>')
        b.append(text(px, 370, label, "label"))
    b.append(text(310, 420, "exact events • sequential network calls", "small"))
    # Regular grid and possible updates.
    for k in range(7):
        px = 690 + k * 65
        b.append(f'<line x1="{px}" y1="287" x2="{px}" y2="333" stroke="{bfs.MUTED}" stroke-width="3"/>')
        if k in (2, 3, 5):
            b.append(f'<circle cx="{px}" cy="270" r="11" fill="{[bfs.GREEN,bfs.BLUE,bfs.AMBER][(k+1)%3]}"/>')
    b.append(text(890, 370, "step size h", "small"))
    b.append(text(890, 420, "parallel updates • discretization bias", "small"))
    b.append(text(600, 515, "Smaller steps reduce bias but increase network evaluations; large exit rates force adaptive steps or event simulation.", "body"))
    save("discgm_simulation_tradeoffs", "\n".join(b), 550)


def main() -> None:
    ctmc_master_equation()
    conditional_marginalization()
    factorized_molecules()
    simulation_tradeoffs()


if __name__ == "__main__":
    main()
