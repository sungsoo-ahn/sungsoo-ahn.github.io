#!/usr/bin/env python3
"""Generate original figures for the deep graph network failure-modes post.

The figures use a flat-icon visual language: rounded geometric forms,
restrained solid colors, consistent strokes, and no gradients or texture.
They do not copy Flaticon assets, lecture slides, or paper figures. The
mathematical ideas follow the author's 2025 GDL/ML4Mol lectures and the
primary papers cited in the post. License: CC BY 4.0 with the blog post.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import blog_figure_style as bfs

OUT = ROOT / "assets" / "img" / "blog"
TEXT = bfs.TEXT
MUTED = bfs.MUTED
SPINE = bfs.SPINE
PURPLE = bfs.PURPLE
PURPLE_SOFT = bfs.PURPLE_SOFT
BLUE = bfs.BLUE
BLUE_LIGHT = bfs.BLUE_LIGHT
TEAL = bfs.TEAL
TEAL_LIGHT = bfs.TEAL_LIGHT
AMBER = bfs.AMBER
AMBER_LIGHT = bfs.AMBER_LIGHT
RED = bfs.RED
RED_LIGHT = bfs.RED_LIGHT


def svg_document(body: str, *, width: int = 1000, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {TEXT}; }}
      .title {{ font-size: 22px; font-weight: 700; }}
      .label {{ font-size: 18px; font-weight: 700; }}
      .body {{ font-size: 16px; }}
      .small {{ font-size: 14px; fill: {MUTED}; }}
      .edge {{ stroke: {SPINE}; stroke-width: 4; stroke-linecap: round; }}
      .arrow {{ stroke: {MUTED}; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def node(x: float, y: float, color: str, *, radius: int = 12, stroke: str = "white") -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="3"/>'


def edge(x1: float, y1: float, x2: float, y2: float, *, color: str = SPINE, width: int = 4, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash}/>'


def panel(x: float, y: float, w: float, h: float, fill: str, title: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{SPINE}" stroke-width="2"/>'
        f'<text x="{x+w/2}" y="{y+40}" text-anchor="middle" class="label">{title}</text>'
    )


def write(name: str, body: str, *, width: int = 1000, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"gnnfail_{name}.svg"
    png_path = OUT / f"gnnfail_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def small_graph(cx: float, cy: float, colors: list[str], *, scale: float = 1.0) -> str:
    pts = [
        (cx - 65 * scale, cy),
        (cx, cy - 55 * scale),
        (cx + 65 * scale, cy),
        (cx, cy + 55 * scale),
    ]
    links = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    return "".join(edge(*pts[i], *pts[j], width=max(2, int(4 * scale))) for i, j in links) + "".join(
        node(x, y, c, radius=max(8, int(13 * scale))) for (x, y), c in zip(pts, colors)
    )


def make_three_failures() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Depth fixes reach, then creates two different information failures</text>
      {panel(30, 82, 295, 410, BLUE_LIGHT, "Under-reaching")}
      {panel(352, 82, 295, 410, PURPLE_SOFT, "Over-smoothing")}
      {panel(675, 82, 295, 410, RED_LIGHT, "Over-squashing")}

      {edge(95, 245, 160, 245)}{edge(160, 245, 225, 245)}
      {node(95,245,AMBER)}{node(160,245,SPINE)}{node(225,245,TEAL)}
      <path d="M 100 210 L 154 210" class="arrow"/>
      <text x="177" y="330" text-anchor="middle" class="body">too few layers</text>
      <text x="177" y="356" text-anchor="middle" class="small">the target lies outside the</text>
      <text x="177" y="377" text-anchor="middle" class="small">current receptive field</text>

      {small_graph(500, 245, [PURPLE, PURPLE, PURPLE, PURPLE], scale=.85)}
      <circle cx="500" cy="245" r="76" fill="none" stroke="{PURPLE}" stroke-width="2" stroke-dasharray="7 7"/>
      <text x="500" y="340" text-anchor="middle" class="body">too much repeated mixing</text>
      <text x="500" y="366" text-anchor="middle" class="small">node representations lose</text>
      <text x="500" y="387" text-anchor="middle" class="small">task-relevant contrast</text>

      {node(735,180,BLUE)}{node(735,230,BLUE)}{node(735,280,BLUE)}{node(735,330,BLUE)}
      {node(820,255,AMBER,radius=15)}
      {node(910,255,TEAL,radius=18)}
      {edge(735,180,820,255)}{edge(735,230,820,255)}{edge(735,280,820,255)}{edge(735,330,820,255)}{edge(820,255,910,255,color=RED,width=7)}
      <text x="823" y="345" text-anchor="middle" class="body">too much compression</text>
      <text x="823" y="371" text-anchor="middle" class="small">many distant signals cross</text>
      <text x="823" y="392" text-anchor="middle" class="small">a narrow topological cut</text>
    '''
    write("three_failures", body)


def make_smoothing_diffusion() -> None:
    colors = [
        [BLUE, TEAL, AMBER, RED, PURPLE],
        ["#6676D1", "#4E82AD", "#8E7F77", "#9B687A", "#816AB0"],
        ["#7370B8", "#7275A8", "#82758F", "#867287", "#7D70A0"],
        ["#77729F", "#79739B", "#7C7494", "#7D7492", "#7A7398"],
    ]
    body = '<text x="500" y="42" text-anchor="middle" class="title">Repeated graph averaging removes high-frequency feature contrast</text>'
    xs = [95, 155, 215, 275, 335]
    ys = [130, 225, 320, 415]
    labels = ["depth 0", "depth 1", "depth 4", "deep limit"]
    for row, y in enumerate(ys):
        body += f'<text x="52" y="{y+5}" text-anchor="middle" class="small">{labels[row]}</text>'
        for x1, x2 in zip(xs[:-1], xs[1:]):
            body += edge(x1, y, x2, y)
        for x, color in zip(xs, colors[row]):
            body += node(x, y, color, radius=16)
    body += f'''
      <rect x="430" y="92" width="510" height="365" rx="20" fill="{PURPLE_SOFT}" stroke="{SPINE}" stroke-width="2"/>
      <text x="685" y="130" text-anchor="middle" class="label">Dirichlet energy</text>
      <line x1="505" y1="385" x2="885" y2="385" stroke="{SPINE}" stroke-width="2"/>
      <line x1="505" y1="385" x2="505" y2="175" stroke="{SPINE}" stroke-width="2"/>
      <path d="M 510 190 C 575 245, 620 288, 690 330 C 760 368, 825 377, 880 380" fill="none" stroke="{PURPLE}" stroke-width="5" stroke-linecap="round"/>
      <circle cx="510" cy="190" r="7" fill="{PURPLE}"/><circle cx="690" cy="330" r="7" fill="{PURPLE}"/><circle cx="880" cy="380" r="7" fill="{PURPLE}"/>
      <text x="695" y="425" text-anchor="middle" class="small">message-passing depth</text>
      <text x="473" y="280" text-anchor="middle" class="small" transform="rotate(-90 473 280)">feature variation across edges</text>
      <text x="695" y="162" text-anchor="middle" class="body">E(H) = tr(HᵀLH)</text>
    '''
    write("smoothing_diffusion", body)


def make_tree_squashing() -> None:
    body = '<text x="500" y="42" text-anchor="middle" class="title">A binary tree turns exponential receptive-field growth into vanishing sensitivity</text>'
    levels = [
        [(105 + i * 52, 120) for i in range(8)],
        [(131 + i * 104, 220) for i in range(4)],
        [(183 + i * 208, 320) for i in range(2)],
        [(287, 420)],
    ]
    for level in range(3):
        for i, (x, y) in enumerate(levels[level]):
            px, py = levels[level + 1][i // 2]
            body += edge(x, y, px, py, color=SPINE, width=3)
    for x, y in levels[0]:
        body += node(x, y, BLUE, radius=11)
    for x, y in levels[1]:
        body += node(x, y, "#867BC2", radius=13)
    for x, y in levels[2]:
        body += node(x, y, AMBER, radius=15)
    body += node(287, 420, RED, radius=19)
    body += f'''
      <rect x="570" y="112" width="350" height="315" rx="22" fill="{RED_LIGHT}" stroke="{SPINE}" stroke-width="2"/>
      <text x="745" y="155" text-anchor="middle" class="label">exact toy update</text>
      <text x="745" y="205" text-anchor="middle" class="body">hparent = ½(hleft + hright)</text>
      <text x="745" y="264" text-anchor="middle" class="body">hroot = 2⁻ʳ ∑ hleaf</text>
      <text x="745" y="323" text-anchor="middle" class="body">∂hroot / ∂hleaf = 2⁻ʳ</text>
      <text x="745" y="372" text-anchor="middle" class="small">each added level doubles the leaves</text>
      <text x="745" y="395" text-anchor="middle" class="small">and halves one leaf's direct influence</text>
      <path d="M 455 265 L 555 265" class="arrow"/>
      <text x="500" y="242" text-anchor="middle" class="small">fixed-width state</text>
      <text x="287" y="477" text-anchor="middle" class="small">root prediction</text>
    '''
    write("tree_squashing", body)


def community_graph(shortcuts: bool = False) -> str:
    left = [(95,180),(145,125),(145,235),(205,145),(205,215)]
    right = [(395,180),(345,125),(345,235),(285,145),(285,215)]
    links = [(0,1),(0,2),(1,3),(2,4),(3,4)]
    out = ""
    for pts in (left, right):
        for i,j in links:
            out += edge(*pts[i], *pts[j], width=3)
    out += edge(205,180,285,180,color=RED,width=7)
    if shortcuts:
        out += edge(145,125,345,125,color=TEAL,width=4)
        out += edge(145,235,345,235,color=TEAL,width=4)
    out += "".join(node(x,y,BLUE if x<250 else PURPLE,radius=10) for x,y in left+right)
    return out


def make_rewiring_tradeoffs() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Rewiring widens a bottleneck, but also changes the model's graph prior</text>
      <g transform="translate(5,55)">{community_graph(False)}</g>
      <g transform="translate(500,55)">{community_graph(True)}</g>
      <text x="250" y="345" text-anchor="middle" class="label">input graph</text>
      <text x="250" y="372" text-anchor="middle" class="small">one bridge carries cross-community information</text>
      <text x="745" y="345" text-anchor="middle" class="label">rewired computational graph</text>
      <text x="745" y="372" text-anchor="middle" class="small">shortcuts reduce path length and effective resistance</text>
      <path d="M 445 230 L 548 230" class="arrow"/>
      <rect x="70" y="425" width="250" height="78" rx="17" fill="{TEAL_LIGHT}"/>
      <text x="195" y="455" text-anchor="middle" class="body">better long-range sensitivity</text>
      <text x="195" y="480" text-anchor="middle" class="small">fewer narrow cuts</text>
      <rect x="375" y="425" width="250" height="78" rx="17" fill="{AMBER_LIGHT}"/>
      <text x="500" y="455" text-anchor="middle" class="body">more computation</text>
      <text x="500" y="480" text-anchor="middle" class="small">more edges and messages</text>
      <rect x="680" y="425" width="250" height="78" rx="17" fill="{RED_LIGHT}"/>
      <text x="805" y="455" text-anchor="middle" class="body">weaker locality bias</text>
      <text x="805" y="480" text-anchor="middle" class="small">unrelated nodes may mix too early</text>
    '''
    write("rewiring_tradeoffs", body)


if __name__ == "__main__":
    make_three_failures()
    make_smoothing_diffusion()
    make_tree_squashing()
    make_rewiring_tradeoffs()
