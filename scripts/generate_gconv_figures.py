#!/usr/bin/env python3
"""Generate original figures for the graph-convolution blog post.

All figures are original explanatory diagrams, drawn as editable SVG and
rendered to PNG for visual inspection. They synthesize mathematical ideas from
the sources below without reproducing any source artwork.

Conceptual sources
------------------
* Shuman et al. (2013), https://doi.org/10.1109/MSP.2012.2235192
* Defferrard et al. (2016), https://proceedings.neurips.cc/paper/2016/hash/
  04df4d434d481c5bb723be1b6df1ee65-Abstract.html
* Kipf and Welling (2017), https://openreview.net/forum?id=SJU4ayYgl
* Maron et al. (2019), https://openreview.net/forum?id=Syx72jC9tm

The diagrams use a flat-icon visual language: rounded silhouettes, restrained
solid colors, consistent strokes, and minimal detail. No third-party icon or
lecture-slide artwork is copied. License: same as the blog.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


OUT = Path("assets/img/blog")
W, H = 1200, 520


def _svg(body: str, *, height: int = H) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{bfs.MUTED}"/>
    </marker>
    <marker id="purple-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{bfs.PURPLE}"/>
    </marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {bfs.TEXT}; }}
      .title {{ font-size: 24px; font-weight: 700; }}
      .label {{ font-size: 19px; font-weight: 600; }}
      .body {{ font-size: 17px; }}
      .small {{ font-size: 15px; fill: {bfs.MUTED}; }}
      .edge {{ stroke: {bfs.SPINE}; stroke-width: 5; stroke-linecap: round; }}
      .arrow {{ stroke: {bfs.MUTED}; stroke-width: 3; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def _text(x: float, y: float, text: str, cls: str = "body", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>'


def _round_box(x: float, y: float, w: float, h: float, fill: str, stroke: str, r: int = 20) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def _node(x: float, y: float, value: str, *, fill: str = bfs.PURPLE_LIGHT,
          stroke: str = bfs.PURPLE, radius: int = 29) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="4"/>'
        + _text(x, y + 7, value, "label")
    )


def _arrow(x1: float, y1: float, x2: float, y2: float, *, purple: bool = False) -> str:
    marker = "purple-arrow" if purple else "arrow"
    color = bfs.PURPLE if purple else bfs.MUTED
    return f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="3" fill="none" marker-end="url(#{marker})"/>'


def _save(stem: str, body: str, *, height: int = H) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(_svg(body, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=W):
        raise RuntimeError(f"Could not render {svg_path}")


def translation_equivariance() -> None:
    """A commuting-square explanation of translation-equivariant convolution."""
    b = []
    b.append(_text(70, 54, "Translation equivariance turns a dense matrix into shared local weights", "title", "start"))

    # Two input/output signals in a commuting square.
    xs = [110, 180, 250, 320, 390, 460]
    vals_top = ["0", "1", "3", "2", "0", "0"]
    vals_bottom = ["0", "0", "1", "3", "2", "0"]
    for x, val in zip(xs, vals_top):
        b.append(_node(x, 145, val, radius=25))
    for x, val in zip(xs, vals_bottom):
        b.append(_node(x, 370, val, fill=bfs.BLUE_LIGHT, stroke=bfs.BLUE, radius=25))
    b.append(_text(285, 105, "signal x", "label"))
    b.append(_text(285, 442, "shifted signal Sx", "label"))
    b.append(_arrow(490, 145, 640, 145, purple=True))
    b.append(_arrow(490, 370, 640, 370, purple=True))
    b.append(_arrow(285, 185, 285, 323))
    b.append(_text(305, 262, "shift S", "small", "start"))

    # Shared filter capsule and outputs.
    b.append(_round_box(520, 90, 100, 110, bfs.AMBER_LIGHT, bfs.AMBER, 28))
    b.append(_text(570, 126, "same", "small"))
    b.append(_text(570, 153, "kernel", "label"))
    b.append(_text(570, 181, "[a b c]", "small"))
    b.append(_round_box(520, 315, 100, 110, bfs.AMBER_LIGHT, bfs.AMBER, 28))
    b.append(_text(570, 351, "same", "small"))
    b.append(_text(570, 378, "kernel", "label"))
    b.append(_text(570, 406, "[a b c]", "small"))

    out_xs = [715, 785, 855, 925, 995, 1065]
    for x, val in zip(out_xs, ["·", "y₁", "y₂", "y₃", "·", "·"]):
        b.append(_node(x, 145, val, fill=bfs.GREEN_LIGHT, stroke=bfs.GREEN, radius=25))
    for x, val in zip(out_xs, ["·", "·", "y₁", "y₂", "y₃", "·"]):
        b.append(_node(x, 370, val, fill=bfs.GREEN_LIGHT, stroke=bfs.GREEN, radius=25))
    b.append(_arrow(890, 185, 890, 323))
    b.append(_text(910, 262, "shift S", "small", "start"))
    b.append(_text(890, 105, "Cx", "label"))
    b.append(_text(890, 442, "SCx = CSx", "label"))
    b.append(_text(600, 495, "The two paths agree exactly when the linear map C commutes with translation: SC = CS.", "body"))
    _save("gconv_translation_equivariance", "\n".join(b))


def spectral_filter() -> None:
    """Graph Fourier transform, spectral multiplier, and inverse transform."""
    b = []
    b.append(_text(70, 52, "A graph filter reshapes a signal in the Laplacian eigenbasis", "title", "start"))
    centers = [(120, 240), (205, 140), (305, 180), (390, 305), (250, 355)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    for i, j in edges:
        x1, y1 = centers[i]; x2, y2 = centers[j]
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge"/>')
    for (x, y), val in zip(centers, ["1.0", "0.2", "−0.8", "−0.4", "0.7"]):
        fill = bfs.AMBER_LIGHT if val.startswith("−") else bfs.PURPLE_LIGHT
        stroke = bfs.AMBER if val.startswith("−") else bfs.PURPLE
        b.append(_node(x, y, val, fill=fill, stroke=stroke, radius=32))
    b.append(_text(250, 430, "node signal f", "label"))
    b.append(_arrow(435, 250, 525, 250, purple=True))
    b.append(_text(480, 225, "Uᵀ", "label"))
    b.append(_text(480, 282, "graph FT", "small"))

    # Spectrum bars.
    base_x, base_y = 565, 370
    heights = [72, 150, 95, 55, 112]
    for i, h in enumerate(heights):
        fill = bfs.PURPLE if i < 2 else bfs.SPINE
        b.append(f'<rect x="{base_x + i*52}" y="{base_y-h}" width="30" height="{h}" rx="8" fill="{fill}"/>')
        b.append(_text(base_x + i*52 + 15, base_y + 26, f"λ{i}", "small"))
    b.append(_text(675, 155, "Fourier coefficients f̂", "label"))
    b.append(_round_box(555, 414, 250, 55, bfs.BLUE_LIGHT, bfs.BLUE, 18))
    b.append(_text(680, 449, "multiply by h(λ)", "label"))
    b.append(_arrow(825, 250, 905, 250, purple=True))
    b.append(_text(865, 225, "U", "label"))
    b.append(_text(865, 282, "inverse FT", "small"))

    # Smoothed output graph.
    centers2 = [(940, 240), (1000, 160), (1080, 195), (1120, 300), (1025, 350)]
    for i, j in edges:
        x1, y1 = centers2[i]; x2, y2 = centers2[j]
        b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge"/>')
    for (x, y), val in zip(centers2, ["0.6", "0.3", "−0.2", "−0.1", "0.4"]):
        fill = bfs.GREEN_LIGHT if not val.startswith("−") else bfs.AMBER_LIGHT
        stroke = bfs.GREEN if not val.startswith("−") else bfs.AMBER
        b.append(_node(x, y, val, fill=fill, stroke=stroke, radius=30))
    b.append(_text(1030, 430, "filtered signal g", "label"))
    b.append(_text(600, 500, "Low-pass filtering suppresses large-eigenvalue modes, so adjacent nodes become more similar.", "body"))
    _save("gconv_spectral_filter", "\n".join(b))


def two_derivations() -> None:
    """Compare spectral and permutation-equivariant derivations."""
    b = []
    b.append(_text(70, 50, "Two derivations impose different symmetries—and therefore answer different questions", "title", "start"))
    b.append(_round_box(50, 88, 520, 340, bfs.PURPLE_SOFT, bfs.PURPLE, 28))
    b.append(_round_box(630, 88, 520, 340, bfs.BLUE_LIGHT, bfs.BLUE, 28))
    b.append(_text(310, 130, "Fixed graph: commute with its Laplacian", "label"))
    b.append(_text(890, 130, "Varying graphs: commute with every relabeling", "label"))

    left = [(125, 230), (205, 175), (290, 230), (220, 310)]
    for i, j in [(0,1),(1,2),(2,3),(3,0),(1,3)]:
        b.append(f'<line x1="{left[i][0]}" y1="{left[i][1]}" x2="{left[j][0]}" y2="{left[j][1]}" class="edge"/>')
    for i, (x, y) in enumerate(left):
        b.append(_node(x, y, str(i+1), radius=25))
    b.append(_arrow(325, 245, 405, 245, purple=True))
    b.append(_round_box(420, 180, 110, 130, bfs.AMBER_LIGHT, bfs.AMBER, 24))
    b.append(_text(475, 218, "h(L)", "label"))
    b.append(_text(475, 253, "spectral", "small"))
    b.append(_text(475, 278, "filter", "small"))
    b.append(_text(310, 365, "Eigenvalues define frequency; polynomials in L become local.", "body"))
    b.append(_text(310, 397, "The basis depends on this particular graph.", "small"))

    # Permuted feature columns and pool/broadcast icon.
    for k, color in enumerate([bfs.PURPLE, bfs.AMBER, bfs.TEAL, bfs.BLUE]):
        b.append(f'<rect x="{690+k*39}" y="183" width="28" height="{60+k*12}" rx="9" fill="{color}"/>')
    b.append(_arrow(850, 240, 925, 240, purple=True))
    b.append(_round_box(945, 170, 155, 145, bfs.GREEN_LIGHT, bfs.GREEN, 26))
    b.append(_text(1022, 210, "αI + β11ᵀ", "label"))
    b.append(_text(1022, 248, "keep each node", "small"))
    b.append(_text(1022, 275, "+ pool globally", "small"))
    b.append(_text(890, 365, "The fixed-point space gives all equivariant linear maps.", "body"))
    b.append(_text(890, 397, "Adjacency enters only through an extra equivariant product AX.", "small"))

    b.append(_round_box(300, 452, 600, 50, bfs.AMBER_LIGHT, bfs.AMBER, 18))
    b.append(_text(600, 484, "Message passing combines shared node maps with the graph-dependent product AX.", "label"))
    _save("gconv_two_derivations", "\n".join(b))


def main() -> None:
    translation_equivariance()
    spectral_filter()
    two_derivations()


if __name__ == "__main__":
    main()
