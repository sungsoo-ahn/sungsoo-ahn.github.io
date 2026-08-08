#!/usr/bin/env python3
"""Generate original figures for the frames and symmetrization blog post.

All figures are original explanatory diagrams, saved as editable SVG with PNG
previews. They use a flat-icon visual language: rounded geometry, restrained
solid colors, consistent strokes, and little decorative detail. No artwork from
the lecture deck or cited papers is reproduced.

Conceptual sources
------------------
* Du et al. (2022), https://proceedings.mlr.press/v162/du22e.html
* Puny et al. (2022), https://arxiv.org/abs/2110.03336
* Kaba et al. (2023), https://proceedings.mlr.press/v202/kaba23a.html
* Kim et al. (2023), https://proceedings.neurips.cc/paper_files/paper/2023/
  hash/ca2d1c4ca195407c1ca9a7e47b9dc27f-Abstract-Conference.html
* Dym et al. (2024), https://proceedings.mlr.press/v235/dym24a.html

License: same as the blog.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


OUT = Path("assets/img/blog")
WIDTH = 1200


def svg_document(body: str, height: int = 540) -> str:
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
      .edge {{ fill: none; stroke: {bfs.SPINE}; stroke-width: 4; stroke-linecap: round; }}
      .axis {{ fill: none; stroke-width: 4; stroke-linecap: round; marker-end: url(#arrow); }}
      .arrow {{ fill: none; stroke: {bfs.MUTED}; stroke-width: 3; marker-end: url(#arrow); }}
      .parrow {{ fill: none; stroke: {bfs.PURPLE}; stroke-width: 3; marker-end: url(#purpleArrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x: float, y: float, w: float, h: float, fill: str, stroke: str, radius: int = 24) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, *, purple: bool = False) -> str:
    cls = "parrow" if purple else "arrow"
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}"/>'


def point(x: float, y: float, color: str, radius: int = 12) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="3"/>'


def triangle(cx: float, cy: float, angle: float = 0.0, scale: float = 1.0) -> str:
    raw = [(-46, 35), (52, 23), (-5, -53)]
    colors = [bfs.PURPLE, bfs.AMBER, bfs.TEAL]
    pts = []
    nodes = []
    c, s = math.cos(angle), math.sin(angle)
    for (x, y), color in zip(raw, colors):
        xr = scale * (c * x - s * y) + cx
        yr = scale * (s * x + c * y) + cy
        pts.append((xr, yr))
        nodes.append(point(xr, yr, color, int(12 * scale)))
    path = " ".join(f"{x},{y}" for x, y in pts)
    return f'<polygon points="{path}" fill="{bfs.PURPLE_LIGHT}" stroke="{bfs.SPINE}" stroke-width="4" stroke-linejoin="round"/>' + "".join(nodes)


def save(stem: str, body: str, height: int = 540) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(svg_document(body, height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=WIDTH):
        raise RuntimeError(f"Could not render {svg_path}")


def orbit_and_quotient() -> None:
    """Show an orbit of poses and a canonical representative in the quotient."""
    b = [text(60, 48, "Canonicalization chooses one representative from an orbit", "title", "start")]
    b.append(f'<ellipse cx="340" cy="275" rx="250" ry="155" fill="none" stroke="{bfs.GRID}" stroke-width="5" stroke-dasharray="10 10"/>')
    for cx, cy, ang in [(155, 230, -0.8), (330, 130, 0.25), (505, 250, 1.2), (330, 395, 2.6)]:
        b.append(triangle(cx, cy, ang, 0.72))
    b.append(text(340, 495, "one object, many coordinate descriptions", "label"))
    b.append(text(340, 520, "orbit G·x", "small"))
    b.append(arrow(590, 275, 735, 275, purple=True))
    b.append(text(660, 245, "quotient", "small"))
    b.append(text(660, 300, "choose a section", "small"))
    b.append(box(780, 120, 330, 310, bfs.PURPLE_SOFT, bfs.PURPLE, 32))
    b.append(triangle(945, 265, 0.0, 1.2))
    b.append(text(945, 160, "canonical representative", "label"))
    b.append(text(945, 385, "c(x) = h(x)⁻¹·x", "label"))
    b.append(text(945, 463, "Invariant predictions can be made on the representative.", "body"))
    b.append(text(945, 490, "Equivariant predictions restore the original pose.", "small"))
    save("framesym_orbit_quotient", "\n".join(b), 550)


def canonicalization_discontinuity() -> None:
    """Show why PCA frames become unstable near repeated eigenvalues."""
    b = [text(60, 48, "A canonical frame becomes unstable near symmetric shapes", "title", "start")]
    panel_data = [
        (45, "distinct eigenvalues", "stable principal axis", 0.0, bfs.GREEN_LIGHT, bfs.GREEN),
        (430, "nearly repeated", "tiny perturbation", 0.0, bfs.AMBER_LIGHT, bfs.AMBER),
        (815, "nearly repeated", "axis switches", math.pi / 2, bfs.RED_LIGHT, bfs.RED),
    ]
    for x0, title1, title2, angle, fill, stroke in panel_data:
        b.append(box(x0, 88, 340, 355, fill, stroke, 28))
        b.append(text(x0 + 170, 127, title1, "label"))
        b.append(text(x0 + 170, 153, title2, "small"))
        cx, cy = x0 + 170, 280
        # Elliptical cloud; the last two are nearly circular.
        sx, sy = ((110, 48) if x0 == 45 else (82, 76))
        pts = []
        for k in range(14):
            t = 2 * math.pi * k / 14
            jitter = 6 * math.sin(3 * t + (0.7 if x0 == 815 else 0.0))
            px = (sx + jitter) * math.cos(t)
            py = (sy - jitter / 2) * math.sin(t)
            c, s = math.cos(angle), math.sin(angle)
            pts.append((cx + c * px - s * py, cy + s * px + c * py))
        for k, (px, py) in enumerate(pts):
            b.append(point(px, py, bfs.PURPLE if k % 2 == 0 else bfs.BLUE, 9))
        # Principal axes.
        c, s = math.cos(angle), math.sin(angle)
        b.append(f'<line x1="{cx-125*c}" y1="{cy-125*s}" x2="{cx+125*c}" y2="{cy+125*s}" stroke="{stroke}" stroke-width="5" stroke-linecap="round"/>')
        b.append(f'<line x1="{cx+75*s}" y1="{cy-75*c}" x2="{cx-75*s}" y2="{cy+75*c}" stroke="{bfs.MUTED}" stroke-width="3" stroke-linecap="round"/>')
    b.append(arrow(390, 270, 422, 270))
    b.append(arrow(775, 270, 807, 270))
    b.append(text(600, 480, "At an exact eigenvalue tie, every basis of the repeated eigenspace is valid.", "body"))
    b.append(text(600, 510, "A deterministic choice must jump somewhere as the shape crosses the degeneracy.", "label"))
    save("framesym_canonicalization_discontinuity", "\n".join(b), 545)


def local_frame_scalarization() -> None:
    """Show scalarization and vectorization through a local orthonormal frame."""
    b = [text(60, 48, "A local frame converts geometric vectors into invariant coordinates", "title", "start")]
    # Geometry and triad.
    ox, oy = 225, 285
    b.append(point(ox, oy, bfs.TEXT, 15))
    b.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+145}" y2="{oy-25}" stroke="{bfs.PURPLE}" class="axis"/>')
    b.append(f'<line x1="{ox}" y1="{oy}" x2="{ox-45}" y2="{oy-125}" stroke="{bfs.TEAL}" class="axis"/>')
    b.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+60}" y2="{oy+115}" stroke="{bfs.AMBER}" class="axis"/>')
    b.append(text(385, 255, "e₁", "label"))
    b.append(text(170, 145, "e₂", "label"))
    b.append(text(295, 420, "e₃", "label"))
    b.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+60}" y2="{oy-95}" stroke="{bfs.RED}" stroke-width="6" marker-end="url(#arrow)"/>')
    b.append(text(300, 175, "v", "label"))
    b.append(text(225, 470, "frame from two non-collinear relative vectors", "small"))

    b.append(arrow(420, 285, 520, 285, purple=True))
    b.append(box(540, 160, 220, 250, bfs.PURPLE_SOFT, bfs.PURPLE, 28))
    b.append(text(650, 202, "scalarize", "label"))
    b.append(text(650, 252, "s₁ = v·e₁", "body"))
    b.append(text(650, 292, "s₂ = v·e₂", "body"))
    b.append(text(650, 332, "s₃ = v·e₃", "body"))
    b.append(text(650, 375, "ordinary nonlinear network", "small"))
    b.append(arrow(780, 285, 870, 285, purple=True))
    b.append(box(890, 160, 250, 250, bfs.GREEN_LIGHT, bfs.GREEN, 28))
    b.append(text(1015, 202, "vectorize", "label"))
    b.append(text(1015, 260, "v′ = a₁e₁ + a₂e₂ + a₃e₃", "body"))
    b.append(f'<line x1="960" y1="350" x2="1065" y2="315" stroke="{bfs.GREEN}" stroke-width="7" marker-end="url(#arrow)"/>')
    b.append(text(1015, 390, "output rotates with the frame", "small"))
    b.append(text(600, 515, "Dot products are rotation invariant because the vector and every frame axis rotate together.", "body"))
    save("framesym_local_frame", "\n".join(b), 550)


def averaging_recipes() -> None:
    """Contrast group averaging, finite frame averaging, and one-pose canonicalization."""
    b = [text(60, 48, "Symmetrization trades backbone evaluations for fewer pose choices", "title", "start")]
    panels = [
        (35, "group average", "all poses", "exact but expensive", bfs.PURPLE_SOFT, bfs.PURPLE),
        (420, "frame average", "finite equivariant set", "exact if the frame law holds", bfs.BLUE_LIGHT, bfs.BLUE),
        (805, "canonicalize", "one selected pose", "fast but can be discontinuous", bfs.GREEN_LIGHT, bfs.GREEN),
    ]
    for x0, title1, title2, foot, fill, stroke in panels:
        b.append(box(x0, 88, 350, 370, fill, stroke, 28))
        b.append(text(x0 + 175, 130, title1, "label"))
        b.append(text(x0 + 175, 157, title2, "small"))
        b.append(box(x0 + 115, 245, 120, 65, "white", stroke, 20))
        b.append(text(x0 + 175, 284, "backbone φ", "label"))
        if title1 == "group average":
            count = 10
            for k in range(count):
                a = 2 * math.pi * k / count
                cx, cy = x0 + 175 + 105 * math.cos(a), 278 + 92 * math.sin(a)
                b.append(triangle(cx, cy, a, 0.24))
                b.append(arrow(cx, cy, x0 + 103 if cx < x0 + 175 else x0 + 247, 278))
        elif title1 == "frame average":
            for cx, cy, a in [(x0+80,220,0), (x0+270,220,math.pi/2), (x0+80,350,math.pi), (x0+270,350,3*math.pi/2)]:
                b.append(triangle(cx, cy, a, 0.34))
                b.append(arrow(cx, cy, x0 + 103 if cx < x0 + 175 else x0 + 247, 278))
        else:
            b.append(triangle(x0 + 175, 205, 0.65, 0.45))
            b.append(arrow(x0 + 175, 230, x0 + 175, 243, purple=True))
            b.append(triangle(x0 + 175, 370, 0.0, 0.45))
        b.append(text(x0 + 175, 425, foot, "small"))
    b.append(text(600, 505, "Each recipe evaluates an arbitrary backbone on transformed inputs, then averages or restores the output pose.", "body"))
    save("framesym_averaging_recipes", "\n".join(b), 540)


def main() -> None:
    orbit_and_quotient()
    canonicalization_discontinuity()
    local_frame_scalarization()
    averaging_recipes()


if __name__ == "__main__":
    main()
