#!/usr/bin/env python3
"""Generate the custom figures for the symmetry and equivariance blog post.

All figures are original, post-specific synthesis diagrams drawn as native SVG.
They use a flat-icon visual language: rounded geometric forms, restrained solid
colors, consistent strokes, and no gradients or texture. They do not copy any
Flaticon asset, slide, or paper artwork. The mathematical ideas follow the
author's 2025 ML4Mol/GDL lectures and the sources cited in the post, especially
Bronstein et al. (2021). License: CC BY 4.0 with the surrounding blog post.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import blog_figure_style as bfs


OUT = ROOT / "assets" / "img" / "blog"
TEXT = bfs.TEXT
MUTED = bfs.MUTED
PURPLE = bfs.PURPLE
PURPLE_SOFT = bfs.PURPLE_SOFT
BLUE = bfs.BLUE
BLUE_LIGHT = bfs.BLUE_LIGHT
TEAL = bfs.TEAL
TEAL_LIGHT = bfs.TEAL_LIGHT
AMBER = bfs.AMBER
AMBER_LIGHT = bfs.AMBER_LIGHT
SPINE = bfs.SPINE


def svg_document(body: str, *, width: int = 960, height: int = 500) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {TEXT}; }}
      .title {{ font-size: 22px; font-weight: 700; }}
      .label {{ font-size: 18px; font-weight: 600; }}
      .body {{ font-size: 16px; }}
      .small {{ font-size: 14px; fill: {MUTED}; }}
      .arrow {{ stroke: {MUTED}; stroke-width: 2.4; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def atom(x: float, y: float, color: str, label: str = "") -> str:
    return f'<circle cx="{x}" cy="{y}" r="17" fill="{color}" stroke="white" stroke-width="3"/><text x="{x}" y="{y + 5}" text-anchor="middle" font-size="13" font-weight="700" fill="white" style="fill:white">{label}</text>'


def molecule(points: list[tuple[float, float]], labels=("A", "B", "C")) -> str:
    bonds = "".join(
        f'<line x1="{points[i][0]}" y1="{points[i][1]}" x2="{points[j][0]}" y2="{points[j][1]}" stroke="{SPINE}" stroke-width="7" stroke-linecap="round"/>'
        for i, j in ((0, 1), (1, 2))
    )
    colors = (PURPLE, BLUE, TEAL)
    return bonds + "".join(atom(x, y, c, lab) for (x, y), c, lab in zip(points, colors, labels))


def write(name: str, body: str, *, width: int = 960, height: int = 500) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"symeq_{name}.svg"
    png_path = OUT / f"symeq_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def make_orbit() -> None:
    configs = [
        [(130, 255), (185, 195), (245, 245)],
        [(420, 260), (360, 205), (405, 145)],
        [(665, 245), (725, 190), (785, 250)],
    ]
    body = f'''
      <text x="480" y="42" text-anchor="middle" class="title">One object, many coordinate descriptions</text>
      {molecule(configs[0])}
      {molecule(configs[1])}
      {molecule(configs[2], labels=("C", "B", "A"))}
      <path d="M 270 220 C 315 175, 335 175, 355 195" class="arrow"/>
      <text x="312" y="160" text-anchor="middle" class="small">rotate + translate</text>
      <path d="M 455 175 C 520 125, 590 150, 650 210" class="arrow"/>
      <text x="548" y="125" text-anchor="middle" class="small">relabel vertices</text>
      <rect x="90" y="330" width="700" height="72" rx="18" fill="{PURPLE_SOFT}" stroke="{PURPLE}" stroke-width="2"/>
      <text x="440" y="359" text-anchor="middle" class="label">same orbit under the symmetry group</text>
      <text x="440" y="384" text-anchor="middle" class="body">an invariant target assigns the same value to every configuration above</text>
      <path d="M 190 286 L 190 326" class="arrow"/><path d="M 395 286 L 395 326" class="arrow"/><path d="M 725 286 L 725 326" class="arrow"/>
    '''
    write("data_orbit", body)


def make_commuting_paths() -> None:
    def box(x, y, w, h, fill, title, subtitle):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{SPINE}" stroke-width="2"/><text x="{x+w/2}" y="{y+35}" text-anchor="middle" class="label">{title}</text><text x="{x+w/2}" y="{y+62}" text-anchor="middle" class="small">{subtitle}</text>'
    body = f'''
      <text x="480" y="42" text-anchor="middle" class="title">Equivariance makes two computational paths agree</text>
      {box(80, 95, 220, 88, BLUE_LIGHT, 'input x', 'coordinates and features')}
      {box(660, 95, 220, 88, TEAL_LIGHT, 'output f(x)', 'prediction')}
      {box(80, 330, 220, 88, BLUE_LIGHT, 'transformed input', 'ρₓ(g)x')}
      {box(660, 330, 220, 88, TEAL_LIGHT, 'transformed output', 'ρᵧ(g)f(x)')}
      <path d="M 305 139 L 650 139" class="arrow"/><text x="478" y="124" text-anchor="middle" class="small">apply f</text>
      <path d="M 305 374 L 650 374" class="arrow"/><text x="478" y="359" text-anchor="middle" class="small">apply f</text>
      <path d="M 190 188 L 190 320" class="arrow"/><text x="210" y="258" class="small">act by g</text>
      <path d="M 770 188 L 770 320" class="arrow"/><text x="790" y="258" class="small">act by g</text>
      <rect x="356" y="212" width="250" height="74" rx="16" fill="{PURPLE_SOFT}" stroke="{PURPLE}" stroke-width="2"/>
      <text x="481" y="243" text-anchor="middle" class="label">f(ρₓ(g)x) = ρᵧ(g)f(x)</text>
      <text x="481" y="268" text-anchor="middle" class="small">the diagram commutes</text>
    '''
    write("commuting_paths", body)


def make_feature_types() -> None:
    body = f'''
      <text x="480" y="42" text-anchor="middle" class="title">A feature type is a rule for how values transform</text>
      <rect x="55" y="95" width="250" height="300" rx="18" fill="{PURPLE_SOFT}" stroke="{SPINE}" stroke-width="2"/>
      <text x="180" y="132" text-anchor="middle" class="label">Scalar</text>
      <circle cx="180" cy="225" r="56" fill="white" stroke="{PURPLE}" stroke-width="3"/>
      <text x="180" y="234" text-anchor="middle" font-size="28" font-weight="700">3.7</text>
      <path d="M 127 315 C 150 345, 210 345, 233 315" class="arrow"/>
      <text x="180" y="375" text-anchor="middle" class="small">unchanged by rotation</text>

      <rect x="355" y="95" width="250" height="300" rx="18" fill="{BLUE_LIGHT}" stroke="{SPINE}" stroke-width="2"/>
      <text x="480" y="132" text-anchor="middle" class="label">Vector</text>
      <circle cx="480" cy="245" r="5" fill="{TEXT}"/>
      <path d="M 480 245 L 480 165" stroke="{BLUE}" stroke-width="5" marker-end="url(#arrow)"/>
      <path d="M 480 245 L 548 205" stroke="{TEAL}" stroke-width="5" marker-end="url(#arrow)"/>
      <path d="M 495 181 A 70 70 0 0 1 538 215" fill="none" stroke="{MUTED}" stroke-width="2"/>
      <text x="480" y="375" text-anchor="middle" class="small">direction rotates with the input</text>

      <rect x="655" y="95" width="250" height="300" rx="18" fill="{TEAL_LIGHT}" stroke="{SPINE}" stroke-width="2"/>
      <text x="780" y="132" text-anchor="middle" class="label">Steerable feature</text>
      <path d="M 705 250 C 730 160, 755 160, 780 250 S 830 340, 855 250" fill="none" stroke="{BLUE}" stroke-width="4"/>
      <path d="M 705 250 C 730 340, 755 340, 780 250 S 830 160, 855 250" fill="none" stroke="{TEAL}" stroke-width="4"/>
      <text x="780" y="330" text-anchor="middle" class="body">[a, b] → D(g)[a, b]</text>
      <text x="780" y="375" text-anchor="middle" class="small">components mix by a representation</text>
    '''
    write("feature_types", body)


def make_hypothesis_class() -> None:
    body = f'''
      <text x="480" y="42" text-anchor="middle" class="title">Architectural symmetry removes functions that contradict the data geometry</text>
      <ellipse cx="455" cy="245" rx="355" ry="165" fill="{BLUE_LIGHT}" stroke="{BLUE}" stroke-width="3"/>
      <text x="240" y="125" text-anchor="middle" class="label">all functions the network can express</text>
      <path d="M 205 280 C 255 170, 315 330, 370 215 S 475 305, 535 190" fill="none" stroke="{SPINE}" stroke-width="3"/>
      <path d="M 160 325 C 245 245, 315 360, 395 275" fill="none" stroke="{SPINE}" stroke-width="3"/>
      <ellipse cx="600" cy="258" rx="170" ry="105" fill="{PURPLE_SOFT}" stroke="{PURPLE}" stroke-width="3"/>
      <text x="600" y="232" text-anchor="middle" class="label">equivariant functions</text>
      <text x="600" y="264" text-anchor="middle" class="body">f(ρₓ(g)x) = ρᵧ(g)f(x)</text>
      <text x="600" y="294" text-anchor="middle" class="small">one prediction constrains an entire orbit</text>
      <path d="M 826 190 C 900 210, 900 300, 827 320" class="arrow"/>
      <text x="875" y="167" text-anchor="middle" class="small">built-in constraint</text>
      <rect x="210" y="435" width="540" height="42" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="480" y="462" text-anchor="middle" class="body">less freedom can improve generalization when the symmetry is correct</text>
    '''
    write("hypothesis_class", body)


if __name__ == "__main__":
    make_orbit()
    make_commuting_paths()
    make_feature_types()
    make_hypothesis_class()
