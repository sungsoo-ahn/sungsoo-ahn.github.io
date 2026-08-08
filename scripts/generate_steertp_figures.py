#!/usr/bin/env python3
"""Generate original figures for the steerable tensor-product blog post.

All outputs are native SVG with PNG previews. They use a flat-icon visual
language: rounded geometric forms, restrained solid colors, consistent strokes,
and no gradients or texture. No Flaticon asset, lecture-slide figure, or paper
figure is copied. The diagrams synthesize standard SO(3) representation theory
and the primary papers cited in the post. License: CC BY 4.0 with the blog post.
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
      .arrow {{ stroke: {MUTED}; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def box(x: float, y: float, w: float, h: float, fill: str, *, radius: int = 20, stroke: str = SPINE) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def node(x: float, y: float, color: str, *, radius: int = 13) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="3"/>'


def write(name: str, body: str, *, width: int = 1000, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"steertp_{name}.svg"
    png_path = OUT / f"steertp_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def make_feature_ladder() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Irreducible feature types are the rotation-equivariant channels</text>
      {box(45,90,275,360,PURPLE_SOFT)}
      {box(362,90,275,360,BLUE_LIGHT)}
      {box(680,90,275,360,TEAL_LIGHT)}

      <text x="182" y="132" text-anchor="middle" class="label">type ℓ = 0</text>
      <circle cx="182" cy="245" r="62" fill="white" stroke="{PURPLE}" stroke-width="4"/>
      <text x="182" y="254" text-anchor="middle" font-size="30" font-weight="700">s</text>
      <path d="M 125 340 C 150 370, 215 370, 240 340" class="arrow"/>
      <text x="182" y="407" text-anchor="middle" class="body">dimension 1</text>
      <text x="182" y="432" text-anchor="middle" class="small">scalar stays fixed</text>

      <text x="500" y="132" text-anchor="middle" class="label">type ℓ = 1</text>
      <circle cx="500" cy="260" r="6" fill="{TEXT}"/>
      <path d="M 500 260 L 500 165" stroke="{BLUE}" stroke-width="6" marker-end="url(#arrow)"/>
      <path d="M 500 260 L 575 205" stroke="{TEAL}" stroke-width="6" marker-end="url(#arrow)"/>
      <path d="M 515 180 A 85 85 0 0 1 563 214" fill="none" stroke="{MUTED}" stroke-width="2"/>
      <text x="500" y="407" text-anchor="middle" class="body">dimension 3</text>
      <text x="500" y="432" text-anchor="middle" class="small">ordinary vector rotates</text>

      <text x="818" y="132" text-anchor="middle" class="label">type ℓ = 2</text>
      <ellipse cx="818" cy="245" rx="92" ry="48" fill="{TEAL}" opacity=".20" stroke="{TEAL}" stroke-width="4"/>
      <ellipse cx="818" cy="245" rx="48" ry="92" fill="{PURPLE}" opacity=".16" stroke="{PURPLE}" stroke-width="4" transform="rotate(35 818 245)"/>
      <circle cx="818" cy="245" r="8" fill="{TEXT}"/>
      <text x="818" y="407" text-anchor="middle" class="body">dimension 5</text>
      <text x="818" y="432" text-anchor="middle" class="small">quadrupolar pattern rotates</text>

      <rect x="185" y="480" width="630" height="46" rx="13" fill="{AMBER_LIGHT}"/>
      <text x="500" y="509" text-anchor="middle" class="body">f⁽ℓ⁾ → D⁽ℓ⁾(R) f⁽ℓ⁾, with 2ℓ + 1 coupled components</text>
    '''
    write("feature_ladder", body)


def make_coupling_rules() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">The Clebsch–Gordan rule determines which output types can exist</text>
      {box(60,92,240,104,BLUE_LIGHT)}
      {box(60,242,240,104,TEAL_LIGHT)}
      <text x="180" y="134" text-anchor="middle" class="label">input type ℓ₁</text>
      <text x="180" y="169" text-anchor="middle" class="body">2ℓ₁ + 1 components</text>
      <text x="180" y="284" text-anchor="middle" class="label">input type ℓ₂</text>
      <text x="180" y="319" text-anchor="middle" class="body">2ℓ₂ + 1 components</text>
      <path d="M 310 144 C 370 144, 370 206, 420 218" class="arrow"/>
      <path d="M 310 294 C 370 294, 370 242, 420 230" class="arrow"/>
      <circle cx="465" cy="225" r="53" fill="{PURPLE_SOFT}" stroke="{PURPLE}" stroke-width="3"/>
      <text x="465" y="217" text-anchor="middle" class="label">CG</text>
      <text x="465" y="243" text-anchor="middle" class="small">couple</text>
      <path d="M 522 225 L 628 225" class="arrow"/>
      {box(648,82,290,310,PURPLE_SOFT)}
      <text x="793" y="123" text-anchor="middle" class="label">allowed output types</text>
      <text x="793" y="165" text-anchor="middle" class="body">|ℓ₁ − ℓ₂| ≤ ℓ ≤ ℓ₁ + ℓ₂</text>
      <text x="793" y="204" text-anchor="middle" class="small">integer steps</text>
      <line x1="700" y1="239" x2="885" y2="239" stroke="{SPINE}" stroke-width="2"/>
      <text x="793" y="278" text-anchor="middle" class="body">m = m₁ + m₂</text>
      <text x="793" y="317" text-anchor="middle" class="small">nonzero coefficients only</text>
      <text x="793" y="357" text-anchor="middle" class="body">parity: p = p₁p₂</text>

      <rect x="70" y="430" width="245" height="80" rx="16" fill="{BLUE_LIGHT}"/>
      <text x="193" y="463" text-anchor="middle" class="body">0 ⊗ 2 = 2</text>
      <text x="193" y="489" text-anchor="middle" class="small">scalar scaling</text>
      <rect x="378" y="430" width="245" height="80" rx="16" fill="{PURPLE_SOFT}"/>
      <text x="500" y="463" text-anchor="middle" class="body">1 ⊗ 1 = 0 ⊕ 1 ⊕ 2</text>
      <text x="500" y="489" text-anchor="middle" class="small">dot, cross, quadrupole</text>
      <rect x="685" y="430" width="245" height="80" rx="16" fill="{TEAL_LIGHT}"/>
      <text x="807" y="463" text-anchor="middle" class="body">1 ⊗ 2 = 1 ⊕ 2 ⊕ 3</text>
      <text x="807" y="489" text-anchor="middle" class="small">three allowed couplings</text>
    '''
    write("coupling_rules", body)


def make_vector_decomposition() -> None:
    cells = ""
    x0, y0, size = 75, 170, 52
    shades = [BLUE_LIGHT, TEAL_LIGHT, PURPLE_SOFT]
    for i in range(3):
        for j in range(3):
            cells += f'<rect x="{x0+j*size}" y="{y0+i*size}" width="{size}" height="{size}" fill="{shades[(i+j)%3]}" stroke="white" stroke-width="3"/>'
            cells += f'<text x="{x0+j*size+size/2}" y="{y0+i*size+32}" text-anchor="middle" class="body">x{i+1}y{j+1}</text>'
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Two vectors contain exactly three irreducible geometric products</text>
      <text x="153" y="110" text-anchor="middle" class="label">outer product x yᵀ</text>
      {cells}
      <text x="153" y="365" text-anchor="middle" class="body">9 Cartesian components</text>
      <path d="M 250 248 L 348 248" class="arrow"/>

      {box(375,80,560,118,PURPLE_SOFT)}
      <circle cx="430" cy="139" r="30" fill="white" stroke="{PURPLE}" stroke-width="3"/>
      <text x="430" y="147" text-anchor="middle" class="label">ℓ=0</text>
      <text x="500" y="130" class="body">scalar trace: x · y</text>
      <text x="500" y="159" class="small">1 component; invariant under rotation</text>

      {box(375,220,560,118,BLUE_LIGHT)}
      <path d="M 420 290 L 450 252" stroke="{BLUE}" stroke-width="5" marker-end="url(#arrow)"/>
      <path d="M 420 290 L 460 300" stroke="{TEAL}" stroke-width="5" marker-end="url(#arrow)"/>
      <text x="500" y="270" class="body">antisymmetric part: x × y</text>
      <text x="500" y="299" class="small">3 components; transforms as a vector</text>

      {box(375,360,560,118,TEAL_LIGHT)}
      <ellipse cx="435" cy="419" rx="44" ry="23" fill="{TEAL}" opacity=".22" stroke="{TEAL}" stroke-width="3"/>
      <text x="500" y="405" class="body">symmetric traceless part</text>
      <text x="500" y="434" class="small">½(xyᵀ + yxᵀ) − ⅓(x · y)I</text>
      <text x="500" y="459" class="small">5 components; transforms as type ℓ=2</text>

      <rect x="200" y="505" width="600" height="38" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="500" y="530" text-anchor="middle" class="body">9 = 1 + 3 + 5, so 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2</text>
    '''
    write("vector_decomposition", body)


def make_layer_pipeline() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">One steerable message-passing layer couples features with edge geometry</text>
      {box(35,95,170,110,BLUE_LIGHT)}
      <text x="120" y="134" text-anchor="middle" class="label">neighbor j</text>
      <text x="120" y="166" text-anchor="middle" class="body">typed hⱼ⁽ℓⁱⁿ⁾</text>
      <text x="120" y="190" text-anchor="middle" class="small">learned channels</text>

      {box(35,280,170,155,TEAL_LIGHT)}
      <text x="120" y="317" text-anchor="middle" class="label">edge rᵢⱼ</text>
      <line x1="80" y1="365" x2="158" y2="335" stroke="{TEAL}" stroke-width="5" marker-end="url(#arrow)"/>
      <text x="120" y="405" text-anchor="middle" class="small">distance + direction</text>

      <path d="M 215 350 L 285 350" class="arrow"/>
      {box(300,270,190,175,PURPLE_SOFT)}
      <text x="395" y="307" text-anchor="middle" class="label">steerable filter</text>
      <text x="395" y="345" text-anchor="middle" class="body">R(rᵢⱼ) Y⁽ℓᶠ⁾(r̂ᵢⱼ)</text>
      <text x="395" y="381" text-anchor="middle" class="small">radial scalar</text>
      <text x="395" y="407" text-anchor="middle" class="small">× angular feature</text>

      <path d="M 215 150 C 350 150, 470 165, 540 205" class="arrow"/>
      <path d="M 495 350 C 530 350, 530 270, 550 248" class="arrow"/>
      <circle cx="600" cy="225" r="62" fill="{AMBER_LIGHT}" stroke="{AMBER}" stroke-width="3"/>
      <text x="600" y="218" text-anchor="middle" class="label">CG</text>
      <text x="600" y="244" text-anchor="middle" class="small">allowed ℓout</text>

      <path d="M 665 225 L 735 225" class="arrow"/>
      {box(750,165,205,120,BLUE_LIGHT)}
      <text x="853" y="205" text-anchor="middle" class="label">sum neighbors</text>
      <text x="853" y="239" text-anchor="middle" class="body">Σⱼ mᵢⱼ⁽ℓᵒᵘᵗ⁾</text>
      <text x="853" y="265" text-anchor="middle" class="small">permutation invariant</text>

      <path d="M 853 295 L 853 350" class="arrow"/>
      {box(750,370,205,120,TEAL_LIGHT)}
      <text x="853" y="410" text-anchor="middle" class="label">mix + gate</text>
      <text x="853" y="444" text-anchor="middle" class="body">g⁽⁰⁾ h⁽ℓ⁾</text>
      <text x="853" y="470" text-anchor="middle" class="small">equivariant output</text>

      <rect x="300" y="490" width="390" height="42" rx="12" fill="{PURPLE_SOFT}"/>
      <text x="495" y="517" text-anchor="middle" class="body">|ℓin − ℓf| ≤ ℓout ≤ ℓin + ℓf</text>
    '''
    write("layer_pipeline", body)


if __name__ == "__main__":
    make_feature_ladder()
    make_coupling_rules()
    make_vector_decomposition()
    make_layer_pipeline()
