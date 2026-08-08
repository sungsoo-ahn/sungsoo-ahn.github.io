#!/usr/bin/env python3
"""Generate original figures for the AlphaFold structure-prediction post.

All outputs are native SVG with PNG previews. Figures use a flat-icon visual
language with rounded forms, restrained solid colors, consistent strokes, and
no gradients. No Flaticon asset, lecture-slide figure, or paper figure is
copied. The diagrams are original explanatory syntheses of concepts described
in the primary sources cited by the post. License: CC BY 4.0 with the blog post.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import blog_figure_style as bfs

OUT = ROOT / "assets" / "img" / "blog"
TEXT, MUTED, SPINE = bfs.TEXT, bfs.MUTED, bfs.SPINE
PURPLE, PURPLE_SOFT = bfs.PURPLE, bfs.PURPLE_SOFT
BLUE, BLUE_LIGHT = bfs.BLUE, bfs.BLUE_LIGHT
TEAL, TEAL_LIGHT = bfs.TEAL, bfs.TEAL_LIGHT
AMBER, AMBER_LIGHT = bfs.AMBER, bfs.AMBER_LIGHT
RED, RED_LIGHT = bfs.RED, bfs.RED_LIGHT


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


def box(x: float, y: float, w: float, h: float, fill: str, *, stroke: str = SPINE, radius: int = 20) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def dot(x: float, y: float, color: str, *, r: int = 11) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="white" stroke-width="2"/>'


def write(name: str, body: str, *, width: int = 1000, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"afstruct_{name}.svg"
    png_path = OUT / f"afstruct_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def make_msa_coevolution() -> None:
    letters = ["AELKQMT", "ADLKQIT", "VELRNMT", "VDLRNIT", "AELKQMT", "VELRNIT"]
    colors = [PURPLE_SOFT, BLUE_LIGHT, TEAL_LIGHT, AMBER_LIGHT]
    grid = ""
    for row, seq in enumerate(letters):
        for col, aa in enumerate(seq):
            fill = colors[(ord(aa) + col) % len(colors)]
            if col in (1, 5):
                fill = RED_LIGHT if aa in "DE" else TEAL_LIGHT
            x, y = 70 + col * 45, 150 + row * 39
            grid += f'<rect x="{x}" y="{y}" width="39" height="32" rx="7" fill="{fill}"/><text x="{x+19.5}" y="{y+22}" text-anchor="middle" class="small">{aa}</text>'
    chain_pts = [(650,190),(700,155),(760,170),(805,225),(770,285),(700,305),(645,270),(620,225)]
    chain = '<polyline points="' + ' '.join(f'{x},{y}' for x,y in chain_pts) + f'" fill="none" stroke="{MUTED}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
    chain += "".join(dot(x,y,[PURPLE,BLUE,TEAL,AMBER][i%4],r=13) for i,(x,y) in enumerate(chain_pts))
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Evolution turns a sequence family into geometric constraints</text>
      {box(35,88,400,375,BLUE_LIGHT)}
      <text x="235" y="126" text-anchor="middle" class="label">multiple-sequence alignment</text>
      {grid}
      <rect x="112" y="139" width="39" height="278" rx="9" fill="none" stroke="{RED}" stroke-width="3"/>
      <rect x="292" y="139" width="39" height="278" rx="9" fill="none" stroke="{TEAL}" stroke-width="3"/>
      <path d="M 151 427 C 195 459, 275 459, 292 427" fill="none" stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow)"/>
      <text x="220" y="448" text-anchor="middle" class="small">coupled substitutions across homologs</text>

      <path d="M 447 275 L 525 275" class="arrow"/>
      <text x="486" y="252" text-anchor="middle" class="small">infer contact</text>
      {box(545,88,420,375,TEAL_LIGHT)}
      <text x="755" y="126" text-anchor="middle" class="label">folded residue graph</text>
      {chain}
      <line x1="700" y1="155" x2="700" y2="305" stroke="{RED}" stroke-width="4" stroke-dasharray="8 6"/>
      <text x="835" y="340" text-anchor="middle" class="body">far in sequence</text>
      <text x="835" y="366" text-anchor="middle" class="body">near in 3D</text>
      <text x="755" y="420" text-anchor="middle" class="small">coevolution is evidence, not a literal force</text>

      <rect x="175" y="496" width="650" height="42" rx="13" fill="{PURPLE_SOFT}"/>
      <text x="500" y="522" text-anchor="middle" class="body">A deep, diverse family records which residue combinations remain compatible with a fold.</text>
    '''
    write("msa_coevolution", body)


def make_pair_reasoning() -> None:
    msa = ""
    for i in range(5):
        for j in range(7):
            msa += f'<rect x="{65+j*28}" y="{155+i*28}" width="24" height="24" rx="5" fill="{[PURPLE_SOFT,BLUE_LIGHT,TEAL_LIGHT][(i+j)%3]}"/>'
    matrix = ""
    for i in range(7):
        for j in range(7):
            fill = PURPLE_SOFT if abs(i-j)<2 else (TEAL_LIGHT if (i+j)%3==0 else "white")
            matrix += f'<rect x="{370+j*29}" y="{145+i*29}" width="26" height="26" rx="4" fill="{fill}" stroke="{SPINE}" stroke-width=".5"/>'
    chain = f'<path d="M 730 295 C 750 205, 820 185, 850 235 C 885 290, 830 350, 765 330 C 720 315, 715 260, 755 240" fill="none" stroke="{TEAL}" stroke-width="8" stroke-linecap="round"/>'
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">AlphaFold 2 alternates family evidence, pair geometry, and coordinates</text>
      {box(35,92,260,365,BLUE_LIGHT)}
      <text x="165" y="130" text-anchor="middle" class="label">MSA representation</text>
      {msa}
      <text x="165" y="330" text-anchor="middle" class="body">features m[s, i]</text>
      <text x="165" y="365" text-anchor="middle" class="small">attention across sequences</text>
      <text x="165" y="390" text-anchor="middle" class="small">and residue positions</text>

      <path d="M 303 265 L 342 265" class="arrow"/>
      {box(350,92,260,365,PURPLE_SOFT)}
      <text x="480" y="130" text-anchor="middle" class="label">pair representation</text>
      {matrix}
      <path d="M 395 362 L 480 395 L 565 362" fill="none" stroke="{PURPLE}" stroke-width="3"/>
      <text x="480" y="424" text-anchor="middle" class="small">triangle updates compare i-k-j paths</text>

      <path d="M 618 265 L 657 265" class="arrow"/>
      {box(665,92,300,365,TEAL_LIGHT)}
      <text x="815" y="130" text-anchor="middle" class="label">structure module</text>
      {chain}
      <circle cx="755" cy="240" r="8" fill="{RED}"/><circle cx="850" cy="235" r="8" fill="{AMBER}"/>
      <text x="815" y="390" text-anchor="middle" class="body">residue frames and atom coordinates</text>
      <text x="815" y="420" text-anchor="middle" class="small">geometry returns through recycling</text>

      <path d="M 815 468 C 730 525, 260 525, 165 468" fill="none" stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow)"/>
      <text x="500" y="519" text-anchor="middle" class="small">recycle the prediction to revise sequence and pair features</text>
    '''
    write("pair_reasoning", body)


def make_decoder_shift() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">The decisive shift is how learned geometry becomes coordinates</text>
      {box(35,90,290,390,BLUE_LIGHT)}
      <text x="180" y="130" text-anchor="middle" class="label">AlphaFold 1</text>
      <rect x="78" y="163" width="204" height="70" rx="15" fill="white" stroke="{BLUE}" stroke-width="2"/>
      <text x="180" y="190" text-anchor="middle" class="body">distograms and torsions</text>
      <text x="180" y="216" text-anchor="middle" class="small">learn pairwise potentials</text>
      <path d="M 180 243 L 180 286" class="arrow"/>
      <path d="M 85 355 C 120 290, 230 300, 270 360" fill="none" stroke="{PURPLE}" stroke-width="7" stroke-linecap="round"/>
      <text x="180" y="405" text-anchor="middle" class="body">optimize coordinates</text>
      <text x="180" y="433" text-anchor="middle" class="small">network plus external solver</text>

      {box(355,90,290,390,PURPLE_SOFT)}
      <text x="500" y="130" text-anchor="middle" class="label">AlphaFold 2</text>
      <g transform="translate(500 220)"><line x1="0" y1="0" x2="58" y2="0" stroke="{RED}" stroke-width="5"/><line x1="0" y1="0" x2="0" y2="-58" stroke="{TEAL}" stroke-width="5"/><circle cx="0" cy="0" r="9" fill="{TEXT}"/></g>
      <g transform="translate(500 330) rotate(35)"><line x1="0" y1="0" x2="58" y2="0" stroke="{RED}" stroke-width="5"/><line x1="0" y1="0" x2="0" y2="-58" stroke="{TEAL}" stroke-width="5"/><circle cx="0" cy="0" r="9" fill="{TEXT}"/></g>
      <path d="M 500 245 L 500 286" class="arrow"/>
      <text x="500" y="405" text-anchor="middle" class="body">iterative residue frames</text>
      <text x="500" y="433" text-anchor="middle" class="small">end-to-end coordinate refinement</text>

      {box(675,90,290,390,TEAL_LIGHT)}
      <text x="820" y="130" text-anchor="middle" class="label">AlphaFold 3</text>
      <circle cx="820" cy="230" r="83" fill="white" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 6"/>
      <circle cx="786" cy="205" r="11" fill="{PURPLE}"/><circle cx="842" cy="190" r="11" fill="{RED}"/><circle cx="860" cy="246" r="11" fill="{AMBER}"/><circle cx="800" cy="267" r="11" fill="{TEAL}"/>
      <path d="M 786 205 L 842 190 L 860 246 L 800 267 Z" fill="none" stroke="{MUTED}" stroke-width="3"/>
      <path d="M 820 323 L 820 350" class="arrow"/>
      <text x="820" y="386" text-anchor="middle" class="body">denoise raw atom coordinates</text>
      <text x="820" y="414" text-anchor="middle" class="small">one decoder for mixed complexes</text>
      <text x="820" y="441" text-anchor="middle" class="small">protein, nucleic acid, ligand, ion</text>

      <rect x="160" y="505" width="680" height="38" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="500" y="530" text-anchor="middle" class="body">All three learn pair geometry; they differ in the inductive bias of coordinate construction.</text>
    '''
    write("decoder_shift", body)


def make_confidence_scope() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Confidence qualifies a structural hypothesis; it does not widen the claim</text>
      {box(35,90,280,360,BLUE_LIGHT)}
      <text x="175" y="130" text-anchor="middle" class="label">pLDDT: local confidence</text>
      <path d="M 78 260 C 110 170, 165 195, 190 260 C 215 330, 260 300, 280 205" fill="none" stroke="{BLUE}" stroke-width="8" stroke-linecap="round"/>
      <circle cx="110" cy="213" r="10" fill="{TEAL}"/><circle cx="190" cy="260" r="10" fill="{AMBER}"/><circle cx="255" cy="282" r="10" fill="{RED}"/>
      <text x="175" y="350" text-anchor="middle" class="body">which residues are locally reliable?</text>
      <text x="175" y="388" text-anchor="middle" class="small">low confidence can mean missing evidence,</text>
      <text x="175" y="411" text-anchor="middle" class="small">disorder, flexibility, or model uncertainty</text>

      {box(345,90,280,360,PURPLE_SOFT)}
      <text x="485" y="130" text-anchor="middle" class="label">PAE: relative placement</text>
      <rect x="405" y="170" width="160" height="160" fill="white" stroke="{SPINE}" stroke-width="2"/>
      <rect x="405" y="170" width="72" height="72" fill="{TEAL}" opacity=".8"/><rect x="493" y="258" width="72" height="72" fill="{TEAL}" opacity=".8"/>
      <rect x="493" y="170" width="72" height="72" fill="{AMBER_LIGHT}"/><rect x="405" y="258" width="72" height="72" fill="{AMBER_LIGHT}"/>
      <text x="485" y="365" text-anchor="middle" class="body">are domains or chains placed reliably?</text>
      <text x="485" y="402" text-anchor="middle" class="small">confident domains can still have an</text>
      <text x="485" y="425" text-anchor="middle" class="small">uncertain relative orientation</text>

      {box(655,90,310,360,RED_LIGHT)}
      <text x="810" y="130" text-anchor="middle" class="label">claims requiring more evidence</text>
      <rect x="700" y="165" width="220" height="48" rx="13" fill="white"/><text x="810" y="195" text-anchor="middle" class="body">state populations and kinetics</text>
      <rect x="700" y="231" width="220" height="48" rx="13" fill="white"/><text x="810" y="261" text-anchor="middle" class="body">binding affinity and selectivity</text>
      <rect x="700" y="297" width="220" height="48" rx="13" fill="white"/><text x="810" y="327" text-anchor="middle" class="body">catalysis and cellular function</text>
      <text x="810" y="385" text-anchor="middle" class="small">need ensembles, energetics, context,</text>
      <text x="810" y="408" text-anchor="middle" class="small">and ultimately experimental tests</text>

      <path d="M 318 270 L 340 270" class="arrow"/><path d="M 628 270 L 650 270" class="arrow"/>
      <rect x="80" y="495" width="840" height="42" rx="13" fill="{TEAL_LIGHT}"/>
      <text x="500" y="521" text-anchor="middle" class="body">High coordinate confidence supports structural interpretation—not thermodynamics or mechanism by itself.</text>
    '''
    write("confidence_scope", body)


if __name__ == "__main__":
    make_msa_coevolution()
    make_pair_reasoning()
    make_decoder_shift()
    make_confidence_scope()
