#!/usr/bin/env python3
"""Generate original figures for the molecular generation/reactions post.

All outputs are native SVG with PNG previews. The figures use a flat-icon
visual language: rounded forms, restrained solid colors, consistent strokes,
and no gradients. No Flaticon asset, lecture-slide figure, or paper figure is
copied. Molecular diagrams are deliberately small schematic examples created
for the post. License: CC BY 4.0 with the blog post.
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


def atom(x: float, y: float, label: str, color: str, *, radius: int = 24) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="3"/><text x="{x}" y="{y+6}" text-anchor="middle" fill="white" font-size="17" font-weight="700">{label}</text>'


def bond(x1: float, y1: float, x2: float, y2: float, *, color: str = MUTED, width: int = 5, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{extra}/>'


def write(name: str, body: str, *, width: int = 1000, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"molgenrxn_{name}.svg"
    png_path = OUT / f"molgenrxn_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def make_representation_contract() -> None:
    graph = bond(117, 215, 197, 215) + bond(197, 215, 277, 215)
    graph += atom(117, 215, "C", TEXT) + atom(197, 215, "C", TEXT) + atom(277, 215, "O", RED)
    invalid = bond(705, 190, 785, 230, color=RED) + bond(705, 270, 785, 230, color=RED) + bond(865, 230, 785, 230, color=RED)
    invalid += atom(705, 190, "C", TEXT, radius=20) + atom(705, 270, "C", TEXT, radius=20)
    invalid += atom(865, 230, "C", TEXT, radius=20) + atom(785, 230, "O", RED, radius=24)
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">A molecular representation is a contract, not just a container</text>
      {box(35,88,290,365,BLUE_LIGHT)}
      <text x="180" y="128" text-anchor="middle" class="label">one molecular graph</text>
      {graph}
      <text x="197" y="270" text-anchor="middle" class="body">ethanol</text>
      <rect x="70" y="305" width="220" height="48" rx="13" fill="white" stroke="{BLUE}" stroke-width="2"/>
      <text x="180" y="336" text-anchor="middle" class="body">SMILES: CCO</text>
      <rect x="70" y="373" width="220" height="48" rx="13" fill="white" stroke="{PURPLE}" stroke-width="2"/>
      <text x="180" y="404" text-anchor="middle" class="body">SMILES: OCC</text>
      <text x="180" y="438" text-anchor="middle" class="small">different strings, same graph</text>

      {box(355,88,290,365,TEAL_LIGHT)}
      <text x="500" y="128" text-anchor="middle" class="label">graph attributes carry chemistry</text>
      <circle cx="430" cy="205" r="26" fill="{TEXT}"/><text x="430" y="212" text-anchor="middle" fill="white" font-size="18" font-weight="700">C</text>
      <circle cx="570" cy="205" r="26" fill="{RED}"/><text x="570" y="212" text-anchor="middle" fill="white" font-size="18" font-weight="700">O</text>
      {bond(456,205,544,205)}
      <text x="500" y="186" text-anchor="middle" class="small">single bond</text>
      <text x="500" y="282" text-anchor="middle" class="body">atom type, charge, aromaticity</text>
      <text x="500" y="314" text-anchor="middle" class="body">bond order, stereochemistry</text>
      <text x="500" y="346" text-anchor="middle" class="body">components and attachment sites</text>
      <rect x="397" y="379" width="206" height="43" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="500" y="406" text-anchor="middle" class="body">decoding must preserve all of them</text>

      {box(675,88,290,365,RED_LIGHT)}
      <text x="820" y="128" text-anchor="middle" class="label">local syntax can still fail</text>
      {invalid}
      <text x="820" y="300" text-anchor="middle" class="body">neutral O with three single bonds</text>
      <text x="820" y="331" text-anchor="middle" class="small">graph-shaped, but valence-invalid</text>
      <line x1="760" y1="365" x2="880" y2="365" stroke="{RED}" stroke-width="5" stroke-linecap="round"/>
      <line x1="820" y1="345" x2="820" y2="385" stroke="{RED}" stroke-width="5" stroke-linecap="round" transform="rotate(45 820 365)"/>
      <text x="820" y="423" text-anchor="middle" class="body">validity is a coupled constraint</text>

      <rect x="155" y="490" width="690" height="46" rx="14" fill="{PURPLE_SOFT}"/>
      <text x="500" y="519" text-anchor="middle" class="body">The generator must be invariant to serialization while remaining sensitive to chemical labels.</text>
    '''
    write("representation_contract", body)


def make_generation_strategies() -> None:
    seq = ""
    for i, (x, label, color) in enumerate([(90, "C", TEXT), (160, "C", TEXT), (230, "O", RED)]):
        if i:
            seq += bond(x - 46, 205, x - 24, 205)
        seq += atom(x, 205, label, color, radius=20)
    noisy_edges = bond(620,230,700,190,dash="7 6") + bond(620,230,700,275,dash="7 6") + bond(700,190,795,240,dash="7 6") + bond(700,275,795,240,dash="7 6")
    clean_edges = bond(820,190,890,240) + bond(820,290,890,240)
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Generation strategy chooses where consistency is enforced</text>
      {box(35,86,440,380,PURPLE_SOFT)}
      <text x="255" y="126" text-anchor="middle" class="label">autoregressive construction</text>
      <text x="255" y="158" text-anchor="middle" class="small">choose one action given the partial molecule</text>
      {seq}
      <path d="M 265 205 L 325 205" class="arrow"/>
      <rect x="337" y="173" width="105" height="64" rx="16" fill="{AMBER_LIGHT}" stroke="{AMBER}" stroke-width="2"/>
      <text x="389" y="199" text-anchor="middle" class="body">STOP</text>
      <text x="389" y="221" text-anchor="middle" class="small">or extend</text>
      <text x="255" y="285" text-anchor="middle" class="body">p(G, pi) = product of action probabilities</text>
      <rect x="75" y="318" width="360" height="112" rx="16" fill="white"/>
      <text x="255" y="348" text-anchor="middle" class="body">strength: constrain each intermediate step</text>
      <text x="255" y="378" text-anchor="middle" class="body">cost: serial decoding and exposure bias</text>
      <text x="255" y="408" text-anchor="middle" class="small">the same graph admits many action orders pi</text>

      {box(525,86,440,380,TEAL_LIGHT)}
      <text x="745" y="126" text-anchor="middle" class="label">one-shot or discrete denoising</text>
      <text x="745" y="158" text-anchor="middle" class="small">update many categorical node and edge variables together</text>
      {noisy_edges}
      {atom(620,230,"?",MUTED,radius=19)}{atom(700,190,"?",MUTED,radius=19)}{atom(700,275,"?",MUTED,radius=19)}{atom(795,240,"?",MUTED,radius=19)}
      <path d="M 806 240 L 835 240" class="arrow"/>
      {clean_edges}{atom(820,190,"C",TEXT,radius=19)}{atom(820,290,"C",TEXT,radius=19)}{atom(890,240,"O",RED,radius=19)}
      <text x="745" y="320" text-anchor="middle" class="body">permutation-equivariant parallel prediction</text>
      <rect x="565" y="342" width="360" height="96" rx="16" fill="white"/>
      <text x="745" y="369" text-anchor="middle" class="body">strength: no canonical construction order</text>
      <text x="745" y="396" text-anchor="middle" class="body">cost: global consistency emerges jointly</text>
      <text x="745" y="422" text-anchor="middle" class="small">size, valence, and connectivity can disagree</text>

      <rect x="167" y="495" width="666" height="42" rx="13" fill="{BLUE_LIGHT}"/>
      <text x="500" y="522" text-anchor="middle" class="body">Motif grammars move the boundary: fewer steps, stronger priors, less access to unseen motifs.</text>
    '''
    write("generation_strategies", body)


def make_reaction_edits_mapping() -> None:
    react = bond(80,210,145,210) + bond(145,210,210,210)
    react += atom(80,210,"C",TEXT,radius=20) + atom(145,210,"C",TEXT,radius=20) + atom(210,210,"Br",AMBER,radius=22)
    react += atom(95,300,"O",RED,radius=22)
    product = bond(390,210,455,210) + bond(455,210,520,210)
    product += atom(390,210,"C",TEXT,radius=20) + atom(455,210,"C",TEXT,radius=20) + atom(520,210,"O",RED,radius=22)
    product += atom(510,300,"Br",AMBER,radius=22)
    equiv_left = bond(735,190,800,225) + bond(735,260,800,225)
    equiv_left += atom(735,190,"O1",RED,radius=22) + atom(735,260,"O2",RED,radius=22) + atom(800,225,"C",TEXT,radius=22)
    equiv_right = bond(875,190,940,225) + bond(875,260,940,225)
    equiv_right += atom(875,190,"O2",RED,radius=22) + atom(875,260,"O1",RED,radius=22) + atom(940,225,"C",TEXT,radius=22)
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Reaction prediction is sparse graph editing with ambiguous bookkeeping</text>
      {box(35,88,565,390,BLUE_LIGHT)}
      <text x="318" y="128" text-anchor="middle" class="label">worked substitution: bromoethane to ethanol</text>
      {react}
      <text x="95" y="345" text-anchor="middle" class="small">hydroxide</text>
      <path d="M 250 235 L 340 235" class="arrow"/>
      <text x="295" y="211" text-anchor="middle" class="small">predict edits</text>
      {product}
      <text x="300" y="388" text-anchor="middle" class="body">delete C-Br; add C-O</text>
      <line x1="217" y1="190" x2="250" y2="223" stroke="{RED}" stroke-width="4"/>
      <line x1="250" y1="190" x2="217" y2="223" stroke="{RED}" stroke-width="4"/>
      <path d="M 425 290 L 455 248" stroke="{TEAL}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
      <text x="318" y="437" text-anchor="middle" class="small">most atoms persist; the reaction center is small</text>

      {box(630,88,335,390,PURPLE_SOFT)}
      <text x="798" y="128" text-anchor="middle" font-size="16" font-weight="700">symmetric atoms allow several mappings</text>
      {equiv_left}
      <path d="M 826 225 L 850 225" class="arrow"/>
      {equiv_right}
      <text x="798" y="310" text-anchor="middle" class="body">equivalent oxygen labels can swap</text>
      <text x="798" y="341" text-anchor="middle" class="small">same unlabeled chemical graph</text>
      <rect x="675" y="375" width="245" height="66" rx="15" fill="{AMBER_LIGHT}"/>
      <text x="798" y="401" text-anchor="middle" class="body">mapping is supervision</text>
      <text x="798" y="425" text-anchor="middle" class="small">inconsistency invents extra edits</text>

      <rect x="170" y="503" width="660" height="38" rx="12" fill="{TEAL_LIGHT}"/>
      <text x="500" y="528" text-anchor="middle" class="body">Chemistry constrains the edit; atom mapping chooses a correspondence among valid symmetries.</text>
    '''
    write("reaction_edits_mapping", body)


def make_evaluation_funnel() -> None:
    stages = [
        (80, 92, 840, 58, BLUE_LIGHT, "1  Parse and sanitize", "syntax, valence, charge, connectivity"),
        (125, 169, 750, 58, PURPLE_SOFT, "2  Audit diversity", "uniqueness, novelty, scaffold and mode coverage"),
        (170, 246, 660, 58, TEAL_LIGHT, "3  Test the design claim", "held-out property models, uncertainty, multi-objective tradeoffs"),
        (215, 323, 570, 58, AMBER_LIGHT, "4  Plan a synthesis", "available building blocks, route length, conditions and precedent"),
        (260, 400, 480, 58, RED_LIGHT, "5  Make and measure", "yield, purity, stability and experimental property"),
    ]
    body = '<text x="500" y="42" text-anchor="middle" class="title">Evaluation should narrow toward the experiment, not stop at validity</text>'
    for x, y, w, h, fill, label, detail in stages:
        body += box(x, y, w, h, fill, stroke="white", radius=16)
        body += f'<text x="{x+22}" y="{y+25}" class="label">{label}</text>'
        body += f'<text x="{x+w-22}" y="{y+38}" text-anchor="end" class="small">{detail}</text>'
    body += f'''
      <path d="M 500 151 L 500 165" class="arrow"/><path d="M 500 228 L 500 242" class="arrow"/>
      <path d="M 500 305 L 500 319" class="arrow"/><path d="M 500 382 L 500 396" class="arrow"/>
      <rect x="180" y="492" width="640" height="46" rx="14" fill="white" stroke="{SPINE}" stroke-width="2"/>
      <text x="500" y="521" text-anchor="middle" class="body">Every lower stage is costlier—and closer to the scientific claim.</text>
    '''
    write("evaluation_funnel", body)


if __name__ == "__main__":
    make_representation_contract()
    make_generation_strategies()
    make_reaction_edits_mapping()
    make_evaluation_funnel()
