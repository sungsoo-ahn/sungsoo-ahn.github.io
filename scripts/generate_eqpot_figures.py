#!/usr/bin/env python3
"""Generate original figures for the equivariant-transformer/MLIP post.

The diagrams are original syntheses of the algebra and physical constraints
described in the cited primary papers. Outputs are editable SVG plus PNG
previews. The visual language uses simple rounded forms, solid colors, and
consistent strokes; no Flaticon, lecture-slide, or paper artwork is copied.
License: CC BY 4.0 with the accompanying blog post.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import blog_figure_style as bfs

OUT = ROOT / "assets" / "img" / "blog"
T, M, S = bfs.TEXT, bfs.MUTED, bfs.SPINE
P, PL = bfs.PURPLE, bfs.PURPLE_LIGHT
B, BL = bfs.BLUE, bfs.BLUE_LIGHT
G, GL = bfs.TEAL, bfs.TEAL_LIGHT
A, AL = bfs.AMBER, bfs.AMBER_LIGHT
R, RL = bfs.RED, bfs.RED_LIGHT


def doc(body: str, width: int = 1000, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x, y, w, h, fill, r=18):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'


def atom(x, y, fill, label=""):
    return f'<circle cx="{x}" cy="{y}" r="20" fill="{fill}" stroke="white" stroke-width="4"/><text x="{x}" y="{y+6}" text-anchor="middle" font-size="16" font-weight="700" fill="white">{label}</text>'


def write(name: str, body: str, height: int = 560):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"eqpot_{name}.svg"
    png = OUT / f"eqpot_{name}.png"
    svg.write_text(doc(body, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=1000):
        raise RuntimeError("SVG renderer unavailable")


def typed_attention():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Geometric attention separates invariant routing from equivariant content</text>
{box(35,80,210,390,PL)}{box(285,80,430,390,BL)}{box(755,80,210,390,GL)}
<text x="140" y="118" text-anchor="middle" class="head">typed inputs</text>
<circle cx="140" cy="178" r="32" fill="{P}"/><text x="140" y="185" text-anchor="middle" font-size="18" fill="white">hᵢ</text>
<text x="140" y="238" text-anchor="middle" class="body">scalar ℓ = 0</text><rect x="85" y="256" width="110" height="25" rx="12" fill="{A}"/>
<text x="140" y="326" text-anchor="middle" class="body">vector ℓ = 1</text><path d="M105 380L180 345" stroke="{B}" stroke-width="6" marker-end="url(#arr)"/>
<text x="500" y="118" text-anchor="middle" class="head">query–key compatibility</text>
{box(320,145,160,83,"white",14)}{box(520,145,160,83,"white",14)}
<text x="400" y="178" text-anchor="middle" class="body">qᵢ⁽ℓ⁾</text><text x="400" y="207" text-anchor="middle" class="small">typed query</text>
<text x="600" y="178" text-anchor="middle" class="body">kᵢⱼ⁽ℓ⁾</text><text x="600" y="207" text-anchor="middle" class="small">typed key</text>
<path d="M480 186L515 186" class="arrow"/>
<rect x="350" y="268" width="300" height="72" rx="18" fill="{AL}" stroke="{A}" stroke-width="2"/>
<text x="500" y="298" text-anchor="middle" class="body">sᵢⱼ = Σℓ ⟨qᵢ⁽ℓ⁾, kᵢⱼ⁽ℓ⁾⟩</text><text x="500" y="325" text-anchor="middle" class="small">one rotation-invariant scalar</text>
<path d="M500 342L500 380" class="arrow"/><text x="500" y="413" text-anchor="middle" class="body">αᵢⱼ = softmaxⱼ(sᵢⱼ)</text><text x="500" y="442" text-anchor="middle" class="small">attention chooses how much, not how to rotate</text>
<text x="860" y="118" text-anchor="middle" class="head">typed output</text><text x="860" y="176" text-anchor="middle" class="body">Σⱼ αᵢⱼ vᵢⱼ⁽ℓ⁾</text>
<path d="M810 265L910 220" stroke="{G}" stroke-width="7" marker-end="url(#arr)"/><path d="M810 325L910 370" stroke="{B}" stroke-width="7" marker-end="url(#arr)"/>
<text x="860" y="424" text-anchor="middle" class="small">each value keeps its type</text><path d="M715 275L750 275" class="arrow"/>
<rect x="180" y="495" width="640" height="42" rx="13" fill="{RL}"/><text x="500" y="522" text-anchor="middle" class="body">invariant weight × equivariant value = equivariant message</text>'''
    write("typed_attention", body)


def rotation_example():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Rotating query, keys, and values leaves attention weights unchanged</text>
{box(45,80,400,375,BL)}{box(555,80,400,375,GL)}
<text x="245" y="116" text-anchor="middle" class="head">original frame</text><text x="755" y="116" text-anchor="middle" class="head">all vectors rotated 90°</text>
<circle cx="245" cy="270" r="6" fill="{T}"/><path d="M245 270L355 270" stroke="{P}" stroke-width="6" marker-end="url(#arr)"/><text x="365" y="275" class="body">q, k₁</text>
<path d="M245 270L245 160" stroke="{B}" stroke-width="6" marker-end="url(#arr)"/><text x="255" y="157" class="body">k₂</text>
<text x="245" y="340" text-anchor="middle" class="body">q·k₁ = 1, q·k₂ = 0</text><text x="245" y="375" text-anchor="middle" class="body">(α₁, α₂) = (0.73, 0.27)</text><text x="245" y="410" text-anchor="middle" class="small">output points mostly right</text>
<circle cx="755" cy="270" r="6" fill="{T}"/><path d="M755 270L755 160" stroke="{P}" stroke-width="6" marker-end="url(#arr)"/><text x="768" y="157" class="body">Rq, Rk₁</text>
<path d="M755 270L645 270" stroke="{B}" stroke-width="6" marker-end="url(#arr)"/><text x="610" y="275" class="body">Rk₂</text>
<text x="755" y="340" text-anchor="middle" class="body">(Rq)·(Rk₁) = 1, (Rq)·(Rk₂) = 0</text><text x="755" y="375" text-anchor="middle" class="body">(α₁, α₂) = (0.73, 0.27)</text><text x="755" y="410" text-anchor="middle" class="small">output rotates upward</text>
<path d="M455 270L535 270" class="arrow"/><text x="495" y="245" text-anchor="middle" class="small">R</text>
<rect x="172" y="488" width="656" height="46" rx="14" fill="{AL}"/><text x="500" y="517" text-anchor="middle" class="body">orthogonality gives (Rq)ᵀ(Rk) = qᵀk</text>'''
    write("rotation_example", body)


def energy_forces():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">An interatomic potential should predict one energy surface, then differentiate it</text>
{box(45,85,360,370,PL)}{box(455,85,500,370,GL)}
<text x="225" y="122" text-anchor="middle" class="head">extensive invariant energy</text>
<line x1="120" y1="230" x2="230" y2="185" stroke="{S}" stroke-width="6"/><line x1="230" y1="185" x2="320" y2="265" stroke="{S}" stroke-width="6"/>
{atom(120,230,P,"C")}{atom(230,185,G,"O")}{atom(320,265,B,"H")}
<rect x="92" y="310" width="266" height="62" rx="18" fill="white" stroke="{P}" stroke-width="2"/><text x="225" y="337" text-anchor="middle" class="body">Eθ = Σᵢ εᵢ</text><text x="225" y="360" text-anchor="middle" class="small">one scalar; unchanged by rigid motion</text>
<text x="705" y="122" text-anchor="middle" class="head">conservative equivariant forces</text>
<line x1="590" y1="260" x2="705" y2="195" stroke="{S}" stroke-width="6"/><line x1="705" y1="195" x2="820" y2="260" stroke="{S}" stroke-width="6"/>
{atom(590,260,P,"C")}{atom(705,195,G,"O")}{atom(820,260,B,"H")}
<path d="M590 292L540 340" stroke="{R}" stroke-width="6" marker-end="url(#arr)"/><path d="M705 163L705 115" stroke="{R}" stroke-width="6" marker-end="url(#arr)"/><path d="M820 292L870 340" stroke="{R}" stroke-width="6" marker-end="url(#arr)"/>
<rect x="540" y="365" width="330" height="62" rx="18" fill="white" stroke="{G}" stroke-width="2"/><text x="705" y="393" text-anchor="middle" class="body">Fᵢ = −∇ᵣᵢ Eθ</text><text x="705" y="416" text-anchor="middle" class="small">one shared potential couples every force</text>
<path d="M410 270L445 270" class="arrow"/>
<rect x="125" y="488" width="750" height="46" rx="14" fill="{AL}"/><text x="500" y="517" text-anchor="middle" class="body">translation invariance ⇒ ΣᵢFᵢ = 0; rotation invariance ⇒ Σᵢ rᵢ × Fᵢ = 0</text>'''
    write("energy_forces", body)


def cutoff_tradeoff():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">The production tradeoff is smooth local physics versus global reach</text>
{box(35,82,290,390,GL)}{box(355,82,290,390,BL)}{box(675,82,290,390,PL)}
<text x="180" y="120" text-anchor="middle" class="head">local + smooth</text><circle cx="180" cy="240" r="95" fill="none" stroke="{G}" stroke-width="3" stroke-dasharray="8 7"/>{atom(180,240,P,"i")}{atom(120,190,G,"j")}{atom(250,275,B,"k")}
<path d="M90 382C130 315 190 435 270 330" fill="none" stroke="{G}" stroke-width="5"/><text x="180" y="431" text-anchor="middle" class="small">envelope and derivative vanish at r꜀</text>
<text x="500" y="120" text-anchor="middle" class="head">global + expressive</text>
<g stroke="{B}" stroke-width="2" opacity=".65"><line x1="410" y1="190" x2="580" y2="190"/><line x1="410" y1="190" x2="455" y2="330"/><line x1="410" y1="190" x2="585" y2="325"/><line x1="580" y1="190" x2="455" y2="330"/><line x1="580" y1="190" x2="585" y2="325"/><line x1="455" y1="330" x2="585" y2="325"/></g>{atom(410,190,P)}{atom(580,190,G)}{atom(455,330,B)}{atom(585,325,A)}
<text x="500" y="402" text-anchor="middle" class="body">long-range response</text><text x="500" y="431" text-anchor="middle" class="small">but dense attention approaches O(N²)</text>
<text x="820" y="120" text-anchor="middle" class="head">equivariant + efficient</text><circle cx="820" cy="240" r="78" fill="{PL}" stroke="{P}" stroke-width="3"/><path d="M820 240L820 148" stroke="{P}" stroke-width="5" marker-end="url(#arr)"/><path d="M820 240L900 278" stroke="{B}" stroke-width="5" marker-end="url(#arr)"/><text x="820" y="353" text-anchor="middle" class="body">edge-aligned SO(2)</text><text x="820" y="382" text-anchor="middle" class="small">higher angular degree at lower cost</text><text x="820" y="431" text-anchor="middle" class="small">still pays for typed channels and gradients</text>
<rect x="135" y="497" width="730" height="39" rx="13" fill="{RL}"/><text x="500" y="523" text-anchor="middle" class="body">no architecture gets smoothness, infinite range, high angular order, and linear cost for free</text>'''
    write("cutoff_tradeoff", body)


if __name__ == "__main__":
    typed_attention(); rotation_example(); energy_forces(); cutoff_tradeoff()
