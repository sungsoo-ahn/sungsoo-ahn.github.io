#!/usr/bin/env python3
"""Generate original SVG-first figures for protein representation learning.

The diagrams are original syntheses of standard protein representation and
evaluation concepts. No lecture-slide, paper, or Flaticon artwork is copied.
Outputs are editable SVG plus PNG previews in a flat-icon visual language.
License: CC BY 4.0 with the accompanying blog post.
"""

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


def doc(body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}.edge{{stroke:{S};stroke-width:4;stroke-linecap:round}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x, y, w, h, fill, r=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'


def residue(x, y, color, label=""):
    return f'<circle cx="{x}" cy="{y}" r="15" fill="{color}" stroke="white" stroke-width="3"/><text x="{x}" y="{y+5}" text-anchor="middle" font-size="12" font-weight="700" fill="white">{label}</text>'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"protrep_{name}.svg"
    png = OUT / f"protrep_{name}.png"
    svg.write_text(doc(body), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=1000):
        raise RuntimeError("SVG renderer unavailable")


def representation_views():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Each protein representation preserves a different neighborhood</text>
{box(25,82,180,360,PL)}{box(218,82,180,360,BL)}{box(411,82,180,360,GL)}{box(604,82,180,360,AL)}{box(797,82,178,360,RL)}
<text x="115" y="120" text-anchor="middle" class="head">sequence</text><g>{residue(62,190,P,'M')}{residue(96,190,B,'K')}{residue(130,190,G,'L')}{residue(164,190,A,'V')}</g><text x="115" y="260" text-anchor="middle" class="body">chain order</text><text x="115" y="290" text-anchor="middle" class="small">evolutionary syntax</text><text x="115" y="318" text-anchor="middle" class="small">no explicit geometry</text>
<text x="308" y="120" text-anchor="middle" class="head">MSA</text><g font-size="16" font-weight="700"><text x="250" y="178">M K L V -</text><text x="250" y="210">M R L I G</text><text x="250" y="242">L K L V G</text></g><rect x="283" y="153" width="22" height="104" rx="6" fill="{B}" opacity=".22"/><text x="308" y="290" text-anchor="middle" class="body">family columns</text><text x="308" y="318" text-anchor="middle" class="small">variation and coupling</text>
<text x="501" y="120" text-anchor="middle" class="head">residue graph</text><g class="edge"><line x1="458" y1="180" x2="540" y2="180"/><line x1="458" y1="180" x2="485" y2="245"/><line x1="540" y1="180" x2="555" y2="250"/><line x1="485" y1="245" x2="555" y2="250"/></g>{residue(458,180,P)}{residue(540,180,B)}{residue(485,245,G)}{residue(555,250,A)}<text x="501" y="290" text-anchor="middle" class="body">spatial contacts</text><text x="501" y="318" text-anchor="middle" class="small">edge definitions matter</text>
<text x="694" y="120" text-anchor="middle" class="head">backbone frames</text><circle cx="694" cy="215" r="19" fill="{P}"/><line x1="694" y1="215" x2="750" y2="215" stroke="{R}" stroke-width="5" marker-end="url(#arr)"/><line x1="694" y1="215" x2="694" y2="160" stroke="{G}" stroke-width="5" marker-end="url(#arr)"/><line x1="694" y1="215" x2="656" y2="250" stroke="{B}" stroke-width="5" marker-end="url(#arr)"/><text x="694" y="290" text-anchor="middle" class="body">pose and direction</text><text x="694" y="318" text-anchor="middle" class="small">equivariant local geometry</text>
<text x="886" y="120" text-anchor="middle" class="head">surface</text><path d="M825 235C820 175 862 146 910 164C957 182 959 238 923 264C886 292 835 276 825 235z" fill="{GL}" stroke="{G}" stroke-width="3"/><circle cx="850" cy="220" r="8" fill="{P}"/><circle cx="910" cy="202" r="8" fill="{A}"/><path d="M842 246C870 229 898 250 930 224" fill="none" stroke="{B}" stroke-width="5"/><text x="886" y="318" text-anchor="middle" class="body">shape and chemistry</text><text x="886" y="346" text-anchor="middle" class="small">interaction-facing boundary</text>
<rect x="150" y="480" width="700" height="45" rx="14" fill="{BL}"/><text x="500" y="508" text-anchor="middle" class="body">multimodal encoders align these views; they do not make them equivalent</text>'''
    write("representation_views", body)


def objectives():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">A pretraining objective decides which information becomes easy to read out</text>
{box(45,82,910,390,PL)}
{box(85,125,215,95,'white',16)}<text x="192" y="160" text-anchor="middle" class="head">masked sequence</text><text x="192" y="192" text-anchor="middle" class="small">recover amino-acid identity</text>
{box(85,245,215,95,'white',16)}<text x="192" y="280" text-anchor="middle" class="head">noisy coordinates</text><text x="192" y="312" text-anchor="middle" class="small">recover local geometry</text>
{box(85,365,215,72,'white',16)}<text x="192" y="397" text-anchor="middle" class="head">paired views</text><text x="192" y="423" text-anchor="middle" class="small">match sequence and structure</text>
<path d="M307 172L405 240" class="arrow"/><path d="M307 292L405 270" class="arrow"/><path d="M307 400L405 292" class="arrow"/>
{box(414,190,180,150,GL)}<text x="504" y="240" text-anchor="middle" class="head">encoder</text><rect x="463" y="265" width="82" height="25" rx="12" fill="{G}"/><rect x="445" y="302" width="118" height="15" rx="8" fill="{B}"/><text x="504" y="365" text-anchor="middle" class="small">shared embedding</text>
<path d="M601 265L694 265" class="arrow"/>
{box(705,125,205,95,AL,16)}<text x="807" y="160" text-anchor="middle" class="head">local readout</text><text x="807" y="192" text-anchor="middle" class="small">residue, contact, site</text>
{box(705,245,205,95,BL,16)}<text x="807" y="280" text-anchor="middle" class="head">global readout</text><text x="807" y="312" text-anchor="middle" class="small">family or phenotype</text>
{box(705,365,205,72,RL,16)}<text x="807" y="397" text-anchor="middle" class="head">discarded detail</text><text x="807" y="423" text-anchor="middle" class="small">not rewarded by the loss</text>
<path d="M696 265C660 220 670 172 697 172" class="arrow"/><path d="M696 265L697 292" class="arrow"/><path d="M696 265C660 330 668 400 697 400" class="arrow"/>
<rect x="160" y="500" width="680" height="38" rx="13" fill="{RL}"/><text x="500" y="525" text-anchor="middle" class="body">pretraining supplies an inductive bias, not a guarantee of downstream causality</text>'''
    write("pretraining_objectives", body)


def split_leakage():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Random splits can turn homology retrieval into apparent generalization</text>
{box(45,82,430,385,RL)}{box(525,82,430,385,GL)}
<text x="260" y="122" text-anchor="middle" class="head">random sequence split</text><text x="740" y="122" text-anchor="middle" class="head">cluster-aware split</text>
<ellipse cx="190" cy="245" rx="105" ry="82" fill="{PL}" stroke="{P}" stroke-width="2"/><ellipse cx="335" cy="280" rx="96" ry="76" fill="{BL}" stroke="{B}" stroke-width="2"/>
<text x="190" y="180" text-anchor="middle" class="small">family A</text><text x="335" y="345" text-anchor="middle" class="small">family B</text>
{residue(150,225,P,'T')}{residue(205,260,P,'V')}{residue(245,215,P,'T')}{residue(305,270,B,'V')}{residue(355,245,B,'T')}{residue(380,310,B,'V')}
<path d="M245 215L305 270" stroke="{R}" stroke-width="4" stroke-dasharray="7 6"/><text x="260" y="410" text-anchor="middle" class="body">near neighbors cross the boundary</text>
<ellipse cx="645" cy="255" rx="88" ry="105" fill="{PL}" stroke="{P}" stroke-width="2"/><ellipse cx="845" cy="255" rx="88" ry="105" fill="{BL}" stroke="{B}" stroke-width="2"/><text x="645" y="175" text-anchor="middle" class="small">training families</text><text x="845" y="175" text-anchor="middle" class="small">held-out families</text>
{residue(620,235,P,'T')}{residue(670,285,P,'T')}{residue(625,320,P,'T')}{residue(820,235,B,'V')}{residue(870,285,B,'V')}{residue(825,320,B,'V')}
<line x1="750" y1="155" x2="750" y2="368" stroke="{S}" stroke-width="3" stroke-dasharray="8 7"/><text x="740" y="410" text-anchor="middle" class="body">entire similarity clusters are held out</text>
<rect x="135" y="493" width="730" height="39" rx="13" fill="{AL}"/><text x="500" y="519" text-anchor="middle" class="body">the split defines whether the question is interpolation, family transfer, or remote function transfer</text>'''
    write("homology_splits", body)


def embedding_claims():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">A useful embedding can encode signal, shortcuts, and nuisance variation together</text>
{box(45,82,910,390,BL)}
<text x="190" y="125" text-anchor="middle" class="head">protein inputs</text><path d="M105 185C130 140 180 235 220 170C250 125 275 205 285 250" fill="none" stroke="{P}" stroke-width="8" stroke-linecap="round"/>
<text x="190" y="300" text-anchor="middle" class="body">sequence + structure</text><text x="190" y="328" text-anchor="middle" class="small">species, family, assay context</text>
<path d="M300 245L405 245" class="arrow"/>
{box(415,145,180,205,PL)}<text x="505" y="185" text-anchor="middle" class="head">embedding z</text><circle cx="470" cy="245" r="16" fill="{P}"/><circle cx="510" cy="220" r="13" fill="{G}"/><circle cx="545" cy="265" r="18" fill="{B}"/><circle cx="490" cy="295" r="12" fill="{A}"/><text x="505" y="328" text-anchor="middle" class="small">entangled coordinates</text>
<path d="M605 245L685 150" class="arrow"/><path d="M605 245L685 245" class="arrow"/><path d="M605 245L685 340" class="arrow"/>
{box(695,105,220,92,GL,16)}<text x="805" y="142" text-anchor="middle" class="head">mechanistic signal</text><text x="805" y="173" text-anchor="middle" class="small">active site, contacts, dynamics</text>
{box(695,215,220,72,AL,16)}<text x="805" y="247" text-anchor="middle" class="head">family identity</text><text x="805" y="273" text-anchor="middle" class="small">often useful, sometimes shortcut</text>
{box(695,305,220,92,RL,16)}<text x="805" y="342" text-anchor="middle" class="head">dataset nuisance</text><text x="805" y="373" text-anchor="middle" class="small">length, taxonomy, structure source</text>
<rect x="125" y="495" width="750" height="38" rx="13" fill="{GL}"/><text x="500" y="520" text-anchor="middle" class="body">probe what transfers after controlling homology and metadata, not only what is linearly decodable</text>'''
    write("embedding_claims", body)


if __name__ == "__main__":
    representation_views()
    objectives()
    split_leakage()
    embedding_claims()
