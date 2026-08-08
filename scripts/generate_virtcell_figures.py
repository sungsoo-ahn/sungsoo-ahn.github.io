#!/usr/bin/env python3
"""Generate original SVG-first figures for genomic models and virtual cells.

These diagrams are original syntheses of standard genomics, single-cell, and
perturbation-modeling concepts. No lecture-slide, paper, or Flaticon artwork is
copied. Outputs are editable SVG plus PNG previews in a flat-icon visual
language. License: CC BY 4.0 with the accompanying blog post.
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
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}.line{{fill:none;stroke:{S};stroke-width:4;stroke-linecap:round}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x, y, w, h, fill, r=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"virtcell_{name}.svg"
    png = OUT / f"virtcell_{name}.png"
    svg.write_text(doc(body), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=1000):
        raise RuntimeError("SVG renderer unavailable")


def representation_ladder():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">A virtual cell needs more than a larger genomic language model</text>
{box(35,92,205,340,PL)}{box(275,92,205,340,BL)}{box(520,92,205,340,GL)}{box(760,92,205,340,AL)}
<text x="137" y="132" text-anchor="middle" class="head">DNA / RNA</text><path d="M72 195C100 155 128 235 158 195C185 155 205 225 215 245" class="line" stroke="{P}"/><text x="137" y="280" text-anchor="middle" class="body">sequence context</text><text x="137" y="310" text-anchor="middle" class="small">motifs and variants</text><text x="137" y="340" text-anchor="middle" class="small">one molecular layer</text>
<text x="377" y="132" text-anchor="middle" class="head">regulatory state</text><g fill="{B}"><rect x="315" y="190" width="18" height="65" rx="7"/><rect x="345" y="165" width="18" height="90" rx="7"/><rect x="375" y="210" width="18" height="45" rx="7"/><rect x="405" y="145" width="18" height="110" rx="7"/><rect x="435" y="188" width="18" height="67" rx="7"/></g><text x="377" y="280" text-anchor="middle" class="body">chromatin + expression</text><text x="377" y="310" text-anchor="middle" class="small">cell-type dependent</text><text x="377" y="340" text-anchor="middle" class="small">assay dependent</text>
<text x="622" y="132" text-anchor="middle" class="head">cell state</text><circle cx="622" cy="214" r="73" fill="white" stroke="{G}" stroke-width="5"/><circle cx="622" cy="214" r="28" fill="{GL}" stroke="{G}" stroke-width="3"/><circle cx="580" cy="185" r="8" fill="{P}"/><circle cx="660" cy="245" r="8" fill="{A}"/><text x="622" y="310" text-anchor="middle" class="body">multi-omic snapshot</text><text x="622" y="340" text-anchor="middle" class="small">context and uncertainty</text>
<text x="862" y="132" text-anchor="middle" class="head">virtual cell</text><circle cx="862" cy="205" r="58" fill="white" stroke="{A}" stroke-width="4"/><path d="M820 205C842 165 874 255 905 190" fill="none" stroke="{R}" stroke-width="6"/><path d="M862 145A60 60 0 0 1 918 205" class="arrow"/><text x="862" y="290" text-anchor="middle" class="body">predict interventions</text><text x="862" y="320" text-anchor="middle" class="small">across time and context</text><text x="862" y="350" text-anchor="middle" class="small">with calibrated distributions</text>
<path d="M242 245L267 245" class="arrow"/><path d="M482 245L512 245" class="arrow"/><path d="M727 245L752 245" class="arrow"/>
<rect x="130" y="478" width="740" height="45" rx="14" fill="{RL}"/><text x="500" y="506" text-anchor="middle" class="body">each step adds measured context; sequence likelihood alone does not simulate cell behavior</text>'''
    write("representation_ladder", body)


def count_observation():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">A single-cell count vector is a noisy observation, not the cell itself</text>
{box(45,85,910,385,BL)}
{box(85,150,180,170,GL)}<text x="175" y="185" text-anchor="middle" class="head">latent cell state</text><circle cx="175" cy="245" r="48" fill="white" stroke="{G}" stroke-width="4"/><circle cx="175" cy="245" r="19" fill="{GL}" stroke="{G}" stroke-width="3"/><text x="175" y="345" text-anchor="middle" class="small">regulation, proteins, history</text>
<path d="M272 235L370 235" class="arrow"/>
{box(380,135,220,200,PL)}<text x="490" y="175" text-anchor="middle" class="head">molecular abundance</text><g fill="{P}"><rect x="425" y="255" width="20" height="44" rx="6"/><rect x="460" y="210" width="20" height="89" rx="6"/><rect x="495" y="240" width="20" height="59" rx="6"/><rect x="530" y="190" width="20" height="109" rx="6"/></g><text x="490" y="320" text-anchor="middle" class="small">unobserved expression lambda</text>
<path d="M608 235L690 180" class="arrow"/><path d="M608 235L690 300" class="arrow"/>
{box(700,112,210,135,AL,16)}<text x="805" y="150" text-anchor="middle" class="head">cell A counts</text><text x="805" y="190" text-anchor="middle" class="body">0  7  1  12</text><text x="805" y="220" text-anchor="middle" class="small">library size 20</text>
{box(700,270,210,135,RL,16)}<text x="805" y="308" text-anchor="middle" class="head">cell B counts</text><text x="805" y="348" text-anchor="middle" class="body">2  1  0  4</text><text x="805" y="378" text-anchor="middle" class="small">library size 7</text>
<rect x="135" y="495" width="730" height="39" rx="13" fill="{GL}"/><text x="500" y="521" text-anchor="middle" class="body">sampling, capture efficiency, batch, and donor effects can separate matched biological states</text>'''
    write("count_observation", body)


def perturbation_model():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Perturbation prediction is a conditional distribution over future cell states</text>
{box(45,85,910,385,PL)}
{box(80,135,205,105,BL,16)}<text x="182" y="173" text-anchor="middle" class="head">baseline state x0</text><text x="182" y="208" text-anchor="middle" class="small">cell type and measured state</text>
{box(80,265,205,105,AL,16)}<text x="182" y="303" text-anchor="middle" class="head">intervention u</text><text x="182" y="338" text-anchor="middle" class="small">drug, dose, edit, time</text>
{box(60,395,245,45,GL,14)}<text x="182" y="423" text-anchor="middle" class="small">context c: donor + environment</text>
<path d="M292 188L405 245" class="arrow"/><path d="M292 317L405 275" class="arrow"/><path d="M292 417L405 300" class="arrow"/>
{box(415,185,190,150,GL)}<text x="510" y="225" text-anchor="middle" class="head">response model</text><text x="510" y="265" text-anchor="middle" class="body">p(x1 | x0, u, c)</text><text x="510" y="300" text-anchor="middle" class="small">distribution, not one vector</text>
<path d="M612 260L700 175" class="arrow"/><path d="M612 260L700 350" class="arrow"/>
{box(710,120,210,115,BL,16)}<text x="815" y="157" text-anchor="middle" class="head">observed regime</text><text x="815" y="190" text-anchor="middle" class="small">matched perturbation + context</text><text x="815" y="215" text-anchor="middle" class="small">interpolation</text>
{box(710,290,210,135,RL,16)}<text x="815" y="327" text-anchor="middle" class="head">counterfactual regime</text><text x="815" y="360" text-anchor="middle" class="small">new intervention or donor</text><text x="815" y="387" text-anchor="middle" class="small">requires coverage or assumptions</text>
<rect x="145" y="497" width="710" height="38" rx="13" fill="{AL}"/><text x="500" y="522" text-anchor="middle" class="small">conditioning labels describe an experiment; they do not by themselves identify its causal mechanism</text>'''
    write("perturbation_distribution", body)


def evaluation_claims():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Evaluation must hold out the factor named in the generalization claim</text>
{box(45,85,910,385,GL)}
<text x="245" y="126" text-anchor="middle" class="head">held-out axis</text><text x="720" y="126" text-anchor="middle" class="head">evidence the test can support</text>
<rect x="90" y="150" width="310" height="55" rx="14" fill="{BL}"/><text x="245" y="184" text-anchor="middle" class="body">cells, same donor and perturbation</text><rect x="455" y="150" width="455" height="55" rx="14" fill="white"/><text x="682" y="184" text-anchor="middle" class="body">technical replicate interpolation</text>
<rect x="90" y="220" width="310" height="55" rx="14" fill="{AL}"/><text x="245" y="254" text-anchor="middle" class="body">entire donor or batch</text><rect x="455" y="220" width="455" height="55" rx="14" fill="white"/><text x="682" y="254" text-anchor="middle" class="body">context transfer</text>
<rect x="90" y="290" width="310" height="55" rx="14" fill="{PL}"/><text x="245" y="324" text-anchor="middle" class="body">entire drug, edit, or combination</text><rect x="455" y="290" width="455" height="55" rx="14" fill="white"/><text x="682" y="324" text-anchor="middle" class="body">perturbation extrapolation</text>
<rect x="90" y="360" width="310" height="55" rx="14" fill="{RL}"/><text x="245" y="394" text-anchor="middle" class="body">cell type + donor + intervention</text><rect x="455" y="360" width="455" height="55" rx="14" fill="white"/><text x="682" y="394" text-anchor="middle" class="body">compositional transfer toward virtual-cell use</text>
<rect x="145" y="497" width="710" height="38" rx="13" fill="{RL}"/><text x="500" y="522" text-anchor="middle" class="body">random cell splits usually leave every biological condition visible during training</text>'''
    write("claim_matched_evaluation", body)


if __name__ == "__main__":
    representation_ladder()
    count_observation()
    perturbation_model()
    evaluation_claims()
