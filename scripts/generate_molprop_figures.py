#!/usr/bin/env python3
"""Generate original SVG-first figures for molecular property prediction.

The diagrams are original editorial syntheses of standard molecular
representations and evaluation practices described by the cited sources. No
lecture-slide, paper, or Flaticon artwork is copied. Outputs are editable SVG
plus PNG previews. License: CC BY 4.0 with the accompanying blog post.
"""

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import blog_figure_style as bfs
OUT=ROOT/'assets'/'img'/'blog'
T,M,S=bfs.TEXT,bfs.MUTED,bfs.SPINE
P,PL=bfs.PURPLE,bfs.PURPLE_LIGHT
B,BL=bfs.BLUE,bfs.BLUE_LIGHT
G,GL=bfs.TEAL,bfs.TEAL_LIGHT
A,AL=bfs.AMBER,bfs.AMBER_LIGHT
R,RL=bfs.RED,bfs.RED_LIGHT


def doc(body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x,y,w,h,fill,r=20): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'
def atom(x,y,c,label='',rad=15): return f'<circle cx="{x}" cy="{y}" r="{rad}" fill="{c}" stroke="white" stroke-width="3"/><text x="{x}" y="{y+5}" text-anchor="middle" font-size="13" font-weight="700" fill="white">{label}</text>'
def write(name,body):
    OUT.mkdir(parents=True,exist_ok=True); svg=OUT/f'molprop_{name}.svg'; png=OUT/f'molprop_{name}.png'
    svg.write_text(doc(body),encoding='utf-8')
    if not bfs.render_svg_preview(svg,png,width=1000): raise RuntimeError('SVG renderer unavailable')


def representation_stack():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Representation determines which molecular distinctions are visible</text>
{box(35,82,285,365,AL)}{box(357,82,285,365,BL)}{box(680,82,285,365,GL)}
<text x="177" y="122" text-anchor="middle" class="head">1D tokens</text><rect x="75" y="170" width="205" height="68" rx="15" fill="white" stroke="{A}" stroke-width="2"/><text x="177" y="211" text-anchor="middle" class="body">CC(=O)Oc1ccccc1</text>
<rect x="80" y="270" width="195" height="42" rx="13" fill="{A}" opacity=".2"/><text x="177" y="297" text-anchor="middle" class="body">sequence context</text><text x="177" y="350" text-anchor="middle" class="small">compact and scalable</text><text x="177" y="380" text-anchor="middle" class="small">syntax is not geometry</text>
<text x="499" y="122" text-anchor="middle" class="head">2D graph</text><g stroke="{S}" stroke-width="6"><line x1="430" y1="205" x2="500" y2="165"/><line x1="500" y1="165" x2="568" y2="210"/><line x1="568" y1="210" x2="550" y2="285"/><line x1="550" y1="285" x2="465" y2="295"/><line x1="465" y1="295" x2="430" y2="205"/></g>{atom(430,205,P,'C')}{atom(500,165,G,'O')}{atom(568,210,B,'C')}{atom(550,285,P,'C')}{atom(465,295,A,'N')}
<text x="499" y="350" text-anchor="middle" class="small">connectivity and local chemistry</text><text x="499" y="380" text-anchor="middle" class="small">no unique conformation</text>
<text x="822" y="122" text-anchor="middle" class="head">3D conformer(s)</text><g stroke="{S}" stroke-width="6"><line x1="748" y1="245" x2="818" y2="175"/><line x1="818" y1="175" x2="892" y2="235"/><line x1="892" y1="235" x2="830" y2="310"/></g>{atom(748,245,P,'C',18)}{atom(818,175,G,'O',18)}{atom(892,235,B,'C',18)}{atom(830,310,A,'N',18)}<path d="M755 325C780 368 860 376 895 330" fill="none" stroke="{G}" stroke-width="3" stroke-dasharray="7 6"/>
<text x="822" y="395" text-anchor="middle" class="small">distances and orientation</text><text x="822" y="421" text-anchor="middle" class="small">conditioned on one or many states</text>
<rect x="130" y="486" width="740" height="48" rx="14" fill="{RL}"/><text x="500" y="507" text-anchor="middle" class="body">more coordinates add signal only when the task and conformer agree</text><text x="500" y="527" text-anchor="middle" class="small">each representation also introduces its own nuisance variation</text>'''
    write('representation_stack',body)


def split_leakage():
    nodes=[(120,175,P),(165,200,P),(105,235,P),(175,265,P),(255,180,G),(290,225,G),(250,270,G),(365,180,B),(405,225,B),(370,275,B)]
    dots=''.join(f'<circle cx="{x}" cy="{y}" r="15" fill="{c}" stroke="white" stroke-width="3"/>' for x,y,c in nodes)
    nodes2=[(610,175,P),(650,205,P),(600,245,P),(665,270,P),(750,180,G),(790,225,G),(750,270,G),(865,180,B),(905,225,B),(870,275,B)]
    dots2=''.join(f'<circle cx="{x}" cy="{y}" r="15" fill="{c}" stroke="white" stroke-width="3"/>' for x,y,c in nodes2)
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">A split defines the kind of generalization being measured</text>{box(45,82,420,370,BL)}{box(535,82,420,370,GL)}
<text x="255" y="122" text-anchor="middle" class="head">random split</text>{dots}<line x1="80" y1="315" x2="430" y2="315" stroke="{R}" stroke-width="3" stroke-dasharray="8 7"/><text x="255" y="350" text-anchor="middle" class="body">related scaffolds cross the boundary</text><text x="255" y="382" text-anchor="middle" class="small">good for interpolation within a library</text><text x="255" y="410" text-anchor="middle" class="small">optimistic for scaffold discovery</text>
<text x="745" y="122" text-anchor="middle" class="head">scaffold / temporal split</text>{dots2}<rect x="575" y="145" width="115" height="160" rx="18" fill="none" stroke="{P}" stroke-width="3"/><rect x="710" y="145" width="105" height="160" rx="18" fill="none" stroke="{G}" stroke-width="3"/><rect x="835" y="145" width="105" height="160" rx="18" fill="none" stroke="{B}" stroke-width="3"/>
<text x="745" y="350" text-anchor="middle" class="body">families or acquisition eras stay intact</text><text x="745" y="382" text-anchor="middle" class="small">harder, but closer to prospective use</text><text x="745" y="410" text-anchor="middle" class="small">duplicates must be resolved first</text>
<rect x="155" y="486" width="690" height="48" rx="14" fill="{AL}"/><text x="500" y="507" text-anchor="middle" class="body">the same predictions can support different claims under different splits</text><text x="500" y="527" text-anchor="middle" class="small">split molecules—not rows, conformers, or repeated measurements</text>'''
    write('split_leakage',body)


def conformer_uncertainty():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">A molecular graph can imply a distribution over three-dimensional states</text>{box(40,82,260,370,BL)}{box(370,82,260,370,PL)}{box(700,82,260,370,GL)}
<text x="170" y="122" text-anchor="middle" class="head">one graph</text><g stroke="{S}" stroke-width="6"><line x1="100" y1="230" x2="170" y2="180"/><line x1="170" y1="180" x2="240" y2="235"/><line x1="240" y1="235" x2="180" y2="300"/></g>{atom(100,230,P)}{atom(170,180,G)}{atom(240,235,B)}{atom(180,300,A)}<text x="170" y="390" text-anchor="middle" class="small">bond graph is fixed</text>
<path d="M305 265L360 265" class="arrow"/><text x="335" y="240" text-anchor="middle" class="small">sample</text>
<text x="500" y="122" text-anchor="middle" class="head">conformer ensemble</text><path d="M410 200L475 165L535 210L590 170" fill="none" stroke="{P}" stroke-width="6"/><path d="M410 265L480 300L535 245L590 300" fill="none" stroke="{B}" stroke-width="6"/><path d="M410 345L470 320L525 365L590 335" fill="none" stroke="{G}" stroke-width="6"/><text x="500" y="405" text-anchor="middle" class="small">energies define populations</text>
<path d="M635 265L690 265" class="arrow"/><text x="663" y="240" text-anchor="middle" class="small">aggregate</text>
<text x="830" y="122" text-anchor="middle" class="head">property distribution</text><path d="M735 340C765 338 770 195 810 195C850 195 848 310 875 310C900 310 905 260 935 260" fill="none" stroke="{G}" stroke-width="6"/><line x1="735" y1="350" x2="935" y2="350" stroke="{S}" stroke-width="2"/><text x="830" y="390" text-anchor="middle" class="small">mean, spread, and tails can matter</text>
<rect x="135" y="486" width="730" height="48" rx="14" fill="{RL}"/><text x="500" y="507" text-anchor="middle" class="body">a single conformer can create irreducible input ambiguity</text><text x="500" y="527" text-anchor="middle" class="small">ensemble averaging should match the physical observable, not convenience</text>'''
    write('conformer_uncertainty',body)


def evaluation_layers():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">One aggregate error can hide four different model failures</text>{box(45,82,910,385,PL)}
<line x1="110" y1="365" x2="890" y2="365" stroke="{S}" stroke-width="3"/><line x1="110" y1="365" x2="110" y2="145" stroke="{S}" stroke-width="3"/>
<circle cx="205" cy="315" r="12" fill="{G}"/><circle cx="290" cy="285" r="12" fill="{G}"/><circle cx="375" cy="250" r="12" fill="{G}"/><circle cx="460" cy="225" r="12" fill="{G}"/><circle cx="545" cy="190" r="12" fill="{G}"/>
<circle cx="630" cy="300" r="12" fill="{R}"/><circle cx="715" cy="170" r="12" fill="{R}"/><circle cx="805" cy="330" r="12" fill="{R}"/>
<line x1="155" y1="335" x2="835" y2="145" stroke="{P}" stroke-width="4"/>
<text x="275" y="405" text-anchor="middle" class="body">dense regime</text><text x="275" y="430" text-anchor="middle" class="small">low average error</text><text x="730" y="405" text-anchor="middle" class="body">rare chemistry / extremes</text><text x="730" y="430" text-anchor="middle" class="small">large, consequential failures</text>
<rect x="95" y="490" width="190" height="42" rx="13" fill="{BL}"/><text x="190" y="516" text-anchor="middle" class="body">central MAE</text><rect x="305" y="490" width="190" height="42" rx="13" fill="{GL}"/><text x="400" y="516" text-anchor="middle" class="body">tail error</text><rect x="515" y="490" width="190" height="42" rx="13" fill="{AL}"/><text x="610" y="516" text-anchor="middle" class="body">calibration</text><rect x="725" y="490" width="190" height="42" rx="13" fill="{RL}"/><text x="820" y="516" text-anchor="middle" class="body">OOD coverage</text>'''
    write('evaluation_layers',body)


if __name__=='__main__': representation_stack(); split_leakage(); conformer_uncertainty(); evaluation_layers()
