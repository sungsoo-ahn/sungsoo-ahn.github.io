#!/usr/bin/env python3
"""Generate original figures for the 3D molecular generation/optimization post.

Figures are original editable SVGs with deterministic PNG previews. They use
flat, rounded forms and the site palette; no paper or lecture artwork is copied.

Conceptual sources
------------------
* Hoogeboom et al. (2022), https://proceedings.mlr.press/v162/hoogeboom22a.html
* Jing et al. (2022), https://openreview.net/forum?id=w6fj2r62r_H
* Gómez-Bombarelli et al. (2018), https://doi.org/10.1021/acscentsci.7b00572

License: same as the blog.
"""
from __future__ import annotations
from html import escape
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs

OUT=Path("assets/img/blog"); WIDTH=1200

def document(body:str,height:int=550)->str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker><marker id="parrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{bfs.TEXT}}}.title{{font-size:24px;font-weight:700}}.label{{font-size:19px;font-weight:600}}.body{{font-size:17px}}.small{{font-size:15px;fill:{bfs.MUTED}}}.arrow{{fill:none;stroke:{bfs.MUTED};stroke-width:3;stroke-linecap:round;marker-end:url(#arrow)}}.parrow{{fill:none;stroke:{bfs.PURPLE};stroke-width:4;stroke-linecap:round;marker-end:url(#parrow)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''

def text(x,y,s,cls="body",anchor="middle"): return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'
def box(x,y,w,h,fill,stroke,r=26): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
def atom(x,y,c,r=17): return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="white" stroke-width="4"/>'
def bond(x1,y1,x2,y2,w=6): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{bfs.MUTED}" stroke-width="{w}" stroke-linecap="round"/>'
def molecule(points,edges):
    b=[]
    for i,j in edges: b.append(bond(*points[i][:2],*points[j][:2]))
    for x,y,c in points: b.append(atom(x,y,c))
    return ''.join(b)
def save(stem,body,height=550):
    OUT.mkdir(parents=True,exist_ok=True); svg=OUT/f"{stem}.svg"; png=OUT/f"{stem}.png"; svg.write_text(document(body,height),encoding="utf-8")
    if not bfs.render_svg_preview(svg,png,width=WIDTH): raise RuntimeError(stem)

def two_generation_tasks():
    b=[text(55,48,"Conformer generation and molecule generation change different variables","title","start")]
    for x,title,fill,stroke in [(45,"fixed graph, new conformer",bfs.BLUE_LIGHT,bfs.BLUE),(625,"new graph and geometry",bfs.GREEN_LIGHT,bfs.GREEN)]:
        b.append(box(x,90,530,380,fill,stroke,28)); b.append(text(x+265,132,title,"label"))
    colors=[bfs.PURPLE,bfs.BLUE,bfs.GREEN,bfs.AMBER]
    p1=[(120,275,colors[0]),(215,220,colors[1]),(310,280,colors[2]),(405,220,colors[3]),(500,275,colors[0])]
    p2=[(120,355,colors[0]),(215,300,colors[1]),(310,335,colors[2]),(405,310,colors[3]),(500,350,colors[0])]
    e=[(0,1),(1,2),(2,3),(3,4)]
    b.append(molecule(p1,e)); b.append(molecule(p2,e)); b.append(text(310,430,"same atoms and bonds • different coordinates","small"))
    q1=[(700,280,bfs.PURPLE),(800,220,bfs.BLUE),(900,280,bfs.RED),(1000,220,bfs.GREEN),(1080,300,bfs.AMBER)]
    q2=[(720,370,bfs.PURPLE),(820,325,bfs.BLUE),(925,375,bfs.GREEN),(1030,340,bfs.RED)]
    b.append(molecule(q1,[(0,1),(1,2),(2,3),(3,4),(1,3)])); b.append(molecule(q2,[(0,1),(1,2),(2,3)])); b.append(text(890,430,"atom types • bonds • coordinates may all change","small"))
    b.append(text(600,520,"The probability space, constraints, and evaluation metric depend on which problem is being solved.","body")); save("mol3dopt_two_tasks","\n".join(b),550)

def coordinates_vs_torsions():
    b=[text(55,48,"Cartesian and torsional models place the prior on different geometry","title","start")]
    for x,title,fill,stroke in [(45,"Cartesian coordinates",bfs.PURPLE_SOFT,bfs.PURPLE),(625,"internal torsions",bfs.AMBER_LIGHT,bfs.AMBER)]: b.append(box(x,90,530,370,fill,stroke,28)); b.append(text(x+265,132,title,"label"))
    p=[(120,300,bfs.PURPLE),(230,220,bfs.BLUE),(345,305,bfs.GREEN),(465,220,bfs.RED)]
    b.append(molecule(p,[(0,1),(1,2),(2,3)]))
    for x,y,_ in p: b.append(f'<path d="M{x},{y-28} L{x+28},{y-55}" class="parrow"/>')
    b.append(text(310,410,"move every atom • remove translation and rotation","small"))
    q=[(700,300,bfs.PURPLE),(810,235,bfs.BLUE),(925,300,bfs.GREEN),(1040,235,bfs.RED)]
    b.append(molecule(q,[(0,1),(1,2),(2,3)])); b.append(f'<circle cx="867" cy="268" r="60" fill="none" stroke="{bfs.AMBER}" stroke-width="4" stroke-dasharray="8 7"/>'); b.append('<path d="M825,213 C875,170 945,205 944,260" class="arrow"/>')
    b.append(text(890,410,"rotate fragments • preserve local bond geometry","small"))
    b.append(text(600,515,"Cartesian models are topology-agnostic; torsional models build chemical constraints into the state space.","body")); save("mol3dopt_coordinates_torsions","\n".join(b),550)

def guided_generation():
    b=[text(55,48,"Guidance bends a learned molecular prior toward a condition","title","start")]
    b.append(box(45,90,1110,370,bfs.PURPLE_SOFT,bfs.PURPLE,30))
    for x,label,color in [(150,"noise",bfs.NEUTRAL),(470,"plausible",bfs.BLUE),(800,"conditioned",bfs.GREEN),(1060,"candidate",bfs.AMBER)]: b.append(f'<circle cx="{x}" cy="270" r="65" fill="white" stroke="{color}" stroke-width="4"/>'); b.append(text(x,365,label,"label"))
    b.append('<path d="M220,270 C280,220 335,220 395,260" class="arrow"/>'); b.append('<path d="M540,260 C610,220 670,225 725,265" class="parrow"/>'); b.append('<path d="M870,270 L985,270" class="arrow"/>')
    # molecule icons
    for x,y,c in [(125,250,bfs.BLUE),(165,290,bfs.RED),(485,240,bfs.PURPLE),(445,290,bfs.BLUE),(785,245,bfs.GREEN),(825,295,bfs.BLUE),(1040,240,bfs.AMBER),(1080,290,bfs.GREEN)]: b.append(atom(x,y,c,13))
    b.append(text(625,190,"property / pocket / scaffold", "label")); b.append('<path d="M625,205 L650,240" class="parrow"/>')
    b.append(text(600,505,"Stronger guidance improves the surrogate condition but can reduce diversity and leave the training distribution.","body")); save("mol3dopt_guided_generation","\n".join(b),540)

def optimization_funnel():
    b=[text(55,48,"A credible optimization loop narrows candidates through independent gates","title","start")]
    stages=[(45,110,210,"generate",bfs.PURPLE_SOFT,bfs.PURPLE),(315,135,210,"score & Pareto",bfs.BLUE_LIGHT,bfs.BLUE),(585,160,210,"synthesis",bfs.GREEN_LIGHT,bfs.GREEN),(855,185,300,"experiment",bfs.AMBER_LIGHT,bfs.AMBER)]
    for x,y,w,label,fill,stroke in stages: b.append(box(x,y,w,190,fill,stroke,26)); b.append(text(x+w/2,y+42,label,"label"))
    for x,y in [(255,210),(525,230),(795,250)]: b.append(f'<path d="M{x},{y} L{x+55},{y+8}" class="arrow"/>')
    for x,y,c in [(90,220,bfs.PURPLE),(150,255,bfs.BLUE),(205,205,bfs.GREEN)]: b.append(atom(x,y,c,13))
    b.append(text(420,210,"potency", "small")); b.append(text(420,245,"toxicity", "small")); b.append(text(420,280,"uncertainty", "small"))
    b.append(text(690,230,"route found?", "small")); b.append(text(690,270,"yield / cost", "small"))
    for i,h in enumerate([45,75,58]): b.append(f'<rect x="{915+i*45}" y="{320-h}" width="28" height="{h}" rx="8" fill="{[bfs.PURPLE,bfs.GREEN,bfs.AMBER][i]}"/>')
    b.append('<path d="M1010,385 C770,470 410,465 160,335" class="parrow"/>'); b.append(text(590,465,"failed assays update the model and the search distribution","small"))
    b.append(text(600,520,"A high oracle score is a proposal; synthesis and experiment decide whether it is a discovery.","body")); save("mol3dopt_optimization_funnel","\n".join(b),550)

def main(): two_generation_tasks(); coordinates_vs_torsions(); guided_generation(); optimization_funnel()
if __name__=="__main__": main()
