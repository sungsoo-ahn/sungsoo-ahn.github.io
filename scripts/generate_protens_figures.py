#!/usr/bin/env python3
"""Generate original protein-ensemble and learned-dynamics blog figures.

Original SVG-first diagrams with deterministic PNG previews. The visual style
uses rounded flat forms, restrained solid colors, and consistent strokes. No
lecture or publication figure is reproduced.

Conceptual sources: Prinz et al. (2011), Noé et al. (2019), Mardt et al.
(2018), Jing et al. (2024). License: same as the blog.
"""
from __future__ import annotations
from html import escape
from pathlib import Path
import math, sys
sys.path.insert(0,str(Path(__file__).resolve().parent)); import blog_figure_style as bfs
OUT=Path("assets/img/blog"); WIDTH=1200
def document(body,height=550):
 return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker><marker id="parrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{bfs.TEXT}}}.title{{font-size:24px;font-weight:700}}.label{{font-size:19px;font-weight:600}}.body{{font-size:17px}}.small{{font-size:15px;fill:{bfs.MUTED}}}.arrow{{fill:none;stroke:{bfs.MUTED};stroke-width:3;stroke-linecap:round;marker-end:url(#arrow)}}.parrow{{fill:none;stroke:{bfs.PURPLE};stroke-width:4;stroke-linecap:round;marker-end:url(#parrow)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
def text(x,y,s,cls="body",anchor="middle"): return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'
def box(x,y,w,h,fill,stroke,r=26): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
def dot(x,y,c,r=10): return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="white" stroke-width="3"/>'
def save(stem,body,height=550):
 OUT.mkdir(parents=True,exist_ok=True); svg=OUT/f"{stem}.svg"; png=OUT/f"{stem}.png"; svg.write_text(document(body,height),encoding="utf-8")
 if not bfs.render_svg_preview(svg,png,width=WIDTH): raise RuntimeError(stem)

def landscape():
 b=[text(55,48,"A protein ensemble occupies basins, not one structure","title","start")]
 b.append(box(45,90,1110,380,bfs.PURPLE_SOFT,bfs.PURPLE,30))
 pts=[]
 for k in range(101):
  x=85+10*k; u=(x-85)/1000*10; y=200+125*(0.35*math.sin(1.15*u)+0.22*math.sin(2.7*u))+70
  pts.append((x,y))
 b.append('<polyline points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in pts)+f'" fill="none" stroke="{bfs.PURPLE}" stroke-width="6" stroke-linejoin="round"/>')
 states=[(245,330,"A",bfs.BLUE),(565,355,"B",bfs.GREEN),(925,315,"C",bfs.AMBER)]
 for x,y,l,c in states:
  for dx,dy in [(-24,-5),(0,8),(24,-8),(-8,-22),(15,-27)]: b.append(dot(x+dx,y+dy,c,8))
  b.append(text(x,420,l,"label"))
 b.append('<path d="M315,260 C390,205 455,210 505,275" class="arrow"/>'); b.append('<path d="M640,285 C735,185 825,205 870,275" class="arrow"/>')
 b.append(text(405,190,"rare transition","small")); b.append(text(760,185,"rare transition","small"))
 b.append(text(600,515,"Free energy sets equilibrium populations; barrier-crossing dynamics sets how quickly they exchange.","body")); save("protens_free_energy_landscape","\n".join(b),550)

def msm():
 b=[text(55,48,"A Markov state model compresses trajectories into lagged transitions","title","start")]
 b.append(box(45,95,400,350,bfs.BLUE_LIGHT,bfs.BLUE,28)); b.append(text(245,135,"molecular trajectory","label"))
 colors=[bfs.BLUE,bfs.BLUE,bfs.GREEN,bfs.GREEN,bfs.GREEN,bfs.AMBER,bfs.AMBER,bfs.GREEN,bfs.BLUE]
 for i,c in enumerate(colors):
  x=85+i*38; y=260+35*math.sin(i*1.3); b.append(dot(x,y,c,11))
  if i<len(colors)-1: b.append(f'<line x1="{x+12}" y1="{y}" x2="{x+26}" y2="{260+35*math.sin((i+1)*1.3)}" stroke="{bfs.MUTED}" stroke-width="3"/>')
 b.append(text(245,395,"cluster configurations into metastable states","small"))
 b.append('<path d="M455,270 L565,270" class="arrow"/>')
 b.append(box(570,95,250,350,"white",bfs.SPINE,28)); b.append(text(695,135,"transition matrix","label"))
 for i,row in enumerate([[.92,.07,.01],[.05,.90,.05],[.01,.09,.90]]):
  for j,v in enumerate(row): b.append(f'<rect x="{615+j*65}" y="{185+i*65}" width="55" height="55" rx="12" fill="{bfs.PURPLE}" opacity="{.15+.75*v:.2f}"/>'); b.append(text(642+j*65,220+i*65,f"{v:.2f}","small"))
 b.append('<path d="M830,270 L930,270" class="arrow"/>')
 b.append(box(935,95,220,350,bfs.GREEN_LIGHT,bfs.GREEN,28)); b.append(text(1045,135,"slow kinetics","label")); b.append(text(1045,210,"stationary weights","small")); b.append(text(1045,265,"implied timescales","small")); b.append(text(1045,320,"transition paths","small")); b.append(text(1045,380,"uncertainty","small"))
 b.append(text(600,505,"The lag time must be long enough for memory within each discrete state to decay.","body")); save("protens_markov_state_model","\n".join(b),540)

def equilibrium_kinetics():
 b=[text(55,48,"Matching equilibrium populations does not determine kinetics","title","start")]
 for x,title,fill,stroke in [(45,"reference dynamics",bfs.GREEN_LIGHT,bfs.GREEN),(625,"fast surrogate",bfs.RED_LIGHT,bfs.RED)]: b.append(box(x,90,530,370,fill,stroke,28)); b.append(text(x+265,132,title,"label"))
 for x0,fast in [(45,False),(625,True)]:
  for x,y,l,c in [(x0+130,270,"A",bfs.BLUE),(x0+400,270,"B",bfs.AMBER)]: b.append(f'<circle cx="{x}" cy="{y}" r="62" fill="white" stroke="{c}" stroke-width="4"/>'); b.append(text(x,y+7,l,"label"))
  if fast:
   b.append(f'<path d="M{x0+200},245 L{x0+330},245" class="parrow"/>'); b.append(f'<path d="M{x0+330},300 L{x0+200},300" class="parrow"/>'); b.append(text(x0+265,390,"many rapid switches","small"))
  else:
   b.append(f'<path d="M{x0+200},260 L{x0+330},260" class="arrow"/>'); b.append(text(x0+265,390,"rare barrier crossings","small"))
  b.append(text(x0+265,180,"50% A • 50% B","small"))
 b.append(text(600,515,"The two models share the same stationary distribution but disagree on relaxation time and pathway statistics.","body")); save("protens_equilibrium_vs_kinetics","\n".join(b),550)

def validation():
 b=[text(55,48,"Protein-ensemble models need layered validation","title","start")]
 stages=[(45,"geometry",bfs.PURPLE_SOFT,bfs.PURPLE),(325,"equilibrium",bfs.BLUE_LIGHT,bfs.BLUE),(605,"kinetics",bfs.GREEN_LIGHT,bfs.GREEN),(885,"experiment",bfs.AMBER_LIGHT,bfs.AMBER)]
 for x,l,fill,stroke in stages: b.append(box(x,130,250,250,fill,stroke,26)); b.append(text(x+125,175,l,"label"))
 labels=[["bonds / clashes","chirality / symmetry"],["state populations","free-energy gaps"],["timescales","paths / correlations"],["NMR / HDX","FRET / kinetics"]]
 for k,(x,*_) in enumerate(stages):
  b.append(text(x+125,245,labels[k][0],"small")); b.append(text(x+125,295,labels[k][1],"small"))
 for x in [300,580,860]: b.append(f'<path d="M{x},255 L{x+22},255" class="arrow"/>')
 b.append('<path d="M1010,405 C770,480 430,480 170,405" class="parrow"/>'); b.append(text(590,408,"failures reveal force-field, sampling, representation, or conditioning gaps","small"))
 b.append(text(600,520,"A model is validated against the scientific claim, not against one convenient structural metric.","body")); save("protens_validation_layers","\n".join(b),550)

def main(): landscape(); msm(); equilibrium_kinetics(); validation()
if __name__=="__main__": main()
