#!/usr/bin/env python3
"""Original SVG-first figures for the crystal prediction/design post.

Flat geometric forms, restrained colors, and consistent strokes are used. No
paper, slide, or Flaticon artwork is copied. PNGs are deterministic previews.
License: CC BY 4.0 with the blog post.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import blog_figure_style as bfs
OUT=ROOT/'assets'/'img'/'blog'
TEXT,MUTED,SPINE=bfs.TEXT,bfs.MUTED,bfs.SPINE
P,PS=bfs.PURPLE,bfs.PURPLE_SOFT; B,BS=bfs.BLUE,bfs.BLUE_LIGHT
T,TS=bfs.TEAL,bfs.TEAL_LIGHT; A,AS=bfs.AMBER,bfs.AMBER_LIGHT; R,RS=bfs.RED,bfs.RED_LIGHT

def doc(body):
 return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{TEXT}}}.title{{font-size:22px;font-weight:700}}.label{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{MUTED}}}.arrow{{stroke:{MUTED};stroke-width:2.5;fill:none;marker-end:url(#arrow)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
def box(x,y,w,h,c): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{c}" stroke="{SPINE}" stroke-width="2"/>'
def lattice(x,y,s=1):
 out=''
 for i in range(4):
  for j in range(4):
   xx=x+(i*50+j*18)*s; yy=y+j*43*s
   out+=f'<circle cx="{xx}" cy="{yy}" r="{9*s}" fill="{[P,T,A][(i+j)%3]}" stroke="white" stroke-width="2"/>'
 return out
def write(n,b):
 OUT.mkdir(parents=True,exist_ok=True); svg=OUT/f'crystgen_{n}.svg'; png=OUT/f'crystgen_{n}.png'; svg.write_text(doc(b),encoding='utf-8')
 if not bfs.render_svg_preview(svg,png,width=1000): raise RuntimeError('SVG renderer unavailable')

def periodic_representation():
 b=f'''<text x="500" y="42" text-anchor="middle" class="title">One infinite crystal has many equivalent finite descriptions</text>
 {box(35,90,280,370,BS)}<text x="175" y="128" text-anchor="middle" class="label">unit cell</text><path d="M80 205L230 175L275 300L125 330Z" fill="white" stroke="{B}" stroke-width="3"/>{lattice(105,205,.65)}<text x="175" y="380" text-anchor="middle" class="body">lattice L + fractional sites S</text><text x="175" y="411" text-anchor="middle" class="small">Cartesian position: x = Ls</text>
 <path d="M325 275L365 275" class="arrow"/>{box(380,90,240,370,PS)}<text x="500" y="128" text-anchor="middle" class="label">equivalent descriptions</text><rect x="418" y="165" width="164" height="54" rx="12" fill="white"/><text x="500" y="198" text-anchor="middle" class="body">s and s + n</text><rect x="418" y="245" width="164" height="54" rx="12" fill="white"/><text x="500" y="278" text-anchor="middle" class="body">permuted sites</text><rect x="418" y="325" width="164" height="54" rx="12" fill="white"/><text x="500" y="358" text-anchor="middle" class="body">rotated cell</text>
 <path d="M630 275L670 275" class="arrow"/>{box(685,90,280,370,TS)}<text x="825" y="128" text-anchor="middle" class="label">one physical prediction</text>{lattice(740,185,.65)}<rect x="740" y="360" width="170" height="54" rx="12" fill="white"/><text x="825" y="394" text-anchor="middle" class="body">energy, band gap, force</text>
 <text x="500" y="515" text-anchor="middle" class="body">A useful representation changes coordinates without changing the material.</text>'''; write('periodic_representation',b)

def periodic_graph():
 b=f'''<text x="500" y="42" text-anchor="middle" class="title">A periodic graph makes neighbors cross the unit-cell boundary</text>
 {box(45,92,430,370,BS)}<text x="260" y="130" text-anchor="middle" class="label">minimum-image neighborhood</text><rect x="100" y="175" width="320" height="220" fill="white" stroke="{B}" stroke-width="3" stroke-dasharray="8 6"/><circle cx="135" cy="285" r="14" fill="{R}"/><circle cx="385" cy="285" r="14" fill="{T}"/><circle cx="455" cy="285" r="14" fill="{R}"/><line x1="135" y1="285" x2="385" y2="285" stroke="{MUTED}" stroke-width="4"/><line x1="385" y1="285" x2="455" y2="285" stroke="{MUTED}" stroke-width="4"/><text x="260" y="425" text-anchor="middle" class="small">the image at the right is the left atom in the next cell</text>
 <path d="M487 275L530 275" class="arrow"/>{box(545,92,410,370,PS)}<text x="750" y="130" text-anchor="middle" class="label">equivariant message passing</text><circle cx="650" cy="275" r="18" fill="{P}"/><circle cx="750" cy="205" r="18" fill="{T}"/><circle cx="850" cy="285" r="18" fill="{A}"/><circle cx="750" cy="355" r="18" fill="{R}"/><path d="M650 275L750 205L850 285L750 355Z" fill="none" stroke="{MUTED}" stroke-width="4"/><text x="750" y="405" text-anchor="middle" class="body">scalar output invariant</text><text x="750" y="432" text-anchor="middle" class="small">vector forces rotate with the crystal</text>
 <text x="500" y="515" text-anchor="middle" class="body">Periodic images are part of the graph; they are not extra atoms in the material.</text>'''; write('periodic_graph',b)

def generative_variables():
 b=f'''<text x="500" y="42" text-anchor="middle" class="title">Crystal generation couples three spaces with different constraints</text>
 {box(45,100,260,330,AS)}<text x="175" y="140" text-anchor="middle" class="label">composition A</text><circle cx="125" cy="220" r="25" fill="{P}"/><circle cx="175" cy="220" r="25" fill="{T}"/><circle cx="225" cy="220" r="25" fill="{A}"/><text x="175" y="290" text-anchor="middle" class="body">discrete elements + counts</text><text x="175" y="330" text-anchor="middle" class="small">charge and stoichiometry</text><text x="175" y="355" text-anchor="middle" class="small">must be chemically plausible</text>
 {box(370,100,260,330,BS)}<text x="500" y="140" text-anchor="middle" class="label">lattice L</text><path d="M420 230L555 190L590 310L455 350Z" fill="white" stroke="{B}" stroke-width="4"/><text x="500" y="385" text-anchor="middle" class="body">lengths, angles, volume</text>
 {box(695,100,260,330,TS)}<text x="825" y="140" text-anchor="middle" class="label">fractional sites S</text>{lattice(760,200,.65)}<text x="825" y="385" text-anchor="middle" class="body">a torus: s ≡ s + n</text>
 <path d="M305 265L360 265" class="arrow"/><path d="M630 265L685 265" class="arrow"/><rect x="165" y="475" width="670" height="48" rx="14" fill="{RS}"/><text x="500" y="505" text-anchor="middle" class="body">A valid sample requires joint compatibility—not three individually plausible outputs.</text>'''; write('generative_variables',b)

def validation_funnel():
 b=f'''<text x="500" y="42" text-anchor="middle" class="title">Generation proposes; relaxation and first-principles calculations adjudicate</text>
 {box(35,100,175,150,BS)}<text x="122" y="140" text-anchor="middle" class="label">generate</text><text x="122" y="180" text-anchor="middle" class="body">A, L, S</text><text x="122" y="215" text-anchor="middle" class="small">many candidates</text>
 {box(260,100,175,150,PS)}<text x="347" y="140" text-anchor="middle" class="label">deduplicate</text><text x="347" y="180" text-anchor="middle" class="body">symmetry + matching</text><text x="347" y="215" text-anchor="middle" class="small">uniqueness, novelty</text>
 {box(485,100,175,150,TS)}<text x="572" y="140" text-anchor="middle" class="label">relax</text><text x="572" y="180" text-anchor="middle" class="body">ML potential</text><text x="572" y="215" text-anchor="middle" class="small">forces → local minimum</text>
 {box(710,100,255,150,AS)}<text x="837" y="140" text-anchor="middle" class="label">DFT validation</text><text x="837" y="180" text-anchor="middle" class="body">energy, forces, properties</text><text x="837" y="215" text-anchor="middle" class="small">convex hull + dynamic tests</text>
 <path d="M210 175L250 175" class="arrow"/><path d="M435 175L475 175" class="arrow"/><path d="M660 175L700 175" class="arrow"/>
 <path d="M830 265C750 350,640 365,560 305" fill="none" stroke="{R}" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#arrow)"/><rect x="654" y="360" width="192" height="28" rx="9" fill="white"/><text x="750" y="379" text-anchor="middle" class="small">large relaxation drift rejects</text>
 <rect x="65" y="320" width="410" height="155" rx="20" fill="{RS}" stroke="{SPINE}" stroke-width="2"/><text x="270" y="360" text-anchor="middle" class="label">selection bias</text><text x="270" y="400" text-anchor="middle" class="body">DFT sees only candidates favored</text><text x="270" y="428" text-anchor="middle" class="body">by predictors and filters</text><text x="270" y="455" text-anchor="middle" class="small">report every denominator and threshold</text>
 <rect x="535" y="410" width="400" height="65" rx="17" fill="white" stroke="{T}" stroke-width="3"/><text x="735" y="439" text-anchor="middle" class="body">stable ≠ synthesizable ≠ useful</text><text x="735" y="463" text-anchor="middle" class="small">kinetics, defects, processing, and conditions remain</text>
 <text x="500" y="525" text-anchor="middle" class="small">The pipeline is a sequence of increasingly expensive attempts to falsify a proposal.</text>'''; write('validation_funnel',b)

if __name__=='__main__': periodic_representation(); periodic_graph(); generative_variables(); validation_funnel()
