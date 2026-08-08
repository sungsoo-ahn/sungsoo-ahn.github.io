#!/usr/bin/env python3
"""Generate original figures for the molecular simulation and ML force-fields post.

All artwork is original, editable SVG with deterministic PNG previews. The
figures use rounded geometric silhouettes, restrained solid colors, consistent
strokes, and minimal visual noise; no lecture or paper figure is reproduced.

Conceptual sources
------------------
* Behler and Parrinello (2007), https://doi.org/10.1103/PhysRevLett.98.146401
* Zhang et al. (2019), https://doi.org/10.1016/j.cpc.2020.107206
* Fu et al. (2025), https://proceedings.mlr.press/v267/fu25b.html

License: same as the blog.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs

OUT = Path("assets/img/blog")
WIDTH = 1200


def document(body: str, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">
 <defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker>
  <marker id="purpleArrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker>
  <style>text {{font-family:Arial,Helvetica,sans-serif;fill:{bfs.TEXT}}}.title{{font-size:24px;font-weight:700}}.label{{font-size:19px;font-weight:600}}.body{{font-size:17px}}.small{{font-size:15px;fill:{bfs.MUTED}}}.arrow{{fill:none;stroke:{bfs.MUTED};stroke-width:3;stroke-linecap:round;marker-end:url(#arrow)}}.parrow{{fill:none;stroke:{bfs.PURPLE};stroke-width:4;stroke-linecap:round;marker-end:url(#purpleArrow)}}</style>
 </defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x: float, y: float, w: float, h: float, fill: str, stroke: str, r: int = 26) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def atom(x: float, y: float, color: str, r: int = 17) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="white" stroke-width="4"/>'


def save(stem: str, body: str, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{stem}.svg"; png = OUT / f"{stem}.png"
    svg.write_text(document(body, height), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=WIDTH):
        raise RuntimeError(f"Could not render {svg}")


def pes_dynamics() -> None:
    b = [text(55, 48, "A potential-energy surface turns geometry into motion", "title", "start")]
    b.append(box(45, 90, 680, 390, bfs.PURPLE_SOFT, bfs.PURPLE, 30))
    # Nested energy contours.
    for rx, ry, color in [(270,150,bfs.SPINE),(220,120,bfs.BLUE),(160,85,bfs.PURPLE),(90,46,bfs.GREEN)]:
        b.append(f'<ellipse cx="385" cy="285" rx="{rx}" ry="{ry}" fill="none" stroke="{color}" stroke-width="4" opacity="0.85"/>')
    b.append('<path d="M120,175 C205,150 255,225 315,210 C390,192 420,330 505,295 C570,268 625,320 665,365" class="parrow"/>')
    for x,y,c in [(120,175,bfs.AMBER),(315,210,bfs.BLUE),(505,295,bfs.RED),(665,365,bfs.GREEN)]: b.append(atom(x,y,c,13))
    b.append(text(385, 455, "force = − energy gradient", "label"))
    b.append(box(775, 90, 380, 390, "white", bfs.SPINE, 30))
    b.append(text(965, 132, "integrator", "label"))
    b.append(text(820, 205, "positions", "body", "start")); b.append(text(1110,205,"R(t)","label","end"))
    b.append('<path d="M825,235 L1100,235" class="arrow"/>')
    b.append(text(820, 285, "forces", "body", "start")); b.append(text(1110,285,"−∇U", "label","end"))
    b.append('<path d="M825,315 L1100,315" class="arrow"/>')
    b.append(text(820, 365, "velocities", "body", "start")); b.append(text(1110,365,"V(t)","label","end"))
    b.append(text(965, 430, "repeat millions of times", "small"))
    b.append(text(600, 525, "The learned model is queried on configurations created by its own previous predictions.", "body"))
    save("molsim_pes_dynamics", "\n".join(b), 560)


def active_learning_loop() -> None:
    b = [text(55, 48, "Reliable training data are discovered in a simulation loop", "title", "start")]
    boxes = [(55,175,220,"reference labels",bfs.BLUE_LIGHT,bfs.BLUE),(360,95,250,"fit potential",bfs.PURPLE_SOFT,bfs.PURPLE),(360,310,250,"explore dynamics",bfs.GREEN_LIGHT,bfs.GREEN),(700,175,220,"uncertainty gate",bfs.AMBER_LIGHT,bfs.AMBER)]
    for x,y,w,label,fill,stroke in boxes:
        b.append(box(x,y,w,105,fill,stroke,24)); b.append(text(x+w/2,y+47,label,"label"))
    b.append('<path d="M280,208 C320,180 330,165 352,155" class="arrow"/>')
    b.append('<path d="M485,205 L485,300" class="arrow"/>')
    b.append('<path d="M620,360 C680,345 710,310 760,285" class="arrow"/>')
    b.append('<path d="M700,218 C565,215 350,275 285,245" class="parrow"/>')
    b.append(text(485, 265, "model rollout", "small"))
    b.append(text(640, 88, "query only informative states", "small"))
    b.append(box(980, 135, 175, 250, "white", bfs.SPINE, 25))
    b.append(text(1067,175,"monitor", "label"))
    for i,(label,color) in enumerate([("committee",bfs.PURPLE),("distance",bfs.BLUE),("physics",bfs.GREEN)]):
        y=225+i*60; b.append(f'<circle cx="1010" cy="{y}" r="10" fill="{color}"/>'); b.append(text(1035,y+6,label,"small","start"))
    b.append(text(600, 475, "Uncertainty is an acquisition signal, not a guarantee that the true error is small.", "body"))
    save("molsim_active_learning_loop", "\n".join(b), 520)


def static_vs_rollout() -> None:
    b = [text(55, 48, "A small static error can still create a catastrophic rollout", "title", "start")]
    panels=[(45,"held-out snapshots",bfs.BLUE_LIGHT,bfs.BLUE),(625,"closed-loop dynamics",bfs.RED_LIGHT,bfs.RED)]
    for x,title,fill,stroke in panels:
        b.append(box(x,90,530,370,fill,stroke,28)); b.append(text(x+265,132,title,"label"))
        b.append(f'<line x1="{x+65}" y1="395" x2="{x+465}" y2="395" stroke="{bfs.SPINE}" stroke-width="3"/>')
        b.append(f'<line x1="{x+65}" y1="395" x2="{x+65}" y2="180" stroke="{bfs.SPINE}" stroke-width="3"/>')
    # Close predictions on sampled points.
    pts=[]
    for k in range(8):
        x=120+k*52; y=365-125*math.exp(-((k-3.5)/2.2)**2); pts.append((x,y))
        b.append(atom(x,y,bfs.BLUE,8)); b.append(atom(x+5,y-5,bfs.PURPLE,7))
    b.append(text(310,425,"low energy / force error", "small"))
    # Stable then blow-up trajectory.
    b.append(f'<path d="M690,350 C760,320 810,345 855,292 C900,242 925,305 960,260 C1000,210 1030,235 1090,155" fill="none" stroke="{bfs.RED}" stroke-width="6" stroke-linecap="round"/>')
    b.append(f'<line x1="945" y1="180" x2="945" y2="395" stroke="{bfs.AMBER}" stroke-width="3" stroke-dasharray="8 7"/>')
    b.append(text(945,420,"first extrapolative state", "small"))
    b.append(text(600,515,"Deployment tests must measure conservation, geometry, uncertainty, and observables over long rollouts.","body"))
    save("molsim_static_vs_rollout", "\n".join(b), 550)


def physics_to_observables() -> None:
    b=[text(55,48,"Validation must connect microscopic physics to ensemble observables","title","start")]
    stages=[(45,"energy model",bfs.PURPLE_SOFT,bfs.PURPLE),(335,"trajectory",bfs.BLUE_LIGHT,bfs.BLUE),(625,"ensemble",bfs.GREEN_LIGHT,bfs.GREEN),(915,"observable",bfs.AMBER_LIGHT,bfs.AMBER)]
    for x,label,fill,stroke in stages:
        b.append(box(x,145,240,190,fill,stroke,26)); b.append(text(x+120,185,label,"label"))
    # energy icon
    b.append('<path d="M85,280 C125,205 170,305 245,225" fill="none" stroke="'+bfs.PURPLE+'" stroke-width="5"/>')
    # trajectory icon
    b.append('<path d="M375,265 C415,205 480,305 535,220" fill="none" stroke="'+bfs.BLUE+'" stroke-width="5"/>')
    for x,y in [(375,265),(445,245),(535,220)]: b.append(atom(x,y,bfs.BLUE,9))
    # ensemble atoms
    for x,y,c in [(675,235,bfs.GREEN),(740,215,bfs.BLUE),(810,260,bfs.AMBER),(705,285,bfs.PURPLE),(835,205,bfs.RED)]: b.append(atom(x,y,c,14))
    # observable bars
    for i,h in enumerate([45,90,65,110]): b.append(f'<rect x="{960+i*38}" y="{300-h}" width="24" height="{h}" rx="8" fill="{[bfs.PURPLE,bfs.BLUE,bfs.GREEN,bfs.AMBER][i]}"/>')
    for x in [290,580,870]: b.append(f'<path d="M{x},240 L{x+35},240" class="arrow"/>')
    b.append(text(165,380,"smooth • conservative","small")); b.append(text(455,380,"stable • equilibrated","small")); b.append(text(745,380,"correct distribution","small")); b.append(text(1035,380,"structure • free energy • transport","small"))
    b.append(text(600,480,"Agreement at the first box is necessary; scientific validity is decided at the last box.","body"))
    save("molsim_physics_to_observables", "\n".join(b), 520)


def main() -> None:
    pes_dynamics(); active_learning_loop(); static_vs_rollout(); physics_to_observables()


if __name__ == "__main__": main()
