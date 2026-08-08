#!/usr/bin/env python3
"""Generate original figures for the materials-discovery blog post.

The SVG-first diagrams are original explanatory drawings. They synthesize the
post's argument rather than reproduce any slide or publication figure. The
convex-hull construction follows the standard thermodynamic definition used by
the Materials Project; the discovery-loop figure is conceptually informed by
Merchant et al. (2023), Szymanski et al. (2023), and Zeni et al. (2025).

Style: rounded flat forms, restrained solid colors, consistent strokes, and
minimal visual noise. License: same as the blog repository.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs

OUT = Path("assets/img/blog")
WIDTH = 1200


def document(body: str, height: int = 600) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.MUTED}"/></marker>
  <marker id="parrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{bfs.PURPLE}"/></marker>
  <style>
    text{{font-family:Arial,Helvetica,sans-serif;fill:{bfs.TEXT}}}
    .title{{font-size:25px;font-weight:700}} .label{{font-size:19px;font-weight:700}}
    .body{{font-size:17px}} .small{{font-size:15px;fill:{bfs.MUTED}}}
    .arrow{{fill:none;stroke:{bfs.MUTED};stroke-width:3;stroke-linecap:round;stroke-linejoin:round;marker-end:url(#arrow)}}
    .parrow{{fill:none;stroke:{bfs.PURPLE};stroke-width:4;stroke-linecap:round;stroke-linejoin:round;marker-end:url(#parrow)}}
  </style>
</defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def text(x, y, value, cls="body", anchor="middle"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def box(x, y, w, h, fill, stroke, radius=26):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'


def atom(x, y, color, radius=16):
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="4"/>'


def bond(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{bfs.SPINE}" stroke-width="6" stroke-linecap="round"/>'


def save(stem: str, body: str, height: int = 600):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"
    svg.write_text(document(body, height), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=WIDTH):
        raise RuntimeError(f"Could not render {stem}")


def composition_structure():
    b = [text(55, 50, "A formula does not specify a material", "title", "start")]
    b.append(box(45, 100, 250, 390, bfs.PURPLE_SOFT, bfs.PURPLE, 30))
    b.append(text(170, 148, "composition", "label"))
    b.append(text(170, 240, "A₂B₂", "title"))
    b.append(atom(128, 310, bfs.BLUE, 23)); b.append(atom(212, 310, bfs.AMBER, 23))
    b.append(text(170, 400, "which atoms", "small")); b.append(text(170, 430, "and how many", "small"))
    b.append('<path d="M305,295 L390,295" class="arrow"/>')

    panels = [(410, "layered polymorph", bfs.BLUE_LIGHT, bfs.BLUE), (795, "network polymorph", bfs.GREEN_LIGHT, bfs.GREEN)]
    for x, label, fill, stroke in panels:
        b.append(box(x, 100, 360, 390, fill, stroke, 30)); b.append(text(x + 180, 148, label, "label"))
        b.append(f'<rect x="{x+68}" y="190" width="224" height="190" rx="10" fill="white" stroke="{stroke}" stroke-width="3" stroke-dasharray="9 7"/>')
    # Layered unit cell.
    x = 410
    for y, ca, cb in [(225, bfs.BLUE, bfs.AMBER), (295, bfs.AMBER, bfs.BLUE), (350, bfs.BLUE, bfs.AMBER)]:
        b.append(bond(x+105, y, x+255, y)); b.append(atom(x+105, y, ca)); b.append(atom(x+255, y, cb))
    # Network unit cell.
    x = 795
    pts = [(x+180,215,bfs.BLUE),(x+105,285,bfs.AMBER),(x+255,285,bfs.AMBER),(x+180,355,bfs.BLUE)]
    for i,j in [(0,1),(0,2),(1,3),(2,3),(1,2)]: b.append(bond(pts[i][0],pts[i][1],pts[j][0],pts[j][1]))
    for px,py,c in pts: b.append(atom(px,py,c))
    b.append(text(590, 430, "same formula • different bands", "small")); b.append(text(975, 430, "same formula • different stiffness", "small"))
    b.append(text(600, 555, "The lattice and fractional coordinates choose a periodic arrangement—and therefore the property.", "body"))
    save("matdisc_composition_structure", "\n".join(b), 590)


def convex_hull():
    b = [text(55, 46, "The convex hull compares a candidate with every competing phase mixture", "title", "start")]
    left, top, right, bottom = 120, 105, 1110, 475
    b.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{bfs.SPINE}" stroke-width="3"/>')
    b.append(f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="{bfs.SPINE}" stroke-width="3"/>')
    b.append(text(615, 535, "composition xᴮ", "body")); b.append(f'<text x="42" y="300" class="body" text-anchor="middle" transform="rotate(-90 42 300)">formation energy (eV/atom)</text>')
    # Scale: y=475 at 0, y=135 at -0.5.
    def px(x): return left + x * (right-left)
    def py(e): return bottom + e * 680
    # Hull A -> AB -> B.
    hull = [(0.0,0.0),(0.5,-0.40),(1.0,0.0)]
    b.append('<polyline points="'+' '.join(f'{px(x):.1f},{py(e):.1f}' for x,e in hull)+f'" fill="none" stroke="{bfs.BLUE}" stroke-width="7" stroke-linejoin="round"/>')
    # Stable points.
    for x,e,label in [(0,0,"A"),(.5,-.40,"AB"),(1,0,"B")]:
        b.append(f'<circle cx="{px(x)}" cy="{py(e)}" r="13" fill="{bfs.BLUE}" stroke="white" stroke-width="4"/>'); b.append(text(px(x),py(e)-24,label,"label"))
    # Candidate and tie line value.
    cx, ce, he = .75, -.12, -.20
    b.append(f'<line x1="{px(cx)}" y1="{py(he)}" x2="{px(cx)}" y2="{py(ce)}" stroke="{bfs.RED}" stroke-width="5" stroke-dasharray="8 6"/>')
    b.append(f'<circle cx="{px(cx)}" cy="{py(ce)}" r="14" fill="{bfs.RED}" stroke="white" stroke-width="4"/>')
    b.append(text(px(cx)+24,py(ce)-10,"AB₃ candidate", "label", "start"))
    b.append(text(px(cx)+24,py(ce)+20,"ΔEᶠ = −0.12", "small", "start"))
    b.append(text(px(cx)-18,(py(he)+py(ce))/2,"0.08", "label", "end"))
    b.append(text(px(cx)-18,(py(he)+py(ce))/2+23,"eV/atom above hull", "small", "end"))
    # Competing mixture annotation.
    b.append(box(150, 120, 270, 96, bfs.GREEN_LIGHT, bfs.GREEN, 20)); b.append(text(285,158,"at xᴮ = 0.75", "small")); b.append(text(285,190,"½ AB + ½ B = −0.20", "label"))
    b.append('<path d="M425,178 C535,170 655,205 852,330" class="arrow"/>')
    b.append(text(600, 575, "Negative formation energy is insufficient: AB₃ still prefers to decompose into AB and B at 0 K.", "body"))
    save("matdisc_convex_hull", "\n".join(b), 610)


def material_reality():
    b = [text(55, 48, "The calculated bulk crystal is one layer of the physical material", "title", "start")]
    stages = [
        (45, "ideal cell", "ordered, periodic", bfs.PURPLE_SOFT, bfs.PURPLE),
        (325, "polymorphs", "competing lattices", bfs.BLUE_LIGHT, bfs.BLUE),
        (605, "defects", "vacancies, dopants", bfs.AMBER_LIGHT, bfs.AMBER),
        (885, "finite T + process", "entropy, pressure, time", bfs.GREEN_LIGHT, bfs.GREEN),
    ]
    for x, title, sub, fill, stroke in stages:
        b.append(box(x, 110, 250, 340, fill, stroke, 28)); b.append(text(x+125,154,title,"label")); b.append(text(x+125,415,sub,"small"))
    # Ideal lattice.
    for yy in [230,300]:
        for xx in [95,165,235]: b.append(atom(xx,yy,bfs.BLUE if (xx+yy)//10%2 else bfs.AMBER,12))
    # Two polymorph icons.
    for dx,dy in [(375,235),(435,235),(495,235)]: b.append(atom(dx,dy,bfs.BLUE,12))
    for dx,dy in [(375,305),(435,275),(495,305)]: b.append(atom(dx,dy,bfs.AMBER,12))
    # Vacancy and dopant.
    for yy in [235,305]:
        for xx in [655,725,795]:
            if (xx,yy)==(725,235): b.append(f'<circle cx="{xx}" cy="{yy}" r="13" fill="white" stroke="{bfs.RED}" stroke-width="3" stroke-dasharray="5 4"/>')
            else: b.append(atom(xx,yy,bfs.GREEN if (xx,yy)==(795,305) else bfs.BLUE,12))
    # Furnace/process icon.
    b.append(f'<path d="M960,330 C930,285 975,260 958,218 C1010,240 1015,286 990,330 Z" fill="{bfs.AMBER}"/>')
    b.append(f'<path d="M1030,330 C1010,295 1042,270 1035,238 C1075,267 1070,305 1055,330 Z" fill="{bfs.RED}"/>')
    b.append(f'<rect x="940" y="340" width="150" height="20" rx="10" fill="{bfs.GREEN}"/>')
    for x in [300,580,860]: b.append(f'<path d="M{x},280 L{x+22},280" class="arrow"/>')
    b.append(text(600, 520, "Each layer can change which phase forms and which electronic, mechanical, or transport property is measured.", "body"))
    save("matdisc_material_reality", "\n".join(b), 560)


def closed_loop():
    b = [text(55, 48, "Materials discovery closes only when experiment changes the next proposal", "title", "start")]
    stages = [
        (60, 145, "propose", "composition + structure", bfs.PURPLE_SOFT, bfs.PURPLE),
        (355, 95, "compute", "relax + properties + hull", bfs.BLUE_LIGHT, bfs.BLUE),
        (650, 145, "synthesize", "precursors + conditions", bfs.AMBER_LIGHT, bfs.AMBER),
        (945, 95, "characterize", "phase + property + yield", bfs.GREEN_LIGHT, bfs.GREEN),
    ]
    for x,y,title,sub,fill,stroke in stages:
        b.append(box(x,y,205,190,fill,stroke,28)); b.append(text(x+102,y+55,title,"label")); b.append(text(x+102,y+105,sub.split(" + ")[0],"small"));
        if " + " in sub: b.append(text(x+102,y+133,"+ "+sub.split(" + ",1)[1],"small"))
    b.append('<path d="M270,225 C300,170 320,160 345,160" class="arrow"/>')
    b.append('<path d="M565,160 C610,165 625,190 642,220" class="arrow"/>')
    b.append('<path d="M860,225 C895,170 915,160 935,160" class="arrow"/>')
    b.append('<path d="M1050,300 C990,490 310,520 165,350" class="parrow"/>')
    b.append(box(405, 370, 390, 92, "white", bfs.PURPLE, 22)); b.append(text(600,405,"active learning", "label")); b.append(text(600,435,"utility • uncertainty • diversity • cost", "small"))
    b.append(text(600, 555, "Failed recipes and impurity phases are data: they update both the synthesis model and the candidate ranking.", "body"))
    save("matdisc_closed_loop", "\n".join(b), 590)


def main():
    composition_structure()
    convex_hull()
    material_reality()
    closed_loop()


if __name__ == "__main__":
    main()
