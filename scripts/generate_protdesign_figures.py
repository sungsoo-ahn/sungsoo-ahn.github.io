#!/usr/bin/env python3
"""Generate original figures for the protein-design post.

The SVG-first diagrams use a flat, restrained visual language and are original
explanatory syntheses. No lecture, paper, or Flaticon artwork is copied.
PNG files are deterministic previews. License: CC BY 4.0 with the blog post.
"""

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


def doc(body: str, width: int = 1000, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{TEXT}}}.title{{font-size:22px;font-weight:700}}.label{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{MUTED}}}.arrow{{stroke:{MUTED};stroke-width:2.5;fill:none;marker-end:url(#arrow)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x, y, w, h, fill):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" stroke="{SPINE}" stroke-width="2"/>'


def chain(x, y, color, scale=1.0):
    pts = [(0, 35), (42, 5), (88, 23), (122, 67), (92, 108), (38, 96), (10, 68)]
    q = " ".join(f"{x+a*scale},{y+b*scale}" for a, b in pts)
    dots = "".join(f'<circle cx="{x+a*scale}" cy="{y+b*scale}" r="{8*scale}" fill="{color}" stroke="white" stroke-width="2"/>' for a,b in pts)
    return f'<polyline points="{q}" fill="none" stroke="{MUTED}" stroke-width="{5*scale}" stroke-linejoin="round" stroke-linecap="round"/>{dots}'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"protdesign_{name}.svg"
    png = OUT / f"protdesign_{name}.png"
    svg.write_text(doc(body), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=1000):
        raise RuntimeError("SVG renderer unavailable")


def variable_choices():
    body = f'''<text x="500" y="42" text-anchor="middle" class="title">Protein design changes when the generated variable changes</text>
    {box(35,90,210,360,BLUE_LIGHT)}<text x="140" y="128" text-anchor="middle" class="label">sequence design</text>
    <text x="140" y="170" text-anchor="middle" class="small">condition: family or property</text><rect x="69" y="210" width="142" height="48" rx="12" fill="white"/><text x="140" y="241" text-anchor="middle" class="body">MKWL...GY</text><path d="M140 278L140 325" class="arrow"/><text x="140" y="359" text-anchor="middle" class="body">sequence</text><text x="140" y="405" text-anchor="middle" class="small">Does it fold and function?</text>
    {box(275,90,210,360,PURPLE_SOFT)}<text x="380" y="128" text-anchor="middle" class="label">inverse folding</text>
    <text x="380" y="170" text-anchor="middle" class="small">condition: backbone X</text>{chain(318,205,PURPLE,.95)}<path d="M380 330L380 362" class="arrow"/><text x="380" y="397" text-anchor="middle" class="body">compatible sequence</text>
    {box(515,90,210,360,TEAL_LIGHT)}<text x="620" y="128" text-anchor="middle" class="label">backbone design</text>
    <text x="620" y="170" text-anchor="middle" class="small">condition: geometry c</text><circle cx="620" cy="250" r="68" fill="white" stroke="{TEAL}" stroke-width="2" stroke-dasharray="7 6"/>{chain(565,207,TEAL,.85)}<text x="620" y="365" text-anchor="middle" class="body">unlabeled structure</text><text x="620" y="405" text-anchor="middle" class="small">Which sequence realizes it?</text>
    {box(755,90,210,360,AMBER_LIGHT)}<text x="860" y="128" text-anchor="middle" class="label">co-design</text>
    <text x="860" y="170" text-anchor="middle" class="small">condition: task c</text>{chain(805,207,AMBER,.85)}<rect x="795" y="345" width="130" height="42" rx="11" fill="white"/><text x="860" y="372" text-anchor="middle" class="body">X and sequence</text>
    <rect x="145" y="490" width="710" height="42" rx="13" fill="{RED_LIGHT}"/><text x="500" y="516" text-anchor="middle" class="body">The conditioning variable determines both what the model promises and how that promise must be tested.</text>'''
    write("variable_choices", body)


def motif_scaffolding():
    motif = '<circle cx="155" cy="235" r="15" fill="%s"/><circle cx="200" cy="205" r="15" fill="%s"/><circle cx="240" cy="245" r="15" fill="%s"/><path d="M155 235L200 205L240 245" fill="none" stroke="%s" stroke-width="6"/>' % (RED, RED, RED, RED)
    noise = "".join(f'<circle cx="{420+(i*47)%170}" cy="{150+(i*73)%210}" r="9" fill="{BLUE_LIGHT}" stroke="{BLUE}"/>' for i in range(11))
    body = f'''<text x="500" y="42" text-anchor="middle" class="title">Motif scaffolding freezes the functional constraint and generates its support</text>
    {box(35,90,260,370,RED_LIGHT)}<text x="165" y="128" text-anchor="middle" class="label">fixed motif</text>{motif}<text x="165" y="325" text-anchor="middle" class="body">catalytic or binding geometry</text><text x="165" y="355" text-anchor="middle" class="small">identity and relative placement persist</text>
    <path d="M305 275L350 275" class="arrow"/>{box(365,90,260,370,BLUE_LIGHT)}<text x="495" y="128" text-anchor="middle" class="label">conditional denoising</text>{noise}<circle cx="455" cy="255" r="11" fill="{RED}"/><circle cx="500" cy="225" r="11" fill="{RED}"/><circle cx="540" cy="265" r="11" fill="{RED}"/><path d="M455 255L500 225L540 265" fill="none" stroke="{RED}" stroke-width="5"/><text x="495" y="400" text-anchor="middle" class="small">move generated residues; preserve motif</text>
    <path d="M635 275L680 275" class="arrow"/>{box(695,90,270,370,TEAL_LIGHT)}<text x="830" y="128" text-anchor="middle" class="label">scaffold candidate</text>{chain(765,190,TEAL,1.05)}<circle cx="799" cy="291" r="12" fill="{RED}"/><circle cx="843" cy="260" r="12" fill="{RED}"/><circle cx="884" cy="300" r="12" fill="{RED}"/><path d="M799 291L843 260L884 300" fill="none" stroke="{RED}" stroke-width="5"/><text x="830" y="385" text-anchor="middle" class="body">global fold supports local motif</text><text x="830" y="415" text-anchor="middle" class="small">geometry is necessary, not sufficient</text>
    <text x="500" y="515" text-anchor="middle" class="body">A successful sample must satisfy the hard local constraint without collapsing global foldability or sequence diversity.</text>'''
    write("motif_scaffolding", body)


def design_loop():
    body = f'''<text x="500" y="42" text-anchor="middle" class="title">A generated backbone is a proposal, not a finished design</text>
    {box(45,105,190,145,TEAL_LIGHT)}<text x="140" y="143" text-anchor="middle" class="label">1. generate</text>{chain(90,165,TEAL,.75)}
    {box(285,105,190,145,PURPLE_SOFT)}<text x="380" y="143" text-anchor="middle" class="label">2. inverse fold</text><rect x="317" y="174" width="126" height="42" rx="11" fill="white"/><text x="380" y="201" text-anchor="middle" class="body">sequences</text>
    {box(525,105,190,145,BLUE_LIGHT)}<text x="620" y="143" text-anchor="middle" class="label">3. refold</text>{chain(568,166,BLUE,.72)}
    {box(765,105,190,145,AMBER_LIGHT)}<text x="860" y="143" text-anchor="middle" class="label">4. filter</text><text x="860" y="184" text-anchor="middle" class="body">geometry + confidence</text><text x="860" y="214" text-anchor="middle" class="small">retain a small subset</text>
    <path d="M235 178L275 178" class="arrow"/><path d="M475 178L515 178" class="arrow"/><path d="M715 178L755 178" class="arrow"/>
    <path d="M860 260L860 322" class="arrow"/>{box(680,335,280,130,RED_LIGHT)}<text x="820" y="375" text-anchor="middle" class="label">5. synthesize and assay</text><text x="820" y="411" text-anchor="middle" class="body">expression, stability, function</text><text x="820" y="439" text-anchor="middle" class="small">the biological verdict</text>
    <path d="M672 400C530 505,250 490,140 270" class="arrow"/><text x="410" y="475" text-anchor="middle" class="small">failures update models, filters, and constraints</text>
    <rect x="55" y="325" width="390" height="92" rx="18" fill="white" stroke="{SPINE}" stroke-width="2"/><text x="250" y="357" text-anchor="middle" class="body">Self-consistency: does a designed sequence</text><text x="250" y="384" text-anchor="middle" class="body">refold near the generated backbone?</text><text x="250" y="405" text-anchor="middle" class="small">useful screen; not proof of function</text>'''
    write("design_loop", body)


def evaluation_funnel():
    body = f'''<text x="500" y="42" text-anchor="middle" class="title">Every filter changes the population on which success is measured</text>
    <path d="M110 105H890L760 185H240Z" fill="{BLUE_LIGHT}" stroke="{SPINE}" stroke-width="2"/><text x="500" y="145" text-anchor="middle" class="body">10,000 generated candidates — diversity and novelty</text>
    <path d="M240 195H760L675 275H325Z" fill="{PURPLE_SOFT}" stroke="{SPINE}" stroke-width="2"/><text x="500" y="237" text-anchor="middle" class="body">2,000 geometry-valid — clashes, bonds, topology</text>
    <path d="M325 285H675L620 365H380Z" fill="{TEAL_LIGHT}" stroke="{SPINE}" stroke-width="2"/><text x="500" y="327" text-anchor="middle" class="body">300 self-consistent — sequence refolds</text>
    <path d="M380 375H620L570 455H430Z" fill="{AMBER_LIGHT}" stroke="{SPINE}" stroke-width="2"/><text x="500" y="417" text-anchor="middle" class="body">24 synthesized — ranked by oracles</text>
    <rect x="450" y="475" width="100" height="48" rx="14" fill="{RED_LIGHT}" stroke="{RED}" stroke-width="2"/><text x="500" y="506" text-anchor="middle" class="body">3 hits</text>
    <path d="M795 215C915 260,915 425,775 480" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#arrow)"/><text x="850" y="302" text-anchor="middle" class="small">selection bias</text><text x="850" y="327" text-anchor="middle" class="small">the assay sees only</text><text x="850" y="352" text-anchor="middle" class="small">the oracle's favorites</text>
    <text x="175" y="487" text-anchor="middle" class="small">Report denominator, filters,</text><text x="175" y="511" text-anchor="middle" class="small">controls, and negative outcomes.</text>'''
    write("evaluation_funnel", body)


if __name__ == "__main__":
    variable_choices(); motif_scaffolding(); design_loop(); evaluation_funnel()
