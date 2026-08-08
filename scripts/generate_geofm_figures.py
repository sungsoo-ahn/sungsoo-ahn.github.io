#!/usr/bin/env python3
"""Generate original SVG-first figures for geometric flow matching.

The diagrams synthesize standard Riemannian geometry and flow-matching ideas
from the primary references cited in the accompanying post. No lecture-slide,
paper, or Flaticon artwork is copied. Outputs are editable SVG plus PNG previews
in a flat-icon visual language. License: CC BY 4.0 with the blog post.
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


def doc(body, height=560):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><defs>
<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker>
<style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x,y,w,h,fill,r=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg=OUT/f"geofm_{name}.svg"; png=OUT/f"geofm_{name}.png"
    svg.write_text(doc(body), encoding="utf-8")
    if not bfs.render_svg_preview(svg,png,width=1000): raise RuntimeError("SVG renderer unavailable")


def paths():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">A manifold path must move through tangent directions</text>
{box(45,80,420,395,BL)}{box(535,80,420,395,GL)}
<text x="255" y="118" text-anchor="middle" class="head">Euclidean interpolation</text><text x="745" y="118" text-anchor="middle" class="head">geodesic interpolation</text>
<circle cx="255" cy="278" r="122" fill="{BL}" stroke="{B}" stroke-width="3"/><circle cx="165" cy="205" r="13" fill="{P}"/><circle cx="350" cy="322" r="13" fill="{G}"/>
<line x1="165" y1="205" x2="350" y2="322" stroke="{R}" stroke-width="5" stroke-dasharray="10 7"/><circle cx="255" cy="262" r="10" fill="{R}"/>
<text x="255" y="424" text-anchor="middle" class="body">(1−t)x₀ + tx₁</text><text x="255" y="449" text-anchor="middle" class="small">the chord leaves the sphere</text>
<circle cx="745" cy="278" r="122" fill="{GL}" stroke="{G}" stroke-width="3"/><circle cx="655" cy="205" r="13" fill="{P}"/><circle cx="840" cy="322" r="13" fill="{G}"/>
<path d="M655 205C716 140 815 202 840 322" fill="none" stroke="{P}" stroke-width="6"/><path d="M740 167L790 184" stroke="{A}" stroke-width="5" marker-end="url(#arr)"/>
<text x="745" y="424" text-anchor="middle" class="body">expₓ₀(t logₓ₀(x₁))</text><text x="745" y="449" text-anchor="middle" class="small">velocity stays tangent to the sphere</text>
<rect x="165" y="498" width="670" height="38" rx="13" fill="{AL}"/><text x="500" y="523" text-anchor="middle" class="body">Euclidean addition is replaced by manifold motion generated in Tₓ𝓜</text>'''
    write("euclidean_geodesic",body)


def explog():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Log chooses an initial tangent velocity; exp follows its geodesic</text>
{box(45,80,910,395,PL)}
<ellipse cx="560" cy="295" rx="310" ry="125" fill="{PL}" stroke="{P}" stroke-width="3"/><path d="M250 295C370 155 600 135 825 260" fill="none" stroke="{P}" stroke-width="6"/>
<circle cx="250" cy="295" r="14" fill="{B}"/><circle cx="825" cy="260" r="14" fill="{G}"/><text x="225" y="328" class="body">x₀</text><text x="842" y="266" class="body">x₁</text>
<line x1="150" y1="355" x2="455" y2="225" stroke="{S}" stroke-width="3"/><text x="165" y="392" class="small">tangent plane Tₓ₀𝓜</text>
<path d="M250 295L405 230" stroke="{A}" stroke-width="7" marker-end="url(#arr)"/><text x="390" y="207" class="body">logₓ₀(x₁)</text>
<path d="M470 180L585 162" class="arrow"/><text x="527" y="142" text-anchor="middle" class="small">expₓ₀</text>
<text x="500" y="432" text-anchor="middle" class="body">x(t) = exp_x0 [ t log_x0(x1) ]</text>
<rect x="175" y="498" width="650" height="38" rx="13" fill="{RL}"/><text x="500" y="523" text-anchor="middle" class="body">at a cut locus, several shortest geodesics can make log multivalued</text>'''
    write("exp_log",body)


def continuity():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">A tangent vector field transports probability without leaving the manifold</text>
{box(45,80,910,395,GL)}
<path d="M105 340C260 145 700 155 895 330" fill="none" stroke="{G}" stroke-width="5"/><path d="M105 340C290 430 690 440 895 330" fill="none" stroke="{G}" stroke-width="3" opacity=".45"/>
<g fill="{P}" opacity=".75"><circle cx="205" cy="292" r="34"/><circle cx="250" cy="250" r="26"/><circle cx="288" cy="292" r="21"/></g><g fill="{B}" opacity=".75"><circle cx="690" cy="250" r="28"/><circle cx="735" cy="285" r="36"/><circle cx="785" cy="264" r="22"/></g>
<path d="M330 265C410 215 505 215 580 250" class="arrow"/><path d="M405 348C490 313 570 320 635 348" class="arrow"/><path d="M585 202L635 215" class="arrow"/>
<text x="245" y="372" text-anchor="middle" class="body">density p(t)</text><text x="735" y="372" text-anchor="middle" class="body">transported density p(t + dt)</text>
<rect x="255" y="405" width="490" height="44" rx="14" fill="white" stroke="{G}" stroke-width="2"/><text x="500" y="433" text-anchor="middle" class="body">d p / d t + div_M(p u) = 0</text>
<rect x="140" y="498" width="720" height="38" rx="13" fill="{AL}"/><text x="500" y="523" text-anchor="middle" class="body">div_M measures local expansion using the Riemannian volume form</text>'''
    write("continuity",body)


def product():
    body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Molecular state spaces are products of different geometries</text>
{box(35,82,280,365,BL)}{box(360,82,280,365,PL)}{box(685,82,280,365,GL)}
<text x="175" y="122" text-anchor="middle" class="head">position ℝ³</text><circle cx="175" cy="255" r="14" fill="{B}"/><path d="M175 255L245 215" stroke="{B}" stroke-width="6" marker-end="url(#arr)"/><path d="M175 255L125 205" stroke="{G}" stroke-width="6" marker-end="url(#arr)"/><path d="M175 255L175 340" stroke="{P}" stroke-width="6" marker-end="url(#arr)"/><text x="175" y="402" text-anchor="middle" class="small">translation of a local frame</text>
<text x="500" y="122" text-anchor="middle" class="head">orientation SO(3)</text><circle cx="500" cy="255" r="92" fill="none" stroke="{P}" stroke-width="4"/><path d="M425 250C450 150 575 150 585 245" fill="none" stroke="{P}" stroke-width="6" marker-end="url(#arr)"/><text x="500" y="272" text-anchor="middle" font-size="28" font-weight="700">R</text><text x="500" y="402" text-anchor="middle" class="small">rotation of a residue frame</text>
<text x="825" y="122" text-anchor="middle" class="head">torsion S¹</text><circle cx="825" cy="255" r="90" fill="none" stroke="{G}" stroke-width="26"/><circle cx="825" cy="255" r="38" fill="white"/><circle cx="888" cy="193" r="12" fill="{A}"/><text x="825" y="264" text-anchor="middle" class="body">χ mod 2π</text><text x="825" y="402" text-anchor="middle" class="small">periodic side-chain angle</text>
<path d="M315 265L352 265" class="arrow"/><text x="334" y="240" text-anchor="middle" class="body">×</text><path d="M640 265L677 265" class="arrow"/><text x="658" y="240" text-anchor="middle" class="body">×</text>
<rect x="115" y="488" width="770" height="48" rx="14" fill="{RL}"/><text x="500" y="509" text-anchor="middle" class="body">product metric: ‖u‖² = wₓ‖uₓ‖² + wᴿ‖uᴿ‖² + wχ‖uχ‖²</text><text x="500" y="528" text-anchor="middle" class="small">the weights set the relative geometry of translation, rotation, and torsion</text>'''
    write("product_manifold",body)


if __name__ == "__main__":
    paths(); explog(); continuity(); product()
