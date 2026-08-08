#!/usr/bin/env python3
"""Generate original SVG-first figures for the drug-discovery chapter.

The diagrams are editorial syntheses of standard pharmacology and development
concepts. No lecture-slide, paper, Flaticon, or other third-party artwork is
copied. Outputs are editable SVG plus PNG previews. License: CC BY 4.0 with the
accompanying blog post.
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
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker><marker id="arrp" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{P}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}.parrow{{fill:none;stroke:{P};stroke-width:3;marker-end:url(#arrp)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''


def box(x, y, w, h, fill, stroke=S, r=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"drugdisc_{name}.svg"
    png = OUT / f"drugdisc_{name}.png"
    svg.write_text(doc(body), encoding="utf-8")
    if not bfs.render_svg_preview(svg, png, width=1000):
        raise RuntimeError("SVG renderer unavailable")


def evidence_chain():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">A medicine requires four linked claims</text>
{box(30,100,205,330,PL,P)}{box(275,100,205,330,BL,B)}{box(520,100,205,330,GL,G)}{box(765,100,205,330,AL,A)}
<text x="132" y="140" text-anchor="middle" class="head">disease biology</text><circle cx="132" cy="220" r="52" fill="white" stroke="{P}" stroke-width="4"/><path d="M100 220C118 183 145 260 165 212" fill="none" stroke="{P}" stroke-width="6"/><text x="132" y="310" text-anchor="middle" class="body">target is causal</text><text x="132" y="343" text-anchor="middle" class="small">genetics and perturbation</text><text x="132" y="370" text-anchor="middle" class="small">relevant patient context</text>
<path d="M240 265L268 265" class="arrow"/>
<text x="377" y="140" text-anchor="middle" class="head">molecular action</text><g stroke="{S}" stroke-width="5"><line x1="335" y1="210" x2="378" y2="180"/><line x1="378" y1="180" x2="421" y2="215"/><line x1="421" y1="215" x2="395" y2="260"/><line x1="395" y1="260" x2="345" y2="252"/><line x1="345" y1="252" x2="335" y2="210"/></g><g fill="{B}" stroke="white" stroke-width="3"><circle cx="335" cy="210" r="14"/><circle cx="378" cy="180" r="14"/><circle cx="421" cy="215" r="14"/><circle cx="395" cy="260" r="14"/><circle cx="345" cy="252" r="14"/></g><text x="377" y="310" text-anchor="middle" class="body">potent and selective</text><text x="377" y="343" text-anchor="middle" class="small">binding, cells, mechanism</text><text x="377" y="370" text-anchor="middle" class="small">reproducible chemistry</text>
<path d="M485 265L513 265" class="arrow"/>
<text x="622" y="140" text-anchor="middle" class="head">exposure</text><path d="M560 270C585 270 587 180 625 180C660 180 665 240 690 260" fill="none" stroke="{G}" stroke-width="6"/><line x1="560" y1="278" x2="690" y2="278" stroke="{S}" stroke-width="2"/><text x="622" y="310" text-anchor="middle" class="body">right tissue and duration</text><text x="622" y="343" text-anchor="middle" class="small">absorption and clearance</text><text x="622" y="370" text-anchor="middle" class="small">free concentration</text>
<path d="M730 265L758 265" class="arrow"/>
<text x="867" y="140" text-anchor="middle" class="head">patient outcome</text><path d="M820 245L852 277L918 190" fill="none" stroke="{A}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/><text x="867" y="310" text-anchor="middle" class="body">benefit exceeds risk</text><text x="867" y="343" text-anchor="middle" class="small">clinical endpoints</text><text x="867" y="370" text-anchor="middle" class="small">defined population</text>
<rect x="135" y="475" width="730" height="50" rx="14" fill="{RL}"/><text x="500" y="505" text-anchor="middle" class="body">strong evidence at one link does not validate the links downstream</text>'''
    write("evidence_chain", body)


def optimization_frontier():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Lead optimization navigates constraints, not one score</text>
{box(40,82,580,400,PL,P)}{box(660,82,300,400,GL,G)}
<line x1="120" y1="405" x2="550" y2="405" stroke="{S}" stroke-width="3"/><line x1="120" y1="405" x2="120" y2="145" stroke="{S}" stroke-width="3"/>
<text x="335" y="450" text-anchor="middle" class="body">potency</text><text x="70" y="275" text-anchor="middle" class="body" transform="rotate(-90 70 275)">developability</text>
<path d="M160 380C235 330 300 285 360 225C420 168 475 150 545 145" fill="none" stroke="{P}" stroke-width="6"/>
<g fill="{B}" stroke="white" stroke-width="3"><circle cx="175" cy="350" r="14"/><circle cx="240" cy="315" r="14"/><circle cx="305" cy="275" r="14"/><circle cx="370" cy="232" r="14"/><circle cx="440" cy="188" r="14"/><circle cx="520" cy="156" r="14"/></g>
<g fill="{R}" stroke="white" stroke-width="3"><circle cx="270" cy="390" r="12"/><circle cx="415" cy="342" r="12"/><circle cx="525" cy="300" r="12"/></g><text x="330" y="115" text-anchor="middle" class="head">constrained Pareto frontier</text><text x="335" y="475" text-anchor="middle" class="small">red candidates fail solubility, clearance, selectivity, or synthesis constraints</text>
<text x="810" y="125" text-anchor="middle" class="head">a viable lead</text><rect x="705" y="155" width="210" height="48" rx="13" fill="white"/><text x="810" y="185" text-anchor="middle" class="body">potent enough</text><rect x="705" y="220" width="210" height="48" rx="13" fill="white"/><text x="810" y="250" text-anchor="middle" class="body">soluble and stable</text><rect x="705" y="285" width="210" height="48" rx="13" fill="white"/><text x="810" y="315" text-anchor="middle" class="body">selective and safe</text><rect x="705" y="350" width="210" height="48" rx="13" fill="white"/><text x="810" y="380" text-anchor="middle" class="body">practical to make</text><text x="810" y="440" text-anchor="middle" class="small">thresholds depend on dose and indication</text>'''
    write("optimization_frontier", body)


def pkpd_bridge():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">Dose, exposure, engagement, and response are different mappings</text>
{box(35,90,210,360,AL,A)}{box(275,90,210,360,BL,B)}{box(515,90,210,360,GL,G)}{box(755,90,210,360,RL,R)}
<text x="140" y="130" text-anchor="middle" class="head">dose</text><rect x="98" y="180" width="84" height="135" rx="36" fill="white" stroke="{A}" stroke-width="4"/><line x1="100" y1="247" x2="180" y2="247" stroke="{A}" stroke-width="4"/><text x="140" y="365" text-anchor="middle" class="body">amount and route</text>
<path d="M250 270L268 270" class="arrow"/>
<text x="380" y="130" text-anchor="middle" class="head">free exposure</text><path d="M310 320C330 320 335 180 377 180C420 180 425 275 455 300" fill="none" stroke="{B}" stroke-width="6"/><line x1="310" y1="330" x2="455" y2="330" stroke="{S}" stroke-width="2"/><text x="380" y="365" text-anchor="middle" class="body">tissue C(t)</text><text x="380" y="395" text-anchor="middle" class="small">absorption and clearance</text>
<path d="M490 270L508 270" class="arrow"/>
<text x="620" y="130" text-anchor="middle" class="head">target engagement</text><path d="M552 315C575 315 580 250 600 225C620 200 650 190 690 188" fill="none" stroke="{G}" stroke-width="6"/><line x1="552" y1="330" x2="690" y2="330" stroke="{S}" stroke-width="2"/><text x="620" y="365" text-anchor="middle" class="body">occupancy or inhibition</text><text x="620" y="395" text-anchor="middle" class="small">affinity, abundance, kinetics</text>
<path d="M730 270L748 270" class="arrow"/>
<text x="860" y="130" text-anchor="middle" class="head">biological response</text><path d="M792 315C812 315 817 270 835 235C855 198 885 185 930 185" fill="none" stroke="{R}" stroke-width="6"/><line x1="792" y1="330" x2="930" y2="330" stroke="{S}" stroke-width="2"/><text x="860" y="365" text-anchor="middle" class="body">benefit and toxicity</text><text x="860" y="395" text-anchor="middle" class="small">pathways and tissue context</text>
<rect x="135" y="485" width="730" height="48" rx="14" fill="{PL}"/><text x="500" y="514" text-anchor="middle" class="body">an in-vitro potency value specifies only one piece of the chain</text>'''
    write("pkpd_bridge", body)


def funnel_feedback():
    body = f'''
<text x="500" y="40" text-anchor="middle" class="title">The funnel narrows, while evidence flows backward</text>
<path d="M80 110H920L810 190H190Z" fill="{PL}" stroke="{P}" stroke-width="2"/><text x="500" y="158" text-anchor="middle" class="head">target hypotheses</text>
<path d="M190 205H810L720 275H280Z" fill="{BL}" stroke="{B}" stroke-width="2"/><text x="500" y="248" text-anchor="middle" class="head">confirmed hits and chemical series</text>
<path d="M280 290H720L650 360H350Z" fill="{GL}" stroke="{G}" stroke-width="2"/><text x="500" y="333" text-anchor="middle" class="head">optimized leads and candidates</text>
<path d="M350 375H650L585 445H415Z" fill="{AL}" stroke="{A}" stroke-width="2"/><text x="500" y="418" text-anchor="middle" class="head">clinical interventions</text>
<path d="M760 410C910 350 930 190 842 120" class="parrow"/><text x="870" y="300" text-anchor="middle" class="small">outcomes revise</text><text x="870" y="322" text-anchor="middle" class="small">mechanism and selection</text>
<path d="M318 410C155 405 80 330 128 250" class="parrow"/><text x="120" y="375" text-anchor="middle" class="small">liabilities revise</text><text x="120" y="397" text-anchor="middle" class="small">chemistry and assays</text>
<rect x="125" y="490" width="750" height="45" rx="14" fill="{RL}"/><text x="500" y="518" text-anchor="middle" class="body">a failed program is informative only when exposure, engagement, and biology can be separated</text>'''
    write("funnel_feedback", body)


if __name__ == "__main__":
    evidence_chain()
    optimization_frontier()
    pkpd_bridge()
    funnel_feedback()
