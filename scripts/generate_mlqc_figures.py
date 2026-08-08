#!/usr/bin/env python3
"""Generate original SVG-first figures for ML and quantum chemistry.

The diagrams are original syntheses of standard electronic-structure and ML
workflows described in the cited primary literature. No lecture-slide, paper,
or Flaticon artwork is copied. Outputs are editable SVG plus PNG previews in a
flat-icon visual language. License: CC BY 4.0 with the blog post.
"""

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import blog_figure_style as bfs
OUT=ROOT/'assets'/'img'/'blog'
T,M,S=bfs.TEXT,bfs.MUTED,bfs.SPINE
P,PL=bfs.PURPLE,bfs.PURPLE_LIGHT; B,BL=bfs.BLUE,bfs.BLUE_LIGHT
G,GL=bfs.TEAL,bfs.TEAL_LIGHT; A,AL=bfs.AMBER,bfs.AMBER_LIGHT; R,RL=bfs.RED,bfs.RED_LIGHT

def doc(body):
 return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560" role="img"><defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{M}"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:{T}}}.title{{font-size:22px;font-weight:700}}.head{{font-size:18px;font-weight:700}}.body{{font-size:16px}}.small{{font-size:14px;fill:{M}}}.arrow{{fill:none;stroke:{M};stroke-width:2.5;marker-end:url(#arr)}}</style></defs><rect width="100%" height="100%" fill="white"/>{body}</svg>'''
def box(x,y,w,h,fill,r=20): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{S}" stroke-width="2"/>'
def atom(x,y,c,label='',rad=15): return f'<circle cx="{x}" cy="{y}" r="{rad}" fill="{c}" stroke="white" stroke-width="3"/><text x="{x}" y="{y+5}" text-anchor="middle" font-size="13" font-weight="700" fill="white">{label}</text>'
def write(name,body):
 OUT.mkdir(parents=True,exist_ok=True); svg=OUT/f'mlqc_{name}.svg'; png=OUT/f'mlqc_{name}.png'; svg.write_text(doc(body),encoding='utf-8')
 if not bfs.render_svg_preview(svg,png,width=1000): raise RuntimeError('SVG renderer unavailable')

def target_boundaries():
 body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Machine learning can enter at different levels of electronic structure</text>
{box(35,82,210,355,PL)}{box(275,82,210,355,BL)}{box(515,82,210,355,GL)}{box(755,82,210,355,AL)}
<text x="140" y="122" text-anchor="middle" class="head">wavefunction</text><path d="M85 235C115 155 145 315 195 205" fill="none" stroke="{P}" stroke-width="7"/><path d="M85 260C120 340 160 170 205 280" fill="none" stroke="{R}" stroke-width="5"/><text x="140" y="345" text-anchor="middle" class="body">psi(r1,...,rN)</text><text x="140" y="385" text-anchor="middle" class="small">solve a variational problem</text>
<text x="380" y="122" text-anchor="middle" class="head">density / orbitals</text><circle cx="380" cy="235" r="88" fill="{BL}" stroke="{B}" stroke-width="3"/><circle cx="350" cy="220" r="40" fill="{B}" opacity=".3"/><circle cx="410" cy="250" r="48" fill="{G}" opacity=".3"/><text x="380" y="345" text-anchor="middle" class="body">rho(r), H, orbitals</text><text x="380" y="385" text-anchor="middle" class="small">retain electronic fields</text>
<text x="620" y="122" text-anchor="middle" class="head">energy surface</text><path d="M550 305C580 175 625 330 690 185" fill="none" stroke="{G}" stroke-width="7"/><circle cx="620" cy="255" r="11" fill="{P}"/><text x="620" y="345" text-anchor="middle" class="body">E(R), forces</text><text x="620" y="385" text-anchor="middle" class="small">surrogate one level of theory</text>
<text x="860" y="122" text-anchor="middle" class="head">property</text><rect x="810" y="185" width="100" height="100" rx="18" fill="white" stroke="{A}" stroke-width="3"/><text x="860" y="244" text-anchor="middle" font-size="30" font-weight="700">y</text><text x="860" y="345" text-anchor="middle" class="body">gap, dipole, rate</text><text x="860" y="385" text-anchor="middle" class="small">fastest and narrowest target</text>
<path d="M245 255L268 255" class="arrow"/><path d="M485 255L508 255" class="arrow"/><path d="M725 255L748 255" class="arrow"/>
<rect x="130" y="485" width="740" height="48" rx="14" fill="{RL}"/><text x="500" y="506" text-anchor="middle" class="body">moving right discards reusable electronic information</text><text x="500" y="526" text-anchor="middle" class="small">moving left increases physical structure and computational burden</text>'''; write('target_boundaries',body)

def vmc_loop():
 body=f'''
<text x="500" y="40" text-anchor="middle" class="title">A neural wavefunction is optimized through variational Monte Carlo</text>{box(50,82,900,375,PL)}
{box(90,155,185,105,BL)}<text x="182" y="191" text-anchor="middle" class="head">antisymmetric</text><text x="182" y="223" text-anchor="middle" class="body">psi_theta(r)</text>
{box(405,135,190,145,GL)}<text x="500" y="172" text-anchor="middle" class="head">sample electrons</text><circle cx="455" cy="225" r="12" fill="{P}"/><circle cx="500" cy="210" r="12" fill="{B}"/><circle cx="545" cy="235" r="12" fill="{G}"/><text x="500" y="263" text-anchor="middle" class="small">r from |psi_theta|^2</text>
{box(725,155,185,105,AL)}<text x="817" y="191" text-anchor="middle" class="head">local energy</text><text x="817" y="225" text-anchor="middle" class="body">H psi / psi</text>
<path d="M278 205L397 205" class="arrow"/><path d="M600 205L717 205" class="arrow"/><path d="M817 270C817 385 182 385 182 270" class="arrow"/>
<rect x="247" y="300" width="506" height="48" rx="14" fill="white"/><text x="500" y="330" text-anchor="middle" class="body">update theta to reduce mean energy and sampling variance</text><rect x="215" y="394" width="570" height="42" rx="13" fill="white" stroke="{P}" stroke-width="2"/><text x="500" y="421" text-anchor="middle" class="body">E_theta >= E_ground when psi_theta is normalized</text>
<rect x="150" y="490" width="700" height="42" rx="14" fill="{RL}"/><text x="500" y="517" text-anchor="middle" class="body">the bound is physical; Monte Carlo and optimization error remain numerical</text>'''; write('vmc_loop',body)

def scf_loop():
 body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Learning inside self-consistency is different from predicting its answer</text>{box(40,82,600,380,GL)}{box(680,82,280,380,BL)}
<text x="340" y="120" text-anchor="middle" class="head">learned exchange-correlation inside SCF</text>
{box(85,160,170,82,'white',15)}<text x="170" y="194" text-anchor="middle" class="body">density rho_k</text><text x="170" y="220" text-anchor="middle" class="small">current iterate</text>
{box(300,145,185,112,AL,15)}<text x="392" y="180" text-anchor="middle" class="body">E_xc_theta[rho]</text><text x="392" y="210" text-anchor="middle" class="small">differentiate for v_xc</text><text x="392" y="235" text-anchor="middle" class="small">build KS Hamiltonian</text>
{box(300,315,185,92,'white',15)}<text x="392" y="350" text-anchor="middle" class="body">solve orbitals</text><text x="392" y="379" text-anchor="middle" class="small">form rho_(k+1)</text>
<path d="M255 200L292 200" class="arrow"/><path d="M392 262L392 306" class="arrow"/><path d="M292 361C220 361 170 320 170 250" class="arrow"/>
<text x="340" y="440" text-anchor="middle" class="small">accuracy and convergence depend on the full loop</text>
<text x="820" y="120" text-anchor="middle" class="head">direct surrogate</text><g stroke="{S}" stroke-width="5"><line x1="745" y1="210" x2="820" y2="165"/><line x1="820" y1="165" x2="890" y2="225"/></g>{atom(745,210,P,'C')}{atom(820,165,G,'O')}{atom(890,225,B,'H')}<path d="M820 265L820 310" class="arrow"/><rect x="748" y="320" width="145" height="72" rx="16" fill="white" stroke="{B}" stroke-width="2"/><text x="820" y="350" text-anchor="middle" class="body">rho or H</text><text x="820" y="376" text-anchor="middle" class="small">one forward pass</text><text x="820" y="440" text-anchor="middle" class="small">fast, but no SCF fixed-point guarantee</text>
<rect x="155" y="492" width="690" height="40" rx="13" fill="{RL}"/><text x="500" y="518" text-anchor="middle" class="body">matching converged outputs does not ensure stable iterative behavior</text>'''; write('scf_loop',body)

def fidelity_cost():
 body=f'''
<text x="500" y="40" text-anchor="middle" class="title">Accuracy and cost must name both the reference and the deployment regime</text>{box(45,82,910,385,BL)}
<line x1="130" y1="390" x2="885" y2="390" stroke="{S}" stroke-width="3"/><line x1="130" y1="390" x2="130" y2="135" stroke="{S}" stroke-width="3"/>
<text x="505" y="445" text-anchor="middle" class="body">compute per system</text><text x="74" y="265" text-anchor="middle" class="body" transform="rotate(-90 74 265)">reference fidelity</text>
<circle cx="250" cy="330" r="24" fill="{A}"/><text x="250" y="335" text-anchor="middle" font-size="13" fill="white">low</text><circle cx="500" cy="245" r="26" fill="{B}"/><text x="500" y="250" text-anchor="middle" font-size="13" fill="white">DFT</text><circle cx="780" cy="155" r="30" fill="{P}"/><text x="780" y="160" text-anchor="middle" font-size="13" fill="white">CC</text>
<path d="M268 320C360 275 405 255 470 248" stroke="{M}" stroke-width="3" fill="none" marker-end="url(#arr)"/><path d="M526 232C610 190 690 160 745 157" stroke="{M}" stroke-width="3" fill="none" marker-end="url(#arr)"/>
<rect x="260" y="160" width="210" height="58" rx="15" fill="{GL}"/><text x="365" y="185" text-anchor="middle" class="body">ML surrogate</text><text x="365" y="206" text-anchor="middle" class="small">cheap after data generation</text>
<rect x="575" y="300" width="220" height="58" rx="15" fill="{RL}"/><text x="685" y="325" text-anchor="middle" class="body">delta correction</text><text x="685" y="346" text-anchor="middle" class="small">baseline + learned residual</text>
<rect x="125" y="490" width="750" height="42" rx="14" fill="{AL}"/><text x="500" y="516" text-anchor="middle" class="body">inference speed excludes labels, training, SCF fallback, and failed extrapolations</text>'''; write('fidelity_cost',body)

if __name__=='__main__': target_boundaries(); vmc_loop(); scf_loop(); fidelity_cost()
