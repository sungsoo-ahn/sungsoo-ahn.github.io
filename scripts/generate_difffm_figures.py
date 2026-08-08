#!/usr/bin/env python3
"""Generate original figures for the diffusion/flow-matching blog post.

All outputs are native SVG with PNG previews. The figures use a flat-icon
visual language: rounded geometric forms, restrained solid colors, consistent
strokes, and no gradients or texture. No Flaticon asset, lecture-slide figure,
or paper figure is copied. The diagrams are original syntheses of standard
diffusion and flow-matching identities discussed in the cited primary papers.
License: CC BY 4.0 with the blog post.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import blog_figure_style as bfs

OUT = ROOT / "assets" / "img" / "blog"
TEXT = bfs.TEXT
MUTED = bfs.MUTED
SPINE = bfs.SPINE
PURPLE = bfs.PURPLE
PURPLE_SOFT = bfs.PURPLE_SOFT
BLUE = bfs.BLUE
BLUE_LIGHT = bfs.BLUE_LIGHT
TEAL = bfs.TEAL
TEAL_LIGHT = bfs.TEAL_LIGHT
AMBER = bfs.AMBER
AMBER_LIGHT = bfs.AMBER_LIGHT
RED = bfs.RED
RED_LIGHT = bfs.RED_LIGHT


def svg_document(body: str, *, width: int = 1000, height: int = 560) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {TEXT}; }}
      .title {{ font-size: 22px; font-weight: 700; }}
      .label {{ font-size: 18px; font-weight: 700; }}
      .body {{ font-size: 16px; }}
      .small {{ font-size: 14px; fill: {MUTED}; }}
      .arrow {{ stroke: {MUTED}; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }}
      .dash {{ stroke: {MUTED}; stroke-width: 2.3; fill: none; stroke-dasharray: 8 7; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  {body}
</svg>'''


def box(x: float, y: float, w: float, h: float, fill: str, *, radius: int = 20, stroke: str = SPINE) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def dot(x: float, y: float, color: str, radius: int = 9) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="2"/>'


def write(name: str, body: str, *, width: int = 1000, height: int = 560) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"difffm_{name}.svg"
    png_path = OUT / f"difffm_{name}.png"
    svg_path.write_text(svg_document(body, width=width, height=height), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=width):
        raise RuntimeError("No SVG renderer found; install rsvg-convert or ImageMagick")


def make_two_views() -> None:
    left_dots = "".join(dot(x, y, PURPLE) for x, y in [(105, 209), (137, 174), (145, 236), (176, 201)])
    noise_dots = "".join(dot(x, y, BLUE) for x, y in [(338, 164), (302, 229), (375, 206), (332, 264)])
    right_dots = "".join(dot(x, y, TEAL) for x, y in [(625, 209), (657, 174), (665, 236), (696, 201)])
    base_dots = "".join(dot(x, y, BLUE) for x, y in [(858, 164), (822, 229), (895, 206), (852, 264)])
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Diffusion and flow matching learn local motion along a probability path</text>
      {box(35,82,440,352,PURPLE_SOFT)}
      <text x="255" y="122" text-anchor="middle" class="label">diffusion: corrupt, then reverse</text>
      <ellipse cx="140" cy="205" rx="74" ry="68" fill="white" stroke="{PURPLE}" stroke-width="3"/>
      {left_dots}
      <ellipse cx="340" cy="210" rx="84" ry="78" fill="white" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 6"/>
      {noise_dots}
      <path d="M 220 190 C 255 160, 270 160, 290 183" class="arrow"/>
      <path d="M 290 249 C 260 273, 240 270, 218 240" class="arrow"/>
      <text x="255" y="315" text-anchor="middle" class="body">learn the score grad log p_t(x)</text>
      <text x="255" y="344" text-anchor="middle" class="small">the reverse drift points toward likely states</text>
      <rect x="92" y="365" width="326" height="43" rx="12" fill="{RED_LIGHT}"/>
      <text x="255" y="392" text-anchor="middle" class="body">sampling uses a reverse SDE or probability-flow ODE</text>

      {box(525,82,440,352,TEAL_LIGHT)}
      <text x="745" y="122" text-anchor="middle" class="label">flow matching: prescribe, then regress</text>
      <ellipse cx="660" cy="205" rx="74" ry="68" fill="white" stroke="{TEAL}" stroke-width="3"/>
      {right_dots}
      <ellipse cx="860" cy="210" rx="84" ry="78" fill="white" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 6"/>
      {base_dots}
      <path d="M 815 236 C 770 275, 735 257, 708 230" stroke="{TEAL}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
      <text x="745" y="315" text-anchor="middle" class="body">learn the velocity u_t(x)</text>
      <text x="745" y="344" text-anchor="middle" class="small">the ODE transports probability along the chosen path</text>
      <rect x="582" y="365" width="326" height="43" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="745" y="392" text-anchor="middle" class="body">sampling integrates one deterministic ODE</text>

      <rect x="190" y="466" width="620" height="55" rx="16" fill="{BLUE_LIGHT}"/>
      <text x="500" y="489" text-anchor="middle" class="body">shared statistical move</text>
      <text x="500" y="512" text-anchor="middle" class="small">replace an intractable marginal target by a tractable conditional regression target</text>
    '''
    write("two_views", body)


def make_denoising_identity() -> None:
    cloud = "".join(dot(x, y, PURPLE, 8) for x, y in [(105, 192), (132, 172), (151, 207), (123, 233), (173, 240)])
    noisy = "".join(dot(x, y, BLUE, 8) for x, y in [(390, 153), (342, 208), (426, 219), (366, 268), (435, 280)])
    arrows = "".join(
        f'<path d="M {x} {y} L 560 270" stroke="{TEAL}" stroke-width="2.2" fill="none" marker-end="url(#arrow)"/>'
        for x, y in [(390, 153), (342, 208), (426, 219), (366, 268), (435, 280)]
    )
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">A conditional Gaussian target reveals the unknown marginal score</text>
      {box(35,88,220,280,PURPLE_SOFT)}
      <text x="145" y="126" text-anchor="middle" class="label">clean sample x_0</text>
      <ellipse cx="140" cy="210" rx="74" ry="62" fill="white" stroke="{PURPLE}" stroke-width="3"/>
      {cloud}
      <text x="145" y="302" text-anchor="middle" class="body">x_0 ~ p_data</text>
      <text x="145" y="332" text-anchor="middle" class="small">sampled from the dataset</text>

      <path d="M 260 208 L 302 208" class="arrow"/>
      <text x="282" y="190" text-anchor="middle" class="small">corrupt</text>
      {box(310,88,220,280,BLUE_LIGHT)}
      <text x="420" y="126" text-anchor="middle" class="label">noisy state x_t</text>
      <ellipse cx="410" cy="226" rx="82" ry="76" fill="white" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 6"/>
      {noisy}
      <text x="420" y="310" text-anchor="middle" class="body">x_t = alpha_t x_0</text>
      <text x="420" y="333" text-anchor="middle" class="body">+ sigma_t epsilon</text>
      <text x="420" y="355" text-anchor="middle" class="small">epsilon ~ N(0, I)</text>

      {arrows}
      <circle cx="600" cy="270" r="64" fill="{TEAL_LIGHT}" stroke="{TEAL}" stroke-width="3"/>
      <text x="600" y="258" text-anchor="middle" class="label">conditional</text>
      <text x="600" y="282" text-anchor="middle" class="body">score target</text>
      <text x="600" y="307" text-anchor="middle" class="small">-epsilon / sigma_t</text>

      <path d="M 669 270 L 727 270" class="arrow"/>
      {box(745,158,220,220,AMBER_LIGHT)}
      <text x="855" y="200" text-anchor="middle" class="label">conditional mean</text>
      <text x="855" y="237" text-anchor="middle" class="body">E[-epsilon/sigma_t | x_t=x]</text>
      <line x1="785" y1="262" x2="925" y2="262" stroke="{SPINE}" stroke-width="2"/>
      <text x="855" y="300" text-anchor="middle" class="label">marginal score</text>
      <text x="855" y="335" text-anchor="middle" class="body">grad_x log p_t(x)</text>

      <rect x="110" y="430" width="780" height="70" rx="16" fill="{PURPLE_SOFT}"/>
      <text x="500" y="458" text-anchor="middle" class="body">The individual arrow depends on the hidden clean example.</text>
      <text x="500" y="485" text-anchor="middle" class="small">Squared-error regression averages those arrows at each noisy location and recovers the field needed for reversal.</text>
    '''
    write("denoising_identity", body)


def make_conditional_marginalization() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">Conditional flows combine into one marginal probability current</text>
      {box(35,90,260,385,BLUE_LIGHT)}
      <text x="165" y="130" text-anchor="middle" class="label">tractable conditional paths</text>
      <circle cx="95" cy="210" r="18" fill="{PURPLE}"/>
      <circle cx="95" cy="280" r="18" fill="{TEAL}"/>
      <circle cx="95" cy="350" r="18" fill="{AMBER}"/>
      <path d="M 120 210 C 178 185, 214 205, 250 235" stroke="{PURPLE}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
      <path d="M 120 280 C 170 280, 205 275, 250 275" stroke="{TEAL}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
      <path d="M 120 350 C 178 375, 214 350, 250 315" stroke="{AMBER}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>
      <text x="165" y="410" text-anchor="middle" class="body">u_t(x | z), p_t(x | z)</text>
      <text x="165" y="438" text-anchor="middle" class="small">condition z is sampled directly</text>

      <path d="M 312 280 L 380 280" class="arrow"/>
      <circle cx="448" cy="280" r="62" fill="{PURPLE_SOFT}" stroke="{PURPLE}" stroke-width="3"/>
      <text x="448" y="266" text-anchor="middle" class="label">average</text>
      <text x="448" y="291" text-anchor="middle" class="body">at fixed x,t</text>
      <text x="448" y="315" text-anchor="middle" class="small">posterior weights</text>

      <path d="M 514 280 L 575 280" class="arrow"/>
      {box(595,90,370,385,TEAL_LIGHT)}
      <text x="780" y="130" text-anchor="middle" class="label">marginal path and field</text>
      <ellipse cx="780" cy="225" rx="108" ry="67" fill="white" stroke="{TEAL}" stroke-width="3"/>
      <circle cx="730" cy="232" r="12" fill="{PURPLE}"/>
      <circle cx="780" cy="194" r="12" fill="{TEAL}"/>
      <circle cx="829" cy="236" r="12" fill="{AMBER}"/>
      <path d="M 730 232 L 757 211" stroke="{PURPLE}" stroke-width="4" marker-end="url(#arrow)"/>
      <path d="M 780 194 L 809 184" stroke="{TEAL}" stroke-width="4" marker-end="url(#arrow)"/>
      <path d="M 829 236 L 850 210" stroke="{AMBER}" stroke-width="4" marker-end="url(#arrow)"/>
      <text x="780" y="335" text-anchor="middle" class="body">p_t(x) = E_z[p_t(x | z)]</text>
      <text x="780" y="376" text-anchor="middle" class="body">u_t(x) = E[u_t(x | Z) | X_t=x]</text>
      <text x="780" y="420" text-anchor="middle" class="small">the average current satisfies the same continuity equation</text>

      <rect x="215" y="502" width="570" height="40" rx="12" fill="{AMBER_LIGHT}"/>
      <text x="500" y="528" text-anchor="middle" class="body">Regress the conditional arrows; obtain the marginal field without evaluating the posterior.</text>
    '''
    write("conditional_marginalization", body)


def make_schedule_solver_tradeoff() -> None:
    body = f'''
      <text x="500" y="42" text-anchor="middle" class="title">The path sets the geometry; the solver decides how faithfully to follow it</text>
      {box(35,88,450,370,PURPLE_SOFT)}
      <text x="260" y="126" text-anchor="middle" class="label">schedule changes path difficulty</text>
      <line x1="92" y1="365" x2="430" y2="365" stroke="{SPINE}" stroke-width="2"/>
      <line x1="92" y1="365" x2="92" y2="165" stroke="{SPINE}" stroke-width="2"/>
      <path d="M 95 350 C 165 345, 230 320, 275 265 C 330 197, 380 180, 425 175" stroke="{PURPLE}" stroke-width="5" fill="none"/>
      <path d="M 95 350 C 155 310, 210 280, 260 252 C 320 218, 375 190, 425 175" stroke="{TEAL}" stroke-width="5" fill="none" stroke-dasharray="8 7"/>
      <text x="111" y="390" class="small">noise</text>
      <text x="396" y="390" class="small">data</text>
      <text x="112" y="185" class="small">signal</text>
      <line x1="110" y1="424" x2="156" y2="424" stroke="{PURPLE}" stroke-width="5"/>
      <text x="166" y="430" class="small">uneven change</text>
      <line x1="285" y1="424" x2="331" y2="424" stroke="{TEAL}" stroke-width="5" stroke-dasharray="8 7"/>
      <text x="341" y="430" class="small">straighter path</text>

      {box(515,88,450,370,BLUE_LIGHT)}
      <text x="740" y="126" text-anchor="middle" class="label">step size trades cost for error</text>
      <path d="M 565 350 C 635 320, 635 205, 710 205 C 785 205, 800 158, 916 175" stroke="{MUTED}" stroke-width="3" fill="none"/>
      <polyline points="565,350 650,285 735,210 820,183 916,175" fill="none" stroke="{RED}" stroke-width="5" stroke-linejoin="round"/>
      <circle cx="565" cy="350" r="8" fill="{RED}"/><circle cx="650" cy="285" r="8" fill="{RED}"/><circle cx="735" cy="210" r="8" fill="{RED}"/><circle cx="820" cy="183" r="8" fill="{RED}"/><circle cx="916" cy="175" r="8" fill="{RED}"/>
      <text x="740" y="385" text-anchor="middle" class="body">few evaluations: fast, larger discretization error</text>
      <text x="740" y="417" text-anchor="middle" class="small">more/adaptive steps improve fidelity but call the network more often</text>

      <rect x="145" y="492" width="710" height="48" rx="14" fill="{TEAL_LIGHT}"/>
      <text x="500" y="522" text-anchor="middle" class="body">Training target, time weighting, parameterization, and numerical solver are coupled design choices.</text>
    '''
    write("schedule_solver_tradeoff", body)


if __name__ == "__main__":
    make_two_views()
    make_denoising_identity()
    make_conditional_marginalization()
    make_schedule_solver_tradeoff()
