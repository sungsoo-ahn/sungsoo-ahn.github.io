"""
Generate figures for the electrocatalysis blog post.

Seven figures covering: energy storage cycle, PEM fuel cell, Gibbs free energy
diagrams, volcano plot, scaling relations, catalyst binding sites, and ML pipeline.

Color convention:
  - Slate blue for structural/neutral elements
  - Warm red/orange for energy barriers and strong binding
  - Green for optimal/favorable regions
  - Teal for cool/weak binding
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from html import escape
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.patches import Arc
import matplotlib.patheffects as pe

import blog_figure_style as bfs

bfs.use_blog_style()


# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
TEXT_COLOR = bfs.TEXT
ARROW_COLOR = bfs.MUTED

# Primary boxes (theme purple)
BOX_MAIN = bfs.PURPLE_LIGHT
EDGE_MAIN = bfs.PURPLE

# Energy / warm
BOX_WARM = bfs.AMBER_LIGHT
EDGE_WARM = bfs.AMBER
COLOR_WARM = bfs.AMBER

# Green / optimal
BOX_GREEN = bfs.GREEN_LIGHT
EDGE_GREEN = bfs.GREEN
COLOR_GREEN = bfs.GREEN

# Red / strong binding
COLOR_RED = bfs.RED
COLOR_RED_LIGHT = bfs.RED_LIGHT

# Teal / weak binding
COLOR_TEAL = bfs.TEAL
COLOR_TEAL_LIGHT = bfs.TEAL_LIGHT

# Special
BOX_CATALYST = bfs.PURPLE_SOFT
EDGE_CATALYST = bfs.PURPLE_STRONG
BOX_ML = bfs.BLUE_LIGHT
EDGE_ML = bfs.BLUE

DENSITY_SLATE = bfs.PURPLE
ANNOTATION_BG = 'white'
ANNOTATION_BG_ALPHA = 0.85

LABEL_FS = 12
SUBLABEL_FS = 10


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────
def _style_axis(ax, xlim, ylim, xlabel=None, ylabel=None):
    """Clean spines, subtle ticks."""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(bfs.SPINE)
    ax.spines['bottom'].set_color(bfs.SPINE)
    ax.tick_params(colors=bfs.MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=TEXT_COLOR, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR, labelpad=6)


def _draw_box(ax, cx, cy, bw, bh, label, fc, ec, rounding=0.12,
              fontsize=10, fontweight='bold', text_color=TEXT_COLOR, lw=1.8):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle=f'round,pad={rounding}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            zorder=4)
    return box


def _draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.6, ms=14, zorder=2):
    """Draw a simple arrow."""
    if color is None:
        color = ARROW_COLOR
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', color=color,
        linewidth=lw, mutation_scale=ms, zorder=zorder)
    ax.add_patch(a)


def _svg_text(x, y, lines, *, size=16, fill=TEXT_COLOR, weight="500", anchor="middle", line_gap=1.18):
    """Return an SVG text block with explicit line breaks."""
    if isinstance(lines, str):
        lines = lines.split("\n")
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_gap
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def _svg_box(x, y, w, h, lines, *, fill, stroke, size=16, weight="700", rx=10):
    mid_y = y + h / 2 - (len(lines) - 1) * size * 0.56
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2.2"/>',
            _svg_text(x + w / 2, mid_y, lines, size=size, weight=weight),
        ]
    )


def _svg_arrow(x1, y1, x2, y2, *, color=ARROW_COLOR, width=2.4):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'marker-end="url(#arrow)"/>'
    )


def _svg_header(width, height):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW_COLOR}"/>
  </marker>
  <marker id="greenArrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{COLOR_GREEN}"/>
  </marker>
  <style>
    text {{ font-family: Arial, Helvetica, DejaVu Sans, sans-serif; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="white"/>
'''


def _write_svg_or_preview(output_path, svg):
    """Write native SVG, and render PNG previews for .png outputs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        svg_path = out.with_suffix(".svg")
        svg_path.write_text(svg, encoding="utf-8")
        if not bfs.render_svg_preview(svg_path, out, width=1600):
            raise RuntimeError("Could not render SVG preview; install rsvg-convert or ImageMagick.")
        print(f"Saved native SVG preview to {out}")
        return
    out.write_text(svg, encoding="utf-8")
    print(f"Saved native SVG figure to {out}")


# ──────────────────────────────────────────────
# Figure 1: Energy Storage Cycle
# ──────────────────────────────────────────────
def generate_energy_cycle_figure(output_path):
    """
    Flowchart: Renewable → Electrolyzer → H2/CH4 → Fuel Cell → Grid
    With a return arrow showing the cycle.
    """
    width, height = 900, 330
    boxes = [
        (54, 128, 138, 72, ["Renewable", "electricity"], BOX_GREEN, EDGE_GREEN),
        (258, 128, 138, 72, ["Electrolyzer"], BOX_WARM, EDGE_WARM),
        (462, 128, 138, 72, ["H2 / CH4", "storage"], BOX_MAIN, EDGE_MAIN),
        (666, 128, 138, 72, ["Fuel cell"], BOX_WARM, EDGE_WARM),
    ]
    parts = [
        _svg_header(width, height),
        _svg_text(width / 2, 48, "Hydrogen energy storage cycle", size=23, weight="700"),
    ]
    for x, y, w, h, lines, fill, stroke in boxes:
        parts.append(_svg_box(x, y, w, h, lines, fill=fill, stroke=stroke, size=17))

    arrow_y = 164
    labels = [
        (225, 104, "H2O + power", EDGE_WARM),
        (429, 104, "chemical energy", EDGE_MAIN),
        (633, 104, "electricity + heat", EDGE_WARM),
    ]
    for left, right in zip(boxes[:-1], boxes[1:]):
        parts.append(_svg_arrow(left[0] + left[2] + 18, arrow_y, right[0] - 18, arrow_y))
    for x, y, label, color in labels:
        parts.append(
            f'<rect x="{x - 66}" y="{y - 18}" width="132" height="28" rx="7" fill="white" opacity="0.92"/>'
        )
        parts.append(_svg_text(x, y, label, size=14, fill=color, weight="700"))

    parts.append(
        f'<line x1="822" y1="{arrow_y}" x2="836" y2="{arrow_y}" stroke="{COLOR_GREEN}" '
        'stroke-width="2.6" stroke-linecap="round" marker-end="url(#greenArrow)"/>'
    )
    parts.append(_svg_text(848, arrow_y + 5, "Grid", size=17, fill=COLOR_GREEN, weight="700", anchor="start"))
    parts.append(
        f'<path d="M 735 224 L 735 252 L 124 252 L 124 215" fill="none" '
        f'stroke="{COLOR_GREEN}" stroke-width="2.4" stroke-linecap="round" '
        'stroke-linejoin="round" marker-end="url(#greenArrow)"/>'
    )
    parts.append(
        _svg_text(430, 283, "water byproduct returns to electrolysis", size=15, fill=COLOR_GREEN, weight="600")
    )
    parts.append("</svg>\n")
    _write_svg_or_preview(output_path, "\n".join(parts))


# ──────────────────────────────────────────────
# Figure 2: PEM Fuel Cell Schematic
# ──────────────────────────────────────────────
def generate_fuel_cell_figure(output_path):
    """
    Simplified PEM fuel cell cross-section:
    Anode | Membrane | Cathode with reactions labeled.
    """
    width, height = 900, 550
    anode = (82, 124, 220, 310)
    pem = (340, 112, 220, 334)
    cathode = (598, 124, 220, 310)
    parts = [
        _svg_header(width, height),
        f'<rect x="{anode[0]}" y="{anode[1]}" width="{anode[2]}" height="{anode[3]}" fill="{COLOR_RED_LIGHT}" stroke="{COLOR_RED}" stroke-width="2.6"/>',
        f'<rect x="{pem[0]}" y="{pem[1]}" width="{pem[2]}" height="{pem[3]}" fill="{BOX_WARM}" stroke="{EDGE_WARM}" stroke-width="2.6"/>',
        f'<rect x="{cathode[0]}" y="{cathode[1]}" width="{cathode[2]}" height="{cathode[3]}" fill="{bfs.BLUE_LIGHT}" stroke="{bfs.BLUE}" stroke-width="2.6"/>',
        _svg_text(192, 96, "Anode", size=17, fill=COLOR_RED, weight="700"),
        _svg_text(450, 86, ["Membrane", "(PEM)"], size=17, fill=EDGE_WARM, weight="700"),
        _svg_text(708, 96, "Cathode", size=17, fill=bfs.BLUE, weight="700"),
    ]

    # External electron circuit.
    parts.append(
        f'<path d="M 192 72 C 290 28, 610 28, 708 72" fill="none" stroke="{ARROW_COLOR}" '
        'stroke-width="3" stroke-linecap="round" marker-end="url(#arrow)"/>'
    )
    parts.append(
        f'<rect x="314" y="20" width="272" height="30" rx="8" fill="white" opacity="0.92"/>'
    )
    parts.append(_svg_text(450, 41, "electrons through external circuit", size=15, fill=ARROW_COLOR, weight="700"))

    # Reactions and ion flow.
    parts.extend(
        [
            _svg_text(192, 222, "H2", size=25, fill=COLOR_RED, weight="700"),
            _svg_text(192, 276, "splits into", size=15, fill=COLOR_RED, weight="600"),
            f'<rect x="136" y="300" width="112" height="34" rx="7" fill="white" opacity="0.88"/>',
            _svg_text(192, 323, "2H+ + 2e-", size=17, fill=COLOR_RED, weight="700"),
            _svg_text(708, 222, "1/2 O2", size=24, fill=bfs.BLUE, weight="700"),
            _svg_text(708, 276, "+ 2H+ + 2e-", size=16, fill=bfs.BLUE, weight="600"),
            f'<rect x="656" y="300" width="104" height="34" rx="7" fill="white" opacity="0.88"/>',
            _svg_text(708, 323, "-> H2O", size=18, fill=bfs.BLUE, weight="700"),
        ]
    )
    for y in (224, 284, 344):
        parts.append(
            f'<line x1="302" y1="{y}" x2="598" y2="{y}" stroke="{EDGE_WARM}" stroke-width="2.4" '
            'stroke-linecap="round" marker-end="url(#arrow)"/>'
        )
    parts.append(f'<rect x="424" y="190" width="52" height="30" rx="7" fill="white" opacity="0.9"/>')
    parts.append(_svg_text(450, 211, "H+", size=17, fill=EDGE_WARM, weight="700"))

    # Catalyst layers.
    for x, label_x in [(318, 318), (582, 582)]:
        parts.append(
            f'<rect x="{x}" y="114" width="18" height="330" fill="{BOX_GREEN}" stroke="{EDGE_GREEN}" '
            'stroke-width="1.5" opacity="0.84"/>'
        )
        parts.append(_svg_text(label_x, 476, "catalyst", size=14, fill=COLOR_GREEN, weight="600"))

    # Inputs and products.
    parts.extend(
        [
            _svg_text(54, 304, "H2 in", size=16, fill=COLOR_RED, weight="700", anchor="end"),
            f'<line x1="60" y1="300" x2="104" y2="300" stroke="{COLOR_RED}" stroke-width="2.5" marker-end="url(#arrow)"/>',
            _svg_text(874, 232, "O2 in", size=16, fill=bfs.BLUE, weight="700", anchor="end"),
            f'<line x1="874" y1="252" x2="804" y2="252" stroke="{bfs.BLUE}" stroke-width="2.5" marker-end="url(#arrow)"/>',
            _svg_text(874, 374, "H2O out", size=16, fill=COLOR_TEAL, weight="700", anchor="end"),
            f'<line x1="804" y1="350" x2="874" y2="350" stroke="{COLOR_TEAL}" stroke-width="2.5" marker-end="url(#arrow)"/>',
        ]
    )
    parts.append("</svg>\n")
    _write_svg_or_preview(output_path, "\n".join(parts))


# ──────────────────────────────────────────────
# Figure 2b: Activation Energy
# ──────────────────────────────────────────────
def generate_activation_energy_figure(output_path):
    """Clean reaction-coordinate diagram for one dissociation step."""
    fig, ax = plt.subplots(figsize=(7.3, 4.25))
    x = np.linspace(0, 1, 500)
    reactant = 0.0
    product = -1.5
    barrier = 0.62
    y = reactant + (product - reactant) / (1 + np.exp(-(x - 0.62) / 0.035))
    y += barrier * np.exp(-((x - 0.44) / 0.10) ** 2)

    ax.plot(x, y, color=EDGE_MAIN, lw=3.0, solid_capstyle='round')
    ax.hlines([reactant, product, barrier], 0.06, 0.94, colors=bfs.NEUTRAL,
              linestyles=':', linewidth=1.5)
    ax.text(0.06, reactant + 0.07, r'O$_2$', ha='left', va='bottom',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)
    ax.text(0.94, product + 0.08, r'O* + O*', ha='right', va='bottom',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)
    ax.text(0.10, reactant - 0.12, '0 eV', ha='left', va='top',
            fontsize=10, color=COLOR_RED, fontweight='bold')
    ax.text(0.83, product - 0.12, '-1.5 eV', ha='center', va='top',
            fontsize=10, color=COLOR_RED, fontweight='bold')

    peak_x = x[np.argmax(y)]
    ax.annotate('', xy=(peak_x + 0.10, barrier), xytext=(peak_x + 0.10, reactant),
                arrowprops=dict(arrowstyle='<->', color=TEXT_COLOR, lw=1.7))
    ax.text(peak_x + 0.14, (barrier + reactant) / 2,
            'activation\nenergy\n0.62 eV', ha='left', va='center',
            fontsize=10.5, color=TEXT_COLOR, fontweight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.0))

    ax.annotate('', xy=(0.15, product), xytext=(0.15, reactant),
                arrowprops=dict(arrowstyle='<->', color=TEXT_COLOR, lw=1.7))
    ax.text(0.19, (reactant + product) / 2,
            'reaction\nfree energy', ha='left', va='center',
            fontsize=10.5, color=TEXT_COLOR,
            bbox=dict(fc='white', ec='none', alpha=0.9, pad=2.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.9, 0.9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved activation energy figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 3: Gibbs Free Energy Diagram
# ──────────────────────────────────────────────
def generate_gibbs_energy_figure(output_path):
    """
    Gibbs free energy diagram for ORR on Pt(111) vs Ni(111).
    Shows energy steps for dissociative ORR pathway.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.55))

    # ORR dissociative pathway steps (approximate values in eV)
    # Reaction: O2 + 4H+ + 4e- → 2H2O
    # Steps: ½O2 → *O, *O + H+e- → *OH, *OH + H+e- → H2O
    steps = ['½O$_2$ + 2H$^+$\n+ 2e$^-$', '*O + H$^+$\n+ e$^-$',
             '*OH', 'H$_2$O']

    # Approximate Gibbs free energies (eV) relative to reference
    # Pt(111) — near-optimal, moderate steps
    G_Pt = [0.0, -0.8, -1.6, -2.46]

    # Ni(111) — binds too strongly, large final desorption barrier
    G_Ni = [0.0, -1.6, -2.5, -2.46]

    # Ideal (no overpotential) — equal steps of ~0.615 eV
    G_ideal = [0.0, -0.615, -1.23, -2.46]

    x_positions = [0, 1, 2, 3]
    platform_width = 0.35

    # Plot energy platforms and connecting lines
    for G_vals, color, label, ls, lw in [
        (G_ideal, bfs.NEUTRAL, 'Ideal (no overpotential)', '--', 1.5),
        (G_Pt, bfs.BLUE, 'Pt(111)', '-', 2.2),
        (G_Ni, COLOR_RED, 'Ni(111)', '-', 2.2),
    ]:
        for i, (xi, gi) in enumerate(zip(x_positions, G_vals)):
            ax.plot([xi - platform_width, xi + platform_width], [gi, gi],
                    color=color, linewidth=lw, linestyle=ls, zorder=3)
            if i < len(x_positions) - 1:
                ax.plot([xi + platform_width, x_positions[i + 1] - platform_width],
                        [gi, G_vals[i + 1]],
                        color=color, linewidth=lw * 0.6, linestyle=ls,
                        alpha=0.5, zorder=2)

        # Legend entry (invisible line for legend)
        ax.plot([], [], color=color, linewidth=lw, linestyle=ls, label=label)

    # Annotate rate-limiting step for Pt (largest drop)
    # For Pt, the first step (½O2 → *O) has ΔG = -0.8 eV
    ax.annotate('', xy=(0.5, G_Pt[1]), xytext=(0.5, G_Pt[0]),
                arrowprops=dict(arrowstyle='<->', color=bfs.BLUE, lw=1.5))
    ax.text(0.65, (G_Pt[0] + G_Pt[1]) / 2, '0.8 eV',
            ha='left', va='center', fontsize=9, color=bfs.BLUE,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=bfs.BLUE, lw=0.5))

    # Annotate rate-limiting step for Ni (*OH → H2O, desorption)
    # Ni has a very large last step
    ax.annotate('', xy=(2.5, G_Ni[2]), xytext=(2.5, G_Ni[3]),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=1.5))
    ax.text(2.28, (G_Ni[2] + G_Ni[3]) / 2 + 0.12, 'small',
            ha='right', va='center', fontsize=8.5, color=COLOR_RED,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=COLOR_RED_LIGHT, lw=0.5))

    # For Ni: annotate the large first drop
    ax.annotate('', xy=(0.5, G_Ni[1]), xytext=(0.5, G_Ni[0]),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=1.5))
    ax.text(0.30, (G_Ni[0] + G_Ni[1]) / 2, '1.6 eV\nstrong binding',
            ha='right', va='center', fontsize=8.5, color=COLOR_RED,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=COLOR_RED_LIGHT, lw=0.5))

    # Step labels at bottom
    for i, label in enumerate(steps):
        ax.text(x_positions[i], -3.0, label, ha='center', va='top',
                fontsize=9.5, color=TEXT_COLOR)

    _style_axis(ax, (-0.6, 3.6), (-3.5, 0.6),
                ylabel='Gibbs free energy, $G$ (eV)')

    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
              edgecolor=bfs.SPINE)

    # Reaction coordinate label
    ax.text(1.5, -3.4, 'Reaction coordinate', ha='center', va='top',
            fontsize=11, color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Gibbs energy figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 4: Volcano Plot
# ──────────────────────────────────────────────
def generate_volcano_plot_figure(output_path):
    """
    Volcano plot: catalytic activity vs oxygen adsorption energy.
    Shows metals labeled, with "too strong" / "too weak" regions.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.95))

    # Approximate data: (ΔE_O in eV relative to Pt, log10(activity))
    # Based on Nørskov et al. 2004 ORR volcano
    metals = {
        'Pt':  (0.0, -0.2),
        'Pd':  (0.3, -0.6),
        'Ir':  (-0.2, -0.5),
        'Rh':  (-0.4, -0.8),
        'Ru':  (-0.8, -1.8),
        'Ni':  (-0.7, -1.5),
        'Co':  (-0.6, -1.3),
        'Fe':  (-1.2, -2.8),
        'Cu':  (0.5, -1.2),
        'Ag':  (1.2, -2.8),
        'Au':  (1.0, -2.2),
        'W':   (-1.5, -3.5),
    }

    # Volcano envelope (two lines meeting at peak)
    x_left = np.linspace(-2.0, 0.0, 100)
    x_right = np.linspace(0.0, 1.8, 100)
    slope_left = 0.9
    slope_right = -1.5
    y_peak = 0.0
    y_left = y_peak + slope_left * x_left
    y_right = y_peak + slope_right * x_right

    # Fill regions
    ax.fill_between(x_left, -4.5, y_left, color=COLOR_RED_LIGHT, alpha=0.15)
    ax.fill_between(x_right, -4.5, y_right, color=COLOR_TEAL_LIGHT, alpha=0.15)

    # Volcano lines
    ax.plot(x_left, y_left, color=COLOR_RED, linewidth=2, linestyle='--',
            alpha=0.6, label='Rate-limiting: desorption')
    ax.plot(x_right, y_right, color=COLOR_TEAL, linewidth=2, linestyle='--',
            alpha=0.6, label='Rate-limiting: adsorption')

    # Plot metals
    for metal, (x, y) in metals.items():
        color = COLOR_RED if x < -0.3 else (COLOR_TEAL if x > 0.3 else COLOR_GREEN)
        ax.plot(x, y, 'o', color=color, markersize=9, zorder=5,
                markeredgecolor='white', markeredgewidth=1.2)
        # Offset labels to avoid overlap
        offsets = {
            'Pt': (0.12, 0.15), 'Pd': (0.12, 0.1), 'Ir': (-0.15, 0.15),
            'Rh': (0.15, 0.1), 'Ru': (0.15, 0.1), 'Ni': (-0.15, 0.15),
            'Co': (0.15, 0.1), 'Fe': (0.15, 0.1), 'Cu': (0.15, 0.1),
            'Ag': (0.12, 0.15), 'Au': (-0.15, 0.15), 'W': (0.15, 0.1),
        }
        dx, dy = offsets.get(metal, (0.1, 0.1))
        ax.text(x + dx, y + dy, metal, fontsize=10, fontweight='bold',
                color=TEXT_COLOR, zorder=6)

    # Region labels
    ax.text(-1.45, -3.85, 'binds too\nstrongly', ha='center', va='center',
            fontsize=11, color=COLOR_RED, fontweight='bold', alpha=0.7,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
    ax.text(1.23, -3.85, 'binds too\nweakly', ha='center', va='center',
            fontsize=11, color=COLOR_TEAL, fontweight='bold', alpha=0.7,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))

    # Optimal region annotation
    ax.annotate('Optimal\nbinding', xy=(0.0, 0.0),
                xytext=(0.95, 0.95),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    _style_axis(ax, (-2.0, 1.8), (-4.5, 1.5),
                xlabel=r'$\Delta E_{\mathrm{O}}$ relative to Pt (eV)',
                ylabel=r'log$_{10}$(activity)')

    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9,
              edgecolor=bfs.SPINE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved volcano plot figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 5: Scaling Relations (*OH vs *OOH)
# ──────────────────────────────────────────────
def generate_scaling_relations_figure(output_path):
    """
    Scaling relations plot: *OOH binding energy vs *OH binding energy.
    Shows linear correlation with metals, and ideal point off the line.
    """
    fig, ax = plt.subplots(figsize=(6.6, 5.25))

    # Approximate data: (ΔG_OH eV, ΔG_OOH eV) relative to Pt
    # Scaling relation: ΔG_OOH ≈ ΔG_OH + 3.2 eV (constant offset)
    metals_scaling = {
        'Pt':  (0.8, 4.0),
        'Pd':  (1.0, 4.2),
        'Ir':  (0.6, 3.8),
        'Rh':  (0.5, 3.7),
        'Ru':  (0.2, 3.3),
        'Ni':  (0.1, 3.3),
        'Co':  (0.0, 3.2),
        'Fe':  (-0.3, 2.9),
        'Cu':  (1.2, 4.4),
        'Ag':  (1.6, 4.8),
        'Au':  (1.5, 4.7),
    }

    # Scaling line
    x_line = np.linspace(-0.8, 2.0, 100)
    y_line = x_line + 3.2  # ΔG_OOH = ΔG_OH + 3.2

    ax.plot(x_line, y_line, color=DENSITY_SLATE, linewidth=2,
            linestyle='--', alpha=0.6,
            label=r'$\Delta G_{*\mathrm{OOH}} = \Delta G_{*\mathrm{OH}} + 3.2$ eV')

    # Plot metals
    for metal, (x, y) in metals_scaling.items():
        ax.plot(x, y, 'o', color=EDGE_MAIN, markersize=9, zorder=5,
                markeredgecolor='white', markeredgewidth=1.2)
        offsets = {
            'Pt': (0.08, 0.12), 'Pd': (0.08, 0.12), 'Ir': (-0.12, -0.18),
            'Rh': (-0.12, 0.12), 'Ru': (0.08, 0.12), 'Ni': (0.08, -0.18),
            'Co': (-0.12, 0.12), 'Fe': (0.08, 0.12), 'Cu': (0.08, 0.12),
            'Ag': (0.08, -0.18), 'Au': (-0.12, 0.12),
        }
        dx, dy = offsets.get(metal, (0.08, 0.1))
        ax.text(x + dx, y + dy, metal, fontsize=10, fontweight='bold',
                color=TEXT_COLOR, zorder=6)

    # Ideal point (off the scaling line — needs different OOH/OH ratio)
    # For optimal ORR: ΔG_OOH - ΔG_OH ≈ 2.46 eV (not 3.2)
    ideal_oh = 0.8
    ideal_ooh = ideal_oh + 2.46
    ax.plot(ideal_oh, ideal_ooh, '*', color=COLOR_GREEN, markersize=16,
            zorder=6, markeredgecolor='white', markeredgewidth=1.0)
    ax.annotate('Ideal catalyst\n(breaks scaling)',
                xy=(ideal_oh, ideal_ooh),
                xytext=(ideal_oh + 0.62, ideal_ooh - 0.62),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    # Shade the gap between scaling line and ideal
    ax.fill_between([ideal_oh - 0.15, ideal_oh + 0.15],
                    [ideal_ooh - 0.15, ideal_ooh - 0.15],
                    [ideal_oh + 3.2 - 0.15, ideal_oh + 3.2 + 0.15],
                    color=COLOR_GREEN, alpha=0.1, zorder=1)

    # Arrow showing the offset
    ax.annotate('', xy=(ideal_oh - 0.15, ideal_ooh),
                xytext=(ideal_oh - 0.15, ideal_oh + 3.2),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=1.5))
    ax.text(ideal_oh - 0.3, (ideal_ooh + ideal_oh + 3.2) / 2,
            '0.74 eV\ngap', ha='right', va='center', fontsize=8.5,
            color=COLOR_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=COLOR_RED_LIGHT, lw=0.5))

    _style_axis(ax, (-0.8, 2.0), (2.2, 5.5),
                xlabel=r'$\Delta G_{*\mathrm{OH}}$ (eV)',
                ylabel=r'$\Delta G_{*\mathrm{OOH}}$ (eV)')

    ax.legend(loc='upper left', fontsize=8.6, framealpha=0.9,
              edgecolor=bfs.SPINE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved scaling relations figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 6: Catalyst Surface Binding Sites
# ──────────────────────────────────────────────
def generate_binding_sites_figure(output_path):
    """
    Top-down view of a catalyst surface (FCC 111) showing
    atop, bridge, and hollow binding sites.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # FCC(111) surface: hexagonal close-packed arrangement
    # Generate hexagonal lattice
    a = 1.0  # lattice spacing
    rows = 7
    cols = 7
    atoms = []

    for row in range(rows):
        for col in range(cols):
            x = col * a + (row % 2) * a / 2
            y = row * a * np.sqrt(3) / 2
            atoms.append((x, y))

    # Draw atoms
    atom_radius = 0.35
    for x, y in atoms:
        circle = plt.Circle((x, y), atom_radius, color=bfs.NEUTRAL,
                           ec=bfs.MUTED, linewidth=1.5, zorder=2)
        ax.add_patch(circle)

    # Highlight binding sites
    # Atop site (on top of an atom)
    atop_x, atop_y = atoms[17]  # pick a central atom
    ax.plot(atop_x, atop_y + 0.02, 's', color=COLOR_RED, markersize=12,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate('Atop\n(1 atom)', xy=(atop_x, atop_y + atom_radius + 0.1),
                xytext=(atop_x + 1.5, atop_y + 1.5),
                fontsize=11, fontweight='bold', color=COLOR_RED,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=COLOR_RED_LIGHT, alpha=0.9, lw=0.8))

    # Bridge site (between two atoms)
    a1_x, a1_y = atoms[24]
    a2_x, a2_y = atoms[25]
    bridge_x = (a1_x + a2_x) / 2
    bridge_y = (a1_y + a2_y) / 2
    ax.plot(bridge_x, bridge_y, 'D', color=EDGE_WARM, markersize=10,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate('Bridge\n(2 atoms)', xy=(bridge_x, bridge_y),
                xytext=(bridge_x + 1.8, bridge_y + 0.8),
                fontsize=11, fontweight='bold', color=EDGE_WARM,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_WARM, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=EDGE_WARM, alpha=0.9, lw=0.8))

    # Hollow site (center of three atoms)
    a1_x, a1_y = atoms[10]
    a2_x, a2_y = atoms[11]
    a3_x, a3_y = atoms[17]
    hollow_x = (a1_x + a2_x + a3_x) / 3
    hollow_y = (a1_y + a2_y + a3_y) / 3
    ax.plot(hollow_x, hollow_y, '^', color=COLOR_GREEN, markersize=12,
            zorder=5, markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate('Hollow\n(3 atoms)', xy=(hollow_x, hollow_y),
                xytext=(hollow_x - 2.0, hollow_y - 1.0),
                fontsize=11, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    # Title
    ax.text(3.0, 6.5, 'Binding Sites on FCC(111) Surface',
            ha='center', va='center', fontsize=13,
            color=TEXT_COLOR, fontweight='bold')

    # Legend-like note
    ax.text(3.0, -0.6, 'Top-down view of close-packed metal surface.\n'
            'Adsorbate can bond at atop (1), bridge (2), or hollow (3) sites.',
            ha='center', va='top', fontsize=9.5, color=bfs.MUTED,
            fontstyle='italic', linespacing=1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved binding sites figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 7: ML Relaxation Pipeline
# ──────────────────────────────────────────────
def generate_ml_pipeline_figure(output_path):
    """
    Three-column comparison:
    (a) DFT relaxation loop (slow)
    (b) ML relaxation (S2EF) — replace DFT with GNN
    (c) Direct prediction (IS2RE) — skip relaxation entirely
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5),
                              gridspec_kw={'wspace': 0.35})

    for ax in axes:
        ax.set_xlim(-2, 4)
        ax.set_ylim(-4.5, 4)
        ax.set_aspect('equal')
        ax.axis('off')

    bw, bh = 3.2, 0.7
    rounding = 0.12
    gap = 0.08

    def right_edge(cx):
        return cx + bw / 2 + rounding + gap

    def left_edge(cx):
        return cx - bw / 2 - rounding - gap

    def top_edge(cy):
        return cy + bh / 2 + rounding + gap

    def bot_edge(cy):
        return cy - bh / 2 - rounding - gap

    CX = 1.0

    # ── Panel (a): DFT Relaxation Loop ──
    ax = axes[0]
    ax.text(CX, 3.7, '(a) DFT Relaxation', ha='center', va='center',
            fontsize=12, fontweight='bold', color=TEXT_COLOR)

    boxes_a = [
        (CX, 2.5, 'Initial structure', BOX_MAIN, EDGE_MAIN),
        (CX, 1.0, 'DFT: compute\nenergy & forces', BOX_WARM, EDGE_WARM),
        (CX, -0.5, 'Update atom\npositions', BOX_MAIN, EDGE_MAIN),
        (CX, -2.0, 'Converged?', BOX_WARM, EDGE_WARM),
        (CX, -3.5, 'Relaxed energy', BOX_GREEN, EDGE_GREEN),
    ]

    for cx, cy, label, fc, ec in boxes_a:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=9.5)

    # Arrows down
    for i in range(len(boxes_a) - 1):
        _draw_arrow(ax, CX, bot_edge(boxes_a[i][1]),
                    CX, top_edge(boxes_a[i + 1][1]))

    # Loop arrow from "Update" back to "DFT"
    loop_x = CX + bw / 2 + rounding + 0.3
    ax.plot([loop_x, loop_x], [boxes_a[2][1], boxes_a[1][1]],
            color=COLOR_RED, lw=1.5, zorder=2)
    ax.annotate('', xy=(CX + bw / 2 + rounding + gap, boxes_a[1][1]),
                xytext=(loop_x, boxes_a[1][1]),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_RED,
                                lw=1.5, mutation_scale=12))
    ax.plot([CX + bw / 2 + rounding + gap, loop_x],
            [boxes_a[2][1], boxes_a[2][1]],
            color=COLOR_RED, lw=1.5, zorder=2)
    ax.text(loop_x + 0.15, (boxes_a[1][1] + boxes_a[2][1]) / 2,
            '50-400\nsteps', ha='left', va='center', fontsize=8,
            color=COLOR_RED, fontweight='bold')

    # Yes/No labels
    ax.text(CX + 0.5, boxes_a[3][1] - bh / 2 - rounding - 0.1,
            'Yes', fontsize=9, color=COLOR_GREEN, fontweight='bold',
            ha='left', va='top')

    # Cost annotation
    ax.text(CX, -4.3, 'Hours to days\nper relaxation',
            ha='center', va='top', fontsize=9, color=COLOR_RED,
            fontstyle='italic')

    # ── Panel (b): ML Relaxation (S2EF) ──
    ax = axes[1]
    ax.text(CX, 3.7, '(b) ML Relaxation (S2EF)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=TEXT_COLOR)

    boxes_b = [
        (CX, 2.5, 'Initial structure', BOX_MAIN, EDGE_MAIN),
        (CX, 1.0, 'GNN: predict\nenergy & forces', BOX_ML, EDGE_ML),
        (CX, -0.5, 'Update atom\npositions', BOX_MAIN, EDGE_MAIN),
        (CX, -2.0, 'Converged?', BOX_ML, EDGE_ML),
        (CX, -3.5, 'Relaxed energy', BOX_GREEN, EDGE_GREEN),
    ]

    for cx, cy, label, fc, ec in boxes_b:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=9.5)

    for i in range(len(boxes_b) - 1):
        _draw_arrow(ax, CX, bot_edge(boxes_b[i][1]),
                    CX, top_edge(boxes_b[i + 1][1]))

    # Loop arrow
    loop_x = CX + bw / 2 + rounding + 0.3
    ax.plot([loop_x, loop_x], [boxes_b[2][1], boxes_b[1][1]],
            color=EDGE_ML, lw=1.5, zorder=2)
    ax.annotate('', xy=(CX + bw / 2 + rounding + gap, boxes_b[1][1]),
                xytext=(loop_x, boxes_b[1][1]),
                arrowprops=dict(arrowstyle='-|>', color=EDGE_ML,
                                lw=1.5, mutation_scale=12))
    ax.plot([CX + bw / 2 + rounding + gap, loop_x],
            [boxes_b[2][1], boxes_b[2][1]],
            color=EDGE_ML, lw=1.5, zorder=2)
    ax.text(loop_x + 0.15, (boxes_b[1][1] + boxes_b[2][1]) / 2,
            'same\nsteps', ha='left', va='center', fontsize=8,
            color=EDGE_ML, fontweight='bold')

    ax.text(CX + 0.5, boxes_b[3][1] - bh / 2 - rounding - 0.1,
            'Yes', fontsize=9, color=COLOR_GREEN, fontweight='bold',
            ha='left', va='top')

    ax.text(CX, -4.3, 'Seconds per step\n(GPU inference)',
            ha='center', va='top', fontsize=9, color=EDGE_ML,
            fontstyle='italic')

    # ── Panel (c): Direct Prediction (IS2RE) ──
    ax = axes[2]
    ax.text(CX, 3.7, '(c) Direct Prediction (IS2RE)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=TEXT_COLOR)

    boxes_c = [
        (CX, 2.5, 'Initial structure', BOX_MAIN, EDGE_MAIN),
        (CX, 0.2, 'GNN: predict\nrelaxed energy', BOX_ML, EDGE_ML),
        (CX, -2.0, 'Relaxed energy', BOX_GREEN, EDGE_GREEN),
    ]

    for cx, cy, label, fc, ec in boxes_c:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=9.5)

    for i in range(len(boxes_c) - 1):
        _draw_arrow(ax, CX, bot_edge(boxes_c[i][1]),
                    CX, top_edge(boxes_c[i + 1][1]))

    # "Skip relaxation" annotation
    ax.annotate('No iterative\nrelaxation!',
                xy=(CX - bw / 2 - rounding - 0.1, 0.2),
                xytext=(CX - bw / 2 - rounding - 0.8, 1.5),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    ax.text(CX, -4.3, 'Single forward pass\n(milliseconds)',
            ha='center', va='top', fontsize=9, color=COLOR_GREEN,
            fontstyle='italic')

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ML pipeline figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 8: OER Catalyst Discovery Workflow
# ──────────────────────────────────────────────
def generate_oer_workflow_figure(output_path):
    """Readable schematic of the oxide OER discovery workflow."""
    width, height = 900, 420
    stages = [
        (48, ["(a) select", "bulk oxide"], BOX_MAIN, EDGE_MAIN),
        (264, ["(b) enumerate", "surface terminations"], BOX_WARM, EDGE_WARM),
        (480, ["(c) place", "adsorbate"], BOX_CATALYST, EDGE_CATALYST),
        (696, ["(d) relax", "structure"], BOX_GREEN, EDGE_GREEN),
    ]
    box_w, box_h = 160, 70
    parts = [
        _svg_header(width, height),
        _svg_text(width / 2, 48, "Typical OER catalyst discovery workflow", size=23, weight="700"),
    ]

    for x, lines, fill, stroke in stages:
        parts.append(_svg_box(x, 94, box_w, box_h, lines, fill=fill, stroke=stroke, size=16))
    for left, right in zip(stages[:-1], stages[1:]):
        parts.append(_svg_arrow(left[0] + box_w + 18, 129, right[0] - 18, 129, width=2.6))

    # Compact oxide/surface sketches: semantic, not a detailed atomistic model.
    sketch_y = 220
    atom_colors = [COLOR_RED, bfs.NEUTRAL, COLOR_RED, bfs.NEUTRAL, COLOR_RED, bfs.NEUTRAL]
    for x, _, _, _ in stages:
        cx = x + box_w / 2
        coords = [
            (cx - 26, sketch_y - 10),
            (cx, sketch_y - 10),
            (cx + 26, sketch_y - 10),
            (cx - 13, sketch_y + 16),
            (cx + 13, sketch_y + 16),
            (cx + 39, sketch_y + 16),
        ]
        for (px, py), color in zip(coords, atom_colors):
            parts.append(f'<circle cx="{px}" cy="{py}" r="9" fill="{color}" stroke="white" stroke-width="1.5"/>')

    # Surface termination choice.
    term_cx = stages[1][0] + box_w / 2
    for idx, color in enumerate([EDGE_MAIN, EDGE_WARM, EDGE_GREEN]):
        y = 260 + idx * 18
        parts.append(
            f'<line x1="{term_cx - 50}" y1="{y}" x2="{term_cx + 50}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        )
    parts.append(
        _svg_text(
            term_cx,
            335,
            ["surface Pourbaix", "selects stable termination"],
            size=14,
            fill=EDGE_WARM,
            weight="600",
        )
    )

    ads_cx = stages[2][0] + box_w / 2
    relax_cx = stages[3][0] + box_w / 2
    parts.append(f'<circle cx="{ads_cx}" cy="194" r="12" fill="{COLOR_RED}" stroke="white" stroke-width="2"/>')
    parts.append(
        _svg_text(ads_cx, 335, ["O*, OH*, OOH*", "intermediates"], size=14, fill=EDGE_CATALYST, weight="600")
    )
    parts.append(
        _svg_text(
            relax_cx,
            335,
            ["adsorption energy", "feeds catalyst ranking"],
            size=14,
            fill=EDGE_GREEN,
            weight="600",
        )
    )
    parts.append(
        _svg_text(
            width / 2,
            392,
            "Oxides require both surface selection and adsorbate placement before relaxation.",
            size=15,
            fill=bfs.MUTED,
            weight="500",
        )
    )
    parts.append("</svg>\n")
    _write_svg_or_preview(output_path, "\n".join(parts))


# ──────────────────────────────────────────────
if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    for ext in ('svg', 'png'):
        generate_energy_cycle_figure(os.path.join(output_dir, f'ec_energy_cycle.{ext}'))
        generate_fuel_cell_figure(os.path.join(output_dir, f'ec_fuel_cell.{ext}'))
        generate_activation_energy_figure(os.path.join(output_dir, f'ec_activation_energy.{ext}'))
        generate_gibbs_energy_figure(os.path.join(output_dir, f'ec_gibbs_energy.{ext}'))
        generate_volcano_plot_figure(os.path.join(output_dir, f'ec_volcano_plot.{ext}'))
        generate_scaling_relations_figure(os.path.join(output_dir, f'ec_scaling_relations.{ext}'))
        generate_binding_sites_figure(os.path.join(output_dir, f'ec_binding_sites.{ext}'))
        generate_ml_pipeline_figure(os.path.join(output_dir, f'ec_ml_pipeline.{ext}'))
        generate_oer_workflow_figure(os.path.join(output_dir, f'ec_oer_workflow.{ext}'))

    print("\nDone! All 9 figures generated.")
