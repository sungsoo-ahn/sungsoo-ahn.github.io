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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import Arc
import matplotlib.patheffects as pe


# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
TEXT_COLOR = '#263238'
ARROW_COLOR = '#455a64'

# Primary boxes (slate blue)
BOX_MAIN = '#dce8f4'
EDGE_MAIN = '#5b7fa5'

# Energy / warm
BOX_WARM = '#fff3e0'
EDGE_WARM = '#e8a030'
COLOR_WARM = '#e07a5f'

# Green / optimal
BOX_GREEN = '#e0f2e9'
EDGE_GREEN = '#4caf50'
COLOR_GREEN = '#388e3c'

# Red / strong binding
COLOR_RED = '#d32f2f'
COLOR_RED_LIGHT = '#ef9a9a'

# Teal / weak binding
COLOR_TEAL = '#1a8a7a'
COLOR_TEAL_LIGHT = '#b2dfdb'

# Special
BOX_CATALYST = '#f3e5f5'
EDGE_CATALYST = '#9c27b0'
BOX_ML = '#e8eaf6'
EDGE_ML = '#5c6bc0'

DENSITY_SLATE = '#5b7fa5'
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
    ax.spines['left'].set_color('#b0bec5')
    ax.spines['bottom'].set_color('#b0bec5')
    ax.tick_params(colors='#78909c', labelsize=9)
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


# ──────────────────────────────────────────────
# Figure 1: Energy Storage Cycle
# ──────────────────────────────────────────────
def generate_energy_cycle_figure(output_path):
    """
    Flowchart: Renewable → Electrolyzer → H2/CH4 → Fuel Cell → Grid
    With a return arrow showing the cycle.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Box dimensions
    bw, bh = 2.4, 0.9
    gap = 0.15

    # Positions (left to right)
    positions = [
        (1.2, 1.5, 'Renewable\nElectricity', BOX_GREEN, EDGE_GREEN),
        (4.8, 1.5, 'Electrolyzer', BOX_WARM, EDGE_WARM),
        (8.4, 1.5, 'H$_2$ / CH$_4$\nStorage', BOX_MAIN, EDGE_MAIN),
        (12.0, 1.5, 'Fuel Cell', BOX_WARM, EDGE_WARM),
    ]

    # Draw boxes
    for cx, cy, label, fc, ec in positions:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=11)

    # Forward arrows
    rounding = 0.12
    for i in range(len(positions) - 1):
        x1 = positions[i][0] + bw / 2 + rounding + gap
        x2 = positions[i + 1][0] - bw / 2 - rounding - gap
        y = positions[i][1]
        _draw_arrow(ax, x1, y, x2, y)

    # Labels on arrows
    arrow_labels = [
        (3.0, 2.15, 'H$_2$O + electricity', EDGE_WARM),
        (6.6, 2.15, 'chemical\nenergy', EDGE_MAIN),
        (10.2, 2.15, 'electricity\n+ heat', EDGE_WARM),
    ]
    for x, y, label, color in arrow_labels:
        ax.text(x, y, label, ha='center', va='bottom',
                fontsize=8.5, color=color, fontstyle='italic',
                linespacing=1.2)

    # Output arrow from fuel cell to "Grid"
    grid_x, grid_y = 14.5, 1.5
    _draw_arrow(ax, 12.0 + bw / 2 + rounding + gap, 1.5,
                grid_x - 0.3, grid_y, color=COLOR_GREEN)
    ax.text(grid_x + 0.1, grid_y, 'Grid', ha='left', va='center',
            fontsize=12, color=COLOR_GREEN, fontweight='bold')

    # Return arrow (bottom) showing cycle
    return_y = -0.3
    # Right end
    ax.annotate('', xy=(1.2, 1.5 - bh / 2 - rounding - gap),
                xytext=(1.2, return_y),
                arrowprops=dict(arrowstyle='-|>', color=COLOR_GREEN,
                                lw=1.4, mutation_scale=12))
    ax.plot([1.2, 12.0], [return_y, return_y],
            color=COLOR_GREEN, lw=1.4, zorder=2)
    ax.plot([12.0, 12.0], [return_y, 1.5 - bh / 2 - rounding - gap],
            color=COLOR_GREEN, lw=1.4, zorder=2)

    ax.text(6.6, return_y - 0.35, 'H$_2$O byproduct recycles',
            ha='center', va='top', fontsize=9, color=COLOR_GREEN,
            fontstyle='italic')

    # Title-like annotation
    ax.text(7.1, 3.2, 'Hydrogen Energy Storage Cycle',
            ha='center', va='center', fontsize=13,
            color=TEXT_COLOR, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved energy cycle figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 2: PEM Fuel Cell Schematic
# ──────────────────────────────────────────────
def generate_fuel_cell_figure(output_path):
    """
    Simplified PEM fuel cell cross-section:
    Anode | Membrane | Cathode with reactions labeled.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Three regions
    regions = [
        (0, 0, 3.5, 6, '#e8eaf6', '#5c6bc0', 'Anode'),
        (3.5, 0, 3, 6, '#fff8e1', '#ffa000', 'Membrane\n(PEM)'),
        (6.5, 0, 3.5, 6, '#fce4ec', '#e53935', 'Cathode'),
    ]

    for x, y, w, h, fc, ec, label in regions:
        rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                         linewidth=2, zorder=1)
        ax.add_patch(rect)
        ax.text(x + w / 2, h + 0.3, label, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=ec)

    # Anode reaction
    ax.text(1.75, 4.5, 'H$_2$', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#5c6bc0')
    ax.text(1.75, 3.5, r'$\rightarrow$ 2H$^+$ + 2e$^-$',
            ha='center', va='center', fontsize=11, color='#5c6bc0')

    # H+ arrows through membrane
    for y_pos in [2.0, 3.0, 4.0]:
        _draw_arrow(ax, 3.2, y_pos, 6.8, y_pos,
                    color='#ffa000', lw=1.2, ms=10)
    ax.text(5.0, 4.6, 'H$^+$', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#ffa000')

    # Cathode reaction
    ax.text(8.25, 4.5, '½O$_2$', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#e53935')
    ax.text(8.25, 3.5, '+ 2H$^+$ + 2e$^-$',
            ha='center', va='center', fontsize=11, color='#e53935')
    ax.text(8.25, 2.5, r'$\rightarrow$ H$_2$O',
            ha='center', va='center', fontsize=12,
            fontweight='bold', color='#e53935')

    # Electron flow (external circuit - top)
    ax.annotate('', xy=(8.25, 6.8), xytext=(1.75, 6.8),
                arrowprops=dict(arrowstyle='-|>', color=ARROW_COLOR,
                                lw=2.0, mutation_scale=14,
                                connectionstyle='arc3,rad=-0.15'))
    ax.text(5.0, 7.2, 'e$^-$ (external circuit)', ha='center', va='bottom',
            fontsize=10, color=ARROW_COLOR, fontweight='bold')

    # Catalyst layers (thin strips)
    for x_pos, label in [(3.2, 'catalyst'), (6.8, 'catalyst')]:
        rect = Rectangle((x_pos - 0.15, 0), 0.3, 6,
                         facecolor='#c8e6c9', edgecolor='#66bb6a',
                         linewidth=1, alpha=0.7, zorder=2)
        ax.add_patch(rect)
        ax.text(x_pos, -0.5, label, ha='center', va='top',
                fontsize=8, color='#388e3c', fontstyle='italic', rotation=0)

    # Input/output labels
    ax.text(-0.3, 3.0, 'H$_2$ in', ha='right', va='center',
            fontsize=10, color='#5c6bc0', fontweight='bold')
    _draw_arrow(ax, -0.1, 3.0, 0.3, 3.0, color='#5c6bc0', lw=1.5)

    ax.text(10.8, 4.0, 'O$_2$ in', ha='left', va='center',
            fontsize=10, color='#e53935', fontweight='bold')
    _draw_arrow(ax, 10.6, 4.0, 10.2, 4.0, color='#e53935', lw=1.5)

    ax.text(10.8, 2.0, 'H$_2$O out', ha='left', va='center',
            fontsize=10, color='#1a8a7a', fontweight='bold')
    _draw_arrow(ax, 10.0, 2.0, 10.6, 2.0, color='#1a8a7a', lw=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved fuel cell figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 3: Gibbs Free Energy Diagram
# ──────────────────────────────────────────────
def generate_gibbs_energy_figure(output_path):
    """
    Gibbs free energy diagram for ORR on Pt(111) vs Ni(111).
    Shows energy steps for dissociative ORR pathway.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

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
        (G_ideal, '#9e9e9e', 'Ideal (no overpotential)', '--', 1.5),
        (G_Pt, '#5c6bc0', 'Pt(111)', '-', 2.2),
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
                arrowprops=dict(arrowstyle='<->', color='#5c6bc0', lw=1.5))
    ax.text(0.65, (G_Pt[0] + G_Pt[1]) / 2, '0.8 eV',
            ha='left', va='center', fontsize=9, color='#5c6bc0',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec='#5c6bc0', lw=0.5))

    # Annotate rate-limiting step for Ni (*OH → H2O, desorption)
    # Ni has a very large last step
    ax.annotate('', xy=(2.5, G_Ni[2]), xytext=(2.5, G_Ni[3]),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=1.5))
    ax.text(2.35, (G_Ni[2] + G_Ni[3]) / 2 + 0.05, 'small\n(easy)',
            ha='right', va='center', fontsize=8.5, color=COLOR_RED,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=COLOR_RED_LIGHT, lw=0.5))

    # For Ni: annotate the large first drop
    ax.annotate('', xy=(0.5, G_Ni[1]), xytext=(0.5, G_Ni[0]),
                arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=1.5))
    ax.text(0.35, (G_Ni[0] + G_Ni[1]) / 2, '1.6 eV\n(too strong)',
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
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
              edgecolor='#b0bec5')

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
    fig, ax = plt.subplots(figsize=(8, 5.5))

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
    ax.text(-1.5, -3.8, 'Binds too\nstrongly', ha='center', va='center',
            fontsize=12, color=COLOR_RED, fontweight='bold', alpha=0.7)
    ax.text(1.3, -3.8, 'Binds too\nweakly', ha='center', va='center',
            fontsize=12, color=COLOR_TEAL, fontweight='bold', alpha=0.7)

    # Optimal region annotation
    ax.annotate('Optimal\nbinding', xy=(0.0, 0.0),
                xytext=(0.8, 0.8),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    _style_axis(ax, (-2.0, 1.8), (-4.5, 1.5),
                xlabel=r'$\Delta E_{\mathrm{O}}$ relative to Pt (eV)',
                ylabel=r'log$_{10}$(activity)')

    ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
              edgecolor='#b0bec5')

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
    fig, ax = plt.subplots(figsize=(7, 6))

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
                xytext=(ideal_oh + 0.5, ideal_ooh - 0.5),
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
            'Gap:\n0.74 eV', ha='right', va='center', fontsize=9,
            color=COLOR_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=COLOR_RED_LIGHT, lw=0.5))

    _style_axis(ax, (-0.8, 2.0), (2.2, 5.5),
                xlabel=r'$\Delta G_{*\mathrm{OH}}$ (eV)',
                ylabel=r'$\Delta G_{*\mathrm{OOH}}$ (eV)')

    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9,
              edgecolor='#b0bec5')

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
        circle = plt.Circle((x, y), atom_radius, color='#b0bec5',
                           ec='#78909c', linewidth=1.5, zorder=2)
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
            ha='center', va='top', fontsize=9.5, color='#607d8b',
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
        (CX, -2.0, 'Converged?', '#fff3e0', EDGE_WARM),
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
        (CX, -2.0, 'Converged?', '#e8eaf6', EDGE_ML),
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
if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    generate_energy_cycle_figure(os.path.join(output_dir, 'ec_energy_cycle.png'))
    generate_fuel_cell_figure(os.path.join(output_dir, 'ec_fuel_cell.png'))
    generate_gibbs_energy_figure(os.path.join(output_dir, 'ec_gibbs_energy.png'))
    generate_volcano_plot_figure(os.path.join(output_dir, 'ec_volcano_plot.png'))
    generate_scaling_relations_figure(os.path.join(output_dir, 'ec_scaling_relations.png'))
    generate_binding_sites_figure(os.path.join(output_dir, 'ec_binding_sites.png'))
    generate_ml_pipeline_figure(os.path.join(output_dir, 'ec_ml_pipeline.png'))

    print("\nDone! All 7 figures generated.")
