"""
Generate figures for the protein design blog post.

Fourteen figures covering: amino acids, protein structure levels, secondary
structure, hydrophobic core, binding interface, antibody structure, energy
landscape, design problems, RFDiffusion, ProteinMPNN, AlphaFold, binder
example, self-consistency workflow, and design funnel.

Color convention (shared with electrocatalysis figures):
  - Slate blue for structural/neutral elements
  - Warm red/orange for energy barriers and strong binding
  - Green for optimal/favorable regions
  - Teal for cool/weak binding
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle, Wedge, Arc, Polygon)
import matplotlib.patheffects as pe
from matplotlib.collections import PatchCollection


# ──────────────────────────────────────────────
# Color palette (same as electrocatalysis)
# ──────────────────────────────────────────────
TEXT_COLOR = '#263238'
ARROW_COLOR = '#455a64'

BOX_MAIN = '#dce8f4'
EDGE_MAIN = '#5b7fa5'

BOX_WARM = '#fff3e0'
EDGE_WARM = '#e8a030'
COLOR_WARM = '#e07a5f'

BOX_GREEN = '#e0f2e9'
EDGE_GREEN = '#4caf50'
COLOR_GREEN = '#388e3c'

COLOR_RED = '#d32f2f'
COLOR_RED_LIGHT = '#ef9a9a'

COLOR_TEAL = '#1a8a7a'
COLOR_TEAL_LIGHT = '#b2dfdb'

BOX_CATALYST = '#f3e5f5'
EDGE_CATALYST = '#9c27b0'
BOX_ML = '#e8eaf6'
EDGE_ML = '#5c6bc0'

DENSITY_SLATE = '#5b7fa5'
ANNOTATION_BG = 'white'
ANNOTATION_BG_ALPHA = 0.85

LABEL_FS = 12
SUBLABEL_FS = 10

# Extra colors for protein figures
COLOR_HYDROPHOBIC = '#5b7fa5'      # slate blue (nonpolar)
COLOR_POLAR = '#4caf50'            # green
COLOR_POSITIVE = '#5c6bc0'         # indigo
COLOR_NEGATIVE = '#e07a5f'         # warm red
COLOR_SPECIAL = '#e8a030'          # amber
BOX_HYDROPHOBIC = '#dce8f4'
BOX_POLAR = '#e0f2e9'
BOX_POSITIVE = '#e8eaf6'
BOX_NEGATIVE = '#fff3e0'
BOX_SPECIAL = '#fff8e1'


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────
def _style_axis(ax, xlim, ylim, xlabel=None, ylabel=None):
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
              fontsize=10, fontweight='bold', text_color=TEXT_COLOR, lw=1.8,
              zorder=3):
    box = FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle=f'round,pad={rounding}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            zorder=zorder + 1)
    return box


def _draw_arrow(ax, x1, y1, x2, y2, color=None, lw=1.6, ms=14, zorder=2):
    if color is None:
        color = ARROW_COLOR
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', color=color,
        linewidth=lw, mutation_scale=ms, zorder=zorder)
    ax.add_patch(a)


# ──────────────────────────────────────────────
# Figure 1: Amino Acids
# ──────────────────────────────────────────────
def generate_amino_acids_figure(output_path):
    """20 amino acids grouped by chemistry with backbone structure."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.2, 'The 20 Standard Amino Acids', ha='center', va='center',
            fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # Backbone structure at top
    backbone_y = 8.0
    bb_labels = [
        (2.5, backbone_y, 'N', '#5c6bc0'),
        (4.5, backbone_y, r'C$_\alpha$', TEXT_COLOR),
        (6.5, backbone_y, "C'", TEXT_COLOR),
        (8.5, backbone_y, 'O', COLOR_RED),
    ]

    for x, y, label, color in bb_labels:
        circle = plt.Circle((x, y), 0.35, facecolor='white', edgecolor=color,
                            linewidth=2.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color, zorder=4)

    # Bonds
    for (x1, _, _, _), (x2, _, _, _) in zip(bb_labels[:-1], bb_labels[1:]):
        ax.plot([x1 + 0.35, x2 - 0.35], [backbone_y, backbone_y],
                color='#78909c', lw=2, zorder=2)

    # R group hanging off Calpha
    ax.plot([4.5, 4.5], [backbone_y - 0.35, backbone_y - 1.1],
            color='#78909c', lw=2, zorder=2)
    r_circle = plt.Circle((4.5, backbone_y - 1.5), 0.4,
                          facecolor=BOX_WARM, edgecolor=EDGE_WARM,
                          linewidth=2.0, zorder=3)
    ax.add_patch(r_circle)
    ax.text(4.5, backbone_y - 1.5, 'R', ha='center', va='center',
            fontsize=12, fontweight='bold', color=EDGE_WARM, zorder=4)
    ax.text(5.2, backbone_y - 1.5, 'Side chain\n(variable)', ha='left',
            va='center', fontsize=9, color=EDGE_WARM, fontstyle='italic')

    # H on N
    ax.plot([2.5, 1.7], [backbone_y + 0.35, backbone_y + 0.9],
            color='#78909c', lw=1.5, zorder=2)
    ax.text(1.5, backbone_y + 1.1, 'H', fontsize=10, color='#78909c',
            ha='center', va='center')

    # Double bond to O
    ax.plot([8.5 + 0.35, 9.2], [backbone_y + 0.05, backbone_y + 0.05],
            color=COLOR_RED, lw=2, zorder=2)
    ax.plot([8.5 + 0.35, 9.2], [backbone_y - 0.05, backbone_y - 0.05],
            color=COLOR_RED, lw=2, zorder=2)

    # Backbone label
    ax.text(10.5, backbone_y, 'General amino acid\nbackbone structure',
            ha='left', va='center', fontsize=10, color='#607d8b',
            fontstyle='italic')

    # Groups of amino acids
    groups = [
        ('Nonpolar / Hydrophobic', BOX_HYDROPHOBIC, COLOR_HYDROPHOBIC,
         [('G', 'Gly'), ('A', 'Ala'), ('V', 'Val'), ('L', 'Leu'),
          ('I', 'Ile'), ('P', 'Pro'), ('F', 'Phe'), ('M', 'Met'), ('W', 'Trp')]),
        ('Polar Uncharged', BOX_POLAR, COLOR_POLAR,
         [('S', 'Ser'), ('T', 'Thr'), ('N', 'Asn'), ('Q', 'Gln'),
          ('Y', 'Tyr'), ('C', 'Cys')]),
        ('Positively Charged', BOX_POSITIVE, COLOR_POSITIVE,
         [('K', 'Lys'), ('R', 'Arg'), ('H', 'His')]),
        ('Negatively Charged', BOX_NEGATIVE, COLOR_NEGATIVE,
         [('D', 'Asp'), ('E', 'Glu')]),
    ]

    group_y_start = 5.8
    group_height = 1.1
    margin_x = 0.3
    aa_box_w = 1.1
    aa_box_h = 0.8

    for gi, (group_name, bg_color, text_color, aas) in enumerate(groups):
        gy = group_y_start - gi * (group_height + 0.35)

        # Group label
        ax.text(margin_x, gy + 0.05, group_name, ha='left', va='center',
                fontsize=10, fontweight='bold', color=text_color)

        # Amino acid boxes
        row_start_x = margin_x
        for ai, (code1, code3) in enumerate(aas):
            x = row_start_x + ai * (aa_box_w + 0.15)
            y = gy - 0.7
            box = FancyBboxPatch(
                (x, y - aa_box_h / 2), aa_box_w, aa_box_h,
                boxstyle='round,pad=0.06',
                facecolor=bg_color, edgecolor=text_color,
                linewidth=1.2, zorder=3, alpha=0.9)
            ax.add_patch(box)
            ax.text(x + aa_box_w / 2, y + 0.08, code1, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color, zorder=4)
            ax.text(x + aa_box_w / 2, y - 0.22, code3, ha='center', va='center',
                    fontsize=7.5, color=text_color, zorder=4, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved amino acids figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 2: Protein Structure Levels
# ──────────────────────────────────────────────
def generate_structure_levels_figure(output_path):
    """Four-panel: primary, secondary, tertiary, quaternary."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5),
                              gridspec_kw={'wspace': 0.3})

    titles = ['Primary\nStructure', 'Secondary\nStructure',
              'Tertiary\nStructure', 'Quaternary\nStructure']
    subtitles = ['Amino acid sequence', 'Local folding patterns',
                 'Complete 3D fold', 'Multi-chain assembly']
    colors = [EDGE_MAIN, COLOR_WARM, COLOR_GREEN, EDGE_ML]

    for ax, title, subtitle, color in zip(axes, titles, subtitles, colors):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 2.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0, 2.0, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)
        ax.text(0, -1.6, subtitle, ha='center', va='center',
                fontsize=9, color='#607d8b', fontstyle='italic')

    # Panel 1: Primary — sequence string
    ax = axes[0]
    seq = 'MKTLLILAVL'
    for i, aa in enumerate(seq):
        x = -1.2 + (i % 5) * 0.55
        y = 0.6 - (i // 5) * 0.55
        box = FancyBboxPatch((x - 0.2, y - 0.2), 0.4, 0.4,
                             boxstyle='round,pad=0.04',
                             facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
                             linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, aa, ha='center', va='center', fontsize=10,
                fontweight='bold', color=EDGE_MAIN, zorder=4)
    # Connecting lines
    for i in range(len(seq) - 1):
        x1 = -1.2 + (i % 5) * 0.55
        y1 = 0.6 - (i // 5) * 0.55
        x2 = -1.2 + ((i + 1) % 5) * 0.55
        y2 = 0.6 - ((i + 1) // 5) * 0.55
        if (i + 1) % 5 != 0:
            ax.plot([x1 + 0.2, x2 - 0.2], [y1, y2],
                    color=EDGE_MAIN, lw=1.5, zorder=2)

    # Panel 2: Secondary — helix and sheet icons
    ax = axes[1]
    # Alpha helix (sinusoidal)
    t = np.linspace(0, 3 * np.pi, 100)
    hx = 0.5 * np.sin(t) - 0.5
    hy = np.linspace(1.0, -0.8, 100)
    ax.plot(hx, hy, color=COLOR_WARM, lw=3, zorder=3)
    ax.text(-0.5, -1.1, r'$\alpha$-helix', ha='center', va='top',
            fontsize=9, color=COLOR_WARM, fontweight='bold')

    # Beta sheet (zigzag arrows)
    for dx in [0.3, 0.7, 1.1]:
        ax.annotate('', xy=(dx, 1.0), xytext=(dx, -0.8),
                    arrowprops=dict(arrowstyle='-|>', color=EDGE_WARM,
                                    lw=2.5, mutation_scale=12))
        # H-bonds between strands
        if dx < 1.0:
            for yy in [0.8, 0.3, -0.2, -0.6]:
                ax.plot([dx + 0.07, dx + 0.23], [yy, yy],
                        color='#bdbdbd', lw=1, linestyle=':', zorder=1)
    ax.text(0.7, -1.1, r'$\beta$-sheet', ha='center', va='top',
            fontsize=9, color=EDGE_WARM, fontweight='bold')

    # Panel 3: Tertiary — compact blob
    ax = axes[2]
    theta = np.linspace(0, 2 * np.pi, 200)
    r = 0.8 + 0.15 * np.sin(5 * theta) + 0.1 * np.cos(3 * theta)
    bx = r * np.cos(theta)
    by = r * np.sin(theta) + 0.1
    ax.fill(bx, by, facecolor=BOX_GREEN, edgecolor=EDGE_GREEN,
            linewidth=2, zorder=3)
    # internal features
    t2 = np.linspace(0.5, 2.5, 40)
    ax.plot(0.3 * np.sin(t2 * 3), np.linspace(0.5, -0.3, 40),
            color=COLOR_GREEN, lw=2, alpha=0.5, zorder=4)
    ax.plot(np.linspace(-0.3, 0.4, 20), [0.0] * 20,
            color=COLOR_GREEN, lw=2, alpha=0.5, zorder=4)

    # Panel 4: Quaternary — multiple chains
    ax = axes[3]
    offsets = [(-0.5, 0.4), (0.5, 0.4), (-0.5, -0.5), (0.5, -0.5)]
    chain_colors = ['#dce8f4', '#e8eaf6', '#e0f2e9', '#fff3e0']
    chain_edges = [EDGE_MAIN, EDGE_ML, EDGE_GREEN, EDGE_WARM]
    for (ox, oy), fc, ec in zip(offsets, chain_colors, chain_edges):
        theta_c = np.linspace(0, 2 * np.pi, 100)
        r_c = 0.45 + 0.05 * np.sin(4 * theta_c)
        cx = r_c * np.cos(theta_c) + ox
        cy = r_c * np.sin(theta_c) + oy
        ax.fill(cx, cy, facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)

    # Interface lines
    ax.plot([0, 0], [-0.1, 0.1], color='#bdbdbd', lw=1.5,
            linestyle='--', zorder=2)
    ax.plot([-0.1, 0.1], [0, 0], color='#bdbdbd', lw=1.5,
            linestyle='--', zorder=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved structure levels figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 3: Secondary Structure (helix + sheet)
# ──────────────────────────────────────────────
def generate_secondary_structure_figure(output_path):
    """Two-panel: alpha helix with H-bonds, beta sheet with H-bonds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'wspace': 0.3})

    # Panel 1: Alpha helix
    ax = axes[0]
    ax.set_xlim(-2.5, 3)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.3, 7.7, r'$\alpha$-Helix', ha='center', va='center',
            fontsize=13, fontweight='bold', color=COLOR_WARM)

    # Draw helix backbone as a series of residues in a helical arrangement
    n_res = 12
    for i in range(n_res):
        angle = i * 100 * np.pi / 180  # ~100 degrees per residue
        rise = i * 0.55  # 1.5 A rise per residue (scaled)
        x = 1.0 * np.cos(angle)
        y = rise
        r = 0.25
        c = plt.Circle((x, y), r, facecolor=BOX_WARM if i % 2 == 0 else BOX_MAIN,
                       edgecolor=EDGE_WARM if i % 2 == 0 else EDGE_MAIN,
                       linewidth=1.5, zorder=4)
        ax.add_patch(c)
        ax.text(x, y, str(i + 1), ha='center', va='center',
                fontsize=7, color=TEXT_COLOR, zorder=5)

        # Connect to next residue
        if i < n_res - 1:
            angle2 = (i + 1) * 100 * np.pi / 180
            rise2 = (i + 1) * 0.55
            x2 = 1.0 * np.cos(angle2)
            y2 = rise2
            ax.plot([x, x2], [y, y2], color='#90a4ae', lw=1.5, zorder=3)

    # H-bonds (every i to i+4)
    for i in range(n_res - 4):
        angle_i = i * 100 * np.pi / 180
        rise_i = i * 0.55
        xi = 1.0 * np.cos(angle_i)
        yi = rise_i

        angle_i4 = (i + 4) * 100 * np.pi / 180
        rise_i4 = (i + 4) * 0.55
        xi4 = 1.0 * np.cos(angle_i4)
        yi4 = rise_i4

        ax.plot([xi, xi4], [yi, yi4], color=COLOR_RED, lw=1.0,
                linestyle=':', alpha=0.6, zorder=2)

    ax.text(-2.0, 3.0, 'H-bond\n(i to i+4)', ha='center', va='center',
            fontsize=9, color=COLOR_RED, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec=COLOR_RED_LIGHT, alpha=0.9, lw=0.8))
    ax.annotate('', xy=(-0.5, 2.0), xytext=(-1.5, 2.7),
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.2))

    ax.text(0.3, -0.3, '3.6 residues per turn\n5.4 A pitch',
            ha='center', va='top', fontsize=9, color='#607d8b',
            fontstyle='italic')

    # Panel 2: Beta sheet
    ax = axes[1]
    ax.set_xlim(-1, 7)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(3.0, 7.7, r'$\beta$-Sheet', ha='center', va='center',
            fontsize=13, fontweight='bold', color=EDGE_MAIN)

    n_strands = 4
    strand_len = 6
    strand_spacing = 1.6

    for s in range(n_strands):
        sx = s * strand_spacing
        direction = 1 if s % 2 == 0 else -1  # antiparallel

        for r in range(strand_len):
            ry = r * 0.9 + 0.5
            rc = plt.Circle((sx, ry), 0.22,
                           facecolor=BOX_MAIN if s % 2 == 0 else BOX_WARM,
                           edgecolor=EDGE_MAIN if s % 2 == 0 else EDGE_WARM,
                           linewidth=1.2, zorder=4)
            ax.add_patch(rc)

            if r < strand_len - 1:
                ax.plot([sx, sx], [ry + 0.22, ry + 0.9 - 0.22],
                        color='#90a4ae', lw=1.5, zorder=3)

        # Arrow showing strand direction
        if direction == 1:
            ax.annotate('', xy=(sx, strand_len * 0.9 + 0.3),
                        xytext=(sx, 0.2),
                        arrowprops=dict(arrowstyle='-|>', color=EDGE_MAIN,
                                        lw=1.5, mutation_scale=10))
        else:
            ax.annotate('', xy=(sx, 0.2),
                        xytext=(sx, strand_len * 0.9 + 0.3),
                        arrowprops=dict(arrowstyle='-|>', color=EDGE_WARM,
                                        lw=1.5, mutation_scale=10))

    # H-bonds between strands
    for s in range(n_strands - 1):
        for r in range(strand_len):
            ry = r * 0.9 + 0.5
            sx1 = s * strand_spacing
            sx2 = (s + 1) * strand_spacing
            ax.plot([sx1 + 0.25, sx2 - 0.25], [ry, ry],
                    color=COLOR_RED, lw=1.0, linestyle=':', alpha=0.5,
                    zorder=2)

    ax.text(3.0, -0.3, 'H-bonds between strands\n(antiparallel shown)',
            ha='center', va='top', fontsize=9, color='#607d8b',
            fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved secondary structure figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 4: Hydrophobic Core
# ──────────────────────────────────────────────
def generate_hydrophobic_core_figure(output_path):
    """Cross-section showing nonpolar core and polar surface."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 5.2, 'Hydrophobic Core', ha='center', va='center',
            fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # Outer boundary (protein surface)
    theta = np.linspace(0, 2 * np.pi, 200)
    r_outer = 3.5 + 0.2 * np.sin(5 * theta) + 0.15 * np.cos(3 * theta)
    ax.fill(r_outer * np.cos(theta), r_outer * np.sin(theta),
            facecolor='#fafafa', edgecolor='#b0bec5', linewidth=2, zorder=1)

    # Inner hydrophobic core boundary
    r_inner = 2.0 + 0.1 * np.sin(4 * theta)
    ax.fill(r_inner * np.cos(theta), r_inner * np.sin(theta),
            facecolor='#eceff1', edgecolor='#90a4ae', linewidth=1.5,
            linestyle='--', zorder=2)

    # Hydrophobic residues in the core (dark circles)
    np.random.seed(42)
    n_core = 12
    for _ in range(n_core):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3, 1.6)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        c = plt.Circle((x, y), 0.25, facecolor=BOX_HYDROPHOBIC,
                       edgecolor=COLOR_HYDROPHOBIC, linewidth=1.5, zorder=4)
        ax.add_patch(c)

    # Polar/charged residues on the surface
    n_surface = 18
    for i in range(n_surface):
        angle = 2 * np.pi * i / n_surface + np.random.uniform(-0.1, 0.1)
        radius = np.random.uniform(2.5, 3.2)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Alternate between polar, positive, negative
        if i % 3 == 0:
            fc, ec = BOX_POLAR, COLOR_POLAR
        elif i % 3 == 1:
            fc, ec = BOX_POSITIVE, COLOR_POSITIVE
        else:
            fc, ec = BOX_NEGATIVE, COLOR_NEGATIVE
        c = plt.Circle((x, y), 0.25, facecolor=fc, edgecolor=ec,
                       linewidth=1.5, zorder=4)
        ax.add_patch(c)

    # Water molecules (outside)
    for i in range(14):
        angle = 2 * np.pi * i / 14 + 0.15
        radius = 4.2 + np.random.uniform(-0.2, 0.3)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ax.text(x, y, 'H$_2$O', ha='center', va='center',
                fontsize=7, color='#42a5f5', alpha=0.6, zorder=5)

    # Legend
    legend_x, legend_y = 3.5, -4.2
    items = [
        (BOX_HYDROPHOBIC, COLOR_HYDROPHOBIC, 'Nonpolar (core)'),
        (BOX_POLAR, COLOR_POLAR, 'Polar (surface)'),
        (BOX_POSITIVE, COLOR_POSITIVE, 'Positive (surface)'),
        (BOX_NEGATIVE, COLOR_NEGATIVE, 'Negative (surface)'),
    ]
    for i, (fc, ec, label) in enumerate(items):
        y = legend_y - i * 0.5
        c = plt.Circle((legend_x - 3.5, y), 0.15, facecolor=fc,
                       edgecolor=ec, linewidth=1.2, zorder=4)
        ax.add_patch(c)
        ax.text(legend_x - 3.1, y, label, ha='left', va='center',
                fontsize=9, color=TEXT_COLOR)

    # Annotation: hydrophobic core
    ax.annotate('Hydrophobic\ncore', xy=(0, 0), xytext=(-3.8, 2.5),
                fontsize=10, fontweight='bold', color=COLOR_HYDROPHOBIC,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_HYDROPHOBIC,
                                lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=EDGE_MAIN, alpha=0.9, lw=0.8))

    # Annotation: polar surface
    ax.annotate('Polar\nsurface', xy=(2.8, 1.5), xytext=(4.5, 3.0),
                fontsize=10, fontweight='bold', color=COLOR_POLAR,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_POLAR, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    # Annotation: water
    ax.text(0, -5.2, 'Aqueous environment',
            ha='center', va='top', fontsize=10, color='#42a5f5',
            fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved hydrophobic core figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 5: Binding Interface
# ──────────────────────────────────────────────
def generate_binding_interface_figure(output_path):
    """Two proteins docked with interface zone highlighted."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 4.2, 'Protein-Protein Binding Interface', ha='center',
            va='center', fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # Protein A (left) — larger
    theta = np.linspace(0, 2 * np.pi, 200)
    # Concave on the right side (interface)
    r_a = 2.5 + 0.3 * np.cos(2 * theta) - 0.5 * np.exp(
        -((theta - 0) ** 2) / 0.3)
    ax_a = r_a * np.cos(theta) - 2.5
    ay_a = r_a * np.sin(theta)
    ax.fill(ax_a, ay_a, facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
            linewidth=2, zorder=2)
    ax.text(-3.5, 0, 'Protein A', ha='center', va='center',
            fontsize=12, fontweight='bold', color=EDGE_MAIN, zorder=4)

    # Protein B (right) — smaller
    r_b = 1.8 + 0.2 * np.cos(3 * theta) - 0.4 * np.exp(
        -((theta - np.pi) ** 2) / 0.3)
    bx_b = r_b * np.cos(theta) + 2.0
    by_b = r_b * np.sin(theta)
    ax.fill(bx_b, by_b, facecolor=BOX_WARM, edgecolor=EDGE_WARM,
            linewidth=2, zorder=2)
    ax.text(2.8, 0, 'Protein B', ha='center', va='center',
            fontsize=12, fontweight='bold', color=EDGE_WARM, zorder=4)

    # Interface zone (highlighted strip)
    interface_x = np.array([-0.3, -0.3, 0.3, 0.3])
    interface_y = np.array([-2.0, 2.0, 2.0, -2.0])
    ax.fill(interface_x, interface_y, facecolor=COLOR_GREEN, alpha=0.15,
            zorder=1)
    ax.plot([-0.3, -0.3], [-2.0, 2.0], color=COLOR_GREEN, lw=1.5,
            linestyle='--', zorder=3)
    ax.plot([0.3, 0.3], [-2.0, 2.0], color=COLOR_GREEN, lw=1.5,
            linestyle='--', zorder=3)

    # Hotspot residues at interface
    hotspots = [(-0.8, 1.0), (-0.7, -0.5), (-0.6, 0.3),
                (0.6, 0.8), (0.7, -0.3), (0.5, -1.0)]
    for i, (hx, hy) in enumerate(hotspots):
        color = COLOR_RED if i < 3 else EDGE_WARM
        c = plt.Circle((hx, hy), 0.18, facecolor=color, edgecolor='white',
                       linewidth=1, zorder=5, alpha=0.8)
        ax.add_patch(c)

    # Labels
    ax.annotate('Binding\ninterface', xy=(0, 0), xytext=(0, 3.2),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    ax.annotate('Hotspot\nresidues', xy=(-0.7, 0.3), xytext=(-4.5, 2.5),
                fontsize=10, fontweight='bold', color=COLOR_RED,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=COLOR_RED_LIGHT, alpha=0.9, lw=0.8))

    # Shape complementarity note
    ax.text(0, -3.8, 'Shape complementarity at the binding interface\n'
            'drives molecular recognition and specificity',
            ha='center', va='top', fontsize=9.5, color='#607d8b',
            fontstyle='italic', linespacing=1.3)

    # Epitope / Paratope labels
    ax.text(-1.5, -2.5, 'Epitope\n(on A)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=EDGE_MAIN,
            bbox=dict(boxstyle='round,pad=0.2', fc=BOX_MAIN,
                      ec=EDGE_MAIN, alpha=0.9, lw=0.8))
    ax.text(1.5, -2.5, 'Paratope\n(on B)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=EDGE_WARM,
            bbox=dict(boxstyle='round,pad=0.2', fc=BOX_WARM,
                      ec=EDGE_WARM, alpha=0.9, lw=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved binding interface figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 6: Antibody Structure
# ──────────────────────────────────────────────
def generate_antibody_structure_figure(output_path):
    """Y-shaped antibody with labeled regions."""
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 6.7, 'Antibody Structure', ha='center', va='center',
            fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # Y-shape: two arms (Fab) + stem (Fc)
    # Heavy chain (darker)
    # Left arm heavy
    left_arm_x = [-0.5, -3.5, -3.8, -1.2, -0.5]
    left_arm_y = [1.5, 4.5, 5.5, 5.5, 1.5]
    ax.fill(left_arm_x, left_arm_y, facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
            linewidth=2, zorder=2)

    # Right arm heavy
    right_arm_x = [0.5, 3.5, 3.8, 1.2, 0.5]
    right_arm_y = [1.5, 4.5, 5.5, 5.5, 1.5]
    ax.fill(right_arm_x, right_arm_y, facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
            linewidth=2, zorder=2)

    # Stem (Fc) - heavy chain continues
    stem_x = [-0.8, -0.8, 0.8, 0.8]
    stem_y = [1.5, -3.5, -3.5, 1.5]
    ax.fill(stem_x, stem_y, facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
            linewidth=2, zorder=2)

    # Light chains (lighter, overlaid on arms)
    # Left arm light chain (outer part)
    left_light_x = [-1.8, -3.5, -3.8, -2.5, -1.8]
    left_light_y = [3.0, 4.5, 5.5, 5.5, 3.0]
    ax.fill(left_light_x, left_light_y, facecolor=BOX_WARM,
            edgecolor=EDGE_WARM, linewidth=2, zorder=3)

    # Right arm light chain
    right_light_x = [1.8, 3.5, 3.8, 2.5, 1.8]
    right_light_y = [3.0, 4.5, 5.5, 5.5, 3.0]
    ax.fill(right_light_x, right_light_y, facecolor=BOX_WARM,
            edgecolor=EDGE_WARM, linewidth=2, zorder=3)

    # CDR loops (tips of arms) — highlighted
    for cx, cy in [(-3.3, 5.3), (3.3, 5.3)]:
        c = plt.Circle((cx, cy), 0.5, facecolor=COLOR_RED_LIGHT,
                       edgecolor=COLOR_RED, linewidth=2, zorder=5)
        ax.add_patch(c)
        ax.text(cx, cy, 'CDR', ha='center', va='center',
                fontsize=8, fontweight='bold', color=COLOR_RED, zorder=6)

    # Antigen at one CDR
    antigen_theta = np.linspace(0, 2 * np.pi, 100)
    antigen_r = 0.7 + 0.1 * np.sin(5 * antigen_theta)
    ax_ag = antigen_r * np.cos(antigen_theta) - 3.3
    ay_ag = antigen_r * np.sin(antigen_theta) + 6.5
    ax.fill(ax_ag, ay_ag, facecolor=BOX_GREEN, edgecolor=EDGE_GREEN,
            linewidth=2, zorder=4)
    ax.text(-3.3, 6.5, 'Antigen', ha='center', va='center',
            fontsize=9, fontweight='bold', color=COLOR_GREEN, zorder=5)

    # Hinge region
    ax.plot([-0.5, -0.5], [1.2, 1.8], color=EDGE_MAIN, lw=2, zorder=2)
    ax.plot([0.5, 0.5], [1.2, 1.8], color=EDGE_MAIN, lw=2, zorder=2)

    # Region labels with arrows
    # Variable region
    ax.annotate('Variable\nregion (V)', xy=(-2.5, 5.0), xytext=(-5.5, 5.5),
                fontsize=10, fontweight='bold', color=COLOR_RED,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=COLOR_RED_LIGHT, alpha=0.9, lw=0.8))

    # Constant region label
    ax.annotate('Constant\nregion (C)', xy=(-1.5, 2.5), xytext=(-5.5, 2.5),
                fontsize=10, fontweight='bold', color=EDGE_MAIN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_MAIN, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_MAIN,
                          ec=EDGE_MAIN, alpha=0.9, lw=0.8))

    # Heavy chain
    ax.annotate('Heavy\nchain', xy=(1.5, 3.0), xytext=(5.0, 3.0),
                fontsize=10, fontweight='bold', color=EDGE_MAIN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_MAIN, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_MAIN,
                          ec=EDGE_MAIN, alpha=0.9, lw=0.8))

    # Light chain
    ax.annotate('Light\nchain', xy=(3.0, 4.5), xytext=(5.5, 5.5),
                fontsize=10, fontweight='bold', color=EDGE_WARM,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_WARM, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_WARM,
                          ec=EDGE_WARM, alpha=0.9, lw=0.8))

    # Fab region bracket
    ax.annotate('Fab', xy=(4.0, 4.0), xytext=(5.2, 1.0),
                fontsize=11, fontweight='bold', color=EDGE_ML,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_ML, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_ML,
                          ec=EDGE_ML, alpha=0.9, lw=0.8))

    # Fc region
    ax.annotate('Fc', xy=(0, -2.0), xytext=(3.5, -2.0),
                fontsize=11, fontweight='bold', color=EDGE_ML,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_ML, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_ML,
                          ec=EDGE_ML, alpha=0.9, lw=0.8))

    # CDR label
    ax.annotate('CDR loops\n(antigen binding)', xy=(3.3, 5.3),
                xytext=(5.5, 6.5),
                fontsize=9, fontweight='bold', color=COLOR_RED,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=COLOR_RED_LIGHT, alpha=0.9, lw=0.8))

    # Hinge label
    ax.text(0, 1.0, 'Hinge', ha='center', va='center', fontsize=9,
            color='#607d8b', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved antibody structure figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 7: Energy Landscape (Funnel)
# ──────────────────────────────────────────────
def generate_energy_landscape_figure(output_path):
    """Funnel-shaped energy landscape."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 7)
    ax.axis('off')

    # Funnel shape: two sides converging to bottom
    # Left side
    left_x = np.array([-4.5, -4.0, -3.2, -2.2, -1.2, -0.5, -0.2])
    left_y = np.array([6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5])
    # Add some ruggedness
    left_x_rugged = left_x + np.array([0, 0.3, -0.2, 0.4, -0.1, 0.2, 0])

    # Right side (mirror)
    right_x = -left_x_rugged[::-1]
    right_y = left_y[::-1]

    # Complete funnel outline
    funnel_x = np.concatenate([left_x_rugged, right_x])
    funnel_y = np.concatenate([left_y, right_y])

    # Fill funnel with gradient-like effect
    for i in range(len(left_y) - 1):
        y_top = left_y[i]
        y_bot = left_y[i + 1]
        x_left_top = left_x_rugged[i]
        x_left_bot = left_x_rugged[i + 1]
        x_right_top = -left_x_rugged[len(left_x_rugged) - 1 - i]
        x_right_bot = -left_x_rugged[len(left_x_rugged) - 2 - i]

        depth = i / (len(left_y) - 1)
        color = plt.cm.RdYlBu(0.3 + 0.6 * depth)
        alpha = 0.15 + 0.1 * depth

        verts = [(x_left_top, y_top), (x_left_bot, y_bot),
                 (x_right_bot, y_bot), (x_right_top, y_top)]
        poly = Polygon(verts, facecolor=color, alpha=alpha, zorder=1)
        ax.add_patch(poly)

    # Funnel outline
    ax.plot(left_x_rugged, left_y, color=EDGE_MAIN, lw=2, zorder=3)
    ax.plot(-left_x_rugged[::-1], left_y[::-1], color=EDGE_MAIN, lw=2,
            zorder=3)

    # Bottom well (native state)
    well_x = np.linspace(-0.6, 0.6, 50)
    well_y = 2.0 * well_x ** 2 - 0.3
    ax.plot(well_x, well_y, color=COLOR_GREEN, lw=3, zorder=4)
    ax.plot(0, -0.3, 'o', color=COLOR_GREEN, markersize=12, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.text(0, -0.8, 'Native state', ha='center', va='top',
            fontsize=11, fontweight='bold', color=COLOR_GREEN)

    # Unfolded states at top
    np.random.seed(123)
    for _ in range(8):
        ux = np.random.uniform(-4, 4)
        uy = np.random.uniform(5.5, 6.5)
        ax.plot(ux, uy, 'o', color=COLOR_WARM, markersize=6,
                alpha=0.6, zorder=4)
    ax.text(0, 6.8, 'Unfolded states\n(many conformations, high energy)',
            ha='center', va='bottom', fontsize=10, color=COLOR_WARM,
            fontweight='bold')

    # Intermediate states
    intermediates = [(-2.0, 3.8), (1.5, 3.2), (-1.0, 2.5)]
    for ix, iy in intermediates:
        ax.plot(ix, iy, 'o', color=EDGE_WARM, markersize=8, alpha=0.7,
                zorder=4, markeredgecolor='white', markeredgewidth=1)
    ax.annotate('Partially folded\nintermediates', xy=(1.5, 3.2),
                xytext=(3.5, 4.0),
                fontsize=9, fontweight='bold', color=EDGE_WARM,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=EDGE_WARM, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=EDGE_WARM, alpha=0.9, lw=0.8))

    # Energy axis
    ax.annotate('', xy=(-4.8, 0), xytext=(-4.8, 6.5),
                arrowprops=dict(arrowstyle='-|>', color=TEXT_COLOR, lw=1.5))
    ax.text(-4.8, 3.5, 'Free energy', ha='right', va='center',
            fontsize=11, color=TEXT_COLOR, rotation=90)

    # Conformational space axis
    ax.annotate('', xy=(4.5, -0.5), xytext=(-4.5, -0.5),
                arrowprops=dict(arrowstyle='-|>', color=TEXT_COLOR, lw=1.5))
    ax.text(0, -1.2, 'Conformational space', ha='center', va='top',
            fontsize=11, color=TEXT_COLOR)

    # Design annotation
    ax.text(0, -1.5, '"Design = finding sequences whose energy\n'
            'minimum matches the target structure"',
            ha='center', va='top', fontsize=9, color=COLOR_GREEN,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                      ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved energy landscape figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 8: Design Problems (3-panel)
# ──────────────────────────────────────────────
def generate_design_problems_figure(output_path):
    """Three panels: forward folding, inverse folding, de novo generation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                              gridspec_kw={'wspace': 0.35})

    bw, bh = 2.8, 0.8
    rounding = 0.12
    gap = 0.1

    panels = [
        {
            'title': 'Forward Folding',
            'color': EDGE_MAIN,
            'input': ('Sequence', BOX_MAIN, EDGE_MAIN),
            'model': ('AlphaFold', BOX_ML, EDGE_ML),
            'output': ('3D Structure', BOX_GREEN, EDGE_GREEN),
        },
        {
            'title': 'Inverse Folding',
            'color': EDGE_WARM,
            'input': ('Backbone', BOX_GREEN, EDGE_GREEN),
            'model': ('ProteinMPNN', BOX_ML, EDGE_ML),
            'output': ('Sequence', BOX_MAIN, EDGE_MAIN),
        },
        {
            'title': 'De Novo Generation',
            'color': EDGE_ML,
            'input': ('Functional spec', BOX_WARM, EDGE_WARM),
            'model': ('RFDiffusion', BOX_ML, EDGE_ML),
            'output': ('Novel backbone', BOX_GREEN, EDGE_GREEN),
        },
    ]

    for ax_i, (ax_obj, panel) in enumerate(zip(axes, panels)):
        ax_obj.set_xlim(-2, 4.5)
        ax_obj.set_ylim(-2, 3)
        ax_obj.set_aspect('equal')
        ax_obj.axis('off')

        # Panel title
        ax_obj.text(1.25, 2.6, panel['title'], ha='center', va='center',
                    fontsize=12, fontweight='bold', color=panel['color'])

        # Input box
        y_input = 1.2
        _draw_box(ax_obj, 1.25, y_input, bw, bh,
                  panel['input'][0], panel['input'][1], panel['input'][2],
                  fontsize=10)

        # Model box
        y_model = -0.1
        _draw_box(ax_obj, 1.25, y_model, bw, bh,
                  panel['model'][0], panel['model'][1], panel['model'][2],
                  fontsize=10)

        # Output box
        y_output = -1.4
        _draw_box(ax_obj, 1.25, y_output, bw, bh,
                  panel['output'][0], panel['output'][1], panel['output'][2],
                  fontsize=10)

        # Arrows
        _draw_arrow(ax_obj, 1.25, y_input - bh / 2 - rounding - gap,
                    1.25, y_model + bh / 2 + rounding + gap)
        _draw_arrow(ax_obj, 1.25, y_model - bh / 2 - rounding - gap,
                    1.25, y_output + bh / 2 + rounding + gap)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved design problems figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 9: RFDiffusion Overview
# ──────────────────────────────────────────────
def generate_rfdiffusion_overview_figure(output_path):
    """Schematic of diffusion process for backbone generation."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-2, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(7, 3.7, 'RFDiffusion: Protein Backbone Generation via Denoising',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=TEXT_COLOR)

    # Stages from noisy to clean
    stages = [
        (1.5, 'Noise\n(t = T)', COLOR_RED),
        (4.5, 'Noisy\n(t ~ T)', COLOR_WARM),
        (7.5, 'Denoising\n(t ~ T/2)', EDGE_WARM),
        (10.5, 'Almost clean\n(t ~ 0)', EDGE_MAIN),
        (13.0, 'Generated\nbackbone', COLOR_GREEN),
    ]

    for sx, label, color in stages:
        # Draw progressively more structured blobs
        theta = np.linspace(0, 2 * np.pi, 100)
        idx = stages.index((sx, label, color))
        noise = 0.5 * (1 - idx / 4)
        np.random.seed(42 + idx)

        if idx == 0:
            # Random cloud
            for _ in range(20):
                px = sx + np.random.normal(0, 0.6)
                py = 1.0 + np.random.normal(0, 0.6)
                ax.plot(px, py, '.', color=color, markersize=4, alpha=0.5)
        elif idx < 4:
            # Progressively more structured
            r = 0.8 + noise * np.random.randn(100) * 0.3
            r += 0.1 * np.sin(5 * theta) * (1 - noise)
            bx = r * np.cos(theta) + sx
            by = r * np.sin(theta) + 1.0
            ax.fill(bx, by, facecolor=color, alpha=0.15 + 0.1 * idx,
                    edgecolor=color, linewidth=1.5, zorder=2)
            # Some internal points
            n_pts = int(8 + idx * 3)
            for _ in range(n_pts):
                px = sx + np.random.normal(0, 0.4 * (1 - idx * 0.15))
                py = 1.0 + np.random.normal(0, 0.4 * (1 - idx * 0.15))
                ax.plot(px, py, '.', color=color, markersize=3 + idx, alpha=0.6)
        else:
            # Clean structure: draw as a compact chain
            n_nodes = 12
            angles = np.linspace(0, 2.5 * np.pi, n_nodes)
            chain_x = 0.6 * np.cos(angles) + sx
            chain_y = 0.6 * np.sin(angles) + 1.0
            ax.plot(chain_x, chain_y, '-o', color=color, lw=2.5,
                    markersize=5, markeredgecolor='white',
                    markeredgewidth=1, zorder=4)

        ax.text(sx, -0.5, label, ha='center', va='top', fontsize=9,
                fontweight='bold', color=color)

    # Arrows between stages
    for i in range(len(stages) - 1):
        sx1 = stages[i][0] + 1.0
        sx2 = stages[i + 1][0] - 1.0
        _draw_arrow(ax, sx1, 1.0, sx2, 1.0, color=ARROW_COLOR, lw=1.5)

    # "Denoise" labels on arrows
    for i in range(len(stages) - 1):
        mx = (stages[i][0] + stages[i + 1][0]) / 2
        ax.text(mx, 1.5, 'denoise', ha='center', va='bottom',
                fontsize=8, color=ARROW_COLOR, fontstyle='italic')

    # Target protein (conditioning)
    target_x, target_y = 10.5, -1.5
    theta_t = np.linspace(0, 2 * np.pi, 80)
    r_t = 0.5 + 0.05 * np.sin(4 * theta_t)
    tx = r_t * np.cos(theta_t) + target_x
    ty = r_t * np.sin(theta_t) + target_y
    ax.fill(tx, ty, facecolor=BOX_WARM, edgecolor=EDGE_WARM,
            linewidth=2, zorder=3)
    ax.text(target_x, target_y, 'Target', ha='center', va='center',
            fontsize=9, fontweight='bold', color=EDGE_WARM, zorder=4)

    # Arrow from target to denoising process
    ax.annotate('condition on', xy=(10.5, 0.2), xytext=(target_x, -0.8),
                fontsize=9, color=EDGE_WARM, ha='center', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=EDGE_WARM, lw=1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved RFDiffusion overview figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 10: ProteinMPNN Overview
# ──────────────────────────────────────────────
def generate_proteinmpnn_overview_figure(output_path):
    """ProteinMPNN: backbone graph -> message passing -> sequence."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(7, 3.5, 'ProteinMPNN: Inverse Folding via Message Passing',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=TEXT_COLOR)

    # Stage 1: Backbone graph (3D coordinates as nodes with edges)
    g_cx, g_cy = 2.0, 1.2
    n_nodes = 10
    np.random.seed(55)
    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    node_x = 0.9 * np.cos(node_angles) + g_cx
    node_y = 0.9 * np.sin(node_angles) + g_cy

    # Edges (k-nearest neighbors style)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist = np.sqrt((node_x[i] - node_x[j]) ** 2 +
                           (node_y[i] - node_y[j]) ** 2)
            if dist < 1.3:
                ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                        color='#b0bec5', lw=1, zorder=1)

    for i in range(n_nodes):
        c = plt.Circle((node_x[i], node_y[i]), 0.15,
                       facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
                       linewidth=1.5, zorder=3)
        ax.add_patch(c)

    ax.text(g_cx, -0.3, 'Backbone graph\n(3D coordinates)', ha='center',
            va='top', fontsize=9, fontweight='bold', color=EDGE_MAIN)

    # Arrow to model
    _draw_arrow(ax, 3.5, 1.2, 5.0, 1.2, color=ARROW_COLOR, lw=2)

    # Stage 2: Message passing (model box)
    _draw_box(ax, 6.5, 1.2, 2.5, 1.2, 'Message\nPassing\nGNN', BOX_ML,
              EDGE_ML, fontsize=10)

    # Arrow to output
    _draw_arrow(ax, 8.0, 1.2, 9.5, 1.2, color=ARROW_COLOR, lw=2)

    # Stage 3: Sequence output (colored residues with probabilities)
    seq_start_x = 10.0
    residues = ['M', 'K', 'T', 'L', 'V', 'I', 'A', 'F']
    residue_colors = [COLOR_HYDROPHOBIC, COLOR_POSITIVE, COLOR_POLAR,
                      COLOR_HYDROPHOBIC, COLOR_HYDROPHOBIC, COLOR_HYDROPHOBIC,
                      COLOR_HYDROPHOBIC, COLOR_HYDROPHOBIC]

    for i, (aa, rc) in enumerate(zip(residues, residue_colors)):
        x = seq_start_x + i * 0.55
        y = 1.2

        # Probability bar (varying heights to show design)
        bar_h = np.random.uniform(0.3, 1.0)
        bar = Rectangle((x - 0.2, y - 0.6), 0.4, bar_h,
                        facecolor=rc, alpha=0.3, zorder=2)
        ax.add_patch(bar)

        # Residue letter
        box = FancyBboxPatch((x - 0.2, y - 0.2), 0.4, 0.4,
                             boxstyle='round,pad=0.04',
                             facecolor='white', edgecolor=rc,
                             linewidth=1.5, zorder=4)
        ax.add_patch(box)
        ax.text(x, y, aa, ha='center', va='center', fontsize=9,
                fontweight='bold', color=rc, zorder=5)

    ax.text(seq_start_x + len(residues) * 0.55 / 2 - 0.27, -0.3,
            'Designed sequence\n(AA probabilities at each position)',
            ha='center', va='top', fontsize=9, fontweight='bold',
            color=COLOR_GREEN)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ProteinMPNN overview figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 11: AlphaFold Overview
# ──────────────────────────────────────────────
def generate_alphafold_overview_figure(output_path):
    """Two-panel: AlphaFold pipeline and outputs (pLDDT + PAE)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                              gridspec_kw={'wspace': 0.35, 'width_ratios': [1, 1.2]})

    # Panel 1: Pipeline
    ax = axes[0]
    ax.set_xlim(-2, 5)
    ax.set_ylim(-5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(1.5, 3.2, 'AlphaFold2 Pipeline', ha='center', va='center',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)

    bw, bh = 3.2, 0.7
    rounding = 0.12
    gap = 0.08
    CX = 1.5

    boxes = [
        (CX, 2.0, 'Sequence / MSA', BOX_MAIN, EDGE_MAIN),
        (CX, 0.5, 'Evoformer\n(attention)', BOX_ML, EDGE_ML),
        (CX, -1.0, 'Structure\nModule', BOX_ML, EDGE_ML),
        (CX, -2.5, 'Predicted\n3D structure', BOX_GREEN, EDGE_GREEN),
        (CX, -4.0, 'Confidence\nmetrics', BOX_WARM, EDGE_WARM),
    ]

    for cx, cy, label, fc, ec in boxes:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=9.5)

    for i in range(len(boxes) - 1):
        _draw_arrow(ax, CX, boxes[i][1] - bh / 2 - rounding - gap,
                    CX, boxes[i + 1][1] + bh / 2 + rounding + gap)

    # Recycling arrow
    loop_x = CX + bw / 2 + rounding + 0.3
    ax.plot([loop_x, loop_x], [boxes[2][1], boxes[1][1]],
            color=EDGE_ML, lw=1.5, zorder=2)
    ax.annotate('', xy=(CX + bw / 2 + rounding + gap, boxes[1][1]),
                xytext=(loop_x, boxes[1][1]),
                arrowprops=dict(arrowstyle='-|>', color=EDGE_ML,
                                lw=1.5, mutation_scale=12))
    ax.plot([CX + bw / 2 + rounding + gap, loop_x],
            [boxes[2][1], boxes[2][1]],
            color=EDGE_ML, lw=1.5, zorder=2)
    ax.text(loop_x + 0.15, (boxes[1][1] + boxes[2][1]) / 2,
            'recycle\n(3x)', ha='left', va='center', fontsize=8,
            color=EDGE_ML, fontweight='bold')

    # Panel 2: Outputs (pLDDT colored structure + PAE matrix)
    ax = axes[1]
    ax.set_xlim(-1, 10)
    ax.set_ylim(-5.5, 3.5)
    ax.axis('off')

    ax.text(4.5, 3.2, 'AlphaFold Outputs', ha='center', va='center',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)

    # pLDDT colored structure (schematic chain)
    ax.text(2.0, 2.2, 'pLDDT confidence', ha='center', va='center',
            fontsize=10, fontweight='bold', color=TEXT_COLOR)

    n_res = 20
    chain_angles = np.linspace(0, 3 * np.pi, n_res)
    chain_x = 0.7 * np.cos(chain_angles) + 2.0
    chain_y = 0.7 * np.sin(chain_angles) * 0.5 + 0.8

    # Color by pLDDT: blue=high, yellow=medium, red=low
    plddt_vals = np.array([90, 92, 88, 85, 70, 60, 55, 65, 80, 90,
                           92, 95, 93, 88, 75, 50, 45, 60, 85, 90])
    plddt_colors = plt.cm.RdYlBu(plddt_vals / 100)

    for i in range(n_res - 1):
        ax.plot([chain_x[i], chain_x[i + 1]], [chain_y[i], chain_y[i + 1]],
                color=plddt_colors[i], lw=4, zorder=3)

    # pLDDT colorbar
    import matplotlib.colors as mcolors
    sm = plt.cm.ScalarMappable(cmap='RdYlBu',
                               norm=plt.Normalize(vmin=0, vmax=100))
    cbar_ax = fig.add_axes([0.56, 0.55, 0.15, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('pLDDT', fontsize=8, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=7)

    # PAE matrix (heatmap)
    ax.text(7.0, 2.2, 'PAE matrix', ha='center', va='center',
            fontsize=10, fontweight='bold', color=TEXT_COLOR)

    # Generate a synthetic PAE matrix
    n_pae = 20
    np.random.seed(77)
    pae = np.random.uniform(0, 5, (n_pae, n_pae))
    # Make diagonal low (high confidence)
    for i in range(n_pae):
        for j in range(n_pae):
            dist = abs(i - j)
            pae[i, j] = min(30, dist * 1.5 + np.random.uniform(0, 3))
            if (i < 10 and j < 10) or (i >= 10 and j >= 10):
                pae[i, j] *= 0.3
    pae = np.minimum(pae, pae.T)

    pae_extent = [5.0, 9.0, -1.5, -5.5]
    ax.imshow(pae, cmap='Greens_r', extent=pae_extent, aspect='auto',
              zorder=2, vmin=0, vmax=15)
    ax.plot([5.0, 9.0, 9.0, 5.0, 5.0], [-1.5, -1.5, -5.5, -5.5, -1.5],
            color='#78909c', lw=1.5, zorder=3)

    ax.text(7.0, -0.8, 'Residue index', ha='center', va='top',
            fontsize=8, color='#607d8b')
    ax.text(4.5, -3.5, 'Residue\nindex', ha='center', va='center',
            fontsize=8, color='#607d8b', rotation=90)

    # pLDDT legend
    ax.text(2.0, -1.5, 'Blue = high confidence\nRed = low confidence',
            ha='center', va='top', fontsize=8, color='#607d8b',
            fontstyle='italic')

    # PAE legend
    ax.text(7.0, -6.0, 'Green = low error (confident)\n'
            'White = high error (uncertain)',
            ha='center', va='top', fontsize=8, color='#607d8b',
            fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved AlphaFold overview figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 12: Binder Example
# ──────────────────────────────────────────────
def generate_binder_example_figure(output_path):
    """Designed binder docked against target protein."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 4.2, 'Designed Protein Binder', ha='center', va='center',
            fontsize=14, fontweight='bold', color=TEXT_COLOR)

    # Target protein (larger, on right)
    theta = np.linspace(0, 2 * np.pi, 200)
    r_target = 2.8 + 0.3 * np.sin(3 * theta) - 0.6 * np.exp(
        -((theta - np.pi) ** 2) / 0.4)
    tx = r_target * np.cos(theta) + 2.0
    ty = r_target * np.sin(theta)
    ax.fill(tx, ty, facecolor=BOX_MAIN, edgecolor=EDGE_MAIN,
            linewidth=2, zorder=2)
    ax.text(3.5, 0, 'Target\nprotein', ha='center', va='center',
            fontsize=12, fontweight='bold', color=EDGE_MAIN, zorder=4)

    # Designed binder (smaller, on left, complementary shape)
    r_binder = 1.5 + 0.15 * np.cos(4 * theta) - 0.35 * np.exp(
        -((theta - 0) ** 2) / 0.35)
    bx = r_binder * np.cos(theta) - 2.0
    by = r_binder * np.sin(theta)
    ax.fill(bx, by, facecolor=BOX_GREEN, edgecolor=EDGE_GREEN,
            linewidth=2, zorder=2)
    ax.text(-2.5, 0, 'Designed\nbinder', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_GREEN, zorder=4)

    # Interface highlight
    interface_x = np.array([-0.3, -0.3, 0.3, 0.3])
    interface_y = np.array([-2.0, 2.0, 2.0, -2.0])
    ax.fill(interface_x, interface_y, facecolor=COLOR_WARM, alpha=0.12,
            zorder=1)

    # Hotspot residues at interface
    hotspots_binder = [(-0.7, 0.8), (-0.6, -0.3), (-0.8, -0.8)]
    hotspots_target = [(0.6, 0.5), (0.7, -0.5), (0.5, -1.2)]

    for hx, hy in hotspots_binder:
        c = plt.Circle((hx, hy), 0.2, facecolor=COLOR_GREEN, edgecolor='white',
                       linewidth=1, zorder=5, alpha=0.8)
        ax.add_patch(c)

    for hx, hy in hotspots_target:
        c = plt.Circle((hx, hy), 0.2, facecolor=EDGE_MAIN, edgecolor='white',
                       linewidth=1, zorder=5, alpha=0.8)
        ax.add_patch(c)

    # Complementary contact lines
    for (bx_h, by_h), (tx_h, ty_h) in zip(hotspots_binder, hotspots_target):
        ax.plot([bx_h, tx_h], [by_h, ty_h], color=COLOR_WARM, lw=1.5,
                linestyle=':', zorder=4, alpha=0.7)

    # Labels
    ax.annotate('Binding\ninterface', xy=(0, 0), xytext=(0, 3.2),
                fontsize=10, fontweight='bold', color=COLOR_WARM,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_WARM, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_WARM,
                          ec=EDGE_WARM, alpha=0.9, lw=0.8))

    ax.annotate('Hotspot\nresidues', xy=(-0.6, -0.3), xytext=(-4.5, -2.5),
                fontsize=10, fontweight='bold', color=COLOR_GREEN,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=COLOR_GREEN, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc=BOX_GREEN,
                          ec=EDGE_GREEN, alpha=0.9, lw=0.8))

    ax.text(0, -4.0, 'Shape complementarity drives tight binding\n'
            'at the designed interface',
            ha='center', va='top', fontsize=9.5, color='#607d8b',
            fontstyle='italic', linespacing=1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved binder example figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 13: Self-Consistency Workflow
# ──────────────────────────────────────────────
def generate_self_consistency_figure(output_path):
    """Horizontal two-row workflow: backbone -> ProteinMPNN -> sequence -> AlphaFold -> structure -> compare."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    ax.text(6, 4.2, 'Self-Consistency Protocol', ha='center', va='center',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)

    bw, bh = 2.2, 0.75

    # Top row (left to right): backbone -> ProteinMPNN -> sequence
    top_y = 3.0
    top_steps = [
        (1.5, top_y, 'Target\nbackbone', BOX_MAIN, EDGE_MAIN),
        (5.0, top_y, 'ProteinMPNN', BOX_WARM, EDGE_WARM),
        (8.5, top_y, 'Designed\nsequence', BOX_ML, EDGE_ML),
    ]

    # Bottom row (right to left): AlphaFold -> predicted structure -> compare
    bot_y = 1.0
    bot_steps = [
        (8.5, bot_y, 'AlphaFold', BOX_WARM, EDGE_WARM),
        (5.0, bot_y, 'Predicted\nstructure', BOX_GREEN, EDGE_GREEN),
        (1.5, bot_y, 'Compare\n(scRMSD)', BOX_MAIN, EDGE_MAIN),
    ]

    # Draw all boxes
    for cx, cy, label, fc, ec in top_steps + bot_steps:
        _draw_box(ax, cx, cy, bw, bh, label, fc, ec, fontsize=9.5)

    # Top row arrows (left to right)
    for i in range(len(top_steps) - 1):
        x1 = top_steps[i][0] + bw / 2 + 0.15
        x2 = top_steps[i + 1][0] - bw / 2 - 0.15
        _draw_arrow(ax, x1, top_y, x2, top_y, color=ARROW_COLOR, lw=1.5, ms=12)

    # Down arrow from sequence to AlphaFold
    _draw_arrow(ax, 8.5, top_y - bh / 2 - 0.15, 8.5, bot_y + bh / 2 + 0.15,
                color=ARROW_COLOR, lw=1.5, ms=12)

    # Bottom row arrows (right to left)
    for i in range(len(bot_steps) - 1):
        x1 = bot_steps[i][0] - bw / 2 - 0.15
        x2 = bot_steps[i + 1][0] + bw / 2 + 0.15
        _draw_arrow(ax, x1, bot_y, x2, bot_y, color=ARROW_COLOR, lw=1.5, ms=12)

    # Decision annotation
    ax.text(11.5, 2.0, 'Accept if\nscRMSD < 2 \u00c5',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=COLOR_GREEN,
            bbox=dict(boxstyle='round,pad=0.4', fc=BOX_GREEN,
                      ec=EDGE_GREEN, alpha=0.9, lw=1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved self-consistency figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 14: Design Funnel
# ──────────────────────────────────────────────
def generate_design_funnel_figure(output_path):
    """Funnel showing pipeline filtering from 10k backbones to 3-5 binders."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-0.3, 5.8)
    ax.axis('off')

    ax.text(5, 5.5, 'Protein Design Pipeline', ha='center', va='center',
            fontsize=13, fontweight='bold', color=TEXT_COLOR)

    stages = [
        ('10,000 backbones', 'RFDiffusion', 10000, BOX_ML, EDGE_ML),
        ('80,000 sequences', 'ProteinMPNN (8/backbone)', 80000, BOX_MAIN, EDGE_MAIN),
        ('~800 pass filters', 'Computational filters (~1%)', 800, BOX_WARM, EDGE_WARM),
        ('~20 ordered', 'Gene synthesis', 20, BOX_GREEN, EDGE_GREEN),
        ('~10 express', 'Protein expression', 10, BOX_GREEN, EDGE_GREEN),
        ('3-5 bind target', 'Experimental validation', 4, COLOR_GREEN, COLOR_GREEN),
    ]

    n_stages = len(stages)
    max_width = 9.0
    min_width = 1.8
    center_x = 5.0
    total_height = 4.8
    stage_height = total_height / n_stages

    for i, (label, method, count, fc, ec) in enumerate(stages):
        y_top = total_height - i * stage_height + 0.3
        y_bot = y_top - stage_height * 0.82

        # Width proportional to log of count
        frac = np.log10(count + 1) / np.log10(80001)
        w = min_width + (max_width - min_width) * frac

        # Trapezoid
        if i < n_stages - 1:
            frac_next = np.log10(stages[i + 1][2] + 1) / np.log10(80001)
            w_next = min_width + (max_width - min_width) * frac_next
        else:
            w_next = w * 0.7

        verts = [
            (center_x - w / 2, y_top),
            (center_x + w / 2, y_top),
            (center_x + w_next / 2, y_bot),
            (center_x - w_next / 2, y_bot),
        ]
        poly = Polygon(verts, facecolor=fc, edgecolor=ec,
                       linewidth=1.5, alpha=0.7, zorder=2)
        ax.add_patch(poly)

        y_mid = (y_top + y_bot) / 2
        ax.text(center_x, y_mid, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT_COLOR, zorder=4)

        ax.text(center_x + max_width / 2 + 0.5, y_mid,
                method, ha='left', va='center',
                fontsize=9, color=ec if ec != COLOR_GREEN else '#388e3c',
                fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved design funnel figure to {output_path}")


# ──────────────────────────────────────────────
if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    generate_amino_acids_figure(os.path.join(output_dir, 'pd_amino_acids.png'))
    generate_structure_levels_figure(os.path.join(output_dir, 'pd_protein_structure_levels.png'))
    generate_secondary_structure_figure(os.path.join(output_dir, 'pd_secondary_structure.png'))
    generate_hydrophobic_core_figure(os.path.join(output_dir, 'pd_hydrophobic_core.png'))
    generate_binding_interface_figure(os.path.join(output_dir, 'pd_binding_interface.png'))
    generate_antibody_structure_figure(os.path.join(output_dir, 'pd_antibody_structure.png'))
    generate_energy_landscape_figure(os.path.join(output_dir, 'pd_energy_landscape.png'))
    generate_design_problems_figure(os.path.join(output_dir, 'pd_design_problems.png'))
    generate_rfdiffusion_overview_figure(os.path.join(output_dir, 'pd_rfdiffusion_overview.png'))
    generate_proteinmpnn_overview_figure(os.path.join(output_dir, 'pd_proteinmpnn_overview.png'))
    generate_alphafold_overview_figure(os.path.join(output_dir, 'pd_alphafold_overview.png'))
    generate_binder_example_figure(os.path.join(output_dir, 'pd_binder_example.png'))
    generate_self_consistency_figure(os.path.join(output_dir, 'pd_self_consistency.png'))
    generate_design_funnel_figure(os.path.join(output_dir, 'pd_design_funnel.png'))

    print("\nDone! All 14 figures generated.")
