"""
Generate figures for the ensembles/thermostats/barostats blog post.

Three figures:
1. Four ensembles — 2×2 grid showing NVE, NVT, NPT, μVT with system boundaries
2. Thermostat spectrum — three panels comparing velocity rescaling, Nosé-Hoover, Langevin
3. ML connection — simulation pipeline vs ML pipeline side-by-side

Color convention:
  - System/box: slate blue (consistent with FP post density curves)
  - Heat bath: warm tones (amber/orange)
  - Mechanical coupling (piston/volume): cool tones (teal)
  - Particle reservoir: soft coral
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ──────────────────────────────────────────────
# Color palette (extends FP post palette)
# ──────────────────────────────────────────────
TEXT_COLOR = '#263238'

# System box
SYSTEM_FILL = '#dce8f4'          # light blue fill
SYSTEM_BORDER = '#5b7fa5'        # slate blue border
SYSTEM_PARTICLE = '#5b7fa5'      # slate blue particles

# Heat bath (warm)
BATH_FILL = '#fff3e0'            # light amber
BATH_BORDER = '#e8a030'          # amber border
BATH_ARROW = '#e8860c'           # amber arrows
BATH_TEXT = '#d4760a'            # warm annotation

# Mechanical coupling (cool)
MECH_FILL = '#e0f2f1'           # light teal
MECH_BORDER = '#4db6ac'         # teal border
MECH_ARROW = '#1a8a7a'          # teal arrows
MECH_TEXT = '#0d7d6c'           # teal annotation

# Particle reservoir (coral)
RESERVOIR_FILL = '#fce4ec'       # light pink
RESERVOIR_BORDER = '#e07a5f'     # soft coral
RESERVOIR_ARROW = '#c0503f'      # darker coral arrows
RESERVOIR_TEXT = '#b7432f'       # coral annotation

# Neutral
NEUTRAL_BORDER = '#b0bec5'
ANNOTATION_BG = 'white'
ANNOTATION_BG_ALPHA = 0.85

LABEL_FS = 13
SUBLABEL_FS = 10.5

# ML figure colors
ML_BLUE = '#5b7fa5'
ML_ORANGE = '#e8860c'
ML_TEAL = '#1a8a7a'


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────
def _draw_particles(ax, cx, cy, w, h, n=12, seed=42):
    """Draw random particles inside a box region."""
    rng = np.random.RandomState(seed)
    margin = 0.08
    xs = cx - w/2 + margin + rng.rand(n) * (w - 2*margin)
    ys = cy - h/2 + margin + rng.rand(n) * (h - 2*margin)
    ax.scatter(xs, ys, s=45, c=SYSTEM_PARTICLE, zorder=5, edgecolors='white',
               linewidths=0.5)
    return xs, ys


def _draw_velocity_arrows(ax, xs, ys, seed=43, scale=0.06):
    """Draw small velocity arrows on particles."""
    rng = np.random.RandomState(seed)
    for x, y in zip(xs, ys):
        dx = (rng.rand() - 0.5) * scale * 2
        dy = (rng.rand() - 0.5) * scale * 2
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=SYSTEM_PARTICLE,
                                    lw=0.8, mutation_scale=6))


def _system_box(ax, cx, cy, w, h, label=None):
    """Draw the system box."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.02',
                          facecolor=SYSTEM_FILL, edgecolor=SYSTEM_BORDER,
                          linewidth=2.0, zorder=2)
    ax.add_patch(rect)
    if label:
        ax.text(cx, cy + h/2 + 0.04, label, ha='center', va='bottom',
                fontsize=SUBLABEL_FS, color=SYSTEM_BORDER, fontweight='bold')


def _heat_bath(ax, cx, cy, w, h, side='right'):
    """Draw a heat bath region adjacent to the system."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.02',
                          facecolor=BATH_FILL, edgecolor=BATH_BORDER,
                          linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(cx, cy, 'Heat\nbath', ha='center', va='center',
            fontsize=9, color=BATH_TEXT, fontstyle='italic', linespacing=1.2)


def _wavy_arrow(ax, x0, y0, x1, y1, color=BATH_ARROW, label=None):
    """Draw a wavy double-headed arrow representing energy exchange."""
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle='<->', color=color,
                            linewidth=2.2, mutation_scale=13,
                            connectionstyle='arc3,rad=0.2', zorder=6)
    ax.add_patch(arrow)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.06, label, ha='center', va='bottom',
                fontsize=8, color=color, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          alpha=0.9, ec='none'))


# ──────────────────────────────────────────────
# Figure 1: Four Ensembles (2×2 grid)
# ──────────────────────────────────────────────
def generate_four_ensembles_figure(output_path):
    """
    2×2 grid showing the four statistical ensembles:
    NVE (isolated), NVT (heat bath), NPT (heat bath + piston), μVT (reservoir).
    Each panel shows what's fixed and what fluctuates.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.25)

    for ax in axes.flat:
        ax.set_xlim(-0.15, 1.20)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect('equal')
        ax.axis('off')

    bw, bh = 0.45, 0.50  # system box size
    bcx, bcy = 0.40, 0.50  # system box center — shifted left for bath space

    # ── NVE (Microcanonical) ──
    ax = axes[0, 0]
    ax.text(0.50, 1.12, '(a) NVE — Microcanonical', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    # Thick insulating walls
    wall = FancyBboxPatch((bcx - bw/2 - 0.04, bcy - bh/2 - 0.04),
                          bw + 0.08, bh + 0.08,
                          boxstyle='round,pad=0.02',
                          facecolor='#eceff1', edgecolor='#78909c',
                          linewidth=3.0, zorder=1)
    ax.add_patch(wall)
    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=10, seed=42)
    _draw_velocity_arrows(ax, xs, ys, seed=43)

    ax.text(bcx, bcy - bh/2 - 0.10, 'Fixed: $N, V, E$',
            ha='center', va='top', fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=NEUTRAL_BORDER, lw=0.6))

    # Bath position (shared across NVT, NPT, μVT)
    bath_cx = bcx + bw/2 + 0.22  # more space between box and bath
    bath_w = 0.30

    # ── NVT (Canonical) ──
    ax = axes[0, 1]
    ax.text(0.50, 1.12, '(b) NVT — Canonical', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    _heat_bath(ax, bath_cx, bcy, bath_w, bh, side='right')
    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=10, seed=44)
    _draw_velocity_arrows(ax, xs, ys, seed=45)

    _wavy_arrow(ax, bcx + bw/2 + 0.03, bcy + 0.05,
                bath_cx - bath_w/2 - 0.01, bcy + 0.05,
                color=BATH_ARROW, label='$Q$')

    ax.text(bcx + 0.05, bcy - bh/2 - 0.10,
            'Fixed: $N, V, T$   Fluctuates: $E$',
            ha='center', va='top', fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=NEUTRAL_BORDER, lw=0.6))

    # ── NPT (Isothermal-isobaric) ──
    ax = axes[1, 0]
    ax.text(0.50, 1.12, '(c) NPT — Isothermal-Isobaric', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    _heat_bath(ax, bath_cx, bcy, bath_w, bh, side='right')
    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=10, seed=46)
    _draw_velocity_arrows(ax, xs, ys, seed=47)

    _wavy_arrow(ax, bcx + bw/2 + 0.03, bcy + 0.05,
                bath_cx - bath_w/2 - 0.01, bcy + 0.05,
                color=BATH_ARROW, label='$Q$')

    # Piston on top
    piston_y = bcy + bh/2
    piston_rect = Rectangle((bcx - bw/2, piston_y), bw, 0.06,
                             facecolor=MECH_FILL, edgecolor=MECH_BORDER,
                             linewidth=1.5, zorder=6)
    ax.add_patch(piston_rect)
    ax.annotate('', xy=(bcx, piston_y + 0.01),
                xytext=(bcx, piston_y + 0.18),
                arrowprops=dict(arrowstyle='->', color=MECH_ARROW,
                                lw=2.5, mutation_scale=14))
    ax.text(bcx + 0.14, piston_y + 0.14, '$P_{\\mathrm{ext}}$',
            ha='left', va='center', fontsize=11, color=MECH_TEXT,
            fontweight='bold')

    ax.text(bcx + 0.05, bcy - bh/2 - 0.10,
            'Fixed: $N, P, T$   Fluctuates: $V, E$',
            ha='center', va='top', fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=NEUTRAL_BORDER, lw=0.6))

    # ── μVT (Grand Canonical) ──
    ax = axes[1, 1]
    ax.text(0.50, 1.12, r'(d) $\mu$VT — Grand Canonical', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    # Heat bath (top-right)
    _heat_bath(ax, bath_cx, bcy + 0.15, bath_w, bh * 0.42, side='right')

    # Particle reservoir (bottom-right)
    res_rect = FancyBboxPatch((bath_cx - bath_w/2, bcy - 0.15 - bh*0.42/2),
                              bath_w, bh * 0.42,
                              boxstyle='round,pad=0.02',
                              facecolor=RESERVOIR_FILL,
                              edgecolor=RESERVOIR_BORDER,
                              linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(res_rect)
    ax.text(bath_cx, bcy - 0.15, 'Particle\nreservoir',
            ha='center', va='center', fontsize=9, color=RESERVOIR_TEXT,
            fontstyle='italic', linespacing=1.2)

    _system_box(ax, bcx, bcy, bw, bh)
    # Dashed right boundary to indicate permeability
    ax.plot([bcx + bw/2, bcx + bw/2],
            [bcy - bh/2 + 0.02, bcy + bh/2 - 0.02],
            color=RESERVOIR_BORDER, linewidth=2.0, linestyle=':', zorder=3)

    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=8, seed=48)
    _draw_velocity_arrows(ax, xs, ys, seed=49)

    # Particles near boundary (entering/leaving)
    ax.scatter([bcx + bw/2 + 0.04, bcx + bw/2 + 0.08],
               [bcy + 0.05, bcy - 0.10],
               s=35, c=RESERVOIR_BORDER, zorder=5, edgecolors='white',
               linewidths=0.5, alpha=0.7)

    # Energy exchange arrow
    _wavy_arrow(ax, bcx + bw/2 + 0.03, bcy + 0.18,
                bath_cx - bath_w/2 - 0.01, bcy + 0.18,
                color=BATH_ARROW, label='$Q$')

    # Particle exchange arrow
    ax.annotate('', xy=(bcx + bw/2 + 0.03, bcy - 0.12),
                xytext=(bath_cx - bath_w/2 - 0.01, bcy - 0.12),
                arrowprops=dict(arrowstyle='<->', color=RESERVOIR_ARROW,
                                lw=1.5, mutation_scale=10))
    ax.text((bcx + bw/2 + bath_cx - bath_w/2) / 2 + 0.01, bcy - 0.20, '$N$',
            ha='center', va='top', fontsize=9, color=RESERVOIR_TEXT,
            fontweight='bold')

    ax.text(bcx + 0.05, bcy - bh/2 - 0.10,
            r'Fixed: $\mu, V, T$   Fluctuates: $N, E$',
            ha='center', va='top', fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=NEUTRAL_BORDER, lw=0.6))

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved four ensembles figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 2: Thermostat Comparison
# ──────────────────────────────────────────────
def generate_thermostat_spectrum_figure(output_path):
    """
    Three-panel figure comparing thermostat methods:
    (a) Velocity rescaling — brute force
    (b) Nosé-Hoover — extended dynamics with friction
    (c) Langevin — stochastic (SDE)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.25)

    box_props = dict(boxstyle='round,pad=0.4', fc=ANNOTATION_BG,
                     alpha=ANNOTATION_BG_ALPHA, ec=NEUTRAL_BORDER, lw=0.8)

    for ax in axes:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.25, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')

    bw, bh = 0.55, 0.40
    bcx, bcy = 0.50, 0.45

    # ── (a) Velocity Rescaling ──
    ax = axes[0]
    ax.text(0.50, 1.02, '(a) Velocity Rescaling', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=8, seed=50)

    # Rescaling arrows (shrink/grow velocities)
    for i in range(min(4, len(xs))):
        dx = (np.random.RandomState(60+i).rand() - 0.5) * 0.08
        dy = (np.random.RandomState(70+i).rand() - 0.5) * 0.08
        ax.annotate('', xy=(xs[i] + dx*0.5, ys[i] + dy*0.5),
                    xytext=(xs[i] + dx, ys[i] + dy),
                    arrowprops=dict(arrowstyle='->', color='#c62828',
                                    lw=1.2, mutation_scale=8))

    # Label: scale factor
    ax.text(bcx, bcy + bh/2 + 0.08,
            r'$\mathbf{v} \leftarrow \lambda\,\mathbf{v}$',
            ha='center', va='bottom', fontsize=12, color='#c62828',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#ffebee',
                      alpha=0.9, ec='#ef9a9a', lw=0.6))

    ax.text(bcx, bcy - bh/2 - 0.06,
            r'$\lambda = \sqrt{T_{\mathrm{target}} / T_{\mathrm{current}}}$',
            ha='center', va='top', fontsize=9.5, color=TEXT_COLOR, bbox=box_props)

    ax.text(bcx, -0.20, 'Simple but wrong\ndistribution',
            ha='center', va='top', fontsize=9, color='#c62828',
            fontstyle='italic', linespacing=1.3)

    # ── (b) Nosé-Hoover ──
    ax = axes[1]
    ax.text(0.50, 1.02, r'(b) Nosé-Hoover', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=8, seed=52)
    _draw_velocity_arrows(ax, xs, ys, seed=53)

    # Friction variable ξ box
    xi_rect = FancyBboxPatch((bcx + bw/2 + 0.05, bcy - 0.12), 0.20, 0.24,
                             boxstyle='round,pad=0.02',
                             facecolor=BATH_FILL, edgecolor=BATH_BORDER,
                             linewidth=1.5, zorder=3)
    ax.add_patch(xi_rect)
    ax.text(bcx + bw/2 + 0.15, bcy, r'$\xi$', ha='center', va='center',
            fontsize=16, color=BATH_TEXT, fontweight='bold')

    # Coupling arrows
    _wavy_arrow(ax, bcx + bw/2 + 0.02, bcy + 0.05,
                bcx + bw/2 + 0.07, bcy + 0.05,
                color=BATH_ARROW)

    # Equations
    ax.text(bcx, bcy + bh/2 + 0.08,
            r'$m\dot{\mathbf{v}} = \mathbf{F} - \xi m\mathbf{v}$',
            ha='center', va='bottom', fontsize=10.5, color=BATH_TEXT,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=BATH_FILL,
                      alpha=0.9, ec=BATH_BORDER, lw=0.6))

    ax.text(bcx, bcy - bh/2 - 0.06,
            r'$\dot{\xi} \propto \mathrm{KE} - \frac{3}{2}Nk_BT$',
            ha='center', va='top', fontsize=9.5, color=TEXT_COLOR, bbox=box_props)

    ax.text(bcx, -0.20, 'Correct canonical\ndistribution',
            ha='center', va='top', fontsize=9, color=BATH_TEXT,
            fontstyle='italic', linespacing=1.3)

    # ── (c) Langevin ──
    ax = axes[2]
    ax.text(0.50, 1.02, '(c) Langevin', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=8, seed=54)

    # Stochastic kicks (wiggly arrows)
    rng = np.random.RandomState(80)
    for i in range(min(5, len(xs))):
        dx = (rng.rand() - 0.5) * 0.10
        dy = (rng.rand() - 0.5) * 0.10
        ax.annotate('', xy=(xs[i] + dx, ys[i] + dy), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color=MECH_ARROW,
                                    lw=1.0, mutation_scale=7,
                                    connectionstyle='arc3,rad=0.3'))

    # Equation
    ax.text(bcx, bcy + bh/2 + 0.08,
            r'$m\dot{\mathbf{v}} = \mathbf{F} - \gamma m\mathbf{v} + \sigma\boldsymbol{\eta}$',
            ha='center', va='bottom', fontsize=10.5, color=MECH_TEXT,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=MECH_FILL,
                      alpha=0.9, ec=MECH_BORDER, lw=0.6))

    ax.text(bcx, bcy - bh/2 - 0.06,
            r'$\sigma = \sqrt{2\gamma k_B T / m}$',
            ha='center', va='top', fontsize=9.5, color=TEXT_COLOR, bbox=box_props)

    ax.text(bcx, -0.20, 'Correct distribution,\nstochastic dynamics',
            ha='center', va='top', fontsize=9, color=MECH_TEXT,
            fontstyle='italic', linespacing=1.3)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved thermostat spectrum figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 3: ML Connection
# ──────────────────────────────────────────────
def generate_ml_connection_figure(output_path):
    """
    Two-column comparison:
    Left: Classical simulation pipeline (energy model → ensemble → MD/MC → observables)
    Right: ML pipeline (energy model → target distribution → generative model → samples)
    Shared layers highlighted.
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')

    # Column positions
    left_x = 0.25
    right_x = 0.75
    col_w = 0.38

    # Column headers
    ax.text(left_x, 0.98, 'Classical Simulation', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=ML_BLUE)
    ax.text(right_x, 0.98, 'ML Approach', ha='center', va='top',
            fontsize=LABEL_FS, fontweight='bold', color=ML_ORANGE)

    # Layer positions (top to bottom)
    layers_y = [0.82, 0.62, 0.42, 0.22]
    layer_h = 0.10
    layer_w = 0.36

    # Layer labels (shared)
    layer_names = ['Layer 1: Physics', 'Layer 2: Statistics',
                   'Layer 3: Algorithm', 'Output']

    # Left column content
    left_labels = [
        r'$U(\mathbf{r})$ — potential energy',
        r'$p \propto e^{-\beta H}$ — Boltzmann',
        'MD / MC sampling',
        r'$\langle A \rangle$ — observables'
    ]
    left_colors = [ML_BLUE, ML_BLUE, ML_BLUE, ML_BLUE]

    # Right column content
    right_labels = [
        r'$E_\theta(\mathbf{x})$ — learned energy',
        r'$p \propto e^{-E_\theta / kT}$ — same!',
        'Normalizing flow / diffusion',
        'Samples + log-likelihood'
    ]
    right_colors = [ML_ORANGE, ML_ORANGE, ML_ORANGE, ML_ORANGE]

    # Shared layer highlighting
    shared = [True, True, False, False]
    shared_fills = ['#e8eaf6', '#e8eaf6', None, None]

    for i, y in enumerate(layers_y):
        # Layer name on far left
        ax.text(-0.02, y, layer_names[i], ha='left', va='center',
                fontsize=9, color='#78909c', fontstyle='italic', rotation=0)

        # Shared background
        if shared[i]:
            bg = FancyBboxPatch((0.06, y - layer_h/2 - 0.015),
                                0.93, layer_h + 0.03,
                                boxstyle='round,pad=0.01',
                                facecolor=shared_fills[i], edgecolor='none',
                                alpha=0.5, zorder=0)
            ax.add_patch(bg)

        # Left box
        lbox = FancyBboxPatch((left_x - layer_w/2, y - layer_h/2),
                              layer_w, layer_h,
                              boxstyle='round,pad=0.02',
                              facecolor=SYSTEM_FILL if shared[i] else '#f5f5f5',
                              edgecolor=left_colors[i],
                              linewidth=1.5, zorder=2)
        ax.add_patch(lbox)
        ax.text(left_x, y, left_labels[i], ha='center', va='center',
                fontsize=10, color=TEXT_COLOR, zorder=3)

        # Right box
        rbox = FancyBboxPatch((right_x - layer_w/2, y - layer_h/2),
                              layer_w, layer_h,
                              boxstyle='round,pad=0.02',
                              facecolor=SYSTEM_FILL if shared[i] else '#fff8e1',
                              edgecolor=right_colors[i],
                              linewidth=1.5, zorder=2)
        ax.add_patch(rbox)
        ax.text(right_x, y, right_labels[i], ha='center', va='center',
                fontsize=10, color=TEXT_COLOR, zorder=3)

        # Arrows between layers
        if i < len(layers_y) - 1:
            for cx in [left_x, right_x]:
                color = ML_BLUE if cx == left_x else ML_ORANGE
                ax.annotate('', xy=(cx, layers_y[i+1] + layer_h/2 + 0.01),
                            xytext=(cx, y - layer_h/2 - 0.01),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.5, mutation_scale=12))

    # "Same" annotation between Layer 1 and 2
    for i in range(2):
        y = layers_y[i]
        ax.annotate('same', xy=(left_x + layer_w/2 + 0.02, y),
                    xytext=(right_x - layer_w/2 - 0.02, y),
                    ha='right', va='center', fontsize=9, color='#7986cb',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='<->', color='#7986cb',
                                    lw=1.2, connectionstyle='arc3,rad=0'))

    # "Different" annotation for Layer 3
    ax.annotate('ML replaces\nthis layer', xy=(left_x + layer_w/2 + 0.02, layers_y[2]),
                xytext=(right_x - layer_w/2 - 0.02, layers_y[2]),
                ha='right', va='center', fontsize=9, color=ML_ORANGE,
                fontweight='bold', linespacing=1.2,
                arrowprops=dict(arrowstyle='<->', color=ML_ORANGE,
                                lw=1.2, connectionstyle='arc3,rad=0'))

    # Key insight at bottom
    ax.text(0.50, 0.05,
            'ML advantage: normalizing flows compute $\\log Z$ (free energy) directly;\n'
            'MD/MC can only estimate expectations $\\langle A \\rangle$, not the partition function.',
            ha='center', va='center', fontsize=10, color=TEXT_COLOR,
            linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.4', fc='#fff8e1',
                      alpha=0.9, ec=ML_ORANGE, lw=0.8))

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ML connection figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 4: System–Environment Boundary
# ──────────────────────────────────────────────
def generate_system_boundary_figure(output_path):
    """
    Left: the full physical picture — a small region of atoms embedded in a
    vast environment (more atoms, container walls, etc.).
    Right: the modeling choice — we simulate the atoms explicitly and replace
    the environment with macroscopic boundary conditions (T, P, μ).
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.subplots_adjust(wspace=0.08)

    for ax in (ax_l, ax_r):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.10, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')

    # ── Left panel: physical reality ──
    ax = ax_l
    ax.text(0.50, 1.03, 'Physical Reality', ha='center', va='bottom',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    # Large environment region
    env = FancyBboxPatch((0.02, 0.02), 0.96, 0.88,
                         boxstyle='round,pad=0.02',
                         facecolor='#f5f5f5', edgecolor='#b0bec5',
                         linewidth=1.5, zorder=0)
    ax.add_patch(env)

    # Environment particles (many, faded)
    rng = np.random.RandomState(100)
    env_n = 80
    ex = 0.04 + rng.rand(env_n) * 0.92
    ey = 0.04 + rng.rand(env_n) * 0.84
    # Remove particles inside the system region
    cx, cy, r = 0.50, 0.47, 0.22
    mask = (ex - cx)**2 + (ey - cy)**2 > (r + 0.04)**2
    ax.scatter(ex[mask], ey[mask], s=20, c='#b0bec5', alpha=0.5,
               zorder=1, edgecolors='white', linewidths=0.3)

    # System region (circle, highlighted)
    circle = plt.Circle((cx, cy), r, facecolor=SYSTEM_FILL,
                         edgecolor=SYSTEM_BORDER, linewidth=2.5,
                         zorder=2, linestyle='-')
    ax.add_patch(circle)

    # System particles (explicit, vibrant)
    sys_n = 15
    rng2 = np.random.RandomState(42)
    angles = rng2.rand(sys_n) * 2 * np.pi
    radii = np.sqrt(rng2.rand(sys_n)) * (r - 0.04)
    sx = cx + radii * np.cos(angles)
    sy = cy + radii * np.sin(angles)
    ax.scatter(sx, sy, s=50, c=SYSTEM_PARTICLE, zorder=5,
               edgecolors='white', linewidths=0.5)

    # Labels
    ax.text(cx, cy - r - 0.06, 'system', ha='center', va='top',
            fontsize=11, color=SYSTEM_BORDER, fontweight='bold')
    ax.text(0.88, 0.85, 'environment', ha='center', va='top',
            fontsize=11, color='#78909c', fontstyle='italic')

    # ── Right panel: modeling choice ──
    ax = ax_r
    ax.text(0.50, 1.03, 'Modeling Choice', ha='center', va='bottom',
            fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR)

    # System box (larger, centered)
    bw, bh = 0.50, 0.55
    bcx, bcy = 0.50, 0.45
    _system_box(ax, bcx, bcy, bw, bh)
    xs, ys = _draw_particles(ax, bcx, bcy, bw, bh, n=15, seed=42)
    _draw_velocity_arrows(ax, xs, ys, seed=43, scale=0.04)

    ax.text(bcx, bcy - bh/2 - 0.04, 'simulate explicitly',
            ha='center', va='top', fontsize=10, color=SYSTEM_BORDER,
            fontweight='bold')

    # Boundary conditions around the box
    pad = 0.06
    bc_style = dict(ha='center', va='center', fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc=BATH_FILL,
                              alpha=0.9, ec=BATH_BORDER, lw=1.0))

    # Top: T
    ax.text(bcx, bcy + bh/2 + pad + 0.04, '$T$ = 300 K',
            color=BATH_TEXT, **bc_style)
    # Right: P
    ax.text(bcx + bw/2 + pad + 0.10, bcy + 0.08, '$P$ = 1 atm',
            color=MECH_TEXT,
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=MECH_FILL,
                      alpha=0.9, ec=MECH_BORDER, lw=1.0))
    # Left: μ
    ax.text(bcx - bw/2 - pad - 0.10, bcy + 0.08, r'$\mu$ = fixed',
            color=RESERVOIR_TEXT,
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc=RESERVOIR_FILL,
                      alpha=0.9, ec=RESERVOIR_BORDER, lw=1.0))

    # Bottom label
    ax.text(bcx, 0.00, 'environment replaced by\nboundary conditions',
            ha='center', va='top', fontsize=10, color='#78909c',
            fontstyle='italic', linespacing=1.3)

    # Arrow between panels
    fig.text(0.50, 0.50, r'$\Rightarrow$', ha='center', va='center',
             fontsize=28, color='#78909c')

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved system boundary figure to {output_path}")


# ──────────────────────────────────────────────
if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    generate_system_boundary_figure(
        os.path.join(output_dir, 'ens_system_boundary.png'))
    generate_four_ensembles_figure(
        os.path.join(output_dir, 'ens_four_ensembles.png'))
    generate_thermostat_spectrum_figure(
        os.path.join(output_dir, 'ens_thermostat_spectrum.png'))
    generate_ml_connection_figure(
        os.path.join(output_dir, 'ens_ml_connection.png'))

    print("Done!")
