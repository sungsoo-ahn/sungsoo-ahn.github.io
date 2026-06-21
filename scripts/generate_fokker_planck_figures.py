"""
Generate figures for the Fokker-Planck equation blog post.

Three figures for the "Reading the Equation: Drift vs. Diffusion" section:
1. Drift advection — curve-based: flux arrows proportional to density, slab with net flux
2. Gaussian smoothing — before/after convolution with gain/loss shading, kernel above peak
3. Diffusion schematic — single density curve with highlighted slab, bidirectional flux arrows

Color convention:
  - Drift/advection figures: warm tones (amber/orange arrows and annotations)
  - Diffusion figures: cool tones (teal/blue arrows and annotations)
  - Density curves: slate blue throughout
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter1d

import blog_figure_style as bfs


# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
TEXT_COLOR = bfs.TEXT

# Density curves (consistent across all figures)
DENSITY_SLATE = bfs.PURPLE       # primary purple for density curves
DENSITY_FILL = bfs.PURPLE_LIGHT  # light purple fill under density
DENSITY_AFTER = bfs.TEAL         # teal for "after" curves

# Warm palette (drift / advection)
WARM_ARROW = bfs.AMBER           # amber for drift arrows
WARM_ANNOT = bfs.AMBER           # warm annotations
WARM_HIGHLIGHT = bfs.AMBER_LIGHT # light amber fill
WARM_BORDER = bfs.AMBER          # amber border

# Cool palette (diffusion)
COOL_ARROW = bfs.TEAL            # teal for diffusion arrows
COOL_ANNOT = bfs.TEAL            # teal for annotations
COOL_HIGHLIGHT = bfs.TEAL_LIGHT  # light teal fill
COOL_BORDER = bfs.TEAL           # teal border

# Gain/loss shading (Gaussian smoothing figure)
GAIN_COLOR = bfs.GREEN
GAIN_ALPHA = 0.45
LOSS_COLOR = bfs.RED
LOSS_ALPHA = 0.45

ANNOTATION_BG = 'white'
ANNOTATION_BG_ALPHA = 0.85

LABEL_FS = 13
SUBLABEL_FS = 10.5

bfs.use_blog_style()


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


def _gauss(x, mu, sigma, amplitude=1.0):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _two_bump(x):
    """Standard two-bump density used across figures for visual continuity."""
    return _gauss(x, 2.5, 0.55, 0.50) + _gauss(x, 5.5, 1.2, 0.28)


def _arrow(ax, start, end, lw=2.0, ms=15, color=bfs.TEXT, zorder=3):
    a = FancyArrowPatch(start, end, arrowstyle='-|>', color=color,
                        linewidth=lw, mutation_scale=ms, zorder=zorder)
    ax.add_patch(a)


def _eqbox(ax, x, y, text, fs=SUBLABEL_FS, ec=bfs.SPINE):
    ax.text(x, y, text, ha='center', va='top', fontsize=fs,
            color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=ec, lw=0.6))


# ──────────────────────────────────────────────
# Figure 1: Drift Advection (curve-based)
# ──────────────────────────────────────────────
def generate_drift_advection_figure(output_path):
    """
    Two-panel figure grounded on density curves:
    (a) Density with flux arrows whose thickness ~ local flux.
    (b) Density with highlighted slab showing incoming/outgoing flux.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5),
                                      gridspec_kw={'wspace': 0.32})

    X_MIN, X_MAX = -0.5, 9.0
    Y_MIN, Y_MAX = -0.14, 0.62

    x = np.linspace(X_MIN, X_MAX, 800)
    p = _two_bump(x)

    # ── Panel (a): Flux arrows along x-axis ──
    ax = ax_a
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    ax.fill_between(x, 0, p, color=DENSITY_FILL, alpha=0.5)
    ax.plot(x, p, color=DENSITY_SLATE, linewidth=2.2)

    # Drift field: constant rightward drift for simplicity
    f_val = 1.0  # drift velocity

    # Draw flux arrows at sampled positions
    arrow_xs = np.linspace(1.0, 8.0, 12)
    arrow_y = -0.04  # just below x-axis
    max_flux = np.max(p) * f_val

    for ax_pos in arrow_xs:
        p_local = np.interp(ax_pos, x, p)
        flux = f_val * p_local
        if flux < 0.02 * max_flux:
            continue
        lw = 1.0 + 3.5 * (flux / max_flux)
        ms = 10 + 8 * (flux / max_flux)
        arrow_len = 0.3 + 0.35 * (flux / max_flux)
        _arrow(ax, (ax_pos - arrow_len / 2, arrow_y),
               (ax_pos + arrow_len / 2, arrow_y),
               lw=lw, ms=ms, color=WARM_ARROW, zorder=4)

    ax.text(4.5, -0.10, r'flux $J = f \cdot p$', ha='center', va='top',
            fontsize=SUBLABEL_FS, color=WARM_ANNOT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=WARM_BORDER, lw=0.6))

    # Drift field label
    ax.annotate(r'$f(x,t) \;\rightarrow$', xy=(7.0, 0.15),
                fontsize=LABEL_FS, color=WARM_ANNOT, fontweight='bold',
                ha='center')

    ax.text(0.0, 0.60, '(a)', fontsize=13, fontweight='bold', color=TEXT_COLOR)

    # ── Panel (b): Slab with incoming/outgoing flux ──
    ax = ax_b
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    ax.fill_between(x, 0, p, color=DENSITY_FILL, alpha=0.5)
    ax.plot(x, p, color=DENSITY_SLATE, linewidth=2.2)

    # Place slab on ascending slope of first peak (flux converging → density increases)
    slab_x = 2.0
    slab_dx = 1.5
    p_left = np.interp(slab_x, x, p)
    p_right = np.interp(slab_x + slab_dx, x, p)

    # Highlighted slab
    slab_mask = (x >= slab_x) & (x <= slab_x + slab_dx)
    ax.fill_between(x[slab_mask], 0, p[slab_mask],
                    color=WARM_HIGHLIGHT, alpha=0.6, zorder=2)
    ax.axvline(slab_x, color=WARM_BORDER, linewidth=1.5,
               linestyle='--', ymin=0.02, ymax=0.72, zorder=2)
    ax.axvline(slab_x + slab_dx, color=WARM_BORDER, linewidth=1.5,
               linestyle='--', ymin=0.02, ymax=0.78, zorder=2)

    # Slab boundary labels — offset from dashed lines with white backgrounds
    _label_bbox = dict(boxstyle='round,pad=0.15', fc='white', alpha=0.9,
                       ec='none')
    ax.text(slab_x - 0.08, -0.04, r'$x$', ha='right', va='top',
            fontsize=10.5, color=WARM_BORDER, fontweight='bold',
            bbox=_label_bbox)
    ax.text(slab_x + slab_dx + 0.08, -0.04, r'$x\!+\!dx$', ha='left',
            va='top', fontsize=10.5, color=WARM_BORDER, fontweight='bold',
            bbox=_label_bbox)

    # Incoming flux (left boundary) — thick
    flux_in = f_val * p_left
    flux_out = f_val * p_right
    arrow_scale = 3.0

    in_y = p_left * 0.55
    in_len = arrow_scale * flux_in / max_flux
    lw_in = 1.5 + 3.0 * (flux_in / max_flux)
    _arrow(ax, (slab_x - in_len - 0.1, in_y), (slab_x - 0.05, in_y),
           lw=lw_in, ms=14, color=WARM_ARROW, zorder=4)
    ax.text(slab_x - in_len / 2 - 0.15, in_y + 0.06,
            r'$J(x)$', ha='center', va='bottom', fontsize=11,
            color=WARM_ANNOT, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.9,
                      ec='none'))

    # Outgoing flux (right boundary) — thicker (density is higher at x+dx)
    out_y = p_right * 0.55
    out_len = arrow_scale * flux_out / max_flux
    lw_out = 1.5 + 3.0 * (flux_out / max_flux)
    _arrow(ax, (slab_x + slab_dx + 0.05, out_y),
           (slab_x + slab_dx + out_len + 0.1, out_y),
           lw=lw_out, ms=14, color=WARM_ARROW, zorder=4)
    # J(x+dx) label — use annotate with leader line to avoid overlap
    ax.annotate(r'$J(x\!+\!dx)$',
                xy=(slab_x + slab_dx + out_len / 2 + 0.1, out_y),
                xytext=(slab_x + slab_dx + out_len + 0.9, out_y + 0.18),
                fontsize=11, color=WARM_ANNOT, fontweight='bold',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color=WARM_ANNOT, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.9,
                          ec='none'))

    # Annotation: on ascending slope, J_in < J_out means flux diverges,
    # but since density is increasing rightward, more flux exits → net depletion
    # Actually: on ascending slope with constant f, J increases with p,
    # so J(x+dx) > J(x) → flux diverges → dp/dt < 0... unless we pick the other side.
    # Let's put the annotation explaining the general equation.
    ax.text(6.8, 0.48,
            r'$\frac{\partial p}{\partial t} = -\frac{J(x{+}dx) - J(x)}{dx}$',
            ha='center', va='top', fontsize=SUBLABEL_FS, color=WARM_ANNOT,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=WARM_BORDER, lw=0.6))

    ax.text(0.0, 0.60, '(b)', fontsize=13, fontweight='bold', color=TEXT_COLOR)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved drift advection figure to {output_path}")


# ──────────────────────────────────────────────
# Figure 2: Gaussian Smoothing
# ──────────────────────────────────────────────
def generate_gaussian_smoothing_figure(output_path):
    """
    Two-bump density before and after convolution.

    The layout is intentionally schematic: the main point is that local
    averaging lowers sharp peaks and fills nearby valleys.
    """
    MU1, SIGMA1, AMP1 = 2.5, 0.55, 0.50
    MU2, SIGMA2, AMP2 = 5.5, 1.2, 0.28
    SMOOTH_SIGMA_PX = 25

    X_MIN, X_MAX = -0.5, 9.0
    Y_MIN, Y_MAX = -0.08, 0.72

    x = np.linspace(X_MIN, X_MAX, 800)
    p_before = _gauss(x, MU1, SIGMA1, AMP1) + _gauss(x, MU2, SIGMA2, AMP2)
    p_after = gaussian_filter1d(p_before, sigma=SMOOTH_SIGMA_PX)
    diff = p_after - p_before

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    # Gain / loss shading
    ax.fill_between(x, p_before, np.where(diff > 0, p_after, p_before),
                    where=(diff > 0), color=GAIN_COLOR, alpha=0.52, zorder=2)
    ax.fill_between(x, np.where(diff < 0, p_after, p_before), p_before,
                    where=(diff < 0), color=LOSS_COLOR, alpha=0.52, zorder=2)

    # Curves
    ax.plot(x, p_before, color=DENSITY_SLATE, linewidth=2.6, zorder=8)
    ax.plot(x, p_after, color=DENSITY_AFTER, linewidth=2.4,
            linestyle='--', zorder=9)
    bfs.curve_label(ax, 5.88, 0.27, r'$p_t(x)$', DENSITY_SLATE, size=10.4)
    bfs.curve_label(ax, 6.16, 0.21, 'smoothed', DENSITY_AFTER, size=10.2)

    ax.text(2.75, 0.58, 'sharp peak\nloses mass',
            ha='center', va='center', fontsize=10.2,
            color=bfs.RED, fontweight='semibold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      alpha=0.92, ec=bfs.RED_LIGHT, lw=0.8),
            zorder=30)
    ax.text(4.05, 0.08, 'nearby valley\ngains mass',
            ha='center', va='center', fontsize=10.2,
            color=bfs.GREEN, fontweight='semibold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      alpha=0.92, ec=bfs.GREEN_LIGHT, lw=0.8),
            zorder=30)

    ax.set_yticks([0.0, 0.25, 0.5])
    plt.tight_layout()
    bfs.save_figure(fig, output_path, dpi=260)


# ──────────────────────────────────────────────
# Figure 3: Diffusion — Taylor expansion intuition
# ──────────────────────────────────────────────
def generate_diffusion_schematic_figure(output_path):
    """
    Two-panel figure matching the Taylor expansion derivation:
    (a) Linear term cancels: on a slope, +ε and -ε kicks give opposite
        density changes that average to zero.
    (b) Quadratic term survives: at a peak, both neighbors are lower,
        so the average of neighbors < p(x) → density decreases.
    """
    MU, SIGMA, AMP = 3.5, 1.3, 0.42
    EPS = 1.0  # kick size ε for visualization

    x = np.linspace(0.0, 7.5, 600)
    p = _gauss(x, MU, SIGMA, AMP)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8),
                                      gridspec_kw={'wspace': 0.30})

    X_MIN, X_MAX = 0.0, 7.5
    Y_MIN, Y_MAX = -0.06, 0.56

    # ── Panel (a): Linear term cancels (point on slope) ──
    ax = ax_a
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    ax.fill_between(x, 0, p, color=DENSITY_FILL, alpha=0.4)
    ax.plot(x, p, color=DENSITY_SLATE, linewidth=2.2)

    # Pick a point on the ascending slope
    x0 = 2.3
    p0 = _gauss(x0, MU, SIGMA, AMP)
    p_left = _gauss(x0 - EPS, MU, SIGMA, AMP)
    p_right = _gauss(x0 + EPS, MU, SIGMA, AMP)

    # Mark x, x-ε, x+ε
    for xi, label, ha in [(x0, r'$x$', 'center'),
                          (x0 - EPS, r'$x - \epsilon$', 'center'),
                          (x0 + EPS, r'$x + \epsilon$', 'center')]:
        pi = _gauss(xi, MU, SIGMA, AMP)
        ax.plot(xi, pi, 'o', color=COOL_ARROW, markersize=7, zorder=5)
        ax.axvline(xi, color=COOL_BORDER, linewidth=0.8,
                   linestyle=':', alpha=0.5, zorder=1)
        ax.text(xi, -0.03, label, ha=ha, va='top', fontsize=9.5,
                color=COOL_ANNOT, fontweight='bold')

    # Show ±ε arrows from x
    arrow_y = p0 + 0.04
    _arrow(ax, (x0, arrow_y), (x0 - EPS, arrow_y),
           lw=2.0, ms=13, color=COOL_ARROW)
    _arrow(ax, (x0, arrow_y), (x0 + EPS, arrow_y),
           lw=2.0, ms=13, color=COOL_ARROW)
    ax.text(x0 - EPS / 2, arrow_y + 0.02, r'$-\epsilon$',
            ha='center', va='bottom', fontsize=10, color=COOL_ANNOT)
    ax.text(x0 + EPS / 2, arrow_y + 0.02, r'$+\epsilon$',
            ha='center', va='bottom', fontsize=10, color=COOL_ANNOT)

    # Show density differences: one positive, one negative
    # Δ_left = p(x-ε) - p(x), Δ_right = p(x+ε) - p(x)
    for xi, pi, sign_label, color in [
        (x0 - EPS, p_left, r'$-\epsilon \, p_t^{\prime}$', bfs.GREEN),
        (x0 + EPS, p_right, r'$+\epsilon \, p_t^{\prime}$', bfs.RED),
    ]:
        # Vertical bar from p0 to pi
        ax.plot([xi, xi], [p0, pi], linewidth=2.5, color=color,
                alpha=0.6, zorder=4)
        # Small horizontal ticks
        ax.plot([xi - 0.08, xi + 0.08], [p0, p0], linewidth=1.5,
                color=color, alpha=0.6, zorder=4)

    # Annotation
    ax.text(3.8, 0.50, 'slope contributions\ncancel out',
            ha='center', va='top', fontsize=9.5,
            color=COOL_ANNOT, fontstyle='italic', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=COOL_BORDER, lw=0.6))

    ax.text(0.2, 0.53, '(a)', fontsize=13, fontweight='bold',
            color=TEXT_COLOR)

    # ── Panel (b): Quadratic term survives (point at peak) ──
    ax = ax_b
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    ax.fill_between(x, 0, p, color=DENSITY_FILL, alpha=0.4)
    ax.plot(x, p, color=DENSITY_SLATE, linewidth=2.2)

    # Pick the peak
    x0 = MU
    p0 = AMP
    p_left = _gauss(x0 - EPS, MU, SIGMA, AMP)
    p_right = _gauss(x0 + EPS, MU, SIGMA, AMP)
    p_avg = (p_left + p_right) / 2

    # Mark x, x-ε, x+ε
    for xi, label, ha in [(x0, r'$x$', 'center'),
                          (x0 - EPS, r'$x - \epsilon$', 'center'),
                          (x0 + EPS, r'$x + \epsilon$', 'center')]:
        pi = _gauss(xi, MU, SIGMA, AMP)
        ax.plot(xi, pi, 'o', color=COOL_ARROW, markersize=7, zorder=5)
        ax.axvline(xi, color=COOL_BORDER, linewidth=0.8,
                   linestyle=':', alpha=0.5, zorder=1)
        ax.text(xi, -0.03, label, ha=ha, va='top', fontsize=9.5,
                color=COOL_ANNOT, fontweight='bold')

    # Show ±ε arrows from x
    arrow_y = p0 + 0.04
    _arrow(ax, (x0, arrow_y), (x0 - EPS, arrow_y),
           lw=2.0, ms=13, color=COOL_ARROW)
    _arrow(ax, (x0, arrow_y), (x0 + EPS, arrow_y),
           lw=2.0, ms=13, color=COOL_ARROW)
    ax.text(x0 - EPS / 2, arrow_y + 0.02, r'$-\epsilon$',
            ha='center', va='bottom', fontsize=10, color=COOL_ANNOT)
    ax.text(x0 + EPS / 2, arrow_y + 0.02, r'$+\epsilon$',
            ha='center', va='bottom', fontsize=10, color=COOL_ANNOT)

    # Dashed horizontal line at average of neighbors
    ax.plot([x0 - EPS - 0.3, x0 + EPS + 0.3], [p_avg, p_avg],
            linewidth=1.5, linestyle='--', color=bfs.RED, alpha=0.7,
            zorder=4)
    ax.text(x0 + EPS + 0.4, p_avg,
            r'avg of neighbors',
            ha='left', va='center', fontsize=9, color=bfs.RED,
            fontstyle='italic')

    # Show gap between p(x) and average
    ax.annotate('', xy=(x0 + 0.15, p_avg), xytext=(x0 + 0.15, p0),
                arrowprops=dict(arrowstyle='<->', color=bfs.RED,
                                lw=1.5))
    ax.text(x0 + 0.30, (p0 + p_avg) / 2,
            r"$\frac{\epsilon^2}{2}\,p_t'' < 0$",
            ha='left', va='center', fontsize=10, color=bfs.RED,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec=bfs.RED_LIGHT, lw=0.5))

    # Annotation
    ax.text(5.8, 0.50, 'both neighbors lower\n' + r'$\rightarrow$ density decreases',
            ha='center', va='top', fontsize=9.5,
            color=COOL_ANNOT, fontstyle='italic', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                      alpha=ANNOTATION_BG_ALPHA, ec=COOL_BORDER, lw=0.6))

    ax.text(0.2, 0.53, '(b)', fontsize=13, fontweight='bold',
            color=TEXT_COLOR)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved diffusion schematic figure to {output_path}")


# ──────────────────────────────────────────────
if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    for ext in ('svg', 'png'):
        generate_drift_advection_figure(
            os.path.join(output_dir, f'fp_drift_advection.{ext}'))
        generate_gaussian_smoothing_figure(
            os.path.join(output_dir, f'fp_gaussian_smoothing.{ext}'))
        generate_diffusion_schematic_figure(
            os.path.join(output_dir, f'fp_diffusion_schematic.{ext}'))

    print("Done!")
