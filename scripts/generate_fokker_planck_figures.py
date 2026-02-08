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


# ──────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────
TEXT_COLOR = '#263238'

# Density curves (consistent across all figures)
DENSITY_SLATE = '#5b7fa5'        # slate blue for density curves
DENSITY_FILL = '#dce8f4'         # light blue fill under density
DENSITY_AFTER = '#e07a5f'        # soft coral for "after" curves

# Warm palette (drift / advection)
WARM_ARROW = '#e8860c'           # amber for drift arrows
WARM_ANNOT = '#d4760a'           # warm orange for annotations
WARM_HIGHLIGHT = '#fff3e0'       # light amber fill
WARM_BORDER = '#e8a030'          # amber border

# Cool palette (diffusion)
COOL_ARROW = '#1a8a7a'           # teal for diffusion arrows
COOL_ANNOT = '#0d7d6c'          # teal for annotations
COOL_HIGHLIGHT = '#e0f2f1'      # light teal fill
COOL_BORDER = '#4db6ac'         # teal border

# Gain/loss shading (Gaussian smoothing figure)
GAIN_COLOR = '#a5d6a7'
GAIN_ALPHA = 0.45
LOSS_COLOR = '#ef9a9a'
LOSS_ALPHA = 0.45

ANNOTATION_BG = 'white'
ANNOTATION_BG_ALPHA = 0.85

LABEL_FS = 13
SUBLABEL_FS = 10.5


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


def _gauss(x, mu, sigma, amplitude=1.0):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _two_bump(x):
    """Standard two-bump density used across figures for visual continuity."""
    return _gauss(x, 2.5, 0.55, 0.50) + _gauss(x, 5.5, 1.2, 0.28)


def _arrow(ax, start, end, lw=2.0, ms=15, color='#37474f', zorder=3):
    a = FancyArrowPatch(start, end, arrowstyle='-|>', color=color,
                        linewidth=lw, mutation_scale=ms, zorder=zorder)
    ax.add_patch(a)


def _eqbox(ax, x, y, text, fs=SUBLABEL_FS, ec='#b0bec5'):
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
# Figure 2: Gaussian Smoothing (improved kernel inset)
# ──────────────────────────────────────────────
def generate_gaussian_smoothing_figure(output_path):
    """
    Two-bump density before and after convolution,
    red/green shading where density changed.
    Gaussian kernel inset positioned directly above the sharp peak.
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

    fig, ax = plt.subplots(figsize=(11, 5.0))
    _style_axis(ax, (X_MIN, X_MAX), (Y_MIN, Y_MAX),
                xlabel=r'$x$', ylabel=r'$p_t(x)$')

    # Gain / loss shading
    ax.fill_between(x, p_before, np.where(diff > 0, p_after, p_before),
                    where=(diff > 0), color=GAIN_COLOR, alpha=GAIN_ALPHA)
    ax.fill_between(x, np.where(diff < 0, p_after, p_before), p_before,
                    where=(diff < 0), color=LOSS_COLOR, alpha=LOSS_ALPHA)

    # Curves
    ax.plot(x, p_before, color=DENSITY_SLATE, linewidth=2.2,
            label=r'$p_t(x)$ (before)')
    ax.plot(x, p_after, color=DENSITY_AFTER, linewidth=2.0,
            linestyle='--', label=r'$p_t(x)$ convolved (after)')

    # Gaussian kernel inset — positioned directly above the sharp peak
    peak_x = MU1  # x-position of the sharp peak
    peak_y = AMP1  # height of the sharp peak

    # Draw kernel centered above the peak
    km = peak_x
    ks = 0.30
    ka = 0.06
    kernel_base_y = peak_y + 0.06  # just above the peak
    xk = np.linspace(km - 3 * ks, km + 3 * ks, 100)
    yk = kernel_base_y + _gauss(xk, km, ks, ka)

    ax.fill_between(xk, kernel_base_y, yk, color=COOL_ARROW, alpha=0.25)
    ax.plot(xk, yk, color=COOL_ARROW, linewidth=1.5)

    # Dashed vertical connector from peak to kernel
    ax.plot([peak_x, peak_x], [peak_y, kernel_base_y],
            color=COOL_BORDER, linewidth=1.0, linestyle=':', zorder=2)

    # Label above kernel
    ax.text(km, kernel_base_y + ka + 0.02,
            'average with neighbors\n' + r'$\rightarrow$ peak erodes',
            ha='center', va='bottom', fontsize=8.5,
            color=COOL_ANNOT, fontstyle='italic', linespacing=1.3)

    # Annotations for loss/gain
    pi = np.argmax(p_before)
    ax.annotate(r'$\Delta p < 0$',
                xy=(x[pi], p_before[pi] - 0.01),
                xytext=(x[pi] + 1.5, p_before[pi] + 0.08),
                fontsize=10.5, color='#c62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                          alpha=ANNOTATION_BG_ALPHA, ec='#ef9a9a', lw=0.6))

    vr = (x > 3.5) & (x < 4.5)
    vi = np.argmin(p_before[vr]) + np.argmax(vr)
    ax.annotate(r'$\Delta p > 0$',
                xy=(x[vi], p_after[vi]),
                xytext=(x[vi] - 0.5, -0.05),
                fontsize=10.5, color='#2e7d32', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.3', fc=ANNOTATION_BG,
                          alpha=ANNOTATION_BG_ALPHA, ec='#a5d6a7', lw=0.6))

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
              edgecolor='#b0bec5')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Gaussian smoothing figure to {output_path}")


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
        (x0 - EPS, p_left, r'$-\epsilon \, p_t^{\prime}$', '#2e7d32'),
        (x0 + EPS, p_right, r'$+\epsilon \, p_t^{\prime}$', '#c62828'),
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
            linewidth=1.5, linestyle='--', color='#c62828', alpha=0.7,
            zorder=4)
    ax.text(x0 + EPS + 0.4, p_avg,
            r'avg of neighbors',
            ha='left', va='center', fontsize=9, color='#c62828',
            fontstyle='italic')

    # Show gap between p(x) and average
    ax.annotate('', xy=(x0 + 0.15, p_avg), xytext=(x0 + 0.15, p0),
                arrowprops=dict(arrowstyle='<->', color='#c62828',
                                lw=1.5))
    ax.text(x0 + 0.30, (p0 + p_avg) / 2,
            r"$\frac{\epsilon^2}{2}\,p_t'' < 0$",
            ha='left', va='center', fontsize=10, color='#c62828',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.85, ec='#ef9a9a', lw=0.5))

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

    generate_drift_advection_figure(
        os.path.join(output_dir, 'fp_drift_advection.png'))
    generate_gaussian_smoothing_figure(
        os.path.join(output_dir, 'fp_gaussian_smoothing.png'))
    generate_diffusion_schematic_figure(
        os.path.join(output_dir, 'fp_diffusion_schematic.png'))

    print("Done!")
