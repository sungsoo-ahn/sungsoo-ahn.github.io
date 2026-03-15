"""
Generate figures for the path measures / non-equilibrium stat mech blog post.

Six figures:
1. Boltzmann overlap — two distributions with minimal overlap (Zwanzig problem)
2. Double-well protocol — time-dependent potential with trajectory
3. Forward trajectory ensemble — many paths from p_A spreading out of equilibrium
4. Work distribution — histogram with ΔF in the tail
5. Crooks intersection — P_F(W) and P_R(−W) crossing at ΔF
6. Unification diagram — four ML methods connected to path measure framework

Color palette matches ensemble/FP posts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

# ──────────────────────────────────────────────
# Color palette (consistent with other posts)
# ──────────────────────────────────────────────
TEXT_COLOR = '#263238'
BLUE = '#5b7fa5'
BLUE_LIGHT = '#dce8f4'
RED = '#c0503f'
RED_LIGHT = '#fce4ec'
AMBER = '#e8860c'
AMBER_LIGHT = '#fff3e0'
TEAL = '#1a8a7a'
TEAL_LIGHT = '#e0f2f1'
CORAL = '#e07a5f'
NEUTRAL = '#b0bec5'

LABEL_FS = 13
SUBLABEL_FS = 10.5
TICK_FS = 9

OUTPUT_DIR = 'assets/img/blog'


def _style_ax(ax, xlabel='', ylabel='', title=''):
    """Apply consistent axis styling."""
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, color=TEXT_COLOR)
    if title:
        ax.set_title(title, fontsize=LABEL_FS, fontweight='bold', color=TEXT_COLOR, pad=10)
    ax.tick_params(labelsize=TICK_FS, colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(NEUTRAL)
    ax.tick_params(colors=TEXT_COLOR)


# ──────────────────────────────────────────────
# Figure 1: Boltzmann overlap
# ──────────────────────────────────────────────
def fig_boltzmann_overlap():
    """Two Boltzmann distributions with minimal overlap — illustrates why FEP fails."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.linspace(-4, 10, 500)
    p_a = norm.pdf(x, loc=0, scale=1.0)
    p_b = norm.pdf(x, loc=6, scale=1.2)

    ax.fill_between(x, p_a, alpha=0.3, color=BLUE, label=r'$p_A(\mathbf{x})$')
    ax.plot(x, p_a, color=BLUE, linewidth=2)
    ax.fill_between(x, p_b, alpha=0.3, color=RED, label=r'$p_B(\mathbf{x})$')
    ax.plot(x, p_b, color=RED, linewidth=2)

    # Shade overlap region
    overlap = np.minimum(p_a, p_b)
    mask = overlap > 1e-4
    ax.fill_between(x[mask], overlap[mask], alpha=0.5, color=AMBER,
                    hatch='///', edgecolor=AMBER, label='Overlap (signal for FEP)')

    ax.annotate('Exponentially\nsmall overlap', xy=(3, 0.02), fontsize=SUBLABEL_FS,
                color=AMBER, ha='center', fontstyle='italic')

    ax.set_xlim(-4, 10)
    ax.set_ylim(0, None)
    _style_ax(ax, xlabel='Configuration space $\\mathbf{x}$', ylabel='Probability density')
    ax.legend(fontsize=SUBLABEL_FS, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_boltzmann_overlap.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_boltzmann_overlap.png')


# ──────────────────────────────────────────────
# Figure 2: Double-well protocol
# ──────────────────────────────────────────────
def fig_double_well_protocol():
    """Time-dependent double well with λ tilting the landscape, plus a trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    x = np.linspace(-2, 2, 300)

    lambdas = [0.0, 0.5, 1.0]
    labels = [r'$\lambda = 0$ (state A)', r'$\lambda = 0.5$', r'$\lambda = 1$ (state B)']
    colors = [BLUE, NEUTRAL, RED]

    for i, (lam, label, col) in enumerate(zip(lambdas, labels, colors)):
        ax = axes[i]
        # Double well: U(x) = (x^2 - 1)^2 - λ * x
        U = (x**2 - 1)**2 - 1.5 * lam * x
        ax.plot(x, U, color=col, linewidth=2.5)
        ax.fill_between(x, U, U.min() - 0.5, alpha=0.08, color=col)

        # Mark the occupied well
        if lam == 0.0:
            min_idx = np.argmin(U[:150])
            ax.plot(x[min_idx], U[min_idx] + 0.15, 'o', color=BLUE, markersize=10, zorder=5)
        elif lam == 1.0:
            min_idx = np.argmin(U[150:]) + 150
            ax.plot(x[min_idx], U[min_idx] + 0.15, 'o', color=RED, markersize=10, zorder=5)
        else:
            ax.plot(0.1, U[np.argmin(np.abs(x - 0.1))] + 0.15, 'o', color=TEXT_COLOR,
                    markersize=10, zorder=5, alpha=0.6)

        ax.set_ylim(-1.5, 3)
        _style_ax(ax, xlabel='$x$')
        ax.set_title(label, fontsize=SUBLABEL_FS, color=col, fontweight='bold')
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('$U(x, \\lambda)$', fontsize=LABEL_FS, color=TEXT_COLOR)

    # Draw arrows between panels
    for i in range(2):
        fig.text(0.365 + i * 0.33, 0.5, '→', fontsize=22, ha='center', va='center',
                 color=TEXT_COLOR, fontweight='bold')

    fig.suptitle('Protocol: tilting the double-well potential', fontsize=LABEL_FS,
                 fontweight='bold', color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_double_well_protocol.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_double_well_protocol.png')


# ──────────────────────────────────────────────
# Figure 2b: Alternating steps (work + MCMC)
# ──────────────────────────────────────────────
def fig_alternating_steps():
    """Alternating potential shifts (work) and particle moves (MCMC) — the mechanism of AIS."""
    rng = np.random.RandomState(42)
    n_particles = 30
    beta = 5.0
    tilt_strength = 3.0
    dt_lang = 0.005
    n_mcmc = 300

    def U(x, lam):
        return (x**2 - 1)**2 - tilt_strength * lam * x

    def dU_dx(x, lam):
        return 4 * x * (x**2 - 1) - tilt_strength * lam

    # Equilibrate particles at λ=0
    particles = -1.0 + 0.3 * rng.randn(n_particles)
    for _ in range(2000):
        f = -dU_dx(particles, 0.0)
        particles = particles + f * dt_lang + np.sqrt(2 * dt_lang / beta) * rng.randn(n_particles)

    # Pick 3 highlighted particles: one near left min, one near barrier, one outlier
    sorted_idx = np.argsort(particles)
    hi_indices = [sorted_idx[5], sorted_idx[n_particles // 2], sorted_idx[-5]]
    hi_colors = [RED, '#7b1fa2', TEAL]  # red, purple, teal
    hi_markers = ['o', 's', 'D']  # circle, square, diamond

    # Build sequence: (λ, particles, title, step_type)
    # Store particle snapshots for highlighted tracking
    states = []
    snapshots = []  # list of particle arrays at each state

    states.append((0.0, 'initial'))
    snapshots.append(particles.copy())

    for lam_new, lam_label in [(0.5, '0.5'), (1.0, '1.0')]:
        # Work step: shift λ, particles unchanged
        states.append((lam_new, 'work'))
        snapshots.append(particles.copy())
        # MCMC step: particles move under new λ
        for _ in range(n_mcmc):
            f = -dU_dx(particles, lam_new)
            particles = particles + f * dt_lang + np.sqrt(2 * dt_lang / beta) * rng.randn(n_particles)
        states.append((lam_new, 'mcmc'))
        snapshots.append(particles.copy())

    titles = [
        r'$\lambda=0$' + '\n(equilibrium)',
        r'$\lambda \to 0.5$' + '\n(potential shifts)',
        r'$\lambda=0.5$' + '\n(particles relax)',
        r'$\lambda \to 1.0$' + '\n(potential shifts)',
        r'$\lambda=1.0$' + '\n(particles relax)',
    ]

    # ── Plot ──
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=(2.7 * n_states, 3.8),
                              gridspec_kw={'wspace': 0.35})
    x_plot = np.linspace(-1.8, 1.8, 300)

    all_u = [U(x_plot, lam) for lam, _ in states]
    y_min = min(u.min() for u in all_u) - 0.5
    y_max = 4.0

    step_colors = {'initial': BLUE, 'work': AMBER, 'mcmc': TEAL}

    for i, ((lam, stype), pts, title) in enumerate(zip(states, snapshots, titles)):
        ax = axes[i]
        col = step_colors[stype]

        # Potential curve
        u = U(x_plot, lam)
        ax.plot(x_plot, u, color=col, linewidth=2.5)
        ax.fill_between(x_plot, u, y_min, alpha=0.06, color=col)

        # Highlighted particles on the curve
        pt_u = U(pts, lam)
        for j, (hi_idx, hi_col, hi_mk) in enumerate(zip(hi_indices, hi_colors, hi_markers)):
            ax.scatter(pts[hi_idx], pt_u[hi_idx], s=70, color='white',
                       edgecolors=hi_col, linewidths=2, zorder=6, marker=hi_mk)

        # In MCMC panels, draw arrows from previous position to current for highlighted
        if stype == 'mcmc' and i >= 1:
            prev_pts = snapshots[i - 1]
            prev_lam = states[i - 1][0]
            for hi_idx, hi_col in zip(hi_indices, hi_colors):
                x_old = prev_pts[hi_idx]
                x_new = pts[hi_idx]
                # Arrow at the bottom of the panel
                y_arrow = y_min + 0.35
                if abs(x_new - x_old) > 0.05:
                    ax.annotate('', xy=(x_new, y_arrow), xytext=(x_old, y_arrow),
                                arrowprops=dict(arrowstyle='->', color=hi_col,
                                                lw=1.8, shrinkA=2, shrinkB=2),
                                zorder=7)

        # In work panels, draw vertical dashed lines for highlighted (same x, curve moved)
        if stype == 'work' and i >= 1:
            prev_lam = states[i - 1][0]
            for hi_idx, hi_col in zip(hi_indices, hi_colors):
                x_pos = pts[hi_idx]
                y_old = U(x_pos, prev_lam)
                y_new = U(x_pos, lam)
                y_lo, y_hi = min(y_old, y_new), max(y_old, y_new)
                ax.plot([x_pos, x_pos], [y_lo, y_hi], color=hi_col,
                        linestyle=':', linewidth=1.5, alpha=0.6, zorder=3)

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=9, color=col, fontweight='bold', linespacing=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax.set_ylabel('$U(x, \\lambda)$', fontsize=LABEL_FS, color=TEXT_COLOR)

    # Arrows between panels
    for i in range(n_states - 1):
        stype_next = states[i + 1][1]
        label = 'shift $\\lambda$\n(work)' if stype_next == 'work' else 'MCMC\n(relax)'
        col = AMBER if stype_next == 'work' else TEAL
        x_mid = (axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2
        fig.text(x_mid, 0.48, '→', fontsize=18, ha='center', va='center',
                 color=col, fontweight='bold', transform=fig.transFigure)
        fig.text(x_mid, 0.38, label, fontsize=7.5, ha='center', va='center',
                 color=col, transform=fig.transFigure)

    fig.savefig(f'{OUTPUT_DIR}/pm_alternating_steps.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_alternating_steps.png')


# ──────────────────────────────────────────────
# Figure 2c: Work trajectories (multiple runs)
# ──────────────────────────────────────────────
def fig_work_trajectories():
    """Many runs of the same double-well protocol: lucky vs unlucky highlighted, rest faint."""
    rng = np.random.RandomState(12)
    n_steps = 500
    dt = 1.0 / n_steps
    t = np.linspace(0, 1, n_steps + 1)
    beta = 5.0
    tilt_strength = 3.0

    def dU_dx(x, lam):
        return 4 * x * (x**2 - 1) - tilt_strength * lam

    def dU_dlam(x, lam):
        return -tilt_strength * x

    # Run many trajectories
    n_traj_show = 30
    n_traj_hist = 2000
    all_work = np.zeros(n_traj_hist)
    saved_x = []
    saved_w = []

    for i in range(n_traj_hist):
        x = -1.0 + 0.2 * rng.randn()
        cumwork = 0.0
        traj_x = [x]
        traj_w = [0.0]
        for k in range(n_steps):
            lam_k = k / n_steps
            dlam = 1.0 / n_steps
            cumwork += dU_dlam(x, lam_k) * dlam
            force = -dU_dx(x, lam_k)
            x = x + force * dt + np.sqrt(2 * dt / beta) * rng.randn()
            if i < n_traj_show:
                traj_x.append(x)
                traj_w.append(cumwork)
        all_work[i] = cumwork
        if i < n_traj_show:
            saved_x.append(np.array(traj_x))
            saved_w.append(np.array(traj_w))

    # Compute ΔF via Jarzynski
    log_weights = -beta * all_work
    log_max = np.max(log_weights)
    delta_f_est = -(1.0 / beta) * (np.log(np.mean(np.exp(log_weights - log_max))) + log_max)

    # Find the luckiest and unluckiest among shown trajectories
    final_works = np.array([sw[-1] for sw in saved_w])
    lucky_idx = np.argmin(final_works)
    unlucky_idx = np.argmax(final_works)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel (a): Trajectories
    ax = axes[0]
    for i in range(n_traj_show):
        if i in (lucky_idx, unlucky_idx):
            continue
        ax.plot(t, saved_x[i], color=NEUTRAL, linewidth=0.6, alpha=0.35)
    ax.plot(t, saved_x[lucky_idx], color=TEAL, linewidth=2.2, label='Lucky (low $W$)', zorder=5)
    ax.plot(t, saved_x[unlucky_idx], color=RED, linewidth=2.2, label='Unlucky (high $W$)', zorder=5)
    ax.axhline(-1, color=BLUE, linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(1, color=RED, linestyle=':', alpha=0.3, linewidth=1)
    ax.text(0.02, -1.15, 'left well', fontsize=8, color=BLUE, va='top')
    ax.text(0.02, 1.15, 'right well', fontsize=8, color=RED, va='bottom')
    ax.legend(fontsize=8.5, loc='center left', framealpha=0.9)
    _style_ax(ax, xlabel='Time $t$', ylabel='Position $x(t)$', title='(a) Trajectories')

    # Panel (b): Cumulative work
    ax = axes[1]
    for i in range(n_traj_show):
        if i in (lucky_idx, unlucky_idx):
            continue
        ax.plot(t, saved_w[i], color=NEUTRAL, linewidth=0.6, alpha=0.35)
    ax.plot(t, saved_w[lucky_idx], color=TEAL, linewidth=2.2, zorder=5)
    ax.plot(t, saved_w[unlucky_idx], color=RED, linewidth=2.2, zorder=5)
    ax.axhline(delta_f_est, color=TEXT_COLOR, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.98, delta_f_est - 0.15, r'$\Delta F$', fontsize=SUBLABEL_FS, color=TEXT_COLOR,
            ha='right', va='top', fontweight='bold')
    _style_ax(ax, xlabel='Time $t$', ylabel='Cumulative work $W(t)$', title='(b) Work along each trajectory')

    # Panel (c): Work histogram
    ax = axes[2]
    ax.hist(all_work, bins=50, density=True, color=BLUE, alpha=0.4, edgecolor=BLUE, linewidth=0.5)
    ax.axvline(np.mean(all_work), color=AMBER, linewidth=2.5, linestyle='--',
               label=r'$\langle W \rangle$ (mean)')
    ax.axvline(delta_f_est, color=RED, linewidth=2.5, linestyle='-',
               label=r'$\Delta F$ (Jarzynski)')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, None)
    _style_ax(ax, xlabel='Work $W$', ylabel='Density', title='(c) Work histogram')

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_work_trajectories.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_work_trajectories.png')


# ──────────────────────────────────────────────
# Figure 3: Forward trajectory ensemble
# ──────────────────────────────────────────────
def fig_trajectory_ensemble():
    """Many forward trajectories from p_A, showing they don't end at p_B."""
    fig, ax = plt.subplots(figsize=(7, 4))

    rng = np.random.RandomState(42)
    n_traj = 40
    n_steps = 100
    t = np.linspace(0, 1, n_steps)

    # Start from p_A centered at -1, "target" p_B centered at +1
    x0s = rng.normal(-1, 0.3, n_traj)

    for i in range(n_traj):
        # Drift + noise: move toward +1 with noise
        x = np.zeros(n_steps)
        x[0] = x0s[i]
        for k in range(n_steps - 1):
            drift = 0.5 * (1.0 - x[k]) * (t[k+1] - t[k])
            noise = 0.3 * rng.randn() * np.sqrt(t[k+1] - t[k])
            x[k+1] = x[k] + drift + noise
        alpha = 0.15 + 0.1 * rng.rand()
        ax.plot(t, x, color=BLUE, alpha=alpha, linewidth=0.7)

    # Mark start and end distributions
    ax.axvline(0, color=BLUE, linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(1, color=RED, linestyle='--', alpha=0.5, linewidth=1)

    # Annotations
    ax.text(0.02, -1.8, '$p_A$\n(equilibrium)', fontsize=SUBLABEL_FS, color=BLUE,
            fontweight='bold', va='top')
    ax.text(0.98, -1.8, 'Endpoint\n(NOT $p_B$)', fontsize=SUBLABEL_FS, color=RED,
            fontweight='bold', va='top', ha='right')

    _style_ax(ax, xlabel='Time $t$', ylabel='Position $x(t)$',
              title='Forward trajectory ensemble (out of equilibrium)')
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_trajectory_ensemble.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_trajectory_ensemble.png')


# ──────────────────────────────────────────────
# Figure 4: Work distribution
# ──────────────────────────────────────────────
def fig_work_distribution():
    """Histogram of work values with ΔF in the tail — shows the variance problem."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    rng = np.random.RandomState(42)
    delta_f = 2.0

    # Work distribution: shifted and skewed (W ≥ ΔF on average, with tail below)
    # Use a shifted distribution: W ~ ΔF + |noise| + small_noise
    n_samples = 5000
    w_samples = delta_f + np.abs(rng.normal(0, 1.5, n_samples)) + rng.normal(0, 0.3, n_samples)

    ax.hist(w_samples, bins=60, density=True, color=BLUE, alpha=0.4, edgecolor=BLUE,
            linewidth=0.5, label='$P_F(W)$')

    # Mark ΔF
    ax.axvline(delta_f, color=RED, linewidth=2.5, linestyle='-', label=r'$\Delta F$', zorder=5)

    # Mark <W>
    mean_w = np.mean(w_samples)
    ax.axvline(mean_w, color=AMBER, linewidth=2, linestyle='--', label=r'$\langle W \rangle$',
               zorder=5)

    # Shade the "signal" region
    w_range = np.linspace(delta_f - 2, delta_f + 0.3, 100)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(w_samples)
    ax.fill_between(w_range, kde(w_range), alpha=0.4, color=RED,
                    label='Signal for Jarzynski\n(rare, high weight)')

    # Annotations
    ax.annotate(r'$\langle W \rangle \geq \Delta F$' + '\n(second law)',
                xy=(mean_w + 0.3, 0.25), fontsize=SUBLABEL_FS, color=AMBER)
    ax.annotate('Rare low-work\ntrajectories dominate\nthe exponential average',
                xy=(delta_f - 1.5, 0.08), fontsize=9, color=RED, fontstyle='italic',
                ha='center')

    _style_ax(ax, xlabel='Work $W$', ylabel='Density')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_work_distribution.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_work_distribution.png')


# ──────────────────────────────────────────────
# Figure 5: Crooks fluctuation theorem
# ──────────────────────────────────────────────
def fig_crooks_intersection():
    """P_F(W) and P_R(−W) crossing at W = ΔF."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    delta_f = 3.0
    sigma = 1.2
    w = np.linspace(-2, 8, 500)

    # Forward: centered above ΔF (dissipation)
    p_f = norm.pdf(w, loc=delta_f + 1.0, scale=sigma)
    # Reverse (−W): centered below ΔF
    p_r = norm.pdf(w, loc=delta_f - 1.0, scale=sigma)

    ax.plot(w, p_f, color=BLUE, linewidth=2.5, label='$P_F(W)$')
    ax.fill_between(w, p_f, alpha=0.15, color=BLUE)
    ax.plot(w, p_r, color=RED, linewidth=2.5, label='$P_R(-W)$')
    ax.fill_between(w, p_r, alpha=0.15, color=RED)

    # Mark intersection at ΔF
    ax.axvline(delta_f, color=TEXT_COLOR, linewidth=1.5, linestyle=':', zorder=5)
    cross_y = norm.pdf(delta_f, loc=delta_f + 1.0, scale=sigma)
    ax.plot(delta_f, cross_y, 'o', color=TEXT_COLOR, markersize=8, zorder=6)

    ax.annotate(r'$W = \Delta F$' + '\n(intersection)',
                xy=(delta_f, cross_y), xytext=(delta_f + 1.5, cross_y + 0.08),
                fontsize=SUBLABEL_FS, color=TEXT_COLOR, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=TEXT_COLOR, lw=1.5))

    _style_ax(ax, xlabel='Work $W$', ylabel='Density')
    ax.legend(fontsize=SUBLABEL_FS, framealpha=0.9)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_crooks_intersection.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_crooks_intersection.png')


# ──────────────────────────────────────────────
# Figure 6: Unification diagram
# ──────────────────────────────────────────────
def fig_unification_diagram():
    """Four ML methods all connected to the central path measure identity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central box
    center_box = FancyBboxPatch((-1.3, -0.45), 2.6, 0.9,
                                 boxstyle="round,pad=0.15",
                                 facecolor=AMBER_LIGHT, edgecolor=AMBER, linewidth=2.5)
    ax.add_patch(center_box)
    ax.text(0, 0.05, r'$\mathcal{P}_F / \mathcal{P}_R = e^{\beta(W - \Delta F)}$',
            fontsize=14, ha='center', va='center', color=TEXT_COLOR, fontweight='bold')

    # Four satellite boxes
    satellites = [
        (0, 2.0, 'AIS\n= Jarzynski', BLUE, BLUE_LIGHT),
        (2.3, 0, 'Flow Matching\n= Optimal Protocol', TEAL, TEAL_LIGHT),
        (0, -2.0, 'GFlowNet TB\n= Crooks', CORAL, RED_LIGHT),
        (-2.3, 0, 'Diffusion Models\n= Fwd/Rev SDE', RED, RED_LIGHT),
    ]

    for cx, cy, label, edge_col, fill_col in satellites:
        box = FancyBboxPatch((cx - 0.95, cy - 0.42), 1.9, 0.84,
                              boxstyle="round,pad=0.12",
                              facecolor=fill_col, edgecolor=edge_col, linewidth=2)
        ax.add_patch(box)
        ax.text(cx, cy, label, fontsize=10, ha='center', va='center',
                color=edge_col, fontweight='bold')

    # Draw connecting lines
    connections = [
        (0, 0.45, 0, 1.58),      # top
        (1.3, 0, 1.35, 0),       # right
        (0, -0.45, 0, -1.58),    # bottom
        (-1.3, 0, -1.35, 0),     # left
    ]
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=TEXT_COLOR, lw=1.8,
                                   connectionstyle='arc3,rad=0'))

    # Shared diagnostic label
    ax.text(0, -2.9, r'Shared diagnostic: $\langle W_{\mathrm{diss}} \rangle'
            r' = \frac{1}{\beta} D_{\mathrm{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$',
            fontsize=11, ha='center', va='center', color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NEUTRAL,
                      linewidth=1))

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/pm_unification_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('  ✓ pm_unification_diagram.png')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating path measures figures...')
    fig_boltzmann_overlap()
    fig_double_well_protocol()
    fig_alternating_steps()
    fig_work_trajectories()
    fig_trajectory_ensemble()
    fig_work_distribution()
    fig_crooks_intersection()
    fig_unification_diagram()
    print('Done.')
