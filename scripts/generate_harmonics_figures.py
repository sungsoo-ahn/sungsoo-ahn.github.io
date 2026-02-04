"""
Generate figures for circular and spherical harmonics with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import factorial

# Consistent color scheme
POSITIVE_COLOR = '#CC4444'  # Lighter red
NEGATIVE_COLOR = '#4444CC'  # Lighter blue


def associated_legendre(l, m, x):
    """
    Compute associated Legendre polynomial P_l^m(x).
    """
    # Use recurrence relations
    m = abs(m)
    pmm = np.ones_like(x)

    if m > 0:
        somx2 = np.sqrt((1 - x) * (1 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = -pmm * fact * somx2
            fact += 2.0

    if l == m:
        return pmm

    pmmp1 = x * (2 * m + 1) * pmm

    if l == m + 1:
        return pmmp1

    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def real_spherical_harmonic(l, m, theta, phi):
    """
    Compute real spherical harmonics.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi]
    """
    # Normalization factor
    abs_m = abs(m)
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) *
                   factorial(l - abs_m) / factorial(l + abs_m))

    # Associated Legendre polynomial
    P = associated_legendre(l, abs_m, np.cos(theta))

    if m > 0:
        Y = norm * np.sqrt(2) * P * np.cos(m * phi)
    elif m < 0:
        Y = norm * np.sqrt(2) * P * np.sin(abs_m * phi)
    else:
        Y = norm * P

    return Y


def generate_circular_harmonics_figure(output_path):
    """
    Generate circular harmonics figure for m = 0, 1, 2, 3.
    Shows both cos(mφ) and sin(mφ) components as polar plots.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={'projection': 'polar'})

    phi = np.linspace(0, 2*np.pi, 500)

    m_values = [0, 1, 2, 3]

    for col, m in enumerate(m_values):
        # Cosine component (top row)
        ax_cos = axes[0, col]
        if m == 0:
            r_cos = np.ones_like(phi)
        else:
            r_cos = np.cos(m * phi)

        # Plot positive and negative parts separately
        r_plot = np.abs(r_cos)
        colors = np.where(r_cos >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

        # Plot as filled regions
        for i in range(len(phi) - 1):
            ax_cos.fill_between([phi[i], phi[i+1]], 0, [r_plot[i], r_plot[i+1]],
                               color=colors[i], alpha=0.8)

        ax_cos.set_ylim(0, 1.3)
        ax_cos.set_yticklabels([])
        ax_cos.set_xticklabels([])
        ax_cos.grid(True, alpha=0.3)
        ax_cos.set_title(f'$\\cos({m}\\phi)$' if m > 0 else '$1$', fontsize=14, pad=10)

        # Sine component (bottom row)
        ax_sin = axes[1, col]
        if m == 0:
            r_sin = np.zeros_like(phi)
        else:
            r_sin = np.sin(m * phi)

        r_plot = np.abs(r_sin)
        colors = np.where(r_sin >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

        for i in range(len(phi) - 1):
            if r_plot[i] > 0.01:  # Avoid plotting near-zero values
                ax_sin.fill_between([phi[i], phi[i+1]], 0, [r_plot[i], r_plot[i+1]],
                                   color=colors[i], alpha=0.8)

        ax_sin.set_ylim(0, 1.3)
        ax_sin.set_yticklabels([])
        ax_sin.set_xticklabels([])
        ax_sin.grid(True, alpha=0.3)
        ax_sin.set_title(f'$\\sin({m}\\phi)$' if m > 0 else '$0$', fontsize=14, pad=10)

    # Add row labels
    fig.text(0.02, 0.72, 'Real', fontsize=14, va='center', rotation=90)
    fig.text(0.02, 0.28, 'Imaginary', fontsize=14, va='center', rotation=90)

    # Add column header
    fig.text(0.5, 0.98, 'Circular Harmonics: $e^{im\\phi} = \\cos(m\\phi) + i\\sin(m\\phi)$',
             fontsize=16, ha='center', va='top')

    # Add m labels
    for col, m in enumerate(m_values):
        fig.text(0.17 + col * 0.22, 0.02, f'$m = {m}$', fontsize=12, ha='center')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved circular harmonics figure to {output_path}")


def generate_spherical_harmonics_figure(output_path):
    """
    Generate spherical harmonics figure in pyramid layout similar to reference.
    """
    # Create figure with specific layout for pyramid
    fig = plt.figure(figsize=(14, 8))

    L_max = 3

    # Calculate positions for pyramid layout
    positions = []
    for l in range(L_max + 1):
        n_harmonics = 2 * l + 1
        row_width = n_harmonics
        start_col = (2 * L_max + 1 - row_width) / 2
        for m_idx, m in enumerate(range(-l, l + 1)):
            positions.append((l, m, start_col + m_idx))

    # Grid resolution for spherical harmonics
    n_theta = 100
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Create axes for each harmonic
    n_cols = 2 * L_max + 1
    n_rows = L_max + 1

    for l, m, col_pos in positions:
        # Calculate subplot position
        row = l
        col = col_pos

        # Create 3D axis with tighter vertical spacing
        left = (col + 0.1) / (n_cols + 0.2)
        bottom = 1 - (row + 1) / (n_rows) * 0.95
        width = 0.85 / (n_cols + 0.2)
        height = 0.95 / (n_rows)

        ax = fig.add_axes([left, bottom, width, height], projection='3d')

        # Compute spherical harmonic
        Y = real_spherical_harmonic(l, m, theta_grid, phi_grid)

        # Convert to Cartesian coordinates with radius = |Y|
        r = np.abs(Y)

        # Normalize for visualization
        if r.max() > 0:
            r = r / r.max() * 0.9

        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)

        # Color based on sign of Y
        colors = np.empty(Y.shape, dtype=object)
        colors_rgba = np.zeros((*Y.shape, 4))

        # Create custom colormap for positive/negative (bright colors)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i, j] >= 0:
                    colors_rgba[i, j] = [0.9, 0.3, 0.3, 1]  # Bright red
                else:
                    colors_rgba[i, j] = [0.3, 0.4, 0.9, 1]  # Bright blue

        # Draw coordinate grid (unit sphere wireframe)
        # Longitude lines
        for lng in np.linspace(0, 2*np.pi, 12, endpoint=False):
            theta_line = np.linspace(0, np.pi, 50)
            x_line = np.sin(theta_line) * np.cos(lng)
            y_line = np.sin(theta_line) * np.sin(lng)
            z_line = np.cos(theta_line)
            ax.plot(x_line * 0.98, y_line * 0.98, z_line * 0.98,
                   color='gray', alpha=0.3, linewidth=0.3)

        # Latitude lines
        for lat in np.linspace(0, np.pi, 7)[1:-1]:  # Skip poles
            phi_line = np.linspace(0, 2*np.pi, 50)
            x_line = np.sin(lat) * np.cos(phi_line)
            y_line = np.sin(lat) * np.sin(phi_line)
            z_line = np.cos(lat) * np.ones_like(phi_line)
            ax.plot(x_line * 0.98, y_line * 0.98, z_line * 0.98,
                   color='gray', alpha=0.3, linewidth=0.3)

        # Plot surface (shade=False to preserve colors)
        ax.plot_surface(x, y, z, facecolors=colors_rgba,
                       rstride=2, cstride=2, antialiased=True, shade=False)

        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        # Force equal aspect ratio in 3D
        ax.set_box_aspect([1, 1, 1])

        # Remove axes for cleaner look
        ax.set_axis_off()

        # Set viewing angle (higher elevation to reduce squashing)
        ax.view_init(elev=30, azim=45)

        # Add label
        ax.set_title(f'$Y_{{{l},{m}}}$', fontsize=11, pad=-5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved spherical harmonics figure to {output_path}")


def generate_circular_harmonics_simple(output_path):
    """
    Generate circular harmonics figure with Y_m labels to match spherical harmonics style.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), subplot_kw={'projection': 'polar'})

    phi = np.linspace(0, 2*np.pi, 500)

    m_values = [0, 1, 2, 3]

    for col, m in enumerate(m_values):
        ax = axes[col]

        if m == 0:
            r = np.ones_like(phi)
        else:
            r = np.cos(m * phi)

        # Plot positive and negative parts with different colors
        r_pos = np.where(r >= 0, np.abs(r), 0)
        r_neg = np.where(r < 0, np.abs(r), 0)

        ax.fill(phi, r_pos, color=POSITIVE_COLOR, alpha=0.8)
        ax.fill(phi, r_neg, color=NEGATIVE_COLOR, alpha=0.8)

        ax.set_ylim(0, 1.2)
        ax.set_yticklabels([])

        # Remove all angular tick labels to avoid overlapping with lobes
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.grid(True, alpha=0.4, color='gray', linestyle='-', linewidth=0.5)
        ax.set_title(f'$Y_{{{m}}}$', fontsize=14, pad=10)
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('gray')
        ax.spines['polar'].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved circular harmonics figure to {output_path}")


def generate_cg_tensor_product_figure(output_path):
    """
    Generate figure illustrating the Clebsch-Gordan tensor product.
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.25, 3.55)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Refined color palette ---
    # Block fills (soft pastels)
    color_l0 = '#a5d6a7'       # sage green
    color_l1 = '#90caf9'       # sky blue
    color_l2 = '#ef9a9a'       # soft coral
    color_tensor = '#cfd8dc'   # blue-gray
    color_cg = '#d1c4e9'       # soft lavender

    # Matching edge colors (mid-tone of each fill)
    edge_l0 = '#66bb6a'
    edge_l1 = '#42a5f5'
    edge_l2 = '#e57373'
    edge_neutral = '#90a4ae'   # for tensor & cg blocks
    arrow_color = '#78909c'

    # Grid
    grid_color = '#b0bec5'
    grid_lw = 0.4

    # Text colors (dark, readable versions of block colors)
    text_color = '#37474f'
    text_l0 = '#2e7d32'
    text_l1 = '#1565c0'
    text_l2 = '#c62828'
    text_cg = '#5e35b1'        # deep purple (matches lavender)
    text_tensor = '#546e7a'    # dark blue-gray
    text_muted = '#90a4ae'

    # Layout constants
    cy = 1.95
    arrow_len = 0.45
    arrow_gap = 0.15
    unit = 0.40
    lw = 1.0

    # --- Helpers ---
    def draw_arrow(x1, x2):
        ax.annotate('', xy=(x2, cy), xytext=(x1, cy),
                    arrowprops=dict(arrowstyle='-|>', color=arrow_color,
                                    lw=1.2, mutation_scale=10))

    def draw_gridded_col(x, n, color, edge):
        """Draw a 1-wide, n-tall column with grid lines."""
        h = n * unit
        r = plt.Rectangle((x, cy - h/2), unit, h, facecolor=color,
                          edgecolor=edge, linewidth=lw)
        ax.add_patch(r)
        for i in range(1, n):
            y = cy - h/2 + i*unit
            ax.plot([x, x + unit], [y, y], color='white',
                   linewidth=0.8, solid_capstyle='butt')
        return x, h

    def draw_gridded_square(x, n, color, edge):
        """Draw an n x n square with grid lines."""
        s = n * unit
        r = plt.Rectangle((x, cy - s/2), s, s, facecolor=color,
                          edgecolor=edge, linewidth=lw)
        ax.add_patch(r)
        for i in range(1, n):
            y = cy - s/2 + i*unit
            ax.plot([x, x + s], [y, y], color='white', linewidth=0.8,
                   solid_capstyle='butt')
            xv = x + i*unit
            ax.plot([xv, xv], [cy - s/2, cy + s/2], color='white',
                   linewidth=0.8, solid_capstyle='butt')
        return x, s

    # ========== SECTION 1: Input ==========
    x1 = 1.0
    draw_gridded_col(x1, 3, color_l1, edge_l1)
    ax.text(x1 + unit/2, cy + 3*unit/2 + 0.08, r'$\ell\!=\!1$',
            ha='center', va='bottom', fontsize=9, color=text_l1)

    ax.text(x1 + unit + 0.25, cy, r'$\otimes$', ha='center', va='center',
            fontsize=14, color=text_color)

    x2 = x1 + unit + 0.5
    draw_gridded_col(x2, 3, color_l1, edge_l1)
    ax.text(x2 + unit/2, cy + 3*unit/2 + 0.08, r'$\ell\!=\!1$',
            ha='center', va='bottom', fontsize=9, color=text_l1)

    # Arrow 1
    a1_s = x2 + unit + arrow_gap
    a1_e = a1_s + arrow_len
    draw_arrow(a1_s, a1_e)

    # ========== SECTION 2: Tensor Product (3x3 grid) ==========
    tp_x = a1_e + arrow_gap
    draw_gridded_square(tp_x, 3, color_tensor, edge_neutral)
    tp_s = 3 * unit

    # Arrow 2
    a2_s = tp_x + tp_s + arrow_gap
    a2_e = a2_s + arrow_len
    draw_arrow(a2_s, a2_e)

    # ========== SECTION 3: CG Transform ==========
    ms = 3 * unit
    dot_gap = 0.28

    cg_x = a2_e + arrow_gap

    draw_gridded_square(cg_x, 3, color_cg, edge_neutral)

    ax.text(cg_x + ms + dot_gap/2, cy, r'$\cdot$', ha='center', va='center',
            fontsize=16, fontweight='bold', color=text_color)

    m2_x = cg_x + ms + dot_gap
    draw_gridded_square(m2_x, 3, color_tensor, edge_neutral)

    ax.text(m2_x + ms + dot_gap/2, cy, r'$\cdot$', ha='center', va='center',
            fontsize=16, fontweight='bold', color=text_color)

    m3_x = m2_x + ms + dot_gap
    draw_gridded_square(m3_x, 3, color_cg, edge_neutral)

    # Arrow 3
    a3_s = m3_x + ms + arrow_gap
    a3_e = a3_s + arrow_len
    draw_arrow(a3_s, a3_e)

    # ========== SECTION 4: Direct Sum of Irreps ==========
    irrep_gap = 0.40

    ix0 = a3_e + arrow_gap
    draw_gridded_col(ix0, 1, color_l0, edge_l0)
    ax.text(ix0 + unit/2, cy + 1*unit/2 + 0.08, r'$\ell\!=\!0$',
            ha='center', va='bottom', fontsize=9, color=text_l0)

    ax.text(ix0 + unit + irrep_gap/2, cy, r'$\oplus$', ha='center', va='center',
            fontsize=14, color=text_color)

    ix1 = ix0 + unit + irrep_gap
    draw_gridded_col(ix1, 3, color_l1, edge_l1)
    ax.text(ix1 + unit/2, cy + 3*unit/2 + 0.08, r'$\ell\!=\!1$',
            ha='center', va='bottom', fontsize=9, color=text_l1)

    ax.text(ix1 + unit + irrep_gap/2, cy, r'$\oplus$', ha='center', va='center',
            fontsize=14, color=text_color)

    ix2 = ix1 + unit + irrep_gap
    draw_gridded_col(ix2, 5, color_l2, edge_l2)
    ax.text(ix2 + unit/2, cy + 5*unit/2 + 0.08, r'$\ell\!=\!2$',
            ha='center', va='bottom', fontsize=9, color=text_l2)

    # ========== Section headers ==========
    h1_y = 3.38

    ax.text((x1 + x2 + unit) / 2, h1_y, 'Input',
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    ax.text(tp_x + tp_s/2, h1_y, 'Tensor Product',
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    cob_center = (cg_x + m3_x + ms) / 2
    ax.text(cob_center, h1_y, 'CG Transform',
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    ds_center = (ix0 + ix2 + unit) / 2
    ax.text(ds_center, h1_y, 'Direct Sum of Irreps',
            ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    # ========== Equations below each element (color-matched) ==========
    eq_y = 0.60

    # --- Input ---
    ax.text(x1 + unit/2, eq_y, r'$\mathbf{x}^{(1)}$',
            ha='center', va='center', fontsize=11, color=text_l1)
    ax.text(x1 + unit + 0.25, eq_y, r'$\otimes$',
            ha='center', va='center', fontsize=11, color=text_color)
    ax.text(x2 + unit/2, eq_y, r'$\mathbf{y}^{(1)}$',
            ha='center', va='center', fontsize=11, color=text_l1)

    # --- CG Transform ---
    ax.text(cg_x + ms/2, eq_y, r'$C$',
            ha='center', va='center', fontsize=12, color=text_cg, fontweight='bold')
    ax.text(cg_x + ms + dot_gap/2, eq_y, r'$\cdot$',
            ha='center', va='center', fontsize=13, color=text_color)
    ax.text(m2_x + ms/2, eq_y, r'$(\cdot)$',
            ha='center', va='center', fontsize=11, color=text_tensor)
    ax.text(m2_x + ms + dot_gap/2, eq_y, r'$\cdot$',
            ha='center', va='center', fontsize=13, color=text_color)
    ax.text(m3_x + ms/2, eq_y, r'$C^{-1}$',
            ha='center', va='center', fontsize=12, color=text_cg, fontweight='bold')

    # --- Direct Sum ---
    ax.text(ix0 + unit/2, eq_y, r'$\mathbf{z}^{(0)}$',
            ha='center', va='center', fontsize=11, color=text_l0)
    ax.text(ix0 + unit + irrep_gap/2, eq_y, r'$\oplus$',
            ha='center', va='center', fontsize=10, color=text_color)
    ax.text(ix1 + unit/2, eq_y, r'$\mathbf{z}^{(1)}$',
            ha='center', va='center', fontsize=11, color=text_l1)
    ax.text(ix1 + unit + irrep_gap/2, eq_y, r'$\oplus$',
            ha='center', va='center', fontsize=10, color=text_color)
    ax.text(ix2 + unit/2, eq_y, r'$\mathbf{z}^{(2)}$',
            ha='center', va='center', fontsize=11, color=text_l2)

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"Saved CG tensor product figure to {output_path}")


def generate_ellipsoid_anisotropy_figure(output_path):
    """
    Generate figure showing how degree-2 spherical harmonics describe anisotropy.
    Four panels: sphere, prolate, oblate, tilted ellipsoid.
    """
    fig = plt.figure(figsize=(14, 4.5))

    # Panel configurations: title, subtitle, {m: coefficient}
    panels = [
        ('Sphere (isotropic)', r'$f^{(2)} = 0$', {}),
        ('Prolate (z-stretched)', r'$f^{(2)}_0 > 0$', {0: 0.35}),
        ('Oblate (z-squashed)', r'$f^{(2)}_0 < 0$', {0: -0.35}),
        ('Tilted stretch', r'$f^{(2)}_2 > 0$', {2: 0.35}),
    ]

    # Grid for surface
    n_theta = 80
    n_phi = 80
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    for idx, (title, subtitle, coeffs) in enumerate(panels):
        left = 0.02 + idx * 0.24
        ax = fig.add_axes([left, 0.08, 0.23, 0.78], projection='3d')

        # Compute r(theta, phi) = 1 + sum c_m * Y_2^m(theta, phi)
        r = np.ones_like(theta_grid)
        for m, c in coeffs.items():
            r = r + c * real_spherical_harmonic(2, m, theta_grid, phi_grid)

        # Cartesian coordinates for deformed surface
        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)

        # Reference wireframe sphere
        r_ref = 0.98
        for lng in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            theta_line = np.linspace(0, np.pi, 50)
            ax.plot(r_ref * np.sin(theta_line) * np.cos(lng),
                    r_ref * np.sin(theta_line) * np.sin(lng),
                    r_ref * np.cos(theta_line),
                    color='gray', alpha=0.3, linewidth=0.3)
        for lat in np.linspace(0, np.pi, 7)[1:-1]:
            phi_line = np.linspace(0, 2 * np.pi, 50)
            ax.plot(r_ref * np.sin(lat) * np.cos(phi_line),
                    r_ref * np.sin(lat) * np.sin(phi_line),
                    r_ref * np.cos(lat) * np.ones_like(phi_line),
                    color='gray', alpha=0.3, linewidth=0.3)

        # Compute deviation-based colors
        deviation = r - 1.0
        colors_rgba = np.zeros((*r.shape, 4))
        max_dev = 0.35  # normalize to known max coefficient

        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                d = deviation[i, j] / max_dev  # in [-1, 1]
                d = np.clip(d, -1, 1)
                if d > 0:
                    # Blend from light gray to red
                    colors_rgba[i, j] = [
                        0.85 + 0.15 * d,   # R: 0.85 -> 1.0
                        0.85 - 0.55 * d,   # G: 0.85 -> 0.30
                        0.85 - 0.55 * d,   # B: 0.85 -> 0.30
                        1.0
                    ]
                elif d < 0:
                    # Blend from light gray to blue
                    ad = -d
                    colors_rgba[i, j] = [
                        0.85 - 0.55 * ad,  # R: 0.85 -> 0.30
                        0.85 - 0.55 * ad,  # G: 0.85 -> 0.30
                        0.85 + 0.15 * ad,  # B: 0.85 -> 1.0
                        1.0
                    ]
                else:
                    colors_rgba[i, j] = [0.85, 0.85, 0.85, 1.0]

        # Plot surface
        ax.plot_surface(x, y, z, facecolors=colors_rgba,
                        rstride=2, cstride=2, antialiased=True, shade=False)

        # Styling
        max_range = 1.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.view_init(elev=25, azim=45)

        # Title and subtitle
        ax.set_title(title, fontsize=11, pad=5)
        ax.text2D(0.5, -0.02, subtitle, transform=ax.transAxes,
                  ha='center', va='top', fontsize=10, color='#555555')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ellipsoid anisotropy figure to {output_path}")


def generate_cg_network_figure(output_path):
    """
    Generate figure showing CG tensor products as neural network layers.
    Single row with two panels side by side: (a) compact with layer blocks,
    (b) expanded with CG connection lines.
    """
    from matplotlib.patches import FancyBboxPatch, Polygon

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.2),
                                      gridspec_kw={'wspace': 0.08})

    # --- Color palette ---
    type_colors = {
        0: ('#c5cae9', '#7986cb'),   # indigo
        1: ('#ffcdd2', '#e57373'),   # pink
        2: ('#c8e6c9', '#66bb6a'),   # green
        3: ('#e1bee7', '#ab47bc'),   # purple
    }
    color_layer = '#b2ebf2'
    edge_layer = '#00acc1'
    text_color = '#37474f'
    text_desc = '#546e7a'
    conn_color = '#78909c'

    lw = 1.2
    rounding = 0.06

    # Shared geometry
    type_labels = [0, 1, 2, 3]
    box_w = 0.85
    box_h = 0.48
    type_gap = 0.14
    stack_h = len(type_labels) * box_h + (len(type_labels) - 1) * type_gap
    col_labels = [None, None, None]

    def draw_box(ax, x, y, w, h, fc, ec, zorder=2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f'round,pad={rounding}',
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))

    def type_cy(t_idx, cy):
        return cy + stack_h/2 - (t_idx + 0.5) * box_h - t_idx * type_gap

    def draw_type_stack(ax, x, cy, label, fontsize=8.5):
        for t_idx, t in enumerate(type_labels):
            y = type_cy(t_idx, cy) - box_h/2
            fill, edge = type_colors[t]
            draw_box(ax, x, y, box_w, box_h, fill, edge)
            ax.text(x + box_w/2, y + box_h/2, f'Type-{t}',
                    ha='center', va='center', fontsize=fontsize,
                    fontweight='bold', color=text_color)
        if label:
            ax.text(x + box_w/2, cy - stack_h/2 - 0.30, label,
                    ha='center', va='center', fontsize=8,
                    color=text_desc, linespacing=1.3)

    def draw_bg_arrow(ax, x1, x2, cy, half_h):
        """Draw a proper polygon arrow shape as background."""
        head_len = 0.45
        shaft_h = half_h * 0.35   # thin shaft
        head_h = half_h * 0.7     # moderate arrowhead
        x_head = x2 - head_len
        verts = [
            (x1, cy - shaft_h),          # bottom-left of shaft
            (x_head, cy - shaft_h),      # bottom-right of shaft
            (x_head, cy - head_h),       # bottom of arrowhead
            (x2, cy),                     # tip
            (x_head, cy + head_h),       # top of arrowhead
            (x_head, cy + shaft_h),      # top-right of shaft
            (x1, cy + shaft_h),          # top-left of shaft
        ]
        ax.add_patch(Polygon(verts, closed=True,
                             facecolor='#eeeeee', edgecolor='none',
                             zorder=0))

    # ============================================================
    # PANEL A: Compact view with layer blocks
    # ============================================================
    ax = ax_a
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-0.2, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')

    cy = 1.9
    layer_w = 0.55
    layer_gap = 0.25

    pair_w = box_w + layer_gap + layer_w + layer_gap
    total_a = pair_w * 2 + box_w
    x0_a = (7 - total_a) / 2

    x_stacks = [x0_a, x0_a + pair_w, x0_a + pair_w * 2]
    x_layers = [x_stacks[0] + box_w + layer_gap,
                x_stacks[1] + box_w + layer_gap]

    draw_bg_arrow(ax, x_stacks[0] - 0.15, x_stacks[2] + box_w + 0.7,
                  cy, stack_h/2 + 0.15)

    for i, x in enumerate(x_stacks):
        draw_type_stack(ax, x, cy, col_labels[i])

    layer_h = stack_h + 0.35
    for i, x in enumerate(x_layers):
        y = cy - layer_h/2
        draw_box(ax, x, y, layer_w, layer_h, color_layer, edge_layer, zorder=1)
        ax.text(x + layer_w/2, cy - layer_h/2 - 0.18, f'Layer {i+1}',
                ha='center', va='top', fontsize=7.5,
                fontweight='bold', color='#00838f')

    ax.text(x_stacks[0] - 0.2, cy + stack_h/2 + 0.30, '(a)',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color=text_color)

    # ============================================================
    # PANEL B: Expanded view with CG connection lines
    # ============================================================
    ax = ax_b
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(-0.2, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')

    col_sep = 2.25
    total_b = 2 * col_sep + box_w
    x0_b = (7 - total_b) / 2

    x_stacks_b = [x0_b, x0_b + col_sep, x0_b + 2 * col_sep]

    draw_bg_arrow(ax, x_stacks_b[0] - 0.15, x_stacks_b[2] + box_w + 0.7,
                  cy, stack_h/2 + 0.15)

    for i, x in enumerate(x_stacks_b):
        draw_type_stack(ax, x, cy, col_labels[i])

    # Connection lines
    connections = [
        ([0, 1], 1),
        ([1], 1),
        ([1, 2], 2),
        ([2, 3], 3),
    ]

    for pair_idx in range(2):
        x_src_r = x_stacks_b[pair_idx] + box_w
        x_dst_l = x_stacks_b[pair_idx + 1]
        x_mid = (x_src_r + x_dst_l) / 2

        for src_types, dst_type in connections:
            y_dst = type_cy(dst_type, cy)
            for st in src_types:
                y_src = type_cy(st, cy)
                rad = 0.0 if y_src == y_dst else 0.1
                ax.annotate('',
                            xy=(x_dst_l - 0.02, y_dst),
                            xytext=(x_src_r + 0.02, y_src),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=conn_color,
                                            lw=0.9,
                                            mutation_scale=8,
                                            connectionstyle=f'arc3,rad={rad}'),
                            zorder=1)

        ax.text(x_mid, cy + stack_h/2 + 0.25,
                'CG tensor products',
                ha='center', va='center', fontsize=7.5,
                color='#d32f2f')

    ax.text(x_stacks_b[0] - 0.2, cy + stack_h/2 + 0.30, '(b)',
            ha='left', va='center', fontsize=11,
            fontweight='bold', color=text_color)

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"Saved CG network figure to {output_path}")


def generate_architecture_figure(output_path):
    """
    Generate high-level architecture diagram showing message-passing and
    spherical equivariant layers interleaved.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(1, 1, figsize=(14, 3.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.2, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Color palette (matching CG figure style) ---
    color_mp = '#90caf9'        # sky blue — message passing
    color_se = '#d1c4e9'        # soft lavender — spherical equivariant
    color_input = '#c8e6c9'     # light green — input
    color_output = '#ffccbc'    # light peach — output

    edge_mp = '#42a5f5'
    edge_se = '#9575cd'
    edge_input = '#66bb6a'
    edge_output = '#ff8a65'

    text_color = '#37474f'
    text_mp = '#1565c0'
    text_se = '#5e35b1'
    text_input = '#2e7d32'
    text_output = '#bf360c'
    arrow_color = '#546e7a'
    text_desc = '#546e7a'       # darker grey for descriptions

    lw = 1.2
    rounding = 0.12

    # --- Helpers ---
    def rounded_box(x, y, w, h, facecolor, edgecolor, zorder=2):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f'round,pad={rounding}',
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=lw, zorder=zorder)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, x2, y=None, color=arrow_color):
        """Draw a horizontal arrow with visible head."""
        if y is None:
            y = cy
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=1.5, mutation_scale=12,
                                    shrinkA=0, shrinkB=0),
                    zorder=3)

    # --- Layout (uniform gap between all elements) ---
    cy = 2.0
    box_h = 1.3
    box_w_io = 1.3       # input/output width
    box_w = 1.6          # layer block width
    gap = 0.5            # uniform gap between adjacent elements
    dots_w = 0.6         # width reserved for the dots region
    arrow_pad = 0.06     # small pad so arrow doesn't touch box edge

    # Compute total width and center it
    total = (box_w_io + gap + box_w + gap + box_w + gap
             + dots_w + gap + box_w + gap + box_w + gap + box_w_io)
    x0 = (14 - total) / 2  # left margin to center everything

    x_input = x0
    x_b1_mp = x_input + box_w_io + gap
    x_b1_se = x_b1_mp + box_w + gap
    x_dots  = x_b1_se + box_w + gap          # start of dots region
    x_b2_mp = x_dots + dots_w + gap
    x_b2_se = x_b2_mp + box_w + gap
    x_output = x_b2_se + box_w + gap

    # --- Helper to draw an MP or SE block ---
    def draw_mp_block(x):
        rounded_box(x, cy - box_h/2, box_w, box_h, color_mp, edge_mp)
        ax.text(x + box_w/2, cy + 0.22, 'Message',
                ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=text_mp)
        ax.text(x + box_w/2, cy - 0.08, 'Passing',
                ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=text_mp)
        ax.text(x + box_w/2, cy - 0.42,
                'Aggregate from\nneighbors',
                ha='center', va='center', fontsize=7.5,
                color=text_desc, linespacing=1.4)

    def draw_se_block(x):
        rounded_box(x, cy - box_h/2, box_w, box_h, color_se, edge_se)
        ax.text(x + box_w/2, cy + 0.22, 'Spherical',
                ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=text_se)
        ax.text(x + box_w/2, cy - 0.08, 'Equivariant',
                ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=text_se)
        ax.text(x + box_w/2, cy - 0.42,
                'CG tensor products,\nnonlinearities',
                ha='center', va='center', fontsize=7.5,
                color=text_desc, linespacing=1.4)

    # --- Draw all blocks ---
    # Input
    rounded_box(x_input, cy - box_h/2, box_w_io, box_h,
                color_input, edge_input)
    ax.text(x_input + box_w_io/2, cy + 0.22, 'Input',
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=text_input)
    ax.text(x_input + box_w_io/2, cy - 0.18, 'Atom types,\npositions',
            ha='center', va='center', fontsize=7.5,
            color=text_desc, linespacing=1.4)

    draw_mp_block(x_b1_mp)
    draw_se_block(x_b1_se)

    # Dots — centered in their reserved region
    dots_cx = x_dots + dots_w / 2
    for i in range(3):
        ax.plot(dots_cx + (i - 1) * 0.20, cy, 'o',
                color='#90a4ae', markersize=4.5, zorder=3)

    draw_mp_block(x_b2_mp)
    draw_se_block(x_b2_se)

    # Output
    rounded_box(x_output, cy - box_h/2, box_w_io, box_h,
                color_output, edge_output)
    ax.text(x_output + box_w_io/2, cy + 0.22, 'Output',
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=text_output)
    ax.text(x_output + box_w_io/2, cy - 0.18, 'Energy,\nforces',
            ha='center', va='center', fontsize=7.5,
            color=text_desc, linespacing=1.4)

    # --- Arrows (uniform: from right edge of box to left edge of next) ---
    def box_right(x, w):
        return x + w + rounding + arrow_pad

    def box_left(x):
        return x - rounding - arrow_pad

    draw_arrow(box_right(x_input, box_w_io), box_left(x_b1_mp))
    draw_arrow(box_right(x_b1_mp, box_w),   box_left(x_b1_se))
    # No arrows into/out of dots — ellipsis implies continuation
    draw_arrow(box_right(x_b2_mp, box_w),   box_left(x_b2_se))
    draw_arrow(box_right(x_b2_se, box_w),   box_left(x_output))

    # --- Annotations below blocks ---
    annot_y = cy - box_h/2 - 0.28
    for bx in [x_b1_mp, x_b2_mp]:
        ax.text(bx + box_w/2, annot_y, 'structural',
                ha='center', va='center', fontsize=8,
                color=text_mp, style='italic')
    for bx in [x_b1_se, x_b2_se]:
        ax.text(bx + box_w/2, annot_y, 'geometric',
                ha='center', va='center', fontsize=8,
                color=text_se, style='italic')

    # --- Repeat bracket on top (spanning full repeated section) ---
    brace_y = cy + box_h/2 + 0.32
    brace_x1 = x_b1_mp - rounding
    brace_x2 = x_b2_se + box_w + rounding
    brace_mid = (brace_x1 + brace_x2) / 2

    tick = 0.10
    bracket_color = '#78909c'
    ax.plot([brace_x1, brace_x1, brace_mid - 0.06],
            [brace_y - tick, brace_y, brace_y],
            color=bracket_color, lw=0.9, solid_capstyle='round')
    ax.plot([brace_mid + 0.06, brace_x2, brace_x2],
            [brace_y, brace_y, brace_y - tick],
            color=bracket_color, lw=0.9, solid_capstyle='round')
    ax.plot([brace_mid - 0.06, brace_mid, brace_mid + 0.06],
            [brace_y, brace_y + tick, brace_y],
            color=bracket_color, lw=0.9, solid_capstyle='round')

    ax.text(brace_mid, brace_y + tick + 0.12,
            r'$\times\; T$ layers',
            ha='center', va='bottom', fontsize=10,
            color=text_color)

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    print(f"Saved architecture figure to {output_path}")


if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    # Generate figures
    generate_circular_harmonics_simple(os.path.join(output_dir, 'circular_harmonics.png'))
    generate_spherical_harmonics_figure(os.path.join(output_dir, 'spherical_harmonics.png'))
    generate_cg_tensor_product_figure(os.path.join(output_dir, 'cg_tensor_product.png'))
    generate_ellipsoid_anisotropy_figure(os.path.join(output_dir, 'ellipsoid_anisotropy.png'))
    generate_architecture_figure(os.path.join(output_dir, 'architecture_overview.png'))
    generate_cg_network_figure(os.path.join(output_dir, 'cg_network.png'))

    print("Done!")
