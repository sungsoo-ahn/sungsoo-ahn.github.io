"""Generate figures for the molecular dynamics enhanced sampling blog post."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


TEXT = "#263238"
BLUE = "#5b7fa5"
BLUE_LIGHT = "#dce8f4"
AMBER = "#e8a030"
AMBER_LIGHT = "#fff3e0"
TEAL = "#1a8a7a"
TEAL_LIGHT = "#e0f2f1"
RED = "#c0503f"
RED_LIGHT = "#fce4ec"
GREEN = "#4caf50"
GREEN_LIGHT = "#e0f2e9"
NEUTRAL = "#b0bec5"

OUTPUT_DIR = Path("assets/img/blog")


def _style_axis(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=12, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=12, color=TEXT)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=8)
    ax.tick_params(colors="#78909c", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRAL)
    ax.spines["bottom"].set_color(NEUTRAL)


def _double_well(x):
    return 1.15 * (x**2 - 1.0) ** 2 + 0.12 * x


def _arrow(ax, start, end, color=TEXT, lw=1.8, ms=14, style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=3,
        zorder=5,
    )
    ax.add_patch(patch)


def _box(ax, xy, width, height, text, fc, ec, fontsize=10.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.7,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT,
        zorder=4,
    )


def _tps_dps_double_well(x, y):
    """Synthetic double-well potential from the TPS-DPS notebook."""
    term_1 = 4 * (1 - x**2 - y**2) ** 2
    term_2 = 2 * (x**2 - 2) ** 2
    term_3 = ((x + y) ** 2 - 1) ** 2
    term_4 = ((x - y) ** 2 - 1) ** 2
    return (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0


def _tps_dps_force(positions):
    """Force field for the synthetic TPS-DPS double well."""
    x = positions[..., 0]
    y = positions[..., 1]
    term_1_dx = -16 * x * (1 - x**2 - y**2)
    term_1_dy = -16 * y * (1 - x**2 - y**2)
    term_2_dx = 8 * x * (x**2 - 2)
    term_3_dx = 4 * (x + y) * ((x + y) ** 2 - 1)
    term_3_dy = term_3_dx
    term_4_dx = 4 * (x - y) * ((x - y) ** 2 - 1)
    term_4_dy = -term_4_dx

    grad_x = (term_1_dx + term_2_dx + term_3_dx + term_4_dx) / 6.0
    grad_y = (term_1_dy + term_3_dy + term_4_dy) / 6.0
    return -np.stack([grad_x, grad_y], axis=-1)


def _make_tps_grid(points=260):
    axis = np.linspace(-1.5, 1.5, points)
    x_grid, y_grid = np.meshgrid(axis, axis)
    potential = _tps_dps_double_well(x_grid, y_grid)
    return axis, x_grid, y_grid, potential


def _draw_tps_landmarks(ax, marker_scale=1.0, zorder=20):
    ax.scatter([-1, 1], [0, 0], edgecolors="#111111", c="white", s=250 * marker_scale, zorder=zorder, linewidths=1.0)
    ax.scatter([0, 0], [1, -1], edgecolors="#111111", c="white", s=420 * marker_scale, zorder=zorder, linewidths=1.0, marker="*")


def _style_tps_animation_axis(ax, xlabel="x", ylabel="y"):
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=13, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="medium")
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.0)


def _simulate_tps_paths(num_paths=24, steps=220, dt=0.015, noise=0.075, seed=0, bias=False):
    rng = np.random.default_rng(seed)
    positions = np.zeros((steps + 1, num_paths, 2))
    positions[0] = np.array([-1.0, 0.0]) + rng.normal(scale=[0.055, 0.095], size=(num_paths, 2))
    centers = np.linspace(-1.0, 1.0, steps)
    kappa = 3.3

    for step in range(steps):
        force = _tps_dps_force(positions[step])
        if bias:
            center = centers[step]
            force[:, 0] += -kappa * (positions[step, :, 0] - center)
        positions[step + 1] = positions[step] + dt * force + noise * np.sqrt(dt) * rng.normal(size=(num_paths, 2))
        positions[step + 1] = np.clip(positions[step + 1], -1.47, 1.47)

    return positions, centers


def _normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def _place_atom(a, b, c, length, angle_deg, dihedral_deg):
    """Place atom d from internal coordinates a-b-c-d."""
    angle = np.deg2rad(angle_deg)
    dihedral = np.deg2rad(dihedral_deg)
    bc = _normalize(b - c)
    normal = _normalize(np.cross(b - a, bc))
    in_plane = np.cross(normal, bc)
    return c + length * (
        np.cos(angle) * bc
        + np.sin(angle) * (np.cos(dihedral) * in_plane + np.sin(dihedral) * normal)
    )


def _dihedral_angle(a, b, c, d):
    b0 = -(b - a)
    b1 = _normalize(c - b)
    b2 = d - c
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.rad2deg(np.arctan2(y, x))


def _wrap_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def _alanine_dipeptide_coords(phi_deg, psi_deg):
    """Build a compact Ace-Ala-Nme backbone schematic with target phi/psi."""
    n_atom = np.array([0.0, 0.0, 0.0])
    ca_atom = np.array([1.46, 0.0, 0.0])
    c_prev = 1.33 * np.array([np.cos(np.deg2rad(121.0)), np.sin(np.deg2rad(121.0)), 0.0])

    c_atom = _place_atom(c_prev, n_atom, ca_atom, 1.52, 111.2, -phi_deg)
    n_next = _place_atom(n_atom, ca_atom, c_atom, 1.33, 116.2, -psi_deg)

    o_prev = _place_atom(ca_atom, n_atom, c_prev, 1.23, 123.0, 175.0)
    methyl_prev = _place_atom(ca_atom, n_atom, c_prev, 1.50, 116.0, -8.0)
    o_atom = _place_atom(n_atom, ca_atom, c_atom, 1.23, 121.0, -(psi_deg + 180.0))
    methyl_next = _place_atom(ca_atom, c_atom, n_next, 1.46, 121.0, 178.0)

    v_n = _normalize(n_atom - ca_atom)
    v_c = _normalize(c_atom - ca_atom)
    normal = _normalize(np.cross(v_n, v_c))
    bisector = _normalize(-(v_n + v_c))
    cb_atom = ca_atom + 1.53 * _normalize(0.82 * bisector + 0.58 * normal)

    coords = {
        "Me_prev": methyl_prev,
        "C_prev": c_prev,
        "O_prev": o_prev,
        "N": n_atom,
        "CA": ca_atom,
        "CB": cb_atom,
        "C": c_atom,
        "O": o_atom,
        "N_next": n_next,
        "Me_next": methyl_next,
    }
    measured_phi = _wrap_angle(_dihedral_angle(coords["C_prev"], coords["N"], coords["CA"], coords["C"]))
    measured_psi = _wrap_angle(_dihedral_angle(coords["N"], coords["CA"], coords["C"], coords["N_next"]))
    return coords, measured_phi, measured_psi


def _draw_bond(ax, coords, start, end, color="#707880", lw=2.6, alpha=1.0):
    p0 = coords[start]
    p1 = coords[end]
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="round",
    )


def _draw_torsion_ring(ax, p0, p1, color, radius=0.34, lw=2.2):
    center = 0.5 * (p0 + p1)
    axis = _normalize(p1 - p0)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u_vec = _normalize(np.cross(axis, helper))
    v_vec = np.cross(axis, u_vec)
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    ring = center + radius * (np.cos(theta)[:, None] * u_vec + np.sin(theta)[:, None] * v_vec)
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=lw, alpha=0.88)


def _draw_alanine_molecule(ax, coords, phi_deg, psi_deg):
    atom_style = {
        "C_prev": ("#5f6872", 78, "C$_{i-1}$"),
        "C": ("#5f6872", 78, "C"),
        "CA": ("#3f4852", 100, "C$_\\alpha$"),
        "CB": (GREEN, 92, "CH$_3$"),
        "Me_prev": ("#a7b0b8", 68, "CH$_3$"),
        "Me_next": ("#a7b0b8", 68, "CH$_3$"),
        "N": ("#2f74c0", 88, "N"),
        "N_next": ("#2f74c0", 88, "N$_{i+1}$"),
        "O_prev": ("#d85040", 76, "O"),
        "O": ("#d85040", 76, "O"),
    }
    bonds = [
        ("Me_prev", "C_prev"),
        ("C_prev", "O_prev"),
        ("C_prev", "N"),
        ("N", "CA"),
        ("CA", "CB"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "N_next"),
        ("N_next", "Me_next"),
    ]

    for start, end in bonds:
        _draw_bond(ax, coords, start, end, color="#8a949d", lw=2.4)
    _draw_bond(ax, coords, "N", "CA", color=TEAL, lw=5.0)
    _draw_bond(ax, coords, "CA", "C", color=AMBER, lw=5.0)
    for start, end in bonds:
        _draw_bond(ax, coords, start, end, color="#eef2f4", lw=1.0, alpha=0.65)

    _draw_torsion_ring(ax, coords["N"], coords["CA"], TEAL, radius=0.36)
    _draw_torsion_ring(ax, coords["CA"], coords["C"], AMBER, radius=0.35)

    for atom, (color, size, label) in atom_style.items():
        point = coords[atom]
        ax.scatter([point[0]], [point[1]], [point[2]], s=size, color=color, edgecolor="white", linewidth=0.8, depthshade=True)
        if atom in {"C_prev", "N", "CA", "C", "N_next", "CB"}:
            ax.text(
                point[0],
                point[1],
                point[2] + 0.18,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=TEXT,
                bbox={"fc": "white", "ec": "none", "alpha": 0.78, "pad": 1.0},
            )

    phi_center = 0.5 * (coords["N"] + coords["CA"])
    psi_center = 0.5 * (coords["CA"] + coords["C"])
    ax.text(phi_center[0], phi_center[1] - 0.32, phi_center[2] + 0.48, rf"$\phi={phi_deg:+.0f}^\circ$", color=TEAL, fontsize=10.5, fontweight="bold")
    ax.text(psi_center[0], psi_center[1] + 0.34, psi_center[2] + 0.46, rf"$\psi={psi_deg:+.0f}^\circ$", color=AMBER, fontsize=10.5, fontweight="bold")
    ax.text2D(0.03, 0.95, "Backbone torsions in alanine dipeptide", transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=TEXT)
    ax.text2D(0.03, 0.07, r"$\phi$: C$_{i-1}$-N-C$_\alpha$-C     $\psi$: N-C$_\alpha$-C-N$_{i+1}$", transform=ax.transAxes, fontsize=9.4, color=TEXT)

    all_points = np.vstack(list(coords.values()))
    center = all_points.mean(axis=0)
    span = 3.3
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - 1.25, center[2] + 1.35)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1.0, 1.0, 0.78))


def _smooth_angle_path(keypoints, frames_per_segment=28):
    path = []
    for start, end in zip(keypoints[:-1], keypoints[1:]):
        for idx in range(frames_per_segment):
            frac = idx / frames_per_segment
            eased = 0.5 - 0.5 * np.cos(np.pi * frac)
            path.append((1.0 - eased) * np.asarray(start) + eased * np.asarray(end))
    return np.asarray(path)


def generate_alanine_dipeptide_cv_gif(output_path: Path):
    """Animate alanine dipeptide phi/psi collective variables."""
    keypoints = np.array(
        [
            [-65.0, -45.0],
            [-125.0, 130.0],
            [55.0, 45.0],
            [-65.0, -45.0],
        ]
    )
    angle_path = _smooth_angle_path(keypoints, frames_per_segment=28)

    fig = plt.figure(figsize=(9.6, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.18)
    ax_mol = fig.add_subplot(gs[0, 0], projection="3d")
    ax_ram = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.12, top=0.94)

    def update(frame):
        phi, psi = angle_path[frame]
        coords, measured_phi, measured_psi = _alanine_dipeptide_coords(phi, psi)

        ax_mol.clear()
        _draw_alanine_molecule(ax_mol, coords, measured_phi, measured_psi)

        ax_ram.clear()
        ax_ram.set_title("The same frame in CV space", fontsize=11.5, fontweight="bold", color=TEXT, pad=8)
        ax_ram.set_xlim(-180, 180)
        ax_ram.set_ylim(-180, 180)
        ax_ram.set_aspect("equal")
        ax_ram.set_xlabel(r"$\phi$ (degrees)", fontsize=10.5, color=TEXT)
        ax_ram.set_ylabel(r"$\psi$ (degrees)", fontsize=10.5, color=TEXT)
        ax_ram.set_xticks([-180, -90, 0, 90, 180])
        ax_ram.set_yticks([-180, -90, 0, 90, 180])
        ax_ram.grid(color="#d7dde1", lw=0.8, alpha=0.8)
        for spine in ax_ram.spines.values():
            spine.set_color("#9aa7af")

        regions = [
            ((-62, -45), 74, 54, -20, RED_LIGHT, RED, r"$\alpha_R$"),
            ((-125, 132), 92, 62, -18, BLUE_LIGHT, BLUE, r"$\beta$"),
            ((58, 42), 64, 58, 18, GREEN_LIGHT, GREEN, r"$\alpha_L$"),
        ]
        for (xy, width, height, angle, face, edge, label) in regions:
            patch = Ellipse(xy, width, height, angle=angle, facecolor=face, edgecolor=edge, lw=1.5, alpha=0.72)
            ax_ram.add_patch(patch)
            ax_ram.text(xy[0], xy[1], label, ha="center", va="center", fontsize=10, color=edge, fontweight="bold")

        ax_ram.plot(angle_path[:, 0], angle_path[:, 1], color="#6b747c", lw=1.5, alpha=0.45)
        trail = angle_path[max(0, frame - 12) : frame + 1]
        if len(trail) > 1:
            ax_ram.plot(trail[:, 0], trail[:, 1], color=AMBER, lw=3.0, alpha=0.86)
        ax_ram.axvline(measured_phi, color=TEAL, lw=1.2, ls="--", alpha=0.75)
        ax_ram.axhline(measured_psi, color=AMBER, lw=1.2, ls="--", alpha=0.75)
        ax_ram.scatter([measured_phi], [measured_psi], s=120, color=AMBER, edgecolor=TEXT, linewidth=1.1, zorder=10)
        ax_ram.text(
            0.03,
            0.04,
            rf"$s(\mathbf{{x}})=(\phi,\psi)=({measured_phi:+.0f}^\circ,{measured_psi:+.0f}^\circ)$",
            transform=ax_ram.transAxes,
            fontsize=9.4,
            color=TEXT,
            bbox={"fc": "white", "ec": "#d7dde1", "alpha": 0.9, "pad": 3},
        )

    animation = FuncAnimation(fig, update, frames=len(angle_path), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=95)
    plt.close(fig)
    print(f"Saved alanine dipeptide CV gif to {output_path}")


def generate_umbrella_sweep_gif(output_path: Path):
    """Animate the umbrella window moving along a collective variable."""
    _, x_grid, y_grid, potential = _make_tps_grid(points=260)
    kappa = 2.9
    centers = np.r_[np.linspace(-1, 1, 46), np.linspace(1, -1, 46)]
    levels = np.linspace(potential.min(), 5.0, 70)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    def update(frame):
        ax.clear()
        center = centers[frame]
        biased = potential + 0.5 * kappa * (x_grid - center) ** 2
        ax.contourf(x_grid, y_grid, biased, levels=levels, vmax=5.0)
        ax.contour(x_grid, y_grid, potential, levels=np.linspace(0, 3, 7), colors="white", linewidths=0.55, alpha=0.55)
        ax.axvspan(center - 0.15, center + 0.15, color="white", alpha=0.28, zorder=6)
        ax.axvline(center, color=AMBER, lw=2.5, zorder=8)
        _draw_tps_landmarks(ax, marker_scale=0.85)
        ax.text(
            0,
            1.32,
            rf"moving umbrella window: $s_k={center:+.2f}$",
            ha="center",
            fontsize=10.2,
            color="#222222",
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 2},
            zorder=30,
        )
        ax.text(
            0,
            -1.34,
            r"$U_{\mathrm{bias}}(x,y)=U(x,y)+\frac{1}{2}\kappa(x-s_k)^2$",
            ha="center",
            fontsize=8.8,
            color="#222222",
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 2},
            zorder=30,
        )
        _style_tps_animation_axis(ax)

    animation = FuncAnimation(fig, update, frames=len(centers), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=95)
    plt.close(fig)
    print(f"Saved umbrella sweep gif to {output_path}")


def generate_md_dynamics_gif(output_path: Path, *, bias=False):
    """Animate toy overdamped MD trajectories with or without a CV bias."""
    _, x_grid, y_grid, potential = _make_tps_grid(points=260)
    positions, centers = _simulate_tps_paths(seed=19 if bias else 13, bias=bias)
    frame_indices = np.arange(0, positions.shape[0], 3)
    levels = np.linspace(potential.min(), 5.0 if bias else 3.0, 70)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    cmap = "viridis"
    path_colors = plt.cm.gist_rainbow(np.linspace(0, 1, positions.shape[1]))

    def update(frame):
        ax.clear()
        step = frame_indices[frame]
        if bias:
            center = centers[min(step, len(centers) - 1)]
            background = potential + 0.5 * 3.3 * (x_grid - center) ** 2
            ax.contourf(x_grid, y_grid, background, levels=levels, vmax=5.0, cmap=cmap)
            ax.axvspan(center - 0.15, center + 0.15, color="white", alpha=0.26, zorder=6)
            ax.axvline(center, color=AMBER, lw=2.5, zorder=8)
            title = "Biased dynamics: moving CV restraint"
            footer = "bias drives trajectories through the window"
        else:
            ax.contourf(x_grid, y_grid, potential, levels=levels, vmax=3.0, cmap=cmap)
            title = "Unbiased dynamics: trapped in basin A"
            footer = "short trajectories rarely reach basin B"

        _draw_tps_landmarks(ax, marker_scale=0.85)
        start = max(0, step - 60)
        segment = positions[start : step + 1]
        for path_idx, color in enumerate(path_colors):
            ax.plot(segment[:, path_idx, 0], segment[:, path_idx, 1], color=color, lw=1.0, alpha=0.55, zorder=12)
        current = positions[step]
        ax.scatter(current[:, 0], current[:, 1], s=16, c=path_colors, edgecolors="none", alpha=0.95, zorder=18)
        ax.text(
            0,
            1.32,
            title,
            ha="center",
            fontsize=10.0,
            color="#222222",
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 2},
            zorder=30,
        )
        ax.text(
            0,
            -1.34,
            footer,
            ha="center",
            fontsize=8.9,
            color="#222222",
            bbox={"fc": "white", "ec": "none", "alpha": 0.84, "pad": 2},
            zorder=30,
        )
        _style_tps_animation_axis(ax)

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=95)
    plt.close(fig)
    label = "biased" if bias else "unbiased"
    print(f"Saved {label} dynamics gif to {output_path}")


def generate_double_well_umbrella(output_path: Path):
    """Collective variables and umbrella sampling on a 2D double well."""
    x = np.linspace(-1.5, 1.5, 420)
    y = np.linspace(-1.5, 1.5, 420)
    x_grid, y_grid = np.meshgrid(x, y)
    potential = _tps_dps_double_well(x_grid, y_grid)

    window_center = 0.0
    kappa = 2.9
    umbrella = 0.5 * kappa * (x_grid - window_center) ** 2
    biased = potential + umbrella

    potential_levels = np.linspace(potential.min(), 3.0, 100)
    guide_levels = np.linspace(potential.min(), 3.0, 36)
    umbrella_levels = np.linspace(0, np.quantile(umbrella, 0.97), 100)
    biased_levels = np.linspace(biased.min(), 5.0, 100)

    fig, axes = plt.subplots(2, 2, figsize=(9.1, 8.1), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.08)
    axes = axes.ravel()

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.0)

    def label_box(alpha=0.84, pad=1.8):
        return {"fc": "white", "ec": "none", "alpha": alpha, "pad": pad}

    def draw_landmarks(ax, zorder=20, labels=False):
        ax.scatter([-1, 1], [0, 0], edgecolors="#111111", c="white", s=390, zorder=zorder, linewidths=1.2)
        ax.scatter([0, 0], [1, -1], edgecolors="#111111", c="white", s=650, zorder=zorder, linewidths=1.2, marker="*")
        if labels:
            ax.text(-1, -0.27, "A", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222222", zorder=zorder + 1)
            ax.text(1, -0.27, "B", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222222", zorder=zorder + 1)
            ax.text(0.18, 1.02, "saddle", ha="left", va="center", fontsize=8.8, color="#222222", bbox=label_box(0.78, 1.2), zorder=zorder + 1)
            ax.text(0.18, -1.02, "saddle", ha="left", va="center", fontsize=8.8, color="#222222", bbox=label_box(0.78, 1.2), zorder=zorder + 1)

    def title(ax, text):
        ax.text(
            0,
            1.34,
            text,
            ha="center",
            va="center",
            fontsize=11.7,
            fontweight="medium",
            color="#222222",
            bbox=label_box(0.84, 2.1),
            zorder=40,
        )

    def panel_tag(ax, letter):
        circle = Circle((-1.32, 1.28), 0.13, fc="white", ec="#111111", lw=1.0, zorder=45)
        ax.add_patch(circle)
        ax.text(-1.32, 1.28, letter, ha="center", va="center", fontsize=10, fontweight="bold", color="#222222", zorder=46)

    ax = axes[0]
    ax.contourf(x_grid, y_grid, potential, levels=potential_levels, vmax=3, zorder=0)
    draw_landmarks(ax, labels=True)
    title(ax, "Original double-well potential")
    panel_tag(ax, "A")
    ax.text(
        0,
        -1.38,
        "start and target basins are separated by two channels",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#222222",
        bbox=label_box(0.78, 1.4),
        zorder=30,
    )

    ax = axes[1]
    ax.contourf(x_grid, y_grid, potential, levels=guide_levels, vmax=3, zorder=0, alpha=0.54)
    draw_landmarks(ax)
    ax.axvspan(-0.16, 0.16, color="white", alpha=0.36, zorder=10)
    ax.axvline(0, color=AMBER, lw=3.0, zorder=28)
    for xpos, color, label, ypos in [
        (-1, "#2878b5", r"$s=-1$", -0.82),
        (0, AMBER, r"$s_k=0$", -0.42),
        (1, "#3b9b5d", r"$s=1$", -0.82),
    ]:
        ax.plot([xpos, xpos], [0, -1.06], linestyle="--", lw=1.7, color=color, alpha=0.88, zorder=25)
        ax.text(xpos, ypos, label, ha="center", va="center", fontsize=10.0, color=color, fontweight="bold", bbox=label_box(0.84, 1.5), zorder=32)
    ax.add_patch(FancyArrowPatch((-1.16, -1.17), (1.16, -1.17), arrowstyle="-|>", mutation_scale=15, lw=2.8, color=AMBER, zorder=30))
    ax.text(
        0,
        -1.36,
        r"collective variable: $s=\xi(x,y)=x$",
        ha="center",
        va="center",
        fontsize=10.5,
        color=AMBER,
        fontweight="bold",
        bbox=label_box(0.86, 1.8),
        zorder=35,
    )
    ax.text(0, 1.05, "project every configuration onto the x-axis", ha="center", va="center", fontsize=9.5, color="#222222", bbox=label_box(0.82, 1.5), zorder=35)
    title(ax, "Define the collective variable")
    panel_tag(ax, "B")

    ax = axes[2]
    ax.contourf(x_grid, y_grid, umbrella, levels=umbrella_levels, cmap="YlOrBr", zorder=0)
    ax.axvspan(-0.16, 0.16, color="white", alpha=0.46, zorder=10)
    ax.axvline(0, color=AMBER, lw=3.1, zorder=20)
    ax.text(0, 0.15, "low restraint\nnear window", ha="center", va="center", fontsize=9.2, color=AMBER, fontweight="bold", bbox=label_box(0.88, 1.5), zorder=31)
    for xpos in [-0.92, 0.92]:
        ax.add_patch(FancyArrowPatch((0.22 * np.sign(xpos), -0.18), (xpos, -0.18), arrowstyle="-|>", mutation_scale=14, lw=2.0, color="#c65b43", zorder=28))
    ax.text(-0.93, -0.42, "higher penalty", ha="center", va="center", fontsize=9.0, color="#c65b43", fontweight="bold", bbox=label_box(0.84, 1.3), zorder=31)
    ax.text(0.93, -0.42, "higher penalty", ha="center", va="center", fontsize=9.0, color="#c65b43", fontweight="bold", bbox=label_box(0.84, 1.3), zorder=31)
    ax.text(0, -1.30, r"$V_k=\frac{1}{2}\kappa(x-s_k)^2,\quad s_k=0$", ha="center", fontsize=9.3, color="#222222", bbox=label_box(0.86, 1.8), zorder=31)
    title(ax, "Umbrella restraint alone")
    panel_tag(ax, "C")

    ax = axes[3]
    ax.contourf(x_grid, y_grid, biased, levels=biased_levels, vmax=5.0, zorder=0)
    ax.contour(x_grid, y_grid, potential, levels=np.linspace(0, 3, 8), colors="white", linewidths=0.65, alpha=0.55, zorder=4)
    draw_landmarks(ax)
    ax.axvspan(-0.16, 0.16, color="white", alpha=0.30, zorder=8)
    ax.axvline(0, color=AMBER, lw=3.1, zorder=22)
    for yy, label in [(1.0, "upper channel"), (-1.0, "lower channel")]:
        ax.add_patch(Circle((0, yy), 0.25, fc="none", ec=AMBER, lw=2.2, zorder=35))
        ax.text(0.36, yy, label, ha="left", va="center", fontsize=8.8, color=AMBER, fontweight="bold", bbox=label_box(0.84, 1.3), zorder=36)
    ax.text(0, -1.35, "basins are penalized; the window is sampled more often", ha="center", fontsize=9.0, color="#222222", bbox=label_box(0.86, 1.8), zorder=36)
    title(ax, r"Biased potential $U+V_k$")
    panel_tag(ax, "D")

    for ax in axes[[0, 2]]:
        ax.set_ylabel("y", fontsize=18, fontweight="medium")
    for ax in axes[2:]:
        ax.set_xlabel("x", fontsize=18, fontweight="medium")

    fig.savefig(output_path, dpi=225, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved double-well umbrella to {output_path}")


def generate_metastability(output_path: Path):
    """Rare events in a double-well landscape."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.32})
    x = np.linspace(-2.1, 2.1, 600)
    u = _double_well(x)

    ax = axes[0]
    ax.plot(x, u, color=BLUE, linewidth=2.6)
    ax.fill_between(x, 0, u, color=BLUE_LIGHT, alpha=0.45)
    ax.axvspan(-1.45, -0.45, color=BLUE_LIGHT, alpha=0.6)
    ax.axvspan(0.55, 1.55, color=GREEN_LIGHT, alpha=0.65)
    ax.plot([-0.92], [_double_well(np.array([-0.92]))[0] + 0.08], "o", color=BLUE, markersize=9)
    ax.plot([0.93], [_double_well(np.array([0.93]))[0] + 0.08], "o", color=GREEN, markersize=9)
    ax.plot([0.0], [_double_well(np.array([0.0]))[0] + 0.08], "*", color=AMBER, markersize=13)
    ax.annotate(
        "high barrier",
        xy=(0.0, _double_well(np.array([0.0]))[0] + 0.08),
        xytext=(-0.65, 1.95),
        arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5),
        fontsize=10.5,
        color=TEXT,
        ha="center",
    )
    _arrow(ax, (-0.85, 0.35), (-0.25, 0.95), color=RED, lw=2.0)
    _arrow(ax, (-0.25, 0.95), (-0.78, 0.36), color=RED, lw=2.0)
    ax.text(-1.02, 0.12, "metastable\nstate A", ha="center", va="bottom", fontsize=10, color=BLUE)
    ax.text(1.02, 0.12, "state B", ha="center", va="bottom", fontsize=10, color=GREEN)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="reaction coordinate", ylabel="free energy", title="Unbiased MD gets trapped")

    ax = axes[1]
    ax.plot(x, u, color=BLUE, linewidth=2.0, alpha=0.8, label="original")
    bias = -0.82 * np.exp(-0.5 * ((x - 0.0) / 0.55) ** 2)
    ub = u + bias
    ax.plot(x, ub, color=TEAL, linewidth=2.7, label="biased")
    ax.fill_between(x, ub, u, where=bias < 0, color=TEAL_LIGHT, alpha=0.75)
    path_x = np.linspace(-0.93, 0.95, 70)
    path_y = np.interp(path_x, x, ub) + 0.12 + 0.03 * np.sin(np.linspace(0, 5 * np.pi, 70))
    ax.plot(path_x, path_y, color=RED, linewidth=2.1)
    for i in [12, 28, 44, 58]:
        _arrow(ax, (path_x[i - 2], path_y[i - 2]), (path_x[i + 2], path_y[i + 2]), color=RED, lw=1.7, ms=12)
    ax.text(0.0, 1.65, "bias lowers the\nsampling barrier", ha="center", fontsize=10.5, color=TEAL)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="reaction coordinate", ylabel="", title="Enhanced sampling changes what is easy")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved metastability to {output_path}")


def generate_cv_metadynamics(output_path: Path):
    """Collective variables and metadynamics biasing."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw={"wspace": 0.36})

    ax = axes[0]
    rng = np.random.default_rng(7)
    mean_a = np.array([-1.0, -0.45])
    mean_b = np.array([1.05, 0.55])
    cloud_a = rng.normal(mean_a, [0.32, 0.23], size=(80, 2))
    cloud_b = rng.normal(mean_b, [0.34, 0.25], size=(80, 2))
    ax.scatter(cloud_a[:, 0], cloud_a[:, 1], s=18, color=BLUE, alpha=0.72)
    ax.scatter(cloud_b[:, 0], cloud_b[:, 1], s=18, color=GREEN, alpha=0.72)
    ax.plot([-1.7, 1.7], [-0.95, 0.95], color=AMBER, linewidth=2.2)
    _arrow(ax, (-1.45, -0.82), (1.45, 0.82), color=AMBER, lw=2.0)
    ax.text(0.0, 1.05, r"CV $s(\mathbf{x})$", ha="center", fontsize=11, color=AMBER, fontweight="bold")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axis(ax, xlabel="coordinates", ylabel="", title="Compress configurations")

    ax = axes[1]
    s = np.linspace(-2.0, 2.0, 600)
    f = 0.95 * (s**2 - 1.0) ** 2 + 0.08 * s
    ax.plot(s, f, color=BLUE, linewidth=2.5)
    ax.fill_between(s, 0, f, color=BLUE_LIGHT, alpha=0.45)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    _style_axis(ax, xlabel=r"CV $s$", ylabel=r"$F(s)$", title="Estimate a free-energy profile")
    ax.text(0.02, 1.22, r"$F(s)=-\beta^{-1}\log p(s)$", ha="center", fontsize=10.5, color=TEXT)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(s, f, color=BLUE, linewidth=2.0, alpha=0.65, label="free energy")
    centers = np.array([-1.0, -0.72, -0.35, 0.02, 0.38])
    heights = np.linspace(0.22, 0.11, len(centers))
    bias = np.zeros_like(s)
    for c, h in zip(centers, heights):
        g = h * np.exp(-0.5 * ((s - c) / 0.18) ** 2)
        bias += g
        ax.fill_between(s, 0, g, color=AMBER_LIGHT, alpha=0.9)
        ax.plot(s, g, color=AMBER, linewidth=1.2, alpha=0.9)
    effective = f + bias
    ax.plot(s, effective, color=TEAL, linewidth=2.5, label="biased surface")
    ax.text(-0.65, 1.18, "metadynamics deposits\nhistory-dependent hills", ha="center", fontsize=10, color=AMBER)
    _arrow(ax, (-0.65, 1.05), (-0.75, 0.34), color=AMBER, lw=1.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    ax.set_yticks([])
    _style_axis(ax, xlabel=r"CV $s$", ylabel="", title="Bias along the CV")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved cv metadynamics to {output_path}")


def generate_method_map(output_path: Path):
    """Map classical enhanced sampling and ML methods."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.05, 0.70), 0.22, 0.15, "Unbiased MD\nphysical dynamics\nrare transitions", BLUE_LIGHT, BLUE)
    _box(ax, (0.39, 0.70), 0.22, 0.15, "CV-based biasing\numbrella, metadynamics,\nOPES, SMD", AMBER_LIGHT, AMBER)
    _box(ax, (0.73, 0.70), 0.22, 0.15, "Free energies\nreweighting recovers\nunbiased statistics", GREEN_LIGHT, GREEN)

    _box(ax, (0.22, 0.36), 0.25, 0.15, "Learn the CV\nBioEmu-CV:\ntime-lagged slow modes", TEAL_LIGHT, TEAL)
    _box(ax, (0.55, 0.36), 0.25, 0.15, "Learn the path bias\nTPS-DPS:\nCV-free path sampling", RED_LIGHT, RED)

    _box(ax, (0.30, 0.08), 0.40, 0.13, "Path-measure view\nJarzynski, AIS, diffusion models,\ntrajectory objectives", "white", NEUTRAL, fontsize=10.3)

    _arrow(ax, (0.27, 0.775), (0.39, 0.775), color=TEXT)
    _arrow(ax, (0.61, 0.775), (0.73, 0.775), color=TEXT)
    _arrow(ax, (0.34, 0.51), (0.45, 0.70), color=TEAL)
    _arrow(ax, (0.67, 0.51), (0.55, 0.70), color=RED)
    _arrow(ax, (0.43, 0.36), (0.43, 0.21), color=NEUTRAL, lw=1.5)
    _arrow(ax, (0.68, 0.36), (0.60, 0.21), color=NEUTRAL, lw=1.5)
    _arrow(ax, (0.86, 0.70), (0.70, 0.21), color=GREEN, lw=1.5)

    ax.text(0.33, 0.82, "add a bias", ha="center", fontsize=9.5, color=TEXT)
    ax.text(0.67, 0.82, "undo the bias", ha="center", fontsize=9.5, color=TEXT)
    ax.text(0.16, 0.60, "classical problem:\nwhere should the bias act?", ha="center", fontsize=10.0, color=TEXT)
    ax.text(0.50, 0.94, "Enhanced sampling is controlled distribution shift", ha="center", fontsize=14, color=TEXT, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved method map to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_metastability(OUTPUT_DIR / "md_metastability_bias.png")
    generate_double_well_umbrella(OUTPUT_DIR / "md_double_well_umbrella.png")
    generate_umbrella_sweep_gif(OUTPUT_DIR / "md_umbrella_sweep.gif")
    generate_md_dynamics_gif(OUTPUT_DIR / "md_unbiased_dynamics.gif", bias=False)
    generate_md_dynamics_gif(OUTPUT_DIR / "md_biased_dynamics.gif", bias=True)
    generate_alanine_dipeptide_cv_gif(OUTPUT_DIR / "md_alanine_dipeptide_cvs.gif")
    generate_cv_metadynamics(OUTPUT_DIR / "md_cv_metadynamics.png")
    generate_method_map(OUTPUT_DIR / "md_sampling_method_map.png")


if __name__ == "__main__":
    main()
