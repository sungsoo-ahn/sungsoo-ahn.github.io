"""Generate figures for the molecular dynamics enhanced sampling blog post."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

import blog_figure_style as bfs


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

bfs.use_blog_style()
LANDSCAPE_CMAP = LinearSegmentedColormap.from_list(
    "blog_landscape",
    ["#f7fbfc", "#dcecf2", "#b9d9df", "#89bfc6", "#5d9ba5"],
)
BIASED_CMAP = LinearSegmentedColormap.from_list(
    "blog_biased_landscape",
    ["#fffaf0", "#f2e0b6", "#d4b66a", "#9e8447"],
)


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
    ax.scatter([-1, 1], [0, 0], edgecolors=TEXT, c="white", s=250 * marker_scale, zorder=zorder, linewidths=0.95)
    ax.scatter([0, 0], [1, -1], edgecolors=TEXT, c="white", s=420 * marker_scale, zorder=zorder, linewidths=0.95, marker="*")


def _style_tps_animation_axis(ax, xlabel="x", ylabel="y"):
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=10.5, fontweight="medium", color=TEXT, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="medium", color=TEXT, labelpad=2)
    for spine in ax.spines.values():
        spine.set_color(bfs.SPINE)
        spine.set_linewidth(0.85)


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
        "C_prev": ("#5f6872", 74),
        "C": ("#5f6872", 74),
        "CA": ("#3f4852", 96),
        "CB": (GREEN, 84),
        "Me_prev": ("#a7b0b8", 60),
        "Me_next": ("#a7b0b8", 60),
        "N": ("#2f74c0", 84),
        "N_next": ("#2f74c0", 84),
        "O_prev": ("#d85040", 68),
        "O": ("#d85040", 68),
    }
    atom_labels = {
        "C_prev": (r"C$_{i-1}$", np.array([-0.08, 0.14, 0.18])),
        "N": ("N", np.array([-0.15, -0.10, 0.18])),
        "CA": (r"C$_\alpha$", np.array([0.02, -0.17, 0.20])),
        "C": ("C", np.array([0.16, -0.10, 0.18])),
        "N_next": (r"N$_{i+1}$", np.array([0.16, 0.12, 0.18])),
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

    for atom, (color, size) in atom_style.items():
        point = coords[atom]
        ax.scatter([point[0]], [point[1]], [point[2]], s=size, color=color, edgecolor="white", linewidth=0.8, depthshade=True)
        if atom in atom_labels:
            label, offset = atom_labels[atom]
            label_point = point + offset
            ax.text(
                label_point[0],
                label_point[1],
                label_point[2],
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color=TEXT,
                bbox=bfs.label_box(alpha=0.88, pad=0.9),
            )

    phi_center = 0.5 * (coords["N"] + coords["CA"])
    psi_center = 0.5 * (coords["CA"] + coords["C"])
    ax.text(
        phi_center[0] - 0.02,
        phi_center[1] - 0.36,
        phi_center[2] + 0.52,
        rf"$\phi={phi_deg:+.0f}^\circ$",
        color=TEAL,
        fontsize=10.0,
        fontweight="bold",
        bbox=bfs.label_box(alpha=0.9, pad=1.1),
    )
    ax.text(
        psi_center[0] + 0.08,
        psi_center[1] + 0.36,
        psi_center[2] + 0.50,
        rf"$\psi={psi_deg:+.0f}^\circ$",
        color=AMBER,
        fontsize=10.0,
        fontweight="bold",
        bbox=bfs.label_box(alpha=0.9, pad=1.1),
    )
    ax.text2D(
        0.04,
        0.91,
        "molecular frame",
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="semibold",
        color=TEXT,
        bbox=bfs.label_box(alpha=0.92, pad=1.4),
    )

    all_points = np.vstack(list(coords.values()))
    center = all_points.mean(axis=0)
    span = 3.55
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

    fig = plt.figure(figsize=(9.35, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.27)
    ax_mol = fig.add_subplot(gs[0, 0], projection="3d")
    ax_ram = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.13, top=0.92)

    def update(frame):
        phi, psi = angle_path[frame]
        coords, measured_phi, measured_psi = _alanine_dipeptide_coords(phi, psi)

        ax_mol.clear()
        _draw_alanine_molecule(ax_mol, coords, measured_phi, measured_psi)

        ax_ram.clear()
        ax_ram.set_title("Ramachandran CV space", loc="left", fontsize=10.8, fontweight="semibold", color=TEXT, pad=8)
        ax_ram.set_xlim(-180, 180)
        ax_ram.set_ylim(-180, 180)
        ax_ram.set_aspect("equal")
        ax_ram.set_xlabel(r"$\phi$ angle")
        ax_ram.set_ylabel(r"$\psi$ angle")
        ax_ram.set_xticks([-180, -90, 0, 90, 180])
        ax_ram.set_yticks([-180, -90, 0, 90, 180])
        bfs.style_axis(ax_ram, grid=True)

        regions = [
            ((-62, -45), 74, 54, -20, bfs.RED_LIGHT, RED, r"$\alpha_R$"),
            ((-125, 132), 92, 62, -18, BLUE_LIGHT, BLUE, r"$\beta$"),
            ((58, 42), 64, 58, 18, GREEN_LIGHT, GREEN, r"$\alpha_L$"),
        ]
        for (xy, width, height, angle, face, edge, label) in regions:
            patch = Ellipse(xy, width, height, angle=angle, facecolor=face, edgecolor=edge, lw=1.1, alpha=0.6)
            ax_ram.add_patch(patch)
            ax_ram.text(
                xy[0],
                xy[1],
                label,
                ha="center",
                va="center",
                fontsize=9.0,
                color=edge,
                fontweight="semibold",
                bbox=bfs.label_box(alpha=0.82, pad=1.0),
            )

        ax_ram.plot(angle_path[:, 0], angle_path[:, 1], color=bfs.NEUTRAL, lw=1.2, alpha=0.45)
        trail = angle_path[max(0, frame - 12) : frame + 1]
        if len(trail) > 1:
            ax_ram.plot(trail[:, 0], trail[:, 1], color=AMBER, lw=2.4, alpha=0.9)
        ax_ram.axvline(measured_phi, color=TEAL, lw=1.0, ls="--", alpha=0.6)
        ax_ram.axhline(measured_psi, color=AMBER, lw=1.0, ls="--", alpha=0.6)
        ax_ram.scatter([measured_phi], [measured_psi], s=80, color=AMBER, edgecolor=TEXT, linewidth=0.8, zorder=10)
        ax_ram.text(
            0.04,
            0.05,
            rf"$({measured_phi:+.0f}^\circ,{measured_psi:+.0f}^\circ)$",
            transform=ax_ram.transAxes,
            fontsize=9.0,
            color=TEXT,
            bbox=bfs.label_box(alpha=0.82, pad=1.0),
        )

    animation = FuncAnimation(fig, update, frames=len(angle_path), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=120)
    plt.close(fig)
    print(f"Saved alanine dipeptide CV gif to {output_path}")


def generate_umbrella_sweep_gif(output_path: Path):
    """Animate the umbrella window moving along a collective variable."""
    _, x_grid, y_grid, potential = _make_tps_grid(points=260)
    kappa = 2.9
    centers = np.r_[np.linspace(-1, 1, 46), np.linspace(1, -1, 46)]
    extent = (-1.5, 1.5, -1.5, 1.5)
    cmap = BIASED_CMAP.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.11, top=0.93)

    def update(frame):
        ax.clear()
        center = centers[frame]
        biased = potential + 0.5 * kappa * (x_grid - center) ** 2
        ax.imshow(
            np.ma.masked_greater(biased, 5.0),
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=potential.min(),
            vmax=5.0,
            interpolation="bilinear",
            zorder=0,
        )
        ax.contour(x_grid, y_grid, potential, levels=np.linspace(0, 3, 5), colors="white", linewidths=0.58, alpha=0.48)
        ax.axvspan(center - 0.12, center + 0.12, color="white", alpha=0.42, zorder=6)
        ax.axvline(center, color=AMBER, lw=2.0, zorder=8)
        _draw_tps_landmarks(ax, marker_scale=0.85)
        bfs.direct_label(ax, 0, 1.34, rf"$s_k={center:+.2f}$", AMBER, size=9.2)
        _style_tps_animation_axis(ax)

    animation = FuncAnimation(fig, update, frames=len(centers), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=120)
    plt.close(fig)
    print(f"Saved umbrella sweep gif to {output_path}")


def generate_md_dynamics_gif(output_path: Path, *, bias=False):
    """Animate toy overdamped MD trajectories with or without a CV bias."""
    _, x_grid, y_grid, potential = _make_tps_grid(points=260)
    positions, centers = _simulate_tps_paths(seed=19 if bias else 13, bias=bias)
    frame_indices = np.arange(0, positions.shape[0], 3)
    extent = (-1.5, 1.5, -1.5, 1.5)
    biased_cmap = BIASED_CMAP.copy()
    biased_cmap.set_bad("white")
    landscape_cmap = LANDSCAPE_CMAP.copy()
    landscape_cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.11, top=0.93)
    path_color = TEAL if bias else BLUE

    def update(frame):
        ax.clear()
        step = frame_indices[frame]
        if bias:
            center = centers[min(step, len(centers) - 1)]
            background = potential + 0.5 * 3.3 * (x_grid - center) ** 2
            ax.imshow(
                np.ma.masked_greater(background, 5.0),
                extent=extent,
                origin="lower",
                cmap=biased_cmap,
                vmin=potential.min(),
                vmax=5.0,
                interpolation="bilinear",
                zorder=0,
            )
            ax.axvspan(center - 0.12, center + 0.12, color="white", alpha=0.36, zorder=6)
            ax.axvline(center, color=AMBER, lw=2.0, zorder=8)
            label = "biased"
        else:
            ax.imshow(
                np.ma.masked_greater(potential, 3.0),
                extent=extent,
                origin="lower",
                cmap=landscape_cmap,
                vmin=potential.min(),
                vmax=3.0,
                interpolation="bilinear",
                zorder=0,
            )
            label = "unbiased"

        _draw_tps_landmarks(ax, marker_scale=0.85)
        start = max(0, step - 60)
        segment = positions[start : step + 1]
        for path_idx in range(positions.shape[1]):
            ax.plot(segment[:, path_idx, 0], segment[:, path_idx, 1], color=path_color, lw=1.0, alpha=0.28, zorder=12)
        current = positions[step]
        ax.scatter(current[:, 0], current[:, 1], s=18, c=path_color, edgecolors="white", linewidth=0.3, alpha=0.95, zorder=18)
        bfs.direct_label(ax, 0, 1.34, label, path_color, size=9.2)
        _style_tps_animation_axis(ax)

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=85)
    animation.save(output_path, writer=PillowWriter(fps=12), dpi=120)
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

    potential_levels = np.linspace(potential.min(), 3.0, 48)
    biased_levels = np.linspace(biased.min(), 5.0, 48)

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), gridspec_kw={"wspace": 0.24})

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        for spine in ax.spines.values():
            spine.set_color(bfs.SPINE)
            spine.set_linewidth(0.8)
        ax.tick_params(length=0, labelsize=8, colors=bfs.MUTED)
        ax.set_xlabel("x", labelpad=2)

    axes[0].set_ylabel("y", labelpad=2)

    ax = axes[0]
    ax.contourf(x_grid, y_grid, potential, levels=potential_levels, vmax=3, zorder=0, cmap=LANDSCAPE_CMAP)
    ax.contour(x_grid, y_grid, potential, levels=np.linspace(0, 3, 6), colors="white", linewidths=0.55, alpha=0.66)
    _draw_tps_landmarks(ax, marker_scale=0.95)
    ax.text(-1, -0.26, "A", ha="center", va="center", fontsize=10, fontweight="semibold", color=TEXT, zorder=35)
    ax.text(1, -0.26, "B", ha="center", va="center", fontsize=10, fontweight="semibold", color=TEXT, zorder=35)
    ax.set_title("Original landscape", loc="left", pad=8, fontweight="semibold")

    ax = axes[1]
    ax.contourf(x_grid, y_grid, potential, levels=np.linspace(potential.min(), 3.0, 38), vmax=3, zorder=0, alpha=0.62, cmap=LANDSCAPE_CMAP)
    _draw_tps_landmarks(ax, marker_scale=0.95)
    ax.axvspan(-0.13, 0.13, color="white", alpha=0.48, zorder=10)
    ax.axvline(0, color=AMBER, lw=2.1, zorder=28)
    ax.annotate(
        "",
        xy=(1.16, -1.12),
        xytext=(-1.16, -1.12),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": AMBER, "mutation_scale": 13},
        zorder=30,
    )
    bfs.callout_label(ax, r"$s=\xi(x,y)=x$", xy=(0.0, 0.1), xytext=(0.48, 1.24), color=AMBER, size=9.1, ha="left", rad=-0.2)
    ax.set_title("Choose a coordinate", loc="left", pad=8, fontweight="semibold")

    ax = axes[2]
    ax.contourf(x_grid, y_grid, biased, levels=biased_levels, vmax=5.0, zorder=0, cmap=BIASED_CMAP)
    ax.contour(x_grid, y_grid, potential, levels=np.linspace(0, 3, 6), colors="white", linewidths=0.55, alpha=0.58, zorder=4)
    _draw_tps_landmarks(ax, marker_scale=0.95)
    ax.axvspan(-0.13, 0.13, color="white", alpha=0.38, zorder=8)
    ax.axvline(0, color=AMBER, lw=2.1, zorder=22)
    for yy in [1.0, -1.0]:
        ax.add_patch(Circle((0, yy), 0.19, fc="white", ec=AMBER, lw=1.7, alpha=0.9, zorder=35))
    bfs.callout_label(ax, r"$U+V_k$", xy=(0.0, -1.0), xytext=(0.54, -1.25), color=AMBER, size=9.2, ha="left", rad=0.15)
    ax.set_title("Add umbrella bias", loc="left", pad=8, fontweight="semibold")

    bfs.save_figure(fig, output_path, dpi=250)


def generate_metastability(output_path: Path):
    """Rare events in a double-well landscape."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.15), gridspec_kw={"wspace": 0.24})
    x = np.linspace(-2.1, 2.1, 800)
    u = _double_well(x)
    y_a = _double_well(np.array([-0.96]))[0]
    y_b = _double_well(np.array([0.95]))[0]
    y_barrier = _double_well(np.array([0.0]))[0]
    y_floor = -0.36

    def draw_energy_gap(ax, x_gap, y_low, y_high, low_anchor_x, high_anchor_x, color):
        guide_style = {
            "color": color,
            "lw": 1.0,
            "alpha": 0.62,
            "linestyle": (0, (2.2, 2.2)),
            "zorder": 24,
        }
        ax.plot([min(x_gap, low_anchor_x), max(x_gap, low_anchor_x)], [y_low, y_low], **guide_style)
        ax.plot([min(x_gap, high_anchor_x), max(x_gap, high_anchor_x)], [y_high, y_high], **guide_style)
        ax.annotate(
            "",
            xy=(x_gap, y_high),
            xytext=(x_gap, y_low),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5, "mutation_scale": 12},
            zorder=30,
        )

    ax = axes[0]
    ax.fill_between(x, y_floor, u, color=BLUE_LIGHT, alpha=0.46, zorder=1)
    ax.plot(x, u, color=BLUE, linewidth=2.6, zorder=8)
    bfs.state_marker(ax, -0.96, y_a + 0.08, "A", BLUE, label_dx=-0.18, label_dy=0.30)
    bfs.state_marker(ax, 0.95, y_b + 0.08, "B", GREEN, label_dx=0.17, label_dy=0.28)
    ax.scatter([0.0], [y_barrier + 0.03], s=120, color="white", edgecolors="none", zorder=24)
    ax.scatter([0.0], [y_barrier + 0.03], s=58, color=AMBER, edgecolors="white", linewidths=1.3, zorder=26)
    draw_energy_gap(ax, -0.46, y_a, y_barrier, -0.96, 0.0, AMBER)
    bfs.curve_label(ax, -0.96, 1.58, "high barrier", AMBER, ha="left", size=9.4)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(y_floor, 2.08)
    ax.set_xticks([])
    ax.set_yticks([])
    bfs.style_axis(ax, xlabel="reaction coordinate", ylabel="free energy", title="Unbiased dynamics")

    ax = axes[1]
    ax.plot(x, u, color=BLUE, linewidth=2.0, alpha=0.44, zorder=5)
    bias = -0.82 * np.exp(-0.5 * ((x - 0.0) / 0.55) ** 2)
    ub = u + bias
    ax.fill_between(x, ub, u, where=bias < 0, color=TEAL_LIGHT, alpha=0.78, zorder=2)
    ax.plot(x, ub, color=TEAL, linewidth=2.7, zorder=9)
    path_x = np.linspace(-0.92, 0.92, 78)
    path_y = np.interp(path_x, x, ub) + 0.14 + 0.035 * np.sin(np.linspace(0, 4.5 * np.pi, 78))
    ax.plot(path_x, path_y, color=RED, linewidth=2.0, zorder=14)
    for i in [14, 33, 52, 67]:
        _arrow(ax, (path_x[i - 2], path_y[i - 2]), (path_x[i + 2], path_y[i + 2]), color=RED, lw=1.7, ms=12)
    y_biased_barrier = np.interp(0.0, x, ub)
    left_biased_min_x = x[np.argmin(np.where(x < 0, ub, np.inf))]
    left_biased_min_y = np.interp(left_biased_min_x, x, ub)
    draw_energy_gap(ax, -0.50, left_biased_min_y, y_biased_barrier, left_biased_min_x, 0.0, TEAL)
    bfs.curve_label(ax, 0.34, 1.34, "original", BLUE, ha="left", size=9.0)
    bfs.curve_label(ax, 0.46, 0.58, "biased", TEAL, ha="left", size=9.0)
    bfs.curve_label(ax, -1.10, 0.78, "lower barrier", TEAL, ha="left", size=9.4)
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(y_floor, 2.08)
    ax.set_xticks([])
    ax.set_yticks([])
    bfs.style_axis(ax, xlabel="reaction coordinate", ylabel="", title="Biased dynamics")

    bfs.save_figure(fig, output_path, dpi=260)


def generate_cv_metadynamics(output_path: Path):
    """Collective variables and metadynamics biasing."""
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.55), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    rng = np.random.default_rng(7)
    mean_a = np.array([-1.0, -0.45])
    mean_b = np.array([1.05, 0.55])
    cloud_a = rng.normal(mean_a, [0.32, 0.23], size=(80, 2))
    cloud_b = rng.normal(mean_b, [0.34, 0.25], size=(80, 2))
    ax.scatter(cloud_a[:, 0], cloud_a[:, 1], s=18, color=BLUE, alpha=0.62, edgecolors="none")
    ax.scatter(cloud_b[:, 0], cloud_b[:, 1], s=18, color=GREEN, alpha=0.62, edgecolors="none")
    ax.plot([-1.7, 1.7], [-0.95, 0.95], color=AMBER, linewidth=2.0)
    _arrow(ax, (-1.45, -0.82), (1.45, 0.82), color=AMBER, lw=2.0)
    bfs.callout_label(ax, r"$s=\xi(\mathbf{x})$", xy=(0.44, 0.25), xytext=(-0.34, 1.06), color=AMBER, size=10.0, rad=0.06)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    bfs.style_axis(ax, xlabel="coordinates", ylabel="", title="Compress")

    ax = axes[1]
    s = np.linspace(-2.0, 2.0, 600)
    f = 0.95 * (s**2 - 1.0) ** 2 + 0.08 * s
    ax.plot(s, f, color=BLUE, linewidth=2.4)
    ax.fill_between(s, 0, f, color=BLUE_LIGHT, alpha=0.48)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    bfs.style_axis(ax, xlabel=r"CV $s$", ylabel=r"$F(s)$", title="Estimate")
    bfs.direct_label(ax, 0.02, 1.22, r"$F(s)$", BLUE)
    ax.set_yticks([])

    ax = axes[2]
    ax.plot(s, f, color=BLUE, linewidth=1.9, alpha=0.55)
    centers = np.array([-1.0, -0.72, -0.35, 0.02, 0.38])
    heights = np.linspace(0.22, 0.11, len(centers))
    bias = np.zeros_like(s)
    for c, h in zip(centers, heights):
        g = h * np.exp(-0.5 * ((s - c) / 0.18) ** 2)
        bias += g
        ax.fill_between(s, 0, g, color=AMBER_LIGHT, alpha=0.84)
        ax.plot(s, g, color=AMBER, linewidth=1.0, alpha=0.82)
    effective = f + bias
    ax.plot(s, effective, color=TEAL, linewidth=2.4)
    bfs.callout_label(ax, "Gaussian hills", xy=(-0.96, 0.23), xytext=(-1.72, 0.73), color=AMBER, size=8.8, ha="left", rad=-0.16)
    bfs.curve_label(ax, 0.66, 1.18, "free energy", BLUE, size=8.7)
    bfs.curve_label(ax, 0.16, 0.58, "biased", TEAL, size=8.9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.15, 1.45)
    ax.set_yticks([])
    bfs.style_axis(ax, xlabel=r"CV $s$", ylabel="", title="Bias")

    bfs.save_figure(fig, output_path, dpi=250)


def generate_method_map(output_path: Path):
    """Map classical enhanced sampling and ML methods."""
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "md": (0.13, 0.62, "Unbiased MD", BLUE),
        "bias": (0.42, 0.62, "CV bias", AMBER),
        "estimate": (0.72, 0.62, "Corrected estimates", GREEN),
        "cv": (0.32, 0.28, "Learn the CV", TEAL),
        "path": (0.62, 0.28, "Learn path bias", RED),
    }

    def draw_node(key, radius=0.09):
        x0, y0, label, color = nodes[key]
        ax.add_patch(Circle((x0, y0), radius, fc="white", ec=color, lw=1.8, zorder=5))
        ax.text(x0, y0, label, ha="center", va="center", fontsize=10.5, fontweight="semibold", color=color, zorder=8)

    def link_points(start, end, color=bfs.NEUTRAL, curve=0.0, lw=1.5):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                lw=lw,
                color=color,
                connectionstyle=f"arc3,rad={curve}",
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )

    link_points((0.24, 0.62), (0.34, 0.62), color=bfs.MUTED, lw=1.8)
    link_points((0.52, 0.62), (0.62, 0.62), color=bfs.MUTED, lw=1.8)
    link_points((0.39, 0.36), (0.42, 0.52), color=TEAL, curve=-0.1)
    link_points((0.62, 0.38), (0.48, 0.52), color=RED, curve=0.18)

    for key in nodes:
        draw_node(key)

    ax.text(0.27, 0.72, "bias", ha="center", fontsize=9.0, color=bfs.MUTED)
    ax.text(0.57, 0.72, "reweight", ha="center", fontsize=9.0, color=bfs.MUTED)
    ax.text(0.47, 0.1, "path-measure view: Jarzynski, AIS, diffusion models, trajectory objectives", ha="center", fontsize=10.0, color=bfs.MUTED)
    ax.text(0.5, 0.91, "Enhanced sampling changes the sampling distribution", ha="center", fontsize=14, color=TEXT, fontweight="semibold")

    bfs.save_figure(fig, output_path, dpi=230)


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
