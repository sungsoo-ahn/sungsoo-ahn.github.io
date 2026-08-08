#!/usr/bin/env python3
"""Generate original figures for the scalar/vector geometric GNN post.

Provenance and design notes
---------------------------
All four figures are original explanatory schematics. They synthesize the
geometric primitives and update rules discussed in the cited primary papers;
they do not reproduce artwork from the lecture decks or papers. Existing
architecture figures were not reused because this post needs custom toy
geometries and one consistent scalar/vector visual language.

Conceptual sources:
  * Schuett et al. (2017),
    https://proceedings.neurips.cc/paper/2017/hash/
    303ed4c69846ab36c2904d3ba8573050-Abstract.html
  * Gasteiger et al. (2020), https://openreview.net/forum?id=B1eWbxStPH
  * Satorras et al. (2021), https://proceedings.mlr.press/v139/satorras21a.html
  * Schuett et al. (2021), https://proceedings.mlr.press/v139/schutt21a.html

The visual treatment uses simple geometric silhouettes, flat fills, rounded
forms, restrained colors, and consistent strokes. No Flaticon asset is copied
or embedded. Outputs are SVG-first with matching PNG previews. License: same
as the blog.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


bfs.use_blog_style()
OUT = Path("assets/img/blog")


def _title(ax, title, subtitle=None):
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="semibold", color=bfs.TEXT)
    if subtitle:
        ax.text(0.02, 0.89, subtitle, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.3, color=bfs.MUTED)


def _atom(ax, xy, label="", *, face=bfs.PURPLE_LIGHT, edge=bfs.PURPLE,
          radius=0.18, zorder=5):
    ax.add_patch(Circle(xy, radius, facecolor=face, edgecolor=edge,
                        linewidth=2.0, zorder=zorder))
    if label:
        ax.text(*xy, label, ha="center", va="center", fontsize=9.4,
                color=bfs.TEXT, fontweight="semibold", zorder=zorder + 1)


def _bond(ax, a, b, *, color=bfs.SPINE, width=2.3, zorder=2):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=width,
            solid_capstyle="round", zorder=zorder)


def _arrow(ax, a, b, *, color=bfs.BLUE, width=2.0, mutation=14,
           shrink_a=0, shrink_b=0, zorder=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=mutation,
                                linewidth=width, color=color, shrinkA=shrink_a,
                                shrinkB=shrink_b, zorder=zorder))


def generate_geometric_scalars():
    """Show distances, angles, and torsions as invariant scalarizations."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45))
    fig.subplots_adjust(wspace=0.14)

    ax = axes[0]
    _title(ax, "Distance", "two atoms")
    a, b = (0.55, 1.45), (2.45, 1.45)
    _bond(ax, a, b)
    _atom(ax, a, "i"); _atom(ax, b, "j", face=bfs.BLUE_LIGHT, edge=bfs.BLUE)
    ax.annotate("", xy=(2.15, 0.86), xytext=(0.85, 0.86),
                arrowprops={"arrowstyle": "<->", "color": bfs.AMBER, "lw": 1.8})
    ax.text(1.5, 0.65, r"$r_{ij}=\|\mathbf{x}_j-\mathbf{x}_i\|$", ha="center", va="center",
            fontsize=10, color=bfs.AMBER, fontweight="semibold")
    ax.set_xlim(0.15, 2.85); ax.set_ylim(0.2, 2.4); ax.axis("off")

    ax = axes[1]
    _title(ax, "Bond angle", "three atoms")
    c, a, b = (1.45, 1.12), (0.45, 2.02), (2.58, 1.82)
    _bond(ax, c, a); _bond(ax, c, b)
    _atom(ax, c, "j", face=bfs.AMBER_LIGHT, edge=bfs.AMBER)
    _atom(ax, a, "i"); _atom(ax, b, "k", face=bfs.BLUE_LIGHT, edge=bfs.BLUE)
    ax.add_patch(Arc(c, 0.95, 0.95, angle=0, theta1=34, theta2=138,
                     color=bfs.TEAL, linewidth=2.0))
    ax.text(1.5, 1.65, "θ", ha="center", va="center", fontsize=12,
            color=bfs.TEAL, fontweight="bold")
    ax.text(1.5, 0.45, r"$\cos\theta=\hat{\mathbf{r}}_{ji}\cdot\hat{\mathbf{r}}_{jk}$", ha="center", va="center",
            fontsize=9.8, color=bfs.TEAL, fontweight="semibold")
    ax.set_xlim(0.1, 2.9); ax.set_ylim(0.15, 2.55); ax.axis("off")

    ax = axes[2]
    _title(ax, "Torsion", "four atoms and two planes")
    pts = [(0.35, 1.78), (1.15, 1.22), (1.95, 1.7), (2.72, 1.0)]
    for p, q in zip(pts[:-1], pts[1:]): _bond(ax, p, q)
    colors = [(bfs.PURPLE_LIGHT, bfs.PURPLE), (bfs.AMBER_LIGHT, bfs.AMBER),
              (bfs.AMBER_LIGHT, bfs.AMBER), (bfs.BLUE_LIGHT, bfs.BLUE)]
    for label, point, (face, edge) in zip("ijkl", pts, colors):
        _atom(ax, point, label, face=face, edge=edge, radius=0.16)
    ax.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]],
            color=bfs.PURPLE, linewidth=1.0, linestyle="--")
    ax.plot([pts[1][0], pts[3][0]], [pts[1][1], pts[3][1]],
            color=bfs.BLUE, linewidth=1.0, linestyle="--")
    ax.text(1.55, 0.47, r"$\phi$ compares the two bond planes", ha="center", va="center",
            fontsize=9.5, color=bfs.MUTED)
    ax.set_xlim(0.05, 3.05); ax.set_ylim(0.15, 2.55); ax.axis("off")

    fig.text(0.5, 0.015,
             "These numbers survive every translation, rotation, and reflection.",
             ha="center", va="bottom", fontsize=10.2, color=bfs.TEXT,
             fontweight="semibold")
    bfs.save_svg_png(fig, OUT / "svgnn_geometric_scalars.svg")


def generate_directional_ambiguity():
    """Work through two neighborhoods with equal radii but different angle."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.85))
    fig.subplots_adjust(wspace=0.18)

    configs = [
        (axes[0], np.deg2rad(60), "Bent neighborhood", r"$\theta=60^\circ,\;\|\mathbf{r}_1+\mathbf{r}_2\|=\sqrt{3}$"),
        (axes[1], np.deg2rad(180), "Linear neighborhood", r"$\theta=180^\circ,\;\|\mathbf{r}_1+\mathbf{r}_2\|=0$"),
    ]
    for ax, theta, title, result in configs:
        _title(ax, title, "both center–neighbor distances equal 1")
        center = np.array([1.75, 1.25])
        p1 = center + np.array([1.0, 0.0])
        p2 = center + np.array([np.cos(theta), np.sin(theta)])
        _bond(ax, center, p1); _bond(ax, center, p2)
        _atom(ax, center, "i", face=bfs.AMBER_LIGHT, edge=bfs.AMBER, radius=0.2)
        _atom(ax, p1, "j", face=bfs.BLUE_LIGHT, edge=bfs.BLUE)
        _atom(ax, p2, "k", face=bfs.PURPLE_LIGHT, edge=bfs.PURPLE)
        resultant = (p1 - center) + (p2 - center)
        if np.linalg.norm(resultant) > 1e-5:
            _arrow(ax, center, center + 0.72 * resultant, color=bfs.TEAL,
                   width=2.5, mutation=15, shrink_a=10)
        else:
            ax.add_patch(Circle(center, 0.34, fill=False, edgecolor=bfs.TEAL,
                                linewidth=2.0, linestyle="--"))
        ax.text(1.75, 0.34, result, ha="center", va="center", fontsize=9.8,
                color=bfs.TEAL, fontweight="semibold")
        ax.set_xlim(0.3, 3.2); ax.set_ylim(0.05, 2.9); ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.01,
             "A radial one-layer update sees the same multiset {1, 1}; an angle or vector sum separates the structures.",
             ha="center", va="bottom", fontsize=9.9, color=bfs.RED,
             fontweight="semibold")
    bfs.save_svg_png(fig, OUT / "svgnn_directional_ambiguity.svg")


def generate_egnn_coordinate_update():
    """Illustrate an EGNN coordinate update as weighted relative vectors."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.95),
                             gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.subplots_adjust(wspace=0.16)
    ax = axes[0]
    _title(ax, "Invariant edge messages", "features + squared distance")
    center = (1.55, 1.35)
    neigh = [(0.52, 2.25), (2.7, 2.05), (0.58, 0.5)]
    colors = [(bfs.PURPLE_LIGHT, bfs.PURPLE), (bfs.BLUE_LIGHT, bfs.BLUE),
              (bfs.TEAL_LIGHT, bfs.TEAL)]
    weights = ["+0.8", "+0.3", "−0.2"]
    for p, (face, edge), w in zip(neigh, colors, weights):
        _bond(ax, center, p)
        _atom(ax, p, face=face, edge=edge)
        mid = ((center[0] + p[0]) / 2, (center[1] + p[1]) / 2)
        ax.text(mid[0], mid[1] + 0.15, w, fontsize=8.8, color=edge,
                ha="center", va="center", fontweight="semibold")
    _atom(ax, center, "i", face=bfs.AMBER_LIGHT, edge=bfs.AMBER, radius=0.21)
    ax.text(1.58, 0.1, r"$m_{ij}=\phi_e(\mathbf{h}_i,\mathbf{h}_j,\|\mathbf{x}_i-\mathbf{x}_j\|^2)$", ha="center", va="center",
            fontsize=9.5, color=bfs.TEXT)
    ax.set_xlim(0.05, 3.2); ax.set_ylim(-0.05, 2.95); ax.axis("off")

    ax = axes[1]
    _title(ax, "Equivariant displacement", "invariant weights scale relative vectors")
    origin = np.array([1.25, 1.32])
    vecs = [np.array([-0.75, 0.68]), np.array([0.95, 0.52]), np.array([-0.7, -0.65])]
    coeffs = [0.8, 0.3, -0.2]
    for vec, coeff, (_, edge) in zip(vecs, coeffs, colors):
        _arrow(ax, origin, origin + vec, color=edge, width=1.8, mutation=12,
               shrink_a=9)
        _arrow(ax, origin, origin + coeff * vec, color=edge, width=3.0,
               mutation=14, shrink_a=9)
    delta = sum(c * v for c, v in zip(coeffs, vecs))
    new = origin + delta
    _atom(ax, origin, r"$\mathbf{x}_i$", face=bfs.AMBER_LIGHT, edge=bfs.AMBER, radius=0.2)
    _arrow(ax, origin, new, color=bfs.RED, width=3.0, mutation=16, shrink_a=12)
    _atom(ax, new, r"$\mathbf{x}'_i$", face=bfs.RED_LIGHT, edge=bfs.RED, radius=0.2)
    ax.text(2.1, 0.3, r"$\Delta\mathbf{x}_i=\sum_j(\mathbf{x}_i-\mathbf{x}_j)\phi_x(m_{ij})$", ha="center", va="center",
            fontsize=9.5, color=bfs.RED, fontweight="semibold")
    ax.set_xlim(0.0, 3.45); ax.set_ylim(-0.05, 2.95); ax.axis("off")
    bfs.save_svg_png(fig, OUT / "svgnn_egnn_coordinate_update.svg")


def generate_scalar_vector_channels():
    """Contrast scalar channels with vector channels under rotation."""
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.6))
    fig.subplots_adjust(wspace=0.16)

    ax = axes[0]
    _title(ax, "Scalar channel", "unchanged by rotation")
    card = FancyBboxPatch((0.48, 0.72), 2.1, 1.25,
                          boxstyle="round,pad=0.1,rounding_size=0.16",
                          facecolor=bfs.PURPLE_LIGHT, edgecolor=bfs.PURPLE,
                          linewidth=1.8)
    ax.add_patch(card)
    ax.text(1.53, 1.55, r"$\mathbf{s}_i$", ha="center", va="center", fontsize=17,
            color=bfs.PURPLE, fontweight="bold")
    ax.text(1.53, 1.1, "charge · energy · atom type", ha="center", va="center",
            fontsize=9.2, color=bfs.MUTED)
    ax.text(1.53, 0.38, r"$\mathbf{s}_i\mapsto\mathbf{s}_i$", ha="center", va="center", fontsize=10,
            color=bfs.TEXT, fontweight="semibold")
    ax.set_xlim(0.15, 2.9); ax.set_ylim(0.1, 2.6); ax.axis("off")

    ax = axes[1]
    _title(ax, "Vector channel", "rotates with the geometry")
    center = (1.45, 1.28)
    _atom(ax, center, face=bfs.BLUE_LIGHT, edge=bfs.BLUE, radius=0.2)
    _arrow(ax, center, (2.35, 2.0), color=bfs.BLUE, width=3.0,
           mutation=16, shrink_a=10)
    ax.add_patch(Arc(center, 1.55, 1.55, theta1=20, theta2=72,
                     color=bfs.AMBER, linewidth=1.8))
    ax.text(2.18, 1.17, "Q", fontsize=10, color=bfs.AMBER, fontweight="bold")
    ax.text(1.5, 0.48, r"$\mathbf{v}_i\mapsto\mathbf{Q}\mathbf{v}_i$", ha="center", va="center", fontsize=10,
            color=bfs.TEXT, fontweight="semibold")
    ax.set_xlim(0.15, 2.9); ax.set_ylim(0.1, 2.6); ax.axis("off")

    ax = axes[2]
    _title(ax, "Safe interactions", "invariants gate equivariant vectors")
    boxes = [(0.25, 1.5, r"$\|\mathbf{v}_i\|$", bfs.TEAL_LIGHT, bfs.TEAL),
             (0.25, 0.65, r"$\mathbf{v}_i\cdot\mathbf{v}_j$", bfs.AMBER_LIGHT, bfs.AMBER),
             (1.83, 1.05, r"$g(\mathbf{s})\mathbf{v}_i$", bfs.BLUE_LIGHT, bfs.BLUE)]
    for x, y, label, face, edge in boxes:
        box = FancyBboxPatch((x, y), 1.05, 0.52, boxstyle="round,pad=0.07",
                             facecolor=face, edgecolor=edge, linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + 0.525, y + 0.26, label, ha="center", va="center",
                fontsize=9.8, color=edge, fontweight="semibold")
    _arrow(ax, (1.35, 1.75), (1.79, 1.43), color=bfs.MUTED, width=1.5)
    _arrow(ax, (1.35, 0.9), (1.79, 1.2), color=bfs.MUTED, width=1.5)
    ax.text(1.5, 0.25, "no componentwise ReLU on vectors", ha="center", va="center",
            fontsize=9.1, color=bfs.RED)
    ax.set_xlim(0.05, 3.05); ax.set_ylim(0.05, 2.6); ax.axis("off")
    bfs.save_svg_png(fig, OUT / "svgnn_scalar_vector_channels.svg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generate_geometric_scalars()
    generate_directional_ambiguity()
    generate_egnn_coordinate_update()
    generate_scalar_vector_channels()


if __name__ == "__main__":
    main()
