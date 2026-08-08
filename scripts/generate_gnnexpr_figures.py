#!/usr/bin/env python3
"""Generate original figures for the GNN expressivity blog post.

Provenance and design notes
---------------------------
All four figures are original explanatory schematics. They do not reproduce
artwork from the lecture decks or cited papers. The concepts follow the
1-WL/MPNN correspondence and the higher-order and subgraph remedies developed
in the sources listed below. The visual treatment uses simple rounded forms,
flat fills, restrained colors, and consistent strokes; no Flaticon asset is
copied or embedded.

Conceptual sources:
  * Xu et al. (2019), https://openreview.net/forum?id=ryGs6iA5Km
  * Morris et al. (2019), https://ojs.aaai.org/index.php/AAAI/article/view/4384
  * Bouritsas et al. (2021), https://openreview.net/forum?id=LT0gkQt1h7
  * Bevilacqua et al. (2022), https://openreview.net/forum?id=6buz2fR0Nw9

Outputs are SVG-first with matching PNG previews. License: same as the blog.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


bfs.use_blog_style()
OUT = Path("assets/img/blog")


def _node(ax, xy, *, face=bfs.PURPLE_LIGHT, edge=bfs.PURPLE, radius=0.18,
          label="", zorder=5):
    ax.add_patch(Circle(xy, radius, facecolor=face, edgecolor=edge,
                        linewidth=2.0, zorder=zorder))
    if label:
        ax.text(*xy, label, ha="center", va="center", fontsize=9.5,
                fontweight="semibold", color=bfs.TEXT, zorder=zorder + 1)


def _edge(ax, a, b, *, color=bfs.SPINE, width=2.2, zorder=2):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=width,
            solid_capstyle="round", zorder=zorder)


def _arrow(ax, a, b, *, color=bfs.MUTED, width=1.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                linewidth=width, color=color, zorder=3))


def _title(ax, title, subtitle=None):
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="semibold", color=bfs.TEXT)
    if subtitle:
        ax.text(0.02, 0.89, subtitle, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.3, color=bfs.MUTED)


def _draw_graph(ax, positions, edges, *, colors=None, labels=None,
                radius=0.18, edge_color=bfs.SPINE):
    for u, v in edges:
        _edge(ax, positions[u], positions[v], color=edge_color)
    for node, xy in positions.items():
        face, edge = (colors or {}).get(node, (bfs.PURPLE_LIGHT, bfs.PURPLE))
        _node(ax, xy, face=face, edge=edge, radius=radius,
              label=(labels or {}).get(node, ""))


def generate_representation_equivalence():
    """Show invariance to relabeling and the desired separation of objects."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55))
    fig.subplots_adjust(wspace=0.18)

    pos = {0: (0.45, 1.75), 1: (1.4, 2.55), 2: (2.45, 1.85),
           3: (2.05, 0.65), 4: (0.8, 0.55)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]

    ax = axes[0]
    _title(ax, "Same object, different names", "A graph has no canonical node order")
    _draw_graph(ax, pos, edges, labels={i: str(i + 1) for i in pos})
    ax.set_xlim(0, 2.9); ax.set_ylim(0.05, 3.25); ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    _title(ax, "Relabeling must preserve the code", "Permutation invariance is non-negotiable")
    labels = {0: "d", 1: "a", 2: "e", 3: "b", 4: "c"}
    _draw_graph(ax, pos, edges, labels=labels,
                colors={i: (bfs.BLUE_LIGHT, bfs.BLUE) for i in pos})
    ax.text(1.43, 0.08, "same graph representation", ha="center", va="center",
            fontsize=9.2, color=bfs.TEAL, fontweight="semibold")
    ax.set_xlim(0, 2.9); ax.set_ylim(0.0, 3.25); ax.set_aspect("equal"); ax.axis("off")

    ax = axes[2]
    _title(ax, "Different objects should separate", "Expressivity asks when they do not")
    p2 = {0: (0.45, 2.1), 1: (1.35, 2.55), 2: (2.25, 2.1),
          3: (2.25, 1.1), 4: (1.35, 0.65), 5: (0.45, 1.1)}
    e2 = [(i, (i + 1) % 6) for i in range(6)]
    _draw_graph(ax, p2, e2, radius=0.15,
                colors={i: (bfs.AMBER_LIGHT, bfs.AMBER) for i in p2})
    ax.text(1.35, 0.15, "distinct graph  →  distinct code?", ha="center", va="center",
            fontsize=9.5, color=bfs.AMBER, fontweight="semibold")
    ax.set_xlim(0, 2.8); ax.set_ylim(-0.05, 3.25); ax.set_aspect("equal"); ax.axis("off")

    bfs.save_svg_png(fig, OUT / "gnnexpr_representation_equivalence.svg")


def generate_wl_refinement():
    """Work through three rounds of 1-WL on a five-node path."""
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.25))
    fig.subplots_adjust(wspace=0.15)
    pos = {i: (0.42 + 0.65 * i, 1.45) for i in range(5)}
    edges = [(i, i + 1) for i in range(4)]
    palettes = [
        {i: (bfs.PURPLE_LIGHT, bfs.PURPLE) for i in pos},
        {0: (bfs.AMBER_LIGHT, bfs.AMBER), 1: (bfs.BLUE_LIGHT, bfs.BLUE),
         2: (bfs.BLUE_LIGHT, bfs.BLUE), 3: (bfs.BLUE_LIGHT, bfs.BLUE),
         4: (bfs.AMBER_LIGHT, bfs.AMBER)},
        {0: (bfs.AMBER_LIGHT, bfs.AMBER), 1: (bfs.BLUE_LIGHT, bfs.BLUE),
         2: (bfs.TEAL_LIGHT, bfs.TEAL), 3: (bfs.BLUE_LIGHT, bfs.BLUE),
         4: (bfs.AMBER_LIGHT, bfs.AMBER)},
    ]
    titles = [("Round 0", "Every node starts with color a"),
              ("Round 1", "Endpoints see {a}; others see {a, a}"),
              ("Round 2", "The center now sees {b, b}")]
    labels = [{i: "a" for i in pos},
              {0: "c", 1: "b", 2: "b", 3: "b", 4: "c"},
              {0: "e", 1: "d", 2: "f", 3: "d", 4: "e"}]
    for ax, colors, (title, subtitle), node_labels in zip(axes, palettes, titles, labels):
        _title(ax, title, subtitle)
        _draw_graph(ax, pos, edges, colors=colors, labels=node_labels, radius=0.19)
        ax.set_xlim(0.05, 3.4); ax.set_ylim(0.45, 2.45); ax.axis("off")
    bfs.save_svg_png(fig, OUT / "gnnexpr_wl_refinement.svg")


def generate_regular_collision():
    """Show the canonical 1-WL collision C6 versus two disjoint triangles."""
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.75))
    fig.subplots_adjust(wspace=0.18)

    ax = axes[0]
    _title(ax, "One six-cycle", "Connected; no triangles")
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    pos = {i: (1.7 + 1.05 * np.cos(a), 1.45 + 1.05 * np.sin(a)) for i, a in enumerate(angles)}
    edges = [(i, (i + 1) % 6) for i in range(6)]
    _draw_graph(ax, pos, edges, labels={i: "a" for i in pos}, radius=0.17)
    ax.text(1.7, 0.05, "six nodes, every degree = 2", ha="center", va="center",
            fontsize=9.5, color=bfs.MUTED)
    ax.set_xlim(0.2, 3.2); ax.set_ylim(-0.15, 3.15); ax.set_aspect("equal"); ax.axis("off")

    ax = axes[1]
    _title(ax, "Two three-cycles", "Disconnected; two triangles")
    pos2 = {}
    for offset, center in [(0, (0.85, 1.5)), (3, (2.72, 1.5))]:
        tri = [(center[0], center[1] + 0.82),
               (center[0] - 0.7, center[1] - 0.45),
               (center[0] + 0.7, center[1] - 0.45)]
        for j, xy in enumerate(tri): pos2[offset + j] = xy
    edges2 = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    _draw_graph(ax, pos2, edges2, labels={i: "a" for i in pos2}, radius=0.17,
                colors={i: (bfs.BLUE_LIGHT, bfs.BLUE) for i in pos2})
    ax.text(1.78, 0.05, "six nodes, every degree = 2", ha="center", va="center",
            fontsize=9.5, color=bfs.MUTED)
    ax.set_xlim(-0.08, 3.68); ax.set_ylim(-0.15, 3.15); ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.01, "1-WL keeps one color forever, so the color histograms remain identical.",
             ha="center", va="bottom", fontsize=10.2, color=bfs.RED, fontweight="semibold")
    bfs.save_svg_png(fig, OUT / "gnnexpr_regular_collision.svg")


def generate_remedy_tradeoff():
    """Organize remedies by the information added and its computational price."""
    fig, ax = plt.subplots(figsize=(10.4, 4.25))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0.2, 4.02, "Every remedy gives the model a larger unit of comparison",
            fontsize=12.5, fontweight="semibold", color=bfs.TEXT, ha="left", va="top")
    ax.text(0.2, 3.68, "The extra information is useful only when it matches the task.",
            fontsize=9.8, color=bfs.MUTED, ha="left", va="top")

    cards = [
        (0.2, "Node tuples", "track relations\namong k nodes", bfs.PURPLE_LIGHT, bfs.PURPLE, "k-WL"),
        (2.75, "Motif counts", "attach cycles\nor patterns", bfs.AMBER_LIGHT, bfs.AMBER, "3"),
        (5.3, "Subgraph views", "run a model on\nperturbed graphs", bfs.BLUE_LIGHT, bfs.BLUE, "S"),
        (7.85, "Positions", "add spectral or\ndistance coordinates", bfs.TEAL_LIGHT, bfs.TEAL, "eig"),
    ]
    for x, title, desc, face, edge, glyph in cards:
        card = FancyBboxPatch((x, 1.25), 2.25, 1.85, boxstyle="round,pad=0.12,rounding_size=0.16",
                              facecolor=face, edgecolor=edge, linewidth=1.8)
        ax.add_patch(card)
        ax.add_patch(Circle((x + 0.42, 2.62), 0.25, facecolor="white", edgecolor=edge, linewidth=1.7))
        ax.text(x + 0.42, 2.62, glyph, ha="center", va="center", fontsize=12,
                fontweight="bold", color=edge)
        ax.text(x + 0.25, 2.2, title, fontsize=10.5, fontweight="semibold", color=bfs.TEXT)
        ax.text(x + 0.25, 1.86, desc, fontsize=8.9, color=bfs.MUTED,
                ha="left", va="top", linespacing=1.3)
    ax.plot([0.55, 9.8], [0.6, 0.6], color=bfs.SPINE, linewidth=2.0,
            solid_capstyle="round")
    ax.text(5.2, 0.27,
            "Different costs: tuple states · motif matching · repeated views · coordinate construction",
            ha="center", va="center", fontsize=9.4, color=bfs.MUTED)
    bfs.save_svg_png(fig, OUT / "gnnexpr_remedy_tradeoff.svg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generate_representation_equivalence()
    generate_wl_refinement()
    generate_regular_collision()
    generate_remedy_tradeoff()


if __name__ == "__main__":
    main()
