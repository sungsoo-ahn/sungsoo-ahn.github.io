#!/usr/bin/env python3
"""Generate original figures for the message-passing GNN blog post.

Provenance and design notes
---------------------------
All three figures are original explanatory schematics drawn for this post; they
do not reproduce artwork from the lecture decks or papers. Existing canonical
architecture figures were not reused because the post needs a single visual
language across several architectures and a custom numerical toy graph.

Conceptual sources:
  * Gilmer et al. (2017), https://proceedings.mlr.press/v70/gilmer17a.html
  * Xu et al. (2019), https://openreview.net/forum?id=ryGs6iA5Km
  * Ying et al. (2021),
    https://proceedings.neurips.cc/paper_files/paper/2021/hash/
    f1c1592588411002af340cbaedd6fc33-Abstract.html

Outputs are SVG-first with matching PNG previews. License: same as the blog.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blog_figure_style as bfs


bfs.use_blog_style()
OUT = Path("assets/img/blog")


def _node(ax, xy, label, *, face=bfs.PURPLE_LIGHT, edge=bfs.PURPLE, radius=0.23,
          text=bfs.TEXT, zorder=5):
    circle = Circle(xy, radius, facecolor=face, edgecolor=edge, linewidth=1.8,
                    zorder=zorder)
    ax.add_patch(circle)
    ax.text(*xy, label, ha="center", va="center", fontsize=10.5,
            fontweight="semibold", color=text, zorder=zorder + 1)


def _edge(ax, a, b, *, color=bfs.SPINE, width=2.0, arrow=False, zorder=2):
    if arrow:
        patch = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                linewidth=width, color=color, shrinkA=17,
                                shrinkB=17, zorder=zorder)
        ax.add_patch(patch)
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=width,
                solid_capstyle="round", zorder=zorder)


def _panel_title(ax, title, subtitle=None):
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="semibold", color=bfs.TEXT)
    if subtitle:
        ax.text(0.02, 0.89, subtitle, transform=ax.transAxes, ha="left", va="top",
                fontsize=9.5, color=bfs.MUTED)


def generate_grid_to_graph():
    """Contrast fixed coordinate slots with a graph's arbitrary node order."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    fig.subplots_adjust(wspace=0.18)

    ax = axes[0]
    _panel_title(ax, "A grid supplies coordinates", "Each location has a fixed offset")
    for i in range(4):
        for j in range(4):
            color = bfs.PURPLE_LIGHT if (i, j) == (2, 2) else "white"
            rect = plt.Rectangle((i, j), 1, 1, facecolor=color,
                                 edgecolor=bfs.SPINE, linewidth=1.2)
            ax.add_patch(rect)
    ax.scatter([2.5], [2.5], s=110, color=bfs.PURPLE, zorder=5)
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ax.arrow(2.5, 2.5, 0.72 * dx, 0.72 * dy, color=bfs.BLUE,
                 width=0.025, head_width=0.13, length_includes_head=True,
                 zorder=4)
    ax.set_xlim(-0.15, 4.15)
    ax.set_ylim(-0.15, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    positions = {"a": (0.5, 2.6), "b": (1.8, 3.3), "c": (3.0, 2.4),
                 "d": (1.0, 1.0), "e": (2.6, 0.8)}
    edges = [("a", "b"), ("a", "d"), ("b", "c"), ("b", "d"),
             ("c", "e"), ("d", "e")]

    ax = axes[1]
    _panel_title(ax, "A graph supplies relations", "No left, right, or first node")
    for u, v in edges:
        _edge(ax, positions[u], positions[v])
    for name, xy in positions.items():
        _node(ax, xy, name)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0.25, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[2]
    _panel_title(ax, "Relabeling changes storage", "The represented object is unchanged")
    relabel = {"a": "4", "b": "1", "c": "5", "d": "2", "e": "3"}
    for u, v in edges:
        _edge(ax, positions[u], positions[v])
    for name, xy in positions.items():
        _node(ax, xy, relabel[name], face=bfs.BLUE_LIGHT, edge=bfs.BLUE)
    ax.text(1.75, 0.23, "same nodes, same edges, different indices",
            ha="center", va="center", fontsize=9.2, color=bfs.MUTED)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0.05, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    bfs.save_svg_png(fig, OUT / "gnnmp_grid_graph_symmetry.svg")


def generate_toy_message_passing():
    """Show a numerical one-layer update on a five-node graph."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1),
                             gridspec_kw={"width_ratios": [1.05, 1.35]})
    fig.subplots_adjust(wspace=0.18)
    pos = {"u": (0.55, 2.45), "v": (2.0, 2.0), "w": (3.35, 2.65),
           "r": (1.1, 0.55), "s": (3.15, 0.55)}
    values = {"u": 2, "v": 1, "w": 3, "r": 4, "s": 0}
    edges = [("u", "v"), ("v", "w"), ("u", "r"), ("v", "r"),
             ("w", "s")]

    ax = axes[0]
    _panel_title(ax, "Messages into node v", "Scalar input shown inside each node")
    for a, b in edges:
        if "v" in (a, b):
            other = b if a == "v" else a
            _edge(ax, pos[other], pos["v"], color=bfs.BLUE, arrow=True, width=2.2)
        else:
            _edge(ax, pos[a], pos[b], color=bfs.GRID, width=1.6)
    for name, xy in pos.items():
        if name == "v":
            _node(ax, xy, str(values[name]), face=bfs.AMBER_LIGHT,
                  edge=bfs.AMBER, radius=0.28)
            ax.text(xy[0], xy[1] - 0.46, "target v", ha="center", va="top",
                    fontsize=9.2, color=bfs.AMBER)
        else:
            _node(ax, xy, str(values[name]), radius=0.25)
    ax.set_xlim(0, 3.9)
    ax.set_ylim(0.05, 3.25)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    _panel_title(ax, "One layer is a learnable set function")
    steps = [
        (0.15, 2.45, 1.2, 0.62, "messages", "$m_{uv}=2$\n$m_{rv}=4$\n$m_{wv}=3$",
         bfs.BLUE_LIGHT, bfs.BLUE),
        (1.72, 2.45, 1.2, 0.62, "aggregate", "2 + 4 + 3 = 9",
         bfs.PURPLE_LIGHT, bfs.PURPLE),
        (3.29, 2.45, 1.25, 0.62, "update", "hᵥ′ = σ(9 + 1)",
         bfs.AMBER_LIGHT, bfs.AMBER),
    ]
    for x, y, w, h, title, content, face, edge in steps:
        box = FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.08",
                             facecolor=face, edgecolor=edge, linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + w / 2, y - 0.16, title, ha="center", va="center",
                fontsize=9.6, fontweight="semibold", color=edge)
        ax.text(x + w / 2, y - 0.42, content, ha="center", va="center",
                fontsize=9.5, color=bfs.TEXT, linespacing=1.15)
    for x0, x1 in [(1.38, 1.69), (2.95, 3.26)]:
        ax.add_patch(FancyArrowPatch((x0, 2.14), (x1, 2.14), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.6,
                                     color=bfs.MUTED))
    ax.text(2.3, 1.28, "Reordering the three incoming messages does not change the sum.",
            ha="center", va="center", fontsize=10, color=bfs.TEXT)
    ax.text(2.3, 0.87, "After two layers, v can depend on nodes up to two hops away.",
            ha="center", va="center", fontsize=10, color=bfs.MUTED)
    ax.set_xlim(0, 4.75)
    ax.set_ylim(0.45, 3.25)
    ax.axis("off")

    bfs.save_svg_png(fig, OUT / "gnnmp_toy_update.svg")


def generate_local_global_structure():
    """Compare sparse messages, global attention, and structure-aware attention."""
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    fig.subplots_adjust(wspace=0.12)
    pos = [(0.45, 1.1), (1.25, 2.35), (2.05, 1.15), (2.95, 2.35), (3.65, 1.05)]
    graph_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

    ax = axes[0]
    _panel_title(ax, "Local message passing", "O(|E|) communication per layer")
    for i, j in graph_edges:
        _edge(ax, pos[i], pos[j], color=bfs.PURPLE, arrow=True, width=1.8)
        _edge(ax, pos[j], pos[i], color=bfs.PURPLE, arrow=True, width=1.8)
    for i, xy in enumerate(pos):
        _node(ax, xy, str(i + 1), radius=0.2)
    ax.set_xlim(0, 4.1)
    ax.set_ylim(0.45, 3.0)
    ax.axis("off")

    ax = axes[1]
    _panel_title(ax, "Unstructured global attention", "O(|V|²) pairs; topology is absent")
    target = 2
    for i, xy in enumerate(pos):
        if i != target:
            _edge(ax, xy, pos[target], color=bfs.BLUE, arrow=True, width=1.25)
    for i, xy in enumerate(pos):
        _node(ax, xy, str(i + 1), radius=0.2,
              face=bfs.AMBER_LIGHT if i == target else bfs.BLUE_LIGHT,
              edge=bfs.AMBER if i == target else bfs.BLUE)
    ax.set_xlim(0, 4.1)
    ax.set_ylim(0.45, 3.0)
    ax.axis("off")

    ax = axes[2]
    _panel_title(ax, "Structure-aware global attention", "Content scores + graph position")
    for i, xy in enumerate(pos):
        if i != target:
            dist = abs(i - target)
            _edge(ax, xy, pos[target], color=bfs.TEAL, arrow=True,
                  width=2.6 if dist == 1 else 1.0)
    for i, j in graph_edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color=bfs.SPINE, linewidth=1.0, linestyle="--", zorder=1)
    for i, xy in enumerate(pos):
        _node(ax, xy, str(i + 1), radius=0.2,
              face=bfs.AMBER_LIGHT if i == target else bfs.TEAL_LIGHT,
              edge=bfs.AMBER if i == target else bfs.TEAL)
    ax.text(2.05, 0.47, "bias by distance, edge type, or spectral coordinates",
            ha="center", va="center", fontsize=8.8, color=bfs.MUTED)
    ax.set_xlim(0, 4.1)
    ax.set_ylim(0.25, 3.0)
    ax.axis("off")

    bfs.save_svg_png(fig, OUT / "gnnmp_local_global.svg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generate_grid_to_graph()
    generate_toy_message_passing()
    generate_local_global_structure()


if __name__ == "__main__":
    main()
