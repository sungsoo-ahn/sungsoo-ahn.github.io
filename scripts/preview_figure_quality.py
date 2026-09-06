#!/usr/bin/env python3
"""Generate analytic QA examples outside site assets; requires an output directory."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import blog_figure_style as bfs


def curve():
    fig, ax = plt.subplots(figsize=(bfs.COLUMN, 2.3), layout="constrained")
    t = np.linspace(0, 4, 160)
    ax.plot(t, np.exp(-t), color=bfs.PURPLE, label="Fast decay")
    ax.plot(t, np.exp(-t / 2), color=bfs.TEAL, linestyle="--", label="Slow decay")
    ax.set(xlabel="Time (s)", ylabel="Relative concentration", ylim=(0, 1.05))
    bfs.clean_axes(ax)
    ax.legend(frameon=False)
    return fig


def panels():
    fig, axes = plt.subplots(1, 2, figsize=(bfs.WIDE, 2.5), layout="constrained", sharey=True)
    t = np.linspace(0, 4, 160)
    for ax, rate, label in zip(axes, (1, .5), ("(a) Fast decay", "(b) Slow decay")):
        ax.plot(t, np.exp(-rate * t), color=bfs.PURPLE)
        ax.set(xlabel="Elapsed time after the\ninitial perturbation (s)", ylim=(0, 1.05))
        ax.set_title(label, loc="left", pad=8)
        bfs.clean_axes(ax)
    axes[0].set_ylabel("Relative concentration")
    return fig


def heatmap():
    fig, ax = plt.subplots(figsize=(bfs.COLUMN, 2.6), layout="constrained")
    x = np.linspace(-2, 2, 80)
    z = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2))
    field = ax.imshow(z, extent=(-2, 2, -2, 2), origin="lower", cmap="Purples", vmin=0, vmax=1)
    ax.set(xlabel="Horizontal position", ylabel="Vertical position")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    fig.colorbar(field, ax=ax, label="Relative intensity", shrink=.9, pad=.04)
    return fig


def diagram():
    fig, ax = plt.subplots(figsize=(bfs.TEXT_WIDTH, 1.7), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    boxes = [(.025, "Initial\nconcentration"), (.385, "Exponential\ndecay"), (.745, "Remaining\nconcentration")]
    for left, label in boxes:
        ax.add_patch(FancyBboxPatch((left, .3), .23, .42,
                                   boxstyle="round,pad=0.008,rounding_size=0.02",
                                   linewidth=.7, edgecolor=bfs.PURPLE, facecolor=bfs.PURPLE_LIGHT))
        ax.text(left + .115, .51, label, ha="center", va="center", fontsize=8.5)
    for left, right in ((.267, .369), (.627, .729)):
        ax.annotate("", xy=(right, .51), xytext=(left, .51),
                    arrowprops={"arrowstyle": "->", "lw": 1, "color": bfs.PURPLE_STRONG})
    return fig


EXAMPLES = (
    ("curve", curve, "Analytic curves: exp(-t) and exp(-t/2), not experimental measurements."),
    ("panels", panels, "Shared scales and reflowed long labels at a two-column width."),
    ("heatmap", heatmap, "Analytic Gaussian field with a separate colorbar; not measured data."),
    ("diagram", diagram, "Editable vector diagram with deliberate text and arrow spacing."),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bfs.use_publication_style()
    findings, cards = {}, []
    for name, create, caption in EXAMPLES:
        fig = create()
        width = float(fig.get_size_inches()[0])
        issues = bfs.audit_figure(fig, target_width_in=width)
        findings[name] = [{"code": issue.code, "message": issue.message} for issue in issues]
        bfs.save_publication_figure(fig, args.output_dir / name, target_width_in=width)
        cards.append(f'<figure><img src="{name}.svg" alt="{html.escape(caption)}" '
                     f'style="width:{width * 96:g}px"><figcaption>{html.escape(caption)}</figcaption></figure>')
    (args.output_dir / "audit.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Figure quality fixtures</title><style>'
        'body{font:16px/1.5 system-ui;color:#231533;margin:32px auto;padding:0 20px;max-width:900px}'
        'figure{margin:32px 0}img{display:block;max-width:100%;height:auto}'
        'figcaption{margin-top:10px;max-width:65ch;color:#665A75}</style>'
        '<h1>Figure quality fixtures</h1><p>Analytic examples for visual review, not research results.</p>'
        + "".join(cards) + "</html>\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
