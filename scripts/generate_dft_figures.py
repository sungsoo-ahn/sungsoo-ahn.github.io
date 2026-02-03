"""
Generate figures for the quantum chemistry / DFT blog post.

SCF Loop Diagram — rectangular flowchart (landscape).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# --- Refined color palette ---
TEXT_COLOR = '#263238'
ARROW_COLOR = '#455a64'

# Main loop boxes: soft slate-blue
BOX_MAIN = '#dce8f4'
EDGE_MAIN = '#5b7fa5'

# Entry box: lighter, muted
BOX_ENTRY = '#eef2f7'
EDGE_ENTRY = '#8da4be'

# Decision box: warm amber
BOX_DECISION = '#fff3e0'
EDGE_DECISION = '#e8a030'

# Output box: soft green
BOX_OUTPUT = '#e0f2e9'
EDGE_OUTPUT = '#4caf50'

# Feedback arrow
COLOR_NO = '#d32f2f'
COLOR_YES = '#388e3c'


def generate_scf_loop_figure(output_path):
    """
    Classic SCF loop as a landscape flowchart.

    Layout (grid-aligned):
      Initial guess → Build density → Construct F(P)
                           ↑  No                ↓
                      Converged?  → Output   Solve FC=SCε
                           ↑                    ↓
                      New orbitals  ←───────────┘
    """
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(-9.5, 8)
    ax.set_ylim(-2.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    rounding = 0.15
    gap = 0.06  # small space between box edge and arrow tip

    # --- Uniform dimensions ---
    BW = 3.2       # width for all loop boxes
    BH = 0.7       # height for all boxes
    BW_ENTRY = 2.8  # entry box slightly narrower
    BW_DEC = 2.2    # decision box

    # --- Grid positions ---
    LEFT_X = -2.5
    RIGHT_X = 3.5
    TOP_Y = 2.0
    BOT_Y = -0.8
    MID_Y = 0.6
    ENTRY_X = -7.2

    all_data = [
        # (label, cx, cy, bw, bh, box_type)
        (r'Initial guess $\mathbf{C}^{(0)}$',
         ENTRY_X, TOP_Y, BW_ENTRY, BH, 'entry'),

        (r'Build density $\mathbf{P} = \mathbf{CC}^\top$',
         LEFT_X, TOP_Y, BW, BH, 'main'),

        (r'Construct $\mathbf{F}(\mathbf{P})$',
         RIGHT_X, TOP_Y, BW, BH, 'main'),

        (r'Solve $\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$',
         RIGHT_X, BOT_Y, BW, BH, 'main'),

        (r'New orbitals $\mathbf{C}$',
         LEFT_X, BOT_Y, BW, BH, 'main'),

        ('Converged?',
         LEFT_X, MID_Y, BW_DEC, BH, 'decision'),
    ]

    centers = [(cx, cy) for _, cx, cy, _, _, _ in all_data]
    dims = [(bw, bh) for _, _, _, bw, bh, _ in all_data]

    # --- Edge helpers: exact box-edge positions (accounts for rounding) ---
    def right_edge(cx, bw):
        return cx + bw / 2 + rounding + gap

    def left_edge(cx, bw):
        return cx - bw / 2 - rounding - gap

    def top_edge(cy, bh):
        return cy + bh / 2 + rounding + gap

    def bot_edge(cy, bh):
        return cy - bh / 2 - rounding - gap

    def draw_arrow(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.6, **kw):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='-|>', color=color,
            linewidth=lw, mutation_scale=14, zorder=2, **kw)
        ax.add_patch(a)

    # --- Color mapping ---
    style_map = {
        'entry':    (BOX_ENTRY,    EDGE_ENTRY),
        'main':     (BOX_MAIN,     EDGE_MAIN),
        'decision': (BOX_DECISION, EDGE_DECISION),
    }

    # --- Draw boxes ---
    for label, cx, cy, bw, bh, btype in all_data:
        fc, ec = style_map[btype]
        box = FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle=f'round,pad={rounding}',
            facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=10.5, color=TEXT_COLOR, fontweight='bold', zorder=4)

    # --- Step number badges (1–5 on loop boxes, indices 1–5) ---
    for i in range(5):
        cx, cy = centers[i + 1]
        bw, _ = dims[i + 1]
        nx = cx - bw / 2 + 0.02
        ny = cy + BH / 2 + 0.20
        badge_color = EDGE_DECISION if i == 4 else EDGE_MAIN
        circle = plt.Circle((nx, ny), 0.17, color=badge_color, zorder=5)
        ax.add_patch(circle)
        ax.text(nx, ny, str(i + 1), ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', zorder=6)

    # --- Explicit arrow routing (strictly horizontal / vertical) ---
    # 0→1: Initial guess → Build density (horizontal, same y=TOP_Y)
    draw_arrow(right_edge(ENTRY_X, BW_ENTRY), TOP_Y,
               left_edge(LEFT_X, BW), TOP_Y)

    # 1→2: Build density → Construct F(P) (horizontal, same y=TOP_Y)
    draw_arrow(right_edge(LEFT_X, BW), TOP_Y,
               left_edge(RIGHT_X, BW), TOP_Y)

    # 2→3: Construct F(P) → Solve (vertical down, same x=RIGHT_X)
    draw_arrow(RIGHT_X, bot_edge(TOP_Y, BH),
               RIGHT_X, top_edge(BOT_Y, BH))

    # 3→4: Solve → New orbitals (horizontal left, same y=BOT_Y)
    draw_arrow(left_edge(RIGHT_X, BW), BOT_Y,
               right_edge(LEFT_X, BW), BOT_Y)

    # 4→5: New orbitals → Converged? (vertical up, same x=LEFT_X)
    draw_arrow(LEFT_X, top_edge(BOT_Y, BH),
               LEFT_X, bot_edge(MID_Y, BH))

    # --- "No" arrow: Converged?(5) → Build density(1) (vertical, offset right) ---
    no_x = LEFT_X + 0.35
    draw_arrow(no_x, top_edge(MID_Y, BH),
               no_x, bot_edge(TOP_Y, BH),
               color=COLOR_NO, lw=1.8)
    ax.text(no_x + 0.40, (MID_Y + TOP_Y) / 2, 'No',
            ha='left', va='center', fontsize=11,
            color=COLOR_NO, fontweight='bold')

    # --- "Yes" arrow: Converged?(5) → output box (horizontal right) ---
    output_cx = 1.5
    output_cy = MID_Y
    output_w = 3.0
    output_h = BH

    output_box = FancyBboxPatch(
        (output_cx - output_w / 2, output_cy - output_h / 2),
        output_w, output_h,
        boxstyle=f'round,pad={rounding}',
        facecolor=BOX_OUTPUT, edgecolor=EDGE_OUTPUT,
        linewidth=1.8, zorder=3)
    ax.add_patch(output_box)
    ax.text(output_cx, output_cy, 'Converged energy & density',
            ha='center', va='center', fontsize=10,
            color='#2E7D32', fontweight='bold', zorder=4)

    draw_arrow(right_edge(LEFT_X, BW_DEC), MID_Y,
               left_edge(output_cx, output_w), MID_Y,
               color=COLOR_YES, lw=1.6)
    mid_yes_x = (right_edge(LEFT_X, BW_DEC) + left_edge(output_cx, output_w)) / 2
    ax.text(mid_yes_x, MID_Y - 0.32, 'Yes',
            ha='center', va='center', fontsize=11,
            color=COLOR_YES, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved SCF loop figure to {output_path}")


if __name__ == '__main__':
    import os

    output_dir = 'assets/img/blog'
    os.makedirs(output_dir, exist_ok=True)

    generate_scf_loop_figure(os.path.join(output_dir, 'scf_loop.png'))

    print("Done!")
