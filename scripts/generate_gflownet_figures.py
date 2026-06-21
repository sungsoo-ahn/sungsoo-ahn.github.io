"""Generate the native SVG detailed-balance diagram for the GFlowNet post.

The other GFlowNet figures are original PNGs and are intentionally not
generated here.
"""

from __future__ import annotations

from html import escape
from math import hypot
from pathlib import Path

import blog_figure_style as bfs


OUT_DIR = Path("assets/img/blog/gflownet")

TEXT = bfs.TEXT
MUTED = bfs.MUTED
GRAY = "#87949a"
LIGHT_GRAY = "#eef3f5"
BLUE = bfs.BLUE
BLUE_LIGHT = bfs.BLUE_LIGHT
AMBER = bfs.AMBER
AMBER_LIGHT = bfs.AMBER_LIGHT
TEAL = bfs.TEAL
TEAL_LIGHT = bfs.TEAL_LIGHT
RED = bfs.RED
RED_LIGHT = bfs.RED_LIGHT
GREEN = bfs.GREEN
GREEN_LIGHT = bfs.GREEN_LIGHT
WHITE = "#ffffff"


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 18,
        weight: int | str = 500,
        fill: str = TEXT,
        anchor: str = "middle",
        lines: list[str] | None = None,
    ) -> None:
        items = lines if lines is not None else value.split("\n")
        line_height = size * 1.22
        start = y - line_height * (len(items) - 1) / 2
        tspans = []
        for i, item in enumerate(items):
            tspans.append(
                f'<tspan x="{x:.1f}" y="{start + i * line_height:.1f}">{escape(item)}</tspan>'
            )
        self.add(
            f'<text text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{"".join(tspans)}</text>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = WHITE,
        stroke: str = LIGHT_GRAY,
        sw: float = 1.4,
        rx: float = 8,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
        )

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        *,
        fill: str = WHITE,
        stroke: str = TEXT,
        sw: float = 3.0,
    ) -> None:
        self.add(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = GRAY,
        sw: float = 3.0,
        marker: str = "arrow-gray",
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw:.1f}" stroke-linecap="round" '
            f'opacity="{opacity:.2f}" marker-end="url(#{marker})"/>'
        )

    def path(
        self,
        d: str,
        *,
        stroke: str = GRAY,
        sw: float = 3.0,
        marker: str = "arrow-gray",
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" '
            f'stroke-linecap="round" opacity="{opacity:.2f}" marker-end="url(#{marker})"/>'
        )

    def svg(self) -> str:
        markers = [
            ("arrow-gray", GRAY),
            ("arrow-blue", BLUE),
            ("arrow-amber", AMBER),
            ("arrow-teal", TEAL),
            ("arrow-red", RED),
            ("arrow-text", TEXT),
        ]
        defs = []
        for mid, color in markers:
            defs.append(
                f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        style = (
            "<style>"
            "text{font-family:Arial,Helvetica,'DejaVu Sans',sans-serif;dominant-baseline:middle}"
            ".caption{fill:#6f7f86;font-size:15px;font-weight:500}"
            "</style>"
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            f'<rect width="100%" height="100%" fill="white"/>'
            f"<defs>{''.join(defs)}</defs>{style}{''.join(self.parts)}</svg>\n"
        )


def _edge_points(p1, p2, r1=23, r2=23):
    x1, y1 = p1
    x2, y2 = p2
    length = hypot(x2 - x1, y2 - y1)
    if length == 0:
        return x1, y1, x2, y2
    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    return x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2


def arrow(svg: Svg, p1, p2, *, color=GRAY, marker="arrow-gray", sw=3.0, opacity=1.0, r1=23, r2=23):
    x1, y1, x2, y2 = _edge_points(p1, p2, r1=r1, r2=r2)
    svg.line(x1, y1, x2, y2, stroke=color, sw=sw, marker=marker, opacity=opacity)


def labeled_node(svg: Svg, x, y, label, *, fill=WHITE, stroke=TEXT, color=TEXT, r=23, size=16):
    svg.circle(x, y, r, fill=fill, stroke=stroke, sw=2.8)
    if label:
        svg.text(x, y + 1, label, size=size, weight=700, fill=color)


def save(svg: Svg, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg.svg(), encoding="utf-8")
    if not bfs.render_svg_preview(svg_path, png_path, width=1700):
        raise RuntimeError("Could not render SVG preview; install rsvg-convert or ImageMagick.")
    print(f"Saved {svg_path} and {png_path}")


DAG_NODES = {
    "s0": (70, 165),
    "a": (185, 78),
    "b": (185, 165),
    "c": (185, 252),
    "d": (315, 165),
    "e": (445, 95),
    "f": (445, 235),
    "x1": (590, 70),
    "x2": (590, 165),
    "x3": (590, 260),
}

DAG_EDGES = [
    ("s0", "a"),
    ("s0", "b"),
    ("s0", "c"),
    ("a", "d"),
    ("b", "d"),
    ("c", "f"),
    ("d", "e"),
    ("d", "f"),
    ("d", "x2"),
    ("e", "x1"),
    ("f", "x2"),
    ("f", "x3"),
]


def draw_dag(svg: Svg, *, highlight=(), reverse=False, show_labels=False, edge_labels=None):
    highlight = set(highlight)
    edge_labels = edge_labels or {}
    for u, v in DAG_EDGES:
        edge = (u, v)
        p1, p2 = DAG_NODES[u], DAG_NODES[v]
        if reverse:
            p1, p2 = p2, p1
        is_high = edge in highlight
        color = BLUE if is_high and not reverse else AMBER if is_high else GRAY
        marker = "arrow-blue" if is_high and not reverse else "arrow-amber" if is_high else "arrow-gray"
        arrow(svg, p1, p2, color=color, marker=marker, sw=5.0 if is_high else 3.0, opacity=1 if is_high else 0.55)
        if edge in edge_labels:
            x = (DAG_NODES[u][0] + DAG_NODES[v][0]) / 2
            y = (DAG_NODES[u][1] + DAG_NODES[v][1]) / 2 - 12
            svg.text(x, y, edge_labels[edge], size=14, weight=700, fill=color)

    for name, (x, y) in DAG_NODES.items():
        if name == "s0":
            labeled_node(svg, x, y, "s0" if show_labels else "", fill=BLUE_LIGHT, stroke=BLUE, color=BLUE)
        elif name.startswith("x"):
            labeled_node(svg, x, y, name if show_labels else "", fill=RED_LIGHT, stroke=RED, color=RED)
        else:
            labeled_node(svg, x, y, name if show_labels else "", fill=WHITE, stroke=TEXT)


def figure_forward_policy():
    svg = Svg(720, 340)
    highlight = [("s0", "b"), ("b", "d"), ("d", "e"), ("e", "x1")]
    draw_dag(svg, highlight=highlight, show_labels=False)
    svg.text(360, 28, "Forward policy samples a construction path", size=21, weight=700)
    svg.text(360, 314, "highlighted path = one trajectory from s0 to a terminal object", size=16, fill=MUTED)
    save(svg, "fig_forward_policy")


def figure_backward_policy():
    svg = Svg(720, 340)
    highlight = [("s0", "b"), ("b", "d"), ("d", "e"), ("e", "x1")]
    draw_dag(svg, highlight=highlight, reverse=True, show_labels=False)
    svg.text(360, 28, "Backward policy decomposes a terminal object", size=21, weight=700)
    svg.text(360, 314, "highlighted arrows trace one reverse construction order", size=16, fill=MUTED)
    save(svg, "fig_backward_policy")


def mini_tree(svg: Svg, ox: float, oy: float, *, nonuniform=False, forward=False, reverse=False):
    pts = {
        "s0": (ox, oy + 80),
        "s1": (ox + 105, oy + 35),
        "s2": (ox + 105, oy + 125),
        "x1": (ox + 225, oy + 12),
        "x2": (ox + 225, oy + 80),
        "x3": (ox + 225, oy + 148),
    }
    if nonuniform:
        edges = [("s0", "s1"), ("s0", "s2"), ("s1", "x1"), ("s1", "x2"), ("s2", "x2"), ("s2", "x3")]
    else:
        edges = [("s0", "s1"), ("s0", "s2"), ("s1", "x1"), ("s2", "x2"), ("s2", "x3")]
    for u, v in edges:
        color = TEXT if forward else GRAY
        marker = "arrow-text" if forward else "arrow-gray"
        p1, p2 = (pts[v], pts[u]) if reverse else (pts[u], pts[v])
        arrow(svg, p1, p2, color=color, marker=marker, sw=2.5, r1=17, r2=17)
    for n, p in pts.items():
        if n == "s0":
            labeled_node(svg, *p, "s0", fill=BLUE_LIGHT, stroke=BLUE, color=BLUE, r=17, size=11)
        elif n.startswith("x"):
            labeled_node(svg, *p, n, fill=RED_LIGHT, stroke=RED, color=RED, r=17, size=12)
        else:
            labeled_node(svg, *p, n, fill=WHITE, stroke=TEXT, r=17, size=12)
    return pts


def panel(svg: Svg, x, y, w, h, title: str):
    svg.rect(x, y, w, h, fill=WHITE, stroke="#d8e0e4", sw=1.2, rx=7)
    svg.text(x + 16, y + 22, title, size=16, weight=700, fill=TEXT, anchor="start")


def bars(svg: Svg, x, y, values, *, max_value=4, color=AMBER):
    for i, (label, value) in enumerate(values):
        yy = y + i * 42
        svg.text(x, yy + 12, label, size=14, weight=700, fill=TEXT, anchor="start")
        svg.rect(x + 64, yy, 132, 24, fill=LIGHT_GRAY, stroke="none", sw=0, rx=5)
        svg.rect(x + 64, yy, 132 * value / max_value, 24, fill=AMBER_LIGHT, stroke=color, sw=1.4, rx=5)
        svg.text(x + 210, yy + 12, str(value), size=15, weight=700, fill=color, anchor="start")


def figure_example_forward():
    svg = Svg(940, 340)
    panel(svg, 20, 26, 278, 270, "A. Backward policy")
    panel(svg, 330, 26, 250, 270, "B. Target weights")
    panel(svg, 612, 26, 306, 270, "C. Forward policy")
    pts = mini_tree(svg, 55, 80, reverse=True)
    svg.text(160, 265, "pB = 1 on each parent edge", size=13, fill=MUTED)
    bars(svg, 365, 96, [("x1", 4), ("x2", 2), ("x3", 1)])
    pts = mini_tree(svg, 650, 80, forward=True)
    labels = [
        ((pts["s0"][0] + pts["s1"][0]) / 2, (pts["s0"][1] + pts["s1"][1]) / 2 - 15, "4/7"),
        ((pts["s0"][0] + pts["s2"][0]) / 2, (pts["s0"][1] + pts["s2"][1]) / 2 + 16, "3/7"),
        ((pts["s1"][0] + pts["x1"][0]) / 2, (pts["s1"][1] + pts["x1"][1]) / 2 - 15, "1"),
        ((pts["s2"][0] + pts["x2"][0]) / 2, (pts["s2"][1] + pts["x2"][1]) / 2 - 15, "2/3"),
        ((pts["s2"][0] + pts["x3"][0]) / 2, (pts["s2"][1] + pts["x3"][1]) / 2 + 16, "1/3"),
    ]
    for x, y, label in labels:
        svg.text(x, y, label, size=14, weight=700, fill=TEXT)
    svg.text(470, 316, "terminal probability is proportional to reward", size=15, fill=MUTED)
    save(svg, "fig_example_forward")


def figure_example_backward():
    svg = Svg(940, 340)
    panel(svg, 20, 26, 300, 270, "A. Non-uniform pB")
    panel(svg, 350, 26, 250, 270, "B. Four trajectories")
    panel(svg, 632, 26, 286, 270, "C. Forward response")
    pts = mini_tree(svg, 60, 80, nonuniform=True, reverse=True)
    svg.text(262, 128, "1", size=13, weight=700, fill=TEXT)
    svg.text(262, 164, "0.5", size=13, weight=700, fill=RED)
    svg.text(262, 192, "0.5", size=13, weight=700, fill=RED)
    svg.text(262, 236, "1", size=13, weight=700, fill=TEXT)
    bars(svg, 385, 78, [("tau1", 4), ("tau2a", 1), ("tau2b", 1), ("tau3", 1)])
    pts = mini_tree(svg, 670, 80, nonuniform=True, forward=True)
    labels = [
        ((pts["s0"][0] + pts["s1"][0]) / 2, (pts["s0"][1] + pts["s1"][1]) / 2 - 16, "5/7"),
        ((pts["s0"][0] + pts["s2"][0]) / 2, (pts["s0"][1] + pts["s2"][1]) / 2 + 16, "2/7"),
        ((pts["s1"][0] + pts["x1"][0]) / 2, (pts["s1"][1] + pts["x1"][1]) / 2 - 15, "4/5"),
        ((pts["s1"][0] + pts["x2"][0]) / 2, (pts["s1"][1] + pts["x2"][1]) / 2 + 12, "1/5"),
        ((pts["s2"][0] + pts["x2"][0]) / 2, (pts["s2"][1] + pts["x2"][1]) / 2 - 12, "1/2"),
        ((pts["s2"][0] + pts["x3"][0]) / 2, (pts["s2"][1] + pts["x3"][1]) / 2 + 15, "1/2"),
    ]
    for x, y, label in labels:
        svg.text(x, y, label, size=13, weight=700, fill=TEXT)
    svg.text(470, 316, "splitting x2 changes the route probabilities", size=15, fill=MUTED)
    save(svg, "fig_example_backward")


def figure_flow_matching():
    svg = Svg(1060, 340)
    panel(svg, 20, 26, 220, 270, "A. Rewards")
    panel(svg, 270, 26, 230, 270, "B. Backward flow")
    panel(svg, 530, 26, 220, 270, "C. Forward flow")
    panel(svg, 780, 26, 260, 270, "D. Policy")
    bars(svg, 55, 100, [("x1", 4), ("x2", 2), ("x3", 1)])
    bars(svg, 305, 100, [("tau1", 4), ("tau2", 2), ("tau3", 1)], color=TEAL)
    svg.text(642, 112, "Z = 7", size=23, weight=700, fill=TEXT)
    bars(svg, 565, 150, [("tau1", 4), ("tau2", 2), ("tau3", 1)], color=BLUE)
    pts = mini_tree(svg, 800, 80, forward=True)
    labels = [
        ((pts["s0"][0] + pts["s1"][0]) / 2, (pts["s0"][1] + pts["s1"][1]) / 2 - 15, "4/7"),
        ((pts["s0"][0] + pts["s2"][0]) / 2, (pts["s0"][1] + pts["s2"][1]) / 2 + 16, "3/7"),
        ((pts["s2"][0] + pts["x2"][0]) / 2, (pts["s2"][1] + pts["x2"][1]) / 2 - 15, "2/3"),
        ((pts["s2"][0] + pts["x3"][0]) / 2, (pts["s2"][1] + pts["x3"][1]) / 2 + 16, "1/3"),
    ]
    for x, y, label in labels:
        svg.text(x, y, label, size=13, weight=700, fill=TEXT)
    svg.text(510, 316, "trajectory balance matches backward and forward flows", size=15, fill=MUTED)
    save(svg, "fig_flow_matching")


def figure_detailed_balance():
    svg = Svg(820, 330)
    svg.text(410, 34, "Detailed balance is a local edge condition", size=22, weight=700)
    svg.rect(90, 100, 170, 92, fill=BLUE_LIGHT, stroke=BLUE, sw=2.2, rx=8)
    svg.text(175, 128, "state s", size=20, weight=700, fill=BLUE)
    svg.text(175, 162, "F(s) = 5", size=18, weight=700, fill=TEXT)
    svg.rect(560, 100, 170, 92, fill=RED_LIGHT, stroke=RED, sw=2.2, rx=8)
    svg.text(645, 128, "state s'", size=20, weight=700, fill=RED)
    svg.text(645, 162, "F(s') = 2", size=18, weight=700, fill=TEXT)
    svg.path("M 270 122 C 370 82, 455 82, 550 122", stroke=BLUE, sw=4.0, marker="arrow-blue")
    svg.path("M 550 170 C 455 212, 370 212, 270 170", stroke=AMBER, sw=4.0, marker="arrow-amber")
    svg.rect(320, 78, 180, 30, fill=WHITE, stroke="none", sw=0, rx=4)
    svg.text(410, 92, "pF(s'|s) = 2/5", size=16, weight=700, fill=BLUE)
    svg.rect(342, 200, 136, 30, fill=WHITE, stroke="none", sw=0, rx=4)
    svg.text(410, 214, "pB(s|s') = 1", size=16, weight=700, fill=AMBER)
    svg.rect(150, 248, 520, 44, fill=WHITE, stroke="#d8e0e4", sw=1.2, rx=6)
    svg.text(410, 270, "local condition:  F(s) * pF(s'|s) = F(s') * pB(s|s')", size=18, weight=700)
    save(svg, "fig_detailed_balance")


def main():
    figure_detailed_balance()


if __name__ == "__main__":
    main()
