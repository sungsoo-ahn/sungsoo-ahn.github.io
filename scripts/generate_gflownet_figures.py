"""Generate editable GFlowNet diagrams with explicit positive target weights.

Superseded lecture PNGs remain in Git history; corrected examples use new filenames.
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


def _worked_graph(*, shared: bool):
    """Forward probabilities derived from path weights 4, 2, 1 or 4, 1, 1, 1."""
    from fractions import Fraction

    svg = Svg(660, 470)
    title = "A shared terminal state" if shared else "One path per terminal state"
    svg.text(330, 30, title, size=25, weight=700)
    svg.text(330, 70, "Positive weights w = exp(R);  Z = 4 + 2 + 1 = 7", size=20)
    pts = {"s0": (65, 250), "s1": (210, 165), "s2": (210, 335),
           "x1": (390, 130), "x2": (390, 250), "x3": (390, 370)}
    paths = [("s0", "s1", "x1"), ("s0", "s1", "x2"), ("s0", "s2", "x3")]
    weights = [4, 2, 1]
    if shared:
        paths.insert(2, ("s0", "s2", "x2"))
        weights = [4, 1, 1, 1]
    flow = {}
    outgoing = {}
    for path, weight in zip(paths, weights):
        for u, v in zip(path, path[1:]):
            flow[u, v] = flow.get((u, v), 0) + weight
            outgoing[u] = outgoing.get(u, 0) + weight
    for (u, v), weight in flow.items():
        arrow(svg, pts[u], pts[v], color=BLUE, marker="arrow-blue", sw=2.6, r1=25, r2=27)
        x = (pts[u][0] + pts[v][0])/2
        y = (pts[u][1] + pts[v][1])/2
        offset = -20 if pts[v][1] < pts[u][1] else 22
        svg.text(x, y+offset, str(Fraction(weight, outgoing[u])), size=20, weight=700)
    for name, (x, y) in pts.items():
        labeled_node(svg, x, y, name, r=25, size=19,
                     fill=RED_LIGHT if name.startswith("x") else BLUE_LIGHT,
                     stroke=RED if name.startswith("x") else BLUE)
    for name, weight in [("x1", 4), ("x2", 2), ("x3", 1)]:
        svg.text(525, pts[name][1], f"w({name}) = {weight}", size=22, weight=700)
    svg.text(330, 430, "Arrows: forward probabilities pF", size=20, fill=MUTED)
    return svg


def figure_example_forward():
    save(_worked_graph(shared=False), "fig_example_tree")


def figure_example_backward():
    save(_worked_graph(shared=True), "fig_example_shared")


def figure_flow_matching():
    svg = Svg(700, 440)
    svg.text(350, 30, "Trajectory balance on the shared-terminal graph", size=23, weight=700)
    headers = [(120, "Path"), (320, "Backward flow"), (525, "Forward probability")]
    for x, title in headers:
        svg.text(x, 92, title, size=20, weight=700)
    rows = [
        ("s0 → s1 → x1", "4 × 1 = 4", "4/7"),
        ("s0 → s1 → x2", "2 × ½ = 1", "1/7"),
        ("s0 → s2 → x2", "2 × ½ = 1", "1/7"),
        ("s0 → s2 → x3", "1 × 1 = 1", "1/7"),
    ]
    for i, row in enumerate(rows):
        y = 150+i*53
        svg.rect(25, y-23, 650, 46, fill=bfs.PURPLE_LIGHT if i%2 == 0 else WHITE,
                 stroke="none", sw=0, rx=4)
        for (x, _), value in zip(headers, row):
            svg.text(x, y, value, size=21)
    svg.text(350, 370, "fB(τ) = w(x) pB(τ | x) = Z pF(τ) = fF(τ)", size=22, weight=700)
    svg.text(350, 408, "w(x) = exp(R(x));  Z = 7", size=20, fill=MUTED)
    save(svg, "fig_trajectory_balance")



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
    figure_example_forward()
    figure_example_backward()
    figure_flow_matching()
    figure_detailed_balance()


if __name__ == "__main__":
    main()
