# Generated figures and print exports

Adapted from OpenResearch's figure guidance and layout audit. See the
[source register](../../../third-party/sources.md) for the pinned source and license.

## Layout at the destination size

Choose a physical width from the destination document for print. The helper's
`COLUMN`, `TEXT_WIDTH`, and `WIDE` presets are conveniences, not venue rules;
custom widths are valid. For the blog, inspect at actual article width and on a
narrow screen. A large canvas shrunk into a column also shrinks every label.

Build a Matplotlib grid with `layout="constrained"` from the start. Do not
follow it with `tight_layout()`, which disables that layout engine. Reserve a
row or suitable outside location for a shared legend or colorbar; automatic
layout does not solve every placement.

Align related panels and share scales when visual comparisons depend on them.
Keep panel letters stable and follow the order used in the text. Group panels
that support one argument; unrelated plots need not occupy one grid.

Let longer labels reflow, widen the relevant space, or simplify the composition.
Do not shrink all text to rescue one crowded label. Use line styles or markers
as well as color when series need to remain distinct in grayscale.

## Tools and style

Use Matplotlib for quantitative marks; use editable SVG for diagrams whose
geometry or labels need deliberate placement. TikZ is an option when the
destination already benefits from LaTeX. Keep diagram positions relative where
practical, separate arrows from labels, and use a consistent stroke/label scale.

Preserve the site's purple-led palette and meaningful color assignments.
A mathematical diagram can have several meaningful accents. Labels and
subtitles that distinguish panels are valid; remove titles that merely duplicate
the caption.

Use `use_blog_style()` for web figures and `use_publication_style()` for
smaller printed figures. The latter keeps the site colors and uses readable
print text with embedded fonts. Use installed fallback fonts; do not make a
particular proprietary font a completion requirement.

## Minimal paper export

Run from the repository root using the existing Python environment:

```python
import numpy as np
import matplotlib.pyplot as plt
from scripts import blog_figure_style as bfs

bfs.use_publication_style()
fig, ax = plt.subplots(figsize=(bfs.COLUMN, 2.3), layout="constrained")
x = np.linspace(0, 3, 100)
ax.plot(x, np.exp(-x), color=bfs.PURPLE)  # explicitly an analytic example
ax.set_xlabel("Time (s)")
ax.set_ylabel("Relative concentration")
issues = bfs.audit_figure(fig, target_width_in=bfs.COLUMN)
# Inspect any issues and the resulting preview.
paths = bfs.save_publication_figure(fig, "output/figures/decay",
                                    target_width_in=bfs.COLUMN)
```

`save_publication_figure` emits PDF, editable SVG, and PNG at the existing
canvas size; it rejects a width that differs from the declared target instead
of silently rescaling. It disables tight bounding-box cropping during export.
Use the manuscript's actual width in both figure construction and inclusion.

Existing `save_svg_png` and `save_figure` retain their web export behavior.
Use the publication exporter when fixed dimensions matter.

## Audit and evidence

`audit_figure` reports text overlap, off-canvas text, small type, and a declared
width mismatch. It returns findings for review rather than certifying a figure.
It does not detect every arrow collision, hidden data mark, or weak visual
hierarchy. An intentional annotation can be excluded by passing that specific
text artist in `ignore`; keep the rationale in the generating code.

State measurements, units, normalization, smoothing, missing runs, and uncertainty
where they affect interpretation. Do not manufacture error bars for a single
run. Preserve the distinction between variation across runs and uncertainty of
an estimate; choose statistics appropriate to the actual sampling design.

After saving, read the PNG/SVG and, for print delivery, inspect the PDF at its
intended size. Keep plots and their source/data together in the deliverable.

When changing the shared helper, run its regression tests and generate the
four analytic examples with `scripts/preview_figure_quality.py`. Pass
`--output-dir` pointing to a fresh temporary directory; inspect the exported
curve, panels, heatmap, and diagram. CI retains these artifacts for review.
