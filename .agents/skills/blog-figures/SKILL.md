---
name: blog-figures
description: Create, source, or improve scientific blog figures and paper-ready exports. Use for plots, diagrams, figure spacing, label legibility, captions, and visual QA; not for full manuscript layout.
---

# Scientific figures for the blog and papers

Make the figure's scientific point legible at its final display size. Preserve
the data, source authority, site palette, and requested visual intent.

## Choose a route

- For plots, editable diagrams, multi-panel layouts, and paper exports, read
  [generated figures](references/generated-figures.md).
- For an existing figure from a paper or the web, read
  [source figures](references/source-figures.md).
- For an illustrative raster scene, read
  [image generation](references/image-generation.md).
- For required reuse from a lecture deck, follow
  [lecture extraction](../lecture-adaptation/references/extraction.md).

For a standard concept, check whether a clear, licensed source figure already
fits. A custom toy construction, synthesis, quantitative plot, or unsuitable
source justifies an original drawing. Confirmed deck reuse takes precedence
over searching for or drawing replacements.

## Shared requirements

- Store blog assets under `assets/img/blog/`; keep generating code in the
  existing `scripts/` workflow. Paper-only outputs go to the requested
  manuscript's figure directory or a durable task output directory.
- Reuse [blog_figure_style.py](../../../scripts/blog_figure_style.py) and
  [the site palette](../../../docs/palette.md).
- Keep vector marks and editable text for ordinary plots and diagrams.
  Raster or hybrid output is appropriate for dense fields, photographs, and
  scientific images. Preserve the reproducible source.
- Empirical results must trace to checked measurements; identify seed counts
  and what uncertainty intervals mean. Explicit educational toy data is valid.
- Introduce the figure where its mechanism becomes relevant. Captions explain
  the point and interpretation, with needed qualifications and verified
  attribution. Caption length follows the figure's complexity.
- Supply concise alt text. Use the
  [Jekyll include pattern](../site-validation/references/jekyll-rendering.md)
  when embedding an asset.

## Completion

Inspect at the intended blog width or paper size, including panel gaps,
legend placement, labels, arrows, and canvas edges. For Matplotlib figures, use
`audit_figure` to find potential collisions and clipping, then judge the rendered
result. Fix visible defects; document intentional overlaps individually.

Validate affected posts with `python3 scripts/validate_blog.py`.
Keep source URLs, licenses, prompts when applicable, and modifications with
scripts or manifests. Selected OpenResearch practices and their local
adaptations are recorded in [the source register](../../third-party/sources.md).
