---
name: generate-blog-figures
description: Create or revise source-first, SVG-first technical blog figures for this Jekyll site. Use when generating quantitative plots, custom explanatory diagrams, figure scripts, PNG previews, captions, or figure-quality checks for blog posts.
---

# Generate Blog Figures

Create publication-quality blog figures. Treat figures as explanatory objects, not notebook screenshots.

## Source-First Rule

- Search for an existing high-quality figure before drawing a well-known concept.
- Prefer stable, license-compatible sources: Wikimedia Commons, official project pages, PMC/open-access paper figures, or author-provided figures with clear terms.
- Do not redraw standard concepts such as amino acid charts, protein structure hierarchy, AlphaFold/RFDiffusion/ProteinMPNN overview figures, or canonical architecture diagrams when a clear licensed source exists.
- Draw a custom figure only when the post needs a new abstraction, a simplified toy plot, a synthesis across sources, or a figure whose existing versions are legally unusable or visually unsuitable.
- Record source URL, license, asset path, and any modifications in the post's Figure Sources section or figure manifest.

## Image-Generation Rule

- Use ChatGPT/image-generation for physical device or process schematics when a polished illustrative rendering communicates better than boxes and arrows. Good candidates include PEM fuel cells, electrolyzers, catalyst reactors, lab workflows, instruments, and molecular scenes where spatial appearance matters.
- Do not use image generation for quantitative plots, architecture diagrams, algorithm pipelines, or figures that require exact geometry, exact text placement, or reproducible data marks.
- Generate the image without embedded text whenever possible. Add labels, arrows, equations, and callouts as an editable SVG overlay or adjacent HTML/SVG layer.
- Keep the prompt, generated base image, final annotated asset, PNG preview, caption, and design/provenance note. The generated base image is the source for the visual rendering; the SVG overlay is the source for labels and explanation.
- If a high-quality licensed source figure already exists and is clearer than a generated illustration, prefer the source figure. Image generation is for making a better explanatory rendering, not for redrawing canonical method figures.

## File Structure

- Script: `scripts/generate_<postname>_figures.py`
- Generated static outputs: `assets/img/blog/<prefix>_<name>.svg` plus `assets/img/blog/<prefix>_<name>.png`
- Image-generated illustration base: `assets/img/blog/<prefix>_<name>_imagegen_base.png`
- Image-generated annotated output: `assets/img/blog/<prefix>_<name>.svg` plus `assets/img/blog/<prefix>_<name>.png`, with the base raster embedded and labels as editable SVG text/shapes.
- Sourced figures: keep the source format and use a filename that indicates provenance when useful, such as `<stem>_source.jpg`
- If an SVG becomes a huge path dump for 3D surfaces, dense contours, raster-like heatmaps, or image composites, embed the PNG preview instead and keep the source code.
- One function per generated figure.
- Use `scripts/blog_figure_style.py` for shared colors, rcParams, axis cleanup, and SVG+PNG saving.

## Drawing Boundaries

- Use Matplotlib for quantitative elements: curves, scatter points, heatmaps, axes, shaded regions, uncertainty bands, and simple legends.
- Do not use Matplotlib for complex architecture diagrams, pipeline diagrams, free-floating callouts, dense arrows, or icons. Use native SVG generation or a source figure.
- Keep Matplotlib base plots sparse. Add complex labels or callouts in SVG coordinates when needed.
- Use at most 2-3 callouts per figure. Prefer a stronger caption over crowded in-plot text.

## House Style

Use `bfs.use_blog_style()` before plotting, where `import blog_figure_style as bfs`.

- Single-column figures: about 3.5-5.5 inches wide.
- Line width: about 2.0-2.5.
- Font size: about 9-11 pt.
- No top/right spines.
- Minimal gridlines or none.
- Direct curve labels when clearer than legends.
- Semantic colors: muted blue for primary, muted orange for comparison, green for positive/increase, red for negative/decrease, gray for auxiliary elements.
- Avoid rainbow colormaps, saturated red/green-only distinctions, and arbitrary decorative colors.

## Captions

Every figure needs:

1. One sentence saying what the figure shows.
2. One sentence explaining the mechanism or interpretation.
3. A short source/license note when the figure is downloaded or adapted.

Inside `{% include figure.liquid %}` captions, use `\(...\)` for math, not `$...$`.

## Workflow

1. Decide source vs draw using the source-first rule.
2. For sourced figures, download the original/highest useful resolution and document provenance.
3. For image-generated physical schematics, generate a no-text base image, save it into `assets/img/blog/`, then add labels/callouts in SVG coordinates. Record the final prompt in the figure script, manifest, or Figure Sources section.
4. For code-generated figures, write or update the figure script and save SVG+PNG with `bfs.save_svg_png(...)` or `bfs.save_figure(...)`.
5. Render/check PNG previews at blog width.
6. Inspect for label collisions, arrow crossings, readable text, semantic colors, and non-default styling.
7. Embed SVG for generated static figures; embed the original format for sourced figures and GIFs.
8. Run `python3 scripts/validate_blog.py`.

## Embed Pattern

Generated static figure:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_figure.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="One-sentence summary. One-sentence mechanism." %}
```

Sourced raster:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_source_figure.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="One-sentence summary. One-sentence interpretation. From Author et al. (Year), CC BY 4.0." %}
```

Image-generated physical schematic:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_device_schematic.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="One-sentence summary. One-sentence mechanism. Base illustration generated with ChatGPT image generation; labels added as editable SVG." %}
```
