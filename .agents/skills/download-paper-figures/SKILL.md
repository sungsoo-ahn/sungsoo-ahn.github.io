---
name: download-paper-figures
description: Find, license-check, download, cite, or redraw figures from papers and web sources for blog posts. Use when incorporating paper figures, Wikimedia/open-license figures, official project figures, or internet-downloaded educational diagrams.
---

# Download and Cite Paper Figures

Use source figures when they are clearer than a custom redraw and legally usable.

## Source-First Rule

- Search before drawing any well-known concept or method overview.
- Prefer high-quality existing figures from Wikimedia Commons, official project pages, PMC/open-access articles, or author pages with explicit licenses.
- Do not copy figures from unclear or restrictive sources. If a useful figure is not license-compatible, redraw the idea approximately in the blog style and cite it as adapted.
- For publisher figures, assume restrictive unless the article or figure explicitly says CC BY or another compatible license.
- For physical device or process schematics, consider whether ChatGPT/image generation would produce a clearer explanatory base than a source figure or hand-drawn boxes. Use this only for illustrative renderings, then add exact labels/callouts as editable SVG text.
- Do not use image generation to substitute for canonical method-overview figures when a licensed high-quality source already exists, such as RFDiffusion, AlphaFold, ProteinMPNN, or standard protein-structure diagrams.

### Rights-Cleared Lecture Decks

When the author supplies a lecture deck and explicitly confirms republication
rights for all embedded figures, treat that confirmation as the reuse authority
for the adaptation. Reuse the deck figure directly instead of finding or
drawing a substitute. Preserve any paper/project attribution already present on
the slide, and record the course, deck, slide number, original cited source, and
extraction method in an agent-facing figure manifest.

This exception applies only to the confirmed deck and its embedded figures. It
does not establish a general license for the same source elsewhere.

#### PowerPoint-native extraction

- Never publish a full-slide screenshot or a region cropped from the lecture
  PDF. The editable PPTX is the source of truth for visual extraction.
- Inspect the slide's actual object tree. Presentation manifests and PPTX media
  archives can contain unused, hidden, duplicated, or alternate resources that
  are not visible on the slide.
- Copy an unmodified browser-safe picture directly from `ppt/media` to preserve
  its original pixels. If PowerPoint applies a crop, mask, rotation, grouped
  composition, or native-shape construction, export the exact picture or
  semantic group with PowerPoint's `save as picture` command.
- Do not rasterize text boxes, equations, paper title cards, citations, or
  tables merely because they appeared on a slide. Recreate those as reader-
  facing HTML/Markdown/MathJax and retain only the scientific visual object.
- Record each published asset with `slide`, `pptx_slide`, `asset_path`,
  `source_media_paths` or `source_shape_name`, `extraction_method`,
  `content_role`, and `reuse_status`. A completed migration must have an exact
  manifest-to-post asset match and no reused PDF-region records.

### Auditing Existing Lecture Figures

- Treat **visual provenance** and **method provenance** as different claims. A
  slide titled with a paper may contain an exact paper figure, a lecturer-made
  derivation of that paper's method, or a composite. Cite the paper as the
  figure source only after matching the actual visual.
- For lecturer-made equations, matrices, graph drawings, and comparison
  diagrams, use a reader-facing note such as “Lecture diagram by the author.”
  Keep the method citation in the prose or references; do not turn it into a
  misleading “Figure source” link.
- Verify exact matches with distinctive labels, panel order, node colors,
  plotted values, table cells, and crop boundaries. Matching a topic, title, or
  broad layout is not enough. When the concept is traceable but the rendering
  is generic, record the concept source as ambiguous and say that the exact
  drawing origin is unclear.
- In chronological method surveys, one primary paper may legitimately source
  several adjacent assets: an architecture panel, result plot, table, or
  equation crop. Record each asset separately, but deduplicate the source
  metadata and reuse the same authoritative paper page.
- A publisher DOI returning an automated 403 is not by itself a broken source.
  Confirm the DOI's title, authors, venue, and year through an authoritative
  index, and record the publisher's license conservatively.

## Workflow

1. Identify the target concept and candidate source figure.
2. Verify license and provenance before downloading:
   - Wikimedia: use the file page license.
   - PMC: use article page license text or `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC...`.
   - arXiv: check paper license; CC BY is usable, all-rights-reserved is not.
   - Publisher sites: do not reproduce directly unless the figure license is explicit and compatible.
3. Download the highest useful resolution or original vector file.
4. Save under `assets/img/blog/` with a stable descriptive filename.
5. Record source URL, license, and modifications in agent-facing notes, a figure-generation script, or a figure manifest. Do not add a rendered source appendix to the post body.
6. Write a two-sentence caption plus a short provenance note.
7. Run `python3 scripts/validate_blog.py`.

If using ChatGPT/image generation instead of a downloaded source figure:

1. Generate a no-text base illustration and save it under `assets/img/blog/<stem>_imagegen_base.png`.
2. Add exact labels, arrows, and equations in a separate editable SVG layer.
3. Save the final annotated SVG plus PNG preview.
4. Record the prompt, asset paths, and design rationale in agent-facing notes, a figure-generation script, or a figure manifest.

## Caption Wording

Direct licensed source:

```liquid
caption="RFDiffusion generates protein backbones through iterative denoising. Conditioning lets the same diffusion model handle unconditional generation, motif scaffolding, symmetry, and binder design. From Watson et al. (2023), CC BY 4.0."
```

Redrawn or reconstructed:

```liquid
caption="The volcano plot relates catalytic activity to adsorption energy. The peak appears because weak binding cannot activate intermediates, while strong binding prevents product release. Adapted from Nørskov et al. (2004)."
```

Use `\(...\)` for math inside Liquid captions.

## Copyright Rules

- Do not reproduce figures directly from ACS, Elsevier, Wiley, Springer/Nature paywalled figures, AAAS, or other restrictive publishers unless the specific article/figure is open under a compatible license.
- PMC open-access articles often expose figure images directly; still verify the article license and third-party material exceptions.
- If the source is a screenshot, a blog image with unclear ownership, or a social-media image, do not use it directly.

## Output Requirements

For each incorporated figure, provide:

- asset path;
- source URL;
- license;
- caption;
- short design/provenance note.

For redrawn figures, also provide source code and SVG+PNG outputs.
For image-generated figures, also provide the prompt, base PNG, final annotated SVG, and PNG preview.
