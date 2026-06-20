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

## Workflow

1. Identify the target concept and candidate source figure.
2. Verify license and provenance before downloading:
   - Wikimedia: use the file page license.
   - PMC: use article page license text or `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC...`.
   - arXiv: check paper license; CC BY is usable, all-rights-reserved is not.
   - Publisher sites: do not reproduce directly unless the figure license is explicit and compatible.
3. Download the highest useful resolution or original vector file.
4. Save under `assets/img/blog/` with a stable descriptive filename.
5. Add or update a Figure Sources entry with source URL, license, and modifications.
6. Write a two-sentence caption plus a short provenance note.
7. Run `python3 scripts/validate_blog.py`.

If using ChatGPT/image generation instead of a downloaded source figure:

1. Generate a no-text base illustration and save it under `assets/img/blog/<stem>_imagegen_base.png`.
2. Add exact labels, arrows, and equations in a separate editable SVG layer.
3. Save the final annotated SVG plus PNG preview.
4. Record the prompt, asset paths, and design rationale in the Figure Sources section.

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
