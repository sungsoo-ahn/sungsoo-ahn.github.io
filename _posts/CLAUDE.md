# Blog Writing Guide

Instructions for writing and editing blog posts in `_posts/`.

**Skills:** `/blog-writing` for prose style, `/jekyll-writing` for rendering rules.

## Post Frontmatter

Every post must include this frontmatter block:

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD
last_updated: YYYY-MM-DD
description: "One-sentence summary for SEO and post listings."
abstract: >
  Optional short abstract shown below the post metadata.
blog_blocks:
  - title: Key points
    content: >
      Optional named block shown before the post body.
post_type: tutorial # tutorial | technical-note | research
authors: ["Sungsoo Ahn"] # list all named authors shown in the post and blog index
order: 1 # legacy field; blog index now sorts by date
series: optional-series-id
series_title: "Optional Series Title"
series_description: "Optional one-sentence reading path description."
series_order: 1
categories: [category-name]
tags: [tag1, tag2, tag3]
toc:
  sidebar: left
related_posts: false
---
```

**MANDATORY: Update `last_updated`** — Whenever you edit a blog post, update the `last_updated` field in the frontmatter to today's date. This is required for every edit, no matter how small.

The blog index sorts posts by date. Series metadata may be kept for organization, but posts do not render bottom reading-path or roadmap blocks.

Use `authors` for every post, even single-author posts. The blog layout also supports the older `author` field as a fallback, but new posts should use `authors`.

Optional frontmatter blocks render between the post metadata and the post body. Use `abstract` for a single abstract-style block. Use `blog_blocks` for additional named blocks such as `Key points`, `Prerequisites`, or `Scope`. Keep these blocks short; Markdown is supported inside each block.

If a post uses AI-assisted material, handle provenance in that post's author note or body only when it is relevant to that specific article.

## Author Note

Posts begin with an author note immediately after the frontmatter:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em
    >Note: Your context here — background, acknowledgments, recommended further
    reading.</em
  >
</p>
```

## Blog-Specific Content Rules

- Collapse related works into paragraphs rather than giving each its own subsection
- Readers likely don't know domain-specific jargon (e.g., "quadrupole") — use plain language
- Footnotes work well for alternate terminology without cluttering the main text
- When referencing publications, add to a References section as you cite, not at the end
- **Footnote IDs: no hyphens.** `[^chemical-potential]` breaks rendering; use `[^chempot]` instead. Short single-word IDs only.
- **Every text citation needs a References entry.** If you write "Bengio et al., 2021" in the text, add the full citation to References immediately.
- **Cross-references: use section names, not numbers.** "See the Scaling Relations section" survives reordering; "see Section 5" doesn't.
- **Figure captions: use `\(...\)` for math**, not `$...$` or `$$...$$`. Liquid consumes `\\`, breaking LaTeX commands.
- **Provenance wording:** use "Redrawn from..." or "Adapted from..." when a figure is reconstructed from a paper or source. Directly licensed external figures should include the source and license.

## Figures

- Store figure images in `assets/img/blog/`
- **Source before drawing:** do not redraw a well-known concept when a high-quality, license-compatible figure already exists online. Prefer Wikimedia Commons, official project pages, PMC/open-access paper figures, or author-provided figures with clear terms.
- Draw custom figures only for post-specific conceptual simplifications, quantitative toy plots, synthesis diagrams, or cases where existing figures are legally unusable or visually unsuitable.
- For generated static figures, write a single Python script with separate functions per figure and export SVG plus PNG preview.
- If the SVG becomes a huge path dump for 3D surfaces, dense contours, raster-like heatmaps, or image composites, embed the PNG preview instead and keep the source code.
- For sourced figures, keep the original useful format and document source URL, license, and any modifications in agent-facing notes, figure-generation scripts, or asset metadata. Do not add a rendered figure-source appendix to the post body.
- Use a consistent color palette across generated figures with matching text colors for equation-figure correspondence.
- Captions should usually have two sentences: one saying what the figure shows, one explaining the mechanism or interpretation. Add concise provenance wording for sourced or adapted figures.
- Embed figures using the Liquid include:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_figure.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="What the figure shows. Why it matters." %}
```

Run `python3 scripts/validate_blog.py` before committing. It fails on broken metadata, missing figures, bad footnote IDs, and unsafe caption math; it warns on unused blog images and provenance wording that should be reviewed.

## CSS

- Distinct h2/h3 styling (border-bottom on h2, different font sizes) for visual hierarchy
- These are one-time additions that carry over to future posts
