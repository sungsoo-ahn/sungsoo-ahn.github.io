# Jekyll rendering

This site uses Kramdown/GFM and MathJax. Check existing source and rendered HTML
when a delimiter interacts with Markdown or Liquid.

## Math, tables, and footnotes

- In Markdown, `$$...$$` on the same line protects inline math with multiple
  underscores through Kramdown. Display equations use their own block with
  blank lines around it.
- Inside raw HTML or a Liquid figure caption, use `\(...\)` for inline math.
  Avoid `$...$` and `$$...$$` in captions, as enforced by the blog validator.
  Use single-backslash LaTeX commands inside the quoted Liquid attribute and
  inspect output when a command needs escaping.
- Use `\lvert`, `\rvert`, or `\mid` instead of literal pipes in inline math
  inside Markdown tables.
- Write currency as USD or dollars when a dollar sign could trigger MathJax.
- Keep footnote IDs short and alphanumeric; hyphens fail the site's checks.
  Each use needs a matching definition.
- Use blank lines around lists, tables, display math, and block HTML.
  Check wide equations and tables for horizontal overflow.

## Figure include

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/example.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="What is visibly present." caption="What the figure establishes and how to interpret it." %}
```

Replace the example asset with an existing file. Math in the caption uses
`\(...\)`. Credit the verified source as appropriate.

## Mermaid

The site supports fenced Mermaid with `mermaid.enabled: true` in frontmatter;
[mermaid-setup.js](../../../../assets/js/mermaid-setup.js) initializes it.
Use a static SVG plus source when a durable figure export is preferable.
Store blog diagram assets under `assets/img/blog/`, not a removed course tree.

Inspect labels and arrows at desktop and narrow widths. Avoid dense crossings
and overly long labels. The existing setup expands viewBoxes; do not assume
that solves every clipping case or use overflowing content that overlaps prose.

## Render verification

Inspect MathJax output, caption math, footnote targets, table borders, and diagram
edges. Check notation and referenced sections as content, too. Use the preview
and build procedures in [site-validation](../SKILL.md).
