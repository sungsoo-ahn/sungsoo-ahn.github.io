# Article layout and visual review

Adapted from Impeccable's Read-mode layout, typesetting, and polish references.
See the [source register](../../../third-party/sources.md). Use the existing
site as the design authority; no separate design framework is required.

## Assess the reading surface

Identify the main reading path and inspect representative rendered content.
Related elements should read as a group; separate a new section more strongly
than a heading from its own paragraph. Judge optical alignment as well as CSS
values. Reuse established spacing values when they already express the role.

Check heading hierarchy, paragraph rhythm, figure/caption proximity, and gaps
around equations and tables. Use the site's existing fonts and palette.
A readable line measure is usually around 45–75 characters, but tune it to the
actual face, mathematical content, and language rather than enforcing a quota.

## Stress the composition

Inspect desktop (about 1280 CSS pixels) and narrow mobile (about 390 pixels),
plus an intermediate width when wrapping or column changes make it relevant.

- Long headings and captions should wrap without colliding with metadata.
- Figures should retain readable labels; do not expand a small diagram solely
  to fill the container.
- Captions should stay grouped with their image and separate from following text.
- Tall equations need room for accents, limits, and equation numbers.
- Wide math and tables may scroll locally; the page should not overflow.
- Check supported light/dark themes, font fallback, and browser zoom.
- Preserve focus visibility, useful alt text, heading semantics, and logical
  reading order.

## Fix and confirm

Fix the narrowest cause: an asset's canvas, its include sizing, a shared caption
rule, or page layout. A clean geometry scan does not prove good hierarchy.
Inspect once, fix observed issues together, and confirm them. Continue only for
specific unresolved defects. Do not rewrite claims or change the site's identity
to make a polish pass more conspicuous.
