# Blog content

Use [blog-writing](../.agents/skills/blog-writing/SKILL.md) for prose and
[lecture-adaptation](../.agents/skills/lecture-adaptation/SKILL.md) when a deck
is authoritative. Figure work uses
[blog-figures](../.agents/skills/blog-figures/SKILL.md).

## Publication requirements

- Keep the filename date and publication `date` aligned. Every post edit,
  including a small correction, updates `last_updated` to today's date.
- Use `authors` even for one author. Preserve author identities and attribution.
- Set `editorial_status` explicitly unless the post is in the independently
  curated Selected collection. New AI-written posts use `ai-generated`.
  Humanizing prose does not establish human review.
- Change `ai-generated` to `human-reviewed` only after an explicit human
  editorial review. Set `selected: true` only when human curation requests it.
- The index sorts by date. Preserve intentional series metadata; do not add
  reading-path or roadmap blocks as an automatic finishing step.
- Begin with a brief reader-facing author note. Include context or source
  attribution that helps the reader; keep production and audit details internal.

For new posts or metadata changes, read the
[frontmatter reference](../.agents/skills/blog-writing/references/post-frontmatter.md).
For math, HTML, tables, footnotes, or figure includes, read the
[Jekyll rendering reference](../.agents/skills/site-validation/references/jekyll-rendering.md).

## House references

- [Fokker–Planck](2026-02-04-fokker-planck-equation.md): compact intuition and derivation.
- [Spherical equivariant layers](2026-02-02-spherical-equivariant-layers.md):
  longer foundations-to-architecture argument.

Use these as examples of depth, voice, and equation-to-explanation rhythm when
creating a tutorial or doing a substantial revision. They do not impose a word
count or require rereading for a typo. An authoritative source deck determines
scientific scope and order.

Each cited paper needs a corresponding References entry. Verify cross-links
against their actual targets and use section names rather than section numbers.
Lecture-derived articles should read as coherent arguments; omit course
logistics, agendas, exercises, and other classroom scaffolding unless requested.

Store figure assets under `assets/img/blog/`. Keep provenance with scripts or
manifests, and put concise verified attribution in captions. Figure creation,
extraction, and styling procedures belong in the figure and lecture skills.

Run `python3 scripts/validate_blog.py` before completing post changes.
