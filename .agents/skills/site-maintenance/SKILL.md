---
name: site-maintenance
description: Maintain this academic homepage's publications, CV, members, courses, pages, and Jekyll theme. Use for site changes and structured-content updates, not article prose or full paper manuscripts.
---

# Site maintenance

Inspect the current personal site rather than assuming all upstream al-folio
examples remain. Follow existing data ownership and regenerate only the
outputs affected by the requested change.

## Structured content

| Task | Input and update |
| --- | --- |
| Publications | Edit `_data/publications.yml`; the website reads it directly. Validate with `uv run python scripts/update_publications.py --check`. Regenerate the CV when its bibliography changes. |
| CV content | Edit `_data/cv_content.yml`; run `uv run python scripts/update_cv.py` to regenerate LaTeX and PDFs. |
| CV design | Edit outside the SYNC blocks in `cv/cv.tex`, then compile through the CV updater. |
| Lab members | Use the workbook selected by `SPML_MEMBERS_XLSX` or the documented default; run `uv run python scripts/update_members.py`. |
| Courses and links | Edit `_data/courses.yml` and the applicable page/social metadata; course notes remain in standalone repositories. |

The CV updater also produces `cv/publications.tex`, `cv/cv.pdf`,
`assets/pdf/cv.pdf`, and `cv/source.sha256`. Keep them synchronized.
`--no-compile` updates LaTeX only; it does not establish that PDFs are current.
`--check --no-compile` is read-only.

The member importer has no dry-run flag and overwrites `_pages/people.md`.
Do not run it as a validation command. If the workbook is unavailable, report
that dependency rather than inventing members or manually patching its output.

## Pages and theme

Use the existing layouts/includes and
[palette](../../../docs/palette.md). For rendering-sensitive Markdown, read
[Jekyll rendering](../site-validation/references/jekyll-rendering.md).

For requested spacing, typography, or layout work, read
[visual review](../site-validation/references/visual-review.md). Preserve the
established site identity and user edits; use observed layout defects to choose
the smallest shared or local change. Theme refinement does not authorize
rewriting factual copy.

Use [site-validation](../site-validation/SKILL.md) for the relevant checks and
preview procedure. A requested local change does not itself request a push;
pushing main deploys the site.
