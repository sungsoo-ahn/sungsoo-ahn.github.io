# Instruction and skill revision — September 6, 2026

This is a historical implementation record, not an additional instruction layer.

## Scope and organization

Implemented the approved Codex-only, five-skill design for this Jekyll repository.
No global Codex settings, manuscript repository, site prose, or deployed site was
changed. The model upgrade motivated simpler guidance, not model-specific
capability assumptions or a new model configuration.

| Previous home | Current home |
| --- | --- |
| Blog prose plus overlapping academic-writing advice | [blog-writing](../../.agents/skills/blog-writing/SKILL.md), with conditional editorial and mathematical references |
| generate-blog-figures and download-paper-figures | [blog-figures](../../.agents/skills/blog-figures/SKILL.md), with generated/source/image routes |
| Long lecture-adaptation entrypoint | [lecture-adaptation](../../.agents/skills/lecture-adaptation/SKILL.md), with extraction and manifest details separated |
| Operational details spread through root instructions | [site-maintenance](../../.agents/skills/site-maintenance/SKILL.md) |
| jekyll-writing and preview/check procedures | [site-validation](../../.agents/skills/site-validation/SKILL.md), with rendering and visual-review references |

Root and post-directory AGENTS files retain local invariants and route to skills.
The six Claude skill adapters, two CLAUDE guides, and two Copilot agent files
were removed as approved. Their committed versions remain recoverable through
Git. Ignored machine-local settings were left alone.

The pre-implementation working tree had 1,257 lines across six skill
entrypoints; the five replacement entrypoints total 271 lines, about 78% less.
Root plus post instructions decreased from 228 to 123 lines. These are line
counts, not tokenizer measurements. Conditional references still cost context
when relevant, so these figures do not claim a 78% reduction for every task.

## Adopted practices

[Source register and licenses](../../.agents/third-party/sources.md) record exact
upstream revisions and local modifications.

- OpenResearch orx-figures: destination-size layout, physical-width-preserving
  exports, text geometry audits, aligned panels, and readable labels.
- Anti-Slop Writing: paragraph relationships, concrete mechanisms, earned emphasis.
- Humanizer: contextual editing that preserves voice, claims, assumptions,
  quantifiers, and uncertainty; no phrase blacklist or detector claims.
- Impeccable: reading-surface spacing, hierarchy, typography, and responsive
  visual review using the site's existing identity.

No extra auto-triggering upstream skills or ORX runtime were installed.
Punctuation, sentence length, and caption length are preferences rather than
universal bans or quotas. Full paper/manuscript layout remains outside this
repository's scope; paper-ready scientific figures are supported.

## Executable changes

[blog_figure_style.py](../../scripts/blog_figure_style.py) adds publication
styling, configurable widths, a geometric text audit, and fixed-size PDF/SVG/PNG
export. Legacy web save helpers and palette aliases retain their behavior.
Audits are review findings, not a certificate of visual correctness.

A baseline in-memory experiment requested 3.25 inches (234 points), but tight
cropping exported 242.39952 points. Publication export now overrides the global
tight-crop setting and preserves 234 points. Tests also cover a custom 4.2-inch
width, editable SVG text, absence of PDF Type 3 fonts, raster dimensions, and
restoration of figure canvas/layout and global settings.

[Instruction validation](../../scripts/validate_agent_hygiene.py) now validates
the canonical Codex tree, actual YAML metadata, and local Markdown resources.
It accepts supported optional metadata and multiline descriptions, skips fenced
examples, and no longer requires adapters or reads ignored local settings.

[Figure examples](../../scripts/preview_figure_quality.py) generate four analytic
fixtures into an explicit output directory, with an HTML gallery and audit JSON.
The [focused CI workflow](../../.github/workflows/agent-tooling.yml) runs the
regressions and retains those exports as review artifacts.

Jekyll excludes instructions, the entire .agents tree, historical agent audits,
and the existing temporary prompt. No active manifests or task notes were moved.

## Validation evidence

- Five skills passed the Skill Creator validator.
- Seven instruction-validator regression tests passed.
- Seven figure-helper regression tests passed, including hidden axes/ticks,
  shared axes, colorbars, multiline labels, arrow annotations, and individually
  ignored intentional overlaps.
- Repository agent hygiene and blog validation passed.
- An isolated Jekyll build passed in approximately 46 seconds. Existing theme
  Sass deprecation warnings remain. The output contained none of the excluded
  instruction/audit/prompt files.
- All four fixtures returned zero geometric findings. All four PDF exports were
  rendered with Poppler and visually inspected: curve, paired panels, heatmap,
  and editable diagram. Text, edges, panel spacing, and colorbar were legible.
- Browser inspection could not run: the browser runtime reported no available
  browsers, confirmed by its discovery list. Desktop/mobile page appearance is
  therefore not claimed as verified. No site CSS was changed by this task.
- Git whitespace validation passed.

The independent read-only forward test covered humanizing scientific prose, a
small typo, mathematical tutorial routing, missing native PPTX assets,
publication/CV propagation, reuse of an existing preview, and long figure
labels. It preserved mathematical meaning and workflow boundaries. It exposed
one operational caveat: existing CV tests can run generators. The validation
skill now distinguishes those tests from explicitly read-only check commands.

Existing draft posts, CSS edits, active ASBS task notes, the temporary prompt,
and lecture manifests were preserved. No CV/member generator, commit, push, or
deployment was performed. The temporary QA HTTP server was stopped; any
pre-existing Jekyll server was not restarted.
