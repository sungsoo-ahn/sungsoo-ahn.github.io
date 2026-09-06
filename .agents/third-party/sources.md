# Adapted upstream sources

Pinned during the September 6, 2026 instruction revision. These are adapted
local resources; no upstream runtime, installer, or automatic update is required.
Review upstream changes before adopting a new revision.

| Source | Revision | Local use | License |
| --- | --- | --- | --- |
| [OpenResearch figures](https://github.com/alphaXiv/OpenResearch/tree/95b2d961966c128b27f9d49753e227791a01c909/agent-skills/orx-figures) | `95b2d961966c128b27f9d49753e227791a01c909` | Figure guidance and selected layout/audit algorithms in [blog_figure_style.py](../../scripts/blog_figure_style.py) | [MIT, alphaXiv](alphaXiv-MIT.txt) |
| [Anti-Slop Writing](https://github.com/adewale/anti-slop-writing/tree/53370ff70b6d1da376e053cf144d39dca8d64f9e) | `53370ff70b6d1da376e053cf144d39dca8d64f9e` | [Editorial review](../skills/blog-writing/references/editorial-review.md), especially paragraph relationships and earned emphasis | [MIT, Ade Oshineye](anti-slop-writing-MIT.txt) |
| [Humanizer](https://github.com/blader/humanizer/tree/e2e92e7b4b8229253ed5c8e81dc65463fdeddda5) | `e2e92e7b4b8229253ed5c8e81dc65463fdeddda5` | Author voice, contextual pattern review, and claim-preservation checks in editorial review | [MIT, Siqi Chen](humanizer-MIT.txt) |
| [Impeccable](https://github.com/pbakaus/impeccable/tree/831cabee8b4bc1a2b66e5ae22003e9a19b57d464) | `831cabee8b4bc1a2b66e5ae22003e9a19b57d464` | Read-mode concepts from layout/typeset/polish, adapted into [visual review](../skills/site-validation/references/visual-review.md) | [Apache 2.0](impeccable-Apache-2.0.txt) |

## Local modifications

OpenResearch's print sizing and visual audit were adapted to the existing site
palette, editable SVG workflow, arbitrary destination widths, and legacy save
helpers. Audits produce review findings. They do not impose an approved-width
list, a proprietary font requirement, a blanket title ban, or empirical-only
data on educational toy plots. Statistical estimation helpers were not imported.

The prose adaptation treats watched phrases and punctuation as contextual
signals. It does not impose dash bans, mandatory critique schemas, artificial
personality, AI-detection claims, or repeated review loops.

The Impeccable adaptation preserves the existing reading surface. Its binaries,
hooks, metadata framework, native-platform references, and design-redirection
instructions were not imported. The upstream NOTICE concerns the unimported
iOS/Android references; it is retained as
[impeccable-NOTICE.md](impeccable-NOTICE.md).

## Supporting guidance

- [Astra prompting and migration guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [Codex skill discovery and progressive disclosure](https://learn.chatgpt.com/docs/build-skills)
- [Matplotlib constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html)

Licenses here govern adapted portions, alongside the repository's own license.
