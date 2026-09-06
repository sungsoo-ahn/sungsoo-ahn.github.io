---
name: lecture-adaptation
description: Adapt an authoritative lecture PDF or PPTX into a source-faithful blog article. Use for deck coverage, source-order revisions, lecture figure reuse, and existing lecture manifests.
---

# Lecture adaptation

The source deck determines the scientific scope and order. Use
[blog-writing](../blog-writing/SKILL.md) for readable prose; house examples
guide quality without imposing a different argument.

## Source fidelity

Inventory slides before outlining a new adaptation. Separate logistics,
literal repetition, and decorative material from substantive content.
Preserve the deck's concepts, equations, examples, comparisons, caveats,
historical reasoning, and final scientific judgment.

Retain cumulative definitions and distinctions between objects or task families.
A recap can be substantive if it changes the comparison or motivates the next
method. Repeated constructions in different geometries may teach an analogy;
do not collapse them merely because the formulas look similar.

Map substantive slides to prose, native math/tables, or figures. Use existing
durable manifest conventions where the post or workflow requires them. Reuse
prior prose only when supported by the source. Add connective explanations
without inventing branches, results, or examples to satisfy a length target.

## Figures and authority

Honor reuse permission already confirmed for the supplied deck. Verify only
unresolved rights or missing sources. A blocked figure does not block independent
inventory or prose work; report the specific outstanding asset.

For extracting or auditing lecture visuals, read
[extraction and manifests](references/extraction.md). The source PDF can guide
prose when it is the supplied authority. A workflow requiring native PPTX figures
needs the corresponding PPTX; do not silently replace required native extraction
with a PDF crop or invented redraw.

## Completion

Check slide coverage and scientific order, then run
`python3 scripts/validate_blog.py` for affected posts and manifests. Inspect
substantial rendered changes for visual legibility and coherent exposition.

Keep the opening source note brief and reader-facing. Put detailed production
and provenance records in the manifest or task audit. Historical review logs
belong in `docs/agent-audits/`, not in skill instructions.
