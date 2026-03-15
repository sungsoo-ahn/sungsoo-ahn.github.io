---
name: blog-writing
description: Write and edit blog posts with a direct, opinionated style — closer to a conference talk than a textbook. Use when drafting or revising blog posts.
---

# Blog Writing Style

Rules for writing and editing blog posts. Blog prose is direct and opinionated — closer to a conference talk than a textbook.

## Prose Style

- **Lead with the point, then justify** — "The BO approximation separates electrons from nuclei" before "because nuclei are 1836x heavier." Don't make readers wait through setup to learn the result.
- **Cut throat-clearing openers** — delete filler like "The equation says:", "What does X look like?", "The fundamental challenge is clear:", "The methods described above form the backbone of...". Just state the content.
- **Don't restate what was just shown** — if the math already demonstrated a property, don't add a sentence restating it in words.
- **Drop dramatic qualifiers** — "fundamental physical flaw", "radically different approach", "enormous complexity" → just state the facts. Let the reader judge significance.
- **Merge redundant statements** — if two consecutive sentences say the same thing in different words, combine into one.
- **Break up stacked parentheticals** — a sentence with three em-dash clauses should be split into separate sentences.

## Clarity

- One idea per sentence, one topic per paragraph
- Cut filler words: "very", "quite", "somewhat", "really", "basically"
- Eliminate redundancy: "completely eliminate" → "eliminate", "first introduce" → "introduce"
- Quantify claims: "improves significantly" → "improves by 5%"
- Avoid passive voice when the actor matters: "The model was trained" → "We trained the model"
- Fix undefined pronouns: "We applied it. It improved." → "We applied Algorithm 1. Accuracy improved by 5%."
- Follow every equation with an explanation of its terms

## Structure and Readability

- **No wall-of-text paragraphs.** If a paragraph exceeds ~8 lines or contains multiple distinct ideas, split it.
- **Sentences under ~40 words.** Split chains of em-dashes or nested parentheticals into separate sentences.
- **Don't bury key points.** Important results belong at the start of a paragraph or in a boxed definition, not mid-paragraph.
- **Signpost transitions.** Before a derivation or definition, add a setup sentence explaining what's coming and why.
- **Don't repeat yourself.** If a point (e.g., "the variance problem") appears in multiple places, state it fully once and back-reference elsewhere.
- **Figures near their discussion.** Place figures immediately after the text that introduces them, not paragraphs later.
- **Notation introductions need breathing room.** Don't introduce 3+ new symbols in one paragraph with no prose between equations.

## Coherence Checklist (apply when reviewing)

- Every forward reference ("Part 3 defines this") — verify the target actually contains what you claim
- Every backward reference ("from Part 2") — verify the source
- Every cited paper — verify it appears in the References section
- Every figure file referenced — verify it exists in `assets/img/blog/`
- Notation consistent throughout: same symbol = same meaning, same formatting convention
