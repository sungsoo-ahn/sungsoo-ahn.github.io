---
name: blog-writing
description: Write and edit blog posts with a direct, opinionated style — closer to a conference talk than a textbook. Use when drafting or revising blog posts.
---

# Blog Writing Style

Rules for writing and editing blog posts. Blog prose is direct and opinionated — closer to a conference talk than a textbook.

## Prose Task Mode

For blog-writing tasks, suppress the normal coding-agent response style. The
post is the output, not a report about the work.

Do not:

- report what you inspected, changed, or verified inside the post;
- organize the prose around tasks completed;
- convert the argument into documentation;
- optimize for exhaustive coverage;
- expose planning, validation, or implementation steps;
- add headings only to make the material scannable;
- conclude every section with a summary.

Write as the author speaking to the reader. Keep only the context, examples,
equations, figures, and caveats that help the intended reader understand the
argument. Do not use a word-count target as a definition of completeness. Stop
when the post has answered its central question at the depth its reader needs.

The quality checklist below diagnoses the argument that belongs in the post. It
is not a content checklist, and it must not cause new sections or background to
be added solely for coverage. When the requested deliverable is prose, return
the prose without appending a coding-agent completion report unless the user
explicitly asks for one.

## Prose Style

- **Lead with the point, then justify** — "The BO approximation separates electrons from nuclei" before "because nuclei are 1836x heavier." Don't make readers wait through setup to learn the result.
- **Cut throat-clearing openers** — delete filler like "The equation says:", "What does X look like?", "The fundamental challenge is clear:", "The methods described above form the backbone of...". Just state the content.
- **Don't restate what was just shown** — if the math already demonstrated a property, don't add a sentence restating it in words.
- **Drop dramatic qualifiers** — "fundamental physical flaw", "radically different approach", "enormous complexity" → just state the facts. Let the reader judge significance.
- **Avoid self-aware scaffolding sentences** — delete sentences like "The boundary of this claim matters", "This distinction is important", or "This raises an interesting question". Replace them with the concrete claim, comparison, or evidence.
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

## Quality Checklist (apply before finalizing)

### Substance

- Every paragraph must contain a concrete claim, result, argument, or necessary explanation.
- If the opening sentence can be deleted without losing information, delete or rewrite it.
- Replace vague praise ("important", "powerful", "compelling", "robust") with the specific mechanism, result, or consequence.
- Support novelty, causality, and superiority claims with evidence, numbers, comparisons, or citations.

### Relevance and Precision

- Every sentence should advance the paragraph's purpose.
- Remove background the intended reader already knows.
- Tie abstract nouns to specific objects, mechanisms, experiments, or numbers.
- Check pronouns such as "this", "it", and "these results" for clear antecedents.

### Formulaic Rhetoric

Flag these patterns; do not ban them when they carry real technical contrast:

- "It is important to note that ..."
- "This highlights/underscores/demonstrates ..."
- "Not only X, but also Y."
- "X is not merely A; it is B."
- "In today's rapidly evolving landscape ..."
- "Taken together", "Overall", or "In summary"
- A final sentence that merely restates the paragraph.
- A short dramatic sentence that tells the reader what to feel: "This matters." "The implication is clear."

### Structure and Rhythm

- Avoid making every paragraph follow the same claim -> explanation -> summary template.
- Watch for sections that are suspiciously equal in length or structure.
- Vary sentence length and syntax when several consecutive sentences sound alike.
- Use transitions only when the logical relation requires them.
- End each paragraph where the argument ends, not with a manufactured conclusion.

### Language

- Prefer direct verbs over noun phrases: "evaluate" instead of "conduct an evaluation of".
- Cut 10-20% of words when meaning survives.
- Use passive voice only when the actor is unknown, irrelevant, or intentionally de-emphasized.
- Remove redundant adjective pairs such as "novel and innovative" or "clear and evident".
- Keep technical terms only when they are necessary and precise.

### Voice

- Ask whether the passage sounds like something the author would actually say or write.
- Preserve legitimate uncertainty instead of converting everything into confident declarative prose.
- Include the author's actual judgment, not only a polished synthesis of conventional observations.
- Compare the passage against two or three known non-AI passages by the author when voice is uncertain.

## Coherence Checklist (apply when reviewing)

- Every forward reference ("Part 3 defines this") — verify the target actually contains what you claim
- Every backward reference ("from Part 2") — verify the source
- Every cited paper — verify it appears in the References section
- Every figure file referenced — verify it exists in `assets/img/blog/`
- Notation consistent throughout: same symbol = same meaning, same formatting convention
