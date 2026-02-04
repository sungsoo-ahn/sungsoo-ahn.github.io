# Blog Writing Guide

Instructions for writing and editing blog posts in `_posts/`.

## Post Frontmatter

Every post must include this frontmatter block:

```yaml
---
layout: post
title: "Your Title Here"
date: YYYY-MM-DD
last_updated: YYYY-MM-DD
description: "One-sentence summary for SEO and post listings."
order: 1  # controls display order (lower = appears later in list)
categories: [category-name]
tags: [tag1, tag2, tag3]
toc:
  sidebar: left
related_posts: false
---
```

## Author Note

Posts begin with an author note immediately after the frontmatter:

```html
<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: Your context here — background, acknowledgments, recommended further reading.</em>
</p>
```

## Content Structure

- Start with a roadmap table mapping sections to "why it's needed" — forces clear narrative dependency
- Lead with concrete examples before abstract formulas
- Introduce every symbol before first use; do a dedicated pass for undefined symbols at the end
- Collapse related works into paragraphs rather than giving each its own subsection
- Footnotes work well for alternate terminology without cluttering the main text
- When referencing publications, add to a References section as you cite, not at the end

## Prose Flow

- **Lead with the point, then justify** — e.g., "The BO approximation separates electrons from nuclei" before "because nuclei are 1836× heavier." Don't make readers wait through the setup to learn the result.
- **Show the destination before the derivation** — when presenting a key equation, state it upfront so readers know what the subsequent steps are building toward, then derive it.
- **Cut throat-clearing openers** — delete filler like "The equation says:", "What does X look like?", "The fundamental challenge is clear:", "The methods described above form the backbone of...". Just state the content.
- **Don't restate what was just shown** — if the math already demonstrated a property (e.g., determinant enforces antisymmetry), don't add a sentence restating it in words.
- **Break up stacked parentheticals** — a sentence with three em-dash clauses (e.g., "Ψ is the unknown — ..., E is the energy — ..., H is the operator — ...") should be split into separate sentences.
- **Merge redundant statements** — if two consecutive sentences say the same thing in different words (e.g., "exponentially large space" then "function from R^3N to C"), combine into one.
- **Be specific in section intros** — "neural networks have been applied at several points" → name the specific targets (XC functional, Hamiltonian, density).
- **Define terms next to first use** — if a symbol appears in an equation, define it immediately after, not "defined below." For symbols defined far earlier, add a brief reminder.
- **Drop dramatic qualifiers** — "fundamental physical flaw", "radically different approach", "enormous complexity" → just state the facts. Let the reader judge significance.

## Writing Quality

- Replace domain jargon with plain-language descriptions (readers likely don't know "quadrupole")
- One idea per sentence, one topic per paragraph
- Cut filler words: "very", "quite", "somewhat", "really", "basically"
- Eliminate redundancy: "completely eliminate" → "eliminate", "first introduce" → "introduce"
- Quantify claims: "improves significantly" → "improves by 5%"
- Avoid passive voice when the actor matters: "The model was trained" → "We trained the model"
- Fix undefined pronouns: "We applied it. It improved." → "We applied Algorithm 1. Accuracy improved by 5%."

## Math & LaTeX

- Use `$$...$$` (not `$...$`) for inline math containing multiple underscores — markdown interprets underscores as italics
- In inline math, use `\lvert...\rvert` instead of `|...|` for absolute values/norms — markdown interprets `|` as table column delimiters. Same applies to bra-ket notation: use `\mid` instead of `|` (e.g., `$$\langle \Psi \mid \hat{O} \mid \Psi \rangle$$`). Display math on its own line is not affected.
- Notation consistency: vectors as lowercase bold, matrices as uppercase bold, don't reuse symbols for different meanings
- Use descriptive subscripts: $$\theta_{\text{enc}}$$ not $$\theta_1$$
- Follow every equation with an explanation of its terms

## Figures

- Write a single Python script with separate functions per figure — easier to regenerate individually
- Use a consistent color palette across all figures with matching text colors for equation-figure correspondence
- Expect multiple rounds of visual iteration per figure; screenshots are the feedback loop
- Write captions that enable standalone understanding — a reader skimming figures alone should grasp the key point
- Store figure images in `assets/img/blog/`
- Embed figures using the Liquid include:

```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_figure.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Your caption here." %}
```

## CSS

- Distinct h2/h3 styling (border-bottom on h2, different font sizes) for visual hierarchy
- Explicit table borders — al-folio defaults are too subtle
- Add `overflow-y: visible` and padding to `mjx-container` CSS to prevent equation clipping
- These are one-time additions that carry over to future posts

## Workflow

- Keep the Jekyll server running and check rendered output frequently
- Commit at natural checkpoints (not after every small edit)
