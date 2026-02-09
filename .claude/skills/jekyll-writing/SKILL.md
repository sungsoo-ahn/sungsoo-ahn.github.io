---
name: jekyll-writing
description: Rendering rules for Jekyll sites with MathJax/KaTeX. Use when editing .md files that will be rendered by Jekyll to avoid common rendering pitfalls.
---

# Jekyll Writing Rules

Rendering rules for writing content on Jekyll sites with MathJax/KaTeX. Apply these whenever editing `.md` files that will be rendered by Jekyll.

## Math & LaTeX

- Use `$$...$$` (not `$...$`) for inline math with multiple underscores — markdown interprets underscores as italics
- In inline math, use `\lvert...\rvert` instead of `|...|` for absolute values/norms — markdown interprets `|` as table column delimiters. Same for bra-ket notation: use `\mid` instead of `|` (e.g., `$$\langle \Psi \mid \hat{O} \mid \Psi \rangle$$`). Display math on its own line is not affected.
- **Do not use `\$` for dollar signs** — MathJax/KaTeX interprets `\$` as a math delimiter, breaking rendering. Write "dollars" or "USD" in prose instead.
- Notation consistency: vectors as lowercase bold ($$\mathbf{x}$$), matrices as uppercase bold ($$\mathbf{X}$$), don't reuse symbols for different meanings

## Mermaid Diagrams

- Use fenced code blocks with `mermaid` language identifier
- Keep node labels short — long labels break diagram layout
- Use `style` directives for consistent coloring

## General Rendering

- Keep the Jekyll server running and check rendered output frequently — do not kill and restart on every edit
- `overflow-y: visible` and padding on `mjx-container` CSS prevents equation clipping
- Explicit table borders may be needed — al-folio defaults are subtle
