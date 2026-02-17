---
name: jekyll-writing
description: Rendering rules for Jekyll sites with MathJax/KaTeX. Use when editing .md files that will be rendered by Jekyll to avoid common rendering pitfalls.
---

# Jekyll Writing Rules

Rendering rules for writing content on Jekyll sites with MathJax/KaTeX. Apply these whenever editing `.md` files that will be rendered by Jekyll.

## Math & LaTeX

- Use `$$...$$` (not `$...$`) for inline math with multiple underscores — markdown interprets underscores as italics
- **Inside raw HTML elements** (`<div>`, `<span>`, figure captions), use `\(...\)` instead of `$$...$$` for inline math — MathJax treats `$$...$$` as display math inside HTML blocks, causing line breaks
- In inline math, use `\lvert...\rvert` instead of `|...|` for absolute values/norms — markdown interprets `|` as table column delimiters. Same for bra-ket notation: use `\mid` instead of `|` (e.g., `$$\langle \Psi \mid \hat{O} \mid \Psi \rangle$$`). Display math on its own line is not affected.
- **Do not use `\$` for dollar signs** — MathJax/KaTeX interprets `\$` as a math delimiter, breaking rendering. Write "dollars" or "USD" in prose instead.
- Notation consistency: vectors as lowercase bold ($$\mathbf{x}$$), matrices as uppercase bold ($$\mathbf{X}$$), don't reuse symbols for different meanings

## Mermaid Diagrams

- **Export as images:** Write `.mmd` file, render to PNG via `mmdc` CLI, and embed as `<img>` tags. Do NOT use fenced Mermaid code blocks in markdown.
- Save `.mmd` source + `.png` output in `assets/img/teaching/protein-ai/mermaid/`
- Use `style` directives for consistent coloring

### Layout

- Prefer `flowchart LR` (left-right) for pipeline/flow diagrams. Use `flowchart TD` (top-down) when LR would be too wide (many parallel branches).
- For mixed layouts, use the primary direction with `direction LR` or `direction TD` inside individual subgraphs
- Avoid dense cross-connections (e.g., `A & B & C --> D & E & F`) — these create wide diagrams. Use subgraph-level arrows or sequential chains instead
- Remove self-loops (`A -.-> A`) — they extend far outside the viewBox and cause severe clipping

### Node Labels

- Keep labels to 1–2 lines. Three-line labels frequently get clipped at the bottom of the diagram
- Shorten verbose labels: "30 edges per node" → "(L nodes, 30·L edges)"; "context-aware representation" → "(context-aware)"
- When a subgraph is at the bottom of a diagram, keep its inner content minimal — Mermaid's viewBox calculation underestimates bottom extent

### ViewBox Clipping (Known Mermaid Bug)

- Mermaid's auto-generated SVG viewBox is consistently too tight, clipping content on all sides (especially bottom)
- This site uses a post-render JS fix in `assets/js/mermaid-setup.js` that expands the viewBox by 80px padding on all sides
- Do NOT set `overflow: visible` on `pre.mermaid` — it causes diagram content to overlap with text below
- The CSS in `_sass/_base.scss` (`pre.mermaid`) must NOT use `!important` on SVG `max-width` — this overrides Mermaid's inline sizing and stretches small diagrams to full container width

### Verification

- Always verify diagram rendering visually — CSS/markdown preview cannot catch viewBox clipping
- Use Playwright (Python) for headless screenshots: take full-page screenshots cropped around each `pre.mermaid` element to check for clipping on all sides

## General Rendering

- Keep the Jekyll server running and check rendered output frequently — do not kill and restart on every edit
- `overflow-y: visible` and padding on `mjx-container` CSS prevents equation clipping
- Explicit table borders may be needed — al-folio defaults are subtle
