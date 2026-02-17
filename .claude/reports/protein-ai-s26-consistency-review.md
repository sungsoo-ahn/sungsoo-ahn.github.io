# Protein & AI Lecture Notes — Consistency Review Report

**Date:** 2026-02-17 (updated from 2026-02-15)
**Scope:** All 10 notes (P1–P4, L1–L6)
**Status:** All notes processed. Jekyll build succeeds.

---

## Summary

| Metric | Count |
|--------|-------|
| Notes reviewed | 10 |
| Total issues found & fixed | 30+ |
| Cross-cutting issues resolved | 10 |
| Jekyll build errors | 0 |

---

## Issues Fixed (2026-02-17 Pass)

### Display Math in HTML Captions (11 captions)
`$$...$$` inside `<div>` elements renders as display math (centered, block-level). Replaced with `\(...\)` for inline rendering.
- **P1:** 3 captions (tensor dimensions, linear regression neuron, gradient descent)
- **P2:** 5 captions (shallow net anatomy, fully connected network, shallow functions, depth composition, activation functions)
- **P3:** 3 captions (loss surface, SGD trajectory, momentum)

**New rendering rule added** to `.claude/skills/jekyll-writing/SKILL.md` and memory.

### Broken `{% cite %}` Tags (12 tags)
Bibliography keys did not exist in `_bibliography/papers.bib`. Replaced with numbered references.
- **L3:** 6 tags → `[1]`–`[5]` (lin2023evolutionary, meier2021language, hu2022lora, rives2021biological, elnaggar2022prottrans)
- **L5:** 6 tags → `[1]`, `[4]`, `[5]`, `[7]` (watson2023novo, baek2021accurate, ho2022classifier, dauparas2022robust)

### Figure Placement (P2)
- Moved "Anatomy of a shallow neural network" from Section 2.4 (Single Neuron) → Section 2.5 (Layers)
- Moved "What a single hidden layer can compute" from Section 2.3 → end of Section 2.5

### Other Fixes
- **P1:** Merged duplicate autograd/chain rule sentence
- **P3:** Fixed citation `[6]` → `[5]` for AdamW paper
- **P3:** Added missing blank line before `### Prerequisites` (was breaking roadmap table)
- **L2:** Removed nonexistent "Exercises" row from roadmap table
- **L6:** Clarified duplicate ProteinMPNN reference (Science [1] vs. bioRxiv preprint [8])

---

## Issues Fixed (2026-02-15 Pass)

| # | Issue | Edits |
|---|-------|-------|
| 1 | "Let us" phrasing → direct constructions | 6 |
| 2 | Auto-generated Mermaid alt text → descriptive | 25 |
| 3 | `## Summary` → `## Key Takeaways` | 2 |
| 4 | Added `## Key Takeaways` to L4, L6 | 2 |
| 5 | Numbered `## 1. Introduction` → unnumbered | 1 (L5) |
| 6 | Numbered `## 13. Key Takeaways` → unnumbered | 1 (L3) |

---

## Knowledge Graph Summary

**55 entities** across 4 types:
- 10 LectureNote entities (P1–P4, L1–L6)
- ~45 Concept entities spanning all notes

### Key Cross-Note Dependencies

| Concept | Introduced | Used In |
|---------|-----------|---------|
| Attention (Q/K/V) | L1 § 2 | L3 (ESM-2), L4 (Evoformer), L5 (IPA) |
| Message Passing | L1 § 6 | L4 (triangular updates), L5, L6 (ProteinMPNN encoder) |
| Diffusion Model | L2 §§ 6–8 | L5 (SE(3) diffusion) |
| SE(3) Equivariance | L1 § 7 | L4 (IPA), L5 (full treatment) |
| Loss Function | P1 § 4 | P3 (MSE/BCE/CE), P4, L5, L6 |
| Classifier-Free Guidance | L2 § 11 | L5 § 9 |

### Notation Consistency

| Symbol | Meaning | Introduced | Consistent? |
|--------|---------|-----------|-------------|
| θ | Model parameters | P1 | ✓ |
| L(θ) | Loss function | P1 | ✓ |
| η | Learning rate | P1 | ✓ |
| σ | Activation function | P2 | ✓ |
| W, b | Weights, bias | P1 | ✓ |
| h | Hidden representation | P2 | ✓ |
| L | Sequence length | P2 | ✓ |
| x_t | Noisy data at step t | L2 | ✓ |
| Q, K, V | Query, key, value | L1 | ✓ |
| (R_i, t_i) | SE(3) rigid frame | L4 | ✓ (L4, L5) |

No notation conflicts detected.

---

## Deferred Issues

| # | Issue | Severity | Reason |
|---|-------|----------|--------|
| 1 | `col-sm` width varies across figures | Low | Cosmetic; 60+ figures affected |
| 2 | Two files share `s26-04` prefix (P4 and L1) | Low | Renaming would break URLs |
| 3 | L3 roadmap uses 2-column format vs standard 3 | Low | Functional as-is |
| 4 | L4, L5 references lack DOI links (L6 has them) | Low | Stylistic consistency |

---

## Quality Checklist

- [x] Every concept defined before first use (or in a prior note)
- [x] Cross-references use correct note numbers and titles
- [x] All `{% cite %}` tags replaced with working numbered references
- [x] No `$$...$$` inside HTML elements
- [x] Section numbering consistent within each note
- [x] All 10 notes have `## Key Takeaways` section
- [x] `jekyll build --incremental` succeeds without errors
