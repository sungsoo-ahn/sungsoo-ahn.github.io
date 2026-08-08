# Tutorial Comparison and Revision Log

This agent-facing log records the sequential comparison of the 27 canonical
lecture-derived tutorials against the site's polished house references.

Permanent anchors:

- `_posts/2026-02-04-fokker-planck-equation.md`
- `_posts/2026-02-02-spherical-equivariant-layers.md`

Each entry records the optional topic-matched reference, concrete gaps found,
generalizable criteria promoted to skills, revisions made, and verification.
Post-specific observations remain here rather than becoming universal rules.

## 01. Chemistry and Physics for Molecular Machine Learning

- **Post:** `_posts/2026-08-08-chemistry-physics-molecular-ml.md`
- **Topic-matched reference:** `_posts/2026-02-03-quantum-chemistry-dft.md`
- **Before:** The post already has a coherent representation-to-dynamics chain
  and strong physical scope. Compared with the anchors, it compresses several
  conceptual transitions: representation loss is described but not calculated;
  symmetry laws lack an explicit constraint example; Born--Oppenheimer and
  force/curvature relations move quickly; Boltzmann populations and barrier
  errors receive only one-step calculations.
- **Criteria promoted:** Long tutorials need an argument rather than a topic
  inventory; derivations should expose the non-obvious step and end with an
  operational check; figures should resolve a conceptual transition and be
  used by the surrounding argument.
- **Planned revision:** Preserve the existing H2 order. Add H3 derivations and
  worked examples for representation ambiguity, symmetry constraints,
  timescale separation, the Morse potential, Boltzmann free-energy/population
  conversion, and transition-state rate sensitivity.
- **After:** Expanded from 3,970 to 5,324 substantive body words while
  preserving every H2. Nine H3 units now derive or calculate the previously
  compressed transitions: ambiguity classes and irreducible error, chirality,
  invariant-energy force constraints, Born--Oppenheimer timescales and
  derivative couplings, the Morse force/curvature limits, finite-temperature
  basin populations, and exponential rate sensitivity. The existing four
  figures remain appropriate because each now sits inside a fuller derivation.
  Citation bijection, primary links, LaTeX audit, blog validation, diff check,
  and a clean rendered Jekyll page passed. No additional general skill rule was
  needed after applying the promoted criteria.
