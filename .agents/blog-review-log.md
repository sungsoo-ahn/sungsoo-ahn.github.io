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

## 02. How Drug Discovery Turns Biological Hypotheses into Molecules

- **Post:** `_posts/2026-08-08-drug-discovery-target-to-clinic.md`
- **Topic-matched reference:** `_posts/2026-03-03-protein-design-for-ml.md`
  for its molecular-design funnel and experiment-facing argument.
- **Before:** The post has a strong four-claim thesis and useful PK/PD equations,
  but at 2,230 substantive words it reads as a compressed map. Target
  validation, occupancy, assay potency, multi-objective SAR, oral exposure,
  therapeutic margin, and clinical inference are introduced independently.
  Compared with the references, no running candidate exposes how evidence and
  uncertainty propagate across the chain, and several equations are stated
  without a worked calculation.
- **Criteria promoted:** Application tutorials benefit from a running scientific
  scenario; model evaluation must name the decision, filtered population, and
  denominator rather than stopping at a benchmark metric.
- **Planned revision:** Preserve all H2s. Add a running target/candidate example,
  occupancy and residence-time calculations, a small SAR/Pareto decision,
  oral one-compartment exposure and free-concentration calculations, a
  therapeutic-window example, and a biomarker-based clinical inference tree.
- **After:** Expanded to 5,991 substantive body words while preserving all nine
  H2s and their order. A single KX/A2 program now carries the argument from a
  causal intervention claim through occupancy, assay correction, binding
  kinetics, SAR/Pareto selection, oral PK, free site exposure, therapeutic
  margin, clinical outcomes, and prospective ML evaluation. New derivations
  include mass-balance occupancy, Cheng--Prusoff, matched-affinity residence
  times, the Bateman concentration curve and steady-state trough, dose-dependent
  benefit versus harm, risk differences, Bayesian updating, and biomarker
  interaction. Added one reproducible quantitative exposure-window figure and
  three references. Figure regeneration, independent arithmetic checks,
  citation bijection and link verification, LaTeX/caption audit, blog validation,
  diff check, and a clean rendered Jekyll build passed. The two promoted skill
  rules directly shaped the running scenario and denominator-aware evaluation;
  no further general rule was needed after revision.

## 03. Materials Discovery Connects Structure, Properties, and Synthesis

- **Post:** `_posts/2026-08-08-materials-discovery-structure-properties-synthesis.md`
- **Topic-matched reference:** `_posts/2026-02-05-electrocatalysis-ml.md` for
  its physical through-line from an application constraint to a descriptor,
  search space, failure of idealized surfaces, and an ML decision.
- **Before:** The chapter has the right eight-stage argument and a useful AB3
  convex-hull calculation, but at 2,666 substantive words most consequences are
  named only once. Composition, state-dependent properties, hull stability,
  defects, periodic learning, inverse design, synthesis, and characterization
  remain adjacent abstractions. Compared with the references, the same
  candidate is not carried across fidelity levels, so finite-temperature,
  defect, process, and measurement corrections never visibly change a decision.
- **Criteria promoted:** When a scientific argument uses nested physical
  approximations, carry the same quantity through multiple fidelity levels and
  show whether restoring a neglected term or reference changes the conclusion.
- **Planned revision:** Preserve every H2 and its order. Carry the hypothetical
  AB3 candidate from equivalent periodic descriptions through state-specific
  electronic/mechanical targets, hull uncertainty and finite-temperature phase
  competition, defect populations, periodic-graph prediction, constrained
  inverse design, a process window, and claim-matched characterization. Add
  derivations and numerical checks where they change the candidate decision;
  add a figure only if the existing four cannot support that reasoning.
- **After:** Expanded from 2,666 to 4,611 substantive words while preserving all
  nine H2s and their exact order. The hypothetical $$\alpha$$-AB$$_3$$ candidate
  now passes through primitive/supercell equivalence, state-specific electronic
  and elastic targets, an expanded hull, correlated energy uncertainty,
  finite-temperature reranking, reservoir-dependent defect populations and
  conductivity, periodic graph construction, joint design feasibility, a
  temperature-dependent synthesis window, and claim-matched characterization.
  The revision explicitly distinguishes Figure 1's generic A$$_2$$B$$_2$$
  illustration and reuses all four existing figures. It adds primary references
  for defect thermodynamics and Rietveld refinement. Independent arithmetic,
  LaTeX/scaffolding, citation bijection and DOI, blog validation, diff, clean
  Jekyll build, and rendered-page checks passed. The promoted fidelity-hierarchy
  rule directly produced several visible decision reversals; no further general
  criterion was needed after revision.

## 04. Graph Neural Networks as Learnable Message Passing

- **Post:** `_posts/2026-08-08-graph-neural-networks-message-passing.md`
- **Topic-matched reference:** No additional older post is close enough to be a
  fair comparator; the spherical-equivariant permanent anchor supplies the
  relevant architecture-building comparison.
- **Before:** The post has a sharp symmetry-first thesis and a clean five-node
  update, but at roughly 3,000 substantive words it becomes an architecture
  catalog after the shared message-passing equation. GCN, GraphSAGE, GAT, and
  GIN are each described correctly on separate abstractions, so the reader never
  sees their different normalization and information loss on the same input.
  Equivariance, receptive-field growth, graph readout, structural attention,
  and computational scaling are stated more often than derived or checked.
- **Criteria promoted:** In a comparative architecture tutorial, run the same
  controlled input through each alternative and expose intermediate outputs;
  holding data and task fixed turns a model list into a causal comparison.
- **Planned revision:** Preserve every H2 and its order. Prove the layer's
  permutation equivariance, deepen the five-node example into a shared
  architecture comparison, derive normalized GCN weights and receptive-field
  propagation, contrast sum/mean/attention multiplicity and degree behavior,
  make graph-level extensivity explicit, work a structural-attention example,
  and quantify sparse/local versus dense/global computation. Keep limitations
  concise and cross-link the dedicated expressivity and failure-mode chapters
  rather than duplicating them. Add a figure only for a real explanatory gap.
- **After:** Expanded from 2,859 to 5,401 substantive body words while
  preserving all eight H2s and their exact order. One fixed five-node graph now
  supports the explicit layer equivariance proof, receptive-field matrix
  powers, extensive versus intensive readout, normalized GCN coefficients,
  GraphSAGE sampling variance, a concrete GAT softmax, GIN, and a controlled
  multiset-information comparison. A shortest-path structural-attention example
  and sparse-versus-dense complexity calculation complete the local/global
  argument, while the limitations now cross-link the dedicated expressivity and
  failure-mode chapters. The three existing figures were sufficient. Arithmetic,
  math/scaffolding, nine-way citation bijection and primary-link resolution,
  blog validation, diff, clean Jekyll build, and parsed rendered-page checks
  passed. The promoted controlled-input rule shaped the entire architecture
  comparison; no further general criterion was needed after revision.

## 05. What Graph Neural Networks Can and Cannot Distinguish

- **Post:** `_posts/2026-08-08-graph-neural-network-expressivity.md`
- **Topic-matched reference:** No additional older post is a fair expressivity
  analogue. The two permanent anchors set the quality standard, and the newly
  polished message-passing chapter is used only as a continuity check.
- **Before:** The post has the right representation-to-WL-to-remedy storyline,
  a clean five-node refinement, and the useful $$C_6$$ versus $$2C_3$$ witness.
  At roughly 2,700 substantive words, however, the ceiling theorem compresses
  its quantifiers into prose, the path refinement omits its actual signatures,
  and the four remedies are introduced on different abstractions rather than
  tested against the same collision. The final cost/generalization tradeoff is
  qualitative.
- **Criteria promoted:** State the quantifiers in an impossibility or
  expressivity result, give a concrete witness pair, separate architectural
  collision from optimization failure, and say which theorem assumption each
  remedy changes.
- **Planned revision:** Preserve every H2 and its order. Formalize the quotient
  induced by a representation, derive multiset collisions and an injective
  bounded encoding, work the full 1-WL signatures on the path, prove the MPNN
  ceiling by induction with explicit quantifiers, and carry $$C_6$$ versus
  $$2C_3$$ through tuple, motif, subgraph, and spectral remedies. Quantify the
  state/computation costs and close with task-dependent approximation examples.
  Cross-link message passing rather than re-explaining its architecture survey;
  add a figure only for a genuine gap.
- **After:** Expanded from 2,763 to 4,790 substantive body words while
  preserving all eight H2s and their exact order. The revision formalizes the
  representation quotient and target-approximation lower bound, derives
  elementary multiset collisions and a bounded injective sum, works every
  $$P_5$$ refinement signature, and states/proves the MPNN ceiling with explicit
  assumptions and quantifiers. The fixed $$C_6$$ versus $$2C_3$$ witness now
  passes through precisely named folklore 2-WL/2-FWL common-neighbor
  refinement, triangle counts, vertex-deletion decks, and Laplacian spectra,
  with computation and target-relative tradeoffs. Added the Maron et al.
  primary citation after resolving the coordinate-wise versus folklore-WL
  nomenclature; reused all four figures. Arithmetic/math/scaffolding, 7/7/7
  citation bijection and link resolution, blog validation, diff, clean Jekyll
  build, and rendered checks passed. The promoted quantifier/witness/assumption
  rule directly shaped the theorem and remedies; no further general criterion
  was needed after revision.
