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

## 06. Depth, Over-Smoothing, and Over-Squashing in Graph Networks

- **Post:** `_posts/2026-08-08-deep-graph-network-failure-modes.md`
- **Topic-matched reference:** No additional older post is a fair analogue.
  The permanent anchors set the quality standard; the revised message-passing
  and expressivity chapters are continuity checks only.
- **Before:** The chapter makes the essential conceptual separation among
  under-reaching, over-smoothing, and over-squashing, and its spectral and
  binary-tree equations are correct. At roughly 2,400 substantive words,
  however, most mechanisms are demonstrated only once and the remedy sections
  read as parallel lists. There is no controlled calculation showing a residual
  path preserve contrast while leaving a topological bottleneck intact, or a
  shortcut improve sensitivity while accelerating diffusion. Diagnostics are
  named but not calibrated on the same graphs.
- **Criteria promoted:** Tutorials about similar-looking failures should use a
  differential diagnosis: mechanism-specific observable, matched intervention,
  and a counterfactual where the wrong remedy fails or worsens another mode.
- **Planned revision:** Preserve every H2 and its order. Work an explicit
  under-reaching task, derive spectral smoothing and Dirichlet-energy decay on a
  small graph, analyze residual/initial-feature paths as graph filters, deepen
  the tree Jacobian and effective-resistance calculations, quantify how a
  shortcut changes resistance and mixing, and finish with a diagnostic matrix
  that maps measurements to remedies and falsifying outcomes. Cross-link the
  preceding graph chapters rather than repeating them; add a figure only for a
  genuine explanatory gap.
- **After:** Expanded from 2,431 to 4,674 substantive body words while
  preserving all seven content H2s and their exact order. The chapter now works
  an exact six-layer under-reaching task, derives the spectral limit and a
  $$C_4$$ Dirichlet-energy decay, compares residual, initial-injection, and
  Jumping-Knowledge filters on the same mode, deepens the binary-tree Jacobian
  bound, and follows $$P_5\to C_5$$ through resistance, commute time, and mixing.
  Residual and rewiring counterfactuals, task-sensitive curvature, and a
  diagnostic/falsification matrix make the failure modes operational. Added the
  primary Chandra et al. commute-time reference and reused all four figures.
  Arithmetic, LaTeX/scaffolding, 9/9/9 citation bijection and links, blog
  validation, diff, clean Jekyll build, and rendered checks passed. The promoted
  differential-diagnosis rule shaped both the controlled counterfactuals and
  final matrix; no further general criterion was needed after revision.

## 07. Symmetry and Equivariance for Geometric Data

- **Post:** `_posts/2026-08-08-symmetry-equivariance-geometric-data.md`
- **Topic-matched reference:** The spherical-equivariant permanent anchor is
  already the direct mature comparator; no additional older post is needed.
  The revised message-passing chapter is used only for permutation continuity.
- **Before:** The chapter has a clean orbit-to-group-to-representation-to-
  equivariance argument and good task-specific cautions. At roughly 2,600
  substantive words, it still moves quickly from definitions to conclusions.
  The semidirect rigid-motion law, representation homomorphism, safe feature
  operations, force transformation, group averaging, and risk restriction are
  mostly stated once. The reader learns what equivariance means but sees little
  of how equivariant primitives compose into an architecture or how a natural
  nonlinear operation violates a feature type.
- **Criteria promoted:** When an abstract property is supposed to constrain an
  entire architecture, prove closure under composition, demonstrate safe
  primitives, and contrast them with a plausible operation that breaks the
  property.
- **Planned revision:** Preserve every H2 and its order. Carry one small
  geometric object through orbit/stabilizer and canonicalization, derive rigid
  motion composition and inverses, verify permutation/vector/tensor
  representations, include parity-sensitive feature types, prove equivariant
  composition and force-from-energy transformation, work safe versus unsafe
  nonlinearities and readouts, derive finite/compact group averaging, and make
  the augmentation-versus-exact-symmetry and wrong-symmetry tradeoffs numerical.
  Cross-link later geometric chapters rather than previewing all their machinery;
  add a figure only for a real gap.
- **After:** Expanded from 2,685 to 4,867 substantive body words while
  preserving all six H2s and their exact order. One typed V-shaped object now
  grounds orbit, stabilizer, quotient, and a narrowly qualified PCA-frame
  discontinuity. The chapter derives the rigid-motion law and inverse,
  distinguishes parity/chirality and feature representations, proves composition
  closure, contrasts safe and unsafe nonlinearities, derives energy-based force
  equivariance and conservation identities, proves group averaging, and works
  finite-augmentation and wrong-symmetry error counterexamples. Fixed a dropped
  `\qquad`, normalized three primary citations, and reused all four figures.
  Arithmetic/LaTeX/scaffolding, citation bijection and links, blog validation,
  diff, clean Jekyll build, rendered HTML, and full-page visual checks passed.
  The promoted closure/safe-primitive rule directly shaped the architecture
  section; no further general criterion was needed after revision.

## 08. Scalar and Vector Geometric Graph Networks

- **Post:** `_posts/2026-08-08-scalar-vector-geometric-gnns.md`
- **Topic-matched reference:** The spherical-equivariant permanent anchor is
  the direct mature comparator. The revised symmetry chapter supplies only the
  transformation-law prerequisites.
- **Before:** The chapter has the correct scalar-to-angle-to-vector storyline,
  a useful bent-versus-linear example, and concise EGNN/PaiNN equations. At
  roughly 2,300 substantive words, the four named designs still read mostly as
  separate descriptions. The text says complete distances can determine a
  geometry but does not derive reconstruction, and it does not quantify why a
  sparse radial layer cannot access the same angle locally. Vector channels are
  motivated but not carried through the same controlled neighborhood or cost
  comparison.
- **Criteria promoted:** Distinguish information-theoretic completeness from
  computational accessibility: prove what a representation determines in
  principle, then show the interactions or depth needed to make that information
  available to the stated architecture.
- **Planned revision:** Preserve every H2 and its order. Reconstruct a centered
  Gram matrix from complete distances, contrast that global result with a
  sparse one-layer radial neighborhood, carry one controlled geometry through
  SchNet-like radial, angle-aware scalar, EGNN, and PaiNN-style updates, derive
  central energy forces and transformation/centroid conditions, quantify
  triplet and vector-channel costs, and close with a decision table separating
  representation completeness, locality, output type, and conservation. Reuse
  the preceding symmetry proofs by cross-link rather than repeating them; add a
  figure only for a genuine gap.
- **After:** Expanded from 2,392 to 4,883 substantive body words while
  preserving all nine content H2s and their exact order. The chapter reconstructs
  centered coordinates from complete distances, works explicit Gram spectra and
  reflection ambiguity, and then carries the same 60-degree/180-degree
  neighborhood through radial, angle-aware, EGNN, and PaiNN-like updates. A
  two-layer radial calculation separates global completeness from local access;
  pair-force signs, zero force/torque, EGNN centroid conditions, an equivariant
  but nonconservative cross-Jacobian witness, and direct-vector conservation
  failure separate output law from physics. Added numerical cost comparisons and
  the Torgerson citation; reused all four figures. Arithmetic/Gram/vector/cost,
  LaTeX/scaffolding, 6/6/6 citation bijection and links, blog validation, diff,
  clean build, and rendered checks passed. The promoted completeness-versus-
  accessibility rule shaped the controlled comparison; no further general
  criterion was needed after revision.

## 09. Steerable Features and Tensor Products

- **Post:** `_posts/2026-08-08-steerable-features-tensor-products.md`
- **Topic-matched reference:** The spherical-equivariant permanent anchor
  overlaps directly and therefore sets both the quality bar and the boundary:
  this chapter should own low-order algebra and one typed layer, while the anchor
  owns broader Wigner conventions, implementation, and architecture families.
- **Before:** The chapter has a good type-system thesis and the correct
  $$1\otimes1=0\oplus1\oplus2$$ decomposition, but at roughly 2,200 substantive
  words it closely shadows the companion anchor's vocabulary. Irreducible linear
  maps, spherical-harmonic typing, Clebsch--Gordan projection, selection rules,
  nonlinearities, and the final layer are each stated once. The low-order
  Cartesian example is not numerically worked or reconstructed, and the cost
  section does not count representation dimensions or coupling paths.
- **Criteria promoted:** Overlapping companion posts must declare a division of
  labor and earn separate existence through a distinct derivation or running
  example; cross-link shared machinery rather than paraphrasing it.
- **Planned revision:** Preserve every H2 and its order. Make this the concrete
  low-order algebra chapter: derive type-preserving linear maps with channel
  multiplicity, evaluate low-degree harmonics on simple directions, reconstruct
  the full vector outer product from trace/cross/STF blocks, carry a numerical
  vector pair through rotations and parity, enumerate selection paths, and work
  one typed edge message through aggregation/readout. Quantify feature storage,
  allowed coupling paths, and channel-cost growth with $$L_{\max}$$. Keep Wigner
  basis conventions and modern architecture survey in the companion anchor;
  reuse existing figures unless a genuine gap appears.
- **After:** Expanded from 2,307 to 4,604 substantive body words while
  preserving all nine H2s and their order. The revision now derives the scoped
  Schur-lemma channel map, evaluates low-degree Cartesian harmonics, reconstructs
  a numerical outer product from its scalar/axial/quadrupole blocks, checks
  rotation and inversion, and carries two neighbors through a parity-safe typed
  update. The final section counts feature storage and allowed coupling paths and
  distinguishes angular degree from depth. The opening now assigns Wigner
  conventions and architecture coverage to the mature companion, so no new
  figures were needed. Arithmetic, citations, validator, clean build, and
  rendered checks passed. The division-of-labor criterion prevented duplication
  and no further general rule was required after revision.
