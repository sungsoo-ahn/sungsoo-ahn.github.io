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

## 10. Equivariant Transformers and Machine-Learned Potentials

- **Post:** `_posts/2026-08-08-equivariant-transformers-machine-learned-potentials.md`
- **Topic-matched reference:** The spherical-equivariant permanent anchor is the
  closest architectural reference and sets the standard for deriving a complete
  legal layer. The polished molecular-dynamics post is a fair application-side
  comparator: it consistently states the sampled object, downstream quantity,
  and operational failure boundary. The new chapter should bridge these two
  levels without duplicating either.
- **Before:** At 2,220 substantive words, the post states the right thesis—
  invariant routing, equivariant content, and a simulation contract—but the two
  halves remain adjacent summaries. The attention proof stops before a complete
  block and the architecture comparison changes several design choices at once.
  The potential half lists extensivity, conservative forces, force supervision,
  smooth cutoffs, range, and cost without carrying one controlled system through
  them. Equivariance, conservation, smoothness, locality, and rollout validity
  therefore read as related virtues rather than distinct guarantees and
  empirical conditions.
- **Criteria promoted:** Architecture-to-workflow tutorials must expose an
  interface contract: the object supplied, downstream assumptions, guarantees
  that transfer, and properties that remain contingent on data, numerical
  methods, or physical approximations.
- **Planned revision:** Preserve all seven H2s and their order. Complete the
  typed-attention closure argument through normalization, softmax, values,
  aggregation, residual/gating, and readout; extend the current rotation example
  to actual vector values and a declared output. Compare architecture families
  against one fixed geometric-computation contract rather than by names alone.
  On the potential side, derive force covariance and zero force/torque from a
  scalar energy, give an explicit nonconservative direct-force witness, quantify
  what energy and force labels constrain on a small potential, and derive cutoff
  regularity with a concrete envelope. Add a locality counterexample, a sparse-
  versus-global cost calculation, and a claim-matched simulation contract that
  cross-links the molecular-simulation chapter. Reuse the four existing figures
  unless a new quantitative figure closes a genuine explanatory gap.
- **After:** Expanded from 2,220 to 5,243 substantive body words while
  preserving all seven H2s and their order. The architectural half now proves a
  complete typed attention block, rotates actual values numerically, and holds
  the representation/cutoff budget fixed while separating routing, angular
  degree, correlation order, and edge-aligned computation. The potential half
  derives force covariance and balance laws, gives a three-atom $$SO(3)$$-
  equivariant but nonconservative witness, quantifies energy-versus-force
  supervision, proves $$C^2$$ cutoff regularity, and constructs an exact locality
  failure with cost arithmetic. A claim-matched table closes the interface from
  symmetry through rollout and observable validity. All four existing figures
  remained sufficient. Arithmetic, citations, validator, clean build, and
  rendered checks passed. The promoted interface-contract rule supplied the
  organizing spine; no additional general criterion was needed after revision.

## 11. ODEs, SDEs, and Probability Flow

- **Post:** `_posts/2026-08-08-odes-sdes-probability-flow.md`
- **Topic-matched reference:** The Fokker--Planck permanent anchor owns the
  physical intuition and two density-equation derivations. The polished path-
  measure post is the fair companion for the distinction between endpoint,
  marginal, and trajectory distributions. This chapter should own the bridge
  from a particle dynamics to probability flow and reverse-time sampling rather
  than repeat either reference.
- **Before:** At 1,869 substantive words, the post contains the correct main
  equations and responsibly states that the probability-flow ODE matches only
  one-time marginals. Most steps, however, are verified only symbolically. The
  constant-velocity example disappears before the SDE sections, the Gaussian
  example arrives late, and no calculation compares conditional kernels,
  quadratic variation, path laws, or reverse-clock moments. The two reverse-time
  sign conventions are stated but not derived from one canonical clock. The
  result is accurate but too compressed to make the factor of two, ensemble
  dependence, or exact scope of marginal equivalence operational.
- **Criteria promoted:** Any claim that stochastic processes match must identify
  the exact equality level and exhibit both a matching calculation and a witness
  for a stronger level that fails.
- **Planned revision:** Preserve all eight H2s and their order. Assign the full
  Fokker--Planck proof to the anchor and path-measure identities to the companion.
  Carry one one-dimensional Gaussian family through ODE pushforward and density
  change, Brownian variance growth, score/current, probability-flow scaling,
  reverse SDE, and reverse ODE. Derive reverse dynamics first in the increasing
  clock $$\tau=T-t$$, then translate once to decreasing $$t$$ and verify the
  factor of two with mean/variance ODEs. Contrast one-time marginals using
  conditional variance, quadratic variation, and joint/path laws; state the
  ensemble dependence and nonuniqueness of continuity velocities. Add numerical
  step-size and CNF likelihood calculations plus an explicit model-error/solver
  contract. Reuse the four existing figures unless a genuinely new equality-
  level figure is necessary.
- **After:** Expanded from 1,869 to 4,589 substantive body words while
  preserving all eight H2s and their order. One $$\mathcal N(2,1)$$ ensemble now
  connects the affine flow map, Jacobian and CNF density change, Brownian
  variance, score/current, probability-flow scaling, and both reverse samplers.
  Conditional variance, two-time covariance, and quadratic variation show
  exactly where marginal agreement stops. The reverse-clock signs and factor of
  two are derived and checked with moment equations; ensemble dependence,
  continuity-velocity nonuniqueness, score support, biased-score error, and Euler
  error set the approximation boundary. The four existing figures remained
  sufficient. Sign/arithmetic/citation audits, validation, clean build, and
  rendered checks passed. The promoted equality-level criterion determined the
  post's spine; no further general criterion was needed after revision.

## 12. Diffusion Models and Flow Matching

- **Post:** `_posts/2026-08-08-diffusion-models-flow-matching.md`
- **Topic-matched reference:** The Fokker--Planck anchor supplies the density
  mechanics and the now-polished probability-flow chapter supplies reverse-time
  and marginal/path distinctions. The older path-measure post remains the fair
  comparator for defining exactly which distribution an objective averages over.
  This chapter should own conditional regression and target conversion, not
  restate stochastic-calculus machinery.
- **Before:** At 2,379 substantive words, the post has the correct unifying
  thesis and both conditional-expectation identities. The MSE decomposition is
  only asserted, however, and the score, noise, data, and velocity
  parameterizations are called algebraically convertible without transforming
  their losses. Diffusion probability flow and conditional flow matching are
  placed side by side but never computed on the same non-Gaussian data mixture.
  Schedules and solvers are listed as interacting choices without a numerical
  target-conditioning or time-reparameterization example. The chapter therefore
  establishes vocabulary more strongly than operational equivalence.
- **Criteria promoted:** Claims of equivalent targets or objectives must name
  the equivalence level, derive the target and loss-weight conversion, and state
  which finite-model or numerical effects remain unequal.
- **Planned revision:** Preserve all nine H2s and their order. Carry a symmetric
  two-point data distribution through one affine Gaussian path. Derive its
  posterior mean, conditional and marginal score, conditional and marginal
  velocity, and compute both regressions at one numerical $$(x,t)$$. Prove the
  squared-error orthogonal decomposition rather than citing conditional means.
  Derive score/noise/data conversions and the loss weights required to preserve
  an objective. For the same $$(\alpha_t,\sigma_t)$$ path, derive the linear SDE
  coefficients and show algebraically that its probability-flow velocity equals
  the flow-matching velocity; qualify the conditions and endpoint singularities.
  Add a time-reparameterization/stiffness calculation and separate exact target,
  finite-capacity, time-sampling, solver, and terminal-prior errors. Reuse the
  four existing figures unless the worked mixture creates a real visual gap.
- **After:** Expanded from 2,379 to 4,566 substantive body words while
  preserving all nine H2s and their order. A symmetric binary mixture now yields
  closed-form posterior odds, score targets, flow targets, and a numerical point
  where posterior averaging and probability-flow conversion agree. The weighted
  MSE decomposition distinguishes exact conditional-versus-marginal loss
  equality from score/noise/data/VP-velocity target conversion; the required
  loss weights and endpoint failures are explicit. The same affine path is
  realized by a derived linear SDE, whose probability-flow field equals the
  flow-matching field pointwise. Time reparameterization, log-SNR arithmetic,
  and a five-layer error contract close the practical argument. Existing figures
  remained sufficient. Math/citation audits, validation, clean build, and render
  checks passed. The promoted objective-equivalence rule supplied the necessary
  distinctions; no further general criterion was needed after revision.

## 13. Geometric Flow Matching on Manifolds

- **Post:** `_posts/2026-08-08-geometric-flow-matching-manifolds.md`
- **Topic-matched reference:** The spherical-equivariant permanent anchor sets
  the standard for typed geometric objects, while the Fokker--Planck anchor sets
  the standard for densities and differential operators. The newly polished
  diffusion/flow-matching chapter supplies the conditional-regression logic.
  This chapter should own the primitive-by-primitive geometric replacement and
  its numerical consequences rather than repeat those references.
- **Before:** At 2,127 substantive words, the post identifies all of the right
  replacements—tangent vectors, metric, Exp/Log, geodesics, volume divergence,
  product spaces, and manifold-aware solvers—but usually states each once. The
  SO(3) and torsion examples verify endpoint interpolation without carrying one
  state through tangent projection, target evaluation, metric loss, density
  change, and a numerical step. Base-point mismatches, coordinate-volume error,
  metric-weight effects, cut-locus target jumps, and retraction error remain
  qualitative. The result is accurate but reads as a map of abstractions rather
  than a 30--45 minute derivation-led chapter.
- **Criteria promoted:** A non-Euclidean generalization must audit and retype
  every primitive used by the Euclidean method, including the loss, reference
  measure, symmetry action, and numerical update—not only its interpolation
  formula.
- **Planned revision:** Preserve all nine H2s and their order. Add a compact
  Euclidean-to-manifold primitive table, then carry a quarter-circle path on
  $$S^2$$ through explicit Log, Exp, tangent velocity, midpoint target, ambient
  projection, metric loss, and intrinsic versus projected Euler updates. Deepen
  the existing rotation and torsion calculations with an invalid matrix
  interpolation and a quantified branch jump. Work one sphere-coordinate
  divergence example to show the volume factor, and prove equivariant flow-map
  commutation from vector-field equivariance. Quantify product-metric units and
  clarify which choices alter paths, loss weighting, and endpoint coupling. End
  with a solver/base/cut-locus/representation contract and cross-link the
  Euclidean chapter. Reuse the four existing figures unless a new figure is
  needed for a genuine geometric step.
- **After:** Expanded from 2,127 to 4,652 substantive body words while
  preserving all nine H2s and their order. A primitive table now retypes the
  Euclidean construction, and one quarter-circle on $$S^2$$ carries Log, Exp,
  tangent projection, metric loss, and intrinsic versus ambient/retracted steps
  through explicit arithmetic. Matrix interpolation and circular branch examples
  quantify constraint and cut-locus failures. A sphere volume-form calculation,
  flow-map equivariance proof, product-metric/coupling reversal, and
  quaternion/chart/base/solver contract complete the system beyond interpolation.
  Existing figures remained sufficient. Geometry/arithmetic/citation audits,
  validation, clean build, and rendered checks passed. The promoted primitive-
  audit rule shaped the full revision; no further general criterion was needed.

## 14. Discrete Flow and Generator Matching

- **Post:** `_posts/2026-08-08-discrete-flow-generator-matching.md`
- **Topic-matched reference:** The Fokker--Planck anchor provides the continuous
  inflow/outflow and conservation standard. The older GFlowNet post is a fair
  discrete companion but owns flow conservation on directed construction DAGs;
  this chapter should distinguish physical-time CTMC rates, cycles, and event
  simulation. The polished conditional-regression chapters provide the target-
  marginalization comparison without needing repetition.
- **Before:** At 2,356 substantive words, the post correctly declares a
  destination-first column convention and includes a useful mask-to-token
  example. It does not instantiate the full generator or short-time transition
  matrix, check positivity and mass conservation in matrix form, or carry the
  same example through the observable generator, reverse rates, hazard-time
  sampling, and invalid-step boundary. The Bregman conditional-mean claim and
  factorized-event tradeoff remain mostly abstract. Generator superposition and
  sampler choice are described without a finite-state witness or numerical cost.
- **Criteria promoted:** Competing operator conventions must be verified with a
  minimal executable matrix whose sign, normalization, conservation, and finite-
  update invariants are all explicit.
- **Planned revision:** Preserve all eight H2s and their order. Carry the
  three-state mask example through $$Q_t$$, $$I+hQ_t$$, mass/positivity checks, an
  observable expectation, integrated hazard, exact jump-time sampling, squared-
  loss/Bregman marginalization, and reverse-clock rates. Quantify when a fixed
  step becomes invalid. Expand single-coordinate factorization with a sequence-
  size calculation and a coupled-edit counterexample; derive permutation
  equivariance for graph rates. Use a three-state circulation to show generator
  nonuniqueness at fixed marginals, then distinguish CTMC generator matching from
  GFlowNet trajectory balance. Quantify event-driven versus fixed-step network
  evaluations on the running schedule. Reuse the four existing figures unless a
  genuine missing mechanism justifies another.
- **After:** Expanded from 2,356 to 4,542 substantive body words while
  preserving all eight H2s and their order. The mask example now supplies an
  explicit column-convention generator, finite transition matrix, mass and
  positivity checks, observable duality, integrated hazard, inverse event-time
  sample, conditional-rate regression, and reverse-clock rates. Sequence-size
  arithmetic and a coupled-edit witness expose factorization; graph-rate
  equivariance, multimodal generator validity, and a stationary circulation
  separate marginal behavior from paths. Event, fixed-step, and tau-leap costs
  are quantified, with GFlowNet DAG balance assigned to its companion post.
  Existing figures remained sufficient. Matrix/arithmetic/citation audits,
  validation, clean build, and rendered checks passed. The promoted executable-
  convention rule caught the core risk; no further general criterion was needed.

## 15. Molecular Data and Property Prediction

- **Post:** `_posts/2026-08-08-molecular-data-property-prediction.md`
- **Topic-matched reference:** The electrocatalysis post is the stronger
  application companion because it carries one physical screening target from
  thermodynamics through a constrained search space and repeatedly quantifies
  why an apparently simple descriptor succeeds or fails. The quantum-chemistry
  post supplies the fidelity hierarchy between physical object, approximation,
  and surrogate. The permanent anchors remain the standard for worked
  derivations and visual pacing.
- **Before:** At 2,495 substantive words, the post has the right outside-in
  thesis and covers representations, conformers, splits, pretraining, metrics,
  and uncertainty. Most sections nevertheless stop after naming the tradeoff.
  The conditional-variance equation has no collision witness; the conformer
  average has no energies, populations, or changed ranking; and random versus
  scaffold or temporal splitting has no finite record family showing exactly
  which dependence crosses the boundary. Metric and calibration advice is also
  detached from a concrete screening decision. The result reads as a precise
  checklist rather than a 30--45 minute argument.
- **Criteria promoted:** A split/generalization discussion must define the
  independent unit, the equivalence relation that induces dependence, and the
  deployment population, then carry one related record family through competing
  partitions and identify the changed estimand and residual leakage.
- **Planned revision:** Preserve all eight H2s and their order. Use one small
  analogue series and one flexible molecule as running examples. Quantify a
  representation collision and Bayes-risk floor; compute a Boltzmann conformer
  average and show how lowest-conformer and uniform aggregation reverse a
  decision. Connect task regimes to label fidelity and inference-time
  availability. Build an explicit molecule/conformer/assay record table and
  carry it through row, molecule, scaffold, and temporal splits, stating the
  deployment claim each estimates. Deepen pretraining with positive-pair
  invariance and overlap arithmetic. Compute MAE/RMSE, tail enrichment, interval
  coverage/sharpness, and an uncertainty-driven acquisition or abstention
  decision on one prediction set. End with a claim contract linking object,
  representation, independent unit, split, metric, and action. Reuse the four
  existing figures unless a missing quantitative relationship genuinely needs
  a new one.
- **After:** Expanded from 2,495 to 4,773 substantive body words while
  preserving all eight H2s and their order. A conformer collision now gives an
  explicit Bayes-risk floor, and a three-state Boltzmann calculation shows how
  lowest-state, uniform, and population-weighted aggregation reverse a screening
  decision. The same six records are carried through row, molecule, scaffold,
  and temporal splits, with the independent unit, residual dependence, and
  deployment risk stated for each. Pretraining exposure is divided into exact,
  analogue, and distant regimes. A six-candidate prediction set connects
  MAE/RMSE, tail enrichment, interval coverage and sharpness, upper-confidence
  acquisition, and abstention to actual experimental actions; a final contract
  reunites the argument. Existing figures remained sufficient. The promoted
  split-semantics rule drove the central revision. Arithmetic/citation audits,
  validation, clean builds, and rendered checks passed.

## 16. Molecular Simulation with Machine-Learned Force Fields

- **Post:** `_posts/2026-08-08-molecular-simulation-machine-learned-force-fields.md`
- **Topic-matched reference:** The Fokker--Planck anchor derives how one
  stochastic update induces an ensemble-density evolution and repeatedly checks
  the limiting algebra. The quantum-chemistry post owns the electronic
  reference hierarchy and Born--Oppenheimer surface. The revised equivariant-
  potentials companion owns architectural symmetry, conservative-force
  construction, and cutoff parameterization; this chapter should own the
  closed-loop numerical and statistical consequences.
- **Before:** At 2,307 substantive words, the post correctly distinguishes a
  potential, integrator, ensemble, trajectory, and observable, and it names the
  major failure modes. It never executes Velocity Verlet on a solvable surface,
  tests its stability boundary, or carries a small learned curvature bias into
  equilibrium weights. Correlated trajectory frames, rollout divergence,
  cutoff regularity, free-energy estimation, diffusion estimation, and block
  uncertainty remain qualitative. The validation hierarchy is accurate but
  does not expose which error source each convergence test can eliminate.
- **Criteria promoted:** A feedback-driven numerical workflow must separate
  model/surface, discretization, mixing/transient, and estimator errors, then
  vary model parameters, step size, and trajectory length on one solvable system
  so static accuracy, stability, integration convergence, and observable
  correctness cannot be conflated.
- **Planned revision:** Preserve all seven H2s and their order. Carry a
  one-dimensional harmonic system through force, exact frequency, a complete
  Velocity-Verlet step, update-matrix stability, and time-step comparison; then
  perturb the learned spring constant to show how a small force bias changes
  the canonical variance despite stable dynamics. Connect energy/force loss and
  reference fidelity to the same surface. Quantify trajectory autocorrelation
  and effective sample size, then show what ensemble disagreement can and cannot
  detect in active learning. Contrast path divergence with distributional error
  and separate integrator failure from model failure. Derive cutoff smoothness
  obligations and retain a nonlocal-physics witness. Carry basin counts through
  free energy, block uncertainty, and a transport estimate, distinguishing
  equilibrium from kinetic claims. End with an error-budget/convergence matrix.
  Reuse the four existing figures unless a genuine missing quantitative
  relationship warrants another.
- **After:** Expanded from 2,307 to 4,568 substantive body words while
  preserving all seven H2s and their order. One harmonic coordinate now carries
  exact force/frequency, a full Velocity-Verlet step, update-matrix stability,
  phase error, and step-size diagnosis. Perturbing the learned curvature shows
  stable integration converging to the wrong frequency and canonical variance,
  while the associated energy/force losses expose what supervision constrains.
  Autocorrelation and ensemble examples quantify effective sample size and
  shared-bias blindness. Path divergence, cutoff differentiability, and a
  nonlocal-information collision sharpen rollout and locality claims. Basin
  counts, block errors, and a diffusion estimate separate equilibrium from
  kinetic evidence. A final matrix assigns reference, surface, discretization,
  mixing, estimator, and finite-size errors to matched convergence tests. The
  four existing figures remained sufficient. The promoted feedback error-budget
  rule shaped the chapter. Arithmetic/citation audits, validation, clean build,
  and rendered checks passed.

## 17. Generating Molecular Graphs and Chemical Reactions

- **Post:** `_posts/2026-08-08-molecular-generation-graphs-reactions.md`
- **Topic-matched reference:** The older protein-design post earns its length by
  following a generated object through a quantitative candidate funnel and
  experimental endpoint. The spherical-equivariant anchor supplies the standard
  for quotienting irrelevant symmetries, while the Fokker--Planck anchor supplies
  the standard for marginalizing alternative histories. The discrete-generator
  companion owns CTMC rate mechanics; this chapter should own chemical validity,
  sparse reaction edits, route search, and the distribution induced by filters.
- **Before:** At 2,731 substantive words, the chapter has a strong unifying
  thesis and covers autoregressive, parallel, conditional, reaction, mapping,
  planning, and evaluation viewpoints. It states the graph-likelihood sum but
  never computes how two construction histories change a molecule probability.
  Constraint masks, valence repair, canonicalization, and route filters are
  described without showing the accepted distribution they induce. The
  bromoethane edit is not carried through atom-indexed adjacency or alternative
  mappings. Route branching, compounded yield, and the final evaluation funnel
  have no budget arithmetic, so the post remains closer to a careful survey than
  a 30--45 minute structured-generation argument.
- **Criteria promoted:** Whenever masks, repair, rejection, canonicalization, or
  downstream filters impose validity, distinguish the raw proposal distribution,
  the transformed/accepted distribution, and the target chemical distribution;
  carry a finite sample through every denominator and expose collisions or
  unreachable modes.
- **Planned revision:** Preserve all nine H2s and their order. Carry one small
  molecular candidate and the existing bromoethane substitution through the
  chapter. Quantify permutation/serialization equivalence and compute a graph
  likelihood by summing construction histories. Use a finite categorical sample
  to compare action masking with parallel valence repair/rejection and show how
  the accepted distribution changes. Turn conditional guidance into a numerical
  prior/proxy tradeoff and separate exact constraints from uncertain oracles.
  Write the substitution as explicit atom-indexed bond edits, carry equivalent
  atom maps through quotient-aware scoring, and show how product-informed maps
  leak the answer. Quantify branching/search budget and route-level yield. Pass a
  generated batch through sanitization, canonical uniqueness, novelty, route,
  synthesis, and measurement with all denominators, then close the generation--
  reaction--planning loop. Reuse the four existing figures unless a genuine
  missing relationship warrants another.
- **After:** Expanded from 2,731 to 5,016 substantive body words while
  preserving all nine H2s and their order. A 2-fluoroethanol example now
  quantifies tensor representatives and alternative construction histories. A
  finite five-outcome decoder distinguishes rejection, deterministic repair,
  and action-mask support, including their induced probability shifts and
  unreachable modes. Numerical guidance exposes prior/proxy/uncertainty tension.
  The bromoethane substitution is written as indexed bond and charge edits;
  equivalent atom maps are marginalized and product-informed mapping leakage is
  made explicit. Route branching, search budget, and compounded yields connect
  one-step prediction to planning. Finally, 1,000 raw graphs pass through every
  canonicalization, novelty, route, synthesis, and measurement denominator to
  five hits, making the selected distribution and feedback loop explicit. The
  existing figures remained sufficient. The promoted constraint-transformation
  rule shaped the central examples. Arithmetic/citation audits, validation,
  clean builds, and rendered checks passed.

## 18. Three-Dimensional Molecular Generation and Optimization

- **Post:** `_posts/2026-08-08-molecular-generation-3d-optimization.md`
- **Topic-matched reference:** The geometric-flow companion owns manifold
  transport primitives, and the property-prediction chapter owns representation,
  split, and uncertainty calibration. The older protein-design post supplies a
  fair generate--filter--experiment comparison with fixed downstream capacity.
  The just-revised graph/reaction chapter owns representation-level validity and
  route-funnel denominators. This chapter should own the distinction between
  conformer probability and molecule search, plus selection-induced oracle bias.
- **Before:** At 2,480 substantive words, the post correctly separates
  $$p(R\mid G)$$ from $$p(G,R)$$, Cartesian from torsional state spaces, prior
  sampling from goal-directed search, and validity from experiment. The
  conformer section has no populated ensemble or observable calculation. Rigid-
  motion and torsion constraints are stated without a numerical state. Guidance,
  GFlowNet temperature, Pareto tradeoffs, oracle budget, and exploitation remain
  qualitative. In particular, the equation $$\widehat R=R+\epsilon$$ never shows
  how maximizing over a pool preferentially selects positive error or how
  independent rescoring changes the winner.
- **Criteria promoted:** An optimization tutorial using a noisy learned oracle
  must quantify selection-induced optimism by carrying one finite candidate pool
  through scoring, adaptive selection, and independent evaluation under an
  explicit query budget; random-test accuracy does not validate the selected
  tail.
- **Planned revision:** Preserve all nine H2s and their order. Use n-butane as a
  fixed-graph conformer example, compute anti/gauche Boltzmann populations and a
  population-dependent observable, and contrast this with changing molecular
  identity. Give centering/equivariant-score checks without duplicating the
  manifold chapter. Quantify the $$179^\circ/-179^\circ$$ torsion wrap and what
  Cartesian/internal coordinates hold fixed. Use a finite guided prior and a
  GFlowNet-temperature calculation to expose diversity/selectivity. Carry a
  small candidate table and a larger noisy-pool estimate through oracle
  maximization, independent rescoring, uncertainty penalties, and best-so-far
  query curves. Work a Pareto table whose scalar optimum flips with weights and
  include conformer-aware scoring. Cross-link the reaction chapter for route
  mechanics, then close with a prospective batch and feedback contract. Reuse
  the four existing figures unless a genuine missing relationship warrants
  another.
- **After:** Expanded from 2,480 to 4,774 substantive body words while
  preserving all nine H2s and their order. An n-butane ensemble now quantifies
  basin degeneracy, Boltzmann populations, and observable aggregation. Numerical
  centering, score equivariance, and torsion-wrap checks make the geometric
  state spaces operational. Finite prior tilting and GFlowNet temperatures show
  selectivity/diversity changes. A Pareto table makes scalar-policy dependence
  and conformer aggregation alter the winner. Most importantly, a Gaussian
  extreme-value estimate and five-candidate ledger carry noisy-oracle selection
  through independent rescoring, uncertainty penalties, an explicit query
  budget, and best-so-far outcomes. The route companion is cross-linked and a
  prospective-batch contract closes the feedback loop. Existing figures remained
  sufficient. The promoted selection-optimism rule shaped the central revision.
  Independent review caught and corrected one Pareto arithmetic typo before
  commit. Arithmetic/citation audits, validation, clean build, and rendered
  checks passed.
