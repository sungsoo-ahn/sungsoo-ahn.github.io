# Tutorial Comparison and Revision Log

## Source-Faithful Deck Migration (supersedes the synthetic-storyline pass)

The 2026-08-09 migration treats each 2025 lecture deck as the content authority.
The polished house posts remain references for prose, caption, equation, and
rendering quality only. The earlier entries below are retained as historical
records; their synthetic running examples and preserved H2 storylines are not
requirements for the replacement deck-level articles.

### ML4Mol 2025 Lecture 2: Chemistry and Physics

- **Source:** `lec2_prelim1.pdf` (57 pages), with media extracted from the
  corresponding PowerPoint.
- **Coverage:** 49 substantive slides covered; one disclaimer and seven title or
  part-transition slides omitted. The six scientific parts remain in source
  order: representations and symmetry, quantum mechanics, electronic
  properties, forces, statistical mechanics, and chemical change.
- **Figures:** 24 of 29 unique substantive slide visuals reused (82.76%). The
  five omitted visuals repeat information already present in a reused slide
  figure. No new explanatory figure was drawn.
- **Synthetic material removed:** the representation-error calculation,
  symmetry-derived force example, Born--Oppenheimer timescale calculation,
  Morse-potential case, basin-population calculation, and barrier-sensitivity
  example from the previous post were absent from the lecture source.
- **General lesson promoted:** source-faithful mode must inventory slides before
  outlining, derive section order from the deck, treat word count as irrelevant,
  and use house references only as an editorial quality check. The blog-writing
  and figure skills now encode this rule.

### ML4Mol 2025 Lecture 3: Drug and Material Discovery

- **Source:** `lec3_prelim2.pdf` (70 pages), with media extracted from the
  corresponding PowerPoint.
- **Coverage:** 61 substantive slides covered; one notice and eight title or
  part-transition slides omitted. The article retains the deck's progression
  from biological organization through PK/PD, the drug-discovery funnel,
  experimental techniques, material properties, synthesis, characterization,
  and applications.
- **Figures:** 50 of 61 unique substantive slide visuals reused (81.97%); the
  remaining eleven repeat information already visible in reused figures. No new
  explanatory figure was drawn.
- **Synthetic material removed:** the hypothetical KX/A2 program, occupancy and
  residence-time calculations, SAR table, Bateman PK calculation, therapeutic
  window calculation, clinical Bayesian example, AB3 convex-hull scenario,
  defect-population derivation, and closed-loop acquisition argument were not in
  the source deck.
- **General lesson:** when one deck deliberately juxtaposes domains, preserve
  the juxtaposition. Splitting the lecture into separately optimized domain
  essays can erase the comparison that motivated the original sequence.

### ML4Mol 2025 Lecture 4: Graph Neural Networks

- **Source:** `lec4_prelim3.pdf` (86 pages), with media extracted from the
  corresponding PowerPoint.
- **Coverage:** 76 substantive slides covered; five cover or part-transition
  slides omitted. The article retains all three source sections: message
  passing, graph discrimination, and graph Transformers; the chronological
  2013--2025 GNN timeline; and the closing critique of applications, graph
  construction, benchmarking, and graph foundation-model claims.
- **Figures:** 73 of 88 unique substantive slide visuals reused (82.95%). The
  fifteen omitted visuals are paper title cards, repeated equations, or visual
  fragments whose scientific content is already represented by a reused figure
  from the same slide. No explanatory figure was drawn.
- **Synthetic material removed:** the fixed five-node architecture comparison,
  numerical GCN/GraphSAGE/GAT updates, two-hop coefficient calculation,
  sparse-versus-dense cost example, formal 1-WL ceiling proof, `C6` versus
  `2C3` collision, remedy cost table, oversmoothing diffusion derivation,
  binary-tree sensitivity example, and rewiring calculations from the previous
  three-post treatment were absent from this deck.
- **General lesson promoted:** source fidelity includes the lecture's rhetorical
  destination. A deck that moves from a technical survey into a methodological
  critique should not be adapted into an architecture tutorial that ends before
  the critique. The blog-writing skill now records this rule.

### ML4Mol 2025 Lecture 5: Geometric Graph Neural Networks

- **Source:** `lec5_prelim4.pdf` (64 pages), with media extracted from the
  corresponding PowerPoint and six source regions recovered from the canonical
  PDF for slide-native diagrams and tables.
- **Coverage:** 58 substantive slides covered; six cover or part-transition
  slides omitted. The original four-part progression remains intact: geometric
  graphs and group language, scalar invariant networks, scalar--vector
  networks, and spherical-tensor networks culminating in TFN and NequIP.
- **Figures:** 47 of 56 unique substantive visuals reused (83.93%). Nine omitted
  visuals duplicate equations or decompositions already represented by a
  reused visual from the same sequence. No explanatory figure was drawn.
- **Synthetic material removed:** the prior posts' orbit and stabilizer running
  example, representation-completeness derivation, numerical EGNN coordinate
  update, nonconservative-force counterexample, fixed-budget architecture
  comparison, and typed-attention calculation were absent from the source deck.
- **General lesson promoted:** legacy PowerPoint formats and slide-native shapes
  are extraction cases, not reasons to omit or redraw scientific visuals. The
  figure skill now requires a reproducible high-resolution canonical-PDF crop
  when direct media extraction fails.

### ML4Mol 2025 Lecture 6: Continuous, Geometric, and Discrete Generative Models

- **Source:** `lec6_prelim5.pdf` (63 pages), with media extracted from the
  corresponding PowerPoint.
- **Coverage:** 56 substantive slides covered; seven cover or part-transition
  slides omitted. The article keeps the deck's dependency chain from ODEs,
  SDEs, Fokker--Planck, and score matching through Euclidean conditional flow
  matching, Riemannian flow matching, and CTMC-based discrete flow matching.
- **Figures:** 49 of 58 unique substantive visuals reused (84.48%). Nine omitted
  visuals are duplicate process strips or equation fragments already shown by a
  reused visual from the same derivation. No explanatory figure was drawn.
- **Synthetic material removed:** the prior posts' constant-velocity transport,
  probability-flow ODE, exact Gaussian variance example, schedule/solver
  comparison, sphere midpoint calculation, SO(3) determinant counterexample,
  torus branch-jump calculation, CTMC mask example, circulation witness, and
  simulation-cost arithmetic were absent from the deck.
- **General lesson promoted:** a repeated derivation can be the source argument
  rather than expendable recap. The deck deliberately carries conditional
  paths, posterior marginalization, and conditional regression from Euclidean
  velocities to Riemannian tangent fields and discrete CTMC rates. The
  blog-writing skill now protects such structural analogies from over-editing.

### ML4Mol 2025 Lecture 7: Molecular Property Prediction and Simulation

- **Source:** `lec7_mpp.pdf` (60 pages), with media extracted from the
  corresponding PowerPoint and three wide model diagrams cropped from the
  canonical PDF where the slide inventory omitted grouped media.
- **Coverage:** 54 substantive slides covered; six cover or part-transition
  slides omitted. The article keeps the deck's five-part progression from 1D
  and 2D benchmarks through 3D datasets, molecular representation learning,
  structure generation, and machine-learning force fields.
- **Figures:** 85 of 100 unique substantive visuals reused (85%). The fifteen
  omitted visuals are duplicate tables, paper title cards, or secondary panels
  whose scientific content is already shown by a retained visual from the same
  slide sequence. No explanatory figure was drawn.
- **Synthetic material removed:** the previous property-prediction chapter's
  representation Bayes-risk example, conformer-population calculation,
  related-record split family, pretraining-overlap arithmetic, six-candidate
  calibration example, and claim-contract table were absent from this deck.
  The replacement instead retains the lecture's named benchmark lineage,
  2D-pretraining and LLM sequence, conformer/docking/crystal generators, four
  architecture families, and MACE--MatterSim--Orb--UMA progression.
- **General lesson:** a deck can use a dataset survey to establish the changing
  input and output contracts of later methods. Do not compress such a survey
  into a benchmark preamble: the sequence from scalar labels to forces,
  spectra, periodic structures, and trajectories is part of the scientific
  argument.

### ML4Mol 2025 Lecture 8: Molecular Simulation

- **Source:** `lec8_molsim.pdf` (16 pages), with media extracted from the
  corresponding PowerPoint and two slide-native diagrams cropped from the
  canonical PDF.
- **Coverage:** 15 substantive slides covered; the repeated lecture-series
  cover omitted. The source sequence remains intact: Langevin dynamics,
  thermodynamic ensembles, the MD workflow, force-field choices, three classes
  of observable, spatial and temporal acceleration, classical coarse-graining,
  enhanced sampling, learned coarse-graining, MDGen, accelerated lithium
  transport, and neural biased dynamics.
- **Figures:** 29 of 31 unique substantive visuals reused (93.55%). Two omitted
  panels repeat the time-coarse transition comparison shown by a retained
  visual. Three paper title cards and one isolated text fragment were treated
  as non-substantive. No explanatory figure was drawn.
- **Synthetic material removed:** the previous simulation chapter's harmonic
  force and frequency example, explicit Verlet step, stability matrix,
  learned-curvature bias, autocorrelation and effective-sample-size arithmetic,
  ensemble-disagreement counterexample, cutoff derivation, and final error
  budget were absent from the lecture source.
- **General lesson:** when a deck classifies methods by what they coarse-grain
  or modify, preserve that axis. Spatial reduction, long-lag transition
  generation, altered exploration, and learned bias forces accelerate different
  objects and should not be flattened into a generic list of efficiency methods.

### ML4Mol 2025 Lecture 9: Molecular Generation

- **Source:** `lec9_molgen.pdf` (38 pages), with media extracted from the
  corresponding PowerPoint and two composite diagrams cropped from the
  canonical PDF.
- **Coverage:** 37 substantive slides covered; the lecture cover omitted. The
  article preserves the source sequence from autoregressive graph models and
  GraphRNN through graph grammars, motifs, SMILES, continuous and discrete graph
  diffusion, reaction prediction, organic and inorganic retrosynthesis,
  multistep planning, electron flow, and chemistry language models.
- **Figures:** 56 of 65 unique substantive visuals reused (86.15%). Nine omitted
  panels are duplicate examples or secondary panels already represented by a
  retained visual from the same method. Paper title cards and decomposed icon
  fragments were treated as non-substantive. No explanatory figure was drawn.
- **Synthetic material removed:** the previous generation chapter's fixed
  five-outcome validity calculation, two-history likelihood comparison,
  guidance ratios, atom-mapping quotient, route-tree arithmetic, compounded
  yield calculation, and 1,000-candidate funnel were absent from the source.
- **General lesson:** representation grammar is not a preliminary encoding
  detail when it determines the model's actions. The adjacency, motif, string,
  diffusion, reaction-edit, and electron-flow representations must remain in
  the source order because each changes the construction or translation problem
  inherited by the next method.

### ML4Mol 2025 Lecture 10: Three-Dimensional Molecular Generation and Optimization

- **Source:** `lec10_molopt.pdf` (55 pages), with media extracted from the
  corresponding PowerPoint and twenty composite slide regions cropped from the
  canonical PDF. Sixteen of those crops preserve the equation-heavy GFlowNet
  derivation as coherent visuals rather than publishing its many disconnected
  PowerPoint text and equation fragments.
- **Coverage:** 51 substantive slides covered; the cover and three part-title
  slides omitted. The source order remains joint 3D generation, crystal and
  symmetry-aware generation, structure-conditioned design, molecular
  optimization, reaction-based construction, and the full GFlowNet derivation
  from reward-proportional sampling through TB, DB, training, and evaluation.
- **Figures:** 80 of 94 unique substantive visuals reused (85.11%). Fourteen
  omitted visuals are duplicate result panels or secondary examples already
  represented by a retained visual. The numerous isolated numerals, equation
  tokens, icons, and paper title fragments in the PowerPoint package were
  treated as pieces of their cropped slide composite rather than independent
  figures. No explanatory figure was drawn.
- **Synthetic material removed:** the prior 3D-generation chapter's torus and
  centered-coordinate derivations, latent-oracle and Pareto examples,
  optimization funnel, oracle-exploitation discussion, and synthesis-budget
  calculations were absent from this lecture. The GFlowNet material now follows
  the lecture's own reward-4/2/1 flow example and forward/backward derivation.
- **General lesson promoted:** PowerPoint media inventory is not identical to
  conceptual figure inventory. When one visual is assembled from dozens of
  equation or text fragments, preserve it as one high-resolution canonical-PDF
  composite; publishing the fragments separately destroys the source argument.

### ML4Mol 2025 Lecture 11: Quantum Chemistry

- **Source:** `lec11_quantum.pdf` (41 pages), with media extracted from the
  corresponding PowerPoint and four equation-heavy diagrams cropped from the
  canonical PDF because their PowerPoint representation was split into small
  text, equation, orbital, and arrow fragments.
- **Coverage:** 36 substantive slides covered; the cover and four part-title
  slides omitted. The source sequence remains the quantum many-body problem,
  Hartree--Fock and self-consistency, antisymmetry and neural wavefunctions,
  variational Monte Carlo, Kohn--Sham DFT, the Skala learned XC functional, and
  supervised prediction of Hamiltonians and electron densities.
- **Figures:** 36 of 43 unique substantive visuals reused (83.72%). Seven
  omitted visuals are paper title cards, decorative media, or secondary panels
  whose scientific content appears in a retained visual. Forty-three unused
  extraction candidates were removed after the composite crops replaced their
  fragmented representations. No explanatory figure was drawn.
- **Synthetic material removed:** the previous quantum-chemistry chapter's
  two-level VMC calculation, scalar SCF contraction example, near-degenerate
  Hamiltonian witness, delta-learning comparison, fidelity-ranking case, and
  amortized break-even analysis were absent from this deck. The replacement
  instead follows the lecture's own chain from the Schrödinger equation through
  WFT and DFT to learned wavefunctions, functionals, operators, and densities.
- **General lesson promoted:** related quantum objects are not interchangeable
  prediction targets. Preserve the source's hierarchy from state to functional,
  operator, field, and basis coefficients, and state which numerical solve or
  query remains downstream of each learned object. The blog-writing skill now
  records this target-and-solver contract.

### ML4Mol 2025 Lecture 12: Protein Structure Prediction

- **Source:** `lec12_protein1.pdf` (80 pages), with 95 published media assets
  extracted directly from the corresponding PowerPoint.
- **Coverage:** 77 substantive slides covered; the cover, attribution notice,
  and one part-title slide omitted. The source progression remains the protein
  sequence--structure--dynamics--design task map, AlphaFold1 learned potentials,
  AlphaFold2 Evoformer and structure module, AlphaFold3 all-atom diffusion,
  open implementations, alternative folding and packing models, affinity
  prediction, protein generation, ensemble prediction, and open data questions.
- **Figures:** 95 of 100 unique substantive visuals reused (95%). The five
  omitted visuals are duplicated overview fragments, logos, or title cards.
  Ten unused extraction candidates were removed. No explanatory figure was
  drawn and no PDF crop was needed because the deck's architecture diagrams
  were preserved as coherent embedded media.
- **Synthetic material removed:** the prior AlphaFold chapter's binary
  common-cause MSA calculation, realizability counterexample, memory arithmetic,
  rigid-transform numerical example, confidence matrix, sample-frequency
  warning, binding-free-energy calculation, and claim-scope table were absent
  from this deck. The replacement retains the much broader source survey from
  protein representations through AlphaFold and into affinity and ensemble
  models.
- **General lesson promoted:** an opening task map is substantive when the later
  storyline crosses its boundaries. Retaining sequence learning, folding,
  ensemble prediction, sequence design, and structure design makes it possible
  to see AlphaFold3, Proteina, Boltz-2, and BioEmu as changes of target rather
  than a flat list of structure architectures. The blog-writing skill now
  records this rule.

### ML4Mol 2025 Lecture 13: Protein Representation Learning and Design

- **Source:** `lec13_protein2.pdf` (62 pages), with 70 published media assets
  extracted from the corresponding PowerPoint and four slide-native diagrams
  cropped from the canonical PDF.
- **Coverage:** 59 substantive slides covered; the cover and attribution notice
  omitted, and one antibody-anatomy slide collapsed as a literal recap. The
  source order remains sequence and structure representation learning,
  multimodal and fitness pretraining, antibody affinity maturation, sequence
  design and inverse folding, backbone generation, joint co-design, de novo
  antibody design, complete design pipelines, and affinity-based experimental
  filtering.
- **Figures:** 74 of 78 unique substantive visuals reused (94.87%). Four omitted
  visuals are duplicate token examples, title media, or decorative animation.
  Three unused extraction candidates were removed. No explanatory figure was
  drawn; canonical-PDF crops recover DiffPreT and the three DSMBind slides whose
  scientific diagrams were built from native slide shapes.
- **Synthetic material removed:** the prior protein-design chapter's abstract
  variable-choice taxonomy, motif-scaffolding derivation, oracle-failure
  argument, evaluation funnel, and selection-bias discussion were absent from
  this deck. The replacement retains the source's extensive method chronology
  and its reported Chai-2, BoltzGen, antibody maturation, and DSMBind laboratory
  validations.
- **General lesson promoted:** a design lecture can distribute its scientific
  claim across a pipeline. Preserve generation, inverse folding, folding,
  confidence or affinity ranking, filtering, synthesis, and wet-lab measurement
  as separate stages with explicit handoffs. The blog-writing skill now records
  this pipeline-fidelity rule.

### ML4Mol 2025 Lecture 14: Genomics and Virtual Cells

- **Source:** `lec14_genome_cell.pdf` (57 pages), with 86 published media assets
  extracted from the corresponding PowerPoint and four AlphaGenome/TxPert
  composites cropped from the canonical PDF.
- **Coverage:** 54 substantive slides covered; the lecture cover and two part
  titles omitted. The source order remains genome and expression biology,
  sequencing and functional annotation, genomic foundation models and design,
  AlphaGenome and RNA structure, virtual-cell desiderata, cell-level measurement
  modalities, cell representation learning, perturbation prediction, and
  language-model reasoning over expression-derived cell sentences.
- **Figures:** 90 of 100 unique substantive visuals reused (90%). Ten omitted
  visuals are duplicate panels or secondary examples whose content is already
  visible in retained slide media. Fourteen unused extraction candidates were
  removed. No explanatory figure was drawn.
- **Synthetic material removed:** the previous virtual-cell chapter's negative
  binomial derivation, causal-identification discussion, donor/batch split
  protocol, perturbation-distribution example, calibration calculation, and
  claim-matched evaluation contract were absent from this deck. The replacement
  follows the source's long genomic-model chronology and its explicit transition
  from genomic sequence to measured cellular perturbation response.
- **General lesson promoted:** measurement slides are scientific content when
  they define what a broad modeling claim can observe. Bulk RNA-seq, single-cell
  RNA-seq, spatial transcriptomics, and fluorescence microscopy expose different
  cell-state projections; retaining them keeps "virtual cell" tied to its actual
  inputs and targets. The blog-writing skill now records this observation-layer
  rule.

### GDL 2025 Lecture 1: An Introduction to Geometric Deep Learning

- **Source:** `lec1_intro_v2.pdf` (33 pages in the canonical manifest), with 31
  published media assets extracted from the corresponding PowerPoint and two
  native diagrams cropped from the canonical PDF.
- **Coverage:** 23 substantive slides covered; nine course-administration slides
  omitted and one AlphaFold motivation slide collapsed as a literal recap. The
  scientific spine remains function approximation, deep-network expressivity
  and overfitting, architecture as data structure, geometry as invariants under
  symmetry, geometric neural and generative models, molecular data domains, and
  current drug, protein, simulation, and materials applications.
- **Figures:** 33 of 39 unique substantive visuals reused (84.62%). Six omitted
  visuals are duplicated application panels or secondary architecture examples.
  Eleven unused extraction candidates were removed. No explanatory figure was
  drawn.
- **Synthetic material removed:** no prior deck-level article existed for this
  course introduction. The new post excludes the staff, schedule, prerequisites,
  grading, blog-project, and peer-review logistics while retaining every
  scientific orientation slide.
- **General lesson promoted:** course-introduction decks require a slide-level
  boundary between administration and science. Logistics should disappear from
  a reader-facing article, but the motivating examples, domain inventory, field
  definition, and architecture-design question often form the deck's scientific
  thesis. The blog-writing skill now records this distinction.

### GDL 2025 Lecture 2: Symmetry and Equivariance

- **Source:** `lec2_prelim1.pdf` (51 pages), with six overview and timeline
  figures extracted from the corresponding PowerPoint and 32 source-native
  diagram or equation composites cropped from the canonical PDF.
- **Coverage:** 40 substantive slides covered; eight course-administration
  slides and three part-title transitions omitted. The source order remains the
  geometric-deep-learning recipe, graph and geometric-graph symmetries,
  algebraic and topological structure, matrix and transformation groups, group
  actions, stabilizers, cosets and quotients, representations, harmonics,
  steerable features, and equivariant operators.
- **Figures:** 38 of 45 unique substantive visuals reused (84.44%). Seven
  omitted visuals are duplicate timeline or secondary feature examples whose
  content is already represented in retained figures. The 109 unused extraction
  fragments removed from the candidate directory were constituents of the
  retained source composites, not additional conceptual figures. No explanatory
  figure was drawn.
- **Synthetic material removed:** the previous symmetry tutorial's typed
  V-shaped running object, signed-volume chirality witness, layer-closure proof,
  force and conservation derivations, and numerical augmentation examples were
  absent from this deck. The replacement follows the source's introductory
  group-theory construction and ends at the definition of equivariance.
- **General lesson promoted:** a formal lecture can make its argument through a
  dependency chain rather than a long derivation. Group, action, stabilizer,
  coset, quotient, representation, and equivariance each supply the object used
  next, so none should be compressed into glossary background. The blog-writing
  skill now records this dependency-chain rule.

### GDL 2025 Lecture 3: Graph Neural Networks

- **Source:** `lec3_gnn1.pdf` (63 pages), represented by 55 canonical PDF
  composites that preserve the equations, diagrams, paper panels, and benchmark
  tables assembled on each substantive slide.
- **Coverage:** 55 substantive slides covered; the cover, overview, five repeated
  agenda slides, and one study-resources slide omitted. The article retains the
  source order: graph targets, message passing and GCN, graph attention, graph
  Transformers and positional encoding, the 2013--2023 architecture survey, and
  dataset regimes from TU collections to knowledge-graph reasoning.
- **Figures:** all 55 source-native substantive visual composites reused (100%).
  Full-slide crops were necessary because equations, annotations, paper
  screenshots, and diagrams form one layout in the source. No extracted fragment
  was substituted and no explanatory figure was drawn.
- **Synthetic material removed:** the previous message-passing tutorial's fixed
  five-node numerical example, layer equivariance proof, aggregation collision
  table, structural-attention calculation, and sparse-versus-dense complexity
  example were absent from this deck. The replacement restores the lecture's
  long architecture and benchmark chronology, including its judgments about
  graph pooling, fragile small-data comparisons, universal-best claims, and the
  changing importance of data scale.
- **General lesson promoted:** a chronological survey can carry an argument by
  showing which limitation each method changes. Preserving only a modern MPNN
  derivation would erase the deck's progression through locality, pooling,
  expressivity, depth, long-range propagation, fair comparison, and scaling.
  The blog-writing skill now records this chronological-survey rule.

### GDL 2025 Lecture 4: Two Routes to Graph Convolution

- **Source:** `lec4_gnn2.pdf` (43 pages), represented by 36 canonical PDF
  composites preserving the equations, signal diagrams, operator bases, and
  annotations assembled on each scientific slide.
- **Coverage:** 36 substantive slides covered; five cover, overview, and outline
  transitions plus two reference-list slides omitted. The source order remains
  MLP versus convolutional linear maps, graph signals and the Laplacian, graph
  Fourier bases and transforms, spectral filtering, Chebyshev and GCN
  approximations, the varying-graph limitation, shift and permutation
  equivariance, higher-order fixed-point equations, Bell-number bases, and the
  comparison with adjacency-conditioned GNNs.
- **Figures:** all 36 source-native substantive visual composites reused (100%).
  Full-slide crops preserve the relationship between equations and diagrams in
  the deck; no explanatory figure was drawn.
- **Synthetic material removed:** the previous graph-convolution tutorial's
  controlled graph example, coefficient ledger, four-index equality-pattern
  correction, basis/sign/scale failure analysis, and cost comparison were absent
  from this deck. The replacement keeps the source's spectral derivation and its
  explicit critique before beginning the independent equivariant-operator route.
- **General lesson promoted:** two derivations of the same operator are not
  redundant when their assumptions differ. The signal-processing route fixes a
  graph Laplacian, while the commutant route fixes a transformation group; the
  lecture's varying-graph critique is the reason for the pivot. The blog-writing
  skill now records this alternate-derivation rule.

### GDL 2025 Lecture 5: Expressive Power of Graph Neural Networks

- **Source:** `lec5_gnn3.pdf` (69 pages), represented by 66 canonical PDF
  composites preserving the graph pairs, refinement steps, theorem statements,
  architecture diagrams, and comparison landscapes on every scientific slide.
- **Coverage:** all 66 substantive slides covered; only the cover, course
  overview, and external reading-list slide omitted. The source order remains
  graph discrimination and approximation, 1-WL and GIN, the k-WL hierarchy,
  sparse local and tuple-restricted variants, IGNs, motif encodings and GSNs,
  cellular lifts and CWNs, subgraph selection policies, DS-GNN and GNN-AK,
  coupled bag symmetries, k-OSAN, and the final proposal to learn rather than
  impose equivariance.
- **Figures:** all 66 source-native substantive visual composites reused (100%).
  The two recap slides were retained because they compare representation
  families and establish the next section's question. No explanatory figure was
  drawn.
- **Synthetic material removed:** the previous expressivity tutorial's fixed P5
  refinement, C6-versus-two-triangles collision, explicit spectra and deletion
  decks, bounded injective encoding proof, and remedy cost ledger were absent
  from this deck. The replacement restores the lecture's much broader sequence
  of formal test--architecture pairs and substructure or subgraph constructions.
- **General lesson promoted:** a slide titled “recap” can be the argumentative
  hinge of a lecture. Here the first recap compares higher-order tensors with
  stochastic identifiers, and the second locates subgraph bags among structural
  encodings and lifts. The blog-writing skill now distinguishes such synthesis
  slides from literal repeated recap.

### GDL 2025 Lecture 6: Failure Modes of Graph Neural Networks

- **Source:** `lec6_gnn4.pdf` (50 pages), represented by 48 canonical PDF
  composites preserving the diagnostic diagrams, energy and sensitivity
  equations, curvature examples, rewiring sequences, benchmark plots, and final
  research-practice evidence.
- **Coverage:** all 48 scientific slides covered; only the cover and course
  overview omitted. The source order remains under-reaching, over-smoothing and
  its competing measures, mitigation families and their risks, over-squashing
  sensitivity, binary-tree and curvature analysis, rewiring, width, graph
  distance versus commute time, dynamic delayed rewiring, long-range benchmarks,
  and the closing critique of application relevance, graph meaning, evaluation
  culture, and graph foundation models.
- **Figures:** all 48 source-native substantive visual composites reused (100%).
  No explanatory figure was drawn.
- **Synthetic material removed:** the previous failure-modes tutorial's custom
  three-failure summary, exact tree sensitivity calculation, Dirichlet numerical
  example, effective-resistance derivation, molecular computation-graph case,
  and architecture remedy table were absent from this deck. The replacement
  retains the lecture's empirical comparisons and its unusually long
  research-practice critique.
- **General lesson promoted:** the final nine slides are not an optional opinion
  appendix. They define when a better information-flow architecture supports a
  meaningful claim at all: the graph must encode a defensible relation and the
  evaluation must survive matched baselines and tuning. The blog-writing skill's
  rhetorical-destination rule now names this form of claim-boundary audit.
  and inflates the apparent figure denominator. The figure skill now records
  this rule.

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

## 19. Machine Learning Meets Quantum Chemistry

- **Post:** `_posts/2026-08-08-machine-learning-quantum-chemistry.md`
- **Topic-matched reference:** The polished quantum-chemistry/DFT post owns the
  derivation from the many-electron Hamiltonian through Hartree--Fock and the
  Kohn--Sham SCF loop. The spherical-equivariant anchor supplies the standard for
  typed angular objects, while the revised equivariant-potential and simulation
  companions own force-field deployment. This chapter should own the
  intervention boundary: what ML outputs, which solver consumes it, and how its
  errors propagate to the scientific observable and cost claim.
- **Before:** At 2,640 substantive words, the post is conceptually well
  organized by learned object and correctly identifies the computation that
  remains. Its equations are mostly endpoint definitions. VMC has no trial-state
  energy/variance calculation. The learned-functional section draws the SCF loop
  but does not show how a derivative perturbation changes fixed-point stability.
  The Hamiltonian warning about near degeneracy has no matrix witness. Density,
  delta-learning, fidelity, transfer, and cost claims likewise remain
  qualitative. The result reads as a high-quality boundary map rather than a
  30--45 minute calculation-led argument.
- **Criteria promoted:** When ML predicts an intermediate object used by a
  downstream solver, inject a controlled perturbation and carry it through the
  derivative, diagonalization, fixed point, or estimator; quantify conditioning
  and final-observable error rather than treating entrywise fit as sufficient.
- **Planned revision:** Preserve all eleven H2s and their order. Use a compact
  two-level electronic example to compute a variational energy, local-energy
  variance, and finite-sample error. Build a scalar SCF fixed-point model whose
  learned derivative changes contraction into divergence, including the role of
  mixing. Carry a small off-diagonal error through a near-degenerate $$2\times2$$
  Hamiltonian to eigenvalues and orbital rotation; pair it with a density
  normalization/dipole witness. Connect potentials/properties to their narrower
  output contracts without duplicating companion posts. Quantify two equal-MAE
  baselines with different residual structure for delta learning, a fidelity-
  dependent candidate ranking, and an out-of-domain dissociation or charge
  example. End with an amortized break-even calculation that includes reference
  data, training, remaining solver work, hardware/output contract, and fallback
  rate. Reuse the four existing figures unless a genuine missing relationship
  warrants another.
- **After:** Expanded from 2,640 to 4,580 substantive body words while
  preserving all eleven H2s and their order. A two-level trial state now carries
  variational energy, local-energy variance, autocorrelation, and effective
  sample error. A scalar learned SCF map preserves the correct fixed point while
  changing contraction into divergence, then shows what mixing repairs. A
  near-degenerate Hamiltonian propagates small matrix MAE to gap error and a
  22.5-degree orbital rotation; density examples separate normalization from
  dipole accuracy. Delta residual structure, omitted charge and dissociation
  tails, fidelity-dependent ranking, and a fallback-aware amortization
  calculation complete the intervention hierarchy. A final interface table maps
  each learned object to its downstream operation, sensitive diagnostic, and
  remaining cost. Existing figures remained sufficient. The promoted
  intermediate-perturbation rule shaped the chapter. Independent review caught
  and corrected two dropped LaTeX slashes before commit. Arithmetic/citation
  audits, validation, clean builds, and rendered checks passed.

## 20. Protein Structure Prediction from AlphaFold 1 to 3

- **Post:** `_posts/2026-08-08-protein-structure-prediction-alphafold.md`
- **Topic-matched reference:** The older protein-design chapter translates
  structural outputs into a fixed computational and experimental funnel, while
  the spherical-equivariant anchor sets the standard for frame-aware geometry.
  The Fokker--Planck anchor supplies the derivation/check rhythm for probabilistic
  claims. This chapter should own the changing AlphaFold representation and the
  exact scope of structural confidence, without becoming a catalog of modules.
- **Before:** At 2,531 substantive words, the post has a coherent AF1-to-AF3
  representation thesis and accurately distinguishes coevolution, distograms,
  pair states, residue frames, confidence, and all-atom diffusion. The MSA
  discussion does not calculate indirect correlation. The AF1 energy has no
  candidate structures or inconsistent distance witness. Evoformer and IPA
  equations are not instantiated numerically. pLDDT and PAE are described well,
  but no two-domain example shows why high local confidence can coexist with an
  uncertain assembly. Complexity, stochastic-sample semantics, and the gap from
  coordinates to affinity/function remain mostly qualitative.
- **Criteria promoted:** A confidence explanation must define its predicted
  random variable, alignment/conditioning convention, granularity, calibration
  population, and supported decision, then exhibit a case where two legitimate
  confidence summaries disagree.
- **Planned revision:** Preserve all ten H2s and their order, including the
  existing H3 claim-boundary subsections. Build a small binary MSA example where
  two positions correlate indirectly through a third, plus an effective-depth
  calculation for redundant sequences. Score two coordinate candidates under a
  finite distogram and show a triangle-inconsistent set of pair marginals. Work
  a scalar outer-product/triangle update and quantify pair/triangle memory and
  compute scaling. Carry residue-local points through one rigid transformation
  to check IPA distance and FAPE invariance. Use a two-domain confidence matrix
  with high pLDDT and high cross-domain PAE, stating exactly which structural
  decisions each supports. Give an AF3 noisy-coordinate/augmentation check and
  separate stochastic structural hypotheses from thermodynamic weights. Compute
  one pose-versus-affinity example and finish with a claim-scope contract from
  coordinates to dynamics, binding, and function. Reuse the four existing
  figures unless a genuine missing relationship warrants another.
- **After:** Expanded from 2,531 to 4,759 substantive body words while
  preserving all ten H2s and the existing claim-boundary H3s. A binary MSA now
  separates marginal from conditionally indirect correlation and quantifies
  redundancy-adjusted depth. Finite distograms score two realizable candidates
  and expose an unrealizable modal triangle. Scalar outer-product and triangle
  updates lead into explicit MSA/pair memory and cubic triangle arithmetic.
  Residue-local points verify IPA distance and FAPE invariance numerically. A
  two-domain pLDDT--PAE matrix implements the promoted confidence-scope rule by
  supporting local folds while withholding relative placement. AF3 corruption
  is checked under a rigid transform, and a 70:30 sample frequency is explicitly
  denied a thermodynamic interpretation. A two-kcal/mol binding example and
  final object/claim contract separate coordinates from affinity, dynamics, and
  function. Existing figures remained sufficient. Arithmetic/citation audits,
  validation, clean build, and rendered checks passed.

## 21. Protein Representation Learning Across Sequence and Structure

- **Post:** `_posts/2026-08-08-protein-representation-learning.md`
- **Topic-matched reference:** The polished protein-design chapter gives every
  representation an operational place in a finite design workflow. The revised
  AlphaFold chapter owns coevolution, pair geometry, residue frames, and
  confidence semantics. The spherical-equivariant anchor owns general geometric
  transformation theory. This chapter should instead own representation
  neighborhoods, pretraining-induced invariances, leakage, probing, and the
  evidentiary ladder behind claims that an embedding “captures” biology.
- **Before:** At 2,870 substantive words, the post has a strong thesis and broad
  coverage of sequence, MSA, graphs, frames, surfaces, objectives, pooling,
  provenance, splits, and probes. It rarely carries one protein or controlled
  record set across these views. Mutation log odds, contrastive loss, pooling,
  frame pose, graph locality, and homology thresholds have no numerical
  consequence. The final probing section correctly warns that decodability can
  reflect shortcuts but does not exhibit a family-matched functional
  counterexample or distinguish decodability, probe accessibility, fine-tuned
  use, and intervention-stable mechanism.
- **Criteria promoted:** A claim that a representation captures a concept must
  separate decodability, restricted-probe accessibility, downstream use, and
  intervention stability, then compare nuisance baselines on matched
  counterfactuals where family or metadata is fixed and mechanism changes.
- **Planned revision:** Preserve all thirteen H2s and their order. Carry one
  small protein/family example through sequence, spatial graph, frame, surface,
  and pooled views, quantifying which neighborhoods and mechanisms each exposes.
  Compute an MLM mutation log-odds without misreading it as free energy. Cross-
  link AlphaFold for coevolution rather than duplicate it, but quantify deep
  versus orphan MSA availability. Work a relative-frame/chirality example and a
  surface-resolution/probe-radius tradeoff. Evaluate one finite InfoNCE batch to
  show how positive and false-negative choices declare invariance. Quantify
  motif dilution under mean/sum/attention pooling. Trace a dated sequence--
  template--predicted-structure provenance path and carry a small record family
  through random, homology-cluster, and temporal splits. Finish with a matched
  homolog/convergent-function quartet, nuisance baseline ladder, probe-capacity
  comparison, and deployment contract. Reuse the four existing figures unless
  a genuine missing relationship warrants another.
- **After:** Expanded from 2,870 to 5,486 substantive body words while
  preserving all thirteen H2s and their order. A controlled 60-residue protein
  now moves through sequence, spatial graph, frame, surface, and pooled views,
  with numerical locality and dilution consequences. MLM log odds are explicitly
  separated from free energy, and deep/orphan MSA regimes are quantified.
  Relative-pose and chirality witnesses, surface resolution/probe radius, and a
  finite InfoNCE batch expose what each representation or objective preserves.
  A dated provenance chain and related-record family distinguish temporal from
  homology-aware splitting. Finally, a matched homolog/convergent-function
  quartet and nuisance/probe ladder implement the promoted distinction among
  decodability, restricted accessibility, downstream use, and intervention
  stability. A deployment contract reunites those claims. Existing figures
  remained sufficient. Arithmetic/citation audits, validation, clean build, and
  rendered checks passed.

## 22. Generative Models for Protein Design

- **Post:** `_posts/2026-08-08-generative-models-protein-design.md`
- **Topic-matched reference:** The polished protein-design chapter owns the
  biological primer, tool landscape, and end-to-end practical funnel. The
  revised AlphaFold and protein-representation chapters own structure-confidence
  semantics and representation evidence; the diffusion companions own
  denoising mechanics. This chapter should instead own the distinctions among
  sequence, inverse-folding, backbone, motif-conditioned, and joint generative
  distributions, plus the interfaces and evidence needed to move between them.
- **Before:** At 2,747 substantive words, the post has a clean nine-section arc
  and already separates the principal conditional distributions. It remains
  mostly qualitative: guidance never meets an epistatic failure, inverse
  folding has no designability calculation, motif constraints have no finite
  geometric witness, and the sequential-versus-joint argument has no proposal
  mass arithmetic. The design funnel counts attrition but not enrichment or
  independence of evidence, so agreement between a generator and a learned
  refolder can read as stronger validation than it is.
- **Criteria promoted:** When one learned model validates or filters another,
  audit shared training data, architectures, representations, and assumptions.
  Separate internal self-consistency from orthogonal computational evidence and
  experiment, and exhibit a correlated-error case where learned models agree
  while an independent check fails.
- **Planned revision:** Preserve all nine H2s and their order. Carry a
  hypothetical three-residue metal-binding design through sequence generation,
  inverse folding, backbone diffusion, motif scaffolding, co-design, filtering,
  and assays. Add a two-locus epistasis failure under additive guidance; a
  fixed-backbone sequence-entropy/designability calculation; a finite motif
  geometry constraint; and a two-region proposal table showing why
  `p(X)p(a\mid X)` can waste probability mass relative to a joint design model.
  Construct a shared-prior generator/refolder failure that passes learned
  self-consistency but fails an orthogonal energy or experimental check. Expand
  the funnel with a randomized or stratified control arm to estimate enrichment
  while preserving expression, folding, and function denominators. Finish with
  a feedback ledger assigning each negative result to the responsible stage.
  Reuse the four existing figures unless a genuine missing relationship
  warrants another.
- **After:** Expanded from 2,747 to 4,601 substantive body words while
  preserving all nine H2s and their order. A three-residue zinc-binding design
  now carries the post from sequence guidance through inverse folding, motif
  scaffolding, co-design, selection, assays, and feedback. Finite calculations
  expose additive guidance under epistasis, conditional sequence entropy,
  distance-only motif failure, and the useful proposal mass lost at a separately
  trained backbone--sequence interface. A shared-prior generator/refolder case
  implements the promoted independence-of-evidence audit by separating learned
  self-consistency from orthogonal chemistry and experiment. A prespecified
  stratified control quantifies ranker enrichment at both synthesis and assay
  denominators, and the final ledger assigns negative results to the earliest
  failed stage. Existing figures remained sufficient. Arithmetic/citation
  audits, validation, clean build, and rendered checks passed.

## 23. Genomic Foundation Models and Virtual Cells

- **Post:** `_posts/2026-08-08-genomic-foundation-models-virtual-cells.md`
- **Topic-matched reference:** There is no older standalone genomics chapter in
  the site. The Fokker–Planck anchor is the fairest mathematical analogue
  because it carefully separates individual paths, transition kernels, and
  population densities; the spherical-equivariant anchor supplies the same
  definition--derivation--check rhythm for representations. This chapter should
  own the ladder from genome sequence to regulatory labels, noisy cell-state
  observations, perturbation-conditioned population response, and the stronger
  evidentiary contract of a virtual cell.
- **Before:** At 2,722 substantive words, the post already rejects the idea that
  a large embedding is automatically a simulator and covers DNA, regulation,
  RNA, counts, multimodality, perturbations, causality, confounding, scaling,
  and evaluation. Most distinctions remain verbal. The masking shortcut has no
  finite token example; the count model never produces a zero probability;
  ranked encodings never visibly lose abundance; and mean response is criticized
  without a bimodal counterexample. Most importantly, destructive before/after
  snapshots are written as `x_0` and `x_t` without showing that the coupling
  between individual cells is unobserved.
- **Criteria promoted:** Destructive or unpaired measurements identify
  cross-sectional population marginals, not individual trajectories. Exhibit
  two couplings with identical observed marginals but different transitions and
  state what additional lineage, longitudinal, randomization, or structural
  assumptions identify the desired dynamic or counterfactual claim.
- **Planned revision:** Preserve all thirteen H2s and their order. Carry one
  hypothetical cytokine-response program from a regulatory sequence variant
  through four-gene single-cell counts, a latent state, an intervention, donor
  transfer, and a prospective decision. Quantify overlapping-k-mer leakage and
  reverse-complement task semantics; compute negative-binomial variance and
  zero probabilities at two library sizes; show two abundance vectors with the
  same rank encoding; and evaluate one finite multimodal contrastive batch.
  Contrast a bimodal responder population with an implausible mean cell. Give
  two transition matrices with the same before/after marginals but opposite
  cell-level dynamics, then name the measurements that distinguish them. Add a
  confounded batch/intervention table, independent-donor versus cell-count
  arithmetic, claim-matched baselines, calibration or interval coverage, and a
  final virtual-cell contract. Reuse the four existing figures unless a genuine
  missing relationship warrants another.
- **After:** Expanded from 2,722 to 4,643 substantive body words while
  preserving all thirteen H2s and their order. A hypothetical cytokine-response
  program now crosses the interfaces from regulatory variant to four-gene
  counts, latent state, perturbation response, donor transfer, calibration, and
  experiment selection. Finite examples expose overlapping-token masking
  leakage, reverse-complement task semantics, negative-binomial zero rates,
  rank-induced scale loss, and contrastive false negatives. A bimodal response
  defeats the mean-cell summary, while stay and swap transition matrices
  produce identical destructive-snapshot marginals and implement the promoted
  path-identification rule. A confounded batch table, cluster effective-size
  calculation, donor-stratified coverage example, probability-object contract,
  and abstention rule close the evidentiary chain. Existing figures remained
  sufficient. Arithmetic/citation audits, skill and blog validation, clean
  build, and rendered checks passed.

## 24. Two Routes to Graph Convolution

- **Post:** `_posts/2026-08-08-graph-convolution-spectral-equivariant.md`
- **Topic-matched reference:** The revised message-passing chapter owns the
  implementation family and controlled GCN/GraphSAGE/GAT/GIN comparisons. The
  revised symmetry chapter owns general group actions, representation closure,
  and architectural equivariance. The spherical-equivariant anchor shows how a
  formal symmetry construction earns intuition through explicit algebra. This
  chapter should instead own the exact overlap and non-equivalence between the
  Laplacian-spectral and permutation-commutant routes to graph convolution.
- **Before:** At 2,747 substantive words, the post has a strong ten-section
  spine and unusually careful caveats about repeated eigenspaces, locality,
  bilinearity, and the limits of permutation symmetry. Its examples are split:
  cyclic convolution, abstract spectral filtering, a four-node GCN update, and
  higher-order fixed points never act on one shared graph signal. The route from
  a Chebyshev filter to GCN compresses rescaling, coefficient tying, and
  renormalization into prose, which can make the practical GCN operator look
  exactly derived rather than deliberately approximated.
- **Criteria promoted:** When multiple derivations motivate one method, carry
  the same object and output through each route, maintain an assumption ledger,
  prove their exact overlap, and locate the approximation or added relational
  input where their guarantees diverge.
- **Planned revision:** Preserve all ten H2s and their order. Use a three-node
  path and one impulse signal to compute its Laplacian eigensystem, graph Fourier
  coefficients, a rational low-pass response, a first-degree localized
  polynomial, and a self-loop-normalized GCN propagation. Start with a finite
  cyclic-shift commutant calculation, and use a four-cycle repeated eigenspace
  to expose basis ambiguity without changing `h(L)`. Derive the ChebNet-to-GCN
  sequence with an explicit ledger for spectral rescaling, first-order
  truncation, coefficient tying, and renormalization; state which equalities are
  exact and which are design choices. Prove polynomial-filter equivariance under
  simultaneous graph relabeling, instantiate the node-feature commutant, and
  connect higher-order equality patterns to graph-dependent bilinear `AX`.
  Quantify sparse-polynomial versus eigendecomposition cost and finish with a
  two-route claim table. Reuse the three existing figures unless a genuine
  missing relationship warrants another.
- **After:** Expanded from 2,747 to 4,799 substantive body words while
  preserving all ten H2s and their order. A single P3 impulse now moves through
  the Laplacian eigensystem, rational low-pass reconstruction, localized
  polynomial, normalized-Laplacian approximation, and self-loop GCN update. A
  finite cyclic commutant and a C4 repeated-eigenspace witness distinguish
  scalar spectral functions from the full Laplacian commutant. The
  ChebNet-to-GCN ledger labels rescaling, truncation, coefficient tying,
  `lambda_max` substitution, and renormalization as exact identities,
  approximations, restrictions, or replacements. Polynomial relabeling
  equivariance, the feature-only commutant obstruction, small-N Bell-number
  boundaries, and the bilinear `AX` contraction then connect the second route
  on the same signal. Cost arithmetic and a final claim table make the overlap
  and blind spots explicit. Existing figures remained sufficient. Matrix,
  citation, skill, blog, build, and rendered audits passed.

## 25. Frames, Canonicalization, and Symmetrization

- **Post:** `_posts/2026-08-08-frames-canonicalization-symmetrization.md`
- **Topic-matched reference:** The revised symmetry chapter owns general orbit,
  stabilizer, group-averaging, and closure arguments. The spherical-equivariant
  anchor owns representation-aware geometric primitives. This chapter should
  instead own the operational trade among choosing one pose, choosing a local
  pose, averaging a finite equivariant frame, and sampling a pose distribution,
  especially near configurations where pose selection becomes singular.
- **Before:** At 2,285 substantive words, the post has a clean seven-section
  progression and already states the stabilizer obstruction, PCA degeneracy,
  local Gram--Schmidt frame, Haar proof, finite frame law, and probabilistic
  pushforward condition. The constructions remain almost entirely symbolic.
  PCA never receives a perturbation whose principal axis jumps; the local frame
  never approaches collinearity; and the four averaging recipes never evaluate
  the same arbitrary backbone. The Monte Carlo paragraph does not separate
  per-draw, coupled-draw, distributional, and expectation equivariance or
  quantify their residuals.
- **Criteria promoted:** For randomized symmetry enforcement, distinguish exact
  per-sample, coupled-sample, distributional, and expectation-level guarantees.
  Compute a finite estimator's equivariance residual and variance, and state how
  randomness is shared across transformed inputs.
- **Planned revision:** Preserve all seven H2s and their order. Carry a centered
  anisotropic four-point cross through orbit, quotient, deterministic PCA pose,
  the eigenvalue tie at zero anisotropy, signed PCA frames, and weighted pose
  distributions. Compute its covariance and 90-degree canonical-axis jump. Use
  one local vector triple to verify scalarization/vectorization under rotation,
  quantify the Gram--Schmidt singularity as the vectors become collinear, and
  check the reflection-parity failure. On a finite four-pose orbit, evaluate one
  unrestricted backbone under canonicalization, full group averaging, finite
  frame averaging, and Monte Carlo symmetrization. Derive expectation and
  residual variance for independent versus coupled samples, plus evaluation
  cost and a final regime-selection table. Reuse the four existing figures
  unless a genuine missing relationship warrants another.
- **After:** Expanded from 2,285 to 5,110 substantive body words while
  preserving all seven H2s and their order. A centered anisotropic cross now
  moves through quotient coordinates, deterministic PCA, a computed eigengap,
  the exact 90-degree pose jump, signed frames, stabilizer closure, and a smooth
  four-pose weight law. A local vector triple verifies scalarization,
  vectorization, collinearity amplification, and reflection parity. One
  unrestricted four-pose backbone then makes canonicalization, finite frame
  averaging, full group averaging, and randomized symmetrization numerically
  distinct. The promoted guarantee ladder separates deterministic,
  coupled-sample, distributional, and expectation equivariance; the independent
  residual has variance `40/M` while coupled poses make it zero without removing
  integration error. Cost formulas and a regime table close the comparison.
  Existing figures remained sufficient. Geometry, pushforward, variance,
  citation, skill, blog, build, and rendered audits passed.

## 26. Protein Ensembles and Learned Molecular Dynamics

- **Post:** `_posts/2026-08-08-protein-ensembles-learned-dynamics.md`
- **Topic-matched reference:** The polished Fokker–Planck chapter owns the
  path-to-density derivation. The revised molecular-simulation chapter owns
  integrators, force-field error, sampling convergence, and observable error
  budgets. The revised probability-flow and geometric-flow chapters own
  generative path semantics. This chapter should instead own the distinction
  among protein equilibrium ensembles, coarse transfer operators, and
  physical-time path laws, with state representation as the interface.
- **Before:** At 2,296 substantive words, the post cleanly separates ensembles,
  transition kernels, and trajectories and already names detailed balance,
  implied timescales, Chapman–Kolmogorov tests, reweighting, and experimental
  forward models. Its equations never close on one finite system. The MSM has
  no count matrix or eigenvalue calculation, and the warning that a learned
  state “hides memory” has no witness. Equilibrium-versus-kinetics uses rates
  but does not derive the transition kernel or relaxation time, while generated
  trajectories are not checked for multi-lag consistency.
- **Criteria promoted:** A representation used as a dynamical state must be
  audited for Markov sufficiency or lumpability. Merge microstates with
  different outgoing laws, compute the history-dependent coarse transition,
  and show what state refinement, lag change, or memory variable repairs it.
- **Planned revision:** Preserve all nine H2s and their order. Carry a
  three-microstate protein switch with two geometrically similar open rotamers
  and one closed state through equilibrium populations, free energy, MD counts,
  a reversible MSM, eigenvalues, implied timescales, and a coarse open/closed
  model. Use an explicit symmetric transition matrix and show that merging the
  two open rotamers violates lumpability because their closed-state transition
  probabilities differ; compare equilibrium and entry-conditioned mixtures and
  a Chapman–Kolmogorov check. Quantify timestep scale and one enhanced-sampling
  reweighting example. Derive a two-state continuous-time kernel showing equal
  stationary populations but 1000-fold different relaxation. Add a
  multi-lag/semigroup inconsistency for a learned trajectory model, equilibrium
  observable and importance-weight arithmetic, experimental forward-model
  nonlinearity, and a final claim-matched diagnostic table. Reuse the four
  existing figures unless a genuine missing relationship warrants another.
- **After:** Expanded from 2,296 to 4,522 substantive body words while
  preserving all nine H2s and their order. A reversible three-microstate protein
  switch now supplies stationary populations, aggregate free energy, exact
  counts, detailed balance, eigenvalues, and implied timescales. Merging its two
  open rotamers creates the promoted non-lumpability witness: equilibrium-,
  entry-, and survival-conditioned hidden mixtures yield different closing
  hazards, and direct two-lag propagation disagrees with the squared coarse
  kernel. Timestep scaling, enhanced-sampling reweighting, generator importance
  weights and ESS, a continuous-time two-state kernel with a 1000-fold clock
  change, and an inconsistent pair of learned lag heads separate ensemble,
  transfer, and physical-time claims. A nonlinear FRET forward model and final
  diagnostic table complete the evidence chain. Existing figures remained
  sufficient. Matrix, probability, citation, skill, blog, build, and rendered
  audits passed.

## 27. Crystal Property Prediction and Generative Design

- **Post:** `_posts/2026-08-08-crystal-property-prediction-generative-design.md`
- **Topic-matched reference:** The polished spherical-equivariant chapter
  supplies the standard for deriving a symmetry constraint and checking the
  transformation law. The revised materials-discovery chapter owns the broader
  composition-to-synthesis pipeline, convex-hull thermodynamics, defects, and
  experimental closure. This chapter should instead own the periodic
  representation contract that connects unit cells, crystal graphs, property
  predictors, joint generators, relaxation, and post-relaxation evaluation.
- **Before:** At 2,346 substantive words, the post has a coherent nine-section
  spine and correctly distinguishes property prediction, generation, symmetry,
  oracle guidance, relaxation, and scientific validation. It says that
  primitive and conventional cells can encode the same crystal but its stated
  invariance law checks only permutation, rigid rotation, and lattice wrapping.
  No finite lattice-basis transformation is carried through fractional sites
  or periodic edges. The graph, generation, space-group, guidance, and
  relaxation sections likewise remain schematic: there is no neighbor-image
  calculation, Wyckoff multiplicity example, guided finite pool, many-to-one
  relaxation witness, or complete finite validation funnel.
- **Criteria promoted:** When coordinates contain an arbitrary basis, gauge, or
  unit-cell choice, define the full equivalence action and carry one nontrivial
  basis change through every dependent object. Transform coordinates,
  neighborhoods, features, densities, and tensor outputs together, then verify
  the claimed invariant or covariant result numerically.
- **Planned revision:** Preserve all nine content H2s and their order. Carry one
  small two-dimensional crystal embedded in three dimensions through an integer
  unimodular lattice-basis change, transformed fractional coordinates, and an
  explicit nearest periodic image. Distinguish intensive, extensive, vector,
  and tensor output laws; compute one periodic message-passing update and state
  the locality boundary. Follow a fixed-composition candidate through wrapped
  coordinate diffusion, lattice validity, Wyckoff multiplicity, a finite
  guidance/Pareto choice, and two raw proposals that relax into one basin.
  Finish with a denominator-preserving raw-to-relaxed-to-stable funnel and a
  claim table that separates proposal coverage, relaxed outcomes, stability,
  and synthesis. Cross-link the materials chapter for the wider physical
  pipeline and reuse the four existing figures unless a real explanatory gap
  remains.
- **After:** Expanded from 2,346 to 4,599 substantive body words while
  preserving all nine content H2s and their order. A single layered AB crystal
  now instantiates the full fixed-site-count action: permutation, rotation,
  common origin shift, integer images, and a nontrivial unimodular basis change
  are carried through Cartesian sites, fractional components, a periodic edge,
  one message update, forces, stress, and model density. Wrapped site noise,
  lattice conditioning, and a Wyckoff-multiplicity calculation make the joint
  generator constraints operational. A three-candidate guidance calculation
  exposes prior tilting, Pareto policy choices, uncertainty, and correlated
  oracle error. Three raw proposals then collapse to two relaxed basins and
  reverse their property ranking before a 10,000-proposal campaign preserves
  every validity, collision, selection, DFT, synthesis, and characterization
  denominator. The four existing figures remained sufficient. Matrix,
  arithmetic, citation, skill, blog, build, and rendered-page audits passed.

## Series consolidation and regression

- **Coverage:** All 27 canonical lecture-derived chapters were reviewed and
  revised in order. Their substantive body lengths now range from 4,522 to
  5,807 words, meeting the 30--45 minute depth diagnostic without adding
  learning objectives, concept checks, exercises, or prerequisite scaffolding.
- **Criteria consolidation:** The promoted rules remain organized by role.
  `blog-writing` owns chapter depth, running examples, comparison discipline,
  decision denominators, evidence boundaries, and reader-facing coherence.
  `academic-writing` owns derivation closure, convention checks, constrained
  domains, identifiability, and transformation-law audits. The final review
  found no duplicate within-skill rule that could be merged without weakening
  its diagnostic trigger. Jekyll and figure rules already covered every
  rendering and visual issue encountered, so no series-specific exception was
  added to those skills.
- **Regression:** The canonical data contain 43 ordered reading-path entries
  resolving to 27 unique posts: 23 chapters in ML4Mol and 20 in GDL, including
  their intentional overlap. Post titles, path membership, categories,
  citations, backlinks, figure assets, and internal post links are consistent.
  All five relevant skills validate, `scripts/validate_blog.py` passes with only
  the pre-existing unused `_site` image warning, and `git diff --check` is clean.
  A clean production Jekyll build with responsive-image generation completed
  successfully. All 27 chapter pages, both reading-path hubs, eight category
  archives, and their local links were verified in the rendered site.
