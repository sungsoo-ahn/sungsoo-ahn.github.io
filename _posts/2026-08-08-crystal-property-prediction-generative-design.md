---
layout: post
title: "Crystal Property Prediction and Generative Design"
date: 2026-08-08
last_updated: 2026-08-08
description: "How periodic representations support crystal property prediction and generative design—and why relaxation, first-principles validation, and synthesis remain decisive."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [materials-science]
lecture_paths: [gdl]
tags: [crystal-graphs, periodic-equivariance, materials-generation, diffusion-models, density-functional-theory]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post focuses on the geometric learning problem inside crystalline materials discovery: how an infinite periodic solid becomes a finite representation, how predictors preserve its symmetries, how generators move composition, lattice, and atomic sites, and why a generated crystal is only the beginning of validation.</em>
</p>

## A Crystal Is Infinite, but Its Description Should Not Be

A molecule is usually represented as a finite collection of atoms. A perfect crystal repeats without end. That single difference changes the representation, the symmetry, and the meaning of generation.

Choose a lattice matrix

$$
L=[\ell_1,\ell_2,\ell_3]\in\mathbb R^{3\times 3},
$$

whose columns are lattice vectors, and fractional coordinates $$s_i\in[0,1)^3$$ for the atoms in one unit cell. The Cartesian coordinate is $$x_i=Ls_i$$. The infinite set of images is

$$
\mathcal C=\{(z_i,L(s_i+n)):i=1,\ldots,N,\ n\in\mathbb Z^3\},
$$

where $$z_i$$ is the chemical element. The finite tuple $$(A,L,S)$$—element types, lattice, and fractional sites—describes the infinite solid.

This description is not unique. Replacing $$s_i$$ by $$s_i+n$$ changes no atom. Reordering sites changes no material. A rigid rotation $$Q$$ sends $$L\mapsto QL$$ and every Cartesian position to $$Qx_i$$ without changing a scalar property. Different choices of primitive or conventional cell can represent the same crystal. Space-group operations add rotations, reflections, screws, glides, and translations that map a crystal onto itself.

{% include figure.liquid loading="eager" path="assets/img/blog/crystgen_periodic_representation.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A periodic crystal is represented by a lattice and sites in one unit cell, but shifts by lattice vectors, site permutations, and rigid changes of frame do not create a new material. A model should return one physical answer across these equivalent descriptions." %}

These are not optional augmentations. They define the quotient space on which learning occurs. For a scalar property $$f$$ such as formation energy per atom or band gap, a basic requirement is

$$
f(A,L,S)=f(\pi A,QL,\pi S+n),
$$

for any site permutation $$\pi$$, rigid rotation $$Q$$, and integer lattice shifts $$n_i$$ applied to fractional sites. Forces differ: they are vector-valued, so they must rotate with the structure. A model that violates these relations can assign different energies to the same physical crystal.

The companion post [Materials Discovery Connects Structure, Properties, and Synthesis]({% post_url 2026-08-08-materials-discovery-structure-properties-synthesis %}) develops the broader scientific loop. Here the narrower concern is what periodic geometry permits a learned model to claim.

## Crystal Graphs Turn Periodicity into Local Neighborhoods

A crystal graph takes sites in a reference cell as nodes and connects atoms through nearby periodic images. For sites $$i$$ and $$j$$, an edge can carry the displacement

$$
r_{ij,n}=L(s_j-s_i+n),\qquad n\in\mathbb Z^3,
$$

when $$\lVert r_{ij,n}\rVert$$ lies below a cutoff. The integer image index $$n$$ matters: two atoms on opposite faces of the displayed unit cell may be nearest neighbors in the infinite crystal. The graph is finite because only images within the cutoff are included.

{% include figure.liquid loading="lazy" path="assets/img/blog/crystgen_periodic_graph.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Periodic message passing includes neighbors across the unit-cell boundary. The image atom is bookkeeping for an interaction in the infinite lattice, not an additional atom. Scalar predictions should be invariant to rigid motion, whereas predicted force vectors must rotate with the crystal." %}

A message-passing predictor updates site features as

$$
h_i^{(k+1)}=\phi\!\left(h_i^{(k)},
\sum_{j,n:\lVert r_{ij,n}\rVert<r_c}
\psi\!\left(h_i^{(k)},h_j^{(k)},e(r_{ij,n})\right)\right),
$$

then pools sites to obtain a crystal representation. If $$e$$ uses only distance expansions, the model is invariant to rigid rotations and reflections. This is sufficient for many scalar properties. CGCNN established this graph-based template for learning formation energies, band gaps, elastic quantities, and related properties directly from crystal structures (<span id="cite-xie2018"></span>[Xie and Grossman, 2018](#ref-xie2018)).

Distance-only messages nevertheless discard angular information unless several layers reconstruct it indirectly. Periodic equivariant networks instead maintain vector or higher-order geometric features. Under a rigid rotation $$Q$$, a scalar feature remains unchanged while a vector feature transforms as $$v_i\mapsto Qv_i$$. Equivariant coordinate or force updates can then use directional information without choosing a preferred laboratory frame.

The distinction matters most when the output is geometric. An energy model should be invariant, $$E(QX)=E(X)$$. If forces are obtained consistently as $$F_i=-\nabla_{x_i}E$$, then

$$
F_i(QX)=QF_i(X).
$$

This gives the right behavior for relaxation and molecular dynamics. Space-group symmetry is more specific than global Euclidean symmetry: symmetry-equivalent sites should receive compatible features and forces. Encoding known space groups or Wyckoff positions can reduce redundancy, but an incorrectly imposed space group can also exclude a lower-symmetry phase.

## Property Prediction Is Conditional on Structure and Data Provenance

For a known relaxed structure, a predictor approximates

$$
\widehat y=f_\theta(A,L,S).
$$

This looks straightforward, but the target encodes a calculation protocol. Formation energies depend on reference states and density-functional settings. Band gaps inherit the systematic bias of the electronic-structure approximation. Magnetic configurations, disorder, temperature, pressure, and defects may be absent from the input even when they control the measurement.

Formation energy alone also does not determine stability against decomposition. For composition $$c$$, stability is judged relative to competing phases through the energy above the convex hull,

$$
E_{\mathrm{hull}}(c)=E(c)-\min_{\{\lambda_k\}}
\sum_k\lambda_kE(c_k),
$$

subject to $$\lambda_k\ge 0$$, $$\sum_k\lambda_k=1$$, and $$\sum_k\lambda_kc_k=c$$. A negative formation energy relative to elements can coexist with a positive hull distance because a mixture of compounds is lower in energy. The result is meaningful only relative to the phase set and calculation consistency used to build the hull.

Generalization is therefore chemical and structural, not merely random-split accuracy. A random split can place nearly identical compositions or prototypes in train and test sets. Performance on a new chemical system, pressure regime, or coordination motif can be substantially worse. Uncertainty estimates help triage expensive calculations, but they must be calibrated on the same kind of distribution shift that deployment will encounter.

## Generation Moves Composition, Cell, and Sites Together

Crystal structure prediction with fixed composition seeks plausible $$(L,S)$$ given $$A$$. Ab initio generation may move all three variables. They inhabit different spaces:

- element types and stoichiometric counts are discrete;
- a valid lattice is a continuous, nonsingular matrix modulo rotations and equivalent cell choices;
- fractional sites live on a three-dimensional torus because $$s_i$$ and $$s_i+n$$ are identical.

{% include figure.liquid loading="lazy" path="assets/img/blog/crystgen_generative_variables.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A crystal generator couples discrete composition, continuous cell geometry, and periodic fractional sites. Charge balance, density, lattice shape, and coordination constrain one another, so separately plausible outputs need not form a plausible crystal." %}

This coupling rules out a naive Cartesian generator. If coordinates are noised independently while the cell is generated separately, density and coordination can drift apart. If fractional coordinates are treated as ordinary Euclidean variables, sites near 0 and 1 appear far apart despite being adjacent on the torus. If atom counts change, the state dimension changes. If the cell matrix is parameterized without constraints, it can become singular or flip handedness.

CDVAE addressed periodic material generation with a latent variable for composition and lattice and a score-based decoder for atomic coordinates (<span id="cite-xie2022"></span>[Xie et al., 2022](#ref-xie2022)). Its conceptual factorization is useful: global variables propose the broad crystal, while a periodic decoder realizes atom positions. But reconstruction quality and latent smoothness do not by themselves establish thermodynamic stability.

DiffCSP instead formulates crystal structure prediction as joint equivariant diffusion of lattice and fractional coordinates conditioned on composition (<span id="cite-jiao2023"></span>[Jiao et al., 2023](#ref-jiao2023)). A schematic forward process is

$$
L_t=\alpha_tL_0+\sigma_t\epsilon_L,
\qquad
s_{i,t}=(s_{i,0}+\sigma_t\epsilon_i)\bmod 1.
$$

The wrapped coordinate process respects periodicity; the denoiser must also preserve permutation and E(3) behavior. Flow matching offers a related construction: learn a time-dependent vector field carrying a simple base distribution to the data distribution. In either case, the interpolation path is part of the inductive bias. It determines what intermediate cells and sites the network must learn to repair.

## Symmetry Can Be Predicted, Conditioned, or Enforced

Space-group information offers a compact description of repeated structure. One strategy generates all sites and later identifies symmetry. Another conditions generation on a target space group. A stronger strategy generates only an asymmetric unit and expands it through space-group operations. If a site $$s$$ occupies a Wyckoff position, the orbit

$$
\mathcal O(s)=\{R_gs+t_g\bmod 1:g\in G\}
$$

produces all symmetry-equivalent sites under space group $$G$$.

Enforcement guarantees exact symmetry and reduces the number of free coordinates. It can also make composition constraints delicate because Wyckoff multiplicities determine how many atoms appear. Conditioning is more flexible but can produce approximate symmetry that disappears after relaxation. Predicting symmetry from data risks treating inconsistent conventional-cell labels as physical differences.

The correct choice depends on the scientific question. Searching within a known high-symmetry family benefits from enforcement. Discovering symmetry-breaking distortions or low-temperature phases requires freedom to leave that family. Symmetry is a prior over the search space, not an unconditional guarantee of stability.

## Property Guidance Creates an Oracle-Optimization Problem

Inverse design asks for crystals satisfying a target property $$y^*$$. A conditional generator approximates $$p_\theta(A,L,S\mid y^*)$$; classifier or energy guidance modifies a base generator using gradients of a learned oracle. MatterGen jointly denoises atom types, coordinates, and the lattice, and supports conditioning on composition, symmetry, and scalar properties (<span id="cite-zeni2025"></span>[Zeni et al., 2025](#ref-zeni2025)). Its experimental synthesis of a generated candidate is important precisely because generative and predictive scores are not the final evidence.

Guidance produces a familiar tradeoff. Stronger pressure toward a target can increase predicted performance while decreasing diversity and moving candidates outside the oracle's training distribution. A model optimized for low predicted formation energy may repeat a narrow prototype family. A band-gap target may be satisfied according to a surrogate whose errors correlate with composition. Multi-objective design introduces further tension: stability, abundance, toxicity, conductivity, mechanical response, and processability rarely improve together.

The appropriate output is therefore a Pareto set rather than a single “best” crystal. Candidate $$u$$ dominates $$v$$ only if it is no worse on every objective and better on at least one. This leaves tradeoffs visible for subsequent calculations and experiments. Collapsing them into one weighted score hides value judgments and creates a sharper target for oracle exploitation.

## Valid, Unique, Novel, Stable, and Useful Are Different Claims

Generative evaluation often reports validity, uniqueness, and novelty. Each requires a precise equivalence rule.

**Validity** may mean only parseable elements, a nonsingular cell, and no severe overlaps. Chemical validity additionally involves oxidation states, charge balance, coordination, and density. **Uniqueness** requires symmetry-aware structure matching; byte-distinct CIF files can encode the same crystal. **Novelty** depends on the database, tolerance, and whether comparison uses composition, prototype, or relaxed structure. A small distortion of a known phase can appear novel before relaxation and identical afterward.

**Stability** is yet another claim. A property predictor can screen formation energy, but relaxation may move the structure far from the generated geometry. DFT can locate a local minimum and estimate hull distance, but zero-temperature thermodynamic stability does not guarantee dynamical stability. Imaginary phonon modes indicate distortions along which the structure can lower its energy. Finite-temperature free energies, pressure, disorder, and metastable kinetic trapping can change the picture.

Finally, a stable crystal need not be synthesizable, and a synthesizable material need not be useful. Precursors, reaction pathways, competing phases, defects, particle size, processing, and environmental stability affect what the laboratory obtains. Device performance can depend on interfaces and microstructure absent from an ideal periodic cell.

## Relaxation Is Part of Evaluation, Not Cosmetic Cleanup

A generated structure $$X_g=(A,L,S)$$ is usually not at a local energy minimum. Relaxation iterates positions and often the cell using predicted or first-principles forces,

$$
X^{(k+1)}=X^{(k)}-\eta_k\nabla_XE(X^{(k)}),
$$

until forces and stresses are small. Machine-learned interatomic potentials can screen many candidates cheaply; selected structures then receive density-functional theory (DFT) relaxation and property calculations.

The displacement between generated and relaxed structures is diagnostic. Small drift suggests that the generator proposed a basin near a local minimum. Large drift means the original coordination or cell was not self-consistent, even if the relaxed endpoint is valid. Reporting only the endpoint gives the generator credit for the optimizer's repair. Conversely, rejecting every large-drift candidate can discard unusual but physically meaningful starting points. The threshold must match the intended claim.

{% include figure.liquid loading="lazy" path="assets/img/blog/crystgen_validation_funnel.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A crystal generator begins a hierarchy of tests: symmetry-aware deduplication, relaxation, and first-principles validation. DFT is usually applied only after learned filters, so the final success rate is conditional on the selection funnel. A low-energy relaxed structure still does not establish synthesis or application performance." %}

This funnel creates selection bias. Suppose one million samples are reduced to ten thousand by a learned stability model, one thousand by novelty and property filters, and one hundred by relaxation before DFT. The DFT success rate describes those hundred candidates, not the generator's unconditional yield. It also cannot show whether a property oracle added value unless random or stratified candidates are calculated as controls.

A rigorous report preserves every denominator: raw samples, valid structures, symmetry-aware unique structures, database-novel structures, successfully relaxed structures, DFT-stable structures, target-property candidates, synthesis attempts, and characterized products. It records thresholds and failures, not only the final shortlist. Comparisons should use the same computational settings and reference phase diagram. Prospective experiments should fix the selection policy before outcomes are known.

## Generative Design Expands the Search, Not the Evidence

Periodic graphs and equivariant networks solve a foundational representation problem: physically equivalent cells should lead to equivalent predictions, and vector outputs should transform correctly. Diffusion and flow models then turn composition, lattice, and sites into movable design variables. Space groups can restrict the search; property models can guide it; relaxation can test whether a proposal sits near an energy minimum.

None of these steps collapses the materials pipeline into a single score. Property predictions inherit the coverage and approximation of their labels. Generated novelty depends on equivalence definitions. Stability is conditional on competing phases and thermodynamic conditions. DFT is evidence about an idealized model, not direct evidence of a synthesis route. Experimental success is conditional on every filter that selected the candidate.

The right role for crystal generation is therefore expansive but precise: propose coherent regions of chemical and structural space that conventional enumeration might miss, then submit those proposals to progressively more independent tests. The generator contributes imagination. Periodic geometry makes that imagination physically legible. Relaxation, first-principles calculations, and synthesis decide how much of it survives.

---

## References

<span id="ref-xie2018"></span>Xie, T. and Grossman, J. C. “Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties.” *Physical Review Letters* 120, 145301 (2018). [doi:10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301) [↩](#cite-xie2018)

<span id="ref-xie2022"></span>Xie, T. et al. “Crystal Diffusion Variational Autoencoder for Periodic Material Generation.” *International Conference on Learning Representations* (2022). [OpenReview](https://openreview.net/forum?id=03RLpj-tc_) [↩](#cite-xie2022)

<span id="ref-jiao2023"></span>Jiao, R. et al. “Crystal Structure Prediction by Joint Equivariant Diffusion.” *Advances in Neural Information Processing Systems* 36 (2023). [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html) [↩](#cite-jiao2023)

<span id="ref-zeni2025"></span>Zeni, C. et al. “A generative model for inorganic materials design.” *Nature* 639, 624–632 (2025). [doi:10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5) [↩](#cite-zeni2025)

---

*Figure provenance: all four diagrams are original explanatory syntheses created for this post with `scripts/generate_crystgen_figures.py`. They reproduce no lecture, paper, or Flaticon assets and are released under CC BY 4.0 with the post.*
