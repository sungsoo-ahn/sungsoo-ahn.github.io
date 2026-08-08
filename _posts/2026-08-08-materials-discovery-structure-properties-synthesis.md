---
layout: post
title: "Materials Discovery Connects Structure, Properties, and Synthesis"
date: 2026-08-08
last_updated: 2026-08-08
description: "A material is more than a chemical formula: discovery must connect periodic structure, competing phases, target properties, processing conditions, and experimental formation."
abstract: "Machine learning can screen and generate crystal structures quickly, but a predicted crystal is not yet a discovered material. This post follows the full argument from periodic representations and structure-dependent properties to convex-hull stability, defects, polymorphs, synthesis conditions, active learning, and experimental closure."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [materials-science]
lecture_paths: [ml4mol, gdl]
tags: [materials-discovery, crystal-structures, geometric-deep-learning, inverse-design, active-learning]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the materials-discovery storyline from my Machine Learning for Molecules and Geometric Deep Learning lectures. It follows the chain from composition and periodic structure through stability, properties, synthesis, and experimental closure.</em>
</p>

## A formula is not a material

Materials discovery is often drawn as an inverse problem: specify a desired property, then search for a chemical formula that provides it. That picture omits the variable that usually decides the answer. A material is a composition arranged into a structure, produced under conditions that may or may not realize that structure.

Consider a crystal with $$N$$ atoms in one unit cell. Let $$\mathbf{L}=[\boldsymbol{\ell}_1,\boldsymbol{\ell}_2,\boldsymbol{\ell}_3]\in\mathbb{R}^{3\times 3}$$ denote the lattice matrix, let $$\mathbf{s}_i\in[0,1)^3$$ denote the fractional coordinate of atom $$i$$, and let $$z_i$$ denote its chemical element. Its Cartesian coordinate is

$$\mathbf{r}_i=\mathbf{L}\mathbf{s}_i.$$

The finite tuple $$X=(\mathbf{L},\{(\mathbf{s}_i,z_i)\}_{i=1}^{N})$$ represents the infinite periodic set

$$\mathcal{C}(X)=\left\{\left(\mathbf{L}(\mathbf{s}_i+\mathbf{n}),z_i\right): i=1,\ldots,N,\;\mathbf{n}\in\mathbb{Z}^3\right\}.$$

The same infinite crystal admits many finite descriptions. We can permute atoms, translate the origin, wrap a fractional coordinate across a cell boundary, or choose another valid unit cell. A property model must give the same scalar prediction for all equivalent descriptions. When it predicts a vector or tensor, such as force or stress, its output must transform consistently with rotations and changes of basis.

Composition discards most of this information. Two crystals can share a reduced formula but differ in lattice, atomic coordination, or space group. Such **polymorphs** can have different band gaps, elastic constants, magnetic order, and kinetic accessibility. Figure 1 makes the missing variable visible.

{% include figure.liquid loading="eager" path="assets/img/blog/matdisc_composition_structure.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 1. A composition specifies atom counts but not their periodic arrangement. Two polymorphs with the same formula can place the same elements in different coordination environments, producing different electronic and mechanical properties." %}

Crystal symmetry compresses this description further, but it does not remove structure. A space group collects the rotations, reflections, inversions, screw operations, glide operations, and translations that leave the crystal unchanged. Wyckoff positions group sites that are equivalent under these operations. Symmetry therefore constrains which coordinates are independent and which tensor components can be nonzero. It does not imply that two structures with the same composition—or even the same space group—have the same energy.

## The target is a property of a state

Materials are judged by different observables because applications ask different physical questions. For an electronic material, one common target is the band gap

$$E_{\mathrm{g}}=\min_{\mathbf{k}}E_{\mathrm{c}}(\mathbf{k})-\max_{\mathbf{k}}E_{\mathrm{v}}(\mathbf{k}),$$

where $$E_{\mathrm{c}}(\mathbf{k})$$ and $$E_{\mathrm{v}}(\mathbf{k})$$ are the lowest conduction-band and highest valence-band energies at wavevector $$\mathbf{k}$$. The minimizing and maximizing wavevectors need not agree, so the definition includes indirect gaps. The value also depends on the electronic-structure approximation: a density-functional-theory band gap is a calculated label, not an error-free experimental truth.

Mechanical response asks a different question. For a small strain tensor $$\boldsymbol{\varepsilon}$$, the energy density can be expanded as

$$\frac{E(\boldsymbol{\varepsilon})}{V}=\frac{E(\mathbf{0})}{V}+\frac{1}{2}\sum_{i,j,k,l}C_{ijkl}\varepsilon_{ij}\varepsilon_{kl}+\mathcal{O}(\lVert\boldsymbol{\varepsilon}\rVert^3),$$

where $$V$$ is the cell volume and $$C_{ijkl}$$ is the elastic tensor. Bulk and shear moduli are reductions of this tensor under specified averaging assumptions. A high bulk modulus does not imply fracture toughness, just as a useful band gap does not imply high carrier mobility.

Thermodynamic targets require another layer. For a phase at temperature $$T$$ and pressure $$P$$, the relevant potential is the Gibbs free energy

$$G(T,P)=E+PV-TS,$$

where $$E$$ is internal energy, $$V$$ is volume, and $$S$$ is entropy. High-throughput databases usually approximate solid-phase comparisons with relaxed electronic energies at zero temperature and low pressure. That approximation is useful because it is standardized and scalable. It cannot settle phase competition driven by vibrational, configurational, magnetic, or electronic entropy.

The choice of target must therefore name both the observable and the state. “Find a good conductor” is incomplete without temperature, carrier concentration, microstructure, and transport direction. “Find a stable crystal” is incomplete without competing phases, chemical environment, and thermodynamic conditions.

## Stability is a comparison, not a score

Formation energy is the first useful comparison. Suppose a crystal contains $$n_a$$ atoms of element $$a$$, with $$N=\sum_a n_a$$. Given elemental reference energies $$\mu_a$$, its formation energy per atom is

$$\Delta E_{\mathrm{f}}(X)=\frac{E(X)-\sum_a n_a\mu_a}{N}.$$

A negative value means that the compound is lower in energy than its separated elements under the chosen reference calculation. It does not mean that the compound is the lowest-energy combination available at its composition. Other compounds may decompose it more favorably.

The convex hull supplies the missing comparison. Let $$\mathbf{x}^{(m)}$$ and $$\Delta E_{\mathrm{f}}^{(m)}$$ denote the composition vector and formation energy per atom of known or computed phase $$m$$. At target composition $$\mathbf{x}$$, the hull energy is

$$E_{\mathrm{hull}}(\mathbf{x})=\min_{\boldsymbol{\lambda}}\sum_m\lambda_m\Delta E_{\mathrm{f}}^{(m)}$$

subject to

$$\lambda_m\geq 0,\qquad \sum_m\lambda_m=1,\qquad \sum_m\lambda_m\mathbf{x}^{(m)}=\mathbf{x}.$$

The coefficients $$\lambda_m$$ describe a phase mixture with the same overall composition. The energy above hull of candidate $$X$$ is

$$\Delta E_{\mathrm{hull}}(X)=\Delta E_{\mathrm{f}}(X)-E_{\mathrm{hull}}(\mathbf{x}(X))\geq 0.$$

For the binary example in Figure 2, the compound AB lies on the hull at $$-0.40$$ eV per atom. A proposed AB₃ polymorph at composition $$x_{\mathrm{B}}=0.75$$ has formation energy $$-0.12$$ eV per atom. At the same composition, an equal mixture of AB and elemental B has energy $$-0.20$$ eV per atom. The candidate is therefore $$0.08$$ eV per atom above the hull even though its formation energy is negative.

{% include figure.liquid loading="eager" path="assets/img/blog/matdisc_convex_hull.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 2. A worked binary convex hull separates formation from decomposition stability. The AB₃ candidate has negative formation energy, but an equal mixture of AB and B is lower by 0.08 eV per atom, so the candidate lies above the 0 K hull." %}

This construction underlies large computed resources such as the Materials Project (<span id="cite-jain2013"></span>[Jain et al., 2013](#ref-jain2013)). It also exposes three boundaries around any claim of stability.

First, a hull is only as complete as its phase set. A newly discovered lower-energy competitor can move an older candidate off the hull. Merchant et al. explicitly observed this effect when expanding the candidate set with large-scale graph-network screening (<span id="cite-merchant2023"></span>[Merchant et al., 2023](#ref-merchant2023)). Second, the energies share the systematic and numerical errors of the chosen electronic-structure workflow. Comparing energies computed with incompatible reference schemes can invalidate the hull. Third, the usual computed hull describes a closed system near zero temperature. Open chemical reservoirs, pressure, and entropy require a different thermodynamic potential or additional free-energy terms.

An energy above hull is thus a decomposition driving force within a declared model, not a universal synthesizability probability. Empirically observed metastability spans chemistry-dependent energy scales; polymorphs and phase-separating compounds show different distributions (<span id="cite-sun2016"></span>[Sun et al., 2016](#ref-sun2016)). Kinetic trapping can preserve an above-hull phase, while a predicted ground state can remain inaccessible because nucleation and diffusion are too slow.

## The physical material exceeds the ideal cell

A relaxed, defect-free periodic cell is an intentionally narrow object. Real specimens may contain vacancies, interstitials, substitutions, dislocations, grain boundaries, surfaces, amorphous regions, and mixtures of polymorphs. Their concentrations depend on chemical potentials and processing history.

For a neutral defect, a simplified formation energy is

$$E_{\mathrm{def}}^{\mathrm{f}}=E_{\mathrm{def}}-E_{\mathrm{bulk}}-\sum_a\Delta n_a\mu_a,$$

where $$E_{\mathrm{def}}$$ and $$E_{\mathrm{bulk}}$$ are matched supercell energies and $$\Delta n_a$$ counts atoms added to the defective cell. Charged defects add Fermi-level and electrostatic finite-size terms. Even the neutral expression shows why a defect concentration is not an intrinsic number attached to a formula: it varies with the allowed chemical reservoirs $$\mu_a$$.

At finite temperature, a more useful solid free-energy approximation may include vibrational and configurational contributions,

$$G(T,P)\approx E_{\mathrm{DFT}}+F_{\mathrm{vib}}(T)-TS_{\mathrm{config}}+PV.$$

These terms can reorder polymorphs that are close in electronic energy. Thermal expansion can change a band structure. A small dopant concentration can dominate conductivity. Grain size can change strength. Figure 3 summarizes why a bulk-cell prediction and a measured material property are different claims.

{% include figure.liquid loading="eager" path="assets/img/blog/matdisc_material_reality.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 3. The ideal periodic cell is progressively modified by polymorphism, defects, finite-temperature effects, and processing conditions. Each layer can change phase identity and the measured electronic, mechanical, or transport response without changing the nominal reduced formula." %}

The distinction prevents two common overclaims. Predicting a low energy for one ordered structure does not establish that the material forms as a pure phase. Predicting a property for that structure does not establish that a processed sample retains the same property.

## Property prediction must respect periodic structure

A structure-aware predictor learns a map $$f_{\theta}:X\mapsto y$$ from a periodic crystal $$X$$ to a property $$y$$. Crystal graph neural networks represent atoms as nodes and connect each atom to neighbors, including periodic images. Xie and Grossman showed that message passing over such local environments can learn crystal properties from structure (<span id="cite-xie2018"></span>[Xie and Grossman, 2018](#ref-xie2018)). Modern variants add angular information, equivariant tensor features, or learned interatomic potentials.

The representation must preserve the equivalences of the crystal. For a scalar property,

$$f_{\theta}(X)=f_{\theta}(g\cdot X)$$

for every transformation $$g$$ that changes the description but not the physical crystal. These transformations include atom permutations, periodic translations, rigid rotations, and valid cell reparameterizations. A model that is invariant to atom order but sensitive to which equivalent unit cell happens to be stored in a file has learned a data convention, not a material property.

Fast predictors enable large screening funnels, but test error alone does not validate a discovery claim. Random train-test splits can place near-duplicate structures or related polymorphs on both sides. A model trained near relaxed minima may fail on the distorted structures encountered during relaxation. A band-gap model inherits the bias of its computational labels. Evaluation should therefore separate chemical systems, structural prototypes, or time periods according to the intended deployment.

Uncertainty is equally conditional. An ensemble disagreement score can flag inputs far from training data, but low disagreement does not reveal a bias shared by every model. Uncertainty should decide which candidate receives a more accurate calculation or experiment; it should not be treated as a certificate of correctness.

## Inverse design generates hypotheses

Forward prediction asks for $$p(y\mid X)$$: what property follows from this crystal? Inverse design seeks structures consistent with a target, which can be written schematically as

$$p(X\mid y^{\star})\propto p(y^{\star}\mid X)p(X).$$

The prior $$p(X)$$ matters. It encodes which compositions, lattices, symmetries, and local environments the generator regards as plausible. A conditional generator can sample element types, lattice parameters, and fractional coordinates, then steer samples toward a target property. MatterGen, for example, jointly denoises atom types, coordinates, and lattices and conditions generation on chemical, symmetry, and scalar-property constraints (<span id="cite-zeni2025"></span>[Zeni et al., 2025](#ref-zeni2025)).

The raw sample is the beginning of a screening funnel. A defensible workflow checks charge and composition constraints, relaxes the structure, removes duplicates, recomputes properties at higher fidelity, and rebuilds the convex hull with relevant competing phases. Many generated crystals move substantially during relaxation or collapse to the same minimum. Novelty before relaxation is not novelty after relaxation.

Multi-objective design makes the funnel narrower. A battery electrolyte may need ionic conductivity, electrochemical stability, mechanical compatibility, low electronic conductivity, available elements, and a viable processing window. These conditions rarely reduce to a single weighted score without value judgments. Pareto fronts keep the tradeoffs visible: a candidate is Pareto optimal when no other candidate improves one objective without worsening another.

The output of inverse design is therefore a ranked hypothesis with a computational audit trail. Calling a generated file a discovered material skips relaxation, phase competition, synthesis, and measurement.

## Synthesizability belongs to a process

Synthesis is better represented as a conditional outcome than as a label on the final structure. Let $$\mathbf{c}$$ describe precursors, stoichiometry, temperature schedule, pressure, atmosphere, mixing, and time. Then a synthesis model asks for

$$p(\text{phase},\text{yield},\text{impurities}\mid X,\mathbf{c}),$$

not merely whether $$X$$ is “synthesizable.” The same target can form from one precursor set and fail from another because stable intermediates block the pathway. Increasing temperature can accelerate diffusion but stabilize a competing phase or volatilize a precursor. Pressure can change both equilibrium and kinetics.

Published recipes provide incomplete supervision for this problem. Successful conditions are reported more often than failed attempts, and small procedural details may be absent. Raccuglia et al. showed that archived unsuccessful reactions contain useful boundaries on crystallization conditions (<span id="cite-raccuglia2016"></span>[Raccuglia et al., 2016](#ref-raccuglia2016)). Negative results are not merely zeros in a table: the observed impurity phase can reveal which branch of the reaction network the experiment followed.

Autonomous laboratories make the conditional nature of synthesis operational. The A-Lab combined computed phase stability, literature-derived synthesis heuristics, robotic solid-state synthesis, X-ray diffraction, and active selection of follow-up recipes (<span id="cite-szymanski2023"></span>[Szymanski et al., 2023](#ref-szymanski2023)). It synthesized 36 of 57 proposed targets during its autonomous campaign. The failures included slow kinetics, precursor volatility, amorphization, and computational error—different causes that a single synthesizability score would merge.

## Experimental closure changes the model

A discovery loop should choose the next experiment by both scientific value and information value. If candidate $$X$$ has predicted utility $$u(X)$$, uncertainty $$\sigma(X)$$, and experimental cost $$c(X)$$, a simple acquisition score is

$$a(X)=u(X)+\kappa\sigma(X)-\lambda c(X),$$

where $$\kappa$$ controls exploration and $$\lambda$$ penalizes cost. In practice, diversity constraints prevent the batch from spending its entire budget on near-duplicates. Feasibility constraints prevent the acquisition function from repeatedly proposing unavailable precursors or incompatible apparatus.

Figure 4 shows the full loop. Computation proposes and ranks structures. Synthesis tests a route under recorded conditions. Characterization identifies phases, impurities, yield, microstructure, and the target property. The outcome then updates the property model, the synthesis model, the uncertainty estimate, or all three.

{% include figure.liquid loading="eager" path="assets/img/blog/matdisc_closed_loop.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Figure 4. A closed materials-discovery loop connects candidate generation, high-fidelity computation, condition-specific synthesis, and characterization. Active learning uses successful and failed outcomes—including impurity phases and yield—to change which candidate or recipe is tested next." %}

Experimental validation must match the original claim. Powder X-ray diffraction can test bulk phase identity and estimate phase fractions, but it may not resolve a small defect population. Microscopy can reveal local defects but samples a limited region. A measured conductivity needs specimen geometry, temperature, density, and contact protocol. Agreement on one observable does not validate the entire predicted structure-property chain.

Prospective evaluation is the strongest test because the candidate and protocol are fixed before the outcome is known. Independent replication is stronger still. The proof-of-concept synthesis in MatterGen measured the target property of a generated material within 20% of the requested value, but the result validates one end-to-end case rather than a universal success rate. Claim boundaries should remain this narrow.

Materials discovery succeeds when a structure exists in more than a database. The composition must admit a periodic arrangement with the desired calculated response. That arrangement must compete favorably enough with other phases under relevant conditions. A process must form it, characterization must identify it, and measurement must confirm the property. Machine learning can accelerate every link, but no link substitutes for the next one.

---

## References

<a id="ref-jain2013"></a>**Jain, A., Ong, S. P., Hautier, G., et al.** (2013). [The Materials Project: A materials genome approach to accelerating materials innovation](https://doi.org/10.1063/1.4812323). *APL Materials*, 1, 011002. [↩](#cite-jain2013)

<a id="ref-xie2018"></a>**Xie, T. & Grossman, J. C.** (2018). [Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties](https://doi.org/10.1103/PhysRevLett.120.145301). *Physical Review Letters*, 120, 145301. [↩](#cite-xie2018)

<a id="ref-sun2016"></a>**Sun, W., Dacek, S. T., Ong, S. P., et al.** (2016). [The thermodynamic scale of inorganic crystalline metastability](https://doi.org/10.1126/sciadv.1600225). *Science Advances*, 2, e1600225. [↩](#cite-sun2016)

<a id="ref-raccuglia2016"></a>**Raccuglia, P., Elbert, K. C., Adler, P. D. F., et al.** (2016). [Machine-learning-assisted materials discovery using failed experiments](https://www.nature.com/articles/nature17439). *Nature*, 533, 73–76. [↩](#cite-raccuglia2016)

<a id="ref-merchant2023"></a>**Merchant, A., Batzner, S., Schoenholz, S. S., et al.** (2023). [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9). *Nature*, 624, 80–85. [↩](#cite-merchant2023)

<a id="ref-szymanski2023"></a>**Szymanski, N. J., Rendy, B., Fei, Y., et al.** (2023). [An autonomous laboratory for the accelerated synthesis of inorganic materials](https://www.nature.com/articles/s41586-023-06734-w). *Nature*, 624, 86–91. [↩](#cite-szymanski2023)

<a id="ref-zeni2025"></a>**Zeni, C., Pinsler, R., Zügner, D., et al.** (2025). [A generative model for inorganic materials design](https://www.nature.com/articles/s41586-025-08628-5). *Nature*, 639, 624–632. [↩](#cite-zeni2025)
