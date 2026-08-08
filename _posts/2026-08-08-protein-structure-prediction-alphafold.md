---
layout: post
title: "Protein Structure Prediction from AlphaFold 1 to 3"
date: 2026-08-08
last_updated: 2026-08-08
description: "How coevolutionary constraints, pairwise geometric reasoning, residue frames, and all-atom diffusion shaped AlphaFold—and where structure prediction stops."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [protein-science]
lecture_paths: [ml4mol, gdl]
tags: [protein-structure-prediction, alphafold, multiple-sequence-alignment, evoformer, diffusion-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post follows the changing representation of the structure problem across AlphaFold 1, 2, and 3. The goal is not a chronology of model releases, but an account of how evolutionary constraints become pair geometry, how geometry becomes coordinates, and how far those coordinates support biological conclusions.</em>
</p>

## Structure Prediction Is Inference Under Constraints

A protein sequence fixes a covalent backbone, but it does not explicitly list the contacts that fold distant residues together. For a chain of length $$L$$, there are $$O(L^2)$$ residue pairs, many possible torsion-angle combinations, and a continuous three-dimensional space of coordinates. The prediction problem is therefore not simply “map one string to one point cloud.” It is to infer a mutually consistent set of local and long-range geometric constraints, then realize them as an all-atom structure.

AlphaFold's three generations changed the second half of that sentence more than the first. AlphaFold 1 predicted pairwise distance distributions and converted them into a potential optimized by an external structure-building procedure. AlphaFold 2 maintained sequence-family and residue-pair representations, reasoned repeatedly about their consistency, and directly constructed coordinates through residue-local frames. AlphaFold 3 retained pair reasoning but replaced the protein-specific coordinate decoder with diffusion over raw atom coordinates, allowing proteins, nucleic acids, ligands, ions, and modified residues to occupy one structural output space.

The common resource is information about which residues should be compatible in three dimensions. Part of that information comes from local chemistry and structural templates. A particularly powerful part comes from evolution.

## An MSA Records Many Natural Perturbation Experiments

A multiple-sequence alignment (MSA) arranges homologous protein sequences so that each column corresponds, approximately, to the same ancestral position. Conserved columns identify positions under strong selection. Correlated columns can reveal a different signal: two positions change together because some combinations preserve structure or function better than others.

Imagine that position $$i$$ is positively charged and position $$j$$ is negatively charged in one branch of a protein family. If mutations replace them together—lysine/glutamate becomes arginine/aspartate—the pair may preserve a favorable interaction. If only one side changed, the interaction might become unfavorable. Across a sufficiently deep and diverse MSA, this compensatory pattern is evidence that $$i$$ and $$j$$ are coupled.

Raw correlation is not enough. If residue $$i$$ contacts $$k$$ and $$k$$ contacts $$j$$, then $$i$$ and $$j$$ may correlate indirectly. Phylogeny also makes related sequences statistically dependent. Direct-coupling methods model the aligned family with a Potts distribution

$$
p(\mathbf{a})
\propto
\exp\!\left[
\sum_i h_i(a_i)+\sum_{i<j}J_{ij}(a_i,a_j)
\right],
$$

where $$a_i$$ is the amino-acid identity at position $$i$$, $$h_i$$ captures single-position preferences, and $$J_{ij}$$ captures direct pair compatibility after accounting for the other positions. Strong couplings often indicate spatial proximity. Early work showed that these evolutionary constraints could be sufficient to construct three-dimensional folds for families with rich sequence data (<span id="cite-marks2011"></span>[Marks et al., 2011](#ref-marks2011)).

{% include figure.liquid loading="eager" path="assets/img/blog/afstruct_msa_coevolution.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A multiple-sequence alignment records residue substitutions accepted across a protein family. When two distant columns change in a coupled manner, their compatibility provides evidence that the corresponding residues interact in the folded structure; coevolution is a statistical constraint, not a literal physical force." %}

The caveat “rich sequence data” matters. A shallow MSA cannot reveal reliable covariance. A deep but redundant MSA can overstate one evolutionary branch. Homologs may share a fold while adopting different oligomeric states or functional conformations. Coevolution compresses family history into structural evidence, but it does not identify which experimental condition or dynamical state produced a particular structure.

## AlphaFold 1 Learned a Geometric Potential

AlphaFold 1 made an important shift from binary contact prediction to a **distogram**: for each residue pair $$(i,j)$$, the network predicted a categorical distribution over distance bins,

$$
p_{ij}^{(b)}
=P\!\left(d_{ij}\in[r_b,r_{b+1})\mid\text{sequence, MSA, templates}\right).
$$

A contact map says only whether $$d_{ij}$$ is below a threshold. A distogram distinguishes a tight contact from medium-range separation and represents uncertainty across several plausible distances. Pairwise features derived from the MSA and templates were processed by a deep residual network, producing distance and torsion distributions.

The predicted distributions did not directly place atoms. They became a learned potential, schematically

$$
E(\mathbf{x})
=-\sum_{i<j}\log p_{ij}\!\left(d_{ij}(\mathbf{x})\right)
+E_{\mathrm{torsion}}(\mathbf{x})
+E_{\mathrm{stereo}}(\mathbf{x}),
$$

and coordinates were found by numerical optimization. The first term rewards structures whose pair distances fall in likely bins; torsional and stereochemical terms discourage locally impossible geometry and steric clashes. AlphaFold 1 therefore separated inference from realization: a neural network inferred geometric restraints, then a solver reconciled them in 3D. This neural-potential strategy was the core of the CASP13 system (<span id="cite-senior2020"></span>[Senior et al., 2020](#ref-senior2020)).

This division is interpretable but brittle. Pairwise distances need not be globally realizable. Torsion errors accumulate along a chain. Optimization can settle in a poor basin even when many individual restraints are correct. The central advance of AlphaFold 2 was to let coordinate construction participate in learned reasoning rather than appear only after it.

## The Evoformer Makes Pair Geometry a Persistent State

AlphaFold 2 carries two main internal representations (<span id="cite-jumper2021"></span>[Jumper et al., 2021](#ref-jumper2021)):

- An MSA representation $$m_{si}$$ stores features for sequence $$s$$ at residue position $$i$$.

- A pair representation $$z_{ij}$$ stores the model's evolving belief about the geometric relationship between residues $$i$$ and $$j$$.

The Evoformer alternates updates within and between these tracks. MSA attention asks which homologs and positions support a residue pattern. An outer-product mean transfers family-level correlations into pair space; schematically,

$$
z_{ij}
\leftarrow z_{ij}
+\frac{1}{N_{\mathrm{seq}}}
\sum_s
\mathbf{a}(m_{si})\otimes\mathbf{b}(m_{sj}).
$$

The outer product preserves interactions between channels at positions $$i$$ and $$j$$ rather than collapsing each column to a single covariance number. Learned projections then turn this family statistic into pair features.

Pair geometry must also be globally consistent. If residues $$i$$ and $$j$$ have a proposed relationship, every third residue $$k$$ offers an indirect path $$i\rightarrow k\rightarrow j$$ against which that proposal can be checked. A triangle multiplicative update has the schematic form

$$
z_{ij}
\leftarrow z_{ij}
+\sum_k
\mathbf{a}(z_{ik})\odot\mathbf{b}(z_{kj}),
$$

with learned gates and projections omitted. Triangle attention performs a related comparison with data-dependent weights. The mechanism resembles geometric consistency: three edges form a triangle, so evidence on two sides constrains the third. It is learned algebra over pair features, not a hard Euclidean triangle solver, but the inductive bias is exactly the right one.

{% include figure.liquid loading="eager" path="assets/img/blog/afstruct_pair_reasoning.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="AlphaFold 2 maintains an MSA representation and a residue-pair representation, repeatedly exchanging information between them. Triangle updates test each pair through alternative two-edge paths, the structure module realizes the resulting geometry in coordinates, and recycling feeds that provisional structure back for revision." %}

The pair representation is therefore more than a distance matrix. It becomes a workspace for orientation, compatibility, template evidence, and long-range constraint propagation. This is why reducing AlphaFold 2 to “a Transformer on an MSA” misses the architectural point: the model devotes comparable effort to reasoning in the complete graph of residue pairs.

## Local Frames Turn Pair Reasoning Into Equivariant Coordinates

The AlphaFold 2 structure module represents each residue backbone as a rigid frame

$$
T_i=(R_i,\mathbf{t}_i),
$$

where $$R_i\in SO(3)$$ is an orientation and $$\mathbf{t}_i\in\mathbb{R}^3$$ is a translation. Side-chain torsions place atoms relative to that frame. The model iteratively updates these frames using single-residue and pair features.

Invariant point attention attaches learned query and key points to local frames. A query point $$\mathbf{q}_{ih}$$ in residue $$i$$'s local coordinates becomes $$R_i\mathbf{q}_{ih}+\mathbf{t}_i$$ globally; a key point on residue $$j$$ is transformed similarly. Their squared distance,

$$
\left\|
R_i\mathbf{q}_{ih}+\mathbf{t}_i
-R_j\mathbf{k}_{jh}-\mathbf{t}_j
\right\|^2,
$$

is unchanged if the whole protein is rotated and translated. Attention can therefore depend on 3D geometry without choosing a privileged laboratory frame. When the input frames undergo a global rigid transformation, the output frames and atoms transform with them: the coordinate construction is equivariant.

The training loss uses the same local-frame logic. Frame-aligned point error compares a predicted atom $$\widehat{\mathbf{x}}_j$$ and true atom $$\mathbf{x}_j$$ as seen from residue $$i$$:

$$
\operatorname{FAPE}_{ij}
=\left\|
\widehat{T}_i^{-1}\widehat{\mathbf{x}}_j
-T_i^{-1}\mathbf{x}_j
\right\|.
$$

A global rotation or translation cancels inside both inverse frames. The loss rewards correct relative geometry and does not require a separate global alignment. Because every residue provides a local viewpoint, it strongly constrains both local side-chain packing and long-range placement.

Recycling closes the loop. Predicted coordinates are converted back into pair and single features, then passed through the network again. The model can revise an MSA-derived hypothesis after seeing the geometry it produced. AlphaFold 2 is thus not a one-pass decoder: it alternates constraint inference and coordinate realization inside the learned system.

## Confidence Is Part of the Prediction

A single coordinate file can look authoritative even where the model has little evidence. AlphaFold counters this with confidence estimates at different scales.

**pLDDT** is a per-residue prediction of local structural accuracy. High pLDDT supports interpreting the local backbone and, at the highest values, often side-chain placement. Low pLDDT is deliberately ambiguous: it may indicate insufficient evolutionary information, an intrinsically disordered region, a flexible linker, an unusual state, or ordinary model error. Applying AlphaFold 2 across the human proteome demonstrated that pLDDT was calibrated against held-out experimental structures and made residue-level confidence essential to large-scale use (<span id="cite-tunyasuvunakool2021"></span>[Tunyasuvunakool et al., 2021](#ref-tunyasuvunakool2021)).

**Predicted aligned error (PAE)** is pairwise. PAE$$(i,j)$$ estimates the position error at residue $$i$$ when prediction and reference are aligned on residue $$j$$. A two-domain protein can have high pLDDT within both domains but high cross-domain PAE: each domain is locally convincing, while their relative orientation is not. Collapsing these signals into one global score discards the distinction a structural biologist needs.

{% include figure.liquid loading="eager" path="assets/img/blog/afstruct_confidence_scope.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="pLDDT estimates local residue accuracy, whereas PAE asks whether residues, domains, or chains are placed reliably relative to one another. Neither score establishes state populations, kinetics, binding free energies, or biological function; those claims require additional models and experiments." %}

Confidence is model- and task-specific. A high score means the model expects its coordinates to resemble a relevant experimental structure under its training distribution. It is not a posterior probability that the protein always adopts this conformation, nor a certificate that an interface forms in the cell.

## AlphaFold 3 Generalizes the Coordinate Space

AlphaFold 2's residue frames and torsion hierarchy are elegant for proteins, but less natural for a complex containing RNA bases, a metal ion, a covalently modified residue, and a flexible small molecule. AlphaFold 3 changes the output representation: its diffusion module operates on raw atom coordinates for a heterogeneous complex (<span id="cite-abramson2024"></span>[Abramson et al., 2024](#ref-abramson2024)).

The trunk still builds single-token and token-pair representations. Polymer residues and ligand atoms or groups become tokens; atom-level features retain the fine chemical detail. A simplified Pairformer performs pair and single updates while omitting the persistent MSA track of the AlphaFold 2 Evoformer. The resulting conditioning is passed to an all-atom denoiser.

For a clean complex with coordinates $$\mathbf{x}_0$$, diffusion training constructs noisy coordinates

$$
\mathbf{x}_\sigma
=\mathbf{x}_0+\sigma\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

after randomizing global rotation and translation. The network predicts a less noisy structure conditioned on sequence, chemistry, MSA-derived information, templates, and pair features. Sampling starts from coordinate noise and repeatedly denoises toward a joint complex.

Unlike the frame-based AlphaFold 2 structure module, the published AlphaFold 3 diffusion module does not use rotational frames or explicitly equivariant processing. Random rigid augmentation teaches the desired behavior statistically. This simplifies the decoder and avoids hand-designing a coordinate hierarchy for every molecular type, but it exchanges exact architectural symmetry for learned symmetry and extensive augmentation.

{% include figure.liquid loading="eager" path="assets/img/blog/afstruct_decoder_shift.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="AlphaFold 1 converts distograms into a potential optimized by an external solver; AlphaFold 2 directly refines protein-specific residue frames; AlphaFold 3 denoises raw atom coordinates for heterogeneous complexes. The pairwise geometric problem persists, while the coordinate decoder becomes progressively more integrated and general." %}

Diffusion also changes the interpretation of multiple outputs. Stochastic sampling can expose alternative coordinate hypotheses and uncertainty. It does not automatically produce a thermodynamic ensemble. A Boltzmann ensemble requires correct relative state probabilities under specified conditions; a kinetic model requires transition rates or time-resolved dynamics. A denoiser trained to reproduce deposited structures is optimized for structural accuracy, not those quantities.

## What a Predicted Structure Does—and Does Not—Establish

A confident structure is a powerful hypothesis. It can suggest domain boundaries, catalytic residues, mutation sites, binding pockets, and plausible interfaces. AlphaFold 3 extends this reach by predicting joint complexes, where ligand and partner geometry can be reasoned about together. But coordinates alone do not close the causal chain from sequence to function.

### Dynamics and state populations

Proteins fluctuate within basins and transition between conformations. Transporters alternate inward- and outward-facing states; kinases switch regulatory conformations; disordered regions may never adopt one dominant fold. A default AlphaFold 2 run typically concentrates on one structurally plausible state. MSA subsampling can encourage alternative conformations for some transporters and receptors, as demonstrated experimentally in a targeted protocol (<span id="cite-delalamo2022"></span>[del Alamo et al., 2022](#ref-delalamo2022)). Those samples are useful hypotheses, but their frequency is not an equilibrium population, and model iteration is not physical time.

### Binding geometry and affinity

A predicted complex can answer “how might these molecules fit?” It does not by itself answer “how tightly do they bind?” Binding free energy includes solvent reorganization, entropy, protonation, competing states, and concentration conventions. The thermodynamic relation

$$
\Delta G^\circ = RT\log K_d
$$

contains information absent from one pose. A geometrically credible protein–ligand complex can still have weak affinity, and a wrong protonation state can reverse an interaction. Pose confidence should guide docking analysis and experiments, not substitute for affinity measurement.

### Structure and function

Structural similarity supports functional hypotheses, but identical folds can support different substrates, regulation, or cellular roles. Catalysis depends on electronic structure and transition states, not just the ground-state arrangement of heavy atoms. Function also depends on expression, localization, partners, modifications, and environmental conditions. A predicted active-site geometry is a reason to run a biochemical assay; it is not the assay result.

These limitations do not diminish structure prediction. They locate it correctly. AlphaFold transformed a scarce experimental input into an abundant computational hypothesis, and confidence scores make those hypotheses unusually actionable. The next scientific step is chosen by the claim: molecular dynamics or ensemble models for conformational populations, free-energy methods and assays for binding, mutagenesis and cellular experiments for mechanism and function.

## The Representation Is the Breakthrough

The durable lesson across AlphaFold 1–3 is not that one neural architecture replaced another. Each generation chose a better intermediate object for the uncertainty it had to resolve.

AlphaFold 1 learned distributions over pair geometry rather than forcing an early coordinate guess. AlphaFold 2 made the pair representation a persistent reasoning space and let local frames connect that space to equivariant coordinates. AlphaFold 3 kept learned pair constraints but broadened the coordinate language to all atoms in mixed biomolecular complexes through diffusion.

MSAs, pair matrices, residue frames, and noisy atom clouds are not interchangeable encodings. Each exposes some constraints and hides others. Their success comes from matching representation to inference: evolution constrains residue compatibility, triangles constrain global geometry, frames remove arbitrary global pose, and diffusion accommodates heterogeneous chemistry. Confidence then marks where the inferred structure is stable enough to use—and where biological interpretation must wait for more evidence.

---

## References

<span id="ref-marks2011"></span>Marks, D. S., Colwell, L. J., Sheridan, R., Hopf, T. A., Pagnani, A., Zecchina, R., & Sander, C. (2011). [Protein 3D Structure Computed from Evolutionary Sequence Variation](https://doi.org/10.1371/journal.pone.0028766). *PLOS ONE, 6*(12), e28766. [↩](#cite-marks2011)

<span id="ref-senior2020"></span>Senior, A. W., Evans, R., Jumper, J., et al. (2020). [Improved Protein Structure Prediction Using Potentials from Deep Learning](https://www.nature.com/articles/s41586-019-1923-7). *Nature, 577*, 706–710. [↩](#cite-senior2020)

<span id="ref-jumper2021"></span>Jumper, J., Evans, R., Pritzel, A., et al. (2021). [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2). *Nature, 596*, 583–589. [↩](#cite-jumper2021)

<span id="ref-tunyasuvunakool2021"></span>Tunyasuvunakool, K., Adler, J., Wu, Z., et al. (2021). [Highly Accurate Protein Structure Prediction for the Human Proteome](https://www.nature.com/articles/s41586-021-03828-1). *Nature, 596*, 590–596. [↩](#cite-tunyasuvunakool2021)

<span id="ref-abramson2024"></span>Abramson, J., Adler, J., Dunger, J., et al. (2024). [Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w). *Nature, 630*, 493–500. [↩](#cite-abramson2024)

<span id="ref-delalamo2022"></span>del Alamo, D., Sala, D., Mchaourab, H. S., & Meiler, J. (2022). [Sampling Alternative Conformational States of Transporters and Receptors with AlphaFold2](https://elifesciences.org/articles/75751). *eLife, 11*, e75751. [↩](#cite-delalamo2022)
