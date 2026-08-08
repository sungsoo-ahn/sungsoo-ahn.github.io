---
layout: post
title: "Generative Models for Protein Design"
date: 2026-08-08
last_updated: 2026-08-08
description: "Protein design as a sequence–structure–function inference problem: inverse folding, backbone diffusion, motif scaffolding, co-design, computational filters, and the experimental evidence that closes the loop."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [protein-science]
lecture_paths: [ml4mol, gdl]
tags: [protein-design, inverse-folding, protein-diffusion, motif-scaffolding, generative-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the protein-design storyline from my Machine Learning for Molecules and Geometric Deep Learning lectures. It treats design as a chain of conditional inference problems rather than a catalog of generators: the central question is which constraints survive generation and which claims remain for computation and experiment to test.</em>
</p>

## Protein Design Is Not Structure Prediction in Reverse

A structure predictor starts with a sequence and asks which three-dimensional arrangement is compatible with it. Protein design begins with a much less complete specification: bind this target, hold these catalytic residues in place, assemble with a chosen symmetry, or merely create a stable fold unlike known ones. There is usually no unique sequence and often no unique backbone that satisfies such a request. The task is therefore not to invert a deterministic map. It is to sample from a large set of molecular solutions while retaining the constraints that matter.

Let $$a=(a_1,\ldots,a_L)$$ denote an amino-acid sequence, $$X$$ a backbone or all-atom structure, and $$y$$ a desired property or function. Several distributions commonly called “protein design” are actually different problems:

$$
\begin{aligned}
\text{sequence generation:} &\qquad p_\theta(a\mid y),\\
\text{inverse folding:} &\qquad p_\theta(a\mid X),\\
\text{backbone generation:} &\qquad p_\theta(X\mid y),\\
\text{co-design:} &\qquad p_\theta(a,X\mid y).
\end{aligned}
$$

The distinction is operational. A sequence model conditioned on a family label must still establish that its sample folds. An inverse-folding model inherits a target backbone, but must find a sequence whose physical energy landscape actually favors that backbone. A backbone generator can satisfy an elegant geometric constraint while leaving unanswered whether any sequence realizes it. A co-design model couples the two variables, but coupling does not make function observable from training data.

{% include figure.liquid loading="eager" path="assets/img/blog/protdesign_variable_choices.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Four distributions that are often grouped under protein design. Each moves a different variable and therefore transfers a different unresolved question to the validation stage. A generated sequence needs folding evidence; a generated backbone needs a realizable sequence; a co-designed pair still needs evidence for the requested function." %}

This viewpoint connects naturally to [protein representation learning]({% post_url 2026-08-08-protein-representation-learning %}): sequence, residue graph, surface, and coordinate representations expose different constraints. It also clarifies why a highly capable [structure predictor]({% post_url 2026-08-08-protein-structure-prediction-alphafold %}) is valuable inside a design pipeline without itself becoming a design oracle. Prediction asks whether a proposed sequence is structurally plausible. Design asks how to search among proposals, and function is usually farther downstream than structure.

## Sequence Models Learn What Evolution Has Already Accepted

The largest source of protein data is sequence. An autoregressive model factors a distribution over amino acids as

$$
p_\theta(a\mid y)=\prod_{i=1}^{L}p_\theta(a_i\mid a_{<i},y),
$$

while a masked or discrete-diffusion model repeatedly reconstructs corrupted residues. In both cases, conditioning $$y$$ may encode a protein family, molecular function, organism, partial sequence, or textual annotation. ProGen demonstrated that a large conditional language model could generate sequences with controllable protein-family labels and that selected artificial lysozymes retained measurable activity despite substantial sequence divergence from natural proteins (<span id="cite-madani2023"></span>[Madani et al., 2023](#ref-madani2023)).

The strength of this formulation is also its limitation. Evolutionary sequence databases contain examples that survived folding, expression, cellular context, and historical selection. A language model compresses these regularities into a prior over plausible sequence space. But the database rarely contains controlled labels for a quantitative engineering objective such as “bind target $$T$$ with nanomolar affinity while avoiding off-target $$T'$$.” Annotation is incomplete, homologous families are unevenly sampled, and phylogeny produces many near-duplicates. A high model likelihood therefore means “resembles patterns accepted in the training distribution,” not “has the requested laboratory phenotype.”

One can formalize property guidance with Bayes' rule:

$$
p(a\mid y)\propto p(y\mid a)\,p(a).
$$

Here $$p(a)$$ is a generative prior and $$p(y\mid a)$$ is a fitness model, classifier, docking score, or another **oracle**. In log space, guided sampling often resembles optimizing

$$
\log p(a)+\lambda\log p(y\mid a),
$$

where $$\lambda$$ controls the pressure toward the requested property. Small $$\lambda$$ leaves the prior dominant; large $$\lambda$$ rewards whatever inputs exploit the oracle. The latter is a familiar failure of optimization under a learned surrogate: sequences can move far outside the oracle's training support and receive extreme predicted scores for accidental reasons. The generator has not discovered biology merely because it has found the classifier's blind spot.

This is especially consequential under epistasis. If the effect of mutating residue $$i$$ depends on residue $$j$$, a fitness surface cannot be reconstructed by adding single-mutation effects. A sequence generator can model such dependencies, but guidance based on a sparse assay may still be unreliable in novel combinations. Plausibility and task fitness are complementary signals, not interchangeable ones.

## Inverse Folding Conditions on the Answer's Geometry

Suppose a backbone $$X$$ is fixed. The inverse-folding problem asks for sequences compatible with that geometry. An autoregressive version uses

$$
p_\theta(a\mid X)=\prod_{i=1}^{L}p_\theta(a_{\pi_i}\mid a_{\pi_{<i}},X),
$$

where $$\pi$$ is a decoding order. A geometric encoder passes messages between residues that are close in 3D, allowing a residue's identity to depend on its spatial environment rather than only its neighbors along the chain. ProteinMPNN showed that this structure-conditioned formulation could design sequences with strong experimental recovery and folding behavior across monomers, oligomers, and protein–protein interfaces (<span id="cite-dauparas2022"></span>[Dauparas et al., 2022](#ref-dauparas2022)).

Consider a small helical bundle whose backbone is already specified. Residues buried in the core need mutually compatible size and hydrophobicity; exposed positions can often vary widely; a charged surface pair may be chosen to improve solubility without changing the fold. The backbone does not uniquely determine the sequence. It defines a conditional family of sequences, and sampling that family is useful precisely because multiple solutions can be screened for expression, stability, immunogenicity, or manufacturing constraints.

Yet inverse folding solves a compatibility problem, not the full thermodynamic problem. Training structures are typically single native conformations. The model learns which sequences occur on such backbones, but not necessarily the energy gap between the target and every competing fold, the rate of aggregation, or the effect of a cellular environment. Recovery of a native amino acid is also an imperfect metric: nature chose one residue under many historical constraints, whereas several alternatives may work. The more direct computational test is **self-consistency**—whether an independently predicted structure for the designed sequence returns to the target backbone. Even that test is evidence about structural agreement, not function.

## Backbone Generation Searches a Geometric Space

Inverse folding becomes more powerful when the backbone itself need not be borrowed from nature. A backbone generator represents a protein as residue coordinates or local rigid frames and learns a distribution over fold-like geometries. In a simplified coordinate diffusion process,

$$
X_t=\alpha_t X_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

and a denoising network estimates either $$X_0$$, the added noise, or the score $$\nabla_{X_t}\log p_t(X_t)$$. Sampling starts from noise and integrates the learned reverse process toward a structured backbone. Real systems must respect more than this schematic equation: global translation and rotation should not change probability, local frames live on rotation groups, chain connectivity produces correlated rather than independent geometry, and chirality must be preserved. These are architectural constraints, not cosmetic data augmentation.

RFdiffusion adapted a structure-prediction network to denoising and generated monomers, symmetric assemblies, binders, and scaffolds around functional motifs, with experimental tests spanning structure and function (<span id="cite-watson2023"></span>[Watson et al., 2023](#ref-watson2023)). Chroma likewise combined structured diffusion with sequence and side-chain design and showed conditional generation under symmetry, shape, substructure, and semantic constraints (<span id="cite-ingraham2023"></span>[Ingraham et al., 2023](#ref-ingraham2023)). The important conceptual advance is not that diffusion produces protein-shaped point clouds. It is that conditioning can turn incomplete geometric specifications into distributions over complete backbones.

Unconditional generation asks the model to invent the entire fold. Conditional generation supplies information that must survive the reverse process. For example, a symmetric oligomer may tie several subunits through group transformations. A binder design fixes the target structure and encourages a complementary interface. A topology condition constrains the coarse arrangement of helices and sheets. These requests differ in how hard the constraint is and in how much freedom remains for diversity.

## Motif Scaffolding Makes the Constraint Local and Explicit

Many functions depend on a small arrangement of atoms: catalytic residues surrounding a transition-state model, metal-coordinating side chains, or a binding epitope presented at a particular orientation. **Motif scaffolding** fixes this local geometry and generates the rest of the protein around it.

Let $$M$$ index motif residues and $$U$$ the residues to generate. The desired conditional distribution is

$$
p_\theta(X_U\mid X_M,c),
$$

where $$X_M$$ is held fixed and $$c$$ can include chain length, symmetry, target context, or secondary-structure preferences. During denoising, coordinates in $$U$$ are updated while the motif coordinates remain clamped or are tightly constrained. The generated scaffold must connect the motif into a coherent chain, bury or expose it appropriately, and create a fold that a sequence can stabilize.

{% include figure.liquid loading="lazy" path="assets/img/blog/protdesign_motif_scaffolding.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Motif scaffolding converts a local functional hypothesis into a conditional geometry problem. The motif remains fixed while the model denoises the surrounding residues. The resulting backbone must support both the local arrangement and a globally realizable fold; preserving three red points alone does not establish catalytic activity or binding affinity." %}

Imagine scaffolding a three-residue catalytic motif. The generator can place a compact fold around the residues while preserving their relative coordinates. Inverse folding can then assign a hydrophobic core and a polar pocket. A structure predictor may confirm that the resulting sequence returns to the designed geometry. Still, catalysis depends on protonation, solvent access, substrate pose, transition-state stabilization, and conformational motion. The scaffold has preserved a geometric hypothesis. It has not yet established the mechanism.

The distinction between a static design target and a distribution of functional conformations is developed in [Protein Ensembles and Learned Molecular Dynamics]({% post_url 2026-08-08-protein-ensembles-learned-dynamics %}). A motif that is perfectly arranged in one predicted structure can be too flexible, inaccessible, or populated too rarely in solution. Conversely, controlled flexibility may be required for binding or turnover. Static geometry is often the right starting constraint, but it is not the whole physical objective.

## Co-Design Couples Discrete Chemistry and Continuous Geometry

The conventional pipeline first generates a backbone and then applies inverse folding. This factorizes a joint model as

$$
p(a,X\mid y)=p(X\mid y)p(a\mid X,y).
$$

The factorization is attractive because each stage has a clear role and can use specialized training data. It also creates an interface failure: the backbone model may sample geometries in a region for which the sequence designer has little support. Repeating inverse folding with several sequences partly addresses this mismatch but does not change the backbone.

Joint or iterative co-design lets sequence and structure revise one another. One can alternate coordinate denoising with discrete residue updates, relax a continuous representation of sequence before discretization, or train a multimodal model over both. In principle this captures feedback: a glycine permits backbone angles that a bulky aromatic residue does not; a buried salt bridge changes which local packing arrangements are plausible. In practice the two spaces have different symmetries, noise processes, and error scales. Coordinates are continuous and equivariant under rigid motion; amino-acid identities are discrete and invariant. A single generation schedule can easily settle one variable before the other has enough information.

There is also a subtler issue. A joint model can make sequence and structure mutually consistent according to the same learned assumptions. This may improve internal scores without adding independent evidence. Coherence within a model is valuable, but it should not be mistaken for calibration against physical reality.

## A Design Pipeline Is a Sequence of Falsification Attempts

In practice, generation is the first stage of a funnel. A backbone generator proposes geometries. An inverse-folding model supplies several sequences per backbone. A structure predictor refolds each sequence. Filters remove clashes, weak interfaces, exposed hydrophobics, poor confidence, undesirable motifs, or manufacturability risks. A much smaller set is synthesized and assayed.

{% include figure.liquid loading="lazy" path="assets/img/blog/protdesign_design_loop.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The common backbone-to-sequence pipeline is best read as repeated attempted falsification. Self-consistent refolding can reject many proposals, but only synthesis and assays test expression, stability, binding, or activity in the relevant experimental setting. Negative results should update both generation and filtering." %}

For a designed sequence $$a$$ and target backbone $$X$$, a common structural score compares a predicted fold $$\widehat X(a)$$ with the target:

$$
s_{\mathrm{sc}}(a,X)
=\operatorname{sim}\!\left(\widehat X(a),X\right),
$$

using RMSD, TM-score, aligned error, or a combination. High self-consistency suggests **designability**: at least one sequence appears able to realize the generated backbone. It does not establish **novelty**, because a close database neighbor may already exist. It does not establish **diversity**, because thousands of samples may collapse to the same topology. And it does not establish **function**, because structure prediction is not an affinity or catalytic assay.

These axes must be reported separately:

- **geometric validity:** chain connectivity, bond geometry, clashes, chirality, and plausible secondary structure;
- **designability:** whether designed sequences independently refold to their intended backbones;
- **diversity:** how broadly samples cover sequence and structural space, rather than how many files were generated;
- **novelty:** distance to training and reference databases under sequence and structural search;
- **constraint satisfaction:** motif RMSD, interface geometry, symmetry, or other task-specific conditions;
- **developability:** expression, solubility, aggregation, immunogenicity, and manufacturability where relevant;
- **experimental function:** binding, activity, selectivity, kinetics, or cellular phenotype under a specified assay.

These objectives conflict. Lower-temperature sampling can raise average model likelihood while reducing diversity. Aggressive novelty thresholds can discard useful variations on known folds or push samples outside regions where predictors are calibrated. Tight motif constraints can reduce the number of globally foldable scaffolds. Selecting only the highest predicted affinity can enrich true binders while also magnifying oracle error. There is no scalar score that preserves every desirable property.

## The Experimental Denominator Matters

Suppose a model generates 10,000 backbones, 2,000 pass geometry filters, 300 obtain self-consistent sequences, 24 are selected by an affinity predictor, and 3 bind experimentally. “Three successful designs” is true. “A 12.5% hit rate” is also true for the selected set. Neither number estimates the probability that an unfiltered generator sample works, and neither reveals whether the affinity predictor improved selection unless appropriate controls were tested.

{% include figure.liquid loading="lazy" path="assets/img/blog/protdesign_evaluation_funnel.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A hypothetical design funnel. Each computational filter changes the evaluated population, so the final hit rate is conditional on the entire selection procedure. Reporting only the synthesized subset hides generator yield, filter selectivity, and whether the ranking oracle adds value." %}

This is **selection bias by construction**. Wet-lab budgets require down-selection, but the resulting observations are concentrated where the pipeline already predicts success. Failed generated candidates are rarely synthesized, so the data cannot distinguish a strong generator from a strong filter applied to a weak generator. Even among synthesized designs, unsuccessful expression may prevent the intended functional assay, creating another conditional denominator.

A persuasive validation therefore reports the full funnel: how many candidates were generated, every filtering threshold, how many independent backbones and sequence clusters survived, what fraction was synthesized, and which assays each construct reached. Random or stratified controls can estimate the value of ranking. Near-neighbor baselines test whether novelty contributes anything beyond known scaffolds. Negative outcomes constrain the model just as importantly as positive structures. Prospective campaigns are strongest when the design and selection protocol is fixed before seeing assay results.

Experimental evidence also has levels. Soluble expression is not proof of a monodisperse folded state. Circular dichroism reports secondary-structure content but not atomic accuracy. A crystal or cryo-EM structure can confirm the fold while leaving solution populations uncertain. Binding at one concentration does not establish affinity, specificity, or biological effect. The claim should stop where the assay stops.

## Design Closes Only When Information Returns from the Lab

Generative models have changed protein design by making structural proposals abundant. Inverse folding can populate a proposed backbone with compatible sequences. Diffusion can complete a scaffold around a functional motif. Co-design can reduce the mismatch between discrete chemistry and continuous geometry. Predictors and learned oracles can compress a vast candidate set into an experimentally manageable one.

But abundance of proposals changes the bottleneck rather than removing it. The difficult question becomes which constraints deserve to enter generation, which independent models can reject failures, which tradeoffs should remain visible, and which experiments discriminate between a plausible molecular picture and the requested function. A successful protein-design system is therefore not a generator in isolation. It is a calibrated loop in which sequence, structure, and function remain distinct claims—and experimental outcomes return information to every stage that made them.

---

## References

<span id="ref-madani2023"></span>Madani, A. et al. “Large language models generate functional protein sequences across diverse families.” *Nature Biotechnology* 41, 1099–1106 (2023). [doi:10.1038/s41587-022-01618-2](https://doi.org/10.1038/s41587-022-01618-2) [↩](#cite-madani2023)

<span id="ref-dauparas2022"></span>Dauparas, J. et al. “Robust deep learning-based protein sequence design using ProteinMPNN.” *Science* 378, 49–56 (2022). [doi:10.1126/science.add2187](https://doi.org/10.1126/science.add2187) [↩](#cite-dauparas2022)

<span id="ref-watson2023"></span>Watson, J. L. et al. “De novo design of protein structure and function with RFdiffusion.” *Nature* 620, 1089–1100 (2023). [doi:10.1038/s41586-023-06415-8](https://doi.org/10.1038/s41586-023-06415-8) [↩](#cite-watson2023)

<span id="ref-ingraham2023"></span>Ingraham, J. B. et al. “Illuminating protein space with a programmable generative model.” *Nature* 623, 1070–1078 (2023). [doi:10.1038/s41586-023-06728-8](https://doi.org/10.1038/s41586-023-06728-8) [↩](#cite-ingraham2023)

---

*Figure provenance: all four diagrams are original explanatory syntheses created for this post with `scripts/generate_protdesign_figures.py`. They use no copied slide, paper, or Flaticon assets and are released under CC BY 4.0 with the post.*
