---
layout: post
title: "Molecular Data and Property Prediction Across 1D, 2D, and 3D"
date: 2026-08-08
last_updated: 2026-08-08
description: "How molecular representations, conformers, data splits, pretraining, and uncertainty determine what a property-prediction benchmark actually measures."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [molecular-science]
lecture_paths: [ml4mol]
tags: [molecular-property-prediction, molecular-representations, molecular-graphs, conformers, uncertainty-quantification]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the molecular property-prediction storyline from
  my Machine Learning for Molecules lecture. It treats representation, data
  splitting, and evaluation as parts of the scientific claim—not as plumbing
  around a neural architecture.</em>
</p>

A molecule can be written as a string, drawn as a graph, or placed in three-dimensional space. These are not three file formats for the same information. Each makes some molecular distinctions explicit, hides others, and introduces its own nuisance variation.

That observation changes how a property-prediction result should be read. A sequence model may exploit a huge unlabeled corpus but spend capacity learning notation. A graph model receives connectivity directly but cannot infer which conformation was measured. A geometric model sees coordinates, yet those coordinates may describe only one arbitrary member of a thermal ensemble. More information can help in principle; in finite data, mismatched information can become a shortcut or a source of variance.

The right starting question is therefore not “Which molecular model is best?” It is: **What physical object generated the label, what part of that object is observed, and what kind of future molecule should the evaluation imitate?**

## A representation defines the prediction problem

Let $$M$$ denote the complete molecular state relevant to a measurement and let $$R=r(M)$$ be the representation supplied to a model. For squared-error prediction, the optimal predictor based on $$R$$ is

$$
f^\star(R)=\mathbb{E}[Y\mid R].
$$

Its irreducible risk is

$$
\mathbb{E}\left[(Y-f^\star(R))^2\right]
=
\mathbb{E}\left[\operatorname{Var}(Y\mid R)\right].
$$

This equation is the cleanest way to think about molecular representation. If two physically different states collapse to the same $$R$$ but have different labels, no architecture can remove the resulting conditional variance. Conversely, adding coordinates reduces ambiguity only when those coordinates are relevant and reliable.

{% include figure.liquid loading="eager" path="assets/img/blog/molprop_representation_stack.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A string exposes sequence context, a graph exposes connectivity, and a conformer exposes one three-dimensional arrangement. Moving right adds geometric information, but also adds sensitivity to notation choices, conformer generation, and the physical state represented by the coordinates. Original diagram." %}

### 1D: strings and sequences

SMILES linearizes a molecular graph into tokens (<span id="cite-weininger1988"></span>[Weininger, 1988](#ref-weininger1988)). It is compact, ubiquitous, and compatible with the machinery developed for language models. Long-range token dependencies can encode ring closures, branches, stereochemical marks, and recurring chemical fragments.

But SMILES order is not molecular order. The same graph can have many valid traversals and therefore many valid strings. A canonicalizer chooses one, while randomized SMILES presents several equivalent views during training. Canonicalization reduces duplication but creates arbitrary sequential regularities; augmentation encourages invariance but increases training cost. A token model must learn both chemistry and the grammar used to serialize it.

Strings are attractive when data scale and metadata dominate geometry: broad bioactivity corpora, reaction records, patents, assay text, and multi-task pretraining. They are less natural when the label depends sensitively on a particular pose or local force. A valid string also does not guarantee a stable three-dimensional molecule under the conditions of interest.

### 2D: graphs and fingerprints

A molecular graph represents atoms as nodes and bonds as edges. Node attributes can include element, formal charge, aromaticity, chirality, and hybridization; edge attributes can include bond order and stereochemistry. Message passing respects atom indexing by construction, and a permutation-invariant readout produces a molecular prediction.

The graph makes connectivity explicit, but it still omits a unique geometry. Cis/trans and tetrahedral stereochemistry can be stored as discrete labels, yet a graph does not specify a protein-bound pose, solvent-dependent conformation, or a distribution of torsional states.

Learned graph representations should also be compared with strong fixed descriptors. Extended-connectivity fingerprints record hashed circular atom environments and remain remarkably competitive on many small molecular datasets (<span id="cite-rogers2010"></span>[Rogers & Hahn, 2010](#ref-rogers2010)). An ECFP model is fast, stable, and difficult to beat when a target is largely determined by local substructures and labeled data are scarce. Beating a weak neural baseline while omitting fingerprints establishes little.

### 3D: conformers and atomic environments

A geometric representation supplies atomic coordinates $$X=(x_1,\ldots,x_N)$$ in addition to atom types and connectivity. Distances, angles, and torsions become observable. An invariant model predicts a scalar property such as energy or polarizability; an equivariant model can predict vectors such as forces or dipoles. The companion post on [scalar and vector geometric graph networks]({% post_url 2026-08-08-scalar-vector-geometric-gnns %}) develops this architectural distinction.

The scientific meaning of “the coordinates” depends on the dataset. QM9 provides optimized geometries and quantum-chemical properties for roughly 134,000 small organic molecules (<span id="cite-ramakrishnan2014"></span>[Ramakrishnan et al., 2014](#ref-ramakrishnan2014)). Its labels describe a specified electronic-structure calculation near an equilibrium structure. A solubility or bioactivity measurement, in contrast, reflects temperature, solvent, protonation, experimental protocol, and often many conformers. Giving such a task one inexpensive generated conformer does not turn it into a clean 3D problem.

## The same molecule may have many relevant conformers

Suppose a molecular graph has conformers $$X_1,\ldots,X_C$$ with free energies $$G_1,\ldots,G_C$$. At inverse temperature $$\beta$$, an idealized population is

$$
\pi_c
=
\frac{\exp(-\beta G_c)}
{\sum_{k=1}^{C}\exp(-\beta G_k)}.
$$

For an observable that averages over interconverting states,

$$
\overline{Y}
\approx
\sum_{c=1}^{C}\pi_c\,Y(X_c).
$$

The equation exposes two uncertainties. The conformer set may be incomplete, and the weights may be inaccurate because the environment and free energies are approximate. Uniformly averaging conformer embeddings silently assumes equal populations; choosing only the lowest-energy conformer assumes the observable is dominated by that state.

{% include figure.liquid loading="lazy" path="assets/img/blog/molprop_conformer_uncertainty.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="One bond graph can generate several three-dimensional conformers with different populations and property values. A single-conformer input collapses this ensemble uncertainty, while an ensemble model must choose an aggregation rule consistent with the measured observable. Original diagram." %}

The correct aggregation is task-dependent. A gas-phase quantum property may be evaluated at one optimized geometry. A room-temperature spectrum averages an ensemble. Binding affinity depends on ligand and receptor conformations, protonation, solvent, and the thermodynamic path—not merely on an isolated ligand conformer. A force-field dataset such as MD17 changes the unit of prediction again: each geometry is a separate state with its own energy and forces, not another noisy row for the same molecule.

Conformer generation can still help a 2D deployment setting. A model can pool several predicted conformers, marginalize their predictions, or use 3D data only during pretraining. 3D Infomax, for example, transfers geometric information into a graph encoder that does not require coordinates at inference (<span id="cite-stark2022"></span>[Stärk et al., 2022](#ref-stark2022)). The important point is to report whether test-time geometry is experimental, computed, generated, or absent.

## Task regimes imply different representations

| Task | Label-generating object | Natural input | Central ambiguity |
|---|---|---|---|
| Aqueous solubility | Molecule in solvent under an assay protocol | 1D/2D plus conditions | Ionization, solid state, and measurement noise |
| Bioactivity classification | Molecule–target–assay interaction | 2D molecule plus target/assay context | Dataset bias, inactive definition, and target shift |
| QM9 energy or orbital property | Optimized gas-phase geometry and computational method | 3D conformer | Limited elements, size, and equilibrium regime |
| Conformer energy ranking | Several states of one molecular graph | 3D conformer ensemble | Missing conformers and energy accuracy |
| Protein–ligand affinity | Bound complex and thermodynamic environment | 3D complex plus conditions | Pose, receptor flexibility, protonation, and entropy |
| Atomic forces | One instantaneous atomic configuration | 3D coordinates | Coverage of off-equilibrium configurations |

The table also explains why a single benchmark suite cannot identify a universally superior molecular representation. MoleculeNet was valuable because it standardized diverse datasets, metrics, featurizations, and splits (<span id="cite-wu2018"></span>[Wu et al., 2018](#ref-wu2018)). Its tasks deliberately span different physical meanings. Averaging ranks across them may summarize engineering breadth, but it does not create one coherent scientific target.

## A split defines the generalization claim

A random row split asks whether the model can interpolate among samples drawn from the same mixture. In chemical datasets, this often places close analogues on both sides of the boundary. If a medicinal-chemistry series differs only by one substituent, the test set may be chemically familiar even when its exact molecules are unseen.

A scaffold split groups molecules by a core scaffold and assigns entire groups to one partition. It is a rougher proxy for discovering new chemical families. A temporal split trains on measurements available before a cutoff and tests on later measurements; when timestamps reflect an actual discovery campaign, this is often closest to prospective deployment. Target-family, laboratory, or acquisition-source splits may be more appropriate when those shifts dominate.

{% include figure.liquid loading="lazy" path="assets/img/blog/molprop_split_leakage.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Random splits can place close scaffold analogues in both training and test sets, measuring interpolation within a chemical series. Scaffold or temporal splits preserve larger groups and better approximate some prospective uses, but only after duplicate molecules, conformers, and repeated measurements are resolved. Original diagram." %}

No split is intrinsically honest. It is honest only relative to a deployment claim. Random splits are reasonable when future samples will come from the same enumerated library. Scaffold splits can be unnecessarily pessimistic when optimization deliberately explores close analogues. Temporal splits can be confounded by changes in assay protocol.

Before splitting, the unit of independence must be defined. Common leakage routes include:

- the same molecule under different SMILES strings;
- salts, tautomers, stereoisomers, or protonation states inconsistently standardized;
- multiple conformers of one molecule divided across partitions;
- repeated assay measurements or near-duplicate records;
- pretraining data that contain downstream test molecules or labels;
- protein–ligand complexes sharing nearly identical proteins and ligand series.

Wallach and Heifets showed that redundancy in ligand benchmarks can reward memorization rather than prospective generalization (<span id="cite-wallach2018"></span>[Wallach & Heifets, 2018](#ref-wallach2018)). The lesson is broader than virtual screening: similarity across the split must be measured, not assumed from row counts.

## Pretraining helps when its invariances match deployment

Molecular pretraining can exploit far more unlabeled structures than any one property dataset. The objective, however, decides what transfers.

Masked atom or bond prediction teaches local chemistry but may be solved through easy neighborhood statistics. Graph-level multi-task supervision teaches correlations across assays but inherits their missingness and bias. Contrastive objectives can align randomized strings, graph augmentations, conformers, images, spectra, or assay descriptions. Geometry-aware pretraining can inject 3D correlations into a 2D encoder.

Hu et al. showed that combining node-level and graph-level objectives can improve molecular graph transfer (<span id="cite-hu2020"></span>[Hu et al., 2020](#ref-hu2020)). The durable principle is not that pretraining always helps. It is that pretraining should expose variations the downstream model must ignore and distinctions it must preserve.

For example, two randomized SMILES of the same molecule are useful positive pairs because traversal order is a nuisance. Two low-energy conformers may be positives for a graph-level identity task, but forcing their embeddings to coincide is questionable if the downstream label is conformer energy. A 3D objective trained only on equilibrium structures may transfer poorly to transition states or strained poses.

Pretraining comparisons should therefore control at least four quantities: architecture, number of unique molecules, label access, and domain overlap. A larger pretrained model may win because it saw more chemistry or even test-set analogues, not because its objective discovered a more general representation.

## Evaluation needs more than one aggregate metric

Regression benchmarks usually report MAE or RMSE. MAE describes a typical absolute miss; RMSE emphasizes large errors. Classification benchmarks often report ROC-AUC, which can remain high under severe class imbalance; precision–recall AUC is more sensitive to performance on a rare active class. These metrics answer different questions and should not be interchanged after looking at results.

Aggregate scores also erase chemical structure. A useful evaluation stratifies error by scaffold novelty, molecular size, charge, element, target family, conformer energy, and distance from the training set. It reports repeated seeds and confidence intervals. For screening or materials discovery, it pays special attention to ranking at the extreme tail, because the top candidates—not the median molecule—drive experiments.

{% include figure.liquid loading="lazy" path="assets/img/blog/molprop_evaluation_layers.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A model can achieve low average error in a dense chemical regime while failing on rare chemistry or extreme property values. Tail error, uncertainty calibration, and coverage under chemical shift reveal failures that a central MAE or ROC-AUC can hide. Original diagram." %}

A strong report also includes simple baselines, computational cost, and data efficiency. A deep model that improves MAE by a negligible amount over ECFP plus gradient boosting while requiring thousands of times more compute may still be useful—but its contribution is different from a claim of better representation learning.

## Uncertainty should change a decision

Molecular labels combine at least two kinds of uncertainty. **Aleatoric uncertainty** comes from measurement noise, uncontrolled conditions, conformer populations, and irreducible ambiguity in the representation. **Epistemic uncertainty** comes from limited training coverage and model uncertainty. More data from the same regime can reduce the second but not necessarily the first.

Suppose a regressor returns mean $$\mu(x)$$ and scale $$\sigma(x)$$. Under a Gaussian predictive model, training may use

$$
-\log p(y\mid x)
=
\frac{(y-\mu(x))^2}{2\sigma(x)^2}
+\frac{1}{2}\log \sigma(x)^2
+\text{constant}.
$$

This objective rewards both accurate means and honest scales. Yet a low negative log-likelihood on an in-distribution test set does not establish calibrated uncertainty under scaffold or temporal shift.

Calibration asks whether intervals behave as advertised. If $$I_{0.9}(x)$$ is a predicted 90% interval, empirical coverage should satisfy

$$
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\{y_i\in I_{0.9}(x_i)\}
\approx 0.9.
$$

Coverage must be paired with sharpness: an interval spanning the entire label range is well covered but useless. Calibration should also be checked by chemical subgroup and distance to training data. Deep ensembles, evidential heads, Bayesian approximations, and conformal intervals make different assumptions; comparative studies find no single method uniformly reliable across molecular tasks (<span id="cite-hirschfeld2020"></span>[Hirschfeld et al., 2020](#ref-hirschfeld2020)).

The operational test is whether uncertainty improves a decision. Can it identify predictions that need a quantum calculation, prioritize experiments with high expected value, or abstain on unsupported chemistry? If not, the uncertainty estimate is decorative.

## What benchmark accuracy hides

A benchmark score compresses an entire experimental system into one number. It usually hides:

- whether labels are experimental, simulated, or mixtures of protocols;
- whether uncertainty comes from the assay, conformers, or chemical novelty;
- whether the split resembles prospective use;
- how many close analogues occur across partitions;
- whether test-time coordinates are realistic and available;
- whether pretraining overlaps the benchmark domain;
- which chemical subgroups dominate the aggregate metric;
- how prediction cost compares with data acquisition cost.

This is why molecular property prediction should be framed from the outside inward. Define the deployment population and label-generating process. Choose the representation that contains the necessary information at inference time. Split at the unit that will truly be new. Compare against strong, cheap baselines. Report error distributions and calibrated uncertainty, not only their averages.

The most sophisticated architecture cannot repair a benchmark that asks the wrong question. Conversely, a simple fingerprint or graph model can be scientifically valuable when the representation, split, and decision are aligned. Molecular machine learning progresses when improvements survive that alignment—not merely when another decimal place moves.

## References

<ol class="bibliography">
  <li id="ref-weininger1988">Weininger, D. (1988). <a href="https://doi.org/10.1021/ci00057a005">SMILES, a Chemical Language and Information System. 1. Introduction to Methodology and Encoding Rules</a>. <em>Journal of Chemical Information and Computer Sciences</em>. <a href="#cite-weininger1988">↩</a></li>
  <li id="ref-rogers2010">Rogers, D., & Hahn, M. (2010). <a href="https://doi.org/10.1021/ci100050t">Extended-Connectivity Fingerprints</a>. <em>Journal of Chemical Information and Modeling</em>. <a href="#cite-rogers2010">↩</a></li>
  <li id="ref-ramakrishnan2014">Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). <a href="https://www.nature.com/articles/sdata201422">Quantum Chemistry Structures and Properties of 134 Kilo Molecules</a>. <em>Scientific Data</em>. <a href="#cite-ramakrishnan2014">↩</a></li>
  <li id="ref-wu2018">Wu, Z. et al. (2018). <a href="https://doi.org/10.1039/C7SC02664A">MoleculeNet: A Benchmark for Molecular Machine Learning</a>. <em>Chemical Science</em>. <a href="#cite-wu2018">↩</a></li>
  <li id="ref-wallach2018">Wallach, I., & Heifets, A. (2018). <a href="https://doi.org/10.1021/acs.jcim.7b00403">Most Ligand-Based Classification Benchmarks Reward Memorization Rather Than Generalization</a>. <em>Journal of Chemical Information and Modeling</em>. <a href="#cite-wallach2018">↩</a></li>
  <li id="ref-hu2020">Hu, W. et al. (2020). <a href="https://openreview.net/forum?id=HJlWWJSFDH">Strategies for Pre-Training Graph Neural Networks</a>. <em>ICLR</em>. <a href="#cite-hu2020">↩</a></li>
  <li id="ref-stark2022">Stärk, H. et al. (2022). <a href="https://proceedings.mlr.press/v162/stark22a.html">3D Infomax Improves GNNs for Molecular Property Prediction</a>. <em>ICML</em>. <a href="#cite-stark2022">↩</a></li>
  <li id="ref-hirschfeld2020">Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R., & Coley, C. W. (2020). <a href="https://doi.org/10.1021/acs.jcim.0c00502">Uncertainty Quantification Using Neural Networks for Molecular Property Prediction</a>. <em>Journal of Chemical Information and Modeling</em>. <a href="#cite-hirschfeld2020">↩</a></li>
</ol>

---

*Figure provenance.* All four `molprop_` diagrams are original SVG illustrations generated by `scripts/generate_molprop_figures.py`. They synthesize standard representation, splitting, conformer, and evaluation concepts described in the cited primary literature; no third-party artwork is reproduced.
