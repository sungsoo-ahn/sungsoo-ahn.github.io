---
layout: post
title: "Genomic Foundation Models and Virtual Cells"
date: 2026-08-08
last_updated: 2026-08-08
description: "From genomic sequence models and noisy single-cell measurements to perturbation-conditioned prediction—and the stronger causal, dynamic, and evaluation requirements of a virtual cell."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [genomics-cell]
lecture_paths: [ml4mol, gdl]
tags: [genomics, single-cell, foundation-models, perturbation-modeling, virtual-cells]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the genomics and virtual-cell storyline from my
  Machine Learning for Molecules and Geometric Deep Learning lectures. It
  separates models of molecular sequence, measured cell state, and response to
  intervention because their inputs, guarantees, and evaluation regimes are
  fundamentally different.</em>
</p>

Calling DNA a language is productive, but incomplete. A genomic sequence has local motifs, long-range dependencies, repeated elements, and variation across organisms. These regularities make masked or autoregressive pretraining useful. Yet a cell is not the text of its genome. Cells with essentially the same DNA can occupy different states because transcription factors, chromatin, proteins, metabolites, spatial signals, developmental history, and environment differ.

The same distinction appears one level later. A single-cell RNA-sequencing vector is not a cell state in full; it is a sparse, destructive measurement of part of that state. And a model that reconstructs or embeds those measurements is not automatically a simulator of intervention. Predicting what happens after a drug, gene edit, or cytokine requires conditional data, temporal assumptions, and evaluation on interventions and contexts absent from training.

{% include figure.liquid loading="eager" path="assets/img/blog/virtcell_representation_ladder.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Genomic sequence, regulatory measurements, cell-state snapshots, and virtual-cell predictions form a ladder of increasingly contextual objects. Scaling a sequence model improves the first map, but a virtual cell must also represent state, intervention, time, and predictive uncertainty. Original diagram." %}

The phrase **virtual cell** should therefore be reserved for a stronger claim: a model that predicts distributions over future cellular behavior under specified interventions and contexts, exposes its uncertainty, and remains useful when some combination of donor, cell type, environment, perturbation, or time was not observed during training.

## DNA language models compress sequence distributions

Let a DNA segment be

$$
\mathbf s=(s_1,\ldots,s_L),
\qquad s_i\in\{\mathrm A,\mathrm C,\mathrm G,\mathrm T\}.
$$

A masked genomic model selects positions $$M$$ and minimizes

$$
\mathcal L_{\mathrm{mask}}
=
-\sum_{i\in M}
\log p_{\theta}
\!\left(s_i\mid\widetilde{\mathbf s}\right).
$$

Early models such as DNABERT tokenized overlapping $$k$$-mers and showed that pretraining on a reference genome could transfer to promoter, splice-site, and transcription-factor-binding tasks (<span id="cite-ji2021"></span>[Ji et al., 2021](#ref-ji2021)). Modern models use single nucleotides or learned tokens, longer context, more species, and architectures whose cost grows more gently with sequence length.

Tokenization changes the inductive bias. Overlapping $$k$$-mers expose motifs but create leakage within a naive masking task: neighboring tokens share most of their nucleotides. Single-base tokens remove that shortcut but demand longer effective context. Learned tokens compress recurring substrings, yet their boundaries need not coincide with regulatory elements.

DNA is double-stranded, so strand handling must also match the task. If a label is invariant to reverse complementation, the predictor should satisfy

$$
f_{\theta}(\mathbf s)
=
f_{\theta}(\operatorname{RC}(\mathbf s)),
$$

or be trained so the two orientations agree. Directional tasks such as transcription relative to a promoter are different: reversing the sequence also changes the coordinate system and output orientation. “Reverse-complement invariance” cannot be applied indiscriminately.

Sequence likelihood is already useful for variant scoring:

$$
\Delta_{\mathrm{LM}}
=
\log p_{\theta}(s_i=s_i^{\mathrm{alt}}\mid\mathbf s_{\setminus i})
-
\log p_{\theta}(s_i=s_i^{\mathrm{ref}}\mid\mathbf s_{\setminus i}).
$$

This measures compatibility with the learned sequence distribution. It is not a causal effect on expression, phenotype, or disease. Conservation, mutational processes, genomic repeats, and sampling across species all contribute to likelihood.

## Regulatory genomics needs labels beyond sequence

Promoters, enhancers, splice sites, and transcription-factor motifs are sequence-defined in part, but their activity depends on cellular context. A long sequence model can predict tracks such as chromatin accessibility, protein binding, or RNA abundance:

$$
f_{\theta}(\mathbf s)
=
\widehat{\mathbf y}
\in\mathbb R^{B\times K},
$$

where $$B$$ indexes genomic bins and $$K$$ indexes assays, tissues, or cell types. Enformer combined convolutional feature extraction with attention over a roughly 200-kilobase input and predicted thousands of functional genomic tracks, demonstrating the value of distal context for expression and variant-effect prediction (<span id="cite-avsec2021"></span>[Avsec et al., 2021](#ref-avsec2021)).

This is supervised multitask learning, not merely genomic language modeling. The targets anchor the representation to measured regulatory function. They also import assay noise, uneven cell-type coverage, and batch structure. A model can predict an ATAC-seq track accurately because similar genomic loci or cell types were seen during training; that result does not by itself show transfer to a new donor or regulatory program.

Long context is a capacity, not proof that distal regulation has been learned. Models may rely primarily on promoter-proximal motifs even when given a megabase. Evaluation should stratify variants and regulatory elements by distance, test experimentally perturbed enhancers, and compare against local-context ablations.

## RNA adds structure, processing, and abundance

RNA occupies several roles at once. Its primary sequence determines base-pairing possibilities and protein-binding motifs. Pre-mRNA is spliced into isoforms. Messenger abundance is a context-dependent output of regulation and degradation. Treating every RNA problem as next-token prediction hides these distinctions.

For splicing, a model may predict a distribution over donor and acceptor choices conditional on genomic sequence. For RNA structure, the output may be a base-pairing matrix or three-dimensional conformation. For transcript abundance, sequence is insufficient without cell context. A variant that alters one splice site can have different consequences across tissues because the relevant RNA-binding proteins differ.

The useful representation therefore follows the claimed output: nucleotide embeddings for local motifs, pair representations for base pairing, long genomic context for isoform regulation, and cellular covariates for expression. A single “RNA embedding” is not a universal object.

## A single-cell matrix is an observation model

For cell $$c$$ and gene $$g$$, RNA sequencing returns an integer count $$Y_{cg}$$. A useful abstraction is

$$
Y_{cg}
\sim
\operatorname{NB}
\!\left(\mu_{cg}=\ell_c\lambda_{cg},\phi_g\right),
$$

where $$\lambda_{cg}$$ is latent relative abundance, $$\ell_c$$ is library size or capture efficiency, and $$\phi_g$$ controls overdispersion. Deep generative models such as scVI explicitly use a probabilistic count model while separating biological latent variables from technical covariates (<span id="cite-lopez2018"></span>[Lopez et al., 2018](#ref-lopez2018)).

{% include figure.liquid loading="lazy" path="assets/img/blog/virtcell_count_observation.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Observed single-cell counts arise from latent molecular abundance through finite sampling, capture efficiency, and technical context. Two count vectors can differ even when their underlying biological states are similar, so reconstruction of counts is not reconstruction of the whole cell. Original diagram." %}

A zero can mean no transcript was present, the transcript was not captured, or sequencing depth was insufficient. Conversely, a large count can partly reflect a large library. Normalization and log transformation stabilize learning but change the statistical object. Ranking genes, as some cell language models do, removes absolute scale and emphasizes relative expression.

The analogy between a cell and a sentence is therefore limited. Genes have a fixed identity rather than a natural word order. Expression is quantitative, and multiple genes can be active simultaneously. A token sequence produced by sorting genes is an encoding choice, not biological chronology.

## Cell foundation models learn reusable state encoders

Large cell models pretrain on millions of transcriptomes using masked genes, ranked expression sequences, binned values, or count-aware objectives. Geneformer trained a rank-based Transformer over single-cell transcriptomes and transferred it to network-biology tasks (<span id="cite-theodoris2023"></span>[Theodoris et al., 2023](#ref-theodoris2023)). scGPT jointly represents gene identities and expression values and applies generative pretraining across single-cell and multi-omic tasks (<span id="cite-cui2024"></span>[Cui et al., 2024](#ref-cui2024)).

Scaling helps when the corpus adds biological coverage: more donors, tissues, diseases, developmental stages, species, and measurement technologies. Merely adding more cells from a dominant cell line reduces optimization noise without expanding the support of the problem. Dataset size should therefore be reported alongside the number and balance of donors, studies, tissues, cell types, and conditions.

There is also no neutral cell embedding. An objective that reconstructs highly expressed genes may prioritize housekeeping programs. Batch integration can deliberately erase study identity but also erase a biological effect confounded with study. Cell-type classification encourages stable identity while a perturbation task needs sensitivity to transient response.

## Multimodal embeddings align partial views of state

Single-cell assays can measure RNA, chromatin accessibility, surface proteins, methylation, spatial location, morphology, or imaging phenotypes. Let $$x_c^{(m)}$$ be modality $$m$$ for cell $$c$$. A multimodal encoder produces

$$
z_c^{(m)}=f_m\!\left(x_c^{(m)}\right)
$$

and may align paired modalities using a contrastive loss:

$$
\mathcal L_c
=
-\log
\frac{\exp(\operatorname{sim}(z_c^{(r)},z_c^{(a)})/\tau)}
{\sum_{c'}\exp(\operatorname{sim}(z_c^{(r)},z_{c'}^{(a)})/\tau)},
$$

where $$r$$ and $$a$$ might denote RNA and ATAC. Alignment is useful for annotation transfer and missing-modality prediction. It does not make RNA and chromatin interchangeable. Their shared latent state may omit modality-specific variation needed for a downstream mechanism.

Spatial context adds another layer. A cell's neighbors, tissue compartment, extracellular signals, and geometry can affect its response. Treating cells as independent tokens discards this coupling. A virtual cell intended for tissue biology eventually needs an interaction model, not only a better isolated-cell embedding.

## Perturbation models predict transitions, not labels

Let $$x_0$$ be a baseline cell state, $$u$$ an intervention, $$c$$ context, and $$t$$ elapsed time. The target is a conditional distribution

$$
p_{\theta}(x_t\mid x_0,u,c,t),
$$

not just one mean expression vector. The intervention may be a gene knockout, CRISPR activation, drug structure, dose, cytokine, combination, or environmental change.

{% include figure.liquid loading="lazy" path="assets/img/blog/virtcell_perturbation_distribution.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A perturbation model conditions a baseline state on an intervention, context, and elapsed time to predict a distribution of responses. Interpolation within observed combinations is different from counterfactual prediction for a new intervention, donor, or environment. Original diagram." %}

A simple latent composition assumes

$$
z_t
=
z_{\mathrm{basal}}(x_0)
+e_u(u)
+e_c(c)
+e_t(t),
$$

followed by a decoder to expression. CPA uses this compositional idea to combine basal state, perturbation, dosage, and covariates, enabling predictions for held-out combinations (<span id="cite-lotfollahi2023"></span>[Lotfollahi et al., 2023](#ref-lotfollahi2023)). The factorization is interpretable and data-efficient, but additivity is an assumption. Drug synergy, gene epistasis, saturation, and state-dependent response require interaction terms or richer dynamics.

Predicting the mean response can also erase biologically meaningful heterogeneity. A treatment may split cells into responding and resistant subpopulations. Two models with identical mean-squared error can differ sharply: one produces both modes, while the other predicts an implausible average cell between them.

## Conditioning is not causal identification

Experimental perturbation gives stronger evidence than observational correlation, but a learned conditional distribution is not automatically a causal model. The desired object is closer to

$$
p\!\left(x_t\mid\operatorname{do}(u),x_0,c\right),
$$

where $$\operatorname{do}(u)$$ denotes an intervention. Identification still depends on experimental design. Cells assigned different treatments may come from different batches. Viability selection changes which cells remain measurable. Baseline state is often observed in control cells rather than the exact cells later destroyed for sequencing. Dose and time may be confounded.

Gene deletion by masking a token in a pretrained transcriptome model is especially easy to overinterpret. Removing a gene from the input asks how the encoder changes under missing information. A biological knockout changes transcription, protein abundance, feedback, growth, and selection over time. Without perturbation data or a defensible mechanistic model, the two operations are not equivalent.

Causal claims should therefore follow the data. Randomized interventions can support response prediction within covered contexts. Compositional generalization requires held-out combinations. Mechanism discovery needs additional evidence: replicated interventions, temporal measurements, mediators, orthogonal assays, and prospective validation.

## Batch, donor, and cell-line signals are predictive shortcuts

Single-cell corpora merge studies generated with different platforms, protocols, laboratories, reference genomes, and quality filters. Biological composition also varies across studies. If every disease sample was processed in one batch and every control in another, removing batch is statistically inseparable from removing disease without additional design.

Three confounders deserve special attention:

- **Donor:** genetic background, age, sex, treatment history, and environment can dominate subtle perturbation effects.
- **Cell line:** immortalized lines carry stable genomic abnormalities and laboratory adaptation; success across wells is not transfer to primary cells.
- **Batch:** library chemistry, sequencing depth, operator, plate, and processing date can be decoded from embeddings and may correlate with labels.

Adversarial removal of these signals does not prove biological invariance. A safer evaluation groups entire donors, cell lines, plates, and studies, then reports both biological preservation and technical mixing. Metadata must be treated as part of the dataset, not an optional appendix.

## Foundation-model scaling does not define a virtual cell

More parameters, longer context, and broader pretraining can improve loss and transfer. They do not determine whether a model answers an interventional question. A large masked-cell model may be an excellent encoder and a poor response simulator. A smaller perturbation-specific model may generalize better because its training data match the intervention.

The relevant scaling axes are not only parameter count:

- diversity and balance of donors, tissues, species, and disease states;
- number and chemical or genetic diversity of perturbations;
- dose, time, combination, and environmental coverage;
- paired modalities and spatial or temporal resolution;
- reproducibility across laboratories and platforms;
- quality of negative controls and intervention assignment.

A million near-replicate cells do not provide a million independent contexts. Effective sample size depends on the unit of generalization.

## Evaluation must match the claim

{% include figure.liquid loading="lazy" path="assets/img/blog/virtcell_claim_matched_evaluation.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The held-out factor determines what an evaluation can establish: cells from an observed condition test technical interpolation, while held-out donors, perturbations, or their compositions test progressively stronger transfer. Random cell splits usually leave every biological condition represented during training. Original diagram." %}

A random cell split is appropriate for denoising within an experiment. It is weak evidence for biological generalization because sibling cells from the same donor, batch, cell type, and perturbation appear on both sides.

Claim-matched evaluation should hold out the named axis:

| Claim | Required holdout | Useful outputs |
|---|---|---|
| new cells in a known condition | individual cells | count likelihood, calibration |
| new donor or batch | entire donor/study | cell-type-stratified response |
| new perturbation | entire drug/edit | differential expression and distribution shift |
| new combination | all occurrences of that combination | synergy and interaction error |
| new cell type under treatment | cell type × intervention block | context-specific response |
| prospective virtual-cell use | future experiment/lab | calibrated predictions and failure detection |

Metrics should also match the output. For average response, compare pseudobulk changes

$$
\Delta_g
=
\mathbb E[X_g\mid u]
-
\mathbb E[X_g\mid u=0]
$$

across genes, not only raw expression dominated by housekeeping genes. Report direction and magnitude on differentially expressed genes, but avoid selecting those genes on the test labels and then treating the subset as fixed. For distributions, use held-out likelihood when calibrated, energy distance or kernel discrepancies, cell-state proportions, and coverage of predictive intervals. For trajectories, test multiple times and recovery after intervention.

Strong baselines are essential. The control mean, nearest observed perturbation, additive single-perturbation model, cell-type average response, and linear dose model often capture much of a benchmark. A foundation model should beat the baseline that corresponds to its claimed novelty.

## What would a virtual cell need to predict?

A useful virtual cell should support at least five operations:

1. **Represent state:** integrate sequence, regulatory, molecular, spatial, and environmental evidence with missing modalities and uncertainty.
2. **Predict interventions:** return calibrated distributions for genetic, chemical, and environmental perturbations across dose and time.
3. **Model dynamics:** distinguish immediate signaling, transcriptional response, adaptation, division, death, and long-term state transitions.
4. **Transfer compositionally:** combine new donors, cell types, perturbations, and environments while recognizing unsupported combinations.
5. **Guide experiments:** identify which measurement would most reduce uncertainty, then update after the result.

This vision, articulated as an agenda for AI virtual cells by Bunne and colleagues, is explicitly multimodal, multiscale, predictive, and queryable (<span id="cite-bunne2024"></span>[Bunne et al., 2024](#ref-bunne2024)). It is closer to a scientific model connected to an experimental loop than to one universal embedding.

Mechanistic fidelity will remain uneven. Some interventions can be predicted statistically from dense screens. Rare state transitions, long-term adaptation, metabolism, morphology, and tissue interactions may require explicit biological structure and new measurements. The model should say which regime it knows.

## The useful boundary

Genomic foundation models compress sequence and can support regulatory annotation, variant scoring, and design. Cell foundation models compress noisy molecular snapshots and can support annotation, integration, retrieval, and task adaptation. Perturbation models learn conditional state changes within the support of intervention data.

These are substantial capabilities. None alone is a virtual cell.

The stronger object begins when the model treats a cell as a partially observed dynamical system, distinguishes association from intervention, predicts response distributions rather than plausible-looking averages, and is evaluated on the contexts named in its claim. Scaling remains valuable, but only when it expands biological and experimental support—not when model size substitutes for the missing experiment.

## References

<ol class="bibliography">
  <li id="ref-ji2021">Ji, Y., Zhou, Z., Liu, H., & Davuluri, R. V. (2021). <a href="https://academic.oup.com/bioinformatics/article/37/15/2112/6128680">DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers Model for DNA-Language in Genome</a>. <em>Bioinformatics</em>. <a href="#cite-ji2021">↩</a></li>
  <li id="ref-avsec2021">Avsec, Ž. et al. (2021). <a href="https://www.nature.com/articles/s41592-021-01252-x">Effective Gene Expression Prediction from Sequence by Integrating Long-Range Interactions</a>. <em>Nature Methods</em>. <a href="#cite-avsec2021">↩</a></li>
  <li id="ref-lopez2018">Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). <a href="https://www.nature.com/articles/s41592-018-0229-2">Deep Generative Modeling for Single-Cell Transcriptomics</a>. <em>Nature Methods</em>. <a href="#cite-lopez2018">↩</a></li>
  <li id="ref-theodoris2023">Theodoris, C. V. et al. (2023). <a href="https://www.nature.com/articles/s41586-023-06139-9">Transfer Learning Enables Predictions in Network Biology</a>. <em>Nature</em>. <a href="#cite-theodoris2023">↩</a></li>
  <li id="ref-cui2024">Cui, H. et al. (2024). <a href="https://www.nature.com/articles/s41592-024-02201-0">scGPT: Toward Building a Foundation Model for Single-Cell Multi-Omics Using Generative AI</a>. <em>Nature Methods</em>. <a href="#cite-cui2024">↩</a></li>
  <li id="ref-lotfollahi2023">Lotfollahi, M. et al. (2023). <a href="https://doi.org/10.15252/msb.202211517">Predicting Cellular Responses to Complex Perturbations in High-Throughput Screens</a>. <em>Molecular Systems Biology</em>. <a href="#cite-lotfollahi2023">↩</a></li>
  <li id="ref-bunne2024">Bunne, C. et al. (2024). <a href="https://www.sciencedirect.com/science/article/pii/S0092867424013321">How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities</a>. <em>Cell</em>. <a href="#cite-bunne2024">↩</a></li>
</ol>

---

*Figure provenance.* All four `virtcell_` diagrams are original SVG illustrations generated by `scripts/generate_virtcell_figures.py`. They synthesize standard genomics, single-cell observation, perturbation-modeling, and evaluation concepts described in the cited primary literature; no third-party artwork is reproduced.
