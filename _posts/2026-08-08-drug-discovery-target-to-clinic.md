---
layout: post
title: "How Drug Discovery Turns Biological Hypotheses into Molecules"
date: 2026-08-08
last_updated: 2026-08-08
description: "Why drug discovery is a sequence of linked inference problems—from target validation and molecular binding to exposure, safety, and clinical benefit."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [molecular-science]
lecture_paths: [ml4mol]
tags: [drug-discovery, target-validation, medicinal-chemistry, pharmacokinetics, clinical-trials]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the drug-discovery storyline from my Machine Learning for Molecules lectures. It follows the chain of evidence from a disease mechanism to a clinical intervention, with particular attention to where molecular machine learning enters—and where a molecular prediction is not yet a drug-discovery result.</em>
</p>

A drug begins as a claim about causality: changing some biological process will improve a disease. A molecule is one possible intervention on that process. Between the claim and a medicine lie several different questions. Is the proposed target causal in patients? Can it be modulated selectively? Can a compound reach the relevant tissue at a tolerable dose? Does changing the target produce clinical benefit?

These questions are coupled, but they are not interchangeable. Excellent binding to a noncausal target is useless. Potent inhibition in a biochemical assay can disappear in cells. A compound that works in cells can be metabolized before reaching the tissue. A drug that engages its target can fail because the disease has an escape route or because toxicity leaves no therapeutic window.

This is why drug discovery is better understood as a **sequence of linked inference problems** than as one molecular-optimization problem. Machine learning can reduce uncertainty at several links. It cannot turn a measurement at one link into evidence for all the others.

## The Starting Point Is a Biological Intervention

A target is a molecule or process whose modulation may alter disease: a receptor, enzyme, ion channel, protein–protein interaction, nucleic acid, or cellular phenotype. Target identification finds candidates. Target validation asks the harder counterfactual question:

> If we intervene on this target in the relevant biological context, does the disease state improve?

Genetic association, perturbation experiments, disease models, human tissue, and prior pharmacology provide different kinds of evidence. Human genetics is especially valuable when allelic variation connects a target to both efficacy and safety phenotypes, but even then the molecular mechanism and therapeutically useful direction of modulation must be resolved (<span id="cite-plenge2013"></span>[Plenge et al., 2013](#ref-plenge2013)). A CRISPR knockout can reveal that a gene is necessary in a cell line, but complete deletion is not the same intervention as partial inhibition in an adult patient. An association can implicate a locus without identifying the causal gene. An animal model can reproduce one mechanism while missing human immune, metabolic, or developmental context.

The desired intervention must therefore be specified along with the target: inhibit or activate, continuously or transiently, in which tissue, cell type, disease stage, and patient population. The target alone is not the hypothesis.

**Druggability** is a separate question. A deep enzyme pocket may admit a conventional small molecule. A flat protein interface may require a macrocycle, peptide, degrader, antibody, or an indirect intervention. “Undruggable” often means that the current modality and assay do not provide enough control, not that biology forbids intervention.

{% include figure.liquid loading="eager" path="assets/img/blog/drugdisc_evidence_chain.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Drug discovery links four claims: the target changes disease biology, a molecule modulates the target, an administered dose produces adequate exposure, and that exposure yields patient benefit with acceptable risk. Evidence does not automatically propagate across the links. Original diagram." %}

## Binding Is an Equilibrium, Not a Pose

For a simple reversible interaction

$$
P+L \rightleftharpoons PL,
$$

the dissociation constant is

$$
K_d=\frac{[P][L]}{[PL]},
\qquad
\Delta G^\circ=RT\log K_d.
$$

A lower $$K_d$$ means that less free ligand is needed to occupy the target at equilibrium. In the simplest one-site model, fractional occupancy is

$$
\theta=\frac{[L]}{K_d+[L]}.
$$

This equation exposes why a plausible docking pose is insufficient. A pose proposes geometry; affinity is a free-energy difference involving the bound and unbound ensembles, solvent, protonation, entropy, and competing states. A model can rank poses within one receptor structure and still miss affinity because the protein reorganizes, water is displaced, or the ligand adopts an expensive conformation.

Affinity is also not cellular potency. A biochemical assay may expose purified protein to a controlled free-ligand concentration. A cell adds membranes, transporters, metabolism, protein binding, target abundance, feedback, and off-targets. The half-maximal concentration reported by an assay—$$IC_{50}$$ or $$EC_{50}$$—depends on the assay design and should not be silently interpreted as $$K_d$$.

Kinetics adds another dimension. The rates

$$
k_{\mathrm{on}}[P][L]
\quad\text{and}\quad
k_{\mathrm{off}}[PL]
$$

can yield the same $$K_d=k_{\mathrm{off}}/k_{\mathrm{on}}$$ for compounds with different residence times. When exposure fluctuates, a slow off-rate may sustain target engagement after plasma concentration falls (<span id="cite-copeland2006"></span>[Copeland et al., 2006](#ref-copeland2006)). Whether this matters depends on target turnover and disease biology; residence time is not universally preferable.

## A Hit Is Evidence, Not a Starting Drug

Hit identification can use high-throughput screening, fragments, phenotypic screening, virtual screening, or prior chemical matter. A credible hit survives orthogonal confirmation and controls for aggregation, fluorescence interference, covalent reactivity, assay artifacts, and sample identity. Its structure and activity should be reproducible.

The next task is to establish a **structure–activity relationship (SAR)**. Chemists make controlled changes and observe how potency, selectivity, solubility, permeability, metabolic stability, and toxicity respond. If large structural changes leave the signal unchanged, the assay may be reporting an artifact. If nearby analogues move activity smoothly, the series offers a path for optimization.

Suppose a lead must achieve target potency $$p(x)$$, aqueous solubility $$s(x)$$, microsomal stability $$m(x)$$, and low off-target activity $$o(x)$$. There is generally no molecule maximizing all four. A scalar score

$$
J(x)=w_p p(x)+w_s s(x)+w_m m(x)-w_o o(x)
$$

hides the choice of weights and can reward an unacceptable tradeoff. Medicinal chemistry instead navigates a Pareto frontier while enforcing hard constraints such as chemical stability, synthetic accessibility, and reactive-group exclusions.

{% include figure.liquid loading="eager" path="assets/img/blog/drugdisc_optimization_frontier.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Lead optimization is multi-objective. Improving potency can expose solubility, clearance, selectivity, or synthesis liabilities; the useful region is a constrained Pareto frontier, not the maximum of one predicted score. Original diagram." %}

This is where property-prediction and generative models are useful. A model can prioritize compounds, estimate uncertainty, propose analogues, and decide which assay would be informative next. But the training labels inherit protocol differences, censoring, batch effects, and series bias. Random molecular splits often place close analogues on both sides of evaluation, measuring interpolation within known chemistry rather than prospective discovery; benchmark studies have shown how readily such similarity turns memorization into apparent generalization (<span id="cite-wallach2018"></span>[Wallach & Heifets, 2018](#ref-wallach2018)). The broader issues of representation, conformers, splits, and uncertainty are developed in [Molecular Data and Property Prediction]({% post_url 2026-08-08-molecular-data-property-prediction %}).

## Exposure Connects a Molecule to a Tissue

Pharmacokinetics describes what the body does to a compound: absorption, distribution, metabolism, and excretion. After an oral dose, concentration often rises as the drug is absorbed, reaches a maximum $$C_{\max}$$, then falls through distribution and clearance. The area under the concentration–time curve (AUC) measures total systemic exposure.

For a simple one-compartment intravenous model,

$$
C(t)=C_0e^{-k_e t},
\qquad
t_{1/2}=\frac{\log 2}{k_e},
\qquad
CL=k_eV_d,
$$

where $$CL$$ is clearance and $$V_d$$ is apparent volume of distribution. Real compounds may require multiple compartments, saturable processes, active transport, or time-varying metabolism, but the simple model makes one point clear: potency has meaning only relative to free concentration at the site of action.

Total plasma concentration can be misleading when most compound is bound to plasma proteins. Brain targets add the blood–brain barrier and efflux transporters. Intracellular targets add membrane permeability and sequestration. Prodrugs deliberately separate the administered molecule from the active species. Metabolites may be inactive, active, or toxic.

Pharmacodynamics describes what exposure does to the biological system. A common saturating response model is

$$
E(C)=E_0+\frac{E_{\max}C^h}{EC_{50}^h+C^h},
$$

with Hill coefficient $$h$$. The response may lag concentration because target turnover and downstream signaling have their own timescales. Biomarkers are valuable when they show that a dose reaches the tissue and engages the intended mechanism, rather than merely producing a correlated change.

{% include figure.liquid loading="eager" path="assets/img/blog/drugdisc_pkpd_bridge.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A dose creates a concentration–time profile; free exposure at the relevant tissue produces target engagement; target engagement produces a downstream response. Potency measured in one assay does not specify any of the other mappings. Original diagram." %}

## Safety Is a Margin, Not a Binary Property

Any pharmacologically active molecule perturbs biology. The practical question is whether desired effects occur at exposures below those causing unacceptable harm. A schematic therapeutic index compares toxic and effective doses,

$$
TI=\frac{TD_{50}}{ED_{50}},
$$

but a single ratio suppresses the shape of both dose–response curves, patient variability, treatment duration, and severity of the adverse event.

Safety failures arise through several routes. **On-target toxicity** occurs when the intended mechanism is harmful in another tissue or at excessive inhibition. **Off-target toxicity** comes from unintended molecular interactions. Reactive metabolites, immune responses, organ accumulation, and drug–drug interactions add mechanisms that may not appear in an acute cellular assay.

Selectivity must therefore be evaluated against a relevant panel, not defined as “stronger binding to the intended target.” A tenfold affinity margin can disappear when the off-target is more abundant, the free tissue concentration differs, or its physiological response is steeper. Chronic dosing can expose liabilities invisible in a short experiment.

Machine-learning toxicity alerts are triage tools. They are strongest inside chemical and assay regimes represented in training and weakest for rare mechanisms, unusual metabolites, and new modalities—the very regions in which confident extrapolation is most dangerous.

## Clinical Trials Test the Whole Causal Chain

Preclinical studies support a first human dose, but clinical development asks successively broader questions. Phase I emphasizes safety, tolerability, exposure, and often target engagement. Phase II asks whether the intervention shows efficacy in an appropriate patient population and refines dose. Phase III tests benefit and risk at the scale and rigor needed for registration. The boundaries vary by disease and modality, but the evidentiary progression remains.

Clinical failure is not one event. A trial may fail because the target was not causal, the drug did not reach or engage it, the dose was limited by toxicity, the endpoint was insensitive, the population was heterogeneous, or the trial was underpowered. Without exposure and engagement biomarkers, a negative efficacy result cannot distinguish a failed biological hypothesis from a failed intervention.

Published analyses show that transition probabilities vary strongly by therapeutic area and development phase, making overall “success rate” a property of a defined portfolio and time window rather than a universal constant (<span id="cite-wong2019"></span>[Wong et al., 2019](#ref-wong2019)). The long decline in output per inflation-adjusted research investment—often called Eroom's law—also warns that adding more screening and computation does not automatically remove biological uncertainty (<span id="cite-scannell2012"></span>[Scannell et al., 2012](#ref-scannell2012)).

{% include figure.liquid loading="eager" path="assets/img/blog/drugdisc_funnel_feedback.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The discovery funnel narrows as evidence accumulates, but late outcomes must feed back to earlier hypotheses. A clinical failure can implicate target biology, molecular mechanism, exposure, safety, patient selection, or trial design; labeling it simply as a failed molecule discards information. Original diagram." %}

## Machine Learning Must Be Attached to a Decision

Drug-discovery data are generated by decisions. Chemists synthesize compounds they expect to work. Assays are run on selected series. Negative results may be censored or absent. Clinical candidates are an extreme, nonrandom subset of discovered molecules. A model trained on this record learns the selection process alongside chemistry.

This creates three common overclaims:

- **Retrospective prediction becomes prospective value.** A random split measures recognition of familiar series, not whether the model changes which compound should be made next.
- **A surrogate becomes the objective.** Generated molecules exploit a learned potency or synthesizability score outside its reliable region.
- **A molecular endpoint becomes a program outcome.** Better docking, affinity, or ADMET prediction is reported as faster drug discovery without tracing the downstream decision.

A useful evaluation starts from the decision. Did prioritization recover active compounds under a time or scaffold split? Did uncertainty identify measurements that improved the series? Did proposed compounds synthesize and reproduce their predicted profile? Did the model reduce experiments, calendar time, or failure risk compared with the actual alternative?

Prospective, blinded tests are unusually valuable because they include the selection step. So are negative results: they reveal the model's applicability boundary and improve the next acquisition round. The central unit is not a benchmark score but a **closed experimental loop**.

## A Drug Is a Chain of Evidence

The discovery pipeline is often drawn as target identification, hit finding, hit-to-lead, lead optimization, preclinical development, and clinical trials. The drawing looks linear; the work is not. Poor cellular translation sends a team back to mechanism or permeability. Toxicity motivates new chemistry or a different modality. Patient biomarkers revise the target hypothesis. Every stage changes the distribution of questions at the next stage.

The most durable view is therefore a chain of claims:

1. intervening on a target changes relevant disease biology;
2. a molecule produces the intended intervention selectively;
3. a feasible dose creates sufficient exposure and engagement;
4. the intervention improves patient outcomes with acceptable risk.

Molecular machine learning operates mainly in the middle of this chain, where representations, property models, docking, simulation, and generation can make experimentation more selective. Its contribution becomes scientifically legible when it states which uncertainty is reduced and which claim remains untested.

That boundary is not a limitation to hide. It is how a computational result becomes useful to a drug-discovery program.

---

## References

<span id="ref-wong2019"></span>Wong, C. H., Siah, K. W., & Lo, A. W. (2019). [Estimation of Clinical Trial Success Rates and Related Parameters](https://doi.org/10.1093/biostatistics/kxx069). *Biostatistics, 20*(2), 273–286. [↩](#cite-wong2019)

<span id="ref-scannell2012"></span>Scannell, J. W., Blanckley, A., Boldon, H., & Warrington, B. (2012). [Diagnosing the Decline in Pharmaceutical R&D Efficiency](https://www.nature.com/articles/nrd3681). *Nature Reviews Drug Discovery, 11*, 191–200. [↩](#cite-scannell2012)

<span id="ref-plenge2013"></span>Plenge, R. M., Scolnick, E. M., & Altshuler, D. (2013). [Validating Therapeutic Targets through Human Genetics](https://www.nature.com/articles/nrd4051). *Nature Reviews Drug Discovery, 12*, 581–594. [↩](#cite-plenge2013)

<span id="ref-copeland2006"></span>Copeland, R. A., Pompliano, D. L., & Meek, T. D. (2006). [Drug–Target Residence Time and Its Implications for Lead Optimization](https://www.nature.com/articles/nrd2082). *Nature Reviews Drug Discovery, 5*, 730–739. [↩](#cite-copeland2006)

<span id="ref-wallach2018"></span>Wallach, I., & Heifets, A. (2018). [Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization](https://doi.org/10.1021/acs.jcim.7b00403). *Journal of Chemical Information and Modeling, 58*(5), 916–932. [↩](#cite-wallach2018)

---

*Figure provenance.* All four `drugdisc_` diagrams are original SVG illustrations generated by `scripts/generate_drugdisc_figures.py`. They synthesize standard pharmacological and drug-development concepts described in the post; no lecture-slide, paper, or third-party icon artwork is reproduced.
