---
layout: post
title: "MADField: Predicting Where Gas Goes in Nanoporous Materials"
date: 2026-06-22
last_updated: 2026-06-23
description: "MADField predicts the 3D equilibrium adsorbate density field in nanoporous materials, recovering gas uptake by integration and screening 270,583 MOFs at five orders of magnitude lower cost than simulation."
post_type: research
authors: ["Yoonho Kim", "Seongsu Kim", "Sungsoo Ahn", "Honghui Kim"]
categories: [machine-learning]
tags: [adsorption, materials, density-functional-theory, neural-operators, nanoporous-materials, multi-fidelity]
toc:
  sidebar: left
related_posts: false
published: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post describes our NeurIPS 2026 paper, MADField: Multi-fidelity Amortized Density Field for Adsorption in Nanoporous Materials. For background on gas adsorption, GCMC, and classical DFT, see our earlier <a href="/blog/2026/adsorption-gcmc-classical-dft/">adsorption tutorial</a>. Code and the cDFT benchmark dataset (280,000 calculations) will be released upon publication.</em>
</p>

## Material "Adsorbent" Discovery

Adsorption is the process by which gas molecules accumulate inside the pores of a porous solid. The solid that holds the gas is the **adsorbent**; the gas it holds is the **adsorbate**. Methane inside a porous crystal, or carbon dioxide pulled from flue gas, are both adsorption.

Finding good adsorbents is one of the large axes of materials discovery. Zeolites, metal-organic frameworks (MOFs), covalent organic frameworks (COFs), and amorphous carbons are all actively explored — each with distinct pore geometries, chemistries, and synthesis routes. The 2025 Nobel Prize in Chemistry, awarded to Robson, Kitagawa, and Yaghi for the design and synthesis of MOFs, reflects how central this class of materials has become.

What makes adsorbents promising is also what makes them hard. MOFs and amorphous carbons span an enormous material space — hundreds of thousands synthesized, and millions more proposed computationally — so the chance that an excellent adsorbent exists somewhere in that space is high. The difficulty is finding it: evaluating that many candidates quickly and accurately, a needle-in-a-haystack search over a space too large to test by hand. This is the central problem of adsorbent discovery.

### Evaluating an adsorbent

Evaluating a candidate means computing its gas **uptake** $$N$$: the amount of gas it adsorbs at a given temperature and pressure. Uptake is the single number used to rank materials in high-throughput screening, since most applications come down to storing or capturing as much gas as possible at a target condition.

The catch is that uptake is not read off directly. It is the outcome of an adsorption equilibrium that has to be solved for each material, adsorbate, and condition. There are two standard ways to solve it, and both are slow.

- **Grand canonical Monte Carlo (GCMC)** is the particle-based gold standard. It randomly inserts, deletes, and moves gas molecules inside the pores, accepting or rejecting each move based on an energy criterion — repeat this for millions of steps and average to get uptake. Reliable, but roughly 3 hours per material per condition.
- **Classical density functional theory (cDFT)** takes a different route: instead of sampling particles, it treats the gas density field $$\rho(\mathbf{r})$$ as the variable and finds the equilibrium by minimizing a free-energy functional via fixed-point iteration. Faster than GCMC, but relies on an approximate functional and still needs an iterative solve for every case.

{% include figure.liquid loading="eager" path="assets/img/blog/adsorption_gcmc_cdft_particle_density.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Two views of adsorption equilibrium. GCMC samples particle configurations and averages them into a density field; cDFT solves for the same density field directly by fixed-point iteration. Both are accurate but too slow to screen a full materials database." %}

Both are accurate, and both are too slow for the search. Screening a database of hundreds of thousands of frameworks with GCMC would take on the order of a century of CPU time. This is the bottleneck MADField removes.

## MADField

MADField is a neural network that predicts the 3D equilibrium gas density field in porous materials. Given a material, an adsorbate, and a thermodynamic condition, it outputs the full density field $$\rho(\mathbf{r})$$ — and gas uptake is computed from that output by integration, without running GCMC or cDFT.

**Can MADField screen a real database?** We test this on ARC-MOF, a standard benchmark of **270,583** real MOF structures, asking how well MADField retrieves the **167 rare high-capacity targets** for CH₄ storage — just 0.06% of the database. MADField recovers **95% of them within the top 1.7%** of candidates. Its average precision is **56× higher** than the best learned baseline, at a cost **five orders of magnitude** below GCMC.

**Uptake accuracy on MOFs.** We benchmark MADField against the strongest existing ML models for uptake prediction. MADField reduces uptake error by **6.0×** over the best baseline on cDFT labels, and by **15.4×** on GCMC labels.

**Generalization to unseen materials.** The real test is whether a model generalizes. We evaluate on disordered materials — amorphous carbons, polymers of intrinsic microporosity (PIMs), hyper-cross-linked polymers, kerogens — that were never seen during training. MADField stays **4–5×** ahead of the best baseline. The strongest MOF baselines collapse: Uni-MOF's error jumps from 7.7 to 31.3 cm³/g on unseen materials. MADField's rises only from 0.82 to 2.71.

## How we solve this?

Two ideas drive these results.

### Predict the density field, not the scalar

Most ML models for adsorption predict scalar uptake $$N$$ directly. We predict the full 3D equilibrium gas density field $$\rho(\mathbf{r})$$ instead, and recover uptake by integrating it:

$$
N = \int_{\mathcal{V}} \rho(\mathbf{r})\,d\mathbf{r}
$$

This matters more than it might seem. A version of MADField that shares the exact same backbone but regresses $$N$$ directly is **7.2× worse**. The accuracy gain comes from the target, not the network.

Density-field prediction is not new, but prior work has a critical gap: existing models predict a **normalized** density that integrates to one by construction. Integrating it gives 1, not uptake. Some models require the uptake $$N$$ itself as an input to work around this. Neither can independently predict how much gas a material holds.

MADField predicts an **unnormalized** density in physical units (molecules per Å³). Integrating it gives the real uptake — no normalization trick, no uptake fed in. This also means the predicted density field can serve as a warm start for the cDFT solver: it cuts solver iterations by **2.0×** and recovers convergence in **42%** of cases that fail from the standard initialization.

### Multi-fidelity: cDFT teaches, GCMC corrects

Predicting a density field rather than a scalar makes the data problem harder: we need density field labels, not just uptake numbers. GCMC-derived density fields are the accurate target, but expensive to generate at scale. cDFT fields are cheap and abundant, but inherit the approximation error of the cDFT functional.

We use both. MADField is first **pre-trained on 280,000 cDFT calculations** spanning 4,000 MOFs and nine adsorbates, then **fine-tuned on a GCMC dataset 14.7× smaller**. cDFT teaches a broad prior — how geometry, adsorbate identity, and thermodynamic conditions shape the density field. GCMC corrects the remaining fidelity gap.

This is what drives generalization. Training on GCMC alone, without the cDFT prior, is **2.9× worse** on in-distribution MOFs and **18.1× worse** on out-of-distribution materials. The cDFT prior is what carries MADField to material classes it never saw during fine-tuning.

## Dataset release

As part of this work, we release the cDFT benchmark dataset of 280,000 PC-SAFT cDFT calculations — density fields, uptake values, and isotherms across nine adsorbates and diverse material classes. Generating it required 3,600 H200 GPU-hours. We hope it serves as a foundation for future multi-fidelity adsorption modeling.

## References

- Y. Kim, S. Kim, S. Ahn, H. Kim. "MADField: Multi-fidelity Amortized Density Field for Adsorption in Nanoporous Materials." *NeurIPS 2026*.
- J. Burner et al. "ARC-MOF: A Diverse Database of Metal-Organic Frameworks with DFT-Derived Partial Charges and Descriptors for Enhanced Machine Learning Predictions." *Chemistry of Materials*, 2023.
- J. Burner et al. "DeepAPD: A Deep Equivariant Neural Network to Predict Adsorbate Probability Density in MOFs." *Journal of Chemical Theory and Computation*, 2026.
- Y. Sun and J. I. Siepmann. "Understanding Adsorption with Machine Learning: Predicting Local Contributions to Henry's Law Adsorption Coefficients." *Journal of Chemical Theory and Computation*, 2024.
- Y. Wang et al. "A comprehensive transformer-based approach for high-accuracy gas adsorption properties prediction in metal-organic frameworks." *Nature Communications*, 2024.
- R. Evans. "The nature of the liquid-vapour interface and other topics in the statistical mechanics of non-uniform, classical fluids." *Advances in Physics*, 28(2):143–200, 1979.
- R. Thyagarajan and D. S. Sholl. "A Database of Porous Rigid Amorphous Materials." *Chemistry of Materials*, 2020.
- J. Rosen et al. "Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery." *Matter*, 2021.
