---
layout: post
title: "MADField: Multi-fidelity Amortized Density Field for Adsorption in Nanoporous Materials"
date: 2026-06-22
last_updated: 2026-06-23
description: "MADField predicts the full 3D adsorbate density field in nanoporous materials, turning slow gas-adsorption simulation into a single forward pass."
post_type: research
authors: ["Yoonho Kim", "Seongsu Kim", "Sungsoo Ahn", "Honghui Kim"]
categories: [machine-learning]
tags: [adsorption, materials, grand-canonical-monte-carlo, classical-density-functional-theory, multi-fidelity]
toc:
  sidebar: left
related_posts: false
published: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post describes our paper, <a href="https://arxiv.org/abs/2606.21284">MADField: Multi-fidelity Amortized Density Field for Adsorption in Nanoporous Materials</a>. For background on gas adsorption, GCMC, and classical DFT, see our earlier <a href="/blog/2026/adsorption-gcmc-classical-dft/">adsorption tutorial</a>.</em>
</p>

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_hero.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>Predicting where gas goes, at database scale.</strong> MADField replaces slow per-material adsorption simulation with a single neural-network forward pass, making it possible to screen hundreds of thousands of porous frameworks for the rare high-capacity ones." %}

**MADField evaluates gas adsorption about 250,000× faster than grand canonical Monte Carlo (GCMC), the gold-standard simulation.** We built it to predict the full 3D adsorbate density field in a tenth of a second — instead of hours of GCMC — and read off uptake by integrating it. On the full **270,583-structure ARC-MOF database** it recovers **95% of the rare high-capacity methane stores in the top 1.7%** of its ranking, **56× more precise than the best previous ML baseline**, and holds up on unseen materials like amorphous carbons.

This post shares the whole story: the problem, what MADField achieves, and how. The how is two ideas — predict the density field, not the scalar uptake, and train multi-fidelity, a broad, cheap cDFT prior corrected by a little GCMC. Together they make MADField fast and accurate enough to drop into real adsorbent-discovery pipelines for gas storage, carbon capture, and separation.

## Material "adsorbent" discovery

Adsorption is the process by which gas molecules accumulate inside the pores of a porous solid. The solid that holds the gas is the **adsorbent**; the gas it holds is the **adsorbate**. Methane inside a porous crystal, or carbon dioxide pulled from flue gas, are both adsorption.

Finding good adsorbents is one of the large axes of materials discovery. Porous materials like zeolites, metal--organic frameworks (MOFs), covalent organic frameworks (COFs), and amorphous carbons are all actively explored. The 2025 Nobel Prize in Chemistry, awarded to Robson, Kitagawa, and Yaghi for the design and synthesis of MOFs (<span id="cite-nobel2025"></span>[Nobel Prize in Chemistry, 2025](#ref-nobel2025)), reflects how central this class of materials has become.

These porous materials span an enormous space — hundreds of thousands synthesized, and millions more proposed computationally (<span id="cite-wilmer2012"></span>[Wilmer et al., 2012](#ref-wilmer2012)) — so the chance that an excellent adsorbent exists somewhere in it is high. The difficulty is finding it: a **needle-in-a-haystack** search over a space too large. This is the central problem of adsorbent discovery.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_csd_mofs.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>MOFs deposited in the Cambridge Structural Database.</strong> The number of MOF crystal structures in the CSD has grown from a few hundred in the 1980s to about 137,000 by 2025. And that is only the synthesized space — before the millions more proposed computationally — which is what makes finding the best adsorbent a needle-in-a-haystack search. Data: the CSD MOF subset, counted by deposition year (CSD 2026.1)." %}

### Bottleneck: evaluating candidates in scale

Evaluating a candidate means computing its gas **uptake** $N$: the amount of gas it adsorbs at a given temperature and pressure. Uptake is the single number used to rank materials in high-throughput screening, since most applications come down to storing or capturing as much gas as possible at a target condition.

The catch is that we can't read uptake off directly. To know *how much* gas a material holds, we first have to know *where* it all goes — the full 3D pattern of gas in the pores, its **density field $\rho(\mathbf{r})$**. Uptake $N$ is just the integral of that field over the unit cell,

$$
N(P, T) = \int_{\mathcal{V}} \rho_{\mathrm{eq}}(\mathbf{r})\, d\mathbf{r},
$$

so the real work is getting $\rho(\mathbf{r})$ — and there are two standard ways, both slow:

<div class="row justify-content-center align-items-end my-4">
  <div class="col-md-6 col-12 mb-3 mb-md-0 text-center">
    <video class="img-fluid rounded z-depth-1" autoplay loop muted playsinline controls src="{{ '/assets/video/madfield_gcmc.mp4' | relative_url }}"></video>
    <p class="mt-2 mb-0"><strong>GCMC</strong></p>
    <p class="text-muted" style="font-size: 0.85em;">Molecules build up and move, then coarse-grain into the density field. Millions of steps.</p>
  </div>
  <div class="col-md-6 col-12 mb-3 mb-md-0 text-center">
    <video class="img-fluid rounded z-depth-1" autoplay loop muted playsinline controls src="{{ '/assets/video/madfield_cdft.mp4' | relative_url }}"></video>
    <p class="mt-2 mb-0"><strong>cDFT</strong></p>
    <p class="text-muted" style="font-size: 0.85em;">Propose a new density to minimize the free-energy functional. Tens–hundreds of iterations.</p>
  </div>
</div>

- **Grand canonical Monte Carlo (GCMC)** (<span id="cite-adams1975"></span>[Adams, 1975](#ref-adams1975); <span id="cite-dubbeldam2016"></span>[Dubbeldam et al., 2016](#ref-dubbeldam2016)) — the particle-based gold standard. Randomly insert, delete, and move gas molecules in the pores over millions of steps, then average the snapshots into $\rho(\mathbf{r})$. Accurate, but several hours per material per condition.
- **Classical density functional theory (cDFT)** — skip the particles and solve for $\rho(\mathbf{r})$ directly, minimizing a free-energy functional (<span id="cite-evans1979"></span>[Evans, 1979](#ref-evans1979)) by fixed-point iteration. Faster, but rests on an approximate functional and still iterates for every case.

A practical screening covers hundreds of thousands of materials, and even cDFT — the faster of the two — costs roughly 5 GPU-minutes per material at a fixed temperature and pressure, so screening 100,000 candidates runs to about 350 GPU-days. GCMC would take far longer even with multicore CPU. And those hundreds of thousands are only the *experimental* structures; the space of *hypothetical* frameworks proposed in silico runs from millions to trillions, where brute-force simulation is hopeless from the start.

## MADField predicts $\rho(\mathbf{r})$ directly

MADField is a neural network that predicts the 3D equilibrium gas density field in porous materials. Given a material, an adsorbate, and a thermodynamic condition, it outputs the full density field $\rho(\mathbf{r})$ in sub-second — and gas uptake $N$ is computed by integration.

<div class="row justify-content-center my-4">
  <div class="col-md-7 col-12">
    {% include figure.liquid loading="eager" path="assets/img/blog/madfield_oneshot.png" class="img-fluid rounded z-depth-1" zoomable=true caption="MADField predicts the converged equilibrium CH₄ density directly — the same field GCMC and cDFT reach by iterating, but in a single ~0.1 s pass with no iteration. Reconstructed from our paper visualizations." %}
  </div>
</div>
### Result 1: benchmark on a practical MOF screening pipeline

For methane storage the quantity that matters is **working capacity** (WC): the CH₄ a material actually delivers between a full and an empty tank — its uptake swing between the storage and depletion pressures,

$$
\mathrm{WC} = N(P_{\mathrm{ads}}, T) - N(P_{\mathrm{des}}, T),
$$

with $P_{\mathrm{ads}} = 65~\mathrm{bar}$, $P_{\mathrm{des}} = 5.8~\mathrm{bar}$, and $T = 298~\mathrm{K}$, so each candidate needs two uptake evaluations. We call a framework a **high-capacity target** when its $\mathrm{WC} \ge 200~\mathrm{cm^3/cm^3}$.

Across the full ARC-MOF database (<span id="cite-burner2023"></span>[Burner et al., 2023](#ref-burner2023)) of **270,583** frameworks, only **167** clear that bar — **0.06%** of the database, a true needle in a haystack. Ranking all of them by predicted WC, MADField recovers **95% of the targets within the top 1.7%** of candidates, at **56× higher average precision**[^ap] than the best learned baseline,[^baselines] and **five orders of magnitude** less cost than GCMC.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_screening.png" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>Screening benchmark on ARC-MOF database.</strong> All 270,583 frameworks ranked by predicted CH₄ working capacity (167 high-capacity targets, 0.06% of the database). The two MADField curves are MADField-GCMC (the full multi-fidelity model) and MADField-cDFT (trained on cDFT alone). (a) MADField-GCMC recovers 95% of the targets after screening just 4,716 frameworks, far above every baseline; (b) it reaches 56× higher average precision than the best learned baseline — and even MADField-cDFT stays well ahead — at a per-MOF cost five orders of magnitude below GCMC. Adapted from our paper." %}

### Result 2: uptake accuracy on unseen materials

MADField generalizes to materials it never saw during training. On in-distribution MOFs every model does reasonably well — MADField tracks the reference isotherm (uptake versus pressure at a fixed temperature) most closely, but strong baselines such as Uni-MOF (<span id="cite-wang2024"></span>[Wang et al., 2024](#ref-wang2024)) stay near it too. The picture splits on unseen materials: for disordered solids like amorphous carbons (<span id="cite-thyagarajan2020"></span>[Thyagarajan & Sholl, 2020](#ref-thyagarajan2020)) — along with polymers of intrinsic microporosity (PIMs), hyper-cross-linked polymers, and kerogens[^kerogen] — MADField still follows the reference closely, while the baselines break down badly.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_isotherms.png" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>Representative CH₄ isotherms (uptake vs. pressure).</strong> For both an in-distribution QMOF and an out-of-distribution amorphous carbon, MADField (red) tracks the cDFT reference (circles) across the full pressure range, while the uptake baselines drift off. Adapted from our paper." %}

### Result 3: density accuracy

MADField predicts the density field itself accurately, not just its integral. The mean Tanimoto similarity to the reference density is **0.996** (cDFT) and **0.965** (GCMC) across MOFs — versus **0.943** and **0.891** for the strongest density baseline, DeepAPD (<span id="cite-burner2026"></span>[Burner et al., 2026](#ref-burner2026)) — and on unseen amorphous carbons the margin widens to **0.885** against **0.492**.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_density.png" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>Predicted CH₄ density fields and their error.</strong> Columns: the framework, the cDFT reference density (gray), and the per-voxel error of MADField, DeepAPD, and SorbIIT (red marks larger error); rows are an in-distribution QMOF and an out-of-distribution amorphous carbon. MADField stays near-white in both — Tanimoto similarity T near 1 — while the baselines accumulate visible error, especially on the unseen material. Adapted from our paper." %}

### Result 4: acceleration of cDFT

Beyond replacing simulation, MADField can also speed it up. Because it predicts an unnormalized density in physical units, that prediction is a ready-made starting guess for the cDFT fixed-point solver: warm-starting from it cuts solver iterations by **2.0×** and recovers convergence in **42%** of cases that fail from the standard initialization.

## Two ideas behind MADField's success

Two ideas drive these results. The first is the prediction target: MADField outputs the full density field $\rho(\mathbf{r})$ rather than the scalar uptake $N$ — a single object that yields uptake, binding sites, and isotherms together, and the source of most of the accuracy. The second is how it is trained: a broad, cheap cDFT prior corrected by a little high-fidelity GCMC, which is what lets it generalize to materials it never saw. We take each in turn.

### $\rho(\mathbf{r})$ as a unified target, not the scalar $N$

Most ML models for adsorption predict scalar uptake $$N$$ directly. We predict the full 3D equilibrium gas density field $$\rho(\mathbf{r})$$ instead, and recover uptake by integrating it:

$$
N = \int_{\mathcal{V}} \rho(\mathbf{r})\,d\mathbf{r}
$$

Integrating the field recovers uptake, but it carries more than its integral: where $$\rho(\mathbf{r})$$ concentrates marks the binding sites, and sweeping pressure traces the full isotherm. All three fall out of one prediction.

This matters more than it might seem. A version of MADField that shares the exact same backbone but regresses $$N$$ directly is **7.2× worse**. The accuracy gain comes from the target, not the network.

Density-field prediction is not new (<span id="cite-sun2024"></span>[Sun & Siepmann, 2024](#ref-sun2024); [Burner et al., 2026](#ref-burner2026)), but prior work has a critical gap: existing models predict a **normalized** density that integrates to one by construction. Integrating it gives 1, not uptake. Some models require the uptake $$N$$ itself as an input to work around this. Neither can independently predict how much gas a material holds.

MADField predicts an **unnormalized** density in physical units (molecules per Å³). Integrating it gives the real uptake — no normalization trick, no uptake fed in.

### Multi-fidelity: cDFT teaches, GCMC corrects

Predicting a density field rather than a scalar makes the data problem harder: we need density-field labels, not just uptake numbers. GCMC-derived density fields are the accurate target, but expensive to generate at scale. cDFT fields are cheap and abundant, but inherit the approximation error of the cDFT functional.

We use both. MADField is first **pre-trained on 280,000 cDFT calculations** spanning 4,000 MOFs and nine adsorbates — we call this cDFT-only model **MADField-cDFT** — then **fine-tuned on a GCMC dataset 14.7× smaller** to give **MADField-GCMC**. cDFT teaches a broad prior — how geometry, adsorbate identity, and thermodynamic conditions shape the density field. GCMC corrects the remaining fidelity gap.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_multifidelity.png" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>Multi-fidelity training.</strong> MADField is pre-trained on a large, low-fidelity cDFT dataset, then fine-tuned on a small, high-fidelity GCMC dataset — combining cDFT's broad coverage with GCMC's accuracy. Adapted from our paper." %}

How good is the cheap, approximate cDFT data on its own? Good enough that **MADField-cDFT** — trained without a single GCMC label — already outranks every learned baseline on the screening above, and on per-MOF working-capacity accuracy it places second only to MADField-GCMC, ahead of Uni-MOF and the rest. Starting from that strong cDFT foundation, the small GCMC set then delivers a decisive gain: fine-tuning raises screening average precision **8×** (MADField-cDFT → MADField-GCMC, 0.068 → 0.557). cDFT supplies the breadth, GCMC the fidelity — the full result needs both.

{% include figure.liquid loading="eager" path="assets/img/blog/madfield_wc_parity.png" class="img-fluid rounded z-depth-1" zoomable=true caption="<strong>Working capacity accuracy across models.</strong> Predicted versus GCMC-reference working capacity over all 270,583 frameworks; the dashed line is \(y=x\) and each panel's MAE is in cm³/cm³. MADField-GCMC tracks the reference most tightly (MAE 4.1), but MADField-cDFT — trained on the approximate cDFT data alone — is already second (6.2), ahead of the strongest learned baseline, Uni-MOF (13.0). Adapted from our paper." %}

This is what drives generalization. We also tried training on GCMC alone, without the cDFT prior, and performance dropped sharply — most of all on out-of-distribution materials. The cDFT prior is what carries MADField to material classes it never saw during fine-tuning.

## Conclusion

MADField reframes adsorption prediction as estimating the 3D equilibrium density field, then recovers gas uptake by integrating it. That one target unifies what a screening pipeline needs: ranking candidates by working capacity, accurate uptake on seen and unseen materials, faithful density fields, and a warm start that accelerates the cDFT solver itself. Multi-fidelity training is what makes it work — a broad, cheap cDFT prior corrected by a little high-fidelity GCMC — and what carries the model from MOFs to disordered solids it never trained on.

Two results are worth underlining. MADField generalizes far beyond the MOFs it was trained on — to amorphous carbons and other disordered porous materials, including frameworks with thousands of atoms and unit cells far larger than anything it saw in training, where the strongest baselines fall apart but MADField stays accurate. And the model gives back to the simulators it learned from: its predicted density is a strong initial guess for the cDFT solver, cutting the iterations to convergence by about half and recovering many cases that otherwise fail to converge at all.

Beyond the model, this work sets the right bar for the problem. Real adsorbent discovery is a needle-in-a-haystack retrieval task, and our database-scale screening benchmark measures exactly that — whether a model surfaces the rare high-capacity materials, not merely whether it reproduces the overall trend. That standard is far stricter than average uptake error, and it is the one a deployable pipeline lives or dies by. By it, capturing the broad behaviour is not nearly enough; only predictions sharp enough to rank the best materials to the very top pay off — and MADField is the first to clear that bar, by a wide margin over prior work.

## References

- Y. Kim, S. Kim, S. Ahn, H. Kim. (2026). MADField: Multi-fidelity Amortized Density Field for Adsorption in Nanoporous Materials. [arXiv:2606.21284](https://arxiv.org/abs/2606.21284).
- <span id="ref-burner2023"></span>Burner, J., et al. (2023). ARC-MOF: A Diverse Database of Metal-Organic Frameworks with DFT-Derived Partial Charges and Descriptors for Enhanced Machine Learning Predictions. *Chemistry of Materials*. <a href="#cite-burner2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-burner2026"></span>Burner, J., Marchand, O., Cicciarella, R., Gibaldi, M. & Woo, T. K. (2026). Rapid prediction of single-site adsorbate probability distributions in metal–organic frameworks using graph neural networks (DeepAPD). *Digital Discovery*. <a href="#cite-burner2026" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-sun2024"></span>Sun, Y. & Siepmann, J. I. (2024). Understanding and predicting the spatially resolved adsorption properties of nanoporous materials (SorbIIT). *Journal of Chemical Theory and Computation*, 20. <a href="#cite-sun2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-wang2024"></span>Wang, J., et al. (2024). A comprehensive transformer-based approach for high-accuracy gas adsorption predictions in metal–organic frameworks (Uni-MOF). *Nature Communications*, 15. <a href="#cite-wang2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kang2023"></span>Kang, Y., Park, H., Smit, B. & Kim, J. (2023). A multi-modal pre-training transformer for universal transfer learning in metal–organic frameworks (MOFTransformer). *Nature Machine Intelligence*, 5. <a href="#cite-kang2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-cui2023"></span>Cui, J., et al. (2023). Direct prediction of gas adsorption via spatial atom interaction learning (DeepSorption). *Nature Communications*, 14. <a href="#cite-cui2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-sarikas2024"></span>Sarikas, A. P., Gkagkas, K. & Froudakis, G. E. (2024). Gas adsorption meets deep learning: voxelizing the potential energy surface of metal–organic frameworks (RetNet). *Scientific Reports*, 14. <a href="#cite-sarikas2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-lin2025"></span>Lin, E., Zhong, Y., Chen, G. & Deng, S. (2025). Unified physio-thermodynamic descriptors via learned CO₂ adsorption properties in metal–organic frameworks (IsothermNet). *npj Computational Materials*, 11. <a href="#cite-lin2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-evans1979"></span>Evans, R. (1979). The nature of the liquid-vapour interface and other topics in the statistical mechanics of non-uniform, classical fluids. *Advances in Physics*, 28(2):143–200. <a href="#cite-evans1979" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-thyagarajan2020"></span>Thyagarajan, R. & Sholl, D. S. (2020). A Database of Porous Rigid Amorphous Materials. *Chemistry of Materials*. <a href="#cite-thyagarajan2020" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-nobel2025"></span>The Royal Swedish Academy of Sciences. (2025). The Nobel Prize in Chemistry 2025 — Susumu Kitagawa, Richard Robson, Omar M. Yaghi. [NobelPrize.org](https://www.nobelprize.org/prizes/chemistry/2025/). <a href="#cite-nobel2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-wilmer2012"></span>Wilmer, C. E., et al. (2012). Large-scale screening of hypothetical metal–organic frameworks. *Nature Chemistry*, 4. <a href="#cite-wilmer2012" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-adams1975"></span>Adams, D. J. (1975). Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. *Molecular Physics*, 29. <a href="#cite-adams1975" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-dubbeldam2016"></span>Dubbeldam, D., Calero, S., Ellis, D. E. & Snurr, R. Q. (2016). RASPA: molecular simulation software for adsorption and diffusion in flexible nanoporous materials. *Molecular Simulation*, 42. <a href="#cite-dubbeldam2016" class="reversefootnote" role="doc-backlink">↩</a>
- Rosen, J., et al. (2021). Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery (QMOF). *Matter*.

---

[^ap]: Average precision summarizes how well a ranking surfaces rare positives near the top. It is the area under the precision–recall curve, so a higher value means the high-capacity targets appear earlier in the screened list.

[^kerogen]: Kerogen is the disordered, insoluble organic matter dispersed in sedimentary rock — the precursor to oil and gas, and a natural microporous adsorbent whose pore structure differs sharply from a crystalline MOF.
[^baselines]: Learned uptake baselines: Uni-MOF ([Wang et al., 2024](#ref-wang2024)), MOFTransformer (<span id="cite-kang2023"></span>[Kang et al., 2023](#ref-kang2023)), DeepSorption (<span id="cite-cui2023"></span>[Cui et al., 2023](#ref-cui2023)), RetNet (<span id="cite-sarikas2024"></span>[Sarikas et al., 2024](#ref-sarikas2024)), and IsothermNet (<span id="cite-lin2025"></span>[Lin et al., 2025](#ref-lin2025)); density-field baselines: DeepAPD ([Burner et al., 2026](#ref-burner2026)) and SorbIIT ([Sun & Siepmann, 2024](#ref-sun2024)).
