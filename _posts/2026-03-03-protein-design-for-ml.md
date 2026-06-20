---
layout: post
title: "Protein Design"
date: 2026-03-03
last_updated: 2026-06-20
description: "An introduction to protein structure, function, and computational design — from amino acids to the RFDiffusion/ProteinMPNN pipeline."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: ml-for-science
series_title: "ML for Science Foundations"
series_description: "A guided route through scientific ML topics: quantum chemistry, equivariant molecular models, electrocatalysis, and protein design."
series_order: 4
categories: [science]
tags: [protein-design, structural-biology, machine-learning, generative-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post is written for ML researchers entering the protein design field. It covers the structural biology you need to read RFDiffusion and ProteinMPNN papers, understand what pLDDT and ipTM measure, and follow a design project from target definition through experimental validation. No biology background is assumed.</em>
</p>

## Introduction

Small-molecule drugs dominate pharmacology, but they have hard limits. A typical small molecule has a molecular weight under 500 Da and contacts its target over ~300 Å$$^2$$ (1 Å = 0.1 nm). Proteins are 100–1000x larger, with binding interfaces spanning 1,000–2,000 Å$$^2$$. That extra contact area gives proteins more selectivity: they encode more geometric and chemical information at the interface, so they can distinguish targets that small molecules cannot.

Many disease-relevant targets — protein–protein interactions, flat surfaces, disordered regions — are "undruggable" by small molecules because there is no deep binding pocket to exploit. Designed proteins can bind these surfaces directly. The prefusion-stabilized COVID-19 spike (Hsieh et al., 2020) was used in multiple approved vaccines, and de novo designed miniprotein inhibitors achieved picomolar (extremely tight) binding to the spike in lab assays (Cao et al., 2020).

For ML researchers, the appeal is structural. Protein design is a well-defined generative modeling problem: the input is a functional specification (target structure, binding constraints), and the output is a sequence of discrete tokens (amino acids) that must satisfy continuous geometric constraints (3D folding).

The training data is the Protein Data Bank, with ~200,000 experimentally solved structures, supplemented by billions of sequences from genomic databases. Structure prediction (AlphaFold) provides a fast oracle for evaluating designs. The experimental feedback loop is also tight: a design campaign can move from computation to lab results in weeks, not years.

Before 2020, computational protein design relied on Rosetta's physics-based energy function and Monte Carlo sampling: slow, expensive, and low-throughput. Between 2021 and 2023, AlphaFold2, ProteinMPNN, and RFDiffusion replaced the core pipeline steps with learned models, increasing success rates from ~1% to ~10–30% and reducing computation from CPU-weeks to GPU-hours. This created an opportunity: the tools work, but they are far from optimal, and the design space is large. Protein design is now a field where ML contributions can have immediate, measurable experimental impact.

The background splits into two parts: the biology and the computational infrastructure around it.

### Overview

A modern design campaign is a pipeline: **RFDiffusion** generates candidate backbone structures conditioned on the target, **ProteinMPNN** designs amino acid sequences for each backbone, **AlphaFold** predicts whether each sequence folds as intended, and the top candidates go to the lab. Roughly 10,000 backbones enter; about 5 confirmed binders come out.

The biology explains why each step works and fails: what proteins are made of, what forces hold the shape together, what makes a good binding interface, and what physical constraints the pipeline must satisfy. The organizing frame is the **sequence → structure → function** triangle.

## What a Protein Is

A protein is a chain of **amino acids**, small molecules linked end-to-end like beads on a string. There are 20 standard types, each identified by a one-letter code (A for alanine, M for methionine, etc.), so a protein sequence reads like `MKVLWAGG...`: a string over a 20-letter alphabet.

Every amino acid shares the same backbone atoms — a repeating N-C$$_\alpha$$-C unit — but differs in its **side chain**, the group that branches off at each C$$_\alpha$$. Side chains vary in size, charge, and hydrophobicity.[^hydrophobicity] This chemical diversity gives proteins their functional range.

[^hydrophobicity]: The 20 amino acids split roughly into four groups: nonpolar/hydrophobic (G, A, V, L, I, M, F, W, P), polar uncharged (S, T, N, Q, Y, C), positively charged (K, R, H), and negatively charged (D, E). The nonpolar residues drive folding by burying themselves away from water.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_amino_acids.svg" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="650px" zoomable=true caption="The standard amino acids share a common backbone but differ in side-chain chemistry. Those side chains determine charge, hydrophobicity, and geometry, which is why sequence controls folding and binding. From Wikimedia Commons (CC BY-SA 3.0)." %}

### Structure Hierarchy

Proteins organize at four levels:

1. **Primary structure** — the amino acid sequence itself.
2. **Secondary structure** — local repeating patterns. Alpha helices are coiled springs stabilized by hydrogen bonds between residue $$i$$ and residue $$i+4$$ (a "residue" is one amino acid in the chain). Beta sheets are flat arrangements of adjacent strands connected by hydrogen bonds. Loops are the flexible connectors between them.
3. **Tertiary structure** — the full 3D shape of a single chain, with helices, sheets, and loops packed together.
4. **Quaternary structure** — the assembly of multiple chains into a complex.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_protein_structure_levels.svg" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="650px" zoomable=true caption="Protein structure is organized from sequence to local motifs, full-chain folds, and multi-chain assemblies. Each level constrains the next: sequence creates local geometry, local geometry packs into a fold, and folds assemble into function. From Wikimedia Commons (public domain)." %}

{% include figure.liquid loading="eager" path="assets/img/blog/pd_secondary_structure_source.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Alpha helices and beta sheets are the main recurring local protein geometries. Hydrogen bonds stabilize these patterns, turning a flexible chain into predictable structural elements. From Wikimedia Commons (CC BY-SA 4.0)." %}

A **fold** (or topology) is the overall arrangement of secondary-structure elements. Two proteins with completely different sequences can share the same fold: different bricks, same floor plan. A **domain** is a compact, independently folding unit within a larger protein; many proteins consist of multiple linked domains.

### Evolution and MSAs

Related proteins across species, called **homologs**, share a common ancestor. Lining up homologous sequences produces a **multiple sequence alignment (MSA)**, which reveals **conservation**: positions that remain fixed across millions of years of evolution are usually structurally or functionally critical.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_msa_source.gif" class="img-fluid rounded z-depth-1" avoid_scaling=true zoomable=true caption="A multiple sequence alignment lines up homologous proteins across species. Conserved columns mark positions where evolution strongly constrained the allowed amino acids. From Wikimedia Commons (CC BY-SA 3.0)." %}

MSAs also reveal **coevolution** — pairs of positions that mutate together, implying physical contact. This was the key insight behind early contact prediction methods and a core input to AlphaFold2. For ML researchers, MSAs are the protein equivalent of a large unlabeled dataset: they encode structural constraints without explicit 3D labels.

The organizing principle of structural biology is the **sequence → structure → function** triangle: sequence determines the 3D fold, and the fold determines what the protein does.

---

## What Holds the Shape Together

A protein folds because the folded state is thermodynamically favorable. The dominant driving force is the **hydrophobic effect**: nonpolar side chains are energetically penalized when exposed to water, so the chain collapses to bury them in a tightly packed interior, the **hydrophobic core**. Disrupting this core usually destroys the protein.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_hydrophobic_core_source.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="The hydrophobic effect drives nonpolar residues away from water and into the protein core. Folding lowers the solvent exposure of hydrophobic side chains while leaving polar residues on the surface. From Wikimedia Commons (CC BY-SA 3.0)." %}

On top of the hydrophobic effect, several other forces contribute:

- **Hydrogen bonds** — weak electrostatic attractions between donor and acceptor atoms. Individually weak, but collectively essential for secondary structure. Every backbone N-H and C=O must either form an H-bond or be exposed to water; an unsatisfied H-bond donor buried in the core is energetically costly.
- **Salt bridges** — attractions between positively charged residues (Lys, Arg) and negatively charged ones (Asp, Glu). Contribute to stability on the protein surface.
- **Disulfide bonds** — covalent bonds between two cysteine residues. Molecular staples that physically lock distant parts of the chain together. Common in antibodies and secreted proteins.
- **Van der Waals interactions** — weak attractions between atoms at close range. Negligible individually but significant when combined, as thousands of atoms pack tightly in the core.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_protein_interactions.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Tertiary structure is stabilized by several side-chain interactions, including hydrophobic packing, hydrogen bonds, salt bridges, and disulfide bridges. A design fails when these interactions are missing, geometrically strained, or placed in the wrong environment." %}

### Stability and Failure Modes

**Stability** measures how hard it is to unfold a protein. The **melting temperature** (T$$_m$$) is the temperature at which half the protein population is unfolded — higher T$$_m$$ means a more robust protein. Designed proteins typically need T$$_m$$ > 60°C to be useful.[^tm]

[^tm]: T$$_m$$ is measured by heating the protein while monitoring secondary structure (e.g., circular dichroism). A well-designed miniprotein might reach T$$_m$$ > 90°C.

The most common failure mode in protein design is **aggregation**: proteins stick to each other and form useless clumps, like egg whites cooking. This usually happens because hydrophobic patches that should be buried are instead exposed on the surface. **Solubility** — whether the protein stays dissolved in water — is closely related. An insoluble protein is useless regardless of how good it looks in simulation.

---

## What Proteins Do — Binding and Function

Most proteins function by **binding** to other molecules — other proteins, small molecules, DNA, or metal ions. The strength of binding is quantified by the **dissociation constant** K$$_d$$:[^kd] lower K$$_d$$ means tighter binding (the two molecules are harder to pull apart).

[^kd]: K$$_d$$ is the concentration at which half the binding sites are occupied at equilibrium. It has units of molar concentration. K$$_d$$ = 1 nM means the binder holds on tightly even at very low concentrations; K$$_d$$ = 1 μM is moderate affinity typical of transient interactions.

| K$$_d$$ range | Binding strength | Typical context |
|:---|:---|:---|
| < 1 nM | Very tight | Therapeutic antibodies |
| 1–100 nM | Tight | Designed binders, drugs |
| 100 nM – 1 μM | Moderate | Signaling interactions |
| > 1 μM | Weak | Transient contacts |

**Specificity** is equally important: a binder that grabs everything is useless. The physical contact surface between two binding partners is the **binding interface**, typically spanning 1,000–2,000 Å$$^2$$. Interface residues do not contribute equally; a handful of **hotspot residues** provide most of the binding energy. Identifying target-surface hotspots is the first step of binder design.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_binding_interface_source.jpg" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="620px" zoomable=true caption="This RNase inhibitor-RNase complex shows a protein-protein binding interface. The interface works because the two surfaces are geometrically and chemically complementary. From Wikimedia Commons (CC BY 3.0; PDB: 1DFJ)." %}

For antibodies specifically, the target surface is called the **epitope** and the matching surface on the antibody is the **paratope**. Different antibodies can target different epitopes on the same target protein.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_antibody_structure.svg" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="620px" zoomable=true caption="An IgG antibody separates target recognition from immune-effector function. The antigen-binding sites sit at the tips of the Fab arms, where CDR loops contact the target. From Wikimedia Commons (CC BY-SA 3.0)." %}

Beyond binding, **enzymes** catalyze chemical reactions. Their **active sites**, small pockets with precisely positioned catalytic residues, accelerate reactions by factors of 10$$^6$$–10$$^{12}$$. Enzyme design is harder than binder design because it requires exact 3D geometry, not merely a good surface fit.

Proteins can also undergo **conformational changes** — shifts in 3D structure triggered by binding. This is how signals propagate through biological systems: binding at one site rearranges the protein to expose or hide a distant functional site.

---

## How Protein Designers Think

ML researchers sometimes imagine protein designers relying on vague biological intuition, built from years of staring at crystal structures. The reality is more specific. Protein designers think in rules. When they inspect a structure, they ask concrete questions: Are the hydrophobic residues buried? Are the hydrogen-bond donors satisfied? Is the backbone in a favorable region of Ramachandran space?[^ramachandran] Much of the expertise is not mystical pattern recognition; it is a mental checklist of physical constraints.

[^ramachandran]: The Ramachandran plot charts the two backbone dihedral angles ($$\phi$$, $$\psi$$) for each residue. Some angle combinations cause atomic clashes and are forbidden. A well-designed protein has all residues in the "allowed" regions of this plot.

Many of these heuristics translate naturally into energy-function terms, loss functions, and model inductive biases.

### Energy Landscapes

The folded state of a protein sits at the bottom of a **free energy landscape**, a surface over all possible conformations. Design means finding sequences whose energy minimum matches a target structure. It is an optimization problem.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_energy_landscape.svg" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="620px" zoomable=true caption="A folding funnel depicts many unfolded conformations collapsing toward a low-energy native state. Protein design asks for a sequence whose energy minimum is the desired target structure. Figure by Thomas Splettstoesser, Wikimedia Commons (CC BY-SA 3.0)." %}

### Packing Geometry

When a designer says "the hydrophobic core is well-packed," they mean atoms fill space tightly with no voids. This is quantified by packing density metrics and reflected in the van der Waals energy term in Rosetta. Empty space in the core means unfavorable energetics.

### Hydrogen Bond Networks

"Satisfying all hydrogen bonds" has a precise meaning: every buried backbone or side-chain donor/acceptor must have a partner. An unsatisfied H-bond donor buried in the core costs roughly 5 kcal/mol, enough to destabilize the entire protein. Designers check these systematically.

### Shape Complementarity

Binding interfaces are scored by geometric fit using the **Sc score**, which measures how well the two surfaces fit together (like interlocking fingers). Sc = 1.0 is a perfect fit; most natural protein–protein interfaces score 0.6–0.7. This is a geometric computation, not a subjective judgment.

### Systematic Enumeration

Rosetta's design protocol is **Monte Carlo sampling over sequence space with a physics-based energy function**. At each position, Rosetta tries different amino acid identities and side-chain rotamers,[^rotamer] accepts or rejects changes based on the energy function, and iterates. The transition from Rosetta to ML-based design was not a complete change in goals; it replaced a physics-based MCMC optimizer with learned models.

[^rotamer]: A rotamer is a preferred side-chain conformation. Side chains don't rotate freely — they snap into a discrete set of low-energy angles, like a dial with set positions. Rosetta's rotamer library catalogs these preferred conformations for each amino acid type.

### Biologist Reasoning Is ML Reasoning

Three examples of "structural biology intuition" translated to ML terms:

| What a biologist says | What they mean computationally |
|:---|:---|
| "This helix should be amphipathic (hydrophobic on one side, hydrophilic on the other)" | The hydrophobic moment vector should point inward — a periodic constraint on sequence hydrophobicity with period 3.6 (the helix repeat) |
| "The core isn't packed well" | Atoms have too much empty space — packing density below threshold, Rosetta vdW energy too high |
| "That loop will be floppy" | High B-factor in crystal structures / low pLDDT in AlphaFold — the model is uncertain about this region's conformation |

The domain knowledge ML researchers admire in structural biologists is largely a set of **quantitative constraints and heuristics** that map to loss terms and architectural priors. When a biologist says something about a structure, there is usually a computable quantity behind it.

---

## The Design Problem — Formulations and Types

Protein design decomposes into three ML problem formulations:

{% include figure.liquid loading="eager" path="assets/img/blog/pd_design_problems.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Protein design splits into forward folding, inverse folding, and de novo backbone generation. The direction of the mapping changes, but each problem links sequence, structure, and functional constraints." %}

**Forward folding** (structure prediction). Input: amino acid sequence. Output: 3D atomic coordinates. Model: AlphaFold2. This is the "check your work" step: given a designed sequence, does the predicted fold match the intended structure?

**Inverse folding** (sequence design). Input: 3D backbone coordinates. Output: amino acid sequence that folds into that backbone. Model: ProteinMPNN. This is the core design step: "I drew the blueprint; now find the bricks."

**De novo backbone generation**. Input: functional specification (target protein, hotspot residues, symmetry constraints). Output: new backbone structure. Model: RFDiffusion. This is the generative step: "create a new shape that binds this target."

The standard validation loop ties these together: generate a backbone (RFDiffusion) → design a sequence for it (ProteinMPNN) → predict the structure of that sequence (AlphaFold) → compare the prediction to the intended backbone. If they match, the design is self-consistent.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_self_consistency.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Self-consistency checks whether a designed sequence folds back to the intended backbone. Low scRMSD means the sequence encodes the target structure rather than merely fitting the design model." %}

### What People Design

Design targets:

- **Miniproteins** (40–80 residues) — small, stable, easy to produce. The "Hello World" of protein design.
- **De novo binders** — proteins that grab a specific target surface. The most active area in computational design.
- **Antibodies** — Y-shaped immune proteins. ~100 approved as drugs. Design focuses on the six **CDR loops** (especially CDR-H3), which contact the antigen. The stable **framework regions** hold the CDRs in position.
- **Nanobodies** (VHH) — single-domain antibodies from camelids (camels, llamas). Smaller, simpler to engineer computationally.
- **Peptides** (< 40 residues) — short, often flexible chains. Many drugs are peptides (e.g., insulin).
- **Enzymes** — proteins that catalyze chemical reactions. Harder than binder design: requires precise 3D geometry at the active site.
- **Vaccine immunogens** — engineered proteins that train the immune system. The COVID-19 spike protein vaccines are a high-profile example.

### Design Constraints

Real designs are constrained:

- **Hotspot residues** — "the designed binder must contact these specific residues on the target."
- **Motif scaffolding** — "build a stable protein around this functional fragment, holding it in the correct 3D position."
- **Symmetric design** — "generate identical subunits that assemble into a ring, cage, or icosahedron (a 20-faced sphere-like shell)."
- **Contig notation** — RFDiffusion's input format. Example: `A1-100/0 30-50/B1-80` means "keep residues 1–100 of chain A, design 30–50 new residues, keep residues 1–80 of chain B."

---

## The Computational Toolkit

Seven tools define the current protein design stack. For each, the key questions are what goes in, what comes out, and when to use it.

### Rosetta

Rosetta is the classic physics-based suite, developed over more than 20 years. It evaluates designs with an energy function that sums van der Waals packing, electrostatics, hydrogen bonds, solvation, and backbone geometry terms. Output is reported in **Rosetta Energy Units (REU)**, where lower is better. Rosetta is still widely used for scoring and refinement, even as ML tools handle generation.

### AlphaFold2 / AlphaFold3

**Input:** sequence (+ MSA for AF2). **Output:** predicted 3D structure with per-residue and per-pair confidence scores.

- **pLDDT** — per-residue confidence (0–100). High pLDDT means the model is certain about local structure.
- **pTM** — overall fold confidence.
- **ipTM** — interface confidence for protein complexes. The key metric for binder design.
- **PAE** — predicted aligned error matrix. Shows expected positional error between all residue pairs. For binder design, check the inter-chain PAE block: low values mean the model is confident about the binding mode.

AF2 handles single chains; AF2-Multimer extends it to multi-chain protein complexes. AF3 extends to protein–nucleic acid and protein–small molecule complexes.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_alphafold_overview_source.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="AlphaFold predicts 3D structure from sequence and evolutionary context. The model combines MSA/template information with a structure module, then reports coordinates and confidence estimates. From Jumper et al. (2021), CC BY 4.0." %}

### ESMFold / ESM3

Meta's protein language models offer a faster structure-prediction route. ESMFold predicts structure from a **single sequence**, with no MSA required, making it much faster than AlphaFold. ESM3 is more general, jointly modeling sequences, structures, and functional annotations. These models are useful for rapid self-consistency screening at scale: when you have 80,000 candidate sequences, filter with ESMFold first and send only the top candidates to AlphaFold.

### ProteinMPNN

**Input:** 3D backbone coordinates (as a graph of residue positions). **Output:** amino acid probability distribution at each position. A message-passing neural network that designs sequences for given backbones. Fast, accurate, and the standard inverse folding tool. Typically generates 8–16 sequences per backbone.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_proteinmpnn_overview_source.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="ProteinMPNN designs sequences for fixed backbones using geometric message passing. Random decoding order and tied positions let the same model handle fixed-context, symmetric, and multi-chain design problems. From Dauparas et al. (2022), CC BY 4.0." %}

### RFDiffusion

**Input:** target protein structure + conditioning constraints (hotspot residues, motifs, symmetry). **Output:** new backbone coordinates. A denoising diffusion model over backbone coordinates — analogous to image diffusion models but operating in SE(3) coordinate space. The most powerful current tool for de novo backbone generation.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_rfdiffusion_overview_source.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="RFDiffusion generates protein backbones through iterative denoising. Conditioning lets the same diffusion model handle unconditional generation, motif scaffolding, symmetry, and binder design. From Watson et al. (2023), CC BY 4.0." %}

### Boltz / Boltz2

Boltz models are alternative structure predictors that provide a second opinion on AlphaFold. Agreement between two separately trained predictors is not experimental proof, but it is a useful computational sanity check before spending lab effort.

### BoltzGen

BoltzGen (Stark et al., 2025) is worth one sentence here because it keeps the same design loop but couples the steps more tightly. The core workflow is still generate, sequence, predict, filter, and validate. Newer systems mostly change how tightly those steps interact and which failure modes they catch before experiments.

### What Does "Good" Look Like?

A reference table for interpreting computational metrics:

| Metric | Good threshold | What it means |
|:---|:---|:---|
| pLDDT | > 80 | High confidence in local structure |
| ipTM | > 0.8 | Confident interface prediction |
| scRMSD | < 2 Å | Self-consistent design (sequence encodes the intended fold) |
| K$$_d$$ | < 100 nM | Tight binding (experimental measurement) |
| Rosetta energy | Negative, comparable to natural proteins | Physically reasonable packing and interactions |

---

## The Design Workflow

The tools above assemble into a five-step pipeline. The important constraint is the **numbers at each stage**: the funnel from millions of computational candidates to a handful of experimental hits shapes every decision in a design project.

### Step 1: Define the problem

What should the protein do? Bind a specific target surface? Catalyze a reaction? Form a symmetric cage? This determines which tools to use, which constraints to set, and how to evaluate success. For binder design, you identify the target protein, choose an epitope (the surface patch to target), and specify hotspot residues.

### Step 2: Generate backbones

RFDiffusion generates ~10,000 backbone structures conditioned on the target and constraints. Each backbone is a candidate protein shape: no sequence yet, just the 3D arrangement of backbone atoms.

### Step 3: Design sequences

ProteinMPNN designs 8–16 amino acid sequences for each backbone, yielding ~80,000–160,000 total candidates. Each is designed to fold into the target backbone.

### Step 4: Computational filtering

The self-consistency check eliminates most candidates. For each designed sequence:
1. Predict its structure with AlphaFold or ESMFold.
2. Compare the predicted structure to the intended backbone (scRMSD).
3. Check confidence metrics (pLDDT, ipTM for binders).

~1% of designs pass all filters — roughly 1,000 candidates from 100,000.

### Step 5: Experimental validation

The top ~20 candidates are ordered as synthetic genes and tested in the lab:

- **Expression** — bacteria (usually *E. coli*) produce the designed protein from synthetic DNA. Many designs fail here: the protein doesn't express or is insoluble.
- **Purification** — the protein is isolated from the bacterial cell contents. Aggregated or insoluble proteins can't be purified.
- **CD (circular dichroism)** — a quick test for secondary structure. If the CD spectrum shows no helices or sheets, the protein didn't fold.
- **SPR (surface plasmon resonance)** — measures binding affinity in real time, providing the K$$_d$$ value. The gold standard for quantifying binder quality.
- **Display methods** (phage display, yeast display) — screen millions of variants simultaneously. Each variant is displayed on the surface of a phage or yeast cell, washed over the target, and only binders stick. Used to improve initial hits.
- **Directed evolution** — the pre-ML baseline. Randomly mutate, test, keep the best, repeat. Won the 2018 Nobel Prize in Chemistry. Computationally, this is what evolutionary algorithms replicate.
- **Cryo-EM / X-ray crystallography** — determine the actual 3D atomic structure. The definitive validation: did the protein fold into the intended shape? Slow and expensive, reserved for the most promising candidates.

Typical hit rates: ~10 of 20 ordered designs express as soluble protein, and 3–5 bind the target. These numbers vary by target difficulty, but the order of magnitude is consistent across published studies.

{% include figure.liquid loading="eager" path="assets/img/blog/pd_design_funnel.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The design funnel narrows thousands of generated backbones to a small number of experimentally tested proteins. The main leverage point is improving computational filters so fewer weak candidates reach the lab." %}

The gap between computational output (millions of candidates) and experimental throughput (tens to hundreds) is the defining constraint of protein design. Every ML improvement that increases the pass rate at Step 4 reduces the cost and time of experimental campaigns. This is the field's main leverage point.

---

## References

- Cao, L., et al. (2020). De novo design of picomolar SARS-CoV-2 miniprotein inhibitors. *Science*, 370(6515), 426–431.
- Hsieh, C.-L., et al. (2020). Structure-based design of prefusion-stabilized SARS-CoV-2 spikes. *Science*, 369(6510), 1501–1505.
- Leman, J. K., et al. (2020). Macromolecular modeling and design in Rosetta: recent methods and frameworks. *Nature Methods*, 17, 665–680.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
- Dauparas, J., et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49–56.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.
- Watson, J. L., et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature*, 620, 1089–1100.
- Abramson, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493–500.
- Wohlwend, J., et al. (2024). Boltz-1: Democratizing Biomolecular Interaction Modeling. *arXiv:2408.00537*.
- Stark, H., et al. (2025). BoltzGen: Toward Universal Binder Design. *bioRxiv:2025.11.20.689494*.

### Figure sources

- Amino acids diagram (`pd_amino_acids.svg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Amino_Acids.svg), CC BY-SA 3.0.
- Protein structure levels (`pd_protein_structure_levels.svg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Main_protein_structure_levels_en.svg), public domain.
- Secondary structure (`pd_secondary_structure_source.png`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Alpha_beta_structure_(full).png), CC BY-SA 4.0.
- Multiple sequence alignment (`pd_msa_source.gif`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RPLP0_90_ClustalW_aln.gif), CC BY-SA 3.0.
- Hydrophobic interaction (`pd_hydrophobic_core_source.jpg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cartoon_of_protein_hydrophobic_interaction.jpg), CC BY-SA 3.0.
- Tertiary-structure interactions (`pd_protein_interactions.svg`): custom simplified schematic generated by `scripts/generate_protein_design_figures.py`; it replaces a visually dense Wikimedia source figure with larger labels and a blog-style palette.
- Protein-protein interface (`pd_binding_interface_source.jpg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:1dfj_RNAseInhibitor-RNAse_complex.jpg), CC BY 3.0.
- Antibody structure (`pd_antibody_structure.svg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Immunoglobulin_basic_unit.svg), CC BY-SA 3.0.
- Folding energy landscape (`pd_energy_landscape.svg`): [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Folding_funnel_schematic.svg) by Thomas Splettstoesser, CC BY-SA 3.0.
- AlphaFold overview (`pd_alphafold_overview_source.jpg`): Figure 1 from Jumper et al. (2021), via [PMC8371605](https://pmc.ncbi.nlm.nih.gov/articles/PMC8371605/), CC BY 4.0.
- ProteinMPNN overview (`pd_proteinmpnn_overview_source.jpg`): Figure 1 from Dauparas et al. (2022), via [PMC9997061](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/), CC BY 4.0.
- RFDiffusion overview (`pd_rfdiffusion_overview_source.jpg`): Figure 1 from Watson et al. (2023), via [PMC10468394](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468394/), CC BY 4.0.
- Custom diagrams (`pd_design_problems.svg`, `pd_self_consistency.svg`, `pd_design_funnel.svg`): generated by `scripts/generate_protein_design_figures.py`.
