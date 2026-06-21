---
layout: post
title: "Cross-Domain Algorithmic Discovery with an AI Co-Scientist"
date: 2026-06-19
last_updated: 2026-06-20
description: "An AI co-scientist reshaped MaskGIT, a vision model, into a state-of-the-art algorithm for crystal structure prediction."
abstract: >
  This post describes how an AI co-scientist transferred MaskGIT, a masked
  generative model from computer vision, to crystal structure prediction. The
  resulting MaskGXT model discretizes crystal structures into tokens, refines
  them with continuous offsets, and reaches state-of-the-art performance on
  standard CSP benchmarks.
post_type: research
authors: ["Kiyoung Seong"]
categories: [machine-learning]
tags: [ai-scientist, generative-models, crystal-structure-prediction, materials, masked-generative-models]
toc:
  sidebar: left
related_posts: false
published: true
---

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_hero.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>A vision model that predicts crystals.</strong> An AI co-scientist carries a masked-prediction idea across domains: from filling in masked patches of an image (left, the vision world of MaskGIT) to filling in the sites of a crystal lattice (right, the materials world of crystal structure prediction)." %}

Asked to predict crystal structures, an **AI co-scientist** reached into computer
vision and brought back a state-of-the-art algorithm. Searching across
generative-modeling ideas from many fields, the autonomous agent selected
**[MaskGIT](https://arxiv.org/abs/2202.04200)**, a masked generative model for
images <d-cite key="chang2022maskgit"></d-cite>, as worth adapting to crystals, a
connection neither a vision nor a materials specialist would naturally make.

The resulting model, the **Masked Generative Crystal Transformer (MaskGXT)** (the
X stands for crystal, after *Xtal*, the conventional shorthand), adapts MaskGIT
from images to crystals and sets a new state of the art on crystal structure
prediction. The agent found the idea, instantiated it,
and refined it largely on its own; a few sparse, high-level human hints carried it
from competitive to best-in-class. This post is about that discovery process and the model it
produced.

## A new state of the art

A single MaskGXT model sets a new state of the art on match rate, the standard
crystal structure prediction metric for whether a generated structure matches the
target, across the standard benchmarks. It also leads on METRe, a stricter,
polymorph-aware metric.[^metre]

| Benchmark | Metric | MaskGXT | Best baseline |
| --- | --- | --- | --- |
| MP-20 | match rate ↑ | **73.79%** | 69.83% |
| MPTS-52 | match rate ↑ | **36.75%** | 28.77% |
| MP-20 | METRe ↑ | **74.78%** | 70.45% |
| MP-20 polymorph split | METRe ↑ | **79.06%** | 70.87% |

Crystal structure prediction (CSP) asks: given a chemical composition, what stable
crystal does it form? A crystal is a pattern of atoms repeated
periodically in space, specified by three things: the atom types, a lattice
(the unit cell that tiles space), and the fractional coordinates of each atom,
which give its position as a fraction along the three lattice vectors so each value
runs from 0 to 1 within the cell.

This representation raises several challenges. The coordinates need high precision
to land on a stable configuration, and should follow the symmetric structure of a
real crystal, captured by its space group. The model should ideally be invariant to periodic translation,
since shifting the unit-cell origin leaves the crystal unchanged, and to atom
ordering, since the atoms have no canonical order.

To handle these challenges, most recent CSP methods use continuous diffusion or
flow models that denoise coordinates. MaskGXT takes a discrete route instead: it
quantizes the crystal into tokens and predicts them, then refines each discrete
prediction with a small continuous offset to recover the precision a coarse bin
would lose. That route originates outside the field.

## How it was discovered

{% include video.liquid path="assets/video/maskgxt_agent_anim.mp4" class="img-fluid rounded z-depth-1 mx-auto d-block" poster="assets/img/blog/maskgxt_agent.png" autoplay=true loop=true muted=true controls=true caption="<strong>The AI co-scientist.</strong> A single orchestrator searches a tree of candidate CSP methods; each node is a complete generative model. The tree grows in chronological order as the orchestrator dispatches idea, draft, debug, and improve operators; a human intervenes only sparsely, passing a high-level mechanism or objective." %}

The decisive move was a deliberate **cross-domain transfer** step. Rather than
tune known CSP methods, the agent searched the broader generative-modeling
literature for a framework *not yet applied to crystals* with a credible mechanism
for the problem. It ran this as a tree-structured search: each node is one candidate
model that the agent coded, trained under a fixed budget, scored on validation
METRe, and used to decide what to try next. Over the course of the search, the
agent proposed fourteen cross-domain generative frameworks from across ML, such as
an autoregressive transformer (language), a Masked Generative Transformer (vision),
and a Mamba/SSM interpolant (sequence modeling). The agent then focused its effort
on the most promising of these, the Masked Generative Transformer, building it into
the core formulation of MaskGXT. Explore the full search tree below, or
[open it full screen](https://kiyoung98.github.io/NanoCSP-agent/) to see every node.

<div class="row justify-content-center my-4">
  <div class="col-12">
    <div class="position-relative">
      <iframe
        src="https://kiyoung98.github.io/NanoCSP-agent/"
        title="MaskGXT search tree"
        loading="lazy"
        class="img-fluid rounded z-depth-1 d-block w-100"
        style="height: 70vh; border: 0;"
        allowfullscreen></iframe>
      <a href="https://kiyoung98.github.io/NanoCSP-agent/" target="_blank" rel="noopener"
         class="btn btn-light position-absolute"
         style="top: 0.5rem; right: 0.5rem; padding: 0.15rem 0.5rem; font-size: 0.7rem; line-height: 1.4; opacity: 0.9;">Open ↗</a>
    </div>
  </div>
</div>

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_trajectory.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>The research trajectory toward MaskGXT.</strong> Validation METRe against the number of trials; the black step line is the running best. The three shaded bands are the search stages, with the per-candidate budget escalating from 2h to 12h training and then 30m of sampling tuning." %}

Although the agent discovered the core formulation on its own, the path to state of
the art required sparse human intervention. On its own the agent already reached a
roughly 70% match rate on MP-20, competitive with the prior state of the art; the
interventions below carried it the rest of the way. In each, we proposed a mechanism
or an objective, pointed to the relevant prior work, and left the implementation to
the agent. Four were decisive. Three supplied a **mechanism**, a piece of knowledge
from crystal generative modeling the agent was missing:

- **Symmetry tokens.** To improve structural accuracy, we proposed encoding symmetry,
  citing DiffCSP++ and WyFormer.
- **Symmetry-preserving augmentation.** To exploit that symmetry in training, we
  proposed orbit permutation over a symmetry-based atom ordering, citing MCFlow.
- **Stratified sampling.** To cover the multiple polymorphs a composition can form,
  we proposed sampling in a non-i.i.d. way.

The fourth supplied an **objective**: we asked the agent to recover the precision
lost to discretization, and it arrived at a continuous sub-bin offset on its own.

## What MaskGXT is

Once the search concluded, we organized the code the agent had written into the core
methodology we describe here.

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_overview.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>How MaskGXT works.</strong> (a) Tokenizing a crystal: one space group token, six lattice tokens, and five tokens per atom site. (b) Training reconstructs randomly masked tokens. (c) Sampling branches over space groups to cover polymorphs, then greedily unmasks the rest." %}

The transferred core is a discrete masked formulation. MaskGXT represents a crystal
as a sequence of discrete tokens and predicts the masked ones with a transformer,
the way MaskGIT fills in masked image patches. Three components adapt this core to
crystals. **Tokenization** maps the lattice, fractional coordinates, space group,
and Wyckoff positions to tokens, encoding crystal symmetry explicitly through the
space group and Wyckoff tokens, with coordinates quantized on a circle so that bins
near 0 and 1 stay adjacent. **Ordinal, circular label smoothing** trains the
coordinate stream to treat neighboring bins as close, respecting that circular
geometry. **Sub-bin refinement** then predicts a small continuous offset within each
bin, recovering the precision that discretization would otherwise lose.

The discrete formulation also brings two advantages a continuous model lacks.
Because the output is a finite vocabulary, **greedy decoding** can unmask the
highest-confidence tokens first and return the single most probable structure. And
because the space group is itself a token, **space group stratified sampling** can
fix it to each likely value and branch generation, so a single composition yields
distinct polymorphs, the source of the large polymorph-coverage gain.

## Why it worked

This search succeeded because CSP offers **cheap, well-aligned validation**:
a candidate trains in hours and is scored by a metric that tracks the goal. That tight loop let the agent run fourteen cross-domain bets
and roughly five hundred refinements without a human judging each one.
MaskGXT is one worked example, and it points toward a shift in how such research
gets done. The AI co-scientist carries out the search; the human sets the goal and intervenes
at a few decisive moments. The loop is harder to close in domains with long training cycles, physical experiments,
or weak proxy objectives; developing more accurate proxy evaluations and determining
how to incorporate human steering in such settings remain open problems.

## References

- Chang et al., 2022. *MaskGIT: Masked Generative Image Transformer.* CVPR.
- Jiao et al., 2024. *Space Group Constrained Crystal Generation* (DiffCSP++). ICLR.
- Kazeev et al., 2025. *Wyckoff Transformer: Generation of Symmetric Crystals* (WyFormer). ICML.
- Martirossyan et al., 2025. *All That Structure Matches Does Not Glitter* (METRe). NeurIPS.
- Seong et al., 2026. *Multimodal Crystal Flow: Any-to-Any Modality Generation for Unified Crystal Modeling* (MCFlow). ICML.

[^metre]: A composition can crystallize into several stable structures, or
    *polymorphs*. Match rate asks for one correct structure; METRe
    (match-everyone-to-reference) is stricter, rewarding recovery of *all* of a
    composition's polymorphs. The polymorph split is the benchmark built to stress
    it, and is where MaskGXT's margin is widest.
