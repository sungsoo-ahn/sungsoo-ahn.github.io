---
layout: post
title: "Cross-Domain Algorithmic Discovery with an AI Co-Scientist"
date: 2026-06-19
last_updated: 2026-06-19
description: "An AI co-scientist reshaped MaskGIT, a vision model, into a state-of-the-art algorithm for crystal structure prediction."
post_type: research
authors: ["Kiyoung Seong"]
categories: [machine-learning]
tags: [ai-scientist, generative-models, crystal-structure-prediction, materials, masked-generative-models]
toc:
  sidebar: left
related_posts: false
published: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em
    >Note: This post is about MaskGXT (paper link TODO), a crystal structure
    prediction method whose core formulation was found not by a human but by an
    AI co-scientist that searched for ideas across machine-learning fields. The
    <a href="https://kiyoung98.github.io/NanoCSP-agent/">interactive search
    tree</a> is the real search log from that run — every node is a
    model the agent trained and scored. I wrote this as a dissemination piece for
    a general ML audience; the crystallography is kept light. Corrections are
    welcome.</em
  >
</p>

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_hero.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>A vision model that predicts crystals.</strong> An AI co-scientist carries a masked-prediction idea across domains: from filling in masked patches of an image (left, the vision world of MaskGIT) to filling in the sites of a crystal lattice (right, the materials world of crystal structure prediction)." %}

Here is a sentence that should not be true: a masked image-generation model from
computer vision is, with the right adaptation, a state-of-the-art crystal
structure predictor. The stranger part is how we found out. The connection was
not proposed by a materials scientist reading the vision literature. It was found
by an **AI co-scientist** — an autonomous agent that, given the goal of
generating crystals from chemical compositions, searched across generative
modeling ideas from many fields and decided that **MaskGIT**, a masked
generative model from vision <d-cite key="chang2022maskgit"></d-cite>, was worth
trying on crystals.

That bet paid off. The resulting model, the **Masked Generative Crystal
Transformer (MaskGXT)**, sets a new state of the art on standard crystal
structure prediction benchmarks. This post is about both halves of that story:
the result, and the discovery process that produced it.

## The result first

Crystal structure prediction (CSP) asks: given a chemical composition, what
stable crystal does it form? On the **MP-20 polymorph split** — the hardest of
the standard benchmarks, because each composition can crystallize into several
distinct structures — MaskGXT reaches **79.06%** METRe accuracy, against
**70.87%** for the strongest baseline we evaluated. It also takes the best match
rate on standard MP-20 and on the larger MPTS-52 benchmark.

| Benchmark | MaskGXT | Best baseline |
| --- | --- | --- |
| MP-20 polymorph split, METRe ↑ | **79.06%** | 70.87% |
| MP-20, METRe ↑ | **74.78%** | 70.45% |
| MP-20, match rate ↑ | **73.79%** | 69.83% |
| MPTS-52, match rate ↑ | **36.75%** | 28.77% |

A single MaskGXT model wins every column, and the margin is largest exactly where
it is hardest — recovering *multiple* valid structures per composition.[^metre]

## What problem is this, really?

A crystal is atoms repeated periodically in space. To specify one you need the
unit cell (a lattice), where each atom sits inside it (fractional coordinates),
and the symmetry of the repeating pattern (a space group and Wyckoff positions).
Two complications make CSP hard. First, coordinates live on a torus — position
0.99 is right next to position 0.01, because the cell wraps around. Second,
**polymorphs**: the same formula, say TiO₂, can form several different stable
crystals, so a good method must propose a *diverse* set, not one guess.

Most recent CSP methods are continuous generative models — diffusion or flow
matching that gradually denoise coordinates. MaskGXT takes a different route,
and that route came from outside the field.

## How it was discovered

The agent ran a tree-structured search. Each node is one candidate model: the
agent wrote the code, trained it under a tight two-hour budget, scored it on a
validation metric, and decided what to try next. Its most important move is a
deliberate **cross-domain transfer** step — instead of tuning known CSP methods,
it surveyed the broader generative-modeling literature across vision, language,
sequence modeling, and more, looking for a framework *not yet applied to
crystals* with a credible mechanism for the problem.

In its opening round it proposed fourteen paradigms from different corners of ML:
a VQ-token autoregressive transformer (from language), a Bayesian Flow Network, a
Schrödinger-bridge model, a few-step consistency model with a Diffusion
Transformer (from vision), a GFlowNet, a Mamba/SSM interpolant (from sequence
modeling), and — as cell #1 of its proposal list — *masked discrete diffusion,
explicitly citing MaskGIT from vision*.

You can explore the **actual search tree** interactively at
**[kiyoung98.github.io/NanoCSP-agent](https://kiyoung98.github.io/NanoCSP-agent/)**.
Colour is run status, node size is the validation score, and clicking a node
shows what that trial changed. The masked-generative lineage is expanded by
default; the other paradigms are collapsed next to the root — click them open to
see how each one did.

The tree tells the first insight on its own. Of the fourteen cross-domain
paradigms, three crashed outright and the rest stalled between roughly 0.03 and
0.36 validation score. Only one — masked generative modeling — climbed past 0.65
and kept going. **Cross-domain transfer is a gamble; the agent's real
contribution was running that gamble fourteen times, cheaply, in parallel, and
keeping only the branch that worked.** No single human would have bet on
fourteen paradigms at once.

There is a second insight hiding in which paradigms lost. The losers include
several of the trendiest tools in generative modeling — consistency models,
Schrödinger bridges, GFlowNets, state-space models. The winner is a 2022 vision
model. Newer is not better; **alignment between the method and the structure of
the problem is what wins**, and masked discrete decoding happens to fit crystals
unusually well.

Crucially, the run was not fully autonomous. At a few moments the agent received
**sparse, high-level human hints** — domain knowledge it could not be expected to
have, such as "crystal symmetry should be reflected in the representation,"
"non-i.i.d. sampling can improve polymorph coverage," and "recover sub-bin
coordinate precision." The agent decided *how* to realize each one. This is the
honest framing of the work: it is a case study in human–AI algorithm discovery,
not a robot scientist.

## What MaskGXT is

The instantiated model treats a crystal as a sequence of **discrete tokens** —
one space group token, six lattice tokens, and five tokens per atom site
(quantized coordinates, a Wyckoff token, an atom type) — and predicts the masked
ones with a transformer, the way MaskGIT fills in masked image patches. At
generation time it fills the highest-confidence tokens first by greedy decoding,
and uses **space-group-stratified sampling** to spread its guesses across
different symmetries, which is what drives the large polymorph-coverage gain.

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_overview.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="MaskGXT represents a crystal — lattice, coordinates, space group, and Wyckoff positions — as one discrete token sequence, then decodes it by confidence-ranked masked prediction. Adapted from the MaskGXT paper." %}

One detail captures the human–AI collaboration nicely. A generic trick from
language modeling, **label smoothing**, *hurt* when applied naively in the
search — it diluted the discrete training signal. The final model keeps the idea
but reshapes it for the domain: an *ordinal, circular* label smoothing that
respects the fact that coordinate bins wrap around and that neighboring bins are
genuinely close. Off-the-shelf application failed; the domain-aware version
helped. That gap — between borrowing an idea and fitting it to the problem — is
where the human hints mattered most.

## What this generalizes to

The most transferable lesson is not "use masked models for crystals." It is
about *when* an AI co-scientist is useful. This search worked because CSP offers
**cheap, fast, and well-aligned validation**: a candidate can be trained in
hours and scored against a metric that genuinely tracks the goal. That tight
feedback loop is what let the agent run fourteen cross-domain bets and a hundred
follow-up refinements without a human in the loop for each one. A safety gate
that ran a few training steps caught models that diverged — for example, a
PFGM++ variant whose loss exploded near its noise floor — and pruned them before
wasting a full run.

So the recipe is portable: in any domain with a cheap, aligned validation signal,
an AI co-scientist can scan ideas from *other* fields, try the ones with a
plausible mechanism, and — paired with a little human domain guidance at the
right moments — surface a transferable principle a specialist might never have
imported. MaskGXT is one worked example. The interactive tree above is what that
process actually looks like.

## References

- Chang et al., 2022. *MaskGIT: Masked Generative Image Transformer.* CVPR.

[^metre]: METRe (match-everyone-to-reference) counts a reference structure as
    recovered if *any* generated structure of the same composition matches it,
    so it rewards covering all polymorphs of a composition rather than just one.
    Match rate is the simpler one-structure-per-reference version.
