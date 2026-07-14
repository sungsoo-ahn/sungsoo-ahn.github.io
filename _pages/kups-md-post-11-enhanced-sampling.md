---
layout: post
permalink: /kups-md-tutorials/post-11-enhanced-sampling/
title: "How Do Adaptive and Nonequilibrium Enhanced-Sampling Methods Work?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible enhanced-sampling diagnostic for metadynamics-style bias filling, nonequilibrium work, Jarzynski estimates, and Crooks crossings."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 11
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 11
categories: [science]
tags: [molecular-dynamics, enhanced-sampling, metadynamics, nonequilibrium-work, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows umbrella sampling by asking how adaptive bias and nonequilibrium pulling change the measure being sampled, and what corrections are needed before a free-energy claim is trustworthy. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Enhanced sampling methods work by changing the probability measure. A
metadynamics-style bias discourages revisiting already sampled regions. A
steered nonequilibrium protocol drives the system along paths whose work values
must be interpreted as a path ensemble, not as equilibrium samples.

For ML researchers working with MLIPs, the important shift is that faster
motion across a barrier is not automatically an unbiased result. The bias,
history, protocol speed, and path weights are part of the estimator. Jarzynski
and Crooks identities give exact nonequilibrium relationships, but their
finite-sample reliability still depends on overlap in work space (<span
id="cite-jarzynski1997"></span>[Jarzynski, 1997](#ref-jarzynski1997); <span
id="cite-crooks1999"></span>[Crooks, 1999](#ref-crooks1999); <span
id="cite-laio2002"></span>[Laio & Parrinello, 2002](#ref-laio2002); <span
id="cite-barducci2008"></span>[Barducci et al., 2008](#ref-barducci2008)).

This draft demonstrates the executable slice of the eleventh tutorial with a
known one-dimensional double-well coordinate. The adaptive-bias diagnostic
shows how history-dependent hills fill wells; the nonequilibrium diagnostic
uses controlled work ensembles to show mean-work dissipation, Jarzynski
estimates, and a Crooks crossing.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/full.json)
- [enhanced-sampling notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-11-enhanced-sampling.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/smoke/enhanced_sampling_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/enhanced_sampling_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-11.md)

## What Changes When The Method Is Adaptive?

The full profile deposits `3000` well-tempered Gaussian hills on a double-well
coordinate. The resulting diagnostic is intentionally compact:

| Diagnostic | Full value | Interpretation |
|---|---:|---|
| final bias range | 6.534 | adaptive bias has filled a substantial free-energy range |
| reconstructed barrier error | 0.092 | final bias gives a reasonable controlled PMF estimate |
| left basin visits | 0.360 | both basins are sampled |
| right basin visits | 0.362 | neither basin dominates the run |
| barrier visits | 0.135 | the barrier region is no longer invisible |

The adaptive trajectory is not an unbiased trajectory. The bias history is part
of the result.

## What Changes When The Method Is Nonequilibrium?

The full pulling diagnostic has true free-energy difference effectively zero,
but the forward and reverse mean works are positive because the finite-speed
protocol dissipates work. Jarzynski and Crooks estimates recover the answer in
this controlled case:

| Estimate | Full value |
|---|---:|
| forward mean work | 0.170 |
| reverse mean work | 0.173 |
| forward Jarzynski estimate | 0.001 |
| reverse Jarzynski estimate | -0.009 |
| Crooks crossing | -0.001 |

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post11_enhanced_sampling_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Enhanced-sampling diagnostics for the committed full profile. Adaptive bias changes where samples are drawn, while nonequilibrium work identities recover the free-energy difference from a path ensemble rather than from mean work." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 11 --profile smoke
uv run kups-tutorial verify 11 --profile smoke
uv run kups-tutorial run 11 --profile full
uv run kups-tutorial verify 11 --profile full
uv run jupyter execute notebooks/post-11-enhanced-sampling.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, enhanced-sampling diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled enhanced-sampling workflows
- committed compact summaries and diagnostic curves
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final 3,500-10,000-word article prose
- rendered desktop and mobile page snapshots
- production MD context and final uncertainty diagnostics
- final citation pass

The rule for this post is that enhanced sampling is a change of measure. Bias
history and path weights are part of the estimator, not implementation details
to hide after the trajectory crosses a barrier.

## References

- <span id="ref-jarzynski1997"></span>Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. *Physical Review Letters*, 78, 2690-2693. <a href="#cite-jarzynski1997" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-crooks1999"></span>Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. *Physical Review E*, 60, 2721-2726. <a href="#cite-crooks1999" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-laio2002"></span>Laio, A. & Parrinello, M. (2002). Escaping free-energy minima. *Proceedings of the National Academy of Sciences*, 99, 12562-12566. <a href="#cite-laio2002" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-barducci2008"></span>Barducci, A., Bussi, G. & Parrinello, M. (2008). Well-tempered metadynamics: a smoothly converging and tunable free-energy method. *Physical Review Letters*, 100, 020603. <a href="#cite-barducci2008" class="reversefootnote" role="doc-backlink">↩</a>
