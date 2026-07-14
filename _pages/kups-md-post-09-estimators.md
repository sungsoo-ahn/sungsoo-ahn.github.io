---
layout: post
permalink: /kups-md-tutorials/post-09-estimators/
title: "What Do Free-Energy Estimators Assume?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible free-energy estimator diagnostic for FEP, BAR, overlap, effective sample size, and estimator failure modes."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 9
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 9
categories: [science]
tags: [molecular-dynamics, free-energy, estimators, bar, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the PMF discussion by asking when free-energy estimators can be trusted, especially when overlap and effective sample size are poor. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Free-energy perturbation is exact as an identity and fragile as an estimator.
The difference is overlap. If samples from state A almost never visit
configurations that matter for state B, the exponential average is controlled
by rare tail events rather than by the typical trajectory frames.

For ML researchers working with MLIPs, this is the useful mental model:
free-energy estimators are diagnostics for probability mass, not just formulas
for energies. BAR improves on one-sided exponential averages by using samples
from both states, and WHAM/MBAR extend the same overlap logic across multiple
biased or intermediate states (<span id="cite-zwanzig1954"></span>[Zwanzig,
1954](#ref-zwanzig1954); <span id="cite-bennett1976"></span>[Bennett,
1976](#ref-bennett1976); <span id="cite-kumar1992"></span>[Kumar et al.,
1992](#ref-kumar1992); <span id="cite-shirts2008"></span>[Shirts & Chodera,
2008](#ref-shirts2008)).

This draft demonstrates the executable slice of the ninth tutorial with
exactly solvable Gaussian state pairs. The true free-energy difference is
known, so the diagnostic can separate estimator assumptions from
physical-model error.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/full.json)
- [estimator notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-09-estimators.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/smoke/estimator_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/estimator_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-09.md)

## What Is Being Estimated?

The current diagnostic compares two unit-variance one-dimensional states. The
second state is displaced and assigned a known free-energy offset:

| Case | Mean shift | True Delta F | Intended regime |
|---|---:|---:|---|
| good_overlap | 0.5 | 0.8 | many useful samples in both states |
| marginal_overlap | 1.5 | 0.8 | estimator looks plausible but ESS warns |
| poor_overlap | 3.0 | 0.8 | rare tails dominate one-sided FEP |

This is not a production alchemical calculation. It is a controlled test for
the failure modes that production calculations must diagnose.

## What Should The Diagnostic Show?

The full run compares forward FEP, reverse FEP, and BAR against the known
answer. It also records overlap coefficients and exponential-weight effective
sample sizes. In the poor-overlap case, the forward effective sample size
collapses to less than one percent of the nominal sample count even though the
simulation contains fifty thousand samples per state.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post09_estimator_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Estimator diagnostics for the committed full profile. BAR remains close to the known \(\Delta F\) in this controlled example, while the overlap and ESS panels show why one-sided FEP becomes fragile as state overlap disappears." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 09 --profile smoke
uv run kups-tutorial verify 09 --profile smoke
uv run kups-tutorial run 09 --profile full
uv run kups-tutorial verify 09 --profile full
uv run jupyter execute notebooks/post-09-estimators.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, estimator diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled estimator workflows
- committed compact estimator summaries and work-sample outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final 3,500-10,000-word article prose
- rendered desktop and mobile page snapshots
- fuller WHAM/MBAR discussion and any final multi-state diagnostic figures
- final citation pass and production-style diagnostics

The rule for this post is that estimator reliability is an overlap question.
More samples help only when they include the configurations that carry the
statistical weight.

## References

- <span id="ref-zwanzig1954"></span>Zwanzig, R. W. (1954). High-temperature equation of state by a perturbation method. *The Journal of Chemical Physics*, 22, 1420-1426. <a href="#cite-zwanzig1954" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bennett1976"></span>Bennett, C. H. (1976). Efficient estimation of free energy differences from Monte Carlo data. *Journal of Computational Physics*, 22, 245-268. <a href="#cite-bennett1976" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021. <a href="#cite-kumar1992" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shirts2008"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. <a href="#cite-shirts2008" class="reversefootnote" role="doc-backlink">↩</a>
