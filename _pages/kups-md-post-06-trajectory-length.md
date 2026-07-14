---
layout: post
permalink: /kups-md-tutorials/post-06-trajectory-length/
title: "When Is a Trajectory Long Enough to Trust?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible trajectory-length diagnostic for molecular dynamics: warmup removal, autocorrelation, effective sample size, block uncertainty, and independent replica agreement."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 6
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 6
categories: [science]
tags: [molecular-dynamics, uncertainty, autocorrelation, equilibration, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the initialization, integrator, error, thermostat, and barostat diagnostics by asking when a finite trajectory has enough independent information to support a numerical claim. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A molecular-dynamics trajectory is long only relative to the question being
asked. Ten million correlated frames can still contain little independent
information if the observable relaxes slowly. A short set of independent
replicas can sometimes say more than a single long trace that has not forgotten
its initial condition.

For ML researchers working with MLIPs, this matters because the simulation may
look stable while the estimator is still biased or overconfident. The practical
question is not how many frames were written. It is whether warmup was removed,
whether autocorrelation was measured, whether uncertainty reflects effective
sample size, and whether independent replicas agree.

This draft demonstrates the executable slice of the sixth tutorial with a
controlled correlated observable. The model has a known equilibrium mean, so it
can expose estimator failure modes cleanly before the final argon/kUPS
observable diagnostics are added.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/full.json)
- [trajectory-length notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-06-trajectory-length.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/smoke/trajectory_length_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/trajectory_length_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-06.md)

## What Does Long Mean?

The current diagnostic compares trajectory-length checkpoints for a correlated
observable with a known answer:

| Choice | Full value | Why it matters |
|---|---:|---|
| true mean | 0.5 | answer key for the controlled diagnostic |
| stationary variance | 1.0 | equilibrium fluctuation scale |
| correlation time | 30 | sets memory and effective sample size |
| warmup steps | 1000 | removes most of the initial-condition bias |
| replicas | 6 | independent agreement check |
| checkpoints | 2000, 6000, 12000, 24000 | short-to-long estimator comparison |

The full run keeps the underlying process fixed and changes only how much data
is allowed into the estimator. This separates the effect of trajectory length
from changes in the model, integrator, or ensemble.

## What Should The Diagnostic Show?

The diagnostic reports naive standard error, autocorrelation-aware standard
error, block standard error, replica standard error, and a conservative
uncertainty used for review. The conservative uncertainty is intentionally not
the smallest number available.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post06_trajectory_length_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Trajectory-length diagnostics for the committed full profile. Running means retain memory, naive uncertainty is overconfident, and effective sample size grows much more slowly than the raw number of retained frames." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 06 --profile smoke
uv run kups-tutorial verify 06 --profile smoke
uv run kups-tutorial run 06 --profile full
uv run kups-tutorial verify 06 --profile full
uv run jupyter execute notebooks/post-06-trajectory-length.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, uncertainty diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled trajectory-length workflows
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for autocorrelation, effective sample size, blocking analysis, and
  equilibration diagnostics
- rendered desktop and mobile page snapshots
- argon/kUPS trajectory-length diagnostics for physical observables

The rule for this post is the same as the rest of the series: a trajectory is
not trustworthy because it is large. It becomes useful when the estimator,
uncertainty, and independent checks support the claim being made.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-flyvbjerg1989"></span>Flyvbjerg, H. & Petersen, H. G. (1989). Error estimates on averages of correlated data. *Journal of Chemical Physics*, 91, 461-466.
