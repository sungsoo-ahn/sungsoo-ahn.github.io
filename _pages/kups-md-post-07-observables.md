---
layout: post
permalink: /kups-md-tutorials/post-07-observables/
title: "How Do Trajectories Become Physical Observables?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible observable-estimator diagnostic for molecular dynamics: RDF normalization, coordination integrals, finite-size support, uncertainty, and velocity autocorrelation functions."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 7
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 7
categories: [science]
tags: [molecular-dynamics, rdf, observables, correlation-functions, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the trajectory-length discussion by asking how sampled configurations and velocities become physical observables with normalization, finite-size limits, and uncertainty. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A trajectory is not an observable by itself. It is a source of samples from
which an observable is estimated. That distinction matters because every
observable has a definition, normalization, finite-size support, and uncertainty
model.

For ML researchers working with MLIPs, the failure mode is familiar: a
simulation produces many frames, a plotting function produces a smooth curve,
and the curve is treated as a physical result. The practical question is
different. What estimator was used? Which samples enter it? What finite-size
region is valid? What error bar belongs on the derived number?

This draft demonstrates the executable slice of the seventh tutorial with
seeded periodic argon FCC cells. It computes radial distribution functions,
coordination integrals, block uncertainties, and a velocity autocorrelation
function before the final argon/kUPS trajectory diagnostic is added.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/full.json)
- [observable notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-07-observables.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/smoke/observable_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/observable_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-07.md)

## What Is Being Estimated?

The current diagnostic keeps the estimator explicit:

| Choice | Full value | Why it matters |
|---|---:|---|
| number density | 0.021 | RDF and coordination normalization |
| displacement scale | 0.10 | seeded thermal-like structural spread |
| small system | 32 atoms | finite-size stress test |
| large system | 256 atoms | larger radial support and lower block noise |
| RDF bin width | 0.05 | resolution versus noise tradeoff |
| coordination cutoff | 4.6 | first-shell integral boundary |
| VACF max lag | 90 | time-correlation support |

The small-cell RDF is not drawn beyond half the periodic box length. Those
radial shells are not valid for a minimum-image RDF estimator, even if a plotting
routine could produce numbers there.

## What Should The Diagnostic Show?

The full run checks three things. The RDF panel shows the normalized pair
estimator rather than a raw distance histogram. The coordination panel turns
that curve into a first-shell integral with a block standard error. The VACF
panel treats time correlation as its own observable, not as a side effect of
the trajectory.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post07_observable_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Observable diagnostics for the committed full profile. The RDF is normalized and finite-size limited, the coordination number carries a block uncertainty, and the velocity autocorrelation function shows how a trajectory becomes a time-correlation estimator." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 07 --profile smoke
uv run kups-tutorial verify 07 --profile smoke
uv run kups-tutorial run 07 --profile full
uv run kups-tutorial verify 07 --profile full
uv run jupyter execute notebooks/post-07-observables.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, observable diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled argon-FCC observable workflows
- committed compact RDF, VACF, and summary outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for RDF normalization, coordination integrals, finite-size effects,
  and time-correlation functions
- rendered desktop and mobile page snapshots
- argon/kUPS trajectory diagnostics for physical observables

The rule for this post is that an observable is a statistical object. The
trajectory provides samples; the estimator, normalization, finite-size support,
and uncertainty determine what can be claimed from those samples.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-allen1987"></span>Allen, M. P. & Tildesley, D. J. (1987). *Computer Simulation of Liquids*. Oxford University Press.
