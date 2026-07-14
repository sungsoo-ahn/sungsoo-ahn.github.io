---
layout: post
permalink: /kups-md-tutorials/post-08-free-energies/
title: "How Do Equilibrium Samples Become Free Energies?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible free-energy diagnostic for molecular dynamics: collective variables, histogram PMFs, binning bias, reweighting, RDF-derived PMFs, and uncertainty."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 8
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 8
categories: [science]
tags: [molecular-dynamics, free-energy, pmf, reweighting, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the observable-estimator discussion by asking how equilibrium samples become free-energy estimates once a collective variable, normalization, binning rule, and uncertainty model are chosen. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Free energy is not read directly from a trajectory. It is inferred from a
probability distribution over a chosen coordinate. That coordinate may be a
distance, an angle, a density, a coordination number, or a learned collective
variable, but the estimator changes with the choice.

For ML researchers working with MLIPs, the important shift is from frames to
probability. A histogram can be converted into a potential of mean force, but
only after asking what was sampled, which bins were empty, how much smoothing or
binning bias was introduced, and what uncertainty should accompany the derived
barrier.

This draft demonstrates the executable slice of the eighth tutorial with a
controlled double-well distribution and a synthetic RDF-derived PMF. It is a
small diagnostic for the estimator mechanics before the final argon/kUPS
free-energy observable is added.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/full.json)
- [free-energy notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-08-free-energies.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/smoke/free_energy_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/free_energy_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-08.md)

## What Is The Free Energy Of?

The current diagnostic keeps the answer key available:

| Choice | Full value | Why it matters |
|---|---:|---|
| temperature | 1.0 | dimensionless \(kT\) |
| samples | 80000 | equilibrium samples from the controlled distribution |
| coordinate domain | -2.5 to 2.5 | support for the collective variable |
| bin widths | 0.06, 0.18, 0.35 | resolution versus bias comparison |
| biased center | 0.9 | simple reweighting test |
| RDF peak radius | 1.2 | minimum of the RDF-derived PMF |

The true double-well barrier is `1.0`, so the diagnostic can separate estimator
error from the physical free-energy definition.

## What Should The Diagnostic Show?

The full run checks three estimator questions. The first panel compares the
true PMF, a direct histogram PMF, and a reweighted PMF. The second panel shows
that bin width changes the estimated barrier even for equilibrium samples. The
third panel shows how an RDF-like \(g(r)\) can become a shifted PMF through
`-kT log g(r)`.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post08_free_energy_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Free-energy diagnostics for the committed full profile. Histogram PMFs depend on binning, reweighting changes the estimate through statistical weights, and an RDF-like pair distribution can be converted into a shifted potential of mean force." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 08 --profile smoke
uv run kups-tutorial verify 08 --profile smoke
uv run kups-tutorial run 08 --profile full
uv run kups-tutorial verify 08 --profile full
uv run jupyter execute notebooks/post-08-free-energies.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, free-energy diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled free-energy workflows
- committed compact PMF curve and summary outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for PMFs, histogram estimators, reweighting, and RDF-derived
  potentials of mean force
- rendered desktop and mobile page snapshots
- argon/kUPS RDF-derived PMF diagnostics linked back to post 07 observables

The rule for this post is that free energy is a property of an estimator over a
chosen coordinate. Changing the coordinate, bins, weights, or sampled support
changes what can be claimed.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021.
