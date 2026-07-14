---
layout: post
permalink: /kups-md-tutorials/post-05-barostats/
title: "How Should Pressure and Cell Degrees of Freedom Be Coupled?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible pressure and cell diagnostic for molecular dynamics: NPT-like volume fluctuations, compressibility, pressure variance, barostat time constants, and cell memory."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 5
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 5
categories: [science]
tags: [molecular-dynamics, npt, pressure, barostat, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post extends the thermostat discussion to pressure and cell degrees of freedom, using a controlled scalar model before the final argon/kUPS NPT diagnostic is added. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Pressure control is not the same kind of operation as setting a scalar
temperature. In small molecular simulations, instantaneous pressure fluctuates
strongly, volume fluctuations encode compressibility, and the barostat time
constant changes how quickly the cell explores those fluctuations.

For ML researchers, the practical question is not whether one pressure number
looks close to a target. It is whether the volume distribution has the right
scale, whether pressure fluctuations are interpreted statistically, and whether
the cell-coupling dynamics distort the observables that will be measured later.

This draft demonstrates the executable slice of the fifth tutorial with a
controlled scalar-volume model. The model is not the final production NPT
workflow; it is a microscope for fluctuation targets and barostat memory.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/full.json)
- [barostat notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-05-barostats.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/smoke/barostat_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/barostat_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-05.md)

## What Should Fluctuate?

The current diagnostic fixes a target pressure, volume, compressibility, and
temperature, then changes only the scalar barostat relaxation time:

| Choice | Full value | Why it matters |
|---|---:|---|
| target pressure | 1.0 | pressure mean for the controlled model |
| equilibrium volume | 1000 | reference cell size |
| compressibility | 0.01 | sets volume fluctuation scale |
| temperature | 1.0 | dimensionless \(kT\) |
| relaxation times | 0.5, 2.0, 8.0 | fast, moderate, and slow cell memory |
| samples | 2500 per run | enough for compact fluctuation checks |

In the NPT ensemble, volume variance is proportional to compressibility. A
barostat that suppresses this variance is not merely "more stable"; it may be
sampling the wrong ensemble.

## What Should The Diagnostic Show?

The full run checks two fluctuation targets: volume variance and pressure
variance. It also reports the integrated autocorrelation time of the scalar
volume process. The slow barostat has the same target distribution but much
longer memory, which means fewer effective samples for the same wall-clock
trajectory length.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post05_barostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Pressure and scalar-cell diagnostics for the committed full profile. The controlled NPT-like model recovers volume and pressure fluctuation scales while showing that slower barostat coupling increases cell memory." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 05 --profile smoke
uv run kups-tutorial verify 05 --profile smoke
uv run kups-tutorial run 05 --profile full
uv run kups-tutorial verify 05 --profile full
uv run jupyter execute notebooks/post-05-barostats.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, scalar barostat diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled scalar barostat workflows
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for NPT ensemble fluctuations, compressibility, barostat coupling,
  and finite-size pressure fluctuations
- rendered desktop and mobile page snapshots
- argon/kUPS NPT diagnostics with actual cell degrees of freedom

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-martyna1994"></span>Martyna, G. J., Tobias, D. J. & Klein, M. L. (1994). Constant pressure molecular dynamics algorithms. *Journal of Chemical Physics*, 101, 4177-4189.
- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
