---
layout: post
permalink: /kups-md-tutorials/post-01-initialization/
title: "How Do You Initialize an MD Simulation Without Biasing the Result?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible first step for molecular dynamics: cell construction, seeded velocities, center-of-mass removal, provenance, and initialization diagnostics."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 1
categories: [science]
tags: [molecular-dynamics, initialization, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. The source repository is <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>; corrections and replication issues should be tracked against that repository.</em>
</p>

## Introduction

Molecular dynamics papers often begin after the most consequential choices have
already been made. The system has a cell, atom positions, velocities, a target
temperature, a seed, maybe a minimization stage, maybe a warmup stage, and
usually some center-of-mass cleanup. These details look procedural, but they
define the distribution from which the trajectory starts.

For ML researchers who already know the equation of motion,

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i, \qquad \frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i,$$

the practical question is different: what exactly did we initialize, and can
someone else reproduce it? If the answer is vague, later claims about
thermostats, equilibration, free energies, or MLIP stability are already on
weak ground.

This draft demonstrates the first reproducible slice of the series using a
small FCC argon smoke profile. The current executable artifacts are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/smoke.json)
- [initialization notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-01-initialization.ipynb)
- [compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/smoke/initialization_summary.json)
- [provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/smoke/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-01.md)

## What Must Be Fixed?

An initial state is not only coordinates. For this tutorial, the committed
configuration fixes:

| Choice | Smoke value | Why it matters |
|---|---:|---|
| system | FCC argon | controlled, cheap CPU example |
| atom count | 32 | small smoke profile for fast checks |
| number density | 0.0213 atoms/angstrom^3 | determines the cell volume |
| temperature | 94.4 K | sets the velocity scale |
| seed | 2026071401 | makes the velocity draw reproducible |
| center-of-mass removal | true | removes bulk translation from the initialized momenta |

The full tutorial will expand this into a larger profile, but the smoke case is
already useful because it exposes the bookkeeping. A later trajectory should
not merely say "argon at 94.4 K." It should say which density, which seed, which
velocity distribution, whether exact temperature rescaling was used, whether
center-of-mass motion was removed, and which code revision produced the state.

## Velocities Are Samples

The current profile samples momenta from the Maxwell-Boltzmann distribution and
does not force the instantaneous kinetic temperature to equal the target
temperature. That is intentional. In a finite system, the kinetic temperature of
one velocity draw fluctuates. Forcing it to match exactly can be useful for
some workflows, but it changes the draw.

This distinction matters for later posts. Thermostat validation, equilibration
diagnostics, and uncertainty estimates all depend on remembering which
quantities were sampled and which were constrained by construction.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post01_initialization_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Initialization diagnostics for the committed smoke profile. The figure checks the FCC cell density, shows the seeded velocity draw as standardized components, and records provenance fields that should remain reproducible." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 01 --profile smoke
uv run kups-tutorial verify 01 --profile smoke
uv run jupyter execute notebooks/post-01-initialization.ipynb --inplace
```

The notebook is deliberately not the source of reusable logic. It imports the
configuration loader, initializer, provenance helper, and figure generator from
`src/kups_md_tutorials/`. That separation keeps the article readable while
making the numerical outputs testable.

## Current Status

This page is not the final article. The implemented pieces are:

- CPU smoke initialization workflow
- committed compact smoke outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, and figure feedback

The missing pieces are:

- full-profile initialization review
- final article prose
- rendered desktop and mobile page snapshots
- broader connection to minimization and warmup choices

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
