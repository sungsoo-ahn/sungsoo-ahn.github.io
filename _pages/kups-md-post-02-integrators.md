---
layout: post
permalink: /kups-md-tutorials/post-02-integrators/
title: "What Does an MD Integrator Actually Approximate?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible integrator diagnostic for molecular dynamics: discrete maps, velocity Verlet, energy error, reversibility, and timestep sensitivity."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 2
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 2
categories: [science]
tags: [molecular-dynamics, integrators, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post assumes the initialization contract from the first tutorial and focuses on the discrete update rule used after an initial state exists. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

The equation of motion is continuous, but an MD trajectory is not. Every saved
state came from a discrete map: positions and momenta at one time were
converted into positions and momenta at the next time. The integrator is that
map.

For ML researchers who already know

$$\dot{\mathbf{r}}_i=\mathbf{v}_i,\qquad m_i\dot{\mathbf{v}}_i=\mathbf{F}_i,$$

the practical question is not whether Newton's equation is correct. It is what
the finite timestep update preserves, what it distorts, and which diagnostics
can tell the difference between bounded discretization error and real
simulation drift.

This draft demonstrates the executable slice of the second tutorial with a
dimensionless harmonic oscillator. The toy system is intentionally simple
because the exact trajectory is known; that makes the integrator error visible
without confusing it with force-field error, neighbor lists, thermostats, or
finite-size effects.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-02/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-02/full.json)
- [integrator notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-02-integrators.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/smoke/integrator_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/full/integrator_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-02.md)

## What Is Being Approximated?

Velocity Verlet is not only a formula for updating coordinates. It is a
specific composition of simpler maps: drift positions, kick momenta with
forces, and drift or kick again depending on the chosen convention. Those
details determine whether the map is time-reversible and symplectic for a
separable Hamiltonian.

The current diagnostic fixes:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | exact reference trajectory is known |
| mass | 1.0 | dimensionless controlled example |
| angular frequency | 1.0 | sets the natural timescale |
| initial position | 1.0 | starts at the turning point |
| initial velocity | 0.0 | makes the phase-space orbit easy to inspect |
| timesteps | 0.02, 0.05, 0.1, 0.2 | exposes timestep sensitivity |
| steps per run | 2000 | separates bounded error from growth |

The final article will connect this back to many-body MD and MLIP force error.
For now, the harmonic oscillator is a microscope: if an update rule fails here,
it should not be trusted as a production MD integrator.

## What Should The Diagnostics Show?

Three checks matter before the prose makes stronger claims.

First, the numerical phase-space orbit should stay close to the exact orbit.
Second, the energy error should be bounded for velocity Verlet on this sweep,
not monotonically amplifying as it does for explicit Euler. Third, reversing the
velocity and applying the same map again should return velocity Verlet to the
initial state up to floating-point roundoff.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post02_integrator_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Integrator diagnostics for the committed full profile. The harmonic oscillator exposes velocity Verlet as a reversible discrete map with bounded energy error on this timestep sweep, while explicit Euler is retained as a negative control." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 02 --profile smoke
uv run kups-tutorial verify 02 --profile smoke
uv run kups-tutorial run 02 --profile full
uv run kups-tutorial verify 02 --profile full
uv run jupyter execute notebooks/post-02-integrators.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, integrator diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full integrator diagnostic workflows
- committed compact summaries and downsampled trajectory samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for Verlet, symplectic integration, and shadow Hamiltonians
- rendered desktop and mobile page snapshots
- connection to timestep, precision, and MLIP force-error diagnostics in the
  next post

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-verlet1967"></span>Verlet, L. (1967). Computer "experiments" on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules. *Physical Review*, 159(1), 98-103.
- <span id="ref-leimkuhler2004"></span>Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.
