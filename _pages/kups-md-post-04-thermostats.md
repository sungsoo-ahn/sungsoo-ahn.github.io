---
layout: post
permalink: /kups-md-tutorials/post-04-thermostats/
title: "How Do Thermostats Change Sampling and Dynamics?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible thermostat diagnostic for molecular dynamics: BAOAB Langevin sampling, canonical moment checks, coupling strength, autocorrelation, and dynamical distortion."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 4
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 4
categories: [science]
tags: [molecular-dynamics, thermostats, langevin, sampling, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post builds on the initialization and integrator diagnostics by asking what changes once a thermostat is added. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A thermostat is often described as "keeping the temperature fixed." That phrase
is too vague for molecular simulation. A thermostat changes the equations being
integrated, the distribution being sampled, and usually the dynamical memory of
the trajectory.

For ML researchers, the practical question is not only whether the kinetic
temperature is close to the target. It is whether the sampled configurational
and kinetic moments match the intended ensemble, and whether the coupling is so
strong that time-correlation functions are no longer interpretable as physical
dynamics.

This draft demonstrates the executable slice of the fourth tutorial with a
controlled harmonic oscillator and BAOAB Langevin dynamics. The oscillator is a
small microscope for the sampling/dynamics distinction; the final article still
needs an argon/kUPS thermostat diagnostic before publication.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/full.json)
- [thermostat notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-04-thermostats.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/smoke/thermostat_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/thermostat_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-04.md)

## What Does Coupling Strength Change?

The current diagnostic fixes the oscillator, timestep, target temperature, and
BAOAB splitting, then changes the Langevin friction:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | canonical targets are known |
| thermostat | BAOAB Langevin | explicit stochastic splitting |
| target temperature | 1.0 | dimensionless \(kT\) |
| timestep | 0.02 | shared across coupling strengths |
| friction values | 0.1, 1.0, 5.0 | weak, moderate, and strong coupling |
| samples | 3500 per run | enough for compact moment checks |

The check is deliberately two-sided. If the moments are wrong, the thermostat
is not sampling the intended canonical distribution. If the moments are right
but autocorrelation changes sharply, the thermostat may still be unsuitable for
dynamical observables.

## What Should The Diagnostic Show?

The full run compares observed position and velocity variances to their
canonical targets. It also compares mean kinetic energy to the \(0.5kT\) target
for one degree of freedom. Finally, it reports the position integrated
autocorrelation time to show that stronger coupling can change dynamical
memory.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post04_thermostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Thermostat diagnostics for the committed full profile. The BAOAB Langevin cases sample near the same canonical moment targets, but strong coupling substantially increases position autocorrelation time." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 04 --profile smoke
uv run kups-tutorial verify 04 --profile smoke
uv run kups-tutorial run 04 --profile full
uv run kups-tutorial verify 04 --profile full
uv run jupyter execute notebooks/post-04-thermostats.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, thermostat diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled BAOAB Langevin workflows
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for Langevin dynamics, BAOAB splitting, canonical sampling, and
  thermostat-induced dynamical distortion
- rendered desktop and mobile page snapshots
- argon/kUPS thermostat diagnostics that connect this controlled testbed to the
  production trajectory workflow

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-leimkuhler2013"></span>Leimkuhler, B. & Matthews, C. (2013). Rational construction of stochastic numerical methods for molecular sampling. *Applied Mathematics Research eXpress*, 2013(1), 34-56.
- <span id="ref-bussi2007"></span>Bussi, G., Donadio, D. & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *Journal of Chemical Physics*, 126, 014101.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
