---
layout: post
permalink: /kups-md-tutorials/post-03-errors/
title: "How Do Timestep, Precision, and Force Error Become Simulation Error?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible simulation-error diagnostic for molecular dynamics: timestep sensitivity, precision floors, force bias, bounded energy oscillation, and drift."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 3
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 3
categories: [science]
tags: [molecular-dynamics, timestep, precision, force-error, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post builds on the velocity-Verlet diagnostic from the previous tutorial and separates timestep, precision, and force-error mechanisms before later argon and MLIP NVE checks. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

An NVE diagnostic often gets compressed into one question: did the energy
drift? That question is too blunt. The same total-energy trace can mix bounded
integrator error, a timestep that is too large, mixed-precision roundoff, force
noise, systematic MLIP bias, and neighbor-list artifacts.

For ML researchers, the useful decomposition is more practical: which part of
the error came from the discrete timestep, which part came from numerical
precision, and which part came from the force model? If these are not separated
early, later claims about thermostat behavior, uncertainty, or free energies
can inherit a hidden simulation artifact.

This draft demonstrates the executable slice of the third tutorial with the
same controlled oscillator used in the integrator post. The system is simple by
design: the exact reference trajectory is known, so the diagnostic can isolate
error mechanisms before the series moves back to argon NVE trajectories.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/full.json)
- [error notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-03-errors.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/smoke/error_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/error_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-03.md)

## What Must Be Separated?

The current diagnostic fixes the oscillator and varies three axes:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | exact reference trajectory is known |
| timesteps | 0.02, 0.05, 0.1, 0.18 | exposes timestep sensitivity |
| steps per run | 3000 | makes drift measurable |
| precision models | float64, float32, rounded grids | separates arithmetic effects |
| force scales | 0.98, 1.0, 1.02 | deterministic force-error proxy |
| total runs | 48 | full grid over all three axes |

The force-scale perturbation is not meant to model every MLIP failure. It is a
controlled negative example: even a simple systematic force bias changes the
energy behavior, so MLIP diagnostics should not hide force error inside a
single timestep-convergence number.

## What Should The Diagnostic Show?

The exact-force float64 runs show the timestep story: the maximum relative
energy error grows as the timestep increases, while remaining bounded on this
sweep. The rounded-precision runs show that arithmetic can set an error floor
even when the analytical force is unchanged. The force-scale cases show that a
biased force can shift normalized energy drift.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_error_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Simulation-error diagnostics for the committed full profile. The figure separates bounded timestep error, precision-induced error floors, and force-bias drift so later NVE tests can report these mechanisms separately." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 03 --profile smoke
uv run kups-tutorial verify 03 --profile smoke
uv run kups-tutorial run 03 --profile full
uv run kups-tutorial verify 03 --profile full
uv run jupyter execute notebooks/post-03-errors.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, error diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled error-diagnostic workflows
- committed compact summaries and downsampled comparison samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final article prose
- citations for timestep stability, backward error analysis, mixed precision,
  and MLIP force-error diagnostics
- rendered desktop and mobile page snapshots
- argon/kUPS NVE diagnostics that connect this controlled testbed to production
  molecular-dynamics practice

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.
- <span id="ref-leimkuhler2004"></span>Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
- <span id="ref-higham2002"></span>Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
