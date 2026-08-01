---
layout: post
permalink: /kups-md-tutorials/post-03-errors/
title: "How Do Timestep, Precision, and Force Error Become Simulation Error?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Separate integration, arithmetic, replica, and force-model error with an exact control and real kUPS Lennard-Jones NVE trajectories."
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
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft remains hidden while the full series is being
validated. It assumes the velocity-Verlet map from Part 2. Corrections and
replication issues belong in
<a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

An NVE energy trace is not a diagnosis. A wider envelope can come from the
timestep, an arithmetic floor, or a discontinuous force. A beautifully flat
trace can still come from the wrong conservative potential. If those cases are
collapsed into one “energy drift” number, the number answers very little.

This tutorial separates the mechanisms twice. An exactly solvable oscillator
lets us change one assumption at a time. Real kUPS Lennard-Jones trajectories
then test timestep, precision, velocity seeds, and a perturbed potential on an
atomic system.

## Three errors, three questions

Write one velocity-Verlet step abstractly as

$$z_{n+1}=\Phi_{\Delta t, p, \widetilde{U}}(z_n),$$

where $$\Delta t$$ is the timestep, $$p$$ denotes the arithmetic policy, and
$$\widetilde{U}$$ is the potential evaluated by the code. These axes answer
different questions:

- **Timestep:** how well does the discrete map approximate the continuous flow?
- **Precision:** how much does finite arithmetic perturb that discrete map?
- **Force model:** is the simulated vector field the one the scientific claim
  requires?

Replica variation is a fourth axis. It does not change the method, but it tells
us whether a conclusion depends on one draw of the initial velocities.

The notebook setup only imports the analytic control and public kUPS runner, so
it stays collapsed.

{% include kups-notebooks/post-03/post03-setup.html %}

## Use an exact control before interpreting atoms

For the unit harmonic oscillator,

$$\ddot q=-q, \qquad E(q,v)=\frac{1}{2}q^2+\frac{1}{2}v^2,$$

the exact position is known at every time. We can therefore report both energy
error and position error, rather than trusting energy alone. The control below
holds the timestep at 0.18 and compares exact float64 arithmetic, deliberately
coarse rounding, and a force scaled by 0.98.

{% include kups-notebooks/post-03/post03-error-control.html %}

The exact-force float64 run has a maximum relative energy error near 0.81%, but
its energy remains bounded. Rounding the state to a $$10^{-3}$$ grid raises the
maximum error to about 2.0%. The 0.98 force scale raises the RMS position error
from about 0.295 to 1.10 and changes the normalized drift by more than an order
of magnitude.

Those are intentionally artificial controls. Their value is that the expected
answer is unambiguous. Across exact-force float64 runs, reducing the
dimensionless timestep from 0.18 to 0.02 lowers the maximum energy error from
about $$8.1\times10^{-3}$$ to $$1.0\times10^{-4}$$. That is the signature of
discretization error. Coarse rounding instead produces an arithmetic floor
(<span id="cite-higham2002"></span>[Higham, 2002](#ref-higham2002)).

## Now ask kUPS the same questions

The next cell calls `kups.application.simulations.md.run` for five 32-atom CPU
smoke cases. It runs two velocity replicas, a 20 fs comparison, a float64
comparison, and a Lennard-Jones epsilon scaled by 0.98. Each case writes a raw
kUPS HDF5 trajectory; the notebook reopens it and prints its frame count,
device, energy error, and hash.

{% include kups-notebooks/post-03/post03-kups-errors.html %}

Eight stored frames are an executable check, not production evidence. The full
profile uses 256 atoms, three independent 2 fs replicas, and 1 ps,
fixed-duration timestep comparisons on GPUs. The perturbed-potential case
reuses the first replica's seed. The float64 case uses the same seed, but dtype
can also change random-number generation, so it is a deployment comparison
rather than a bitwise-identical initial-state test.

<div class="table-responsive" markdown="1">

| Run | Max stored-frame $$\lvert\Delta E/E_0\rvert$$ | Energy span |
|---|---:|---:|
| 2 fs, float32, replica 0 | 0.528% | 0.385 meV/atom |
| 2 fs, float32, replica 1 | 0.544% | 0.392 meV/atom |
| 2 fs, float32, replica 2 | 0.545% | 0.399 meV/atom |
| 0.5 fs, float32 | 0.527% | 0.385 meV/atom |
| 20 fs, float32 | 0.613% | 0.447 meV/atom |
| 2 fs, float64 | 0.540% | 0.397 meV/atom |
| 2 fs, float32, $$0.98\epsilon$$ | 0.533% | 0.380 meV/atom |

</div>

Every full case records `production_gpu_ready: true`, observed NVIDIA RTX A5000
devices, the kUPS entry point, input and HDF5 hashes, dataset shapes, frame
counts, and HDF5-derived energy traces. The table reports the maximum deviation
from the first stored total energy. It is a diagnostic for this initialized
crystal and time window, not a universal timestep limit.

All traces share an early rise near 0.5%, so the absolute envelope is dominated
by a common initialization transient. The 20 fs case is visibly wider. The 0.5
and 2 fs cases are nearly coincident, the float64 result lies inside replica
variation, and the altered conservative potential conserves its own energy just
as well. Those are the useful comparisons; none justifies assigning the shared
transient to timestep truncation alone.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_error_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The first three panels are exact harmonic-oscillator controls for timestep, arithmetic, and inconsistent-force error. The lower-right panel is derived from real kUPS Lennard-Jones NVE HDF5 trajectories, including independent replicas, precision, timestep, and conservative potential-scale comparisons." %}

## A flat trace can certify the wrong Hamiltonian

The oscillator's scaled force is deliberately inconsistent with the energy
used for diagnosis: it integrates $$-0.98q$$ but reports
$$E=(q^2+v^2)/2$$. Drift is expected because the force is not the gradient of
the reported energy.

The real kUPS perturbation is different. Scaling Lennard-Jones epsilon changes
both energy and force consistently. Velocity Verlet can conserve that altered
Hamiltonian extremely well. The trajectory may therefore have a flat energy
trace while representing the wrong material model.

This matters directly for machine-learned potentials. NVE conservation tests
whether energy and force are numerically consistent along the sampled path. It
does **not** establish that either is accurate relative to reference quantum
mechanics. Static energy/force validation, extrapolation diagnostics, and
observable checks remain separate obligations.

## Read envelopes, slopes, and replicas separately

A symplectic integrator need not conserve the reported physical energy at every
step. It often follows a nearby modified Hamiltonian, giving a bounded energy
envelope rather than monotonic drift
(<span id="cite-hairer2006"></span>[Hairer et al., 2006](#ref-hairer2006)).
That is why this workflow records both maximum excursion and a fitted drift.

Replicas answer another question. Two trajectories with different initial
velocities will not agree point by point, and chaotic separation makes that an
unhelpful target. What should agree are protocol-level conclusions: whether the
energy envelope is bounded, whether its scale changes under timestep reduction,
and eventually whether observable estimates agree within uncertainty.

A defensible error audit therefore proceeds in this order:

1. Verify the update rule against an exact or otherwise controlled problem.
2. Sweep timesteps while holding initial state, duration, and output cadence
   fixed.
3. Repeat the relevant cases under the deployed precision policy.
4. Change velocity seeds before treating one trace as typical.
5. Validate the force model against an external reference; do not infer model
   accuracy from energy conservation.
6. Repeat the comparison for the observable that supports the scientific
   claim.

The last step prevents a common shortcut. A timestep acceptable for mean
potential energy is not automatically acceptable for a diffusion coefficient,
vibrational spectrum, or rare-event rate.

## Reproduce the result

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 03 --profile smoke
uv run kups-tutorial verify 03 --profile smoke
uv run kups-tutorial verify-notebooks --posts 03 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 03 --check
```

The compact [smoke](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/smoke/kups_md_summary.json)
and [full](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/kups_md_summary.json)
kUPS summaries contain the values above. The
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/full.json),
[full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/manifest.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-03-errors.ipynb),
[figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post03_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-03.md)
complete the chain.

## References

- <span id="ref-higham2002"></span>Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM. <a href="#cite-higham2002" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. <a href="#cite-hairer2006" class="reversefootnote" role="doc-backlink">↩</a>
