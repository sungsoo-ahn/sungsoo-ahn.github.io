---
layout: post
permalink: /kups-md-tutorials/post-02-integrators/
title: "What Does an MD Integrator Actually Approximate?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Use an exact oscillator and real kUPS Lennard-Jones NVE trajectories to separate the equations of motion from their discrete update."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 2
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 2
categories: [science]
tags: [molecular-dynamics, integrators, velocity-verlet, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft remains hidden while the full series is being
validated. It assumes the initialization contract from Part 1. Corrections and
replication issues belong in
<a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

An MD program does not solve Newton's equations continuously. It repeatedly
applies a finite-timestep map. That map—not the differential equation written
in a paper—is what determines stability, reversibility, and energy error.

This tutorial uses two complementary tests. An exactly solvable harmonic
oscillator exposes the numerical method without force-field ambiguity. Real
kUPS Lennard-Jones NVE trajectories then show what the public Verlet path does
for 256 atoms on GPUs.

## The computational object is a map

For a timestep $$\Delta t$$, an integrator replaces the exact flow with

$$(\mathbf{r}_{n+1},\mathbf{v}_{n+1}) =
\Phi_{\Delta t}(\mathbf{r}_n,\mathbf{v}_n).$$

Velocity Verlet applies a half kick, a drift, and another half kick:

$$\mathbf{v}_{n+1/2}=\mathbf{v}_n+
\frac{\Delta t}{2m}\mathbf{F}(\mathbf{r}_n),$$

$$\mathbf{r}_{n+1}=\mathbf{r}_n+\Delta t\,\mathbf{v}_{n+1/2},$$

$$\mathbf{v}_{n+1}=\mathbf{v}_{n+1/2}+
\frac{\Delta t}{2m}\mathbf{F}(\mathbf{r}_{n+1}).$$

The second force evaluation is essential. Reusing a stale force changes the
map, even if the code still calls itself Verlet. Position-Verlet, leapfrog, and
velocity-Verlet conventions can describe equivalent trajectories, but only
when their half-step and integer-step state conventions are handled
consistently (<span id="cite-verlet1967"></span>[Verlet,
1967](#ref-verlet1967)).

The notebook setup is collapsed; it only imports the analytic and real-kUPS
interfaces used below.

{% include kups-notebooks/post-02/post02-setup.html %}

## First test the structure on an exact problem

For a unit harmonic oscillator,

$$\ddot q=-q, \qquad E=\frac{1}{2}v^2+\frac{1}{2}q^2,$$

the exact trajectory is known. This makes three integrator properties directly
measurable:

- trajectory error against the exact orbit;
- bounded energy excursion over many steps;
- return error after forward integration, velocity reversal, and backward
  integration.

{% include kups-notebooks/post-02/post02-verlet-control.html %}

At dimensionless timestep 0.2, the committed control has a maximum relative
energy error of about 1% and a forward/backward state error below
$$10^{-15}$$. Across timesteps 0.02, 0.05, 0.1, and 0.2, the maximum Verlet
energy error grows from $$10^{-4}$$ to $$10^{-2}$$. Explicit Euler instead
expands the oscillator orbit and eventually diverges.

Reversibility and symplectic structure do not mean exact energy conservation.
A stable symplectic method often conserves a nearby shadow Hamiltonian, so the
reported physical energy oscillates within a bounded envelope rather than
drifting without limit (<span id="cite-hairer2006"></span>[Hairer et al.,
2006](#ref-hairer2006)). The size of that envelope still depends on the
timestep.

## Then run the real library path

An analytic control cannot prove that kUPS ran. The next cell calls
`kups.application.simulations.md.run` twice with the `verlet` integrator. Both
32-atom smoke cases use the same seeded initial momenta and physical-unit
Lennard-Jones potential; only the timestep changes. The raw HDF5 trajectories
remain ignored, while their schema, hashes, frame counts, devices, and compact
energy diagnostics are committed.

{% include kups-notebooks/post-02/post02-kups-nve.html %}

Eight frames prove executable CPU integration and HDF5 analysis, not a
production timestep choice. The full comparison holds the simulated duration
and stored-frame interval fixed at 10 ps and 20 fs, respectively:

<div class="table-responsive" markdown="1">

| Timestep | Steps / frames | Max stored-frame $$|\Delta E/E_0|$$ | Energy span |
|---:|---:|---:|---:|
| 0.5 fs | 20,000 / 500 | 0.528% | 0.383 meV/atom |
| 2 fs | 5,000 / 500 | 0.528% | 0.383 meV/atom |
| 20 fs | 500 / 500 | 0.633% | 0.459 meV/atom |

</div>

All three full manifests record `production_gpu_ready: true` and eight observed
NVIDIA RTX A5000 devices. The 0.5 and 2 fs traces are nearly indistinguishable
at this resolution; 20 fs produces a wider energy envelope. That is evidence
for this initialized crystal and 10 ps window, not a universal safe-timestep
table.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post02_integrator_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The exact oscillator exposes phase-space geometry, scheme-dependent energy error, and velocity-Verlet reversibility; the lower-right panel is derived from real kUPS Lennard-Jones NVE HDF5 trajectories. Holding duration and output cadence fixed reveals a wider stored-frame energy envelope at 20 fs, while 0.5 and 2 fs are nearly coincident for this protocol." %}

## Energy drift is a symptom, not a diagnosis

A sloping energy trace can come from a timestep that is too large, but it can
also come from discontinuous cutoffs, stale neighbor lists, mixed precision,
constraint tolerances, or a force model that is not the gradient of the
reported energy. Conversely, a bounded energy trace does not establish that
the positions are accurate over long times; chaotic trajectories separate even
when their ensemble statistics remain useful.

The diagnostic should therefore be staged:

1. Verify the update rule on a problem with an exact solution.
2. Sweep timesteps on the real potential and initialized system.
3. Hold initial state, simulated duration, and output cadence fixed.
4. Inspect both bounded excursions and fitted drift.
5. Repeat after changing precision, neighbor-list policy, or the force model.

For a machine-learned potential, the final step is indispensable. Reducing
$$\Delta t$$ cannot repair a systematically wrong or non-conservative force.
Part 3 separates those error sources.

## Reproduce the result

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 02 --profile smoke
uv run kups-tutorial verify 02 --profile smoke
uv run kups-tutorial verify-notebooks --posts 02 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 02 --check
```

The compact [smoke](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/smoke/kups_md_summary.json)
and [full](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/full/kups_md_summary.json)
kUPS summaries contain the values above. The
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-02/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-02/full.json),
[full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-02/full/manifest.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-02-integrators.ipynb),
[figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post02_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-02.md)
complete the chain.

## References

- <span id="ref-verlet1967"></span>Verlet, L. (1967). Computer “experiments” on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules. *Physical Review*, 159(1), 98–103. <a href="#cite-verlet1967" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. <a href="#cite-hairer2006" class="reversefootnote" role="doc-backlink">↩</a>
