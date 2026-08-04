---
layout: post
permalink: /kups-md-tutorials/post-03-errors/
title: "Where Does Error Enter an MD Trajectory?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Separate timestep, arithmetic, initial-condition, and force-model error with transparent JAX controls and matched kUPS trajectories."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 3
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 3
categories: [science]
tags: [molecular-dynamics, timestep, precision, force-error, jax, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

An energy trace is evidence, not a verdict. A wider trace can come from a large
timestep, accumulated arithmetic error, or forces that do not match the
reported energy. A flat trace can belong to the wrong conservative potential.
One number called "energy drift" cannot distinguish those mechanisms.

We will separate them on a harmonic oscillator whose exact trajectory is
known. Then we will compare matched kUPS trajectories atom by atom. The goal is
to replace the question "is the simulation accurate?" with smaller questions
that an experiment can answer.

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- how timestep, arithmetic, initial-state, and force-model errors differ;
- why velocity Verlet can have bounded energy error but growing phase error;
- how to expose the force function and arithmetic policy in JAX;
- why conserving energy does not validate a potential;
- how matched seeds and atom-level displacement reveal trajectory divergence.

**Prerequisites:** the initialized state from [Post 01]({{ '/kups-md-tutorials/post-01-initialization/' | relative_url }})
and the velocity-Verlet map from [Post 02]({{ '/kups-md-tutorials/post-02-integrators/' | relative_url }}).
</div>

## "Trajectory error" is not one object

Let $$z(t)=(\mathbf{R}(t),\mathbf{P}(t))$$ be the exact trajectory under a
chosen potential $$U$$. A program produces discrete states

$$
z_{n+1}=\Phi_{\Delta t,p,\widetilde U}(z_n).
$$

Here $$\Phi$$ is the implemented update, $$\Delta t$$ is the timestep, $$p$$
denotes the arithmetic policy, and $$\widetilde U$$ is the energy model used by
the program. Each subscript creates a different comparison:

- **Timestep error:** compare the discrete map with continuous dynamics under
  the same $$U$$.
- **Arithmetic error:** compare implementations of the same discrete map under
  different dtypes or rounding policies.
- **Force-model error:** compare trajectories under $$\widetilde U$$ and the
  scientifically intended $$U$$.
- **Initial-state variation:** compare independent draws from the same stated
  initialization distribution.

The first three change the computed dynamics. The fourth does not indicate a
fault; it measures how much the protocol depends on one sampled state.

We also need to choose **what** is compared. Position error, energy error, a
radial distribution, and a diffusion coefficient can rank two methods
differently. Accuracy only has meaning relative to an observable and a time
window.

## Timestep error changes the discrete flow

Velocity Verlet approximates a continuous path with finite kick–drift–kick
updates. For a smooth system, reducing $$\Delta t$$ usually reduces local
truncation error. After many steps, the numerical trajectory can still move
out of phase with the exact one.

The unit harmonic oscillator makes that phase error visible:

$$
\ddot q=-q,
\qquad
q(t)=\cos t,
\qquad
E(q,p)=\frac{q^2+p^2}{2},
$$

for $$q(0)=1$$ and $$p(0)=0$$. The exact energy is $$1/2$$ at every time. A
symplectic method such as velocity Verlet usually oscillates around that value
instead of producing monotonic loss or gain. It follows the flow of a nearby
modified Hamiltonian
(<span id="cite-hairer2006"></span>[Hairer et al.,
2006](#ref-hairer2006)).

Bounded energy therefore does not imply a small position error at late times.
The numerical oscillator can have the right amplitude and the wrong phase.

## Arithmetic error perturbs every update

Floating-point numbers keep a finite number of significant bits. Each force,
sum, and state update is rounded to a representable value. The effect depends
on dtype, hardware, operation order, system size, and the number of steps
(<span id="cite-higham2002"></span>[Higham,
2002](#ref-higham2002)).

Roundoff is not equivalent to a larger timestep:

- timestep error changes the mathematical map before arithmetic is applied;
- roundoff perturbs the evaluation of that chosen map;
- reducing $$\Delta t$$ increases the number of updates needed for a fixed
  physical duration, so smaller steps do not monotonically remove roundoff.

JAX defaults and accelerator support matter here. Enabling float64 in Python
does not guarantee that every deployed kernel has the same throughput or
reduction order. Precision is part of the execution contract, not a cosmetic
array property.

## Force error can be conservative or inconsistent

Suppose the intended energy is $$U$$ but the program uses
$$\widetilde U=0.98U$$. If forces are computed consistently,

$$
\widetilde{\mathbf F}(\mathbf R)
=-\nabla_{\mathbf R}\widetilde U(\mathbf R),
$$

the simulation can conserve $$\widetilde U+K$$ extremely well. It still
follows the wrong Hamiltonian for a claim about $$U$$.

A different failure occurs when the code reports $$U$$ but propagates with a
force that is not $$-\nabla U$$. Energy exchange between potential and kinetic
terms is then inconsistent. An NVE trace can expose that mismatch, but it
cannot prove that a consistently differentiated model is physically accurate.

This distinction is central for machine-learned interatomic potentials. Force
consistency is a software and differentiation property. Agreement with
reference quantum mechanics is a model-validation property.

## Put the error sources into JAX

The setup chooses a CPU backend, enables float64 for the analytic control, and
imports the real kUPS workflow. It begins collapsed.

{% include kups-notebooks/post-03/post03-setup.html %}

The open cell makes the force function an explicit input to Verlet. An optional
quantizer rounds positions and momenta after every step. This quantizer is
deliberately crude; it creates an arithmetic floor large enough to inspect in
a short example.

{% include kups-notebooks/post-03/post03-jax-errors.html %}

At dimensionless timestep 0.18, the exact-force run keeps its maximum energy
excursion at 0.784% but accumulates an RMS position error of 0.295 over 540
time units. Rounding the state to a $$10^{-3}$$ grid expands the energy envelope
to 7.32%.

The weak force produces the most useful contrast. Judged against the original
physical energy, its maximum energy mismatch is 2.69%. Judged against its own
matching $$0.98U$$ energy, the envelope returns to 0.769%. Both measurements
come from the same wrong trajectory, whose RMS position error is 1.10. A flat
trace certifies internal consistency with the simulated Hamiltonian—not the
choice of Hamiltonian.

## Ask the same questions of real kUPS trajectories

The next cell runs five 32-atom CPU cases through
`kups.application.simulations.md.run`. It reopens each raw HDF5 trajectory and
reports frames, observed device, energy excursion, and content hash.

{% include kups-notebooks/post-03/post03-kups-errors.html %}

The smoke run is an execution check. Its eight stored frames already show the
20 fs case widening from a roughly $$10^{-6}$$ baseline envelope to
$$1.52\times10^{-3}$$, while the 0.98 potential conserves its own energy at the
$$10^{-6}$$ level.

The quantitative comparison uses the committed full GPU profile: 256 atoms,
50 stored frames over 1 ps, three 2 fs replicas, and matched-seed timestep,
precision, and potential cases. Every full trajectory records an NVIDIA RTX
A5000 device and hashes of its input and HDF5 output.

<div class="table-responsive" markdown="1">

| Full kUPS case | Maximum stored-frame $$\lvert\Delta E/E_0\rvert$$ | Energy span |
|---|---:|---:|
| 2 fs, float32, replica 0 | 0.528% | 0.385 meV/atom |
| 2 fs, float32, replica 1 | 0.544% | 0.392 meV/atom |
| 2 fs, float32, replica 2 | 0.545% | 0.399 meV/atom |
| 0.5 fs, float32 | 0.527% | 0.385 meV/atom |
| 20 fs, float32 | 0.613% | 0.447 meV/atom |
| 2 fs, float64 | 0.540% | 0.397 meV/atom |
| 2 fs, float32, $$0.98\epsilon$$ | 0.533% | 0.380 meV/atom |

</div>

All cases share an early rise near 0.5%, so the absolute envelope contains a
common initialization transient. The comparisons still carry information:
20 fs is wider than the matched 2 fs run, float64 lies inside replica
variation, and the altered potential conserves its own total energy.

The float64 case uses the same integer seed, but dtype can change random-number
generation and initialization. Treat it as a deployment comparison, not a
bitwise-identical initial state.

## Watch matched atoms separate

Energy compresses an atomic trajectory into one scalar per frame. The next
figure returns to positions. The 20 fs and 0.98-potential cases reuse the
baseline seed and storage times, so each atom can be compared with its baseline
counterpart using the periodic minimum-image displacement.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_trajectory_divergence.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Atomic displacement vectors from matched kUPS trajectories beside RMS trajectory separation over one picosecond" caption="Blue points show one layer of the final 2 fs baseline, while orange and red arrows show matched-atom differences for the 20 fs and 0.98-potential trajectories. Arrow lengths are magnified 100 times for visibility; the right panel reports unscaled periodic minimum-image RMS separation. After 1 ps, the timestep case differs by 0.005 angstrom RMS and the altered-potential case by 0.017 angstrom RMS, so a similarly flat energy trace does not imply the same atomic path." %}

The 2 fs baseline is not an exact solution. The plot measures **separation from
that reference trajectory**, not absolute truth. Its value is causal control:
the same initial seed and observation times isolate the changed timestep or
potential.

The altered potential separates atoms more than the 20 fs map over this 1 ps
window, even though its energy span is slightly smaller. That is the atom-level
version of the JAX oscillator result: conserving the wrong Hamiltonian can look
numerically cleaner while producing a less relevant trajectory.

## Replicas test conclusions, not pointwise paths

Two velocity seeds should not produce matching coordinates. Chaotic dynamics
makes pointwise agreement a poor expectation even when both trajectories are
valid. Replicas instead test whether a protocol-level conclusion survives
initial-state variation.

For the full 2 fs float32 runs, maximum energy excursions range from 0.528% to
0.545%. The float64 result at 0.540% sits inside that range. On this metric and
time window, changing precision does not exceed ordinary replica variation.
That statement is narrower than "float32 is accurate." A longer run or a
different observable may resolve another effect.

## Build an error audit in the right order

Use controls that isolate one mechanism:

1. Test the update against an exact or high-accuracy reference problem.
2. Sweep timesteps at fixed initial state, duration, and storage cadence.
3. Compare deployed precision policies on the target hardware.
4. Repeat independent seeds before treating one envelope as typical.
5. Check that reported energies and propagated forces are differentiably
   consistent.
6. Validate energy and force predictions against an external reference.
7. Repeat the convergence study for the observable supporting the scientific
   claim.

The final step prevents a common shortcut. A timestep acceptable for mean
potential energy need not resolve a vibrational spectrum, diffusion
coefficient, or rare-event rate.

## Check your understanding

Before changing the notebook, predict each outcome:

1. Halve the timestep while doubling the number of steps so final time remains
   fixed. Which error should decrease first?
2. Keep the weak force but switch the measured energy between $$U$$ and
   $$0.98U$$. Which energy trace is flatter, and why is the trajectory unchanged?
3. Change only the velocity key. Should matched-atom RMS separation remain a
   useful error metric?
4. Replace the artificial quantizer with float32 arrays. What additional
   hardware and reduction-order evidence would you record?

The second prediction separates dynamics from diagnosis. The same propagated
states can appear inconsistent or conservative depending on which Hamiltonian
is measured.

## Error claims need a named mechanism and observable

Post 02 showed that an integrator defines a discrete path. This chapter adds
three more facts: arithmetic perturbs that path, a potential selects the vector
field, and initial states create legitimate variation. No single energy number
identifies all three.

A defensible result states what changed, what stayed fixed, which observable
was compared, and which reference supports the conclusion. That is the error
contract the remaining chapters will use for thermostats, barostats, sampling,
and learned potentials.

<details class="kups-reproducibility" markdown="1">
<summary>Reproducibility record and complete diagnostic dashboard</summary>

Run and verify the CPU profile from a locked environment:

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

The complete four-panel audit retains the analytic timestep sweep, artificial
rounding cases, force-consistency control, energy traces, drift fits, and
replica comparisons:

{% include figure.liquid loading="lazy" path="assets/img/blog/kups_md_post03_error_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four-panel error validation dashboard for oscillator controls and kUPS trajectories" caption="The audit dashboard separates oscillator timestep, arithmetic, and force-consistency checks before showing all real kUPS energy traces. These panels support numerical validation, while the main figure explains trajectory separation at atomic resolution." %}

Source and evidence:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/full.json)
- [smoke compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/smoke/error_summary.json)
- [full compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/error_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/manifest.json)
- [executed notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-03-errors.ipynb)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post03_figures.py)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-03.md)
- [source repository](https://github.com/sungsoo-ahn/kups-md-tutorials)

</details>

## References

- <span id="ref-higham2002"></span>Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM. <a href="#cite-higham2002" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. <a href="#cite-hairer2006" class="reversefootnote" role="doc-backlink">↩</a>
