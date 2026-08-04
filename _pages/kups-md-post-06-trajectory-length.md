---
layout: post
permalink: /kups-md-tutorials/post-06-trajectory-length/
title: "When Is a Trajectory Long Enough to Trust?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Use warmup, autocorrelation, effective sample size, and independent kUPS replicas to decide what a molecular-dynamics trajectory can support."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 6
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 6
categories: [science]
tags: [molecular-dynamics, autocorrelation, effective-sample-size, uncertainty, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The physical examples are bounded trajectory-length diagnostics, not converged argon thermodynamics.</em>
</p>

## Frame Count Is the Wrong Question

A million MD frames can contain less information than a thousand independent
samples. Adjacent states share positions, velocities, and local structure. If
an observable relaxes slowly, saving it more often makes a larger file—not a
more precise estimate.

“How many steps did you run?” is therefore incomplete. The useful questions are:

- Which early frames were discarded, and why?
- What is the autocorrelation time of the reported observable?
- How many effective samples remain?
- Do independent replicas agree?
- Does the uncertainty shrink when the trajectory is extended?

The last question has a subtle answer. Uncertainty should shrink in the
long-run limit, but finite checkpoint estimates need not improve monotonically.
A longer prefix can reveal slower correlation or larger replica disagreement
that the shorter prefix missed. That is information, not a failed diagnostic.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-06-trajectory-length.ipynb),
[smoke kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/smoke/kups_trajectory_length_summary.json),
[full kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/kups_trajectory_length_summary.json),
[full observable trace](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/kups_observable_samples.csv),
[full manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/manifest.json),
[figure generator](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post06_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-06.md).

## Calibrate the Statistics Before the Physics

The notebook begins with a correlated process whose equilibrium mean is known.
It starts from a deliberate bias, relaxes toward the target, and retains
temporal memory after warmup. This is not molecular dynamics. It is an answer
key for the uncertainty machinery.

{% include kups-notebooks/post-06/post06-setup.html %}

{% include kups-notebooks/post-06/post06-statistical-control.html %}

At 24,000 steps, the control has 34,500 retained values across six replicas but
only about 2,296 effective samples. Its naive standard error is 0.0054; the
review error is 0.0208. Treating frames as independent would report roughly
four times too much precision.

For a stationary series, a common schematic is

$$
N_{\mathrm{eff}} \approx \frac{N}{\tau_{\mathrm{int}}},
$$

where the precise convention for $$\tau_{\mathrm{int}}$$ determines whether a
factor of two appears. The tutorial uses one convention consistently and
records it with the results. The scientific point does not depend on notation:
correlation reduces information.<sup id="cite-sokal"><a href="#ref-sokal">1</a></sup>

## Warmup Solves Only One Problem

Discarding warmup can reduce initial-condition bias. It does not make the
remaining frames independent, prove that every slow coordinate equilibrated,
or force replicas into the same metastable basin.

Equilibration is observable-specific. Temperature can settle while density,
defect populations, coordination, or a conformational coordinate continues to
drift. A reliable review varies the discard moderately and checks whether the
conclusion survives. Automated truncation methods can help, but their job is to
balance bias against lost effective samples—not to certify the physics by
themselves.<sup id="cite-chodera"><a href="#ref-chodera">2</a></sup>

Independent replicas are especially useful here. One long trajectory may look
smooth because it remains trapped. Several replicas expose between-run
disagreement that no within-run autocorrelation estimate can recover.

## Run the Same Review on Real kUPS Data

The physical protocol uses Lennard-Jones argon at 100 K and fixed volume. Each
replica runs the kUPS `baoab_langevin` integrator with a 2 fs timestep and an
independent seed. kUPS writes positions and potential energy to HDF5. The
tutorial derives two observables from those stored frames:

- potential energy in eV per atom;
- mean coordination inside a 5.1 Å cutoff.

The full profile runs three 256-atom replicas, discards 200 warmup steps, and
stores 80 production frames per replica at 20 fs spacing. The smoke profile is
smaller and runs on CPU. Its open notebook cell actually launches kUPS; it does
not load committed JSON as proof of execution.

{% include kups-notebooks/post-06/post06-kups-length.html %}

Potential energy and coordination do not have to share an autocorrelation time.
That is why “the trajectory ESS” is not a complete concept. Effective sample
size belongs to an observable and estimator.

## What the GPU Checkpoints Showed

Every full worker observed `gpu:NVIDIA RTX A5000`. The three HDF5 files have
distinct SHA-256 digests and contain 80 frames for 256 atoms. Checkpoints reuse
the first 20, 40, and 80 frames of each replica.

<div class="table-responsive" markdown="1">

| Frames per replica | Time (fs) | Raw frames | PE ESS | PE 95% half-width (eV/atom) | Coord. ESS | Coord. 95% half-width |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 400 | 60 | 12.1 | 0.000311 | 22.6 | 0.0498 |
| 40 | 800 | 120 | 14.3 | 0.000360 | 16.5 | 0.0382 |
| 80 | 1600 | 240 | 36.9 | 0.000199 | 32.8 | 0.0238 |

</div>

The final prefix contains 240 raw frames but only about 37 effective
potential-energy samples and 33 effective coordination samples. Both final
uncertainties are smaller than at the first checkpoint, but the middle prefix
is not uniformly better: the energy interval grows slightly, and the estimated
coordination ESS falls. The longer prefix revealed more memory and replica
variation before the final checkpoint accumulated enough information to win.

The checkpoint means remain compact:

<div class="table-responsive" markdown="1">

| Frames per replica | Mean PE (eV/atom) | Mean coordination |
|---:|---:|---:|
| 20 | -0.069983 | 13.513 |
| 40 | -0.069810 | 13.548 |
| 80 | -0.069795 | 13.540 |

</div>

Those stable-looking means do not erase the ESS result. A mean can change
little while its uncertainty estimate changes substantially.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post06_trajectory_length_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Trajectory-length diagnostics for the committed full profile. The first three panels calibrate warmup, correlation-aware uncertainty, and effective sample size on a process with a known mean. The final panel shows checkpointed potential-energy and coordination uncertainty derived from three real kUPS GPU HDF5 replicas." %}

## What Counts as “Long Enough”?

There is no universal step count. A trajectory is long enough only relative to
a claim and tolerance. A mean potential energy may converge before a rare
transition rate. Coordination may decorrelate faster than a collective phase
coordinate. A static average may be usable while a diffusion coefficient is
not.

A defensible stopping rule states four things:

1. the observable and estimator;
2. the warmup rule;
3. an autocorrelation- or block-aware uncertainty target;
4. a replica-agreement check.

Blocking is useful because block means become less correlated as the block size
grows. A plateau in the estimated error is evidence that blocks are long enough
to capture memory.<sup id="cite-flyvbjerg"><a href="#ref-flyvbjerg">3</a></sup>
It is not magic: too few blocks produce a noisy error estimate, and metastable
replica disagreement can dominate any within-run block calculation.

This Post 06 run is intentionally bounded. Eighty stored frames are enough to
show that raw count and ESS diverge, to compare two observables, and to prove an
observed GPU kUPS path. They are not enough to claim a converged argon equation
of state, long-time dynamics, or a universal 5.1 Å coordination definition.

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 06 --profile smoke
uv run kups-tutorial verify 06 --profile smoke

# Requires an observed JAX GPU backend; CPU fallback fails this profile.
uv run kups-tutorial run 06 --profile full
uv run kups-tutorial verify 06 --profile full

uv run kups-tutorial verify-notebooks --posts 06
uv run python scripts/generate_post06_figures.py
```

The verifier requires matched kUPS replica counts, finite HDF5 evidence, at
least eight stored frames per case, increasing checkpoint sizes, positive
observable ESS, and an observed GPU for the full profile. The manifest records
the config, entry point, seeds, dataset schemas, device evidence, raw HDF5
hashes, and compact output names.

The practical rule is blunt: report effective samples and replica agreement,
not just trajectory length. Warmup removes selected history. It does not turn a
correlated simulation into independent data.

## References

1. <span id="ref-sokal"></span>Sokal, A. D. (1997). Monte Carlo methods in statistical mechanics: foundations and new algorithms. In *Functional Integration*. <a href="#cite-sokal" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-chodera"></span>Chodera, J. D. (2016). A simple method for automated equilibration detection in molecular simulations. *Journal of Chemical Theory and Computation*, 12, 1799–1805. [doi:10.1021/acs.jctc.5b00784](https://doi.org/10.1021/acs.jctc.5b00784). <a href="#cite-chodera" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-flyvbjerg"></span>Flyvbjerg, H. & Petersen, H. G. (1989). Error estimates on averages of correlated data. *Journal of Chemical Physics*, 91, 461–466. [doi:10.1063/1.457480](https://doi.org/10.1063/1.457480). <a href="#cite-flyvbjerg" class="reversefootnote" role="doc-backlink">↩</a>
