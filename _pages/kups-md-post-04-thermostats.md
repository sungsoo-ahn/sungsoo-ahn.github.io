---
layout: post
permalink: /kups-md-tutorials/post-04-thermostats/
title: "How Do Thermostats Change Sampling and Dynamics?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Run BAOAB Langevin and CSVR with kUPS, then separate temperature control, canonical sampling, and dynamical distortion."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 4
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 4
categories: [science]
tags: [molecular-dynamics, thermostats, langevin, csvr, sampling, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft remains hidden while the full series is being
validated. Corrections and replication issues belong in
<a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

A thermostat does more than keep a temperature trace near a target. It changes
the distribution sampled by the trajectory and, usually, its dynamical memory.
The right diagnostic therefore asks two different questions:

1. Does the trajectory sample the intended ensemble?
2. Does the thermostat preserve the dynamics needed by the observable?

This tutorial answers the first question with real kUPS simulations and uses a
harmonic oscillator to isolate the second. A short CPU run proves that the
configuration, HDF5 trajectory, and analysis pipeline work together. A
256-atom GPU run supplies the production statistics.

## A thermostat changes the trajectory

For one coordinate, Langevin dynamics can be written as

$$dx = v\,dt,$$

$$dv = \frac{F(x)}{m}\,dt - \gamma v\,dt
     + \sqrt{\frac{2\gamma k_{B}T}{m}}\,dW_t.$$

Here $$\gamma$$ removes momentum memory while the Wiener increment $$dW_t$$
injects thermal noise. The fluctuation-dissipation relation couples those two
terms so that the canonical distribution is stationary. Changing $$\gamma$$
does not merely change how quickly a displayed temperature settles; it changes
the time correlations of the path.

kUPS provides both the BAOAB Langevin splitting and canonical stochastic
velocity rescaling (CSVR). BAOAB applies local friction and noise to momenta.
CSVR acts through the kinetic energy and a thermostat time constant. Both can
target the canonical ensemble, but they do not generate identical dynamics.

The notebook imports the experiment and plotting interfaces in a collapsed
setup cell. The source is exported from a fresh notebook execution rather than
copied into this article by hand.

{% include kups-notebooks/post-04/a106b2a9.html %}

## Run BAOAB and CSVR through kUPS

The smoke experiment uses 32 argon atoms at a number density of
$$0.0275\ \mathrm{\mathring{A}}^{-3}$$. The Lennard-Jones parameters are
$$\sigma=3.405\ \mathrm{\mathring{A}}$$ and
$$\epsilon=0.010326\ \mathrm{eV}$$. Each case starts from the same periodic
structure and uses a fixed seed, a 2 fs timestep, eight warmup blocks, and
eight stored production frames.

The following cell calls `kups.application.simulations.md.run` for each smoke
thermostat, then loads the hash-pinned production summaries. Each run writes an
ignored raw HDF5 file. The tutorial reopens that file with the kUPS analysis API
and records its SHA-256 digest, schema, frame count, device, and compact
thermodynamic statistics.

{% include kups-notebooks/post-04/bc8d26c8.html %}

The smoke output establishes a narrow but useful fact: both thermostat
definitions run through kUPS 1.0.3 on the CPU and produce finite, analyzable
trajectories. Their mean temperatures are about 88.7 K for a 100 K target.
Eight frames are too few to interpret that difference as thermostat bias.

The production run uses 256 atoms, 20,000 warmup steps, 20,000 production
steps, and 1,000 stored frames per thermostat:

<div class="table-responsive" markdown="1">

| Thermostat | Mean temperature | Observed device | HDF5 SHA-256 prefix |
|---|---:|---|---|
| BAOAB Langevin | 99.970 ± 0.279 K | NVIDIA RTX A5000 | `79f8fe809975` |
| CSVR | 99.851 ± 0.424 K | NVIDIA RTX A5000 | `2d63432d3018` |

</div>

Both full-profile manifests record `production_gpu_ready: true`. The close
agreement with 100 K supports a low-order kinetic-temperature claim for this
protocol; it does not establish that BAOAB and CSVR preserve the same
time-dependent observables.

The HDF5 digest is more important than it may look. It binds the displayed
statistics to the raw trajectory that produced them without committing that
large trajectory to Git. A result cannot pass verification if it merely imports
kUPS, records a version string, or substitutes a NumPy integrator.

## Temperature is only a moment check

Temperature in classical MD is inferred from kinetic energy. A thermostat can
make that single number look reasonable while the configurational distribution
or temporal correlations remain wrong. A useful report checks at least three
surfaces:

| Surface | Diagnostic | Failure it exposes |
|---|---|---|
| kinetic distribution | temperature and kinetic-energy moments | incorrect heat exchange or degree count |
| configuration | energy and structural distributions | biased ensemble sampling |
| memory | autocorrelation and effective sample size | distorted dynamics or inefficient sampling |

For a harmonic oscillator with unit mass, unit angular frequency, and
$$k_{B}T=1$$, the canonical targets are exact:

$$\operatorname{Var}(x)=1, \qquad
  \operatorname{Var}(v)=1, \qquad
  \left\langle K\right\rangle=\frac{1}{2}.$$

That makes the oscillator a useful analytic control. The next cell runs the
tutorial-owned BAOAB reference integrator for a few steps. It is not presented
as kUPS evidence; its purpose is to make the stochastic splitting and its
sampled state explicit.

{% include kups-notebooks/post-04/adc44d5e.html %}

The longer committed oscillator experiment compares weak, moderate, and strong
coupling. All three runs remain near the canonical moments, but their memory is
different. The strong-coupling trajectory has a position integrated
autocorrelation time of about 52.7 saved samples, compared with about 10.1 for
weak coupling. Its effective sample count falls from roughly 348 to 66 despite
having the same number of stored points.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post04_thermostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The analytic oscillator separates canonical moments from dynamical memory; the fourth panel is HDF5-derived real kUPS evidence. Strong Langevin coupling damps velocity memory but increases position autocorrelation time, while the 1,000-frame BAOAB and CSVR GPU runs both remain near the 100 K target." %}

## Choose the thermostat for the observable

Stronger coupling does not mean better sampling in every coordinate. In the
overdamped limit, velocities decorrelate quickly while positions can move
slowly. A thermostat may therefore be useful for equilibration and poor for a
time-correlation measurement.

For static canonical observables, leaving a validated thermostat on during
production can be appropriate. For diffusion coefficients, vibrational
spectra, velocity autocorrelations, and other dynamical quantities, a common
workflow is to equilibrate under NVT and then hand the final state to an NVE
segment. The handoff must preserve positions and momenta, and the NVE segment
must pass the energy checks from the integrator tutorials.

The same rule matters for machine-learned potentials. A thermostat can remove
energy injected by noisy or extrapolative forces. A stable temperature trace
can therefore hide a model failure that appears immediately in NVE. Ensemble
control is not a repair mechanism for an invalid force field.

CSVR, Nosé-Hoover chains, Andersen collisions, and Langevin variants make
different compromises. The choice should be justified by the ensemble and
observable, not by habit. BAOAB is used here because its split is transparent
and its configurational behavior is well understood in simple systems
(<span id="cite-leimkuhler2013"></span>[Leimkuhler & Matthews,
2013](#ref-leimkuhler2013)). CSVR is included because it provides a useful
global canonical comparison (<span id="cite-bussi2007"></span>[Bussi et al.,
2007](#ref-bussi2007)).

## Reproduce the result

The notebook is the presentation layer; the reusable runner, HDF5 inspection,
and verification logic live in the Python package.

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 04 --profile smoke
uv run kups-tutorial verify 04 --profile smoke
uv run kups-tutorial verify-notebooks --posts 04 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 04 --check
```

The compact [smoke](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/smoke/kups_md_summary.json)
and [production](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/kups_md_summary.json)
summaries record the values shown above. The
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/full.json),
[full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/manifest.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-04-thermostats.ipynb),
[figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post04_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-04.md)
complete the reproducibility chain.

## References

- <span id="ref-leimkuhler2013"></span>Leimkuhler, B. & Matthews, C. (2013). Rational construction of stochastic numerical methods for molecular sampling. *Applied Mathematics Research eXpress*, 2013(1), 34–56. <a href="#cite-leimkuhler2013" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bussi2007"></span>Bussi, G., Donadio, D. & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *Journal of Chemical Physics*, 126, 014101. <a href="#cite-bussi2007" class="reversefootnote" role="doc-backlink">↩</a>
