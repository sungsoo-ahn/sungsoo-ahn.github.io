---
layout: post
permalink: /kups-md-tutorials/post-05-barostats/
title: "How Should Pressure and Cell Degrees of Freedom Be Coupled?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Run isotropic and fully flexible NPT paths in kUPS, then inspect pressure, temperature, and moving-cell response from recorded HDF5 trajectories."
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
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. Every numerical claim below comes from committed smoke or GPU-profile artifacts. The short runs test implementation and response; they do not establish converged argon thermodynamics.</em>
</p>

## A Pressure Target Is Not a Flat Pressure Trace

If an NPT trajectory sits exactly at its target pressure, be suspicious.
Pressure in a small atomistic box is noisy. The barostat controls a distribution
by moving the cell; it does not clamp each frame to one number.

That distinction changes the review question. “Did the pressure reach 10 MPa?”
is too weak. We need to ask:

- Did the cell move in the expected direction?
- Did the thermostat keep the kinetic temperature plausible?
- How fast did each pressure-coupling choice respond?
- Are volume and pressure still fluctuating after the transient?
- Does the stored evidence support an isotropic-volume claim, a cell-shape
  claim, or only an implementation check?

The last question matters especially for ML potentials. An NPT integrator can
run perfectly while the learned stress is wrong outside its training domain.
Energy conservation cannot validate a pressure model, and a moving box cannot
validate an MLIP under strain.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-05-barostats.ipynb),
[smoke kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/smoke/kups_md_summary.json),
[full kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/kups_md_summary.json),
[full NPT trace](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/kups_npt_samples.csv),
[full manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/manifest.json),
[figure generator](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post05_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-05.md).

## First Calibrate the Statistic

Before touching atomistic NPT, the notebook checks a scalar stochastic process
whose volume variance is known. All three control runs have the same target
distribution. Only the relaxation time changes. The variance ratios stay near
one, while the integrated autocorrelation time grows from about 2 to 23 stored
samples.

{% include kups-notebooks/post-05/post05-setup.html %}

{% include kups-notebooks/post-05/post05-scalar-control.html %}

This control is not molecular dynamics. It is a ruler. It makes one lesson
unambiguous: the number of stored frames is not the number of independent cell
samples. A slow barostat can produce a smooth, long trajectory with very little
statistical information.

For a real NPT ensemble, volume fluctuations are related to the isothermal
compressibility. The exact estimator and finite-size corrections depend on the
system, but suppressing volume fluctuations merely because they look noisy is
not a valid stability fix.<sup id="cite-frenkel"><a href="#ref-frenkel">1</a></sup>

## kUPS Exposes Two Different Cell Models

kUPS 1.0.3 provides two NPT paths used here:

<div class="table-responsive" markdown="1">

| kUPS integrator | Cell motion | Thermostat/barostat idea |
|---|---|---|
| `csvr_npt` | isotropic scaling | CSVR thermostat plus stochastic cell rescaling |
| `baoab_npt_langevin` | fully flexible cell | atom and cell Langevin dynamics with BAOAB splitting |

</div>

The fully flexible path follows the extended-variable formulation described by
Gao, Fang, and Wang.<sup id="cite-gao"><a href="#ref-gao">2</a></sup> These
are not interchangeable switches. Isotropic scaling assumes one scalar cell
mode. A fully flexible cell can represent shear and anisotropic strain, which
is useful for solids but also exposes more ways for a bad stress model to fail.

The experiment uses Lennard-Jones argon in physical units:

<div class="table-responsive" markdown="1">

| Parameter | Full-profile value |
|---|---:|
| atoms | 256 |
| initial number density | 0.0275 Å\(^{-3}\) |
| temperature target | 100 K |
| pressure target | 10 MPa |
| timestep | 2 fs |
| warmup | 200 steps |
| production | 800 steps |
| stored frames | 80 per case |
| compressibility parameter | \(5\times10^{-10}\) Pa\(^{-1}\) |
| pressure-coupling times | 0.5, 1.0, and 2.0 ps |
| target runtime | GPU |

</div>

Ten MPa is deliberately high enough for a short response test. This is not a
one-bar equation-of-state calculation. The starting FCC cell is dense, the
trajectory is only 1.6 ps after warmup, and no result below is presented as an
equilibrium argon property.

## Run the Moving Cell

The open notebook cell runs both NPT implementations through
`kups.application.simulations.md.run`. Each case executes in a separate process
because JAX donates buffers during kUPS propagation. kUPS writes the trajectory
to HDF5; the tutorial then extracts volume, virial stress, kinetic temperature,
and total energy.

{% include kups-notebooks/post-05/post05-kups-npt.html %}

There is an easy unit bug here. kUPS stores the virial stress in its internal
eV/Å³ units. The tutorial converts by
\(1\ \mathrm{Pa}=6.241509\times10^{-12}\ \mathrm{eV}/\text{Å}^3\) before a
column is named `pressure_pa`. The wrapper tests this boundary explicitly; it
does not attach SI labels to raw internal values.

The smoke result is intentionally small: 32 atoms, 16 stored frames, and a CPU
backend. Its job is to prove that both algorithms compile, move the cell, and
produce finite HDF5 datasets. The full profile is the evidence used for the
scientific interpretation.

## What the GPU Run Actually Showed

All three full-profile workers observed `gpu:NVIDIA RTX A5000`. Each wrote 80
frames for 256 atoms. The HDF5 hashes differ across cases, so the table is tied
to three distinct raw trajectories.

<div class="table-responsive" markdown="1">

| Case | Mean \(T\) (K) | Mean \(P\) (MPa) | Final \(V/V_{first}\) | Final \(P\) (MPa) |
|---|---:|---:|---:|---:|
| CSVR–NPT, \(\tau_P=0.5\) ps | 95.0 | 38.2 | 1.0970 | 30.8 |
| CSVR–NPT, \(\tau_P=2.0\) ps | 100.6 | 139.0 | 1.0915 | 61.6 |
| BAOAB–NPT, \(\tau_P=1.0\) ps | 99.1 | 3.1 | 1.0921 | 7.6 |

</div>

The mean pressure is not expected to equal the target in this transient. The
slow isotropic case still remembers its high-pressure starting state. The
faster isotropic case responds more quickly. The flexible-cell trajectory
crosses the target and fluctuates on both sides, giving a mean below 10 MPa but
a final stored pressure near it. These are response-time observations, not a
ranking of integrators and not evidence that the flexible algorithm has
equilibrated better.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post05_barostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Real kUPS NPT evidence from the committed full GPU profile. The cell expands from the first stored production frame, instantaneous pressure remains noisy, temperature is checked independently, and the short-run volume response is separated from the span of stored fluctuations." %}

The figure also explains why reporting only a mean pressure is dangerous. The
2 ps case has a reasonable temperature but remains far from the pressure target
over the bounded window. A stable thermostat does not imply an equilibrated
barostat.

## What This Does Not Validate

The run exercises the fully flexible kUPS code path, but the stable per-step
HDF5 surface currently stores positions, energies, virial stress, and volume—not
the full cell matrix. Therefore this article validates moving-cell volume
response for that path. It does not claim to have measured cell angles, shear
relaxation, or a flexible-cell shape distribution.

The trajectory is also too short for a compressibility estimate. Eighty stored
frames can show motion and fluctuations, but an NPT ensemble claim needs longer
runs, warmup sensitivity, multiple seeds, autocorrelation-aware uncertainty,
and usually finite-size checks. In a crystal, it also needs a direct review of
cell vectors and stress anisotropy. In an MLIP workflow, it needs stress
validation against reference calculations in the strained configurations the
barostat visits.

The useful hierarchy is:

1. **Smoke:** both NPT paths compile and write finite HDF5 evidence.
2. **Response:** the cell moves in a physically interpretable direction while
   temperature remains controlled.
3. **Sampling:** longer replicas recover stable means, variances, and effective
   sample counts.
4. **Model validity:** reference forces and stresses support the thermodynamic
   interpretation.

This post completes the first two gates. It deliberately leaves the latter two
as requirements for a material-specific production study.

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 05 --profile smoke
uv run kups-tutorial verify 05 --profile smoke

# Requires a working JAX GPU backend; CPU fallback does not satisfy this profile.
uv run kups-tutorial run 05 --profile full
uv run kups-tutorial verify 05 --profile full

uv run kups-tutorial verify-notebooks --posts 05
uv run python scripts/generate_post05_figures.py
```

The verifier requires the configured integrator set, at least eight frames per
case, finite positive volumes, the expected HDF5 evidence surface, and an
observed GPU for the GPU-targeted profile. The manifest records the config hash,
kUPS version, entry point, runtime device, per-case HDF5 SHA-256 digest, and
elapsed time.

The practical rule is simple: do not call NPT successful because the code ran
or because one pressure average looks plausible. Show the cell response, keep
temperature and pressure diagnostics separate, estimate cell memory, and state
exactly which cell degrees of freedom the stored data can support.

## References

1. <span id="ref-frenkel"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-gao"></span>Gao, X., Fang, J. & Wang, H. (2016). Sampling the isothermal-isobaric ensemble by Langevin dynamics. *Journal of Chemical Physics*, 144, 124113. [arXiv:1601.01044](https://arxiv.org/abs/1601.01044). <a href="#cite-gao" class="reversefootnote" role="doc-backlink">↩</a>
