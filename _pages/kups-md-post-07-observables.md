---
layout: post
permalink: /kups-md-tutorials/post-07-observables/
title: "How Do Trajectories Become Physical Observables?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Estimate RDF, coordination, and velocity autocorrelation from real kUPS trajectories without losing normalization, periodic support, or uncertainty."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 7
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 7
categories: [science]
tags: [molecular-dynamics, rdf, observables, correlation-functions, kups]
toc:
  sidebar: left
related_posts: false
nav: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The physical examples are bounded estimator diagnostics, not converged argon structure or transport calculations.</em>
</p>

## A Trajectory Is Not a Result

An MD file contains microscopic states. It does not contain an RDF, a
coordination number, or a diffusion coefficient waiting to be read from a
column. Those quantities appear only after we choose an estimator.

That choice carries obligations:

- define the microscopic inputs;
- state the normalization and units;
- respect the support allowed by the periodic cell;
- account for correlation and replica disagreement;
- preserve enough provenance to repeat the measurement.

This distinction is easy to miss because a plotting function can turn almost
any trajectory into a smooth curve. Smoothness is not validation. A raw
pair-distance histogram can look like an RDF while having the wrong radial
normalization. A velocity-autocorrelation tail can look stable while being
supported by very few time origins.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-07-observables.ipynb),
[smoke kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/smoke/kups_observable_summary.json),
[full kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_observable_summary.json),
[RDF samples](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_rdf_samples.csv),
[VACF samples](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_vacf_samples.csv),
[full manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/manifest.json),
[figure generator](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post07_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-07.md).

## Test the Estimator Before the Physics

The notebook begins with periodic FCC cells whose local structure is known.
Each frame adds a seeded displacement to the lattice. This is not a liquid
simulation. It is an answer key for the RDF normalization, coordination
integral, finite-size mask, and block uncertainty.

{% include kups-notebooks/post-07/post07-setup.html %}

{% include kups-notebooks/post-07/post07-estimator-control.html %}

Both controlled systems recover a first-shell coordination near 12. The
32-atom cell, however, requests an RDF radius 1.39 times its half-box length.
Those unsupported bins are masked. The 256-atom cell has a ratio of 0.695 and
therefore supports the requested radius.

That check matters because periodic boundaries do not create an infinite
isotropic sphere around every atom. With a minimum-image estimator, the safe
radial support ends at half the shortest cell length. Drawing a curve beyond
that point does not create information.

## An RDF Is a Normalized Pair Estimator

A pair histogram counts separations. An RDF asks how the observed pair density
compares with the bulk density. For a bin bounded by (r_i) and (r_{i+1}),
the tutorial uses the shell volume directly:

\[
\Delta V_i = \frac{4\pi}{3}\left(r_{i+1}^3-r_i^3\right).
\]

If each unordered pair is counted once, a schematic finite-sample estimator is

\[
g_i = \frac{n_i}
{\tfrac{1}{2}N\rho\,\Delta V_i\,N_{\mathrm{frames}}}.
\]

The factor of one half changes if pairs are counted twice. The convention is
less important than using it consistently. Omitting the shell volume makes
large-radius bins look artificially populated; omitting density or frame count
makes otherwise identical systems incomparable.<sup id="cite-allen"><a href="#ref-allen">1</a></sup>

Coordination then inherits every RDF choice:

\[
n_c(r_c) = 4\pi\rho \int_0^{r_c} r^2 g(r)\,dr.
\]

The cutoff is part of the observable. Moving it across a minimum or the next
shell changes the reported coordination even when the trajectory is unchanged.

## Run the Same Estimators on kUPS HDF5

The physical workflow uses Lennard-Jones argon at 100 K and fixed volume. kUPS
runs the `baoab_langevin` integrator with a 2 fs timestep. Every stored frame is
20 fs apart. The analysis reads four HDF5 quantities rather than a precomputed
plot:

- positions for minimum-image pair distances;
- cell volume for density and half-box support;
- momenta and masses for velocities;
- replica identity for between-run uncertainty.

The open notebook cell launches two new CPU smoke replicas. It does not load a
committed physical summary as proof of execution.

{% include kups-notebooks/post-07/post07-kups-observables.html %}

The full profile uses three independent seeds, 256 atoms per replica, and 80
stored frames per replica. Its 8.0 Å RDF range is below the 10.52 Å half-box
limit. Every worker observed `gpu:NVIDIA RTX A5000`.

<div class="table-responsive" markdown="1">

| Replica | Frames | Mean temperature (K) | Runtime (s) | HDF5 SHA-256 prefix | Device |
|---|---:|---:|---:|---|---|
| 0 | 80 | 99.89 | 53.43 | `3aa322ef0943` | RTX A5000 |
| 1 | 80 | 98.45 | 50.00 | `d065bbd2cda2` | RTX A5000 |
| 2 | 80 | 102.23 | 50.46 | `342396071b71` | RTX A5000 |

</div>

Distinct seeds and file hashes do not prove independence of every physical
mode, but they do prove that the uncertainty band is not three copies of one
trajectory.

## What the GPU Observables Showed

The mean RDF has its first peak at 3.64 Å with height 4.61. Integrating through
5.1 Å gives a coordination of 13.696. Across the three replicas, the
coordination ranges from 13.678 to 13.722, giving a replica standard error of
0.013.

The number is not presented as a universal argon coordination. It belongs to
this density, temperature, potential, cutoff, warmup, and bounded trajectory.
The useful evidence is the complete estimator path: periodic positions become
a normalized RDF, the RDF becomes a cutoff-dependent integral, and independent
replicas attach uncertainty.

The VACF is estimated from velocities obtained as momentum divided by mass:

\[
C_v(t) =
\frac{\left\langle \mathbf{v}(0)\!\cdot\!\mathbf{v}(t)\right\rangle}
{\left\langle \mathbf{v}(0)\!\cdot\!\mathbf{v}(0)\right\rangle}.
\]

At the first stored lag, 20 fs, the mean correlation is 0.799. It crosses zero
near 140 fs. Integrating the displayed 0–600 fs window gives 7.06 fs with a
replica standard error of 0.33 fs. That integral is a diagnostic of the sampled
VACF, not a diffusion coefficient: the sampling interval is coarse, the tail
is short, and no hydrodynamic finite-size correction is attempted.<sup id="cite-kubo"><a href="#ref-kubo">2</a></sup>

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post07_observable_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Observable diagnostics for the committed full profile. The upper panels validate RDF normalization, finite-size support, and coordination uncertainty on controlled periodic cells. The lower panels show VACF and RDF estimates derived from three real 256-atom kUPS GPU HDF5 replicas, with between-replica standard-deviation bands." %}

## Curves Need Uncertainty Too

A scalar error bar is not enough for a curve. RDF bins share atoms and frames;
VACF lags share time origins. The full figure therefore plots the mean curve
and the between-replica standard deviation at every bin or lag. The maximum
replica standard deviation is 0.077 for the RDF and 0.0088 for the normalized
VACF.

Replica variation answers a different question from a within-trajectory block
estimate. Blocks probe finite-time correlation inside one run. Replicas expose
initialization and trajectory-to-trajectory disagreement. A serious observable
study often needs both, especially when metastability makes one long trajectory
look deceptively smooth.<sup id="cite-frenkel"><a href="#ref-frenkel">3</a></sup>

The bounded run here is enough to demonstrate the measurement contract and an
observed-GPU kUPS path. It is not long enough for a converged liquid structure,
a transport coefficient, or a model-quality claim. Those would require longer
warmup, longer trajectories, sensitivity to bin width and cutoff, more
replicas, and a finite-size study designed around the target observable.

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 07 --profile smoke
uv run kups-tutorial verify 07 --profile smoke

# Requires an observed JAX GPU backend; CPU fallback fails this profile.
uv run kups-tutorial run 07 --profile full
uv run kups-tutorial verify 07 --profile full

uv run kups-tutorial verify-notebooks --posts 07
uv run python scripts/generate_post07_figures.py
```

The verifier requires real kUPS evidence, matching replica counts, RDF support
inside the half box, a physical RDF peak, positive coordination and replica
uncertainty, finite VACF statistics, and an observed GPU for the full profile.
The manifest records configs, seeds, HDF5 schemas and hashes, device evidence,
entry point, and compact output names.

The practical rule is simple: never report a trajectory-derived curve without
also reporting how it was normalized, where it is valid, and how uncertain it
is.

## References

1. <span id="ref-allen"></span>Allen, M. P. & Tildesley, D. J. (1987). *Computer Simulation of Liquids*. Oxford University Press. <a href="#cite-allen" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-kubo"></span>Kubo, R. (1957). Statistical-mechanical theory of irreversible processes. I. *Journal of the Physical Society of Japan*, 12, 570–586. [doi:10.1143/JPSJ.12.570](https://doi.org/10.1143/JPSJ.12.570). <a href="#cite-kubo" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-frenkel"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation* (2nd ed.). Academic Press. <a href="#cite-frenkel" class="reversefootnote" role="doc-backlink">↩</a>
