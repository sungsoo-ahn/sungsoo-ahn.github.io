---
layout: post
permalink: /kups-md-tutorials/post-08-free-energies/
title: "How Do Equilibrium Samples Become Free Energies?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Turn probabilities and real kUPS radial distribution functions into free-energy profiles without hiding binning, reweighting, support, or replica uncertainty."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 8
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 8
categories: [science]
tags: [molecular-dynamics, free-energy, potential-of-mean-force, reweighting, kups]
toc:
  sidebar: left
related_posts: false
nav: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The real-kUPS example is a bounded RDF-to-PMF diagnostic, not a converged argon free-energy calculation.</em>
</p>

## A Free-Energy Curve Is a Claim About Probability

Free energy amplifies weak sampling. If a probability estimate is too small by
a factor of two, the error becomes an additive (k_B Tlog 2). If a histogram
bin is empty, its logarithm is not “a very high barrier.” It is undefined from
the available samples.

That makes free-energy plots unusually easy to over-interpret. A smooth curve
can hide arbitrary binning, missing probability support, the wrong reweighting
sign, or an unreported additive shift. The right workflow makes each choice
visible before interpreting a barrier or basin.

This post uses two layers:

1. a double well with a known answer to test histograms, bootstrap uncertainty,
   and bias removal;
2. independent kUPS argon trajectories to test (W(r)=-k_B Tlog g(r)),
   low-(g(r)) support, and replica uncertainty.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-08-free-energies.ipynb),
[smoke control summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/smoke/free_energy_summary.json),
[full control summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/free_energy_summary.json),
[smoke kUPS PMF summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/smoke/kups_rdf_pmf_summary.json),
[full kUPS PMF summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/kups_rdf_pmf_summary.json),
[full curves](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/free_energy_curves.csv),
[full manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/manifest.json),
[figure generator](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post08_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-08.md).

## Calibrate the Logarithm on a Known Answer

For a coordinate (s) sampled from an equilibrium density (p(s)), the
relative free energy is

\[
F(s) = -k_B T\log p(s) + C.
\]

The constant (C) is arbitrary. The tutorial shifts each finite profile so
its minimum is zero. It does not fill empty bins with an invented energy.

{% include kups-notebooks/post-08/post08-setup.html %}

{% include kups-notebooks/post-08/post08-free-energy-control.html %}

The full control draws 80,000 samples from a double well whose barrier is
exactly 1 in reduced units.

<div class="table-responsive" markdown="1">

| Bin width | Estimated barrier | Bootstrap SE | Barrier error | Curve RMSE |
|---:|---:|---:|---:|---:|
| 0.06 | 0.985 | 0.032 | -0.015 | 0.171 |
| 0.18 | 0.976 | 0.017 | -0.024 | 0.366 |
| 0.35 | 0.915 | 0.017 | -0.085 | 0.591 |

</div>

The coarsest histogram has a small bootstrap error bar but the largest bias.
More samples from the same bins would not repair that discretization error.
Conversely, very narrow bins reduce smoothing bias but create noisy or empty
regions. Bin width is part of the estimator, not a plotting preference.

## Reweighting Must Undo the Bias

Suppose sampling used a bias (V_b(s)), so the observed density is

\[
p_b(s) \propto \exp\{-\beta[F(s)+V_b(s)]\}.
\]

Recovering the unbiased density therefore weights samples by

\[
w(s)=\exp\{+\beta V_b(s)\}.
\]

The sign is easy to reverse and difficult to diagnose from a pretty curve. In
the known-answer control, reweighting gives a barrier of 1.123: useful, but not
exact. The remaining 0.123 error reflects finite overlap and sampling, which
reweighting cannot manufacture.<sup id="cite-torrie"><a href="#ref-torrie">1</a></sup>

## An RDF Defines a Pair PMF Only Where It Has Support

For an isotropic homogeneous system, the radial distribution function removes
the spherical-shell reference measure. Its pair potential of mean force is

\[
W(r) = -k_B T\log g(r) + C.
\]

This is not the bare pair potential. It is an equilibrium, many-body quantity
for the selected state point. It also becomes unstable as (g(r)) approaches
zero. The code therefore declares a minimum supported RDF value and masks
smaller bins. Replacing zero with a tiny epsilon would create a finite number,
but that number would be controlled by the epsilon rather than the data.

The smoke notebook cell launches two new CPU kUPS replicas, estimates their
RDFs from stored positions and volumes, transforms the mean RDF, and propagates
replica variation through the logarithm.

{% include kups-notebooks/post-08/post08-kups-rdf-pmf.html %}

## What the GPU RDF-to-PMF Run Showed

The full profile uses three independent 256-atom Langevin replicas at 100 K.
Each stores 80 frames at 20 fs spacing after 200 warmup steps. The 8.0 Å RDF
range lies below the 10.52 Å half-box limit. Every worker observed
`gpu:NVIDIA RTX A5000`.

<div class="table-responsive" markdown="1">

| Replica | Frames | Mean temperature (K) | Runtime (s) | HDF5 SHA-256 prefix | Device |
|---|---:|---:|---:|---|---|
| 0 | 80 | 100.48 | 53.40 | `07d605fcf78f` | RTX A5000 |
| 1 | 80 | 98.84 | 50.17 | `e4a1d621418a` | RTX A5000 |
| 2 | 80 | 99.40 | 50.76 | `942f4c0281ed` | RTX A5000 |

</div>

The mean RDF peak is 4.679 at 3.64 Å, so the shifted PMF minimum is also at
3.64 Å. Under the primary (g(r)\ge 0.05) rule, 60 radial bins remain finite
and the displayed PMF range is 37.98 meV. The maximum between-replica PMF
standard deviation is only 0.85 meV.

That narrow replica band is not the whole uncertainty story. Changing only the
low-RDF support rule gives:

| Minimum supported (g(r)) | Finite bins | Shifted PMF range (meV) | PMF minimum (Å) |
|---:|---:|---:|---:|
| 0.02 | 61 | 45.50 | 3.64 |
| 0.05 | 60 | 37.98 | 3.64 |
| 0.10 | 57 | 31.27 | 3.64 |

The 14.24 meV span across support rules is much larger than the replica spread.
The minimum location is stable, but the high-free-energy range is not. A plot
that reports only the 0.05 curve would hide the dominant analysis sensitivity.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post08_free_energy_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Free-energy diagnostics for the committed full profile. The upper and lower-left panels calibrate histogram binning, reweighting, and the RDF logarithm on known answers. The lower-right panel transforms the mean RDF from three real 256-atom kUPS GPU replicas, showing the replica band and three explicit low-RDF support rules." %}

## What This Result Does and Does Not Mean

The controlled double well proves that the implementation can recover a known
barrier and reveal binning bias. The kUPS layer proves that real HDF5 positions
can flow through periodic RDF normalization, a physical (k_B T), support
masking, and replica uncertainty.

It does not prove a converged argon PMF. Eighty stored frames are deliberately
short. A production study would require longer warmup and trajectories, more
replicas, stability to RDF bin width, finite-size checks, and a physical
argument for the radial region being interpreted. If the target were a rare
transition rather than an RDF, enhanced sampling and overlap diagnostics would
be part of the estimator too.<sup id="cite-frenkel"><a href="#ref-frenkel">2</a></sup>

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 08 --profile smoke
uv run kups-tutorial verify 08 --profile smoke

# Requires an observed JAX GPU backend; CPU fallback fails this profile.
uv run kups-tutorial run 08 --profile full
uv run kups-tutorial verify 08 --profile full

uv run kups-tutorial verify-notebooks --posts 08
uv run python scripts/generate_post08_figures.py
```

The verifier checks the known-answer controls, bootstrap uncertainty,
reweighting tolerance, real kUPS provenance, periodic RDF support, finite PMF
bins and range, replica uncertainty, support-threshold sensitivity, and an
observed GPU for the full profile. The manifest records configs, seeds, HDF5
schemas and hashes, device evidence, entry point, and compact output names.

The practical rule is blunt: never turn a low-probability bin into a confident
free-energy barrier without showing the probability support that created it.

## References

1. <span id="ref-torrie"></span>Torrie, G. M. & Valleau, J. P. (1977). Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling. *Journal of Computational Physics*, 23, 187–199. [doi:10.1016/0021-9991(77)90121-8](https://doi.org/10.1016/0021-9991(77)90121-8). <a href="#cite-torrie" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-frenkel"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation* (2nd ed.). Academic Press. <a href="#cite-frenkel" class="reversefootnote" role="doc-backlink">↩</a>
