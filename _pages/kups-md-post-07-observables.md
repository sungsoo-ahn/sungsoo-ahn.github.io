---
layout: post
permalink: /kups-md-tutorials/post-07-observables/
title: "How Do Trajectories Become Physical Observables?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Implement a periodic RDF and coordination estimator in JAX, then measure structure and velocity memory from real kUPS trajectories."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 7
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 7
categories: [science]
tags: [molecular-dynamics, rdf, observables, correlation-functions, jax, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
collapse_code: true
---

An MD trajectory is a sequence of microscopic states. It stores where atoms
were and how they moved. It does not contain a radial distribution function,
coordination number, or diffusion coefficient waiting in a column. Those
quantities appear only after an estimator turns the stored state into a
measurement.

That distinction matters because almost any trajectory can produce a smooth
curve. A raw distance histogram can resemble an RDF while missing the shell
volume and density normalization. A velocity-correlation tail can look stable
even when only a few time origins support it. Plotting is the last step, not the
validation.

We will make the estimator visible. First, a short JAX function will apply
minimum-image geometry, count unordered pairs, normalize spherical shells, and
integrate coordination. Then the same operations will be applied to real kUPS
HDF5 positions, while momenta and masses provide a velocity autocorrelation.

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- how a microscopic trajectory becomes a macroscopic estimator;
- how periodic minimum-image distances enter an RDF;
- why pair counts must be divided by shell volume, density, frames, and pair
  convention;
- how coordination is an integral with a user-chosen cutoff;
- how velocity autocorrelation uses many overlapping time origins;
- how support limits and replica uncertainty constrain every curve.

**Prerequisites:** periodic cells from the
[foundations lesson]({% link _pages/kups-md-foundations.md %}), real HDF5 state
from [Post 01]({% link _pages/kups-md-post-01-initialization.md %}), and
observable-specific effective samples from
[Post 06]({% link _pages/kups-md-post-06-trajectory-length.md %}).
</div>

## An observable is a rule applied to state

At stored frame $$t$$, let the atom positions be
$$\mathbf R_t=(\mathbf r_{1,t},\ldots,\mathbf r_{N,t})$$ and momenta be
$$\mathbf P_t$$. A scalar observable is a function of that state:

$$
A_t=A(\mathbf R_t,\mathbf P_t,\mathbf h_t),
$$

where $$\mathbf h_t$$ is the periodic cell. An equilibrium average estimates
an ensemble expectation by averaging the time series,

$$
\langle A\rangle \approx \frac{1}{N_f}\sum_{t=0}^{N_f-1} A_t.
$$

The function $$A$$ is the scientific definition. It specifies geometry,
normalization, units, parameters, and valid support. The average adds sampling
assumptions: warmup, correlation, and replica coverage. Two groups can analyze
the same HDF5 file and obtain different “coordination numbers” simply because
they used different cutoffs.

This chapter uses two kinds of observable:

- **structural:** the RDF and its coordination integral depend on positions;
- **dynamical:** the velocity autocorrelation depends on momenta at separated
  times.

The first asks how atoms are arranged. The second asks how motion remembers
itself.

## Start from periodic pair displacement

For atoms $$i$$ and $$j$$ in a cubic cell of length $$L$$, the raw displacement
is $$\Delta\mathbf r_{ij}=\mathbf r_i-\mathbf r_j$$. Periodic copies make that
vector ambiguous. The minimum-image convention chooses the nearest copy:

$$
\Delta\mathbf r_{ij}^{\mathrm{MIC}}
=
\Delta\mathbf r_{ij}
-L\operatorname{round}\!\left(\frac{\Delta\mathbf r_{ij}}{L}\right).
$$

The pair distance is its norm. This componentwise expression is appropriate
for the cubic cells used here. General triclinic cells are handled more safely
through fractional coordinates and the complete cell matrix.

We count each unordered pair once, so only indices $$i<j$$ enter the
histogram. Counting both $$(i,j)$$ and $$(j,i)$$ is also valid, but the expected
count must then change by a factor of two.

Periodic boundaries do not grant unlimited radial support. Under this cubic
minimum-image estimator, a complete spherical shell must satisfy

$$
r \le \frac{L}{2}.
$$

Beyond half the box, corners of the nominal shell cannot be represented
uniformly by minimum images. Drawing an RDF line there does not create the
missing pair geometry.

## Pair counts become an RDF only after normalization

Let radial bin $$b$$ span $$[r_b,r_{b+1})$$. Its exact spherical-shell volume
is

$$
\Delta V_b
=\frac{4\pi}{3}\left(r_{b+1}^3-r_b^3\right).
$$

For $$N$$ atoms, number density $$\rho=N/V$$, $$N_f$$ frames, and unordered
pairs, an ideal-gas reference contributes the expected count

$$
n_b^{\mathrm{ideal}}
=\frac{1}{2}N_fN\rho\,\Delta V_b.
$$

The RDF estimator is therefore

$$
g_b=\frac{n_b}{n_b^{\mathrm{ideal}}}.
$$

The shell volume removes the geometric growth of available space with radius.
The density makes systems comparable, the frame count converts accumulated
counts to a per-frame estimate, and the factor one-half matches the unordered
pair convention. In a spatially uniform ideal gas, $$g(r)\rightarrow1$$ away
from finite-size and sampling noise
(<span id="cite-allen"></span>[Allen & Tildesley, 1987](#ref-allen)).

## Implement periodic RDF and coordination in JAX

The collapsed setup selects the CPU backend, imports the known FCC system, and
loads the real kUPS workflow.

{% include kups-notebooks/post-07/post07-setup.html %}

The open cell below performs every central operation. Broadcasting constructs
all frame/atom/atom displacements, `round` applies the minimum image,
`triu` keeps each pair once, and `histogram` counts radial bins. Unsupported
bins are set to `NaN` instead of being silently plotted.

{% include kups-notebooks/post-07/post07-jax-rdf.html %}

The control is one exact 32-atom FCC frame at the same density as the physical
runs. Its first pair shell appears at 3.72 Å. Integrating through 5.12 Å returns
exactly 12 neighbors, the FCC nearest-neighbor count. The last fully supported
bin center is 5.16 Å because the half-box boundary lies at 5.26 Å.

The coordination calculation uses the same shell volumes as the RDF:

$$
n_c(r_c)
=4\pi\rho\int_0^{r_c}r^2g(r)\,dr
\approx
\rho\sum_{r_{b+1}\le r_c}g_b\Delta V_b.
$$

The cutoff $$r_c$$ is part of the observable. It is usually placed at a
physically motivated minimum after the first RDF peak. Moving it into the next
shell changes the reported coordination even when no trajectory frame changes.

The transparent JAX function assumes a fixed cubic cell. The production
analysis reads the stored volume and uses the same pair and shell convention;
a moving or noncubic cell would require per-frame cell geometry.

## Run the estimators on new kUPS trajectories

The physical workflow runs Lennard-Jones argon at 100 K and fixed volume. Each
replica uses the kUPS `baoab_langevin` path with a 2 fs timestep and an
independent momentum seed. One frame is stored every ten steps, or 20 fs.

The analysis reads rather than invents its microscopic inputs:

- positions for minimum-image pair distances;
- volume for density and radial support;
- momenta and per-atom masses for velocities;
- replica identity for between-run uncertainty.

The next cell launches two fresh 32-atom CPU smoke replicas through
`kups.application.simulations.md.run` and analyzes their HDF5 files.

{% include kups-notebooks/post-07/post07-kups-observables.html %}

The smoke output has 16 frames per replica. It places the first RDF peak at
3.75 Å with height 4.99, estimates coordination as
$$12.60\pm0.08$$ by replica standard error, and obtains a 42.1 fs normalized
VACF integral. These short-run values prove the code path; they are not
production estimates.

The full profile uses three independent 256-atom replicas and 80 frames per
replica. Its requested 8.0 Å RDF range is below the 10.52 Å half-box support.
Every worker observed an NVIDIA RTX A5000:

<div class="table-responsive" markdown="1">

| Replica | Frames | Mean temperature | Runtime | HDF5 SHA-256 prefix | Device |
|---|---:|---:|---:|---|---|
| 0 | 80 | 99.89 K | 53.43 s | `3aa322ef0943` | RTX A5000 |
| 1 | 80 | 98.45 K | 50.00 s | `d065bbd2cda2` | RTX A5000 |
| 2 | 80 | 102.23 K | 50.46 s | `342396071b71` | RTX A5000 |

</div>

Distinct seeds and HDF5 hashes establish separate executions. They do not by
themselves prove that every slow physical mode was sampled independently.

## See one neighbor shell become the RDF

The left panel below selects atom 0 in the final frame of full replica 0 and
places it at the origin. Every displayed displacement uses the periodic
minimum image. Orange atoms lie inside the three-dimensional 5.1 Å coordination
sphere; pale atoms are outside the cutoff but inside the 8 Å RDF range. Marker
size encodes the magnitude of the hidden $$z$$ displacement.

The right panel then replaces this single neighborhood with all unordered
pairs from all frames and replicas. Shell normalization produces the RDF, and
the green cutoff selects the bins used by the coordination integral.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post07_neighbor_shell.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Actual periodic neighbor shell beside the normalized radial distribution function from three kUPS replicas" caption="One actual full-profile atom has 14 neighbors inside 5.1 angstrom in the displayed frame. That instantaneous count is not the reported coordination. The RDF pools all atoms and frames from three replicas, normalizes each spherical shell, and integrates to 13.696 through the same cutoff. The first RDF peak is at 3.64 angstrom with height 4.61; the pale band is the between-replica standard deviation." %}

The distinction between 14 and 13.696 is important. The first is one atom in
one frame. The second averages a defined RDF estimator over all atoms, 80
frames, and three replicas. An atom-level picture builds intuition, but the
reported scalar needs the complete estimator and sampling record.

Across replicas, full coordination ranges from 13.678 to 13.722, giving a
replica standard error of 0.013. It is not a universal argon coordination. It
belongs to this potential, density, temperature, warmup, cutoff, bin width, and
1.6 ps stored window.

## Velocity memory is another trajectory estimator

Positions describe structure. Dynamics require time ordering. kUPS stores
momenta $$\mathbf p_{i,t}$$ and masses $$m_i$$, so velocities are

$$
\mathbf v_{i,t}=\frac{\mathbf p_{i,t}}{m_i}.
$$

For lag $$k$$, the normalized velocity autocorrelation function (VACF) averages
over atoms and all time origins that still have a partner:

$$
C_v(k)
=
\frac{
  \dfrac{1}{N\,(N_f-k)}\sum_{t=0}^{N_f-k-1}\sum_i
  \mathbf v_{i,t}\cdot\mathbf v_{i,t+k}
}{
  \dfrac{1}{N\,N_f}\sum_{t=0}^{N_f-1}\sum_i
  \mathbf v_{i,t}\cdot\mathbf v_{i,t}
}.
$$

The numerator has fewer time origins as $$k$$ grows. Long-lag noise therefore
increases even before physical finite-size effects enter. Replica bands and a
reported maximum lag are part of the estimator, not cosmetic additions.

The full mean VACF is 0.799 at the first 20 fs lag and first crosses zero near
140 fs. Its displayed 0--600 fs integral is
$$7.06\pm0.33$$ fs by replica standard error.

That integral is not a diffusion coefficient. For an isotropic fluid, the
Green--Kubo relation uses the unnormalized velocity correlation:

$$
D
=\frac{1}{3}\int_0^\infty
\left\langle\mathbf v(0)\cdot\mathbf v(t)\right\rangle dt
=\frac{\langle v^2\rangle}{3}\int_0^\infty C_v(t)\,dt.
$$

This tutorial has a coarse 20 fs storage interval, a short tail, and no
hydrodynamic finite-size correction. It therefore reports the normalized
integral only as a dynamical-memory diagnostic
(<span id="cite-kubo"></span>[Kubo, 1957](#ref-kubo)).

## Every curve needs support and uncertainty

An error bar on coordination does not quantify the entire RDF. RDF bins share
atoms and frames, and VACF lags share time origins. The full analysis therefore
records the mean and between-replica standard deviation at every RDF bin and
VACF lag. The maximum replica standard deviation is 0.077 for the RDF and
0.0088 for the normalized VACF.

Replica spread and within-run correlation answer different questions. Blocks
probe finite-time memory inside one run. Independent replicas expose
trajectory-to-trajectory disagreement. A serious observable study often needs
both, especially if metastability makes one long trajectory deceptively smooth
(<span id="cite-frenkel"></span>[Frenkel & Smit, 2001](#ref-frenkel)).

Before accepting a trajectory-derived observable, record:

1. microscopic inputs and their units;
2. pair, normalization, and time-origin conventions;
3. bin width, cutoff, and maximum lag;
4. periodic-cell support and finite-size assumptions;
5. warmup, storage interval, effective samples, and replica spread;
6. sensitivity to analysis parameters and trajectory extension.

The bounded full run demonstrates this measurement contract and a real GPU
kUPS path. It does not establish converged liquid structure, a transport
coefficient, or force-model validity.

## Check your understanding

1. If ordered pairs are counted instead of unordered pairs, which factor in
   the ideal reference count must change?
2. Why does a raw pair histogram grow with radius even in a uniform gas?
3. Can two coordination numbers computed from the same RDF disagree without
   either implementation being numerically wrong?
4. Why are the last VACF lags noisier than the first even before considering
   slow physics?

The answers follow from estimator definitions: ordered pairs double the count,
spherical shells grow as $$r^2$$, coordination depends on its cutoff, and long
lags have fewer time origins.

## A plot becomes a result only after its estimator is explicit

Positions do not become an RDF until periodic pair geometry is normalized
against shell volume and density. An RDF does not become coordination until a
cutoff and quadrature are chosen. Momenta do not become a transport result
until velocities, time origins, tail support, and Green--Kubo normalization are
specified.

A reproducible observable reports that entire path from microscopic arrays to
the final quantity. Post 08 will invert a different measured object—a
probability distribution—to obtain relative free energy.

<details class="kups-reproducibility" markdown="1">
<summary>Reproducibility record and complete observable dashboard</summary>

Run and verify the CPU profile from the locked environment:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 07 --profile smoke
uv run kups-tutorial verify 07 --profile smoke
uv run kups-tutorial verify-notebooks --posts 07 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 07 --check
```

The complete audit dashboard retains the known FCC estimator controls,
finite-size support check, block uncertainty, full VACF, and full RDF with
replica bands:

{% include figure.liquid loading="lazy" path="assets/img/blog/kups_md_post07_observable_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four-panel observable audit dashboard with periodic-cell controls and full kUPS RDF and VACF curves" caption="The upper panels validate normalization, half-box support, and coordination uncertainty on controlled periodic cells. The lower panels use momenta, masses, positions, and volumes from three real 256-atom kUPS HDF5 replicas. These bounded curves validate the estimator path, not converged argon structure or transport." %}

Source and evidence:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/full.json)
- [smoke kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/smoke/kups_observable_summary.json)
- [full kUPS summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_observable_summary.json)
- [full RDF samples](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_rdf_samples.csv)
- [full VACF samples](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/kups_vacf_samples.csv)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/manifest.json)
- [executed notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-07-observables.ipynb)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post07_figures.py)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-07.md)
- [source repository](https://github.com/sungsoo-ahn/kups-md-tutorials)

</details>

## References

- <span id="ref-allen"></span>Allen, M. P. & Tildesley, D. J. (1987). *Computer Simulation of Liquids*. Oxford University Press. <a href="#cite-allen" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kubo"></span>Kubo, R. (1957). Statistical-mechanical theory of irreversible processes. I. *Journal of the Physical Society of Japan*, 12, 570–586. <a href="#cite-kubo" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-frenkel"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel" class="reversefootnote" role="doc-backlink">↩</a>
