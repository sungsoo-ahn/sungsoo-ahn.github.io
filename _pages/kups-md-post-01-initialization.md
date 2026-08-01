---
layout: post
permalink: /kups-md-tutorials/post-01-initialization/
title: "How Do You Initialize an MD Simulation Without Biasing the Result?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Construct a seeded kUPS MD state, then check the cell, momentum distribution, center-of-mass motion, and provenance."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 1
categories: [science]
tags: [molecular-dynamics, initialization, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft remains hidden while the full series is being
validated. Corrections and replication issues belong in
<a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

Initialization is part of an MD experiment, not boilerplate before it. Two
runs called “argon at 94.4 K” can start from different probability
distributions because they use different cells, velocity draws, constraints,
or warmup histories. A finite trajectory may remember those choices long
enough to change the result.

This tutorial builds a periodic FCC argon cell and passes it to kUPS 1.0.3's
public `md_state_from_ase` adapter. The evidence records the seed, returned
array shapes and hashes, observed JAX device, kinetic temperature, and residual
center-of-mass speed. A separate ASE calculation remains as a transparent
control for the statistics of a finite Maxwell-Boltzmann draw.

## Specify the state before you sample it

An MD state contains more than positions. At minimum, it fixes

- atomic identities and coordinates;
- the simulation cell and boundary conditions;
- masses and momenta;
- the choices used to construct or transform those quantities.

For this example, the number density $$\rho$$ determines the volume:

$$V = \frac{N}{\rho}.$$

The smoke profile contains 32 atoms; the full profile contains 500. Both use
$$\rho=0.0213\ \mathrm{atoms\,\mathring{A}^{-3}}$$. Changing that value changes
the periodic distances, pressure, coordination environment, and the region in
which a machine-learned potential is evaluated. Density is therefore a model
input, not merely a plotting label.

The notebook setup is collapsed because it only finds the repository and
imports reusable interfaces. Its source comes from a fresh notebook execution,
not from a second handwritten copy in this article.

{% include kups-notebooks/post-01/post01-setup.html %}

## A target temperature is not an exact finite temperature

At temperature $$T$$, a Cartesian momentum component is commonly drawn from

$$p(p_{i,\alpha}) \propto
\exp\left[-\frac{p_{i,\alpha}^{2}}{2m_i k_B T}\right].$$

The target sets the distribution's width. It does not force every finite draw
to have exactly the target kinetic energy. For a state with $$f$$ unconstrained
velocity degrees of freedom,

$$T_{\mathrm{inst}} = \frac{2K}{f k_B}.$$

Removing the three center-of-mass components gives $$f=3N-3$$ here. The
following ASE control draws momenta at 94.4 K with a fixed seed, removes the
center-of-mass motion, and reports the realized temperature.

{% include kups-notebooks/post-01/post01-finite-draw.html %}

The 32-atom control lands at 103.23 K. That is not evidence of failed
initialization; it is the expected variability of a small random sample.
Exact-temperature rescaling would make the printed value prettier by imposing
an additional constraint on total kinetic energy. If that constraint is used,
it belongs in the methods description.

## Construct the state through kUPS

The decisive test is whether the real library interface produces a complete,
finite state. The next cell writes the configured structure, starts an isolated
CPU worker, and calls
`kups.application.md.data.md_state_from_ase` with a JAX random key. It then
prints diagnostics derived from the returned kUPS arrays.

{% include kups-notebooks/post-01/post01-kups-state.html %}

The kUPS and ASE temperatures differ because they use different random-number
generators; the seed is reproducible within each implementation, not a promise
that two libraries emit identical Gaussian samples. In the committed full
profile, the kUPS draw gives 95.37 K for the 94.4 K target over 500 atoms. Its
position and momentum arrays both have shape `(500, 3)`, and the residual
center-of-mass speed is $$3.32\times10^{-10}$$ in the recorded internal velocity
units. The verifier uses a tolerance appropriate to the returned float32 state.

The array hashes make the claim sharper. Repeating the pinned code and config
must reproduce the same state; silently changing a seed, structure, library
path, or initialization rule changes a digest. A version string alone would
not prove that kUPS actually constructed the state.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post01_initialization_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The FCC projection and velocity histogram expose the cell and one finite ASE control draw; the third panel records evidence from the real kUPS state adapter. The 500-atom control lies close to the 94.4 K target, while the kUPS panel binds the returned arrays to their device, shapes, and momentum hash." %}

## Remove only the motion you intend to remove

A random draw generally contains a small net momentum. For a periodic bulk
system, translating the entire box is usually not the internal motion of
interest, so kUPS removes center-of-mass momentum. This changes the effective
degrees of freedom and must be reported.

Other transformations need the same treatment. Energy minimization finds a
nearby low-energy structure; it does not sample a thermal ensemble. Warmup
evolves the state and discards an initial interval. Restraints, heating ramps,
and pressure equilibration likewise change the state that enters production.
The useful question is not “did initialization run?” but “which distribution
produced the first measured frame?” (<span id="cite-frenkel2001"></span>[Frenkel
& Smit, 2001](#ref-frenkel2001); <span id="cite-tuckerman2010"></span>[Tuckerman,
2010](#ref-tuckerman2010)).

For controlled algorithm comparisons, reuse the same initial state so that a
different velocity draw does not obscure the method change. For uncertainty
estimation, change the seed and run independent replicas. These are different
experimental designs, and a single seed cannot serve both purposes.

## Record a reproducible initialization contract

A compact report should answer five questions:

| Question | This tutorial records |
|---|---|
| Where did coordinates and the cell come from? | FCC construction, atom count, density, periodic cell |
| How were momenta sampled? | Maxwell-Boltzmann draw at 94.4 K |
| Which randomness was used? | fixed profile seed and momentum-array hash |
| Which constraints were applied? | center-of-mass momentum removal; no exact-temperature rescaling |
| Which implementation built the MD state? | kUPS 1.0.3 `md_state_from_ase` on an observed CPU device |

This contract does not prove equilibration. It proves something earlier and
equally necessary: the simulation begins from the state-generation procedure
that the analysis claims it used.

## Reproduce the result

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 01 --profile smoke
uv run kups-tutorial verify 01 --profile smoke
uv run kups-tutorial verify-notebooks --posts 01 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 01 --check
```

The compact [smoke](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/smoke/kups_initialization_summary.json)
and [full](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/kups_initialization_summary.json)
kUPS summaries contain the evidence quoted above. The
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/full.json),
[full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/manifest.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-01-initialization.ipynb),
[figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post01_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-01.md)
complete the chain.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel2001" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press. <a href="#cite-tuckerman2010" class="reversefootnote" role="doc-backlink">↩</a>
