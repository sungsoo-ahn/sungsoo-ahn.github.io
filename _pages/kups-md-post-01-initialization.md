---
layout: post
permalink: /kups-md-tutorials/post-01-initialization/
title: "What Distribution Does an MD Simulation Start From?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Build an atomic state, derive a Maxwell–Boltzmann momentum draw in JAX, remove box translation, and inspect the state returned by kUPS."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 1
categories: [science]
tags: [molecular-dynamics, initialization, jax, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

Initialization chooses the first probability distribution in an MD
experiment. It is not housekeeping before the "real" simulation. A cell that
is too dense changes every periodic distance. A velocity draw with net drift
moves the whole box. A minimization or warmup changes which configurations can
appear in the measured trajectory.

This lesson constructs an argon crystal, implements its thermal momentum draw
in JAX, removes rigid translation, and then performs the same state
construction through kUPS. The important question is not whether an
initialization function returns without error. It is: **what distribution
produced the first measured frame?**

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- which parts of an atomic state must be fixed before momenta are sampled;
- why a target temperature sets a distribution rather than an exact finite value;
- how the Maxwell–Boltzmann draw and center-of-mass correction become JAX code;
- what `md_state_from_ase` adds when it constructs the real kUPS state;
- why a valid initial state is not yet an equilibrated sample.

**Prerequisites:** atomic positions, momenta, masses, cells, JAX arrays, and
explicit random keys from the [foundations lesson]({{ '/kups-md-tutorials/foundations/' | relative_url }}).
</div>

## First choose atoms, coordinates, and a periodic cell

An MD state needs atomic numbers $$\mathbf{z}$$, positions $$\mathbf{R}$$,
momenta $$\mathbf{P}$$, masses $$\mathbf{m}$$, and a cell $$\mathbf{H}$$. We
must choose the first four deterministic ingredients before sampling
$$\mathbf{P}$$:

1. the chemical identities and masses;
2. the arrangement of atoms;
3. the cell vectors;
4. which directions are periodic.

The example is face-centered-cubic argon. Its number density $$\rho$$ fixes
the cell volume:

$$
V = \det(\mathbf{H}) = \frac{N}{\rho}.
$$

Here $$N$$ is the atom count, $$V$$ is measured in cubic ångström, and
$$\rho=0.0213\ \mathrm{atoms\,\mathring{A}^{-3}}$$. The smoke calculation
uses 32 atoms; the full state uses 500. Density is a physical input: changing
it changes periodic distances, pressure, coordination, and the environments
seen by a learned potential.

## Temperature sets a momentum variance

For a classical atom $$i$$ of mass $$m_i$$, one Cartesian momentum component
at temperature $$T$$ follows

$$
p_{i,\alpha} \sim
\mathcal{N}\!\left(0,\,m_i k_B T\right),
$$

where $$\alpha\in\{x,y,z\}$$ and $$k_B$$ is Boltzmann's constant. Heavy atoms
therefore have wider momentum distributions. Their **velocity** distribution
is narrower because $$v_{i,\alpha}=p_{i,\alpha}/m_i$$.

For one sampled state, kinetic energy is

$$
K(\mathbf{P}) = \sum_{i=1}^{N}\sum_{\alpha=1}^{3}
\frac{p_{i,\alpha}^{2}}{2m_i}.
$$

If $$f$$ independent velocity degrees of freedom remain, the instantaneous
kinetic temperature is defined by

$$
T_{\mathrm{inst}} = \frac{2K}{f k_B}.
$$

This is an estimator computed from one random draw. It need not equal the
target $$T$$. Forcing exact agreement by rescaling all momenta imposes another
constraint on $$K$$; it creates a different finite-state distribution and
must be reported.

## The draw is three JAX operations

The setup below only selects the repository and imports JAX and the kUPS
workflow. It begins collapsed because none of those lines defines the
algorithm.

{% include kups-notebooks/post-01/post01-setup.html %}

The open cell implements the complete reference operation. First it draws
standard normal values. Then it multiplies each atom's row by
$$\sqrt{m_i k_B T}$$. Finally it subtracts rigid translation and measures the
remaining kinetic temperature.

{% include kups-notebooks/post-01/post01-jax-draw.html %}

`jax.random.PRNGKey(7)` makes randomness an explicit input. Reusing that key
repeats the draw; splitting it creates independent keys. This functional
randomness matters under `jit`, `vmap`, and parallel replica generation,
because there is no hidden global generator for a compiled function to
advance.

The 32-atom result misses 94.4 K by a visible amount. That is the expected
finite-sample fluctuation of $$3N-3=93$$ quadratic degrees of freedom, not an
initialization failure. More atoms narrow this fluctuation, but they do not
turn a random variable into a deterministic number.

## Remove box translation, not thermal motion

A raw draw usually gives the box a nonzero center-of-mass velocity:

$$
\mathbf{V}_{\mathrm{COM}}
=\frac{\sum_i \mathbf{p}_i}{\sum_i m_i}.
$$

Subtracting the corresponding momentum from every atom gives

$$
\mathbf{p}'_i = \mathbf{p}_i-m_i\mathbf{V}_{\mathrm{COM}}.
$$

The corrected total momentum is exactly zero in exact arithmetic:

$$
\sum_i \mathbf{p}'_i
=\sum_i \mathbf{p}_i
-\mathbf{V}_{\mathrm{COM}}\sum_i m_i
=\mathbf{0}.
$$

This removes three degrees of freedom, so the temperature denominator becomes
$$f=3N-3$$. It does **not** set individual velocities to zero. The atoms retain
relative thermal motion, which is what drives internal dynamics.

Removing rotation is a different operation and is usually inappropriate for a
periodic bulk crystal. Likewise, preserving temperature after drift removal
rescales the remaining momenta. Each transformation changes the distribution
and should be an explicit choice.

## kUPS constructs the production state

The reference JAX code makes the mathematics visible. The actual tutorial run
uses the public kUPS adapter. It reads the ASE structure, builds kUPS particle
and cell PyTrees, draws momenta from a JAX key, and removes center-of-mass
motion.

{% include kups-notebooks/post-01/post01-kups-state.html %}

The cell executes `kups.application.md.data.md_state_from_ase`; it does not
load a committed summary and call that execution. The output reports the
observed JAX device, array shapes, realized temperature, residual box speed,
and a digest of the returned momenta.

The full 500-atom run returns position and momentum arrays of shape `(500, 3)`.
Its realized kinetic temperature is 95.37 K for a 94.4 K target, and its
center-of-mass speed is $$3.32\times10^{-10}$$ in the recorded internal
velocity units. These values establish a finite, stationary state. They do not
establish equilibrium.

## Look at the state before propagating it

The next figure reads the positions, momenta, masses, and cell serialized from
the full kUPS return value. The left panel selects one layer so velocity arrows
remain legible. Their lengths are scaled for display and do not represent an
elapsed trajectory. The right panel uses all 1,500 Cartesian momentum
components.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post01_initialization_state.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Top-down argon layer with initial velocity arrows beside a standardized momentum histogram" caption="The blue points and orange velocity arrows come from the 500-atom state returned by kUPS; the arrows are scaled vectors, not propagated displacements. Removing center-of-mass motion leaves local thermal motion while reducing the aggregate box speed to numerical zero. Across all atoms, standardized momentum components follow the target Gaussian with ordinary finite-sample variation." %}

The absence of a preferred arrow direction is the atom-level meaning of the
small center-of-mass speed. The approximate Gaussian is a distribution-level
check. Neither plot can tell us whether the **positions** are equilibrated for
the chosen potential and temperature.

## Initialization and equilibration answer different questions

A syntactically valid state can still be physically poor. The crystal may be
too dense, contain overlapping atoms, or sit far from the target potential's
typical energy range. Thermal momenta do not repair those problems.

Common follow-up operations change the state in distinct ways:

These distinctions are standard in molecular-simulation methodology
(<span id="cite-frenkel2001"></span>[Frenkel & Smit,
2001](#ref-frenkel2001); <span id="cite-tuckerman2010"></span>[Tuckerman,
2010](#ref-tuckerman2010)).

<div class="table-responsive" markdown="1">

| Operation | What it changes | What it does not prove |
|---|---|---|
| energy minimization | moves positions toward a nearby local minimum | thermal sampling |
| velocity rescaling | fixes total kinetic energy | correct configurational distribution |
| thermostat warmup | evolves momenta and positions under a temperature controller | independence from the starting structure |
| pressure equilibration | evolves the cell and density | unbiased production statistics |

</div>

The production trajectory should begin only after the relevant transient has
been diagnosed and the discarded interval has been declared. For a controlled
integrator comparison, reuse the same initialized state. For uncertainty
estimation, change the key and run independent replicas. Those are different
experimental designs.

## Check your understanding

Suppose every atom receives the same added velocity $$\mathbf{u}$$ before the
center-of-mass correction.

1. Predict the corrected momenta without running code.
2. Does the correction recover the original relative velocities for atoms of
   different masses?
3. Change the notebook from 32 to 3,200 atoms. Predict what happens to the
   spread of $$T_{\mathrm{inst}}$$ across independent keys, then test it with
   `jax.vmap`.

The first two answers follow directly from
$$\mathbf{p}'_i=\mathbf{p}_i-m_i\mathbf{V}_{\mathrm{COM}}$$. The third turns
the phrase "finite-sample fluctuation" into a measurable scaling experiment.

## What to carry into the trajectory

An initialization record must bind coordinates, cell, identities, masses,
momentum rule, random key, constraints, and implementation. In this run, the
state digest changes if any of those inputs or operations changes. That makes
the starting point reproducible.

It is still only a starting point. Post 02 now asks how forces turn this state
into a discrete trajectory, while later chapters decide which transient frames
to discard and which observables to trust.

<details class="kups-reproducibility" markdown="1">
<summary>Reproducibility record and complete diagnostic dashboard</summary>

Run and verify the CPU profile from a locked environment:

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

The complete audit retains the FCC projection, the earlier ASE control, and
the kUPS evidence panel:

{% include figure.liquid loading="lazy" path="assets/img/blog/kups_md_post01_initialization_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Three-panel initialization diagnostic dashboard" caption="The audit dashboard retains the independent ASE finite-draw control, cell projection, and kUPS evidence record used during validation. It is secondary to the returned-state velocity field because ASE and JAX do not promise identical Gaussian samples from the same integer seed." %}

Source and evidence:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/full.json)
- [smoke compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/smoke/kups_initialization_summary.json)
- [full compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/kups_initialization_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/manifest.json)
- [executed notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-01-initialization.ipynb)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post01_figures.py)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-01.md)
- [source repository](https://github.com/sungsoo-ahn/kups-md-tutorials)

</details>

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel2001" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press. <a href="#cite-tuckerman2010" class="reversefootnote" role="doc-backlink">↩</a>
