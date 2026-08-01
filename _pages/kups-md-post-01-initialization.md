---
layout: post
permalink: /kups-md-tutorials/post-01-initialization/
title: "How Do You Initialize an MD Simulation Without Biasing the Result?"
date: 2026-07-14
last_updated: 2026-07-29
description: "Why coordinates, density, velocity sampling, and center-of-mass removal define the distribution an MD trajectory starts from."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for ML researchers who are new to simulation."
series_order: 1
categories: [science]
tags: [molecular-dynamics, initialization, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Part 1 of the kUPS Molecular Dynamics Tutorials. The executable example is maintained in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

Two simulations can both be described as "500 argon atoms at 94.4 K" and
still begin from different distributions. One may use a denser cell, rescale
every velocity draw to exactly 94.4 K, or remove the motion of the center of
mass. Another may do none of these things. The descriptions sound equivalent,
but the trajectories do not start from the same state.

Initialization does not permanently determine a long, well-equilibrated
trajectory. A good sampler can lose memory of its starting point. In practice,
however, trajectories are finite and warmup is imperfect. Initialization can
change early dynamics, equilibration time, apparent drift, and comparisons
between methods. The first step in a reliable simulation is therefore to say
which distribution produced the initial coordinates and velocities.

The example in this tutorial is deliberately simple: a periodic FCC argon
crystal with velocities drawn at a target temperature. Argon lets us isolate
initialization from chemical complexity. The same choices become harder to
debug when the force provider is a machine-learned interatomic potential
(MLIP), because a small change in density or local geometry can also change how
far the system lies from the model's training data.

Consider a comparison between two integrators. If each run begins from a
different velocity draw, their early energy traces differ before the
integrators have had a chance to reveal their numerical behavior. Reusing the
same initial state removes that source of variation. The opposite choice is
useful when estimating uncertainty: independent velocity seeds show how much an
observable changes across replicas. Initialization is therefore part of the
experimental design, not merely the command that runs before the experiment.

## An Initial State Is a Distribution

Newton's equations evolve positions and velocities,

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i, \qquad
\frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i,$$

but they do not choose the state at time zero. That state has three parts:

1. the atom positions;
2. the simulation cell and boundary conditions;
3. the atom velocities.

Each part can be fixed or sampled. A crystal builder fixes positions from a
lattice rule. A packing program samples positions subject to geometric
constraints. A previous trajectory supplies a state with its own history.
Velocities are usually random samples from a temperature-dependent
distribution. The cell may be copied from an experiment, relaxed under a
pressure target, or constructed from a chosen density.

These choices define the probability measure from which the trajectory starts
(<span id="cite-frenkel2001"></span>[Frenkel & Smit,
2001](#ref-frenkel2001); <span id="cite-tuckerman2010"></span>[Tuckerman,
2010](#ref-tuckerman2010)). Reporting only the chemical formula and target
temperature leaves most of that measure unspecified.

## Build the Cell Before Drawing Velocities

The tutorial constructs a periodic FCC argon supercell. The full example uses
500 atoms at a number density of 0.0213 atoms per cubic angstrom. For a fixed
atom count $$N$$ and number density $$\rho$$, the volume is

$$V = \frac{N}{\rho}.$$

The resulting volume is about 23,474 cubic angstroms. That number is not a
formatting detail. Periodic boundary conditions replicate this cell in every
direction, so the cell determines which atomic images can interact and how
often atoms encounter one another. Changing the density changes the pressure,
coordination structure, collision frequency, and possibly the region explored
by an MLIP.

Starting from a crystal makes this first example easy to inspect. Every atom is
placed by the same lattice construction, and the density fixes the cell. A
liquid or biomolecular system needs more preparation: molecule packing,
solvation, ion placement, removal of atomic overlaps, and often a relaxation
stage. The principle is unchanged. The coordinate source and cell construction
are part of the initial condition, not clerical details that can be omitted.

Unit conventions deserve the same care. Number density, mass density, molar
concentration, and reduced Lennard-Jones density can describe very different
cells with similar-looking numbers. Recording the generated volume provides a
simple check that the intended convention reached the simulation.

## Temperature Sets a Velocity Distribution

At temperature $$T$$, each Cartesian velocity component is commonly drawn from
a Maxwell-Boltzmann distribution,

$$p(v_{i,\alpha}) \propto
\exp\left(-\frac{m_i v_{i,\alpha}^{2}}{2k_{B}T}\right).$$

The target temperature sets the width of this distribution. It does not require
one finite draw to have exactly the target kinetic temperature. In the full
argon example, the target is 94.4 K, while the sampled velocities correspond to
an instantaneous temperature of about 90.9 K. This difference is expected. It
is finite-sample variation, not a failed initialization.

The instantaneous value is computed from the kinetic energy $$K$$,

$$T_{\mathrm{inst}} = \frac{2K}{f k_{B}},$$

where $$f$$ is the number of unconstrained velocity degrees of freedom. Before
constraints, a system of $$N$$ atoms has $$3N$$ Cartesian components. Removing
the three components of center-of-mass motion reduces that count to $$3N-3$$.
For small systems, both the random kinetic energy and the choice of $$f$$ are
visible in the reported temperature. For large systems, the relative
fluctuation becomes smaller, but it never turns a random draw into a fixed
number. This is why a methods description should distinguish the temperature
used to draw velocities from the temperature measured immediately afterward.

Some programs optionally rescale the sampled velocities so that the
instantaneous temperature matches the target exactly. Rescaling can be useful
for a controlled comparison, but it changes the draw by imposing a constraint
on the total kinetic energy. A plain Maxwell-Boltzmann draw and an exactly
rescaled draw should not be described as the same initialization procedure.

The random seed fixes which draw is produced. Reusing the seed helps isolate a
method change because two runs can start from identical velocities. Changing
the seed produces an independent replica and reveals how sensitive a result is
to the initial momenta. The right choice depends on the question: matched seeds
for controlled algorithm comparisons, independent seeds for uncertainty.

A random velocity draw can also give the finite system a small net momentum.
For a periodic bulk system, this center-of-mass translation is usually removed
because it is not the internal motion of interest. The tutorial subtracts the
center-of-mass velocity after sampling, leaving a center-of-mass speed near
zero. The order matters: drawing velocities, removing the center-of-mass motion,
and optionally rescaling temperature are distinct transformations.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post01_initialization_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The FCC construction fixes the atom count and density, while the velocity histogram shows one finite draw from the target distribution. The final checks confirm that the requested seed was used and center-of-mass motion was removed." %}

## Minimization and Warmup Change the Starting Point

Real systems often cannot begin production dynamics immediately. Packed
molecules may overlap. A predicted structure may contain strained bonds. A
solvated protein may need the solvent to relax around it. Energy minimization
and warmup address these problems, but they also change the state that enters
the measured trajectory.

Minimization follows an optimization objective toward a nearby low-energy
structure. It does not sample the target thermal ensemble. Warmup evolves the
system, often while temperature or restraints change, and discards the early
frames. If production begins after either step, then the minimized or warmed-up
state is the effective initial state for the reported measurement.

There is no universal requirement to minimize every system or discard a fixed
amount of warmup. A clean lattice under a compatible potential may not need
minimization. A dense molecular packing probably does. The defensible rule is
to record each state-changing step and justify it from the system rather than
from habit.

## Record the Choices That Change the Distribution

A reader does not need every internal field from the simulation program. The
useful record contains the choices that would make a second trajectory start
from a different distribution.

| Choice | What to record |
|---|---|
| coordinates | source file, structure identifier, or construction rule |
| cell | dimensions or density, plus boundary conditions |
| velocities | distribution, target temperature, and random seed |
| constraints | center-of-mass removal and exact-temperature rescaling |
| preprocessing | minimization, heating, restraints, and discarded warmup |
| replicas | which fields stay fixed and which seeds change |

For this example, the compact description is: 500 FCC argon atoms at a number
density of 0.0213 atoms per cubic angstrom; velocities sampled from the
Maxwell-Boltzmann distribution at 94.4 K with seed 2026071401; no exact
temperature rescaling; center-of-mass momentum removed after the draw; no
minimization or warmup.

That description is short, but it distinguishes the important alternatives. A
colleague can reproduce the same state or create an independent replica without
guessing which choices were scientific and which were incidental.

## Run the Example

The smoke profile uses the same initialization policy with 32 atoms so the
complete path runs quickly on a CPU:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 01 --profile smoke
uv run kups-tutorial verify 01 --profile smoke
```

The repository also contains the [full
configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-01-initialization.ipynb),
and [recorded
results](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/results/post-01/full).

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel2001" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press. <a href="#cite-tuckerman2010" class="reversefootnote" role="doc-backlink">↩</a>
