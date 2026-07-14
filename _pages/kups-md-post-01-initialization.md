---
layout: post
permalink: /kups-md-tutorials/post-01-initialization/
title: "How Do You Initialize an MD Simulation Without Biasing the Result?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible first step for molecular dynamics: cell construction, seeded velocities, center-of-mass removal, provenance, and initialization diagnostics."
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
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. The source repository is <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>; corrections and replication issues should be tracked against that repository.</em>
</p>

## Introduction

Molecular dynamics papers often begin after the most consequential choices have
already been made. The system has a cell, atom positions, velocities, a target
temperature, a seed, maybe a minimization stage, maybe a warmup stage, and
usually some center-of-mass cleanup. These details look procedural, but they
define the distribution from which the trajectory starts.

For ML researchers who already know the equation of motion,

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i, \qquad \frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i,$$

the practical question is different: what exactly did we initialize, and can
someone else reproduce it? If the answer is vague, later claims about
thermostats, equilibration, free energies, or MLIP stability are already on
weak ground.

Initialization is not a cosmetic prelude to simulation. It is a statement about
the first probability measure used by the trajectory. Coordinates define which
region of configuration space is being sampled. Velocities define a kinetic
energy distribution and a total momentum. Cell shape and density define the
volume available to the atoms and, in periodic systems, the neighboring images
that every force evaluation will see. Minimization and warmup, when used, decide
which parts of the nominal initial distribution are discarded before production.
The random seed decides whether a colleague can reproduce the exact same
starting point or only the same recipe.

The running example here is deliberately modest: FCC argon at a fixed density
and temperature. Argon is not the scientific destination of the series. It is a
cheap controlled system for making the bookkeeping impossible to skip. The same
questions become more expensive, not less important, when the potential is a
machine-learned interatomic potential and the system is aluminum, an electrolyte,
or a defected solid.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-01/full.json)
- [initialization notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-01-initialization.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/smoke/initialization_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/initialization_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-01/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-01.md)

## What Is the Initialization Contract?

An initial state is not only coordinates. It is a contract between the
configuration, the code, the random number generator, and the later analysis.
For this tutorial, the committed full configuration fixes:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | FCC argon | controlled, cheap CPU example |
| atom count | 500 | larger profile for smoother diagnostics |
| number density | 0.0213 atoms/angstrom^3 | determines the cell volume |
| target temperature | 94.4 K | sets the velocity scale |
| seed | 2026071401 | makes the velocity draw reproducible |
| exact temperature rescaling | false | preserves the finite-system velocity draw |
| center-of-mass removal | true | removes bulk translation from the initialized momenta |

The smoke case uses the same density, temperature, seed, and velocity policy
with a smaller 32-atom cell. It is not meant to be a statistically convincing
argon simulation. It is a fast test of the contract: can the code construct the
system, write the compact summary, record provenance, and verify the result
without requiring a long run?

This is the level of detail that should be present before a trajectory is
interpreted. A paper, notebook, or internal report should not merely say "argon
at 94.4 K." It should say which density, which seed, which velocity
distribution, whether exact temperature rescaling was used, whether
center-of-mass motion was removed, and which code revision produced the state.
Otherwise the first row of the production table is already under-specified.

The full committed summary records 500 atoms, number density
0.021300000000000034 atoms/angstrom^3, volume 23474.178403755832
angstrom^3, target temperature 94.4 K, instantaneous kinetic temperature
90.89870854463463 K, kinetic energy 5.8747814906665905 eV, seed 2026071401,
and center-of-mass speed about \(9.35 \times 10^{-19}\) in the configured units.
The exact number of digits is not aesthetically important. What matters is that
the numerical record is machine-checkable, not reconstructed from prose.

## Why Start From a Simple Crystal?

The first initialization choice is the coordinate distribution. In a molecular
liquid, one might read a previously equilibrated structure, pack molecules with
a placement algorithm, or start from a crystal and melt it. In a solid, one may
start from a relaxed primitive cell, a supercell with defects, or a strained
cell. In a biomolecular simulation, the initial coordinates are often a
processed experimental or predicted structure with added hydrogens, solvent,
ions, and missing atoms. These choices are scientific, not clerical.

Here the coordinate distribution is intentionally narrow. The system builder
constructs an FCC argon supercell at a specified number density. That makes the
cell volume reproducible and avoids hiding initialization issues inside a
complicated packing or relaxation procedure. The full profile uses a 5 by 5 by
5 conventional FCC replication, giving 500 atoms. The smoke profile uses a 2 by
2 by 2 replication, giving 32 atoms. Both profiles use the same physical density.

The benefit of a simple crystal is control. If a later diagnostic fails, the
failure is less likely to come from a hidden overlap, an ambiguous input
structure, or an undocumented relaxation. The cost is that the system is not an
equilibrated liquid or a realistic production target. That cost is acceptable
for the first tutorial because the point is to define what must be recorded.
Later posts will deliberately disturb this clean starting point with
integrators, thermostats, barostats, longer trajectories, observables, and
free-energy estimators.

This separation is especially useful for MLIP work. A learned potential may
behave well near a training-like crystal and fail under strain, high
temperature, defects, or extrapolative coordination environments. If the
initial state is not pinned, it becomes hard to distinguish a model failure from
a setup difference. A reproducible coordinate builder is a baseline for asking
whether the potential, the integration, or the sampling protocol changed the
result.

## What Does the Density Fix?

Periodic molecular dynamics does not only need positions. It needs a cell. In a
periodic calculation, the cell defines the replicated universe. It determines
which atom images can become neighbors, how distances are wrapped, and how
volume-dependent quantities such as pressure are later interpreted.

For a fixed atom count \(N\) and number density \(\rho\), the volume is

$$V = \frac{N}{\rho}.$$

The full profile has \(N=500\) and \(\rho=0.0213\) atoms/angstrom^3, so the
recorded volume is about 23474.1784 angstrom^3. If one only reports the
temperature and chemical formula, this volume is lost. Two simulations with the
same temperature and atom count but different densities are not replicas of the
same ensemble. They sample different environments and will produce different
pressure, coordination, collision frequency, and transport behavior.

Density is also an early place where unit mistakes appear. A density expressed
in atoms/angstrom^3, grams/cm^3, mol/L, or reduced Lennard-Jones units can lead
to the same-looking prose and very different cells. The tutorial keeps the
configuration explicit and writes the compact summary so the generated state
can be checked after the fact. A serious MD workflow should make this kind of
mistake boring to find.

The figure below uses the full profile because 500 atoms make the velocity
diagnostic smoother than the smoke profile while remaining cheap. It checks the
cell construction, visualizes the seeded velocity draw, and prints provenance
fields that should not change silently.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post01_initialization_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Initialization diagnostics for the committed full profile. The figure checks the FCC cell density, shows the seeded velocity draw as standardized components, and records provenance fields that should remain reproducible." %}

## Velocities Are Samples

After coordinates and the cell, the next question is momentum. The common
canonical choice is to draw velocities from the Maxwell-Boltzmann distribution
at a target temperature. For atom \(i\), each Cartesian velocity component has a
variance set by \(k_B T / m_i\). The draw is random, even though the random
number generator is seeded.

This distinction is easy to blur. A target temperature is a parameter of the
distribution. The instantaneous kinetic temperature of one finite velocity
draw is a random variable. The full profile targets 94.4 K and obtains an
instantaneous kinetic temperature of about 90.90 K. That is not a failed
initialization. It is the finite-system draw produced by the documented seed
and velocity policy.

The configuration therefore sets `force_exact_temperature` to `false`. That
choice preserves the sampled Maxwell-Boltzmann draw. Some workflows instead
rescale velocities so the instantaneous kinetic temperature exactly matches
the target. That can be reasonable when one wants a deterministic initial
kinetic energy or a tightly controlled comparison. But it is a constraint
applied after the draw, and it should be reported as such. It changes the
distribution of the initial kinetic energy.

For ML researchers, the analogy is straightforward. Sampling a minibatch from a
data distribution and then forcing one statistic of that minibatch to equal the
population statistic are different operations. Both can be useful. They should
not be confused in the experimental record.

The seed is part of this record. If a thermostat later appears to produce the
wrong kinetic-energy distribution, or if two replicas disagree, the first
question should be whether the difference comes from the intended stochastic
draw or from a hidden change in setup. A seed does not make the physics more
valid. It makes the debugging path shorter.

## Why Remove Center-of-Mass Motion?

Velocity draws can give the whole finite system a small net momentum. For a
bulk periodic simulation, this center-of-mass translation is usually not the
motion of interest. It can waste kinetic energy in a trivial drift mode and
complicate comparisons across replicas.

The tutorial removes center-of-mass momentum after the velocity draw. The
recorded center-of-mass speed is effectively zero, about
\(9.35 \times 10^{-19}\). This is a bookkeeping choice, but it is still a
choice. If it is applied after exact temperature rescaling, before exact
temperature rescaling, or not at all, the final kinetic state can differ.

This matters because the later trajectory is interpreted through conserved or
controlled quantities. An NVE diagnostic checks energy behavior. A thermostat
diagnostic checks kinetic-energy statistics. A diffusion calculation interprets
long-time displacement. A free-energy calculation assumes the sampled ensemble
is the one claimed. Small setup choices can leak into these diagnostics, and
center-of-mass cleanup is one of the easiest to document.

The figure review for this page found a useful failure mode. An earlier version
of the diagnostic tried to report residual total momentum from the saved
structure file. The file round-tripped momenta with limited precision, so the
panel was more about text serialization than physics. The reviewed figure now
uses the JSON summary for the checklist instead. That is the kind of
self-review this series is meant to enforce: the figure should support the
claim, not merely look plausible.

## Where Do Minimization and Warmup Enter?

This page initializes a clean FCC argon configuration and velocity draw. It
does not pretend that every production simulation can start integrating from
the same kind of state. Real workflows often include energy minimization,
short restrained dynamics, solvent relaxation, cell relaxation, heating, or a
warmup trajectory before data collection.

Those stages should be treated as transformations of the initial state, not as
unreported preliminaries. Minimization changes coordinates by following an
optimization objective, not by sampling the target ensemble. Heating changes
the kinetic distribution over a schedule. Warmup discards early trajectory
segments because the starting point is not trusted to be representative.
Equilibration diagnostics in later posts will ask when discarding is justified,
but the first requirement is simpler: say what was done.

For example, a reproducible report might distinguish:

| Stage | Example record | Why it belongs in provenance |
|---|---|---|
| coordinate construction | FCC argon, density 0.0213 atoms/angstrom^3 | fixes the starting cell and positions |
| velocity initialization | Maxwell-Boltzmann at 94.4 K, seed 2026071401 | fixes the initial momenta distribution |
| center-of-mass cleanup | removed after velocity draw | removes trivial bulk translation |
| minimization | not used in this profile | avoids implying an unrecorded coordinate transform |
| warmup | not used in this profile | keeps this page focused on the initial state |

The absence of minimization and warmup is not a universal recommendation. It is
a scoped decision for this first page. If a later tutorial uses a warmup
trajectory, the warmup length, ensemble, thermostat parameters, discarded
frames, and resulting state must be part of the artifact trail.

## What Should Be in the Provenance?

A reproducible initialization needs more than a JSON configuration. It needs the
configuration, the code version, the dependency environment, and compact
outputs that can be audited without storing huge trajectories. The full
manifest records the config path, config SHA-256, lockfile SHA-256, git
revision, Python version, platform, precision policy, runtime device, ASE
version, kUPS version, and NumPy version.

The manifest is intentionally mundane. It is not a substitute for scientific
judgment, but it prevents avoidable ambiguity. If a result changes after a
dependency update, the lock hash says so. If a CPU smoke result is compared to a
GPU full result, the runtime device says so. If a colleague reruns the notebook
from a different commit, the revision says so.

The compact summary is equally important. A future reader should not have to
load a trajectory just to learn the atom count, density, seed, instantaneous
temperature, or center-of-mass speed. The summary is small enough to commit and
specific enough to test. That is the pattern used throughout the series:
large artifacts may be regenerated, but compact numerical claims should be
tracked.

This is also why the notebook is not the source of reusable logic. The notebook
imports the configuration loader, initializer, provenance helper, and figure
generator from `src/kups_md_tutorials/`. The notebook is a readable execution
surface. The package code is the testable implementation. Mixing those roles is
fine for exploration and fragile for publication.

## What Can Go Wrong If This Is Underspecified?

Many MD disagreements begin as invisible initialization differences. One group
uses a different density. Another removes center-of-mass motion every step
instead of only at initialization. One workflow rescales velocities exactly,
another preserves the velocity draw. A notebook silently changes the random
seed when a cell is re-executed. A structure is minimized before production, but
the minimization criterion is missing from the methods text. None of these
differences necessarily indicates bad faith or bad science. They are ordinary
ways for an under-specified procedure to diverge.

The consequences show up downstream. In an integrator study, a different
initial kinetic energy can change early energy traces. In a thermostat study,
exact rescaling can make the initial kinetic distribution look artificially
tight. In a barostat study, a density mismatch can dominate pressure
relaxation. In free-energy work, a hidden warmup or poor initial overlap can
affect apparent convergence. In MLIP deployment, a small coordinate or cell
change can move the system closer to or farther from the training distribution.

The point is not that initialization determines everything. Long, well-mixed
trajectories can lose memory of many initial details. The point is that one
cannot know which details have become irrelevant unless they were recorded well
enough to test. Reproducibility is not only about rerunning the final command.
It is about preserving the chain of choices that made the command meaningful.

## What Should a Methods Section Say?

A useful methods section does not need to dump every internal field from a
configuration file. It should give enough information for a reader to decide
whether two trajectories are meant to be comparable. For initialization, that
usually means separating the coordinate state, the velocity state, and any
state-changing preprocessing.

For the coordinate state, report the source of the structure and the cell. In
this page that means an FCC argon builder, 500 atoms in the full profile,
number density 0.0213 atoms/angstrom^3, and the resulting periodic cell volume.
For a more realistic system it might mean a database identifier, a relaxation
procedure, a solvent box construction method, ion placement rules, or a defect
construction script. If the coordinates were read from a file, the file should
have a stable path, hash, or repository revision.

For the velocity state, report the distribution and the constraints. Here the
velocities are sampled from a Maxwell-Boltzmann distribution at 94.4 K with
seed 2026071401. The instantaneous temperature is not forced to equal 94.4 K.
Center-of-mass momentum is removed after the draw. Those three statements are
more informative than simply writing "initialized at 94.4 K" because they tell
the reader which quantities are stochastic and which are constrained.

For preprocessing, report whether minimization, heating, warmup, or
equilibration was performed before measurements began. This profile uses none
of those stages. That does not make it a universal template; it makes the first
state easy to audit. If a production workflow uses 50 ps of NVT warmup before
collecting NVE data, then the warmup ensemble, timestep, thermostat parameters,
discarded interval, and final state should be recorded. Otherwise the
"initial" state seen by the production estimator is not the state described in
the methods text.

One practical rule is to ask whether a colleague could create a second replica
without guessing. A second replica should usually change the velocity seed, and
possibly the coordinate seed if the coordinates are sampled. It should not
accidentally change the density, exact-temperature policy, minimization
criterion, or warmup length. Those are protocol changes unless they are
deliberately part of the experimental design.

Another rule is to decide what must be identical across controlled comparisons.
When the goal is to compare integrators, the initial coordinates and velocities
should often be reused so early differences can be attributed to the integrator.
When the goal is to estimate uncertainty of an observable, independent seeds
are useful because the variation across replicas is part of the estimate. When
the goal is to test MLIP robustness, one may intentionally create strained or
high-temperature initial states, but those states should be labeled as
different regimes, not silent variants of the same setup.

This is why the tutorial series keeps smoke and full profiles. The smoke
profile is a fast contract test: it should reveal broken paths, missing
provenance, and obvious unit mistakes. The full profile is the stronger
diagnostic target for figures and prose. The two profiles are not meant to
produce the same finite-sample numbers. They are meant to implement the same
initialization policy at different sizes.

## How Does This Connect to Later Tutorials?

The later posts will look more sophisticated, but they inherit this contract.
The integrator post will ask what velocity Verlet actually approximates. That
question is cleaner when the initial state is fixed and the energy trace is not
confounded by a different velocity draw. The timestep and precision post will
separate bounded energy oscillation from drift. That separation needs a
known starting cell, known initial kinetic energy, and known force model.

The thermostat post will ask whether a dynamics scheme samples the intended
kinetic-energy distribution. That question is muddled if the initial
distribution was exactly rescaled but described as a plain Maxwell-Boltzmann
sample. The barostat post will ask how pressure and cell degrees of freedom are
controlled. That question depends directly on the starting density and cell.
The trajectory-length post will ask when warmup can be discarded and when a
running average has stabilized. That question requires knowing where the run
started.

The free-energy and enhanced-sampling posts depend on initialization in a more
subtle way. A biased simulation, umbrella window, or nonequilibrium pulling
trajectory can look reproducible while still starting from a poorly documented
state. If the first configuration in each window was generated by an
unreported relaxation or copied from a previous run with unknown history, then
overlap and hysteresis diagnostics become harder to interpret. The initialization
record is part of the free-energy estimator's evidence chain.

The MLIP capstone makes the same point with higher stakes. A learned potential
can have acceptable static validation metrics and still behave badly when a
trajectory enters an extrapolative region. To diagnose that failure, the series
needs to know whether the state was initialized near the training distribution,
heated into a new regime, strained by the cell, or perturbed by the integrator.
The humble argon initialization in this page is the template for that later
audit.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 01 --profile smoke
uv run kups-tutorial verify 01 --profile smoke
uv run kups-tutorial run 01 --profile full
uv run kups-tutorial verify 01 --profile full
uv run jupyter execute notebooks/post-01-initialization.ipynb --inplace
```

The corresponding review path is:

```bash
uv run python scripts/generate_post01_figures.py
uv run kups-tutorial verify-reviews
python3 scripts/validate_kups_pages.py  # from the website repository
```

The rendered desktop and mobile page snapshots are captured through the website
workflow while local browser dependencies remain unavailable on the current
machine. The review note records the workflow run, artifact name, manifest
coverage, and representative snapshot feedback.

## Current Status

This page is not the final article. It is a substantially expanded
hidden draft that remains outside site navigation while the rest of the series
is brought to the same standard. The implemented pieces are:

- CPU smoke initialization workflow
- committed compact smoke and full outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, and figure feedback
- rendered desktop and mobile page snapshots for the hidden draft
- expanded prose connecting initialization to density, velocity sampling,
  center-of-mass cleanup, minimization, warmup, provenance, and later ensemble
  diagnostics

The remaining non-publication pieces are:

- final all-post consistency pass once the other eleven articles are expanded
- final rendered desktop and mobile page snapshots after that consistency pass
- complete inspection of every captured page snapshot, not only representative
  samples
- public indexing decision after the series is ready as a unit

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
