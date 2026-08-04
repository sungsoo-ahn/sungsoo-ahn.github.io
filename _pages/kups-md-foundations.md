---
layout: post
permalink: /kups-md-tutorials/foundations/
title: "What Does a Molecular-Dynamics Program Actually Store and Update?"
date: 2026-08-04
last_updated: 2026-08-04
description: "Build the mental model behind MD: atomic state, potential energy, JAX forces, discrete trajectories, periodic cells, ensembles, and the corresponding kUPS objects."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 0
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 0
categories: [science]
tags: [molecular-dynamics, jax, kups, foundations, trajectories]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

Molecular dynamics is a loop over arrays. Positions enter a potential-energy
function, differentiation produces forces, an integrator updates positions and
momenta, and selected states become a trajectory. The apparent complexity of
an MD package comes from making this loop fast, physical, and difficult to
misuse.

This chapter builds that mental model with two atoms. The example is small
enough to inspect line by line, but it uses the same JAX operations and the same
kinds of state that later chapters use for hundreds of atoms.

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- what an MD state stores and what a trajectory stores;
- how a scalar potential energy becomes one force vector per atom;
- why JAX uses pure state updates, `grad`, `jit`, `scan`, and explicit random keys;
- how kUPS packages particle arrays, system arrays, a potential, and a propagator;
- how a trajectory differs from an ensemble and an observable.

**Prerequisites:** Python arrays, derivatives of scalar functions, and Newton's
second law. You do not need prior MD or kUPS experience.
</div>

## The state is more than a set of coordinates

For $$N$$ atoms, the position array has shape $$N\times3$$. Row $$i$$ stores the
Cartesian position $$\mathbf{r}_i=(x_i,y_i,z_i)$$. An MD state usually also
contains momenta $$\mathbf{p}_i$$, masses $$m_i$$, atomic numbers $$Z_i$$, and a
simulation cell $$\mathbf{H}$$:

$$
X = \left(\mathbf{R},\mathbf{P},\mathbf{m},\mathbf{Z},\mathbf{H},\ldots\right).
$$

Here $$\mathbf{R}$$ and $$\mathbf{P}$$ are $$N\times3$$ arrays. The cell
$$\mathbf{H}$$ is a $$3\times3$$ matrix whose rows or columns, depending on the
library convention, describe the box vectors. The ellipsis can contain a step
counter, thermostat variables, cell momentum, neighbor data, or cached
potential outputs.

Positions alone are not a complete dynamical state. Two systems with identical
positions and opposite momenta immediately move in opposite directions.
Atomic numbers and masses also play different roles: the potential generally
uses chemical identity, while Newton's equation uses mass to convert force into
acceleration.

Units are part of this state contract. In the kUPS examples, positions enter
through ASE in ångström, masses use atomic mass units, energies use electron
volts, and user-facing timesteps are specified in femtoseconds. kUPS converts
these values into its internally consistent representation. A bare array does
not carry a unit label, so mixing conventions can produce a numerically smooth
but physically meaningless trajectory.

## Energy becomes force through a gradient

An interatomic potential maps all atomic positions to one scalar:

$$
U:\mathbf{R}\longmapsto U(\mathbf{R}).
$$

The force on atom $$i$$ is the negative derivative of that scalar with respect
to the atom's position:

$$
\mathbf{F}_i=-\frac{\partial U}{\partial\mathbf{r}_i}.
$$

The minus sign points the force downhill on the energy surface. Because the
input has shape $$N\times3$$, the gradient and force arrays have that same
shape.

For the two-atom example, let $$r=\lVert\mathbf{r}_2-\mathbf{r}_1\rVert$$ and
use a harmonic bond,

$$
U(r)=\frac{1}{2}k(r-r_0)^2.
$$

The equilibrium distance is $$r_0=1$$, the spring constant is $$k=2$$, and the
initial distance is $$1.2$$ in dimensionless units. The extension is therefore
$$0.2$$, giving an energy of $$0.04$$ and a force magnitude of $$0.4$$.

{% include kups-notebooks/post-00/post00-setup.html %}

{% include kups-notebooks/post-00/post00-energy-to-force.html %}

The two force vectors are equal and opposite. Their sum is zero because
translating both atoms by the same amount does not change their separation or
the energy. This simple check catches sign, indexing, and broadcasting errors
before a trajectory hides them among many updates.

JAX performs the derivative in `forces_from_energy` with `jax.grad`. This is
automatic differentiation, not a finite-difference approximation. The energy
function remains the primary object; JAX constructs the derivative program
needed by the integrator.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post00_atomic_trajectory.svg" class="img-fluid rounded z-depth-1" alt="Animated two-atom JAX trajectory with a harmonic bond and force arrows" caption="A real JAX velocity-Verlet trajectory moves two atoms through stretched, equilibrium, and compressed configurations. The arrows show \(\mathbf{F}=-\nabla U\); they shrink near equilibrium and reverse after the bond becomes compressed. Motion is shown at physical scale, and the SVG provides a static fallback when reduced motion is requested." %}

## An integrator turns one state into the next

Newton's equations describe continuous time,

$$
\frac{d\mathbf{r}_i}{dt}=\frac{\mathbf{p}_i}{m_i},
\qquad
\frac{d\mathbf{p}_i}{dt}=\mathbf{F}_i.
$$

A computer cannot apply an infinitesimal change. An integrator chooses a finite
timestep $$\Delta t$$ and constructs a discrete map

$$
X_{n+1}=\Phi_{\Delta t}(X_n).
$$

The sequence $$X_0,X_1,\ldots,X_M$$ is a trajectory. It approximates the
continuous solution at specific times. It is not the exact motion with some
frames omitted.

The reference code represents the state with three JAX arrays and repeatedly
applies velocity Verlet. `jax.lax.scan` carries the new state into the next
step and stacks the recorded states into trajectory arrays.

{% include kups-notebooks/post-00/post00-jax-trajectory.html %}

The result has shape `(400, 2, 3)`: 400 stored states, two atoms, and three
coordinates. The maximum relative energy excursion is about
$$4.0\times10^{-4}$$ for this timestep. A small bounded excursion is expected
from velocity Verlet; it does not mean the discrete trajectory is exact. Post
02 opens the step itself and shows how the error changes with timestep and
integration scheme.

An MD run often performs many steps between stored frames. If the block size is
ten, the engine may update the state ten times and save only the tenth state.
The trajectory file then has fewer frames than integration steps. This
distinction becomes essential when interpreting time axes, correlations, and
storage cost.

## JAX changes how the loop is expressed

Ordinary Python code often mutates arrays inside a `for` loop. JAX works best
when a step is a pure function: the old state enters and a new state leaves.
That functional form enables several transformations used throughout the
series:

- `jax.grad` turns a scalar energy program into a force calculation;
- `jax.jit` compiles a state update for the selected device and array shapes;
- `jax.lax.scan` expresses a repeated update without a Python dispatch per step;
- `jax.vmap` applies the same observable or model across frames or replicas;
- explicit PRNG keys make stochastic updates reproducible and composable.

These operations do not change the physics. They change how the computation is
staged, differentiated, compiled, and parallelized. A correct JAX program must
still use correct forces, units, boundaries, and update equations.

## Periodic boundaries make the cell active

Bulk simulations use a finite box as a repeating tile. An atom leaving one side
re-enters through the opposite side, and an atom near the left face can be a
close neighbor of an atom near the right face.

The potential must therefore measure displacement using the selected periodic
cell rather than raw Cartesian subtraction. The integrator must also decide
whether to store wrapped positions inside the primary cell or an unwrapped path
that records boundary crossings. Both representations can describe the same
physical motion, but they support different analyses.

Periodic boundaries do not make a small system identical to an infinite one.
The box length limits meaningful pair distances, changes which fluctuations
fit inside the cell, and can couple an atom to its own periodic images. Later
chapters make these finite-size limits explicit when computing radial
distribution functions and free energies.

## kUPS gives the arrays a production structure

The minimal `ParticleState` is useful because every operation is visible. A
production engine must also represent multiple chemical species, one or more
systems, cell geometry, particle-to-system indices, boundary flows,
thermostat/barostat parameters, potential caches, logging, and recovery from
invalid updates.

kUPS stores these pieces in structured JAX PyTrees. A PyTree is a nested Python
object whose numerical leaves are arrays. JAX transformations can operate on
the leaves while kUPS preserves their physical roles.

The next cell constructs an ASE system and passes it through the public kUPS
state adapter. This is a real kUPS object, not a dictionary made to resemble
one.

{% include kups-notebooks/post-00/post00-kups-state.html %}

The particle table contains the same position, momentum, and mass arrays used
by the minimal example. The system table adds the periodic cell and integrator
parameters. The full state contains fourteen numerical leaves for this tiny
case because kUPS also tracks indices, gradients, cell data, and system-level
quantities.

A kUPS MD calculation can be read as four objects:

1. **State:** the current particle and system data.
2. **Potential:** a function that returns energy and the derivatives required
   by dynamics.
3. **Propagator:** a composable state update such as velocity Verlet, a
   thermostat, or a barostat.
4. **Logger:** a rule for selecting and writing frames and observables.

The later notebooks will keep this mapping visible. They first implement the
central operation with small JAX arrays, then show the corresponding kUPS
potential or propagator, and finally run the library path that writes the
recorded evidence.

## A trajectory is not an ensemble or an observable

A trajectory is an ordered sequence of states. An ensemble is a probability
distribution over states under specified constraints, such as fixed particle
number, volume, and energy. An observable is a quantity estimated from states,
such as temperature, pressure, a radial distribution function, or a velocity
correlation.

One long trajectory may provide samples from an ensemble, but only after the
initial transient has decayed and only to the extent that stored frames contain
independent information. Saving ten times as many highly correlated frames
does not create ten times as much evidence. Posts 04--08 develop these
distinctions using real kUPS trajectories.

## Check your understanding

Before running the code, predict what happens if both atomic masses are
doubled while the potential and initial positions remain fixed.

The energy and force are unchanged because neither depends on mass in this
example. The acceleration is halved because $$\mathbf{a}=\mathbf{F}/m$$, so the
bond oscillates more slowly. Changing the spring constant instead changes both
the force and the oscillation rate.

You can test the prediction by replacing `jnp.ones(2)` with
`2 * jnp.ones(2)` in the trajectory cell. The final separation at a fixed step
count changes, but the opposite-force and zero-net-force checks still hold.

Post 01 now has a narrower job: construct a physically controlled initial state
for many atoms. Post 02 then opens the velocity-Verlet map and connects its JAX
implementation to the kUPS propagator.

<details class="kups-reproducibility">
<summary>Reproducibility record</summary>
<div markdown="1">

The [executable notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-00-md-jax-foundations.ipynb)
uses the locked Python environment and a CPU JAX backend. The reference
algorithms live in
[`jax_reference.py`](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/jax_reference.py).
The animated SVG and PNG poster are generated by
[`generate_post00_figures.py`](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post00_figures.py),
which records the full initial state, timestep, frame count, displacement scale,
and position hash.

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run pytest tests/test_jax_reference.py tests/test_foundation_visuals.py
uv run kups-tutorial verify-notebooks --posts 00 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 00 --check
```

</div>
</details>
