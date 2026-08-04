---
layout: post
permalink: /kups-md-tutorials/post-11-enhanced-sampling/
title: "Bias the Landscape or Drive the System?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Separate adaptive metadynamics bias from nonequilibrium steering, implement both central updates in JAX, and interpret real kUPS Ar-pair paths."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 11
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 11
categories: [science]
tags: [molecular-dynamics, enhanced-sampling, metadynamics, nonequilibrium-work, jarzynski, jax, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
collapse_code: true
---

A molecular-dynamics trajectory can remain trapped in one free-energy basin
for far longer than we can afford to simulate. Enhanced sampling changes the
calculation so that rare regions become easier to visit. The dangerous leap is
to confuse *visiting a region* with *measuring its equilibrium probability*.

This chapter studies two ways of changing the calculation:

- **adaptive bias** changes the energy landscape according to the trajectory's
  accumulated history;
- **nonequilibrium steering** prescribes a time-dependent protocol and records
  the work performed along each path.

They solve related sampling problems, but they generate different data and
require different estimators. A metadynamics bias history is not a work path.
A steered path is not a biased equilibrium histogram. We will implement the
central update for each method in JAX, then use real kUPS trajectories to see
what a moving restraint actually does to two atoms.

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- how a collective variable compresses atomic configurations into a coordinate;
- why well-tempered Gaussian hills become shorter as local bias accumulates;
- why adaptive bias and nonequilibrium steering are distinct sampling procedures;
- how a discrete restraint-center change performs work at a fixed coordinate;
- how kUPS makes the restraint center depend on the MD step;
- why slower pulling can reduce lag and hysteresis without becoming equilibrium;
- how Jarzynski's equality turns a path ensemble into a tail-sensitive estimate;
- why effective sample size and forward--reverse agreement do not prove convergence.

**Prerequisites:** biased distributions from
[Post 08]({% link _pages/kups-md-post-08-free-energies.md %}), overlap and
exponential weights from
[Post 09]({% link _pages/kups-md-post-09-estimators.md %}), harmonic restraints
from [Post 10]({% link _pages/kups-md-post-10-umbrella-sampling.md %}), and
Langevin dynamics from
[Post 04]({% link _pages/kups-md-post-04-thermostats.md %}).
</div>

The collapsed setup selects JAX CPU for the teaching kernels and imports the
same workflow that launches the real kUPS smoke experiment later.

{% include kups-notebooks/post-11/setup.html %}

## Part I: adaptive bias remembers where the path has been

Let $$\mathbf R$$ collect all atomic positions. A **collective variable**

$$
s=\xi(\mathbf R)
$$

maps that high-dimensional configuration to a smaller coordinate. It could be
an atom-pair distance, a torsion angle, a coordination number, or a learned
descriptor. Choosing $$s$$ is a scientific assumption: motion hidden from
$$s$$ can remain slow even when $$s$$ appears to move freely.

Metadynamics builds a history-dependent bias by depositing Gaussian hills at
previously visited values $$s_i=s(t_i)$$:

$$
V_t(s)
=
\sum_{t_i<t}
w_i
\exp\left[-\frac{(s-s_i)^2}{2\sigma^2}\right].
$$

Here $$V_t$$ is the accumulated bias at time $$t$$, $$w_i$$ is the height of
hill $$i$$, and $$\sigma$$ is its width in collective-variable units. A new
positive hill makes the visited neighborhood less favorable. Repeated hills
therefore encourage the path to leave an already explored basin.

The time subscript matters. A fixed umbrella samples one stationary modified
Hamiltonian. Metadynamics changes its Hamiltonian while sampling. The path at
time $$t$$ depends on every earlier deposition event.

### Well tempering slows the deposition

Depositing hills of constant height forever would keep forcing the landscape
away from a steady limit. Well-tempered metadynamics reduces a new hill using
the bias already present at the sampled point:<sup id="cite-metadynamics"><a href="#ref-metadynamics">1</a></sup>

$$
w_i
=
w_0
\exp\left[
-\frac{V_{t_i}(s_i)}{k_{\mathrm B}\Delta T}
\right],
\qquad
\gamma=\frac{T+\Delta T}{T}.
$$

The initial height is $$w_0$$, the physical temperature is $$T$$, and
$$\Delta T=(\gamma-1)T$$ sets how quickly hills shrink. Large accumulated
bias means a smaller next hill. The dimensionless bias factor $$\gamma$$ must
exceed one.

The JAX function below performs exactly one deposition event. `jnp.interp`
reads the current local bias, the exponential tempers the height, and the final
line adds a Gaussian across the grid.

{% include kups-notebooks/post-11/post11-jax-adaptive-bias.html %}

With no accumulated bias, the deposited height equals the requested 0.03000.
When the local bias is already $$2k_{\mathrm B}T$$ and $$\gamma=10$$, the same
request deposits only 0.02402. This is the adaptive mechanism—not yet a
free-energy estimate.

In the long-time well-tempered limit, the standard reconstruction is

$$
F(s)
=
-\frac{\gamma}{\gamma-1}V(s)+C,
$$

where $$C$$ is an arbitrary additive constant. The relation is asymptotic. It
does not excuse a poor collective variable, an unconverged bias history, or
missing support in a hidden slow coordinate.

### Test adaptive bias against an answer key

The full analytic control uses a one-dimensional double well whose barrier is
known before sampling. It deposits 3,000 hills with $$\gamma=10$$. The final
bias range is 6.378 reduced-energy units, both basins receive nearly equal
support, and the reconstructed barrier differs from the answer by 0.0749.

<div class="table-responsive" markdown="1">

| Adaptive-bias check | Full value | Interpretation |
|---|---:|---|
| deposited hills | 3,000 | length of the history |
| final bias range | 6.378 | the landscape changed substantially |
| left / right basin visits | 0.360 / 0.362 | both basins were explored |
| barrier-region visits | 0.134 | the barrier region gained support |
| barrier-height error | 0.0749 | known-answer reconstruction error |

</div>

Equal basin counts alone would not validate the method. The answer-key error
does. On a molecular problem without an answer key, repeatability across
independent bias histories and stability to hill height, width, bias factor,
and deposition interval become essential.

This metadynamics example is an analytic teaching control. The physical kUPS
experiment below performs steered MD, not metadynamics. Keeping that boundary
explicit prevents evidence from one method being borrowed to support another.

## Part II: steering prescribes a protocol

For the kUPS experiment, the collective variable is the minimum-image distance
$$r(\mathbf R)$$ between two argon atoms. A harmonic restraint has a center
$$c(t)$$ that moves from 3.8 Å to 7.5 Å:

$$
U_{\mathrm{bias}}(\mathbf R,t)
=
\frac{K}{2}\left[r(\mathbf R)-c(t)\right]^2.
$$

The restraint center is an energetic preference, not a position constraint.
At every MD step, Lennard--Jones forces, thermal noise, inertia, and the bias
force jointly determine the next atomic state. The realized distance can lag,
overshoot, or fluctuate around the center.

### Work occurs when the protocol changes

The implementation keeps the center fixed within each stored MD block. At a
block boundary it changes the center from $$c_i$$ to $$c_{i+1}$$ while holding
the stored coordinate $$r_i$$ fixed. Work is the instantaneous energy change
caused by that parameter update:

$$
\Delta W_i
=
U_{\mathrm{bias}}(r_i,c_{i+1})
-U_{\mathrm{bias}}(r_i,c_i)
=
\frac{K}{2}
\left[
(r_i-c_{i+1})^2-(r_i-c_i)^2
\right].
$$

The atoms do not move during this accounting event. Coordinate propagation at
a fixed center contributes heat and internal-energy change, but not protocol
work. The last stored frame has no following center change, so its increment
is zero.

{% include kups-notebooks/post-11/post11-jax-work.html %}

For the three-event control, the two center changes contribute 0.0096 and
0.0725 eV. The final zero is not padding; it encodes the fact that no later
protocol value exists. Summing this array reconstructs the total work from the
stored distances and center schedule.

### Map that definition to kUPS

The real worker implements the same event sequence:

1. read the kUPS MD step counter and select the corresponding center;
2. evaluate the harmonic pair restraint with that center;
3. add it to the Lennard--Jones energy with `sum_potentials`;
4. obtain forces from the combined energy inside the kUPS propagator;
5. advance BAOAB Langevin dynamics with `run_md`;
6. store positions, reconstruct minimum-image distances, and evaluate the JAX
   work formula above at block boundaries.

kUPS defines its harmonic bond as $$k(r-r_0)^2$$, whereas this chapter writes
$$K(r-r_0)^2/2$$. The worker therefore passes $$k=K/2$$. Before production, it
compares the dynamic first-center potential with an independently built static
restraint in both energy and gradient. It also verifies that zero drive gives
zero work and that a kUPS energy difference matches the analytic increment.

Those checks establish the implemented Hamiltonian and accounting convention.
They do not show that the protocol is slow or that the path ensemble is large
enough.

## Run actual steering trajectories

The fresh-kernel notebook launches four CPU-sized kUPS paths: fast and slow
protocols in both forward and reverse directions. Each isolated worker
constructs the combined potential, propagates real MD, writes HDF5 positions,
and returns a compact work trace. The notebook then invokes the same verifier
used by the command-line workflow.

{% include kups-notebooks/post-11/smoke-run.html %}

The new smoke run stores 160 frames. Its fast hysteresis gap is 0.2032 eV and
its slow gap is 0.0230 eV. Zero drive gives exactly zero work, and the largest
kUPS-versus-analytic increment error is $$3.90\times10^{-9}$$ eV. This is
bounded execution evidence on CPU; the independent-replica GPU record carries
the quantitative comparison.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post11_steered_atom_path.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A moving restraint guides rather than prescribes the atomic path. Left: 50 frames selected from one real 200-frame slow-forward kUPS trajectory, with atom 0 fixed at the origin and atom 1 shown by its minimum-image x-y displacement. Right: mean Ar--Ar distance and one-standard-deviation ribbons from all four forward replicas at each speed. The gray dashed line is the prescribed center." %}

The left panel is an actual atomic trajectory, not a schematic interpolation
between the endpoints. The three-dimensional pair displacement is projected
onto the x-y plane, so radial pulling and rotational diffusion both appear.
The dashed circles show only the initial and final restraint centers.

The right panel separates the protocol from the response. Across the four
forward paths, the mean absolute distance-to-center lag is 0.74 Å for fast
pulling and 0.52 Å for slow pulling. Slowing the schedule lets the coordinate
track its moving preference more closely, but the shaded replica spread makes
clear that thermal fluctuations remain.

## Hysteresis asks whether slower was actually gentler

For a forward path, the second law implies

$$
\langle W_{\mathrm F}\rangle\geq\Delta F.
$$

For the reverse direction,

$$
\langle W_{\mathrm R}\rangle\geq-\Delta F.
$$

Their sum defines a loop-width diagnostic,

$$
H
=
\langle W_{\mathrm F}\rangle
+\langle W_{\mathrm R}\rangle.
$$

An ideal quasistatic pair of protocols has $$H=0$$. Finite-speed driving,
coordinate lag, and relaxation hidden from the chosen coordinate make the loop
wider. Hysteresis is not itself a free-energy estimator; it answers a more
limited question: did allocating more MD steps reduce dissipation?

The full experiment uses two Ar atoms in a periodic 30 Å cube at 100 K,
Lennard--Jones parameters $$\sigma=3.405$$ Å and $$\epsilon=0.010326$$ eV,
and $$K=0.02$$ eV/Å$$^2$$. Four independent replicas are run for every
speed--direction combination, giving 16 GPU paths and 2,000 stored frames.

{% include kups-notebooks/post-11/production-evidence.html %}

<div class="table-responsive" markdown="1">

| Production check | Full-profile value | What it supports |
|---|---:|---|
| observed device | NVIDIA RTX A5000 | required GPU execution |
| fast hysteresis | 0.05258 ± 0.02600 eV | four-path mean and path SEM |
| slow hysteresis | 0.01113 ± 0.00866 eV | smaller than fast |
| fast / slow ratio | 4.72 | slower protocol narrows this loop |
| static energy / gradient error | 0 / 0 | dynamic first center matches static bias |
| zero-drive work error | 0 eV | no parameter change means no work |
| maximum work-increment error | $$6.82\times10^{-10}$$ eV | stored-frame accounting matches kUPS |

</div>

The uncertainty is not decorative. One fast-forward replica records 0.0848 eV
of work while another records -0.0205 eV. Four paths support the limited claim
that this slower protocol reduces the loop width in this teaching system. They
do not determine the tails of a work distribution precisely.

## Jarzynski is exact and tail-sensitive

Jarzynski's equality relates a nonequilibrium path ensemble to the equilibrium
free-energy difference:<sup id="cite-jarzynski"><a href="#ref-jarzynski">2</a></sup>

$$
e^{-\beta\Delta F}
=
\left\langle e^{-\beta W}\right\rangle,
\qquad
\beta=\frac{1}{k_{\mathrm B}T}.
$$

The average is over independently initialized paths generated by the same
protocol. Taking the logarithm gives the finite-sample estimator

$$
\widehat{\Delta F}
=
-k_{\mathrm B}T
\log\left[
\frac{1}{N}\sum_{n=1}^{N}
e^{-W_n/(k_{\mathrm B}T)}
\right].
$$

Low-work paths receive exponentially larger weights. Numerically, the JAX
implementation uses `logsumexp` so tiny weights do not underflow. It also
reports the normalized effective sample size

$$
\frac{N_{\mathrm{eff}}}{N}
=
\frac{\left(\sum_n a_n\right)^2}
{N\sum_n a_n^2},
\qquad
a_n=e^{-W_n/(k_{\mathrm B}T)}.
$$

{% include kups-notebooks/post-11/post11-jax-work-estimator.html %}

When all 12 control paths have the same work, Jarzynski must return that work
and every path carries equal weight. The output recovers 0.070 eV and an ESS
fraction of 1.000. Real nonequilibrium work is neither constant nor so kind.

For the radial Ar-pair answer key, $$\Delta F=-0.00570$$ eV between the two
restrained endpoints. Only four slow paths per direction give forward and
reverse Jarzynski estimates of -0.01126 and -0.01238 eV. They agree with each
other to 0.00112 eV, yet both miss the answer by roughly 0.006--0.007 eV. The
ESS fractions are 0.546 and 0.672.

This is the central warning: directional agreement and a moderate ESS do not
manufacture an unsampled exponential tail. A defensible free-energy claim
would need more independent equilibrated starting states, path-count and tail
stability tests, and preferably a bidirectional estimator using forward--reverse
overlap.<sup id="cite-crooks"><a href="#ref-crooks">3</a></sup>

## A prediction to test

Before changing code, predict the effect of each intervention:

1. Halve the metadynamics hill width while keeping its height fixed. Which
   parts of the reconstructed free energy should become noisier?
2. Double the number of MD steps in the slow steering protocol without changing
   its endpoints. Should the restraint-center lag, hysteresis, and thermal
   trajectory spread all decrease in the same way?
3. Add many typical-work paths but no unusually low-work paths. Can the ordinary
   work mean stabilize while the Jarzynski estimate remains biased?

The useful answer to the second question is “not necessarily.” Slower driving
should reduce systematic lag and dissipation, but it does not remove equilibrium
thermal fluctuations. For the third, Jarzynski can remain unstable because its
information lives in a different tail than the ordinary mean.

<details class="kups-code-block kups-code-block--collapsed">
<summary>Reproducibility record and full diagnostic dashboard</summary>
<div markdown="1">

The committed source of truth includes the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-11-enhanced-sampling.ipynb),
[smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/smoke/enhanced_sampling_summary.json),
[production summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/enhanced_sampling_summary.json),
[stored steering trace](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/kups_steering_samples.csv),
[manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/manifest.json),
[kUPS worker](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/kups_steering_worker.py),
[JAX reference algorithms](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/jax_reference.py),
[figure source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/steering_visuals.py),
[figure-generation entry point](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post11_figures.py),
and [review record](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-11.md).

```bash
uv run kups-tutorial run 11 --profile smoke
uv run kups-tutorial verify 11 --profile smoke
uv run kups-tutorial run 11 --profile full
uv run kups-tutorial verify 11 --profile full
uv run kups-tutorial verify-notebooks --posts 11 --timeout 180
```

Each of the 16 production paths records the input CIF and HDF5 SHA-256, raw
byte count and dataset schema, seed, speed, direction, replica, frame count,
observed device, runtime, block thermodynamics, and work controls. The compact
CSV retains every center, distance, work increment, and cumulative work value.
The figure provenance records the selected HDF5 hash, frame indices,
minimum-image convention, projection, and all-forward-replica trace hash.

The multi-panel dashboard below preserves the analytic work controls, complete
metadynamics reconstruction, driven-coordinate traces, and cumulative work.
It is useful for audit, but the two-panel atomic figure carries the main
physical argument.

{% include figure.liquid loading="lazy" path="assets/img/blog/kups_md_post11_enhanced_sampling_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Full Post 11 diagnostic dashboard. The analytic metadynamics, work-identity, and finite-speed controls are separated from the real kUPS Ar-pair coordinate and cumulative protocol work." %}

</div>
</details>

## What this chapter establishes—and what it does not

The adaptive control shows that the standard well-tempered update can fill and
approximately reconstruct a known double well. It does not show that kUPS ran
metadynamics on an atomistic system.

The physical experiment shows that kUPS ran a time-dependent harmonic
restraint on real Ar-pair trajectories, that work is reconstructible at the
actual discrete parameter changes, and that a slower schedule reduced lag and
forward--reverse hysteresis. It does not show that four paths converge an
exponential-work free energy or that pair distance is a challenging reaction
coordinate.

That separation is the point. Enhanced sampling is trustworthy when the
modified dynamics, stored evidence, and estimator correspond to the same
mathematical experiment.

## Closing

Adaptive bias and steering both make rare motion easier by changing the
measure that generates trajectories. The correction must match the change.

For metadynamics, preserve and test the bias history. For steered MD, define
work at the same protocol events the code executes and collect an ensemble of
independent paths. In both cases, a visually successful crossing is only the
beginning. The result starts when you can explain what distribution generated
the data and why the estimator has support there.

## References

1. <span id="ref-metadynamics"></span>Barducci, A., Bussi, G. & Parrinello, M. (2008). Well-tempered metadynamics: A smoothly converging and tunable free-energy method. [Physical Review Letters 100, 020603](https://doi.org/10.1103/PhysRevLett.100.020603). <a href="#cite-metadynamics" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-jarzynski"></span>Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. [Physical Review Letters 78, 2690](https://doi.org/10.1103/PhysRevLett.78.2690). <a href="#cite-jarzynski" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-crooks"></span>Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. [Physical Review E 60, 2721](https://doi.org/10.1103/PhysRevE.60.2721). <a href="#cite-crooks" class="reversefootnote" role="doc-backlink">↩</a>
