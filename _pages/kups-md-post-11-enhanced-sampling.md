---
layout: post
permalink: /kups-md-tutorials/post-11-enhanced-sampling/
title: "How Do Adaptive and Nonequilibrium Enhanced-Sampling Methods Work?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Build a state-dependent restraint into kUPS, reconstruct protocol work from stored frames, and test whether slower pulling reduces hysteresis."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 11
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 11
categories: [science]
tags: [molecular-dynamics, enhanced-sampling, nonequilibrium-work, jarzynski, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The double-well examples are known-answer controls. The Ar-pair result comes from real kUPS trajectories on the devices reported below.</em>
</p>

## A Transition Is Not Yet an Estimate

Enhanced sampling can make an impressive movie and a bad measurement.

A history-dependent bias can push a trajectory out of a free-energy basin. A
moving restraint can drag a coordinate from A to B. Neither observation says
how probable A or B is in equilibrium. The simulation has changed its own
sampling distribution; the estimator must remember exactly how.

This post separates three claims that are often blurred together:

1. an analytic metadynamics-style control shows what history-dependent bias
   does to a known double well;
2. analytic work ensembles test Jarzynski and Crooks identities without an MD
   implementation in the loop;
3. a physical Ar--Ar coordinate is driven by a state-dependent harmonic
   restraint inside kUPS, with work reconstructed from the stored trajectory.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-11-enhanced-sampling.ipynb),
[smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/smoke/enhanced_sampling_summary.json),
[production summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/enhanced_sampling_summary.json),
[stored steering trace](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/kups_steering_samples.csv),
[provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/manifest.json),
[kUPS worker](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/kups_steering_worker.py),
[figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post11_figures.py),
and [review record](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-11.md).

Run and verify either committed profile with the same public commands used by
the review gate:

```bash
uv run kups-tutorial run 11 --profile smoke
uv run kups-tutorial verify 11 --profile smoke
uv run kups-tutorial run 11 --profile full
uv run kups-tutorial verify 11 --profile full
```

{% include kups-notebooks/post-11/setup.html %}

## Adaptive Bias Changes During Sampling

Suppose a collective variable $$s=\xi(\mathbf{x})$$ compresses the atomic
configuration into a coordinate we can bias. Metadynamics deposits Gaussian
hills along that coordinate:

$$
V_t(s)=\sum_{t_i<t} w_i
\exp\left[-\frac{(s-s_{t_i})^2}{2\sigma^2}\right].
$$

The important subscript is $$t$$. The potential used tomorrow depends on where
the trajectory went yesterday. Unlike a fixed umbrella window, this is not a
stationary biased ensemble while hills are still being deposited.

The known-answer control uses a one-dimensional double well. In the full
profile, 3,000 well-tempered hills produce these diagnostics:

<div class="table-responsive" markdown="1">

| Check | Full value | What it establishes |
|---|---:|---|
| final bias range | 6.534 | the procedure substantially changed the landscape |
| reconstructed barrier error | 0.092 | the final bias recovers the controlled barrier reasonably |
| left / right basin visits | 0.360 / 0.362 | both basins received support |
| barrier-region visits | 0.135 | the barrier is no longer invisible |

</div>

A large bias range is not an accurate PMF. It proves only that the adaptive
policy acted strongly. Accuracy comes from a reconstruction test, repeated
runs, and stability with respect to hill height, width, and bias factor. In a
real molecular system without an answer key, disagreement between repeats is
evidence, not nuisance variation to average away.<sup id="cite-metadynamics"><a href="#ref-metadynamics">1</a></sup>

## A Moving Restraint Defines Protocol Work

For the physical experiment, the collective variable is the minimum-image
distance $$r(\mathbf{x})$$ between two argon atoms. A harmonic restraint has a
center $$c(t)$$ that moves according to a prescribed schedule:

$$
U_{\mathrm{bias}}(\mathbf{x},t)
=\frac{K}{2}\left[r(\mathbf{x})-c(t)\right]^2.
$$

kUPS defines its harmonic bond as $$k(r-r_0)^2$$, so the worker passes
$$k=K/2$$. More importantly, the worker does not generate a path in NumPy and
attach a device label afterward. It creates a parameter view that reads the MD
step counter, selects the current center, composes the restraint with the kUPS
Lennard--Jones potential through `sum_potentials`, and integrates BAOAB
Langevin dynamics through `run_md`.

The center remains fixed within each stored block. At the boundary after frame
$$i$$, it changes from $$c_i$$ to $$c_{i+1}$$ while the coordinate is held at
the stored value $$r_i$$. The work increment is therefore

$$
\Delta W_i
=\frac{K}{2}
\left[
(r_i-c_{i+1})^2-(r_i-c_i)^2
\right].
$$

The last frame has no following center change, so its increment is zero. This
definition makes total work exactly reconstructible from the HDF5 positions
and committed center schedule:

{% include kups-notebooks/post-11/work-definition.html %}

Before any trajectory runs, the worker checks that the dynamic schedule at its
first center matches an independently constructed static restraint in both
energy and gradient. It also checks that a zero-length center move gives zero
work and that the kUPS energy difference across one center change matches the
analytic expression above.

These are Hamiltonian and accounting tests. They do not prove that the pulling
protocol is slow enough.

## The Notebook Launches Real kUPS Paths

The fresh-kernel notebook launches four CPU-sized paths: fast and slow
protocols in both forward and reverse directions. Each isolated worker writes
a real kUPS HDF5 trajectory. The parent process reads minimum-image distances,
reconstructs every work increment, and invokes the same verification gate used
by the command-line workflow:

{% include kups-notebooks/post-11/smoke-run.html %}

The smoke profile stores 160 frames in total. Its fast hysteresis gap is
0.2032 eV, while the slow gap is 0.0230 eV. The zero-drive error is exactly
zero, and the maximum kUPS-versus-analytic work error is
$$3.90\times10^{-9}$$ eV. This run proves that the executable path works on a
CPU. Publication evidence has a separate GPU-only gate and independent
replicas.

## Hysteresis Tests Protocol Speed

For a forward protocol, equilibrium thermodynamics gives the free-energy
difference $$\Delta F$$ and nonequilibrium work satisfies

$$
\langle W_{\mathrm F}\rangle \geq \Delta F.
$$

For the reverse protocol,

$$
\langle W_{\mathrm R}\rangle \geq -\Delta F.
$$

Their sum is a useful loop-width diagnostic:

$$
H=\langle W_{\mathrm F}\rangle+
\langle W_{\mathrm R}\rangle.
$$

An ideally quasistatic pair of protocols has $$H=0$$. Finite-speed driving,
coordinate lag, and hidden relaxation produce positive hysteresis. This scalar
does not reconstruct a free energy, but it asks a sharp engineering question:
does spending more MD steps actually make the driven ensemble less
dissipative?

The production experiment uses the following physical protocol:

<div class="table-responsive" markdown="1">

| Parameter | Full profile |
|---|---:|
| system | two Ar atoms in a periodic 30 Å cube |
| temperature | 100 K |
| Lennard--Jones $$\sigma / \epsilon$$ | 3.405 Å / 0.010326 eV |
| restraint path | 3.8 Å $$\leftrightarrow$$ 7.5 Å |
| restraint strength $$K$$ | 0.02 eV/Å$$^2$$ |
| integrator | BAOAB Langevin, 1 fs |
| warmup | 2,000 steps per path |
| fast / slow production | 2,000 / 8,000 steps |
| independent paths | 4 per speed and direction |
| stored frames | 2,000 across 16 paths |

</div>

{% include kups-notebooks/post-11/production-evidence.html %}

The full results pass the production gates:

<div class="table-responsive" markdown="1">

| Check | Full-profile value | Gate |
|---|---:|---:|
| observed device | NVIDIA RTX A5000 | GPU required |
| fast hysteresis gap | 0.05258 ± 0.02600 eV | reported with path SEM |
| slow hysteresis gap | 0.01113 ± 0.00866 eV | less than fast |
| fast / slow gap ratio | 4.72 | > 1.10 |
| static-schedule energy / gradient error | 0 / 0 | near zero |
| zero-drive work error | 0 eV | ≤ $$10^{-12}$$ eV |
| kUPS work-increment error | $$6.82\times10^{-10}$$ eV | ≤ $$10^{-6}$$ eV |

</div>

The path-level spread is visible in the uncertainty. One fast forward replica
did 0.0848 eV of work; another did -0.0205 eV. Four replicas are enough to
demonstrate a slower-protocol improvement in this teaching system, but not
enough to pretend the work distribution has a precisely known tail.

## Jarzynski Is Exact and Still Data-Hungry

Jarzynski's equality converts a nonequilibrium work ensemble into an
equilibrium free-energy difference:<sup id="cite-jarzynski"><a href="#ref-jarzynski">2</a></sup>

$$
e^{-\beta\Delta F}=\left\langle e^{-\beta W}\right\rangle.
$$

The exponential gives rare low-work paths enormous leverage. Exactness of the
identity does not imply low variance of its finite-sample estimator. Crooks'
forward--reverse relation makes the overlap problem more visible, but it also
needs supported work distributions.<sup id="cite-crooks"><a href="#ref-crooks">3</a></sup>

The radial known-answer calculation gives
$$\Delta F=-0.00570$$ eV between the two restrained endpoints. From only four
slow paths per direction, the forward and reverse Jarzynski estimates are
-0.01126 and -0.01238 eV. They agree with each other to 0.00112 eV, yet both
remain roughly 0.006--0.007 eV from the answer key. That is the lesson in one
line: directional agreement does not manufacture a sampled exponential tail.

The corresponding exponential-weight ESS fractions are 0.546 and 0.672. They
are useful diagnostics, not certificates of convergence. The article reports
the Jarzynski values because hiding an imperfect small ensemble would teach
the wrong habit. A production free-energy claim would require more independent
equilibrated starting configurations, tail-stability checks, and preferably a
bidirectional estimator.<sup id="cite-hummer"><a href="#ref-hummer">4</a></sup>

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post11_enhanced_sampling_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Post 11 diagnostics. The top row contains known-answer metadynamics and work-identity controls. The lower-left panel is an analytic finite-speed control. The lower-middle and lower-right panels show the actual Ar--Ar coordinate and discrete cumulative work from real kUPS GPU trajectories." %}

The physical distance does not track the dashed restraint center exactly. It
lags, overshoots, and relaxes stochastically. That is precisely why a steered
trajectory is path data rather than a table of target coordinates.

## What the Evidence Chain Preserves

Raw HDF5 trajectories are too large for the site repository, but each of the
16 production runs leaves an identifiable record:

- input CIF and raw HDF5 SHA-256 hashes;
- raw HDF5 byte count, dataset names, shapes, and dtypes;
- seed, speed, direction, replica, frame count, and atom count;
- observed JAX device and elapsed time;
- thermodynamic block summaries and all four work controls.

The committed steering CSV retains every center, distance, work increment, and
cumulative work value used in the analysis. The manifest hashes that CSV, the
summary, and the plotting curves. Verification rejects a malformed raw hash,
missing HDF5 schema field, frame-count drift, output-hash mismatch, CPU full
run, failed zero-drive test, failed kUPS work check, or a slow protocol that
does not improve hysteresis.

A hash does not validate the physics. It prevents the figure, summary, and raw
trajectory identity from quietly drifting apart.

## What to Report

A defensible time-dependent enhanced-sampling result should answer these
questions:

- What coordinate was biased, in what units, and with what periodic-image
  convention?
- How does the simulation code select the time-dependent parameter?
- At what instant is work evaluated relative to coordinate propagation and
  parameter changes?
- Can total work be reconstructed from stored frames and the schedule?
- Does zero drive give zero work, and does the energy change match the work
  formula?
- How were endpoint configurations equilibrated and independent paths seeded?
- Do slower protocols reduce forward--reverse hysteresis?
- Are Jarzynski or Crooks estimates supported by work-space overlap and tail
  diagnostics?
- Which device ran each trajectory, and which compact artifacts identify the
  raw data?

The method name is not the evidence. The path ensemble and its accounting are.

## Closing

Adaptive bias and nonequilibrium driving solve a sampling problem by changing
the measure. That trade is useful only when the change is explicit.

In metadynamics, preserve the bias history. In steered MD, define work at the
same discrete events the code executes. In both cases, compare independent
runs and expose the support of the estimator. A trajectory that crosses a
barrier is a beginning. A result starts when you can say what distribution
generated it and how the correction was tested.

## References

1. <span id="ref-metadynamics"></span>Laio, A. & Parrinello, M. (2002). Escaping free-energy minima. [PNAS 99, 12562](https://doi.org/10.1073/pnas.202427399). <a href="#cite-metadynamics" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-jarzynski"></span>Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. [Physical Review Letters 78, 2690](https://doi.org/10.1103/PhysRevLett.78.2690). <a href="#cite-jarzynski" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-crooks"></span>Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. [Physical Review E 60, 2721](https://doi.org/10.1103/PhysRevE.60.2721). <a href="#cite-crooks" class="reversefootnote" role="doc-backlink">↩</a>
4. <span id="ref-hummer"></span>Hummer, G. & Szabo, A. (2001). Free energy reconstruction from nonequilibrium single-molecule pulling experiments. [PNAS 98, 3658](https://doi.org/10.1073/pnas.071034098). <a href="#cite-hummer" class="reversefootnote" role="doc-backlink">↩</a>
