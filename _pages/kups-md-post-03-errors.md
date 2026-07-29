---
layout: post
permalink: /kups-md-tutorials/post-03-errors/
title: "How Do Timestep, Precision, and Force Error Become Simulation Error?"
date: 2026-07-14
last_updated: 2026-07-29
description: "How timestep, numerical precision, force bias, and neighbor lists leave different signatures in an MD trajectory."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 3
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for ML researchers who are new to simulation."
series_order: 3
categories: [science]
tags: [molecular-dynamics, timestep, precision, force-error, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Part 3 of the kUPS Molecular Dynamics Tutorials. The executable example is maintained in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>


An NVE diagnostic often gets compressed into one question: did the energy
drift? That question is too blunt. The same total-energy trace can mix bounded
integrator error, a timestep that is too large, mixed-precision roundoff, force
noise, systematic MLIP bias, and neighbor-list artifacts.

For ML researchers, the useful decomposition is more practical: which part of
the error came from the discrete timestep, which part came from numerical
precision, and which part came from the force model? If these are not separated
early, later claims about thermostat behavior, uncertainty, or free energies
can inherit a hidden simulation artifact.

This page uses the same controlled oscillator as the integrator post because
the exact reference is known. That is a feature, not a simplification to hide
behind. A many-body trajectory can make every mechanism happen at once. A
controlled oscillator lets us isolate mechanisms before adding atomistic
complexity back in. The full run now adds a larger
reduced-unit argon NVE protocol check so the article includes physical
many-body energy traces with timestep and replica variation. The committed run
records CPU fallback because this environment does not have a CUDA-enabled
stack, so the final series still needs real GPU kUPS production checks and, in
the capstone, MLIP-specific extrapolation diagnostics. This hidden draft is the
mechanism-level layer that those later checks should use.

The core distinction is between four words that often get blurred: error,
drift, instability, and uncertainty. Error is any difference from a reference or
target quantity. Drift is a systematic trend, often measured after normalizing
by time and energy scale. Instability is qualitative failure: a trajectory that
leaves the physically meaningful regime or a diagnostic that blows up.
Uncertainty is what remains after estimating a quantity from finite data. A
good MD report should not use one number to stand in for all four.
The numerical-analysis literature separates geometric timestep error from
finite-precision arithmetic for exactly this reason
(<span id="cite-hairer2006"></span>[Hairer et al., 2006](#ref-hairer2006);
<span id="cite-leimkuhler2004"></span>[Leimkuhler & Reich,
2004](#ref-leimkuhler2004); <span id="cite-higham2002"></span>[Higham,
2002](#ref-higham2002)).

## What Must Be Separated?

The current diagnostic fixes the oscillator and varies three axes:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | exact reference trajectory is known |
| timesteps | 0.02, 0.05, 0.1, 0.18 | exposes timestep sensitivity |
| steps per run | 3000 | makes drift measurable |
| precision models | float64, float32, rounded grids | separates arithmetic effects |
| force scales | 0.98, 1.0, 1.02 | deterministic force-error proxy |
| total runs | 48 | full grid over all three axes |

The full profile also includes a many-body NVE protocol check:

| Choice | Full value | Why it matters |
|---|---:|---|
| protocol label | gpu_ready_lj_nve_replicas | explicit production-style diagnostic path |
| target device | cuda_or_cpu_fallback | records that this run used CPU fallback here |
| runtime device | jax:cpu;devices:cpu | generated artifact provenance |
| production GPU ready | false | CUDA/GPU target was not satisfied in this environment |
| argon cell | 256 atoms | larger than the initial compact 108-atom check |
| replicas | 3 velocity seeds | exposes initialization sensitivity |
| timesteps | 0.0015, 0.003, 0.006 | checks timestep-dependent energy behavior |
| steps | 1200 | compact but nontrivial many-body NVE trace |

The force-scale perturbation is not meant to model every MLIP failure. It is a
controlled negative example: even a simple systematic force bias changes the
energy behavior, so MLIP diagnostics should not hide force error inside a
single timestep-convergence number.

The separation is the point. Timestep error comes from replacing the continuous
flow with a finite update. Precision error comes from representing and
combining numbers with finite arithmetic. Force error comes from evaluating the
wrong force, whether because the model is approximate, the neighbor list is
stale, the cutoff is discontinuous, or the learned potential is extrapolating.
These mechanisms can compensate or mask each other in a single trajectory.

For the harmonic oscillator, the exact force is simple. If the potential is
one-half x squared, the exact force is minus x. The diagnostic perturbs that
force by multiplying it by 0.98 or 1.02. That is not a realistic MLIP error
model, but it is deliberately interpretable: one run is slightly too soft, one
is correct, and one is slightly too stiff. The resulting drift and phase error
can then be read without debating what the model learned.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_timestep_error.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The controlled sweep shows how bounded integration error grows with timestep. A stable-looking trace can still be quantitatively inaccurate." %}

## Why Is Energy Drift Not One Thing?

Energy in a finite-timestep Hamiltonian simulation is subtle. The exact NVE
system conserves the physical Hamiltonian. A stable symplectic integrator such
as velocity Verlet does not conserve that exact energy at every step. It often
tracks a nearby modified energy while the reported physical energy oscillates.
That bounded oscillation is not the same failure mode as a monotonic drift.

The diagnostic therefore records both maximum absolute relative energy error
and normalized energy drift. The maximum error asks how far the reported energy
gets from the initial value. The normalized drift asks whether the final energy
has shifted systematically relative to elapsed simulated time. Both are useful,
and neither is complete by itself.

For the exact-force float64 runs, the full profile shows the timestep story.
The maximum relative energy error increases from about 1.0e-4 at dt = 0.02 to
about 8.1e-3 at dt = 0.18. That is the expected direction: larger timesteps
give larger bounded discretization error. But the normalized final drift
remains small, about -1.56e-7 at dt = 0.02 and about -2.04e-6 at dt = 0.18.
Those values describe a controlled oscillator test, not a universal production
timestep rule.

The biased-force cases tell a different story. At dt = 0.18, the low-force
case has normalized drift around -4.51e-5 in float64, much larger in magnitude
than the exact-force case. The high-force case changes the sign and magnitude.
That shift is not caused by the timestep alone. The map is integrating a
different force than the energy diagnostic assumes. This is the simplest
version of a common production problem: the conserved quantity one plots may
not correspond to the force actually used by the dynamics.

## What Does Precision Change?

Precision is often discussed too vaguely. "Use float64" and "float32 is fine"
are not scientific claims unless they are attached to a diagnostic. Precision
affects roundoff, accumulation, force evaluation, reductions, and sometimes the
branching behavior of neighbor-list or model code. In a simple oscillator, most
of that complexity is absent, so the diagnostic uses explicit precision models.

The full profile includes float64, float32, rounded_1e-4, and rounded_1e-3.
The rounded models are intentionally crude. They create a visible arithmetic
floor by rounding the state or intermediate values to a grid. This is not meant
to be a hardware model. It is a readable stress test showing that arithmetic
can dominate once the discretization error is small enough.

At dt = 0.18 with exact force, the maximum relative energy error is about
8.1e-3 for float64 and float32, about 1.08e-2 for rounded_1e-4, and about
2.0e-2 for rounded_1e-3. The rounded_1e-3 result is not a timestep failure in
the same sense as increasing dt. It is a precision-induced floor. Reducing the
timestep will not necessarily remove a floor caused by coarse arithmetic.

This distinction matters for MLIP simulations because many learned-potential
workflows mix precisions. Neighbor construction, model inference, force
accumulation, and integration may not all use the same dtype. A model can be
trained in one precision and deployed in another. GPU kernels may use fused
operations or reduced precision internally. The right question is not whether a
label says "float32" or "float64"; it is whether the resulting trajectory-level
diagnostics meet the tolerance needed for the scientific claim.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_precision_floor.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Reducing the timestep eventually exposes a precision-dependent error floor. Smaller steps cannot remove rounding error once arithmetic dominates discretization." %}

## How Does Force Error Enter?

Force error is different from roundoff and timestep truncation. The integrator
can be implemented correctly and run at a reasonable timestep while still
following the wrong vector field. In classical force fields, this can happen
because parameters or cutoffs are wrong. In MLIP simulations, it can happen
because the learned potential is inaccurate, extrapolative, noisy, or
inconsistent with the energy used for diagnostics.

The force-scale cases in this page make that mechanism visible. A 2 percent
force-scale perturbation is simple enough to understand, but it changes phase
and energy behavior. The low-force oscillator is too soft; the high-force
oscillator is too stiff. The position error grows even when the run is not
unstable. That is important: not every model error produces an immediate crash.
Some errors produce plausible but biased trajectories.

This is why static force RMSE is not enough for MD. A low average error on a
held-out set may hide systematic bias in the region visited by dynamics. A
model can have acceptable static metrics and still produce drift or wrong
kinetics when integrated. Conversely, a trajectory-level drift diagnostic does
not identify the source by itself. It must be paired with force-error,
extrapolation, precision, and timestep checks.

Post 12 will need this distinction for the MACE/fcc-Al capstone. The current
post does not claim to be that capstone. It provides a vocabulary: force bias,
precision floor, bounded timestep error, normalized drift, phase error, and
instability are different diagnoses.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_force_bias.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A persistent force bias produces directional energy drift rather than bounded oscillation. This signature differs from the timestep error of a reversible symplectic update." %}

## What Is Normalized Drift?

Raw energy drift is hard to compare across systems. A change of 0.001 eV may
be large for one small system and irrelevant for another. A drift over 10 ps is
not the same as the same drift over 1 ns. Normalized drift divides by an energy
scale and elapsed time so that comparisons are less dependent on arbitrary run
length and system size.

The exact normalization depends on the project convention, but the intent is
stable: measure systematic energy change per unit simulated time relative to a
meaningful energy scale. This is different from the maximum bounded energy
oscillation. A stable symplectic trajectory can have a visible maximum energy
error and very small normalized drift. A biased force can have a drift sign and
magnitude that reveal the dynamics are following the wrong vector field.

For a production report, the normalized drift number should be paired with the
energy trace. A scalar can hide transients, warmup behavior, rare jumps, or
instability. The trace can show whether the drift is monotonic, oscillatory,
step-like, or dominated by an early event. Later posts use this logic when
thermostats, barostats, and sampling diagnostics are added.

## What About Phase Error?

Energy is not the only error. The oscillator can have small energy error and
still gradually move out of phase with the exact solution. For observables that
depend on dynamics, timing, or correlation functions, phase error can matter.
This is one reason the summary records final position error and RMS position
error in addition to energy metrics.

In many-body MD, exact position references are usually unavailable beyond very
short times because trajectories are chaotic. That does not make phase error
irrelevant. It means the diagnostic changes form. One may compare short-time
force consistency, time-correlation functions, conserved quantities, replica
statistics, or observable convergence rather than a one-to-one position trace.

The controlled oscillator is useful because it keeps phase error visible. At
larger dt or with biased forces, position error grows even when the run remains
finite. That distinction warns against judging a simulation only by whether it
crashed or whether the energy stayed within a loose band.

## How Should This Guide Timestep Choice?

A timestep is not chosen by tradition alone. It should be justified against the
fastest relevant modes, the desired observable, the force smoothness, the
precision policy, and the acceptable error. In classical MD, constraints and
hydrogen mass repartitioning may allow larger timesteps for some observables.
In reactive or MLIP simulations, sharp force changes or extrapolative regimes
may require more conservative choices.

A practical workflow is:

| Step | Question | Evidence |
|---|---|---|
| exact or controlled test | Does timestep error shrink as dt shrinks? | timestep sweep |
| precision check | Does arithmetic set an error floor? | dtype or rounding comparison |
| force check | Does force perturbation mimic observed drift? | biased-force or model comparison |
| short NVE run | Does the real system show bounded energy behavior? | energy trace and normalized drift |
| observable check | Is the target observable stable to dt? | repeated analysis at smaller dt |

The last row matters. A timestep can look acceptable by energy diagnostics but
still bias a time-correlation function or rare-event rate. Conversely, a small
bounded energy oscillation may be acceptable for a structural observable. The
scientific claim determines the tolerance.

## How Should an NVE Error Report Be Read?

A useful NVE error report should let the reader separate mechanism from
judgment. The mechanism is what happened: bounded oscillation, monotonic drift,
rare jumps, instability, phase error, or precision floor. The judgment is
whether that behavior is acceptable for the scientific claim. Those two layers
should not be collapsed.

For example, an energy trace with bounded oscillation can be acceptable if the
oscillation is small relative to the observable of interest and shrinks under a
timestep reduction. The same trace may be unacceptable if the observable is a
high-frequency dynamical quantity. A monotonic drift is more concerning because
it suggests systematic energy injection or removal. A single jump may point to
a neighbor-list rebuild, a discontinuity, a model extrapolation event, or a
numerical overflow. An unstable run that exits the physical regime is not a
large uncertainty bar; it is a failed protocol.

The report should therefore include at least four pieces. First, show the
energy trace or a compact diagnostic derived from it. Second, report normalized
drift so the run length and energy scale are visible. Third, compare at least
one smaller timestep or tighter precision setting when the claim depends on
numerical stability. Fourth, connect the diagnostic to the observable being
reported. A timestep acceptable for mean potential energy is not automatically
acceptable for a diffusion coefficient, vibrational spectrum, or rare-event
rate.

This is where the controlled oscillator helps. It teaches the shapes. Bounded
velocity-Verlet error has one shape. Rounded arithmetic floors have another.
Force bias changes drift and phase behavior. When those shapes appear in a real
trajectory, the report can describe them with more precision than "the energy
looks okay."

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_argon_nve.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The kUPS argon NVE replicas show bounded normalized energy error across the tested timesteps. Replica bands separate numerical variation from a single favorable trace." %}

## Where Do Neighbor Lists and Cutoffs Enter?

The current executable diagnostic does not include neighbor lists or cutoffs,
but production MD almost always does. A pair potential, graph neural network,
or equivariant message-passing model usually decides which atoms interact
through a cutoff or neighbor construction. That machinery can create error
mechanisms that look like integrator problems if they are not tested
separately.

A stale neighbor list means the force is evaluated from an outdated interaction
set. A discontinuous cutoff means the force can jump when a pair crosses the
cutoff radius. A skin distance that is too small can miss interactions between
rebuilds. A learned potential may use neighbor features whose smoothness
depends on cutoff envelopes, radial bases, or graph construction. These are
force-evaluation issues, but they show up in trajectory diagnostics as drift,
jumps, or instability.

This is why an NVE test should record neighbor-list policy as part of
provenance. Rebuild interval, skin distance, cutoff, precision, and model
version are not incidental implementation details. They define the force field
that the integrator actually sees. If two runs use the same timestep but
different neighbor-list settings, they are not clean timestep replicas.

The oscillator cannot test these effects directly. Its value is to establish
the language: if the integrator has bounded error in a controlled case, but the
atomistic trajectory has jumps, the next suspect may be force continuity or
neighbor bookkeeping. If the drift appears only in a learned-potential run and
not in a classical or exact-force control, the force model and its deployment
settings deserve attention.

## How Does This Apply to MLIP Workflows?

Machine-learned interatomic potentials make this separation more important, not
less. A model is usually evaluated by static metrics: force RMSE, energy RMSE,
stress error, or ranking on a held-out set. Those metrics are useful, but they
do not answer the trajectory question by themselves. MD composes force errors
over many steps. It asks the model to behave smoothly under its own generated
states, not only under a curated validation set.

Three failure modes are especially easy to confuse. The first is timestep
failure: even a good model can be integrated with a timestep too large for the
forces it produces. The second is model failure: a reasonable timestep can
still produce drift if forces are biased, noisy, discontinuous, or
extrapolative. The third is deployment failure: neighbor construction,
precision, batching, device kernels, or unit conversion can differ from the
environment in which the model was validated.

The practical response is to build a ladder of controls. Start with a simple
system and exact or trusted force where the timestep behavior is understood.
Then add the target force model while keeping the same initialization and
timestep. Then vary the timestep. Then vary precision or device if deployment
requires it. Then inspect extrapolation or uncertainty diagnostics. A single
green static validation metric should not skip this ladder.

For the final MACE/fcc-Al capstone, the relevant questions will be: does the
model stay in-domain under the initialized dynamics, does normalized energy
drift remain acceptable in NVE-style checks, does the force error correlate
with uncertainty or extrapolation signals, and do static errors predict
trajectory reliability? The oscillator in this post does not answer those
questions. It defines how the answers should be separated.

## Run the Example

The smoke profile follows the same protocol with a smaller CPU workload:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 03 --profile smoke
uv run kups-tutorial verify 03 --profile smoke
```

The repository also contains the [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/full.json), [notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/notebooks), and [recorded results](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/results/post-03/full).

## References

- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. <a href="#cite-hairer2006" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-leimkuhler2004"></span>Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press. <a href="#cite-leimkuhler2004" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-higham2002"></span>Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM. <a href="#cite-higham2002" class="reversefootnote" role="doc-backlink">↩</a>
