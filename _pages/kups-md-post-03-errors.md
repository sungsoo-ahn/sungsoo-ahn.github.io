---
layout: post
permalink: /kups-md-tutorials/post-03-errors/
title: "How Do Timestep, Precision, and Force Error Become Simulation Error?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible simulation-error diagnostic for molecular dynamics: timestep sensitivity, precision floors, force bias, bounded energy oscillation, and drift."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 3
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 3
categories: [science]
tags: [molecular-dynamics, timestep, precision, force-error, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post builds on the velocity-Verlet diagnostic from the previous tutorial and separates timestep, precision, force-error, and compact argon NVE mechanisms before later GPU kUPS and MLIP production checks. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

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
complexity back in. The current executable workflow now adds a larger
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

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-03/full.json)
- [error notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-03-errors.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/smoke/error_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/error_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-03/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-03.md)

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

## What Should the Diagnostic Show?

The exact-force float64 runs show the timestep story: the maximum relative
energy error grows as the timestep increases, while remaining bounded on this
sweep. The rounded-precision runs show that arithmetic can set an error floor
even when the analytical force is unchanged. The force-scale cases show that a
biased force can shift normalized energy drift.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post03_error_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Simulation-error diagnostics for the committed full profile. The figure separates bounded oscillator timestep error, precision-induced error floors, force-bias drift, and 256-atom argon NVE energy drift with three-replica uncertainty bands so later production checks can report these mechanisms separately." %}

The figure has four jobs. The timestep panel shows that exact-force velocity
Verlet has increasing but bounded energy error as dt grows. The precision panel
shows that rounded arithmetic can raise the error floor. The force-bias panel
shows that a deterministic force perturbation can move normalized drift away
from the exact-force reference. The argon NVE panel checks the same reporting
vocabulary on a 256-atom many-body Lennard-Jones argon cell rather than another
one-dimensional oscillator, and the band shows the standard deviation across
three independent velocity seeds. The panel legend also reports that this
artifact is a CPU-fallback run, not a completed GPU production run.

For the full NVE protocol, the maximum relative energy error across all
timestep/replica runs is about `2.65e-4`, the maximum absolute normalized drift
is about `3.12e-5`, and the largest replica drift standard error is about
`2.79e-6`. No configured NVE run is flagged as unstable. These values support a
bounded-energy diagnostic for the committed reduced-unit protocol; they do not
prove that a future MLIP or CUDA production run is ready.

The figure review caught a small but real readability issue. In an earlier
version, the precision labels were sorted alphabetically, which put
rounded_1e-3 before rounded_1e-4. That made the mechanism harder to scan. The
reviewed figure now orders the precision models as float64, float32,
rounded_1e-4, and rounded_1e-3. A figure that supports a mechanism should make
the mechanism easy to read.

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

## What Should Be Recorded?

The compact summary for this page records the post, profile, config hash,
initial state, mass, oscillator frequency, and one row per run. Each run records
timestep, precision model, force case, force scale, initial and final energy,
maximum relative energy error, normalized energy drift, final position error,
RMS position error, and whether the run was unstable. That is the minimal
shape of a useful error diagnostic: it records mechanism-specific fields rather
than only a pass/fail flag.

The manifest records the environment: config path, config hash, lockfile hash,
git revision, Python version, platform, precision policy, runtime device, kUPS
version, and NumPy version. This is especially important for precision
diagnostics. If the same experiment is rerun on a different device, with a
different precision policy, or after a dependency change, the manifest should
make that visible.

For a production atomistic version, the same pattern should include system
size, density or cell, timestep, integrator, force model, cutoff, neighbor-list
policy, precision, device, seed, warmup policy, and compact energy/drift
summaries. The raw trajectory may be too large to commit. The diagnostic record
should not be.

## What Should Not Be Inferred?

This page does not prove that the final GPU kUPS or MLIP simulations are ready.
The 256-atom argon NVE protocol is a stronger physical sanity check than the
initial compact trace, but it still ran through the committed CPU fallback path
in this environment. The full summary now records `target_device =
cuda_or_cpu_fallback`, `runtime_device = jax:cpu;devices:cpu`,
`production_gpu_ready = false`, and the blocking reason: the target requested
CUDA/GPU, but the generated artifact runtime was CPU. The review note keeps
real GPU kUPS production diagnostics as a final-release item. The oscillator
remains the mechanism-level diagnostic that makes the error vocabulary
testable before production complexity is added.

It also does not imply that MLIP errors are simple force-scale errors. Real
learned potentials can have local extrapolation, nonuniform bias, inconsistent
energy-force behavior, discontinuities from neighbor features, and uncertainty
that is not calibrated. A 2 percent scale perturbation is only a controlled
proxy that shows how force error can enter trajectory diagnostics.

Finally, this page does not replace statistical uncertainty analysis. A stable
trajectory can still be too short. A low drift number does not give an
effective sample size. A precise force model does not remove autocorrelation.
Those questions appear later in the curriculum. The present task is narrower:
keep timestep, precision, and force-error mechanisms from being collapsed into
one vague "simulation error" bucket.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 03 --profile smoke
uv run kups-tutorial verify 03 --profile smoke
uv run kups-tutorial run 03 --profile full
uv run kups-tutorial verify 03 --profile full
uv run jupyter execute notebooks/post-03-errors.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, error diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. It is a substantially expanded hidden draft
that remains outside site navigation while the rest of the series is brought to
the same standard. The implemented pieces are:

- smoke and full controlled error-diagnostic workflows
- 256-atom argon NVE reduced-unit energy-drift workflow with 3 velocity-seed
  replicas
- machine-readable runtime-device and GPU-readiness provenance for the NVE
  protocol
- committed compact summaries and downsampled comparison samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback
- expanded prose separating timestep sensitivity, precision floors, force
  bias, normalized drift, compact argon NVE behavior, phase error, and
  final-release limitations

The remaining non-publication pieces are:

- real CUDA/GPU kUPS production NVE diagnostic before treating this post as
  final
- final all-post consistency pass once the other articles are expanded
- final rendered desktop and mobile page snapshots after that consistency pass
- public indexing decision after the series is ready as a unit

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer.
- <span id="ref-leimkuhler2004"></span>Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
- <span id="ref-higham2002"></span>Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
