---
layout: post
permalink: /kups-md-tutorials/post-04-thermostats/
title: "How Do Thermostats Change Sampling and Dynamics?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible thermostat diagnostic for molecular dynamics: BAOAB Langevin sampling, canonical moment checks, coupling strength, autocorrelation, and dynamical distortion."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 4
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 4
categories: [science]
tags: [molecular-dynamics, thermostats, langevin, sampling, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post builds on the initialization and integrator diagnostics by asking what changes once a thermostat is added, first in a controlled oscillator and then in a compact reduced-unit argon Langevin check. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A thermostat is often described as "keeping the temperature fixed." That phrase
is too vague for molecular simulation. A thermostat changes the equations being
integrated, the distribution being sampled, and usually the dynamical memory of
the trajectory.

For ML researchers, the practical question is not only whether the kinetic
temperature is close to the target. It is whether the sampled configurational
and kinetic moments match the intended ensemble, and whether the coupling is so
strong that time-correlation functions are no longer interpretable as physical
dynamics.

This page uses a harmonic oscillator with BAOAB Langevin dynamics as a
controlled microscope. The oscillator has known canonical moments, so the
thermostat can be checked against targets rather than judged by visual
temperature traces. That does not make the oscillator a production MD system.
It makes the sampling/dynamics tradeoff visible before the series moves back to
argon and later to MLIP aluminum.

The main distinction is simple: a thermostat is a sampling device, not only a
temperature clamp. It can help draw from a canonical distribution, remove or
inject kinetic energy, randomize momenta, change autocorrelation times, and
destroy the dynamics that one might otherwise interpret as physical. A correct
temperature is therefore necessary but not sufficient evidence.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/full.json)
- [thermostat notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-04-thermostats.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/smoke/thermostat_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/thermostat_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-04.md)

## What Does a Thermostat Change?

In an NVE trajectory, the deterministic map tries to follow a constant-energy
Hamiltonian system. A thermostat changes that map. Langevin dynamics adds a
friction term and a random noise term whose balance is chosen so that the
canonical distribution is stationary. In continuous time, a one-dimensional
Langevin oscillator can be written schematically as

$$dx = v\,dt,$$

$$dv = F(x)\,dt - \gamma v\,dt + \sqrt{2\gamma kT/m}\,dW_t.$$

The friction removes momentum memory. The noise injects thermal fluctuations.
The fluctuation-dissipation relation ties the two together. If the relation is
wrong, the thermostat will not sample the intended temperature. If the
relation is right but the coupling is too strong for the observable, the
trajectory may sample reasonable moments while its dynamics are distorted.

BAOAB Langevin is a particular splitting of this stochastic dynamics into
position updates, force kicks, and Ornstein-Uhlenbeck momentum updates. Like
velocity Verlet, the ordering matters. The method is popular because it has
good configurational sampling properties for many problems, especially in
simple tests such as the harmonic oscillator. But "BAOAB" is not a magic word;
the timestep, friction, random seed, warmup, and sampling interval still define
the numerical experiment.

The thermostat therefore changes at least four things:

| Quantity | What changes | Why it matters |
|---|---|---|
| kinetic energy | random noise and friction reshape momenta | target temperature is sampled, not fixed exactly |
| configuration | canonical sampling changes position statistics | structural observables depend on the ensemble |
| memory | friction changes autocorrelation | dynamical observables can be distorted |
| reproducibility | stochastic noise requires a seed | replicas must separate randomness from protocol |

This is why a thermostat report should not stop at average temperature. It
should check whether the intended distribution is sampled and whether the
dynamics remain suitable for the planned analysis.

## What Does Coupling Strength Change?

The current diagnostic fixes the oscillator, timestep, target temperature, and
BAOAB splitting, then changes the Langevin friction:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | canonical targets are known |
| thermostat | BAOAB Langevin | explicit stochastic splitting |
| target temperature | 1.0 | dimensionless kT |
| timestep | 0.02 | shared across coupling strengths |
| friction values | 0.1, 1.0, 5.0 | weak, moderate, and strong coupling |
| full run length | 40000 steps | enough for compact moment checks |
| warmup | 5000 steps | discarded before sampling |
| samples | 3500 per run | sampled every 10 steps |

The check is deliberately two-sided. If the moments are wrong, the thermostat
is not sampling the intended canonical distribution. If the moments are right
but autocorrelation changes sharply, the thermostat may still be unsuitable for
dynamical observables.

The full profile compares gamma = 0.1, 1.0, and 5.0. Those are not universal
recommendations. They are weak, moderate, and strong coupling cases for this
dimensionless oscillator. The useful result is the trend: stronger coupling can
preserve the target moments while changing the correlation structure of the
trajectory.

The random seed is also part of the protocol. A stochastic thermostat is not
reproducible unless the random stream is reproducible. Independent replicas
should change seeds intentionally, not accidentally. If two thermostat runs
disagree, the first question should be whether the difference is expected
stochastic variation or a protocol change.

## What Are the Canonical Targets?

For the dimensionless oscillator used here, the target temperature is kT = 1,
mass is 1, and angular frequency is 1. The canonical distribution has position
variance 1 and velocity variance 1. The mean kinetic energy for one degree of
freedom is 0.5. These are simple targets, so a thermostat diagnostic can be
direct:

| Moment | Target | Interpretation |
|---|---:|---|
| position mean | 0 | distribution centered at oscillator minimum |
| velocity mean | 0 | no net momentum bias |
| position variance | 1 | configurational sampling target |
| velocity variance | 1 | kinetic sampling target |
| kinetic mean | 0.5 | one quadratic velocity degree of freedom |

The full run is not exact because it is finite. Sampling error remains, and the
samples are correlated. The observed position variances are about 0.921, 1.053,
and 1.042 for weak, moderate, and strong coupling. The observed velocity
variances are about 0.945, 1.072, and 1.039. The mean kinetic energies are
about 0.472, 0.536, and 0.519. These are all within roughly 8 percent of the
targets in this compact run.

That result supports a limited claim: the BAOAB Langevin diagnostic samples
near the expected canonical moments for this oscillator and run length. It does
not prove that every distributional detail is exact. The executable workflow
now adds a compact reduced-unit argon Langevin check so the hidden draft also
shows one many-body kinetic-temperature response. That compact check is still
not a GPU kUPS production thermostat benchmark, and the review note keeps the
larger production diagnostic as a final-release blocker.

## Why Is Temperature Alone Not Enough?

Temperature in MD is usually inferred from kinetic energy. A thermostat can
make that number look reasonable while other parts of the distribution are
wrong. A velocity-rescaling scheme, a Langevin thermostat, a Nosé-Hoover chain,
and a stochastic velocity-rescaling thermostat can all target temperature, but
they do not impose the same dynamics or always fail in the same way.

For canonical sampling, the configurational distribution matters. A thermostat
that gives the right kinetic temperature but biases positions is not sampling
the intended ensemble. For dynamics, the memory matters. A thermostat that
samples the right static distribution may still distort diffusion,
time-correlation functions, vibrational spectra, or transport coefficients.

This page therefore checks both moments and autocorrelation. The moment checks
ask whether the sampled distribution is plausible. The autocorrelation check
asks how much dynamical memory remains. Both are required because "temperature
equals target" can hide too much.

The same warning applies to MLIP workflows. If a learned potential has noisy or
biased forces, a thermostat can mask energy drift by continually exchanging
heat with the system. The kinetic temperature may look stable while the model
is producing wrong structural statistics or wrong dynamics. A thermostat is not
a repair mechanism for an invalid force model.

## What Should the Diagnostic Show?

The full run compares observed position and velocity variances to their
canonical targets. It also compares mean kinetic energy to the 0.5 kT target
for one degree of freedom. Finally, it reports the position integrated
autocorrelation time to show that stronger coupling can change dynamical
memory.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post04_thermostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Thermostat diagnostics for the committed full profile. The BAOAB Langevin oscillator cases sample near the same canonical moment targets while strong coupling substantially increases position autocorrelation time; the compact argon panel checks many-body kinetic-temperature response." %}

The figure has a specific scope. It is moment-focused and memory-focused. It
does not yet show the full kinetic-energy distribution. The argon panel is a
compact reduced-unit response check, not a final production calculation. That
is acceptable for the hidden draft because the prose makes moment-level and
temperature-response claims. If the final article makes stronger
distribution-shape claims, the review note already flags a possible
kinetic-energy histogram or empirical CDF as future work.

The most important panel is the autocorrelation panel. The strong-coupling run
has position integrated autocorrelation time about 52.7, while the weak and
moderate cases are about 10.1 and 12.7. Strong coupling is not simply "more
temperature control." It changes how slowly the position samples decorrelate.
The effective sample count drops from roughly 348 in the weak case and 275 in
the moderate case to about 66 in the strong case.

## How Does Coupling Distort Dynamics?

Langevin friction damps velocity memory. At small friction, the system retains
more Hamiltonian-like motion and exchanges heat slowly. At large friction,
momenta are randomized more aggressively. That can help sampling in some
coordinates, but it can also slow configurational exploration or distort
time-dependent observables.

The full diagnostic shows this split. The strong-coupling case has velocity
lag-1 autocorrelation about 0.355, much lower than about 0.958 for the weak
case and 0.803 for the moderate case. Velocity memory is strongly damped. Yet
the position autocorrelation time is much larger for strong coupling. The
system does not simply become "better mixed" because the thermostat is stronger.

This is familiar from stochastic dynamics. In the overdamped limit, position
motion can become diffusive and slow even though velocities decorrelate quickly.
For molecular observables, this matters. A thermostat may be acceptable for
equilibrating or sampling static structure but inappropriate for measuring
real-time dynamics.

The practical rule is to align thermostat choice with the observable. If the
goal is canonical sampling of static quantities, a thermostat can be part of
the production ensemble. If the goal is a dynamical observable, a common
workflow is to equilibrate with a thermostat, then switch to NVE production
after checking energy behavior. The exact decision depends on the observable,
system, and acceptable bias.

## When Should One Switch to NVE?

Switching from thermostatted equilibration to NVE production is not a ritual.
It is a way to separate ensemble preparation from dynamics measurement. The
thermostat helps bring the system to the desired temperature distribution.
The NVE production run then tests whether the resulting trajectory evolves
without artificial heat exchange.

This switch is most important for time-correlation functions, transport
coefficients, vibrational spectra, and any observable where physical dynamics
matter. If the thermostat remains on, the measured dynamics include the
thermostat's friction and noise. That may be the intended stochastic dynamics,
but it should not be silently reported as unthermostatted Newtonian dynamics.

The switch is not automatic. Before trusting NVE production, one needs an
energy diagnostic like the earlier posts describe: bounded energy behavior,
acceptable normalized drift, and no obvious instability. If the NVE trajectory
drifts badly after thermostat removal, the thermostat was not the real problem;
it was hiding a timestep, precision, force, or model issue.

For this hidden draft, a compact argon Langevin thermostat diagnostic is now
included. The final GPU kUPS production pass remains open. That final pass
should show how the oscillator lesson transfers to the target argon trajectory
family at production settings: warmup under a thermostat, moment checks, a
coupling sweep if needed, and an NVE handoff check for dynamics.

## What Does This Mean for Enhanced Sampling?

Thermostats also appear inside enhanced-sampling workflows. Umbrella sampling,
metadynamics, steered dynamics, and replica methods often use thermostatted
dynamics while also adding bias forces or exchanging configurations. In those
settings, the thermostat is part of the sampling algorithm.

That makes documentation more important. A biased simulation should report the
thermostat, friction or coupling time, timestep, random seed policy, warmup,
and whether samples are decorrelated enough for the estimator. If a thermostat
changes autocorrelation, it changes effective sample size. If it changes
dynamics, it can affect nonequilibrium work distributions or path-dependent
observables.

The later free-energy posts will return to this point. A thermostat can help
maintain a target temperature in biased windows, but it does not guarantee
overlap, independence, or estimator validity. Moment checks and autocorrelation
checks remain part of the evidence chain.

## What Should Be Recorded?

The compact summary for this page records the seed, timestep, sample interval,
target temperature, gamma values, observed means, variances, kinetic energies,
autocorrelations, integrated autocorrelation times, and effective sample
counts. That is the right shape for a thermostat diagnostic: it records both
sampling targets and dynamical memory.

The manifest records the configuration path, config hash, lock hash, git
revision, runtime device, precision policy, Python version, kUPS version, and
NumPy version. For stochastic dynamics, the seed and software versions matter.
Different random streams or numerical kernels can produce different finite-run
statistics even when the target distribution is the same.

For a production atomistic thermostat report, the same record should include
system size, density or cell, initialization source, warmup length, thermostat
method, coupling constants, timestep, sample interval, removed degrees of
freedom, constraints, precision policy, and whether the production segment is
thermostatted or NVE. Without those fields, a temperature number is not enough
to reproduce the simulation.

## How Do Thermostat Families Differ?

Different thermostat families encode different assumptions. Langevin dynamics
adds stochastic noise and friction. Stochastic velocity rescaling controls the
kinetic-energy distribution while preserving a global rescaling structure.
Nosé-Hoover methods introduce extended variables that can produce deterministic
canonical sampling when the dynamics are sufficiently ergodic. Andersen-style
methods randomize velocities through collision-like events. Local thermostats
and global thermostats act differently on collective motion.

These differences matter because the thermostat is not an interchangeable
accessory. A global velocity-rescaling thermostat can maintain the kinetic
energy while leaving some slow modes poorly sampled. A local Langevin
thermostat can damp hydrodynamic behavior. A Nosé-Hoover chain can preserve
more deterministic structure but may require care with ergodicity and chain
parameters. A strongly damped Langevin thermostat can be excellent for
relaxation and poor for dynamics.

The current diagnostic uses BAOAB Langevin because its stochastic splitting is
easy to explain and because the harmonic oscillator has clear canonical
targets. That does not make it the only thermostat worth using. It makes it a
good first microscope for separating moment checks from memory checks. Later
production workflows should justify the thermostat family in terms of the
observable, not in terms of habit.

For MLIP simulations, the thermostat family can also interact with model
failure. Strong stochastic coupling may prevent immediate blow-up by removing
excess kinetic energy, but that can mask a force model that would drift in NVE.
Weak coupling may reveal instabilities sooner but equilibrate slowly. Extended
system thermostats may behave poorly if the potential has rough or noisy
forces. The thermostat choice should therefore be part of model deployment
validation.

## What Would a Stronger Sampling Check Add?

Moment checks are a first pass, not a full distributional proof. For a harmonic
oscillator, the canonical position and velocity distributions are Gaussian.
Checking means and variances is informative, but a stronger diagnostic could
also compare histograms, empirical CDFs, quantiles, or kinetic-energy
distributions. The review note already flags a possible kinetic-energy
histogram if the final article makes stronger distribution-shape claims.

For many-body systems, the analogous checks are broader. One might inspect
kinetic-energy distributions, temperature distributions by degrees of freedom,
RDFs, coordination statistics, potential-energy histograms, pressure
fluctuations, or conserved quantities after thermostat removal. The correct
check depends on the claimed ensemble and observable. There is no universal
single thermostat plot.

Autocorrelation should also be part of sampling validation. If the position
integrated autocorrelation time is about 52.7, as in the strong-coupling case
here, then 3500 saved samples are not 3500 independent samples. Effective
sample size is the bridge between thermostat diagnostics and uncertainty
reporting. A thermostat can produce reasonable moments while leaving too few
independent samples for a precise estimate.

This is why trajectory-length diagnostics come later in the curriculum. The
thermostat determines part of the Markov chain's memory. The trajectory-length
post asks how much data are enough given that memory. The observable post then
asks which estimators are valid for the stored trajectory.

## What Are Common Thermostat Failure Modes?

One failure mode is wrong temperature control. The kinetic energy can be biased
because the noise/friction relation is wrong, degrees of freedom are counted
incorrectly, constraints are mishandled, or center-of-mass motion is included
or removed inconsistently. This is the obvious failure mode, and it is still
worth checking.

A second failure mode is correct kinetic energy but wrong configuration
statistics. This can happen if the integrator splitting, timestep, or coupling
interacts badly with the potential. In a simple oscillator, position variance
is the configurational check. In a molecular system, one needs structural or
energy-distribution checks.

A third failure mode is dynamical distortion. The thermostat samples plausible
static moments, but diffusion, velocity autocorrelation, vibrational spectra,
or reaction rates are no longer those of the intended physical dynamics. This
failure is easy to miss if the report only shows temperature and potential
energy.

A fourth failure mode is masking. The thermostat hides energy drift that would
appear in NVE. This is particularly relevant for MLIPs. If a learned potential
injects energy because of force noise or extrapolation, a thermostat can
continually remove that energy. The simulation may look temperature-stable
while the model is unreliable. An NVE handoff is a useful way to expose that
case.

## How Should Replicas Be Used?

Stochastic thermostats make replica design explicit. If the protocol is fixed,
changing the random seed creates another realization of the same stochastic
dynamics. That is useful for estimating uncertainty. If the thermostat
parameters, warmup length, timestep, or initialization also change, then the
replicas are no longer only stochastic replicas; they are protocol variants.

For a thermostat study, a small replica set can separate finite-sample noise
from systematic coupling effects. If one gamma value has a high autocorrelation
time in every replica, that is a protocol property. If a moment is off in one
short run but not in longer or independent runs, it may be sampling noise. The
current compact tutorial uses one seeded run per coupling strength for a fast
executable diagnostic, but a final production analysis should broaden this
when numerical claims need uncertainty.

Replicas also help avoid over-tuning. It is easy to choose a thermostat
parameter because one run looked good. A robust choice should survive different
seeds, reasonable warmup changes, and shorter timestep checks. This becomes
more important in enhanced sampling, where biased trajectories and thermostat
noise interact with estimator variance.

## What Should Not Be Inferred?

This page does not prove that the final argon thermostat workflow is complete.
The oscillator has known targets and cheap sampling; the compact argon panel
adds a many-body kinetic-temperature response check but still has finite-size,
short-trajectory, and reduced-unit limitations. The review note keeps the
larger GPU kUPS production thermostat diagnostic as a final-release blocker.

It also does not claim that BAOAB Langevin is always the right thermostat.
Different thermostats have different strengths. CSVR, Nosé-Hoover chains,
Langevin variants, Andersen-like collisions, and local thermostats can all be
appropriate or inappropriate depending on the task. The article's claim is
narrower: thermostat diagnostics should distinguish canonical moment checks
from dynamical distortion.

Finally, this page does not replace uncertainty analysis. Autocorrelation
reduces effective sample size. If samples are correlated, 3500 saved points are
not 3500 independent samples. The strong-coupling case makes that clear. Later
posts on trajectory length and observables will turn this into a broader error
bar workflow.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 04 --profile smoke
uv run kups-tutorial verify 04 --profile smoke
uv run kups-tutorial run 04 --profile full
uv run kups-tutorial verify 04 --profile full
uv run jupyter execute notebooks/post-04-thermostats.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, thermostat diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. It is a substantially expanded hidden draft
that remains outside site navigation while the rest of the series is brought to
the same standard. The implemented pieces are:

- smoke and full controlled BAOAB Langevin workflows
- compact argon Langevin reduced-unit temperature-response workflow
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback
- expanded prose separating canonical moment checks, coupling strength,
  autocorrelation, compact argon temperature response, thermostat-induced
  dynamical distortion, NVE handoff, and final-release limitations

The remaining non-publication pieces are:

- rendered desktop and mobile page snapshots for this updated expanded draft
- larger GPU kUPS production thermostat and NVE handoff diagnostic before
  treating this post as final
- final all-post consistency pass once the other articles are expanded
- final rendered desktop and mobile page snapshots after that consistency pass
- public indexing decision after the series is ready as a unit

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-leimkuhler2013"></span>Leimkuhler, B. & Matthews, C. (2013). Rational construction of stochastic numerical methods for molecular sampling. *Applied Mathematics Research eXpress*, 2013(1), 34-56.
- <span id="ref-bussi2007"></span>Bussi, G., Donadio, D. & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *Journal of Chemical Physics*, 126, 014101.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
