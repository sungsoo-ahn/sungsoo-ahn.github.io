---
layout: post
permalink: /kups-md-tutorials/post-05-barostats/
title: "How Should Pressure and Cell Degrees of Freedom Be Coupled?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible pressure and cell diagnostic for molecular dynamics: NPT-like volume fluctuations, compressibility, pressure variance, barostat time constants, and cell memory."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 5
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 5
categories: [science]
tags: [molecular-dynamics, npt, pressure, barostat, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post extends the thermostat discussion to pressure and cell degrees of freedom, using a controlled scalar model plus compact reduced-unit argon cell-response and moving-cell checks before the final kUPS production NPT article is added. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Pressure control is not the same kind of operation as setting a scalar
temperature. In small molecular simulations, instantaneous pressure fluctuates
strongly, volume fluctuations encode compressibility, and the barostat time
constant changes how quickly the cell explores those fluctuations.

For ML researchers, the practical question is not whether one pressure number
looks close to a target. It is whether the volume distribution has the right
scale, whether pressure fluctuations are interpreted statistically, and whether
the cell-coupling dynamics distort the observables that will be measured later.

This draft demonstrates the executable slice of the fifth tutorial with a
controlled scalar-volume model, a reduced-unit argon pressure-volume sweep,
and a compact isotropic moving-cell argon check. The model is not the final
production NPT workflow; it is a microscope for fluctuation targets, barostat
memory, virial pressure signs, and density relaxation under a moving box. That
scope matters. A scalar stochastic cell coordinate cannot teach every detail of
anisotropic stress, elastic response, or cell-shape coupling. The compact
argon moving-cell check cannot replace a kUPS production NPT trajectory.
Together they make three first failure modes visible: confusing pressure
control with pressure clamping, forgetting to validate pressure response under
cell scaling, and treating a moving cell as production-ready before its
relaxation and effective samples have been reviewed.

The intended reader already knows what forces, velocities, temperature, and
MLIP-driven trajectories are. The missing pieces are usually more operational:
what a barostat is allowed to fluctuate, which diagnostic should be checked,
how pressure variance should be interpreted, and why a slow cell variable can
make a trajectory look long while providing few effective samples. Those are
the issues this page emphasizes.

In the constant-pressure ensemble, the simulation cell is part of the sampled
state. The box is not merely a container that adjusts until the pressure number
looks right. Its fluctuations are thermodynamic observables. If the system is
too small, if the compressibility is wrong, if the coupling time is too short
or too long, or if the cell is constrained in a way that does not match the
physical question, the sampled ensemble can be wrong even when the mean
pressure is close to the target.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-05/full.json)
- [barostat notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-05-barostats.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/smoke/barostat_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/barostat_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-05/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-05.md)

## What Is a Barostat Actually Doing?

An NVT trajectory samples configurations at fixed particle number, volume, and
temperature. An NPT trajectory samples configurations at fixed particle number,
pressure, and temperature. That replacement of volume by pressure is not a
minor bookkeeping change. It introduces cell degrees of freedom, changes the
equilibrium distribution, and changes how one should interpret the trajectory.

At a practical level, a barostat supplies equations of motion or stochastic
updates for the simulation cell. In an isotropic barostat, the scalar volume or
single box scale factor changes. In a semi-isotropic barostat, different groups
of directions can be coupled separately. In a fully flexible-cell barostat,
cell shape can also fluctuate, so the cell matrix carries shear-like degrees
of freedom. These choices are physical assumptions, not only numerical
options.

For a liquid, isotropic volume changes are often a reasonable starting point.
For a membrane, one might need semi-isotropic coupling because the normal
direction and in-plane directions represent different mechanical constraints.
For a crystal under stress, a flexible cell may be necessary if the question
involves lattice constants, elastic response, or shear relaxation. The wrong
cell constraint can suppress exactly the relaxation mode one wants to study.

Pressure is also not a smooth observable in atomistic simulation. The virial
pressure contains kinetic and configurational terms, and instantaneous values
are noisy in small boxes. A single snapshot pressure can be far from the target
without indicating failure. A useful NPT diagnostic therefore looks at the
distribution: the mean pressure, volume mean, volume variance, pressure
variance, and cell autocorrelation.

This is why the phrase "set pressure to 1 bar" is incomplete. A production
protocol should say which ensemble is intended, which cell degrees of freedom
are active, how the thermostat and barostat are coupled, what coupling times
are used, how long equilibration lasts, what data are discarded as warmup, and
which fluctuation checks passed.

## What Should Fluctuate?

The current diagnostic fixes a target pressure, volume, compressibility, and
temperature, then changes only the scalar barostat relaxation time:

| Choice | Full value | Why it matters |
|---|---:|---|
| target pressure | 1.0 | pressure mean for the controlled model |
| equilibrium volume | 1000 | reference cell size |
| compressibility | 0.01 | sets volume fluctuation scale |
| temperature | 1.0 | dimensionless kT |
| relaxation times | 0.5, 2.0, 8.0 | fast, moderate, and slow cell memory |
| samples | 2500 per run | enough for compact fluctuation checks |

The same configuration also carries a compact atomistic check:

| Choice | Full value | Why it matters |
|---|---:|---|
| argon cell | 3x3x3 FCC conventional repeats | 108-atom reduced-unit crystal |
| reference density | 1.0 | compressed enough to show a clear pressure response |
| volume factors | 0.90 to 1.10 | affine compression and expansion around V0 |
| pressure model | Lennard-Jones virial | tests coordinates, PBC, cell scaling, and pressure signs |

In the NPT ensemble, volume variance is proportional to compressibility. A
barostat that suppresses this variance is not merely "more stable"; it may be
sampling the wrong ensemble.

For an isotropic scalar-volume model, the useful relation is that the volume
variance should scale with kT times the isothermal compressibility times the
mean volume. The constants and units depend on the convention used by the
model, but the diagnostic idea does not: compressible systems should fluctuate
more in volume than stiff systems, larger systems should have larger absolute
volume variance but smaller relative volume fluctuations, and finite
trajectories should show sampling uncertainty around the target variance.

The committed full profile sets the expected volume variance to 10.0 and the
expected pressure variance to 0.1 in dimensionless units. These are not meant
as argon values. They are controlled targets. The scalar model is constructed
so that the correct answer is known before the run begins. That lets the page
ask whether the diagnostic code can recover the intended fluctuation scale.

The full-profile results are compact but informative:

| Barostat | Relaxation time | Volume mean | Volume variance | Pressure mean | Pressure variance |
|---|---:|---:|---:|---:|---:|
| fast | 0.5 | 999.676 | 9.788 | 1.032 | 0.0979 |
| moderate | 2.0 | 1000.099 | 8.606 | 0.990 | 0.0861 |
| slow | 8.0 | 1000.439 | 8.520 | 0.956 | 0.0852 |

The volume means are close to the reference volume of 1000. The pressure means
are near the target pressure of 1.0. The variance estimates are within about
15 percent of the analytical targets in this finite run. That is the level of
claim this executable slice supports.

Notice what it does not claim. It does not say pressure should be constant. It
does not say that a real material with an MLIP will have this compressibility.
It does not say that the cell matrix in a fully flexible simulation is
validated. It says the diagnostic machinery can check known fluctuation targets
and expose how coupling time changes memory.

## Why Is Pressure So Noisy?

Many first NPT workflows are derailed by the same visual pattern: the pressure
trace jumps wildly, so the user tightens the barostat until the trace looks
less alarming. That instinct can be counterproductive. In a microscopic system,
instantaneous pressure is expected to fluctuate strongly. The time average and
the fluctuation statistics are the quantities to interpret.

The pressure estimator is noisy because it depends on momenta, forces, and
pair or many-body virial terms. In an MLIP simulation, it also depends on how
the model predicts stress or how stress is derived from the learned energy.
Small force or virial errors can show up strongly in pressure. A stable
temperature trace does not guarantee a reliable pressure trace.

There is also a finite-size issue. Pressure fluctuations become easier to
average as the system grows, but small tutorial boxes are deliberately small.
They are useful for speed and debugging, not for claiming high-precision
equations of state. A pressure diagnostic should therefore state system size,
trajectory length, sampling interval, equilibration discard, and uncertainty
or at least effective sample count.

For the scalar part of this page, the controlled model avoids the atomistic
virial entirely. That is an advantage for teaching the diagnostic and a
limitation for production science. The compact argon sweep adds a direct virial
pressure check under affine cell scaling. The new moving-cell check then starts
from a compressed reduced-unit argon cell and lets the isotropic volume factor
relax under a noisy pressure-feedback update. It is a useful review harness for
cell motion, density relaxation, pressure mean, and volume effective samples,
but it still does not replace a final kUPS production NPT trajectory.

## What Does the Relaxation Time Change?

The relaxation time controls how quickly the cell variable responds. A very
short relaxation time can make the cell chase instantaneous noise. A very long
relaxation time can make the cell change so slowly that the trajectory has few
independent volume samples. Both failures can pass a superficial "mean
pressure near target" check.

The scalar diagnostic is designed to isolate this point. The target
distribution is the same for all three barostat settings. Only the relaxation
time changes. If the model is behaving correctly, the long-run mean and
variance targets should be shared, while the autocorrelation structure should
change.

That is what the full results show:

| Barostat | Lag-1 volume autocorrelation | Integrated autocorrelation time | Effective volume samples |
|---|---:|---:|---:|
| fast | 0.338 | 1.991 | 1255.7 |
| moderate | 0.761 | 7.247 | 345.0 |
| slow | 0.925 | 22.531 | 111.0 |

All three runs store 2500 samples, but the slow barostat provides only about
111 effective volume samples by this estimator. The files are the same size.
The trajectory lengths are the same. The statistical content is not the same.

This is the main lesson for production workflows. A slow barostat can look
gentle and stable while producing a long-memory trajectory. If the observable
depends on volume or density, that memory changes how much independent
information has actually been collected. The barostat time constant therefore
belongs in the methods section and in the review checklist.

The opposite limit is also important. A barostat that is too aggressive can
drive unphysical cell oscillations, amplify noisy pressure estimates, or couple
badly to the thermostat and integrator. The current scalar model does not
attempt to demonstrate every instability mode. It gives a minimal,
reproducible place to see why coupling strength is a statistical design
choice.

## How Should a Simulation Be Initialized Before NPT?

NPT is usually not the first step in a reliable MD workflow. A practical
initialization sequence separates geometry cleanup, temperature preparation,
volume relaxation, and production sampling. The exact order depends on the
system, but the logic is consistent.

First, remove severe overlaps or unreasonable starting geometries. A barostat
should not be asked to fix a bad structure. If forces are enormous because the
initial geometry is pathological, changing the cell volume can create new
pathologies rather than repair the old ones. Minimization or a conservative
short warmup is often needed before any pressure coupling is trusted.

Second, initialize velocities for the target temperature and remove net
translation if appropriate. The thermostat can then equilibrate kinetic
degrees of freedom in a fixed cell. This NVT step lets one check whether the
timestep, precision, and force model are stable before the box itself is
allowed to move.

Third, turn on pressure coupling for equilibration, not immediately for
measurement. The density and cell shape may need time to relax. Early parts of
an NPT trajectory can be dominated by the initial volume guess. Those frames
should usually be discarded from production averages.

Fourth, decide what production ensemble is actually needed. If the target
observable is an equation-of-state quantity, density, thermal expansion, or
pressure-dependent structure, NPT production may be the right choice. If the
target observable is a dynamical quantity, one may equilibrate with NPT, fix
the equilibrated volume, then collect NVE or weakly thermostatted production
depending on the scientific question.

For MLIP work, there is an additional model-validity check. The pressure or
stress behavior should be inside the model's training domain. If a learned
potential was trained mostly near fixed-volume configurations, a flexible-cell
NPT run can explore strained states where the model is less reliable. The
barostat may then expose model errors rather than ordinary thermodynamic
fluctuations.

## Which Cell Degrees of Freedom Should Be Coupled?

The scalar model in this draft corresponds most closely to isotropic volume
coupling. That is the simplest case: the box expands or contracts uniformly.
It is appropriate for some liquids and cubic systems when shape fluctuations
are not part of the question. It is not a universal default.

Anisotropic and flexible-cell barostats introduce more choices. The cell
matrix can change lengths independently, and in fully flexible schemes it can
also change angles. These extra degrees of freedom are necessary for some
solids, interfaces, and stressed systems, but they can also introduce rotation,
shear, or shape instabilities if used without care.

The important question is what mechanical boundary condition the simulation is
supposed to represent. If the material is a bulk isotropic liquid, suppressing
cell-shape fluctuations may be fine. If the material is a crystal under
external stress, suppressing shear relaxation can bias the result. If the
material is a slab or membrane, coupling the normal and lateral directions
together can impose an unphysical constraint.

A good tutorial should therefore avoid saying "use NPT" as though that is a
complete method. It should say whether the cell is isotropic, semi-isotropic,
anisotropic diagonal, or fully flexible; whether the target is a scalar
pressure or stress tensor; and whether cell angles are allowed to change. This
page is a first step toward that vocabulary, not the final treatment.

The committed argon diagnostics now add actual atomistic coordinates, periodic
boundaries, affine cell scaling, and a compact moving-cell trajectory. The full
profile uses a 108-atom reduced-unit FCC argon cell, scales the volume from
0.90 to 1.10 times the reference volume for the static response check, and
starts three moving-cell replicas from `V/V0 = 0.90`. In the current full run,
the fitted reduced-unit bulk response is about 42.1, the pressure span is about
8.7, the moving-cell mean pressure is about 0.925 +/- 0.005 against a target of
1.0, the mean kinetic temperature is 0.699 against a target of 0.70, and the
volume-factor effective sample count is about 96. The maximum absolute
replica-level total-energy change across the sampled moving-cell traces is
about 0.121 per atom. This is a useful moving-cell wiring and relaxation
check, not a final kUPS production NPT result.

The final kUPS production diagnostic should add real atomistic NPT dynamics
with documented thermostat/barostat settings, timestep, warmup, and sampling
interval. A minimal production version could compare NVT at fixed density with
isotropic NPT density relaxation, then check pressure mean, density mean,
volume fluctuations, and energy/temperature behavior. A more advanced version
could later add flexible-cell solids, but that is beyond what the current
committed workflow supports.

## How Should Thermostat and Barostat Choices Interact?

Constant-pressure simulation also needs temperature control. The barostat and
thermostat are not independent decorations; together they define the extended
or stochastic dynamics. Changing one can affect the behavior of the other.

In many methods, the thermostat controls particle momenta while the barostat
controls volume or cell momenta. Some schemes also thermostat the barostat
variables. The numerical splitting, timestep, and coupling constants determine
how energy flows between physical and cell degrees of freedom. If the
thermostat is too aggressive, it can mask integration or force-model problems.
If the barostat is too aggressive, it can inject cell noise that the thermostat
then removes.

The simple review habit is to check NVT before NPT. If NVT does not sample or
conserve as expected, NPT will be harder to interpret. Once NPT is enabled,
check temperature, pressure, volume, and energy-like diagnostics together. A
single scalar temperature or pressure mean is too weak.

The choice of timestep also matters. A timestep that is acceptable in NVT may
still be risky in NPT if cell motion changes neighbor distances or stress
response rapidly. For MLIPs, neighbor lists, cutoff behavior, and stress
calculation details can become visible when the box changes. Reproducible NPT
workflows should therefore record the timestep, neighbor-list settings when
relevant, precision policy, thermostat parameters, and barostat parameters.

## What Should The Diagnostic Show?

The full run checks two fluctuation targets: volume variance and pressure
variance. It also reports the integrated autocorrelation time of the scalar
volume process. The slow barostat has the same target distribution but much
longer memory, which means fewer effective samples for the same wall-clock
trajectory length.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post05_barostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Pressure and scalar-cell diagnostics for the committed full profile. The controlled NPT-like model recovers volume and pressure fluctuation scales, slower barostat coupling increases cell memory, and the compact argon panel now checks three reduced-unit moving-cell replicas with volume uncertainty, kinetic temperature, pressure SEM, and effective-sample annotations." %}

The figure has four roles. The volume panel checks whether the scalar cell
samples the expected fluctuation scale. The pressure panel checks the
corresponding pressure variance in the controlled model. The memory panel
shows why a trajectory with the same number of stored frames can contain very
different amounts of independent information. The argon panel checks that a
compressed reduced-unit cell can move, relax its density, report replica-level
pressure uncertainty, keep the kinetic temperature near the target, and still
expose the effective number of independent volume-factor samples.

The target lines in the variance panels are essential. Without them, the
figure would be only a comparison among three arbitrary runs. With them, the
diagnostic asks whether the simulation recovers a known ensemble property. In
a real atomistic workflow, the target may not always be known from first
principles, but the same style of thinking applies: compare against
compressibility estimates, equations of state, independent references, or
larger-system checks when available.

The visual result also shows why a barostat diagnostic should not rely on a
single time trace. A pressure trace may look noisy even when the variance is
correct. A volume trace may look smooth because the barostat is slow, not
because the ensemble is better. The summary statistics and the figure must be
read together.

## How Would This Extend to Atomistic Argon?

The compact argon panel returns to the atomistic thread used in the earlier
tutorials, now with both a static cell-response diagnostic and a reduced-unit
moving-cell trajectory. A production diagnostic would start from a validated
initialized structure, run NVT equilibration at fixed volume, then run
isotropic NPT with a documented thermostat, barostat, timestep, warmup, and
sampling interval.

The review should include at least these checks:

| Check | Why it belongs in the review |
|---|---|
| density or volume mean | verifies the relaxed state is plausible |
| pressure mean with uncertainty | avoids over-interpreting noisy instantaneous pressure |
| volume variance | checks whether compressibility-scale fluctuations are plausible |
| temperature distribution | catches thermostat/barostat coupling failures |
| energy behavior | catches hidden integration or model problems |
| cell autocorrelation | estimates how much independent density sampling exists |
| snapshot rendering | verifies the public figure and prose match the data |

For MLIP-driven argon or aluminum, stress validation should also be explicit.
If the model does not provide reliable virials or stress, an NPT result is not
trustworthy merely because the trajectory runs. The training data, stress
labels, finite-difference checks, or comparison to a reference potential may
be needed before pressure-dependent conclusions are credible.

That final dynamic diagnostic is intentionally not invented in this draft. The
current page reports committed scalar-model outputs and a committed static
argon pressure-volume sweep. The open item remains open because the review
standard for the series is that the article, data, figure, and rendered page
must all describe the same executed workflow.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 05 --profile smoke
uv run kups-tutorial verify 05 --profile smoke
uv run kups-tutorial run 05 --profile full
uv run kups-tutorial verify 05 --profile full
uv run jupyter execute notebooks/post-05-barostats.ipynb --inplace
uv run python scripts/generate_post05_figures.py
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, scalar barostat diagnostics, and figure generator from
`src/kups_md_tutorials/`. The committed full manifest records the configuration
hash, source Git revision, lockfile hash, Python version, platform, precision
policy, runtime device, and package versions. In the current full run, the
source revision recorded by the manifest is
`ffbf49effecb7ccac823145c58c073e0c8cd731c`, the config hash is
`cc78ccac72341e718683ccef58ee73f3ba2418b8c8c188bb9442f63ee79857fa`, and the
runtime device is CPU.

That provenance is useful because pressure and cell diagnostics are sensitive
to small protocol changes. A different seed, timestep, sampling interval,
warmup length, precision policy, or implementation of the stochastic update can
change the finite-run estimates. The review note records what was checked so
the article can be revised when the final atomistic workflow is added.

## Practical Checklist

Before treating an NPT run as production data, the following questions should
have concrete answers:

| Question | A weak answer | A stronger answer |
|---|---|---|
| What cell can move? | "NPT was used." | Isotropic, semi-isotropic, anisotropic, or flexible cell is specified. |
| What pressure is targeted? | "1 bar." | Units, tensor/scalar target, and coupling scheme are stated. |
| What is equilibrated? | "The pressure stabilized." | Warmup discard, density relaxation, and pressure uncertainty are reported. |
| What fluctuates? | "The box changes." | Volume mean, variance, and autocorrelation are checked. |
| What is measured? | "Averages after NPT." | Observable-specific ensemble choice is justified. |
| What is reproducible? | "The notebook runs." | Configs, summaries, manifests, figures, and page snapshots are reviewed. |

This checklist is especially important for ML researchers using MD as an
experimental substrate. A learned potential can make it easy to run long
trajectories, but long trajectories do not compensate for ambiguous ensemble
definition. The more automated the workflow becomes, the more explicit the
ensemble and diagnostic metadata should be.

The current scalar diagnostic is small enough to rerun quickly. That is
intentional. The page is not trying to replace production NPT simulation; it is
trying to make the review habits executable before the workflow becomes more
complex.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled scalar barostat workflows
- compact reduced-unit argon pressure-volume response workflow
- compact reduced-unit argon moving-cell density-relaxation workflow with
  three full-profile replicas, kinetic-temperature samples, and energy-like
  trace diagnostics
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final kUPS production NPT dynamics diagnostics with full atomistic
  thermostat/barostat settings, GPU provenance, and production stress/cell
  checks
- citations for NPT ensemble fluctuations, compressibility, barostat coupling,
  and finite-size pressure fluctuations
- rendered desktop and mobile page snapshots after the final production NPT
  diagnostic is added
- final consistency pass after the production dynamics diagnostic is added

The rule for this series is simple: a result is not ready because the code ran.
It is ready only after the code, data, figure, prose, and rendered page have
all been reviewed against the same reproducibility contract.

## References

- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-martyna1994"></span>Martyna, G. J., Tobias, D. J. & Klein, M. L. (1994). Constant pressure molecular dynamics algorithms. *Journal of Chemical Physics*, 101, 4177-4189.
- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
