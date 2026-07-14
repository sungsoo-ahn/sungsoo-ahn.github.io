---
layout: post
permalink: /kups-md-tutorials/post-07-observables/
title: "How Do Trajectories Become Physical Observables?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible observable-estimator diagnostic for molecular dynamics: RDF normalization, coordination integrals, finite-size support, uncertainty, and velocity autocorrelation functions."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 7
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 7
categories: [science]
tags: [molecular-dynamics, rdf, observables, correlation-functions, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the trajectory-length discussion by asking how sampled configurations and velocities become physical observables with normalization, finite-size limits, and uncertainty. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A trajectory is not an observable by itself. It is a source of samples from
which an observable is estimated. That distinction matters because every
observable has a definition, normalization, finite-size support, and uncertainty
model.

For ML researchers working with MLIPs, the failure mode is familiar: a
simulation produces many frames, a plotting function produces a smooth curve,
and the curve is treated as a physical result. The practical question is
different. What estimator was used? Which samples enter it? What finite-size
region is valid? What error bar belongs on the derived number?

This draft demonstrates the executable slice of the seventh tutorial with
seeded periodic argon FCC cells. It computes radial distribution functions,
coordination integrals, block uncertainties, and a velocity autocorrelation
function before the final argon/kUPS trajectory diagnostic is added.

The target reader already knows that MD produces positions and velocities.
The missing skill is turning those arrays into quantities with physical
meaning. That conversion is not automatic. It requires a definition of the
observable, a normalization convention, a domain where the estimator is valid,
and an uncertainty model. The same trajectory can support a structural claim,
fail to support a dynamical claim, and be ambiguous for a finite-size-sensitive
quantity.

This page uses displaced periodic FCC argon cells rather than a production
liquid trajectory. That scope is deliberate. The controlled structure makes
normalization and finite-size support easy to see. It also keeps the page from
pretending that a polished curve is already a production measurement. The
final version must still add an actual argon/kUPS trajectory observable
diagnostic before any liquid-like or dynamical MD claim is final.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-07/full.json)
- [observable notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-07-observables.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/smoke/observable_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/observable_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-07/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-07.md)

## What Is an Observable?

An observable is a function of the microscopic state, or sometimes a function
of a sequence of states. Potential energy is a function of positions. Kinetic
temperature is a function of velocities. Density is a function of particle
number and cell volume. The radial distribution function is a normalized
estimator built from pair distances. A velocity autocorrelation function is a
time-correlation estimator built from velocities at separated times.

That definition sounds formal, but it prevents a common mistake. A trajectory
file is not itself a result. It is a collection of samples. To get a result,
one must choose an estimator and then ask whether the samples are appropriate
for that estimator. If the estimator is normalized incorrectly, if invalid
finite-size regions are included, or if uncertainty is ignored, the result can
look smooth and still be wrong.

For MLIP users, this matters because the model often sits upstream of a
scientific observable. A force MAE or energy RMSE is rarely the final
quantity. The final quantity might be a structure factor, coordination number,
diffusion coefficient, density, free-energy difference, or reaction rate. The
model can look accurate on static validation data while the observable
estimator is biased by sampling, finite-size effects, or analysis choices.

This post therefore shifts the emphasis from "run MD" to "define the
measurement." A reproducible observable report should state:

| Item | Why it matters |
|---|---|
| microscopic quantity | positions, velocities, energies, virials, or derived coordinates |
| estimator | the formula applied to finite samples |
| normalization | units and reference density or probability measure |
| valid support | finite-size and boundary-condition limits |
| uncertainty | block, replica, bootstrap, or model-aware error estimate |
| provenance | configuration, seed, code path, manifest, and figure source |

The current diagnostic is built around three observable types: an RDF as a
normalized structural estimator, a coordination number as an integral derived
from the RDF, and a VACF as a time-correlation estimator.

## What Is Being Estimated?

The current diagnostic keeps the estimator explicit:

| Choice | Full value | Why it matters |
|---|---:|---|
| number density | 0.021 | RDF and coordination normalization |
| displacement scale | 0.10 | seeded thermal-like structural spread |
| small system | 32 atoms | finite-size stress test |
| large system | 256 atoms | larger radial support and lower block noise |
| RDF bin width | 0.05 | resolution versus noise tradeoff |
| coordination cutoff | 4.6 | first-shell integral boundary |
| VACF max lag | 90 | time-correlation support |

The small-cell RDF is not drawn beyond half the periodic box length. Those
radial shells are not valid for a minimum-image RDF estimator, even if a plotting
routine could produce numbers there.

The full profile compares two periodic FCC cells. The small system has 32 atoms
and box length about 11.51. The large system has 256 atoms and box length about
23.01. Both use the same number density, displacement scale, RDF bin width,
coordination cutoff, and frame count. That makes the system-size comparison an
estimator comparison rather than a change in physical state.

The full summary reports:

| System | Atoms | Box length | First RDF peak radius | First RDF peak value | Coordination |
|---|---:|---:|---:|---:|---:|
| small cell | 32 | 11.507 | 4.075 | 7.757 | 11.9990 |
| large cell | 256 | 23.015 | 4.075 | 7.666 | 11.9988 |

Both systems recover the first-neighbor coordination near 12, as expected for
the displaced FCC construction and cutoff. The figure and summary do not claim
that this is a liquid argon RDF. They claim that the RDF normalization,
coordination integration, finite-size mask, and uncertainty workflow are
functioning on a controlled periodic structure.

## Why Is an RDF Not Just a Histogram?

A raw pair-distance histogram counts how many pairs fall into each radial bin.
An RDF divides that count by the shell volume, density, number of reference
particles, number of frames, and any pair-counting convention. The goal is to
estimate how the local pair density at radius r compares with the bulk density.
For a uniform ideal gas, the RDF should be near 1 away from finite-size and
sampling artifacts. For a structured material, peaks and troughs encode local
order.

The normalization is what turns a count into a physical estimator. Without it,
larger radii may appear to have more neighbors simply because spherical shells
have larger volume. Larger systems may appear different simply because they
contain more pairs. More frames may change the histogram height even when the
underlying structure is unchanged.

A schematic RDF estimator has the form:

$$g(r) \approx \frac{\mathrm{pair\ counts\ in\ shell}}{\mathrm{frames}
\times \mathrm{particles} \times \rho \times 4\pi r^2 \Delta r}.$$

The details depend on whether pairs are counted once or twice, how periodic
boundaries are handled, and how shell volumes are approximated for finite bin
widths. Those details are not cosmetic. They determine whether the curve has
the right units and limiting behavior.

For a finite periodic cell using a minimum-image convention, the RDF should
not be interpreted beyond half the shortest box length. Beyond that distance,
the set of available pair separations is constrained by the periodic geometry,
and the spherical shell is not sampled as if it lived in an infinite isotropic
system. The first implementation for this post drew invalid small-cell shells;
the review corrected that by masking small-cell RDF bins beyond half the box
length before the reviewed figure was committed.

The small cell therefore has less valid radial support than the large cell.
That is not a plotting annoyance. It is a finite-size property of the
estimator. The summary records the finite-size shell fraction as about 1.39 for
the small system and about 0.70 for the large system, making the support issue
explicit.

## How Does an RDF Become a Coordination Number?

The coordination number up to a cutoff is an integral of the RDF against the
spherical shell measure and the number density. In words, it asks how many
neighbors are expected inside a chosen radius around a reference particle. For
a first-shell coordination number, the cutoff is often placed near the first
minimum after the first RDF peak.

In this diagnostic, the cutoff is 4.6 in the same distance units as the
constructed cells. That cutoff includes the first FCC neighbor shell. The full
summary gives coordination numbers of about 11.9990 for the small cell and
11.9988 for the large cell. The values are close to 12 because the controlled
structure is a displaced FCC lattice and the cutoff captures the first
neighbor shell.

The important lesson is not the number 12 by itself. The lesson is that a
derived observable inherits every analysis choice that went into the RDF:
normalization, bin width, finite-size support, frame selection, and uncertainty
estimation. If the RDF is invalid at some radii, the coordination integral
over those radii is also invalid. If the cutoff is moved across a peak or
minimum, the coordination number changes.

The committed workflow estimates block standard errors for the coordination
number. They are very small in this controlled setup: about 0.00042 for the
small cell and 0.000058 for the large cell. Those small uncertainties should
not be generalized to a liquid trajectory. They reflect a controlled displaced
crystal-like construction with seeded frames, not a slow structural relaxation
problem.

For production MD, a coordination report should say how the cutoff was chosen,
whether the RDF minimum is stable across replicas, whether block or replica
uncertainty is reported, and whether the system size supports the radial range
being integrated.

## What Makes a Time-Correlation Observable Different?

A time average of a static observable uses samples at individual times. A
time-correlation function uses pairs of times separated by a lag. The velocity
autocorrelation function, for example, compares velocities at time t with
velocities at time t plus a lag. It asks how quickly velocity memory is lost.

That changes the estimator. Long lags have fewer time origins than short lags.
The noise usually grows with lag. The tail can matter for transport
coefficients, but the tail is also where the estimator can be least stable.
Therefore a time-correlation function should not be treated as just another
smooth curve.

The current VACF diagnostic uses the large cell and a seeded correlated
velocity process with configured correlation time 12.0. The full summary
reports lag-1 autocorrelation about 0.921, first zero crossing at lag 68, and
normalized integral about 12.04. Those values support the limited claim that
the VACF estimator recovers the configured decay scale in this controlled
workflow.

They do not support a diffusion claim for real argon. A diffusion coefficient
from a VACF integral would require a physical velocity trajectory, units,
careful tail treatment, finite-size analysis, and uncertainty on the integral.
The final article should add actual kUPS trajectory observables before making
any physical dynamical claim.

## What Should The Diagnostic Show?

The full run checks three things. The RDF panel shows the normalized pair
estimator rather than a raw distance histogram. The coordination panel turns
that curve into a first-shell integral with a block standard error. The VACF
panel treats time correlation as its own observable, not as a side effect of
the trajectory.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post07_observable_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Observable diagnostics for the committed full profile. The RDF is normalized and finite-size limited, the coordination number carries a block uncertainty, and the velocity autocorrelation function shows how a trajectory becomes a time-correlation estimator." %}

The figure is intentionally estimator-focused. The RDF panel compares the 32-
and 256-atom cells while respecting the small-cell finite-size limit. The
coordination panel shows that the first-shell integral is close to the expected
FCC value and that block uncertainty is part of the reported number. The VACF
panel shows a decaying time-correlation function with a configured memory
scale.

The figure supports a narrow but important mechanism: observables require
analysis definitions. The same stored frames can produce different results if
the RDF normalization is wrong, if invalid radial bins are drawn, if the
coordination cutoff is changed, or if a time-correlation tail is over-read.

## How Should Uncertainty Be Attached?

Uncertainty belongs on derived observables, not only on raw measurements. For
an RDF curve, uncertainty can be estimated by block averages, independent
replicas, bootstrap over blocks, or other correlation-aware methods. For a
coordination number, uncertainty should propagate the variation in the RDF or
be estimated directly from block-level coordination values. For a VACF,
uncertainty depends on lag and time-origin count.

The current diagnostic uses block uncertainty for the coordination number
because that scalar derived value is easy to audit. It does not yet put error
bands on the entire RDF curve or VACF curve. That is acceptable for this hidden
draft because the claims are focused on normalization, finite-size support, and
the first-shell coordination value. A final production article should expand
uncertainty treatment for curves if the curve shape is used to support a
scientific claim.

The uncertainty method should match the sampling problem. If frames are
correlated, treating every frame as independent underestimates uncertainty. If
there are independent replicas, between-replica variation can expose slow modes
or initialization dependence. If a quantity is derived from a cutoff or tail
integral, uncertainty in that analysis choice may be as important as
statistical noise.

For MLIP studies, there is another layer. The observable uncertainty from
finite sampling is not the same as model uncertainty from the learned
potential. A structural observable may be precisely estimated for the wrong
potential. A reproducible analysis should therefore separate sampling
uncertainty from model validation evidence.

## How Do Finite-Size Effects Enter?

Finite-size effects enter observables in several ways. The box limits which
length scales exist. Periodic boundaries impose artificial repetition. Long
wavelength fluctuations may be suppressed. Time-correlation functions and
transport estimates can have system-size-dependent tails. RDFs have limited
radial support under minimum-image analysis.

The small-versus-large comparison in this post is deliberately simple. Both
systems are built from the same periodic FCC motif at the same density. The
small cell has fewer atoms and a shorter half-box distance. The large cell has
more atoms and a longer valid radial interval. The finite-size support
difference is visible even before asking about physical finite-size
corrections.

The lesson for production is to report the support of the estimator. If the
RDF is plotted to a radius larger than half the box length, the analysis should
justify the method used to handle periodic geometry. If a coordination cutoff
approaches the finite-size limit, the result is suspect. If a correlation
function reaches a significant fraction of the trajectory length, the tail may
be dominated by estimator noise.

Finite-size checks do not always require enormous simulations. Sometimes a
small controlled comparison is enough to show whether an observable is
sensitive to box size. Sometimes physics requires larger cells. The point is
to make the scale limitation visible rather than hiding it behind a smooth
plot.

## What Are Common Observable Mistakes?

The most common mistake is treating a plotting routine as the definition of an
observable. A library may produce an RDF, VACF, or coordination curve, but the
scientific result still depends on the assumptions passed to that routine.
Defaults for bin width, cutoff, normalization, wrapping, and time origins are
not universal physical choices.

Another mistake is comparing curves with different estimators. If one RDF uses
a different density, different bin width, different finite-size mask, or
different frame selection, curve differences may reflect analysis choices
rather than material behavior. This is especially risky when comparing an MLIP
trajectory to a reference simulation or experiment. The estimator should be
matched before the model is judged.

A third mistake is reporting derived scalar values without the analysis path.
A coordination number without the RDF, cutoff, and uncertainty is hard to
interpret. A diffusion-like number from a VACF without lag support and tail
treatment is similarly fragile. Derived values are useful because they compress
curves, but compression should not hide the choices that produced them.

The last mistake is using smoothness as a proxy for correctness. Averaging,
interpolation, broad bins, and many frames can all make a figure look clean.
None of them proves that the estimator is normalized, the finite-size support
is valid, or the uncertainty is honest.

## How Would This Extend to kUPS Trajectories?

The final post should replace or augment the displaced-FCC estimator with
observables from an actual argon/kUPS trajectory. A natural extension would use
the initialization, integrator, thermostat, barostat, and trajectory-length
checks from earlier posts, then compute RDF, coordination number, and possibly
VACF on a physically generated trajectory.

That extension should record:

| Artifact | What it should prove |
|---|---|
| trajectory protocol | ensemble, timestep, warmup, sampling interval, and seed |
| RDF estimate | normalized structural observable with valid radial support |
| coordination number | cutoff choice, uncertainty, and replica/block evidence |
| VACF or other time correlation | lag support, tail behavior, and uncertainty |
| finite-size comparison | whether the observable changes with box size |
| model-health checks | whether the kUPS trajectory remains in a credible regime |

The current controlled workflow is still useful after that extension. It acts
as a unit test for the observable-estimator logic. If the production figure
changes, the controlled estimator can remain as a simpler reference for
normalization, support masking, and notebook regeneration.

## What Belongs in the Methods Paragraph?

An observable methods paragraph should be specific enough that another person
can reproduce the estimator, not only the trajectory. For an RDF, that means
reporting the density, bin width, maximum radius, periodic-boundary convention,
whether invalid shells were masked, and the frame set used for averaging. For a
coordination number, it means reporting the cutoff, integration convention, and
uncertainty estimator. For a VACF, it means reporting the velocity source,
normalization, lag range, time-origin handling, and any integration limit.

The methods text should also separate physical interpretation from analysis
mechanics. "The first-shell coordination is near 12" is a physical statement
for this controlled FCC-like structure. "The RDF was masked beyond half the
box length" is an estimator-validity statement. Both are needed. Omitting the
second makes the first harder to trust.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 07 --profile smoke
uv run kups-tutorial verify 07 --profile smoke
uv run kups-tutorial run 07 --profile full
uv run kups-tutorial verify 07 --profile full
uv run jupyter execute notebooks/post-07-observables.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, observable diagnostics, and figure generator from
`src/kups_md_tutorials/`. The committed full manifest records the configuration
hash, source Git revision, lockfile hash, Python version, platform, precision
policy, runtime device, and package versions. For the current full profile, the
configuration hash is
`240a48a5693bdb1390ec17bcc33de0d3ebfca7c48b9a8dd1bcc528c38caa98db`, the
recorded source revision is `ffbf49effecb7ccac823145c58c073e0c8cd731c`, and
the runtime device is CPU.

The compact outputs include the summary JSON plus RDF and VACF sample tables.
Those files are committed so the notebook and website figure can be
regenerated without raw trajectory archives. Raw trajectories remain out of
scope for the repository because the plan commits compact summaries and figure
sources, not bulky intermediate data.

## Practical Checklist

Before accepting an observable from a trajectory, record concrete answers to
these questions:

| Question | Evidence to record |
|---|---|
| What is the observable definition? | Formula, units, and code path. |
| What samples enter the estimator? | Frames, warmup discard, stride, and replicas. |
| What normalization is used? | Density, shell volume, time-origin convention, or reference measure. |
| What is the valid support? | Half-box RDF limit, lag support, cutoff, or finite-size bound. |
| What uncertainty is attached? | Block, replica, bootstrap, or correlation-aware interval. |
| What derived choices matter? | RDF bin width, coordination cutoff, VACF integration range. |
| What does the figure claim? | Caption and plotted quantities must match the summary values. |

The checklist is intentionally stricter than "plot the observable." A
scientific observable is a statistical claim about a system, not a visual
artifact produced by a plotting function.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled argon-FCC observable workflows
- committed compact RDF, VACF, and summary outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- argon/kUPS trajectory diagnostics for physical observables
- citations for RDF normalization, coordination integrals, finite-size effects,
  and time-correlation functions beyond the current starter references
- rendered desktop and mobile page snapshots for this expanded prose
- final consistency pass after the production trajectory-observable diagnostic
  is added

The rule for this post is that an observable is a statistical object. The
trajectory provides samples; the estimator, normalization, finite-size support,
and uncertainty determine what can be claimed from those samples.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-allen1987"></span>Allen, M. P. & Tildesley, D. J. (1987). *Computer Simulation of Liquids*. Oxford University Press.
