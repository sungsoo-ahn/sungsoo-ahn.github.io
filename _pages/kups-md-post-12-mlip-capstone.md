---
layout: post
permalink: /kups-md-tutorials/post-12-mlip-capstone/
title: "What Changes When the Potential Is a Machine-Learned Interatomic Potential?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible MLIP reliability diagnostic for fcc aluminum: static force error, extrapolation, drift, uncertainty calibration, and artifact provenance."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 12
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 12
categories: [science]
tags: [molecular-dynamics, machine-learned-potentials, mace, aluminum, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post closes the series by asking what changes when the potential is a machine-learned interatomic potential rather than a fixed analytic or classical model. The current numerical diagnostic is a deterministic CPU surrogate for MLIP reliability checks; the MACE artifact metadata is now pinned and hash-verified, but the final public article still needs a real GPU production pass. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

An MLIP changes the failure modes of molecular dynamics. The equations of
motion, thermostat, barostat, observables, free-energy estimators, and enhanced
sampling diagnostics still matter, but now they sit behind a learned potential
whose accuracy depends on the local environment being inside the model's
training support.

For ML researchers, the practical lesson is that static test error is not
deployment validation. A model can have acceptable force error on familiar fcc
aluminum configurations while still showing extrapolation, drift, ensemble
temperature bias, neighbor-list risk, or biased free-energy shifts when the
simulation leaves the familiar regime. Prior work on high-dimensional neural
network potentials, Gaussian approximation potentials, equivariant neural
network potentials, and MACE motivates this reliability view (<span
id="cite-behler2007"></span>[Behler & Parrinello, 2007](#ref-behler2007);
<span id="cite-bartok2010"></span>[Bartok et al., 2010](#ref-bartok2010);
<span id="cite-batzner2022"></span>[Batzner et al., 2022](#ref-batzner2022);
<span id="cite-batatia2022"></span>[Batatia et al., 2022](#ref-batatia2022)).
Validation guidance for MLIPs makes the same deployment point: a potential
should be reviewed against the physical task it will be used for, not only
against a single static test metric (<span id="cite-morrow2023"></span>[Morrow
et al., 2023](#ref-morrow2023)).

This draft demonstrates the executable slice of the twelfth tutorial with
three fcc-Al reliability regimes. The configured model artifact is recorded as
`mace-mp-0b3-medium.model` from `mace-foundations/mace-mp-0`, pinned at
revision `e291ace`, with SHA-256
`2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012`.
The artifact belongs to the MACE-MP-0 foundation-model family, which is useful
for broad initial exploration but still needs task-specific validation before
quantitative production claims (<span id="cite-batatia2025"></span>[Batatia
et al., 2025](#ref-batatia2025)).

The target reader already knows how MLIPs are trained and evaluated on static
structures. The capstone question is different: what evidence is needed before
using a learned potential as the force provider inside an MD workflow? The
answer is not one test-set RMSE. It is a chain of deployment checks that connect
static error, dynamical stability, ensemble control, observable bias, and
provenance.

This page keeps that chain visible. The current numbers are deterministic
surrogate diagnostics, not final MACE/fcc-Al production results. That
limitation is intentional in the hidden draft. It lets the page describe the
review protocol before the final GPU pass replaces the surrogate numerical
diagnostic with real production trajectories.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/full.json)
- [MLIP capstone notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-12-mlip-capstone.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/smoke/mlip_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/mlip_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/manifest.json)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post12_figures.py)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-12.md)

## What Changes In The Capstone?

The full profile compares three regimes:

| Regime | Force RMSE | NVE drift | Extrapolation fraction | Neighbor risk |
|---|---:|---:|---:|---:|
| in_domain_fcc | 0.030 | 0.0026 | 0.0001 | 0.000 |
| strained_cell | 0.069 | 0.0144 | 0.9945 | 0.150 |
| extrapolative_hot | 0.153 | 0.0191 | 1.0000 | 0.971 |

The diagnostic is not claiming that these are production MACE numbers. It is
showing the shape of the checks the final production run must pass or fail
honestly.

The three regimes are deliberately ordered. The `in_domain_fcc` case represents
small thermal displacements around a familiar fcc aluminum environment. The
`strained_cell` case changes the cell enough that most samples are flagged as
extrapolative. The `extrapolative_hot` case combines larger strain and larger
thermal displacement, producing saturated extrapolation flags and high
neighbor-list risk.

The table makes the main lesson visible. Force RMSE increases from about 0.030
to 0.069 to 0.153. That is a static error trend. But the deployment trend is
broader: normalized NVE-like drift grows, ensemble temperature drift grows,
and neighbor-list risk grows from zero to nearly one. A potential can fail MD
through mechanisms that are not summarized by force RMSE alone.

The capstone therefore reviews the MLIP as part of the simulation method. The
potential is not an isolated model file. It interacts with timestep, precision,
neighbor lists, thermostat and barostat settings, observable estimators,
free-energy reconstruction, and enhanced sampling. Every earlier tutorial
becomes a diagnostic lens for the learned potential.

## Why Is Static Test Error Not Enough?

Static force and energy errors are necessary checks. If a model cannot
reproduce reference forces on relevant configurations, there is no reason to
trust its MD. But static tests are usually evaluated on a dataset distribution,
not on the distribution induced by a simulation. MD generates a sequence of
new configurations whose probability depends on the model's own forces.

This feedback loop changes the meaning of validation. A small static force
error on familiar structures does not guarantee stable trajectories. A small
energy error does not guarantee a good pressure distribution. A good average
metric can hide rare configurations that trigger large forces, bad neighbor
updates, or unphysical heating.

The full diagnostic illustrates this by separating force RMSE from drift and
extrapolation. The in-domain case has force RMSE about 0.030 and normalized
drift about 0.0026. The strained case has force RMSE about 0.069, but its
extrapolation fraction is already about 0.994. The hot extrapolative case has
force RMSE about 0.153 and neighbor-list risk about 0.971. Static error gets
worse, but the deployment warnings become urgent earlier than the force RMSE
alone would suggest.

For a machine-learning researcher, this is the difference between held-out
prediction and closed-loop control. MD is closed-loop deployment. The model's
errors change the states the model will see next.

## How Should Extrapolation Be Treated?

Extrapolation is not a binary moral judgment on a model. It is a diagnostic
that asks whether the current configuration resembles the model's training or
calibration support. In a production MLIP workflow, extrapolation might be
measured with committee disagreement, latent-distance metrics, local
environment descriptors, uncertainty estimates, or explicit domain checks.

The current surrogate uses an extrapolation fraction. It is almost zero in the
in-domain case, about 0.994 in the strained case, and one in the hot case. The
exact surrogate rule is less important than the habit: every trajectory should
report how often it enters regions where model support is weak.

The interpretation should be conservative. A high extrapolation fraction does
not automatically prove the trajectory is wrong, but it changes the burden of
evidence. The run should not be treated as a clean production result unless
additional checks explain why the model remains reliable there. Those checks
could include reference calculations on sampled frames, active-learning
updates, comparison with another potential, or a reduced claim about the
trajectory.

The opposite mistake is also common. A low extrapolation flag does not prove
that the result is correct. The extrapolation metric may miss a failure mode,
or the model may be confidently biased within its nominal domain. That is why
the capstone pairs extrapolation with force error, drift, uncertainty
coverage, and observable/free-energy shifts.

## Why Does Drift Matter For MLIPs?

Energy drift was introduced earlier as an integrator and force-quality
diagnostic. In the MLIP setting, drift has an additional role: it can reveal
deployment errors that are not obvious in static snapshots. A learned force
field may produce small average force errors but still create systematic
energy injection or removal over a trajectory.

The full surrogate diagnostic records normalized NVE-like drift. It increases
from about 0.0026 in the in-domain case to about 0.0144 in the strained case
and about 0.0191 in the hot case. These values are not final MACE production
claims. They are teaching signals that connect model regime to dynamical
stability.

Drift should be interpreted alongside bounded fluctuations. A symplectic
integrator with a good force field can show bounded energy oscillations without
long-term drift. A model or numerical setup that produces monotonic drift is a
different problem. The review should separate timestep instability, precision
issues, neighbor-list discontinuities, and model extrapolation rather than
collapsing them into one failure label.

For MLIPs, this separation is practical. If drift disappears with a smaller
timestep, the issue may be integration stability. If drift correlates with
large extrapolation flags, the issue may be model support. If drift jumps at
neighbor-list rebuilds or near cutoff boundaries, the issue may be the model
or simulation plumbing. Each case has a different fix.

## What Changes For Ensemble Control?

Thermostats and barostats do not make a bad potential reliable. They can hide
some symptoms by controlling temperature or pressure, but they cannot turn an
extrapolative force field into a physically validated model. The capstone
therefore keeps ensemble diagnostics separate from static error.

The full summary records ensemble temperature drift. The in-domain case has
about 0.506 K drift, while the strained and hot cases reach about 13.58 K and
16.06 K. In a real workflow, such a pattern would force a review of model
validity, timestep, thermostat coupling, and sampled configurations before any
observable or free-energy claim was accepted.

The key point is that ensemble control diagnostics are downstream of the
potential. If the learned potential changes the effective forces in a biased
way, a thermostat can maintain a target kinetic temperature while structural
or thermodynamic observables remain biased. A barostat can maintain an average
pressure while the equation of state is wrong.

This is why the earlier ensemble posts matter in the capstone. The question is
not only whether the thermostat ran. It is whether the model, integrator, and
ensemble controller together sample the intended distribution.

## Why Is Neighbor-List Risk Included?

Neighbor lists look like implementation details until they become scientific
failure modes. MLIPs often use local environments within a cutoff. If the
neighbor list, cutoff behavior, skin distance, or rebuild schedule is poorly
matched to the dynamics, the force seen by the integrator can change
discontinuously or miss important local information.

The current diagnostic records a neighbor-list risk fraction. It is zero in
the in-domain case, about 0.150 in the strained case, and about 0.971 in the
hot case. This is a surrogate risk metric, but it encodes a real review
question: does the trajectory move through local environments where neighbor
handling is likely to matter?

In production, this should be connected to concrete simulation settings:
cutoff, skin, rebuild frequency, maximum displacement between rebuilds, and
whether the model was trained with compatible local environments. If a hot or
strained trajectory pushes atoms into unusual coordination shells, the model
and neighbor logic must be reviewed together.

The practical failure is subtle. A trajectory can run without crashing and
still accumulate biased forces from neighbor-list artifacts. The capstone
therefore treats neighbor-list risk as part of provenance, not as a backend
detail to omit from the article.

## What Should Uncertainty Mean For An MLIP?

Uncertainty is useful only if it is calibrated to the failure being claimed.
An uncertainty estimate that grows in extrapolative regions is a warning
signal. An uncertainty estimate that stays small when force errors are large
is a dangerous confidence signal. Both cases need to be reported.

The current surrogate records mean uncertainty and two-sigma coverage. Mean
uncertainty grows from about 0.042 to 0.146 to 0.404 across the three regimes.
Two-sigma coverage is high in all cases: about 0.993, 1.000, and 1.000. In
this controlled diagnostic, the uncertainty signal is intentionally calibrated
enough to act as a warning rather than a hidden failure.

That does not mean final MLIP uncertainty will be so clean. Production
uncertainty might come from an ensemble of models, dropout-like approximations,
latent distances, local environment scores, or calibrated residual models.
Each method has assumptions. The review should state what uncertainty means,
what it was calibrated against, and where it is expected to fail.

For MD, uncertainty also needs temporal interpretation. A few high-uncertainty
frames may identify rare transitions or bad contacts. Persistent uncertainty
may indicate that the entire trajectory has left the model's domain. A
free-energy calculation that relies on high-uncertainty barrier configurations
needs stronger validation than one whose important regions are well supported.

## How Do MLIP Errors Reach Observables And Free Energies?

The earlier posts treated observables and free energies as estimators from a
trajectory. With an MLIP, the trajectory distribution itself may be biased by
model error. That means an observable can be statistically well estimated for
the wrong potential.

The full surrogate records a free-energy barrier shift that grows from
effectively zero in the in-domain case to about 0.00395 and 0.01514 in the
strained and hot cases. These values are small in the controlled diagnostic,
but they encode an important mechanism. A learned potential can perturb the
relative probabilities of basins and barriers. Even if sampling and WHAM/MBAR
are implemented correctly, the result belongs to the learned potential unless
model error is controlled.

This applies to RDFs, coordination numbers, diffusion estimates, PMFs,
umbrella windows, metadynamics biases, and pulling work distributions. The
estimator can be mathematically correct while the force provider is not
validated for the configurations that dominate the estimate.

The capstone's final public version should therefore connect every result to a
model-support check. If a free-energy barrier uses extrapolative configurations,
that should be stated directly. If an observable is insensitive to the risky
regions, that is also useful evidence. Either way, model diagnostics and
statistical diagnostics should appear together.

## What Provenance Must Be Frozen?

For a classical analytic potential, the model can often be named compactly.
For an MLIP, a name is not enough. The artifact, repository revision, training
or release identifier, downloaded file hash, model settings, neighbor/cutoff
settings, precision, device, and software versions all affect reproducibility.

The current page pins `mace-mp-0b3-medium.model` from
`mace-foundations/mace-mp-0` at revision `e291ace` and records the downloaded
file's SHA-256 hash. That fixes model-file provenance for this hidden draft,
but it does not turn the surrogate numerical diagnostic into a production
MACE/fcc-Al result. The validation question remains tied to the intended task:
the same artifact can be adequate for a qualitative pilot and inadequate for a
published fcc-Al free-energy or dynamics claim (<span id="cite-morrow2023b"></span>[Morrow et al.,
2023](#ref-morrow2023); <span id="cite-batatia2025b"></span>[Batatia et al., 2025](#ref-batatia2025)).

The final GPU pass should freeze:

| Item | Why it matters |
|---|---|
| model repository and revision | identifies the code and release state |
| downloaded artifact hash | proves the exact weights used |
| kUPS and dependency versions | records simulation and model interfaces |
| device and precision policy | affects forces, drift, and reproducibility |
| neighbor/cutoff settings | defines the local environment construction |
| random seeds and configs | makes smoke/full comparisons reproducible |

Without this provenance, another researcher cannot know whether a discrepancy
comes from physics, model version, precision, or a changed artifact.

## What Should The Diagnostic Show?

The full run checks three ideas. Static force metrics worsen as the case leaves
the in-domain regime. Dynamics and extrapolation metrics expose failure modes
that static force error alone does not explain. Uncertainty calibration must be
checked against realized force errors rather than treated as a decorative model
output.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post12_mlip_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="MLIP reliability diagnostics for the committed full profile. Static errors, dynamics drift, extrapolation flags, neighbor-list risk, and uncertainty calibration must be reviewed together before trusting MD or free-energy claims from a learned potential." %}

The figure has three roles. The static-error panel shows that force and energy
metrics worsen across regimes. The dynamics/extrapolation panel shows that
deployment warnings can become severe even when a single scalar drift metric
does not look dramatic. The uncertainty panel asks whether uncertainty grows
with realized error.

The artifact annotation is also part of the figure review. It exposes the
pinned model file and repository revision so the hidden draft carries exact
artifact provenance even though the plotted numerical diagnostics are still a
controlled surrogate rather than a completed MACE production result.

The intended reader should leave the figure with one habit: never review an
MLIP trajectory from a single metric. Static error, drift, extrapolation,
neighbor risk, uncertainty, and provenance answer different questions. A
credible MD claim needs the relevant subset of those answers.

## What Would Make The Capstone Production-Ready?

The final capstone should replace the deterministic surrogate with a real
MACE/fcc-Al production run. The controlled workflow should remain as a smoke
and review harness, but the public scientific claims should be based on GPU
outputs from the verified model artifact.

That final pass should be planned before the run starts. The protocol should
state the exact MACE artifact and repository revision, the fcc-Al cell and
initialization path, timestep and precision policy, neighbor/cutoff settings,
thermostat or NVE handoff, trajectory length and replica plan, model-support
diagnostics, observable/free-energy targets, and the rule for rejecting or
narrowing claims when extrapolation, drift, uncertainty, or neighbor risk is
too large. The current surrogate diagnostic exercises that review structure,
but it is not a substitute for the real GPU production evidence.

The production pass should include:

| Requirement | Evidence |
|---|---|
| pinned MACE artifact | repository revision and SHA-256 hash |
| real fcc-Al trajectories | compact summaries and provenance manifests |
| stability diagnostics | NVE drift, bounded fluctuations, and failures |
| ensemble diagnostics | temperature/pressure checks where applicable |
| model-support diagnostics | extrapolation or uncertainty over sampled frames |
| observable/free-energy links | how model errors affect downstream claims |
| rendered review artifacts | figure snapshots and page snapshots |

If any of those items fails, the final article should report the failure. A
negative result is scientifically useful: it tells readers where a learned
potential is not yet ready for a specific MD claim.

## How Does This Close The Series?

The capstone is not a separate ML benchmark appended to an MD tutorial. It is a
stress test of the whole workflow. Initialization still matters because an MLIP
can be pushed out of domain by a bad starting density, poor minimization, or
unreviewed velocity draw. Integrators still matter because force noise and
cutoff behavior can turn a tolerable static error into energy drift.

Thermostats and barostats still matter because learned forces define the
distribution those controllers act on. A thermostat can regulate kinetic
temperature while structural observables remain biased. A barostat can sample
a cell distribution that is internally consistent for the learned potential
but wrong relative to the reference physics.

Trajectory-length and observable diagnostics still matter because MLIP errors
can create long autocorrelation, hidden metastability, or biased estimator
inputs. A clean-looking RDF or coordination number is not enough unless the
sampled local environments are inside the model's credible domain.

Free-energy and enhanced-sampling diagnostics still matter most in the
regions where MLIPs are easiest to overtrust. Barriers, strained geometries,
rare coordination states, and driven paths often sit at the edge of training
support. If an umbrella window, metadynamics bias, or pulling protocol depends
on those regions, the free-energy estimator inherits the model-risk question.

The series therefore ends with a practical rule: an MLIP does not replace MD
validation. It adds a model-validity layer to every MD validation step.

This also changes how limitations should be written. If the final GPU pass
finds extrapolation, drift, or artifact uncertainty, the article should not
hide that behind a polished capstone story. The useful result is the complete
diagnostic chain: what was checked, what passed, what failed, and which MD
claims remain defensible for the pinned model.

## Practical Checklist

Before trusting an MLIP-driven MD result, record concrete answers to these
questions:

| Question | Evidence to record |
|---|---|
| Which exact model ran? | repository revision, artifact hash, model settings |
| Is the trajectory in domain? | extrapolation or uncertainty diagnostics |
| Are forces accurate enough? | static force/energy checks on relevant frames |
| Is dynamics stable? | NVE drift and bounded fluctuation diagnostics |
| Is ensemble control credible? | temperature, pressure, and cell checks |
| Are neighbor settings safe? | cutoff, skin, rebuild, and risk diagnostics |
| Do observables depend on risky frames? | per-region model-support review |
| Are free energies model-limited? | support checks in basins and barriers |

The checklist is not a replacement for physics judgment. It is a way to keep
MLIP validation attached to the MD claim being made.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 12 --profile smoke
uv run kups-tutorial verify 12 --profile smoke
uv run kups-tutorial run 12 --profile full
uv run kups-tutorial verify 12 --profile full
uv run jupyter execute notebooks/post-12-mlip-capstone.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, MLIP capstone diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled MLIP reliability workflows
- pinned `mace-mp-0b3-medium.model` artifact metadata with verified SHA-256
- committed compact summaries and diagnostic samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- rendered desktop and mobile page snapshots for the hidden draft
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- real MACE/fcc-Al GPU production run
- final 3,500-10,000-word article prose
- rendered desktop and mobile page snapshots after final production diagnostics
  or any public-indexing change
- additional citations if the final production article adds new scientific
  claims beyond the current controlled MLIP-reliability and protocol discussion

The rule for this post is that an MLIP is part of the simulation method, not a
drop-in oracle. Provenance, extrapolation, drift, and uncertainty diagnostics
are part of the scientific result.

## References

- <span id="ref-behler2007"></span>Behler, J. & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98, 146401. <a href="#cite-behler2007" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bartok2010"></span>Bartok, A. P., Payne, M. C., Kondor, R. & Csanyi, G. (2010). Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. *Physical Review Letters*, 104, 136403. <a href="#cite-bartok2010" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-batzner2022"></span>Batzner, S. et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453. <a href="#cite-batzner2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-batatia2022"></span>Batatia, I. et al. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *NeurIPS Workshop*. <a href="#cite-batatia2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-morrow2023"></span>Morrow, J. D., Gardner, J. L. A. & Deringer, V. L. (2023). How to validate machine-learned interatomic potentials. *The Journal of Chemical Physics*, 158, 121501. <a href="#cite-morrow2023" class="reversefootnote" role="doc-backlink">↩</a> <a href="#cite-morrow2023b" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-batatia2025"></span>Batatia, I. et al. (2025). A foundation model for atomistic materials chemistry. *The Journal of Chemical Physics*, 163, 184110. <a href="#cite-batatia2025" class="reversefootnote" role="doc-backlink">↩</a> <a href="#cite-batatia2025b" class="reversefootnote" role="doc-backlink">↩</a>
