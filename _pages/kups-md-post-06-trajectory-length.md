---
layout: post
permalink: /kups-md-tutorials/post-06-trajectory-length/
title: "When Is a Trajectory Long Enough to Trust?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible trajectory-length diagnostic for molecular dynamics: warmup removal, autocorrelation, effective sample size, block uncertainty, and independent replica agreement."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 6
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 6
categories: [science]
tags: [molecular-dynamics, uncertainty, autocorrelation, equilibration, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the initialization, integrator, error, thermostat, and barostat diagnostics by asking when a finite trajectory has enough independent information to support a numerical claim. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

A molecular-dynamics trajectory is long only relative to the question being
asked. Ten million correlated frames can still contain little independent
information if the observable relaxes slowly. A short set of independent
replicas can sometimes say more than a single long trace that has not forgotten
its initial condition.

For ML researchers working with MLIPs, this matters because the simulation may
look stable while the estimator is still biased or overconfident. The practical
question is not how many frames were written. It is whether warmup was removed,
whether autocorrelation was measured, whether uncertainty reflects effective
sample size, and whether independent replicas agree. The diagnostic language
used here follows standard molecular-simulation reporting practice for
trajectory observables (<span id="cite-frenkel2001"></span>[Frenkel & Smit,
2001](#ref-frenkel2001); <span id="cite-tuckerman2010"></span>[Tuckerman,
2010](#ref-tuckerman2010); <span id="cite-allen2017"></span>[Allen &
Tildesley, 2017](#ref-allen2017)).

This draft demonstrates the executable slice of the sixth tutorial with a
controlled correlated observable and a compact reduced-unit argon
physical-observable check. The controlled model has a known equilibrium mean,
so it can expose estimator failure modes cleanly. The argon diagnostic then
asks the same trajectory-length question for potential energy per atom and a
nearest-neighbor coordination number across independent atomistic replicas.
That is still not the final GPU kUPS production workflow, but it is no longer
only a toy answer-key diagnostic.

The target reader already knows what an MD trajectory is. The missing question
is how to decide whether the trajectory is evidence. That decision cannot be
made from wall-clock time, number of frames, or visual smoothness alone. It
depends on the observable, the initialization, the equilibration discard, the
autocorrelation time, the number of independent replicas, and the uncertainty
estimator used for the final claim.

This page is deliberately about estimators rather than about a new force
field. The earlier posts asked how to initialize, integrate, thermostat, and
barostat a simulation. This post asks what happens after a trajectory exists:
which part of it should be discarded, how strongly the retained frames are
correlated, how many effective samples remain, and when independent replicas
contradict the story told by a single long run.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-06/full.json)
- [trajectory-length notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-06-trajectory-length.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/smoke/trajectory_length_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/trajectory_length_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-06/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-06.md)

## What Does "Long Enough" Mean?

"Long enough" is not a property of a trajectory file. It is a property of an
estimator for a specified observable. A trajectory can be long enough for the
mean kinetic temperature but too short for a rare defect count, a slow density
relaxation, an RDF coordination integral, or a time-correlation tail. The
observable defines the relevant timescale.

For a stationary process, an average over correlated samples is still a valid
estimator, but its uncertainty is not the same as the uncertainty from
independent samples. If adjacent frames are nearly identical, writing frames
more often mostly increases storage. It does not proportionally increase
statistical information. A review should therefore ask for effective sample
size, not only raw sample count.

There are two separate problems. The first is equilibration bias: early samples
remember the initial condition. The second is sampling uncertainty: retained
samples fluctuate around the equilibrium mean and are correlated. Warmup
removal addresses the first problem. Autocorrelation, blocking, and replicas
address the second. Treating them as one vague "trajectory length" issue
usually leads to overconfidence.

The distinction matters for MLIP workflows. A learned potential can run fast
enough to generate many frames, but speed does not remove slow modes. If the
observable relaxes on a long timescale, faster frame writing only creates more
correlated data. If the MLIP has a subtle bias in forces, stress, or phase
stability, a single trajectory may agree with itself for a long time while
independent replicas expose a problem.

The practical definition used here is: a trajectory is long enough for a claim
when the warmup choice is justified, the uncertainty estimator accounts for
correlation, independent replicas agree within that uncertainty, and the
result is stable enough across predeclared checkpoints for the intended
precision. This is stricter than "the plot looks flat," and it is the standard
this tutorial series tries to make executable.

## What Does the Controlled Diagnostic Test?

The current diagnostic compares trajectory-length checkpoints for a correlated
observable with a known answer:

| Choice | Full value | Why it matters |
|---|---:|---|
| true mean | 0.5 | answer key for the controlled diagnostic |
| stationary variance | 1.0 | equilibrium fluctuation scale |
| correlation time | 30 | sets memory and effective sample size |
| warmup steps | 1000 | removes most of the initial-condition bias |
| replicas | 6 | independent agreement check |
| checkpoints | 2000, 6000, 12000, 24000 | short-to-long estimator comparison |

The same full profile also includes a compact physical-observable check:

| Choice | Full value | Why it matters |
|---|---:|---|
| argon cell | 3x3x3 FCC conventional repeats | 108-atom reduced-unit system |
| observables | potential energy per atom; coordination number at rc = 1.5 | physical trajectory averages, not answer-key scalars |
| replicas | 5 | independent agreement check |
| checkpoints | 3000, 6000, 12000 | short-to-long atomistic estimator comparison |
| dynamics | Langevin, gamma = 1.0 | simple thermalized reduced-unit trajectory |

The physical-observable summary also records the runtime limitation explicitly:

| Field | Full-profile value |
|---|---|
| target device | `cuda_or_cpu_fallback` |
| runtime device | `jax:cpu;devices:cpu` |
| production GPU ready | `false` |
| blocking reason | target requested CUDA/GPU, but the generated runtime was CPU |

The full run keeps the underlying process fixed and changes only how much data
is allowed into the estimator. This separates the effect of trajectory length
from changes in the model, integrator, or ensemble.

The controlled observable is not an atomistic energy, density, or RDF value.
It is a stochastic process with a known mean and known correlation structure.
That makes it an answer-key diagnostic. In real MD, the true equilibrium mean
is not known. Here, the answer key lets us test whether the review workflow can
detect overconfident estimators and finite-trajectory failure modes. The argon
observable removes the answer key and forces the review to rely on
autocorrelation and replica agreement.

The full profile uses six independent replicas. Each replica starts with the
same statistical model and a controlled initial bias. The warmup removes the
earliest portion of the trajectory before estimates are computed. The
checkpoints then ask what would have been concluded if the run stopped at
2000, 6000, 12000, or 24000 steps.

The resulting summary is:

| Checkpoint | Mean estimate | Absolute error | Conservative SE | 95% half-width | Effective samples |
|---:|---:|---:|---:|---:|---:|
| 2000 | 0.4094 | 0.0906 | 0.1617 | 0.3170 | 148.9 |
| 6000 | 0.5108 | 0.0108 | 0.0571 | 0.1120 | 580.7 |
| 12000 | 0.4901 | 0.0099 | 0.0422 | 0.0827 | 996.8 |
| 24000 | 0.4936 | 0.0064 | 0.0208 | 0.0408 | 2295.6 |

The estimates happen to be close to the known mean after 6000 steps, but the
important point is not the small error at one checkpoint. The important point
is the uncertainty structure. The early checkpoint has wide replica spread.
Later checkpoints have more effective samples and narrower conservative
intervals. The raw number of retained samples grows much faster than the
effective sample count.

## Why Is Warmup Removal Not Enough?

Warmup removal is necessary when the trajectory starts away from equilibrium,
but it is not a proof that the retained samples are independent or precise. It
only says that a chosen early segment is excluded from the estimator. The
remaining trajectory may still have long memory, slow drift, or inadequate
replica agreement.

In this diagnostic, the initial bias decays with a time constant of 120. The
full profile discards 1000 steps before sampling. That is a deliberately
generous warmup for the controlled process. It removes most of the
initial-condition memory before the checkpoint estimates are made. Yet the
early retained data still have large uncertainty because the process remains
correlated and the replicas have not accumulated much independent information.

In atomistic MD, warmup is harder. One rarely knows the exact equilibration
time. Temperature may equilibrate quickly while density, phase composition,
defect populations, adsorption states, or slow conformational modes lag
behind. A single scalar trace can look equilibrated while the observable of
interest remains biased.

A practical equilibration review should therefore be observable-specific:

| Question | Why it matters |
|---|---|
| What observable defines equilibration? | Temperature alone is rarely enough. |
| What initial condition is being forgotten? | Bad density, bad velocities, or wrong phase require different checks. |
| How was warmup chosen before seeing the final estimate? | Retrospective trimming can bias the analysis. |
| Do independent replicas agree after warmup? | A single trace can hide metastability. |
| Does the estimate change when the discard is varied moderately? | Fragile estimates are not ready for production claims. |

Automated equilibration detection can be useful when it is treated as an
analysis choice to review, not as a substitute for scientific judgment. The
standard idea is to choose a truncation point that trades initial-condition bias
against the loss of effective samples (<span id="cite-chodera2016"></span>[Chodera,
2016](#ref-chodera2016)).

The review note for this hidden draft now separates two levels of evidence.
The controlled model proves the estimator machinery against an answer key. The
compact argon diagnostic applies the machinery to a physical observable. A
larger GPU kUPS production trajectory remains open before public release.

## Why Do Frames Not Equal Samples?

If a trajectory is sampled every timestep, adjacent frames are usually almost
the same. If it is sampled every few timesteps, adjacent frames may still be
strongly correlated. The effective sample size is the number of independent
samples that would give comparable uncertainty for the same estimator.

A schematic relation is:

$$N_{\mathrm{eff}} \approx \frac{N}{2\tau_{\mathrm{int}}},$$

where N is the number of retained samples and tau-int is an integrated
autocorrelation time measured in sample units. The exact prefactor and
estimator details depend on convention, but the lesson is stable: correlation
reduces independent information. This is the same statistical issue discussed
in Monte Carlo autocorrelation-time analyses and in molecular-simulation error
estimation for correlated averages (<span id="cite-sokal1997"></span>[Sokal,
1997](#ref-sokal1997); <span id="cite-flyvbjerg1989"></span>[Flyvbjerg &
Petersen, 1989](#ref-flyvbjerg1989)).

The full diagnostic makes this visible. At the final checkpoint, the retained
data contain 34500 raw samples across replicas. The effective sample estimate
is about 2296. That is a useful amount of information for this controlled
mean, but it is not 34500 independent measurements. At the 2000-step
checkpoint, 1500 retained samples correspond to only about 149 effective
samples.

This is why saving every frame can be misleading. High-frequency output is
valuable for debugging, visualization, short-time dynamics, and some
time-correlation functions. It is not automatically valuable for estimating a
long-time mean. If the correlation time is long, one must either run longer,
use more independent replicas, improve sampling, or change the scientific
claim.

For MLIP simulations, this point is easy to miss because model inference can be
fast. A trajectory with millions of frames may still have poor independent
sampling of a slow structural coordinate. The output size can be large while
the statistical evidence is small.

## What Should The Diagnostic Show?

The diagnostic reports naive standard error, autocorrelation-aware standard
error, block standard error, replica standard error, and a conservative
uncertainty used for review. The conservative uncertainty is intentionally not
the smallest number available.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post06_trajectory_length_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Trajectory-length diagnostics for the committed full profile. Running means retain memory, naive uncertainty is overconfident, effective sample size grows more slowly than retained frames, and the compact CPU-fallback argon panel shows how replica disagreement can remain visible for a physical observable." %}

The figure is designed to make four separate claims. The running-mean panel
shows that estimates can retain early-history memory. The uncertainty panel
shows that naive standard error is too small for correlated data. The
effective-sample panel shows that independent information grows more slowly
than stored frames. The argon panel shows checkpointed potential energy per
atom and coordination number with conservative uncertainty from independent
reduced-unit argon replicas.

The full summary gives the numerical version of the same story. At 24000
steps, the naive standard error is about 0.00537. The autocorrelation-aware
standard error is about 0.0208, and the conservative review standard error is
also about 0.0208. At 2000 steps, the naive standard error is about 0.0271,
while the conservative standard error is about 0.162. A naive analysis would
be much too confident early in the run.

The controlled panels do not claim that the answer-key observable is a
physical quantity. They are estimator diagnostics. The argon panel is a
physical-observable wiring check: in the current full run, potential-energy
effective samples increase from about 35 to 115, while coordination-number
effective samples increase from about 55 to 368. The coordination mean remains
near 12.30 neighbors at rc = 1.5, but the conservative 95 percent half-width
widens from about 0.074 to 0.115 because the review interval uses the largest
available uncertainty signal. That is exactly the kind of warning a
trajectory-length review should surface.

## How Should Block Averages Be Used?

Blocking is a practical way to estimate uncertainty when samples are
correlated. The trajectory is divided into contiguous blocks, each block is
averaged, and the variance among block means is used to estimate uncertainty.
If blocks are much longer than the correlation time, block means are closer to
independent than individual frames. The block-averaging citation is not just a
bibliography item here: it is the reason this page treats naive frame-count
standard errors as a failure mode rather than a minor reporting choice
(<span id="cite-flyvbjerg1989b"></span>[Flyvbjerg & Petersen,
1989](#ref-flyvbjerg1989)).

The method is useful because it asks a concrete question: if I group adjacent
frames into chunks, how variable are the chunk means? If the block size is too
small, block means remain correlated and the uncertainty is underestimated. If
the block size is too large, there may be too few blocks to estimate the
variance reliably. Blocking is therefore a diagnostic, not a mechanical
formula.

In the committed full summary, the block standard error decreases from about
0.0731 at the 2000-step checkpoint to about 0.0157 at the 24000-step
checkpoint. The block estimate is consistently larger than the naive standard
error. That is the expected direction for correlated data.

A production workflow should report enough information to make the block
analysis interpretable: block size, number of blocks, warmup discard, sampling
interval, and whether the estimate is stable across reasonable block choices.
If the estimated uncertainty changes wildly when the block size changes, the
trajectory is probably too short for that observable.

## Why Use Independent Replicas?

Independent replicas are one of the simplest ways to catch problems that a
single long trajectory hides. They should start from independent velocity
seeds, and when possible from meaningfully different initial configurations or
sampling histories. The goal is not to average away all differences. The goal
is to test whether the claimed ensemble estimate is reproducible.

Replica disagreement can mean several things. It can mean the trajectories are
too short. It can mean warmup was insufficient. It can mean the system has
metastable states that were not sampled evenly. It can also mean one replica
encountered a numerical or model failure. These cases require different
responses, but all are more visible when replicas are reviewed separately.

The full diagnostic reports the minimum and maximum replica means at each
checkpoint. At 2000 steps, replica means range from about -0.033 to 0.886,
which is a large spread around the true mean of 0.5. At 24000 steps, they
range from about 0.460 to 0.526. The later replicas are much more consistent.

This supports a practical rule: do not collapse replicas too early. Plot or
tabulate per-replica estimates before reporting the pooled mean. If one
replica strongly disagrees, the first response should be diagnosis, not
automatic averaging. Averaging can hide a failed run just as easily as it can
reduce random noise.

## How Should Checkpoints Be Chosen?

A checkpoint is a predeclared opportunity to ask what conclusion the current
amount of data supports. It is not a license to stop the first time the answer
looks favorable. For honest uncertainty reporting, checkpoints should be chosen
before looking at the final result, or at least recorded transparently.

The full profile uses checkpoints at 2000, 6000, 12000, and 24000 steps. These
are not universal recommendations. They are a compact sequence for the
controlled process. A real MD workflow might use physical time, numbers of
correlation times, replica wall-clock budgets, or observable-specific
convergence criteria.

A checkpoint review should ask:

| Check | Interpretation |
|---|---|
| Has the warmup rule changed? | If yes, record why and rerun the analysis. |
| Did the mean shift by more than uncertainty predicts? | The trajectory may still be relaxing. |
| Did uncertainty shrink as expected? | If not, correlation or replica spread may dominate. |
| Did effective sample size increase? | Raw frame count alone is not enough. |
| Do replicas agree better? | Persistent disagreement suggests metastability or failure. |

The final checkpoint in this controlled run supports the modest claim that the
mean is estimated to within a conservative 95 percent half-width of about
0.0408. It does not support every possible observable, nor does it prove that a
real MD system would be equilibrated on the same schedule.

## What Changes for Physical Observables?

In real MD, the observable has units, physical meaning, and failure modes.
Potential energy, kinetic temperature, density, pressure, coordination number,
diffusion, RDF peak heights, and free-energy estimates do not share the same
relaxation time. A trajectory can be long enough for one and insufficient for
another.

Static averages and dynamical observables also require different care. A
static mean can often be estimated from equilibrium samples after accounting
for autocorrelation. A time-correlation function has a lag-time axis, and the
tail can be noisy even when the mean of the underlying observable is precise.
Transport coefficients can be especially sensitive to long-time behavior and
finite-size effects.

For a larger argon/kUPS extension, a useful diagnostic would report additional
equilibration-sensitive and structural observables. Density could track
relaxation from initialization and NPT/NVT preparation. RDF coordination could
show how a trajectory estimator and its uncertainty behave for a physically
interpretable structural quantity. A time-correlation example could be added
later, but it should not be mixed into the same claim without separate
uncertainty analysis.

For MLIP production, the trajectory-length question also interacts with model
validity. If a learned potential slowly drifts into a region outside its
training distribution, longer simulation can make the estimate worse rather
than better. Uncertainty reporting should therefore sit next to model-health
checks: force ranges, energy ranges, stress ranges, neighbor environments, and
any available model uncertainty proxy.

## What Should Be Reported in Methods?

A trajectory-length decision should be reproducible from the methods text and
the committed artifacts. "We ran for 1 ns" is not enough. The reader needs to
know what was averaged, how often samples were saved, how much warmup was
discarded, how many replicas were used, and how uncertainty was estimated.

For a mean observable, the report should include the estimator, the number of
retained samples, the sampling interval, the estimated autocorrelation time or
block protocol, the effective sample size, and the confidence interval or
standard error. If replicas are used, report the per-replica estimates or at
least their range before reporting the pooled estimate. If a replica is
discarded, the discard rule should be stated before the result is interpreted.

For a convergence claim, the report should include checkpoints. A final number
without the path to that number hides too much. If the estimate changes
substantially between checkpoints, the final interval should not be used to
pretend the earlier drift did not happen. If uncertainty stops shrinking, the
analysis should say whether correlation, replica disagreement, or metastability
is limiting the calculation.

For a hidden tutorial draft, the same rule applies to figures and prose. The
figure caption should not claim more than the diagnostic supports. The current
figure supports claims about estimator behavior for a controlled correlated
observable, potential-energy-per-atom uncertainty, and a compact coordination
number for reduced-unit argon. It does not yet support public-release claims
about GPU kUPS argon equilibration, RDF convergence, or transport-property
uncertainty. Those
claims require the production diagnostic listed in the open items.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 06 --profile smoke
uv run kups-tutorial verify 06 --profile smoke
uv run kups-tutorial run 06 --profile full
uv run kups-tutorial verify 06 --profile full
uv run jupyter execute notebooks/post-06-trajectory-length.ipynb --inplace
uv run python scripts/generate_post06_figures.py
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, uncertainty diagnostics, and figure generator from
`src/kups_md_tutorials/`. The committed full manifest records the configuration
hash, source Git revision, lockfile hash, Python version, platform, precision
policy, target device, runtime device, GPU-readiness state, and package
versions. For the current full profile, the
configuration hash is
`6f78f4d49c0337b35ed454cab07e69b445423c1c8fba7fefc641370d4e539aa6`, the
recorded source revision is `a382fdf8dbbd3c105f67d2063fbfd66750832cbb`, the
target device is `cuda_or_cpu_fallback`, the runtime device is
`jax:cpu;devices:cpu`, and the production GPU readiness flag is `false`.

Those details matter because uncertainty estimates are protocol-dependent.
Changing the seed, replica count, sampling interval, warmup length, or
checkpoint schedule changes the finite-run summary. The review artifact keeps
those choices visible so later atomistic diagnostics can be compared against
the controlled estimator baseline.

## Practical Checklist

Before accepting a trajectory-length claim, record concrete answers to these
questions:

| Question | Evidence to record |
|---|---|
| What observable is being estimated? | Definition, units, and code path. |
| What samples were discarded? | Warmup rule and justification. |
| How correlated are retained samples? | Autocorrelation time or blocking evidence. |
| How many effective samples remain? | ESS estimate and sampling interval. |
| Do replicas agree? | Per-replica estimates before pooling. |
| How large is uncertainty? | Conservative interval and estimator choice. |
| Is the conclusion stable across checkpoints? | Predeclared checkpoint table. |
| Is the figure faithful? | Snapshot review of labels, scales, and caption. |

The checklist is intentionally analysis-heavy. MD failures often survive
because the simulation code ran and the output file is large. This tutorial
series treats the analysis as part of the experiment, not as a decorative
summary after the fact.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled trajectory-length workflows
- compact reduced-unit argon potential-energy-per-atom and coordination-number
  trajectory-length workflow
- machine-readable target-device, runtime-device, GPU-readiness, and blocking
  reason provenance for the compact argon physical-observable diagnostic
- committed compact summaries and downsampled samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback
- final citations for autocorrelation, effective sample size, blocking
  analysis, equilibration diagnostics, and physical-observable convergence

The missing pieces are:

- larger GPU kUPS trajectory-length diagnostics for physical observables such
  as density, RDF coordination, or time-correlation estimates
- rendered desktop and mobile page snapshots after the final production
  physical-observable diagnostic is added
- final consistency pass after the production physical-observable diagnostic is
  added

The rule for this post is the same as the rest of the series: a trajectory is
not trustworthy because it is large. It becomes useful when the estimator,
uncertainty, and independent checks support the claim being made.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. <a href="#cite-frenkel2001" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press. <a href="#cite-tuckerman2010" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-allen2017"></span>Allen, M. P. & Tildesley, D. J. (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press. <a href="#cite-allen2017" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-flyvbjerg1989"></span>Flyvbjerg, H. & Petersen, H. G. (1989). Error estimates on averages of correlated data. *Journal of Chemical Physics*, 91, 461-466. <a href="#cite-flyvbjerg1989" class="reversefootnote" role="doc-backlink">↩</a> <a href="#cite-flyvbjerg1989b" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-sokal1997"></span>Sokal, A. (1997). Monte Carlo methods in statistical mechanics: foundations and new algorithms. In C. DeWitt-Morette, P. Cartier & A. Folacci (Eds.), *Functional Integration*, NATO ASI Series, vol. 361, pp. 131-192. Springer. <a href="#cite-sokal1997" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-chodera2016"></span>Chodera, J. D. (2016). A simple method for automated equilibration detection in molecular simulations. *Journal of Chemical Theory and Computation*, 12, 1799-1805. <a href="#cite-chodera2016" class="reversefootnote" role="doc-backlink">↩</a>
