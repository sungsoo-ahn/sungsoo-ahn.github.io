---
layout: post
permalink: /kups-md-tutorials/post-08-free-energies/
title: "How Do Equilibrium Samples Become Free Energies?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible free-energy diagnostic for molecular dynamics: collective variables, histogram PMFs, binning bias, reweighting, RDF-derived PMFs, and uncertainty."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 8
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 8
categories: [science]
tags: [molecular-dynamics, free-energy, pmf, reweighting, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the observable-estimator discussion by asking how equilibrium samples become free-energy estimates once a collective variable, normalization, binning rule, and uncertainty model are chosen. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Free energy is not read directly from a trajectory. It is inferred from a
probability distribution over a chosen coordinate. That coordinate may be a
distance, an angle, a density, a coordination number, or a learned collective
variable, but the estimator changes with the choice.

For ML researchers working with MLIPs, the important shift is from frames to
probability. A histogram can be converted into a potential of mean force, but
only after asking what was sampled, which bins were empty, how much smoothing or
binning bias was introduced, and what uncertainty should accompany the derived
barrier.

This draft demonstrates the executable slice of the eighth tutorial with a
controlled double-well distribution, a synthetic RDF-derived PMF with an answer
key, and a compact reduced-unit argon trajectory RDF transformed into a PMF.
It is still a diagnostic for estimator mechanics, not a production
free-energy calculation from a long GPU kUPS trajectory.

The target reader already knows that equilibrium samples are distributed
according to Boltzmann weights. The practical gap is usually the step from
that principle to a reproducible free-energy estimate. A PMF is not a
trajectory, and it is not simply a smoothed plot. It is an estimate of a
probability distribution over a chosen coordinate, shifted into free-energy
units and interpreted only where the coordinate was sampled well enough.

This page therefore treats free energy as an analysis object. The same raw
equilibrium samples can support different claims depending on the collective
variable, binning rule, support, normalization, and uncertainty estimator. The
controlled example has an answer key, so it can show estimator bias directly.
Real MD usually lacks that answer key, which makes the review habits more
important, not less.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-08/full.json)
- [free-energy notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-08-free-energies.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/smoke/free_energy_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/free_energy_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-08/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-08.md)

## What Is the Free Energy Of?

The first question in a free-energy calculation is not which estimator to use.
It is what coordinate the free energy is defined over. A PMF over a distance is
not the same object as a PMF over an angle, density, coordination number, or
learned latent coordinate. Each coordinate compresses the microscopic state in
a different way, and the free energy includes everything that has been
integrated out.

For a coordinate s, the basic relation is:

$$F(s) = -kT \log p(s) + C,$$

where p(s) is the equilibrium probability density of that coordinate and C is
an arbitrary additive constant. The constant does not affect barriers or
differences, but the coordinate and probability measure do. A histogram over s
is an estimator for p(s), not the exact distribution.

This is why the phrase "the free energy" is incomplete. One should ask: free
energy of which coordinate, under which ensemble, at which temperature, using
which estimator, and over which sampled support? Without those details, a PMF
figure can look precise while the scientific claim remains ambiguous.

For MLIP workflows, the coordinate can be especially consequential. A learned
potential may make long equilibrium sampling cheaper, but it does not choose
the collective variable. If the chosen coordinate misses a slow orthogonal
mode, the PMF along that coordinate can be misleading. If the coordinate enters
a region where the MLIP is poorly trained, the free-energy estimate may be
precise for the wrong potential.

## Why Does Coordinate Choice Change the Claim?

Choosing a collective variable is a modeling decision. A distance coordinate
may be natural for a pair association problem, but it can hide orientational
states. A density coordinate may describe a phase transition but hide local
defect structures. A coordination number may summarize local structure but
erase which neighbors are involved. A learned coordinate may separate states
well while being harder to interpret physically.

The PMF includes all microscopic degrees of freedom that are not shown on the
coordinate axis. That is the meaning of "mean force" in potential of mean
force: forces and entropic contributions from hidden variables are averaged
conditional on the coordinate. If those hidden variables are not equilibrated
at each coordinate value, the PMF estimate can be biased even when the
histogram along the displayed coordinate looks well populated.

This matters when interpreting barriers. A barrier in one coordinate is not
automatically a transition-state free energy. It may be a projection artifact,
or it may merge several pathways into one apparent bottleneck. A low barrier
can also be misleading if the coordinate ignores a slow orthogonal rearrangement.
The PMF should therefore be described as a free-energy profile along the chosen
coordinate, not as a universal free-energy landscape.

For this hidden draft, the coordinate is one-dimensional by construction. That
is useful for testing histogram and reweighting mechanics. The final article
should be more careful when discussing RDF-derived PMFs, because pair distance
is a specific coordinate with specific interpretation. It can explain radial
pair preferences, but it does not describe every many-body structural pathway.
That limitation should be visible in the caption.

## What Does the Controlled Diagnostic Test?

The current diagnostic keeps the answer key available:

| Choice | Full value | Why it matters |
|---|---:|---|
| temperature | 1.0 | dimensionless kT |
| samples | 80000 | equilibrium samples from the controlled distribution |
| coordinate domain | -2.5 to 2.5 | support for the collective variable |
| bin widths | 0.06, 0.18, 0.35 | resolution versus bias comparison |
| biased center | 0.9 | simple reweighting test |
| RDF peak radius | 1.2 | minimum of the RDF-derived PMF |
| compact argon trajectory | 108 atoms, 551 sampled frames | real time-correlated RDF support |

The true double-well barrier is `1.0`, so the diagnostic can separate estimator
error from the physical free-energy definition.

The full run uses three histogram bin widths. The finest width, 0.06, creates
84 bins across the domain, with 65 occupied by samples. The middle width, 0.18,
creates 28 bins, with 23 occupied. The coarse width, 0.35, creates 15 bins,
with 12 occupied. These differences are not only cosmetic: they change the
estimated barrier and the RMSE relative to the known curve.

The committed full summary reports:

| Bin width | Occupied bins | Barrier estimate | Barrier error | Bootstrap SE | RMSE vs true |
|---:|---:|---:|---:|---:|---:|
| 0.06 | 65 | 0.985 | -0.0146 | 0.0320 | 0.171 |
| 0.18 | 23 | 0.976 | -0.0239 | 0.0166 | 0.366 |
| 0.35 | 12 | 0.915 | -0.0848 | 0.0174 | 0.591 |

The coarser histogram has smaller-looking statistical error on the barrier but
larger bias relative to the known PMF shape. That is the central lesson:
uncertainty bars do not automatically include discretization bias, smoothing
bias, poor support, or bad coordinate choices. A free-energy estimate needs
both statistical uncertainty and estimator review.

## How Does a Histogram Become a PMF?

A histogram starts as counts. Those counts become probabilities only after
normalization by the total sample count and bin width. The probabilities then
become free energies after applying the negative log and choosing an additive
zero. Empty bins create a special problem: the log of zero is undefined, and
empty regions often mean the trajectory did not sample that coordinate range.

In practice, one usually shifts the PMF so its minimum is zero. This makes
barriers and relative differences easy to read. The shift should not be
confused with an absolute free energy. PMFs from histograms are usually
relative free energies along a coordinate.

The bin width controls a tradeoff. Narrow bins preserve resolution but increase
noise and empty-bin risk. Wide bins reduce noise but smear features and bias
barriers. In the full diagnostic, increasing bin width from 0.06 to 0.35 moves
the estimated barrier from about 0.985 to about 0.915 even though all estimates
come from equilibrium samples of the same underlying distribution.

The correct response is not always "use the finest bin." A very fine histogram
can produce noisy PMFs, disconnected support, or unstable barriers. A better
review asks whether the result is stable over reasonable bin choices, whether
the relevant transition region has enough samples, and whether the uncertainty
estimate reflects the analysis choices being reported.

## Why Do Empty and Low-Count Bins Matter?

Free-energy estimators are harsh on low probabilities because of the logarithm.
Small changes in a low-count bin can become large changes in free energy. Empty
bins are even more severe: without regularization or a different estimator,
they have no finite PMF value.

This matters most near barriers and tails. The regions that determine rare
events, transition states, or high free-energy penalties are often the regions
with the fewest samples. A PMF may look smooth near its minimum while being
unreliable near the barrier that the paper wants to discuss.

The controlled full run avoids making claims in unsupported regions. The true
double-well has high walls near the domain edge, and the reviewed figure
focuses on the sampled support. The review note keeps this as an explicit
article point: missing or empty bins should be explained rather than hidden by
plot styling.

For production MD, the same issue appears when an RDF-derived PMF uses radii
where g(r) is near zero, when a distance coordinate rarely visits the barrier,
or when a collective variable misses a metastable basin. These are not
formatting problems. They are sampling limitations.

## What Does Reweighting Change?

Reweighting tries to estimate an unbiased distribution from samples drawn from
a biased distribution. If the bias potential is known, each sample can be
assigned a statistical weight that compensates for the bias. In the simplest
case, the weight is proportional to the exponential of the bias energy divided
by kT, with normalization handled by the estimator.

The current diagnostic uses a simple biased-sample reweighting test with bias
center 0.9 and bias strength 2.0. The full summary reports a reweighted
barrier of about 1.123, compared with the true barrier 1.0 and the direct
histogram estimates near 0.985, 0.976, and 0.915 for the three bin widths. The
reweighted estimate is close enough to demonstrate the mechanism, but it is
not identical to the direct equilibrium estimate.

That difference is useful. Reweighting is not magic. It depends on overlap
between biased samples and the target distribution, accurate bias energies,
stable weights, and enough effective samples after weighting. If the biased
simulation never visits an important region, reweighting cannot create
information there. If a few samples carry most of the weight, the effective
sample size can be small even when the raw sample count is large.

This page keeps reweighting at the basics level. Post 09 is reserved for
free-energy estimators and their assumptions, including overlap, Bennett
acceptance ratio, WHAM/MBAR ideas, and failure diagnostics. Here the point is
only that weights change the probability estimator before the PMF is formed.

## How Does an RDF Become a Free Energy?

Post 07 treated the RDF as a normalized pair-distribution estimator. Once an
RDF g(r) is available, one can form an RDF-derived potential of mean force:

$$W(r) = -kT \log g(r) + C.$$

This object is a free-energy-like profile for pair separation in the sampled
environment. It is useful, but it inherits every limitation of the RDF:
finite-size support, bin width, density normalization, frame correlation,
cutoff choices, and low-count bins. If g(r) is invalid or noisy, the derived
PMF is invalid or noisy too.

The controlled post 08 workflow includes a synthetic RDF-like profile with a
configured peak radius of 1.2 and width 0.16. The full summary reports a
synthetic RDF-derived PMF minimum at radius about 1.2004, matching the
configured peak. It also reports a synthetic RDF-derived PMF barrier height of
about 1.705 in the shifted profile. This verifies the transformation for the
answer-key example.

The refreshed workflow also computes an RDF from a compact reduced-unit argon
trajectory using the same physical support logic introduced in post 07. The
full-profile argon run uses 108 atoms and 551 sampled frames at number density
0.85 and temperature 0.70. Its RDF first peak is near radius `1.125` with
`g(r)` about `3.01`; applying `-kT log g(r)` after masking low-RDF bins gives a
shifted PMF minimum at radius `1.125` and a finite-bin range of about `1.64`
in reduced energy units. Four contiguous blocks give a maximum PMF block SEM
of about `0.029`, and three independent seed-shifted replicas give a maximum
local PMF standard deviation of about `0.062`. This is a trajectory-derived
PMF diagnostic with explicit uncertainty checks, but it is not yet a long
production kUPS free-energy result.

## What Should The Diagnostic Show?

The full run checks four estimator questions. The first panel compares the
true PMF, a direct histogram PMF, and a reweighted PMF. The second panel shows
that bin width changes the estimated barrier even for equilibrium samples. The
third panel shows how an RDF-like g(r) can become a shifted PMF through
negative kT times the logarithm of g(r). The fourth panel repeats that
transformation on compact time-correlated argon trajectory frames.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post08_free_energy_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Free-energy diagnostics for the committed full profile. Histogram PMFs depend on binning, reweighting changes the estimate through statistical weights, and both synthetic and compact argon RDFs can be converted into shifted potentials of mean force only where the RDF has support; the trajectory RDF-PMF panel now overlays block SEM and replica disagreement." %}

The figure is intentionally an estimator figure. It does not claim to be a
production free-energy calculation for a molecular process. The first panel
checks whether the histogram and reweighting pipelines produce plausible
profiles against a known answer. The second panel isolates binning effects on
barrier estimates. The third panel connects the observable-estimator language
from post 07 to free-energy interpretation. The fourth panel shows the same
transform on actual compact argon trajectory frames, making the support and
low-count-bin problem visible.

The figure also shows what a review should not ignore. Barrier values depend
on binning. Reweighted estimates can differ from direct estimates. RDF-derived
PMFs are shifted profiles, not absolute free energies, and the PMF line should
break where `g(r)` is too small to support a stable logarithm. A clean figure
can still hide estimator assumptions unless those assumptions are written into
the caption, prose, and review note.

## How Should Uncertainty Be Reported?

The committed workflow uses bootstrap replicates to estimate uncertainty in
the histogram barrier. Bootstrap uncertainty is useful because it asks how the
estimate changes when the sampled data are resampled. In the full profile,
bootstrap barrier standard errors are positive for all bin widths and are
recorded in the summary rather than inferred from visual smoothness.

Bootstrap is not a universal cure. If trajectory samples are correlated,
ordinary sample-level bootstrap can be overconfident. Blocks or replicas may be
needed. If the collective variable has poor overlap between basins, resampling
the same poor data does not fix missing support. If the estimator has binning
bias, bootstrap uncertainty may not include that bias.

For a defensible PMF report, uncertainty should be paired with estimator
sensitivity:

| Question | Evidence |
|---|---|
| Are samples correlated? | block or replica-aware uncertainty |
| Are barriers stable to bin width? | bin-width sensitivity table |
| Are important bins populated? | occupancy or support report |
| Are weights stable? | effective sample size after reweighting |
| Is the coordinate sufficient? | physical interpretation and failure modes |

This is why the full diagnostic reports bootstrap standard errors, bin-width
dependence, blockwise trajectory PMF SEM, and replica PMF disagreement. The
coarse-bin estimate can have a modest bootstrap standard error while still
being biased downward relative to the known barrier. The compact argon PMF can
also have small block SEM while showing where independent seed-shifted
replicas disagree locally.
The review should catch both.

## What Are Common PMF Mistakes?

The first common mistake is treating a PMF as if it were independent of the
coordinate. A barrier along one coordinate may disappear, split, or change
height along another coordinate. That does not mean one of the plots is
automatically wrong; it means the coordinate defines the question.

The second mistake is hiding unsupported regions. Empty bins, nearly empty
bins, invalid RDF radii, and poor overlap regions should be marked or excluded,
not connected with a smooth line as though they were measured. A free-energy
plot is especially prone to this because the logarithm turns small
probabilities into visually important high free energies.

The third mistake is reporting a barrier without estimator sensitivity. A
single number should be accompanied by bin-width, block, replica, overlap, or
bootstrap evidence appropriate to the estimator. If the barrier shifts more
under reasonable analysis choices than its reported statistical error, the
analysis uncertainty is not captured by the error bar.

The fourth mistake is mixing sampling uncertainty with model validity. A PMF
can be well converged for a learned potential that is wrong in the sampled
region. For MLIP studies, the PMF should be read together with evidence that
the model is credible for the relevant configurations.

## What Belongs in the Methods Paragraph?

A free-energy methods paragraph should make the estimator reproducible. It
should name the coordinate, temperature, ensemble, sampling protocol, warmup
discard, binning or smoothing rule, normalization, reference zero, support
criteria, reweighting formula if used, and uncertainty estimator. If barriers
are reported, it should say how minima and barrier locations were identified.

For RDF-derived PMFs, the methods should also report the RDF estimator from
post 07: density, bin width, finite-size cutoff, frame selection, and
uncertainty. The phrase "-kT log g(r)" is not enough. The g(r) itself is an
estimator with assumptions.

For MLIP work, the methods should separate sampling uncertainty from model
validity. A PMF may be converged for the learned potential while still wrong
relative to a reference method. If a free-energy barrier matters, model checks
should be reported near the PMF analysis, not only in a separate benchmark
table.

## How Would This Extend to Larger kUPS Trajectories?

The current hidden draft now includes a compact reduced-unit argon
trajectory-derived PMF linked to the post 07 RDF workflow. A larger kUPS
extension would run or reuse validated GPU trajectories, compute RDFs with
finite-size-aware support, transform them into shifted PMFs, and report
uncertainty or at least block/replica sensitivity on the RDF and derived PMF.

That extension should record:

| Artifact | What it should prove |
|---|---|
| trajectory protocol | ensemble, timestep, warmup, sampling interval, and seed |
| RDF estimator | normalization, finite-size support, and uncertainty |
| PMF transform | shifted negative-log profile with valid support only |
| low-count handling | explicit policy for empty or fragile bins |
| comparison controls | bin-width or block sensitivity |
| model-health checks | whether the kUPS trajectory is credible for the sampled states |

The current controlled workflow remains useful because it isolates estimator
mechanics with an answer key. The compact trajectory-derived PMF sits beside
that lesson; a production extension should show how the same review habits
transfer to longer physical trajectories and model-credibility checks.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 08 --profile smoke
uv run kups-tutorial verify 08 --profile smoke
uv run kups-tutorial run 08 --profile full
uv run kups-tutorial verify 08 --profile full
uv run jupyter execute notebooks/post-08-free-energies.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, free-energy diagnostics, and figure generator from
`src/kups_md_tutorials/`. The committed full manifest records the configuration
hash, source Git revision, lockfile hash, Python version, platform, precision
policy, runtime device, and package versions. For the current full profile, the
configuration hash is
`ce962a26cd12d9dfd62c36c8114f8de7c6c784519749fd84d1bbd59664e59aa7`, the
recorded source revision is `77ddc60ae82aa8dca9d2cd0ad4029fa493b7a188`, and
the runtime device is CPU.

The compact outputs include the summary JSON and PMF curve table, including
the argon RDF and argon RDF-PMF columns. Those files are committed so the
notebook and website figure can be regenerated without raw samples or bulky
intermediate data.

## Practical Checklist

Before accepting a PMF claim, record concrete answers to these questions:

| Question | Evidence to record |
|---|---|
| What is the coordinate? | definition, units, and code path |
| What distribution was sampled? | ensemble, temperature, bias, and support |
| What estimator was used? | histogram, KDE, reweighting, RDF-derived transform, or other method |
| How were bins or smoothing chosen? | bin-width sensitivity or smoothing rationale |
| What regions are unsupported? | empty bins, low counts, finite-size limits, or poor overlap |
| What uncertainty is attached? | bootstrap, block, replica, or estimator-specific interval |
| What is the additive zero? | shifted minimum or other reference |
| What model checks matter? | MLIP validity over sampled coordinate states |

The checklist is deliberately more detailed than a single PMF plot. A
free-energy profile is a statistical summary, and its credibility depends on
the estimator and support as much as on the trajectory.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled free-energy workflows
- committed compact PMF curve and summary outputs
- compact reduced-unit argon trajectory RDF-derived PMF transformation with
  block and replica uncertainty overlays
- executable notebook
- generated four-panel SVG/PNG figure and snapshot review
- rendered desktop and mobile page snapshots for the refreshed hidden draft
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- citations for PMFs, histogram estimators, reweighting, and RDF-derived
  potentials of mean force beyond the current starter references
- larger GPU kUPS RDF-derived PMF diagnostics and final production consistency
  pass before public indexing

The rule for this post is that free energy is a property of an estimator over a
chosen coordinate. Changing the coordinate, bins, weights, or sampled support
changes what can be claimed.

## References

- <span id="ref-frenkel2001"></span>Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press.
- <span id="ref-tuckerman2010"></span>Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021.
