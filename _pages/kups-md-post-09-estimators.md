---
layout: post
permalink: /kups-md-tutorials/post-09-estimators/
title: "What Do Free-Energy Estimators Assume?"
date: 2026-07-14
last_updated: 2026-07-15
description: "A reproducible free-energy estimator diagnostic for FEP, BAR, overlap, effective sample size, and estimator failure modes."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 9
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 9
categories: [science]
tags: [molecular-dynamics, free-energy, estimators, bar, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the PMF discussion by asking when free-energy estimators can be trusted, especially when overlap and effective sample size are poor. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Free-energy perturbation is exact as an identity and fragile as an estimator.
The difference is overlap. If samples from state A almost never visit
configurations that matter for state B, the exponential average is controlled
by rare tail events rather than by the typical trajectory frames.

For ML researchers working with MLIPs, this is the useful mental model:
free-energy estimators are diagnostics for probability mass, not just formulas
for energies. BAR improves on one-sided exponential averages by using samples
from both states, and WHAM/MBAR extend the same overlap logic across multiple
biased or intermediate states (<span id="cite-zwanzig1954"></span>[Zwanzig,
1954](#ref-zwanzig1954); <span id="cite-bennett1976"></span>[Bennett,
1976](#ref-bennett1976); <span id="cite-kumar1992"></span>[Kumar et al.,
1992](#ref-kumar1992); <span id="cite-shirts2008"></span>[Shirts & Chodera,
2008](#ref-shirts2008)).

This draft demonstrates the executable slice of the ninth tutorial with
exactly solvable Gaussian state pairs. The true free-energy difference is
known, so the diagnostic can separate estimator assumptions from
physical-model error.

The target reader already understands that a free-energy difference can be
written as a ratio of partition functions. The practical issue is whether the
finite samples in hand contain the probability mass needed to estimate that
ratio. In ordinary MD language, the hard part is not writing the formula. It is
diagnosing when the formula is being driven by configurations that were almost
never sampled.

This page is therefore about estimator failure modes. Post 08 introduced PMFs
from equilibrium samples along one coordinate. Post 09 asks what happens when
one wants a free-energy difference between states, Hamiltonians, biased
windows, or alchemical endpoints. The answer is not "use the most advanced
estimator." The answer is to understand what overlap structure the estimator
requires and what diagnostics would reveal a failure.

The executable artifacts for this page are:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/full.json)
- [estimator notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-09-estimators.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/smoke/estimator_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/estimator_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-09.md)

## What Is Being Estimated?

The current diagnostic compares two unit-variance one-dimensional states. The
second state is displaced and assigned a known free-energy offset:

| Case | Mean shift | True Delta F | Intended regime |
|---|---:|---:|---|
| good_overlap | 0.5 | 0.8 | many useful samples in both states |
| marginal_overlap | 1.5 | 0.8 | estimator looks plausible but ESS warns |
| poor_overlap | 3.0 | 0.8 | rare tails dominate one-sided FEP |

This is not a production alchemical calculation. It is a controlled test for
the failure modes that production calculations must diagnose.

The three cases keep the true free-energy difference fixed at 0.8 in
dimensionless units and change only the displacement between two unit-variance
states. That means estimator degradation can be attributed to overlap loss, not
to changing physics. Each full-profile case uses 50000 samples from each state.
The nominal sample count is therefore large, but the useful weighted sample
count can still collapse.

The full summary reports:

| Case | Overlap | Forward ESS fraction | Reverse ESS fraction | Forward FEP error | BAR error |
|---|---:|---:|---:|---:|---:|
| good overlap | 0.803 | 0.779 | 0.778 | -0.00245 | 0.00036 |
| marginal overlap | 0.453 | 0.0966 | 0.0862 | 0.00458 | 0.00810 |
| poor overlap | 0.134 | 0.00274 | 0.00182 | 0.0472 | 0.0326 |

The important number is not only the estimator error. In the marginal case,
the forward FEP error is small in this seeded run, but the effective sample
size fraction is already below 0.1. That is a warning sign. A favorable finite
sample does not prove the estimator is reliable when its weights are dominated
by a small subset of samples.

## What Does FEP Assume?

Free-energy perturbation estimates the free-energy difference from samples of
one state and energy differences to another state. In one direction, it asks:
if I sample state A, can I estimate how much probability mass state B assigns
to those same configurations? The exponential average makes rare favorable
samples extremely important.

This creates the classic one-sided failure. The typical configurations of
state A may not be typical configurations of state B. If the important B-like
configurations are rare under A, a finite trajectory may never see them. The
exponential average then becomes an estimator of missing tail events. It may
look numerically stable until another seed reveals that the tail was not
sampled.

The controlled diagnostic makes this visible through work distributions. In
the good-overlap case, forward work values have mean about 0.923 and standard
deviation about 0.502. In the poor-overlap case, forward work values have mean
about 5.334 and standard deviation about 2.997. The exponential average in the
poor case depends on rare low-work samples, not on the typical forward work.

This is why the review records effective sample size fractions. In the
good-overlap case, roughly 78 percent of the nominal forward samples remain
useful under the exponential weights. In the poor-overlap case, the forward
fraction falls to about 0.274 percent. With 50000 nominal samples, that is only
about 137 effective weighted samples by this diagnostic. More raw samples help
only if they actually sample the important tail.

## Why Does Reverse FEP Not Automatically Fix It?

One can also run the perturbation in reverse, sampling state B and estimating
state A. Sometimes one direction is much better than the other. But reverse
FEP is still a one-sided exponential average. It can fail for the same reason:
the important tail under the sampled state may be rare.

In the full diagnostic, both forward and reverse ESS fractions collapse in the
poor-overlap case. The reverse fraction is about 0.182 percent, even lower
than the forward fraction. The reverse FEP error is about 0.0201 in this
seeded run, which looks smaller than the forward error of about 0.0472, but
the ESS diagnostic still says the estimate is fragile.

This is a useful review pattern. If one FEP direction and the other direction
disagree, something is wrong. If they agree but both have poor effective sample
size, the apparent agreement may be accidental. A report should show both
directions or explain why a one-direction estimate is justified.

For MLIP-based free-energy work, reverse checks are also model checks. If the
learned potential produces configurations in one state that have unreasonable
energies under another state, the work distribution can reveal extrapolation
or instability. That failure should not be averaged away.

## What Does BAR Add?

Bennett's acceptance ratio uses samples from both states and solves for the
free-energy difference that best balances the two work distributions. It is
more statistically efficient than choosing one exponential average when both
states have useful samples. Conceptually, it estimates the free-energy
difference from the region where the two sampled distributions overlap.

BAR does not remove the need for overlap. If the two states barely overlap,
there is little bridge information for BAR to use. In this controlled example,
BAR remains close to the answer key even in poor overlap, with error about
0.0326. That is good behavior for this particular seeded Gaussian diagnostic,
not proof that BAR can rescue arbitrary endpoint calculations.

The right interpretation is conditional: BAR can be much better than one-sided
FEP when both states contribute useful bridging samples, but it still needs
overlap diagnostics. A BAR estimate without overlap, ESS, or work-distribution
review is incomplete.

The full diagnostic shows the pattern. In good overlap, BAR error is about
0.00036. In marginal overlap, it grows to about 0.0081. In poor overlap, it is
about 0.0326. The estimator degrades as overlap disappears even though the
sample count is constant. The figure makes this trend visible by placing
estimator values, overlap coefficients, ESS fractions, and work distributions
side by side.

## How Should Effective Sample Size Be Read?

Effective sample size for weighted estimates asks how many equally weighted
samples would carry comparable information. If one sample carries most of the
weight, the ESS is near one no matter how many samples were written to disk.
If weights are nearly uniform, ESS is close to the raw sample count.

ESS is not a perfect certificate. It depends on the estimator and on how
weights are defined. It also does not prove that a coordinate or Hamiltonian
path is physically meaningful. But it is a useful warning light. A tiny ESS
fraction says the estimate is dominated by rare events and should not be
reported as though all frames contributed equally.

In this page, the collapse is dramatic:

| Case | Nominal samples per direction | Approximate forward effective samples |
|---|---:|---:|
| good overlap | 50000 | 38949 |
| marginal overlap | 50000 | 4832 |
| poor overlap | 50000 | 137 |

Those counts are approximate because they multiply the nominal sample count by
the reported ESS fraction. They are enough to explain the mechanism. The poor
case has many samples but very little weighted information.

## What Do WHAM and MBAR Assume?

WHAM and MBAR generalize the same overlap logic across multiple states,
windows, or biasing conditions. Instead of asking only whether endpoint A
overlaps endpoint B, they ask whether a connected network of sampled states can
support estimates across the range of interest. Intermediate states can make a
hard free-energy difference easier by replacing one large overlap problem with
several smaller ones.

But intermediate states are not automatically sufficient. A chain with one
broken link is still broken. If adjacent windows do not overlap, WHAM/MBAR
cannot reliably infer the free-energy difference across that gap. If a window
has poor sampling, wrong bias metadata, or a hidden slow mode, the multi-state
estimate can inherit that failure.

The practical WHAM/MBAR review should include overlap matrices or equivalent
diagnostics, per-window sample counts, effective sample sizes, bias parameters,
and checks for hysteresis or replica disagreement. The estimator name alone is
not evidence. The overlap network is the evidence.
Disconnected windows make the final estimate a protocol failure, not a result.

The refreshed executable diagnostic now adds a minimal multi-state bridge
test. It samples harmonic biased windows from a known Gaussian target and
compares a connected dense bridge with an endpoint-only sparse bridge. The
dense bridge has minimum adjacent overlap about 0.184 in the full profile,
whereas the sparse bridge has effectively zero adjacent overlap and one broken
edge. This is not a full production WHAM/MBAR analysis, but it makes the
network assumption concrete: without adjacent overlap, a multi-state estimator
has a protocol failure before it has a trustworthy result.

## What Does Estimator Failure Look Like?

Estimator failure does not always announce itself as a numerical exception.
Often the output is a plausible number with too little evidence behind it.
That is why this tutorial emphasizes diagnostics rather than only estimator
formulas.

One symptom is directional disagreement. If forward and reverse perturbation
estimates differ by more than their uncertainty, at least one direction is not
sampling the important configurations. Another symptom is tiny effective sample
size. A weighted estimate can have thousands of nominal samples and still be
controlled by a handful of frames. A third symptom is seed sensitivity: a new
replica changes the estimate because rare tail samples appear or disappear.

Work distributions are often the clearest visual diagnostic. In healthy
overlap, the important work values sit in a region sampled by both states. In
poor overlap, the estimator depends on a tail that is far from the typical
work. The average work can be very different from the free-energy difference
because free energy depends on an exponential average, not an arithmetic mean
of work.

Another failure mode is a false sense of security from BAR or MBAR. These
estimators are powerful, but they are still estimators over sampled
distributions. If the sampled distributions do not connect the relevant
regions, an advanced estimator cannot infer missing probability mass. A report
should therefore show overlap evidence even when using a statistically optimal
method.

## How Should Intermediate States Be Designed?

Intermediate states are a practical response to poor endpoint overlap. Instead
of estimating a difficult free-energy difference in one jump, one designs a
path of states where neighboring states overlap well. The total free-energy
difference is then assembled from smaller, better-supported comparisons.

The spacing of intermediate states should be driven by diagnostics, not by a
decorative lambda grid. If adjacent states have poor overlap, add states where
the gap is largest. If some states have much lower ESS than others, concentrate
sampling or redesign the path there. If a state introduces unstable
configurations, the path may be physically or numerically inappropriate.

For biased sampling, intermediate states may be umbrella windows. For
alchemical transformations, they may be lambda states. For a PMF
reconstruction, they may be overlapping windows along a coordinate. In all
cases, the review question is the same: does the sampled network provide a
connected bridge of probability mass?

A good intermediate-state report includes the list of states, sample counts,
bias or lambda parameters, adjacent overlap, and a failure policy. The failure
policy matters because one weak link can dominate the final uncertainty. If a
window has poor overlap, one should not hide it inside a global estimate.

## What Is Special About MLIP Free Energies?

MLIPs make free-energy work attractive because they can reduce the cost of
sampling. They also add a layer of risk. Free-energy estimators often emphasize
rare or high-leverage configurations, exactly the configurations that may be
underrepresented in a model's training data. A model can have good average
force accuracy and still be unreliable in the overlap region that controls a
free-energy estimate.

This is especially important for reweighting and alchemical calculations. A
small subset of configurations can carry most of the statistical weight. If
those configurations are outside the MLIP's reliable domain, the estimator may
be both statistically fragile and physically wrong. The free-energy workflow
should therefore record not only estimator ESS, but also model-health evidence
for high-weight configurations.

Useful checks include energy and force ranges, stress ranges when pressure or
cell degrees matter, neighbor-environment diagnostics, comparison to a
reference potential for selected high-weight frames, and any model uncertainty
proxy available. These checks should be tied to the estimator, not reported
only as global validation metrics.

The controlled Gaussian diagnostic in this post does not test MLIP model
validity. It isolates estimator assumptions. The final capstone work should
combine both views: estimator overlap and MLIP reliability in the same
high-weight regions.

## What Are Common Reporting Mistakes?

The first mistake is reporting only the final free-energy number. A free-energy
difference without overlap, ESS, and uncertainty evidence is not reviewable.
The reader cannot tell whether the estimator used typical samples or rare tail
events.

The second mistake is treating sample count as evidence. Fifty thousand samples
per state are enough in the good-overlap case here and fragile in the
poor-overlap case. The difference is not the file size. It is the weighted
information content.

The third mistake is hiding unfavorable diagnostics because the final estimate
looks close to a reference. The marginal-overlap case has a small FEP error in
this seeded run, but the ESS fractions are already low. That warning belongs
in the prose. Favorable finite-sample luck should not be converted into a
general recommendation.

The fourth mistake is mixing estimator failure with physical interpretation.
If overlap is poor, the immediate conclusion is about the estimator and
sampling protocol. It is premature to interpret the resulting number as a
physical free-energy barrier or model error until the estimator has enough
support.

The fifth mistake is omitting the failed attempts. If intermediate states were
added, windows were moved, or samples were discarded because diagnostics failed,
that history explains the final protocol. A concise failure log can be more
useful than a polished final table alone.

## What Should The Diagnostic Show?

The full run compares forward FEP, reverse FEP, and BAR against the known
answer. It also records overlap coefficients and exponential-weight effective
sample sizes. In the poor-overlap case, the forward effective sample size
collapses to less than one percent of the nominal sample count even though the
simulation contains fifty thousand samples per state.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post09_estimator_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Estimator diagnostics for the committed full profile. BAR remains close to the known Delta F in this controlled example, while the overlap, ESS, work-tail, and multi-state bridge panels show why estimators need connected probability mass." %}

The figure has four roles. The estimator panel shows that all methods are
close in good overlap and that errors grow as overlap decreases. The overlap
and ESS panel shows the hidden statistical problem more directly than the
estimator values alone. The work-distribution panel shows why the exponential
average becomes fragile: poor-overlap estimates are controlled by rare tail
samples rather than typical work values. The multi-state bridge panel compares
a connected dense set of biased windows with a sparse endpoint-only protocol;
the sparse curve has a visible missing middle because the adjacent-window
overlap network is broken.

This is the main lesson for free-energy calculations. A good-looking estimate
is not enough. It must be accompanied by diagnostics that show the estimator
had the probability mass it needed.

## What Belongs in a Methods Paragraph?

A free-energy estimator methods paragraph should report the sampled states,
temperature, number of samples, energy or work definitions, estimator formula,
overlap diagnostics, ESS diagnostics, uncertainty method, and any discarded or
failed states. If BAR, WHAM, or MBAR is used, the report should say which
states were included and how their overlap was checked.

For alchemical or biased simulations, the methods should also record lambda
schedules or bias centers, force-field or MLIP versions, restraint or bias
parameters, equilibration discard, sampling interval, and whether estimates
were checked in both directions. If only one direction was used, the report
should explain why.

For MLIP studies, model validity belongs near the estimator. A free-energy
estimate can be statistically precise for a learned potential that is wrong in
the important overlap region. The methods should therefore separate estimator
uncertainty, sampling uncertainty, and model error.

## Practical Checklist

Before accepting a free-energy estimator result, record concrete answers to
these questions:

| Question | Evidence to record |
|---|---|
| What states are compared? | Hamiltonians, biases, coordinates, and temperature |
| What estimator is used? | FEP, BAR, WHAM, MBAR, or another estimator |
| Is there overlap? | overlap coefficient, overlap matrix, or work-distribution plot |
| How many weighted samples remain? | ESS fraction or equivalent diagnostic |
| Do directions agree? | forward/reverse checks where available |
| Are intermediates connected? | adjacent-window overlap or multi-state network |
| What uncertainty is reported? | bootstrap, asymptotic, block, or replica estimate |
| What model checks matter? | MLIP validity in important sampled regions |

The checklist is intentionally estimator-focused. A free-energy estimate is a
claim about probability ratios. It is only as good as the overlap evidence
supporting those ratios.
For hidden drafts, those diagnostics should be recorded even when the final
production estimator is still pending, because they determine which claims the
draft is allowed to make.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 09 --profile smoke
uv run kups-tutorial verify 09 --profile smoke
uv run kups-tutorial run 09 --profile full
uv run kups-tutorial verify 09 --profile full
uv run jupyter execute notebooks/post-09-estimators.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, estimator diagnostics, and figure generator from
`src/kups_md_tutorials/`. The committed full manifest records the configuration
hash, source Git revision, lockfile hash, Python version, platform, precision
policy, runtime device, and package versions. For the current full profile, the
configuration hash is
`54f9c7456965f1eb75ff0f47960d59c3eccd1c5dfa192c5298919e3fc04ed125`, the
recorded source revision is `98dc7cb2b3a6828141117f80de81bb9a242e57aa`, and
the runtime device is `jax:cpu;devices:cpu`.

| Provenance field | Value |
|---|---|
| configuration hash | `54f9c7456965f1eb75ff0f47960d59c3eccd1c5dfa192c5298919e3fc04ed125` |
| source revision | `98dc7cb2b3a6828141117f80de81bb9a242e57aa` |
| runtime device | `jax:cpu;devices:cpu` |
| precision policy | `jax_enable_x64=false;env_JAX_ENABLE_X64=unset` |

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled estimator workflows
- committed compact estimator summaries, work-sample outputs, and
  multi-state bridge curves
- executable notebook
- generated SVG/PNG four-panel figure and snapshot review
- rendered desktop and mobile page snapshots for the refreshed bridge panel
  and provenance section
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final production-style estimator diagnostics if a later public article needs
  a chemistry-specific WHAM/MBAR or alchemical example

The rule for this post is that estimator reliability is an overlap question.
More samples help only when they include the configurations that carry the
statistical weight.

## References

- <span id="ref-zwanzig1954"></span>Zwanzig, R. W. (1954). High-temperature equation of state by a perturbation method. *The Journal of Chemical Physics*, 22, 1420-1426. <a href="#cite-zwanzig1954" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bennett1976"></span>Bennett, C. H. (1976). Efficient estimation of free energy differences from Monte Carlo data. *Journal of Computational Physics*, 22, 245-268. <a href="#cite-bennett1976" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021. <a href="#cite-kumar1992" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shirts2008"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. <a href="#cite-shirts2008" class="reversefootnote" role="doc-backlink">↩</a>
