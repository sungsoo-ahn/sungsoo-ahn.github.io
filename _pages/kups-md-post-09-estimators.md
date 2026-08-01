---
layout: post
permalink: /kups-md-tutorials/post-09-estimators/
title: "What Do Free-Energy Estimators Assume?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Use FEP, BAR, overlap, effective sample size, and connected-state diagnostics without mistaking a plausible free-energy number for a supported result."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 9
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 9
categories: [science]
tags: [molecular-dynamics, free-energy, estimators, bar, overlap, kups]
toc:
  sidebar: left
related_posts: false
nav: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The Gaussian examples below are estimator controls, not molecular simulations. The physical curve is reused from Post 08 through hash-pinned kUPS GPU evidence.</em>
</p>

## A Plausible Number Can Still Be Unsupported

Free-energy perturbation is an exact identity and a fragile estimator. The
difference is probability overlap. If state A almost never visits the
configurations that matter to state B, an exponential average asks a few rare
frames—or frames never sampled at all—to determine the answer.

That failure can look deceptively healthy. A calculation may contain 50,000
samples, return a value near a reference, and still have fewer than 200 useful
weighted samples. BAR uses both directions more efficiently, but it cannot
infer missing probability mass either.

This post separates two jobs:

1. known-answer Gaussian controls test FEP, BAR, effective sample size, and a
   connected multi-state bridge;
2. a hash-pinned curve from three real kUPS GPU trajectories shows that the
   same support questions survive outside the control.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-09/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-09-estimators.ipynb),
[smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/smoke/estimator_summary.json),
[full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/estimator_summary.json),
[shared kUPS curve](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/shared_kups_pmf.csv),
[full manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-09/full/manifest.json),
[figure generator](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post09_figures.py),
and [review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-09.md).

## Hold the Answer Fixed and Remove Overlap

The control compares two unit-variance states,

$$
u_A(x)=\frac{x^2}{2}, \qquad
u_B(x)=\frac{(x-d)^2}{2}+\Delta F,
$$

with the exact reduced free-energy difference fixed at
$$\Delta F=0.8$$. Only the displacement $$d$$ changes. This isolates
estimator failure from physical-model error.

{% include kups-notebooks/post-09/post09-setup.html %}

{% include kups-notebooks/post-09/post09-estimator-control.html %}

The full profile draws 50,000 independent samples from each state. The final
errors remain modest in this seeded run, but the diagnostics do not:

<div class="table-responsive" markdown="1">

| Case | Displacement | Overlap | Forward ESS fraction | Approx. effective samples | Forward FEP error | BAR error |
|---|---:|---:|---:|---:|---:|---:|
| good | 0.5 | 0.803 | 0.7790 | 38,949 | -0.0025 | 0.0004 |
| marginal | 1.5 | 0.453 | 0.0966 | 4,832 | 0.0046 | 0.0081 |
| poor | 3.0 | 0.134 | 0.0027 | 137 | 0.0472 | 0.0326 |

</div>

The marginal FEP estimate is numerically close to the answer, yet its forward
ESS has already fallen below 10% of the nominal sample count. That is favorable
finite-sample luck, not evidence that the estimator is healthy.

## FEP Is a Tail Estimator

Forward FEP uses samples from A and reduced work
$$w=u_B-u_A$$:

$$
\Delta F = -\log \left\langle e^{-w}\right\rangle_A.
$$

The exponential rewards unusually low-work samples. With good overlap, those
samples are common enough to estimate. With poor overlap, they live in a rare
tail. More frames help only if the trajectory reaches that tail often enough.

The weighted effective sample size makes the concentration visible:

$$
N_{\mathrm{eff}}=
\frac{\left(\sum_i q_i\right)^2}{\sum_i q_i^2},
\qquad q_i=e^{-w_i}.
$$

It is not a universal uncertainty certificate, and correlated MD frames would
reduce the independent information further. It is a blunt warning that a large
trajectory file can contain very little information for a particular
reweighting calculation.

## BAR Is Better, Not Magical

BAR balances forward and reverse work samples and is statistically more
efficient than choosing one one-sided exponential average when both states
sample the overlap region.<sup id="cite-bennett"><a href="#ref-bennett">1</a></sup>
In this control, BAR remains closest to the answer as separation grows.

But its error still rises from 0.0004 to 0.0326 as overlap disappears. BAR can
combine bridging information; it cannot create that information. A BAR result
without work distributions, directional checks, or overlap diagnostics is
still incomplete.

For an MLIP, this matters twice. The highest-weight configurations may be both
statistically rare and far from the model's training distribution. Estimator
ESS and model-validity checks should therefore be evaluated on the same
high-leverage frames.

## A Multi-State Path Is Only as Strong as Its Weakest Link

WHAM and MBAR replace one difficult endpoint comparison with a network of
intermediate states.<sup id="cite-kumar"><a href="#ref-kumar">2</a></sup>
<sup id="cite-shirts"><a href="#ref-shirts">3</a></sup> The relevant question
is not how many states were listed. It is whether adjacent sampled
distributions form a connected bridge.

The control compares seven dense harmonic windows with two endpoint-only
windows. It is a support-aware bridge diagnostic, not a production MBAR
implementation.

<div class="table-responsive" markdown="1">

| Protocol | Windows | Minimum adjacent overlap | Broken edges | Support-aware PMF RMSE |
|---|---:|---:|---:|---:|
| dense bridge | 7 | 0.184 | 0 | 1.303 |
| sparse bridge | 2 | 0.000 | 1 | 3.314 |

</div>

The sparse protocol has a visible missing middle. That is a protocol failure,
not a high-uncertainty free-energy result. Post 10 turns this idea into biased
kUPS umbrella windows and adds zero-bias and force checks.

## Reuse Production Evidence Without Losing Its Identity

Post 09 is allowed to reuse GPU samples from an adjacent free-energy post. It
does not pretend that the Gaussian controls are kUPS dynamics, and it does not
launch a redundant production trajectory. Instead, both Post 09 configurations
pin three Post 08 files by exact SHA-256:

- the kUPS execution summary, including raw-HDF5 hashes and observed devices;
- the RDF-to-PMF summary;
- the compact PMF curves consumed here.

The notebook reruns the smoke estimator workflow from a fresh kernel. Before it
reads the physical curve, the workflow recomputes all three hashes and rejects
any mismatch.

{% include kups-notebooks/post-09/post09-shared-kups.html %}

The shared source contains three independent 256-atom Langevin replicas, each
with 80 stored frames. Every worker observed `gpu:NVIDIA RTX A5000`; the three
raw HDF5 hashes remain in the pinned execution summary. The primary
$$g(r)\ge 0.05$$ rule leaves 60 finite PMF bins and a 37.98 meV shifted range.

This curve is physical evidence, but not a BAR answer key. It shows the
practical consequence of the same principle: the high-free-energy region is
controlled by weak support. Changing the RDF threshold from 0.02 to 0.10 moves
the displayed range from 45.50 to 31.27 meV, a 14.24 meV analysis span that is
much larger than the 0.85 meV maximum replica standard deviation.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post09_estimator_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Estimator diagnostics for the committed full profile. The first four panels are known-answer controls: FEP/BAR estimates, overlap and weighted ESS, work tails, and connected versus broken multi-state bridges. The fifth panel is the hash-pinned RDF-derived PMF from three real 256-atom kUPS GPU replicas, with between-replica variation." %}

## What to Report

A defensible free-energy result should make its evidence inspectable:

<div class="table-responsive" markdown="1">

| Question | Minimum evidence |
|---|---|
| What states are compared? | Hamiltonians, biases, coordinates, and temperature |
| Is probability mass shared? | work distributions, overlap matrix, or adjacent-window overlap |
| How many weighted samples remain? | estimator-specific ESS and trajectory correlation analysis |
| Do directions agree? | forward/reverse estimates or a reason one direction is unavailable |
| Is the state network connected? | per-edge diagnostics and an explicit failure threshold |
| How uncertain is the result? | blocks or replicas plus analysis-choice sensitivity |
| Can the MLIP be trusted there? | checks on the configurations carrying the largest weights |
| Can the evidence be identified? | config, source revision, device record, and artifact hashes |

</div>

The estimator name is not evidence. The sampled overlap network is.

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 09 --profile smoke
uv run kups-tutorial verify 09 --profile smoke
uv run kups-tutorial run 09 --profile full
uv run kups-tutorial verify 09 --profile full

uv run kups-tutorial verify-notebooks --posts 09
uv run kups-tutorial export-notebook-cells --posts 09 \
  --site-root ../sungsoo-ahn.github.io --check
uv run python scripts/generate_post09_figures.py
```

The Post 09 verifier checks known-answer estimator behavior, ESS collapse,
connected and disconnected bridge protocols, every shared-source hash, real
kUPS engine provenance, observed GPU execution, raw-HDF5 hashes, replica and
frame counts, and hashes for all compact Post 09 outputs.

The practical rule is simple: if the samples do not connect the probability
mass that controls the estimate, narrow the claim or redesign the path.

## References

1. <span id="ref-bennett"></span>Bennett, C. H. (1976). Efficient estimation of free energy differences from Monte Carlo data. *Journal of Computational Physics*, 22, 245–268. [doi:10.1016/0021-9991(76)90078-4](https://doi.org/10.1016/0021-9991(76)90078-4). <a href="#cite-bennett" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-kumar"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011–1021. [doi:10.1002/jcc.540130812](https://doi.org/10.1002/jcc.540130812). <a href="#cite-kumar" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-shirts"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177). <a href="#cite-shirts" class="reversefootnote" role="doc-backlink">↩</a>
