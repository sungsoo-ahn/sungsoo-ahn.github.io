---
layout: post
permalink: /kups-md-tutorials/post-11-enhanced-sampling/
title: "How Do Adaptive and Nonequilibrium Enhanced-Sampling Methods Work?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible enhanced-sampling diagnostic for metadynamics-style bias filling, nonequilibrium work, Jarzynski estimates, and Crooks crossings."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 11
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 11
categories: [science]
tags: [molecular-dynamics, enhanced-sampling, metadynamics, nonequilibrium-work, kups]
toc:
  sidebar: left
related_posts: false
nav: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows umbrella sampling by asking how adaptive bias and nonequilibrium pulling change the measure being sampled, and what corrections are needed before a free-energy claim is trustworthy. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

Enhanced sampling methods work by changing the probability measure. A
metadynamics-style bias discourages revisiting already sampled regions. A
steered nonequilibrium protocol drives the system along paths whose work values
must be interpreted as a path ensemble, not as equilibrium samples.

For ML researchers working with MLIPs, the important shift is that faster
motion across a barrier is not automatically an unbiased result. The bias,
history, protocol speed, and path weights are part of the estimator. Jarzynski
and Crooks identities give exact nonequilibrium relationships, but their
finite-sample reliability still depends on overlap in work space (<span
id="cite-jarzynski1997"></span>[Jarzynski, 1997](#ref-jarzynski1997); <span
id="cite-crooks1999"></span>[Crooks, 1999](#ref-crooks1999); <span
id="cite-laio2002"></span>[Laio & Parrinello, 2002](#ref-laio2002); <span
id="cite-barducci2008"></span>[Barducci et al., 2008](#ref-barducci2008)).

This draft demonstrates the executable slice of the eleventh tutorial with a
known one-dimensional double-well coordinate. The adaptive-bias diagnostic
shows how history-dependent hills fill wells; the nonequilibrium diagnostic
uses controlled work ensembles to show mean-work dissipation, Jarzynski
estimates, and a Crooks crossing.

The target reader already knows why equilibrium MD can miss rare transitions.
The more subtle question is what is being estimated after we deliberately make
transitions easier. Enhanced sampling is not a permission slip to ignore the
sampling distribution. It is a controlled way to collect samples or paths from
a modified distribution, then correct the result using the bias or protocol
metadata.

This distinction matters for MLIP workflows. A learned potential can make a
long trajectory cheap, but a cheap trajectory can still be trapped. An adaptive
bias can move the system across barriers, but the resulting trajectory is
history dependent. A pulling protocol can produce many transitions, but the
mean work is not the free energy. The estimator has to remember how the data
were generated.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-11/full.json)
- [enhanced-sampling notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-11-enhanced-sampling.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/smoke/enhanced_sampling_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/enhanced_sampling_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-11/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-11.md)

## What Changes When The Method Is Adaptive?

The full profile deposits `3000` well-tempered Gaussian hills on a double-well
coordinate. The resulting diagnostic is intentionally compact:

| Diagnostic | Full value | Interpretation |
|---|---:|---|
| final bias range | 6.534 | adaptive bias has filled a substantial free-energy range |
| reconstructed barrier error | 0.092 | final bias gives a reasonable controlled PMF estimate |
| left basin visits | 0.360 | both basins are sampled |
| right basin visits | 0.362 | neither basin dominates the run |
| barrier visits | 0.135 | the barrier region is no longer invisible |

The adaptive trajectory is not an unbiased trajectory. The bias history is part
of the result.

The compact table hides an important conceptual difference from umbrella
sampling. In umbrella sampling, each biased window has a fixed bias potential.
The samples in that window are drawn from a stationary biased distribution once
the window is equilibrated. In metadynamics-style sampling, the bias changes
while the trajectory is running. The trajectory therefore depends on its own
past.

The current diagnostic deposits many small Gaussian hills along the coordinate.
Early hills are placed where the trajectory spends time, usually in a basin.
Those hills raise the effective free energy of the visited region and make it
less attractive to return. Over time, the bias fills wells and encourages the
trajectory to visit regions that were rare under the original PMF.

The final bias range of about 6.534 is not a free-energy estimate by itself.
It says that the adaptive procedure has built a substantial history-dependent
potential across the coordinate. The reconstructed barrier error of about
0.092 says that, in this controlled example, the final bias contains useful
information about the known PMF. Those are different claims. A large bias
range proves that the method pushed the sampling distribution; it does not by
itself prove that the reconstructed PMF is accurate.

The visitation fractions provide a second check. The left and right basins
each account for about 0.36 of the sampled points, and the barrier region
accounts for about 0.135. That is the practical outcome one wants from this
toy adaptive run: the barrier is no longer invisible. In a production MD
system, the corresponding diagnostic would usually include the biased
coordinate trace, the accumulated bias, basin counts, and a convergence
measure for the reconstructed free-energy surface.

## What Does Well-Tempered Bias Change?

Plain metadynamics keeps adding bias hills with fixed height. That can produce
aggressive filling and large history dependence. Well-tempered metadynamics
reduces the effective hill height as bias accumulates, so the bias growth slows
down in heavily visited regions. The bias factor controls that tempering.

The full post 11 configuration uses a bias factor of 10.0, hill height 0.030,
and hill width 0.12. These numbers are dimensionless in the controlled
double-well example, but their roles mirror real MD parameters. Hill height
sets how strongly each deposition changes the future path. Hill width sets
the coordinate scale over which each deposition spreads. The bias factor sets
how quickly the method stops adding large corrections to already biased
regions.

The failure modes are predictable. Hills that are too high can push the system
through the coordinate in a way that looks like exploration but mostly records
protocol artifacts. Hills that are too narrow can create a rough bias surface
with poor generalization between nearby coordinate values. Hills that are too
wide can smear over meaningful barriers or wells. A bias factor that is too
aggressive can slow useful exploration, while one that is too weak can leave a
strongly history-dependent trajectory.

For a machine-learning audience, the useful analogy is online data collection.
The adaptive bias changes the data-generating process in response to previous
samples. That can improve coverage, but it also makes the dataset conditional
on the acquisition policy. The policy has to be part of the provenance.

## What Is The Adaptive Estimator Actually Using?

The adaptive estimator uses more than a coordinate histogram. It uses the bias
history that produced the trajectory. If the bias is ignored, the sampled
histogram answers the wrong question: where did the adaptive process spend its
time, not what is the unbiased PMF?

In this post, the controlled answer key is a double-well PMF, so the final
bias-derived reconstruction can be compared directly to the known barrier.
Dense physical systems do not provide such an answer key. The replacement is a
set of internal checks: repeated runs, bias convergence, stability of the
reconstructed surface, sensitivity to hill parameters, and comparison with
fixed-bias methods such as umbrella sampling when feasible.

A common mistake is to show only the coordinate trace and say the barrier was
crossed. Barrier crossing is a sampling event, not a free-energy estimate. The
result becomes a free-energy claim only after the bias history, estimator, and
uncertainty are included. This is why the review artifact treats the
metadynamics diagnostic as a measure-changing example, not merely a faster
trajectory.

Another mistake is to compare biased and unbiased trajectories frame by frame.
An adaptive trajectory may visit high-free-energy regions more often by design.
That does not mean those regions are probable in equilibrium. It means the
sampling process has been made useful for estimating the free-energy landscape,
provided the correction is valid.

## What Changes When The Method Is Nonequilibrium?

The full pulling diagnostic has true free-energy difference effectively zero,
but the forward and reverse mean works are positive because the finite-speed
protocol dissipates work. Jarzynski and Crooks estimates recover the answer in
this controlled case:

| Estimate | Full value |
|---|---:|
| forward mean work | 0.170 |
| reverse mean work | 0.173 |
| forward Jarzynski estimate | 0.001 |
| reverse Jarzynski estimate | -0.009 |
| Crooks crossing | -0.001 |

The first line to notice is not the Jarzynski estimate. It is the mean work.
Both forward and reverse directions have positive mean work around 0.17 even
though the true free-energy difference is essentially zero. The extra work is
dissipation from the finite-speed protocol. If one mistakes mean work for free
energy, the answer is biased.

Jarzynski's equality says that an exponential average of nonequilibrium work
can recover an equilibrium free-energy difference. Crooks' fluctuation theorem
relates the forward and reverse work distributions; the crossing point of the
properly defined distributions identifies the free-energy difference. Both
relationships are exact in the ideal ensemble, but finite-sample behavior can
be unforgiving.

This post uses controlled Gaussian work ensembles for the Jarzynski/Crooks
identity check rather than a full steered MD integrator. That is intentional
for the hidden draft. It isolates the statistical identity: work values are
path-ensemble data, and the free-energy estimate is not the mean work. The
diagnostic now also includes a separate path-level steered-trajectory
hysteresis check, where fast and slow protocols are generated by a simple
time-dependent restraint model. A final production MD version should replace
that controlled path model with real atomistic steered trajectories while
keeping the same diagnostic language.

## Why Is Exponential Averaging Fragile?

The Jarzynski estimator weights work values exponentially. That makes rare
low-work trajectories disproportionately important. If those trajectories are
not sampled, the estimator can look stable while being wrong. This is the
nonequilibrium analogue of missing overlap in umbrella sampling.

The full controlled diagnostic records forward and reverse exponential-weight
ESS fractions of about 0.716 and 0.722. Those high values are deliberately
benign. They say that the exponential average is not being carried by a tiny
number of rare work values in this teaching example. A production pulling
calculation would not be assumed to have that property. It would need to show
work histograms, effective sample sizes, direction agreement, and sensitivity
to pulling speed.

The worst case is a fast pulling protocol with little overlap between forward
and reverse work distributions. The paths may be visually dramatic and the
mean work may be easy to compute, but the equilibrium free-energy estimate can
depend on rare events that were never observed. This is why Crooks-style
diagnostics are more informative than reporting a single mean.

The practical rule is simple: nonequilibrium estimators need path overlap in
work space. Umbrella sampling needs overlap in coordinate space. Free-energy
perturbation needs overlap in energy-difference space. The mathematical
objects differ, but the review habit is the same: check whether the data cover
the statistically important region of the estimator.

## What Does Crooks Add Beyond Jarzynski?

Jarzynski can be applied with one direction of work values, but one-direction
estimates can hide missing rare events. Crooks uses paired forward and reverse
protocols. If the work distributions overlap, their crossing gives a direct
visual and numerical check on the free-energy difference.

In the full diagnostic, the forward Jarzynski estimate is about 0.001, the
reverse estimate is about -0.009, and the Crooks crossing is about -0.001. The
agreement is good because the synthetic work ensembles are designed to be a
well-behaved teaching example. The point is not that pulling is always this
easy. The point is that agreement across estimators is part of the evidence.

If the forward and reverse estimates disagree, the response should not be to
average them silently. Disagreement can mean insufficient paths, too fast a
protocol, poor reaction coordinate choice, hysteresis, or hidden slow modes.
For MLIP simulations, it can also mean the model is unreliable along one of the
driven paths. The diagnostic should preserve that ambiguity until additional
checks resolve it.

Crooks diagnostics are especially useful because they show the work
distributions themselves. A single scalar free-energy estimate cannot reveal
whether the crossing is supported by many samples or extrapolated from noisy
tails. In a publication figure, the histogram or density panel should make
that support visible.

## How Should Pulling Protocols Be Designed?

A pulling protocol defines a path through control-parameter space. In a
steered MD calculation, that might mean moving a harmonic restraint center from
one coordinate value to another. The protocol speed, restraint strength, start
and end states, equilibration, and number of repeated paths all affect the
work distribution.

The full post 11 diagnostic uses start and end centers at -1.1 and 1.1, a
trap force constant of 18.0, 260 path steps, and 20000 synthetic paths. In a
real MD calculation, each path would be an actual trajectory generated under
the time-dependent restraint. The work would be accumulated along that path,
and the protocol would need independent starting configurations from the
equilibrium end states.

The separate steered-trajectory hysteresis diagnostic compares a fast
65-step protocol to a slow 520-step protocol using 6000 path replicas. The
fast protocol has mean forward-plus-reverse loop width about 33.74, while the
slow protocol has loop width about 5.55. The exact values are model-specific;
the point is the ordering. Slower pulling reduces the forward/reverse
hysteresis loop, but it does not remove the need for estimator checks.

Slow pulling tends to reduce dissipation but costs more simulation time per
path. Fast pulling generates more dissipated work and wider distributions, so
more paths may be needed to observe the important low-work tail. Strong
restraints can keep the coordinate close to the protocol but may introduce
large forces and model-validity concerns. Weak restraints can give gentler
paths but may produce broad coordinate lag.

The design question is therefore not only "did the system move?" It is "did
the path ensemble support the estimator?" A good protocol makes the work
distributions interpretable, gives enough overlap for the intended estimator,
and does not drive the model into unphysical regions.

## How Does This Connect To Hysteresis?

Hysteresis appears when the result depends on direction or history. Both
adaptive bias and nonequilibrium pulling are history-sensitive by design, so
the diagnostic has to separate intended history dependence from uncontrolled
sampling failure.

In adaptive bias, hysteresis can appear when repeated runs build different
biases or converge to different free-energy surfaces. That can happen because
the coordinate misses slow orthogonal modes, because bias parameters are too
aggressive, or because the initial state traps the system in a submanifold.
The remedy is not just longer trajectories. It may require better collective
variables, multiple walkers, replicas, or a less aggressive deposition
schedule.

In pulling, hysteresis appears when forward and reverse work distributions are
far apart or produce inconsistent free-energy estimates. Some dissipation is
expected at finite speed. The problem is when the path ensemble needed by the
estimator is poorly sampled. A forward pull that follows one structural route
and a reverse pull that follows another may not define a useful pair of
protocols for the claimed free energy.

For MLIP workflows, hysteresis also tests the learned potential. Driven paths
can visit strained or high-energy configurations that were rare in training
data. If forward and reverse paths explore different extrapolative regions,
the free-energy disagreement may be a model problem rather than only a
sampling problem.

## What Should The Diagnostic Show?

The figure combines adaptive and nonequilibrium diagnostics because the two
families share a review pattern: the data are generated from a modified
measure, and the estimator must expose that modification.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post11_enhanced_sampling_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Enhanced-sampling diagnostics for the committed full profile. Adaptive bias changes where samples are drawn, nonequilibrium work identities recover the free-energy difference from a path ensemble rather than from mean work, and the steered-trajectory panel shows how faster pulling increases forward/reverse hysteresis." %}

The adaptive-bias panel compares the true PMF, a bias-derived reconstruction,
and the scaled final bias. It is meant to show that the final bias carries
information about the landscape, while also reminding the reader that the
trajectory was generated under a changing potential.

The bias-growth panel shows history dependence directly. It makes visible that
the bias is not a static restraint chosen before the run; it accumulates from
the sampled path. The basin and barrier visitation annotations connect that
history to the practical goal of escaping basins.

The work-distribution panel shows the nonequilibrium side. The forward and
reverse work ensembles have positive mean work, while the Jarzynski and Crooks
markers sit near the true free-energy difference. That contrast is the lesson:
mean work measures dissipative protocol cost; the free-energy estimator uses
an exponential or bidirectional path identity.

The steered-trajectory panel is a protocol diagnostic rather than a
free-energy estimator. It compares the forward/reverse loop width for fast and
slow time-dependent restraints. In the committed full profile, the fast
protocol's hysteresis gap is about 6.08 times the slow gap. That panel is the
warning label for any production pulling calculation: if the loop stays wide,
the protocol is still far from reversible even when the paths look smooth.

## What Should Be Reported In Methods?

For adaptive bias, a methods paragraph should report the collective variable,
bias functional form, hill height, hill width, deposition stride or count,
tempering parameters, temperature, warmup treatment, bias reconstruction
method, convergence checks, and whether multiple walkers or replicas were
used. If the bias was restarted or parameters were changed after a pilot run,
that history belongs in the report.

For nonequilibrium pulling, the methods should report the start and end
states, control parameter schedule, restraint strength, pulling speed or number
of steps, number of paths, initialization of path endpoints, work definition,
estimator, direction agreement, and work-overlap diagnostics. Reporting only a
single free-energy number is not enough.

For MLIP simulations, methods should also describe model-health checks along
biased or driven configurations. Enhanced sampling often intentionally visits
rare regions. Those regions may be exactly where a learned potential has the
least support. The simulation protocol and model-validity protocol should be
reviewed together.

## What Should Not Be Claimed?

Enhanced sampling results are easy to overstate. A trajectory that crosses a
barrier more often does not prove that the equilibrium barrier is low. A
visually flat biased histogram does not prove that the reconstructed PMF is
converged. A small Jarzynski number does not prove reliability unless the
exponential average has enough effective samples. A forward pull that reaches
the target coordinate does not prove that the reverse path, hidden modes, or
model validity are under control.

The safer claim is narrower and more informative: the protocol changed the
sampling measure in a documented way, the recorded metadata support the chosen
estimator, and the diagnostics show where the estimate is credible. If the
diagnostics are incomplete, the prose should say that the run is a controlled
demonstration or pilot result, not a final free-energy measurement.

## How Would This Extend To Production MD?

The final version of this post should connect the controlled double-well
diagnostic to a real MD enhanced-sampling workflow. A practical extension
would use a coordinate from the earlier observable/free-energy posts, run a
short adaptive-bias protocol, compare it with a fixed-window or unbiased
baseline, and then run paired pulling paths with direction agreement checks.

That extension should record:

| Artifact | What it should prove |
|---|---|
| coordinate definition | the biased or driven coordinate is physically meaningful |
| bias protocol | hills, tempering, and history are explicit |
| adaptive diagnostics | bias growth and visitation are not hidden |
| pulling protocol | path schedule and work definition are reproducible |
| overlap diagnostics | work distributions support Jarzynski/Crooks estimates |
| model checks | MLIP predictions are credible along rare or driven paths |

The controlled workflow remains useful because it isolates estimator logic.
It shows that accelerated barrier crossing, mean work, and free energy are
different objects. The production workflow should add molecular realism
without losing that separation.

## Practical Checklist

Before accepting an enhanced-sampling result, record concrete answers to these
questions:

| Question | Evidence to record |
|---|---|
| What measure was changed? | adaptive bias, pulling protocol, or both |
| What metadata corrects it? | bias history, work values, path weights |
| Did the method improve support? | basin/barrier visits or path overlap |
| Is the estimator stable? | replicas, blocks, ESS, or direction agreement |
| Is mean work separated from free energy? | Jarzynski/Crooks or equivalent diagnostics |
| Are rare configurations credible? | MLIP validity checks along biased paths |
| What failed in pilots? | revised hill settings, speeds, or coordinates |

The checklist is intentionally estimator-centered. Enhanced sampling is useful
because it changes what the simulation sees. The result is trustworthy only
when the correction for that change is visible and reviewed.

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 11 --profile smoke
uv run kups-tutorial verify 11 --profile smoke
uv run kups-tutorial run 11 --profile full
uv run kups-tutorial verify 11 --profile full
uv run jupyter execute notebooks/post-11-enhanced-sampling.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, enhanced-sampling diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled enhanced-sampling workflows
- committed compact summaries and diagnostic curves
- executable notebook
- generated SVG/PNG four-panel figure and snapshot review
- rendered desktop and mobile page snapshots for the latest four-panel figure
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final 3,500-10,000-word article prose
- production MD context with real atomistic steered trajectories
- final uncertainty diagnostics and citation pass

The rule for this post is that enhanced sampling is a change of measure. Bias
history and path weights are part of the estimator, not implementation details
to hide after the trajectory crosses a barrier.

## References

- <span id="ref-jarzynski1997"></span>Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. *Physical Review Letters*, 78, 2690-2693. <a href="#cite-jarzynski1997" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-crooks1999"></span>Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. *Physical Review E*, 60, 2721-2726. <a href="#cite-crooks1999" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-laio2002"></span>Laio, A. & Parrinello, M. (2002). Escaping free-energy minima. *Proceedings of the National Academy of Sciences*, 99, 12562-12566. <a href="#cite-laio2002" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-barducci2008"></span>Barducci, A., Bussi, G. & Parrinello, M. (2008). Well-tempered metadynamics: a smoothly converging and tunable free-energy method. *Physical Review Letters*, 100, 020603. <a href="#cite-barducci2008" class="reversefootnote" role="doc-backlink">↩</a>
