---
layout: post
permalink: /kups-md-tutorials/post-10-umbrella-sampling/
title: "What Does Umbrella Sampling Actually Sample?"
date: 2026-07-14
last_updated: 2026-07-29
description: "How biased windows sample local distributions and how overlap controls umbrella reconstruction and hysteresis."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 10
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for ML researchers who are new to simulation."
series_order: 10
categories: [science]
tags: [molecular-dynamics, umbrella-sampling, free-energy, wham, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Part 10 of the kUPS Molecular Dynamics Tutorials. The executable example is maintained in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>


Umbrella sampling does not sample the target PMF directly. Each window samples
a biased ensemble. The unbiased PMF is reconstructed later, and that
reconstruction only works when neighboring biased ensembles overlap enough to
form a connected bridge across the collective variable.

For ML researchers working with MLIPs, this is the useful shift: window
placement is a statistical design problem. A large trajectory in each window is
not enough if the windows do not exchange probability mass through their tails.
WHAM and MBAR formalize this reweighting problem across biased or intermediate
states (<span id="cite-torrie1977"></span>[Torrie & Valleau,
1977](#ref-torrie1977); <span id="cite-kumar1992"></span>[Kumar et al.,
1992](#ref-kumar1992); <span id="cite-shirts2008"></span>[Shirts & Chodera,
2008](#ref-shirts2008)). Production umbrella-sampling reviews emphasize the
same operational point: biasing is only useful when the reaction coordinate,
window protocol, reconstruction method, and convergence checks are all reported
as part of the free-energy calculation (<span id="cite-kaestner2011"></span>[Kästner,
2011](#ref-kaestner2011)).

This draft demonstrates the executable slice of the tenth tutorial with a
known one-dimensional double-well PMF. Dense and sparse umbrella protocols are
run against the same answer key, so the diagnostic can isolate overlap and
window placement from physical-model error. The full run
also adds a compact pair-distance umbrella diagnostic with Lennard-Jones
contact physics, window-overlap checks, replica disagreement, and explicit
CPU-fallback runtime provenance.

The target reader already knows why enhanced sampling is needed: equilibrium
trajectories can spend too much time in basins and too little time near
barriers. The practical question is what umbrella sampling actually gives you.
It does not give an unbiased trajectory in each window. It gives biased samples
whose relationship to the target ensemble must be reconstructed with the bias
metadata and enough overlap.

This distinction is easy to miss in automated workflows. One can place window
centers, run many steps per window, and obtain a smooth PMF-looking curve. That
curve is not trustworthy unless the windows form a connected statistical
bridge, replicas agree, and the reconstruction is checked against the sampled
support. This page treats window placement and overlap diagnostics as part of
the result.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/full.json)
- [umbrella notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-10-umbrella-sampling.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/smoke/umbrella_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/umbrella_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/manifest.json)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post10_figures.py)
- [self-analysis](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-10.md)

## What Is Being Sampled?

The current diagnostic compares two harmonic umbrella protocols on the same
double-well coordinate:

| Protocol | Windows | Minimum adjacent overlap | Barrier error |
|---|---:|---:|---:|
| dense_windows | 9 | 0.3552 | 0.0106 |
| sparse_windows | 4 | 0.0003 | -0.2554 |
| pair_distance_umbrella | 8 | 0.3109 | well-depth error 0.0053 |

Both protocols draw many samples from every biased window. The difference is
whether those biased samples connect neighboring regions of the collective
variable.

An umbrella window samples a biased distribution. If the unbiased potential of
mean force along coordinate s is F(s), a harmonic umbrella adds a restraint
centered at a chosen value. The sampled distribution is shaped by the sum of
the underlying PMF and the bias. The sampled mean is therefore not guaranteed
to equal the umbrella center.

That point appears directly in the full summary. In the dense protocol, the
window centered at -1.6 has sample mean about -1.339, not -1.6. The window
centered at -1.2 has mean about -1.115. Near the middle, the window centered
at 0.0 has mean about -0.003. The center defines the bias parameter, not the
exact sampled coordinate average.

The dense protocol uses nine windows from -1.6 to 1.6 at spacing 0.4. The
sparse protocol uses four windows at -1.6, -0.8, 0.8, and 1.6. Both use the
same force constant, temperature, bin width, and samples per window. The
comparison therefore isolates window placement.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_biased_windows.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Each umbrella window samples a shifted local distribution rather than the unbiased target. The reconstruction must account for the bias applied in every window." %}

## Why Is Overlap the Central Diagnostic?

Umbrella sampling reconstructs a global PMF from local biased samples. Adjacent
windows must share probability mass so that their relative free-energy offsets
can be inferred. If one region has no bridge, the reconstruction can align
local pieces incorrectly or smooth over a gap that was never sampled.

In the full profile, the dense protocol has minimum adjacent overlap about
0.355 and mean adjacent overlap about 0.443. The sparse protocol has mean
overlap about 0.121, but its minimum adjacent overlap is only 0.0003. That
near-zero bridge is the failure. It says that at least one adjacent pair of
windows barely shares support.

The consequence appears in the reconstructed barrier. The true barrier height
is 1.0. Dense windows reconstruct about 1.0106. Sparse windows reconstruct
about 0.7446, biasing the barrier downward by about 0.255. Every sparse window
has thousands of samples, but the global reconstruction is still wrong because
the windows do not connect the barrier region.

This is the core lesson: local sample count is not global evidence. A window
can be well sampled around its own biased distribution and still fail to
support the full PMF if neighboring windows do not overlap.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_window_overlap.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Adjacent-window overlap reveals whether samples can connect neighboring regions. Sparse windows leave gaps even when every individual histogram looks smooth." %}

## What Should Window Placement Achieve?

Window placement should be designed around the sampled distributions, not only
around geometric centers. A common first pass places centers uniformly along a
coordinate. That can work if the PMF curvature and bias strength are uniform
enough. Barriers, steep wells, hard boundaries, and coordinate-dependent
diffusion can all make uniform spacing inadequate.

The full dense protocol is deliberately simple. It places windows every 0.4
units and uses a force constant of 18.0. This produces adjacent overlap above
0.35 throughout the coordinate range in the controlled double well. The sparse
protocol skips the bridge through the barrier by jumping from -0.8 to 0.8.
That gap is too large for the chosen bias strength and sampled widths.

The design variables interact:

| Choice | Failure if too weak | Failure if too strong |
|---|---|---|
| window spacing | poor adjacent overlap | unnecessary cost if too dense |
| force constant | windows too broad or poorly targeted | windows too narrow to overlap |
| sample length | noisy local histograms | wasted cost if overlap is absent |
| coordinate choice | hidden barriers remain | over-constrained interpretation |

The goal is not maximal overlap everywhere. The goal is enough overlap to
support reconstruction while keeping windows efficient and interpretable.

## What Does WHAM-Style Reconstruction Need?

WHAM-style reconstruction combines biased histograms by accounting for the bias
potential in each window and solving for consistent window free-energy offsets.
The operational requirement is straightforward: bins that determine the PMF
must be supported by windows that connect to each other through overlap.

The current diagnostic is not a full production WHAM implementation tutorial.
It is a controlled reconstruction test. The known PMF supplies an answer key,
so the reconstruction can be judged by barrier error and RMSE. Dense windows
have PMF RMSE about 0.173 versus the known PMF. Sparse windows have RMSE about
0.223. The sparse RMSE is not catastrophic, but the barrier error is large
because the sparse protocol fails in the most important bridge region.

This illustrates a useful diagnostic. A global RMSE can hide a localized
barrier problem. A barrier error can hide local shape errors elsewhere. An
overlap plot can reveal the protocol failure that caused both. A free-energy
review should not rely on a single scalar diagnostic.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_reconstruction.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Dense overlapping windows recover the controlled free-energy profile more accurately than sparse windows. Reconstruction quality follows connectivity, not the number of plotted curves." %}

## Why Are Replicas Useful?

Replica checks ask whether the result depends on which independent samples
were collected. In the full summary, the dense protocol has forward/reverse
replica PMF RMSE about 0.115 and maximum replica mean difference about 0.0021.
The sparse protocol has forward/reverse RMSE about 0.235 and maximum replica
mean difference about 0.0019.

The small mean differences show that local window means are reproducible in
this controlled run. The larger sparse forward/reverse PMF RMSE shows that
global reconstruction can still be less consistent when overlap is poor. Local
window reproducibility is not the same as global PMF reliability.

In production MD, replicas are even more important. Different replicas can
start from different basin histories, cross barriers at different rates, or
fail in different windows. If replicas disagree in a region, the PMF should
show that uncertainty or the protocol should be revised before publication.
That is why a production umbrella result should report convergence evidence,
not only the final reweighted curve (<span id="cite-kaestner2011b"></span>[Kästner, 2011](#ref-kaestner2011)).

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_replicas.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Replica disagreement localizes unstable regions of the reconstructed profile. A global error score can hide these window-specific failures." %}

## What Does the Pair-Distance Diagnostic Add?

The compact pair-distance diagnostic is not a final production umbrella
calculation, but it makes the tutorial less abstract. Instead of only sampling
a generic double-well coordinate, it reconstructs a one-dimensional
pair-interaction PMF along reduced pair distance \(r/\sigma\). The target
profile is a Lennard-Jones contact well shifted to zero at its minimum. Harmonic
umbrella windows then reconstruct that profile from biased pair-distance
samples.

For the full run, eight windows span \(r/\sigma = 0.98\) to
2.35 with force constant 45.0 and 12,000 samples per window. The reconstructed
contact-well depth differs from the known pair PMF by about 0.005, the PMF RMSE
is about 0.138, and the minimum adjacent overlap is about 0.311. Those numbers
are not presented as a production molecular free energy. They are a compact
atomistic-coordinate check that the same diagnostics apply once the
coordinate has physical meaning: window centers, biased means, overlap,
replica disagreement, and runtime provenance must all be reported.

The runtime provenance is intentionally visible. The full profile targets
`cuda_or_cpu_fallback`, but this artifact was generated on
`jax:cpu;devices:cpu`, so production GPU readiness is `false`. The blocking
reason is recorded in the summary rather than left implicit.

## What Are Common Umbrella Mistakes?

The first mistake is treating umbrella centers as sampled coordinates. The
center is a parameter of the bias potential. The sampled distribution is
determined by the bias plus the underlying PMF. Reporting only centers hides
whether the windows actually cover the intended regions.

The second mistake is assuming that many samples per window imply a good PMF.
The sparse protocol has 20000 samples per window in the full run. That is not
enough because the missing bridge is a design problem, not only a sampling
length problem.

The third mistake is reconstructing first and diagnosing later. Overlap,
per-window support, and replica consistency should be checked before the PMF is
interpreted. If an overlap gap exists, the correct response is usually to add
or move windows, change bias strength, or revise the coordinate, not to polish
the plot.

The fourth mistake is reporting only the final PMF curve. A methods section
should include window centers, force constants, sample counts, overlap
diagnostics, reconstruction method, uncertainty method, and any failed or
revised windows.

## How Does This Connect to Hysteresis?

Hysteresis appears when the result depends on sampling direction, initialization,
or history. In umbrella sampling, one common check is whether windows started
from different directions or replicas produce compatible histograms and PMFs.
If a window is trapped in a metastable substate, it may appear locally stable
while failing to represent the intended biased ensemble.

The current controlled diagnostic uses simple one-dimensional biased samples,
so hysteresis is not the main failure mode. The review still records
forward/reverse replica consistency because production protocols should not
skip that habit. A final MD article should discuss hysteresis explicitly when
windows are initialized along a pulling path or when slow orthogonal modes can
remain trapped.

For MLIP workflows, hysteresis can also indicate model problems. If a learned
potential permits one pathway in one direction and a different pathway in
another, the issue may be sampling, coordinate choice, or model validity. The
diagnostic should not collapse those cases into a single smooth PMF.

## How Should Windows Be Initialized?

Window initialization is part of the sampling protocol, not a clerical detail.
For a real MD system, the biased coordinate rarely starts at every desired
window center by accident. A common workflow is to generate starting structures
along a pulling path, by restrained relaxation, or by selecting frames from an
earlier trajectory. Each choice can introduce history into the windows.

The safe habit is to separate three stages. First, generate candidate starting
structures and record how they were produced. Second, equilibrate each biased
window under its own restraint and discard the warmup. Third, sample the
production segment used for WHAM or MBAR. If those stages are mixed together,
the reconstruction can include transients that are not representative of the
intended biased ensembles.

A window initialized from a neighboring window can look smooth while carrying a
memory of the path used to create it. This is especially dangerous when an
orthogonal slow coordinate is hidden from the chosen collective variable. The
sampled coordinate may sit near the target value, but the molecular structure
may still belong to the previous basin. Overlap along the biased coordinate is
necessary, but it is not sufficient to prove that all relevant slow modes are
equilibrated.

For that reason, production umbrella protocols should record initialization
metadata in the same way they record force constants and sample counts. The
question is not only whether a window exists. It is whether the window's
starting structure, warmup, trajectory length, and replica behavior make the
biased ensemble credible.

## How Would This Extend to Production MD?

The final version of this post should connect the controlled double-well
diagnostic to an actual MD umbrella workflow. A practical extension would pick
a simple coordinate from the earlier observable/free-energy posts, define
windows, run biased trajectories, reconstruct the PMF, and report overlap and
replica consistency.

For a real atomistic or MLIP-driven system, that extension would need a written
sampling protocol before any long run is launched. The protocol should name the
collective variable and units, define the admissible coordinate range, choose
initial windows from pilot sampling rather than convenience alone, specify how
starting structures are generated, separate equilibration from production,
state the WHAM or MBAR inputs, and define the revision rule for low-overlap or
high-disagreement windows. The compact pair-distance diagnostic on this page is
useful because it exercises those bookkeeping habits on a physical coordinate,
but it does not replace an atomistic umbrella campaign with equilibrated
windows, model-domain checks, and production uncertainty intervals.

That extension should record:

| Artifact | What it should prove |
|---|---|
| coordinate definition | the biased coordinate is physically meaningful |
| window protocol | centers, force constants, and initialization are explicit |
| overlap diagnostics | adjacent windows form a connected bridge |
| reconstruction | WHAM/MBAR inputs and outputs are reproducible |
| replica checks | independent runs give compatible PMFs |
| model checks | kUPS/MLIP predictions are credible in sampled windows |

The current controlled workflow remains useful as an answer-key diagnostic.
It shows what a window-placement failure looks like without conflating it with
model error, slow MD relaxation, or coordinate ambiguity.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_pair_umbrellas.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The kUPS pair-distance umbrellas reconstruct the Lennard-Jones contact well from biased trajectories. The physical run links window overlap to a molecular coordinate." %}

## What Should Uncertainty Mean Here?

Uncertainty in umbrella sampling should not mean only the noise on the final
plotted curve. There are several distinct questions. Are individual window
histograms noisy? Are adjacent offsets well determined? Do independent replicas
give compatible PMFs? Does the result change when warmup, bin width, or window
spacing changes? These are related but not identical diagnostics.

The controlled post 10 summary already separates some of them. The dense
protocol has lower barrier error and better forward/reverse PMF consistency
than the sparse protocol. That does not mean the dense protocol is final MD
evidence. It means the controlled statistical design is healthy enough for the
answer-key example. A production result would still need block or bootstrap
uncertainty, replica-level confidence intervals, and sensitivity checks for
window placement and reconstruction settings.

Uncertainty should also be localized. A single average error can make a PMF
look reliable even when the barrier region is weakly supported. In an umbrella
review, the uncertainty display should make the fragile coordinate regions
visible. If uncertainty grows near a barrier, endpoint, or low-overlap bridge,
that is a result. It should drive either additional sampling or a more cautious
interpretation.

For MLIP-driven MD, there is a second uncertainty layer. Statistical
uncertainty tells us whether the biased samples are sufficient for the chosen
potential. Model uncertainty asks whether the potential is credible for the
biased configurations. A high-force umbrella can pull structures into regions
that were rare or absent in training data. The final public article should keep
those two checks separate rather than presenting one error bar as if it covers
all failure modes.

## What Should You Do When Overlap Fails?

A low-overlap bridge is not a plotting problem. It is usually a protocol
problem. The first response should be to identify which adjacent pair fails and
where that gap sits on the coordinate. In the sparse protocol here, the
problem is the jump across the barrier region. The natural fix is to add
intermediate windows or change the restraint strength so the biased
distributions share support.

Changing the force constant is not automatically better than adding windows.
A weaker restraint can broaden each window, but it may also allow a window to
wander into multiple regions or produce poor local resolution. A stronger
restraint can localize sampling, but it narrows distributions and can make
overlap worse unless windows are closer together. The useful design variable
is the combined protocol, not any single knob.

When overlap fails in production, rerunning the same sparse protocol for a
longer time may not solve the problem. Longer sampling helps if the tails are
present but noisy. It does little if the biased distributions barely touch.
That distinction is why overlap diagnostics should be checked before
interpreting the final PMF. If the diagnostic says the bridge is missing, the
protocol should be revised before more expensive production runs are launched.

The revision should be recorded. A publication-quality workflow should show
which pilot windows failed, which windows were added or moved, and which final
diagnostics justify accepting the revised protocol. Hidden failed pilots are a
source of bias, especially when the final PMF is used to support a mechanistic
claim.

## Run the Example

The smoke profile follows the same protocol with a smaller CPU workload:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 10 --profile smoke
uv run kups-tutorial verify 10 --profile smoke
```

The repository also contains the [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/full.json), [notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/notebooks), and [recorded results](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/results/post-10/full).

## References

- <span id="ref-torrie1977"></span>Torrie, G. M. & Valleau, J. P. (1977). Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling. *Journal of Computational Physics*, 23, 187-199. <a href="#cite-torrie1977" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021. <a href="#cite-kumar1992" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shirts2008"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. <a href="#cite-shirts2008" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kaestner2011"></span>Kästner, J. (2011). Umbrella sampling. *WIREs Computational Molecular Science*, 1, 932-942. <a href="#cite-kaestner2011" class="reversefootnote" role="doc-backlink">↩</a> <a href="#cite-kaestner2011b" class="reversefootnote" role="doc-backlink">↩</a>
