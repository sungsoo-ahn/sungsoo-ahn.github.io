---
layout: post
permalink: /kups-md-tutorials/post-10-umbrella-sampling/
title: "What Does Umbrella Sampling Actually Sample?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Build a harmonic umbrella into a kUPS potential, run biased Ar-pair trajectories, and reject PMFs without connected windows or replica agreement."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 10
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 10
categories: [science]
tags: [molecular-dynamics, umbrella-sampling, free-energy, wham, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable draft is hidden from site navigation until the full kUPS MD series passes its release review. The double-well example is a known-answer control. The Ar-pair result comes from real kUPS trajectories on the devices reported below.</em>
</p>

## A Smooth PMF Is Not the Result

Umbrella sampling makes a hard trajectory easier by changing the ensemble. It
does not make the sampling problem disappear.

Each window sees a different biased potential. A reconstruction such as WHAM
can align those windows only where their sampled distributions overlap. If one
neighboring pair does not share probability mass, no estimator can invent the
missing bridge. Twenty thousand frames in every disconnected window are still
disconnected data.

This post makes that failure concrete in two stages:

1. a double-well answer key holds the target PMF fixed and removes windows;
2. a physical Ar--Ar coordinate composes a harmonic restraint with a kUPS
   Lennard--Jones potential and runs 16 independent GPU trajectories.

The executable artifacts are the
[smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/smoke.json),
[full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/full.json),
[notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-10-umbrella-sampling.ipynb),
[smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/smoke/umbrella_summary.json),
[production summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/umbrella_summary.json),
[stored distance samples](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/kups_umbrella_samples.csv),
[provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/manifest.json),
[kUPS worker](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/kups_umbrella_worker.py),
[figure source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/figures.py),
and [review record](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-10.md).

{% include kups-notebooks/post-10/setup.html %}

## A Window Samples the Biased Ensemble

Let $$r$$ be the collective variable and $$F(r)$$ its unbiased free energy. A
window centered at $$r_i$$ adds

$$
U_i^{\mathrm{bias}}(r)=\frac{K}{2}(r-r_i)^2.
$$

The window therefore samples

$$
p_i(r) \propto
\exp\left[-\beta\left(F(r)+U_i^{\mathrm{bias}}(r)\right)\right].
$$

The center $$r_i$$ is a parameter, not the expected sample mean. The underlying
free-energy slope pulls the distribution away from the center; boundaries and
hidden slow coordinates can distort it further. A table of window centers is a
protocol description, not coverage evidence.

kUPS defines its harmonic bond energy as $$k(r-r_0)^2$$. The tutorial defines
the more common $$K(r-r_0)^2/2$$ form, so the worker passes $$k=K/2$$. The
notebook evaluates that convention directly:

{% include kups-notebooks/post-10/harmonic-bias.html %}

That factor of two is small enough to miss in prose and large enough to change
every sampled distribution. It belongs in executable code.

## Composition Needs Force Tests

The production worker builds the base Lennard--Jones potential with kUPS,
builds a fixed-edge harmonic potential for particles 0 and 1, and combines the
two with `sum_potentials`. It then asks two questions before running MD:

- Does a zero-strength restraint leave the base energy and position gradient
  unchanged?
- Does the nonzero restraint gradient agree with a central finite difference?

The full GPU run reports zero energy error and zero maximum gradient error for
the first check. The nonzero-bias finite-difference error is
$$2.39\times10^{-6}$$ eV/Å. These are implementation tests, not convergence
diagnostics. Passing them says that the intended Hamiltonian was composed
correctly; it says nothing yet about whether the windows sampled it well.

The fresh-kernel notebook goes further than loading a JSON file. Its smoke cell
launches eight isolated kUPS workers, reads minimum-image pair distances from
their HDF5 trajectories, runs WHAM, and invokes the same verification gate used
by the command-line workflow:

{% include kups-notebooks/post-10/smoke-run.html %}

The smoke profile is deliberately CPU-sized. It proves the path executes. The
publication record below must pass a separate GPU-only gate.

## Sparse Windows Fail with Plenty of Samples

The answer-key control uses the same double-well PMF, temperature, force
constant, and sample count for two protocols. Only the window grid changes.

<div class="table-responsive" markdown="1">

| Protocol | Windows | Minimum adjacent overlap | Barrier error | PMF RMSE | Replica PMF RMSE |
|---|---:|---:|---:|---:|---:|
| dense | 9 | 0.3552 | 0.0106 | 0.1730 | 0.1148 |
| sparse | 4 | 0.0003 | -0.2554 | 0.2229 | 0.2352 |

</div>

The sparse protocol jumps from $$-0.8$$ to $$0.8$$ across the barrier. Its
weakest overlap is essentially zero. WHAM still returns a curve, but the
barrier is underestimated by 0.255. The estimator did not fail to run. The
sampling design failed to identify the relative offset across the gap.

This is why a single global error is not enough. The sparse PMF RMSE is only
about 29% worse than the dense value, while its barrier error is 24 times
larger. The overlap plot identifies the cause and location of the failure.
WHAM and MBAR are powerful ways to combine connected states; neither can
recover probability mass that no state sampled.<sup id="cite-wham"><a href="#ref-wham">1</a></sup>
<sup id="cite-mbar"><a href="#ref-mbar">2</a></sup>

## The Physical Coordinate Has a Measure

For the physical experiment, two argon atoms occupy a periodic 30 Å cube. The
base interaction is Lennard--Jones with
$$\sigma=3.405$$ Å and $$\epsilon=0.010326$$ eV. Eight harmonic windows span
3.40--7.95 Å at 100 K with $$K=0.015$$ eV/Å$$^2$$.

There is an important trap here. A histogram of three-dimensional pair distance
does not reconstruct the bare pair potential $$U_{\mathrm{LJ}}(r)$$. Spherical
shells contribute a factor of $$r^2$$, so

$$
p(r) \propto r^2 e^{-\beta U_{\mathrm{LJ}}(r)}
$$

and the radial PMF is

$$
F_r(r)=U_{\mathrm{LJ}}(r)-2k_{\mathrm B}T\log r+C.
$$

The workflow compares WHAM to this radial answer key. To assess the recovered
Lennard--Jones well, it adds the $$2k_{\mathrm B}T\log r$$ measure term back
before measuring the well depth. Comparing the raw radial PMF directly with
the bare interaction would mix statistical error with a known coordinate
Jacobian.

## The Production Record

Each full-profile window has two independently seeded BAOAB Langevin
trajectories. Every trajectory uses 3,000 warmup steps followed by 8,000
production steps at 1 fs; storing every tenth step leaves 800 frames per run.
The complete record therefore contains 8 windows × 2 replicas × 800 frames =
12,800 stored frames.

{% include kups-notebooks/post-10/production-evidence.html %}

The acceptance numbers are:

<div class="table-responsive" markdown="1">

| Check | Full-profile value | Gate |
|---|---:|---:|
| observed device | NVIDIA RTX A5000 | GPU required |
| minimum adjacent overlap | 0.4550 | > 0.20 |
| radial-PMF RMSE | 0.00359 eV | ≤ 0.01 eV |
| independent-replica PMF RMSE | 0.00407 eV | ≤ 0.01 eV |
| corrected LJ well-depth error | 0.00296 eV | ≤ 0.005 eV |
| largest window replica-mean shift | 0.326 Å | ≤ 0.50 Å |
| zero-bias energy / gradient error | 0 / 0 | near zero |
| bias force finite-difference error | $$2.39\times10^{-6}$$ eV/Å | ≤ $$10^{-4}$$ eV/Å |

</div>

Those gates were not chosen after viewing only the final curve. The first
smoke protocol exposed a zero-overlap bridge. A later full run passed the loose
numerical gates but showed a large replica-mean shift in one soft window. The
Langevin friction was reduced from 0.01 to 0.002 fs$$^{-1}$$ to improve
coordinate relaxation, and both profiles were rerun. Minimum overlap increased
to 0.455 and independent-replica PMF disagreement fell to 0.00407 eV. The
failed pilots changed the protocol; they were not polished out of the story.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_umbrella_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Post 10 diagnostics. The top and lower-left panels are known-answer controls showing how a disconnected sparse grid corrupts a PMF. The lower-middle panel reconstructs the radial Ar-pair PMF from 16 real kUPS GPU trajectories. The final panel records the execution and force-test evidence." %}

The physical reconstruction is intentionally plotted beside the analytic
radial PMF, not as an isolated attractive curve. The overlap bars, replica
difference, device, frame count, and force checks are part of the result.

## What the Hashes Prove

Raw HDF5 trajectories are too large for the site repository, but they do not
vanish from the evidence chain. The production summary contains one record per
window and replica with:

- input CIF SHA-256;
- raw kUPS HDF5 SHA-256 and byte count;
- stable dataset names, shapes, and dtypes;
- seed, window center, frame count, and atom count;
- observed JAX device and elapsed time;
- thermodynamic block summaries and force-test results.

The compact distance CSV retains every stored pair distance used by WHAM. The
manifest hashes that CSV, the summary, the PMF curves, and the window table.
Verification rejects a malformed raw hash, missing HDF5 schema field, frame
count drift, output hash mismatch, CPU full run, broken overlap, failed force
test, or excessive replica disagreement.

A hash does not make a simulation correct. It makes the evidence identifiable.

## When a Window Grid Fails

Low overlap is a protocol problem before it is an estimator problem. The
response depends on the diagnostic:

<div class="table-responsive" markdown="1">

| Symptom | Likely issue | Useful response |
|---|---|---|
| adjacent overlap near zero | windows too far apart or too stiff | add/move windows or broaden restraints |
| mean far from center | strong PMF slope, boundary, or poor relaxation | inspect the distribution; retune center, bias, or equilibration |
| replicas have shifted means | slow relaxation or hidden states | extend/revise equilibration and inspect orthogonal coordinates |
| local histograms look stable but PMFs disagree | offsets are weakly identified | improve the overlap network; do not average away the conflict |
| statistical checks pass but forces are wrong | bias composition bug | stop; fix zero-bias and finite-difference tests |
| biased structures leave the MLIP domain | model extrapolation | add model checks or change the scientific scope |

</div>

Longer sampling helps when overlapping tails exist but are noisy. It does not
repair a geometric gap between biased ensembles. A weaker spring is not always
the answer either: it broadens windows but can reduce local control and expose
new slow modes. Window spacing, restraint strength, initialization, warmup,
and production length form one design.

## What to Report

A defensible umbrella result should let a reader answer these questions:

- What coordinate was biased, in what units, and with what periodic-image
  convention?
- What bias function, centers, and force constants were used?
- How were starting structures generated, and which frames were discarded as
  warmup?
- Do adjacent windows form a connected overlap network?
- Do independent replicas agree locally and after reconstruction?
- Was the radial or configurational measure handled correctly?
- Which configurations carry the free-energy inference, and is the MLIP
  credible there?
- Can every compact result be traced to a config, device record, and raw-file
  hash?

The PMF line is the last item produced by that evidence chain, not a substitute
for it. Practical umbrella reviews make the same point: coordinate choice,
window protocol, reconstruction, and convergence evidence belong together.<sup id="cite-kaestner"><a href="#ref-kaestner">3</a></sup>

## Reproduce It

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync

uv run kups-tutorial run 10 --profile smoke
uv run kups-tutorial verify 10 --profile smoke

# Requires a working JAX GPU runtime and launches 16 kUPS trajectories.
uv run kups-tutorial run 10 --profile full
uv run kups-tutorial verify 10 --profile full

uv run kups-tutorial verify-notebooks --posts 10
uv run python scripts/generate_post10_figures.py
```

The notebook completed from a fresh kernel in 106.5 seconds without changing
its source. Its five code cells all executed and produced four outputs. The
full production workflow observed `gpu:NVIDIA RTX A5000`; there is no CPU
fallback accepted for this post.

## Takeaway

An umbrella is not evidence of coverage. It is a rule for collecting biased
evidence. Trust the reconstruction only when the implemented bias preserves the
base force field, neighboring windows connect, independent replicas agree, the
coordinate measure is handled correctly, and the sampled configurations remain
scientifically credible.

## References

1. <span id="ref-wham"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011--1021. <a href="#cite-wham" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-mbar"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. <a href="#cite-mbar" class="reversefootnote" role="doc-backlink">↩</a>
3. <span id="ref-kaestner"></span>Kästner, J. (2011). Umbrella sampling. *WIREs Computational Molecular Science*, 1, 932--942. <a href="#cite-kaestner" class="reversefootnote" role="doc-backlink">↩</a>
