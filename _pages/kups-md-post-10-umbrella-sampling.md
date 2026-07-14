---
layout: post
permalink: /kups-md-tutorials/post-10-umbrella-sampling/
title: "What Does Umbrella Sampling Actually Sample?"
date: 2026-07-14
last_updated: 2026-07-14
description: "A reproducible umbrella-sampling diagnostic for biased windows, adjacent overlap, WHAM-style reconstruction, and sparse-window failure modes."
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
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post follows the free-energy-estimator discussion by asking what biased umbrella windows actually sample, how adjacent overlap controls reconstruction, and how sparse windows can fail even with many samples. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>

## Introduction

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
2008](#ref-shirts2008)).

This draft demonstrates the executable slice of the tenth tutorial with a
known one-dimensional double-well PMF. Dense and sparse umbrella protocols are
run against the same answer key, so the diagnostic can isolate overlap and
window placement from physical-model error.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-10/full.json)
- [umbrella notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-10-umbrella-sampling.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/smoke/umbrella_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/umbrella_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-10/full/manifest.json)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-10.md)

## What Is Being Sampled?

The current diagnostic compares two harmonic umbrella protocols on the same
double-well coordinate:

| Protocol | Windows | Minimum adjacent overlap | Barrier error |
|---|---:|---:|---:|
| dense_windows | 9 | 0.3552 | 0.0106 |
| sparse_windows | 4 | 0.0003 | -0.2554 |

Both protocols draw many samples from every biased window. The difference is
whether those biased samples connect neighboring regions of the collective
variable.

## What Should The Diagnostic Show?

The full run compares the known PMF to dense and sparse WHAM-style
reconstructions. It also records adjacent histogram overlap and per-window
sampling means. The sparse protocol intentionally skips the bridge through the
barrier region, making the reconstruction less reliable even though every
window has local support.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post10_umbrella_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Umbrella-sampling diagnostics for the committed full profile. Dense windows maintain adjacent overlap and recover the known PMF, while sparse windows leave a near-zero-overlap bridge and bias the reconstructed barrier downward." %}

## Reproduction

The current executable path is:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 10 --profile smoke
uv run kups-tutorial verify 10 --profile smoke
uv run kups-tutorial run 10 --profile full
uv run kups-tutorial verify 10 --profile full
uv run jupyter execute notebooks/post-10-umbrella-sampling.ipynb --inplace
```

The notebook is deliberately not the implementation source. It imports the
configuration loader, umbrella diagnostics, and figure generator from
`src/kups_md_tutorials/`.

## Current Status

This page is not the final article. The implemented pieces are:

- smoke and full controlled umbrella-sampling workflows
- committed compact summaries, PMF curves, and window-overlap outputs
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- final 3,500-10,000-word article prose
- rendered desktop and mobile page snapshots
- production MD context and final uncertainty diagnostics
- final citation pass

The rule for this post is that umbrella sampling is only as trustworthy as the
biased ensembles that connect the coordinate. Window placement, overlap, and
replica consistency are part of the result, not optional diagnostics.

## References

- <span id="ref-torrie1977"></span>Torrie, G. M. & Valleau, J. P. (1977). Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling. *Journal of Computational Physics*, 23, 187-199. <a href="#cite-torrie1977" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kumar1992"></span>Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H. & Kollman, P. A. (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. *Journal of Computational Chemistry*, 13, 1011-1021. <a href="#cite-kumar1992" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shirts2008"></span>Shirts, M. R. & Chodera, J. D. (2008). Statistically optimal analysis of samples from multiple equilibrium states. *The Journal of Chemical Physics*, 129, 124105. <a href="#cite-shirts2008" class="reversefootnote" role="doc-backlink">↩</a>
