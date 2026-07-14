---
layout: post
permalink: /kups-md-tutorials/post-12-mlip-capstone/
title: "What Changes When the Potential Is a Machine-Learned Interatomic Potential?"
date: 2026-07-14
last_updated: 2026-07-14
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
<em>Note: This is an early draft page for the executable kUPS MD tutorial series. It is intentionally hidden from site navigation while the simulations, notebooks, figures, and review artifacts mature. This post closes the series by asking what changes when the potential is a machine-learned interatomic potential rather than a fixed analytic or classical model. The current diagnostic is a deterministic surrogate for MLIP reliability checks; the final public article must replace the placeholder MACE artifact metadata with a verified model hash from the GPU production pass. Corrections and replication issues should be tracked in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
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

This draft demonstrates the executable slice of the twelfth tutorial with
three fcc-Al reliability regimes. The configured model artifact is recorded as
`mace-mp-0-medium` from `ACEsuit/mace`, but the revision and hash are
placeholders until the final GPU artifact pass.

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/full.json)
- [MLIP capstone notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-12-mlip-capstone.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/smoke/mlip_summary.json)
- [full summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/mlip_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/manifest.json)
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

## What Should The Diagnostic Show?

The full run checks three ideas. Static force metrics worsen as the case leaves
the in-domain regime. Dynamics and extrapolation metrics expose failure modes
that static force error alone does not explain. Uncertainty calibration must be
checked against realized force errors rather than treated as a decorative model
output.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post12_mlip_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="MLIP reliability diagnostics for the committed full profile. Static errors, dynamics drift, extrapolation flags, neighbor-list risk, and uncertainty calibration must be reviewed together before trusting MD or free-energy claims from a learned potential." %}

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
- committed compact summaries and diagnostic samples
- executable notebook
- generated SVG/PNG figure and snapshot review
- self-review note covering code, science, notebook, and figure feedback

The missing pieces are:

- real MACE/fcc-Al GPU production run
- verified MACE artifact revision and hash
- final 3,500-10,000-word article prose
- rendered desktop and mobile page snapshots
- final citation pass

The rule for this post is that an MLIP is part of the simulation method, not a
drop-in oracle. Provenance, extrapolation, drift, and uncertainty diagnostics
are part of the scientific result.

## References

- <span id="ref-behler2007"></span>Behler, J. & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98, 146401. <a href="#cite-behler2007" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bartok2010"></span>Bartok, A. P., Payne, M. C., Kondor, R. & Csanyi, G. (2010). Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. *Physical Review Letters*, 104, 136403. <a href="#cite-bartok2010" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-batzner2022"></span>Batzner, S. et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13, 2453. <a href="#cite-batzner2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-batatia2022"></span>Batatia, I. et al. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *NeurIPS Workshop*. <a href="#cite-batatia2022" class="reversefootnote" role="doc-backlink">↩</a>
