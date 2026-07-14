---
layout: page
permalink: /kups-md-tutorials/
title: kUPS MD Tutorials
description: Executable molecular-dynamics tutorials for MLIP-aware machine-learning researchers.
nav: false
nav_order: 4
---

This page tracks the executable tutorial repository for a planned series on
molecular-dynamics practice with
[kUPS](https://github.com/cusp-ai-oss/kUPS).

The material is aimed at machine-learning researchers who already know
machine-learned interatomic potentials, force fields, and the equations of MD,
but want a practical understanding of initialization, integrators, ensemble
control, uncertainty, free-energy estimation, enhanced sampling, and MLIP
failure modes.

## Current repository

- Repository: [sungsoo-ahn/kups-md-tutorials](https://github.com/sungsoo-ahn/kups-md-tutorials)
- Status: early scaffold
- Python: 3.13
- Primary simulation library: kUPS 1.0.3
- Current plan: twelve executable posts, from initialization through enhanced
  sampling and an MLIP aluminum capstone

## Local reproduction

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run pytest
```

The repository is the source of truth for simulations, notebooks, tests,
configuration, compact numerical summaries, and figure-generation code. Final
articles and publication-ready assets will live on this website.
