---
layout: post
permalink: /kups-md-tutorials/post-04-thermostats/
title: "What Does a Thermostat Really Do?"
date: 2026-07-14
last_updated: 2026-08-04
description: "Derive BAOAB Langevin dynamics, implement it transparently in JAX, and compare real kUPS BAOAB and CSVR trajectories."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 4
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "An executable introduction from physical ideas to JAX algorithms and kUPS simulations."
series_order: 4
categories: [science]
tags: [molecular-dynamics, thermostats, langevin, baoab, csvr, jax, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

A thermostat is not a correction that pushes a temperature display toward a
number. It changes momenta throughout the trajectory. With the right balance
of friction and noise, that altered dynamics samples the canonical ensemble.
With a different coupling strength or algorithm, it can sample similar static
distributions while producing different time correlations.

That distinction decides whether a trajectory can support a claim. Mean
energy, structure, and other equilibrium averages need the right ensemble.
Diffusion, spectra, and rate estimates also depend on how motion unfolds in
time. We will make both requirements visible: first in a transparent JAX
implementation of BAOAB Langevin dynamics, then in real BAOAB and CSVR kUPS
runs.

<div class="kups-learning-box" markdown="1">
<div class="kups-learning-box__title">What you will learn</div>

- why temperature is a fluctuating kinetic-energy estimator, not a controlled
  state variable at every step;
- how friction and random force combine through fluctuation--dissipation;
- what the five letters in BAOAB do to positions and momenta;
- how BAOAB and CSVR can share an ensemble target but alter dynamics
  differently;
- which evidence is needed before using a thermostatted trajectory for a
  static or dynamical observable.

**Prerequisites:** the state and temperature estimator from
[Post 01]({{ '/kups-md-tutorials/post-01-initialization/' | relative_url }}), the integrator
map from [Post 02]({{ '/kups-md-tutorials/post-02-integrators/' | relative_url }}), and the
error distinctions from [Post 03]({{ '/kups-md-tutorials/post-03-errors/' | relative_url }}).
</div>

## NVE and NVT answer different physical questions

For positions $$\mathbf R$$, momenta $$\mathbf P$$, and Hamiltonian

$$
H(\mathbf R,\mathbf P)
= U(\mathbf R)
+ \sum_i \frac{\lVert\mathbf p_i\rVert^2}{2m_i},
$$

an ideal microcanonical trajectory keeps $$H$$ fixed. It represents an
isolated system with fixed particle number, volume, and energy: the NVE
ensemble.

A system in thermal contact with a much larger heat bath can exchange energy.
At temperature $$T$$, its canonical density is

$$
\pi(\mathbf R,\mathbf P)
= \frac{1}{Z}
  \exp\!\left[-\beta H(\mathbf R,\mathbf P)\right],
\qquad
\beta=\frac{1}{k_{\mathrm B}T}.
$$

This is the NVT ensemble. A thermostat modifies the equations of motion so
that this density is stationary. It should not conserve the system's ordinary
total energy: energy must enter and leave through the modelled bath.

The practical question is therefore not "does energy drift under NVT?" It is
"does the stationary trajectory represent the intended canonical density, and
what dynamical information survives the coupling?"

## Temperature is a noisy observable

Classical MD infers an instantaneous kinetic temperature from

$$
K=\sum_i\frac{\lVert\mathbf p_i\rVert^2}{2m_i},
\qquad
T_{\mathrm{inst}}=\frac{2K}{f k_{\mathrm B}},
$$

where $$f$$ is the number of active degrees of freedom. If center-of-mass
translation was removed, three degrees are usually subtracted. Constraints
remove more.

Even a perfect canonical sampler does not make $$T_{\mathrm{inst}}=T$$ in
every frame. In the canonical ensemble,

$$
\frac{2K}{k_{\mathrm B}T}\sim\chi_f^2,
\qquad
\langle K\rangle=\frac{f}{2}k_{\mathrm B}T.
$$

The fluctuations are part of the ensemble. A controller that clamps kinetic
energy exactly can suppress the distribution it is meant to sample. We care
about the distribution and its correlations, not whether a temperature curve
looks smooth.

## Langevin dynamics balances forgetting and noise

In momentum form, Langevin dynamics is

$$
d\mathbf R_i=\frac{\mathbf p_i}{m_i}\,dt,
$$

$$
d\mathbf p_i
= \mathbf F_i(\mathbf R)\,dt
- \gamma\mathbf p_i\,dt
+ \sqrt{2\gamma m_i k_{\mathrm B}T}\,d\mathbf W_i.
$$

The friction $$-\gamma\mathbf p_i$$ erases momentum memory. The Wiener
increment $$d\mathbf W_i$$ adds random momentum. Their amplitudes cannot be
chosen independently: fluctuation--dissipation is what leaves the Maxwell
momentum distribution stationary.

The timescale $$\gamma^{-1}$$ gives useful intuition. Weak friction allows
long ballistic or oscillatory memory. Strong friction rapidly randomizes
momenta, but positions can then diffuse slowly through an overdamped
landscape. Stronger coupling is not automatically faster sampling.

## BAOAB makes the heat exchange explicit

BAOAB splits one timestep into five exactly named operations
(<span id="cite-leimkuhler2013"></span>[Leimkuhler & Matthews,
2013](#ref-leimkuhler2013)). Let
$$c=\exp(-\gamma\Delta t)$$ and draw a fresh standard normal vector
$$\boldsymbol\xi_n$$:

$$
\begin{aligned}
\mathbf p^{(1)}
  &= \mathbf p_n + \frac{\Delta t}{2}\mathbf F(\mathbf R_n),
  &&\text{B: force kick},\\
\mathbf R^{(1)}
  &= \mathbf R_n + \frac{\Delta t}{2}\frac{\mathbf p^{(1)}}{\mathbf m},
  &&\text{A: half drift},\\
\mathbf p^{(2)}
  &= c\mathbf p^{(1)}
   + \sqrt{\mathbf m k_{\mathrm B}T(1-c^2)}\,\boldsymbol\xi_n,
  &&\text{O: heat bath},\\
\mathbf R_{n+1}
  &= \mathbf R^{(1)} + \frac{\Delta t}{2}\frac{\mathbf p^{(2)}}{\mathbf m},
  &&\text{A: half drift},\\
\mathbf p_{n+1}
  &= \mathbf p^{(2)} + \frac{\Delta t}{2}\mathbf F(\mathbf R_{n+1}),
  &&\text{B: force kick}.
\end{aligned}
$$

Only the O step exchanges heat. Its variance explains the noise amplitude. If
$$\operatorname{Var}(\mathbf p)=\mathbf m k_{\mathrm B}T$$ before the step,
then

$$
\operatorname{Var}(c\mathbf p+\sigma\boldsymbol\xi)
=c^2\mathbf m k_{\mathrm B}T
+\mathbf m k_{\mathrm B}T(1-c^2)
=\mathbf m k_{\mathrm B}T.
$$

The deterministic B and A operations are the same force and drift ideas used
by velocity Verlet. The O operation is the new physics.

## Write those five operations in JAX

The collapsed setup chooses the CPU backend, enables float64 for the analytic
control, and imports the real kUPS workflow.

{% include kups-notebooks/post-04/post04-setup.html %}

The open cell below is the algorithm rather than a wrapper around it. The PRNG
key is explicit, `jax.random.split` supplies one independent key per step, and
`jax.lax.scan` carries the immutable state. Forces still come from
`-jax.grad(energy_fn)`.

{% include kups-notebooks/post-04/post04-jax-baoab.html %}

For the unit harmonic oscillator, the exact canonical targets are
$$\operatorname{Var}(q)=\operatorname{Var}(p)=1$$ and
$$\langle K\rangle=1/2$$. The moderate-coupling run reports 0.997, 1.000, and
0.500. The weak and strong runs deviate more in this finite record: their
position variances are 1.102 and 0.862.

Those deviations do not by themselves rank the thermostats. The saved points
are correlated, so 3,500 stored values are fewer than 3,500 independent
draws. In the longer committed control, the position effective sample count is
about 348 for weak coupling but only 66 for strong coupling. Uncertainty must
use effective samples, not the raw frame count.

## BAOAB and CSVR do not modify momentum the same way

BAOAB applies an independent Ornstein--Uhlenbeck refresh to each momentum
component. Canonical stochastic velocity rescaling (CSVR) instead evolves the
total kinetic energy with a stochastic process and applies one shared scale
factor,

$$
\mathbf p_i' = \alpha\mathbf p_i,
$$

to all momenta. The random construction of $$\alpha$$ preserves the canonical
kinetic-energy distribution rather than forcing one deterministic kinetic
energy (<span id="cite-bussi2007"></span>[Bussi et al.,
2007](#ref-bussi2007)).

<div class="table-responsive" markdown="1">

| Property | BAOAB Langevin | CSVR |
|---|---|---|
| Bath action | component-wise friction and noise | one stochastic global rescaling |
| Main coupling parameter | friction $$\gamma$$ | relaxation time $$\tau_T$$ |
| Canonical target | positions and momenta | kinetic energy, coupled to MD positions |
| Dynamical effect | local momentum decorrelation | global kinetic-mode coupling |
| Same trajectories expected? | no | no |

</div>

Both can be valid canonical samplers. Neither gives thermostat-independent
real-time dynamics merely because the mean temperature is correct.

## Run both thermostats through kUPS

The next cell performs two new 32-atom CPU runs through
`kups.application.simulations.md.run`. It reopens each HDF5 file and prints the
integrator, frame and atom counts, block temperature estimate, observed
device, and content hash. It also prints the committed full-profile record.

{% include kups-notebooks/post-04/post04-kups-thermostats.html %}

The smoke cases produce only eight stored frames. Their 92.55 K BAOAB and
94.06 K CSVR estimates confirm executable, finite HDF5 analysis; they are not
convergence evidence for a 100 K target.

The quantitative evidence comes from 256 atoms, 20,000 warmup steps, 20,000
production steps, and 1,000 stored frames per thermostat:

<div class="table-responsive" markdown="1">

| Full kUPS case | Coupling | Mean temperature | Stored frames | Observed device |
|---|---:|---:|---:|---|
| BAOAB Langevin | $$\gamma=0.01\ \mathrm{fs}^{-1}$$ | 99.970 ± 0.279 K | 1,000 | NVIDIA RTX A5000 |
| CSVR | $$\tau_T=100\ \mathrm{fs}$$ | 99.851 ± 0.424 K | 1,000 | NVIDIA RTX A5000 |

</div>

The ± values are block standard errors of the mean, not the width of
instantaneous temperature fluctuations. Both means support a narrow claim:
under these protocols, the low-order kinetic-temperature estimate is
consistent with 100 K. They do not establish identical configurational
sampling, efficiency, or dynamics.

## See where the bath enters an atomic trajectory

The top row of the next figure turns the equations into a state transition:
B changes momentum through force, A moves the atom, and O replaces part of the
old momentum with a thermal draw. The lower panels use actual atom positions
from the two full kUPS HDF5 files.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post04_thermostat_mechanism.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Five BAOAB momentum and position substeps above atom-level trails from full kUPS BAOAB and CSVR trajectories" caption="BAOAB inserts one heat-bath momentum update between deterministic kicks and drifts. The lower panels unwrap periodic displacements for 24 atoms near the final z=0 plane over the last 1 ps of each 256-atom full trajectory; darker points are the final positions. The BAOAB and CSVR cases use different configured seeds, so the trail shapes illustrate atom-level motion and are not a paired pathwise comparison." %}

The trails are short because a solid-like argon atom moves around its local
environment over this window. A thermostat changes those motions through
momenta, but an endpoint picture cannot validate an ensemble. Distributional
and correlation diagnostics remain necessary.

## Correlation determines how much information was collected

For a sampled observable $$A_n$$ with normalized autocorrelation $$\rho(k)$$,
a common integrated autocorrelation estimate is

$$
\tau_{\mathrm{int}}
=1+2\sum_{k=1}^{k_{\max}}\rho(k),
\qquad
N_{\mathrm{eff}}\approx\frac{N}{\tau_{\mathrm{int}}}.
$$

In the committed oscillator control, increasing $$\gamma$$ from 0.1 to 5.0
reduces one-step velocity correlation from 0.958 to 0.355. Yet the position
autocorrelation time grows from 10.1 to 52.7 saved samples. Momentum forgets
quickly while the coordinate explores slowly: the overdamped regime.

This is why a thermostat should be tuned against the observable of interest.
Temperature relaxation alone cannot reveal sampling efficiency.

## Choose a production protocol from the claim

For equilibrium averages, a validated NVT production run can be appropriate.
Check kinetic moments, configurational observables, equilibration, correlation
times, and independent replicas.

For diffusion coefficients, velocity autocorrelation functions, vibrational
spectra, or rates, thermostat perturbations can enter the answer directly. A
common design is:

1. equilibrate positions and momenta under NVT;
2. transfer that exact final state into NVE;
3. verify NVE energy behavior;
4. compute the dynamical observable over multiple independent segments.

The current real kUPS evidence in this chapter validates separate NVT runs; it
does not demonstrate an exact kUPS NVT-to-NVE state handoff. A reduced-unit
control exists in the audit artifacts, but it is not a substitute for that
production operation.

The same caution applies to machine-learned potentials. A strong thermostat
can continually remove energy injected by noisy or extrapolative forces. A
stable temperature is therefore not evidence that the force model is valid.

## Check your understanding

1. If $$\gamma$$ doubles, which BAOAB line changes directly? Which two terms
   must change together to preserve the same momentum variance?
2. Why is a perfectly flat instantaneous-temperature trace suspicious in an
   NVT simulation?
3. Two thermostats give the same mean temperature. What additional checks are
   needed before comparing a diffusion coefficient?
4. Why can 3,500 stored oscillator states contain only about 66 effective
   position samples?

The first question is the algorithmic core: friction without matching noise
cools the system, while noise without matching friction heats it.

## A thermostat defines sampling dynamics, not just temperature

BAOAB makes the intervention easy to locate. Four substeps follow mechanics;
one substep exchanges momentum with a bath. CSVR performs a different
stochastic intervention through global kinetic energy. Both choices can reach
the canonical target, but they need not preserve the same temporal path.

A defensible thermostat claim therefore names the ensemble, algorithm,
coupling time, equilibration period, sampled observable, correlation analysis,
and hardware-backed execution evidence. Post 05 will add a second bath
variable—the simulation cell—and ask the same questions about pressure.

<details class="kups-reproducibility" markdown="1">
<summary>Reproducibility record and complete diagnostic dashboard</summary>

Run and verify the CPU profile from the locked environment:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked

uv run kups-tutorial run 04 --profile smoke
uv run kups-tutorial verify 04 --profile smoke
uv run kups-tutorial verify-notebooks --posts 04 --output-dir notebook-runs
uv run kups-tutorial export-notebook-cells \
  --executed-notebooks-dir notebook-runs \
  --site-root ../sungsoo-ahn.github.io --posts 04 --check
```

The full audit dashboard retains the canonical variance, kinetic moment,
position-memory, and real kUPS temperature panels:

{% include figure.liquid loading="lazy" path="assets/img/blog/kups_md_post04_thermostat_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true alt="Four-panel thermostat audit for canonical moments, autocorrelation, and real kUPS temperature" caption="The analytic oscillator checks canonical position, momentum, and kinetic moments while exposing the loss of effective position samples under strong coupling. The final panel reports the two 1,000-frame full kUPS GPU temperature estimates." %}

Source and evidence:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/smoke.json)
- [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-04/full.json)
- [smoke compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/smoke/kups_md_summary.json)
- [full compact summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/kups_md_summary.json)
- [full provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-04/full/manifest.json)
- [executed notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-04-thermostats.ipynb)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post04_figures.py)
- [self-review note](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-04.md)
- [source repository](https://github.com/sungsoo-ahn/kups-md-tutorials)

</details>

## References

- <span id="ref-leimkuhler2013"></span>Leimkuhler, B. & Matthews, C. (2013). Rational construction of stochastic numerical methods for molecular sampling. *Applied Mathematics Research eXpress*, 2013(1), 34–56. <a href="#cite-leimkuhler2013" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bussi2007"></span>Bussi, G., Donadio, D. & Parrinello, M. (2007). Canonical sampling through velocity rescaling. *Journal of Chemical Physics*, 126, 014101. <a href="#cite-bussi2007" class="reversefootnote" role="doc-backlink">↩</a>
