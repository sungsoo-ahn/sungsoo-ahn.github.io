---
layout: post
title: "Molecular Simulation with Machine-Learned Force Fields"
date: 2026-08-08
last_updated: 2026-08-08
description: "How learned potential-energy surfaces become molecular dynamics, why rollout stability differs from static accuracy, and how to validate the resulting scientific observables."
abstract: >
  A machine-learned force field is useful only after it survives repeated integration, stays inside a physically covered configuration space, and reproduces the ensemble observables that motivated the simulation.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [molecular-science]
lecture_paths: [ml4mol]
tags: [molecular-dynamics, machine-learned-force-fields, active-learning, uncertainty-quantification, thermodynamic-observables]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the molecular-simulation storyline from my 2025 Machine Learning for Molecules lecture. Architectural symmetry and energy-derived forces are developed in <a href="{% post_url 2026-08-08-equivariant-transformers-machine-learned-potentials %}">Equivariant Transformers and Machine-Learned Potentials</a>; the stochastic density dynamics behind Langevin simulation are developed in <a href="{% post_url 2026-02-04-fokker-planck-equation %}">The Fokker–Planck Equation</a>.</em>
</p>

A force-field benchmark asks whether a model predicts energies and forces on held-out configurations. A molecular simulation asks a harder question: what happens after the model's prediction moves the atoms, the new geometry is fed back into the model, and this loop is repeated millions of times?

That feedback changes the meaning of error. A small force error on a familiar structure may be harmless. A small systematic bias can alter an equilibrium distribution. One bad extrapolative force can push a bond into an unphysical region, after which every subsequent query is farther from the training data. Static accuracy is a property of predictions on a dataset; simulation reliability is a property of a learned dynamical system coupled to an integrator and an ensemble protocol.

The useful object is therefore not merely a neural network. It is a chain of commitments: a reference electronic-structure method defines a potential-energy surface; a model approximates its energy and derivatives; an integrator turns those derivatives into trajectories; a thermostat or barostat determines the sampled ensemble; and statistical estimators turn the trajectory into structural, thermodynamic, and kinetic observables. Validation has to follow the same chain.

## A potential-energy surface turns geometry into force

Consider $$N$$ atoms with positions

$$
\mathbf{R}=(\mathbf{r}_1,\ldots,\mathbf{r}_N)
$$

and masses $$m_1,\ldots,m_N$$. Under the Born–Oppenheimer approximation, the electronic problem assigns a scalar potential energy $$U(\mathbf{R})$$ to each nuclear geometry. The force on atom $$i$$ is

$$
\mathbf{F}_i(\mathbf{R})
=-\nabla_{\mathbf{r}_i}U(\mathbf{R}).
$$

Classical Newtonian dynamics then follows

$$
m_i\frac{d^2\mathbf{r}_i}{dt^2}
=\mathbf{F}_i(\mathbf{R}).
$$

The potential-energy surface is high dimensional—roughly $$3N$$ coordinates before removing rigid motion—but the local interpretation is simple. Its gradient gives the direction and magnitude of acceleration. Minima are stable configurations, saddles organize transitions, and barriers set the rarity of activated events.

{% include figure.liquid loading="eager" path="assets/img/blog/molsim_pes_dynamics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A potential-energy surface maps each atomic geometry to an energy, and its negative gradient supplies the forces used by the integrator. Because each updated geometry becomes the next model input, molecular dynamics is a closed-loop deployment of the potential. Original diagram." %}

Numerical integration replaces the differential equation by finite updates. Velocity Verlet, for example, uses

$$
\begin{aligned}
\mathbf{r}_i^{n+1}
&=\mathbf{r}_i^n
+\Delta t\,\mathbf{v}_i^n
+\frac{\Delta t^2}{2m_i}\mathbf{F}_i^n,\\
\mathbf{v}_i^{n+1}
&=\mathbf{v}_i^n
+\frac{\Delta t}{2m_i}
\left(\mathbf{F}_i^n+\mathbf{F}_i^{n+1}\right).
\end{aligned}
$$

The time step must resolve the fastest relevant motion. Making the force evaluation faster does not permit an arbitrarily larger $$\Delta t$$: stiff bond vibrations and steep repulsive walls remain stiff. An unstable integrator and an unstable force field can look similar, so the time step must be converged independently.

In ideal isolated dynamics, the Hamiltonian

$$
H(\mathbf{R},\mathbf{V})
=\sum_i\frac{m_i}{2}\lVert\mathbf{v}_i\rVert^2
+U(\mathbf{R})
$$

is conserved. This defines the microcanonical, or NVE, ensemble after equilibration. A thermostat instead exchanges energy with an idealized environment. Underdamped Langevin dynamics takes the form

$$
d\mathbf{r}_i=\mathbf{v}_i\,dt,
$$

$$
m_i\,d\mathbf{v}_i
=\mathbf{F}_i\,dt
-m_i\gamma\mathbf{v}_i\,dt
+\sqrt{2m_i\gamma k_{\mathrm B}T}\,d\mathbf{W}_i.
$$

Friction removes kinetic energy and Brownian forcing restores it. Their fluctuation–dissipation relation makes the canonical density proportional to $$\exp[-H/(k_{\mathrm B}T)]$$ stationary under suitable conditions. This is the NVT ensemble. A barostat adds volume or cell degrees of freedom to target pressure, producing an NPT ensemble. The [Fokker–Planck post]({% post_url 2026-02-04-fokker-planck-equation %}) explains how the Langevin SDE induces this probability evolution.

The thermostat is not a repair mechanism for a bad potential. It controls ensemble sampling; it should not be expected to hide energy discontinuities, runaway forces, or missing physics.

## A machine-learned force field approximates a reference surface

First-principles electronic-structure calculations provide accurate energies and forces but are too expensive for many large or long simulations. Classical force fields are much cheaper because they prescribe bonded, electrostatic, and dispersion terms, but their fixed functional forms can limit chemical accuracy and transferability. Machine-learned interatomic potentials aim at the middle: learn the reference potential with an evaluation cost closer to an empirical model.

The influential Behler–Parrinello construction writes the energy as a sum of atomic contributions (<span id="cite-behler2007"></span>[Behler & Parrinello, 2007](#ref-behler2007)):

$$
U_\theta(\mathbf{Z},\mathbf{R})
=\sum_{i=1}^N \varepsilon_{\theta,i}.
$$

Here $$\mathbf{Z}$$ contains chemical species. Local descriptors or a geometric neural network encode each environment, and shared parameters make the model size-extensive. Modern equivariant potentials improve data efficiency by letting vectors and higher-order geometric features transform consistently (<span id="cite-batzner2022"></span>[Batzner et al., 2022](#ref-batzner2022)). Their architectural details are the subject of the [equivariant-potentials post]({% post_url 2026-08-08-equivariant-transformers-machine-learned-potentials %}).

For simulation, forces should usually be differentiated from the same scalar energy:

$$
\mathbf{F}_{\theta,i}
=-\nabla_{\mathbf{r}_i}U_\theta.
$$

This guarantees a conservative learned force field up to numerical differentiation. Translation invariance implies zero net internal force, rotation invariance implies the correct force transformation and zero net internal torque, and mixed derivatives obey the symmetries of an energy Hessian. A direct vector-force model may be accurate at sampled points while violating these integrability constraints between them.

Training commonly combines energy and force errors:

$$
\mathcal{L}(\theta)
=w_E\left(U_\theta-U^\star\right)^2
+\frac{w_F}{3N}
\sum_{i=1}^N
\left\lVert
\mathbf{F}_{\theta,i}-\mathbf{F}_i^\star
\right\rVert^2.
$$

Energy labels anchor relative basin depths; force labels constrain the local slope in $$3N$$ coordinate directions. Stresses or virials are also important when cell response and pressure matter. The weights are not mere tuning constants: they determine which parts of the reference surface are reproduced most faithfully.

The reference method itself remains part of the model definition. An ML potential cannot systematically exceed the physics of its labels. Density-functional approximation, basis settings, dispersion treatment, spin state, charge state, and boundary conditions all propagate into the learned surface.

## Training data should cover a distribution of environments, not frames from one movie

Consecutive frames of an ab initio trajectory are strongly correlated. Collecting more adjacent frames may increase dataset size without adding new local environments. Conversely, a rare compressed bond or transition-state geometry can be more valuable than thousands of equilibrium snapshots.

A useful dataset spans the conditions the production simulation will visit: phases, compositions, temperatures, pressures, strains, interfaces, defects, reaction coordinates, and high-energy repulsive configurations that act as guardrails. The split should respect this structure. Randomly splitting neighboring frames leaks almost identical geometries into train and test sets and produces an overly optimistic error estimate.

When the relevant configuration space is not known in advance, simulation and data generation can be coupled. A concurrent-learning loop trains several models, explores with the current potential, measures disagreement, labels selected configurations with the expensive reference method, and retrains. DP-GEN formalized this explore–label–train cycle for deep potentials (<span id="cite-zhang2020"></span>[Zhang et al., 2020](#ref-zhang2020)).

{% include figure.liquid loading="eager" path="assets/img/blog/molsim_active_learning_loop.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Active or concurrent learning lets the evolving simulation propose configurations. An uncertainty or extrapolation gate sends only informative states for expensive reference labeling, after which the enlarged dataset is used to refit the potential. Original diagram." %}

For an ensemble of $$M$$ models, a simple force-disagreement score is

$$
u(\mathbf{R})
=\max_i
\sqrt{
\frac{1}{M}
\sum_{m=1}^M
\left\lVert
\mathbf{F}^{(m)}_i
-\overline{\mathbf{F}}_i
\right\rVert^2
}.
$$

Large disagreement is evidence that the models are not constrained to the same answer. Small disagreement is weaker evidence: all members may share the same bias, representation limit, or missing physical term. Uncertainty is therefore useful for triage, not a certificate of correctness. It must be calibrated on genuinely shifted validation sets and supplemented by physical monitors such as minimum distances, maximum forces, energy drift, coordination changes, and known conservation laws.

## Static error and rollout stability answer different questions

Suppose two potentials have the same force RMSE on a held-out set. One may still have a rough energy surface between test points, a discontinuous cutoff, or an unphysical low-energy pocket just outside the sampled domain. Molecular dynamics actively searches such defects because forces direct the trajectory toward whatever the model declares favorable.

{% include figure.liquid loading="eager" path="assets/img/blog/molsim_static_vs_rollout.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Held-out snapshots test interpolation at fixed configurations. A rollout feeds every prediction back into the next input, so the first extrapolative state can trigger a rapidly compounding failure that static error statistics never probe. Original diagram." %}

This creates several distinct failure modes:

- **Integrator instability:** $$\Delta t$$ is too large for the fastest modeled mode.
- **Surface roughness:** energy is accurate but its derivatives change sharply, especially near a cutoff.
- **Uncovered configuration:** the trajectory enters chemistry or geometry absent from training.
- **Spurious attraction:** the model invents an artificial low-energy structure and the dynamics collapses into it.
- **Systematic thermodynamic bias:** the trajectory remains visually plausible but basin populations or pressures are wrong.

Fu et al. show that lower held-out errors do not automatically improve downstream physical predictions and propose energy conservation in molecular dynamics as an additional gate (<span id="cite-fu2025"></span>[Fu et al., 2025](#ref-fu2025)). Conservation is especially diagnostic in NVE simulation, where thermostatting cannot obscure the drift. It is necessary but not sufficient: a smooth conservative model can conserve the wrong Hamiltonian perfectly.

Rollout validation should therefore be adversarial. Start from multiple basins and temperatures; test longer than the training trajectories; perturb bonds and cell parameters; cross phase boundaries if they are in scope; and compare failure time as well as average error. Monitor the distribution of uncertainty over time, not only its mean. A rare peak often matters more than a low average.

## Local models need explicit answers for long-range physics

Most scalable ML potentials use a finite neighborhood radius. This makes evaluation approximately linear in atom count and matches the locality of many short-range interactions. It also creates two obligations.

First, energy and its required derivatives must vanish smoothly at the cutoff. If an edge disappears abruptly, the force changes discontinuously. Repeating that discontinuity over many neighbor-list crossings produces heating or energy drift.

Second, a local environment is not always sufficient. Electrostatics, polarization, dispersion, and charge transfer can couple distant regions. Two atoms may have identical neighbors inside the cutoff but belong to systems with different total charge or electric field. No amount of local training data can resolve information excluded from the representation.

The remedy depends on the physics: add explicit electrostatics or dispersion, predict environment-dependent charges and solve a global charge-equilibration problem, use reciprocal-space components, or combine local and global message passing. Fourth-generation high-dimensional neural-network potentials demonstrate how nonlocal charge equilibration can complement local energy terms (<span id="cite-ko2021"></span>[Ko et al., 2021](#ref-ko2021)). The important decision is explicit: identify which interactions are delegated to learning and which are retained as structured physics.

Long-range adequacy should be tested at the observable level. A model can have excellent short-range force errors yet fail dielectric response, interfacial energetics, phonons, or diffusion because a weak missing interaction accumulates coherently across the system.

## The simulation workflow defines what can be measured

A production trajectory is not obtained by placing atoms in a box and pressing run. A typical workflow minimizes severe overlaps, ramps temperature, equilibrates the target ensemble, and only then accumulates production statistics. Initialization is consequential because a metastable system may retain memory of its starting basin for far longer than the nominal equilibration period.

Observables fall into several classes. Structural quantities include radial distribution functions,

$$
g(r)
=\frac{1}{4\pi r^2\rho N}
\left\langle
\sum_{i\neq j}\delta(r-r_{ij})
\right\rangle,
$$

coordination numbers, angular distributions, and conformational populations. Thermodynamic quantities include mean energy, pressure, heat capacity, phase stability, and free-energy differences. A free-energy difference between regions $$A$$ and $$B$$ can be written as

$$
\Delta F_{A\to B}
=-k_{\mathrm B}T
\log\frac{P(B)}{P(A)},
$$

provided the simulation samples both regions with the correct equilibrium weights. Dynamic quantities include relaxation times and transport coefficients. For example, the Einstein relation gives the self-diffusion coefficient in three dimensions:

$$
D
=\lim_{t\to\infty}
\frac{1}{6t}
\left\langle
\lVert\mathbf{r}_i(t)-\mathbf{r}_i(0)\rVert^2
\right\rangle.
$$

These formulas expose two different validation requirements. Equilibrium averages require correct stationary weights and adequate mixing. Kinetic quantities additionally require faithful time correlations. An aggressive thermostat, coarse-graining, or biased enhanced-sampling force may preserve or recover equilibrium statistics after reweighting while distorting real dynamics.

{% include figure.liquid loading="eager" path="assets/img/blog/molsim_physics_to_observables.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Scientific validation has four levels: a smooth conservative energy, a stable and equilibrated trajectory, the correct ensemble distribution, and agreement of the final structural, thermodynamic, or transport observable. Passing an earlier level does not guarantee the next. Original diagram." %}

Rare events make the distinction sharper. If a barrier crossing is much slower than the accessible trajectory, an unbiased simulation may be stable but statistically useless. Replica exchange, metadynamics, umbrella sampling, and learned bias potentials accelerate exploration, but their bias and reweighting rules become part of the estimator. Faster sampling does not relax the need for an accurate underlying potential in the visited regions.

## Validation should follow the intended claim

A simulation-ready validation report should separate four layers.

At the **reference layer**, verify energy, force, virial, and relative-energy errors on splits organized by trajectory, composition, phase, and thermodynamic condition. At the **dynamical layer**, measure NVE drift, temperature and pressure control, constraint violations, neighbor-list smoothness, and catastrophic-failure time across seeds. At the **distribution layer**, compare structural distributions and basin populations, with block averaging and effective sample sizes to account for autocorrelation. At the **observable layer**, compare the actual claimed quantities—free energies, diffusion, elastic constants, phonons, reaction rates, or phase boundaries—with reference calculations or experiment, including statistical and model uncertainty.

This hierarchy prevents a common category error. A low force RMSE supports the claim that local derivatives interpolate well on a particular test distribution. It does not by itself support a melting temperature, binding free energy, or diffusion coefficient. Those claims require the entire trajectory-to-estimator pipeline.

Machine-learned force fields are powerful precisely because they turn costly electronic-structure information into long trajectories. The leverage is enormous, and so is the opportunity for errors to compound. The right standard is therefore not whether a network is accurate once. It is whether the learned surface, numerical dynamics, sampled ensemble, and final observable remain mutually consistent for the full simulation the scientist intends to trust.

---

## References

<ol class="bibliography">
  <li id="ref-behler2007">Behler, J., &amp; Parrinello, M. (2007). <a href="https://doi.org/10.1103/PhysRevLett.98.146401">Generalized neural-network representation of high-dimensional potential-energy surfaces</a>. <em>Physical Review Letters</em>, 98, 146401. <a href="#cite-behler2007">↩</a></li>
  <li id="ref-batzner2022">Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., &amp; Kozinsky, B. (2022). <a href="https://www.nature.com/articles/s41467-022-29939-5">E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials</a>. <em>Nature Communications</em>, 13, 2453. <a href="#cite-batzner2022">↩</a></li>
  <li id="ref-zhang2020">Zhang, Y., Wang, H., Chen, W., Zeng, J., Zhang, L., Wang, H., &amp; E, W. (2020). <a href="https://doi.org/10.1016/j.cpc.2020.107206">DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models</a>. <em>Computer Physics Communications</em>, 253, 107206. <a href="#cite-zhang2020">↩</a></li>
  <li id="ref-fu2025">Fu, X., Wood, B. M., Barroso-Luque, L., Levine, D. S., Gao, M., Dzamba, M., &amp; Zitnick, C. L. (2025). <a href="https://proceedings.mlr.press/v267/fu25h.html">Learning smooth and expressive interatomic potentials for physical property prediction</a>. <em>Proceedings of the 42nd International Conference on Machine Learning</em>, 17875–17893. <a href="#cite-fu2025">↩</a></li>
  <li id="ref-ko2021">Ko, T. W., Finkler, J. A., Goedecker, S., &amp; Behler, J. (2021). <a href="https://www.nature.com/articles/s41467-020-20427-2">A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer</a>. <em>Nature Communications</em>, 12, 398. <a href="#cite-ko2021">↩</a></li>
</ol>

---

*Figure provenance.* All four `molsim_` diagrams are original SVG illustrations generated by `scripts/generate_molsim_figures.py`. They synthesize standard simulation identities and the validation principles discussed in the cited primary literature; no third-party artwork is reproduced.
