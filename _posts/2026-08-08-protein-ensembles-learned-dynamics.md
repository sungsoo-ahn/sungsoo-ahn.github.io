---
layout: post
title: "Protein Ensembles and Learned Molecular Dynamics"
date: 2026-08-08
last_updated: 2026-08-08
description: "How metastable protein conformations become equilibrium ensembles and kinetic models, and what learned samplers must preserve beyond structural plausibility."
abstract: >
  Proteins occupy distributions of conformations connected by rare transitions. Learning those distributions can accelerate equilibrium sampling, while learning the dynamics additionally requires the correct transition pathways and timescales.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [protein-science]
lecture_paths: [gdl]
tags: [protein-ensembles, molecular-dynamics, markov-state-models, learned-dynamics, generative-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the protein-ensemble and learned-dynamics storyline from my 2025 Geometric Deep Learning lecture, together with the simulation material from Machine Learning for Molecules. The mechanics of reliable trajectories are developed in <a href="{% post_url 2026-08-08-molecular-simulation-machine-learned-force-fields %}">Molecular Simulation with Machine-Learned Force Fields</a>; the stochastic density view appears in <a href="{% post_url 2026-02-04-fokker-planck-equation %}">The Fokker–Planck Equation</a>; geometric generative paths appear in <a href="{% post_url 2026-08-08-geometric-flow-matching-manifolds %}">Geometric Flow Matching on Manifolds</a>.</em>
</p>

A protein structure is not a single object. Even at fixed sequence, solvent, temperature, protonation, and binding partners, the atoms fluctuate. Flexible loops move, side chains exchange rotamers, domains open and close, and disordered regions occupy broad families of conformations. Some states exchange in picoseconds; others remain separated for milliseconds or longer.

This makes “predict the structure distribution” ambiguous. An equilibrium model should reproduce how often each conformation occurs. A dynamical model should additionally reproduce how conformations are connected and how long transitions take. A set of beautiful structures can satisfy the first criterion badly and the second criterion not at all.

The distinction matters for learned models. A diffusion or flow model can draw independent equilibrium-like structures without defining physical time. A transfer model can predict a future state at a chosen lag. A trajectory generator can synthesize an entire path. These objects answer different scientific questions, require different training data, and demand different validation.

## Metastable states organize a protein ensemble

Let $$x$$ denote all molecular coordinates after fixing the thermodynamic system and removing irrelevant rigid motion. At temperature $$T$$, the canonical equilibrium density is

$$
\mu(x)
=\frac{1}{Z}\exp[-\beta U(x)],
\qquad
\beta=(k_{\mathrm B}T)^{-1},
$$

where $$U(x)$$ is the potential energy and $$Z$$ is the partition function. A low-dimensional collective variable $$z=\xi(x)$$—a torsion, contact pattern, inter-domain distance, or learned coordinate—induces the marginal

$$
p(z)
=\int \mu(x)\,\delta(z-\xi(x))\,dx.
$$

Its free energy is

$$
F(z)
=-k_{\mathrm B}T\log p(z)+C.
$$

Basins of $$F$$ correspond to frequently occupied conformational families. High barriers separate **metastable states**: the system mixes rapidly inside a state but leaves it rarely. The protein therefore has two scales at once—fast local fluctuations and slow state exchange.

{% include figure.liquid loading="eager" path="assets/img/blog/protens_free_energy_landscape.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A protein ensemble concentrates in metastable free-energy basins. Basin depth controls equilibrium population, while barrier height and dynamical friction control how rarely the system crosses between states. Original diagram." %}

Free energy alone does not determine kinetics. Two systems can share $$p(z)$$ but have different diffusivity along $$z$$, different hidden barriers orthogonal to $$z$$, and different transition mechanisms. Projecting onto a poor collective variable can even make a non-Markovian process look like motion on a simple landscape.

This is the first recurring warning: a landscape is a marginal description, not a literal track on which a protein moves.

## Molecular dynamics supplies both ensembles and time order

Molecular dynamics integrates forces from a molecular Hamiltonian or machine-learned potential. With a thermostat, a sufficiently long and ergodic trajectory can sample an equilibrium ensemble. Unlike an unordered structural dataset, the trajectory also records temporal correlations and transition pathways.

The [molecular-simulation post]({% post_url 2026-08-08-molecular-simulation-machine-learned-force-fields %}) develops initialization, integration, force-field error, and ensemble validation. Here the central limitation is timescale separation. A femtosecond step is needed to resolve fast atomic motion, while a biologically interesting conformational change may take milliseconds. One event can require roughly a trillion integration steps.

Running many shorter trajectories in parallel improves coverage, especially when seeded from diverse states, but it does not automatically solve rare-event sampling. Enhanced-sampling methods deliberately alter exploration. Replica exchange changes temperature or Hamiltonian across replicas. Umbrella sampling restrains chosen collective variables. Metadynamics deposits a history-dependent bias that discourages revisiting explored regions (<span id="cite-laio2002"></span>[Laio & Parrinello, 2002](#ref-laio2002)).

Biased samples cannot be treated as ordinary equilibrium frames. Recovering unbiased populations requires the method's reweighting formula and adequate overlap. Kinetics is even more delicate: a bias that accelerates barrier crossing usually changes physical transition times. Enhanced sampling may give a reliable free-energy difference while destroying the natural path timing.

The force field remains another source of uncertainty. A perfectly converged simulation samples the equilibrium distribution of its chosen Hamiltonian, which may disagree with the real protein because of water balance, ion parameters, protonation, polarization, or missing chemistry. Longer sampling removes statistical error but not Hamiltonian bias.

## Markov state models compress long-time kinetics

A Markov state model partitions configuration space into states $$S_1,\ldots,S_K$$ and estimates transition probabilities at lag time $$\tau$$:

$$
T_{ij}(\tau)
=\Pr(X_{t+\tau}\in S_j\mid X_t\in S_i).
$$

Given transition counts $$C_{ij}(\tau)$$, the simplest estimator normalizes rows,

$$
\widehat T_{ij}
=\frac{C_{ij}}{\sum_j C_{ij}},
$$

although reversible and Bayesian estimators are often preferable. The stationary distribution satisfies

$$
\boldsymbol{\pi}^{\mathsf T}T
=\boldsymbol{\pi}^{\mathsf T}.
$$

At equilibrium, detailed balance is

$$
\pi_iT_{ij}
=\pi_jT_{ji}.
$$

The nontrivial eigenvalues $$1>\lambda_2\geq\lambda_3\geq\cdots$$ encode relaxation times,

$$
t_k
=-\frac{\tau}{\log |\lambda_k|}.
$$

Thus one compact object provides state populations, slow timescales, and transition-path statistics. Prinz et al. develop the construction and its validation in detail (<span id="cite-prinz2011"></span>[Prinz et al., 2011](#ref-prinz2011)).

{% include figure.liquid loading="eager" path="assets/img/blog/protens_markov_state_model.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A Markov state model clusters molecular configurations, counts transitions separated by lag time tau, and estimates a transition matrix. Its stationary vector describes equilibrium weights; its slow eigenmodes describe long-timescale kinetics. Original diagram." %}

The Markov approximation is not automatic. If a state combines configurations that relax slowly relative to $$\tau$$, the next-state distribution retains memory of where the trajectory entered. Increasing $$\tau$$ reduces this memory but discards temporal resolution and reduces the number of observed transitions. Implied timescales should plateau across a range of lag times, and Chapman–Kolmogorov tests should compare multi-step predictions $$T(\tau)^n$$ with directly observed transitions at $$n\tau$$.

Clustering geometry is equally consequential. RMSD may split a kinetically coherent basin or merge distinct states separated by a hidden barrier. A useful representation emphasizes slow coordinates rather than merely large geometric variance.

## Learned latent dynamics search for slow coordinates

Traditional pipelines choose molecular features, perform a time-lagged dimensional reduction, cluster, and estimate transitions. Learned latent models replace some or all of these steps with a neural encoder

$$
z_t=f_\theta(x_t).
$$

A reconstruction objective alone tends to preserve coordinates that explain structural variance, which need not be kinetically slow. Time-lagged objectives instead reward representations that predict $$x_{t+\tau}$$ or approximate the leading singular functions of the transfer operator.

VAMPnets learn soft state memberships and a kinetic model end to end using a variational score for Markov processes (<span id="cite-mardt2018"></span>[Mardt et al., 2018](#ref-mardt2018)). Their latent states are optimized for slow dynamics rather than visual separation. Related time-lagged autoencoders, Koopman models, and neural transfer operators learn a coordinate system in which long-time evolution is approximately linear or easier to propagate.

This compression can pool many short trajectories into estimates of slow behavior. It cannot infer unobserved mechanisms without assumptions. If no trajectory crosses between two basins, a learned transition rate is extrapolation. A model may infer a plausible bridge from related proteins or structural priors, but the evidence is no longer contained in the trajectory data alone.

Protein generalization adds a second challenge. A latent coordinate learned for one sequence may describe a particular loop motion; the analogous functional coordinate in another protein can involve different residues. Sequence-conditioned geometric encoders must align these motions without erasing protein-specific states.

## Equilibrium generators skip the physical path

An equilibrium generator aims to draw independent samples from $$\mu(x)$$. Boltzmann generators use invertible flows to map a simple latent distribution into molecular configurations and correct the generated distribution with statistical-mechanical weights (<span id="cite-noe2019"></span>[Noé et al., 2019](#ref-noe2019)). If samples are independent, they can cross free-energy barriers without waiting for local dynamics.

Diffusion and flow models extend this strategy to flexible geometric architectures. AlphaFlow repurposes a protein structure predictor inside a flow-matching framework and learns sequence-conditioned ensemble variation from structural and molecular-dynamics data (<span id="cite-alphaflow2024"></span>[Jing et al., 2024](#ref-alphaflow2024)). The model can generate a diverse ensemble much faster than integrating atomistic dynamics for the same number of decorrelated frames.

The [geometric flow post]({% post_url 2026-08-08-geometric-flow-matching-manifolds %}) explains how translations, rotations, and internal geometry constrain these paths. For proteins, rigid residue frames or backbone coordinates must transform equivariantly, while the output density remains invariant to a global laboratory frame.

Independent ensemble samples are valuable for equilibrium observables:

$$
\langle A\rangle_\mu
=\int A(x)\mu(x)\,dx
\approx\frac{1}{M}\sum_{m=1}^M A(x^{(m)}).
$$

But the denoising or flow time used to generate $$x^{(m)}$$ is an algorithmic coordinate, not physical time. A path from Gaussian noise to a folded structure does not describe how the protein folds. Equilibrium generation can answer “which conformations and with what weights?” without answering “how do they interconvert?”

## Trajectory generators model ordered paths

A dynamical surrogate instead models a transition kernel,

$$
p_\theta(x_{t+\Delta}\mid x_t),
$$

or a joint path distribution conditioned on endpoints or partial frames. Autoregressive transition models repeatedly sample the next coarse time step. Conditional normalizing flows can learn large-lag transfer operators. Diffusion and flow models can treat a trajectory as a geometric time series and generate many frames jointly.

MDGen demonstrates the joint-trajectory view for forward simulation, endpoint-conditioned transition paths, temporal upsampling, and inpainting (<span id="cite-mdgen2024"></span>[Jing et al., 2024](#ref-mdgen2024)). Joint generation reduces the accumulation of one-step errors and permits noncausal conditioning. It also creates a harder consistency problem: the sampled path must be geometrically valid at every frame and statistically compatible across time.

Conditioning is scientifically useful. Given two metastable endpoints, a model can propose transition paths. Given sparse experimental restraints, it can generate compatible ensembles. Given the trajectory of one protein region, it can inpaint coupled motion elsewhere. But conditioning changes the target distribution. Endpoint-conditioned transition paths are rare-event bridges, not ordinary equilibrium trajectories; property-guided samples need correction before being interpreted as unbiased populations.

## Equilibrium and kinetics can disagree completely

Consider two states $$A$$ and $$B$$ with stationary probabilities $$\pi_A=\pi_B=1/2$$. A reference process might have rates

$$
k_{A\to B}=k_{B\to A}=10^{-3}\ \mathrm{ns}^{-1},
$$

while a surrogate uses

$$
\widetilde k_{A\to B}
=\widetilde k_{B\to A}
=1\ \mathrm{ns}^{-1}.
$$

Both satisfy detailed balance and have the same equilibrium distribution. Their relaxation times differ by a factor of one thousand. A snapshot-based metric cannot tell them apart.

{% include figure.liquid loading="eager" path="assets/img/blog/protens_equilibrium_vs_kinetics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Two models can assign identical stationary probability to states A and B while predicting radically different exchange rates. Equilibrium validation sees the same populations; kinetic validation sees different relaxation times and transition counts. Original diagram." %}

The reverse failure also occurs. A local transition model can predict short-time fluctuations accurately while drifting toward the wrong stationary distribution after repeated rollout. Enforcing detailed balance or training against equilibrium data can help, but consistency must be tested rather than assumed.

This yields a clean hierarchy. An **ensemble model** needs the correct stationary distribution. A **transfer model** needs the correct conditional distribution at its specified lag and the correct stationary distribution under iteration. A **physical-time trajectory model** additionally needs multi-time correlations, pathway statistics, and calibrated time units.

## Validation must match the scientific claim

Structural plausibility is the first gate: bond geometry, stereochemistry, excluded volume, secondary structure, and global frame symmetry. A generated backbone can have good RMSD while containing local clashes or unrealistic peptide geometry.

Equilibrium validation compares state populations, free-energy differences, contact and distance distributions, radius of gyration, solvent exposure, and higher-order correlations. Coverage and precision must both be reported. A broad model can cover every reference state by producing many unphysical structures; a sharp model can look precise while missing rare functional basins.

Kinetic validation compares implied timescales, autocorrelation functions, mean first-passage times, transition-path ensembles, committor statistics, and Chapman–Kolmogorov consistency. Evaluation should use held-out temporal blocks or independent trajectories, not random frames from the same correlated run.

{% include figure.liquid loading="eager" path="assets/img/blog/protens_validation_layers.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Validation proceeds from local geometry to equilibrium populations, kinetic observables, and independent experiments. Failure at a later layer can reveal errors in the force field, sampling protocol, state representation, learned dynamics, or conditioning assumptions. Original diagram." %}

Experimental observables are ensemble averages filtered through a measurement model. NMR chemical shifts and order parameters, hydrogen–deuterium exchange, single-molecule FRET, SAXS, cryo-EM heterogeneity, and kinetic rate measurements each see different aspects of the ensemble. Agreement with one observable does not uniquely identify the full distribution. Forward models from structure to experiment have their own uncertainty, and several ensembles may fit the same low-dimensional measurement.

The strongest test combines complementary observables and predicts data not used for conditioning. If experimental restraints were supplied to the generator, reproducing them is a constraint-satisfaction check, not independent validation.

## Failure modes are usually category errors

Several recurring failures follow directly from confusing the target objects:

- **Flow time is treated as physical time.** A generative path is interpreted as a folding trajectory without kinetic calibration.
- **Diversity is mistaken for equilibrium weight.** Many distinct structures are generated, but their frequencies do not follow the desired ensemble.
- **Biased simulation is treated as unbiased data.** Enhanced-sampling frames train a model without reweighting or labeling the bias.
- **Random-frame splitting leaks trajectories.** Near-duplicate temporal neighbors appear in both training and test sets.
- **A learned state hides memory.** The latent partition looks compact but fails lag-time and Chapman–Kolmogorov tests.
- **Protein-level generalization is overstated.** Homologous sequences or nearly identical folds cross the split boundary.
- **Force-field truth is confused with experimental truth.** A model perfectly emulates a simulation whose Hamiltonian is systematically wrong.

Uncertainty should be decomposed accordingly: finite sampling, force-field error, generative-model error, state-discretization error, and experimental forward-model error are not interchangeable confidence intervals.

Proteins are ensembles because thermal motion and competing interactions populate many conformations. They exhibit dynamics because those conformations are connected by structured, often rare transitions. Learned models can accelerate both tasks, but only if the distinction remains explicit. An equilibrium generator may leap between basins and still be scientifically valid. A kinetic model must earn the right to attach a clock to those leaps.

---

## References

<ol class="bibliography">
  <li id="ref-laio2002">Laio, A., &amp; Parrinello, M. (2002). <a href="https://doi.org/10.1073/pnas.202427399">Escaping free-energy minima</a>. <em>Proceedings of the National Academy of Sciences</em>, 99(20), 12562–12566. <a href="#cite-laio2002">↩</a></li>
  <li id="ref-prinz2011">Prinz, J.-H., Wu, H., Sarich, M., Keller, B., Senne, M., Held, M., Chodera, J. D., Schütte, C., &amp; Noé, F. (2011). <a href="https://doi.org/10.1063/1.3565032">Markov models of molecular kinetics: Generation and validation</a>. <em>The Journal of Chemical Physics</em>, 134, 174105. <a href="#cite-prinz2011">↩</a></li>
  <li id="ref-mardt2018">Mardt, A., Pasquali, L., Wu, H., &amp; Noé, F. (2018). <a href="https://www.nature.com/articles/s41467-017-02388-1">VAMPnets for deep learning of molecular kinetics</a>. <em>Nature Communications</em>, 9, 5. <a href="#cite-mardt2018">↩</a></li>
  <li id="ref-noe2019">Noé, F., Olsson, S., Köhler, J., &amp; Wu, H. (2019). <a href="https://doi.org/10.1126/science.aaw1147">Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning</a>. <em>Science</em>, 365(6457), eaaw1147. <a href="#cite-noe2019">↩</a></li>
  <li id="ref-alphaflow2024">Jing, B., Berger, B., &amp; Jaakkola, T. (2024). <a href="https://proceedings.mlr.press/v235/jing24a.html">AlphaFold meets flow matching for generating protein ensembles</a>. <em>Proceedings of the 41st International Conference on Machine Learning</em>, 22277–22303. <a href="#cite-alphaflow2024">↩</a></li>
  <li id="ref-mdgen2024">Jing, B., Stärk, H., Jaakkola, T., &amp; Berger, B. (2024). <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/478b06f60662d3cdc1d4f15d4587173a-Abstract-Conference.html">Generative modeling of molecular dynamics trajectories</a>. <em>Advances in Neural Information Processing Systems</em>, 37. <a href="#cite-mdgen2024">↩</a></li>
</ol>

---

*Figure provenance.* All four `protens_` diagrams are original SVG illustrations generated by `scripts/generate_protens_figures.py`. They synthesize standard statistical-mechanical and kinetic-modeling concepts described in the cited primary literature; no third-party artwork is reproduced.
