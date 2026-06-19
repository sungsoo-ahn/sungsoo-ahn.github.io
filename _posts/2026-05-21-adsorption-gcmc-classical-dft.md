---
layout: post
title: "Adsorption, GCMC, and Classical DFT"
date: 2026-05-21
last_updated: 2026-06-19
description: "Gas adsorption simulation: uptake, grand canonical Monte Carlo, classical density functional theory, and density-field learning."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
series: ml-for-science
series_title: "ML for Science Foundations"
series_description: "A guided route through scientific ML topics: quantum chemistry, equivariant molecular models, electrocatalysis, and protein design."
series_order: 5
categories: [science]
tags: [adsorption, grand-canonical-monte-carlo, classical-density-functional-theory, molecular-simulation, machine-learning]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post introduces gas adsorption simulation for ML researchers who encounter GCMC, isotherms, uptake, and classical DFT in porous-material papers. It is meant to sit between my earlier posts on <a href="/blog/2026/ensembles-thermostats-barostats/">ensembles and Monte Carlo</a> and <a href="/blog/2026/quantum-chemistry-dft/">quantum-chemistry DFT</a>. The background comes partly from work in our group on learning adsorbate density fields, where GCMC provides particle-simulation references and classical DFT provides cheaper density-field supervision. Corrections are welcome.</em>
</p>

## Introduction

Gas adsorption looks like a scalar prediction problem at first: given a porous material, a gas species, a temperature, and a pressure, predict how much gas the material stores.

That scalar is called **uptake**. It is the main number used in high-throughput screening for methane storage, carbon capture, and gas separations. But uptake is only the integral of a richer object: the equilibrium density field of gas molecules inside the pore.

If $$\rho(\mathbf{r})$$ is the local number density of adsorbate molecules at position $$\mathbf{r}$$, then the total loading is

$$N = \int_{\mathcal{V}} \rho(\mathbf{r})\,d\mathbf{r}$$

where $$\mathcal{V}$$ is the unit-cell volume. The scalar uptake $$N$$ tells you how many molecules are present. The field $$\rho(\mathbf{r})$$ tells you where they sit, which binding sites are occupied, which regions are inaccessible, and how adsorption changes with pressure.

Two standard routes compute that density:

- **Grand canonical Monte Carlo (GCMC)** samples explicit particle configurations and estimates density by averaging those samples.
- **Classical density functional theory (cDFT)** skips particle sampling and solves directly for the equilibrium density field by minimizing a free-energy functional.

For ML, this distinction matters because it changes the learning target. Predicting uptake is scalar regression. Predicting $$\rho(\mathbf{r})$$ is learning a thermodynamic density operator.

To keep the notation anchored, I will use one running example: **methane adsorption in MOF-5**. The framework is fixed, methane is the adsorbate, and the thermodynamic condition is a chosen temperature and pressure. Changing the pressure changes the reservoir chemical potential; the number of methane molecules inside the unit cell is the thing we want to predict.

### Overview

The adsorption problem is an open-system equilibrium problem. A porous material sits in contact with a gas reservoir, so the number of molecules inside the pore fluctuates. GCMC handles this by sampling particle configurations in the grand canonical ensemble. cDFT handles it by optimizing a density field that minimizes a grand-potential functional. Modern ML methods can use both views: broad cDFT labels teach a cheap density prior, while sparse GCMC labels correct toward particle-simulation fidelity.

The post builds this picture in four steps. The Adsorption Problem section defines uptake, density, and the grand canonical boundary condition. The GCMC section explains the particle sampler. The Classical DFT section explains the density variational problem and fixed-point equation. The final sections explain why unnormalized density fields are a better ML target than scalar uptake alone.

{% include figure.liquid loading="eager" path="assets/img/blog/adsorption_gcmc_cdft_particle_density.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Particle and density views of adsorption equilibrium. GCMC samples particle configurations in the grand canonical ensemble, then averages those samples into uptake or a density field. cDFT solves directly for the equilibrium density by fixed-point iteration. Adapted from internal manuscript materials on multi-fidelity adsorbate density learning." %}

---

## The Adsorption Problem

Adsorption occurs when guest molecules accumulate on a surface or inside a porous host. The host is the **adsorbent**. The guest fluid is the **adsorbate**.[^adswords] In the running example, MOF-5 is the adsorbent and methane is the adsorbate. Carbon dioxide inside a zeolite and xenon inside activated carbon are the same kind of problem with different host-guest chemistry.

The physical setup is open. The material is in contact with a large gas reservoir at temperature $$T$$ and pressure $$P$$. Methane molecules enter and leave the pores until equilibrium. The number of adsorbed methane molecules is not fixed; it is an outcome.

This is why adsorption is usually modeled in the **grand canonical ensemble**, also called $$\mu VT$$:

- $$T$$ is fixed by a heat bath.
- $$V$$ is fixed by the simulation cell.
- $$\mu$$ is fixed by the gas reservoir.
- $$N$$ fluctuates.

The chemical potential $$\mu$$ is the free-energy cost of adding one molecule to the reservoir.[^chempot] In practice, for a pure gas at fixed $$T$$, specifying pressure $$P$$ determines $$\mu$$ through an equation of state. For methane in MOF-5, "simulate at pressure $$P$$" means "set the methane reservoir chemical potential corresponding to $$P$$." This is why adsorption papers often use pressure language while the formal ensemble is written with $$\mu$$.

### What Makes Adsorption Hard?

Two effects must be modeled together.

First, the porous material creates a heterogeneous external potential $$V_{\mathrm{ext}}(\mathbf{r})$$. In MOF-5, methane sees favorable regions near parts of the framework and unfavorable regions where the framework atoms block space. The density can change by orders of magnitude across a single unit cell.

Second, adsorbate molecules interact with each other. At low pressure, a methane molecule mostly feels the framework. At high pressure, methane molecules crowd the pore and exclude volume from one another. This many-body structure is exactly what makes adsorption nontrivial.

If we ignore fluid-fluid interactions, each molecule independently follows the framework potential. The density has the ideal-gas Boltzmann form:

$$\rho_{\mathrm{Boltz}}(\mathbf{r}) = \rho_{\mathrm{bulk}}(T,P)\exp[-\beta V_{\mathrm{ext}}(\mathbf{r})], \qquad \beta = \frac{1}{k_{B}T}$$

Here $$\rho_{\mathrm{bulk}}(T,P)$$ is the number density of the gas reservoir. This baseline says: put more density where the framework potential is low. Real adsorption adds many-body corrections on top of this baseline.

---

## GCMC: The Particle View

GCMC samples adsorption equilibrium by constructing a Markov chain over particle configurations. Adams introduced grand canonical Monte Carlo for Lennard-Jones fluids in 1975; modern adsorption packages such as RASPA use the same ensemble logic with richer force fields and move sets. For methane in MOF-5, a state contains a variable number of methane molecules inside the unit cell:

$$\mathbf{x} = (N, \mathbf{r}_1, \ldots, \mathbf{r}_N, \Omega_1, \ldots, \Omega_N)$$

where $$\mathbf{r}_i$$ is the molecular position and $$\Omega_i$$ denotes orientation for a non-spherical molecule. Methane is often treated as nearly spherical in simple force fields, but keeping $$\Omega_i$$ in the notation makes the same state representation work for molecules such as carbon dioxide.

The target distribution is the grand canonical distribution:

$$p(N,\mathbf{r}^N) \propto \frac{1}{N! \Lambda^{3N}}\exp[\beta \mu N - \beta U_N(\mathbf{r}^N)]$$

where $$U_N$$ is the potential energy of the $$N$$-molecule configuration and $$\Lambda$$ is the thermal de Broglie wavelength. For most ML purposes, the important part is the energy-chemical-potential trade-off:

$$p(N,\mathbf{r}^N) \propto \exp[-\beta(U_N(\mathbf{r}^N) - \mu N)]$$

Increasing $$N$$ is rewarded by $$\mu N$$ but penalized if the inserted molecules raise the interaction energy too much. At higher methane pressure, the reward for adding molecules increases; at high loading, the overlap and crowding penalties also increase.

### The Moves

GCMC uses Metropolis-Hastings moves that leave the grand canonical distribution invariant:[^gcmc]

- **Translation and rotation:** move an existing molecule.
- **Insertion:** propose a new molecule at a random position and orientation.
- **Deletion:** remove an existing molecule.

The insertion/deletion moves are the defining feature. They let particle number fluctuate, which is exactly what adsorption needs. A trajectory of GCMC samples for methane in MOF-5 looks like a stack of snapshots, each with a different number of methane molecules in the pore.

{% include figure.liquid loading="eager" path="assets/img/blog/adsorption_gcmc_cdft_moves.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Core GCMC move types for the methane-in-MOF-5 running example. Translation and rotation explore configurations at fixed particle count, while insertion and deletion move between states with different \(N\)." %}

The uptake is the ensemble average:

$$\langle N \rangle \approx \frac{1}{M}\sum_{k=1}^{M} N_k$$

The density field is the ensemble average of particle locations:

$$\rho(\mathbf{r}) = \left\langle \sum_{i=1}^{N} \delta(\mathbf{r} - \mathbf{r}_i)\right\rangle_{T,\mu}$$

On a computer, the delta functions are binned onto a voxel grid or smoothed with a kernel. For the running example, each accepted methane configuration contributes methane centers to the grid. Averaging those grids turns particle samples into a density field.

{% include figure.liquid loading="eager" path="assets/img/blog/adsorption_gcmc_cdft_snapshots_to_density.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Coarse-graining GCMC samples into a density field. Each accepted particle snapshot contributes methane positions to a grid; averaging snapshots estimates \(\rho(\mathbf{r})\), whose integral gives uptake." %}

### Why GCMC Is Expensive

GCMC is the reference method because, for a chosen force field and enough sampling, it converges to the correct ensemble average. The price is sampling cost.

Insertion becomes hard in dense pores. A random proposed methane molecule can land too close to an existing methane molecule or a framework atom, producing a huge energy increase and near-zero acceptance probability. Narrow pores are also difficult because the allowed volume is a small fraction of the cell. The Markov chain then spends many steps proposing moves that do not change the state.

This is why high-throughput adsorption screening is expensive. One material, one gas, one temperature, and one pressure is already a simulation. An isotherm needs many pressures. A screening campaign needs thousands or millions of materials.

---

## Classical DFT: The Density View

Classical DFT solves the same equilibrium problem without sampling particles. It treats $$\rho(\mathbf{r})$$ as the optimization variable. In the running example, $$\rho(\mathbf{r})$$ is the methane number density over the MOF-5 unit cell. Evans' 1979 review is the classic starting point for this density-functional view of non-uniform classical fluids.

The word "classical" matters.[^cdft] This is not Kohn-Sham DFT for electrons. Quantum DFT uses the electron density to avoid the many-electron wavefunction. Classical DFT uses the molecular number density to avoid sampling many-particle configurations.

> **Classical density functional theory.** For a classical fluid at fixed $$T$$ and $$\mu$$, cDFT defines a grand-potential functional $$\Omega[\rho]$$. The equilibrium density is the density field that minimizes it:
>
> $$\rho_{\mathrm{eq}} = \arg\min_{\rho \geq 0} \Omega[\rho]$$
{: .block-definition }

For adsorption in a rigid framework such as MOF-5, the grand potential is

$$\Omega[\rho] = F_{\mathrm{id}}[\rho] + F_{\mathrm{exc}}[\rho] + \int \rho(\mathbf{r})\left[V_{\mathrm{ext}}(\mathbf{r}) - \mu\right]\,d\mathbf{r}$$

Each term has a direct interpretation.

**The ideal term $$F_{\mathrm{id}}[\rho]$$ is known exactly.** It is the entropy of a non-interacting classical gas:

$$F_{\mathrm{id}}[\rho] = k_{B}T \int \rho(\mathbf{r})\left(\log(\rho(\mathbf{r})\Lambda^3) - 1\right)\,d\mathbf{r}$$

This term spreads density out because entropy favors many accessible configurations.

**The external term couples density to the framework.** The integral

$$\int \rho(\mathbf{r})V_{\mathrm{ext}}(\mathbf{r})\,d\mathbf{r}$$

puts methane density in attractive regions and removes it from repulsive regions.

**The chemical-potential term controls loading.** The term

$$-\mu \int \rho(\mathbf{r})\,d\mathbf{r}$$

rewards adding methane molecules when the reservoir chemical potential is high.

**The excess term $$F_{\mathrm{exc}}[\rho]$$ contains fluid-fluid interactions.** This is the hard part. It accounts for excluded volume, dispersion attraction, chain connectivity, and other many-body effects. Unlike the ideal term, it is not known exactly for realistic fluids, so every practical cDFT method chooses an approximation.

The structure should feel familiar if you have seen variational inference or energy-based models. We define an objective over distributions, split it into a tractable reference part plus an interaction correction, then optimize the object we want directly.

### The cDFT Fixed Point

At equilibrium, the functional derivative of $$\Omega[\rho]$$ vanishes. This gives an Euler-Lagrange equation. Rearranged into fixed-point form, it becomes

$$\rho_{\mathrm{eq}}(\mathbf{r}) = \rho_{\mathrm{bulk}}\exp\left[-\beta V_{\mathrm{ext}}(\mathbf{r}) - \beta \left.\frac{\delta F_{\mathrm{exc}}}{\delta \rho(\mathbf{r})}\right|_{\rho_{\mathrm{eq}}} + \beta\mu_{\mathrm{exc}}^{\mathrm{bulk}}\right]$$

The first exponential term is the ideal-gas Boltzmann density. The functional-derivative term adds the many-body correction from fluid-fluid interactions. The bulk excess chemical potential aligns the confined fluid with the reservoir.

Because the right-hand side depends on $$\rho_{\mathrm{eq}}$$ through $$F_{\mathrm{exc}}$$, the equation must be solved iteratively. A simple Picard iteration is

$$\rho^{(n+1)}(\mathbf{r}) = \rho_{\mathrm{bulk}}\exp\left[-\beta V_{\mathrm{ext}}(\mathbf{r}) - \beta \frac{\delta F_{\mathrm{exc}}}{\delta \rho(\mathbf{r})}\bigg\rvert_{\rho^{(n)}} + \beta\mu_{\mathrm{exc}}^{\mathrm{bulk}}\right]$$

Starting from $$\rho^{(0)} = \rho_{\mathrm{Boltz}}$$, the solver repeatedly evaluates the many-body correction and updates the methane density until $$\rho^{(n+1)} \approx \rho^{(n)}$$.

{% include figure.liquid loading="eager" path="assets/img/blog/adsorption_gcmc_cdft_fixed_point.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The cDFT fixed-point loop. The current density \(\rho^{(n)}\) defines the many-body correction, the update produces \(\rho^{(n+1)}\), and the loop stops when the density no longer changes appreciably." %}

This is the density analogue of a self-consistent field loop. In quantum DFT, the electron density defines an effective Hamiltonian, whose orbitals define a new density. In classical DFT, the adsorbate density defines a fluid-fluid correction, which defines a new density.

### What Is the Functional?

The practical quality of cDFT depends on $$F_{\mathrm{exc}}[\rho]$$. For adsorption of small non-polar molecules, a common choice is a cDFT realization of **PC-SAFT**: perturbed-chain statistical associating fluid theory.

PC-SAFT represents a molecule as a chain of tangent Lennard-Jones segments.[^pcsaft] Methane is a small non-polar adsorbate, so much of the species dependence is summarized by three parameters:

- $$m$$: number of segments.
- $$\sigma$$: segment diameter.
- $$\varepsilon$$: dispersion energy.

The same parameters affect the framework potential. A typical external potential is a pairwise Lennard-Jones sum over framework atoms:

$$V_{\mathrm{ext}}(\mathbf{r}) = m \sum_{j \in \mathrm{frame}} 4\varepsilon_{fj}\left[\left(\frac{\sigma_{fj}}{\left\lVert \mathbf{r} - \mathbf{r}_j \right\rVert}\right)^{12} - \left(\frac{\sigma_{fj}}{\left\lVert \mathbf{r} - \mathbf{r}_j \right\rVert}\right)^{6}\right]$$

where cross parameters often use Lorentz-Berthelot mixing:

$$\sigma_{fj} = \frac{1}{2}(\sigma_f + \sigma_j), \qquad \varepsilon_{fj} = \sqrt{\varepsilon_f\varepsilon_j}$$

So the adsorbate identity enters twice: directly in the fluid functional and indirectly in the framework potential. This is useful for ML because the species dependence is not arbitrary. A model can condition on a small vector like $$(m,\sigma,\varepsilon)$$ while still predicting a full 3D density field. The methane-in-MOF-5 case is one point in that larger conditional mapping.

### The Trade-Off

GCMC and cDFT fail differently.

GCMC has the right target distribution for the chosen force field, but it pays with Markov-chain sampling. It can be painfully slow when insertions are rarely accepted.

cDFT is deterministic and often much faster, but it depends on the approximate functional. If $$F_{\mathrm{exc}}$$ is wrong for a regime, cDFT converges quickly to the wrong answer. If the fixed-point iteration is unstable, it may fail to converge at all.

This makes the two methods complementary:

- GCMC is high fidelity but expensive.
- cDFT is lower fidelity but cheaper and density-native.

That complementarity is exactly what makes the setting attractive for machine learning.

---

## What Should ML Predict?

Most ML models for adsorption predict scalar uptake. This is a reasonable first target, but it throws away physical structure.

The density field gives three observables at once:

- **Uptake:** integrate $$\rho(\mathbf{r})$$.
- **Binding sites:** inspect where $$\rho(\mathbf{r})$$ concentrates.
- **Isotherms:** evaluate the density predictor over a pressure grid and integrate at each pressure.

The critical detail is that the density should be **unnormalized**. A probability density normalized to integrate to one can tell you where molecules prefer to sit, but it cannot tell you how many molecules are present. For adsorption, the integral is the answer.

This suggests the supervised learning problem:

$$\left(\text{framework}, \text{adsorbate}, T, P\right) \longmapsto \rho_{\mathrm{eq}}(\mathbf{r})$$

For the running example, this is $$(\text{MOF-5}, \text{methane}, T, P) \mapsto \rho_{\mathrm{eq}}(\mathbf{r})$$. The input contains geometry, chemistry, and thermodynamic condition. The output is a 3D field in physical units.

### Multi-Fidelity Density Learning

The natural data sources have different cost and fidelity. cDFT can generate many solver-converged density fields across materials, gases, and pressures, while GCMC provides more expensive particle-simulation references. After coarse-graining GCMC samples into density grids, the learning problem becomes a multi-fidelity correction: use broad cDFT coverage to learn the geometry-to-density map, then use sparse GCMC labels to correct toward particle-simulation behavior.

One concrete connection for us is work from our group on this density-field view of adsorption. The idea is modest: predict $$\rho_{\mathrm{eq}}(\mathbf{r})$$ because it preserves uptake, binding-site information, and pressure-dependent behavior in one object. The prediction can also warm-start a cDFT solve rather than replace the physics solver outright.

---

## Connections to ML

Adsorption simulation is a compact example of familiar ML ideas in physical clothing. GCMC is MCMC on an open system whose particle number changes. cDFT is a variational solver that maps a potential field and thermodynamic condition to an optimal density. The density field is a richer supervised target than scalar uptake because uptake is just its integral.

The main ML lesson is to respect the physics hierarchy. Cheap approximate solvers can provide broad coverage, expensive simulations can provide correction, and the functional itself is an inductive bias rather than a nuisance to ignore. This is the same pattern that appears in many scientific ML problems.

---

## Closing

For ML researchers, the clean way to remember adsorption is:

- Adsorption is an open-system equilibrium problem.
- The natural ensemble is $$\mu VT$$ because particle number fluctuates.
- GCMC samples particles from the grand canonical distribution.
- cDFT optimizes the density field that minimizes the grand potential.
- Uptake is the integral of the density, not a separate physical object.

The methane-in-MOF-5 example is only one instance. The methodological lesson is broader than adsorption. Whenever a scientific field reports a scalar observable, ask whether it is the projection of a richer object. In adsorption, that richer object is $$\rho(\mathbf{r})$$. Learning it gives the model more structure, more supervision, and a more natural bridge between simulation and prediction.

## References

- Adams, D. J. (1975). Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. *Molecular Physics, 29*(1), 307-311. [DOI](https://doi.org/10.1080/00268977500100211).
- Evans, R. (1979). The nature of the liquid-vapour interface and other topics in the statistical mechanics of non-uniform, classical fluids. *Advances in Physics, 28*(2), 143-200. [DOI](https://doi.org/10.1080/00018737900101365).
- Roth, R. (2010). Fundamental measure theory for hard-sphere mixtures: a review. *Journal of Physics: Condensed Matter, 22*(6), 063102. [DOI](https://doi.org/10.1088/0953-8984/22/6/063102).
- Rappe, A. K., Casewit, C. J., Colwell, K. S., Goddard III, W. A. & Skiff, W. M. (1992). UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations. *Journal of the American Chemical Society, 114*(25), 10024-10035. [DOI](https://doi.org/10.1021/ja00051a040).
- Gross, J. & Sadowski, G. (2001). Perturbed-chain SAFT: An equation of state based on a perturbation theory for chain molecules. *Industrial & Engineering Chemistry Research, 40*(4), 1244-1260. [DOI](https://doi.org/10.1021/ie0003887).
- Dubbeldam, D., Calero, S., Ellis, D. E. & Snurr, R. Q. (2016). RASPA: molecular simulation software for adsorption and diffusion in flexible nanoporous materials. *Molecular Simulation, 42*(2), 81-101. [DOI](https://doi.org/10.1080/08927022.2015.1010082).
- Sauer, E. & Gross, J. (2017). Classical density functional theory for liquid-fluid interfaces and confined systems. *Industrial & Engineering Chemistry Research, 56*(14), 4119-4135. [DOI](https://doi.org/10.1021/acs.iecr.6b04551).
- Dufour-Decieux, V., Rehner, P., Schilling, J., Moubarak, E., Gross, J. & Bardow, A. (2025). Classical density functional theory as a fast and accurate method for adsorption property prediction of porous materials. *AIChE Journal, 71*(6), e18779.
- Ran, Y. A., et al. (2024). RASPA3: A Monte Carlo code for computing adsorption and diffusion in nanoporous materials and thermodynamics properties of fluids. *The Journal of Chemical Physics, 161*(11).
- Thiele, N., et al. (2026). Efficient prediction of multicomponent adsorption isotherms and enthalpies of adsorption in MOFs using classical density functional theory. *The Journal of Physical Chemistry B*.

---

[^adswords]: The naming is easy to mix up: the adsorbent is the material doing the adsorbing; the adsorbate is the molecule being adsorbed.

[^mof]: Metal-organic frameworks are crystalline porous materials made from metal nodes connected by organic linkers. Their large internal surface areas make them common targets for gas-storage and separation studies.

[^chempot]: In the grand canonical ensemble, $$\mu$$ is imposed by the reservoir. The adsorbed system then chooses its average particle number by balancing the reward for adding particles against the energetic and entropic cost of fitting them into the pore.

[^gcmc]: The exact insertion and deletion acceptance probabilities include volume, particle-number, and de Broglie-wavelength factors. The simplified expression in the main text keeps the part most relevant for ML intuition: the competition between energy $$U$$ and chemical potential $$\mu N$$.

[^cdft]: The abbreviation cDFT is overloaded. In electronic-structure papers, "cDFT" can mean constrained DFT. In this post it always means classical density functional theory.

[^pcsaft]: SAFT stands for statistical associating fluid theory. PC-SAFT extends it to chain molecules; in adsorption cDFT it supplies an approximate excess free-energy functional for the confined fluid.
