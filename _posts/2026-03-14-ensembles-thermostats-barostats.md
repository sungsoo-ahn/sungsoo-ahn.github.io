---
layout: post
title: "Ensembles, Thermostats, and Barostats for ML Researchers"
date: 2026-03-14
last_updated: 2026-03-15
description: "Statistical mechanics for ML researchers — from Newton's equations to ensembles, thermostats, barostats, Monte Carlo, and connections to generative modeling."
order: 1
categories: [science]
tags: [statistical-mechanics, molecular-dynamics, monte-carlo]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post introduces statistical mechanics and molecular simulation for ML researchers who encounter terms like NVT, NPT, and GCMC in scientific papers but find textbooks starting from Carnot cycles unhelpful. I build bottom-up from Newton's equations using probability language that ML people already know, and close with connections to generative modeling. This complements my earlier posts on <a href="/blog/2026/fokker-planck-equation/">the Fokker-Planck equation</a> and <a href="/blog/2026/electrocatalysis-ml/">electrocatalysis</a>. Corrections are welcome.</em>
</p>

## Introduction

If you work on molecular generative models, you have seen acronyms like NVT, NPT, and GCMC in every simulation paper. These refer to **ensembles** — probability distributions over molecular configurations — and the algorithms that sample from them. Understanding what they mean is essential for knowing what your model's training data actually represents and what physical quantities your model can (and cannot) predict.

The problem is that statistical mechanics textbooks start from thermodynamics: Carnot cycles, heat engines, the second law. This is backwards for ML researchers. We already think in terms of probability distributions, sampling, and expectations. The natural entry point is: what distribution are we sampling from, and why?

This post takes that approach. We start from atoms and forces, define macroscopic quantities as expectations, introduce ensembles as modeling choices about which distribution to use, and then describe the algorithms (thermostats, barostats, Monte Carlo) that sample from these distributions.

### Roadmap

| Section | What It Explains |
|---------|-----------------|
| **The Microscopic Picture** | Phase space, potential energy, Newton's equations |
| **Macroscopic Quantities as Averages** | Temperature, pressure, energy, entropy — which are easy and which are hard |
| **Why Ensembles?** | Ensembles as modeling choices; the system-boundary picture |
| **The Four Ensembles** | NVE, NVT, NPT, $$\mu$$VT — what's fixed, what fluctuates, when to use each |
| **Thermostats** | Velocity rescaling, Nosé-Hoover, Langevin — controlling temperature |
| **Barostats** | Berendsen, Parrinello-Rahman — controlling pressure |
| **Monte Carlo** | Skip dynamics entirely, sample directly from the target distribution |
| **Connections to Generative Modeling** | EBMs, Langevin sampling, HMC, annealing, free energy estimation |

---

## Part I: From Atoms to Macroscopic Quantities

### The Microscopic Picture

A molecular system is $$N$$ atoms, each with a position $$\mathbf{r}_i \in \mathbb{R}^3$$ and velocity $$\mathbf{v}_i \in \mathbb{R}^3$$. The complete state lives in **phase space**: a $$6N$$-dimensional vector $$(\mathbf{r}^N, \mathbf{v}^N)$$.

The atoms interact through a **potential energy** function $$U(\mathbf{r}^N)$$ — the total energy stored in bonds, angles, electrostatics, and van der Waals interactions. Forces are the negative gradient:

$$\mathbf{F}_i = -\nabla_{\mathbf{r}_i} U(\mathbf{r}^N)$$

Newton's second law gives the equations of motion:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i, \qquad \frac{d\mathbf{r}_i}{dt} = \mathbf{v}_i$$

These are deterministic ODEs. Given initial conditions, the trajectory is fully determined. The total energy is the **Hamiltonian**:

$$H(\mathbf{r}^N, \mathbf{v}^N) = \underbrace{\sum_{i=1}^N \frac{1}{2}m_i |\mathbf{v}_i|^2}_{\text{kinetic energy}} + \underbrace{U(\mathbf{r}^N)}_{\text{potential energy}}$$

Newton's equations conserve $$H$$ exactly — energy flows between kinetic and potential forms but the total is constant. In practice, we integrate numerically using a **symplectic integrator** (typically velocity Verlet), which preserves this conservation up to bounded oscillations over arbitrarily long simulations. Energy conservation defines the simplest ensemble (NVE).

### Macroscopic Quantities as Averages

Macroscopic quantities — temperature, pressure, free energy — are not properties of a single microstate. They are **observables of an ensemble**: averages over the probability distribution of microstates that a system visits at equilibrium. We write $$\langle A \rangle$$ for this average — concretely, a time average along a long simulation trajectory. We will define ensembles precisely in Part II; for now, the point is that macroscopic quantities are expectations, and different quantities vary in how easy they are to estimate from samples.

**Temperature** is defined through the equipartition theorem:[^equipartition]

$$\frac{3}{2}Nk_{B}T = \left\langle \sum_{i=1}^N \frac{1}{2} m_i |\mathbf{v}_i|^2 \right\rangle$$

The right-hand side is the average total kinetic energy. Each quadratic degree of freedom contributes $$\frac{1}{2}k_{B}T$$ on average. Temperature is a property of the velocity distribution — not of any single configuration.

**Pressure** comes from the virial equation:[^virial]

$$PV = Nk_{B}T + \frac{1}{3}\left\langle \sum_{i=1}^N \mathbf{r}_i \cdot \mathbf{F}_i \right\rangle$$

The first term is the ideal gas contribution (momentum transfer to walls); the second term accounts for intermolecular forces.

**Internal energy** is the average of the Hamiltonian: $$\langle H \rangle$$.

**Entropy** measures how spread out the distribution over microstates is:

$$S = -k_{B} \langle \ln p \rangle$$

This is exactly the Shannon entropy from information theory, scaled by Boltzmann's constant $$k_{B}$$. A system with many accessible microstates (gas) has high entropy; one confined to a few states (crystal) has low entropy. The intuition is the same as in ML: entropy measures uncertainty about which microstate the system is in.

**Free energy** answers a different question: of the system's total energy $$\langle H \rangle$$, how much is available to do useful work? The answer is the **Helmholtz free energy**:

$$F = \langle H \rangle - TS$$

For the Boltzmann distribution, this equals $$F = -k_{B}T \ln Z$$, where $$Z = \int e^{-\beta H(\mathbf{r}^N, \mathbf{v}^N)} \, d\mathbf{r}^N d\mathbf{v}^N$$ is the **partition function** and $$\beta = 1/k_{B}T$$. Why is $$TS$$ energy you cannot use? Recall that $$S$$ measures your uncertainty about which microstate the system is in. To extract work, you need to direct energy in a controlled way — but if the system could be in any of many microstates, the energy is spread across random thermal motions that you cannot coordinate. The more uncertain you are (higher $$S$$), the more energy is locked in uncontrollable thermal motion. The temperature $$T$$ sets the conversion rate: each unit of entropy costs $$T$$ units of energy. So $$TS$$ is the total tax that thermal randomness imposes. What remains ($$F = \langle H \rangle - TS$$) is the energy that is "free" to do work.

At equilibrium, a system at constant $$T$$ and $$V$$ minimizes $$F$$, balancing two competing drives: lowering energy (favoring ordered, low-$$\langle H \rangle$$ states) and increasing entropy (favoring disordered, high-$$S$$ states).

Both quantities are fundamentally hard to compute. Entropy requires knowing the distribution $$p$$, not just samples from it. Free energy requires the partition function $$Z$$ — an integral over the entire phase space, intractable for any nontrivial system.

All of these quantities — easy and hard alike — are expectations over some probability distribution of microstates. The next question is: which distribution?

---

## Part II: Ensembles — Choosing What to Control

### Why Ensembles?

We want to simulate a molecular system to study its microscopic behavior — how atoms move, what configurations they visit, how structures form. But we cannot model the entire universe. We have to draw a boundary around our system and make a choice: which quantities do we simulate explicitly at the atomic level, and which macroscopic quantities do we treat as fixed boundary conditions?

For example, we might simulate 1,000 water molecules (microscopic, explicit) while assuming they sit at a fixed temperature of 300 K (macroscopic, imposed from outside). Temperature is not something we track atom by atom — it is a condition we enforce, abstracting away the vast surrounding environment (the rest of the liquid, the container walls, the room) that would maintain it in reality.

{% include figure.liquid loading="eager" path="assets/img/blog/ens_system_boundary.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The modeling choice behind every simulation. Left: the physical reality — a small region of atoms embedded in a vast environment. Right: we simulate the atoms explicitly and replace the environment with macroscopic boundary conditions (\(T\), \(P\), \(\mu\)). An ensemble is a specific choice of which conditions to impose." %}

Why is this conditioning valid? Because the environment is enormous compared to the system. When our 1,000 water molecules exchange energy with $$10^{23}$$ surrounding molecules, the environment's temperature barely changes — it acts as an effectively infinite reservoir. This separation of scales is what lets us replace a detailed atomistic environment with a single number ($$T = 300$$ K). The same logic applies to pressure (the atmosphere is much larger than our simulation box) and chemical potential (the gas reservoir is vast compared to a surface).

But which macroscopic quantities can serve as boundary conditions? A system can exchange three things with its surroundings — **energy**, **volume**, and **matter**. For each, there is a quantity that equalizes at equilibrium, and that quantity is the natural boundary condition:

| Exchange | What equalizes | Formal definition | Intuition |
|----------|---------------|-------------------|-----------|
| Energy | Temperature $$T$$ | $$(\partial E / \partial S)_{N,V}$$ | Energy cost per unit of entropy gained |
| Volume | Pressure $$P$$ | $$-(\partial F / \partial V)_{T,N}$$ | Free energy cost of expansion |
| Particles | Chemical potential $$\mu$$ | $$(\partial F / \partial N)_{T,V}$$ | Free energy cost of adding one particle |

An **ensemble** is a choice of which of these to fix. It specifies a probability distribution over microstates, defined by which macroscopic quantities are held fixed (boundary conditions) and which are allowed to fluctuate. Different choices answer different questions:

- Want to study an isolated molecule in vacuum? → NVE (fix energy)
- Want to model a protein at body temperature? → NVT (fix temperature)
- Want to match experimental lab conditions? → NPT (fix temperature and pressure)
- Want to model gas adsorption on a surface? → $$\mu$$VT (fix temperature and chemical potential)

### The Four Ensembles

{% include figure.liquid loading="eager" path="assets/img/blog/ens_four_ensembles.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The four standard ensembles. (a) NVE: an isolated system with fixed energy — thick insulating walls. (b) NVT: the system exchanges heat \(Q\) with a thermal bath at temperature \(T\). (c) NPT: heat exchange plus a movable piston maintaining external pressure. (d) \(\mu\)VT: heat and particle exchange with a reservoir at chemical potential \(\mu\) — the boundary is permeable." %}

Each ensemble is defined by its probability distribution over microstates. The naming convention tells you which macroscopic quantities are fixed — the letters are the boundary conditions.

**NVE (Microcanonical).** Fixed particle number $$N$$, volume $$V$$, total energy $$E$$. The distribution is uniform over all microstates on the constant-energy surface:

$$p(\mathbf{r}^N, \mathbf{v}^N) \propto \delta\!\left(H(\mathbf{r}^N, \mathbf{v}^N) - E\right)$$

This is the simplest ensemble — all states with the correct energy are equally likely. It describes a completely isolated system: no energy enters or leaves. Plain molecular dynamics (Verlet integration) samples this distribution, because Newton's equations conserve $$H$$. However, NVE rarely matches experimental conditions, since real systems are not perfectly insulated — energy flows between the system and its surroundings.

**NVT (Canonical).** Fixed $$N$$, $$V$$, temperature $$T$$. Now the system can exchange energy with a large environment. But why is temperature — rather than, say, average energy — the right quantity to fix as a boundary condition?

Consider two systems that can exchange energy, with fixed total $$E_1 + E_2 = E_\text{total}$$. The second law says the combined system maximizes total entropy $$S_1(E_1) + S_2(E_2)$$. The maximum occurs when:

$$\frac{\partial S_1}{\partial E_1} = \frac{\partial S_2}{\partial E_2}$$

This derivative $$\partial S / \partial E$$ is the entropy gained per unit of energy absorbed. Energy flows from the system where each joule buys less entropy to where it buys more, until the rate equalizes on both sides. The quantity that equalizes is $$1/T$$ — and this is precisely what temperature *is*:

$$T = \left(\frac{\partial E}{\partial S}\right)_{N,V}$$

Temperature is the natural boundary condition for energy exchange because it is what equalizes at thermal equilibrium. When we fix $$T$$, we are saying: the environment has this temperature, and the system will equilibrate to match it.

The distribution becomes the Boltzmann distribution:

> **Boltzmann distribution.** The probability of microstate $$(\mathbf{r}^N, \mathbf{v}^N)$$ is
>
> $$p(\mathbf{r}^N, \mathbf{v}^N) = \frac{1}{Z} e^{-\beta H(\mathbf{r}^N, \mathbf{v}^N)}, \qquad Z = \int e^{-\beta H(\mathbf{r}^N, \mathbf{v}^N)}\,d\mathbf{r}^N d\mathbf{v}^N$$
>
> where $$\beta = 1/k_{B}T$$. Low-energy states are exponentially favored. The partition function $$Z$$ normalizes the distribution but is intractable to compute.
{: .block-definition }

This is the workhorse ensemble. Temperature is the natural control variable — it's what we set in an experiment, and it determines which states are thermally accessible. Energy fluctuates around its mean; the fluctuations scale as $$\sqrt{N}$$.[^fluctuations]

The parameter $$\beta = 1/k_{B}T$$ controls how sharply the distribution concentrates on low-energy states. At low temperature (large $$\beta$$), the exponential $$e^{-\beta H(\mathbf{r}, \mathbf{v})}$$ decays steeply — only the lowest-energy states have significant probability. At high temperature (small $$\beta$$), many states become accessible. Operationally, $$\beta$$ encodes how much energy the environment is willing to supply through thermal fluctuations: a hot environment freely donates energy, a cold one confines the system near its energy minimum.

*There is a circularity that confused me initially.* In Part I, we *defined* temperature through the equipartition theorem — as a property of the velocity distribution, computed from the average kinetic energy. But in NVT, we *fix* the temperature as a boundary condition. How can we both define $$T$$ from the system's behavior and impose it from outside?

The resolution: there are two different roles for $$T$$. The $$T$$ in the Boltzmann distribution is the **environment temperature** — a property of the enormous reservoir surrounding the system, not of the system itself. We impose it as a boundary condition. The system then evolves, exchanging energy with this environment. The equipartition result $$\langle \text{kinetic energy} \rangle = \frac{3}{2}Nk_{B}T$$ is a *consequence* — a theorem that says: if the system is in thermal equilibrium with an environment at temperature $$T$$, then its average kinetic energy will be $$\frac{3}{2}Nk_{B}T$$. The two uses of $$T$$ are consistent, but their logical roles are different: one is an input (boundary condition), the other is an output (a measurable prediction). The thermostat (Part III) is the algorithm that enforces this coupling in a simulation.

**NPT (Isothermal-Isobaric).** Fixed $$N$$, pressure $$P$$, temperature $$T$$. The name says exactly what it does: "isothermal" means constant temperature (Greek *iso* = equal, *thermos* = heat), and "isobaric" means constant pressure (*baros* = weight/pressure). Now the volume $$V$$ also fluctuates — the system can expand or compress against the environment.

By the same equilibrium logic as temperature: when two systems can exchange volume, the boundary shifts until the free energy cost of expansion equalizes — that is, until both have the same $$P = -(\partial F / \partial V)_{T,N}$$. Pressure is what equalizes at mechanical equilibrium, so it is the natural boundary condition for volume exchange.

The distribution is:

$$p(\mathbf{r}^N, \mathbf{v}^N, V) \propto e^{-\beta(H(\mathbf{r}^N, \mathbf{v}^N) + PV)}$$

This matches standard laboratory conditions — most experiments are done at atmospheric pressure and controlled temperature. The $$PV$$ term is the work required to maintain a volume $$V$$ against the external pressure $$P$$. The distribution balances two drives: the system wants to minimize its energy $$H$$, but expanding the volume costs $$PV$$ of work against the atmosphere. The equilibrium volume is where these competing pressures balance.

**$$\mu$$VT (Grand Canonical).** Fixed chemical potential $$\mu$$, volume $$V$$, temperature $$T$$. Both energy and particle number $$N$$ fluctuate — the system can exchange particles with the environment. The distribution is:

$$p(\mathbf{r}^N, \mathbf{v}^N, N) \propto e^{-\beta(H(\mathbf{r}^N, \mathbf{v}^N) - \mu N)}$$

Completing the pattern: when two systems can exchange particles, particles migrate until the free energy cost of adding one particle equalizes — that is, until both have the same $$\mu = (\partial F / \partial N)_{T,V}$$. Chemical potential is what equalizes at chemical equilibrium, so it is the natural boundary condition for particle exchange. (Note: $$T$$ uses internal energy $$E$$ and entropy $$S$$ in its derivative, while $$P$$ and $$\mu$$ use free energy $$F$$. This is because $$F = E - TS$$ already contains $$T$$, so using $$F$$ to define $$T$$ would be circular.)

As with temperature, $$\mu$$ here is a property of the *environment*, not of the system itself — it is imposed as a boundary condition. Since $$N$$ is an integer, the derivative $$\partial F / \partial N$$ requires some interpretation: for large $$N$$, the discrete difference $$F(N+1) - F(N)$$ is well approximated by the derivative. Operationally, $$\mu$$ answers: "how much does the free energy change when I add one more particle?"

When the environment's $$\mu$$ is high, it readily donates particles; when $$\mu$$ is low, extracting particles is costly. For an ideal gas, $$\mu = k_{B}T \ln(P/P_0)$$ where $$P_0$$ is a reference pressure — so fixing $$\mu$$ is equivalent to specifying the external gas pressure, a concrete experimentally controllable quantity.

In the distribution, the $$-\mu N$$ term rewards having more particles — each additional particle lowers the exponent by $$\beta\mu$$. The equilibrium particle number balances this reward against the energy cost of accommodating an extra particle (the increase in $$H$$).

Why do we need this ensemble at all? NVE and NVT fix the number of particles, which is fine for a closed system like a protein in a water box. But many important systems are **open**: gas molecules adsorb onto and desorb from a catalyst surface; ions flow through a membrane channel; solvent molecules enter and leave a porous material. In these systems, $$N$$ is not something we control — it is an outcome determined by the balance between the system and its environment. The $$\mu$$VT ensemble models this by fixing the chemical potential (set by the gas pressure or solution concentration) and letting $$N$$ fluctuate to its equilibrium value.[^gcmc]

| Ensemble | Fixed | Fluctuates | Distribution | Typical use |
|----------|-------|------------|--------------|-------------|
| NVE | $$N, V, E$$ | — | Uniform on energy surface | Isolated systems, basic MD |
| NVT | $$N, V, T$$ | $$E$$ | $$\propto e^{-\beta H(\mathbf{r}, \mathbf{v})}$$ | Most simulations |
| NPT | $$N, P, T$$ | $$V, E$$ | $$\propto e^{-\beta(H(\mathbf{r}, \mathbf{v}) + PV)}$$ | Lab conditions |
| $$\mu$$VT | $$\mu, V, T$$ | $$N, E$$ | $$\propto e^{-\beta(H(\mathbf{r}, \mathbf{v}) - \mu N)}$$ | Adsorption, open systems |

---

## Part III: Making It Work — Thermostats, Barostats, and Monte Carlo

An ensemble defines a probability distribution, but how do we actually draw samples from it? The **ergodic hypothesis** provides one bridge: for a system in equilibrium, time averages along a single long trajectory equal ensemble averages over the distribution. So if we can construct dynamics whose long-time statistics match the target ensemble, running the simulation long enough gives us the samples we need.

In NVE (plain Newton's equations), the trajectory corresponds to *physical time* — each step advances the system by a real time increment (typically 1–2 femtoseconds). This means we can compute **non-equilibrium properties** — quantities that depend on how the system evolves in time, not just what states it visits at equilibrium: diffusion coefficients from how fast particles spread, reaction rates from how often transitions occur, viscosities from time-correlation functions.

However, all thermostatted methods modify the physical dynamics to some degree — the added friction (Nosé-Hoover) or noise (Langevin) perturbs the true trajectory. When dynamical properties matter, a common practice is to use a thermostat only during equilibration (to reach the target temperature), then switch to NVE for the production run where the trajectory is physically meaningful. Monte Carlo, as we will see, abandons dynamics entirely.

### Thermostats (Controlling Temperature)

The word "thermostat" comes from Greek *thermos* (heat) + *statos* (standing/fixed) — literally, "keeping heat fixed." Just like the thermostat in your house maintains a target temperature by turning heating on and off, a simulation thermostat maintains the target temperature by adding or removing kinetic energy from the atoms.

Plain MD conserves energy (NVE). To sample the NVT ensemble, we need a **thermostat** — a modification to the equations of motion that mimics coupling to a heat bath (i.e., a large environment at temperature $$T$$).

**Velocity rescaling** is the simplest approach: at each step, multiply all velocities by $$\lambda = \sqrt{T_\text{target}/T_\text{current}}$$. This forces the instantaneous temperature to exactly equal $$T_\text{target}$$. The problem: it kills fluctuations. In the true canonical ensemble, the instantaneous kinetic temperature fluctuates from step to step. Forcing it to be exactly $$T_\text{target}$$ at every instant produces the wrong distribution — the energy fluctuations are too small.

**Nosé-Hoover** introduces a fictional **friction variable** $$\xi$$ that couples to the particle velocities:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i - \xi m_i \mathbf{v}_i$$

$$\frac{d\xi}{dt} = \frac{1}{Q}\left(\sum_{i=1}^N m_i |\mathbf{v}_i|^2 - 3Nk_{B}T\right)$$

where $$Q$$ is a fictitious "mass" controlling how quickly $$\xi$$ responds. The feedback mechanism: when kinetic energy exceeds the target, $$\xi$$ increases, damping the velocities; when kinetic energy is too low, $$\xi$$ goes negative, accelerating them. Nosé and Hoover showed that this extended system samples the canonical distribution exactly.[^nosehoover]

**Langevin dynamics** takes a different approach — add friction and random noise:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_i - \gamma m_i \mathbf{v}_i + \sigma \boldsymbol{\eta}_i(t)$$

where $$\gamma$$ is the friction coefficient, $$\boldsymbol{\eta}_i(t)$$ is white noise, and $$\sigma = \sqrt{2\gamma k_{B}T m_i}$$ satisfies the **fluctuation-dissipation relation** — the balance between energy removed by friction and energy injected by noise. This is an SDE. The friction term dissipates energy; the noise term injects it; the balance produces the Boltzmann distribution as the stationary distribution. If you have read my post on the [Fokker-Planck equation](/blog/2026/fokker-planck-equation/), this is the same structure: an SDE whose density dynamics are governed by a Fokker-Planck PDE, and the stationary solution is $$p(\mathbf{r}, \mathbf{v}) \propto e^{-\beta H(\mathbf{r}, \mathbf{v})}$$.

The trade-off: Langevin dynamics destroys dynamical information (the stochastic noise scrambles the true trajectory), but it is robust and easy to implement. Nosé-Hoover is deterministic (no random noise), but can get stuck in non-ergodic oscillations for small systems.

### Barostats (Controlling Pressure)

Similarly, "barostat" comes from *baros* (weight/pressure) + *statos* — "keeping pressure fixed." A **barostat** maintains the target pressure by adjusting the simulation box volume, just as a piston maintains pressure by moving in response to force imbalances.

The NPT ensemble requires controlling pressure in addition to temperature.

**Berendsen barostat** rescales the box dimensions toward the target pressure:

$$\frac{dV}{dt} = \frac{V}{\tau_P}(P_\text{current} - P_\text{target})$$

where $$\tau_P$$ is a relaxation time. Like velocity rescaling for temperature, this is simple and effective for equilibration but produces the wrong distribution — the volume fluctuations are too small.

**Parrinello-Rahman** treats the box dimensions as dynamical variables with their own equations of motion, analogous to how Nosé-Hoover treats the friction $$\xi$$. The simulation cell vectors evolve according to:

$$W\ddot{\mathbf{h}} = (P_\text{current} - P_\text{target})V\,\mathbf{h}^{-T}$$

where $$\mathbf{h}$$ is the cell matrix (columns are box vectors), $$W$$ is a fictitious mass, and $$P_\text{current}$$ includes the virial contribution from interatomic forces. This produces the correct NPT distribution, including proper volume fluctuations.[^parrinellorahman]

The pattern is the same as for thermostats, and it is a general principle in simulation design: **crude methods** (Berendsen, velocity rescaling) force the target value directly — fast equilibration, wrong distribution. **Extended-variable methods** (Parrinello-Rahman, Nosé-Hoover) introduce auxiliary dynamical variables that couple to the system — slower to converge but produce the statistically correct ensemble.

### Monte Carlo — The Alternative

Everything above modifies Newton's equations to sample the target ensemble. Monte Carlo (MC) takes a fundamentally different approach: skip dynamics entirely and sample the distribution directly.

> **Metropolis-Hastings.** To sample from $$p(\mathbf{r}) \propto e^{-\beta U(\mathbf{r})}$$:
>
> 1. Propose a random move: $$\mathbf{r}' = \mathbf{r} + \delta\mathbf{r}$$
> 2. Compute $$\Delta U = U(\mathbf{r}') - U(\mathbf{r})$$
> 3. Accept with probability $$\min\!\left(1, \; e^{-\beta \Delta U}\right)$$
>
> Moves that lower the energy are always accepted. Moves that raise it are accepted with probability $$e^{-\beta \Delta U}$$ — exponentially unlikely for large energy increases, but allowed for small fluctuations.
{: .block-definition }

MC is philosophically clean: you define a target distribution and construct a Markov chain whose stationary distribution matches it. No fictional friction variables, no fluctuation-dissipation relations, no symplectic integrators. The acceptance criterion guarantees **detailed balance**, which guarantees convergence to the correct distribution.[^detailedbalance]

**GCMC (Grand Canonical Monte Carlo)** extends this to the $$\mu$$VT ensemble by adding particle insertion and deletion moves:

- **Insert:** Place a new particle at a random position. Accept with probability $$\min\!\left(1, \; \frac{V}{N+1} e^{-\beta(\Delta U - \mu)}\right)$$.
- **Delete:** Remove a random particle. Accept with the inverse probability.

This is how adsorption isotherms are computed — how many gas molecules adsorb onto a surface at a given chemical potential (i.e., gas pressure). GCMC is the standard method for the $$\mu$$VT ensemble because changing $$N$$ in dynamics-based approaches is difficult.[^gcmcapps]

The trade-off: MC gives no dynamical information. Its moves (random displacements, insertions) are fictitious — there is no physical time associated with a MC step. It tells you equilibrium properties (averages, distributions) but nothing about rates, diffusion, or kinetics. If you need to know *how fast* something happens, you need MD, because MD integrates the real equations of motion and its trajectory corresponds to physical time. If you only need to know *what happens at equilibrium*, MC is often simpler.

---

## Closing

This post covered two layers of molecular simulation:

1. **Statistical mechanics** provides the probability distributions (ensembles) that connect microscopic states to macroscopic observables — the Boltzmann distribution, the NPT distribution, the grand canonical distribution. The choice of ensemble is a modeling decision about what to simulate explicitly and what to impose as a boundary condition.
2. **Simulation algorithms** are the tools that sample from these distributions — thermostats and barostats modify the equations of motion; Monte Carlo bypasses dynamics entirely and samples directly.

A third layer — **thermodynamics** — tells you *what* happens at equilibrium (which phase is stable, whether a reaction is spontaneous) without reference to microscopic details. We did not cover it here, but the quantities it works with (free energy, entropy, chemical potential) are the same ones that appeared throughout this post.

### Connections to Generative Modeling

Many ideas in this post have direct counterparts in modern generative models — the intellectual traffic goes both ways.

**The Boltzmann distribution is an energy-based model.** The canonical distribution $$p(\mathbf{x}) \propto e^{-\beta U(\mathbf{x})}$$ is exactly the form of an energy-based model (EBM) in ML. The potential energy $$U$$ is the learned energy function, and $$\beta$$ controls the sharpness of the distribution. The intractable partition function $$Z$$ is the same normalization problem that makes EBM training difficult.

**Langevin dynamics is score-based sampling.** The overdamped Langevin equation $$d\mathbf{x} = -\frac{1}{\gamma}\nabla U(\mathbf{x})\,dt + \sqrt{2k_{B}T/\gamma}\,d\mathbf{W}$$ (the high-friction limit of the thermostat in Part III, where inertia is negligible) is the same Langevin MCMC used to sample from unnormalized densities in ML. Score-based diffusion models (Song & Ermon, 2019) use this connection explicitly: the score $$\nabla_\mathbf{x} \log p(\mathbf{x}) = -\beta \nabla U(\mathbf{x})$$ is the force field, and sampling is a Langevin simulation at the learned energy landscape.

**Metropolis-Hastings is the backbone of MCMC in ML.** The accept/reject mechanism from Monte Carlo is the same algorithm used for posterior sampling in Bayesian inference. Hamiltonian Monte Carlo (HMC) — widely used in probabilistic programming (Stan, PyMC) — borrows directly from molecular dynamics: it runs short NVE trajectories as proposals, using the Hamiltonian structure to make large, low-rejection moves through parameter space.

**Simulated annealing is temperature scheduling.** Varying $$\beta$$ during sampling — starting at high temperature (flat distribution, broad exploration) and cooling toward low temperature (sharp distribution, concentrated on modes) — is simulated annealing. The same idea appears in diffusion models, where the noise schedule interpolates between a broad prior and the data distribution.

**Free energy estimation connects to variational inference.** The identity $$F = -k_{B}T \ln Z$$ means estimating free energy differences is equivalent to estimating ratios of partition functions — the same problem as computing marginal likelihoods in Bayesian model comparison. Techniques like thermodynamic integration and free energy perturbation have ML counterparts in annealed importance sampling and variational bounds.

---

## References

- Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation: From Algorithms to Applications*. Academic Press. The standard reference for MD and MC methods.
- Tuckerman, M. E. (2010). *Statistical Mechanics: Theory and Molecular Simulation*. Oxford University Press.
- Nosé, S. (1984). A unified formulation of the constant temperature molecular dynamics methods. [J. Chem. Phys. 81, 511](https://doi.org/10.1063/1.447334).
- Parrinello, M. & Rahman, A. (1981). Polymorphic transitions in single crystals: A new molecular dynamics method. [J. Appl. Phys. 52, 7182](https://doi.org/10.1063/1.328693).
- Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. [NeurIPS 2019](https://arxiv.org/abs/1907.05600).

---

[^equipartition]: Strictly, each quadratic degree of freedom contributes $$\frac{1}{2}k_{B}T$$. For $$N$$ atoms in 3D, there are $$3N$$ translational degrees of freedom, giving $$\langle \text{kinetic energy} \rangle = \frac{3}{2}Nk_{B}T$$. Constraints (e.g., rigid bonds) reduce the count.

[^virial]: The virial equation follows from the virial theorem of classical mechanics. The sum $$\sum_i \mathbf{r}_i \cdot \mathbf{F}_i$$ is the virial — it measures how much the interatomic forces contribute to pressure beyond the ideal gas term.

[^fluctuations]: In the canonical ensemble, the relative energy fluctuation is $$\sigma_E / \langle E \rangle \sim 1/\sqrt{N}$$. For $$N \sim 10^{23}$$ (a mole of atoms), fluctuations are negligible and NVE and NVT give identical results. The distinction matters for small systems — which is most of what we simulate.

[^nosehoover]: Nosé (1984) introduced the extended Lagrangian; Hoover (1985) reformulated it as coupled first-order equations. The combined "Nosé-Hoover thermostat" is standard in all major MD codes (GROMACS, LAMMPS, OpenMM).

[^parrinellorahman]: Parrinello and Rahman (1981) originally introduced the method for studying structural phase transitions in crystals, where the box shape (not just size) changes.

[^gcmc]: Grand canonical Monte Carlo is particularly important for studying gas storage in porous materials (zeolites, metal-organic frameworks) and ion channel selectivity in biology.

[^detailedbalance]: Detailed balance means that the probability of being in state $$A$$ and transitioning to $$B$$ equals the probability of being in $$B$$ and transitioning to $$A$$: $$p(A)\,T(A \to B) = p(B)\,T(B \to A)$$. The Metropolis criterion is designed to satisfy this for $$p \propto e^{-\beta U}$$.

[^gcmcapps]: In practice, GCMC is the only practical method for the $$\mu$$VT ensemble. Inserting or deleting particles in MD would require discontinuous changes to the equations of motion, while MC handles it naturally through the accept/reject framework.
