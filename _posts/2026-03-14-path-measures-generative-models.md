---
layout: post
title: "From Jarzynski's Equality to Diffusion Models"
date: 2026-03-14
last_updated: 2026-03-15
description: "From Jarzynski's equality to diffusion models — path measures unify free energy estimation, AIS, diffusion models, and GFlowNets as instances of the same mathematics."
order: 1
categories: [science]
tags: [non-equilibrium-statistical-mechanics, path-measures, free-energy, generative-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post connects two fields I've worked in from different entry points — non-equilibrium statistical mechanics from the molecular simulation side, and generative models from the ML side. The punchline is that several ML methods (annealed importance sampling, diffusion models, GFlowNet trajectory balance) are instances of the same mathematical framework that physicists developed in the 1990s–2000s to understand systems driven out of equilibrium. The connection became concrete for me while working on <a href="https://arxiv.org/abs/2405.19961">transition path sampling with diffusion models</a> — we wanted to use diffusion models to connect folded and unfolded protein states and estimate the free energy difference between them, which led us directly to Jarzynski's equality. I learned the broader framework from studying and collaborating with <a href="https://chertkov.github.io/">Michael Chertkov</a>, whose work on fluctuation theorems and path integral control deeply shaped how I think about these connections. I wrote this post because the connection is underappreciated, and making it explicit improves both how we design methods and how we understand what they compute. This complements my earlier posts on <a href="/blog/2026/fokker-planck-equation/">the Fokker-Planck equation</a> and <a href="/blog/2026/ensembles-thermostats-barostats/">ensembles, thermostats, and barostats</a>. Corrections are welcome.</em>
</p>

## Introduction

A diffusion model transforms noise into data by learning to reverse a noising process. The forward process (data $$\to$$ noise) and the reverse process (noise $$\to$$ data) are two stochastic processes running in opposite directions — two probability distributions over trajectories, not single points. The training loss turns out to be the KL divergence between these trajectory distributions, and a perfectly trained model is one where the forward and reverse trajectory distributions are identical.

This structure — two processes running in opposite directions, their ratio encoding something useful — is not unique to diffusion models. It is exactly the mathematical framework that physicists developed in the 1990s to understand systems driven out of equilibrium. In physics, state A might be a protein with a drug unbound and state B the drug bound; in diffusion models, state A is the data distribution and state B is Gaussian noise. The mathematics is the same.

My previous post on ensembles ended with a claim: free energy is fundamentally hard to compute because it requires the partition function $$Z$$ — an integral over the entire phase space. But free energy *differences* between two states are exactly what we need in practice. The sign of $$\Delta F$$ determines which state nature prefers: does a protein fold, does a drug bind, is one crystal form more stable than another? (Part I defines these precisely.)

The equilibrium approach computes $$\Delta F$$ directly, which requires sampling from both endpoints — intractable when the two states are separated by high barriers. The key insight of non-equilibrium statistical mechanics is that you don't need equilibrium samples at all. You can *drive* the system from state A to state B along any path — fast, slow, out of equilibrium — and still recover the exact free energy difference.

Here is how. Imagine smoothly deforming the potential energy from $$U_A$$ to $$U_B$$ over some time interval — like slowly stretching a spring, or gradually turning on an electric field. As the potential changes, the system's particles move in response, and the energy you must supply to carry out this deformation is the **work** $$W$$. Concretely, at each instant you change the potential by a small amount, and the work is the integral of that change along the trajectory the system actually follows (Part I gives the precise formula). The crucial point: $$W$$ depends on the *entire trajectory*, not just the endpoints. Two runs of the same protocol give different $$W$$ because thermal noise makes each microscopic trajectory different — work is a random variable.

On average, the second law guarantees $$\langle W \rangle \geq \Delta F$$: you always waste some energy as heat (dissipation). But Jarzynski (1997) showed that an *exponential* average recovers $$\Delta F$$ exactly:

$$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$

where $$\beta = 1/k_BT$$ is the inverse temperature (defined precisely in Part I).

The quantity $$e^{-\beta W}$$ — the exponentiated negative work — reweights each trajectory by how "reversible" it was. Trajectories where the system got lucky (low work, close to $$\Delta F$$) receive high weight; trajectories where dissipation was large receive low weight. The exponential average is unbiased regardless of how far from equilibrium you drove the system.

This sounds like a simulation trick. It's not. The mathematical framework behind Jarzynski — path measures, Radon-Nikodym derivatives between forward and reverse trajectory distributions, work-weighted reweighting — is already standard in ML, hiding in plain sight. **Annealed importance sampling** (AIS), the method most commonly used to evaluate normalizing flows and energy-based models, *is* Jarzynski's equality. The importance weights in AIS are precisely $$e^{-\beta W}$$ — the exponentiated negative work. The annealing schedule is the protocol $$\lambda(t)$$. And the variance problem that plagues AIS (rare high-weight samples dominating the estimate) is the same variance problem physicists have studied since 1997: most trajectories dissipate too much work, so the exponential average is dominated by rare low-work runs.

Once you see AIS as Jarzynski, you notice the same mathematics appearing elsewhere — in diffusion models and GFlowNets. Part V makes these connections precise.

**What this post is not.** This is not a survey of free energy methods in molecular simulation (thermodynamic integration, free energy perturbation, metadynamics, umbrella sampling). Those are *applications* of the framework. This post develops the *framework itself* — path measures and non-equilibrium equalities — and then maps it onto generative models.

### Roadmap

| Section | What It Explains |
|---------|-----------------|
| **Thermodynamic States and Free Energy** | Boltzmann distribution, partition function, Helmholtz free energy — self-contained definitions |
| **Free Energy Differences** | Zwanzig's identity, the overlap problem, why direct comparison fails |
| **Non-Equilibrium Processes and Work** | Protocols, work, dissipation — driving a system out of equilibrium |
| **Path Measures, RND, Girsanov** | Path integral intuition, Radon-Nikodym derivatives, Girsanov's theorem, discrete verification |
| **Forward-Backward SDEs** | Forward-backward RND (Vargas et al.), Nelson's relation, discrete forward-reverse ratio |
| **Non-Equilibrium Equalities** | The work identity, Jarzynski, Crooks, dissipation as KL divergence |
| **Connections to Generative Models** | AIS, diffusion models, GFlowNets |
| **What This Unification Buys You** | Practical implications for method design and diagnostics |

Parts II–IV develop the mathematics in detail. If you are primarily interested in the ML connections, you can read Part I for the physics setup, then skip to Part V — each connection there summarizes the relevant mathematics as needed.

---

## Part I: The Free Energy Problem

### Configurations, States, and the Boltzmann Distribution

Consider $$N$$ atoms with positions $$\mathbf{x} = (\mathbf{r}_1, \ldots, \mathbf{r}_N) \in \mathbb{R}^{3N}$$. A single assignment of all positions — one snapshot of the system — is a **configuration**. A potential energy function $$U(\mathbf{x})$$ assigns an energy to each configuration based on the interactions between atoms (bonds, electrostatics, van der Waals forces, etc.).

In ML terms, think of $$\mathbf{x}$$ as a data point in $$\mathbb{R}^{3N}$$, and $$U(\mathbf{x})$$ as a negative log-probability (up to a constant). A **thermodynamic state** is not a single configuration — it is the *entire probability distribution* over configurations defined by $$U$$. At temperature $$T$$, this distribution is the **Boltzmann distribution**:

> **Boltzmann distribution.**
>
> $$p(\mathbf{x}) = \frac{e^{-\beta U(\mathbf{x})}}{Z}, \qquad Z = \int e^{-\beta U(\mathbf{x})} \, d\mathbf{x}, \qquad \beta = \frac{1}{k_B T}$$
>
> The Boltzmann factor $$e^{-\beta U(\mathbf{x})}$$ assigns high probability to low-energy configurations. The **partition function** $$Z$$ normalizes the distribution — it sums the Boltzmann weight over all possible configurations.
{: .block-definition }

A state is therefore an energy-based model: $$U$$ defines the unnormalized log-density, and $$Z$$ is the intractable normalizing constant — for even $$N = 100$$ atoms, it is an integral over $$\mathbb{R}^{300}$$.

**Why different states have different potentials.** The potential $$U(\mathbf{x})$$ encodes *everything* about the physical setup: which atoms are present, how they interact, and any external conditions. Changing the physical setup changes $$U$$, which defines a new state. For example:

- **Drug binding.** State A: protein and drug molecule simulated separately in solvent — $$U_A$$ includes protein-solvent and drug-solvent interactions but no protein-drug interactions. State B: protein and drug simulated together — $$U_B$$ adds the protein-drug interaction terms. Same atoms, different $$U$$ because the interaction terms change.
- **Alchemical transformation.** To compare two drug candidates, state A uses the force field parameters of molecule 1 and state B uses those of molecule 2. The potential $$U$$ changes because the atomic charges, Lennard-Jones parameters, or even the number of atoms differ.
- **Crystal polymorphism.** Same molecule, but $$U_A$$ and $$U_B$$ include different periodic boundary conditions and lattice geometries, giving different packing interactions.

In each case, the configuration space $$\mathbf{x} \in \mathbb{R}^{3N}$$ is the same (or can be made the same via dummy atoms), but $$U_A(\mathbf{x}) \neq U_B(\mathbf{x})$$. This gives two Boltzmann distributions $$p_A(\mathbf{x})$$ and $$p_B(\mathbf{x})$$ with partition functions $$Z_A$$ and $$Z_B$$.

*AIS parallel: In annealed importance sampling, state A is the tractable prior $$p_0$$ (e.g., a Gaussian) and state B is the intractable target $$p_K$$ (e.g., the model distribution whose normalizing constant you want to estimate). The "potential" $$U$$ is $$-\log \hat{p}$$ (the negative log-unnormalized density), and different states correspond to different distributions in the annealing chain.*

### Free Energy

The **Helmholtz free energy** packages the intractable partition function into a single thermodynamic quantity:

> **Helmholtz free energy.**
>
> $$F = -k_B T \ln Z = \langle U \rangle - TS$$
>
> where $$\langle U \rangle = \mathbb{E}_{p}[U(\mathbf{x})]$$ is the average potential energy under the Boltzmann distribution and $$S$$ is the entropy. Free energy balances energy (low $$U$$ — the system finds favorable interactions) against entropy (high $$S$$ — many configurations are accessible). The equilibrium state minimizes $$F$$.
{: .block-definition }

**Notation.** Throughout this post, angle brackets $$\langle \cdot \rangle$$ denote expectations (averages): $$\langle f \rangle = \mathbb{E}_p[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \, d\mathbf{x}$$. A subscript indicates which distribution the average is over — $$\langle \cdot \rangle_A$$ means averaging over the Boltzmann distribution of state A.

Since each state has its own $$U$$ and $$Z$$, each state has its own free energy: $$F_A = -k_B T \ln Z_A$$ and $$F_B = -k_B T \ln Z_B$$. We rarely need $$F_A$$ or $$F_B$$ individually — we need their *difference*. The sign of $$\Delta F = F_B - F_A$$ tells us which state is thermodynamically favored: if $$\Delta F < 0$$, state B is more stable (lower free energy); if $$\Delta F > 0$$, state A wins. The magnitude tells us *how much* more stable — whether the preference is marginal or overwhelming.

*AIS parallel: The free energy difference is $$\Delta F = -k_BT \ln(Z_B/Z_A)$$. In AIS, this is $$-k_BT$$ times the log-ratio of normalizing constants — exactly what AIS estimates. If $$p_0$$ is a standard Gaussian with known $$Z_0$$, then estimating $$Z_B/Z_A$$ gives $$Z_B$$ directly.*

This makes $$\Delta F$$ the central quantity in three classes of problems:

- **Protein folding.** State A is the unfolded ensemble (high entropy, many disordered conformations). State B is the folded state (low energy, compact structure with favorable contacts). The protein folds spontaneously if $$\Delta F_{\text{unfolded} \to \text{folded}} < 0$$ — the energy gain from forming contacts outweighs the entropy loss from ordering. Typical folding free energies are small: 5–15 kcal/mol, the difference between large opposing terms. Predicting the sign correctly requires getting both the energy and entropy right.

- **Drug binding.** State A is the drug and protein separated in solution. State B is the drug bound in the protein's active site. The **binding free energy** $$\Delta F_{\text{bind}}$$ determines affinity: a drug with $$\Delta F_{\text{bind}} = -10$$ kcal/mol binds $$\sim 10^7$$ times more tightly than one with $$\Delta F_{\text{bind}} = -1$$ kcal/mol (since the equilibrium constant goes as $$K \propto e^{-\beta \Delta F}$$). Binding free energy calculation is the gold standard for computational prediction of binding affinity — it is the only approach that is both rigorous (grounded in statistical mechanics, not heuristic scoring) and accounts for the entropic costs of binding (loss of translational, rotational, and conformational freedom). Pharmaceutical companies routinely use free energy perturbation (FEP) calculations to prioritize drug candidates before synthesis, and achieving $$\sim$$1 kcal/mol accuracy in $$\Delta\Delta F$$ predictions is considered state-of-the-art.

- **Crystal polymorphism.** The same molecule can pack into different crystal structures (polymorphs). State A and B are two such packing arrangements, each with its own $$U_A, U_B$$. The polymorph with lower $$F$$ is the one that forms at equilibrium. Getting this wrong has real consequences — the wrong polymorph of a pharmaceutical can have different solubility, bioavailability, or stability (famously, ritonavir had to be reformulated after a more stable polymorph appeared).

In all three cases, the computational challenge is identical: compute $$\Delta F$$ from the potential energy functions $$U_A$$ and $$U_B$$.

### Free Energy Differences and Why They're Hard

The free energy difference between states A and B is:

> **Free energy difference.**
>
> $$\Delta F = F_B - F_A = -k_B T \ln \frac{Z_B}{Z_A}$$
>
> where $$Z_A = \int e^{-\beta U_A(\mathbf{x})} \, d\mathbf{x}$$ and $$Z_B = \int e^{-\beta U_B(\mathbf{x})} \, d\mathbf{x}$$.
{: .block-definition }

Computing $$Z_A$$ and $$Z_B$$ individually is intractable, but their *ratio* can in principle be estimated. Zwanzig (1954) showed how:

> **Zwanzig's identity (free energy perturbation).**
>
> $$e^{-\beta \Delta F} = \left\langle e^{-\beta (U_B - U_A)} \right\rangle_A$$
>
> where $$\langle \cdot \rangle_A$$ denotes an average over the equilibrium distribution of state A.
{: .block-definition }

This is exact but useless in practice. The average is dominated by rare configurations where $$U_B(\mathbf{x}) - U_A(\mathbf{x})$$ is small — configurations that lie in the overlap between the two Boltzmann distributions. When A and B are very different (the interesting case), this overlap is exponentially small, and the estimator has exponentially large variance.

{% include figure.liquid loading="eager" path="assets/img/blog/pm_boltzmann_overlap.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Two Boltzmann distributions with minimal overlap. The shaded region is where the Zwanzig estimator gets its signal — exponentially small when A and B are far apart." %}

The core problem: we need a bridge between A and B that doesn't require direct overlap between their equilibrium distributions. Enter non-equilibrium methods.

---

### Non-Equilibrium Processes and Work

The idea: instead of comparing A and B directly, *continuously deform* one into the other. Define a time-dependent potential $$U(\mathbf{x}, \lambda(t))$$ controlled by a parameter $$\lambda$$ that interpolates from state A ($$\lambda = 0$$) to state B ($$\lambda = 1$$) over time $$t \in [0, \tau]$$:

$$U(\mathbf{x}, \lambda) = (1 - \lambda) U_A(\mathbf{x}) + \lambda \, U_B(\mathbf{x})$$

This linear interpolation is the simplest choice; other interpolation schemes (geometric, optimized) are possible and sometimes preferable. The function $$\lambda(t)$$ is the **protocol** — our choice of how fast to switch between states.

*AIS parallel: The protocol is the **annealing schedule** $$\beta_0, \beta_1, \ldots, \beta_K$$. The intermediate distributions are $$p_k(\mathbf{x}) \propto p_0(\mathbf{x})^{1-\beta_k} p_K(\mathbf{x})^{\beta_k}$$ — a geometric interpolation, analogous to the linear interpolation of potentials above. More intermediate steps (larger $$K$$) correspond to a slower protocol.*

**The dynamics.** While the protocol runs, the particles don't just sit still — they move according to the forces from the current potential, plus random thermal kicks from the heat bath. This is **overdamped Langevin dynamics**:

$$d\mathbf{x}(t) = -\nabla U(\mathbf{x}(t), \lambda(t)) \, dt + \sqrt{2/\beta} \, d\mathbf{w}(t)$$

The first term is deterministic: the particles slide downhill on the current potential energy surface $$U(\mathbf{x}, \lambda(t))$$. The second term is stochastic: $$\mathbf{w}(t)$$ is a Wiener process (continuous-time white noise) representing random collisions with solvent molecules at temperature $$T$$. Together, these define a stochastic trajectory $$\mathbf{x}(t)$$ — the path the system actually takes as the potential is being deformed.

The system starts in thermal equilibrium at state A: $$\mathbf{x}(0) \sim p_A(\mathbf{x}) = e^{-\beta U_A(\mathbf{x})}/Z_A$$. As the protocol advances, the potential changes and the particles respond — but if the protocol is fast, the particles can't keep up, and the system falls out of equilibrium.

*AIS parallel: In discrete time, the dynamics correspond to MCMC transitions. At each annealing level $$k$$, AIS applies one or more MCMC steps (e.g., Metropolis-Hastings, HMC) targeting $$p_k$$ to the current sample $$\mathbf{x}_k$$. The MCMC transition is the discrete analogue of the Langevin SDE above — it moves the sample toward equilibrium at the current distribution, with randomness from the proposal.*

{% include figure.liquid loading="eager" path="assets/img/blog/pm_double_well_protocol.png" class="img-fluid rounded z-depth-1" zoomable=true caption="A non-equilibrium protocol: the parameter \(\lambda\) smoothly tilts a double-well potential from state A (left well deeper) to state B (right well deeper). The particle (dot) starts in equilibrium at A and is driven to B." %}

**Work: the energy cost of driving the system.** As the protocol runs, the potential changes underneath the particles. At each instant, the particles sit at positions $$\mathbf{x}(t)$$ (determined by the SDE above), and the protocol shifts the potential by $$\dot{\lambda}(t) \, dt$$. The **work** is the total energy cost of these shifts, accumulated along the trajectory:

> **Work along a trajectory.**
>
> $$W[\mathbf{x}(\cdot)] = \int_0^\tau \frac{\partial U}{\partial \lambda}(\mathbf{x}(t), \lambda(t)) \cdot \dot{\lambda}(t) \, dt$$
>
> The integrand $$\partial U / \partial \lambda$$ measures how sensitive the potential energy is to a change in $$\lambda$$ at the current configuration $$\mathbf{x}(t)$$. The rate $$\dot{\lambda}(t)$$ measures how fast we are changing the protocol. Their product is the instantaneous rate at which we inject energy into the system. For the linear interpolation above, $$\partial U / \partial \lambda = U_B(\mathbf{x}) - U_A(\mathbf{x})$$: the work at each instant measures the energy difference between B and A at the configuration the system happens to occupy.
{: .block-definition }

A physical analogy: imagine slowly stretching a rubber band (the potential changes) while a heavy ball sits on it (the configuration). The work you do depends on *where the ball is* at each moment you stretch — if the ball happens to be near the attachment point, stretching costs little energy; if it's far away, it costs a lot. Run the experiment twice with thermal noise jiggling the ball, and you get different work values. Work is a **random variable** — a functional of the stochastic trajectory $$\mathbf{x}(\cdot)$$, not a function of the endpoints alone.

*AIS parallel: The discrete-time work is the negative log importance weight. At each annealing step, AIS computes the ratio $$p_{k+1}(\mathbf{x}_k) / p_k(\mathbf{x}_k)$$ — how much the unnormalized density changes at the current sample when the distribution shifts from $$p_k$$ to $$p_{k+1}$$. The total importance weight is $$w = \prod_k p_{k+1}(\mathbf{x}_k) / p_k(\mathbf{x}_k)$$, so $$\log w = \sum_k [\log p_{k+1}(\mathbf{x}_k) - \log p_k(\mathbf{x}_k)]$$. Compare with the continuous work integral: $$W = \int_0^\tau (\partial U / \partial \lambda) \dot{\lambda} \, dt$$. Both accumulate the change in log-density (or energy) at the current configuration as the distribution is deformed. Different MCMC chains give different $$w$$ values, just as different trajectories give different $$W$$.*

**Quasistatic vs. driven processes.** How fast we run the protocol determines whether we can recover $$\Delta F$$:

- **Quasistatic (infinitely slow):** The system remains in equilibrium at every instant — the Boltzmann distribution adjusts to the current potential $$U(\mathbf{x}, \lambda(t))$$ before $$\lambda$$ changes further. In this limit, $$W = \Delta F$$ exactly. But this takes infinite time.
- **Finite-time (driven):** The system falls out of equilibrium — the Boltzmann distribution lags behind the changing potential. On average, $$\langle W \rangle > \Delta F$$ — the average excess is the **dissipated work** $$\langle W_{\text{diss}} \rangle = \langle W \rangle - \Delta F \geq 0$$. This is the second law of thermodynamics restated for driven processes: you always waste some energy as heat when you drive a system at finite speed.

Even though $$\langle W \rangle \geq \Delta F$$ (second law), there is a specific *exponential average* of $$W$$ that gives $$\Delta F$$ exactly. This is Jarzynski's equality, and proving it requires reasoning about path measures.

---

## Part II: Path Measures, Radon-Nikodym Derivatives, and Girsanov's Theorem

**The big picture.** Part I ended with a question: how do you extract $$\Delta F$$ from stochastic work measurements? The answer requires comparing two processes — the forward protocol (A $$\to$$ B) and the reverse protocol (B $$\to$$ A) — and asking how their *trajectory-level* statistics differ. This part builds the mathematical tools for that comparison: path measures (probability distributions over trajectories), Radon-Nikodym derivatives (how to compute ratios between them), and Girsanov's theorem (the specific formula for SDEs). These are the same tools that underlie importance weighting in AIS, the ELBO in diffusion models, and trajectory balance in GFlowNets.

Everything so far in this series has been about probability distributions over *states* — single configurations $$\mathbf{x}$$. But work is defined on *trajectories* — entire time-histories $$\mathbf{x}(\cdot) = \{\mathbf{x}(t) : t \in [0, T]\}$$. To reason about work, we need probability distributions over trajectories. These are **path measures**.[^pathmeasure]

*AIS parallel: A path measure is the distribution over entire AIS sample chains $$(\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_K)$$, not individual samples $$\mathbf{x}_k$$. The chain includes the initial draw from $$p_0$$ and all $$K$$ MCMC transitions. Two runs of AIS produce different chains — the path measure describes the distribution over all possible chains.*

[^pathmeasure]: I use "path measure" throughout this post. Physicists often say "path integral" for the same concept — summing/integrating over all possible trajectories weighted by an action. The mathematical content is closely related to Feynman's path integral in quantum mechanics, but our context is classical stochastic dynamics rather than quantum amplitudes.

### The Path Integral Picture

Physicists have a powerful way to think about path measures: the **Feynman-Kac path integral**. Consider a particle diffusing in a time-dependent potential $$U(\mathbf{x}, \lambda(t))$$. The probability of observing a specific trajectory $$\mathbf{x}(\cdot)$$ is weighted by an exponential of the **action** along that path:

$$\mathcal{P}[\mathbf{x}(\cdot)] \propto \exp\left(-\frac{1}{2} \int_0^T \lvert \dot{\mathbf{x}}(t) + \nabla U(\mathbf{x}(t), \lambda(t)) \rvert^2 \, dt\right)$$

This is the **Onsager-Machlup action** for overdamped Langevin dynamics. Each trajectory gets a weight determined by how "surprising" it is — trajectories that follow the force field ($$\dot{\mathbf{x}} \approx -\nabla U$$) have low action and high weight, while trajectories that fight the forces have high action and low weight.[^underdamped]

[^underdamped]: The underdamped case adds velocity degrees of freedom and a kinetic energy term to the action, but the conceptual structure is the same. Overdamped Langevin is the standard setting for the ML connections because it matches the dynamics used in diffusion models and score-based methods.

**Discrete derivation of the Onsager-Machlup action.** To see where this comes from, discretize time into $$N$$ steps of size $$\Delta t$$. At each step, the Langevin SDE says $$\mathbf{x}_{k+1} = \mathbf{x}_k - \nabla U_k \Delta t + \sqrt{2/\beta} \, \boldsymbol{\xi}_k$$ where $$\boldsymbol{\xi}_k \sim \mathcal{N}(0, \Delta t \, \mathbf{I})$$. The transition probability is Gaussian:

$$P(\mathbf{x}_{k+1} \mid \mathbf{x}_k) \propto \exp\left(-\frac{\beta}{4 \Delta t} \lvert \mathbf{x}_{k+1} - \mathbf{x}_k + \nabla U_k \Delta t \rvert^2\right)$$

The full path probability is the product over all steps:

$$\mathcal{P}_F[\mathbf{x}_0, \ldots, \mathbf{x}_N] = p_A(\mathbf{x}_0) \cdot \prod_{k=0}^{N-1} P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$

Substituting the Gaussian form and collecting the exponents:

$$\propto p_A(\mathbf{x}_0) \cdot \exp\left(-\frac{\beta}{4\Delta t} \sum_{k=0}^{N-1} \lvert \Delta \mathbf{x}_k + \nabla U_k \Delta t \rvert^2\right)$$

In the continuous limit ($$N \to \infty$$, $$\Delta t \to 0$$), the discrete sum $$\frac{\beta}{4\Delta t} \sum_k \lvert \Delta\mathbf{x}_k + \nabla U_k \Delta t \rvert^2$$ becomes $$\frac{1}{2}\int_0^T \lvert \dot{\mathbf{x}} + \nabla U \rvert^2 dt$$ (using $$\beta/4 = 1/(2\sigma^2)$$ with $$\sigma^2 = 2/\beta$$) — the Onsager-Machlup action. The discrete version is what we actually compute; the continuous version is the formal notation.

*AIS parallel: The discrete path probability is exactly the AIS chain probability. In AIS, the chain $$(\mathbf{x}_0, \ldots, \mathbf{x}_K)$$ has probability $$p_0(\mathbf{x}_0) \cdot \prod_k T_k(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$, where $$T_k$$ is the MCMC transition kernel at level $$k$$. The Gaussian transition kernel $$P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$ above is a specific choice of $$T_k$$ — one step of unadjusted Langevin dynamics. Other MCMC kernels (Metropolis-Hastings, HMC) give different $$T_k$$ but the same path measure structure.*

The path integral picture gives us a way to *assign weights* to individual trajectories. But what we actually need for Jarzynski and Crooks is the *ratio* of weights between the forward process (A $$\to$$ B) and the reverse process (B $$\to$$ A). Computing this ratio rigorously requires tools beyond the path integral — which is where Radon-Nikodym derivatives and Girsanov's theorem come in.

### Why the Path Integral Picture Is Not Enough

The path integral formula $$\mathcal{P}[\mathbf{x}(\cdot)] \propto \exp(-\text{action})$$ is a powerful heuristic, but it is not mathematically rigorous as written. The problem is fundamental: **there is no uniform measure on the space of continuous paths**.

In finite dimensions, we can write a density $$p(x) \propto e^{-U(x)}$$ because there is a natural reference measure — the Lebesgue measure $$dx$$ on $$\mathbb{R}^d$$. The "$$\propto$$" means "up to a normalizing constant with respect to Lebesgue." On path space $$C([0, T]; \mathbb{R}^d)$$, no such flat reference exists. The "integral over all paths" $$\int \mathcal{D}[\mathbf{x}(\cdot)]$$ that appears in path integral formulas is not a well-defined mathematical object — it is a formal notation that must be given meaning through a limiting procedure.

Physicists handle this by discretizing time ($$N$$ steps, each a finite-dimensional Gaussian), computing, and taking $$N \to \infty$$. This produces correct results, but it requires justifying the interchange of limits and integrals — which is subtle and often swept under the rug.

The rigorous approach is to never write down an individual path measure's "density" at all. Instead, work with **ratios** between path measures. This is the **Radon-Nikodym derivative**.

### Radon-Nikodym Derivatives: The Right Way to Compare Path Measures

The key insight is that while individual path measures have no density with respect to a flat reference, two path measures that share the same noise structure *do* have well-defined densities with respect to *each other*. This is the same idea as importance sampling: we don't need $$p(x)$$ and $$q(x)$$ individually — we need their ratio $$p(x)/q(x)$$.

Given two probability measures $$\mathbb{P}$$ and $$\mathbb{Q}$$ on the same space, the **Radon-Nikodym derivative** $$d\mathbb{P}/d\mathbb{Q}$$ is the density of $$\mathbb{P}$$ with respect to $$\mathbb{Q}$$ — the function that reweights $$\mathbb{Q}$$-samples to produce $$\mathbb{P}$$-expectations:

$$\mathbb{E}_{\mathbb{P}}[f(X)] = \mathbb{E}_{\mathbb{Q}}\left[\frac{d\mathbb{P}}{d\mathbb{Q}}(X) \cdot f(X)\right]$$

For distributions on $$\mathbb{R}^d$$, this is just the likelihood ratio $$p(x)/q(x)$$. For path measures — distributions on $$C([0, T]; \mathbb{R}^d)$$ — the Radon-Nikodym derivative is a functional of the entire trajectory. It is well-defined whenever the two processes share the same diffusion coefficient (same noise), even though neither process has a "density" in isolation.[^ommeasure]

*AIS parallel: In discrete time, the Radon-Nikodym derivative between the forward chain (prior $$\to$$ target) and the reverse chain (target $$\to$$ prior) is exactly the importance weight $$w = \prod_k p_{k+1}(\mathbf{x}_k) / p_k(\mathbf{x}_k)$$. This is a product of density ratios at each step — the discrete analogue of the stochastic integral in Girsanov's theorem.*

[^ommeasure]: The Onsager-Machlup action can be made rigorous in a limited sense: it characterizes the *most probable path* (the trajectory that maximizes the path measure density with respect to the Wiener measure). But using it to compute expectations, partition functions, or free energies requires the Radon-Nikodym / Girsanov framework.

### Girsanov's Theorem: Change of Measure for SDEs

**Girsanov's theorem** solves this problem for diffusion processes. Consider two SDEs with the same noise but different drifts:

$$dX_t = a(X_t, t) \, dt + \sigma \, dW_t \qquad \text{vs.} \qquad dX_t = \tilde{a}(X_t, t) \, dt + \sigma \, dW_t$$

Girsanov's theorem states that the Radon-Nikodym derivative between their path measures is:

> **Girsanov's theorem (one direction).**
>
> $$\ln \frac{d\mathbb{P}^a}{d\mathbb{P}^{\tilde{a}}}(X) = \frac{1}{\sigma^2} \int_0^T (a_t - \tilde{a}_t)(X_t) \cdot dX_t - \frac{1}{2\sigma^2} \int_0^T \lvert a_t - \tilde{a}_t \rvert^2(X_t) \, dt$$
{: .block-definition }

The first term is a stochastic integral (the "martingale part"); the second is a deterministic correction. The key property: the diffusion coefficient $$\sigma$$ must be the same for both processes — Girsanov changes the drift, not the noise. This is why the noise cancels in the forward/reverse ratio: both processes have the same $$\sigma$$.

The continuous-time proof requires the theory of exponential martingales, Novikov's condition, and absolute continuity of measures on path space — machinery I have not invested time into fully understanding myself. See Øksendal (*Stochastic Differential Equations*, Chapter 8) or Revuz & Yor (*Continuous Martingales and Brownian Motion*, Chapter VIII) for the full treatment. In practice, I find it more useful to verify the result from the discrete side, where every step is elementary — and then trust that the continuous limit is justified by the theorem.

### Discrete Derivation of Girsanov's Formula

Discretize both SDEs with Euler-Maruyama (step size $$\Delta t$$). Both processes share the same Gaussian noise — they differ only in the drift. The transition kernel for the first process is:

$$P^a(\mathbf{x}_{k+1} \mid \mathbf{x}_k) \propto \exp\left(-\frac{1}{2\sigma^2 \Delta t} \lvert \mathbf{x}_{k+1} - \mathbf{x}_k - a_k \Delta t \rvert^2\right)$$

and similarly for $$P^{\tilde{a}}$$ with $$\tilde{a}_k$$ replacing $$a_k$$. The log-ratio at step $$k$$ is:

$$\ln \frac{P^a(\mathbf{x}_{k+1} \mid \mathbf{x}_k)}{P^{\tilde{a}}(\mathbf{x}_{k+1} \mid \mathbf{x}_k)} = -\frac{1}{2\sigma^2 \Delta t}\left[\lvert \Delta\mathbf{x}_k - a_k \Delta t \rvert^2 - \lvert \Delta\mathbf{x}_k - \tilde{a}_k \Delta t \rvert^2\right]$$

where $$\Delta\mathbf{x}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$$. Expanding both squares, the $$\lvert \Delta\mathbf{x}_k \rvert^2$$ terms cancel (same noise). What remains:

$$= \frac{1}{\sigma^2}(a_k - \tilde{a}_k) \cdot \Delta\mathbf{x}_k - \frac{1}{2\sigma^2}\lvert a_k - \tilde{a}_k \rvert^2 \Delta t$$

Summing over all steps and taking the continuous limit:

$$\ln \frac{d\mathbb{P}^a}{d\mathbb{P}^{\tilde{a}}}(X) = \frac{1}{\sigma^2} \int_0^T (a_t - \tilde{a}_t) \cdot dX_t - \frac{1}{2\sigma^2} \int_0^T \lvert a_t - \tilde{a}_t \rvert^2 \, dt$$

This reproduces Girsanov's formula exactly. The discrete sum $$\sum_k (a_k - \tilde{a}_k) \cdot \Delta\mathbf{x}_k$$ becomes the stochastic integral (an Itô integral, since the integrand is evaluated at the left endpoint $$\mathbf{x}_k$$); the sum $$\sum_k \lvert a_k - \tilde{a}_k \rvert^2 \Delta t$$ becomes the deterministic correction.

*AIS parallel: This is the discrete importance weight between two different MCMC proposal chains. If both chains use Langevin proposals but with different drift functions (e.g., targeting different distributions), the log importance weight is a sum of drift-difference terms at each step — exactly the discrete Girsanov formula.*

Note that Girsanov compares two *forward* processes. Part III uses it as an internal tool to derive the *forward-reverse* path measure ratio — applying Girsanov twice against a common reference to compare a forward SDE against a backward SDE.

---

## Part III: Forward-Backward SDEs

**The big picture.** Part II gave us general tools for comparing probability distributions over trajectories. But Jarzynski and Crooks involve a specific structure: a *forward* process (driving the system from A to B) and a *reverse* process (driving from B to A), paired together. This part introduces that paired structure — forward-backward SDEs — and derives the formula for their Radon-Nikodym derivative. In diffusion models, this is the forward noising SDE paired with the reverse denoising SDE. In AIS, this is the forward annealing chain paired with the reverse chain. The framework from Vargas et al. (2024) treats all of these uniformly.

### Forward-Backward SDEs

We adopt the notation from Vargas et al. (2024). Consider a pair of forward and backward Itô SDEs:

> **Forward-backward SDEs.**
>
> $$d X_t = a(X_t, t) \, dt + \sigma \, \fwd{d} W_t, \qquad X_0 \sim \mu \qquad \Rightarrow \qquad X \sim \fwd{\mathbb{P}}^{\mu, a}$$
>
> $$d X_t = b(X_t, t) \, dt + \sigma \, \bwd{d} W_t, \qquad X_T \sim \nu \qquad \Rightarrow \qquad X \sim \bwd{\mathbb{P}}^{\nu, b}$$
>
> where $$\fwd{d} W_t$$ and $$\bwd{d} W_t$$ denote forward and backward Itô integration (see the [Fokker-Planck post](/blog/2026/fokker-planck-equation/) for definitions of Itô calculus and stochastic integrals), and $$\fwd{\mathbb{P}}^{\mu, a}$$, $$\bwd{\mathbb{P}}^{\nu, b}$$ are the associated path measures on $$C([0, T]; \mathbb{R}^d)$$.
{: .block-definition }

The forward SDE generates trajectories from $$\mu$$ at time 0; the backward SDE generates trajectories from $$\nu$$ at time $$T$$. For the physics setting, $$a_t = \sigma^2 \nabla \log \pi_t$$ with $$\pi_t$$ interpolating from $$\pi_0$$ to $$\pi_T$$ — exactly the non-equilibrium protocol from Part I. For diffusion models, $$a_t$$ is the noising drift and $$b_t$$ is the learned denoising drift.

**Discrete-time counterpart.** Discretizing the forward SDE with step size $$\Delta t$$ gives the Euler-Maruyama chain:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + a(\mathbf{x}_k, t_k) \Delta t + \sigma \boldsymbol{\xi}_k, \qquad \boldsymbol{\xi}_k \sim \mathcal{N}(0, \Delta t \, \mathbf{I})$$

with path measure $$\fwd{\mathbb{P}}^{\mu, a}[\mathbf{x}_0, \ldots, \mathbf{x}_N] = \mu(\mathbf{x}_0) \cdot \prod_k P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$. The backward SDE discretizes analogously, starting from $$\mathbf{x}_N \sim \nu$$ and stepping in reverse:

$$\mathbf{x}_{k} = \mathbf{x}_{k+1} + b(\mathbf{x}_{k+1}, t_k) \Delta t + \sigma \boldsymbol{\xi}_k$$

with path measure $$\bwd{\mathbb{P}}^{\nu, b}[\mathbf{x}_0, \ldots, \mathbf{x}_N] = \nu(\mathbf{x}_N) \cdot \prod_k P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})$$. The reverse transition $$P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})$$ is a Gaussian centered at $$\mathbf{x}_{k+1} + b(\mathbf{x}_{k+1}, t_k) \Delta t$$ — the same noise variance, but the drift is evaluated at $$\mathbf{x}_{k+1}$$ and pushes backward. This is the reverse kernel used in the discrete derivation below.

*AIS parallel: The forward chain is exactly AIS: draw $$\mathbf{x}_0 \sim p_0$$, then apply MCMC transitions $$T_k(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$ targeting $$p_1, p_2, \ldots, p_K$$ in sequence. The backward chain is the reverse: draw $$\mathbf{x}_K \sim p_K$$, apply reverse transitions $$\tilde{T}_k(\mathbf{x}_k \mid \mathbf{x}_{k+1})$$ targeting $$p_{K-1}, \ldots, p_0$$. The Euler-Maruyama kernel above is one specific choice of $$T_k$$ (unadjusted Langevin); Metropolis-Hastings or HMC give other choices. The importance weight $$w = \mathcal{P}_F / \mathcal{P}_R$$ is the Radon-Nikodym derivative between forward and backward chains.*

### The Forward-Backward Radon-Nikodym Derivative

The central question: what is the log-ratio between the forward and backward path measures? The answer is given by (Vargas et al., 2024, Proposition 2.2):

> **Forward-backward Radon-Nikodym derivative.** Given a reference path measure $$\fwd{\mathbb{P}}^{\Gamma_0, \gamma^+} = \bwd{\mathbb{P}}^{\Gamma_T, \gamma^-}$$:
>
> $$\ln \frac{d\fwd{\mathbb{P}}^{\mu, a}}{d\bwd{\mathbb{P}}^{\nu, b}}(X) = \ln \frac{d\mu}{d\Gamma_0}(X_0) - \ln \frac{d\nu}{d\Gamma_T}(X_T)$$
>
> $$\quad + \frac{1}{\sigma^2} \int_0^T (a_t - \gamma_t^+)(X_t) \cdot \left(\fwd{d}X_t - \frac{1}{2}(a_t + \gamma_t^+)(X_t) \, dt\right)$$
>
> $$\quad - \frac{1}{\sigma^2} \int_0^T (b_t - \gamma_t^-)(X_t) \cdot \left(\bwd{d}X_t - \frac{1}{2}(b_t + \gamma_t^-)(X_t) \, dt\right)$$
{: .block-definition }

This generalizes Girsanov's theorem from Part II. The proof applies Girsanov twice — once for the forward process, once for the backward — using the reference to bridge between them. The key insight is that the reference $$(\Gamma_0, \gamma^\pm)$$ can be chosen freely; different choices redistribute weight between boundary terms and path integrals, enabling different computational strategies.[^otchoice]

[^otchoice]: The connection to optimal transport is direct: the Benamou-Brenier formula characterizes optimal transport as a variational problem over path measures, and the Schrödinger bridge problem — finding the path measure closest to a reference that matches given marginals — is a regularized version of OT. See Vargas et al. (2024, Section 3.1) for details.

### Nelson's Relation: When the RND Is Trivial

A natural question: when does the forward-backward RND equal 1 (i.e., the two path measures are identical)? Nelson (1967) gave the answer:

> **Nelson's relation.** $$\fwd{\mathbb{P}}^{\mu, a} = \bwd{\mathbb{P}}^{\nu, b}$$ if and only if $$\nu = \fwd{\mathbb{P}}^{\mu, a}_T$$ and
>
> $$b_t = a_t - \sigma^2 \nabla \log \rho_t^{\mu, a}, \qquad \forall \, t \in (0, T]$$
>
> where $$\rho_t^{\mu, a}$$ is the time-marginal density of the forward process.
{: .block-definition }

This is the continuous-time analogue of detailed balance. When $$a_t = \sigma^2 \nabla \log \pi_t$$, the natural reverse drift is $$b_t = -\sigma^2 \nabla \log \pi_t$$. If $$\pi_t$$ is the true marginal at time $$t$$, the forward and reverse path measures coincide — the process is reversible. In the physics language, this is the quasistatic limit: the system stays in equilibrium at every instant, so the work equals $$\Delta F$$ exactly and the RND is $$e^0 = 1$$.

*AIS parallel: Nelson's relation holds when each MCMC transition in the AIS chain runs long enough to reach equilibrium at $$p_k$$ before moving to $$p_{k+1}$$. In practice, we use a single (or few) MCMC step(s) per level — the chain never equilibrates, the forward and reverse path measures diverge, and the importance weights compensate with high variance.*

### Discrete Derivation of the Forward-Reverse Path Measure Ratio

We now specialize to the physics setting (overdamped Langevin with potential $$U(\mathbf{x}, \lambda)$$, inverse temperature $$\beta$$) and derive the forward-reverse path measure ratio in discrete time. This provides the discrete counterpart to the continuous-time work identity derived in Part IV.

Using the discrete forward and backward path measures defined above:

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R} = \ln \frac{p_A(\mathbf{x}_0)}{p_B(\mathbf{x}_N)} + \sum_{k=0}^{N-1} \ln \frac{P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)}{P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})}$$

Each kernel is Gaussian with the same variance $$(2/\beta)\Delta t$$, so the log-ratio at step $$k$$ is:

$$\ln \frac{P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)}{P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})} = -\frac{\beta}{4\Delta t}\left[\lvert \Delta\mathbf{x}_k + \nabla U_k \Delta t \rvert^2 - \lvert \Delta\mathbf{x}_k - \nabla U_k \Delta t \rvert^2\right]$$

Expanding the squares, the $$\lvert \Delta\mathbf{x}_k \rvert^2$$ and $$\lvert \nabla U_k \rvert^2 \Delta t^2$$ terms cancel — same noise, same cancellation as in Part II's discrete Girsanov derivation. What remains is the cross term $$-\beta \nabla U_k \cdot \Delta\mathbf{x}_k$$, which sums to $$-\beta \int_0^T \nabla U \cdot d\mathbf{x}$$ in the continuous limit.

Decompose $$\nabla U_k \cdot \Delta\mathbf{x}_k$$ using the discrete total difference $$\Delta U_k = \nabla U_k \cdot \Delta\mathbf{x}_k + (\partial U / \partial \lambda)_k \Delta\lambda_k$$, and add the boundary term $$\ln p_A(\mathbf{x}_0) - \ln p_B(\mathbf{x}_N) = \beta(U_B(\mathbf{x}_N) - U_A(\mathbf{x}_0)) - \beta \Delta F$$. All potential energy terms cancel, leaving:

> **Path measure ratio (discrete).**
>
> $$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R} = \beta \underbrace{\sum_{k=0}^{N-1} \frac{\partial U}{\partial \lambda}(\mathbf{x}_k, \lambda_k) \Delta\lambda_k}_{= \, W_N} - \beta \Delta F = \beta(W_N - \Delta F)$$
>
> where $$W_N$$ is the discrete work — a Riemann sum that converges to $$W = \int_0^T (\partial U / \partial \lambda) \dot{\lambda} \, dt$$ as $$N \to \infty$$.
{: .block-definition }

*AIS parallel: This is the AIS importance weight identity. The MCMC transition kernels cancel in the ratio (same noise cancellation), leaving only the density ratios: $$\log w = \sum_k [\log p_{k+1}(\mathbf{x}_k) - \log p_k(\mathbf{x}_k)]$$. The AIS importance weight $$w$$ is the exponentiated path measure ratio $$\mathcal{P}_F / \mathcal{P}_R$$.*

---

## Part IV: The Non-Equilibrium Equalities

**The big picture.** Parts II and III built the mathematical machinery. This part delivers the payoff: we plug the physics (potential $$U$$, temperature $$\beta$$, protocol $$\lambda(t)$$) into the general forward-backward RND from Part III and derive three results that are central to both statistical mechanics and ML:

1. **The work identity** — the forward-reverse path measure ratio equals $$e^{\beta(W - \Delta F)}$$, connecting trajectory-level statistics to thermodynamic quantities.
2. **Jarzynski's equality** — taking expectations gives $$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$, the exact relation between non-equilibrium work and equilibrium free energy (= the AIS estimator).
3. **Crooks' fluctuation theorem** — the full work distribution of the forward process is related to the reverse by $$P_F(W)/P_R(-W) = e^{\beta(W - \Delta F)}$$.

We return to the physics notation from Parts I–II. The two notation systems used in this post are the same objects:

| Physics (Parts I–II, IV) | General (Part III) | AIS (parallels) |
|---|---|---|
| $$\mathcal{P}_F$$ | $$\fwd{\mathbb{P}}^{\mu, a}$$ | Forward chain distribution |
| $$\mathcal{P}_R$$ | $$\bwd{\mathbb{P}}^{\nu, b}$$ | Reverse chain distribution |
| $$U(\mathbf{x}, \lambda)$$ | $$-\sigma^2 \log \hat{\pi}_t(\mathbf{x})$$ | $$-\log \hat{p}_k(\mathbf{x})$$ |
| $$\beta = 1/k_BT$$ | $$1/\sigma^2$$ | 1 (absorbed into densities) |
| Work $$W$$ | $$\mathcal{W}_T / \sigma^2$$ | $$-\log w$$ |
| $$\Delta F$$ | $$-\sigma^2 \ln(Z_T/Z_0)$$ | $$-\log(Z_K/Z_0)$$ |

### The Work Identity

We specialize the forward-backward RND from Part III to the non-equilibrium physics setting. Recall from Part I the time-dependent potential $$U(\mathbf{x}, \lambda(t))$$ with Boltzmann distribution $$\pi_t(\mathbf{x}) = e^{-\beta U(\mathbf{x}, \lambda(t))} / Z_t$$. The forward SDE is overdamped Langevin under this potential, and the backward SDE reverses the drift:

$$\text{Forward:} \quad dX_t = -\nabla U(X_t, \lambda(t)) \, dt + \sqrt{2/\beta} \, \fwd{d}W_t, \qquad X_0 \sim \pi_0$$

$$\text{Backward:} \quad dX_t = +\nabla U(X_t, \lambda(t)) \, dt + \sqrt{2/\beta} \, \bwd{d}W_t, \qquad X_T \sim \pi_T$$

with reference $$\Gamma_0 = \pi_0$$, $$\Gamma_T = \pi_T$$, $$\gamma^+ = \gamma^- = 0$$. The forward-backward RND simplifies to a Stratonovich integral. The key step applies the Stratonovich chain rule to $$t \mapsto -\beta U(X_t, \lambda(t))$$:

$$d[-\beta U(X_t, \lambda(t))] = -\beta \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt - \beta \nabla U \circ dX_t$$

The $$\nabla U \circ dX_t$$ term telescopes with the path integral from the RND, and the $$(\partial U / \partial \lambda) \dot{\lambda}$$ term is the work integrand from Part I. After cancellation:

> **Path measure ratio (fundamental identity, rigorous form).**
>
> $$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = -\beta \Delta F + \beta W[X]$$
>
> where the **work** is $$W[X] = \int_0^T \frac{\partial U}{\partial \lambda}(X_t, \lambda(t)) \dot{\lambda}(t) \, dt$$ and the **free energy difference** is $$\Delta F = F_T - F_0 = -\frac{1}{\beta} \ln \frac{Z_T}{Z_0}$$.
{: .block-definition }

Part II derived the Girsanov formula for two *forward* SDEs with different drifts. Part III gave the general forward-backward RND. This work identity is the result of *specializing* the forward-backward RND to the physics setting — choosing the specific drifts $$\pm \nabla U$$ from the non-equilibrium protocol and using the Stratonovich chain rule to collapse the path integrals into the work functional $$W$$ from Part I.

<details>
<summary><strong>Full rigorous derivation (click to expand)</strong></summary>

<p><strong>Step 1: The Radon-Nikodym derivative via Stratonovich integral.</strong> Starting from the forward-backward RND formula from Part III with reference \(\Gamma_0 = \pi_0\), \(\Gamma_T = \pi_T\), \(\gamma^+ = \gamma^- = 0\):</p>

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = \ln \frac{\pi_0(X_0)}{\pi_T(X_T)} - \beta \int_0^T \nabla U(X_t, \lambda(t)) \circ dX_t$$

<p>The Stratonovich form arises from combining the forward and backward Itô integrals. This step requires Girsanov's theorem to rigorously define the change of measure from the reference process to each SDE.</p>

<p><strong>Step 2: Apply the Stratonovich chain rule to \(U(X_t, \lambda(t))\).</strong> The Stratonovich chain rule (which takes the same form as the ordinary chain rule, unlike the Itô formula) gives:</p>

$$dU(X_t, \lambda(t)) = \nabla U \circ dX_t + \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt$$

<p>Rearranging and integrating from \(0\) to \(T\):</p>

$$-\beta \int_0^T \nabla U \circ dX_t = -\beta \bigl[U(X_T, \lambda_T) - U(X_0, \lambda_0)\bigr] + \beta \int_0^T \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt$$

<p>The second term is \(\beta W\) — the work integral from Part I.</p>

<p><strong>Step 3: Combine and cancel.</strong> Substituting into Step 1 and using \(\pi_t = e^{-\beta U(\cdot, \lambda(t))}/Z_t\):</p>

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = \underbrace{\ln \frac{\pi_0(X_0)}{\pi_T(X_T)}}_{-\beta U_0(X_0) + \beta U_T(X_T) + \ln(Z_T/Z_0)} \underbrace{- \beta[U_T(X_T) - U_0(X_0)]}_{\text{from chain rule}} + \beta W$$

<p>The potential energy terms cancel exactly:</p>

$$= \ln \frac{Z_T}{Z_0} + \beta W = -\beta \Delta F + \beta W = \beta(W - \Delta F)$$

<p>∎</p>

<p><strong>Why Stratonovich?</strong> The Stratonovich integral preserves the ordinary chain rule, which is why the telescoping in Step 2 works cleanly. Using the Itô integral instead would introduce an additional \(\frac{1}{2}\sigma^2 \Delta U\) correction term (the Itô-Stratonovich conversion), which must then be tracked and cancelled — possible but messier.</p>

</details>

### Jarzynski's Equality

The derivation from the path measure ratio takes three lines. Start from $$\mathcal{P}_F / \mathcal{P}_R = e^{\beta(W - \Delta F)}$$. Rearrange:

$$\frac{\mathcal{P}_R}{\mathcal{P}_F} = e^{-\beta(W - \Delta F)}$$

Integrate both sides over all trajectories with respect to $$\mathcal{P}_F$$:

$$\int \frac{\mathcal{P}_R[\mathbf{x}(\cdot)]}{\mathcal{P}_F[\mathbf{x}(\cdot)]} \, \mathcal{P}_F[\mathbf{x}(\cdot)] \, \mathcal{D}[\mathbf{x}(\cdot)] = \int \mathcal{P}_R[\mathbf{x}(\cdot)] \, \mathcal{D}[\mathbf{x}(\cdot)] = 1$$

$$\int e^{-\beta(W - \Delta F)} \, \mathcal{P}_F[\mathbf{x}(\cdot)] \, \mathcal{D}[\mathbf{x}(\cdot)] = 1$$

$$\left\langle e^{-\beta W} \right\rangle_F = e^{-\beta \Delta F}$$

> **Jarzynski's equality (1997).**
>
> $$\left\langle e^{-\beta W} \right\rangle_F = e^{-\beta \Delta F}$$
>
> where $$\langle \cdot \rangle_F$$ denotes an average over all forward trajectories — i.e., an expectation under the forward path measure $$\mathcal{P}_F$$. The exponential average of the work over forward non-equilibrium trajectories gives the exact equilibrium free energy difference, regardless of how far from equilibrium the process is driven.
{: .block-definition }

Three properties make this remarkable:

1. $$\Delta F$$ is an equilibrium quantity. $$W$$ is measured from non-equilibrium trajectories. The equality holds for *any* protocol speed.
2. The second law $$\langle W \rangle \geq \Delta F$$ follows from Jensen's inequality applied to the convex function $$e^{-x}$$: $$\langle e^{-\beta W} \rangle \geq e^{-\beta \langle W \rangle}$$, so $$e^{-\beta \Delta F} \geq e^{-\beta \langle W \rangle}$$, giving $$\langle W \rangle \geq \Delta F$$.
3. The Jensen bound is tight only when $$W$$ is constant across trajectories — the quasistatic limit where the system remains in equilibrium throughout.

*AIS parallel: Jarzynski's equality in discrete time is $$\langle w \rangle = Z_B / Z_A$$, where $$w = \prod_k p_{k+1}(\mathbf{x}_k) / p_k(\mathbf{x}_k)$$ is the AIS importance weight. This is exactly how AIS estimates normalizing constant ratios — average the importance weights over many forward chains. The three properties above translate directly: (1) the estimate is valid for any number of annealing steps $$K$$; (2) Jensen gives $$\langle \log w \rangle \leq \log(Z_B/Z_A)$$, the ELBO; (3) the bound is tight only when all weights are equal — perfect annealing.*

**The variance problem.** In practice, the exponential average is dominated by rare trajectories with anomalously low work. For fast protocols (far from equilibrium), most trajectories have $$W \gg \Delta F$$, and the rare ones with $$W \approx \Delta F$$ carry exponentially large weight. The estimator is unbiased but has exponentially large variance. This is the *same* overlap problem as Zwanzig's identity — transferred from configuration space to trajectory space.

*AIS parallel: This is the well-known problem of AIS weight degeneracy. With too few annealing steps (fast protocol), one chain dominates the importance weight sum while the rest contribute negligibly. The effective sample size collapses. Adding more intermediate distributions (slower protocol) reduces variance but increases cost — the same speed-accuracy tradeoff as in non-equilibrium physics.*


### Crooks' Fluctuation Theorem

A direct consequence of the path measure ratio identity. Consider the probability of observing work $$W$$ in the forward process, $$P_F(W)$$, versus work $$-W$$ in the reverse process, $$P_R(-W)$$.

> **Crooks' fluctuation theorem (1999).**
>
> $$\frac{P_F(W)}{P_R(-W)} = e^{\beta(W - \Delta F)}$$
>
> The ratio of the probability of observing work $$W$$ in the forward direction to the probability of observing work $$-W$$ in the reverse direction is exponentially related to how far $$W$$ deviates from $$\Delta F$$.
{: .block-definition }

Crooks is a stronger statement than Jarzynski — it relates the *entire work distribution* of the forward process to the reverse, not just an exponential average. Jarzynski is recovered by integrating both sides of Crooks over $$W$$.

<details>
<summary><strong>Deriving Jarzynski from Crooks (click to expand)</strong></summary>

<p>Start from Crooks: \(P_F(W) = e^{\beta(W - \Delta F)} P_R(-W)\). Multiply both sides by \(e^{-\beta W}\) and integrate over all \(W\):</p>

$$\int e^{-\beta W} P_F(W) \, dW = \int e^{-\beta W} \cdot e^{\beta(W - \Delta F)} P_R(-W) \, dW = e^{-\beta \Delta F} \int P_R(-W) \, dW$$

<p>The left side is \(\langle e^{-\beta W} \rangle_F\). The right side simplifies because \(\int P_R(-W) \, dW = 1\) (normalization of the reverse work distribution, via substitution \(W' = -W\)):</p>

$$\langle e^{-\beta W} \rangle_F = e^{-\beta \Delta F}$$

<p>which is Jarzynski's equality. ∎</p>

</details>

**Physical intuition.** Trajectories where the forward work is less than $$\Delta F$$ (the system "got lucky" — thermal fluctuations helped the protocol) are exponentially rare, but they are *exactly as probable* as the corresponding reverse trajectories where the reverse work exceeds $$\Delta F$$. The crossing point $$P_F(W) = P_R(-W)$$ occurs at $$W = \Delta F$$. This gives a graphical method for estimating $$\Delta F$$ — the Bennett acceptance ratio (BAR) method.

*AIS parallel: Crooks relates the forward and reverse work distributions via a single identity. The **Bennett acceptance ratio** (BAR) exploits this by combining forward AIS runs (prior $$\to$$ target) with reverse AIS runs (target $$\to$$ prior) into a single estimator for $$\Delta F$$. Shirts et al. (2003) showed that BAR is the minimum-variance estimator given samples from both directions. In practice, the reverse direction requires samples from the target, which may not always be available; but when it is (e.g., evaluating a trained generative model that can produce samples), BAR can improve over unidirectional AIS.*

{% include figure.liquid loading="eager" path="assets/img/blog/pm_crooks_intersection.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Crooks' fluctuation theorem: the forward work distribution \(P_F(W)\) and reverse work distribution \(P_R(-W)\) intersect at \(W = \Delta F\). This crossing point is the basis of the Bennett acceptance ratio (BAR) method." %}

### The Second Law and Dissipation as KL Divergence

From Jarzynski and Jensen:

$$\langle W \rangle_F \geq \Delta F$$

The gap is the average dissipated work $$\langle W_{\text{diss}} \rangle = \langle W \rangle_F - \Delta F$$. This gap has an information-theoretic interpretation:

> **Dissipated work as KL divergence.**
>
> $$\langle W_{\text{diss}} \rangle = \frac{1}{\beta} D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$
>
> The average work wasted beyond the free energy difference equals (up to temperature) the KL divergence between the forward and reverse path measures. Reversible processes (zero dissipation) have $$\mathcal{P}_F = \mathcal{P}_R$$.
{: .block-definition }

This follows directly from the path measure ratio identity. The KL divergence between $$\mathcal{P}_F$$ and $$\mathcal{P}_R$$ is:

$$D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R) = \left\langle \ln \frac{\mathcal{P}_F}{\mathcal{P}_R} \right\rangle_F = \left\langle \beta(W - \Delta F) \right\rangle_F = \beta \langle W_{\text{diss}} \rangle$$

Irreversibility = information loss = KL divergence between forward and reverse path measures.

*AIS parallel: The dissipated work in AIS is $$\langle W_{\text{diss}} \rangle = \langle -\log w \rangle - \log(Z_A/Z_B) = \log(Z_B/Z_A) - \langle \log w \rangle$$. This is the gap between the true $$\log(Z_B/Z_A)$$ and the ELBO $$\langle \log w \rangle$$. The identity says this gap equals the KL divergence between the forward AIS chain distribution and the reverse chain distribution. A tighter ELBO means less dissipation means the forward and reverse chains are more similar.*

---

## Part V: Connections to Generative Models

### Connection 1: Annealed Importance Sampling = Jarzynski

Annealed importance sampling (Neal, 2001) estimates $$Z_B / Z_A$$ by constructing a sequence of intermediate distributions $$p_0, p_1, \ldots, p_K$$ bridging $$p_A$$ to $$p_B$$, and running MCMC transitions at each level. The mapping to non-equilibrium statistical mechanics is exact:

| Non-eq. stat mech | AIS |
|---|---|
| Protocol $$\lambda(t)$$ | Annealing schedule $$\beta_0, \ldots, \beta_K$$ |
| Potential $$U(\mathbf{x}, \lambda)$$ | $$-\log p(\mathbf{x}; \beta_k)$$ (interpolated log-density) |
| Forward trajectory | AIS sample chain $$\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_K$$ |
| Work $$W$$ | Negative log importance weight: $$W = \sum_k [\log p_k(\mathbf{x}_k) - \log p_{k+1}(\mathbf{x}_k)]$$ |
| Jarzynski: $$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$ | AIS: $$\langle w \rangle = Z_B / Z_A$$ where $$w = \prod_k p_{k+1}(\mathbf{x}_k) / p_k(\mathbf{x}_k)$$ |

When you evaluate a normalizing flow or energy-based model using AIS, you are running Jarzynski's equality. The importance weights are the exponentiated negative work. The variance problem (rare high-weight samples dominating) is the same variance problem as in the non-equilibrium work estimator.

This connection is not a recent observation. Neal himself noted it in his AIS paper (Neal, 2001), citing Jarzynski (1997) and pointing out that AIS can be viewed as a discrete version of the non-equilibrium work identity. The physics and ML communities developed the same idea in parallel, in different notation, for different applications.

**Practical implication.** All the tricks physicists developed for improving Jarzynski estimates — bidirectional methods (BAR = Bennett acceptance ratio), optimal protocol design, targeted free energy perturbation — are directly transferable to improving AIS for generative model evaluation.

### Connection 2: Diffusion Models = Forward/Reverse Non-Equilibrium Processes

A diffusion model has two stochastic processes running in opposite directions — exactly the forward-backward SDE pair from Part III:

- **Forward (noising):** $$d\mathbf{x} = f(\mathbf{x}, t) \, dt + g(t) \, d\mathbf{w}$$. Gradually destroys data structure. State A = data distribution, state B = Gaussian noise. This is the "protocol" that drives the system from A to B.
- **Reverse (denoising):** $$d\mathbf{x} = [f - g^2 \nabla \log p_t] \, dt + g \, d\bar{\mathbf{w}}$$. Learned process that reconstructs data from noise. The score $$\nabla \log p_t$$ plays the role of the drift in the backward SDE.

The mapping to non-equilibrium statistical mechanics:

| Non-eq. stat mech | Diffusion model |
|---|---|
| State A | Data distribution $$p_{\text{data}}$$ |
| State B | Gaussian noise $$\mathcal{N}(0, \mathbf{I})$$ |
| Forward protocol (A $$\to$$ B) | Forward noising process |
| Reverse protocol (B $$\to$$ A) | Learned reverse (denoising) process |
| Forward path measure $$\mathcal{P}_F$$ | Distribution over all noising trajectories |
| Backward path measure $$\mathcal{P}_R$$ | Distribution over all denoising trajectories |
| Protocol speed | Number of diffusion steps (more steps = slower, less dissipation) |
| Score $$\nabla \log p_t$$ | The time-dependent score function learned by the neural network |
| Nelson's relation | Perfect score estimation $$\Rightarrow$$ reversible process ($$\mathcal{P}_F = \mathcal{P}_R$$) |
| Dissipated work $$\langle W_{\text{diss}} \rangle$$ | Gap between ELBO and true log-likelihood |

**The ELBO is a path-measure KL divergence.** The variational bound used in diffusion model training (the DDPM loss) can be derived as:

$$D_{\text{KL}}(q(\mathbf{x}_{0:T}) \| p_\theta(\mathbf{x}_{0:T}))$$

where $$q$$ is the forward path measure (the joint distribution over all noising steps $$\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T$$) and $$p_\theta$$ is the reverse generative path measure (the learned denoising chain). This is the Radon-Nikodym derivative from Part III, applied to the discrete chain. The KL decomposes into the sum of per-step KL terms:

$$D_{\text{KL}}(q(\mathbf{x}_{0:T}) \| p_\theta(\mathbf{x}_{0:T})) = \sum_{t=1}^{T} \mathbb{E}_q\left[D_{\text{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))\right] + \text{const.}$$

This is the DDPM weighted denoising loss. In the physics language, this is the same decomposition as breaking the work integral into per-step contributions in the discrete Jarzynski framework — each step contributes a "local work" term.

**Dissipation = imperfect score.** From Part IV, $$\langle W_{\text{diss}} \rangle = (1/\beta) D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$. For diffusion models, this identity says:

- The gap between the ELBO and the true log-likelihood = how far the learned score is from the true score = how far the process is from being reversible.
- A perfectly trained diffusion model has zero dissipation: the learned score matches the true score at every time step, the forward and reverse path measures coincide (Nelson's relation from Part III), and the ELBO equals the true log-likelihood.
- More diffusion steps (slower protocol) reduce dissipation even with an imperfect score — the same speed-accuracy tradeoff as in non-equilibrium physics and AIS.

*AIS parallel: The DDPM loss decomposes the path-measure KL into per-step terms, just as the AIS importance weight decomposes into per-level density ratios. In both cases, adding more intermediate steps reduces the per-step contribution and tightens the overall bound.*

The connection is precise: Song et al. (2021) formulated diffusion models as continuous-time SDEs, and Huang et al. (2021) showed that the variational perspective on diffusion models is equivalent to the path-measure KL framework.

### Connection 3: GFlowNet Trajectory Balance = Path Measure Balance

GFlowNets (Bengio et al., 2021) solve a different problem than diffusion models: instead of learning a continuous-time SDE, they learn a *discrete* construction policy that builds objects (molecules, graphs, sequences) step by step, sampling each object with probability proportional to a reward $$R(x)$$. But the mathematical structure is the same.

A GFlowNet constructs an object $$x$$ by a sequence of actions $$(a_0, a_1, \ldots, a_T)$$, defining a trajectory $$\tau = (s_0, s_1, \ldots, s_T = x)$$ through a directed acyclic graph (DAG) of partial constructions. The **forward policy** $$P_F(a_t \mid s_t)$$ defines a distribution over construction trajectories. The **backward policy** $$P_B(a_t \mid s_{t+1})$$ defines a distribution over *deconstruction* trajectories — ways to take apart the object step by step.

The **trajectory balance** condition (Malkin et al., 2022) states that these must satisfy:

$$Z \cdot P_F(\tau) = R(x) \cdot P_B(\tau)$$

where $$Z = \sum_{x} R(x)$$ is the partition function (total reward) and $$P_F(\tau) = \prod_t P_F(a_t \mid s_t)$$, $$P_B(\tau) = \prod_t P_B(a_t \mid s_{t+1})$$ are the full trajectory probabilities.

**This is Crooks' theorem in discrete form.** The mapping is:

| Non-eq. stat mech | GFlowNet |
|---|---|
| Forward path measure $$\mathcal{P}_F$$ | Forward policy trajectory distribution $$P_F(\tau)$$ |
| Reverse path measure $$\mathcal{P}_R$$ | Backward policy trajectory distribution $$P_B(\tau)$$ |
| $$e^{-\beta \Delta F} = Z_B / Z_A$$ | $$Z$$ (the partition function / total reward) |
| Work $$W[\mathbf{x}(\cdot)]$$ | $$\log R(x)$$ (the log-reward of the terminal object) |
| Crooks: $$\mathcal{P}_F = e^{\beta(W - \Delta F)} \mathcal{P}_R$$ | Trajectory balance: $$Z \cdot P_F(\tau) = R(x) \cdot P_B(\tau)$$ |

To see the identity explicitly, rewrite Crooks for a single trajectory:

$$\frac{\mathcal{P}_F}{\mathcal{P}_R} = e^{\beta(W - \Delta F)}$$

Rearrange: $$e^{\beta \Delta F} \cdot \mathcal{P}_F = e^{\beta W} \cdot \mathcal{P}_R$$. Identify $$Z = e^{-\beta \Delta F}$$ and $$R(x) = e^{\beta W}$$:

$$Z \cdot P_F = R \cdot P_B$$

The GFlowNet community derived trajectory balance from first principles in the discrete DAG setting. The non-equilibrium stat mech community had it for continuous systems since Crooks (1999). They are the same theorem.

**What this buys you.** The correspondence runs deeper than the trajectory balance condition itself:

- **Detailed balance** (the GFlowNet local condition $$F(s) P_F(a \mid s) = F(s') P_B(a \mid s')$$) corresponds to the **microscopic reversibility** condition in physics — detailed balance on the transition level rather than the trajectory level.
- **Sub-trajectory balance** (Madan et al., 2023) corresponds to applying the Crooks relation on *sub-intervals* of the protocol rather than the full trajectory — a coarse-grained path measure ratio.
- The **variance of the trajectory balance loss** $$\text{Var}[\log(Z \cdot P_F(\tau)) - \log(R(x) \cdot P_B(\tau))]$$ is exactly the log-variance divergence between forward and reverse path measures — the same objective used in controlled diffusion samplers (Vargas et al., 2024).

---

## Part VI: What This Unification Buys You

### A Shared Diagnostic: Dissipation = KL Divergence = Quality Gap

The identity $$\langle W_{\text{diss}} \rangle = (1/\beta) D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$ gives a universal measure of "how well is our generative process doing":

- **Diffusion models:** High dissipation = imperfect score estimation = noisy samples.
- **GFlowNets:** High dissipation = forward policy doesn't match the Boltzmann-weighted backward policy = poor mode coverage.
- **AIS:** High dissipation = loose bound on $$\log Z$$ = unreliable evaluation.

In all cases, the diagnostic is the same: measure how different the forward and reverse path measures are.

### Optimal Protocols = Optimal Schedules

Physicists have studied: given a fixed time budget $$\tau$$, what protocol $$\lambda(t)$$ minimizes $$\langle W_{\text{diss}} \rangle$$? The answer involves the **thermodynamic metric tensor** (Sivak & Crooks, 2012) — a Riemannian metric on the space of equilibrium states that measures the "friction" of driving the system.

The ML translation:

- **Diffusion models:** What noise schedule minimizes the path KL? This is the question of optimal noise schedules, studied empirically (cosine schedules, learned schedules). The thermodynamic metric provides a principled framework for this choice.
- **AIS:** What annealing schedule minimizes estimator variance? The thermodynamic length framework gives an answer.

### The Bidirectional Trick

Crooks tells you that combining forward and reverse work measurements gives a better free energy estimate than either alone — this is the Bennett acceptance ratio (BAR). The ML analogue: using both the generative (noise $$\to$$ data) and inference (data $$\to$$ noise) directions simultaneously. This is already done in some VAE and flow training procedures, and the connection to BAR explains *why* it helps.

### Variance Reduction Across Fields

Physicists have decades of work on reducing variance of Jarzynski estimators. Each technique has an ML analogue:

- **Targeted free energy perturbation** — choose intermediate states that maximize overlap between neighbors. ML analogue: choosing the annealing schedule in AIS to minimize the KL between consecutive distributions, or designing the noise schedule in diffusion models.
- **Steered molecular dynamics with optimal pulling protocols** — optimize the protocol $$\lambda(t)$$ to minimize dissipation (the thermodynamic metric tensor of Sivak & Crooks, 2012). ML analogue: learning the interpolation path between prior and target in flow-based methods.
- **Bidirectional estimators (BAR, MBAR)** — combine forward and reverse work measurements to reduce variance below what either direction achieves alone. ML analogue: training with both the generative and inference directions simultaneously.

Recognizing these as instances of the same mathematical structure means that improvements in one field can be translated to the other.

---

## Conclusion

The summary of the mapping:

| Physics concept | ML incarnation |
|---|---|
| Forward path measure $$\mathcal{P}_F$$ | Generative model's trajectory distribution |
| Reverse path measure $$\mathcal{P}_R$$ | Inference / backward process |
| Work $$W$$ | Log-likelihood ratio along trajectory |
| Free energy difference $$\Delta F$$ | Log normalizing constant $$\ln Z$$ |
| Jarzynski's equality | AIS / importance-weighted ELBO |
| Crooks' fluctuation theorem | GFlowNet trajectory balance |
| Dissipated work | Training loss / KL gap |
| Optimal protocol | Optimal schedule / interpolation |
| Second law ($$\langle W \rangle \geq \Delta F$$) | ELBO $$\leq \log Z$$ |

These are not analogies. They are the same mathematics applied in different contexts. Physicists derived these results for atoms in the 1990s; ML researchers rediscovered them for data in the 2020s. Recognizing the identity means we can:

1. Import decades of physics intuition about what makes a good non-equilibrium protocol.
2. Reason about AIS, diffusion models, and GFlowNets using a single conceptual framework.

In physics, the central problem is moving a system between thermodynamic states efficiently. In ML, the central problem is transporting probability mass between distributions efficiently. These are the same problem, and the unified language of path measures makes this precise.

---

## References

- C. Jarzynski, "Nonequilibrium equality for free energy differences," *Physical Review Letters*, 1997.
- G. E. Crooks, "Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences," *Physical Review E*, 1999.
- R. M. Neal, "Annealed importance sampling," *Statistics and Computing*, 2001.
- R. W. Zwanzig, "High-temperature equation of state by a perturbation method," *Journal of Chemical Physics*, 1954.
- E. Nelson, *Dynamical Theories of Brownian Motion*, Princeton University Press, 1967.
- M. R. Shirts, E. Bair, G. Hooker, and V. S. Pande, "Equilibrium free energies from nonequilibrium measurements using maximum-likelihood methods," *Physical Review Letters*, 2003.
- D. A. Sivak and G. E. Crooks, "Thermodynamic metrics and optimal paths," *Physical Review Letters*, 2012.
- F. Vargas, S. Padhy, D. Blessing, and N. Nüsken, "Transport meets variational inference: Controlled Monte Carlo Diffusions," *ICLR*, 2024. ([arXiv:2307.01050](https://arxiv.org/abs/2307.01050))
- N. Malkin, M. Jain, E. Bengio, C. Sun, and Y. Bengio, "Trajectory balance: Improved credit assignment in GFlowNets," *ICML*, 2022.
- Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, "Score-based generative modeling through stochastic differential equations," *ICLR*, 2021.
- C.-W. Huang, J. H. Lim, and A. Courville, "A variational perspective on diffusion-based generative models and score matching," *NeurIPS*, 2021.

---

## Footnotes
