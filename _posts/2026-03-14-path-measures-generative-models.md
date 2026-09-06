---
layout: post
title: "From Jarzynski's Equality to Diffusion Models"
date: 2026-03-14
last_updated: 2026-09-06
description: "How path measures connect Jarzynski's equality, free-energy estimation, annealed importance sampling, diffusion models, and GFlowNets."
post_type: technical-note
selected: true
authors: ["Sungsoo Ahn"]
order: 1
series: stochastic-generative-models
series_title: "Stochastic Processes and Generative Models"
series_description: "A reading path from stochastic dynamics to statistical mechanics, path measures, and generative modeling."
series_order: 3
categories: [generative-modeling]
tags: [non-equilibrium-statistical-mechanics, path-measures, free-energy, generative-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post connects non-equilibrium statistical mechanics and generative modeling: AIS, diffusion models, and GFlowNet trajectory balance can all be read through the same forward/reverse path-measure ratio. The continuous-time view follows <a href="https://openreview.net/forum?id=PP1rudnxiW">Controlled Monte Carlo Diffusions</a> (CMCD, Vargas et al., 2024), and the connection became concrete for me through work from our group on <a href="https://openreview.net/forum?id=WQV9kB1qSU">transition path sampling with diffusion models</a>.</em>
</p>

## Introduction

A diffusion model transforms noise into data by learning to reverse a noising process. The forward process (data $$\to$$ noise) and the reverse process (noise $$\to$$ data) are stochastic processes running in opposite directions: probability distributions over trajectories, not single points. In a variational formulation, a path-space KL contributes to the training objective, and an exact reverse model, with the correct starting marginal, reproduces the forward path law.

Diffusion models are one instance of a broader pattern: two opposing stochastic processes whose path-measure ratio encodes useful information. Physicists developed the same framework in the 1990s to understand systems driven out of equilibrium. In physics, state A might be an unbound protein-drug system and state B the bound complex; in diffusion models, state A is the data distribution and state B is Gaussian noise. The useful common object is a ratio of path measures; interpreting it as physical work requires the thermodynamic assumptions developed below.

The free-energy problem makes this comparison concrete. We often need a normalizer ratio $$Z_B/Z_A$$ rather than either normalizer separately. Equilibrium reweighting can estimate it from one endpoint, but poor overlap makes the estimate noisy. Annealed importance sampling (AIS) introduces intermediate distributions and accumulates a weight $$w$$ along each chain.

Under the AIS assumptions, $$\langle w\rangle=Z_B/Z_A$$, while Jensen's inequality gives $$\langle\log w\rangle\leq\log(Z_B/Z_A)$$. The gap is a KL divergence between forward and reverse chain measures. Constant weights make the bound tight; equilibrating between finitely sized distribution changes need not do so.

For Boltzmann intermediates and suitable relaxation dynamics, $$\log w=-\beta W$$, where $$W$$ is the work done by changing the potential and $$\beta=1/(k_BT)$$. The normalizer identity then becomes Jarzynski's equality (<span id="cite-jarzynski1997"></span>[Jarzynski, 1997](#ref-jarzynski1997)):

$$\langle e^{-\beta W}\rangle=e^{-\beta\Delta F},
\qquad \Delta F=-\beta^{-1}\log(Z_B/Z_A).$$

The derivation below starts with the free-energy problem, proves the discrete AIS ratio, and then develops the continuous-time tools. The final comparison with diffusion models and GFlowNets separates this shared mathematics from the additional assumptions needed to interpret a ratio as physical work.

## Part 1: The Free Energy Problem

### Configurations, States, and the Boltzmann Distribution

Consider $$N$$ atoms with positions $$\mathbf{x} = (\mathbf{r}_1, \ldots, \mathbf{r}_N) \in \mathbb{R}^{3N}$$. A single assignment of all positions, one snapshot of the system, is a configuration. A potential energy function $$U(\mathbf{x})$$ assigns each configuration an energy based on interactions between atoms: bonds, electrostatics, van der Waals forces, and so on.

In ML terms, think of $$\mathbf{x}$$ as a data point in $$\mathbb{R}^{3N}$$ and $$U(\mathbf{x})$$ as a negative log-probability up to a constant. A thermodynamic state is not a single configuration; it is the *entire probability distribution* over configurations defined by $$U$$. At temperature $$T$$, this distribution is the Boltzmann distribution:

> **Boltzmann distribution.**
>
> $$p(\mathbf{x}) = \frac{e^{-\beta U(\mathbf{x})}}{Z}, \qquad Z = \int e^{-\beta U(\mathbf{x})} \, d\mathbf{x}, \qquad \beta = \frac{1}{k_B T}$$
>
> The Boltzmann factor $$e^{-\beta U(\mathbf{x})}$$ assigns high probability to low-energy configurations. The partition function $$Z$$ normalizes the distribution — it sums the Boltzmann weight over all possible configurations.
{: .block-definition }

A state is therefore an energy-based model: $$U$$ defines the unnormalized log-density, and $$Z$$ is the intractable normalizing constant — for even $$N = 100$$ atoms, it is an integral over $$\mathbb{R}^{300}$$.

**How states are represented.** Thermodynamic states depend on the potential, temperature, boundary conditions, and any restriction to a basin. In alchemical calculations we construct a common configuration space and change its potential; physical folding or binding states can instead be restricted regions of the same potential. Standard-state and restraint corrections are needed when relating an alchemical calculation to a binding affinity. For example:

- **Drug binding.** State A: protein and drug molecule simulated separately in solvent — $$U_A$$ includes protein-solvent and drug-solvent interactions but no protein-drug interactions. State B: protein and drug simulated together — $$U_B$$ adds the protein-drug interaction terms. Same atoms, different $$U$$ because the interaction terms change.
- **Alchemical transformation.** To compare two drug candidates, state A uses the force field parameters of molecule 1 and state B uses those of molecule 2. The potential $$U$$ changes because the atomic charges, Lennard-Jones parameters, or even the number of atoms differ.
- **Crystal polymorphism.** Same molecule, but $$U_A$$ and $$U_B$$ include different periodic boundary conditions and lattice geometries, giving different packing interactions.

In each case, the configuration space $$\mathbf{x} \in \mathbb{R}^{3N}$$ is the same (or can be made the same via dummy atoms), but $$U_A(\mathbf{x}) \neq U_B(\mathbf{x})$$. This gives two Boltzmann distributions $$p_A(\mathbf{x})$$ and $$p_B(\mathbf{x})$$ with partition functions $$Z_A$$ and $$Z_B$$.


### Free Energy

The Helmholtz free energy packages the intractable partition function into a single thermodynamic quantity:

> **Helmholtz free energy.**
>
> $$F = -k_B T \ln Z = \langle U \rangle - TS$$
>
> where $$\langle U \rangle = \mathbb{E}_{p}[U(\mathbf{x})]$$ is the average potential energy under the Boltzmann distribution and $$S$$ is the entropy. Free energy balances energy (low $$U$$ — the system finds favorable interactions) against entropy (high $$S$$ — many configurations are accessible). The equilibrium state minimizes $$F$$.
{: .block-definition }

**Notation.** Throughout this post, angle brackets $$\langle \cdot \rangle$$ denote expectations (averages): $$\langle f \rangle = \mathbb{E}_p[f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x}) \, d\mathbf{x}$$. A subscript indicates which distribution the average is over — $$\langle \cdot \rangle_A$$ means averaging over the Boltzmann distribution of state A.

Since each state has its own $$U$$ and $$Z$$, each state has its own free energy: $$F_A = -k_B T \ln Z_A$$ and $$F_B = -k_B T \ln Z_B$$. We rarely need $$F_A$$ or $$F_B$$ individually; we need their *difference*. The sign of $$\Delta F = F_B - F_A$$ tells us which state is thermodynamically favored: if $$\Delta F < 0$$, state B is more stable (lower free energy); if $$\Delta F > 0$$, state A wins. The magnitude tells us whether the preference is marginal or overwhelming.


This makes $$\Delta F$$ the central quantity in three classes of problems:

- **Protein folding.** State A is the unfolded ensemble (high entropy, many disordered conformations). State B is the folded state (low energy, compact structure with favorable contacts). The protein folds spontaneously if $$\Delta F_{\text{unfolded} \to \text{folded}} < 0$$ — the energy gain from forming contacts outweighs the entropy loss from ordering. Typical folding free energies are small: 5–15 kcal/mol, the difference between large opposing terms. Predicting the sign correctly requires getting both the energy and entropy right.

- **Drug binding.** State A is the drug and protein separated in solution. State B is the drug bound in the protein's active site. The binding free energy $$\Delta F_{\text{bind}}$$ determines affinity: a drug with $$\Delta F_{\text{bind}} = -10$$ kcal/mol binds $$\sim 10^7$$ times more tightly than one with $$\Delta F_{\text{bind}} = -1$$ kcal/mol (since the equilibrium constant goes as $$K \propto e^{-\beta \Delta F}$$). Binding free energy calculation is the gold standard for computational drug design — pharmaceutical companies routinely use free energy perturbation (FEP) to prioritize candidates before synthesis.[^fep]

[^fep]: FEP includes energetic and entropic contributions through an equilibrium average. Its practical accuracy depends on sampling, the force field, and the thermodynamic cycle; it is not the only statistical-mechanical route to a binding free energy.

- **Crystal polymorphism.** The same molecule can pack into different crystal structures (polymorphs). State A and B are two such packing arrangements, each with its own $$U_A, U_B$$. The polymorph with lower $$F$$ is the one that forms at equilibrium. Getting this wrong has real consequences — the wrong polymorph of a pharmaceutical can have different solubility, bioavailability, or stability.[^ritonavir]

[^ritonavir]: Famously, ritonavir had to be reformulated after a more stable polymorph appeared unexpectedly, disrupting supply of the HIV drug for months.

In all three cases, the computational challenge is identical: compute $$\Delta F$$ from the potential energy functions $$U_A$$ and $$U_B$$.

### Free Energy Differences and Why They're Hard

The free energy difference between states A and B is:

> **Free energy difference.**
>
> $$\Delta F = F_B - F_A = -k_B T \ln \frac{Z_B}{Z_A}$$
>
> where $$Z_A = \int e^{-\beta U_A(\mathbf{x})} \, d\mathbf{x}$$ and $$Z_B = \int e^{-\beta U_B(\mathbf{x})} \, d\mathbf{x}$$.
{: .block-definition }

Computing $$Z_A$$ and $$Z_B$$ individually is intractable, but their *ratio* can in principle be estimated by rewriting it as an expectation under one endpoint's distribution. Zwanzig showed how (<span id="cite-zwanzig1954"></span>[Zwanzig, 1954](#ref-zwanzig1954)):

> **Zwanzig's identity (free energy perturbation).**
>
> $$e^{-\beta \Delta F} = \left\langle e^{-\beta (U_B - U_A)} \right\rangle_A$$
>
> where $$\langle \cdot \rangle_A$$ denotes an average over the equilibrium distribution of state A.
{: .block-definition }

This is exact but often useless in practice. The average is dominated by rare configurations where $$U_B(\mathbf{x}) - U_A(\mathbf{x})$$ is small, meaning configurations that lie in the overlap between the two Boltzmann distributions. When A and B are very different, which is usually the interesting case, the relevant overlap can be very small and the estimator can have large, or even unbounded, variance.

{% include figure.liquid loading="eager" path="assets/img/blog/pm_boltzmann_overlap.svg" alt="Separated equilibrium densities have only a small overlap region that contributes to reweighting." class="img-fluid rounded z-depth-1" zoomable=true caption="Two Boltzmann distributions with minimal overlap. The shaded region is where the Zwanzig estimator gets its signal — small for the separated densities shown here; the shape is illustrative." %}

The core problem is to bridge A and B without requiring direct overlap between their equilibrium distributions. AIS supplies that bridge.

---

## Part 2: The AIS Framework

AIS (<span id="cite-neal2001"></span>[Neal, 2001](#ref-neal2001)) fixes the overlap problem from direct free-energy comparison by bridging states A and B through intermediate distributions. Neighboring distributions overlap even when the endpoints do not. The resulting forward and reverse path measures yield three identities; in the physics specialization, these become Jarzynski's equality, Crooks' fluctuation theorem, and the second law.

### The AIS Setup

AIS bridges two distributions by constructing intermediates $$\hat{p}_0, \hat{p}_1, \ldots, \hat{p}_K$$ — unnormalized densities with unknown normalizing constants $$Z_k$$. The forward chain works as follows:

1. Draw $$\mathbf{x}_0 \sim p_0 = \hat{p}_0 / Z_0$$.
2. For $$k = 1, \ldots, K$$: apply an MCMC transition $$T_k(\mathbf{x}_k \mid \mathbf{x}_{k-1})$$ that leaves $$p_k$$ invariant.
3. Record the log importance weight:

$$\log w = \sum_{k=0}^{K-1} \bigl[\log \hat{p}_{k+1}(\mathbf{x}_k) - \log \hat{p}_k(\mathbf{x}_k)\bigr]$$

Two runs give different weights because each MCMC chain follows a different random path — $$w$$ is a random variable. With many intermediates (slow annealing), each density ratio is close to 1 and $$\log w$$ clusters tightly around $$\log(Z_K/Z_0)$$. With few intermediates (fast annealing), the ratios fluctuate wildly and the estimate degrades.

### Forward and Reverse Path Measures

The forward path measure is the joint probability of the entire chain $$(\mathbf{x}_0, \ldots, \mathbf{x}_K)$$:

$$\mathcal{P}_F[\mathbf{x}_0, \ldots, \mathbf{x}_K] = p_0(\mathbf{x}_0) \prod_{k=1}^{K} T_k(\mathbf{x}_k \mid \mathbf{x}_{k-1})$$

The reverse path measure starts from $$p_K$$ and steps backward:

$$\mathcal{P}_R[\mathbf{x}_0, \ldots, \mathbf{x}_K] = p_K(\mathbf{x}_K) \prod_{k=1}^{K} \tilde{T}_k(\mathbf{x}_{k-1} \mid \mathbf{x}_k)$$

where $$\tilde{T}_k$$ is the time-reversal of $$T_k$$ under its invariant density $$p_k$$, defined by $$p_k(\mathbf{x}) T_k(\mathbf{y} \mid \mathbf{x}) = p_k(\mathbf{y}) \tilde{T}_k(\mathbf{x} \mid \mathbf{y})$$.

### The Path Measure Ratio

The log-ratio decomposes as:

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R} = \ln \frac{p_0(\mathbf{x}_0)}{p_K(\mathbf{x}_K)} + \sum_{k=1}^{K} \ln \frac{T_k(\mathbf{x}_k \mid \mathbf{x}_{k-1})}{\tilde{T}_k(\mathbf{x}_{k-1} \mid \mathbf{x}_k)}$$

This reversal identity gives $$\ln \frac{T_k(\mathbf{x}_k \mid \mathbf{x}_{k-1})}{\tilde{T}_k(\mathbf{x}_{k-1} \mid \mathbf{x}_k)} = \ln p_k(\mathbf{x}_k) - \ln p_k(\mathbf{x}_{k-1})$$. Substituting:

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R} = \ln p_0(\mathbf{x}_0) - \ln p_K(\mathbf{x}_K) + \sum_{k=1}^{K} \bigl[\ln p_k(\mathbf{x}_k) - \ln p_k(\mathbf{x}_{k-1})\bigr]$$

The MCMC kernels have cancelled completely — the ratio depends only on the densities $$p_k$$, not the transition kernels $$T_k$$.

The sum telescopes when we group terms by sample point. Each $$\mathbf{x}_j$$ contributes $$+\ln p_j(\mathbf{x}_j)$$ from the $$k=j$$ term and $$-\ln p_{j+1}(\mathbf{x}_j)$$ from the $$k=j+1$$ term:

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R} = -\sum_{k=0}^{K-1} \bigl[\ln p_{k+1}(\mathbf{x}_k) - \ln p_k(\mathbf{x}_k)\bigr]$$

Writing $$\ln p_k = \ln \hat{p}_k - \ln Z_k$$ and recognizing the importance weight:

> **Path measure ratio.**
>
> $$\log \frac{\mathcal{P}_F}{\mathcal{P}_R} = \log \frac{Z_K}{Z_0} - \log w$$
>
> The MCMC kernels cancel by the reversal identity (ordinary detailed balance is the case where the reverse kernel equals the forward kernel), and the path measure ratio depends only on the importance weight $$w$$ and the normalizing constant ratio $$Z_K / Z_0$$. (All logs in the AIS framework are natural logarithms.)
{: .block-definition }

### Unbiasedness of AIS

From $$\mathcal{P}_R / \mathcal{P}_F = w / (Z_K/Z_0)$$, sum over all trajectories weighted by $$\mathcal{P}_F$$:

$$\left\langle \frac{\mathcal{P}_R}{\mathcal{P}_F} \right\rangle_F = \sum_{\text{traj}} \mathcal{P}_R = 1$$

$$\left\langle \frac{w}{Z_K/Z_0} \right\rangle_F = 1$$

> **Unbiasedness of AIS.**
>
> $$\langle w \rangle_F = \frac{Z_K}{Z_0}$$
>
> The average importance weight over forward chains is exactly the normalizing constant ratio — regardless of the number of intermediates $$K$$ or the quality of the MCMC transitions.
{: .block-definition }

This guarantee assumes an exact draw from $$p_0$$, kernels that leave their assigned $$p_k$$ invariant, finite normalizers, and support sufficient to reweight the reverse measure from the forward one. Full equilibration at each level is not required.

### The ELBO and Its Gap

Taking logs and applying Jensen's inequality ($$\log \langle w \rangle \geq \langle \log w \rangle$$):

$$\langle \log w \rangle_F \leq \log \frac{Z_K}{Z_0}$$

The left side is the ELBO — the evidence lower bound. The gap has an information-theoretic interpretation:

> **ELBO gap = KL divergence.**
>
> $$\log \frac{Z_K}{Z_0} - \langle \log w \rangle_F = D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$
>
> The gap between the true log-normalizing-constant ratio and the ELBO equals the KL divergence between forward and reverse path measures. The bound is tight if and only if $$\mathcal{P}_F = \mathcal{P}_R$$.
{: .block-definition }

This follows directly:

$$\begin{aligned}
D_{\text{KL}}(\mathcal{P}_F \Vert \mathcal{P}_R)
&= \langle \log(\mathcal{P}_F / \mathcal{P}_R) \rangle_F\\
&= \log(Z_K/Z_0) - \langle \log w \rangle_F.
\end{aligned}$$

The Jensen bound is tight exactly when $$w$$ is constant almost surely under the forward measure. Full equilibration at each of finitely many intermediate distributions is not sufficient. For example, a one-step change from $$\mathcal{N}(0,1)$$ to $$\mathcal{N}(m,1)$$ has $$\log w=mx_0-m^2/2$$ with $$x_0\sim\mathcal{N}(0,1)$$. Its variance is $$m^2$$ even if the subsequent transition draws an exact equilibrium sample. A finely spaced schedule with adequate mixing can reduce the gap; the schedule and dynamics matter together.

**The variance problem.** With too few intermediates, most chains have $$w \ll Z_K/Z_0$$, and rare ones with $$w \gg Z_K/Z_0$$ dominate the average. The estimator is unbiased under the assumptions above but can have very large variance — Zwanzig's overlap problem from the free-energy section, transferred from configuration space to trajectory space.

---

## Part 3: Non-Equilibrium Processes and Work

Non-equilibrium physics is a special case of the AIS identities: choose Boltzmann intermediates along a protocol and Langevin dynamics as the transition kernel. This specialization turns the log importance weight into work, and the same path-measure ratio becomes the thermodynamic identities derived in the Non-Equilibrium Equalities section.

**Intermediates = Boltzmann distributions along a protocol.** Define a protocol parameter $$\lambda_k$$ interpolating from $$\lambda_0 = 0$$ (state A) to $$\lambda_K = 1$$ (state B), with:

$$\hat{p}_k(\mathbf{x}) = e^{-\beta U(\mathbf{x}, \lambda_k)}, \qquad U(\mathbf{x}, \lambda) = (1 - \lambda) U_A(\mathbf{x}) + \lambda \, U_B(\mathbf{x})$$

**Relaxation by Langevin dynamics.** First change the protocol from $$\lambda_k$$ to $$\lambda_{k+1}$$ at fixed $$\mathbf{x}_k$$, then relax at the new potential. With mobility set to one, an Euler–Maruyama approximation to that relaxation is

$$\mathbf{x}_{k+1}=\mathbf{x}_k-\nabla U(\mathbf{x}_k,\lambda_{k+1})\Delta t+\sqrt{2/\beta}\,\boldsymbol{\xi}_k,\qquad \boldsymbol{\xi}_k\sim\mathcal{N}(0,\Delta t\,\mathbf I).$$

An unadjusted Euler step does **not** generally leave the Boltzmann density invariant at finite $$\Delta t$$. Exact finite-step AIS requires an invariant kernel, such as a Metropolis-adjusted Langevin transition or an exact frozen-potential transition. The unadjusted equation is a discretization of the continuous-time physics, not an exact finite-step AIS kernel.

**The importance weight becomes work.** With the distributions and dynamics specified, we can evaluate the per-step log-density ratio from the AIS framework in this physical setting:

$$\log \hat{p}_{k+1}(\mathbf{x}_k) - \log \hat{p}_k(\mathbf{x}_k) = -\beta\bigl[U(\mathbf{x}_k, \lambda_{k+1}) - U(\mathbf{x}_k, \lambda_k)\bigr]$$

So $$\log w = -\beta \sum_{k=0}^{K-1} [U(\mathbf{x}_k, \lambda_{k+1}) - U(\mathbf{x}_k, \lambda_k)]$$. Physicists call this sum the work:

> **Work (discrete).**
>
> $$W = \sum_{k=0}^{K-1} \bigl[U(\mathbf{x}_k, \lambda_{k+1}) - U(\mathbf{x}_k, \lambda_k)\bigr]$$
>
> The total change in potential energy due to protocol steps, evaluated at the current configuration. Each term is the energy cost of shifting the potential from $$\lambda_k$$ to $$\lambda_{k+1}$$ while the particle sits at $$\mathbf{x}_k$$. The importance weight is $$w = e^{-\beta W}$$.
{: .block-definition }

For the linear interpolation,

$$\begin{aligned}
&U(\mathbf{x}, \lambda_{k+1}) - U(\mathbf{x}, \lambda_k)\\
&\qquad= \Delta\lambda_k \cdot [U_B(\mathbf{x}) - U_A(\mathbf{x})].
\end{aligned}$$

The work at each step is the energy difference between B and A, scaled by the protocol step size.

{% include figure.liquid loading="eager" path="assets/img/blog/pm_alternating_steps.svg" alt="Alternating protocol changes and relaxation steps move sample configurations through changing energy landscapes." class="img-fluid rounded z-depth-1" zoomable=true caption="The mechanism of a non-equilibrium process (= AIS). Amber panels are work steps, where the potential shifts while particles stay fixed; teal panels are relaxation steps, where particles move under the current potential." %}

Now run many independent copies. Each receives different noise, producing different trajectories and different work values:

- **Lucky run (low work, high $$w$$).** The sample lands in a high-density region of $$p_{k+1}$$ before the distribution shifts much — per-step ratios are close to 1, $$\log w$$ stays near $$\log(Z_K/Z_0)$$, and work $$W$$ is close to $$\Delta F$$.
- **Unlucky run (high work, low $$w$$).** The chain gets trapped in a mode of $$p_k$$ with low density under $$p_{k+1}$$ — each step gives a large negative density ratio, $$\log w$$ falls far below $$\log(Z_K/Z_0)$$, and work $$W$$ far exceeds $$\Delta F$$.

{% include figure.liquid loading="eager" path="assets/img/blog/pm_work_trajectories.svg" alt="Toy trajectories, their cumulative work, and a work histogram illustrate variation between switching runs." class="img-fluid rounded z-depth-1" zoomable=true caption="Many runs of the same protocol produce different trajectories and work values. Low-work trajectories are rare but dominate the exponential average, which is why Jarzynski's equality is statistically demanding." %}

**Quasistatic vs. driven processes.** Protocol speed and relaxation jointly control the work distribution:

- **Quasistatic limit:** Infinitesimal protocol changes with adequate equilibration can make work concentrate at $$\Delta F$$. Merely increasing $$K$$ at fixed physical duration does not slow the protocol, and exact equilibration between finitely sized changes does not eliminate work variance.
- **Driven protocol:** The system can lag behind the instantaneous equilibrium density. The mean excess work $$\langle W\rangle-\Delta F\geq0$$ equals the path-space KL divided by $$\beta$$. Equality or strict inequality is determined by the path measures, not by finite $$K$$ alone.

**Notation bridge.** The two notational systems used throughout this post map as follows:

<div class="table-responsive" markdown="1">

| AIS | Physics |
|---|---|
| Unnormalized density $$\hat{p}_k$$ | Boltzmann factor $$e^{-\beta U(\mathbf{x}, \lambda_k)}$$ |
| Log importance weight $$\ln w$$ | $$-\beta W$$ (negative work times inverse temperature) |
| Log normalizing constant ratio $$\ln(Z_K/Z_0)$$ | $$-\beta \Delta F$$ (negative free energy difference) |
| Path measure ratio $$\ln(Z_K/Z_0) - \ln w$$ | $$\beta(W - \Delta F)$$ |

</div>

**The AIS identities become physics.** The path measure ratio $$\ln(\mathcal{P}_F / \mathcal{P}_R) = \ln(Z_K/Z_0) - \ln w$$ becomes $$\beta(W - \Delta F)$$, where $$\Delta F = -(1/\beta)\ln(Z_K/Z_0)$$ is the free energy difference. The three AIS identities from the AIS framework now have direct physical names:

<div class="table-responsive" markdown="1">

| AIS identity | Physics name | Statement |
|---|---|---|
| Unbiasedness: $$\langle w \rangle = Z_K/Z_0$$ | Jarzynski's equality | $$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$ |
| ELBO: $$\langle \log w \rangle \leq \log(Z_K/Z_0)$$ | Second law | $$\langle W \rangle \geq \Delta F$$ |
| ELBO gap = KL | Dissipation = KL | $$\langle W \rangle - \Delta F = (1/\beta) \, D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$ |

</div>

The path measure ratio contains more than expectations: it relates the full *distribution* of $$W$$ under the forward and reverse processes. This is Crooks' fluctuation theorem, the strongest of the three results, developed fully in the non-equilibrium equalities section.

---

## Part 4: Continuous-Time Machinery

Discrete chains are enough for AIS, but Langevin dynamics live in continuous time. To carry the same path-measure ratio to SDEs, we need path integrals, Radon-Nikodym derivatives, Girsanov's theorem, and forward-backward SDEs.

AIS introduced path measures in discrete time — products of MCMC kernels over finite chains. But continuous-time dynamics, such as the Langevin SDE, produce trajectories in $$C([0, T]; \mathbb{R}^d)$$, where the discrete product formula no longer applies. We need a continuous-time theory of path measures.[^pathmeasure]


[^pathmeasure]: I use "path measure" throughout this post. Physicists often say "path integral" for the same concept — summing/integrating over all possible trajectories weighted by an action. The mathematical content is closely related to Feynman's path integral in quantum mechanics, but our context is classical stochastic dynamics rather than quantum amplitudes.

### The Path Integral Picture

A finite-grid path has an ordinary joint density. For overdamped Langevin dynamics with mobility one, the Euler–Maruyama conditional density is

$$P(\mathbf{x}_{k+1}\mid\mathbf{x}_k)
=(4\pi\Delta t/\beta)^{-d/2}
\exp\!\left[-\frac{\beta}{4\Delta t}
\left|\Delta\mathbf{x}_k+\nabla U(\mathbf{x}_k,\lambda_k)\Delta t\right|^2\right],$$

where $$d$$ is the configuration dimension and $$\Delta\mathbf{x}_k=\mathbf{x}_{k+1}-\mathbf{x}_k$$. The joint density of the discretized path is

$$\mathcal P_\Delta(\mathbf{x}_{0:N})=p_A(\mathbf{x}_0)\prod_{k=0}^{N-1}P(\mathbf{x}_{k+1}\mid\mathbf{x}_k).$$

Taking its logarithm yields a sum of squared drift residuals. This motivates action-based pictures of trajectories: a large residual is a less likely increment under the Gaussian transition. Underdamped dynamics requires positions and velocities instead.[^underdamped]

### Why the Path Integral Picture Is Not Enough

Brownian sample paths are almost surely nondifferentiable. We therefore cannot replace the finite-grid sum by an ordinary integral involving $$\lvert\dot{\mathbf{x}}+\nabla U\rvert^2$$ and call it the density of an exact continuous path. There is no flat Lebesgue-like reference measure on this infinite-dimensional path space.

Onsager–Machlup formulas can describe small-tube probabilities around sufficiently regular comparison paths, with the appropriate correction terms and convention. They are not ordinary probability densities of Brownian sample paths. For the expectation identities here, the useful object is a density **relative to another path measure**, obtained through a Radon–Nikodym derivative.

[^underdamped]: Underdamped Langevin dynamics adds velocities; noise usually acts only on velocities. Its change-of-measure conditions must respect that noise structure, so one cannot change arbitrary position drifts with the scalar-noise formula below.

### Radon-Nikodym Derivatives: The Right Way to Compare Path Measures

Individual path measures have no density with respect to a flat reference. Two path measures can nevertheless have densities relative to each other when the required absolute continuity holds. Matching diffusion coefficients is important, but it is not sufficient by itself. This is the same idea as importance sampling: we don't need $$p(x)$$ and $$q(x)$$ individually — we need their ratio $$p(x)/q(x)$$.

Given two probability measures $$\mathbb{P}$$ and $$\mathbb{Q}$$ on the same space, the Radon-Nikodym derivative $$d\mathbb{P}/d\mathbb{Q}$$ is the density of $$\mathbb{P}$$ with respect to $$\mathbb{Q}$$ — the function that reweights $$\mathbb{Q}$$-samples to produce $$\mathbb{P}$$-expectations:

$$\mathbb{E}_{\mathbb{P}}[f(X)] = \mathbb{E}_{\mathbb{Q}}\left[\frac{d\mathbb{P}}{d\mathbb{Q}}(X) \cdot f(X)\right]$$

For distributions on $$\mathbb{R}^d$$, this is just the likelihood ratio $$p(x)/q(x)$$. For path measures — distributions on $$C([0, T]; \mathbb{R}^d)$$ — the Radon-Nikodym derivative is a functional of the entire trajectory. It exists when the numerator measure is absolutely continuous with respect to the denominator. Initial-law support and drift-integrability conditions matter as well as the diffusion coefficient.



### Girsanov's Theorem: Change of Measure for SDEs

Consider two forward SDEs with the same initial law and constant scalar noise amplitude $$\sigma>0$$:

$$\begin{aligned}
dX_t &= a(X_t,t)\,dt+\sigma\,dW_t,\\
dX_t &= \tilde a(X_t,t)\,dt+\sigma\,d\widetilde W_t.
\end{aligned}$$

Write $$\delta_t=a(X_t,t)-\tilde a(X_t,t)$$. Assume the SDEs are well posed and the change-of-measure exponential is a true martingale. A sufficient integrability condition is Novikov's condition under the reference law:

$$\mathbb E_{\mathbb P^{\tilde a}}\!\left[
\exp\!\left(\frac{1}{2\sigma^2}\int_0^T|\delta_t|^2dt\right)\right]<\infty.$$

> **Girsanov's formula.**
>
> $$\begin{aligned}
> \log\frac{d\mathbb P^a}{d\mathbb P^{\tilde a}}(X)
> ={}&\frac{1}{\sigma^2}\int_0^T\delta_t\cdot
> \bigl(dX_t-\tilde a(X_t,t)\,dt\bigr)\\
> &-\frac{1}{2\sigma^2}\int_0^T|\delta_t|^2dt.
> \end{aligned}$$
{: .block-definition }

Under the reference process, $$dX_t-\tilde a_tdt=\sigma dW_t$$. Thus the first term is a zero-mean stochastic integral when square integrable. The second is an ordinary time integral, but remains random because its integrand depends on the path. If the initial laws differ, their log density ratio must also be included.

Equivalently, expanding the reference-drift term gives

$$\log\frac{d\mathbb P^a}{d\mathbb P^{\tilde a}}(X)
=\frac{1}{\sigma^2}\int_0^T(a_t-\tilde a_t)\cdot dX_t
-\frac{1}{2\sigma^2}\int_0^T\bigl(|a_t|^2-|\tilde a_t|^2\bigr)dt.$$

The final integrand is a **difference of squared norms**, not the squared norm of the drift difference. The discrete calculation makes that distinction explicit.

### Discrete Derivation of Girsanov's Formula

At a step of length $$\Delta t$$, the two Euler–Maruyama kernels have means $$\mathbf{x}_k+a_k\Delta t$$ and $$\mathbf{x}_k+\tilde a_k\Delta t$$, with common covariance $$\sigma^2\Delta t\,\mathbf I$$. Their Gaussian normalizers cancel:

$$\begin{aligned}
\log\frac{P^a(\mathbf{x}_{k+1}\mid\mathbf{x}_k)}
{P^{\tilde a}(\mathbf{x}_{k+1}\mid\mathbf{x}_k)}
&=-\frac{|\Delta\mathbf{x}_k-a_k\Delta t|^2
-|\Delta\mathbf{x}_k-\tilde a_k\Delta t|^2}{2\sigma^2\Delta t}\\
&=\frac{(a_k-\tilde a_k)\cdot\Delta\mathbf{x}_k}{\sigma^2}
-\frac{|a_k|^2-|\tilde a_k|^2}{2\sigma^2}\Delta t.
\end{aligned}$$

Summing over steps gives the discretized path log-ratio. Under conditions justifying the limit, the left-endpoint sum becomes the Itô integral above. This algebra checks the formula; the martingale and absolute-continuity conditions justify it for continuous paths.

As a sign check, take $$a=2$$, $$\tilde a=1$$, $$\sigma=1$$, $$T=1$$, and $$X_0=0$$. The path log-ratio is $$X_1-3/2$$, exactly the log-ratio of the endpoint densities $$\mathcal N(2,1)$$ and $$\mathcal N(1,1)$$. Using $$\lvert a-\tilde a\rvert^2$$ in the expanded formula would incorrectly give $$X_1-1/2$$.

Girsanov compares processes running in the same time direction. Next we need to compare forward and backward descriptions.

### Forward-Backward SDEs

Girsanov's theorem compares two processes running in the *same* direction — two forward SDEs with different drifts. But we need to compare processes running in *opposite* directions: the forward protocol (A $$\to$$ B) against the reverse protocol (B $$\to$$ A). This requires pairing a forward Itô SDE with a backward Itô SDE, which introduces a new notational distinction (forward vs. backward integration). We adopt the notation from <span id="cite-vargas2024"></span>[Vargas et al., 2024](#ref-vargas2024):

> **Forward-backward SDEs.**
>
> $$d X_t = a(X_t, t) \, dt + \sigma \, \fwd{d} W_t, \qquad X_0 \sim \mu \qquad \Rightarrow \qquad X \sim \fwd{\mathbb{P}}^{\mu, a}$$
>
> $$d X_t = b(X_t, t) \, dt + \sigma \, \bwd{d} W_t, \qquad X_T \sim \nu \qquad \Rightarrow \qquad X \sim \bwd{\mathbb{P}}^{\nu, b}$$
>
> where $$\fwd{d} W_t$$ and $$\bwd{d} W_t$$ denote forward and backward Itô integration (see the [Fokker-Planck post](/blog/2026/fokker-planck-equation/) for definitions of Itô calculus and stochastic integrals), and $$\fwd{\mathbb{P}}^{\mu, a}$$, $$\bwd{\mathbb{P}}^{\nu, b}$$ are the associated path measures on $$C([0, T]; \mathbb{R}^d)$$.
{: .block-definition }

The forward SDE generates trajectories from $$\mu$$ at time 0; the backward SDE generates trajectories from $$\nu$$ at time $$T$$. For the physics setting, $$a_t = \frac{\sigma^2}{2} \nabla \log \pi_t = -\nabla U_t$$ with $$\pi_t$$ interpolating from $$\pi_0$$ to $$\pi_T$$ — exactly the non-equilibrium protocol from the work section. For diffusion models, $$a_t$$ is the noising drift and $$b_t$$ is the learned denoising drift.

**Discrete-time counterpart.** Discretizing the forward SDE with step size $$\Delta t$$ gives the Euler-Maruyama chain:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + a(\mathbf{x}_k, t_k) \Delta t + \sigma \boldsymbol{\xi}_k, \qquad \boldsymbol{\xi}_k \sim \mathcal{N}(0, \Delta t \, \mathbf{I})$$

with path measure $$\fwd{\mathbb{P}}^{\mu, a}[\mathbf{x}_0, \ldots, \mathbf{x}_N] = \mu(\mathbf{x}_0) \cdot \prod_k P(\mathbf{x}_{k+1} \mid \mathbf{x}_k)$$. The backward SDE discretizes analogously, starting from $$\mathbf{x}_N \sim \nu$$ and stepping in reverse:

$$\mathbf{x}_{k} = \mathbf{x}_{k+1} - b(\mathbf{x}_{k+1}, t_{k+1}) \Delta t + \sigma \boldsymbol{\xi}_k,\qquad \Delta t=t_{k+1}-t_k>0.$$

with path measure $$\bwd{\mathbb{P}}^{\nu, b}[\mathbf{x}_0, \ldots, \mathbf{x}_N] = \nu(\mathbf{x}_N) \cdot \prod_k P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})$$. The reverse transition $$P_R(\mathbf{x}_k \mid \mathbf{x}_{k+1})$$ is a Gaussian centered at $$\mathbf{x}_{k+1} - b(\mathbf{x}_{k+1}, t_{k+1}) \Delta t$$ — the same noise variance, but the drift is evaluated at $$\mathbf{x}_{k+1}$$ and pushes backward. The minus sign is essential because simulation steps toward smaller values of the original time coordinate. This Gaussian discretization need not equal the exact invariant-density reversal kernel in the AIS framework at finite step size.

*AIS parallel: For the invariant-kernel construction in the AIS framework, $$w=(Z_K/Z_0)\mathcal P_R/\mathcal P_F$$. Thus the normalized weight is the reverse-to-forward density ratio, not the forward-to-reverse ratio.*

### The Forward-Backward Radon-Nikodym Derivative

The remaining object is the log-ratio between the forward and backward path measures. Vargas et al. (2024, Proposition 2.2) give it as:

> **Forward-backward Radon-Nikodym derivative.** Given a reference path measure $$\fwd{\mathbb{P}}^{\Gamma_0, \gamma^+} = \bwd{\mathbb{P}}^{\Gamma_T, \gamma^-}$$:
>
> $$\ln \frac{d\fwd{\mathbb{P}}^{\mu, a}}{d\bwd{\mathbb{P}}^{\nu, b}}(X) = \ln \frac{d\mu}{d\Gamma_0}(X_0) - \ln \frac{d\nu}{d\Gamma_T}(X_T)$$
>
> $$\quad + \frac{1}{\sigma^2} \int_0^T (a_t - \gamma_t^+)(X_t) \cdot \left(\fwd{d}X_t - \frac{1}{2}(a_t + \gamma_t^+)(X_t) \, dt\right)$$
>
> $$\quad - \frac{1}{\sigma^2} \int_0^T (b_t - \gamma_t^-)(X_t) \cdot \left(\bwd{d}X_t - \frac{1}{2}(b_t + \gamma_t^-)(X_t) \, dt\right)$$
{: .block-definition }

This generalizes Girsanov's theorem. The proof applies Girsanov twice — once for the forward process, once for the backward — using the reference to bridge between them. The reference can vary, but its endpoints and two drifts must describe one consistent path measure, and the required absolute continuity must hold. They cannot be selected independently. Different valid references redistribute terms in the same ratio.[^otchoice]

[^otchoice]: The connection to optimal transport is direct: the Benamou-Brenier formula characterizes optimal transport as a variational problem over path measures, and the Schrödinger bridge problem — finding the path measure closest to a reference that matches given marginals — is a regularized version of OT. See Vargas et al. (2024, Section 3.1) for details.

### Nelson's Relation: When the RND Is Trivial

Nelson's relation characterizes when the forward-backward RND equals 1, meaning the two path measures are identical (<span id="cite-nelson1967"></span>[Nelson, 1967](#ref-nelson1967)):

> **Nelson's relation.** $$\fwd{\mathbb{P}}^{\mu, a} = \bwd{\mathbb{P}}^{\nu, b}$$ if and only if $$\nu = \fwd{\mathbb{P}}^{\mu, a}_T$$ and
>
> $$b_t = a_t - \sigma^2 \nabla \log \rho_t^{\mu, a}, \qquad \forall \, t \in (0, T]$$
>
> where $$\rho_t^{\mu, a}$$ is the time-marginal density of the forward process.
{: .block-definition }

This relation describes the exact time reversal of a process, including one far from equilibrium. It does not imply thermodynamic reversibility. The density $$\rho_t^{\mu,a}$$ is the process's actual marginal, which generally differs from the instantaneous Boltzmann density $$\pi_t$$ during a driven protocol.

For example, with $$a_t=-\nabla U_t$$ and $$\sigma^2=2/\beta$$, the exact backward drift is $$-\nabla U_t-\sigma^2\nabla\log\rho_t^{\mu,a}$$. The physical reverse protocol instead uses $$+\nabla U_t$$ in backward-time notation and starts from equilibrium at the final potential. These are the same only under the appropriate equilibrium/reversibility conditions. Crooks compares the physical protocols, not a process with its tautologically identical exact time reversal.

### The Work Identity: Plugging in the Physics

We now plug the physics, potential $$U$$, temperature $$\beta$$, and protocol $$\lambda(t)$$, into the general forward-backward RND and recover the identity $$\ln(\mathcal{P}_F / \mathcal{P}_R) = \beta(W - \Delta F)$$. This is the continuous-time version of the AIS framework's path measure ratio $$\ln(Z_K/Z_0) - \ln w$$. The Stratonovich chain rule collapses the path integrals into the work functional, generalizing beyond any particular discretization.

We now return to the physics notation. The three descriptions use the following correspondences:

<div class="table-responsive" markdown="1">

| Physics | General SDEs | AIS |
|---|---|---|
| $$\mathcal{P}_F$$ | $$\fwd{\mathbb{P}}^{\mu, a}$$ | Forward chain distribution |
| $$\mathcal{P}_R$$ | $$\bwd{\mathbb{P}}^{\nu, b}$$ | Reverse chain distribution |
| $$U(\mathbf{x}, \lambda)$$ | $$-\frac{\sigma^2}{2} \ln \hat{\pi}_t(\mathbf{x})$$ | $$-\log \hat{p}_k(\mathbf{x})$$ |
| $$\beta = 1/k_BT$$ | $$2/\sigma^2$$ | 1 (absorbed into densities) |
| Work $$W$$ | $$\int_0^T \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt$$ | $$-\log w$$ |
| $$\Delta F$$ | $$-\frac{\sigma^2}{2} \ln(Z_T/Z_0)$$ | $$-\log(Z_K/Z_0)$$ |

</div>

We specialize the forward-backward RND to the non-equilibrium physics setting. Recall from the work section the time-dependent potential $$U(\mathbf{x}, \lambda(t))$$ with Boltzmann distribution $$\pi_t(\mathbf{x}) = e^{-\beta U(\mathbf{x}, \lambda(t))} / Z_t$$. The forward SDE is overdamped Langevin under this potential, and the backward SDE reverses the drift:

$$\text{Forward:} \quad dX_t = -\nabla U(X_t, \lambda(t)) \, dt + \sqrt{2/\beta} \, \fwd{d}W_t, \qquad X_0 \sim \pi_0$$

$$\text{Backward:} \quad dX_t = +\nabla U(X_t, \lambda(t)) \, dt + \sqrt{2/\beta} \, \bwd{d}W_t, \qquad X_T \sim \pi_T$$

The backward process describes the physical reverse protocol in the original time coordinate: stepping backward applies the force $$-\nabla U$$ at the reversed schedule. Assume a smooth protocol, normalizable endpoint Boltzmann densities, well-posed dynamics, and the absolute-continuity and integrability conditions needed for the path ratios. Both runs begin in equilibrium at their respective starting endpoints.

Combining the conditional forward/reverse ratio with these equilibrium endpoint densities gives the work identity below. One must not set both reference drifts to zero while independently prescribing arbitrary endpoint laws; that would not define a consistent reference process.

> **Path measure ratio (core identity, rigorous form).**
>
> $$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = -\beta \Delta F + \beta W[X]$$
>
> where the work is $$W[X] = \int_0^T \frac{\partial U}{\partial \lambda}(X_t, \lambda(t)) \dot{\lambda}(t) \, dt$$ and the free energy difference is $$\Delta F = F_T - F_0 = -\frac{1}{\beta} \ln \frac{Z_T}{Z_0}$$.
{: .block-definition }

The general forward-backward RND was given above. This work identity is the result of *specializing* it to the physics setting — choosing the specific drifts $$\pm \nabla U$$ from the non-equilibrium protocol and using the Stratonovich chain rule to collapse the path integrals into the work functional $$W$$ from the work section.

<details>
<summary><strong>Derivation of the work identity (click to expand)</strong></summary>

<p><strong>Step 1: Conditional path ratio.</strong> Let \(g_k=\nabla U(\mathbf{x}_k,\lambda_k)\). The forward Gaussian kernel has residual \(\Delta\mathbf{x}_k+g_k\Delta t\); the reversed physical kernel has residual \(-\Delta\mathbf{x}_k+g_{k+1}\Delta t\). Expanding their log-ratio gives a midpoint force term and a squared-force difference:</p>

$$-\frac{\beta}{2}(g_k+g_{k+1})\cdot\Delta\mathbf{x}_k
-\frac{\beta\Delta t}{4}\bigl(|g_k|^2-|g_{k+1}|^2\bigr).$$

<p>The squared-force terms telescope to a vanishing boundary term as the grid is refined, under the stated regularity conditions. The midpoint sum converges to a Stratonovich integral. Adding the equilibrium starting densities gives:</p>

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = \ln \frac{\pi_0(X_0)}{\pi_T(X_T)} - \beta \int_0^T \nabla U(X_t, \lambda(t)) \circ dX_t$$

<p>This is a continuous-time identity. The finite Euler kernels illustrate its limiting algebra; without an invariant-kernel adjustment or discretization correction, they do not give the exact finite-step work identity.</p>

<p><strong>Step 2: Apply the Stratonovich chain rule to \(U(X_t, \lambda(t))\).</strong> The Stratonovich integral \(\circ dX_t\) preserves the ordinary chain rule from calculus — \(df(X_t) = f'(X_t) \circ dX_t\) — which is why the following telescoping works. (The Itô integral does not preserve the chain rule; it would add a correction term \(\frac{1}{2}\sigma^2 \Delta U\), which must then be tracked separately.) Applying the Stratonovich chain rule:</p>

$$dU(X_t, \lambda(t)) = \nabla U \circ dX_t + \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt$$

<p>Rearranging and integrating from \(0\) to \(T\):</p>

$$-\beta \int_0^T \nabla U \circ dX_t = -\beta \bigl[U(X_T, \lambda_T) - U(X_0, \lambda_0)\bigr] + \beta \int_0^T \frac{\partial U}{\partial \lambda} \dot{\lambda} \, dt$$

<p>The second term is \(\beta W\) — the work from the work section.</p>

<p><strong>Step 3: Combine and cancel.</strong> Substituting into Step 1 and using \(\pi_t = e^{-\beta U(\cdot, \lambda(t))}/Z_t\):</p>

$$\ln \frac{\mathcal{P}_F}{\mathcal{P}_R}(X) = \underbrace{\ln \frac{\pi_0(X_0)}{\pi_T(X_T)}}_{-\beta U_0(X_0) + \beta U_T(X_T) + \ln(Z_T/Z_0)} \underbrace{- \beta[U_T(X_T) - U_0(X_0)]}_{\text{from chain rule}} + \beta W$$

<p>The potential energy terms cancel exactly:</p>

$$= \ln \frac{Z_T}{Z_0} + \beta W = -\beta \Delta F + \beta W = \beta(W - \Delta F)$$

<p>∎</p>

<p><strong>Why Stratonovich?</strong> The Stratonovich integral preserves the ordinary chain rule, which is why the telescoping in Step 2 works cleanly. Using the Itô integral instead would introduce an additional \(\frac{1}{2}\sigma^2 \Delta U\) correction term (the Itô-Stratonovich conversion), which must then be tracked and cancelled — possible but messier.</p>

</details>

The continuous-time path measure ratio has the same form $$\beta(W - \Delta F)$$ as the discrete AIS ratio $$\ln(Z_K/Z_0) - \ln w$$ from the AIS framework, with $$\ln w = -\beta W$$ and $$\Delta F = -(1/\beta)\ln(Z_K/Z_0)$$. The continuous-time derivation adds no new equality. It relates the exact SDE identity to discrete approximations. Exact finite-step AIS has its own invariant-kernel proof; convergence of a numerical scheme alone is not a guarantee that finite-step exponential-work estimates are unbiased. The next section extracts the three named equalities from this identity.

---

## Part 5: Non-Equilibrium Equalities

The discrete and continuous derivations give the same identity:

- **Discrete (the AIS framework):** $$\ln(\mathcal{P}_F / \mathcal{P}_R) = \ln(Z_K/Z_0) - \ln w$$, from detailed balance cancellation in AIS chains.
- **Continuous (the continuous-time section):** $$\ln(\mathcal{P}_F / \mathcal{P}_R) = \beta(W - \Delta F)$$, from the Stratonovich chain rule applied to the Langevin SDE.

These are the same identity in different notation ($$\ln w = -\beta W$$, $$\ln(Z_K/Z_0) = -\beta \Delta F$$). This part extracts the three named results that follow from it.

### Jarzynski's Equality

The derivation is the same as the AIS framework's unbiasedness proof, now in physics notation. From $$\mathcal{P}_F / \mathcal{P}_R = e^{\beta(W - \Delta F)}$$, integrating $$\mathcal{P}_R / \mathcal{P}_F$$ against $$\mathcal{P}_F$$ gives 1:

$$\left\langle e^{-\beta(W - \Delta F)} \right\rangle_F = 1$$

> **Jarzynski's equality (Jarzynski, 1997).**
>
> $$\left\langle e^{-\beta W} \right\rangle_F = e^{-\beta \Delta F}$$
>
> The exponential average of work over forward non-equilibrium trajectories gives the exact equilibrium free energy difference — regardless of protocol speed, number of steps, or how far from equilibrium the process is driven.
{: .block-definition }

The equality has three useful consequences:

1. $$\Delta F$$ is an equilibrium quantity. $$W$$ is measured from non-equilibrium trajectories. The equality holds for *any* protocol.
2. The second law $$\langle W \rangle \geq \Delta F$$ follows from Jensen's inequality: $$e^{-\beta \Delta F} = \langle e^{-\beta W} \rangle \geq e^{-\beta \langle W \rangle}$$.
3. The bound is tight only when $$W$$ is constant — the quasistatic limit where every trajectory gives $$W = \Delta F$$.

In AIS terms: $$\langle w \rangle = Z_K/Z_0$$ (unbiasedness), $$\langle \log w \rangle \leq \log(Z_K/Z_0)$$ (the ELBO), and the bound is tight when all importance weights are equal (perfect annealing).

**The variance problem.** This is the same overlap issue encountered in Parts 1 and 2, now in the work variable: rare low-work trajectories carry exponentially large weight, and the effective sample size collapses for fast protocols.

### Crooks' Fluctuation Theorem

The path measure ratio relates the *entire distribution* of work, not only its expectations. For a given work value $$W$$, group all trajectories that produce that value:

> **Crooks' fluctuation theorem (<span id="cite-crooks1999"></span>[Crooks, 1999](#ref-crooks1999)).**
>
> $$\frac{P_F(W)}{P_R(-W)} = e^{\beta(W - \Delta F)}$$
>
> The ratio of the probability of observing work $$W$$ in the forward direction to the probability of observing work $$-W$$ in the reverse direction is exponentially related to how far $$W$$ deviates from $$\Delta F$$.
{: .block-definition }

Crooks is stronger than Jarzynski: it relates the *entire work distribution*, rather than only an exponential average. Jarzynski is recovered by integrating both sides over $$W$$.

<details>
<summary><strong>Deriving Jarzynski from Crooks (click to expand)</strong></summary>

<p>Start from Crooks: \(P_F(W) = e^{\beta(W - \Delta F)} P_R(-W)\). Multiply both sides by \(e^{-\beta W}\) and integrate over all \(W\):</p>

$$\int e^{-\beta W} P_F(W) \, dW = e^{-\beta \Delta F} \int P_R(-W) \, dW = e^{-\beta \Delta F}$$

<p>The left side is \(\langle e^{-\beta W} \rangle_F\), giving Jarzynski's equality. ∎</p>

</details>

**Physical intuition.** For $$W<\Delta F$$, Crooks gives $$P_F(W)<P_R(-W)$$; the two densities are equal only at $$W=\Delta F$$. Their crossing illustrates the free-energy difference. Bennett-style bidirectional estimation instead uses the likelihood-ratio relation across the work samples, rather than locating the intersection of noisy histograms. Shirts et al. develop the maximum-likelihood estimator and its variance properties (<span id="cite-shirts2003"></span>[Shirts et al., 2003](#ref-shirts2003)).

{% include figure.liquid loading="eager" path="assets/img/blog/pm_crooks_intersection.svg" alt="Forward and sign-reversed reverse work densities cross at the free-energy difference." class="img-fluid rounded z-depth-1" zoomable=true caption="Crooks' fluctuation theorem makes the free-energy difference visible as the crossing of \(P_F(W)\) and \(P_R(-W)\). The crossing illustrates the identity; BAR uses bidirectional likelihood information across samples rather than just a histogram intersection." %}

### Dissipation as KL Divergence

From Jarzynski and Jensen, the second law gives $$\langle W \rangle_F \geq \Delta F$$. The gap is the average dissipated work $$\langle W_{\text{diss}} \rangle = \langle W \rangle_F - \Delta F$$. This gap has an information-theoretic interpretation:

> **Dissipated work as KL divergence.**
>
> $$\langle W_{\text{diss}} \rangle = \frac{1}{\beta} D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R)$$
>
> The average work wasted beyond the free energy difference equals (up to temperature) the KL divergence between the forward and reverse path measures. Reversible processes (zero dissipation) have $$\mathcal{P}_F = \mathcal{P}_R$$.
{: .block-definition }

This is the ELBO gap identity from the AIS framework in physics notation. The same information-geometric view also underlies thermodynamic metric and optimal-protocol results (<span id="cite-sivak2012"></span>[Sivak & Crooks, 2012](#ref-sivak2012)):

$$D_{\text{KL}}(\mathcal{P}_F \| \mathcal{P}_R) = \left\langle \ln \frac{\mathcal{P}_F}{\mathcal{P}_R} \right\rangle_F = \left\langle \beta(W - \Delta F) \right\rangle_F = \beta \langle W_{\text{diss}} \rangle$$

For the paired physical protocols above, the path-space KL equals dissipated work times $$\beta$$. Zero KL means equal path measures and constant work $$W=\Delta F$$ almost surely. For a learned generative process, a similar KL measures model mismatch; calling it physical dissipation requires an additional thermodynamic interpretation.

---

## Part 6: Connections to Generative Models

The AIS-Jarzynski connection has been the throughline of this post since the AIS framework. The same path-measure language also helps read diffusion models and GFlowNets. A diffusion model has a forward noising path measure and a learned reverse denoising path measure; its variational loss can be read as a KL between those path measures. A GFlowNet (<span id="cite-bengio2021"></span>[Bengio et al., 2021](#ref-bengio2021)) has a forward construction path measure and a backward deconstruction path measure; trajectory balance (<span id="cite-malkin2022"></span>[Malkin et al., 2022](#ref-malkin2022)) asks their ratio to match the terminal reward up to the partition function.

The diagnostic is the same in all three cases: how different are the forward and reverse path measures? In AIS this appears as loose importance weights, in diffusion models as the gap between the learned reverse process and the true reverse process, and in GFlowNets as variance in the trajectory-balance log-ratio. This shared diagnostic is useful, but physical dissipation applies only when the forward and reverse measures represent the thermodynamic protocols specified above.

---

## Closing

AIS, diffusion models, and GFlowNets can all be analyzed by comparing path measures. The useful questions are concrete: which measure generates the samples, which ratio is being estimated, and what assumptions make that ratio valid?

For AIS, exact initialization and invariant transitions support the normalizer identity. For SDEs, absolute continuity and stochastic-integral conventions matter. Only in the specified thermodynamic setting does the ratio become $$\beta(W-\Delta F)$$. Keeping those conditions visible makes the analogy useful without treating every generative-model loss as literal physical work.

---

## References

- <span id="ref-jarzynski1997"></span>C. Jarzynski, "Nonequilibrium equality for free energy differences," *Physical Review Letters*, 1997. [DOI](https://doi.org/10.1103/PhysRevLett.78.2690). <a href="#cite-jarzynski1997" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-crooks1999"></span>G. E. Crooks, "Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences," *Physical Review E*, 1999. [DOI](https://doi.org/10.1103/PhysRevE.60.2721). <a href="#cite-crooks1999" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-neal2001"></span>R. M. Neal, "Annealed importance sampling," *Statistics and Computing*, 2001. [DOI](https://doi.org/10.1023/A:1008923215028). <a href="#cite-neal2001" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-zwanzig1954"></span>R. W. Zwanzig, "High-temperature equation of state by a perturbation method," *Journal of Chemical Physics*, 1954. [DOI](https://doi.org/10.1063/1.1740409). <a href="#cite-zwanzig1954" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-nelson1967"></span>E. Nelson, *Dynamical Theories of Brownian Motion*, Princeton University Press, 1967. [PDF](https://web.math.princeton.edu/~nelson/books/bmotion.pdf). <a href="#cite-nelson1967" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shirts2003"></span>M. R. Shirts, E. Bair, G. Hooker, and V. S. Pande, "Equilibrium free energies from nonequilibrium measurements using maximum-likelihood methods," *Physical Review Letters*, 2003. [DOI](https://doi.org/10.1103/PhysRevLett.91.140601). <a href="#cite-shirts2003" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-sivak2012"></span>D. A. Sivak and G. E. Crooks, "Thermodynamic metrics and optimal paths," *Physical Review Letters*, 2012. [DOI](https://doi.org/10.1103/PhysRevLett.108.190602). <a href="#cite-sivak2012" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-vargas2024"></span>F. Vargas, S. Padhy, D. Blessing, and N. Nüsken, "Transport meets variational inference: Controlled Monte Carlo Diffusions," *ICLR*, 2024. [ICLR 2024](https://openreview.net/forum?id=PP1rudnxiW). <a href="#cite-vargas2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bengio2021"></span>E. Bengio, M. Jain, M. Korablyov, D. Precup, and Y. Bengio, "Flow network based generative models for non-iterative diverse candidate generation," *NeurIPS*, 2021. [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html). <a href="#cite-bengio2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-malkin2022"></span>N. Malkin, M. Jain, E. Bengio, C. Sun, and Y. Bengio, "Trajectory balance: Improved credit assignment in GFlowNets," *NeurIPS*, 2022. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract-Conference.html). <a href="#cite-malkin2022" class="reversefootnote" role="doc-backlink">↩</a>

---

## Footnotes
