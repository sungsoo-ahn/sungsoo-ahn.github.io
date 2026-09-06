---
layout: post
title: "Understanding the Adjoint Schrödinger Bridge Sampler"
date: 2026-08-30
last_updated: 2026-09-06
description: "How ASBS combines Schrödinger bridges, stochastic control, adjoint matching, and a learned endpoint corrector for energy-based sampling."
post_type: technical-note
editorial_status: ai-generated
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
tags: [incomplete, energy-based-models, schrodinger-bridges, stochastic-control, diffusion-samplers]
draft: true
sitemap: false
noindex: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This post explains the NeurIPS 2025 Adjoint Schrödinger Bridge Sampler from its path-space objective to its practical matching losses. The intended reader knows diffusion models and need not know stochastic control or Schrödinger bridges. Claims about convergence and scalability remain tied to their proof assumptions.</em>
</p>

## Introduction

I wrote this post because ASBS compresses several difficult ideas into a short algorithmic description. Schrödinger bridges come from stochastic transport, adjoint variables from optimal control, conditional score regression from diffusion modeling, and iterative proportional fitting from information geometry. The method becomes hard to read when those concepts are introduced only at the point where they enter a loss function.

The endpoint corrector carries a source-dependent normalization. An energy gradient identifies favorable terminal states. It does not describe how the source and reference process allocate probability among those states. I found few intuitive explanations that begin with this normalization problem and derive both matching objectives. This post therefore rebuilds ASBS from discrete endpoint algebra before moving to path measures, SDE controls, and neural regressions.

| Key concept | Role in its field | Role in ASBS and diffusion sampling | Discrete counterpart |
|---|---|---|---|
| Energy-based target | Defines a distribution up to an unknown partition function | Supplies terminal energy and its gradient without target samples | Unnormalized weights on a finite state space |
| Schrödinger bridge | Entropic transport closest to reference dynamics | Defines the desired source-to-target diffusion path law | KL projection of a Markov path law with two endpoint constraints |
| Schrödinger potentials | Normalize both endpoint constraints | Produce the forward control and the missing terminal correction | Row and column scaling factors of an endpoint matrix |
| Adjoint Matching | Transports terminal sensitivity backward in control theory | Learns a Markov drift from terminal energy and corrector signals | Backward sensitivity recursion followed by conditional averaging |
| Corrector Matching | Uses conditional-score regression to recover a marginal field | Learns the source-dependent endpoint potential | Posterior averaging under a transition matrix |
| Half bridges and IPF | Alternate KL projections onto marginal constraints | Alternate the learned control and corrector updates | Alternating endpoint normalization |

## The information problem in energy-based diffusion sampling

An energy-based model specifies relative probability rather than normalized probability. Evaluating one state's energy is local. Sampling also requires a global normalizer and global movement of probability mass.

> **Energy-defined sampling problem.** Let $$E\colon\mathbb R^d\to\mathbb R$$ be differentiable and suppose
>
> $$
> 0<Z=\int_{\mathbb R^d}e^{-E(x)}dx<\infty.
> $$
>
> The target probability density is
>
> $$
> \nu(x)=\frac{e^{-E(x)}}{Z}.
> $$
>
> ASBS assumes access to samples from a source law $$\mu$$, evaluations of $$E$$ and $$\nabla E$$, and tractable simulation and transition scores for a reference diffusion. It does not assume access to samples from $$\nu$$ or to $$Z$$. The objective is to learn a controlled path law with initial marginal $$\mu$$ and terminal marginal $$\nu$$.
{: .block-definition }

### Discrete counterpart with unnormalized state weights

On a finite state space, assign each state the weight

$$
\widetilde\nu_i=e^{-E_i}.
$$

The normalized target is

$$
\nu_i
=
\frac{\widetilde\nu_i}{\sum_j\widetilde\nu_j}.
$$

Energy differences determine probability ratios because $$\nu_i/\nu_j=e^{-(E_i-E_j)}$$, even when the denominator is unavailable. A finite-state sampler can query these relative weights during transition construction. In continuous space, the energy gradient supplies the corresponding local direction. Energy values and gradients do not provide target samples.

### Continuous formulation and available information

Suppose the target is known only through an energy.

$$
\nu(x)=\frac{e^{-E(x)}}{Z},
\qquad
Z=\int e^{-E(x)}dx.
$$

We can evaluate $$E(x)$$ and usually $$\nabla E(x)$$. We cannot sample directly from $$\nu$$, and the normalizing constant $$Z$$ is unknown. We do have samples from a source distribution $$\mu$$, such as a Gaussian or a molecular harmonic prior.

The information pattern is precise. Training may use

- independent samples $$X_0\sim\mu$$.
- evaluations of $$E(x)$$ and $$\nabla E(x)$$.
- simulation and transition scores from a chosen Gaussian reference process.
- endpoint pairs generated by the current controlled sampler.

It may not use samples from $$\nu$$ or the value of $$Z$$. The time-dependent control $$u_t(x)$$ is initially unknown. The terminal corrector $$h(x)\approx\nabla\log\hat\varphi_1(x)$$ is also unknown. ASBS represents both with neural networks. Their training targets use only the four available ingredients above.

The Adjoint Schrödinger Bridge Sampler (ASBS) uses stochastic control and Schrödinger-bridge theory to train a diffusion without target samples. The method of <span id="cite-liu2025"></span>[Liu et al., 2025](#ref-liu2025) uses two coupled regressions. Adjoint Matching learns the forward control. Corrector Matching learns the endpoint correction required for a nontrivial source.

The correction follows from the two-potential factorization of a bridge. A bridge with two prescribed marginals has two Schrödinger potentials. The forward control uses one potential. ASBS estimates the terminal gradient of the other.

## Schrödinger bridge as the closest path law with two endpoints

A Schrödinger bridge chooses a probability law over trajectories. The law contains many possible trajectories from the source to the target. The two endpoint distributions are fixed. Path-space KL keeps the dynamics close to a reference process.

> **ASBS Schrödinger bridge problem.** Let $$R$$ be the path law of
>
> $$
> dX_t=f_t(X_t)dt+\sigma_t dW_t,
> \qquad X_0\sim\mu,
> $$
>
> on $$t\in[0,1]$$. The Schrödinger bridge from $$\mu$$ to $$\nu$$ is
>
> $$
> P^*=\arg\min_P
> \left\{
> D_{\mathrm{KL}}(P\|R)
> \ \middle|\ P_0=\mu,\ P_1=\nu
> \right\}.
> $$
>
> If an admissible control generates
>
> $$
> dX_t=[f_t(X_t)+\sigma_tu_t(X_t)]dt+\sigma_t dW_t,
> $$
>
> Girsanov's theorem gives
>
> $$
> D_{\mathrm{KL}}(P^u\|R)
> =\frac12\mathbb E_{P^u}\!\left[
> \int_0^1\lVert u_t(X_t)\rVert^2dt
> \right].
> $$
>
> Thus the bridge is the minimum-energy equal-noise control that satisfies both endpoint constraints, subject to the usual existence and absolute-continuity conditions.
{: .block-definition }

### Discrete counterpart with endpoint-constrained path KL

For a finite Markov chain with reference path law $$R(x_{0:N})$$, the discrete bridge is

$$
P^*
=
\arg\min_{P:\,P_0=\mu,\,P_N=\nu}
D_{\mathrm{KL}}(P\|R).
$$

The endpoint joint matrix has the scaled form

$$
P_{0,N}^*(i,j)=a_iR_{0,N}(i,j)b_j,
$$

where one factor enforces the source marginal and the other enforces the target marginal. Conditional on the endpoint pair $$(i,j)$$, the optimal law keeps the reference distribution over intermediate states. The continuous bridge uses the same decomposition, with matrices replaced by densities and path measures.

### Control convention and Girsanov cost

ASBS writes the reference SDE as

$$
dX_t=f_t(X_t)dt+\sigma_t dW_t,
\qquad X_0\sim\mu,
$$

and the controlled SDE as

$$
dX_t
=
\bigl[f_t(X_t)+\sigma_tu_t(X_t)\bigr]dt
+\sigma_t dW_t,
\qquad X_0\sim\mu,
$$

for $$t\in[0,1]$$. The paper uses a scalar positive noise schedule $$\sigma_t$$. Let $$R$$ be the reference path measure and $$P^u$$ the controlled path measure.

Three quantities must stay separate.

- $$u_t$$ is the control in noise coordinates.
- $$\sigma_tu_t$$ is the actual correction to the SDE drift.
- A potential gradient such as $$\nabla\log\varphi_t$$ becomes a control only after the paper's factor $$\sigma_t$$ is applied.

Because the two SDEs have the same diffusion coefficient, Girsanov's theorem gives

$$
D_{\mathrm{KL}}(P^u\|R)
=
\mathbb E_{P^u}\!\left[
\frac12\int_0^1\lVert u_t(X_t)\rVert^2dt
\right],
$$

assuming the usual absolute-continuity and integrability conditions. The Schrödinger bridge problem is therefore

$$
\min_{P:\,P_0=\mu,\;P_1=\nu}
D_{\mathrm{KL}}(P\|R),
$$

or equivalently, among controls that hit both endpoint marginals,

$$
\min_u
\mathbb E_{P^u}\!\left[
\frac12\int_0^1\lVert u_t(X_t)\rVert^2dt
\right].
$$

The solution changes the reference dynamics as little as possible in path-space KL. This is an exact population statement. A neural implementation only approximates its control and endpoints.

### Two Schrödinger potentials

> **Schrödinger potentials.** For a Markov reference with transition density $$r_{t\mid s}(x\mid y)$$, the positive bridge potentials satisfy
>
> $$
> \varphi_t(x)
> =\int r_{1\mid t}(y\mid x)\varphi_1(y)dy,
> $$
>
> $$
> \hat\varphi_t(x)
> =\int r_{t\mid0}(x\mid y)\hat\varphi_0(y)dy.
> $$
>
> Their boundary products impose the endpoint constraints.
>
> $$
> \varphi_0\hat\varphi_0=\mu,
> \qquad
> \varphi_1\hat\varphi_1=\nu.
> $$
>
> The bridge endpoint density and the optimal forward control are
>
> $$
> p_{0,1}^*(x_0,x_1)
> =\hat\varphi_0(x_0)
> r_{1\mid0}(x_1\mid x_0)
> \varphi_1(x_1),
> $$
>
> $$
> u_t^*(x)=\sigma_t\nabla\log\varphi_t(x).
> $$
>
> The actual state-space drift correction is $$\sigma_t^2\nabla\log\varphi_t$$. The factors are unique up to reciprocal positive scaling.
{: .block-definition }

The endpoint factorization can be checked by integration.

Integrating over $$x_1$$ replaces the last two factors by $$\varphi_0(x_0)$$ and gives $$p_0^*=\hat\varphi_0\varphi_0=\mu$$. Integrating over $$x_0$$ replaces the first two factors by $$\hat\varphi_1(x_1)$$ and gives $$p_1^*=\hat\varphi_1\varphi_1=\nu$$. The two factors are exactly the two endpoint normalizations needed to make one joint law satisfy both constraints.

ASBS follows the convention that absorbs the reference marginal into the forward factor. The bridge marginal is then

$$
p_t^*(x)=\varphi_t(x)\hat\varphi_t(x).
$$

The backward potential $$\varphi_t$$ measures the future terminal weight available from state $$x$$. Its Doob transform determines the optimal forward control.

$$
u_t^*(x)=\sigma_t\nabla\log\varphi_t(x).
$$

The actual drift correction is consequently

$$
\sigma_tu_t^*(x)
=
\sigma_t^2\nabla\log\varphi_t(x).
$$

The forward potential $$\hat\varphi_t$$ propagates corrected source mass toward the current time. Both factors are needed because the bridge has two independent marginal constraints. Established Schrödinger-bridge theory gives this factorization and preservation of reference conditional bridges (<span id="cite-leonard2014"></span>[Léonard, 2014](#ref-leonard2014)).

The same endpoint expression also describes the bridge between its endpoints. Conditional on $$(X_0,X_1)=(x_0,x_1)$$, the endpoint factors are constants and cancel during normalization. The intermediate path therefore follows the reference conditional bridge. ASBS changes the probability of each endpoint pair. It reuses the reference's stochastic interpolation between each pair.

### Derivation of the terminal-cost control problem

> **Endpoint corrector and corrected terminal cost.** The terminal boundary equation $$\varphi_1\hat\varphi_1=\nu$$ implies
>
> $$
> g(x)=-\log\varphi_1(x)
> =\log\frac{\hat\varphi_1(x)}{\nu(x)}
> =E(x)+\log\hat\varphi_1(x)+\log Z.
> $$
>
> Therefore
>
> $$
> \nabla g(x)=\nabla E(x)+\nabla\log\hat\varphi_1(x).
> $$
>
> The bridge control solves the unconstrained terminal-cost problem
>
> $$
> \inf_u\mathbb E_{P^u}\!\left[
> \frac12\int_0^1\lVert u_t(X_t)\rVert^2dt
> +\log\frac{\hat\varphi_1(X_1)}{\nu(X_1)}
> \right].
> $$
>
> ASBS calls $$\nabla\log\hat\varphi_1$$ the endpoint corrector. The unknown constant $$\log Z$$ has no effect on the gradient or optimal control.
{: .block-definition }

To derive this terminal cost, define the value function

$$
V_t(x)=-\log\varphi_t(x).
$$

With zero running state cost and the paper's unit temperature, its Hopf–Cole transform is exactly the backward potential. The terminal cost is

$$
g(x)=V_1(x)=-\log\varphi_1(x).
$$

Use the terminal boundary equation $$\varphi_1\hat\varphi_1=\nu$$.

$$
g(x)
=
\log\frac{\hat\varphi_1(x)}{\nu(x)}.
$$

For $$\nu(x)=e^{-E(x)}/Z$$,

$$
g(x)
=
E(x)+\log\hat\varphi_1(x)+\log Z.
$$

The unknown $$\log Z$$ is additive. It does not affect $$\nabla g$$ or the optimal control, so ASBS never needs to estimate it.

The endpoint correction can be checked by direct cancellation. Since $$g=-\log\varphi_1$$,

$$
e^{-g(x_1)}
=\frac{\nu(x_1)}{\hat\varphi_1(x_1)}
=\varphi_1(x_1).
$$

For a fixed source point $$x_0$$, terminal-cost control exponentially tilts the reference transition by this weight. Its source-dependent normalizer is

$$
\int r_{1\mid0}(x_1\mid x_0)e^{-g(x_1)}dx_1
=\int r_{1\mid0}(x_1\mid x_0)\varphi_1(x_1)dx_1
=\varphi_0(x_0).
$$

Starting from $$X_0\sim\mu$$, the resulting endpoint density is therefore

$$
\begin{aligned}
p_{0,1}^g(x_0,x_1)
&=\mu(x_0)
r_{1\mid0}(x_1\mid x_0)
\frac{\varphi_1(x_1)}{\varphi_0(x_0)}\\
&=\hat\varphi_0(x_0)
r_{1\mid0}(x_1\mid x_0)
\varphi_1(x_1),
\end{aligned}
$$

where the second line uses $$\mu=\hat\varphi_0\varphi_0$$. This is exactly the bridge endpoint joint law. A final integration verifies the desired target.

$$
\begin{aligned}
p_1^g(x_1)
&=\varphi_1(x_1)
\int r_{1\mid0}(x_1\mid x_0)\hat\varphi_0(x_0)dx_0\\
&=\varphi_1(x_1)\hat\varphi_1(x_1)
=\nu(x_1).
\end{aligned}
$$

Theorem 3.1 of the paper states that the bridge control solves

$$
\min_u
\mathbb E_{P^u}\!\left[
\frac12\int_0^1\lVert u_t(X_t)\rVert^2dt
+\log\frac{\hat\varphi_1(X_1)}{\nu(X_1)}
\right].
$$

This turns an endpoint-constrained path optimization into an ordinary terminal-cost control problem when $$\hat\varphi_1$$ is known. The computational problem is now concentrated in its score $$\nabla\log\hat\varphi_1$$. An unrelated reference marginal does not provide a valid replacement. Its source-dependent normalizer would not cancel against $$\mu/\varphi_0$$.

## Endpoint correction beyond the energy gradient

An energy-only terminal weight treats every source state as if it sent mass toward the target in the same way. A finite-time reference process generally remembers where it started. The terminal tilt must therefore be normalized separately for each source state, and those normalizers alter the endpoint mixture.

> **Source-dependent terminal normalization.** Let $$R^{x_0}$$ be the reference path law conditioned on $$X_0=x_0$$. A terminal cost $$g$$ defines the source-preserving tilted law
>
> $$
> P^g(d\omega\mid X_0=x_0)
> =\frac{e^{-g(X_1)}}{Z_g(x_0)}
> R(d\omega\mid X_0=x_0),
> $$
>
> where
>
> $$
> Z_g(x_0)
> =\mathbb E_R[e^{-g(X_1)}\mid X_0=x_0].
> $$
>
> Its terminal density is
>
> $$
> p_1^g(x_1)
> =e^{-g(x_1)}
> \int \mu(x_0)
> \frac{r_{1\mid0}(x_1\mid x_0)}{Z_g(x_0)}dx_0.
> $$
>
> A single marginal weight reaches the desired target only when these conditional normalizers cancel correctly. Endpoint independence is one sufficient case. The Schrödinger corrector supplies the factor needed for a general finite-memory reference.
{: .block-definition }

### Discrete counterpart with source-dependent normalizers

Let $$K_{ij}=R(X_1=j\mid X_0=i)$$ be a finite-state reference transition and let $$\mu_i$$ be the source law. Its terminal marginal is $$r_j=\sum_i\mu_iK_{ij}$$. The naive terminal weight that would correct this marginal in one global step is

$$
w_j=\frac{\nu_j}{r_j}.
$$

Keeping the source state fixed requires the row-specific normalizer

$$
Z_i=\sum_jK_{ij}w_j.
$$

After tilting each row and mixing over the source, the new terminal law is

$$
p_j^w
=
w_j\sum_i\mu_i\frac{K_{ij}}{Z_i}.
$$

If every row of $$K$$ equals $$r$$, then all $$Z_i=1$$ and $$p_j^w=\nu_j$$. For a source-dependent transition matrix, the $$Z_i$$ need not agree, so they remain inside the mixture and the same global weight misses the target.

### Formal continuous derivation

For a generic terminal cost $$g$$, exponential tilting of the reference conditional law gives

$$
P^*(d\omega\mid X_0=x_0)
=
\frac{e^{-g(X_1)}}{Z(x_0)}
R(d\omega\mid X_0=x_0),
$$

where

$$
Z(x_0)
=
\mathbb E_R[e^{-g(X_1)}\mid X_0=x_0].
$$

The initial marginal must remain $$\mu$$, so normalization occurs separately for each $$x_0$$. If $$Z(x_0)$$ varies with the starting state, it changes the terminal mixture after integrating over $$X_0$$.

The ASBS paper calls the reference **memoryless** when its endpoint joint law factorizes.

$$
R_{0,1}(x_0,x_1)=R_0(x_0)R_1(x_1).
$$

This means $$X_0$$ and $$X_1$$ are independent under the reference endpoint law. This property differs from the Markov property. A Markov process can retain strong dependence between its endpoints. The Markov property says that the present screens the future from the past. It does not require exact independence between distant endpoints.

Under endpoint independence, choosing

$$
g(x)=\log\frac{r_1(x)}{\nu(x)}
$$

gives $$e^{-g}=\nu/r_1$$ and

$$
Z(x_0)
=
\int r_1(x_1)\frac{\nu(x_1)}{r_1(x_1)}dx_1
=1.
$$

The tilt reaches $$\nu$$. If the endpoint conditional $$r_{1\mid0}(x_1\mid x_0)$$ depends on $$x_0$$, the same cancellation fails.

The failure can be derived without choosing a finite state table. Let $$r_1$$ be the terminal marginal of the reference process started from $$\mu$$, and use the naive weight

$$
w(x_1)=\frac{\nu(x_1)}{r_1(x_1)}.
$$

Preserving the initial marginal requires a separate normalizer for every source state.

$$
Z(x_0)
=\int r_{1\mid0}(x_1\mid x_0)
\frac{\nu(x_1)}{r_1(x_1)}dx_1.
$$

After tilting each source-conditioned transition and mixing over $$X_0\sim\mu$$, the terminal density becomes

$$
\begin{aligned}
p_1^g(x_1)
&=\int \mu(x_0)
r_{1\mid0}(x_1\mid x_0)
\frac{w(x_1)}{Z(x_0)}dx_0\\
&=\frac{\nu(x_1)}{r_1(x_1)}
\int \mu(x_0)
\frac{r_{1\mid0}(x_1\mid x_0)}{Z(x_0)}dx_0.
\end{aligned}
$$

If the reference forgets its source endpoint, then $$r_{1\mid0}(x_1\mid x_0)=r_1(x_1)$$ and every $$Z(x_0)=1$$. The integral reduces to $$r_1(x_1)$$, which cancels the denominator and leaves $$p_1^g=\nu$$.

For a general finite-horizon reference, $$Z(x_0)$$ varies with the starting state. The factor cannot leave the integral, so the reference marginal $$r_1$$ no longer cancels. The naive energy tilt then produces a different terminal law.

This dependence is natural for molecular sources. A harmonic prior may place different initial conformations near different metastable basins. Over a finite diffusion horizon, those conformations do not have equal access to every low-energy region. The expected terminal weight therefore depends on the initial geometry, and a single global correction $$r_1/\nu$$ cannot normalize all source-conditioned transports.

The contrast is clearest source by source. Take two initial conformations $$x_0$$ and $$x_0'$$. Endpoint independence makes the same terminal weight integrate to one under both conditional transitions. A finite-memory reference instead gives normalizers $$Z(x_0)$$ and $$Z(x_0')$$ that need not agree. Mixing the separately normalized conditional laws then introduces the factors $$1/Z(x_0)$$ and $$1/Z(x_0')$$, changing their relative contribution at the endpoint. That source-dependent reweighting is the bias.

For a Dirac source, $$X_0$$ is constant and the issue disappears trivially. Adjoint Sampling exploited this restricted case with Brownian reference dynamics and terminal correction $$\log(r_1/\nu)$$ (<span id="cite-havens2025"></span>[Havens et al., 2025](#ref-havens2025)). ASBS replaces $$r_1$$ by the bridge's learned $$\hat\varphi_1$$, allowing a general sampleable source instead of enforcing endpoint independence.

## Adjoint Matching for turning terminal sensitivity into feedback

Adjoint Matching starts with a terminal derivative and transports it backward. The terminal energy gradient is observed at $$X_1$$. The sampler needs a control at the current pair $$(t,X_t)$$. The adjoint equation propagates sensitivity along a path. Conditional expectation then converts the pathwise sensitivities into a Markov feedback field.

> **Adjoint Matching population objective.** Let $$h(x)$$ approximate $$\nabla\log\hat\varphi_1(x)$$, and freeze a current sampler $$\bar u$$. Draw
>
> $$
> t\sim\rho,
> \qquad
> (X_0,X_1)\sim P_{0,1}^{\bar u},
> \qquad
> X_t\sim R_{t\mid0,1}(\cdot\mid X_0,X_1).
> $$
>
> For the zero-base-drift ASBS reference, the terminal adjoint is
>
> $$
> a_t=a_1=\nabla E(X_1)+h(X_1).
> $$
>
> The population regression is
>
> $$
> \mathcal L_{\mathrm{AM}}(v;\bar u,h)
> =\mathbb E\!\left[
> \left\lVert
> v_t(X_t)+\sigma_t[\nabla E+h](X_1)
> \right\rVert^2
> \right].
> $$
>
> Its unique $$L^2$$ minimizer is
>
> $$
> v_t^*(x)
> =-\sigma_t\mathbb E\!\left[
> [\nabla E+h](X_1)
> \ \middle|\ t,X_t=x
> \right].
> $$
>
> At an ideal fixed point with the exact corrector, this field is the forward bridge control in noise coordinates.
{: .block-definition }

### Discrete counterpart with backward sensitivity

For deterministic discrete dynamics $$x_{k+1}=F_k(x_k)$$ and terminal cost $$g(x_N)$$, a perturbation obeys

$$
\delta x_{k+1}=\nabla F_k(x_k)\delta x_k.
$$

Define the terminal adjoint $$a_N=\nabla g(x_N)$$. Preserving the first-order pairing $$a_k^\top\delta x_k$$ across one step gives

$$
a_k
=
\nabla F_k(x_k)^\top a_{k+1}.
$$

This is reverse-mode differentiation through the dynamics. With stochastic transitions, many future paths can pass through the same current state. Their backward sensitivities must be averaged conditionally on that state before they define a Markov control.

### Formal adjoint dynamics and regression

Consider first a terminal-cost control problem with

$$
g(x)=E(x)+\log\hat\varphi_1(x)+\text{constant}.
$$

The stochastic maximum principle introduces an adjoint state. Intuitively, $$a_t$$ records how terminal cost changes under a small perturbation of the state at time $$t$$. A perturbation transported by the passive drift satisfies $$d(\delta X_t)=\nabla f_t(X_t)\delta X_tdt$$. Requiring the inner product $$a_t^\top\delta X_t$$ to preserve the terminal sensitivity gives the backward equation used by Adjoint Matching.

$$
-da_t
=
(\nabla f_t(X_t))^\top a_tdt,
\qquad
a_1=\nabla g(X_1).
$$

ASBS specializes its sampler to zero base drift, $$f_t=0$$. The adjoint is then constant along the path.

$$
a_t=a_1
=
\nabla E(X_1)+\nabla\log\hat\varphi_1(X_1).
$$

Adjoint Matching turns the optimality condition into regression. At the population solution,

$$
u_t^*(x)
=
-\sigma_t
\mathbb E_{P^*}[a_t\mid X_t=x].
$$

ASBS therefore minimizes

$$
\mathcal L_{\mathrm{AM}}(u;h)
=
\mathbb E\!\left[
\left\lVert
u_t(X_t)
+\sigma_t\bigl(\nabla E+h\bigr)(X_1)
\right\rVert^2
\right],
$$

where $$h\approx\nabla\log\hat\varphi_1$$. The regression minimizer is the **negative conditional expectation** of the endpoint target after the $$\sigma_t$$ scaling. The network output is $$u_t$$. The SDE adds $$\sigma_tu_t$$ to its drift.

The conditional expectation follows from the geometry of squared loss, not from a guess about the network. Define

$$
Y_t=-\sigma_t(\nabla E+h)(X_1).
$$

For any candidate function $$u_t$$,

$$
\begin{aligned}
\mathbb E[\lVert u_t(X_t)-Y_t\rVert^2]
&=\mathbb E\!\left[
\left\lVert
u_t(X_t)-\mathbb E[Y_t\mid X_t]
\right\rVert^2
\right]\\
&\quad+
\mathbb E\!\left[
\left\lVert
Y_t-\mathbb E[Y_t\mid X_t]
\right\rVert^2
\right].
\end{aligned}
$$

The cross term vanishes after conditioning on $$X_t$$. The second term does not depend on $$u_t$$, so the population minimizer is

$$
u_t^*(x)
=\mathbb E[Y_t\mid X_t=x]
=-\sigma_t
\mathbb E[(\nabla E+h)(X_1)\mid X_t=x].
$$

The target is an endpoint gradient. The learned control is a function of the current state and time. Conditional averaging turns the nonlocal terminal signal into a Markov feedback field.

The sampling law in this regression matters. In one AM update, ASBS samples

$$
t\sim\rho(t),
\qquad
(X_0,X_1)\sim P_{0,1}^{\bar u},
\qquad
X_t\sim R_{t\mid0,1}(\cdot\mid X_0,X_1).
$$

Here $$\rho$$ is the chosen training-time distribution. The current controlled model supplies the endpoint pair. The tractable **reference conditional bridge** supplies the intermediate state. For the zero-drift Gaussian reference, define

$$
\kappa_{t\mid s}=\int_s^t\sigma_r^2dr.
$$

Conditioning the jointly Gaussian increments gives

$$
R_{t\mid0,1}(\cdot\mid x_0,x_1)
=\mathcal N\!\left(
x_0+\frac{\kappa_{t\mid0}}{\kappa_{1\mid0}}(x_1-x_0),
\frac{\kappa_{t\mid0}\kappa_{1\mid t}}{\kappa_{1\mid0}}I
\right).
$$

The mean interpolates according to accumulated variance instead of clock time. The covariance vanishes at both endpoints. The construction can therefore resample intermediate states from stored endpoints. It does not require differentiation through every numerical step of a controlled SDE trajectory.

The endpoint distribution inside the expectation also depends on $$u$$. The notation

$$
\bar u=\operatorname{stopgrad}(u)
$$

means that the current sampler supplies training data. Its sampling law remains frozen during differentiation of the regression loss. This is a fixed-point or self-consistency update. It does not compute a pathwise gradient through data generation. The unique-critical-point result for the ideal Adjoint Matching functional relies on its function-space assumptions (<span id="cite-domingo2025"></span>[Domingo-Enrich et al., 2025](#ref-domingo2025)). It does not say that every finite neural optimization run finds that point.

## Corrector Matching for learning the missing potential

Corrector Matching recovers the source-side Schrödinger potential from endpoint pairs generated by the current sampler. It has the same structure as denoising score matching. A conditional expectation of a tractable transition score represents the unknown marginal field.

> **Corrector Matching population objective.** Suppose
>
> $$
> \hat\varphi_t(x)
> =\int r_{t\mid0}(x\mid x_0)\hat\varphi_0(x_0)dx_0.
> $$
>
> Under the corresponding bridge joint law,
>
> $$
> \nabla\log\hat\varphi_t(x)
> =\mathbb E\!\left[
> \nabla_x\log r_{t\mid0}(x\mid X_0)
> \ \middle|\ X_t=x
> \right].
> $$
>
> Therefore the unique $$L^2$$ minimizer of
>
> $$
> \mathbb E\!\left[
> \left\lVert
> h_t(X_t)-\nabla_{X_t}\log r_{t\mid0}(X_t\mid X_0)
> \right\rVert^2
> \right]
> $$
>
> is $$h_t^*(x)=\nabla\log\hat\varphi_t(x)$$. ASBS uses the terminal case $$t=1$$ and endpoint pairs from its current half-bridge stage.
{: .block-definition }

### Discrete counterpart with posterior averaging

A finite state space has no spatial score, so there is no literal discrete gradient $$\nabla\log\hat\varphi_t$$. The conditioning identity still has a discrete form. If

$$
\hat\varphi_t(j)
=
\sum_i K_{0,t}(j\mid i)\hat\varphi_0(i),
$$

then the source posterior associated with state $$j$$ is

$$
p(i\mid j)
=
\frac{K_{0,t}(j\mid i)\hat\varphi_0(i)}
{\hat\varphi_t(j)}.
$$

Any source-to-state statistic can therefore be converted into a function of the current state by averaging it under this posterior. In continuous space, choosing the statistic to be the transition score and differentiating the potential gives the corrector identity.

### Formal identity from Bayes' rule

The forward potential has the integral form

$$
\hat\varphi_t(x)
=
\int r_{t\mid0}(x\mid x_0)\hat\varphi_0(x_0)dx_0.
$$

At an exact bridge, the joint density of the source endpoint and the state at time $$t$$ is

$$
p_{0,t}^*(x_0,x)
=\hat\varphi_0(x_0)
r_{t\mid0}(x\mid x_0)
\varphi_t(x).
$$

Divide this by $$p_t^*(x)=\hat\varphi_t(x)\varphi_t(x)$$. The backward conditional is

$$
p_{0\mid t}^*(x_0\mid x)
=\frac{
r_{t\mid0}(x\mid x_0)\hat\varphi_0(x_0)
}{\hat\varphi_t(x)}.
$$

This Bayes identity is the missing step between the integral potential and a regression target.

Differentiate its logarithm.

$$
\begin{aligned}
\nabla\log\hat\varphi_t(x)
&=
\frac{1}{\hat\varphi_t(x)}
\int \nabla_x r_{t\mid0}(x\mid x_0)
\hat\varphi_0(x_0)dx_0\\
&=
\int
\nabla_x\log r_{t\mid0}(x\mid x_0)
p_{0\mid t}^*(x_0\mid x)dx_0.
\end{aligned}
$$

Thus

$$
\nabla\log\hat\varphi_t(x)
=
\mathbb E_{P^*}\!\left[
\nabla_x\log r_{t\mid0}(x\mid X_0)
\middle|X_t=x
\right].
$$

Conditional expectation is the minimizer of squared error. The identity becomes the matching objective

$$
\min_{h_t}
\mathbb E_{P^*}\!\left[
\left\lVert
h_t(X_t)-\nabla_{X_t}\log r_{t\mid0}(X_t\mid X_0)
\right\rVert^2
\right].
$$

To verify the minimizer, set

$$
S_t=\nabla_{X_t}\log r_{t\mid0}(X_t\mid X_0).
$$

The same orthogonal decomposition used above gives

$$
\mathbb E[\lVert h_t(X_t)-S_t\rVert^2]
=\mathbb E[\lVert h_t(X_t)-\mathbb E[S_t\mid X_t]\rVert^2]
+\mathbb E[\lVert S_t-\mathbb E[S_t\mid X_t]\rVert^2].
$$

Only the first term depends on $$h_t$$, so $$h_t^*(x)=\mathbb E[S_t\mid X_t=x]=\nabla\log\hat\varphi_t(x)$$. This is the same regression structure as denoising score matching. It regresses a marginal score-like field on a tractable conditional score. It is also the Markovian-projection step used in bridge matching (<span id="cite-shi2023"></span>[Shi et al., 2023](#ref-shi2023)). The conditioning direction is from the current state back to the source endpoint.

ASBS needs only $$t=1$$ because the adjoint terminal condition contains $$\nabla\log\hat\varphi_1(X_1)$$. Its Corrector Matching loss is

$$
\mathcal L_{\mathrm{CM}}(h)
=
\mathbb E_{(X_0,X_1)\sim P^u_{0,1}}\!\left[
\left\lVert
h(X_1)-\nabla_{X_1}\log r_{1\mid0}(X_1\mid X_0)
\right\rVert^2
\right].
$$

The target on the right is available from the reference transition kernel. The endpoint pairs come from the current model, not from $$\nu$$. At an exact half-bridge stage, this regression recovers the potential associated with that stage. During finite training, it is a moving approximation rather than an oracle for the final bridge.

For the zero-drift Gaussian reference used by ASBS, define the accumulated variance

$$
\kappa_{1\mid0}=\int_0^1\sigma_s^2ds.
$$

Then

$$
r_{1\mid0}(x_1\mid x_0)
=\mathcal N(x_1;x_0,\kappa_{1\mid0}I),
$$

and its score is available in closed form.

$$
\nabla_{x_1}\log r_{1\mid0}(x_1\mid x_0)
=-\frac{x_1-x_0}{\kappa_{1\mid0}}.
$$

Corrector Matching regresses this source-to-endpoint displacement on $$X_1$$. One pair supplies a noisy regression target. The population minimizer averages these targets over the posterior distribution of source points.

## Half bridges and alternating endpoint constraints

Neither matching problem solves the full bridge alone. Adjoint Matching enforces the source side and preserves one conditional direction. Corrector Matching enforces the target side and preserves the opposite conditional direction. Iterative proportional fitting alternates these two half bridges until both endpoint constraints agree.

> **Forward and backward half-bridge projections.** Given a backward law $$Q^{(k-1)}$$, the forward half bridge is
>
> $$
> P^{(k)}
> =\arg\min_{P:\,P_0=\mu}
> D_{\mathrm{KL}}(P\|Q^{(k-1)}).
> $$
>
> Given this forward law, the backward half bridge is
>
> $$
> Q^{(k)}
> =\arg\min_{Q:\,Q_1=\nu}
> D_{\mathrm{KL}}(P^{(k)}\|Q).
> $$
>
> The first projection replaces the source marginal and preserves the forward conditional law given $$X_0$$. The second replaces the terminal marginal and preserves the backward conditional law given $$X_1$$. Their alternating limit is the Schrödinger bridge under the standard IPF assumptions.
{: .block-definition }

### Discrete counterpart with alternating endpoint normalization

For an endpoint matrix $$\pi_{ij}$$, a target projection rescales each column.

$$
\pi_{ij}^{\mathrm{target}}
=
\pi_{ij}\frac{\nu_j}{\sum_{i'}\pi_{i'j}}.
$$

This sets the column sum to $$\nu_j$$ and preserves the source conditional given the endpoint. A source projection rescales each row.

$$
\pi_{ij}^{\mathrm{source}}
=
\pi_{ij}\frac{\mu_i}{\sum_{j'}\pi_{ij'}}.
$$

It sets the row sum to $$\mu_i$$ and preserves the endpoint conditional given the source. One projection usually disturbs the constraint imposed by the other, which is why the two normalizations must alternate.

### Practical ASBS cycle

The control needs the corrector. The corrector needs endpoint pairs from the controlled process. ASBS initializes $$h^{(0)}=0$$. Stage $$k$$ then performs the following cycle.

1. Freeze $$h^{(k-1)}$$ and train $$u^{(k)}$$ with Adjoint Matching. Endpoint pairs are supplied by the current stopped-gradient control during this fixed-point optimization.
2. Roll out the updated controlled SDE from fresh $$X_0\sim\mu$$ to obtain $$(X_0,X_1)\sim P_{0,1}^{u^{(k)}}$$.
3. Freeze $$u^{(k)}$$ and train $$h^{(k)}$$ with Corrector Matching on those endpoint pairs and the analytic reference transition score.
4. Use $$h^{(k)}$$ in the terminal adjoint target of the next AM stage.

Because $$h^{(0)}=0$$, the first AM stage uses only $$\nabla E(X_1)$$ in its terminal signal. The first CM stage then estimates how that energy-only transport distorted the source-conditioned endpoint normalizers. Later stages feed that correction back into the control.

### Population meaning of the practical cycle

The paper's Theorems 4.1 and 4.2 identify the ideal population updates with the two projections in the definition block. Adjoint Matching realizes the forward half bridge. Corrector Matching realizes the backward half bridge. The two KL orientations differ because each update preserves a different conditional law.

### Derivation of the preserved conditional laws

The orientations follow from which conditional law a half bridge must preserve. Disintegrate the first KL at the source endpoint.

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q^{(k-1)})
&=D_{\mathrm{KL}}(P_0\|Q_0^{(k-1)})\\
&\quad+
\mathbb E_{X_0\sim P_0}
\left[
D_{\mathrm{KL}}(
P(\cdot\mid X_0)
\|Q^{(k-1)}(\cdot\mid X_0))
\right].
\end{aligned}
$$

Once $$P_0=\mu$$ is imposed, the first term is fixed. The conditional KL is minimized by copying the previous law conditional on $$X_0$$. Thus the forward half bridge replaces the source marginal. It preserves the forward conditional path law.

For the second projection, disintegrate at the terminal endpoint.

$$
\begin{aligned}
D_{\mathrm{KL}}(P^{(k)}\|Q)
&=D_{\mathrm{KL}}(P_1^{(k)}\|Q_1)\\
&\quad+
\mathbb E_{X_1\sim P_1^{(k)}}
\left[
D_{\mathrm{KL}}(
P^{(k)}(\cdot\mid X_1)
\|Q(\cdot\mid X_1))
\right].
\end{aligned}
$$

Now $$Q_1=\nu$$ makes the first term fixed with respect to the remaining choice of $$Q$$. Minimization copies $$P^{(k)}(\cdot\mid X_1)$$ and replaces only the terminal marginal. That operation is naturally written with the variable law in the second KL argument. The two stages therefore preserve opposite conditional directions, which is why their formulas cannot be made identical by swapping labels.

One ideal iteration therefore has two conditional-preserving operations. Adjoint Matching starts from $$Q^{(k-1)}$$, imposes the source marginal $$\mu$$, and preserves the law of the future path conditional on its source. Corrector Matching starts from the resulting $$P^{(k)}$$, imposes the terminal marginal $$\nu$$, and preserves the law of the past path conditional on its endpoint. The neural control and corrector regressions approximate these two path-law projections, respectively.

Alternating these endpoint projections is iterative proportional fitting (IPF). Classical Schrödinger-bridge theory and modern diffusion IPF analyses establish convergence under positivity, absolute-continuity, and integrability conditions (<span id="cite-debortoli2021"></span>[De Bortoli et al., 2021](#ref-debortoli2021)). ASBS supplies matching objectives that realize the two projections without direct target samples.

Theorem 3.2 states convergence to the bridge **provided all matching stages achieve their critical points**. This is an exact-stage, infinite-alternation result. It does not cover incomplete optimization, finite networks, finite minibatches, discretized SDEs, or stale replay data as an unconditional guarantee.

## Scope and contributions of ASBS

### What belongs to ASBS

The conceptual stack has three layers.

**Established path-space theory.** Schrödinger bridges minimize $$D_{\mathrm{KL}}(P\|R)$$ with two endpoint constraints. They use two potentials, preserve the reference bridge conditional on endpoints, and can be computed by alternating endpoint projections.

**Established control theory.** Girsanov equates equal-noise path KL with quadratic control energy. Hopf–Cole maps a matched HJB value to positive desirability. The Doob transform converts that global desirability into the local drift $$\sigma_t^2\nabla\log\varphi_t$$.

**Prior matching machinery.** Adjoint Matching provides a regression whose population critical point is an optimal terminal-cost control under its assumptions. Adjoint Sampling applies it to an energy-defined target with a Dirac source and memoryless Brownian construction.

**The ASBS contribution.** The paper combines a general sampleable source and an energy-defined target in a matching-based Schrödinger-bridge implementation. Its learned corrector replaces the memoryless shortcut. The two matching stages are the half-bridge projections of IPF. The paper and its predecessors support this claim without direct target samples or importance-weighted target estimation in the proposed training objectives.

That is narrower than saying ASBS invented bridge sampling, stochastic control for energies, or IPF. Its contribution is the computational realization that connects them under the energy-only information pattern.

### What the experiments establish

The paper tests multi-well and Lennard–Jones systems, alanine dipeptide, and amortized molecular conformer generation. These settings exercise the claimed benefit of a nontrivial source. In the conformer experiments, a harmonic prior can encode molecular connectivity before diffusion starts. A Dirac source cannot encode this structure.

Across the reported benchmarks, ASBS improves several distributional and structural metrics over the included diffusion-sampler baselines. The alanine-dipeptide results also show the boundary of those metrics. The method matches the dominant Ramachandran regions and misses some low-density modes. The additional 40-mode Gaussian experiment obtains broader coverage on another multimodal target. It does not supply a general mode-coverage guarantee.

The empirical conclusion is therefore conditional. Learned source geometry and the corrector can improve finite-horizon transport in the studied systems. The results do not identify how training cost or mode coverage scales asymptotically with dimension, barrier height, or the number of separated modes.

### Limitations that matter in use

**The energy must be differentiable.** Adjoint Matching uses $$\nabla E(X_1)$$. Nondifferentiable simulators or black-box energies require an estimator or a different objective, potentially with substantial variance.

**The reference must remain computationally usable.** A source may be general and sampleable. Corrector Matching also needs the transition score $$\nabla_{x_1}\log r_{1\mid0}(x_1\mid x_0)$$. The AM construction needs tractable reference conditional bridges. The paper uses zero-drift Gaussian transitions and known noise schedules. An arbitrary source does not imply an arbitrary opaque path simulator.

**Exact projection theory meets approximate training.** The convergence theorem assumes each AM and CM stage reaches its critical point. In practice, error enters at four distinct layers. The function classes may not contain the population control or corrector.

Finite minibatches approximate the two conditional expectations. Optimization may stop away from either regression optimum. Numerical SDE discretization changes the endpoint law that supplies the next stage's data. Gradient clipping and replay add further bias. The theorem does not bound how these errors accumulate across alternations.

**Sampling is on-policy in theory and hybrid in practice.** Fresh endpoint pairs come from the current controlled model. This can be expensive and can reinforce regions already found by the sampler. The implementation stores AM and CM examples in replay buffers. The paper describes this approach as a hybrid of on-policy and off-policy learning. Replay improves data reuse. It also makes the data distribution stale relative to the exact-stage analysis.

**Multimodality remains difficult.** A source and noise schedule that rarely reach a mode provide little regression signal there. The paper reports missed low-density modes in its alanine-dipeptide experiment and calls the behavior mode-seeking. Results on the reported benchmarks support those settings. They do not provide a dimension-independent mixing guarantee.

**A learned sampler is not automatically an unbiased equilibrium method.** After training, one SDE solve gives fast approximate samples. Finite model and discretization error can leave a biased terminal law. There is no Metropolis correction in the basic sampler. The paper derives possible importance weights as future work. Its current gradient parametrization does not directly provide all potential values required to compute them.

A useful source can place mass near relevant geometry. A learned finite-horizon diffusion can amortize expensive exploration. The limitations bound the justified claim. ASBS learns an approximate fast transport toward an equilibrium target. It does not certify independent, unbiased equilibrium samples after a fixed amount of neural training.

## Role of the endpoint corrector

An energy gradient can tell the control which terminal states look favorable. It cannot by itself account for how the source and reference dynamics distribute probability across those states. That missing dependence is $$\nabla\log\hat\varphi_1$$.

ASBS learns it by alternating two conditional-expectation regressions. In the ideal limit, those regressions are the two half-bridge projections of IPF. In practice, both targets are computable from energies, reference transitions, and current model samples. The algorithm does not require a target dataset.

---

## References

- <span id="ref-liu2025"></span>Liu, G.-H., Choi, J., Chen, Y., Miller, B. K. & Chen, R. T. Q. (2025). Adjoint Schrödinger Bridge Sampler. [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/174692c52dc84fad2b2e99dd8637ce6a-Abstract-Conference.html). <a href="#cite-liu2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-leonard2014"></span>Léonard, C. (2014). A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport. [Discrete and Continuous Dynamical Systems A](https://doi.org/10.3934/dcds.2014.34.1533). <a href="#cite-leonard2014" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-havens2025"></span>Havens, A., Miller, B. K., Yan, B., Domingo-Enrich, C., Sriram, A., Wood, B., Levine, D., Hu, B., Amos, B., Karrer, B., Fu, X., Liu, G.-H. & Chen, R. T. Q. (2025). Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching. [arXiv:2504.11713](https://arxiv.org/abs/2504.11713). <a href="#cite-havens2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-domingo2025"></span>Domingo-Enrich, C., Drozdzal, M., Karrer, B. & Chen, R. T. Q. (2025). Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control. [ICLR 2025](https://openreview.net/forum?id=8cftXa1ijb). <a href="#cite-domingo2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-shi2023"></span>Shi, Y., De Bortoli, V., Campbell, A. & Doucet, A. (2023). Diffusion Schrödinger Bridge Matching. [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract-Conference.html). <a href="#cite-shi2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-debortoli2021"></span>De Bortoli, V., Thornton, J., Heng, J. & Doucet, A. (2021). Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html). <a href="#cite-debortoli2021" class="reversefootnote" role="doc-backlink">↩</a>
