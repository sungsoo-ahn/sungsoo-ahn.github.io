---
layout: post
title: "Path Measures for Comparing, Reweighting, and Constraining Stochastic Processes"
date: 2026-08-30
last_updated: 2026-09-06
description: "A discrete-to-continuous derivation of path likelihoods, Girsanov control cost, Doob transforms, and Schrödinger bridges."
post_type: tutorial
editorial_status: ai-generated
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
tags: [incomplete, path-measures, stochastic-processes, girsanov, schrodinger-bridges]
draft: true
sitemap: false
noindex: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This tutorial develops path-law comparison from discrete Markov chains to continuous diffusions, local guidance, and endpoint-constrained dynamics.</em>
</p>

## Introduction

Most descriptions of a diffusion focus on its time-dependent density $$p_t(x)$$. This density tells us where the process can be at one time.

Many questions depend on the complete trajectory $$X_{0:T}$$. Barrier crossings, route choices, accumulated work, and guidance costs all depend on how consecutive states connect.

This distinction became concrete during our work on [transition path sampling with diffusion models](https://openreview.net/forum?id=WQV9kB1qSU). Our target was a distribution over reactive trajectories between two metastable states.

I found the same path-space calculation in diffusion guidance, stochastic control, and rare-event sampling. Clear explanations that connect these uses were difficult to find.

I wrote this post to build the connection from a discrete Markov chain. The discrete calculation will show where each continuous-time formula comes from.

Let $$X_0$$ be uniform on the two states $$\{0,1\}$$. Consider two Markov chains with transition matrices

$$
\begin{aligned}
K^{\mathrm{stay}}
&=
\begin{pmatrix}
0.9 & 0.1\\
0.1 & 0.9
\end{pmatrix},\\[0.6em]
K^{\mathrm{switch}}
&=
\begin{pmatrix}
0.1 & 0.9\\
0.9 & 0.1
\end{pmatrix}.
\end{aligned}
$$

The first chain usually stays in its current state. The second chain usually switches states. Both matrices preserve the uniform distribution, so

$$
P(X_k=0)=P(X_k=1)=\frac12
$$

at every time $$k$$ for both chains. Their time marginals are identical.

Their trajectories are different. For example, the persistent path $$(0,0,0)$$ has probabilities

$$
\begin{aligned}
\frac12(0.9)^2&=0.405,\\
\frac12(0.1)^2&=0.005.
\end{aligned}
$$

respectively. The expected number of switches is also different. Neither quantity follows from the time marginals.

The transition probabilities determine how the time marginals are coupled. A path measure records this missing temporal dependence.

We will compare two discrete path laws by multiplying their transition ratios. The logarithm converts this product into a sum of local costs.

For equal-noise diffusions, the continuous-time limit gives Girsanov's theorem. A terminal preference gives the Doob $$h$$-transform after backward propagation.

Two prescribed endpoint distributions require two coupled factors. Their minimum-KL path law is the Schrödinger bridge.

## From time marginals to path measures

On a time grid, a trajectory is the random vector $$X_{0:N}=(X_0,\ldots,X_N)$$. A time marginal describes one coordinate. A path law describes their joint distribution.

For a Markov chain with initial distribution $$r_0$$ and transition kernels $$K_k$$, the joint law factorizes as

$$
R(x_{0:N})
=r_0(x_0)
\prod_{k=0}^{N-1}K_k(x_{k+1}\mid x_k).
$$

The product records every local transition. A path-dependent functional can use the complete vector. Examples include a switch count, accumulated energy, or first hitting time.

Continuous time replaces the vector with a random function. We do not assign an ordinary probability to one exact continuous path. Such a path usually has probability zero.

We access the law through finite observations. For times $$0\leq t_0<\cdots<t_n\leq T$$ and measurable sets $$A_0,\ldots,A_n$$, a path law assigns the joint probability

$$
P(X_{t_0}\in A_0,\ldots,X_{t_n}\in A_n).
$$

These finite-time events are cylinder events. Their probabilities specify how states at different times are coupled.

> **Continuous-time path measure.** Let $$\Omega=C([0,T],\mathbb R^d)$$ and $$\mathcal F=\mathcal B(\Omega)$$. A path measure $$P$$ is a probability measure on $$(\Omega,\mathcal F)$$. The coordinate map
>
> $$
> X_t(\omega)=\omega(t)
> $$
>
> gives the process state at time $$t$$.
{: .block-definition }

The path measure is the object needed for any quantity that depends on how a process moves. Time marginals alone cannot evaluate such a quantity.

## Comparing path laws

Let $$P$$ and $$R$$ be Markov-chain laws on the same grid. Use $$p_0,Q_k$$ for $$P$$ and $$r_0,K_k$$ for $$R$$. Their path probabilities are

$$
P(x_{0:N})
=p_0(x_0)\prod_{k=0}^{N-1}Q_k(x_{k+1}\mid x_k),
$$

$$
R(x_{0:N})
=r_0(x_0)\prod_{k=0}^{N-1}K_k(x_{k+1}\mid x_k).
$$

Suppose every path with positive probability under $$P$$ also has positive probability under $$R$$. Division gives

$$
\frac{P(x_{0:N})}{R(x_{0:N})}
=
\frac{p_0(x_0)}{r_0(x_0)}
\prod_{k=0}^{N-1}
\frac{Q_k(x_{k+1}\mid x_k)}{K_k(x_{k+1}\mid x_k)}.
$$

Each transition contributes one local likelihood ratio. Define its logarithm by

$$
\ell_k(x,y)
=\log\frac{Q_k(y\mid x)}{K_k(y\mid x)}.
$$

Taking the logarithm of the path ratio turns the product into a sum.

$$
\begin{aligned}
\log\frac{P(x_{0:N})}{R(x_{0:N})}
&=\log\frac{p_0(x_0)}{r_0(x_0)}\\
&\quad+\sum_{k=0}^{N-1}\ell_k(x_k,x_{k+1}).
\end{aligned}
$$

The continuous analogue requires absolute continuity. The notation $$P\ll R$$ means that every event with zero probability under $$R$$ also has zero probability under $$P$$.

> **Path likelihood and path-space KL.** If $$P\ll R$$, the Radon–Nikodym derivative
>
> $$
> L(\omega)=\frac{dP}{dR}(\omega)
> $$
>
> is the path likelihood ratio. For every integrable path functional $$F$$,
>
> $$
> \mathbb E_P[F]
> =
> \mathbb E_R[LF].
> $$
>
> The path-space relative entropy is
>
> $$
> D_{\mathrm{KL}}(P\|R)
> =\mathbb E_P[\log L].
> $$
>
> If $$P\not\ll R$$, then $$D_{\mathrm{KL}}(P\|R)=+\infty$$.
{: .block-definition }

The reweighting identity gives the likelihood ratio its operational role. It converts an expectation under $$P$$ into a weighted expectation under $$R$$.

The conditional expectation of one log ratio is a local transition KL. Define

$$
\begin{aligned}
d_k(x)
&=\sum_yQ_k(y\mid x)\ell_k(x,y)\\
&=D_{\mathrm{KL}}\!\left(
Q_k(\cdot\mid x)\|K_k(\cdot\mid x)
\right).
\end{aligned}
$$

Conditioning on $$X_k=x$$ gives

$$
\mathbb E_P\!\left[
\ell_k(X_k,X_{k+1})\mid X_k=x
\right]
=d_k(x).
$$

The tower property then gives

$$
\mathbb E_P[\ell_k(X_k,X_{k+1})]
=\mathbb E_{X_k\sim P_k}[d_k(X_k)].
$$

Taking the expectation of the complete log ratio gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|R)
&=D_{\mathrm{KL}}(p_0\|r_0)\\
&\quad+\sum_{k=0}^{N-1}
\mathbb E_{X_k\sim P_k}[d_k(X_k)].
\end{aligned}
$$

The outer expectation uses $$P_k$$ because $$P$$ determines which states the changed process visits. The global discrepancy is an expected sum of local transition discrepancies.

## Girsanov's theorem

The Markov-chain decomposition suggests how to compare diffusions. We first compare their short Gaussian transitions and then pass to continuous time.

A realized noise increment favors the changed process when it aligns with the added drift. A quadratic term normalizes this local preference.

Consider the reference and controlled SDEs

$$
\begin{aligned}
dX_t&=f_t(X_t)dt+\sigma\,dW_t,\\
dX_t&=\left[f_t(X_t)+\sigma u_t(X_t)\right]dt+\sigma\,dW_t.
\end{aligned}
$$

For now, let $$\sigma\in\mathbb R^{d\times d}$$ be constant and invertible. Define $$a=\sigma\sigma^\top$$. The vector $$u_t\in\mathbb R^d$$ is a control in noise coordinates. The state-space drift correction is $$\sigma u_t$$.

### Gaussian transition ratios

Euler–Maruyama with step $$\Delta t$$ gives the reference transition

$$
\begin{aligned}
m_k^R(x)&=x+f_k(x)\Delta t,\\
K_k(\cdot\mid x)&=\mathcal N(m_k^R(x),a\Delta t).
\end{aligned}
$$

and the controlled transition

$$
\begin{aligned}
m_k^u(x)
&=x+[f_k(x)+\sigma u_k(x)]\Delta t,\\
Q_k(\cdot\mid x)
&=\mathcal N(m_k^u(x),a\Delta t).
\end{aligned}
$$

The means differ by $$\sigma u_k\Delta t$$. Both covariances equal $$a\Delta t$$.

Under the reference transition, define the normalized innovation

$$
\Delta W_k^R
=\sigma^{-1}
\left[
X_{k+1}-X_k-f_k(X_k)\Delta t
\right].
$$

This increment is Gaussian with covariance $$\Delta t I$$ under $$R$$. Expanding the two Gaussian exponents gives

$$
\begin{aligned}
\ell_k(X_k,X_{k+1})
&=u_k(X_k)^\top\Delta W_k^R\\
&\quad-\frac12\lVert u_k(X_k)\rVert^2\Delta t.
\end{aligned}
$$

The local KL follows from the equal-covariance Gaussian formula.

$$
\delta m_k(x)
=m_k^u(x)-m_k^R(x)
=\sigma u_k(x)\Delta t.
$$

$$
\begin{aligned}
d_k(x)
&=\frac12\delta m_k(x)^\top
(a\Delta t)^{-1}\delta m_k(x)\\
&=\frac12\lVert u_k(x)\rVert^2\Delta t.
\end{aligned}
$$

The mean shift is order $$\Delta t$$. Its squared size is order $$(\Delta t)^2$$. The inverse covariance contributes $$1/\Delta t$$, so one factor $$\Delta t$$ remains.

There are approximately $$T/\Delta t$$ transitions. Summing their order-$$\Delta t$$ costs gives a finite time integral.

Write the state-space drift correction as $$v_k=\sigma u_k$$. Then

$$
\lVert u_k\rVert^2
=v_k^\top a^{-1}v_k.
$$

The covariance sets the geometry of the control cost. A correction costs less in a direction where the reference process has more noise.

### Continuous-time limit

Summing the discrete log ratios suggests

$$
\begin{aligned}
\log\frac{dP^u}{dR}
&=\log\frac{dp_0}{dr_0}(X_0)\\
&\quad+\int_0^T u_t^\top dW_t^R\\
&\quad-\frac12\int_0^T\lVert u_t\rVert^2dt.
\end{aligned}
$$

The stochastic sum becomes an Itô integral. The ordinary sum becomes a time integral. This discretize-and-limit argument is heuristic because it does not control convergence or integrability.

Under the controlled law $$P^u$$, define

$$
W_t^{P^u}
=W_t^R-\int_0^t u_sds.
$$

Girsanov's theorem states conditions under which $$W^{P^u}$$ is Brownian motion. Therefore,

$$
dW_t^R=dW_t^{P^u}+u_tdt.
$$

Substitute this relation into the log density.

$$
\begin{aligned}
\log\frac{dP^u}{dR}
&=\log\frac{dp_0}{dr_0}(X_0)\\
&\quad+\int_0^T u_t^\top dW_t^{P^u}\\
&\quad+\frac12\int_0^T\lVert u_t\rVert^2dt.
\end{aligned}
$$

The stochastic integral has zero expectation under $$P^u$$ when $$u$$ is adapted and square-integrable. Taking the $$P^u$$ expectation gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P^u\|R)
&=D_{\mathrm{KL}}(p_0\|r_0)\\
&\quad+\frac12
\mathbb E_{P^u}\!\left[
\int_0^T\lVert u_t\rVert^2dt
\right].
\end{aligned}
$$

This identity is the continuous counterpart of the conditional KL sum. The factor $$1/2$$ comes from expressing the density with Brownian motion under the controlled law.

> **Girsanov change of drift.** Under $$R$$, suppose
>
> $$
> dX_t=f_t(X_t)dt+\sigma_t(X_t)dW_t^R.
> $$
>
> Let $$u_t$$ be progressively measurable and suppose Novikov's condition holds.
>
> $$
> \mathbb E_R\!\left[
> \exp\!\left(
> \frac12\int_0^T\lVert u_t\rVert^2dt
> \right)
> \right]<\infty.
> $$
>
> For equal initial laws, define
>
> $$
> \begin{aligned}
> \log\frac{dP^u}{dR}
> &=\int_0^T u_t^\top dW_t^R\\
> &\quad-\frac12\int_0^T\lVert u_t\rVert^2dt.
> \end{aligned}
> $$
>
> Then $$W_t^{P^u}=W_t^R-\int_0^t u_sds$$ is Brownian motion under $$P^u$$, and
>
> $$
> \begin{aligned}
> dX_t
> &=\left[f_t(X_t)+\sigma_t(X_t)u_t\right]dt\\
> &\quad+\sigma_t(X_t)dW_t^{P^u}.
> \end{aligned}
> $$
{: .block-definition }

The theorem permits an adapted control that depends on the observed past. A Markov control $$u_t(X_t)$$ is a common special case. Standard statements give additional technical conditions (<span id="cite-oksendal2003"></span>[Øksendal, 2003](#ref-oksendal2003)).

> **Why the diffusion coefficient stays fixed.** A drift change alters the finite-variation part of a path. A diffusion change alters its quadratic variation. In one dimension, $$dX_t=\sigma dW_t$$ satisfies
>
> $$
> [X]_T=\sigma^2T
> $$
>
> almost surely. A process with $$\widetilde\sigma\neq\sigma$$ concentrates on a different quadratic-variation set. The two path laws are usually singular even when every fixed-time Gaussian density overlaps.
{: .block-note }

## Doob h-transform

Girsanov's theorem evaluates a drift correction that is already known. The Doob transform starts from a desired future outcome and finds its local drift correction.

Let $$w(x)\geq 0$$ assign a weight to terminal state $$x$$. A terminal cost can define this weight as $$w(x)=e^{-C_T(x)/\lambda}$$.

The preference is defined at time $$T$$. The process must choose each earlier transition locally. The backward weight connects these two time scales.

The globally reweighted path law has density

$$
\frac{dP^{\mathrm{tilt}}}{dR}
=\frac{w(X_T)}{Z},
\qquad
Z=\mathbb E_R[w(X_T)].
$$

Define the backward weight

$$
h_t(x)=\mathbb E_R[w(X_T)\mid X_t=x].
$$

If $$X_0$$ is random, the global tilt changes its distribution according to

$$
P_0^{\mathrm{tilt}}(dx)
=r_0(dx)\frac{h_0(x)}{Z}.
$$

Sometimes the initial law must remain equal to $$r_0$$. Conditional normalization gives a different path law.

$$
\frac{dP^h}{dR}
=\frac{w(X_T)}{h_0(X_0)}.
$$

The conditional expectation of this ratio given $$X_0$$ equals one. The law $$P^h$$ therefore preserves the reference initial marginal.

### Discrete derivation

Consider the reference Markov chain

$$
R(x_{0:N})
=r_0(x_0)
\prod_{k=0}^{N-1}K_k(x_{k+1}\mid x_k).
$$

Set $$h_N(x)=w(x)$$. The Markov property gives the backward recursion

$$
h_k(x)
=\sum_yK_k(y\mid x)h_{k+1}(y).
$$

The value $$h_k(x)$$ is the expected terminal weight from state $$x$$ at time $$k$$. It propagates the future preference backward.

Define the transformed transition

$$
K_k^h(y\mid x)
=K_k(y\mid x)
\frac{h_{k+1}(y)}{h_k(x)}.
$$

The numerator favors next states with larger expected terminal weight. The denominator normalizes the transition.

$$
\begin{aligned}
\sum_yK_k^h(y\mid x)
&=\frac{1}{h_k(x)}
\sum_yK_k(y\mid x)h_{k+1}(y)\\
&=1.
\end{aligned}
$$

Multiplying the transformed transitions gives

$$
\begin{aligned}
P^h(x_{0:N})
&=r_0(x_0)
\prod_{k=0}^{N-1}K_k^h(x_{k+1}\mid x_k)\\
&=R(x_{0:N})
\prod_{k=0}^{N-1}
\frac{h_{k+1}(x_{k+1})}{h_k(x_k)}\\
&=R(x_{0:N})
\frac{w(x_N)}{h_0(x_0)}.
\end{aligned}
$$

Every intermediate $$h_k$$ appears once in a numerator and once in a denominator. The product telescopes to the conditionally normalized terminal weight.

For terminal conditioning on a set $$B$$, choose $$w(x)=\mathbf 1_B(x)$$. Then $$h_k(x)$$ is the reference probability of reaching $$B$$ from the current state.

### Continuous derivation

Let the reference diffusion be

$$
\begin{aligned}
dX_t&=f_t(X_t)dt+\sigma_t(X_t)dW_t,\\
a_t(x)&=\sigma_t(x)\sigma_t(x)^\top.
\end{aligned}
$$

Its generator acts on a smooth test function $$\phi$$ as

$$
\begin{aligned}
\mathcal L_t\phi(x)
&=f_t(x)^\top\nabla\phi(x)\\
&\quad+\frac12\operatorname{tr}\!\left(
a_t(x)\nabla^2\phi(x)
\right).
\end{aligned}
$$

Under standard regularity conditions, the backward weight satisfies the backward Kolmogorov equation

$$
\partial_t h_t+\mathcal L_t h_t=0,
\qquad
h_T=w.
$$

Let $$K_{t,s}\phi(x)=\mathbb E_R[\phi(X_s)\mid X_t=x]$$. The continuous analogue of the transformed transition rule is

$$
K_{t,t+\Delta t}^h\phi(x)
=\frac{1}{h_t(x)}
K_{t,t+\Delta t}(h_{t+\Delta t}\phi)(x).
$$

Expand the right side to first order in $$\Delta t$$.

$$
\begin{aligned}
K_{t,t+\Delta t}^h\phi(x)
&=\phi(x)\\
&\quad+\frac{\Delta t}{h_t(x)}
\partial_t(h_t\phi)(x)\\
&\quad+\frac{\Delta t}{h_t(x)}
\mathcal L_t(h_t\phi)(x)\\
&\quad+o(\Delta t).
\end{aligned}
$$

The generator product rule is

$$
\mathcal L_t(h_t\phi)
=h_t\mathcal L_t\phi
+\phi\mathcal L_t h_t
+(\nabla h_t)^\top a_t\nabla\phi.
$$

The backward equation cancels $$\phi(\partial_t h_t+\mathcal L_t h_t)$$. The transformed generator is

$$
\mathcal L_t^h\phi
=\mathcal L_t\phi
+\left(a_t\nabla\log h_t\right)^\top\nabla\phi.
$$

The second-order term does not change. The additional first-order term is a state-space drift correction.

$$
b_t^h(x)
=f_t(x)+a_t(x)\nabla\log h_t(x).
$$

> **Doob h-transform for a diffusion.** Suppose $$h_t$$ is positive for $$t<T$$ and sufficiently smooth. The conditionally normalized law $$P^h$$ satisfies
>
> $$
> \begin{aligned}
> dX_t
> &=b_t^h(X_t)dt\\
> &\quad+\sigma_t(X_t)dW_t^h,
> \end{aligned}
> $$
>
> where $$W^h$$ is Brownian motion under $$P^h$$.
{: .block-definition }

The gradient points toward states with a larger relative increase in expected terminal weight. The covariance converts this direction into a drift in state coordinates.

In a diffusion sampler, $$w$$ can represent a terminal likelihood, condition, or energy. The term $$a_t\nabla\log h_t$$ gives exact guidance when $$h_t$$ is known.

For conditional diffusion sampling, $$w$$ can be an observation likelihood. Then $$h_t$$ is the expected observation likelihood from the current noisy state (<span id="cite-denker2024"></span>[Denker et al., 2024](#ref-denker2024)).

### Brownian bridge

A Brownian bridge shows how the backward weight produces time-dependent guidance. Start with

$$
dX_t=\sqrt{\varepsilon}\,dW_t
$$

and condition on $$X_T=y$$. This exact endpoint event has probability zero, so we use its transition density as the backward weight.

For $$t<T$$, the transition density is

$$
h_t(x)
\propto
\exp\!\left(
-\frac{\lVert y-x\rVert^2}{2\varepsilon(T-t)}
\right).
$$

Its Doob drift correction is

$$
\varepsilon\nabla\log h_t(x)
=\frac{y-x}{T-t}.
$$

The conditioned process therefore satisfies

$$
dX_t
=\frac{y-X_t}{T-t}dt
+\sqrt{\varepsilon}\,dW_t^h.
$$

The correction is small when much time remains. It grows near $$T$$ and forces the endpoint toward $$y$$.

The discrete transform repeatedly favors states with a larger remaining chance of reaching $$y$$. The continuous drift applies the same preference at every instant.

## Schrödinger bridges

A terminal weight imposes one preference. Prescribing both endpoint distributions requires two coupled factors.

> **Schrödinger bridge.** Let $$R$$ be a reference path measure. Let $$\mu$$ and $$\nu$$ be prescribed distributions at times $$0$$ and $$T$$. The dynamic Schrödinger problem is
>
> $$
> P^*
> =\arg\min_{\substack{P_0=\mu\\P_T=\nu}}
> D_{\mathrm{KL}}(P\|R).
> $$
>
> Under suitable positivity and integrability conditions, the minimizer is unique and has density
>
> $$
> \frac{dP^*}{dR}
> =\widehat\varphi_0(X_0)\varphi_T(X_T).
> $$
{: .block-definition }

The reference law specifies plausible local dynamics. The endpoint laws specify where paths start and finish (<span id="cite-leonard2014"></span>[Léonard, 2014](#ref-leonard2014)).

The KL objective preserves as much of the reference path law as the endpoint constraints permit.

The expectation in $$D_{\mathrm{KL}}(P\|R)$$ uses $$P$$. The bridge pays for each selected path according to its likelihood under the reference law.

### Matrix scaling

Let the endpoint spaces be finite. Write the reference endpoint probabilities as the positive matrix

$$
M_{ij}=R(X_0=i,X_T=j).
$$

We seek a coupling $$\pi$$ with row sums $$\mu_i$$ and column sums $$\nu_j$$. The closest coupling to $$M$$ solves

$$
\min_{\pi}
\sum_{i,j}\pi_{ij}\log\frac{\pi_{ij}}{M_{ij}}
$$

subject to

$$
\sum_j\pi_{ij}=\mu_i,
\qquad
\sum_i\pi_{ij}=\nu_j.
$$

Introduce multipliers $$\alpha_i$$ and $$\beta_j$$. Stationarity gives

$$
\log\frac{\pi_{ij}}{M_{ij}}
+1-\alpha_i-\beta_j=0.
$$

After constants are absorbed into two positive vectors, the solution has the form

$$
\pi_{ij}^*=a_iM_{ij}b_j.
$$

The row factor enforces the initial marginal. The column factor enforces the terminal marginal. One factor cannot generally satisfy both constraints.

### Conditional reference bridges

The finite matrix suggests an endpoint reweighting of a complete path law. The path-space KL chain rule shows what the optimizer changes.

Write $$P^{x,y}$$ for the conditional path law given $$X_0=x$$ and $$X_T=y$$. Define $$R^{x,y}$$ in the same way. Disintegration gives

$$
P(d\omega)
=P_{0,T}(dx,dy)P^{x,y}(d\omega).
$$

The KL chain rule gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|R)
&=D_{\mathrm{KL}}(P_{0,T}\|R_{0,T})\\
&\quad+
\mathbb E_{(x,y)\sim P_{0,T}}\!\left[
D_{\mathrm{KL}}(P^{x,y}\|R^{x,y})
\right].
\end{aligned}
$$

The endpoint constraints affect only the first term. Any change to a conditional bridge increases the second term without improving feasibility. The optimum therefore satisfies

$$
(P^*)^{x,y}=R^{x,y}.
$$

The optimizer does not change the conditional route law after the endpoints are fixed. It changes how often each endpoint pair is selected.

The dynamic problem reduces to the endpoint coupling problem

$$
P_{0,T}^*
=\arg\min_{\pi\in\Pi(\mu,\nu)}
D_{\mathrm{KL}}(\pi\|R_{0,T}).
$$

The matrix derivation then gives the two endpoint factors. Their normalizing constant is absorbed into either factor.

### Schrödinger potentials

Propagate the terminal factor backward and the initial factor forward under the reference law.

$$
\varphi_t(x)
=\mathbb E_R[\varphi_T(X_T)\mid X_t=x],
$$

$$
\widehat\varphi_t(x)
=\mathbb E_R[\widehat\varphi_0(X_0)\mid X_t=x].
$$

The Markov property makes the past and future conditionally independent given $$X_t$$. Hence

$$
\frac{dP_t^*}{dR_t}(x)
=\widehat\varphi_t(x)\varphi_t(x).
$$

If $$r_t$$ is the reference marginal density, then

$$
p_t^*(x)
=\widehat\varphi_t(x)\varphi_t(x)r_t(x).
$$

The boundary conditions are

$$
\mu(x)
=\widehat\varphi_0(x)\varphi_0(x)r_0(x),
$$

$$
\nu(y)
=\widehat\varphi_T(y)\varphi_T(y)r_T(y).
$$

Multiplying every $$\widehat\varphi$$ by a positive constant and dividing every $$\varphi$$ by that constant leaves the path law unchanged.

For forward dynamics, the past factor cancels after conditioning on the current state. The remaining terminal factor gives a Doob transform with backward weight $$\varphi_t$$.

$$
b_t^*(x)
=f_t(x)+a_t(x)\nabla\log\varphi_t(x).
$$

The forward SDE is

$$
\begin{aligned}
dX_t
&=b_t^*(X_t)dt\\
&\quad+\sigma_t(X_t)dW_t^*.
\end{aligned}
$$

The second endpoint factor selects the initial distribution. The backward terminal factor supplies the forward drift correction.

### Iterative proportional fitting

The two factors are coupled, so one normalization generally breaks the other. Iterative proportional fitting alternates between the two constraints.

Starting from $$P^{(0)}=R$$, the path-space projections are

$$
P^{(2n+1)}
=\arg\min_{P:\,P_T=\nu}
D_{\mathrm{KL}}(P\|P^{(2n)}),
$$

$$
P^{(2n+2)}
=\arg\min_{P:\,P_0=\mu}
D_{\mathrm{KL}}(P\|P^{(2n+1)}).
$$

At the endpoint-coupling level, the terminal projection is

$$
\pi^{(2n+1)}(x,y)
=\pi^{(2n)}(x,y)
\frac{\nu(y)}{\pi_T^{(2n)}(y)}.
$$

Integration over $$x$$ gives the terminal marginal $$\nu$$. The multiplier depends only on $$y$$, so the conditional distribution of $$x$$ given $$y$$ stays fixed.

The next initial projection is

$$
\pi^{(2n+2)}(x,y)
=\pi^{(2n+1)}(x,y)
\frac{\mu(x)}{\pi_0^{(2n+1)}(x)}.
$$

Integration over $$y$$ gives the initial marginal $$\mu$$. This multiplier depends only on $$x$$, so the conditional distribution of $$y$$ given $$x$$ stays fixed.

The second projection usually changes the terminal marginal again. Alternation continues until both endpoint constraints hold at the same time.

Diffusion Schrödinger bridge methods approximate these projections with learned scores or drifts (<span id="cite-debortoli2021"></span>[De Bortoli et al., 2021](#ref-debortoli2021); <span id="cite-vargas2021"></span>[Vargas et al., 2021](#ref-vargas2021)).

For molecular rare events, the reference diffusion supplies plausible local motion. The endpoint laws select the reactant and product ensembles.

## A compact map

| Question | Discrete object | Continuous object |
|---|---|---|
| How do two dynamics differ? | Product of transition ratios | Path density and Girsanov formula |
| What is the cost of changing dynamics? | Sum of conditional transition KL terms | Expected control energy |
| How does a terminal preference change dynamics? | Backward recursion and transformed kernel | Doob drift correction |
| How are both endpoint laws enforced? | Matrix scaling | Schrödinger potentials and IPF |

## Conclusion

The opening chains have equal marginals and different path laws because their transition couplings differ. Products of those transitions produce path likelihood ratios. Their expected log ratios decompose path-space KL into local costs.

For equal-noise diffusions, Gaussian transition ratios converge to the Girsanov density and quadratic control energy. The Doob transform realizes a terminal weight through local drift.

Schrödinger bridges extend the same construction to two endpoint constraints. They select endpoint pairs and retain the reference route law between each pair.

---

## References

- <span id="ref-oksendal2003"></span>Øksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications, 6th ed. [Springer](https://link.springer.com/book/10.1007/978-3-642-14394-6). <a href="#cite-oksendal2003" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-denker2024"></span>Denker, A., Vargas, F., Padhy, S., et al. (2024). DEFT: Efficient Fine-Tuning of Diffusion Models by Learning the Generalised h-transform. [arXiv:2406.01781](https://arxiv.org/abs/2406.01781). <a href="#cite-denker2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-leonard2014"></span>Léonard, C. (2014). A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport. [Discrete and Continuous Dynamical Systems A](https://doi.org/10.3934/dcds.2014.34.1533). <a href="#cite-leonard2014" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-debortoli2021"></span>De Bortoli, V., Thornton, J., Heng, J. & Doucet, A. (2021). Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html). <a href="#cite-debortoli2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-vargas2021"></span>Vargas, F., Thodoroff, P., Lamacraft, A. & Lawrence, N. D. (2021). Solving Schrödinger Bridges via Maximum Likelihood. [arXiv:2106.02081](https://arxiv.org/abs/2106.02081). <a href="#cite-vargas2021" class="reversefootnote" role="doc-backlink">↩</a>
