---
layout: post
title: "The Fokker-Planck Equation: How Densities Evolve Under SDEs"
date: 2026-02-04
last_updated: 2026-02-04
description: "Three routes to the Fokker-Planck equation — intuition, heuristic discretization, and rigorous Itô calculus — building from physical pictures to mathematical proof."
order: 2
categories: [generative_model]
tags: [fokker-planck, stochastic-differential-equations, diffusion-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post presents the Fokker-Planck equation in three layers: physical intuition, a heuristic derivation via discretization (following <a href="https://arxiv.org/abs/2510.21890">Lai et al., 2025</a>), and a rigorous derivation via Itô calculus. The first two layers use only multivariate calculus; the third introduces stochastic calculus for readers who want the full proof.</em>
</p>

## Introduction

Diffusion models — DDPM, score-based models, and their ODE counterparts like flow matching — are built on stochastic differential equations (SDEs) and their associated density dynamics. A forward SDE gradually corrupts data into noise; a learned reverse process turns noise back into data. The **Fokker-Planck equation** is the PDE that connects the two sides: given the SDE describing how individual samples move, it tells us how the probability density $$p_t(\mathbf{x})$$ evolves over time. It is the starting point for deriving the probability flow ODE, reverse-time SDEs, and score matching objectives.

Most diffusion model tutorials state the Fokker-Planck equation without proof and move on. Fully understanding where it comes from is surprisingly involved — the rigorous derivation requires Itô calculus, a branch of stochastic analysis that is not part of the standard ML curriculum. This post aims to bridge that gap, building from physical intuition to a complete proof in three layers: (1) a visual explanation of what each term means, (2) a heuristic derivation using only multivariate calculus, and (3) a rigorous derivation via Itô's lemma for readers who want the full argument.

### Roadmap

| Section | Why It's Needed |
|---------|-----------------|
| **Setup: SDEs and Their Transition Kernels** | Define the forward SDE and its Gaussian transition kernel via Euler-Maruyama discretization |
| **Reading the Equation: Drift vs. Diffusion** | State the Fokker-Planck equation and build intuition for each term before proving it |
| **Deriving the Fokker-Planck Equation** | Heuristic derivation: Chapman-Kolmogorov → change of variables → Taylor-Gaussian smoothing → take limits |
| **Rigorous Derivation via Itô Calculus** | The same result, proved rigorously: Itô's lemma → test functions → integration by parts |

---

## Setup: SDEs and Their Transition Kernels

Consider a stochastic process $$\{\mathbf{x}(t)\}_{t \in [0,T]}$$ in $$\mathbb{R}^D$$ governed by the **forward SDE**:

> **Forward SDE.** The process evolves according to
>
> $$d\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t), t)\,dt + g(t)\,d\mathbf{w}(t)$$
>
> where $$\mathbf{f}: \mathbb{R}^D \times [0,T] \to \mathbb{R}^D$$ is the **drift** (a deterministic force pushing the particle), $$g: [0,T] \to \mathbb{R}$$ is the **diffusion coefficient** (the intensity of random noise), and $$\mathbf{w}(t)$$ is a standard $$D$$-dimensional Wiener process (Brownian motion). The initial condition is $$\mathbf{x}(0) \sim p_0$$.
{: .block-definition }

The SDE describes how a single particle moves: at each instant, it drifts by $$\mathbf{f}(\mathbf{x},t)\,dt$$ and receives a random kick of magnitude $$g(t)\,d\mathbf{w}$$. Our goal is to determine the PDE governing the marginal density $$p_t(\mathbf{x})$$ — the probability of finding the particle at position $$\mathbf{x}$$ at time $$t$$.

**Euler-Maruyama discretization.** For a small time step $$\Delta t$$, the SDE is approximated by:

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\,\Delta t + g(t)\sqrt{\Delta t}\;\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This means the **transition kernel** — the conditional distribution of the next state given the current one — is Gaussian:

> **Transition kernel.** The conditional distribution is Gaussian:
>
> $$p(\mathbf{x}_{t+\Delta t} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t+\Delta t};\; \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\,\Delta t,\; g^2(t)\,\Delta t\;\mathbf{I}\right)$$
>
> The mean is the current position shifted by the drift, and the variance is $$g^2(t)\,\Delta t$$ in each coordinate.
{: .block-definition }

This Gaussian kernel is the building block of the derivation.

---

## Reading the Equation: Drift vs. Diffusion

The Fokker-Planck equation for the SDE above is:

> **Fokker-Planck Equation.** The density $$p_t(\mathbf{x})$$ evolves according to
>
> $$\displaystyle\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \bigl[\mathbf{f}(\mathbf{x},t)\,p_t(\mathbf{x})\bigr] + \frac{g^2(t)}{2}\,\Delta_{\mathbf{x}}\,p_t(\mathbf{x})$$
>
> The first term is **advection by drift**; the second is **spreading by diffusion**. Here $$\nabla_{\mathbf{x}} \cdot [\,\cdot\,]$$ is the divergence operator and $$\Delta_{\mathbf{x}} = \sum_{i=1}^{D} \frac{\partial^2}{\partial x_i^2}$$ is the Laplacian.
{: .block-lemma }

We build intuition for each term before deriving the equation in the next section.

### Advection by Drift

The quantity $$\mathbf{f}\,p_t$$ is a **probability flux**: density times velocity. The negative divergence $$-\nabla \cdot (\mathbf{f}\,p_t)$$ measures net inflow. Where flux converges, density accumulates; where it diverges, density depletes. In one dimension, this is the finite-difference statement $$\partial p / \partial t = -(J(x{+}dx) - J(x))/dx$$: density in a slab changes by the net flux through its boundaries. With no noise ($$g = 0$$), this reduces to the continuity equation from fluid dynamics.

{% include figure.liquid loading="eager" path="assets/img/blog/fp_drift_advection.png" class="img-fluid rounded z-depth-1" zoomable=true caption="(a) Probability flux is density times velocity: $J = f \cdot p$. Arrow thickness is proportional to local flux — thicker where density is high. (b) The divergence measures net flux imbalance across the boundaries of an infinitesimal slab: $\partial p / \partial t = -(J(x{+}dx) - J(x))/dx$." %}

### Spreading by Diffusion

At each instant, the SDE's noise kicks every particle by a symmetric random displacement $$g\,d\mathbf{w}$$. The net effect is like Gaussian blurring: peaks erode and valleys fill.

{% include figure.liquid loading="eager" path="assets/img/blog/fp_gaussian_smoothing.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Diffusion smooths a two-bump density. The solid blue curve is the original density $p_t(x)$; the dashed coral curve is the density after Gaussian convolution (kernel shown above the peak). Peaks erode (red shading) and valleys fill (green shading). The derivation below explains why." %}

**Why does blurring produce a second derivative?** In a small time step $$\Delta t$$, the noise kicks each particle from $$y$$ to $$y + \epsilon$$, where $$\epsilon \sim \mathcal{N}(0,\, g^2\,\Delta t)$$. A particle arrives at $$x$$ only if it started at $$y$$ and received kick $$\epsilon = x - y$$. Summing over all starting positions, weighted by the density $$p_t(y)$$ and the probability of the required kick:

$$p_{t+\Delta t}(x) = \int p_t(y)\;\mathcal{N}\!\left(x - y;\;0,\,g^2\,\Delta t\right)dy = \mathbb{E}_\epsilon\!\left[p_t(x - \epsilon)\right]$$

where the second equality substitutes $$\epsilon = x - y$$. Taylor-expanding $$p_t(x - \epsilon)$$ around $$x$$:

$$p_t(x - \epsilon) \approx p_t(x) \;-\; \epsilon\,p_t'(x) \;+\; \frac{\epsilon^2}{2}\,p_t''(x)$$

Now take the expectation term by term:

- **The linear term** $$-\epsilon\,p_t'(x)$$ **vanishes.** A kick $$+\epsilon$$ to the right is exactly as likely as $$-\epsilon$$ to the left, so $$\mathbb{E}[\epsilon] = 0$$ and the slope contributions cancel.

- **The quadratic term** $$\frac{\epsilon^2}{2}\,p_t''(x)$$ **survives.** Whether the particle goes left or right, $$\epsilon^2$$ is positive — the direction cancels but the magnitude does not. This term detects **curvature**: whether neighbors on both sides have higher density than $$x$$ ($$p_t'' > 0$$, valley) or lower ($$p_t'' < 0$$, peak).

{% include figure.liquid loading="eager" path="assets/img/blog/fp_diffusion_schematic.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Why only the second derivative survives. (a) On a slope, a kick $+\epsilon$ raises the density by the same amount that $-\epsilon$ lowers it — the linear (slope) contributions cancel. (b) At a peak, both neighbors have lower density than $x$. The average of neighbors falls below $p_t(x)$, so the quadratic (curvature) term $\frac{\epsilon^2}{2}p_t'' < 0$ drives density down." %}

After taking the expectation, only the curvature term remains: $$p_{t+\Delta t}(x) = p_t(x) + \frac{g^2\,\Delta t}{2}\,p_t''(x)$$. Rearranging:

> **The diffusion PDE (1D).** Pure diffusion in one dimension gives
>
> $$\displaystyle\frac{\partial p_t}{\partial t} = \frac{g^2}{2}\,\frac{\partial^2 p_t}{\partial x^2}$$
>
> In $$D$$ dimensions, the noise components $$dw_1, \ldots, dw_D$$ are independent, so the same argument applies along each coordinate, giving $$\frac{g^2}{2}\,\Delta_{\mathbf{x}}\,p_t$$.
{: .block-lemma }

---

## Deriving the Fokker-Planck Equation

This section gives a **heuristic derivation** based on Euler-Maruyama discretization. It produces the correct equation and builds understanding of where each term comes from, but it sweeps regularity conditions under the rug — we freely exchange limits, integrals, and Taylor expansions without justifying when these operations are valid. The next section provides a rigorous derivation via Itô calculus.

The derivation proceeds from the Gaussian transition kernel in three steps.

### Step 1: Chapman-Kolmogorov

The marginal density at time $$t + \Delta t$$ is obtained by integrating the transition kernel against the current density:

> **Chapman-Kolmogorov equation.** The marginal density at time $$t + \Delta t$$ is
>
> $$p_{t+\Delta t}(\mathbf{x}) = \int \mathcal{N}\!\left(\mathbf{x};\; \mathbf{y} + \mathbf{f}(\mathbf{y},t)\,\Delta t,\; g^2(t)\,\Delta t\;\mathbf{I}\right) p_t(\mathbf{y})\,d\mathbf{y}$$
>
> This is marginalization: summing over all previous positions $$\mathbf{y}$$, weighted by their density $$p_t(\mathbf{y})$$ and the transition probability $$\mathbf{y} \to \mathbf{x}$$. The Markov property justifies using only the one-step kernel.
{: .block-definition }

### Step 2: Change of Variables

The Gaussian kernel is centered at $$\mathbf{y} + \mathbf{f}(\mathbf{y},t)\,\Delta t$$, which depends on $$\mathbf{y}$$ in a complicated way. To simplify, introduce a new integration variable that absorbs the drift:

$$\mathbf{u} := \mathbf{y} + \mathbf{f}(\mathbf{y},t)\,\Delta t$$

Now the Gaussian is centered at $$\mathbf{u}$$: $$\mathcal{N}(\mathbf{x};\;\mathbf{u},\;g^2(t)\,\Delta t\;\mathbf{I})$$. For small $$\Delta t$$, the map $$\mathbf{y} \mapsto \mathbf{u}$$ is invertible. The inverse follows from substituting back into the definition of $$\mathbf{u}$$ and dropping $$\mathcal{O}(\Delta t^2)$$ cross-terms:

$$\mathbf{y} = \mathbf{u} - \mathbf{f}(\mathbf{u},t)\,\Delta t + \mathcal{O}(\Delta t^2)$$

The Jacobian determinant uses the identity $$\det(\mathbf{I} - \mathbf{A}\,\Delta t) = 1 - \text{tr}(\mathbf{A})\,\Delta t + \mathcal{O}(\Delta t^2)$$:

$$\left\lvert\det \frac{\partial \mathbf{y}}{\partial \mathbf{u}}\right\rvert = 1 - (\nabla_{\mathbf{u}} \cdot \mathbf{f})(\mathbf{u},t)\,\Delta t + \mathcal{O}(\Delta t^2)$$

Substituting into the Chapman-Kolmogorov integral and Taylor-expanding $$p_t(\mathbf{y})$$ around $$\mathbf{u}$$:

$$p_t(\mathbf{y}) = p_t(\mathbf{u}) - \Delta t\;\mathbf{f}(\mathbf{u},t) \cdot \nabla_{\mathbf{u}} p_t(\mathbf{u}) + \mathcal{O}(\Delta t^2)$$

Combining with the Jacobian determinant produces two $$\Delta t$$ corrections with distinct origins: one from the Taylor expansion of $$p_t$$ (density varies across space) and one from the Jacobian (the change of variables distorts volume elements):

$$p_{t+\Delta t}(\mathbf{x}) = \int \mathcal{N}\!\left(\mathbf{x};\;\mathbf{u},\;g^2(t)\,\Delta t\;\mathbf{I}\right) \left[p_t(\mathbf{u}) - \Delta t\;\mathbf{f}(\mathbf{u},t) \cdot \nabla_{\mathbf{u}} p_t(\mathbf{u}) - \Delta t\;(\nabla_{\mathbf{u}} \cdot \mathbf{f})(\mathbf{u},t)\;p_t(\mathbf{u})\right] d\mathbf{u} + \mathcal{O}(\Delta t^2)$$

### Step 3: Taylor-Gaussian Smoothing

Each term in the bracket is now convolved against a Gaussian $$\mathcal{N}(\mathbf{x};\;\mathbf{u},\;\sigma^2\mathbf{I})$$ with $$\sigma^2 = g^2(t)\,\Delta t$$. This is the same Taylor expansion we used to build intuition in the previous section, now applied in $$D$$ dimensions.

> **Taylor-Gaussian smoothing.** For any smooth function $$\phi: \mathbb{R}^D \to \mathbb{R}$$ and $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$:
>
> $$\int \mathcal{N}(\mathbf{x};\;\mathbf{u},\;\sigma^2\mathbf{I})\;\phi(\mathbf{u})\,d\mathbf{u} = \mathbb{E}\!\left[\phi(\mathbf{x} + \sigma\mathbf{z})\right] = \phi(\mathbf{x}) + \frac{\sigma^2}{2}\,\Delta_{\mathbf{x}}\phi(\mathbf{x}) + \mathcal{O}(\sigma^4)$$
>
> Gaussian convolution equals the original function plus a Laplacian correction. The first-order term vanishes by symmetry ($$\mathbb{E}[\mathbf{z}] = \mathbf{0}$$); the second-order term survives because $$\mathbb{E}[z_i z_j] = \delta_{ij}$$.
{: .block-lemma }

This follows from Taylor-expanding $$\phi(\mathbf{x} + \sigma\mathbf{z})$$ to second order:

$$\phi(\mathbf{x} + \sigma\mathbf{z}) = \phi(\mathbf{x}) + \sigma\,\nabla_{\mathbf{x}}\phi(\mathbf{x}) \cdot \mathbf{z} + \frac{\sigma^2}{2}\,\mathbf{z}^\top \nabla_{\mathbf{x}}^2 \phi(\mathbf{x})\,\mathbf{z} + \mathcal{O}(\sigma^3)$$

and taking expectations. The components of $$\mathbf{z}$$ are independent standard normals, so $$\mathbb{E}[\mathbf{z}^\top \mathbf{A}\,\mathbf{z}] = \text{tr}(\mathbf{A})$$. The trace of the Hessian is the Laplacian.

Now apply this identity to each term in the bracket. Since $$\sigma^2 = g^2(t)\,\Delta t$$, the Gaussian smoothing of the leading term $$p_t(\mathbf{u})$$ contributes a Laplacian at order $$\Delta t$$:

$$\int \mathcal{N}(\mathbf{x};\;\mathbf{u},\;\sigma^2\mathbf{I})\;p_t(\mathbf{u})\,d\mathbf{u} = p_t(\mathbf{x}) + \frac{g^2(t)\,\Delta t}{2}\,\Delta_{\mathbf{x}}\,p_t(\mathbf{x}) + \mathcal{O}(\Delta t^2)$$

For the two terms already at order $$\Delta t$$ (the drift gradient and divergence terms), the Gaussian smoothing leaves them unchanged at leading order — the $$\sigma^2$$ correction would produce $$\mathcal{O}(\Delta t^2)$$ terms. Collecting everything:

$$p_{t+\Delta t}(\mathbf{x}) - p_t(\mathbf{x}) = -\Delta t\;\mathbf{f}(\mathbf{x},t) \cdot \nabla_{\mathbf{x}} p_t(\mathbf{x}) - \Delta t\;(\nabla_{\mathbf{x}} \cdot \mathbf{f})(\mathbf{x},t)\;p_t(\mathbf{x}) + \frac{g^2(t)}{2}\,\Delta t\;\Delta_{\mathbf{x}}\,p_t(\mathbf{x}) + \mathcal{O}(\Delta t^2)$$

The first two terms combine via the product rule: $$\mathbf{f} \cdot \nabla p + (\nabla \cdot \mathbf{f})\,p = \nabla \cdot (\mathbf{f}\,p)$$. Dividing both sides by $$\Delta t$$ and taking $$\Delta t \to 0$$:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \bigl[\mathbf{f}(\mathbf{x},t)\,p_t(\mathbf{x})\bigr] + \frac{g^2(t)}{2}\,\Delta_{\mathbf{x}}\,p_t(\mathbf{x})$$

This is the Fokker-Planck equation.

---

## Rigorous Derivation via Itô Calculus

The heuristic derivation discretized the SDE, Taylor-expanded, and took limits — producing the right answer but without controlling error terms or justifying the interchange of limits and integrals. This section derives the same equation rigorously using Itô calculus.

### What Is Itô Calculus?

Ordinary calculus assumes smooth paths: differentiation and the chain rule require the function to have well-defined derivatives. Brownian motion $$\mathbf{w}(t)$$ violates this — its sample paths are continuous but nowhere differentiable. Writing $$d\mathbf{w}/dt$$ is meaningless, so expressions like $$d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{w}$$ cannot be interpreted through classical calculus.

**Itô calculus** resolves this by redefining integration rather than differentiation:

> **Itô integral.** The stochastic integral is defined as the limit
>
> $$\displaystyle\int_0^T H(t)\,d\mathbf{w}(t) = \lim_{n \to \infty} \sum_{k=0}^{n-1} H(t_k)\bigl[\mathbf{w}(t_{k+1}) - \mathbf{w}(t_k)\bigr]$$
>
> The integrand $$H(t_k)$$ is evaluated at the **left endpoint** of each subinterval — this ensures the integrand does not "look into the future" (it is **adapted** to the filtration).
{: .block-definition }

This left-endpoint choice has two consequences that drive everything that follows:

1. **The Itô integral is a martingale.** Its expectation is zero: $$\mathbb{E}\!\left[\int_0^T H(t)\,d\mathbf{w}(t)\right] = 0$$. This is because each increment $$\mathbf{w}(t_{k+1}) - \mathbf{w}(t_k)$$ is independent of $$H(t_k)$$ and has zero mean. In the Fokker-Planck derivation below, this property lets us kill the stochastic integral by taking expectations.

2. **Quadratic variation is non-trivial.** For smooth paths, $$(dx)^2$$ is negligible compared to $$dx$$. For Brownian motion, $$(d\mathbf{w})^2 = dt$$ — the increments are of order $$\sqrt{dt}$$, so their squares accumulate at order $$dt$$. This means second-order Taylor terms survive, producing the Itô correction in the chain rule.

These two facts — martingale property and non-vanishing quadratic variation — are the only tools we need. The following example shows both in action.

> **Example: computing $$\int_0^T W_t\,dW_t$$.** Consider the simplest non-trivial Itô integral in one dimension. (The multi-dimensional case follows coordinate-by-coordinate.)
>
> Write the Itô sum and use the algebraic identity $$a(b-a) = \frac{1}{2}(b^2 - a^2) - \frac{1}{2}(b-a)^2$$:
>
> $$\sum_k W_{t_k}(W_{t_{k+1}} - W_{t_k}) = \frac{1}{2}\sum_k \bigl(W_{t_{k+1}}^2 - W_{t_k}^2\bigr) - \frac{1}{2}\sum_k (W_{t_{k+1}} - W_{t_k})^2$$
>
> The first sum telescopes to $$\frac{1}{2}W_T^2 - \frac{1}{2}W_0^2$$. The second sum is the **quadratic variation**: each squared increment $$(W_{t_{k+1}} - W_{t_k})^2$$ has mean $$t_{k+1} - t_k$$, and the variance of the full sum shrinks to zero as the partition refines. The sum converges to $$\frac{1}{2}T$$, giving:
>
> $$\int_0^T W_t\,dW_t = \frac{1}{2}W_T^2 - \frac{1}{2}W_0^2 - \frac{1}{2}T$$
>
> Had we instead evaluated at the **right endpoint** $$W_{t_{k+1}}$$ (the backward Itô convention), the identity $$b(b-a) = \frac{1}{2}(b^2 - a^2) + \frac{1}{2}(b-a)^2$$ gives $$\frac{1}{2}W_T^2 - \frac{1}{2}W_0^2 + \frac{1}{2}T$$. The two answers differ by $$T$$ — the full quadratic variation of Brownian motion over $$[0,T]$$. For smooth paths this difference vanishes and the evaluation point is irrelevant; for Brownian paths it is not.
>
> Rearranging the Itô result reveals a stochastic chain rule:
>
> $$d\!\left[\tfrac{1}{2}W_t^2\right] = W_t\,dW_t + \tfrac{1}{2}\,dt$$
>
> In ordinary calculus, $$d[\frac{1}{2}x^2] = x\,dx$$. The extra $$\frac{1}{2}\,dt$$ is the **Itô correction** — a direct consequence of $$(dW_t)^2 = dt$$.
{: .block-example }

### Itô's Lemma

The example above showed that the ordinary chain rule $$d[\frac{1}{2}x^2] = x\,dx$$ picks up an extra $$\frac{1}{2}\,dt$$ for stochastic processes. Itô's lemma generalizes this to arbitrary smooth functions.

> **Itô's lemma.** For the SDE $$d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{w}$$ and a smooth function $$\varphi(\mathbf{x}, t)$$:
>
> $$d\varphi = \left(\frac{\partial \varphi}{\partial t} + \mathbf{f} \cdot \nabla_{\mathbf{x}} \varphi + \frac{g^2}{2}\,\Delta_{\mathbf{x}}\,\varphi\right)dt + g\,\nabla_{\mathbf{x}} \varphi \cdot d\mathbf{w}$$
>
> The $$\frac{g^2}{2}\,\Delta_{\mathbf{x}}\,\varphi$$ is the **Itô correction** — the second-order Taylor term that survives because $$(d\mathbf{w})^2 = dt$$.
{: .block-lemma }

One can verify the lemma against the example: setting $$\varphi(x) = \frac{1}{2}x^2$$ and $$dx = dW_t$$ (pure Brownian motion, no drift) gives $$d\varphi = x\,dW_t + \frac{1}{2}\,dt$$, matching $$d[\frac{1}{2}W_t^2] = W_t\,dW_t + \frac{1}{2}\,dt$$. This is the stochastic analogue of the second-order Taylor term from the heuristic derivation: the Taylor expansion of $$\mathbb{E}[p_t(\mathbf{x} - \boldsymbol{\epsilon})]$$ produced the same $$\frac{g^2}{2} \cdot \text{(second derivative)}$$ structure. Itô's lemma makes this rigorous by tracking the quadratic variation exactly rather than through a discretize-and-hope argument.

**Derivation.** Start from the second-order Taylor expansion of $$d\varphi = \varphi(\mathbf{x} + d\mathbf{x}, t + dt) - \varphi(\mathbf{x}, t)$$:

$$d\varphi = \frac{\partial \varphi}{\partial t}\,dt + \nabla_{\mathbf{x}}\varphi \cdot d\mathbf{x} + \frac{1}{2}\,d\mathbf{x}^\top \nabla_{\mathbf{x}}^2 \varphi\,d\mathbf{x} + \cdots$$

For smooth paths the quadratic term is $$\mathcal{O}(dt^2)$$ and vanishes. For the SDE $$d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{w}$$, the **stochastic multiplication rules** change this:

> **Stochastic multiplication rules.** $$dw_i \, dw_j = \delta_{ij}\,dt$$, and $$dt\,dt = dt\,dw_i = 0$$. Squared Brownian increments accumulate at rate $$dt$$; all other products are higher-order and vanish.
{: .block-definition }

Substituting $$d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{w}$$ into the quadratic term:

$$d\mathbf{x}^\top \nabla_{\mathbf{x}}^2 \varphi\,d\mathbf{x} = g^2 \sum_{i,j} \frac{\partial^2 \varphi}{\partial x_i \partial x_j}\,dw_i\,dw_j + \mathcal{O}(dt^{3/2}) = g^2 \sum_i \frac{\partial^2 \varphi}{\partial x_i^2}\,dt = g^2\,\Delta_{\mathbf{x}}\varphi\,dt$$

The rule $$dw_i\,dw_j = \delta_{ij}\,dt$$ kills all off-diagonal Hessian entries, leaving only the Laplacian. Collecting the $$dt$$ and $$d\mathbf{w}$$ terms separately yields Itô's lemma as stated above.

### Deriving FP from Itô's Lemma

The strategy is the **test function method**: apply Itô's lemma to a smooth, compactly supported function $$\varphi(\mathbf{x})$$ (no explicit $$t$$-dependence), then transfer derivatives from $$\varphi$$ onto the density $$p_t$$ via integration by parts.

**Step 1: Apply Itô's lemma.** Since $$\varphi$$ does not depend on $$t$$, the $$\partial\varphi/\partial t$$ term vanishes:

$$d\varphi(\mathbf{x}(t)) = \left(\mathbf{f} \cdot \nabla \varphi + \frac{g^2}{2}\,\Delta\varphi\right)dt + g\,\nabla\varphi \cdot d\mathbf{w}$$

**Step 2: Integrate and take expectations.** Integrating from $$t$$ to $$t + \Delta t$$ and taking expectations, the $$d\mathbf{w}$$ integral vanishes — it is a martingale with zero expectation:

$$\mathbb{E}\!\left[\varphi(\mathbf{x}(t+\Delta t))\right] - \mathbb{E}\!\left[\varphi(\mathbf{x}(t))\right] = \int_t^{t+\Delta t} \mathbb{E}\!\left[\mathbf{f} \cdot \nabla\varphi + \frac{g^2}{2}\,\Delta\varphi\right] ds$$

**Step 3: Rewrite expectations as integrals against the density** $$p_s$$:

$$\int \varphi(\mathbf{x})\,p_{t+\Delta t}(\mathbf{x})\,d\mathbf{x} - \int \varphi(\mathbf{x})\,p_t(\mathbf{x})\,d\mathbf{x} = \int_t^{t+\Delta t}\!\!\int \left(\mathbf{f} \cdot \nabla\varphi + \frac{g^2}{2}\,\Delta\varphi\right) p_s(\mathbf{x})\,d\mathbf{x}\,ds$$

**Step 4: Integration by parts.** Move derivatives from the test function $$\varphi$$ onto the density $$p_s$$. Since $$\varphi$$ is compactly supported, the boundary terms vanish:

$$\int \mathbf{f} \cdot \nabla\varphi \;\cdot\; p_s\,d\mathbf{x} = -\int \varphi\;\nabla \cdot (\mathbf{f}\,p_s)\,d\mathbf{x}$$

$$\int \Delta\varphi \;\cdot\; p_s\,d\mathbf{x} = \int \varphi\;\Delta p_s\,d\mathbf{x}$$

The first identity is the divergence theorem (one integration by parts); the second applies integration by parts twice, which returns the Laplacian without a sign change. Substituting back:

$$\int \varphi(\mathbf{x})\bigl[p_{t+\Delta t}(\mathbf{x}) - p_t(\mathbf{x})\bigr]\,d\mathbf{x} = \int_t^{t+\Delta t}\!\!\int \varphi(\mathbf{x})\left[-\nabla \cdot (\mathbf{f}\,p_s) + \frac{g^2}{2}\,\Delta p_s\right]d\mathbf{x}\,ds$$

**Step 5: Extract the PDE.** Dividing by $$\Delta t$$ and sending $$\Delta t \to 0$$, the left side becomes $$\int \varphi\,\partial_t p_t\,d\mathbf{x}$$. Since the equation holds for **all** smooth, compactly supported test functions $$\varphi$$, the integrands must be equal:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla_{\mathbf{x}} \cdot \bigl[\mathbf{f}(\mathbf{x},t)\,p_t(\mathbf{x})\bigr] + \frac{g^2(t)}{2}\,\Delta_{\mathbf{x}}\,p_t(\mathbf{x})$$

This is the Fokker-Planck equation — the same result as the heuristic derivation, now established rigorously. The test function approach works in the **weak (distributional) sense**: it avoids requiring $$p_t$$ to be classically differentiable, needing only that the integrated identity holds for all smooth test functions.

---

## Summary

1. **From particles to densities.** The Fokker-Planck equation converts a stochastic description (individual particles following an SDE) into a deterministic one (the density $$p_t$$ evolving under a PDE).

2. **Two competing effects shape the density.** The divergence of the drift flux advects density (transport without spreading), while the Laplacian term smooths density (spreading without net transport). Their balance determines the density's evolution.

3. **Two derivations, one equation.** The heuristic route (discretize → Chapman-Kolmogorov → Taylor expand → take limits) builds understanding of where each term comes from. The Itô calculus route (Itô's lemma → test functions → integration by parts) makes the argument rigorous without requiring the density to be classically differentiable. Both produce the same PDE.

4. **Connection to diffusion models.** The Fokker-Planck equation characterizes how the data density evolves under the forward SDE. This is the starting point for deriving the probability flow ODE and reverse-time SDE used in score-based generative modeling.

---

## References

- Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. [NeurIPS 2020](https://arxiv.org/abs/2006.11239).
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. [ICLR 2021](https://arxiv.org/abs/2011.13456).
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2023). Flow Matching for Generative Modeling. [ICLR 2023](https://arxiv.org/abs/2210.02747).
- Lai, C.-H., Song, Y., Kim, D., Mitsufuji, Y. & Ermon, S. (2025). The Principles of Diffusion Models. [arXiv:2510.21890](https://arxiv.org/abs/2510.21890).
