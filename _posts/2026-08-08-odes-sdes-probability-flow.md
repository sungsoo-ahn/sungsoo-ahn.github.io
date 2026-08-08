---
layout: post
title: "ODEs, SDEs, and Probability Flow"
date: 2026-08-08
last_updated: 2026-08-08
description: "How deterministic and stochastic dynamics transport probability, why the score appears in reverse-time diffusion, and how a probability-flow ODE shares every marginal with an SDE."
abstract: >
  An ODE and an SDE can trace very different sample paths while producing the same probability density at every time. The bridge is probability flux, and the score converts diffusion into an equivalent deterministic velocity.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
lecture_paths: [ml4mol, gdl]
tags: [ordinary-differential-equations, stochastic-differential-equations, probability-flow, reverse-time-sde, score-models]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the ODE and SDE storyline from my 2025 Machine Learning for Molecules and Geometric Deep Learning lectures. The Fokker–Planck equation is used as the bridge between dynamics and densities; its intuition and full derivation appear in <a href="{% post_url 2026-02-04-fokker-planck-equation %}">The Fokker–Planck Equation</a>.</em>
</p>

Generative models need a way to move probability. A deterministic model can move every sample along a velocity field. A stochastic model can add random Brownian kicks while it moves. These mechanisms produce different trajectories, but trajectories are not the final object of interest. What matters for generation is the probability distribution reached at each time.

The distinction between trajectories and distributions leads to a useful equivalence. A diffusion process has a deterministic **probability-flow ODE** whose marginal density matches the diffusion at every time. The reverse-time diffusion and the reverse probability flow can therefore start from the same noise distribution and arrive at the same data distribution, even though one produces noisy paths and the other produces smooth paths.

The score $$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$ makes this equivalence possible. It measures how the current density varies in space. In the reverse SDE, the score compensates for diffusion while time runs backward. In the probability-flow ODE, half of the same correction converts diffusive spreading into deterministic transport.

## An ODE transports particles through a flow map

Let $$\mathbf{X}_t\in\mathbb{R}^{d}$$ follow the ordinary differential equation

$$
d\mathbf{X}_t
=\mathbf{v}(\mathbf{X}_t,t)\,dt.
$$

The vector field $$\mathbf{v}:\mathbb{R}^{d}\times[0,T]\to\mathbb{R}^{d}$$ assigns a velocity to every position and time. Under standard regularity conditions, each initial point $$\mathbf{X}_0=\mathbf{x}_0$$ determines one trajectory. We write the resulting flow map as

$$
\boldsymbol{\psi}_t(\mathbf{x}_0)=\mathbf{X}_t.
$$

If the initial state is random, $$\mathbf{X}_0\sim p_0$$, the same map pushes the entire distribution forward:

$$
p_t=(\boldsymbol{\psi}_t)_{\#}p_0.
$$

The pushforward notation says that a sample drawn from $$p_0$$ and transformed by $$\boldsymbol{\psi}_t$$ has density $$p_t$$. The ODE operates on particles, while the pushforward describes their ensemble.

{% include figure.liquid loading="eager" path="assets/img/blog/probflow_transport_continuity.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An ODE moves each particle along a velocity field and pushes the full density through the resulting flow map. Probability is conserved, while expansion or contraction of particle spacing changes the local density. Original diagram." %}

A constant velocity gives the simplest example. In one dimension, let

$$
dX_t=c\,dt,
\qquad
X_0\sim p_0.
$$

The solution is $$X_t=X_0+ct$$, so the density is translated without changing shape:

$$
p_t(x)=p_0(x-ct).
$$

The particle equation and density equation contain the same information in different forms. Differentiating the translated density gives

$$
\frac{\partial p_t(x)}{\partial t}
=-c\frac{\partial p_t(x)}{\partial x}
=-\frac{\partial}{\partial x}\bigl(c\,p_t(x)\bigr).
$$

For a general vector field, this becomes the **continuity equation**:

$$
\frac{\partial p_t(\mathbf{x})}{\partial t}
=-\nabla_{\mathbf{x}}\cdot
\bigl[p_t(\mathbf{x})\mathbf{v}(\mathbf{x},t)\bigr].
$$

The product $$p_t\mathbf{v}$$ is probability flux. Its divergence measures net outflow from a small region. Positive divergence depletes density; negative divergence accumulates it.

## Neural ODEs learn the velocity field

A neural ODE parameterizes the vector field with a network $$\mathbf{v}_{\theta}$$ (<span id="cite-chen2018"></span>[Chen et al., 2018](#ref-chen2018)):

$$
d\mathbf{X}_t
=\mathbf{v}_{\theta}(\mathbf{X}_t,t)\,dt.
$$

Numerical integration replaces a fixed sequence of residual blocks with evaluations of a continuous-time vector field. When the ODE is used as a continuous normalizing flow, we also need the density change along each trajectory.

Apply the chain rule to $$\log p_t(\mathbf{X}_t)$$ and substitute the continuity equation:

$$
\begin{aligned}
\frac{d}{dt}\log p_t(\mathbf{X}_t)
&=\frac{\partial}{\partial t}\log p_t(\mathbf{X}_t)
+\mathbf{v}(\mathbf{X}_t,t)^{\mathsf T}
\nabla\log p_t(\mathbf{X}_t)\\
&=-\nabla\cdot\mathbf{v}(\mathbf{X}_t,t).
\end{aligned}
$$

The terms involving $$\mathbf{v}^{\mathsf T}\nabla\log p_t$$ cancel. Density decreases along a trajectory when the local flow expands and increases when it contracts. Integrating this scalar equation gives an exact change-of-variables formula:

$$
\log p_T(\mathbf{X}_T)
=\log p_0(\mathbf{X}_0)
-\int_0^T
\nabla\cdot\mathbf{v}(\mathbf{X}_t,t)\,dt.
$$

This formula is one reason deterministic probability flows are useful: the same ODE solver can transport samples and accumulate likelihoods.

## An SDE adds randomness at order square root of time

A stochastic differential equation adds a Brownian increment:

$$
d\mathbf{X}_t
=\mathbf{f}(\mathbf{X}_t,t)\,dt
+g(t)\,d\mathbf{W}_t.
$$

Here $$\mathbf{f}$$ is the drift, $$g(t)\geq 0$$ is a scalar diffusion coefficient, and $$\mathbf{W}_t$$ is standard $$d$$-dimensional Brownian motion. We restrict attention to isotropic, state-independent diffusion because it exposes the probability-flow identity without extra matrix-divergence terms.

Over a small step $$\Delta t$$, Euler–Maruyama gives

$$
\mathbf{X}_{t+\Delta t}
\approx
\mathbf{X}_t
+\mathbf{f}(\mathbf{X}_t,t)\Delta t
+g(t)\sqrt{\Delta t}\,\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

The deterministic displacement is order $$\Delta t$$, while the random displacement is order $$\sqrt{\Delta t}$$. Brownian paths are continuous but nowhere differentiable, so the SDE does not define an ordinary velocity along each realization.

{% include figure.liquid loading="eager" path="assets/img/blog/probflow_ode_sde_paths.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An ODE assigns one smooth trajectory to each initial state, while an SDE assigns a distribution over irregular trajectories because every time step receives a fresh random increment. Both mechanisms induce a density at each time. Original diagram." %}

The absence of a deterministic flow map does not prevent us from tracking the density. The density follows a partial differential equation rather than an ODE.

## The Fokker–Planck equation exposes probability flux

Let $$p_t(\mathbf{x})$$ be the density of the SDE above. Its Fokker–Planck equation is

$$
\frac{\partial p_t}{\partial t}
=-\nabla\cdot(\mathbf{f}p_t)
+\frac{g^2(t)}{2}\Delta p_t.
$$

The first term transports density with the drift. The second smooths density through diffusion. The separate post on the [Fokker–Planck equation]({% post_url 2026-02-04-fokker-planck-equation %}) develops both terms and derives the PDE from Itô calculus. Here we need one algebraic rewrite.

Define the **score** of the marginal density as

$$
\mathbf{s}_t(\mathbf{x})
=\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
$$

Since $$\nabla p_t=p_t\nabla\log p_t$$, the diffusion term can be written as a divergence:

$$
\frac{g^2}{2}\Delta p_t
=-\nabla\cdot\left[
-\frac{g^2}{2}\mathbf{s}_t p_t
\right].
$$

The full Fokker–Planck equation therefore becomes a continuity equation:

$$
\frac{\partial p_t}{\partial t}
=-\nabla\cdot\left[
p_t\left(
\mathbf{f}-\frac{g^2}{2}\mathbf{s}_t
\right)
\right].
$$

The bracketed quantity is the probability current

$$
\mathbf{J}_t
=p_t\mathbf{f}-\frac{g^2}{2}\nabla p_t,
$$

and the density equation is simply $$\partial_t p_t=-\nabla\cdot\mathbf{J}_t$$. Dividing current by density gives an effective velocity. The equation no longer displays an explicit Laplacian because diffusive spreading has been absorbed into that density-dependent velocity.

## The probability-flow ODE matches the SDE marginals

The continuity form identifies a deterministic vector field:

$$
\mathbf{v}_{\mathrm{pf}}(\mathbf{x},t)
=\mathbf{f}(\mathbf{x},t)
-\frac{g^2(t)}{2}
\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
$$

The corresponding **probability-flow ODE** is

$$
d\mathbf{Z}_t
=\mathbf{v}_{\mathrm{pf}}(\mathbf{Z}_t,t)\,dt.
$$

If $$\mathbf{Z}_0$$ and $$\mathbf{X}_0$$ have the same distribution, the ODE and SDE have the same marginal density at every time (<span id="cite-song2021"></span>[Song et al., 2021](#ref-song2021)). The claim concerns one-time marginals:

$$
\mathbf{Z}_t\overset{d}{=}\mathbf{X}_t
\qquad\text{for each }t.
$$

It does not say that their trajectories, transition kernels, or joint distributions across multiple times agree. The SDE continually injects fresh randomness. The ODE deterministically couples its initial and final samples through one flow map.

The probability-flow vector field is an ensemble-level object. Two initial distributions evolved under the same drift and diffusion generally have different scores, so they induce different probability-flow ODEs. We cannot construct the ODE from $$\mathbf{f}$$ and $$g$$ alone; we also need the current marginal density or its score. Where the density vanishes, the logarithmic form requires care, although the probability current can remain well defined.

The factor $$1/2$$ follows directly from the Fokker–Planck diffusion coefficient. Losing it is a common mistake: the reverse-time SDE uses a full $$g^2\mathbf{s}_t$$ correction, while the probability-flow ODE uses half.

## Reversing a diffusion requires the score

Suppose the forward SDE gradually converts data at time $$0$$ into a simple noise distribution at time $$T$$. Sampling requires dynamics that run from $$T$$ back to $$0$$. Reversing only the drift is not enough because Brownian increments are not differentiable paths that can be played backward.

Define a new increasing clock $$\tau=T-t$$ and the reverse process $$\mathbf{Y}_{\tau}=\mathbf{X}_{T-\tau}$$. Its density is $$q_{\tau}=p_{T-\tau}$$. The reverse-time diffusion has drift (<span id="cite-anderson1982"></span>[Anderson, 1982](#ref-anderson1982))

$$
d\mathbf{Y}_{\tau}
=\left[
-\mathbf{f}(\mathbf{Y}_{\tau},T-\tau)
+g^2(T-\tau)\nabla\log p_{T-\tau}(\mathbf{Y}_{\tau})
\right]d\tau
+g(T-\tau)d\overline{\mathbf{W}}_{\tau}.
$$

The first term reverses the forward drift. The score term points toward regions of larger forward marginal density and compensates for the new Brownian noise added during reverse simulation.

The same formula is often written using the original time variable integrated from $$T$$ down to $$0$$:

$$
d\mathbf{X}_t
=\left[
\mathbf{f}(\mathbf{X}_t,t)
-g^2(t)\nabla\log p_t(\mathbf{X}_t)
\right]dt
+g(t)d\overline{\mathbf{W}}_t,
\qquad dt<0.
$$

The sign convention becomes unambiguous once the integration direction is stated. In the increasing reverse clock $$\tau$$, the score appears with a plus sign. In the decreasing original clock $$t$$, it appears inside $$\mathbf{f}-g^2\mathbf{s}_t$$.

The probability-flow ODE can also be integrated backward. In the increasing reverse clock, its drift is

$$
-\mathbf{v}_{\mathrm{pf}}(\mathbf{y},T-\tau)
=-\mathbf{f}(\mathbf{y},T-\tau)
+\frac{g^2(T-\tau)}{2}
\nabla\log p_{T-\tau}(\mathbf{y}).
$$

{% include figure.liquid loading="eager" path="assets/img/blog/probflow_shared_marginals.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A forward SDE spreads a data density toward noise. The reverse SDE follows noisy trajectories and the reverse probability-flow ODE follows smooth trajectories, yet both recover the same marked marginal densities when supplied with the exact score. Original diagram." %}

The reverse SDE uses twice the score correction of the reverse ODE because it also injects diffusion. Its stronger inward drift must offset that extra spreading. The deterministic ODE needs only the weaker drift that reproduces the same net probability flux.

## A Gaussian example makes the factor of two visible

Consider one-dimensional variance-exploding diffusion with constant noise scale:

$$
dX_t=\sigma\,dW_t,
\qquad
X_0\sim\mathcal{N}(m,s_0^2),
\qquad s_0>0.
$$

Brownian increments are Gaussian, so

$$
p_t(x)
=\mathcal{N}\!\left(x;m,s_0^2+\sigma^2t\right).
$$

The score is

$$
\nabla_x\log p_t(x)
=-\frac{x-m}{s_0^2+\sigma^2t}.
$$

The forward SDE has zero drift. Its probability-flow ODE is therefore

$$
\frac{dZ_t}{dt}
=\frac{\sigma^2}{2(s_0^2+\sigma^2t)}(Z_t-m).
$$

This linear equation has the exact solution

$$
Z_t-m
=\sqrt{\frac{s_0^2+\sigma^2t}{s_0^2}}
(Z_0-m).
$$

The flow leaves the mean fixed and scales every deviation by the ratio of standard deviations. If $$Z_0\sim\mathcal{N}(m,s_0^2)$$, then $$Z_t\sim\mathcal{N}(m,s_0^2+\sigma^2t)$$, exactly matching the SDE.

{% include figure.liquid loading="eager" path="assets/img/blog/probflow_gaussian_example.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Brownian diffusion broadens a Gaussian from variance s0 squared to s0 squared plus sigma squared times t. The probability-flow ODE produces the same Gaussian by deterministically scaling each sample's displacement from the mean. Original diagram." %}

In the increasing reverse clock, the reverse SDE drift is

$$
-\frac{\sigma^2}{s_0^2+\sigma^2(T-\tau)}(Y_{\tau}-m),
$$

while the reverse probability-flow drift is half as large. Both contract the distribution toward $$m$$. The reverse SDE contracts more strongly because its Brownian term continues to spread samples while the clock moves toward the data.

## A neural score turns the identities into models

The exact score is unknown because the intermediate marginal $$p_t$$ is unknown. Score-based generative models replace it with a neural estimate

$$
\mathbf{s}_{\theta}(\mathbf{x},t)
\approx\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
$$

The same score network can be inserted into either reverse dynamic:

$$
\text{reverse SDE:}
\qquad
\mathbf{f}-g^2\mathbf{s}_{\theta},
$$

$$
\text{probability-flow ODE:}
\qquad
\mathbf{f}-\frac{g^2}{2}\mathbf{s}_{\theta},
$$

where both expressions use the original time variable integrated backward. Training the score and choosing the forward noise schedule are the subjects of diffusion modeling rather than stochastic calculus; they are developed in the next chapter of the reading path.

With an approximate score and finite numerical steps, the exact marginal equivalence no longer holds automatically. The two samplers can respond differently to model error and discretization. The ODE gives deterministic samples for a fixed initial noise and supports likelihood computation through its divergence. The SDE keeps stochasticity during generation and admits stochastic corrector steps. These are algorithmic differences layered on top of an exact population-level identity.

The useful conceptual hierarchy is now clear. An ODE specifies particle velocities and induces a continuity equation. An SDE specifies drift and diffusion and induces a Fokker–Planck equation. Rewriting the Fokker–Planck equation as continuity exposes a density-dependent velocity. Time reversal changes that velocity through the score. The resulting reverse SDE and probability-flow ODE are different mechanisms for transporting the same sequence of distributions.

---

## References

<ol class="bibliography">
  <li id="ref-chen2018">Chen, R. T. Q., Rubanova, Y., Bettencourt, J., &amp; Duvenaud, D. (2018). <a href="https://arxiv.org/abs/1806.07366">Neural ordinary differential equations</a>. <em>Advances in Neural Information Processing Systems</em>, 31. <a href="#cite-chen2018">↩</a></li>
  <li id="ref-anderson1982">Anderson, B. D. O. (1982). <a href="https://doi.org/10.1016/0304-4149(82)90051-5">Reverse-time diffusion equation models</a>. <em>Stochastic Processes and their Applications</em>, 12(3), 313–326. <a href="#cite-anderson1982">↩</a></li>
  <li id="ref-song2021">Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., &amp; Poole, B. (2021). <a href="https://arxiv.org/abs/2011.13456">Score-based generative modeling through stochastic differential equations</a>. <em>International Conference on Learning Representations</em>. <a href="#cite-song2021">↩</a></li>
</ol>
