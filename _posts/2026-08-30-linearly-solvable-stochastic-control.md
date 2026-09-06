---
layout: post
title: "Why Some Stochastic Optimal Control Problems Become Linear"
date: 2026-08-30
last_updated: 2026-09-06
description: "A formal-first derivation of Bellman equations, HJB, Hopf–Cole linearization, desirability, and path-integral control."
post_type: tutorial
editorial_status: ai-generated
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
tags: [incomplete, stochastic-control, hjb, hopf-cole, path-integral-control]
draft: true
sitemap: false
noindex: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This tutorial develops linearly solvable stochastic control from finite-state Bellman recursion to continuous-time HJB. It is for machine-learning researchers who know dynamic programming and SDEs. The post identifies when desirability, Feynman–Kac, and controlled diffusion formulas describe the same problem.</em>
</p>

## Introduction

I wrote this post because several ideas that are often taught separately describe one control problem. Bellman recursion belongs to dynamic programming. HJB belongs to continuous optimal control. Hopf–Cole belongs to nonlinear PDEs.

Feynman–Kac belongs to stochastic analysis. The Doob transform belongs to probability. Diffusion-sampler papers use all of them when a reference SDE is guided toward low-cost or high-probability outcomes. Many explanations state the equivalence only after introducing the continuous-time machinery.

The discrete counterpart is much easier to see. A controller changes a Markov transition kernel, pays KL divergence from a passive kernel, and exponentiates the value function. The resulting recursion is linear. I could not find one intuitive account that carried this finite-state calculation to the controlled diffusion. This post builds the chain from scratch.

| Key concept | Role in its field | Role in diffusion samplers | Discrete counterpart |
|---|---|---|---|
| Bellman recursion | Decomposes sequential decisions into one-step problems | Defines the value of changing the sampler from each time and state | Backward cost-to-go recursion |
| KL-regularized control | Penalizes deviation from passive dynamics | Keeps a guided sampler close to a reference diffusion | Choosing a transition distribution with a KL penalty |
| HJB equation | Continuous-time Bellman equation | Describes the optimal drift correction through the value gradient | Small-time limit of one Bellman step |
| Desirability and Hopf–Cole | Linearize a matched nonlinear control PDE | Convert cost-to-go into a positive guidance function | Exponential transform of the soft Bellman value |
| Feynman–Kac formula | Represents a linear PDE by path expectations | Evaluates guidance by averaging exponential costs under the passive SDE | Repeated kernel averaging |
| Doob control and path KL | Realize a tilted path law by local dynamics | Produce the drift of the optimally guided sampler | Reweighted transition kernel and accumulated transition KL |

The Hamilton–Jacobi–Bellman equation is nonlinear because choosing a control changes the future distribution. A useful class of stochastic control problems becomes linear after exponentiating the value function. The derivation below shows where HJB comes from. It also identifies the structure required for the exponential transformation.

Stochastic optimal control is not linear in general. The cancellation needs quadratic control cost, matching control and noise directions, and a fixed diffusion covariance. Removing one ingredient prevents the ordinary Hopf–Cole calculation from closing.

## Bellman recursion as one-step optimization

Bellman's principle says that an optimal future remains optimal after the first transition. The current state contains the information needed for the remaining decisions. Thus a long-horizon problem becomes one decision followed by a shorter instance of the same problem.

> **Finite-horizon Markov control problem.** Let $$Q_k(dx'\mid x,u)$$ be controlled transition kernels, let $$c_k(x,u)$$ be stage costs, and let $$g(x)$$ be a terminal cost. For an admissible Markov policy $$\pi$$, define
>
> $$
> J_k^\pi(x)
> =\mathbb E^\pi\!\left[
> \sum_{j=k}^{N-1}c_j(X_j,U_j)+g(X_N)
> \ \middle|\ X_k=x
> \right].
> $$
>
> The value function is $$V_k(x)=\inf_\pi J_k^\pi(x)$$. Under the usual measurability and integrability conditions, it satisfies
>
> $$
> V_N(x)=g(x),
> $$
>
> $$
> V_k(x)
> =\inf_u\left\{
> c_k(x,u)+\int V_{k+1}(x')Q_k(dx'\mid x,u)
> \right\}.
> $$
{: .block-definition }

### Discrete intuition for the cost-to-go

Consider a controlled Markov chain. At time $$k$$, the process is in state $$x$$, chooses action $$u_k$$, pays cost $$c_k(x,u_k)$$, and samples the next state from

$$
X_{k+1}\sim Q_k(\cdot\mid x,u_k).
$$

Let $$g$$ be the terminal cost. The optimal cost-to-go is

$$
V_k(x)
=
\inf_\pi
\mathbb E^\pi\!\left[
\sum_{j=k}^{N-1}c_j(X_j,u_j)+g(X_N)
\middle|X_k=x
\right],
$$

where $$\pi$$ ranges over policies from time $$k$$ onward.

### Derivation of Bellman recursion by conditioning

Split the sum after the first decision. Once $$u_k$$ and $$X_{k+1}$$ are fixed, the best remaining expected cost is $$V_{k+1}(X_{k+1})$$. The tower property gives

$$
V_k(x)
=
\min_{u_k}
\left\{
c_k(x,u_k)
+
\mathbb E\!\left[
V_{k+1}(X_{k+1})
\mid X_k=x,u_k
\right]
\right\},
$$

with $$V_N=g$$. This is Bellman recursion. It is the repeated use of conditional expectation plus the fact that the current state summarizes the past for future decisions.

The minimization is generally nonlinear even when the state dynamics are linear. The value inside the expectation depends on the controlled next-state distribution, and the minimizing action depends on its gradient or finite-state analogue.

## KL-regularized transition control and desirability

Linearly solvable control begins by changing what the controller chooses. Instead of selecting an action whose effect on the transition law may be complicated, let the action be the next-state distribution itself. KL divergence measures the effort required to move that distribution away from passive dynamics.

> **KL-regularized transition control.** At time $$k$$ and state $$x$$, let $$K_k(\cdot\mid x)$$ be a passive transition law. The controller chooses a law $$Q_k(\cdot\mid x)\ll K_k(\cdot\mid x)$$ and solves
>
> $$
> V_k(x)=q_k(x)+\inf_Q\left\{
> \int V_{k+1}(x')Q(dx')
> +\lambda D_{\mathrm{KL}}(Q\|K_k(\cdot\mid x))
> \right\}.
> $$
>
> For $$\lambda>0$$, the optimizer is
>
> $$
> Q_k^*(dx'\mid x)
> =\frac{K_k(dx'\mid x)e^{-V_{k+1}(x')/\lambda}}
> {\int K_k(dy\mid x)e^{-V_{k+1}(y)/\lambda}}.
> $$
>
> With desirability $$\varphi_k=e^{-V_k/\lambda}$$, the Bellman equation becomes
>
> $$
> \varphi_k(x)
> =e^{-q_k(x)/\lambda}
> \int K_k(dx'\mid x)\varphi_{k+1}(x').
> $$
>
> The terminal condition is $$\varphi_N=e^{-g/\lambda}$$.
{: .block-definition }

### Discrete explanation with transition reweighting

Before taking a diffusion limit, let the control be the next-state distribution itself. At state $$x$$, a passive kernel $$K_k(\cdot\mid x)$$ describes the uncontrolled transition. The controller chooses $$Q_k(\cdot\mid x)$$ and pays

$$
q_k(x)
+
\lambda D_{\mathrm{KL}}\!\left(
Q_k(\cdot\mid x)\|K_k(\cdot\mid x)
\right).
$$

Bellman recursion becomes

$$
\begin{aligned}
V_k(x)
=q_k(x)+\min_{Q_k}
\Bigl\{&
\mathbb E_{X'\sim Q_k(\cdot\mid x)}[V_{k+1}(X')]\\
&+\lambda D_{\mathrm{KL}}(Q_k(\cdot\mid x)\|K_k(\cdot\mid x))
\Bigr\}.
\end{aligned}
$$

### Derivation of the linear recursion by Gibbs reweighting

The finite Gibbs identity is worth deriving because it contains the whole construction. Fix $$x$$, abbreviate the successor states by $$i$$, and add a multiplier $$\eta$$ for $$\sum_iQ_i=1$$. The part of the Bellman objective that depends on $$Q$$ is

$$
\mathcal J(Q,\eta)
=\sum_iQ_iV_{k+1}(i)
+\lambda\sum_iQ_i\log\frac{Q_i}{K_i}
+\eta\left(\sum_iQ_i-1\right).
$$

At a successor with $$K_i>0$$, stationarity gives

$$
0=\frac{\partial\mathcal J}{\partial Q_i}
=V_{k+1}(i)
+\lambda\left(\log\frac{Q_i}{K_i}+1\right)
+\eta.
$$

Solving for $$Q_i$$ produces a common normalization constant.

$$
Q_i
=K_i e^{-V_{k+1}(i)/\lambda}
e^{-1-\eta/\lambda}.
$$

The constraint $$\sum_iQ_i=1$$ determines that constant. Restoring the state variables, the optimum is

$$
Q_k^*(x'\mid x)
=
\frac{
K_k(x'\mid x)e^{-V_{k+1}(x')/\lambda}
}{
\sum_yK_k(y\mid x)e^{-V_{k+1}(y)/\lambda}
}.
$$

The passive probability is multiplied by future desirability. Define

$$
\varphi_k(x)=e^{-V_k(x)/\lambda}.
$$

Substitution of $$Q_k^*$$ into the one-step objective gives the soft minimum

$$
\min_Q\left\{
\mathbb E_Q[V_{k+1}]
+\lambda D_{\mathrm{KL}}(Q\|K_k)
\right\}
=-\lambda\log\sum_{x'}K_k(x'\mid x)e^{-V_{k+1}(x')/\lambda}.
$$

Therefore Bellman recursion reads

$$
V_k(x)
=q_k(x)
-\lambda\log\sum_{x'}K_k(x'\mid x)e^{-V_{k+1}(x')/\lambda}.
$$

Exponentiating both sides gives

$$
\varphi_k(x)
=
e^{-q_k(x)/\lambda}
\sum_{x'}K_k(x'\mid x)\varphi_{k+1}(x').
$$

The recursion is linear in $$\varphi_{k+1}$$ apart from the known multiplication by the running-cost weight. The optimal kernel can be written as

$$
Q_k^*(x'\mid x)
=
K_k(x'\mid x)
\frac{\varphi_{k+1}(x')}
{\sum_yK_k(y\mid x)\varphi_{k+1}(y)}.
$$

For zero running cost, the denominator is $$\varphi_k(x)$$ and this is a Doob $$h$$-transform. The passive kernel determines the support. If $$K_k(x'\mid x)=0$$, no finite KL payment can make the transition available. Todorov used this structure to define linearly solvable Markov decision processes (<span id="cite-todorov2006"></span>[Todorov, 2006](#ref-todorov2006)).

The ratio between any two successor probabilities shows how cost and passive dynamics interact. For successors $$x_A$$ and $$x_B$$,

$$
\frac{Q_k^*(x_A\mid x)}{Q_k^*(x_B\mid x)}
=
\frac{K_k(x_A\mid x)}{K_k(x_B\mid x)}
\exp\!\left(
-\frac{V_{k+1}(x_A)-V_{k+1}(x_B)}{\lambda}
\right).
$$

The first factor is the passive odds. The exponential factor changes those odds according to future cost. A lower-cost successor receives more probability. A transition forbidden by $$K_k$$ remains forbidden. Control reweights the passive dynamics. It does not create new support.

The diffusion problem below is the small-step version of this kernel control. Two Gaussian Euler transitions with the same covariance $$a\Delta t$$ and mean difference $$Bu\Delta t$$ have

$$
D_{\mathrm{KL}}(Q\|K)
=
\frac{\Delta t}{2}u^\top B^\top a^\dagger Bu.
$$

Weighting this KL by $$\lambda$$ reproduces $$\frac12u^\top Mu\Delta t$$ when the control parametrization is compatible with $$a=\lambda BM^{-1}B^\top$$. Thus the discrete KL penalty, continuous quadratic effort, and covariance matching are three views of the same local restriction.

## Hamilton–Jacobi–Bellman equation as the continuous Bellman limit

HJB follows from dynamic programming. Apply Bellman recursion over a short time interval. Expand the next-state value and retain every contribution of order $$\Delta t$$.

> **Continuous-time control problem and HJB equation.** Consider the controlled diffusion
>
> $$
> dX_s=[f_s(X_s)+B_s(X_s)u_s]ds+\sigma_s(X_s)dW_s,
> \qquad a_s=\sigma_s\sigma_s^\top,
> $$
>
> and the value function
>
> $$
> V_t(x)=\inf_u\mathbb E_{t,x}^u\!\left[
> g(X_T)+\int_t^T
> \left(q_s(X_s)+\frac12u_s^\top M_s(X_s)u_s\right)ds
> \right].
> $$
>
> If $$V$$ is a classical solution and $$M_t$$ is positive definite, dynamic programming gives
>
> $$
> \begin{aligned}
> -\partial_tV_t
> =q_t+f_t\cdot\nabla V_t
> +\frac12\operatorname{tr}(a_t\nabla^2V_t)
> +\inf_u\left\{
> \frac12u^\top M_tu+u^\top B_t^\top\nabla V_t
> \right\},
> \end{aligned}
> $$
>
> with terminal condition $$V_T=g$$. The minimizing feedback is $$u_t^*=-M_t^{-1}B_t^\top\nabla V_t$$. When classical derivatives do not exist, the equation is understood in the viscosity sense.
{: .block-definition }

### Discrete counterpart from one short Bellman step

Now consider the controlled diffusion

$$
dX_t
=
\bigl(f_t(X_t)+B_t(X_t)u_t\bigr)dt
+\sigma_t(X_t)dW_t,
$$

with diffusion covariance $$a_t=\sigma_t\sigma_t^\top$$. The running cost is

$$
q_t(x)+\frac12u_t^\top M_t(x)u_t,
$$

where $$M_t$$ is positive definite on the control space. The terminal cost is $$V_T(x)=g(x)$$.

Apply one Bellman step of duration $$\Delta t$$.

$$
\begin{aligned}
V_t(x)=\min_u\Bigl\{&
\left(q_t(x)+\frac12u^\top M_tu\right)\Delta t\\
&+\mathbb E[V_{t+\Delta t}(X_{t+\Delta t})\mid X_t=x,u_t=u]
\Bigr\}.
\end{aligned}
$$

The Euler increment is

$$
\Delta X=(f_t+B_tu)\Delta t+\sigma_t\sqrt{\Delta t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Expand the future value around $$(t,x)$$.

$$
\begin{aligned}
V_{t+\Delta t}(x+\Delta X)
&=V_t(x)+\partial_tV_t\Delta t
+\nabla V_t^\top\Delta X\\
&\quad+\frac12\Delta X^\top\nabla^2V_t\Delta X
+o(\Delta t).
\end{aligned}
$$

The noise has zero mean and covariance $$a_t\Delta t$$. Therefore

$$
\mathbb E[\nabla V_t^\top\Delta X]
=
\nabla V_t^\top(f_t+B_tu)\Delta t,
$$

$$
\mathbb E[\Delta X^\top\nabla^2V_t\Delta X]
=
\operatorname{tr}(a_t\nabla^2V_t)\Delta t+o(\Delta t).
$$

Only the noise covariance survives in the second-order term. The deterministic increment $$(f_t+B_tu)\Delta t$$ has squared size $$O((\Delta t)^2)$$. Its cross term with the noise has mean zero and size $$O((\Delta t)^{3/2})$$. The noise increment has squared size $$O(\Delta t)$$. Therefore only this term remains after division by $$\Delta t$$. This order counting produces the diffusion Hessian in HJB.

Substitute these expectations into Bellman recursion, cancel $$V_t(x)$$, divide by $$\Delta t$$, and take the limit. Using backward time, the HJB equation is

$$
\begin{aligned}
-\partial_tV_t
=q_t+f_t\cdot\nabla V_t
+\frac12\operatorname{tr}(a_t\nabla^2V_t)\\
+\min_u\left\{
\frac12u^\top M_tu
+u^\top B_t^\top\nabla V_t
\right\}.
\end{aligned}
$$

Every sign follows from the terminal-value convention. We set $$V_T=g$$ and solve the equation backward from $$T$$ to $$0$$.

This Taylor argument assumes enough smoothness to differentiate $$V$$ twice in space and once in time. Value functions can develop kinks even from smooth data. HJB is then interpreted as a viscosity solution. This interpretation preserves dynamic programming without requiring every derivative pointwise. The algebra below describes the classical regime. Weaker versions need their own regularity hypotheses.

### Derivation of the control nonlinearity

Differentiate the expression inside the minimum with respect to $$u$$.

$$
M_tu+B_t^\top\nabla V_t=0.
$$

Equivalently, complete the square.

$$
\begin{aligned}
\frac12u^\top M_tu+u^\top B_t^\top\nabla V_t
&=\frac12
\left(u+M_t^{-1}B_t^\top\nabla V_t\right)^\top
M_t
\left(u+M_t^{-1}B_t^\top\nabla V_t\right)\\
&\quad-
\frac12\nabla V_t^\top
B_tM_t^{-1}B_t^\top\nabla V_t.
\end{aligned}
$$

Positive definiteness of $$M_t$$ makes the first term nonnegative and pins down its unique minimizer.

The minimizing control is

$$
u_t^*
=
-M_t^{-1}B_t^\top\nabla V_t.
$$

The control $$u_t^*$$ lives in the action space. The actual drift correction is $$B_tu_t^*$$. Substituting the minimizer gives

$$
\begin{aligned}
-\partial_tV_t
=q_t+f_t\cdot\nabla V_t
&-\frac12\nabla V_t^\top
B_tM_t^{-1}B_t^\top\nabla V_t\\
&+\frac12\operatorname{tr}(a_t\nabla^2V_t).
\end{aligned}
$$

The quadratic gradient is the optimized control cost plus its effect on future value. It is also the obstruction to a linear PDE.

## Desirability and the Hopf–Cole transform

The discrete desirability $$\varphi_k=e^{-V_k/\lambda}$$ turned a soft Bellman recursion into a linear kernel update. The continuous-time transform needs one exact coefficient match. The control gradient term must cancel the quadratic term from differentiating the logarithm.

> **Matching condition and Hopf–Cole linearization.** Let $$\lambda>0$$ and suppose
>
> $$
> a_t=\lambda B_tM_t^{-1}B_t^\top.
> $$
>
> Define the positive desirability function by
>
> $$
> \varphi_t(x)=e^{-V_t(x)/\lambda}.
> $$
>
> Then the optimized HJB equation is equivalent to the linear terminal-value problem
>
> $$
> \partial_t\varphi_t
> +f_t\cdot\nabla\varphi_t
> +\frac12\operatorname{tr}(a_t\nabla^2\varphi_t)
> -\frac{q_t}{\lambda}\varphi_t=0,
> \qquad
> \varphi_T=e^{-g/\lambda}.
> $$
>
> The optimal control and its state-space drift correction are
>
> $$
> u_t^*=\lambda M_t^{-1}B_t^\top\nabla\log\varphi_t,
> \qquad
> B_tu_t^*=a_t\nabla\log\varphi_t.
> $$
{: .block-definition }

### Why the matching condition is restrictive

The matching condition says that passive noise and control span the same state-space directions. Their magnitudes must also agree with the quadratic cost through one matrix identity. Having both noise and control is not sufficient.

For compactness, write $$G_t=B_tM_t^{-1}B_t^\top$$. The condition is then $$a_t=\lambda G_t$$.

The inverse relation $$V_t=-\lambda\log\varphi_t$$ converts positive desirability back to additive cost-to-go.

### Derivation of the quadratic cancellation

Its derivatives are

$$
\partial_tV_t
=
-\lambda\frac{\partial_t\varphi_t}{\varphi_t},
\qquad
\nabla V_t
=
-\lambda\frac{\nabla\varphi_t}{\varphi_t},
$$

$$
\nabla^2V_t
=
-\lambda\frac{\nabla^2\varphi_t}{\varphi_t}
+\lambda\frac{\nabla\varphi_t\nabla\varphi_t^\top}{\varphi_t^2}.
$$

The optimized-control term becomes

$$
-\frac{\lambda^2}{2\varphi_t^2}
\nabla\varphi_t^\top
B_tM_t^{-1}B_t^\top
\nabla\varphi_t.
$$

The Hessian contributes its own quadratic term.

$$
+\frac{\lambda}{2\varphi_t^2}
\nabla\varphi_t^\top a_t\nabla\varphi_t.
$$

These are the two terms that must cancel. Substituting every derivative into HJB, multiplying by $$\varphi_t/\lambda$$, and moving the linear terms to the left gives the more revealing equation

$$
\begin{aligned}
\partial_t\varphi_t
+f_t\cdot\nabla\varphi_t
+\frac12\operatorname{tr}(a_t\nabla^2\varphi_t)
-\frac{q_t}{\lambda}\varphi_t
=\frac{1}{2\varphi_t}
\nabla\varphi_t^\top(a_t-\lambda G_t)\nabla\varphi_t.
\end{aligned}
$$

The right-hand side gives the exact matching condition. It vanishes when $$a_t=\lambda B_tM_t^{-1}B_t^\top$$. The remaining equation is linear.

$$
\partial_t\varphi_t
+f_t\cdot\nabla\varphi_t
+\frac12\operatorname{tr}(a_t\nabla^2\varphi_t)
-\frac{q_t}{\lambda}\varphi_t
=0,
$$

with terminal condition

$$
\varphi_T(x)=e^{-g(x)/\lambda}.
$$

This exponential change of variables is the Hopf–Cole transform. The HJB and linear desirability equations are two coordinate descriptions of one matched control problem. Kappen's path-integral control derivation makes the same cancellation and its covariance requirement explicit (<span id="cite-kappen2005"></span>[Kappen, 2005](#ref-kappen2005)).

The equivalence can be followed in either direction. Dynamic programming turns the path objective into nonlinear HJB. Hopf–Cole maps the additive value $$V$$ to the positive desirability $$\varphi$$. The matched covariance removes the residual quadratic term.

Taking $$\nabla\log\varphi$$ recovers the feedback drift. Girsanov identifies that local drift change with the globally tilted path law. Every step depends on the same covariance-matching condition.

## Feynman–Kac representation of the linear equation

Feynman–Kac turns the linear desirability equation into an expectation over uncontrolled futures. The PDE evolves a function backward in time. The probabilistic formula starts at the current state, samples passive paths forward, and averages their exponential cost. Both descriptions apply the same linear evolution operator.

> **Feynman–Kac representation.** Let the passive process satisfy
>
> $$
> dX_s=f_s(X_s)ds+\sigma_s(X_s)dW_s.
> $$
>
> Suppose the SDE is well posed and the exponential cost is integrable. The solution of
>
> $$
> (\partial_t+\mathcal L_t)\varphi_t
> -\frac{q_t}{\lambda}\varphi_t=0,
> \qquad
> \varphi_T=e^{-g/\lambda},
> $$
>
> has the representation
>
> $$
> \varphi_t(x)
> =\mathbb E_R\!\left[
> \exp\!\left(
> -\frac{g(X_T)+\int_t^Tq_s(X_s)ds}{\lambda}
> \right)
> \ \middle|\ X_t=x
> \right],
> $$
>
> where $$\mathcal L_t\psi=f_t\cdot\nabla\psi+\frac12\operatorname{tr}(a_t\nabla^2\psi)$$ is the passive generator.
{: .block-definition }

### Discrete counterpart through repeated passive averaging

Unroll the discrete desirability recursion from time $$k$$ to $$N$$. With terminal value $$\varphi_N(x)=e^{-g(x)/\lambda}$$,

$$
\varphi_k(x)
=
\mathbb E_R\!\left[
\exp\!\left(
-\frac{\sum_{j=k}^{N-1}q_j(X_j)+g(X_N)}{\lambda}
\right)
\middle|X_k=x
\right].
$$

Each backward kernel application adds one passive transition and one running-cost factor. The continuous Feynman–Kac formula replaces the sum by a time integral and the Markov chain by a diffusion.

### Continuous limit of repeated averaging

As the time step decreases, the discrete sum converges to a time integral and the passive chain converges to the diffusion. The repeated kernel average becomes the conditional path expectation in the definition block.

### Derivation with Itô's formula

The representation follows directly from Itô's formula. Along a passive trajectory starting from $$X_t=x$$, define

$$
Y_s
=
\exp\!\left(-\frac{1}{\lambda}
\int_t^sq_r(X_r)dr\right)
\varphi_s(X_s),
\qquad t\leq s\leq T.
$$

Applying the product rule and Itô's formula gives

$$
\begin{aligned}
dY_s
&=e^{-\lambda^{-1}\int_t^sq_r(X_r)dr}
\left[
\partial_s\varphi_s
+f_s\cdot\nabla\varphi_s
+\frac12\operatorname{tr}(a_s\nabla^2\varphi_s)
-\frac{q_s}{\lambda}\varphi_s
\right]ds\\
&\quad+
e^{-\lambda^{-1}\int_t^sq_r(X_r)dr}
(\nabla\varphi_s)^\top\sigma_s\,dW_s.
\end{aligned}
$$

The bracketed drift is zero by the linear desirability PDE, so $$Y_s$$ is a local martingale and, under the integrability assumptions below, a true martingale. At the terminal time,

$$
Y_T
=\exp\!\left(
-\frac{g(X_T)+\int_t^Tq_s(X_s)ds}{\lambda}
\right).
$$

Taking the conditional expectation of $$Y_T$$ and using $$Y_t=\varphi_t(x)$$ produces the stated formula. The negative potential term $$-(q_t/\lambda)\varphi_t$$ compensates for accumulated running cost.

The representation assumes that the passive SDE is well posed and that the exponential cost is integrable. Smooth bounded coefficients give one sufficient regime. They are not necessary. Standard stochastic-calculus treatments give sharper variants (<span id="cite-oksendal2003"></span>[Øksendal, 2003](#ref-oksendal2003)).

The function $$\varphi_t(x)$$ is total future desirability. It averages the exponential reward of every passive future starting at $$x$$. The value

$$
V_t(x)=-\lambda\log\varphi_t(x)
$$

is the corresponding soft future cost. Low-cost futures receive exponentially more weight. Many futures still contribute.

The expectation is a path integral in the probabilistic sense. It integrates an exponential path cost over passive trajectories. Numerical path-integral control methods estimate it with samples, importance sampling, or adaptive controlled proposals. Linearity replaces a nonlinear PDE with an expectation that can have high variance. High-dimensional rare events remain difficult.

### Closed-form case with a Gaussian terminal cost

A one-dimensional example shows all three coordinate systems without a numerical PDE. Let

$$
dX_t=u_tdt+\sqrt{\lambda}\,dW_t
$$

and penalize

$$
\mathbb E\!\left[
\frac12\int_t^Tu_s^2ds
+\frac{\kappa}{2}(X_T-y)^2
\right].
$$

Here $$B=M=1$$ and $$a=\lambda$$, so the matching condition holds. Set $$\tau=T-t$$. Under the passive process,

$$
X_T=x+\sqrt{\lambda\tau}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,1).
$$

Let $$d=x-y$$. Combining the standard-normal density with the exponential terminal cost gives the exponent

$$
\begin{aligned}
\frac{\epsilon^2}{2}
+\frac{\kappa}{2\lambda}
(d+\sqrt{\lambda\tau}\,\epsilon)^2
&=\frac{1+\kappa\tau}{2}
\left(
\epsilon+
\frac{\kappa d\sqrt{\tau/\lambda}}
{1+\kappa\tau}
\right)^2\\
&\quad+
\frac{\kappa d^2}
{2\lambda(1+\kappa\tau)}.
\end{aligned}
$$

The first term integrates as a shifted Gaussian with precision $$1+\kappa\tau$$. It contributes the factor $$(1+\kappa\tau)^{-1/2}$$. The remaining constant stays in the exponential. Therefore the Feynman–Kac integral is

$$
\varphi_t(x)
=
\frac{1}{\sqrt{1+\kappa\tau}}
\exp\!\left[
-\frac{\kappa(x-y)^2}
{2\lambda(1+\kappa\tau)}
\right].
$$

Taking its negative log gives the value

$$
V_t(x)
=
\frac{\lambda}{2}\log(1+\kappa\tau)
+\frac{\kappa(x-y)^2}{2(1+\kappa\tau)}.
$$

The first term is the expected price of terminal noise. The second is a softened quadratic distance to $$y$$. The optimal drift can be computed from $$-\partial_xV_t$$ or $$a\partial_x\log\varphi_t$$.

$$
u_t^*(x)
=
-\frac{\kappa(x-y)}{1+\kappa(T-t)}.
$$

Three checks catch most sign or scale mistakes in this calculation. If $$\kappa=0$$, then $$\varphi=1$$ and $$V=0$$. There is no task to control. If $$t=T$$, then $$\tau=0$$ and $$\varphi_T=e^{-\kappa(x-y)^2/(2\lambda)}$$. This is exactly the terminal condition. Finally, differentiating the displayed value gives the same control as differentiating the log desirability.

Far from the terminal time, passive noise has time to explore and the restoring drift is weaker. As $$t\to T$$, the drift approaches the terminal quadratic force $$-\kappa(x-y)$$. The global exponential weighting of terminal locations has become a state-local feedback rule.

## Temperature and path selection

Temperature $$\lambda$$ controls the compromise between staying close to the passive process and concentrating on low-cost paths. Large temperature preserves passive diversity. Small temperature amplifies cost differences. It can choose only among trajectories that the passive process can generate.

> **Temperature in path-space control.** Let $$R$$ be the passive path law and let
>
> $$
> C(\omega)=g(X_T)+\int_0^Tq_t(X_t)dt.
> $$
>
> For $$\lambda>0$$ and $$0<Z_\lambda=\mathbb E_R[e^{-C/\lambda}]<\infty$$, define
>
> $$
> \frac{dP_\lambda^*}{dR}
> =\frac{e^{-C/\lambda}}{Z_\lambda}.
> $$
>
> This law uniquely minimizes
>
> $$
> \mathbb E_P[C]+\lambda D_{\mathrm{KL}}(P\|R)
> $$
>
> over $$P\ll R$$. The value is $$-\lambda\log Z_\lambda$$. Therefore $$\lambda$$ controls the strength of both the path-space KL penalty and the exponential path selection.
{: .block-definition }

### Discrete counterpart with route odds

Partition the passive trajectories into two route families $$A$$ and $$B$$. Suppose every path in a family has the same representative cost, with $$C_A<C_B$$. Exponential tilting changes the route odds according to

$$
\frac{P^*(A)}{P^*(B)}
=
\frac{R(A)}{R(B)}
\exp\!\left(
\frac{C_B-C_A}{\lambda}
\right).
$$

The passive odds remain in the formula. A low-cost route that is extremely rare under the reference may still receive little mass unless its cost advantage compensates for that rarity. Temperature $$\lambda$$ controls how strongly cost can overturn the passive preference.

Both limits follow from elementary expansions. For small $$\lambda$$, factor the minimum cost $$C_{\min}$$ out of the partition function.

$$
-\lambda\log\mathbb E_R[e^{-C/\lambda}]
=C_{\min}
-\lambda\log\mathbb E_R[e^{-(C-C_{\min})/\lambda}].
$$

Only minimum-cost paths survive in the second expectation as $$\lambda\to0$$. For large $$\lambda$$, expand the exponential and logarithm instead.

$$
-\lambda\log\mathbb E_R[e^{-C/\lambda}]
=\mathbb E_R[C]
-\frac{\operatorname{Var}_R(C)}{2\lambda}
+O(\lambda^{-2}).
$$

Thus the high-temperature value approaches the passive mean cost, and the controlled distribution approaches the passive distribution. Temperature continuously interpolates between selecting the cheapest supported paths and accepting the passive ensemble.

Lowering $$\lambda$$ only redistributes probability among paths supported by $$R$$. If $$R(A)=0$$, then exponential reweighting cannot assign positive probability to route $$A$$, regardless of its cost. The low-temperature limit is therefore a minimum over the passive support, not an unconstrained search over every imaginable trajectory.

Temperature has two linked readings. In the global path objective it weights the KL penalty. In the local problem it fixes the ratio between diffusion covariance and control cost. Calling it temperature is useful only when both roles are kept together.

"Linear" refers to the backward equation for $$\varphi$$. The optimal feedback $$a\nabla\log\varphi$$ can remain highly nonlinear in $$x$$. Evaluating $$\varphi$$ may require a difficult rare-event expectation. The transform removes one mathematical nonlinearity. The learned controller and the required computation can remain nonlinear.

## Doob control and path-space KL

The desirability function has two jobs. Its logarithm determines the local drift correction. Its path expectation determines the global exponential reweighting. The Doob transform and Girsanov's theorem show that these local and global descriptions agree.

> **Doob-controlled diffusion.** Fix a starting pair $$(t,x)$$ and let $$R^{t,x}$$ be the passive diffusion law from that pair. Define
>
> $$
> C_{t,T}=g(X_T)+\int_t^Tq_s(X_s)ds,
> \qquad
> \varphi_t(x)=\mathbb E_{R^{t,x}}[e^{-C_{t,T}/\lambda}].
> $$
>
> The cost-tilted conditional path law is
>
> $$
> \frac{dP^{*,t,x}}{dR^{t,x}}
> =\frac{e^{-C_{t,T}/\lambda}}{\varphi_t(x)}.
> $$
>
> Under the matching condition, this law is generated by
>
> $$
> dX_s=
> [f_s(X_s)+a_s(X_s)\nabla\log\varphi_s(X_s)]ds
> +\sigma_s(X_s)dW_s.
> $$
>
> This is the Feynman–Kac extension of a Doob transform. With zero running cost, $$\varphi$$ is space-time harmonic and the construction is the ordinary Doob h-transform.
{: .block-definition }

### The optimal drift is a Doob transform

Insert $$\nabla V_t=-\lambda\nabla\log\varphi_t$$ into the optimal control.

$$
u_t^*
=
\lambda M_t^{-1}B_t^\top\nabla\log\varphi_t.
$$

Multiplying by $$B_t$$ and using the matching condition gives

$$
B_tu_t^*
=
a_t\nabla\log\varphi_t.
$$

Thus the optimally controlled diffusion is

$$
dX_t
=
\left[f_t(X_t)+a_t(X_t)\nabla\log\varphi_t(X_t)\right]dt
+\sigma_t(X_t)dW_t.
$$

This is the continuous Doob transform associated with future desirability. Hopf–Cole and Doob do different jobs. Hopf–Cole converts an additive value into a positive function and linearizes the matched HJB. The Doob transform uses that positive function to turn a global future weighting into local transition dynamics.

### Global path KL and local control are the same objective

> **Control energy as path-space relative entropy.** Let $$R$$ and $$P^u$$ have the same initial law and diffusion coefficient. Suppose their drifts differ by $$B_tu_t$$, this correction lies in the range of $$\sigma_t$$, and Girsanov's conditions hold. Then
>
> $$
> D_{\mathrm{KL}}(P^u\|R)
> =\frac12\mathbb E_{P^u}\!\left[
> \int_0^T
> \lVert\sigma_t^\dagger B_tu_t\rVert^2dt
> \right].
> $$
>
> If $$a_t=\lambda B_tM_t^{-1}B_t^\top$$ and the control parametrization has no redundant directions, then
>
> $$
> \lambda D_{\mathrm{KL}}(P^u\|R)
> =\mathbb E_{P^u}\!\left[
> \frac12\int_0^Tu_t^\top M_tu_tdt
> \right].
> $$
{: .block-definition }

Suppose the controlled and passive diffusions have the same initial law and diffusion coefficient. Girsanov gives

$$
D_{\mathrm{KL}}(P^u\|R)
=
\mathbb E_{P^u}\!\left[
\frac12\int_0^T
\lVert\sigma_t^\dagger B_tu_t\rVert^2dt
\right],
$$

where $$\sigma_t^\dagger$$ is the inverse or pseudoinverse on the noise range. The norm can be written in state-space coordinates as

$$
\lVert\sigma_t^\dagger B_tu_t\rVert^2
=u_t^\top B_t^\top a_t^\dagger B_tu_t,
$$

provided $$B_tu_t$$ lies in the range of $$\sigma_t$$. The matching condition supplies the missing matrix identity. If $$B_t$$ has full column rank, set $$C_t=B_tM_t^{-1/2}$$. Then

$$
a_t=\lambda C_tC_t^\top,
$$

and the pseudoinverse identity $$C_t^\top(C_tC_t^\top)^\dagger C_t=I$$ gives

$$
\lambda B_t^\top a_t^\dagger B_t=M_t.
$$

Consequently, under this compatible parametrization, $$\lambda$$ times the path KL is exactly the quadratic control cost

$$
\mathbb E_{P^u}\!\left[
\frac12\int_0^Tu_t^\top M_tu_t\,dt
\right].
$$

If the action parametrization contains redundant directions in the null space of $$B_t$$, those directions change neither the state drift nor the path law. The minimum-cost representation sets them to zero. A nonzero redundant component would charge a motion that the process never makes. If $$B_tu_t$$ leaves the noise range, no Brownian shift realizes the new drift. The controlled law is then not absolutely continuous with respect to the passive law.

Adding state and terminal costs gives the global problem

$$
\min_{P\ll R}
\left\{
\mathbb E_P\!\left[g(X_T)+\int_0^Tq_t(X_t)dt\right]
+\lambda D_{\mathrm{KL}}(P\|R)
\right\}.
$$

The Gibbs variational identity says its optimal path law is the passive law exponentially tilted by total cost. Dynamic programming says the same optimum obeys HJB. Feynman–Kac evaluates its desirability under the passive law. Girsanov translates its path KL into local quadratic effort. The Hopf–Cole cancellation is the PDE trace of those path-space identities, not an unrelated algebraic trick.

This equivalence also motivates the phrase **control as inference**. One may infer a cost-biased posterior over passive trajectories. A controlled Markov process can then realize that posterior. This interpretation retains the support and integrability assumptions required by the change of measure.

## Scope of linearly solvable control

### Neighboring control formalisms

Several names emphasize different pieces of the same matched structure.

- **Linearly solvable MDPs** start in discrete time with KL-penalized transition control and a linear desirability recursion.
- **Path-integral control** emphasizes the Feynman–Kac expectation over passive trajectories.
- **KL-regularized control** emphasizes the global penalty for changing the passive path law.
- **Control as inference** treats the exponential cost as a likelihood on trajectories.

The stochastic maximum principle takes a different route. It introduces a costate process and derives local first-order optimality conditions, often as a forward-backward SDE. Dynamic programming instead seeks a value function over all states and produces HJB. Under sufficient regularity, the costate is related to $$\nabla V_t(X_t)$$. The maximum principle does not create the Hopf–Cole linearization by itself. The covariance and cost matching remain necessary.

### Where linearization fails

The transform applies under restrictive assumptions.

**Unmatched directions.** If $$a_t\neq\lambda B_tM_t^{-1}B_t^\top$$, the residual

$$
\frac{1}{2\varphi_t}
\nabla\varphi_t^\top
(a_t-\lambda B_tM_t^{-1}B_t^\top)
\nabla\varphi_t
$$

remains in the desirability PDE. Even the scalar system $$dX_t=u_tdt+\sigma dW_t$$ with effort $$\frac{m}{2}u_t^2$$ matches only when $$\lambda=m\sigma^2$$. Noise-free control directions are also outside the passive path support, so their path KL is not finite.

**Nonquadratic control cost.** Minimizing a different convex cost produces its convex conjugate in HJB. For example,

$$
\min_u\left\{\frac14u^4+bu\right\}
=-\frac34\lvert b\rvert^{4/3}.
$$

With $$b=B_t^\top\nabla V_t$$, HJB contains a $$4/3$$-power gradient term rather than a quadratic. The Hessian of $$-\lambda\log\varphi$$ has no matching term to cancel it.

**Controlled diffusion.** If the action changes $$\sigma_t$$, the Hessian term itself depends on the control. Diffusions with different quadratic variation are usually singular as continuous-time path measures. Therefore the simple Girsanov KL identity fails.

**Hard constraints.** If $$\lVert u\rVert\leq u_{\max}$$, the unconstrained optimizer $$-M^{-1}B^\top\nabla V$$ must be projected or clipped whenever it exceeds the bound. The minimized Hamiltonian then changes formula at the clipping threshold. State barriers and pathwise safety constraints can likewise impose boundary conditions or admissible-path restrictions incompatible with the unconstrained linear Feynman–Kac problem.

**Absolute-continuity failure.** A passive process cannot be reweighted into trajectories outside its support. Exponential tilting changes probabilities, not reachability.

General stochastic optimal control still has Bellman recursion and HJB. It is **linearly solvable** only when the optimized control nonlinearity matches the diffusion curvature under Hopf–Cole.

## One problem in three coordinate systems

The additive value $$V_t$$ is natural for Bellman recursion and optimization. The positive desirability $$\varphi_t$$ is natural for linear evolution and conditional expectations. The controlled path law is natural for KL projection and exponential reweighting.

Under the matching condition, these are exact descriptions of one another. HJB, Feynman–Kac, and the Doob-controlled drift encode the same optimum. The computational choice has three options. One can solve a nonlinear value equation, estimate a linear expectation that is sensitive to rare events, or learn the local drift.

---

## References

- <span id="ref-todorov2006"></span>Todorov, E. (2006). Linearly-Solvable Markov Decision Problems. [NIPS 2006](https://proceedings.neurips.cc/paper/2006/hash/d806ca13ca3449af72a1ea5aedbed26a-Abstract.html). <a href="#cite-todorov2006" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kappen2005"></span>Kappen, H. J. (2005). Path Integrals and Symmetry Breaking for Optimal Control Theory. [Journal of Statistical Mechanics](https://doi.org/10.1088/1742-5468/2005/11/P11011). <a href="#cite-kappen2005" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-oksendal2003"></span>Øksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications, 6th ed. [Springer](https://link.springer.com/book/10.1007/978-3-642-14394-6). <a href="#cite-oksendal2003" class="reversefootnote" role="doc-backlink">↩</a>
