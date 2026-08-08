---
layout: post
title: "Diffusion Models and Flow Matching"
date: 2026-08-08
last_updated: 2026-08-08
description: "A unified derivation of diffusion and flow matching through conditional probability paths, marginalization identities, and simulation-free regression."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
lecture_paths: [ml4mol, gdl]
tags: [diffusion-models, score-matching, flow-matching, stochastic-differential-equations, continuous-normalizing-flows]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the diffusion and flow-matching storyline from my
    Machine Learning for Molecules and Geometric Deep Learning lectures, using a
    common conditional-regression principle. For the density equations underneath the
    construction, see <a href="{% post_url 2026-02-04-fokker-planck-equation %}">The
    Fokker–Planck Equation</a>; for a closer comparison of deterministic and
    stochastic dynamics, see <a href="{% post_url 2026-08-08-odes-sdes-probability-flow %}">ODEs,
    SDEs, and Probability Flow</a>.</em>
</p>

## Two Ways to Learn Motion Through Probability Space

A generative model has to turn a simple distribution, usually a standard Gaussian, into a complicated data distribution. Diffusion models and flow-matching models appear to solve this problem differently. A diffusion model first destroys data with a stochastic process, learns the score of every intermediate noisy distribution, and then reverses the process. A flow-matching model specifies a probability path and learns the velocity field whose ordinary differential equation transports probability along it.

That distinction is real at sampling time, but it hides the more useful connection at training time. In both cases, the global object we need is an average over unknown data origins. The marginal score $$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$ is unknown because the noisy marginal $$p_t$$ is unknown. The marginal flow velocity $$\mathbf{u}_t(\mathbf{x})$$ is unknown because it averages many conditional trajectories that can pass through the same point. Yet after conditioning on a clean data sample, both targets become elementary. Squared-error regression then performs the required marginalization automatically.

{% include figure.liquid loading="eager" path="assets/img/blog/difffm_two_views.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Diffusion and flow matching attach different local objects to an intermediate probability path: a score for reversing an SDE, or a velocity for integrating an ODE. Their training logic is shared: expose a tractable conditional target, then let regression recover its conditional mean at the observed state." %}

This is the thread of the post. We will first make corruption analytically tractable, derive denoising score matching, and use the learned score to reverse a diffusion. We will then prescribe conditional Gaussian paths, derive their velocities, and show why conditional flow matching learns the correct marginal ODE without simulating it during training. The final comparison separates the mathematical identities from the practical choices—schedule, prediction target, and numerical solver—that determine whether either model works well.

Throughout, time $$t\in[0,1]$$ is **corruption time**: $$t=0$$ is data and $$t=1$$ is the Gaussian base distribution. Diffusion naturally uses this convention. Many flow-matching papers instead put noise at $$t=0$$ and data at $$t=1$$; replacing $$t$$ by $$1-t$$ converts between the conventions and flips the velocity sign. Keeping one convention lets us compare the two methods without silently exchanging endpoints.

## Forward Corruption Makes a Hard Density Easy to Perturb

Let $$\mathbf{x}_0\sim p_{\mathrm{data}}$$. A continuous diffusion corrupts it through the Itô SDE

$$
d\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t,t)\,dt + g(t)\,d\mathbf{W}_t,
$$

where $$\mathbf{f}$$ is a chosen drift, $$g$$ controls the noise rate, and $$\mathbf{W}_t$$ is Brownian motion. The coefficients are not learned. They are designed so that the terminal marginal $$p_1$$ is close to a distribution from which we can sample directly. Score-based SDE models make this construction explicit and derive both stochastic and deterministic reverse dynamics from the same intermediate marginals (<span id="cite-song2021"></span>[Song et al., 2021](#ref-song2021)).

The most convenient processes have a closed-form perturbation kernel. Their state at any time can be written

$$
\mathbf{X}_t = \alpha_t\mathbf{X}_0 + \sigma_t\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

so that

$$
p_{t|0}(\mathbf{x}\mid\mathbf{x}_0)
=\mathcal{N}\!\left(\mathbf{x};\alpha_t\mathbf{x}_0,\sigma_t^2\mathbf{I}\right).
$$

The boundary conditions $$\alpha_0=1,\sigma_0=0$$ preserve data, while $$\alpha_1\approx0,\sigma_1\approx1$$ erase it. Variance-preserving diffusions attenuate the signal while adding noise; variance-exploding diffusions primarily increase noise. Discrete denoising diffusion probabilistic models are time-discretized members of the same broad family (<span id="cite-ho2020"></span>[Ho et al., 2020](#ref-ho2020)).

The marginal at time $$t$$ is a mixture of these kernels:

$$
p_t(\mathbf{x})
=\int p_{t|0}(\mathbf{x}\mid\mathbf{x}_0)
       p_{\mathrm{data}}(\mathbf{x}_0)\,d\mathbf{x}_0.
$$

We can sample this mixture by drawing a data point and Gaussian noise. We generally cannot evaluate its density because the integral ranges over the entire data distribution. Diffusion training succeeds because it needs neither the density nor its normalization constant. It needs only the **score**, the spatial gradient $$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$$.

## Denoising Score Matching Is a Marginalization Identity

The conditional Gaussian score is known exactly:

$$
\nabla_{\mathbf{x}}\log p_{t|0}(\mathbf{x}\mid\mathbf{x}_0)
=-\frac{\mathbf{x}-\alpha_t\mathbf{x}_0}{\sigma_t^2}.
$$

For a perturbed sample $$\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol{\epsilon}$$, this reduces to $$-\boldsymbol{\epsilon}/\sigma_t$$. The score points from the noisy observation back toward the mean of its particular corruption kernel. It is not yet the marginal score: two different clean samples can produce the same noisy location and suggest different directions.

Differentiate the mixture under the integral and divide by $$p_t(\mathbf{x})$$:

$$
\begin{aligned}
\nabla_{\mathbf{x}}\log p_t(\mathbf{x})
&=\frac{1}{p_t(\mathbf{x})}
  \int \nabla_{\mathbf{x}}p_{t|0}(\mathbf{x}\mid\mathbf{x}_0)
  p_{\mathrm{data}}(\mathbf{x}_0)\,d\mathbf{x}_0 \\
&=\int \nabla_{\mathbf{x}}\log p_{t|0}(\mathbf{x}\mid\mathbf{x}_0)
  p(\mathbf{x}_0\mid\mathbf{x})\,d\mathbf{x}_0 \\
&=\mathbb{E}\!\left[
  \nabla_{\mathbf{x}}\log p_{t|0}(\mathbf{x}\mid\mathbf{X}_0)
  \mid \mathbf{X}_t=\mathbf{x}\right].
\end{aligned}
$$

This denoising identity says that the marginal score is the posterior average of conditional scores. Crucially, training never has to calculate the posterior $$p(\mathbf{x}_0\mid\mathbf{x}_t)$$. Consider the loss

$$
\mathcal{L}_{\mathrm{DSM}}(\theta)
=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}
\left[
\lambda(t)\left\|
\mathbf{s}_\theta(\mathbf{x}_t,t)
+\frac{\boldsymbol{\epsilon}}{\sigma_t}
\right\|^2
\right].
$$

At every fixed $$(\mathbf{x}_t,t)$$, the minimizer of squared error is the conditional mean of the target. It is therefore the marginal score. More precisely, the conditional-target loss equals a marginal-score regression loss plus the conditional variance of the target, which is independent of $$\theta$$. This is the score-matching–denoising connection developed by <span id="cite-vincent2011"></span>[Vincent, 2011](#ref-vincent2011).

{% include figure.liquid loading="eager" path="assets/img/blog/difffm_denoising_identity.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A Gaussian corruption gives the exact conditional score (-\epsilon/\sigma_t), but that arrow depends on the hidden clean sample. Averaging all compatible conditional arrows at a fixed noisy state yields the marginal score \(\nabla_x\log p_t(x)\); squared-error regression performs this averaging without evaluating the posterior." %}

Implementations often predict $$\boldsymbol{\epsilon}$$, $$\mathbf{x}_0$$, or a linear combination called the velocity parameter rather than the score itself. These parameterizations contain equivalent information when $$\alpha_t$$ and $$\sigma_t$$ are known, but they scale errors differently across noise levels. The weight $$\lambda(t)$$ is therefore not cosmetic: together with the output parameterization, it decides which portions of the path dominate optimization.

## The Score Turns Corruption Around

The forward SDE spreads probability. To generate, start at $$\mathbf{X}_1\sim p_1$$ and integrate back toward $$t=0$$. The reverse-time SDE is

$$
d\mathbf{X}_t
=\left[\mathbf{f}(\mathbf{X}_t,t)
-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{X}_t)\right]dt
+g(t)\,d\overline{\mathbf{W}}_t,
\qquad t:1\rightarrow0.
$$

Here $$dt$$ is negative along numerical integration, and $$\overline{\mathbf{W}}_t$$ denotes reverse-time Brownian motion. The score correction compensates for the probability spreading of the forward noise. Replacing the unknown score by $$\mathbf{s}_\theta$$ produces the generative sampler.

The same marginal densities can also be generated by the probability-flow ODE

$$
\frac{d\mathbf{X}_t}{dt}
=\mathbf{f}(\mathbf{X}_t,t)
-\frac{1}{2}g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{X}_t).
$$

Why does the factor change from $$1$$ to $$1/2$$? The reverse SDE retains a diffusion term, which continues to spread density and needs the full score correction. The ODE has no random spreading, so half the correction produces the same Fokker–Planck evolution. This is an equality of one-time marginals, not of individual trajectories: stochastic and deterministic sample paths can look entirely different while visiting the same distributions at every time.

This already narrows the apparent gap to flow matching. Once a score is known, diffusion admits an ODE velocity. Flow matching asks whether we can train such a velocity directly, without first introducing an SDE or deriving its score.

## Flow Matching Starts From a Conditional Path

An ODE

$$
\frac{d\mathbf{X}_t}{dt}=\mathbf{u}_t(\mathbf{X}_t)
$$

transports its marginal density according to the continuity equation

$$
\partial_t p_t(\mathbf{x})
=-\nabla_{\mathbf{x}}\cdot
  \left[p_t(\mathbf{x})\mathbf{u}_t(\mathbf{x})\right].
$$

If we knew a velocity field connecting data to noise, we could sample by integrating it backward. Directly designing the marginal path $$p_t$$ and solving for $$\mathbf{u}_t$$ is hard. Flow matching instead builds a family of easy conditional paths indexed by a condition $$\mathbf{z}$$—usually a data endpoint, or a paired data-and-noise endpoint—and then marginalizes them.

For a clean sample $$\mathbf{x}_0$$ and independent noise $$\boldsymbol{\epsilon}$$, use the same affine construction as above:

$$
\mathbf{x}_t
=\alpha_t\mathbf{x}_0+\sigma_t\boldsymbol{\epsilon}.
$$

This equation now specifies an ODE trajectory rather than merely sampling the marginal of an SDE. Differentiate the trajectory:

$$
\dot{\mathbf{x}}_t
=\dot{\alpha}_t\mathbf{x}_0
+\dot{\sigma}_t\boldsymbol{\epsilon}.
$$

The right-hand side is a tractable conditional velocity target. If we want it as a function of the current state and data endpoint, eliminate $$\boldsymbol{\epsilon}=(\mathbf{x}_t-\alpha_t\mathbf{x}_0)/\sigma_t$$:

$$
\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)
=\frac{\dot{\sigma}_t}{\sigma_t}\mathbf{x}
+\left(\dot{\alpha}_t
-\frac{\dot{\sigma}_t}{\sigma_t}\alpha_t\right)\mathbf{x}_0.
$$

For the straight interpolation $$\alpha_t=1-t$$ and $$\sigma_t=t$$, every conditional trajectory moves at the constant velocity $$\boldsymbol{\epsilon}-\mathbf{x}_0$$ from data to noise. Generation integrates the learned field in the reverse direction. With the more common noise-to-data time convention, the same path is written $$(1-t)\boldsymbol{\epsilon}+t\mathbf{x}_0$$ and its velocity changes sign.

## Conditional Velocities Produce the Correct Marginal Current

Suppose each conditional density $$p_t(\mathbf{x}\mid\mathbf{z})$$ satisfies a continuity equation with velocity $$\mathbf{u}_t(\mathbf{x}\mid\mathbf{z})$$. Its marginal is

$$
p_t(\mathbf{x})
=\int p_t(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

Differentiate, insert the conditional continuity equation, and move the spatial divergence outside the integral:

$$
\begin{aligned}
\partial_t p_t(\mathbf{x})
&=-\nabla_{\mathbf{x}}\cdot
\int \mathbf{u}_t(\mathbf{x}\mid\mathbf{z})
p_t(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z} \\
&=-\nabla_{\mathbf{x}}\cdot
\left[p_t(\mathbf{x})
\mathbb{E}[\mathbf{u}_t(\mathbf{x}\mid\mathbf{Z})
\mid\mathbf{X}_t=\mathbf{x}]\right].
\end{aligned}
$$

Therefore a valid marginal velocity is

$$
\mathbf{u}_t(\mathbf{x})
=\mathbb{E}\!\left[
\mathbf{u}_t(\mathbf{x}\mid\mathbf{Z})
\mid\mathbf{X}_t=\mathbf{x}\right].
$$

This is the flow analogue of the denoising score identity. Conditional paths contribute probability **currents**, $$p_t(\mathbf{x}\mid\mathbf{z})\mathbf{u}_t(\mathbf{x}\mid\mathbf{z})$$. Their sum is the marginal current. Dividing by the marginal density turns that current into a posterior-weighted average velocity.

{% include figure.liquid loading="eager" path="assets/img/blog/difffm_conditional_marginalization.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Many tractable conditional paths can overlap at the same state. Their probability-weighted currents add, so the marginal velocity is the posterior average \(u_t(x)=\mathbb{E}[u_t(x\mid Z)\mid X_t=x]\); conditional flow matching learns this average without computing the posterior." %}

The ideal flow-matching objective would regress the neural field $$\mathbf{u}_\theta$$ onto this marginal velocity, but the posterior average is intractable. The conditional flow-matching objective is directly sampleable:

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=\mathbb{E}_{t,\mathbf{z},\mathbf{x}\sim p_t(\cdot\mid\mathbf{z})}
\left[
\left\|\mathbf{u}_\theta(\mathbf{x},t)
-\mathbf{u}_t(\mathbf{x}\mid\mathbf{z})\right\|^2
\right].
$$

Again, squared error returns the conditional mean at fixed $$(\mathbf{x},t)$$. The conditional loss differs from marginal velocity regression only by a $$\theta$$-independent variance term, so their gradients are identical. This gives simulation-free training: draw endpoints and a time, construct $$\mathbf{x}_t$$ in one algebraic step, compute its conditional velocity, and regress. No ODE trajectory has to be integrated during training. Flow matching formalized this construction for broad Gaussian paths and continuous normalizing flows (<span id="cite-lipman2023"></span>[Lipman et al., 2023](#ref-lipman2023)).

“Simulation-free” applies to training, not generation. Sampling still requires integrating the learned ODE from noise to data. It also does not mean the learned marginal trajectories are straight. Even when every conditional path is a straight line, their posterior average can bend because the set of plausible endpoints changes along the route.

## The Common Core—and the Real Difference

The two regression identities can now be placed side by side:

$$
\underbrace{\nabla\log p_t(\mathbf{x})}_{\text{marginal score}}
=\mathbb{E}\!\left[
\underbrace{\nabla\log p_t(\mathbf{x}\mid\mathbf{X}_0)}_{\text{conditional score}}
\mid\mathbf{X}_t=\mathbf{x}\right],
$$

$$
\underbrace{\mathbf{u}_t(\mathbf{x})}_{\text{marginal velocity}}
=\mathbb{E}\!\left[
\underbrace{\mathbf{u}_t(\mathbf{x}\mid\mathbf{Z})}_{\text{conditional velocity}}
\mid\mathbf{X}_t=\mathbf{x}\right].
$$

Both models choose a conditional probability path whose local target is tractable. Both train a neural network on samples from that path. Both rely on conditional expectation to recover a global field that is never evaluated directly. The distinction lies in what that field means.

The score describes the instantaneous geometry of a density: it points in the direction of greatest log-density increase. Combined with the known forward coefficients, it defines a reverse SDE and a probability-flow ODE. A flow-matching velocity directly describes probability transport: it specifies where states move so that the continuity equation follows the chosen path. A score alone is not a velocity, and a velocity need not be a score gradient. Diffusion connects them through a specific dynamical construction.

This also explains why “diffusion versus flow matching” is not a clean opposition. Flow matching can use diffusion probability paths. A diffusion score model can sample through a deterministic probability-flow ODE. The training target and sampling dynamics are modular choices, although changing one can alter conditioning and numerical error.

## Schedules and Solvers Decide the Practical Model

The path schedule $$(\alpha_t,\sigma_t)$$ determines more than the pictures between the endpoints. It determines the conditional target magnitude, the signal-to-noise ratio seen by the network, how strongly different times are weighted, and whether the sampling dynamics become stiff near an endpoint. A schedule that changes almost nothing for most of the interval and then moves rapidly near $$t=0$$ forces a solver to resolve that narrow region. A straighter or more evenly parameterized path can often be traversed with fewer evaluations, but straight conditional paths do not guarantee a simple marginal field.

The neural parameterization changes the same conditioning. Score prediction divides the noise residual by $$\sigma_t$$ and can become large near the data endpoint. Noise prediction removes that division. Data prediction emphasizes reconstruction. Velocity parameterizations balance signal and noise differently. These are algebraically convertible only when the schedule is known; their optimization landscapes and finite-capacity errors are not identical.

At sampling time, the solver introduces another approximation:

- A reverse SDE sampler injects randomness at every step. Predictor–corrector or other stochastic schemes may improve exploration, but reproducibility and step allocation become additional choices.

- A probability-flow or flow-matching ODE is deterministic for a fixed initial noise sample. Higher-order or adaptive solvers can reduce discretization error, but every function evaluation calls the neural network.

- Fewer steps reduce cost but can skip regions where the vector field changes rapidly. More steps reduce integration error only until model error dominates.

{% include figure.liquid loading="eager" path="assets/img/blog/difffm_schedule_solver_tradeoff.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The schedule shapes where the probability path changes rapidly, while the numerical solver approximates motion along the learned field. Uneven or curved dynamics demand smaller or adaptive steps; increasing the number of network evaluations reduces discretization error but cannot repair an inaccurate learned target." %}

It is therefore misleading to attribute performance to the high-level framework alone. Noise schedule, time sampling, loss weighting, network preconditioning, target parameterization, and solver order interact. The diffusion design-space analysis of <span id="cite-karras2022"></span>[Karras et al., 2022](#ref-karras2022) is valuable precisely because it separates these components rather than treating a sampler as an indivisible recipe.

## The Lasting Unification

Diffusion and flow matching are best understood as two constructions of a learnable probability path. Diffusion chooses a forward stochastic process whose transition kernels are tractable. Denoising score matching turns their conditional scores into the marginal score, which supplies the missing reverse drift. Flow matching chooses tractable conditional paths directly. Conditional velocity regression turns their local motions into a marginal vector field, which supplies the ODE.

The central trick is the same: design a conditional object that is easy to sample and differentiate, then use regression to marginalize over the hidden condition. Once that trick is visible, the field stops looking mysterious. The remaining questions are concrete engineering and modeling decisions: which path exposes a well-conditioned target, which parameterization distributes error sensibly over time, and which solver follows the learned dynamics accurately enough for the available compute.

---

## References

<span id="ref-song2021"></span>Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS). *International Conference on Learning Representations*. [↩](#cite-song2021)

<span id="ref-ho2020"></span>Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html). *Advances in Neural Information Processing Systems, 33*. [↩](#cite-ho2020)

<span id="ref-vincent2011"></span>Vincent, P. (2011). [A Connection Between Score Matching and Denoising Autoencoders](https://doi.org/10.1162/NECO_a_00142). *Neural Computation, 23*(7), 1661–1674. [↩](#cite-vincent2011)

<span id="ref-lipman2023"></span>Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). [Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t). *International Conference on Learning Representations*. [↩](#cite-lipman2023)

<span id="ref-karras2022"></span>Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). [Elucidating the Design Space of Diffusion-Based Generative Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html). *Advances in Neural Information Processing Systems, 35*. [↩](#cite-karras2022)
