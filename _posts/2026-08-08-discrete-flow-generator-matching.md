---
layout: post
title: "Discrete Flow and Generator Matching"
date: 2026-08-08
last_updated: 2026-08-08
description: "How continuous-time Markov chains transport categorical probability, how conditional paths make their rates learnable, and how generator matching extends the construction across modalities."
abstract: >
  A discrete state has no infinitesimal displacement, but its probability can still move continuously in time. The right analogue of a velocity field is a jump generator: a collection of rates whose master equation transports mass between states.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [generative-modeling]
lecture_paths: [ml4mol, gdl]
tags: [continuous-time-markov-chains, discrete-flow-matching, generator-matching, discrete-generative-models, molecular-generation]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the discrete-flow storyline from my 2025 Machine Learning for Molecules and Geometric Deep Learning lectures. For the continuous counterpart—velocity fields, probability flux, and probability-flow ODEs—see <a href="{% post_url 2026-08-08-odes-sdes-probability-flow %}">ODEs, SDEs, and Probability Flow</a>.</em>
</p>

A continuous flow moves a point by an infinitesimal vector. That picture breaks on a categorical state space. There is no state halfway between carbon and nitrogen, no amino acid infinitesimally close to glycine, and no fractional bond order hiding between two graph categories.

Probability can nevertheless evolve continuously in time. The state remains discrete, but the *chance of jumping* changes smoothly. A continuous-time Markov chain replaces velocity by a collection of transition rates. Its master equation replaces the continuity equation. Once that replacement is made, much of flow matching survives: choose a tractable path of probability distributions, construct simple dynamics conditioned on a data endpoint, and regress their posterior average.

The generator is the object that makes this analogy precise. On a finite state space it is a rate matrix. On Euclidean or geometric spaces it becomes a differential operator. Generator matching therefore does more than rename discrete flow matching. It isolates the common infinitesimal structure behind ODEs, diffusions, jumps, and mixtures of them.

## A rate is an infinitesimal probability of jumping

Let $$\mathcal{S}$$ be a finite state space and let $$X_t\in\mathcal{S}$$. For two distinct states $$x,y\in\mathcal{S}$$, define the time-dependent jump rate

$$
q_t(y\mid x)\geq 0,
\qquad y\neq x.
$$

Our notation places the destination first: $$q_t(y\mid x)$$ is the rate of jumping *from* $$x$$ *to* $$y$$. Over a short interval of length $$h>0$$,

$$
\Pr(X_{t+h}=y\mid X_t=x)
=h q_t(y\mid x)+o(h),
\qquad y\neq x.
$$

The total exit rate from $$x$$ is

$$
\lambda_t(x)=\sum_{y\neq x}q_t(y\mid x).
$$

Consequently, the probability of staying at $$x$$ is

$$
\Pr(X_{t+h}=x\mid X_t=x)
=1-h\lambda_t(x)+o(h).
$$

It is convenient to define the diagonal entry as

$$
q_t(x\mid x)=-\lambda_t(x).
$$

The matrix $$Q_t$$ with entries $$(Q_t)_{yx}=q_t(y\mid x)$$ then has nonnegative off-diagonal entries and columns that sum to zero. Some references put sources in rows and write $$p_tQ_t$$ instead. Nothing substantive changes, but mixing the two conventions silently transposes every equation.

{% include figure.liquid loading="eager" path="assets/img/blog/discgm_ctmc_master_equation.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A continuous-time Markov chain assigns an instantaneous rate to each permitted jump. The diagonal generator entry is minus the sum of outgoing rates, ensuring that the short-time transition probabilities sum to one. Original diagram." %}

Let $$p_t(x)=\Pr(X_t=x)$$ and regard $$p_t$$ as a column vector. Probability obeys the **master equation**, also called the Kolmogorov forward equation:

$$
\frac{d}{dt}p_t(y)
=\sum_{x\in\mathcal{S}}q_t(y\mid x)p_t(x).
$$

Separating the diagonal term exposes the probability balance:

$$
\frac{d}{dt}p_t(y)
=\underbrace{\sum_{x\neq y}q_t(y\mid x)p_t(x)}_{\text{inflow}}
-\underbrace{\lambda_t(y)p_t(y)}_{\text{outflow}}.
$$

In matrix notation, $$\dot p_t=Q_t p_t$$. This is the discrete analogue of a continuity equation. A continuous velocity transports density through spatial flux; a generator transports probability through jumps between named states.

## The generator describes what every observable does next

A rate matrix is useful on a finite set, but the more durable definition acts on test functions. For any observable $$f:\mathcal{S}\to\mathbb{R}$$, define

$$
(\mathcal{L}_t f)(x)
=\lim_{h\downarrow 0}
\frac{\mathbb{E}[f(X_{t+h})\mid X_t=x]-f(x)}{h}.
$$

Substituting the short-time transition law gives

$$
(\mathcal{L}_t f)(x)
=\sum_{y\neq x}q_t(y\mid x)\bigl[f(y)-f(x)\bigr].
$$

The generator answers a local question: if the process is currently at $$x$$, how quickly will the expected value of *any* observable change? The master equation is its dual statement about probability. With

$$
\langle p,f\rangle
=\sum_{x\in\mathcal{S}}p(x)f(x),
$$

the weak forward equation is

$$
\frac{d}{dt}\langle p_t,f\rangle
=\langle p_t,\mathcal{L}_t f\rangle.
$$

Writing $$\mathcal{L}_t^*$$ for the adjoint yields $$\partial_t p_t=\mathcal{L}_t^*p_t$$. On a finite state space, this is exactly $$\dot p_t=Q_tp_t$$. For an ODE, the generator is $$\mathcal{L}_t f=\nabla f^{\mathsf T}v_t$$. For a diffusion, it adds a Hessian term. For a jump process, it sums or integrates finite differences. Generator matching begins from this operator-level identity (<span id="cite-holderrieth2025"></span>[Holderrieth et al., 2025](#ref-holderrieth2025)).

## A mask-to-token chain is a complete worked example

Consider the three-state space

$$
\mathcal{S}=\{M,A,B\},
$$

where $$M$$ is a mask token. Let the data endpoint $$Z$$ equal $$A$$ with probability $$0.6$$ and $$B$$ with probability $$0.4$$. We want a path that begins fully masked and ends at this data distribution. Choose an increasing schedule $$\kappa_t$$ satisfying $$\kappa_0=0$$ and $$\kappa_1=1$$, and define the endpoint-conditioned path

$$
p_t(x\mid Z=z)
=(1-\kappa_t)\,\delta_M(x)
+\kappa_t\,\delta_z(x).
$$

Conditioned on $$z$$, a chain realizes this path by jumping from $$M$$ to $$z$$ at rate

$$
r_t=\frac{\dot\kappa_t}{1-\kappa_t},
$$

with $$z$$ absorbing. To verify the construction, let $$s_t$$ be the conditional probability of still being masked. The master equation gives

$$
\dot s_t=-r_ts_t.
$$

Since $$s_t=1-\kappa_t$$,

$$
-r_ts_t
=-\frac{\dot\kappa_t}{1-\kappa_t}(1-\kappa_t)
=-\dot\kappa_t,
$$

which is exactly $$d(1-\kappa_t)/dt$$.

Marginalizing over $$Z$$ gives

$$
p_t(M)=1-\kappa_t,
\qquad
p_t(A)=0.6\kappa_t,
\qquad
p_t(B)=0.4\kappa_t.
$$

While the current state is $$M$$, its endpoint remains hidden. The marginal chain averages the two conditional possibilities:

$$
q_t(A\mid M)=0.6r_t,
\qquad
q_t(B\mid M)=0.4r_t.
$$

Take the linear schedule $$\kappa_t=t$$. At $$t=0.5$$, $$r_t=2$$ and the current marginal is

$$
p_{0.5}=(0.5,0.3,0.2)^{\mathsf T}
$$

in the order $$(M,A,B)$$. A forward Euler step of size $$h=0.1$$ gives

$$
p_{0.6}
\approx p_{0.5}+h\dot p_{0.5}
=(0.5,0.3,0.2)^{\mathsf T}
+0.1(-1,0.6,0.4)^{\mathsf T}
=(0.4,0.36,0.24)^{\mathsf T}.
$$

This matches the designed path at $$\kappa_{0.6}=0.6$$. The example also exposes a numerical issue: $$r_t=1/(1-t)$$ diverges near the endpoint. The integrated hazard is correct—the surviving masked mass shrinks exactly like $$1-t$$—but a fixed-step sampler must shorten its step or force the final unmasking.

## Conditional paths make an intractable marginal generator learnable

The toy calculation scales because it has a general form. Draw a target endpoint $$Z\sim p_{\mathrm{data}}$$ and choose tractable conditional paths $$p_t(x\mid z)$$. Their mixture defines the desired marginal path:

$$
p_t(x)
=\sum_z p_{\mathrm{data}}(z)p_t(x\mid z).
$$

Suppose $$q_t^z(y\mid x)$$ generates the conditional path for endpoint $$z$$. Then the marginal rate is the posterior average

$$
q_t(y\mid x)
=\sum_z q_t^z(y\mid x)p_t(z\mid x)
=\mathbb{E}\!\left[q_t^Z(y\mid x)\mid X_t=x\right].
$$

The identity follows directly from the master equation:

$$
\begin{aligned}
\dot p_t(y)
&=\sum_z p_{\mathrm{data}}(z)
  \sum_x q_t^z(y\mid x)p_t(x\mid z)\\
&=\sum_x p_t(x)
  \sum_z q_t^z(y\mid x)p_t(z\mid x).
\end{aligned}
$$

The posterior $$p_t(z\mid x)$$ is usually intractable, which is exactly why the marginal rates cannot be written down. Training avoids evaluating it. We sample $$Z$$ from data, sample $$X_t$$ from the tractable conditional path, and regress the known conditional rate target.

{% include figure.liquid loading="eager" path="assets/img/blog/discgm_conditional_marginalization.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Each endpoint defines a simple conditional probability path and a known conditional generator. Averaging those rates under the posterior over endpoints produces the marginal generator; conditional training estimates that average without computing the marginal density. Original diagram." %}

Let $$q_\theta(\cdot\mid x,t)$$ denote the network's vector of outgoing rates. With a Bregman divergence $$D$$, the conditional discrete flow-matching objective has the form

$$
\mathcal{J}_{\mathrm{CDFM}}(\theta)
=\mathbb{E}_{t,Z,X_t}
\left[
D\!\left(q_t^Z(\cdot\mid X_t),
q_\theta(\cdot\mid X_t,t)\right)
\right].
$$

The Bregman conditional-mean property makes its population minimizer the posterior mean of the conditional rates—the marginal rate above. More strongly, the conditional and inaccessible marginal objectives have the same expected parameter gradient under the usual differentiability conditions. This is the discrete counterpart of conditional flow matching and is the central construction in discrete flow matching (<span id="cite-gat2024"></span>[Gat et al., 2024](#ref-gat2024)).

The network must still produce a valid generator. A common parameterization applies a nonnegative map such as softplus to every off-diagonal rate, then sets the diagonal to minus their sum. Predicting a normalized categorical distribution is not enough: the exit rate controls *when* a jump occurs, while normalized off-diagonal rates control *where* it goes.

## Reverse-time rates are discrete density ratios

Discrete diffusion often starts with a known forward corruption chain and learns to reverse it (<span id="cite-campbell2022"></span>[Campbell et al., 2022](#ref-campbell2022)). Suppose the forward process has rate $$q_t(y\mid x)$$ and marginal $$p_t$$. For distinct states with positive marginal probability, the reverse-clock rate at the corresponding forward time is

$$
q_t^{\mathrm{rev}}(x\mid y)
=q_t(y\mid x)\frac{p_t(x)}{p_t(y)}.
$$

The formula follows by comparing the infinitesimal joint probability of seeing $$x$$ before a jump and $$y$$ after it:

$$
p_t(x)q_t(y\mid x)h
=p_t(y)q_t^{\mathrm{rev}}(x\mid y)h+o(h).
$$

This is the discrete analogue of the score correction in a reverse SDE. A continuous score is a local log-density derivative; reverse CTMC rates contain ratios of neighboring state probabilities. Both modify a known forward dynamic using information about the current marginal.

The formula also identifies the fragile cases. Reverse rates can become large when $$p_t(y)$$ is small, and they are undefined outside the marginal support. Practical models choose corruption paths with adequate support, predict denoising posteriors rather than raw density ratios, and regularize or clip numerically extreme rates.

Discrete flow matching can instead build a source-to-data path directly. The two views are compatible: one emphasizes reversing a noising generator, while the other emphasizes constructing a generating path from conditional bridges. Their learned objects are still rates satisfying the same master equation.

## Factorized jumps avoid an exponential output space

A sequence of length $$d$$ over an alphabet $$\mathcal{A}$$ has $$|\mathcal{A}|^d$$ states. A generator with an unrestricted rate between every pair of sequences is impossible to represent. The practical restriction is to permit one coordinate to change at a time:

$$
q_t(y\mid x)
=\sum_{i=1}^{d}
\mathbf{1}\{y^{-i}=x^{-i}\}
q_t^i(y^i\mid x).
$$

Here $$x^{-i}$$ denotes every coordinate except $$i$$. The output now scales as $$d|\mathcal{A}|$$ rather than $$|\mathcal{A}|^{2d}$$. This does **not** make the model independent across coordinates: each rate $$q_t^i(\cdot\mid x)$$ can inspect the entire current sequence through a Transformer or graph network. The factorization restricts the elementary event, not the context used to choose it.

{% include figure.liquid loading="eager" path="assets/img/blog/discgm_factorized_molecules.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="High-dimensional discrete generators become tractable when each elementary event edits one token, node label, or edge label. A shared context network can still couple all coordinates, while the rate output grows only linearly with the number of categorical variables. Original diagram." %}

The restriction has a real cost. A tightly coupled edit—changing an atom type and several incident bonds—must occur through a sequence of intermediate graphs. Those intermediates may be chemically invalid or simply inefficient. One can add structured multi-coordinate events, constrain permitted transitions, or use corrector dynamics, but every richer event family increases the rate space and simulation cost.

Molecular graphs add symmetry to this tradeoff. Atom and bond rates must be equivariant to node relabeling, and variable-size generation needs explicit birth, death, padding, or masking conventions. Discrete flow matching has been adapted to graph generation while respecting these symmetries (<span id="cite-qin2025"></span>[Qin et al., 2025](#ref-qin2025)). The master equation itself does not guarantee chemical validity; validity enters through state representation, architecture, allowed jumps, training data, and possibly guidance.

Protein co-design reveals a second advantage. A protein combines discrete amino-acid identities with continuous translations and rotations. MultiFlow assigns a CTMC generator to sequence and geometric flows to structure, then learns them jointly (<span id="cite-campbell2024"></span>[Campbell et al., 2024](#ref-campbell2024)). This is difficult to express as a single vector field because the modalities do not share one tangent space. It is natural as a sum of generators on a product state space.

## Generator matching unifies rather than flattens modalities

Suppose a state contains continuous coordinates, orientations, and categorical labels. Each component may have its own generator:

$$
\mathcal{L}_t
=\mathcal{L}_t^{\mathrm{flow}}
+\mathcal{L}_t^{\mathrm{diffusion}}
+\mathcal{L}_t^{\mathrm{jump}}.
$$

Because the weak forward equation is linear in $$\mathcal{L}_t$$, the sum is again a valid generator under suitable regularity conditions. The terms need not ignore one another: their coefficients may all condition on the full multimodal state. A sequence jump rate can depend on 3D geometry, while a geometric drift can depend on amino-acid identity.

Conditional generator matching mirrors the discrete derivation. Construct a conditional generator $$\mathcal{L}_t^z$$ that transports a simple source toward endpoint $$z$$. The marginal generator acts as the posterior average

$$
(\mathcal{L}_t f)(x)
=\mathbb{E}\!\left[
(\mathcal{L}_t^Z f)(x)
\mid X_t=x
\right].
$$

When the generator depends linearly on learnable coefficients—velocity, diffusion tensor, or jump kernel—conditional regression learns their marginal posterior average. The framework preserves what is different about each modality instead of embedding everything into a continuous surrogate and pretending every change is a vector displacement.

There is also non-uniqueness. Many generators can trace the same marginal probability path, just as multiple continuous velocity fields can share a continuity equation. Adding a component whose adjoint annihilates $$p_t$$ changes trajectories without changing the prescribed marginals. This freedom supports correctors and generator superposition, but it also means that matching marginals alone does not determine pathwise behavior, event count, or simulation efficiency.

## Simulation turns rates into an algorithmic tradeoff

Once rates are learned, there is no single obligatory sampler.

An **event-driven sampler** uses the total exit rate $$\lambda_t(x)$$. For time-homogeneous rates, the waiting time is exponential with parameter $$\lambda(x)$$, and the destination is drawn with probability

$$
\Pr(Y=y\mid X=x,\text{jump})
=\frac{q(y\mid x)}{\lambda(x)}.
$$

This Gillespie-style simulation represents individual events exactly. Time-dependent neural rates require inversion, thinning, or local approximations, and every accepted event may require another network evaluation. The method is accurate but inherently sequential.

A **fixed-step sampler** chooses $$h$$ and uses

$$
\Pr(X_{t+h}=y\mid X_t=x)
\approx
\begin{cases}
hq_t(y\mid x), & y\neq x,\\
1-h\lambda_t(x), & y=x.
\end{cases}
$$

Validity requires $$h\lambda_t(x)\leq 1$$. Smaller steps reduce first-order bias but increase the number of network evaluations. Updating several coordinates in parallel is fast on modern hardware, yet two nominally simultaneous changes omit the within-step dependence that a true CTMC would resolve sequentially. The resulting error is order $$h^2$$ locally but can matter when rates are large or molecular constraints are tight.

{% include figure.liquid loading="eager" path="assets/img/blog/discgm_simulation_tradeoffs.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Event-driven simulation resolves irregular jump times and is exact for a fixed generator, but it is sequential. Fixed-step simulation admits batched or parallel updates, but introduces discretization bias and must keep each step small relative to the total exit rate. Original diagram." %}

Adaptive steps, tau-leaping, predictor-corrector updates, and schedule redesign occupy the space between these extremes. The best choice depends on more than asymptotic correctness. Network evaluations, parallelism, endpoint stiffness, event sparsity, and the cost of invalid intermediate structures all matter.

The central idea is therefore not that discrete data should imitate continuous trajectories. It is that both can be described infinitesimally. A vector field says how a point moves. A jump generator says how probability leaves one state and enters another. The master equation translates those local rates into a global probability path. Conditional paths make the otherwise intractable marginal generator learnable, and generator matching extends the construction without erasing the distinct geometry of each modality.

---

## References

<ol class="bibliography">
  <li id="ref-campbell2022">Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deligiannidis, G., &amp; Doucet, A. (2022). <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html">A continuous time framework for discrete denoising models</a>. <em>Advances in Neural Information Processing Systems</em>, 35. <a href="#cite-campbell2022">↩</a></li>
  <li id="ref-campbell2024">Campbell, A., Yim, J., Barzilay, R., Rainforth, T., &amp; Jaakkola, T. (2024). <a href="https://proceedings.mlr.press/v235/campbell24a.html">Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design</a>. <em>Proceedings of the 41st International Conference on Machine Learning</em>, 5453–5512. <a href="#cite-campbell2024">↩</a></li>
  <li id="ref-gat2024">Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G., Adi, Y., &amp; Lipman, Y. (2024). <a href="https://openreview.net/forum?id=GTDKo3Sv9p">Discrete flow matching</a>. <em>Advances in Neural Information Processing Systems</em>, 37. <a href="#cite-gat2024">↩</a></li>
  <li id="ref-holderrieth2025">Holderrieth, P., Havasi, M., Yim, J., Shaul, N., Gat, I., Jaakkola, T., Karrer, B., Chen, R. T. Q., &amp; Lipman, Y. (2025). <a href="https://arxiv.org/abs/2410.20587">Generator matching: Generative modeling with arbitrary Markov processes</a>. <em>International Conference on Learning Representations</em>. <a href="#cite-holderrieth2025">↩</a></li>
  <li id="ref-qin2025">Qin, Y., Madeira, M., Thanou, D., &amp; Frossard, P. (2025). <a href="https://proceedings.mlr.press/v267/qin25d.html">DeFoG: Discrete flow matching for graph generation</a>. <em>Proceedings of the 42nd International Conference on Machine Learning</em>, 50269–50326. <a href="#cite-qin2025">↩</a></li>
</ol>
