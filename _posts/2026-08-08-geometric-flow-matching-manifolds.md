---
layout: post
title: "Geometric Flow Matching on Manifolds"
date: 2026-08-08
last_updated: 2026-08-08
description: "Flow matching beyond Euclidean space, from tangent velocity fields and geodesic conditional paths to product manifolds for molecular geometry."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [geometric-deep-learning]
lecture_paths: [ml4mol, gdl]
tags: [flow-matching, riemannian-manifolds, geometric-generative-models, molecular-generation, equivariance]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the geometric flow-matching storyline from my
  Machine Learning for Molecules and Geometric Deep Learning lectures. The aim
  is not to turn differential geometry into a checklist, but to identify the few
  replacements that make ordinary flow matching valid on rotations, periodic
  angles, and other curved state spaces. The preceding chapter on
  <a href="{% post_url 2026-08-08-diffusion-models-flow-matching %}">diffusion models and flow matching</a>
  develops the Euclidean construction in more detail.</em>
</p>

Flow matching is unusually easy to describe in Euclidean space. Sample noise $$x_0$$ and data $$x_1$$, connect them by a straight line, and train a velocity field to follow that line. The construction depends on operations so familiar that they become invisible: subtracting two points, adding a vector to a point, and measuring a squared error with one global inner product.

None of those operations is globally available on a general manifold. Two rotations cannot be averaged entry by entry and still remain rotations. The difference between two angles must respect periodicity. A velocity attached to one point on a sphere does not automatically live in the tangent space of another point.

Riemannian flow matching preserves the logic of flow matching while replacing its hidden Euclidean assumptions. Velocities live in tangent spaces. The exponential and logarithmic maps replace addition and subtraction. Geodesics replace straight lines. The Riemannian metric measures velocity error, and Riemannian divergence describes how the flow transports density. This is a small conceptual change with large consequences for molecular and geometric generation.

## Euclidean flow matching learns a velocity field

Let $$p_0$$ be a tractable base distribution on $$\mathbb{R}^d$$ and $$p_1$$ the data distribution. A time-dependent vector field $$u_t:\mathbb{R}^d\to\mathbb{R}^d$$ defines the ODE

$$
\frac{d x_t}{dt}=u_t(x_t),
\qquad x_0\sim p_0.
$$

Its flow map $$\psi_t$$ pushes the base distribution forward to a probability path $$p_t=(\psi_t)_\#p_0$$. At the density level, the same evolution satisfies the continuity equation

$$
\partial_t p_t(x)
+
\nabla\cdot\left(p_t(x)u_t(x)\right)
=0.
$$

Directly constructing the marginal velocity $$u_t$$ is usually difficult. Flow matching instead chooses tractable conditional paths. For independently sampled endpoints $$x_0\sim p_0$$ and $$x_1\sim p_1$$, the simplest path is

$$
x_t=(1-t)x_0+t x_1,
\qquad
u_t(x_t\mid x_0,x_1)=x_1-x_0.
$$

Equivalently, when the current point and endpoint are known,

$$
u_t(x_t\mid x_1)=\frac{x_1-x_t}{1-t}.
$$

A neural field $$v_\theta(t,x)$$ is trained with the conditional flow-matching objective

$$
\mathcal{L}_{\mathrm{CFM}}(\theta)
=
\mathbb{E}
\left[
\left\lVert
v_\theta(t,x_t)
-u_t(x_t\mid x_0,x_1)
\right\rVert^2
\right].
$$

Although the target depends on a particular endpoint pair, averaging these targets at a fixed $$x_t$$ recovers the marginal velocity needed to transport the marginal path. This conditional-to-marginal identity is the central simplification of flow matching (<span id="cite-lipman2023"></span>[Lipman et al., 2023](#ref-lipman2023)).

## A manifold is locally linear, not globally linear

Let $$\mathcal{M}$$ be a smooth manifold. Around each point it resembles $$\mathbb{R}^d$$, but its global topology may be spherical, periodic, or more complicated. The valid instantaneous directions at $$x\in\mathcal{M}$$ form a vector space called the tangent space $$T_x\mathcal{M}$$.

A time-dependent manifold vector field therefore has type

$$
u_t(x)\in T_x\mathcal{M}.
$$

This dependence on $$x$$ is essential. A tangent vector is not merely a length-$$d$$ array; it is a direction attached to a particular base point. A manifold ODE asks for a curve $$\gamma:[0,1]\to\mathcal{M}$$ satisfying

$$
\dot\gamma(t)=u_t(\gamma(t))
\in T_{\gamma(t)}\mathcal{M}.
$$

The curve remains on the manifold because its velocity is always tangent. In an extrinsic implementation, a network may first emit an ambient vector $$\widetilde u_t(x)\in\mathbb{R}^D$$ and then project it:

$$
u_t(x)=\Pi_x\widetilde u_t(x),
$$

where $$\Pi_x$$ projects onto $$T_x\mathcal{M}$$. On the unit sphere,

$$
\Pi_x=\mathbf{I}-xx^{\mathsf T},
$$

because a tangent vector must be orthogonal to the radius $$x$$.

{% include figure.liquid loading="eager" path="assets/img/blog/geofm_euclidean_geodesic.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A Euclidean chord between two points on a sphere immediately leaves the data manifold. Geodesic interpolation instead follows a curve whose instantaneous velocity lies in the tangent space at every point. Original diagram." %}

A Riemannian metric $$g_x$$ supplies an inner product on each tangent space. It defines the local squared norm

$$
\lVert v\rVert_{g_x}^2=g_x(v,v)
$$

and therefore lengths of curves, angles, distances, and shortest paths. The metric is part of the model specification: changing it changes which paths count as straight and how errors along different directions are weighted.

## Exponential and logarithmic maps replace addition and subtraction

The exponential map starts from a point and a tangent velocity:

$$
\operatorname{Exp}_x:T_x\mathcal{M}\to\mathcal{M}.
$$

The point $$\operatorname{Exp}_x(v)$$ is reached after following the geodesic that begins at $$x$$ with velocity $$v$$ for unit time. In Euclidean space it reduces to $$\operatorname{Exp}_x(v)=x+v$$.

Locally, the logarithmic map reverses this operation:

$$
\operatorname{Log}_x(y)
\in T_x\mathcal{M},
\qquad
\operatorname{Exp}_x(\operatorname{Log}_x(y))=y.
$$

It is the geometric analogue of $$y-x$$. Given endpoint pair $$(x_0,x_1)$$, a constant-speed geodesic conditional path is

$$
x_t
=
\operatorname{Exp}_{x_0}
\left(t\operatorname{Log}_{x_0}(x_1)\right).
$$

At an intermediate point, its target velocity can be written

$$
u_t(x_t\mid x_1)
=
\frac{1}{1-t}
\operatorname{Log}_{x_t}(x_1).
$$

The factor $$1/(1-t)$$ does not imply that the velocity actually diverges along an exact constant-speed path: the remaining log displacement shrinks proportionally to $$1-t$$. Numerically, however, evaluating this quotient very near the endpoint can be ill-conditioned, so implementations often use equivalent endpoint-pair formulas or avoid sampling $$t=1$$ exactly.

{% include figure.liquid loading="lazy" path="assets/img/blog/geofm_exp_log.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The logarithmic map sends the target point to an initial velocity in the tangent space at the starting point, and the exponential map follows the associated geodesic back onto the manifold. The construction is local: at a cut locus, the shortest logarithm need not be unique. Original diagram." %}

The Riemannian conditional flow-matching loss is now almost forced:

$$
\mathcal{L}_{\mathrm{RCFM}}(\theta)
=
\mathbb{E}
\left[
g_{x_t}\left(
v_\theta(t,x_t)-u_t(x_t\mid x_0,x_1),
v_\theta(t,x_t)-u_t(x_t\mid x_0,x_1)
\right)
\right].
$$

Both vectors are compared in the same tangent space $$T_{x_t}\mathcal{M}$$. Chen and Lipman show how this construction extends flow matching to general geometries without computing density divergences during training (<span id="cite-chen2024"></span>[Chen & Lipman, 2024](#ref-chen2024)).

## Worked examples: a rotation and a periodic angle

For $$SO(3)$$, tangent directions may be represented by skew-symmetric matrices. Let $$R_0=I$$ and let $$R_1$$ be a rotation by angle $$\omega=2\pi/3$$ around the $$z$$-axis. Write

$$
\Omega
=
\omega
\begin{bmatrix}
0&-1&0\\
1&0&0\\
0&0&0
\end{bmatrix}.
$$

Then $$R_1=\exp(\Omega)$$ and the geodesic path is

$$
R_t=R_0\exp(t\Omega)=\exp(t\Omega).
$$

At $$t=1/2$$ the orientation has rotated by $$60^\circ$$, and every $$R_t$$ remains orthogonal with determinant one. Entrywise interpolation $$(1-t)I+tR_1$$ does not preserve either constraint. At angle $$\pi$$, however, the axis–angle logarithm reaches a cut locus: equivalent shortest representations make the choice of target velocity discontinuous.

For a torsion angle $$\phi\in S^1$$, the shortest signed displacement is

$$
\Delta(\phi_0,\phi_1)
=
\operatorname{atan2}
\left(
\sin(\phi_1-\phi_0),
\cos(\phi_1-\phi_0)
\right).
$$

The geodesic is

$$
\phi_t
=
\left(\phi_0+t\Delta\right)\bmod 2\pi.
$$

Take $$\phi_0=170^\circ$$ and $$\phi_1=-170^\circ$$. Ordinary subtraction suggests a $$-340^\circ$$ journey. The circular logarithm gives $$\Delta=20^\circ$$ and correctly crosses the periodic boundary. At exactly $$180^\circ$$ separation, clockwise and counterclockwise paths tie. Molecular conformer models exploit this toroidal geometry for collections of torsions (<span id="cite-jing2022"></span>[Jing et al., 2022](#ref-jing2022)).

## Probability still obeys a continuity equation

Let $$d\mathrm{vol}_g$$ be the Riemannian volume measure and let $$p_t$$ denote density with respect to that measure. Probability conservation becomes

$$
\partial_t p_t
+
\operatorname{div}_{\mathcal{M}}(p_tu_t)
=0.
$$

In local coordinates with metric matrix $$G(x)=[g_{ij}(x)]$$,

$$
\operatorname{div}_{\mathcal{M}} u
=
\frac{1}{\sqrt{\lvert G\rvert}}
\sum_i
\partial_{x^i}
\left(
\sqrt{\lvert G\rvert}
u^i
\right).
$$

The factor $$\sqrt{\lvert G\rvert}$$ corrects for how coordinate volume changes across the manifold. Omitting it treats chart coordinates as though they had uniform Euclidean volume, which generally gives the wrong density dynamics. Riemannian continuous normalizing flows use precisely this geometric change-of-variables structure (<span id="cite-mathieu2020"></span>[Mathieu & Nickel, 2020](#ref-mathieu2020)).

{% include figure.liquid loading="lazy" path="assets/img/blog/geofm_continuity.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A tangent vector field transports probability along the manifold, while Riemannian divergence measures local expansion or compression relative to the manifold's volume form. Conditional flow matching can train the velocity without evaluating this divergence, although likelihood computation still requires it. Original diagram." %}

This separates two tasks that are often conflated. **Sampling** only requires solving the manifold ODE. **Exact likelihood evaluation** requires integrating the Riemannian divergence along trajectories. Flow matching avoids divergence during training, not necessarily during density evaluation.

## Equivariance constrains the vector field

Manifold geometry and data symmetry are related but distinct. The fact that rotations live on $$SO(3)$$ describes the state space. The claim that rotating an entire molecule should rotate a generated structure describes an external group action.

Let a group element $$h$$ act on $$\mathcal{M}$$ by an isometry $$\rho_h$$. Its differential

$$
d\rho_h\big\rvert_x:
T_x\mathcal{M}
\to
T_{\rho_h(x)}\mathcal{M}
$$

maps tangent velocities between the appropriate tangent spaces. A vector field is equivariant when

$$
v_\theta(t,\rho_h(x))
=
d\rho_h\big\rvert_x
v_\theta(t,x).
$$

If the base distribution and conditional-path construction respect the same action, integrating this field commutes with the action. This is the continuous-time analogue of equivariant message passing. In protein generation, FoldFlow combines flow matching with the geometry and symmetry of residue frames in $$SE(3)$$ (<span id="cite-bose2024"></span>[Bose et al., 2024](#ref-bose2024)).

## Molecular state spaces are product manifolds

A molecule rarely lives on one homogeneous manifold. A residue frame contains a translation in $$\mathbb{R}^3$$ and an orientation in $$SO(3)$$. Side-chain torsions live on copies of $$S^1$$. A schematic protein state space is therefore

$$
\mathcal{M}
=
\left(\mathbb{R}^3\times SO(3)\right)^N
\times
\left(S^1\right)^K.
$$

Its tangent space factors componentwise:

$$
T_x\mathcal{M}
\cong
\prod_{i=1}^{N}
\left(\mathbb{R}^3\times\mathfrak{so}(3)\right)
\times\mathbb{R}^K.
$$

A product metric can weight these parts:

$$
\lVert u\rVert_g^2
=
w_x\lVert u_x\rVert^2
+w_R\lVert u_R\rVert^2
+w_\phi\lVert u_\phi\rVert^2.
$$

{% include figure.liquid loading="lazy" path="assets/img/blog/geofm_product_manifold.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A molecular state can combine Euclidean translations, rotational frames, and periodic torsions in one product manifold. The product metric's weights determine the relative training geometry of errors measured in length, rotation angle, and torsional angle. Original diagram." %}

Those weights are not mere loss hyperparameters. They define relative path length and therefore change the regression target emphasized by flow matching. A one-radian orientation error and a one-ångström translation error have no canonical exchange rate. The choice should reflect coordinate scaling, downstream structure quality, and solver behavior.

The endpoint coupling matters too. Independent pairs yield simple training, but their geodesics may cross low-density regions or take unnecessarily long paths. Approximate optimal-transport couplings can shorten conditional paths and reduce target variance, at the cost of solving a batch-level matching problem. On products, the metric weights directly affect that coupling.

## Topology and numerics are part of the model

The geodesic formula is elegant, but several complications determine whether it works in practice.

**There may be no global logarithm.** On a sphere, the antipode of $$x$$ has infinitely many shortest geodesics from $$x$$. On $$SO(3)$$, rotations by angle $$\pi$$ sit at the analogous ambiguity. A chosen principal logarithm is discontinuous across the cut locus. These sets may have zero volume, yet samples near them can still produce large target variation.

**The base distribution is geometry-dependent.** There is no universal manifold Gaussian. Compact homogeneous spaces admit a uniform base; other settings use wrapped distributions, heat kernels, or a distribution defined through charts or an embedding. The base affects both coverage and how difficult the learned transport must be.

**A generic ODE step can leave the manifold.** The geometric Euler update

$$
x_{t+\Delta t}
\approx
\operatorname{Exp}_{x_t}
\left(\Delta t\,v_\theta(t,x_t)\right)
$$

respects the manifold by construction. Exact exponential maps can be expensive, so implementations use retractions, projection, Lie-group integrators, or chart-based solvers. Projection is simple but can distort the intended vector field; charts can introduce singularities; repeated matrix exponentials can dominate runtime.

**Curvature limits comfortable step sizes.** A tangent-space linearization is only local. Large solver steps are less reliable where curvature is high or the vector field changes rapidly. Adaptive solvers reduce error but make the number of neural evaluations unpredictable.

**Topology cannot be wished away by coordinates.** Unwrapping an angle to the real line introduces an artificial boundary. Representing a rotation by a quaternion introduces a double cover, because $$q$$ and $$-q$$ describe the same rotation. Every parameterization trades one difficulty for another; the model, loss, and solver must agree on that choice.

## The point of the geometry

Geometric flow matching is not ordinary flow matching followed by a projection at the end. The geometry determines the valid velocity at every time, the conditional path used as supervision, the norm in the loss, the volume form in the continuity equation, and the numerical update used for sampling.

The clean recipe is therefore short:

1. Specify the actual state manifold and its symmetries.
2. Produce a tangent, equivariant velocity field.
3. Construct conditional paths with exponential and logarithmic maps—or a valid alternative when shortest geodesics are unsuitable.
4. Measure velocity errors with the chosen metric.
5. Integrate with a solver that respects the manifold.

What makes the problem interesting is everything hidden inside “chosen”: the metric on a product space, the branch of the logarithm, the base distribution, the endpoint coupling, and the compromise between exact geometry and affordable simulation. Flow matching removes the need to simulate a diffusion process during training. It does not remove the topology of the space being generated.

## References

<ol class="bibliography">
  <li id="ref-lipman2023">Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). <a href="https://openreview.net/forum?id=PqvMRDCJT9t">Flow Matching for Generative Modeling</a>. <em>ICLR</em>. <a href="#cite-lipman2023">↩</a></li>
  <li id="ref-chen2024">Chen, R. T. Q., & Lipman, Y. (2024). <a href="https://openreview.net/forum?id=g7ohDlTITL">Flow Matching on General Geometries</a>. <em>ICLR</em>. <a href="#cite-chen2024">↩</a></li>
  <li id="ref-mathieu2020">Mathieu, E., & Nickel, M. (2020). <a href="https://proceedings.neurips.cc/paper_files/paper/2020/hash/1aa3d9c6ce672447e1e5d0f1b5207e85-Abstract.html">Riemannian Continuous Normalizing Flows</a>. <em>NeurIPS</em>. <a href="#cite-mathieu2020">↩</a></li>
  <li id="ref-bose2024">Bose, A. J. et al. (2024). <a href="https://openreview.net/forum?id=kJFIH23hXb">SE(3)-Stochastic Flow Matching for Protein Backbone Generation</a>. <em>ICLR</em>. <a href="#cite-bose2024">↩</a></li>
  <li id="ref-jing2022">Jing, B., Corso, G., Chang, J., Barzilay, R., & Jaakkola, T. (2022). <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/994545b2308bbbbc97e3e687ea9e464f-Abstract-Conference.html">Torsional Diffusion for Molecular Conformer Generation</a>. <em>NeurIPS</em>. <a href="#cite-jing2022">↩</a></li>
</ol>

---

*Figure provenance.* All four `geofm_` diagrams are original SVG illustrations generated by `scripts/generate_geofm_figures.py`. They synthesize standard Riemannian geometry and flow-matching constructions described in the cited primary literature; no third-party artwork is reproduced.
