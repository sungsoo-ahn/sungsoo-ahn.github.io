---
layout: post
title: "Equivariant Transformers and Machine-Learned Potentials"
date: 2026-08-08
last_updated: 2026-08-08
description: "How invariant attention scores, equivariant values, and energy-based force prediction turn geometric Transformers into practical interatomic potentials."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [geometric-deep-learning]
lecture_paths: [ml4mol, gdl]
tags: [equivariant-transformers, attention, interatomic-potentials, molecular-dynamics, force-fields]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the equivariant-attention and interatomic-potential
  storyline from my Machine Learning for Molecules and Geometric Deep Learning
  lectures. The central distinction is between architectural symmetry and a
  simulation-ready potential: the former constrains transformations, while the
  latter must also be conservative, smooth, scalable, and physically adequate
  over the configurations it will visit.</em>
</p>

Attention has an appealing interpretation for atomistic systems: an atom should decide which neighbors matter before combining what they say. Geometry makes that sentence harder than it sounds. A Transformer's attention weight is a scalar, while its hidden features may be vectors and higher-order spherical tensors. An arbitrary query–key comparison can silently destroy rotation equivariance.

The resolution is a useful design principle: **routing must be invariant, while content may be equivariant**. Queries and keys are built as typed geometric features, but their compatibility is reduced to an invariant scalar. That scalar can safely modulate an equivariant value. This principle connects the SE(3)-Transformer to Equiformer and its descendants.

Yet a model that respects rotations is not automatically a good interatomic potential. Molecular dynamics asks for more: a scalar energy that is extensive, forces that are derivatives of that energy, smooth behavior as neighbors cross a cutoff, enough range to capture the relevant physics, and low enough cost to evaluate for millions of time steps. The real design problem begins where equivariance ends.

## Attention has to respect feature types

Let atom $$i$$ carry a collection of spherical features

$$
\mathbf{h}_i
=
\bigoplus_{\ell=0}^{L}
\mathbf{h}_i^{(\ell)},
\qquad
\mathbf{h}_i^{(\ell)}
\mapsto
\mathbf{D}^{(\ell)}(\mathbf{R})
\mathbf{h}_i^{(\ell)}
$$

under a rotation $$\mathbf{R}\in SO(3)$$. Type $$\ell=0$$ features are scalars, type $$\ell=1$$ features rotate as vectors, and higher types carry finer angular structure. The relative position

$$
\mathbf{r}_{ij}=\mathbf{r}_j-\mathbf{r}_i
$$

removes global translation. Radial functions of $$r_{ij}=\lVert\mathbf{r}_{ij}\rVert$$ and spherical harmonics of $$\widehat{\mathbf{r}}_{ij}$$ then provide invariant and equivariant edge information.

Ordinary dot-product attention begins with queries, keys, and values. The geometric version keeps the same roles, but every object has a type:

$$
\begin{aligned}
\mathbf{q}_i^{(\ell)}
&= \mathbf{W}_Q^{(\ell)}\mathbf{h}_i^{(\ell)},\\
\mathbf{k}_{ij}^{(\ell)}
&= \Phi_K^{(\ell)}
\left(\mathbf{h}_j, r_{ij},
\mathbf{Y}(\widehat{\mathbf{r}}_{ij})\right),\\
\mathbf{v}_{ij}^{(\ell)}
&= \Phi_V^{(\ell)}
\left(\mathbf{h}_j, r_{ij},
\mathbf{Y}(\widehat{\mathbf{r}}_{ij})\right).
\end{aligned}
$$

Here $$\Phi_K$$ and $$\Phi_V$$ are equivariant maps, typically assembled from tensor products, radial networks, and type-preserving linear layers. Under rotation, each query, key, and value block transforms through the same $$\mathbf{D}^{(\ell)}(\mathbf{R})$$ associated with its type.

The attention logit must not transform at all. One valid construction is

$$
s_{ij}
=
\sum_{\ell=0}^{L}
\frac{1}{\sqrt{d_\ell}}
\left\langle
\mathbf{q}_i^{(\ell)},
\mathbf{k}_{ij}^{(\ell)}
\right\rangle,
\qquad
\alpha_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{k\in\mathcal{N}(i)}\exp(s_{ik})}.
$$

In a real orthonormal irrep basis, Wigner matrices are orthogonal. Therefore

$$
\left\langle
\mathbf{D}^{(\ell)}(\mathbf{R})\mathbf{q},
\mathbf{D}^{(\ell)}(\mathbf{R})\mathbf{k}
\right\rangle
=
\mathbf{q}^{\mathsf T}
\mathbf{D}^{(\ell)}(\mathbf{R})^{\mathsf T}
\mathbf{D}^{(\ell)}(\mathbf{R})
\mathbf{k}
=
\langle\mathbf{q},\mathbf{k}\rangle.
$$

The logits and softmax weights are invariant. The value aggregation

$$
\mathbf{m}_i^{(\ell)}
=
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij}\mathbf{v}_{ij}^{(\ell)}
$$

is consequently equivariant: multiplying a rotating feature by an invariant scalar changes its magnitude, not its transformation law.

{% include figure.liquid loading="eager" path="assets/img/blog/eqpot_typed_attention.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Geometric attention uses typed queries and keys to produce a scalar compatibility score, then uses that score to route typed values. Because the attention weight is invariant, every weighted value retains its original transformation law. Original diagram." %}

This is the essential move in the SE(3)-Transformer (<span id="cite-fuchs2020"></span>[Fuchs et al., 2020](#ref-fuchs2020)). The architecture is more elaborate than these equations—its keys and values are steerable convolutions—but the proof reduces to the same sentence: invariant weights modulate equivariant values.

## A worked rotation example

The invariance of a typed inner product is easiest to see for type-1 features. Take a two-dimensional slice of three-dimensional space and set

$$
\mathbf{q}=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\mathbf{k}_1=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\mathbf{k}_2=\begin{bmatrix}0\\1\end{bmatrix}.
$$

The logits are $$s_1=1$$ and $$s_2=0$$, so

$$
(\alpha_1,\alpha_2)
=
\operatorname{softmax}(1,0)
\approx
(0.73,0.27).
$$

Now rotate everything by $$90^\circ$$ with

$$
\mathbf{R}
=
\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
$$

The query and keys change direction, but

$$
(\mathbf{R}\mathbf{q})^{\mathsf T}
(\mathbf{R}\mathbf{k}_a)
=
\mathbf{q}^{\mathsf T}\mathbf{k}_a
$$

for either key $$a$$. The attention weights stay at $$(0.73,0.27)$$. If the values are vectors, their weighted sum rotates by $$\mathbf{R}$$. The model makes the same relational choice in the new frame, while its geometric output follows the frame.

{% include figure.liquid loading="lazy" path="assets/img/blog/eqpot_rotation_example.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A type-1 query is aligned with the first key and orthogonal to the second, giving logits (1) and (0). Rotating query, keys, and vector values together leaves the softmax weights unchanged while rotating the aggregated output. Original diagram." %}

Inner-product attention is not the only possibility. Equiformer generates attention logits with an invariant nonlinear network after equivariant tensor-product processing, which gives the scoring function more freedom than a single dot product (<span id="cite-liao2023"></span>[Liao & Smidt, 2023](#ref-liao2023)). The non-negotiable requirement is not a particular formula; it is that the final logit belongs to type $$0$$.

## Architectures differ in where they spend geometric computation

Once invariant routing is available, the next question is how much angular structure to put in the keys and values, and how efficiently to compute it.

The **SE(3)-Transformer** starts from steerable convolutions and adds data-dependent invariant attention. It provides the conceptual template: values carry geometry, weights select neighbors. Its tensor-product operations become expensive as the maximum degree and channel count grow.

**Equiformer** recasts more of the Transformer stack—attention, normalization, feed-forward updates—as operations on irreducible features. It uses nonlinear invariant attention and depth-wise tensor products, analogous to depth-wise convolution, to reduce channel mixing cost. **EquiformerV2** then imports the edge-alignment idea from eSCN and improves normalization and nonlinear processing for higher-degree features (<span id="cite-liao2024"></span>[Liao et al., 2024](#ref-liao2024)).

The key efficiency result behind **eSCN** is geometric rather than approximate. Align the local coordinate axis with an edge, perform the expensive coupling in that frame, and the $$SO(3)$$ convolution becomes a much sparser $$SO(2)$$ computation. Passaro and Zitnick reduce the dependence on maximum degree from $$O(L^6)$$ to $$O(L^3)$$ for the relevant tensor products (<span id="cite-passaro2023"></span>[Passaro & Zitnick, 2023](#ref-passaro2023)). Higher angular resolution becomes practical, although rotations into and out of edge frames still cost time.

**MACE** spends its capacity differently. Instead of relying primarily on attention to choose pairwise messages, it constructs higher-correlation equivariant features inspired by the atomic cluster expansion. One layer can explicitly couple several neighbors around a center, so higher body order need not emerge only after many rounds of pairwise message passing (<span id="cite-batatia2022"></span>[Batatia et al., 2022](#ref-batatia2022)). This can trade depth for wider and more structured local computation.

| Design | Where expressivity enters | Main advantage | Main cost |
|---|---|---|---|
| SE(3)-Transformer | Invariant attention over steerable values | Clear, general equivariant-attention construction | Dense tensor products at high degree |
| Equiformer / V2 | Nonlinear invariant attention plus typed feed-forward blocks | Strong geometric routing and high-degree features | Complex kernels, memory, and normalization |
| eSCN-style convolution | Edge-aligned $$SO(2)$$ coupling | Better angular-degree scaling | Frame rotations and specialized implementation |
| MACE | Explicit higher-correlation local products | High body order with few message-passing layers | Wider local expansions and contraction cost |

These are not four incompatible philosophies. Modern models combine them: attention can route higher-order values, edge-aligned convolutions can accelerate an equivariant Transformer, and higher-order products can enrich a local energy model. The useful comparison is which operation dominates accuracy, memory, and wall-clock time on the target system.

## A force field is an energy model with derivatives

For atom types $$\mathbf{Z}=(Z_1,\ldots,Z_N)$$ and coordinates $$\mathbf{r}=(\mathbf{r}_1,\ldots,\mathbf{r}_N)$$, an interatomic potential approximates a scalar potential-energy surface

$$
E_\theta(\mathbf{Z},\mathbf{r}).
$$

Rigid translations and rotations must leave this scalar unchanged. For systems whose energy is size-extensive, the usual local decomposition is

$$
E_\theta(\mathbf{Z},\mathbf{r})
=
\sum_{i=1}^{N}
\varepsilon_{\theta,i},
$$

where each atomic contribution is read from the invariant part of atom $$i$$'s final representation. Two noninteracting copies of a system then have twice the energy. This decomposition also permits cost that is approximately linear in $$N$$ when each atom has a bounded number of neighbors.

Forces should be obtained from the same energy:

$$
\mathbf{F}_{\theta,i}
=
-\nabla_{\mathbf{r}_i}
E_\theta(\mathbf{Z},\mathbf{r}).
$$

This one equation enforces several facts at once. The force field is conservative because it is a gradient. Rotational invariance of energy makes forces equivariant. Translation invariance implies zero net internal force, and rotation invariance implies zero net internal torque:

$$
\sum_i \mathbf{F}_{\theta,i}=\mathbf{0},
\qquad
\sum_i \mathbf{r}_i\times\mathbf{F}_{\theta,i}=\mathbf{0}.
$$

{% include figure.liquid loading="lazy" path="assets/img/blog/eqpot_energy_forces.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An extensive potential sums invariant atomic energy contributions and obtains all force vectors by differentiating the shared scalar energy. Translation and rotation invariance then imply zero net internal force and torque, respectively. Original diagram." %}

A direct vector head can be equivariant without being conservative. It may predict plausible forces at isolated configurations yet fail to be the gradient of any scalar energy. Around a closed path $$\mathcal{C}$$ in configuration space, a conservative model satisfies

$$
\oint_{\mathcal{C}}
\sum_i
\mathbf{F}_{\theta,i}\cdot
d\mathbf{r}_i
=0,
$$

whereas a direct force model has no reason to do so. This distinction matters because a molecular-dynamics trajectory repeatedly composes the model with a numerical integrator. Tiny inconsistencies can accumulate into systematic energy drift.

Energy-derived forces are not sufficient by themselves. Fu et al. show that low held-out errors need not predict stable downstream simulations, and identify smoothness choices that materially affect energy conservation (<span id="cite-fu2025"></span>[Fu et al., 2025](#ref-fu2025)). Their eSEN model is an important reminder: the derivative behavior of the learned surface matters, not merely its values on test structures.

## Force supervision constrains the shape of the surface

A practical objective combines energies and forces:

$$
\mathcal{L}(\theta)
=
\lambda_E
\left(E_\theta-E^\star\right)^2
+
\frac{\lambda_F}{N}
\sum_{i=1}^{N}
\left\lVert
-\nabla_{\mathbf{r}_i}E_\theta
-\mathbf{F}_i^\star
\right\rVert^2.
$$

One energy label constrains one scalar value. A force-labeled configuration provides $$3N$$ derivative components, revealing the local slope of the potential-energy surface. This is why force training can be extremely data-efficient. It is also why training is expensive: backpropagating through a force loss requires derivatives of coordinate gradients with respect to model parameters.

Consider a stretched diatomic molecule with distance $$r$$. Near equilibrium $$r_0$$, a harmonic approximation is

$$
E(r)=E_0+\frac{k}{2}(r-r_0)^2,
\qquad
F_r=-\frac{dE}{dr}=-k(r-r_0).
$$

An energy sample at $$r_0$$ fixes the minimum value but says little about curvature $$k$$. A force sample at $$r_0+\delta$$ directly constrains $$k\delta$$. The same relationship holds in a many-atom system, only the surface has $$3N$$ coordinate directions.

## Cutoffs turn locality into a smoothness problem

Most scalable potentials build a neighbor graph with cutoff radius $$r_c$$. The local environment is finite, and for fixed density the number of edges grows approximately linearly with the number of atoms. A hard cutoff, however, creates a discontinuity when a pair crosses $$r_c$$. Even if energy itself is continuous, a nonzero slope at the boundary creates a force jump.

Let an edge contribution be

$$
\widetilde{\phi}(r)
=
s(r)\phi(r),
$$

where the envelope satisfies

$$
s(r_c)=0,
\qquad
s'(r_c)=0.
$$

Then both the contribution and its first derivative vanish smoothly as the edge disappears. Higher-order simulation properties may require controlling further derivatives. The envelope is not cosmetic windowing; it determines whether the learned potential remains differentiable under a changing neighbor list.

Locality also limits physical reach. Covalent bonding and short-range repulsion are often well represented inside a moderate cutoff, but electrostatics, dispersion, charge transfer, and collective response can extend farther. Stacking local message-passing layers grows the receptive field, but does not make the induced interaction identical to a true long-range term. Fully global attention restores reach at $$O(N^2)$$ edge cost. Practical models therefore use larger but sparse neighborhoods, explicit electrostatic or dispersion terms, hierarchical propagation, reciprocal-space components, or other local–global hybrids.

{% include figure.liquid loading="lazy" path="assets/img/blog/eqpot_cutoff_tradeoff.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Local cutoffs provide near-linear scaling only when interactions vanish smoothly at the boundary; global attention supplies long-range communication but approaches quadratic edge cost. Edge-aligned equivariant operations reduce angular coupling cost, yet typed channels and coordinate derivatives remain part of the computational budget. Original diagram." %}

## What equivariance buys—and what it does not

Equivariance ties together every rotated copy of a configuration. A model trained on one orientation already knows how its internal vectors and predicted forces should behave in another. NequIP demonstrated the resulting data efficiency for interatomic potentials (<span id="cite-batzner2022"></span>[Batzner et al., 2022](#ref-batzner2022)). Equivariant attention adds adaptive neighbor selection without giving up that guarantee.

But equivariance is a constraint, not a complete account of the physics. It does not determine the cutoff, resolve missing long-range interactions, ensure conservative direct-force predictions, prevent extrapolation into untrained chemistry, or make high-degree tensor products cheap. A scalar-only model trained on vast and diverse data can outperform a small equivariant model on some benchmarks; an accurate equivariant model can still produce unstable dynamics if its cutoff is nonsmooth.

The right model is therefore chosen against a simulation contract:

- If trajectories must conserve energy, predict a smooth scalar energy and differentiate it.
- If the system is large, budget neighbor count, angular degree, tensor-product width, and force-gradient cost together.
- If long-range physics matters, add a mechanism for it rather than hoping local depth will imitate it.
- If data are scarce or rotations are strongly out of distribution, equivariance is especially valuable.
- If throughput dominates, measure wall-clock simulation stability, not only single-structure error.

The deepest connection between equivariant Transformers and machine-learned potentials is not that attention is universally superior. It is that a legal geometric architecture exposes exactly what may be learned freely and what must remain fixed. Attention may learn **which interaction matters**. Symmetry dictates **how its content transforms**. The energy construction dictates **where forces come from**. Smoothness, range, and computational scale determine whether those elegant equations survive a real molecular-dynamics run.

## References

<ol class="bibliography">
  <li id="ref-fuchs2020">Fuchs, F. B., Worrall, D. E., Fischer, V., & Welling, M. (2020). <a href="https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html">SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks</a>. <em>NeurIPS</em>. <a href="#cite-fuchs2020">↩</a></li>
  <li id="ref-liao2023">Liao, Y.-L., & Smidt, T. (2023). <a href="https://openreview.net/forum?id=KwmPfARgOTD">Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs</a>. <em>ICLR</em>. <a href="#cite-liao2023">↩</a></li>
  <li id="ref-liao2024">Liao, Y.-L., Wood, B., Das, A., & Smidt, T. (2024). <a href="https://arxiv.org/abs/2306.12059">EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations</a>. <em>ICLR</em>. <a href="#cite-liao2024">↩</a></li>
  <li id="ref-passaro2023">Passaro, S., & Zitnick, C. L. (2023). <a href="https://proceedings.mlr.press/v202/passaro23a.html">Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs</a>. <em>ICML</em>. <a href="#cite-passaro2023">↩</a></li>
  <li id="ref-batatia2022">Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html">MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields</a>. <em>NeurIPS</em>. <a href="#cite-batatia2022">↩</a></li>
  <li id="ref-fu2025">Fu, X., Wood, B. M., Barroso-Luque, L., Levine, D. S., Gao, M., Dzamba, M., & Zitnick, C. L. (2025). <a href="https://proceedings.mlr.press/v267/fu25b.html">Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction</a>. <em>ICML</em>. <a href="#cite-fu2025">↩</a></li>
  <li id="ref-batzner2022">Batzner, S. et al. (2022). <a href="https://www.nature.com/articles/s41467-022-29939-5">E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials</a>. <em>Nature Communications</em>. <a href="#cite-batzner2022">↩</a></li>
</ol>

---

*Figure provenance.* All four `eqpot_` diagrams are original SVG illustrations generated by `scripts/generate_eqpot_figures.py`. They synthesize standard geometric-attention identities and physical constraints described in the cited primary literature; no third-party artwork is reproduced.
