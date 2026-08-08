---
layout: post
title: "Symmetry and Equivariance: How Geometry Constrains Learning"
date: 2026-08-08
last_updated: 2026-08-08
description: "A concrete account of group actions, invariance, equivariance, and feature types for geometric machine learning."
abstract: >
  Geometry tells us which transformations preserve the identity of a data point.
  Equivariant models turn that statement into a constraint on the functions they can learn.
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 1
categories: [foundations]
lecture_paths: [ml4mol, gdl]
tags: [geometric-deep-learning, symmetry, equivariance, group-theory, molecular-machine-learning]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the storyline of my 2025 Machine Learning for Molecules and Geometric Deep Learning lectures. It focuses on the general principle behind equivariant models; the later machinery for spherical equivariant layers is developed in <a href="{% post_url 2026-02-02-spherical-equivariant-layers %}">a separate post</a>.</em>
</p>

A geometric model should not spend parameters learning that a rotated molecule is still the same molecule. The coordinates have changed, but the object has not. If the target is energy, the prediction should stay fixed. If the target is force, the prediction should rotate with the atoms.

This observation sounds like data augmentation: rotate each training example and ask the network to fit all copies. But the stronger approach is to constrain the function itself. The model should give a consistent answer for *every* allowed transformation, including those absent from the training set. Geometry then becomes an architectural principle rather than a source of extra examples.

The same principle applies beyond molecules. Reordering the vertices of a graph changes its adjacency matrix but not the graph. Translating a point cloud changes every coordinate but not its internal shape. Shifting an image moves its pixels but preserves a local visual pattern. Each domain comes with transformations that change its description without changing the underlying object.

## Geometry begins with a choice of sameness

Geometry specifies which properties survive a chosen class of transformations. Euclidean geometry, for example, studies distances and angles because rotations, translations, and reflections preserve them. For machine learning, the useful question is more operational: **which transformations should leave the meaning of a data point unchanged?** Bronstein et al. (2021) call this the geometric prior of the domain.

That question cannot be answered from coordinates alone. A molecule stored as an array contains both physical structure and arbitrary bookkeeping. Its row order is arbitrary, its origin is arbitrary, and its orientation is arbitrary. A model that treats these choices as signal sees many descriptions where a scientist sees one object.

The collection of descriptions connected by allowed transformations is called an **orbit**. The figure below shows a toy geometric graph under rigid motions and vertex relabeling. Each drawing belongs to the same orbit even though its coordinate array and row order differ.

{% include figure.liquid loading="eager" path="assets/img/blog/symeq_data_orbit.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A geometric object has many coordinate descriptions related by symmetry transformations. An invariant target is constant across this orbit, while an equivariant target changes according to a prescribed transformation rule. Original diagram." %}

An invariant function effectively operates on orbits rather than individual coordinate arrays. It maps all points in an orbit to one answer. Mathematicians describe the set of orbits as a **quotient space**: each equivalence class is treated as a single point. This viewpoint explains why an invariant model can be data-efficient. Learning one value on an orbit replaces learning a separate value at every rotated, translated, or relabeled copy.

One might try to obtain the same effect by choosing a canonical description for each object. We could center every molecule, align it with its principal axes, and sort its atoms. This strategy is fragile. Symmetric molecules can have non-unique principal axes, and small coordinate perturbations can abruptly swap axes or sorting order. Canonicalization then turns a smooth physical change into a discontinuous input change. Equivariance avoids this artificial choice: the model accepts every coordinate frame and relates their predictions algebraically.

A symmetry is therefore never just a property of the input. It is a claim about the task. Rotating a molecule should preserve its energy, but it should rotate its dipole moment. Permuting atoms of the same element should preserve molecular properties, but exchanging carbon with oxygen is not an allowed permutation. A reflection may be appropriate for an isolated potential-energy model, while a chirality-sensitive classifier must distinguish mirror images. Choosing the wrong symmetry discards real information.

## Groups turn transformations into algebra

To constrain a function for all transformations, we need to know how transformations compose. A **group** is a set of transformations with four properties: two transformations can be composed, composition is associative, an identity transformation does nothing, and every transformation has an inverse.

The definition is abstract because the same algebra describes different domains. The symmetric group $$S_N$$ contains all permutations of $$N$$ objects. The rotation group $$SO(3)$$ contains all orientation-preserving rotations in three dimensions. The Euclidean group $$E(3)$$ adds translations and reflections, while the special Euclidean group $$SE(3)$$ contains rotations and translations but excludes reflections.

For rigid motions, we can represent a group element as a pair $$(\mathbf{R}, \mathbf{t})$$. Here, $$\mathbf{R}$$ is a rotation matrix and $$\mathbf{t}$$ is a translation vector. Acting on a point $$\mathbf{x} \in \mathbb{R}^3$$ gives

$$
(\mathbf{R}, \mathbf{t}) \cdot \mathbf{x}
= \mathbf{R}\mathbf{x} + \mathbf{t}.
$$

The group law follows from applying two rigid motions in sequence:

$$
(\mathbf{R}_2, \mathbf{t}_2)(\mathbf{R}_1, \mathbf{t}_1)
=
(\mathbf{R}_2\mathbf{R}_1,
 \mathbf{R}_2\mathbf{t}_1 + \mathbf{t}_2).
$$

The second translation is not simply added to the first. The first translation must also be rotated by $$\mathbf{R}_2$$. This coupling is why it is useful to treat rigid motions as group elements rather than as an informal list of augmentations.

A **group action** specifies how a group transforms a particular space. The rotation group acts on coordinates by matrix multiplication. The permutation group acts on a matrix of node features by reordering its rows. If $$\mathbf{P}$$ is a permutation matrix, node features $$\mathbf{H}$$ transform as

$$
\mathbf{H} \mapsto \mathbf{P}\mathbf{H}.
$$

The same permutation acts on an adjacency matrix $$\mathbf{A}$$ by reordering both endpoints of every edge:

$$
\mathbf{A} \mapsto \mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf{T}}.
$$

The input is one object, but its components transform in different ways. Coordinates, node features, edges, and targets therefore need separate actions.

A three-node graph makes the two actions concrete. Suppose the rows of $$\mathbf{H}$$ store features for nodes $$(1,2,3)$$, and a permutation changes the stored order to $$(3,1,2)$$. Multiplication by $$\mathbf{P}$$ performs exactly this row reordering. Each entry $$A_{ij}$$ describes a pair of nodes, so the same relabeling must act on both indices; this gives $$\mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf{T}}$$. The transformed matrices may look different, but every edge still connects the same pair of physical nodes.

Now consider one message-passing update:

$$
\mathbf{h}'_i
=
\phi\!\left(
\mathbf{h}_i,\,
\sum_{j \in \mathcal{N}(i)}
\psi(\mathbf{h}_i,\mathbf{h}_j)
\right).
$$

Here, $$\psi$$ constructs a message from neighbor $$j$$, the sum aggregates all incoming messages, and $$\phi$$ updates node $$i$$. The functions $$\phi$$ and $$\psi$$ are shared across nodes, while the sum ignores neighbor order. Relabeling the input therefore only relabels the outputs: $$\mathbf{H}' \mapsto \mathbf{P}\mathbf{H}'$$. Permutation equivariance is not an emergent property of training; it follows from weight sharing and symmetric aggregation.

## Representations describe how features transform

A group action on a vector space is called a **representation** when every transformation acts linearly. A representation $$\rho$$ assigns a matrix $$\rho(g)$$ to each group element $$g$$ and preserves composition:

$$
\rho(g_2g_1) = \rho(g_2)\rho(g_1).
$$

Representations are not an extra symmetry imposed on the data. They are the bookkeeping system that states how each quantity responds to a symmetry already chosen for the task.

Three feature types cover much of what appears in geometric neural networks. A **scalar** does not change under rotation, so its representation is the number one. Energy, mass, and atom type are common scalar features. An ordinary **vector** transforms with the rotation matrix itself. Positions, velocities, forces, and electric fields are vector quantities. A rank-two Cartesian tensor, such as a stress tensor, transforms on both indices:

$$
\mathbf{T} \mapsto \mathbf{R}\mathbf{T}\mathbf{R}^{\mathsf{T}}.
$$

The term **steerable feature** covers any feature vector whose components mix through a known representation matrix. Scalars, ordinary vectors, and higher-order tensor features are all steerable features. The name emphasizes that after transforming the input, we can *steer* the feature to its new value without recomputing it from scratch.

{% include figure.liquid loading="eager" path="assets/img/blog/symeq_feature_types.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Scalars, vectors, and higher-order steerable features differ by their transformation laws. A representation matrix \(D(g)\) specifies how the components of a steerable feature mix under a group element \(g\). Original diagram." %}

The transformation law matters more than the coordinate format. A three-dimensional array is not necessarily a geometric vector: three unrelated scalar channels should not rotate into one another. Conversely, a vector can be stored in a basis other than the Cartesian axes. Circular and spherical harmonics use coefficients whose components mix under rotations, yet they still carry precise geometric meaning. Their representation matrices make the transformation law explicit.

This distinction prevents a common modeling error. Applying an arbitrary multilayer perceptron to the three components of a force vector does not generally produce another vector. The nonlinear map can break the rotation law. Equivariant architectures restrict how different feature types mix so that every intermediate feature retains a defined transformation rule.

## Invariance and equivariance constrain functions

Let $$f: X \rightarrow Y$$ be a function from an input space $$X$$ to an output space $$Y$$. Suppose a group $$G$$ acts on the input through $$\rho_X$$ and on the output through $$\rho_Y$$. The function is **equivariant** when

$$
f\bigl(\rho_X(g)x\bigr)
=
\rho_Y(g)f(x)
\qquad \text{for every } g \in G.
$$

The equation says that two computational paths agree. We can transform the input and then evaluate the function, or evaluate the function and then transform the output. The result must be the same.

{% include figure.liquid loading="eager" path="assets/img/blog/symeq_commuting_paths.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An equivariant function commutes with the group action: transforming before prediction matches transforming after prediction. Invariance is the special case in which the output action is the identity. Original diagram." %}

**Invariance** is a special case of equivariance. If the output should not change, then $$\rho_Y(g)$$ is the identity and

$$
f\bigl(\rho_X(g)x\bigr) = f(x).
$$

This compact equation separates the symmetry of a task into two decisions: how the input transforms and how the output transforms. The group alone is not enough. The same rotation group leads to invariance for energy and equivariance for force.

The distinction also appears in graphs. A graph-level prediction such as molecular solubility should be invariant to vertex ordering:

$$
f(\mathbf{P}\mathbf{H},
  \mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf{T}})
= f(\mathbf{H}, \mathbf{A}).
$$

A node-level prediction such as an atom-wise charge should instead be permutation-equivariant:

$$
f(\mathbf{P}\mathbf{H},
  \mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf{T}})
= \mathbf{P}f(\mathbf{H}, \mathbf{A}).
$$

The output values are physically unchanged, but their rows must follow the reordered atoms. Summation provides a simple invariant operation, while message passing with a shared update rule provides a simple equivariant operation. This is the algebraic reason graph neural networks share parameters across vertices and aggregate neighbors without using their order.

Euclidean symmetry combines several constraints. Let $$\mathbf{X} \in \mathbb{R}^{N \times 3}$$ contain atomic coordinates as rows, and let $$\mathbf{F}(\mathbf{X})$$ contain predicted forces. Under the row-vector convention, a rotation and translation act as

$$
\mathbf{X} \mapsto
\mathbf{X}\mathbf{R}^{\mathsf{T}}
+ \mathbf{1}\mathbf{t}^{\mathsf{T}}.
$$

An invariant energy model $$E$$ and an equivariant force model $$\mathbf{F}$$ should satisfy

$$
E\!\left(
\mathbf{X}\mathbf{R}^{\mathsf{T}}
+ \mathbf{1}\mathbf{t}^{\mathsf{T}}
\right)
= E(\mathbf{X}),
$$

$$
\mathbf{F}\!\left(
\mathbf{X}\mathbf{R}^{\mathsf{T}}
+ \mathbf{1}\mathbf{t}^{\mathsf{T}}
\right)
= \mathbf{F}(\mathbf{X})\mathbf{R}^{\mathsf{T}}.
$$

Translations disappear from the force transformation because forces have direction but no position. If forces come from the gradient of an invariant energy, $$\mathbf{F} = -\nabla_{\mathbf{X}}E$$, rotational equivariance follows by the chain rule. This relation is one reason molecular potentials often predict a scalar energy and obtain conservative forces by differentiation.

Consider a water molecule as a concrete case. Place the oxygen at the origin and the two hydrogens in the $$xy$$-plane. A rigid rotation changes all three coordinate rows, yet the O–H distances and H–O–H angle stay fixed. An energy model should return the same scalar before and after the rotation. Its force on each atom should preserve magnitude and rotate by the same matrix as the molecule. A model that predicts the correct energy but leaves its force vectors fixed in the laboratory frame is not physically consistent.

Relative displacements remove translations without discarding rotations:

$$
\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i,
\qquad
\mathbf{r}_{ij} \mapsto \mathbf{R}\mathbf{r}_{ij}.
$$

Their lengths $$\lVert\mathbf{r}_{ij}\rVert$$ are invariant, while the displacements themselves are equivariant vectors. A distance-only model can build an invariant energy from the lengths. A vector-aware model can also propagate directional information through $$\mathbf{r}_{ij}$$. The second design is necessary when pairwise distances alone fail to distinguish configurations relevant to the target.

## Architectural symmetry changes the hypothesis class

An unconstrained neural network searches over many functions that disagree on symmetry-related inputs. An equivariant architecture removes those functions before training. Its **hypothesis class**—the set of functions it can express—is smaller, but every remaining function respects the chosen transformation law.

{% include figure.liquid loading="eager" path="assets/img/blog/symeq_hypothesis_class.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Exact equivariance restricts learning to the symmetry-compatible part of a larger hypothesis class. When the symmetry matches the data-generating process, one observed configuration constrains predictions across its entire orbit. Original diagram." %}

Convolutional networks provide the familiar case. A standard convolution applies the same filter at every image location, so translating the input translates the feature map. Cohen and Welling (2016) extended this construction from translations to larger transformation groups. Geometric graph networks follow the same design logic on permutations and continuous rotations, although their equivariant operations need not look like image convolutions.

There are three common ways to impose a symmetry, and they make different guarantees. The first is to build invariant inputs, such as pairwise distances. This route is simple, but any information removed by the invariant map is unavailable downstream. The second is to use equivariant intermediate features and equivariant layers, then convert to invariants only when the target requires them. This route preserves more information but requires structured linear maps, tensor products, or other representation-aware operations. The third is to average an arbitrary predictor over transformed inputs:

$$
\bar{f}(x)
=
\int_G \rho_Y(g)^{-1}
f\!\left(\rho_X(g)x\right)\,d\mu(g),
$$

where $$\mu$$ is a uniform group measure when such a normalized measure exists. The averaged predictor is equivariant under standard conditions. Exact integration is often infeasible for continuous groups, so practical systems approximate it with samples. Architectural equivariance enforces the same relation at each forward pass without enumerating rotations.

This restriction changes how information propagates from data. After seeing one molecular conformation, an exactly equivariant model already knows how its prediction must transform under every rotation. An unconstrained model must infer that relation from augmented samples and optimization. Data augmentation can encourage symmetry, but finite augmentation does not make the learned function exactly equivariant between sampled transformations.

The benefit is not just aesthetic. Elesedy and Zaidi (2021) show that, under symmetry assumptions on the data distribution and loss, averaging a predictor over the group cannot increase its expected risk and can strictly reduce it. Architectural equivariance performs an analogous restriction by construction. In practice, the gain depends on whether the architecture remains expressive within the equivariant class and whether the assumed symmetry is correct.

Exact symmetry is therefore a bias, not a free theorem about nature. A surface breaks translation symmetry in the direction normal to it. An external electric field selects a preferred direction. A labeled anchor atom can break permutation symmetry. A model that erases these signals will underfit regardless of how elegant its equations look. The right architecture preserves the transformations that leave the *task* unchanged and exposes the variables that break them.

There is also a computational tradeoff. Invariant models based only on pairwise distances are simple and efficient, but distances can discard orientation and chirality. Models with vector or higher-order steerable features retain more geometric information, at the cost of more structured operations. The appropriate feature types depend on the output and on which geometric distinctions the target actually uses.

Equivariance also changes failure modes. An unconstrained model can violate a symmetry arbitrarily far from its training orientations. An equivariant model cannot make that particular mistake, but it can still fit the wrong invariant dependence—for example, the wrong relationship between bond length and energy. Symmetry removes a family of inconsistent solutions; it does not supply the missing chemistry, data coverage, or optimization.

Symmetry changes learning because it changes what counts as a possible function. Group actions state how inputs and outputs transform. Representations state how features transform. Equivariance then requires the model to respect both statements at every point in its domain. Once those choices are explicit, geometry stops being vague intuition and becomes a testable contract for an architecture.

## References

- Bronstein, M. M., Bruna, J., Cohen, T., and Veličković, P. (2021). [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478). *arXiv:2104.13478*.
- Cohen, T. S., and Welling, M. (2016). [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html). *Proceedings of ICML*, 2990–2999.
- Elesedy, B., and Zaidi, S. (2021). [Provably Strict Generalisation Benefit for Equivariant Models](https://proceedings.mlr.press/v139/elesedy21a.html). *Proceedings of ICML*, 2959–2969.
