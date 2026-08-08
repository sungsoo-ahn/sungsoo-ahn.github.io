---
layout: post
title: "Steerable Features and Tensor Products"
date: 2026-08-08
last_updated: 2026-08-08
description: "How irreducible rotation types, spherical harmonics, and Clebsch–Gordan tensor products create expressive equivariant neural-network layers."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [geometric-deep-learning]
lecture_paths: [ml4mol, gdl]
tags: [equivariance, steerable-features, spherical-harmonics, clebsch-gordan, tensor-products]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the algebraic core of my 2025 Machine Learning
  for Molecules and Geometric Deep Learning lectures. The companion post
  <a href="{% post_url 2026-08-08-symmetry-equivariance-geometric-data %}">Symmetry and Equivariance</a>
  motivates the transformation laws; <a href="{% post_url 2026-02-02-spherical-equivariant-layers %}">Spherical Equivariant Layers</a>
  goes further into Wigner matrices, implementation details, and modern architectures.</em>
</p>

An equivariant network cannot treat every hidden array as an ordinary feature vector. Some channels are scalars and should stay fixed under rotation. Others encode directions and should rotate as vectors. Higher-order channels contain angular patterns whose components mix in less familiar but equally precise ways.

This creates a type system for geometry. Every feature carries a rule for how it transforms, and every layer must preserve those rules. Linear maps alone can mix channels of the same type, but they cannot create the interactions that make a deep network expressive. The missing operation is multiplication.

The Clebsch–Gordan tensor product is multiplication for rotation-typed features. It couples two input types, decomposes their pairwise products into valid output types, and exposes exactly which couplings are allowed. Combined with spherical harmonics on edges and invariant gates, it gives a reusable recipe for building nonlinear equivariant layers.

## Rotation equivariance imposes a feature type system

Let $$\mathbf{f}^{(\ell)}$$ denote a feature of type $$\ell$$. Under a three-dimensional rotation $$\mathbf{R}\in SO(3)$$, it transforms as

$$
\mathbf{f}^{(\ell)}
\mapsto
\mathbf{D}^{(\ell)}(\mathbf{R})
\mathbf{f}^{(\ell)}.
$$

The matrix $$\mathbf{D}^{(\ell)}(\mathbf{R})$$ is the degree-$$\ell$$ irreducible representation of $$SO(3)$$. It has size $$(2\ell+1)\times(2\ell+1)$$. “Irreducible” means that no fixed change of basis can split these components into smaller subspaces that remain independent under every rotation.

The first three types have concrete interpretations:

- Type $$\ell=0$$ has one component. Its representation is the number one, so it is a scalar.
- Type $$\ell=1$$ has three components. It transforms as an ordinary vector.
- Type $$\ell=2$$ has five components. It describes a quadrupolar angular pattern, equivalent to the independent components of a symmetric traceless $$3\times3$$ tensor.

{% include figure.liquid loading="eager" path="assets/img/blog/steertp_feature_ladder.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The first three irreducible feature types of \(SO(3)\) have dimensions \(1\), \(3\), and \(5\). Their components form one coupled block that transforms through \(\mathbf{D}^{(\ell)}(\mathbf{R})\); they are not independent scalar channels. Original diagram." %}

A network usually stores multiple channels of each type. If there are $$C_\ell$$ type-$$\ell$$ channels, the feature block has shape

$$
\mathbf{F}^{(\ell)}
\in
\mathbb{R}^{C_\ell\times(2\ell+1)}.
$$

The channel index selects different learned features. The magnetic index $$m\in\{-\ell,\ldots,\ell\}$$ selects components within one geometric feature. Rotations mix the $$m$$ components but do not mix learned channels. Confusing these two axes is a common source of broken implementations.

Cohen and Welling (2017) describe this representation-theoretic organization as a type system for steerable networks (<span id="cite-cohen2017"></span>[Cohen & Welling, 2017](#ref-cohen2017)). Once the types are known, equivariance sharply restricts the legal weights. A linear map between repeated copies of the same irreducible type has the form

$$
\mathbf{F}^{(\ell)}
\mapsto
\mathbf{W}^{(\ell)}
\mathbf{F}^{(\ell)},
$$

where $$\mathbf{W}^{(\ell)}\in\mathbb{R}^{C'_\ell\times C_\ell}$$ mixes channels and applies the same coefficients to every $$m$$ component. The layer may learn how much of one type-$$\ell$$ channel enters another, but it cannot assign arbitrary weights to the components inside the irrep. That constraint is what makes the map commute with rotation.

## Spherical harmonics turn directions into typed features

The direction from node $$i$$ to node $$j$$ is

$$
\widehat{\mathbf{r}}_{ij}
=
\frac{\mathbf{r}_j-\mathbf{r}_i}
{\lVert\mathbf{r}_j-\mathbf{r}_i\rVert}.
$$

Its Cartesian coordinates form a type-1 vector, but one direction can generate every angular type. The degree-$$\ell$$ spherical harmonics collect $$2\ell+1$$ functions into a vector

$$
\mathbf{Y}^{(\ell)}
\left(
\widehat{\mathbf{r}}
\right)
=
\left[
Y_{\ell,-\ell}(\widehat{\mathbf{r}}),
\ldots,
Y_{\ell,\ell}(\widehat{\mathbf{r}})
\right]^{\mathsf{T}}.
$$

Under rotation, this vector transforms through the same irreducible representation:

$$
\mathbf{Y}^{(\ell)}
\left(
\mathbf{R}\widehat{\mathbf{r}}
\right)
=
\mathbf{D}^{(\ell)}(\mathbf{R})
\mathbf{Y}^{(\ell)}
\left(
\widehat{\mathbf{r}}
\right),
$$

up to the chosen real or complex harmonic convention. Spherical harmonics therefore convert edge direction into a feature with a declared rotation type.

Distance supplies the complementary invariant information. A learnable radial function $$R_c^{(\ell)}(r_{ij})$$ depends only on

$$
r_{ij}=\lVert\mathbf{r}_j-\mathbf{r}_i\rVert.
$$

Multiplying the angular feature by this scalar produces a steerable filter:

$$
\mathbf{F}_{c}^{(\ell)}
\left(
\mathbf{r}_{ij}
\right)
=
R_c^{(\ell)}(r_{ij})
\mathbf{Y}^{(\ell)}
\left(
\widehat{\mathbf{r}}_{ij}
\right).
$$

The filter separates what changes under rotation from what does not. The spherical harmonic carries orientation, while the radial network learns how interaction strength varies with distance. Tensor Field Networks use exactly this construction to obtain rotation-, translation-, and permutation-equivariant point-cloud layers (<span id="cite-thomas2018"></span>[Thomas et al., 2018](#ref-thomas2018)).

## Linear maps cannot create new geometric interactions

Typed linear maps are necessary but not sufficient. A stack of linear equivariant maps remains linear, and mixing only channels of the same type cannot create a scalar from two vectors or a higher-order angular feature from lower-order inputs.

An ordinary componentwise nonlinearity is not a solution. For a type-1 vector, ReLU generally fails the equivariance equation:

$$
\operatorname{ReLU}
\left(
\mathbf{R}\mathbf{v}
\right)
\neq
\mathbf{R}
\operatorname{ReLU}
\left(
\mathbf{v}
\right).
$$

A two-dimensional subcase shows the failure. Let

$$
\mathbf{v}
=
\begin{bmatrix}
1\\-1
\end{bmatrix},
\qquad
\mathbf{R}
=
\begin{bmatrix}
0&-1\\
1&0
\end{bmatrix}.
$$

Then $$\operatorname{ReLU}(\mathbf{R}\mathbf{v})=(1,1)^{\mathsf{T}}$$, while $$\mathbf{R}\operatorname{ReLU}(\mathbf{v})=(0,1)^{\mathsf{T}}$$. ReLU treats coordinate axes as physically meaningful, so it destroys the vector transformation law.

We need a nonlinear operation that knows the types of both inputs and outputs. Products provide that operation because rotations distribute over multiplication. The remaining problem is to reorganize those products into irreducible blocks.

## The tensor product is equivariant but reducible

Consider a type-$$\ell_1$$ feature $$\mathbf{x}^{(\ell_1)}$$ and a type-$$\ell_2$$ feature $$\mathbf{y}^{(\ell_2)}$$. Their ordinary tensor product contains every pairwise component product:

$$
\mathbf{x}^{(\ell_1)}
\otimes
\mathbf{y}^{(\ell_2)}.
$$

After rotation,

$$
\left(
\mathbf{D}^{(\ell_1)}(\mathbf{R})\mathbf{x}^{(\ell_1)}
\right)
\otimes
\left(
\mathbf{D}^{(\ell_2)}(\mathbf{R})\mathbf{y}^{(\ell_2)}
\right)
=
\left(
\mathbf{D}^{(\ell_1)}(\mathbf{R})
\otimes
\mathbf{D}^{(\ell_2)}(\mathbf{R})
\right)
\left(
\mathbf{x}^{(\ell_1)}
\otimes
\mathbf{y}^{(\ell_2)}
\right).
$$

The product is already equivariant. Its problem is organization: the Kronecker-product representation is reducible. Its components contain several irreducible types mixed together.

Clebsch–Gordan coefficients define the change of basis that separates those types. The type-$$\ell$$ output is

$$
z_m^{(\ell)}
=
\sum_{m_1=-\ell_1}^{\ell_1}
\sum_{m_2=-\ell_2}^{\ell_2}
C^{\ell m}_{\ell_1 m_1,\ell_2 m_2}
x_{m_1}^{(\ell_1)}
y_{m_2}^{(\ell_2)}.
$$

Here, $$C^{\ell m}_{\ell_1 m_1,\ell_2 m_2}$$ is a fixed Clebsch–Gordan coefficient. It is not learned. The learned parameters decide which channels interact and how strongly; the coefficients decide how component products must combine to transform as type $$\ell$$.

## Selection rules remove impossible couplings

Most coefficients in the component formula are zero. Two rules state which outputs can survive.

The **triangle rule** restricts the output degree:

$$
\lvert\ell_1-\ell_2\rvert
\leq
\ell
\leq
\ell_1+\ell_2.
$$

The **magnetic-index rule** requires

$$
m=m_1+m_2.
$$

The allowed output representation is therefore

$$
\mathbf{D}^{(\ell_1)}
\otimes
\mathbf{D}^{(\ell_2)}
\cong
\bigoplus_{\ell=\lvert\ell_1-\ell_2\rvert}^{\ell_1+\ell_2}
\mathbf{D}^{(\ell)}.
$$

{% include figure.liquid loading="eager" path="assets/img/blog/steertp_coupling_rules.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Clebsch–Gordan coupling produces only types that satisfy the triangle rule, with nonzero components further restricted by \(m=m_1+m_2\). For \(O(3)\) features, output parity must also satisfy \(p=p_1p_2\). Original diagram." %}

These rules are the geometric analogue of shape checking. A scalar times a type-2 feature can only produce type 2:

$$
0\otimes2=2.
$$

A vector times a type-2 feature can produce three types:

$$
1\otimes2
=
1\oplus2\oplus3.
$$

No parameter choice can make a type-0 output from this pair because type 0 is absent from the decomposition.

Reflections add one more label. An $$O(3)$$ feature carries parity $$p\in\{+1,-1\}$$, which says whether it stays fixed or changes sign under inversion. Tensor products multiply parity:

$$
p_{\text{out}}
=
p_1p_2.
$$

The degree-$$\ell$$ spherical harmonic has parity $$(-1)^\ell$$. A polar vector is odd, while the cross product of two polar vectors is an even axial vector. Two features can share the same rotation degree and still differ under reflection, so parity must be tracked when the task has full $$O(3)$$ symmetry.

## Two vectors contain scalar, vector, and quadrupole information

The coupling

$$
1\otimes1
=
0\oplus1\oplus2
$$

is the most useful low-order example. Two three-dimensional vectors have a $$3\times3=9$$ dimensional outer product. The Clebsch–Gordan change of basis separates it into blocks of dimensions

$$
9=1+3+5.
$$

In Cartesian notation, the three blocks are familiar.

The type-0 block is the trace, proportional to the dot product:

$$
s=\mathbf{x}\cdot\mathbf{y}.
$$

The type-1 block is the antisymmetric part, represented by the cross product:

$$
\mathbf{a}
=
\mathbf{x}\times\mathbf{y}.
$$

The type-2 block is the symmetric traceless part:

$$
\mathbf{Q}
=
\frac{1}{2}
\left(
\mathbf{x}\mathbf{y}^{\mathsf{T}}
+\mathbf{y}\mathbf{x}^{\mathsf{T}}
\right)
-
\frac{1}{3}
\left(
\mathbf{x}\cdot\mathbf{y}
\right)\mathbf{I}.
$$

{% include figure.liquid loading="eager" path="assets/img/blog/steertp_vector_decomposition.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The nine components of \(\mathbf{x}\mathbf{y}^{\mathsf{T}}\) split into a one-dimensional scalar trace, a three-dimensional antisymmetric vector, and a five-dimensional symmetric traceless tensor. This Cartesian decomposition is the low-order content of \(1\otimes1=0\oplus1\oplus2\). Original diagram." %}

This example explains why tensor products add expressivity. A scalar-only network can measure the angle between two directions through the dot product, but it discards their oriented plane. The type-1 output keeps that orientation through the cross product. The type-2 output retains a quadrupolar pattern that neither a scalar nor a single vector can represent.

Kondor et al. (2018) make this bilinear coupling the source of nonlinearity in Clebsch–Gordan networks (<span id="cite-kondor2018"></span>[Kondor et al., 2018](#ref-kondor2018)). Repeated products can build higher-order interactions while every intermediate block retains an exact transformation type.

## Equivariant nonlinearities act through invariants

Tensor products are not the only legal nonlinear operation. A scalar feature is invariant, so any ordinary activation can act on type 0:

$$
s\mapsto\sigma(s).
$$

A higher-order feature can be scaled by an invariant gate:

$$
\mathbf{f}^{(\ell)}
\mapsto
\sigma(g)
\mathbf{f}^{(\ell)},
\qquad
g\text{ is type }0.
$$

The scalar gate changes magnitude without changing the rotation law. Its value may come from a separate scalar channel, a norm, or a tensor product projected to type 0.

A norm nonlinearity makes the same idea explicit:

$$
\mathbf{f}^{(\ell)}
\mapsto
\eta\!\left(
\left\lVert
\mathbf{f}^{(\ell)}
\right\rVert
\right)
\frac{
\mathbf{f}^{(\ell)}
}{
\left\lVert
\mathbf{f}^{(\ell)}
\right\rVert+\varepsilon
}.
$$

The norm is invariant, while the normalized direction remains type $$\ell$$. Gated and norm-based nonlinearities are standard components of three-dimensional steerable CNNs (<span id="cite-weiler2018"></span>[Weiler et al., 2018](#ref-weiler2018)).

Each option makes a different tradeoff. Scalar-only activations are cheap but leave higher types linear between coupling operations. Gates are efficient but cannot create a new type. Tensor products create new angular content, but their cost grows with the number of input channels and allowed coupling paths. Practical layers combine all three.

## One layer couples node features with edge geometry

The pieces now assemble into a steerable message-passing layer.

For each edge $$(i,j)$$, compute the invariant distance $$r_{ij}$$ and direction $$\widehat{\mathbf{r}}_{ij}$$. A radial network and spherical harmonics produce a filter of type $$\ell_f$$:

$$
\mathbf{F}_{ij}^{(\ell_f)}
=
R^{(\ell_f)}(r_{ij})
\mathbf{Y}^{(\ell_f)}
\left(
\widehat{\mathbf{r}}_{ij}
\right).
$$

Couple a neighbor feature of type $$\ell_{\text{in}}$$ with this filter and retain an allowed output type:

$$
\mathbf{m}_{ij}^{(\ell_{\text{out}})}
=
\sum_{\ell_{\text{in}},\ell_f}
w_{\ell_{\text{in}},\ell_f,\ell_{\text{out}}}
(r_{ij})
\left[
\mathbf{h}_j^{(\ell_{\text{in}})}
\otimes_{\mathrm{CG}}
\mathbf{Y}^{(\ell_f)}
\left(
\widehat{\mathbf{r}}_{ij}
\right)
\right]^{(\ell_{\text{out}})}.
$$

The bracket selects the type-$$\ell_{\text{out}}$$ Clebsch–Gordan block. The learned radial weight $$w$$ mixes channels and controls distance dependence. The selection rule requires

$$
\lvert
\ell_{\text{in}}-\ell_f
\rvert
\leq
\ell_{\text{out}}
\leq
\ell_{\text{in}}+\ell_f.
$$

The low-order cases give this equation a physical interpretation. Suppose node $$j$$ carries a scalar charge-like feature $$q_j$$ and the edge filter has type $$\ell_f=1$$. The only allowed output is type 1:

$$
0\otimes1=1.
$$

Up to basis normalization, the message is

$$
\mathbf{m}_{ij}^{(1)}
\propto
R(r_{ij})q_j
\widehat{\mathbf{r}}_{ij}.
$$

The scalar controls magnitude and the edge direction controls orientation. Rotating the molecule rotates the message.

Now suppose node $$j$$ carries a vector $$\mathbf{v}_j$$. Coupling it to the same type-1 edge filter gives

$$
1\otimes1=0\oplus1\oplus2.
$$

The scalar output is proportional to the radial projection

$$
\mathbf{v}_j\cdot\widehat{\mathbf{r}}_{ij}.
$$

The vector output is proportional to the oriented tangential component

$$
\mathbf{v}_j\times\widehat{\mathbf{r}}_{ij},
$$

and the type-2 output is their symmetric traceless product. One neighbor feature and one edge direction therefore yield three geometrically distinct messages. Learned channel weights can retain all three, suppress some paths, or send them to different downstream channels without changing their transformation laws.

Sum messages over neighbors:

$$
\mathbf{m}_i^{(\ell)}
=
\sum_{j\in\mathcal{N}(i)}
\mathbf{m}_{ij}^{(\ell)}.
$$

Summation is permutation-invariant and preserves the rotation type because it adds features that transform through the same matrix. A type-preserving channel mix and an invariant gate then produce the next node state.

{% include figure.liquid loading="eager" path="assets/img/blog/steertp_layer_pipeline.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A steerable message-passing layer combines a typed neighbor feature with an edge filter \(R(r_{ij})\mathbf{Y}^{(\ell_f)}(\widehat{\mathbf{r}}_{ij})\), projects the product onto allowed output types, sums neighbors, and applies type-preserving mixing and invariant gates. Original diagram." %}

Every step has a distinct symmetry role. Relative positions remove translation. Spherical harmonics encode rotation. Clebsch–Gordan coefficients couple types. Neighbor summation removes ordering. Scalar gates add nonlinearity without choosing a coordinate axis.

For an invariant graph target such as molecular energy, the final readout uses type-0 channels. Vector or tensor targets read out the corresponding types. Intermediate higher-order channels can still improve a scalar prediction because later tensor products can couple them back to type 0.

## Angular resolution and cost are coupled

The maximum degree $$L_{\max}$$ limits the angular frequencies the network can represent. Increasing it adds finer directional structure, but each type has $$2\ell+1$$ components and participates in more tensor-product paths. Computational cost therefore grows through both representation size and coupling count.

More types are not automatically better. A model can gain capacity by adding channels at low degree, by increasing radial resolution, or by stacking more coupling layers. These choices represent different kinds of complexity. High degree adds angular detail within one layer; depth composes lower-order interactions; channel multiplicity stores more learned patterns of the same type.

The useful design question is which geometric interactions the target needs. Scalar energies may benefit from vector and type-2 intermediates even though the output is invariant. A force model must retain type-1 information to the output. Reflection-sensitive targets require parity labels rather than only $$SO(3)$$ degrees.

The resulting architecture is not a generic MLP with rotation augmentation. It is a typed program. Irreducible representations define the data types, spherical harmonics create typed edge inputs, Clebsch–Gordan products implement legal multiplication, and invariant gates provide stable nonlinear control. The companion [Spherical Equivariant Layers]({% post_url 2026-02-02-spherical-equivariant-layers %}) post continues from this algebra to detailed Wigner-matrix conventions, computational strategies, and architecture families.

## References

- <span id="ref-cohen2017"></span>Cohen, T. S., & Welling, M. (2017). Steerable CNNs. [ICLR](https://openreview.net/forum?id=rJQKYt5ll). <a href="#cite-cohen2017" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-thomas2018"></span>Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds. [arXiv:1802.08219](https://arxiv.org/abs/1802.08219). <a href="#cite-thomas2018" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-weiler2018"></span>Weiler, M., Geiger, M., Welling, M., Boomsma, W., & Cohen, T. S. (2018). 3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data. [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html). <a href="#cite-weiler2018" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kondor2018"></span>Kondor, R., Lin, Z., & Trivedi, S. (2018). Clebsch–Gordan Nets: A Fully Fourier Space Spherical Convolutional Neural Network. [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/a3fc981af450752046be179185ebc8b5-Abstract.html). <a href="#cite-kondor2018" class="reversefootnote" role="doc-backlink">↩</a>
