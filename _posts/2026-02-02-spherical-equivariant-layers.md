---
layout: post
title: "Spherical Equivariant Layers for 3D Atomic Systems"
date: 2026-02-02
last_updated: 2026-06-19
description: "Understanding the spherical equivariant layers that power modern molecular neural networks, from group theory foundations to Clebsch-Gordan tensor products."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 2
series: ml-for-science
series_title: "ML for Science Foundations"
series_description: "A guided route through scientific ML topics: quantum chemistry, equivariant molecular models, electrocatalysis, and protein design."
series_order: 2
categories: [machine-learning]
tags: [geometric-deep-learning, equivariance, spherical-harmonics]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: I wrote this post while studying this material with <a href="https://seongsukim-ml.github.io/">Seongsu Kim</a>. For readers who want to go deeper, I highly recommend <a href="https://arxiv.org/abs/2512.13927">Sophia Tang's tutorial</a>, <a href="https://uvagedl.github.io/">Erik Bekkers' lecture series</a>, and <a href="https://arxiv.org/abs/2312.07511">Duval et al.'s Hitchhiker's Guide</a>—all three provide comprehensive treatments with beautiful figures.</em>
</p>

## Introduction

Neural networks for 3D atomic systems, from Tensor Field Networks to modern architectures such as MACE and eSCN, achieve strong accuracy by building rotational equivariance into the model. When a molecule rotates in space, their internal representations rotate with it, so vector predictions such as forces transform correctly.

### High-Level Architecture

These networks can be understood as having two interleaved components:

1. **Message-passing layers**: Aggregate information between atoms by collecting features from neighbors and updating each atom's representation. This handles the *structural* question: how information flows through the molecular graph.

2. **Spherical equivariant layers**: Transform feature vectors while preserving equivariance. These layers use spherical-harmonic features with well-defined rotation rules, so every component has a known response when the molecule rotates. This handles the *geometric* question: how features remain tied to 3D space.

{% include figure.liquid loading="eager" path="assets/img/blog/architecture_overview.png" class="img-fluid rounded z-depth-1" zoomable=true caption="High-level architecture of spherical equivariant networks. Message-passing layers aggregate information from neighboring atoms (structural), while spherical equivariant layers transform features using CG tensor products and nonlinearities (geometric). These two components are interleaved for \(T\) layers." %}

The message-passing structure is straightforward: sum over neighbors and apply learned weights. The hard part is the spherical equivariant layer: **how do we transform features expressively without breaking their rotational behavior?**

Arbitrary neural-network operations are not safe on geometric features. If a feature represents a direction, such as a 3D vector, a standard MLP would destroy its geometric meaning: after the MLP, the feature would no longer rotate properly with the molecule. Equivariant models therefore need operations whose algebra preserves the rotation law.

> **Equivariance.** A function $f$ is **equivariant** with respect to a symmetry (such as rotation) if transforming the input and then applying $f$ gives the same result as applying $f$ first and then transforming the output. For instance, if we rotate a molecule and then predict forces, we get the same answer as predicting forces first and then rotating them:
>
> $$f(\text{rotate}(\mathbf{x})) = \text{rotate}(f(\mathbf{x})) \quad \text{for every rotation}$$
>
> We will formalize this using the language of *groups* and *representations* below.
{: .block-definition }

### Overview

As in any graph neural network, each atom $$i$$ carries a feature vector $$\mathbf{h}_i$$, and each layer updates it by aggregating messages $$\mathbf{m}_{ij}$$ from neighboring atoms $$j$$. Equivariance restricts what $$\mathbf{h}_i$$ and $$\mathbf{m}_{ij}$$ can be: they must be **spherical tensors**, vectors whose transformation under any rotation is exactly described by a **Wigner-D matrix**. Spherical tensors are organized by degree $$\ell = 0, 1, 2, \ldots$$ (scalars, 3D vectors, and higher angular patterns), and Wigner-D matrices are the **representation** matrices of the rotation **group** $$SO(3)$$.

The central operation is therefore tensor combination: how do we combine two spherical tensors and get another spherical tensor out? The answer is the **Clebsch-Gordan tensor product** $$\otimes_{\text{cg}}$$. It plays the role that matrix multiplication plays in standard neural networks: the core operation from which layers are built. A typical message takes the form:

$$\mathbf{m}_{ij} = W(r_{ij}) \left( \mathbf{h}_j \otimes_{\text{cg}} Y(\hat{\mathbf{r}}_{ij}) \right)$$

where $$Y(\hat{\mathbf{r}}_{ij})$$ encodes the direction between atoms as a spherical tensor and $$W(r_{ij})$$ is a learned weight that depends only on distance.

## Mathematical Foundations

### Groups and Symmetries

For 3D atomic systems, the key symmetry is rotation: rotating a molecule shouldn't change its predicted energy, and predicted forces should rotate along with the molecule. We need a precise way to talk about the full collection of 3D rotations and how they combine.

A **group** is the natural mathematical structure for a set of transformations. For 3D rotations, any two rotations compose to another rotation, every rotation has an inverse, and one rotation does nothing. Closure, inverses, and an identity element are exactly the group axioms. The **special orthogonal group** $SO(3)$ is the group of all 3D rotations, represented concretely as $3 \times 3$ orthogonal matrices with determinant $+1$. This is the primary symmetry group for equivariant neural networks on 3D atomic systems. Reflections and translations can also be included, but they are outside the scope of this post.

A **group action** describes how a group transforms the elements of some space $X$. For each group element $g \in G$ and each point $x \in X$, the action produces a transformed point $g \cdot x \in X$, satisfying two properties: the identity element does nothing ($e \cdot x = x$), and composing group elements before acting is the same as acting sequentially ($(g_1 \cdot g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$). For example, $SO(3)$ acts on 3D space by rotating vectors: $R \cdot \mathbf{v} = R\mathbf{v}$. Representations, which we define next, are a special case of group actions where the transformations are linear maps on a vector space.

### Group Representations

Once the symmetry is fixed, the next question is how different features transform under it. A scalar such as energy does not change when the molecule rotates. A vector such as force rotates with the molecule. Higher-order quantities such as quadrupole moments transform in more complex ways. *Group representations* formalize these transformation laws, letting us categorize features by their response to rotation.

> **Group representation.** A **representation** of a group $G$ assigns to each group element $g$ a matrix $D(g)$ acting on a vector space $V$, such that the matrices respect the group structure: $D(g_1 \cdot g_2) = D(g_1) D(g_2)$.
>
> The vector space $V$ is called the **carrier space**.[^carrier] A representation has two components:
> - The **carrier space** $V$: the vector space whose elements get transformed
> - The **representation matrices** $D(g)$: the linear transformations that act on $V$
{: .block-definition }

For example, consider 3D vectors like position or velocity. The carrier space is $\mathbb{R}^3$, and for each rotation $R \in SO(3)$, the representation matrix is the $3 \times 3$ rotation matrix itself. When we rotate a vector $\mathbf{v}$, we compute $R\mathbf{v}$—the rotation matrix acts on elements of the carrier space.

For feature vectors to transform predictably, each feature type needs an associated representation. The simplest is the **trivial representation**, where every group element maps to the identity matrix; the carrier space is $\mathbb{R}^1$ (scalars), and rotations leave scalars unchanged. The **standard representation** of $SO(3)$ uses the $3 \times 3$ rotation matrices on the carrier space $\mathbb{R}^3$, describing how ordinary 3D vectors transform.

These are only two examples from an infinite family. The natural next question is whether a representation can be broken into simpler pieces.

> **Irreducible representation.** A representation $D$ is **reducible** if there exists a change-of-basis matrix $P$ such that:
>
> $$P\,D(g)\,P^{-1} = \begin{bmatrix} D_1(g) & 0 \\ 0 & D_2(g) \end{bmatrix} \quad \text{for all } g \in G$$
>
> In the new basis, the representation is a **direct sum** $$D_1 \oplus D_2$$, meaning the carrier space splits into independent subspaces that don't mix under the group action. A representation that *cannot* be decomposed this way is called **irreducible**.[^irreps]
{: .block-definition }

Irreducible representations (irreps) are the basic building blocks: any representation can be decomposed into a direct sum of irreps by an appropriate change of basis.

For $SO(3)$, the irreps are labeled by non-negative integers $\ell = 0, 1, 2, \ldots$. The $\ell$-th irrep has a carrier space of dimension $2\ell + 1$: the $\ell = 0$ irrep is the trivial representation (1D, scalars), the $\ell = 1$ irrep is the standard representation (3D, vectors), and higher $\ell$ correspond to increasingly complex transformation properties.

### Wigner-D Matrices

What are the concrete representation matrices for these irreps? For each degree $\ell$, the **Wigner-D matrix** $$D^{(\ell)}: SO(3) \to \mathbb{R}^{(2\ell+1) \times (2\ell+1)}$$ provides the representation matrix that acts on the $(2\ell+1)$-dimensional carrier space. We write $$D^{(\ell)}_{mm'}(R)$$ for the $(m, m')$-th entry, where both indices range from $-\ell$ to $\ell$.[^indexing]

The Wigner-D matrices satisfy the representation property: composing two rotations and then computing the representation matrix gives the same result as multiplying the individual representation matrices:

$$D^{(\ell)}(R_1 R_2) = D^{(\ell)}(R_1) D^{(\ell)}(R_2)$$

At each degree the Wigner-D matrix takes a specific form:
- $D^{(0)}(R) = 1$ for all rotations (scalars are invariant)
- $D^{(1)}(R)$ is equivalent to the ordinary $3 \times 3$ rotation matrix
- $D^{(2)}(R)$ is a $5 \times 5$ matrix that transforms the independent components of a traceless symmetric tensor

But what do these carrier spaces look like concretely? Spherical harmonics provide natural basis functions that span them.

---

## Spherical Harmonics and Spherical Tensors

### Spherical Harmonics

Representations tell us how features should transform; spherical harmonics tell us how to construct such features from geometry. Given two neighboring atoms, the network needs to encode the *direction* from one atom to the other. Spherical harmonics are functions on the unit sphere that do exactly this: they take a direction and return numbers with predictable rotation behavior. Evaluating them at the direction $$\hat{\mathbf{r}}_{ij}$$ between atoms $$i$$ and $$j$$ gives the network its first geometric features, and later layers build on them.

To build intuition, consider the simpler case of the circle first. The **circular harmonics** $e^{im\phi} = \cos(m\phi) + i\sin(m\phi)$ form a basis for functions on the circle $S^1$. Any function on the circle can be written as a sum of these basis functions (this is the Fourier series). Crucially, each circular harmonic has a simple transformation property under 2D rotations: rotating by angle $\alpha$ multiplies $e^{im\phi}$ by $e^{im\alpha}$. Different values of $m$ transform independently. We write $Y_m(\phi)$ for the real part $\cos(m\phi)$ of each circular harmonic.

{% include figure.liquid loading="eager" path="assets/img/blog/circular_harmonics.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Circular harmonics \(Y_m\) for \(m = 0, 1, 2, 3\). Red indicates positive values, blue indicates negative values. The number of lobes increases with \(m\). Under rotation, each harmonic gets multiplied by a phase factor proportional to \(m\)." %}

**Spherical harmonics** extend this idea from the circle to the sphere. Circular harmonics form a basis for functions on $S^1$; spherical harmonics form a basis for functions on the unit sphere $S^2$. The analogy also preserves the key symmetry property: circular harmonics transform simply under 2D rotations, and spherical harmonics transform simply under 3D rotations.

The **spherical harmonics** $Y_\ell^m$ are special functions on the unit sphere $S^2$ that form a complete orthonormal basis.[^realsh] Each $Y_\ell^m: S^2 \to \mathbb{R}$ takes a direction and returns a real number. They are indexed by degree $\ell \geq 0$ and order $-\ell \leq m \leq \ell$, so for each degree $\ell$ there are $2\ell + 1$ spherical harmonics.

{% include figure.liquid loading="eager" path="assets/img/blog/spherical_harmonics.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Spherical harmonics for degrees \(\ell = 0, 1, 2, 3\). Red indicates positive values, blue indicates negative values. Each row shows all \(2\ell+1\) harmonics for a given degree. Higher degrees capture increasingly complex angular patterns." %}

Each degree captures a different level of angular complexity:

| $\ell$ | Dim | Intuition |
|-------|-----|-----------|
| 0 | 1 | A single number that doesn't change under rotation (e.g., temperature, charge) |
| 1 | 3 | A direction in 3D space—rotates like an arrow (e.g., velocity, electric field) |
| 2 | 5 | An orientation or anisotropy—describes how something stretches along different axes (e.g., polarizability, stress) |
| 3 | 7 | A more complex angular pattern with finer directional structure |

The $$\ell = 1$$ row has a concrete interpretation: the three degree-1 spherical harmonics are proportional to the Cartesian coordinates $$x$$, $$y$$, $$z$$ evaluated on the unit sphere. Specifically, $$Y_1^{-1} \propto y$$, $$Y_1^0 \propto z$$, and $$Y_1^1 \propto x$$. A type-1 feature vector $$(f_{-1}^{(1)}, f_0^{(1)}, f_1^{(1)})$$ is literally a 3D direction — it rotates like an arrow.

To build intuition for the $$\ell = 2$$ row, consider drawing a surface whose radius at each direction $$\hat{\mathbf{r}}$$ is determined by a combination of type-0 and type-2 spherical harmonics:

$$r(\hat{\mathbf{r}}) = f^{(0)} Y_0^0(\hat{\mathbf{r}}) + \sum_{m=-2}^{2} f_m^{(2)} Y_2^m(\hat{\mathbf{r}})$$

When only the type-0 coefficient is nonzero, $$r$$ is constant in every direction: a perfect sphere. Adding type-2 coefficients deforms the sphere into an ellipsoid, capturing directional stretching or compression. The sign and component choice determine whether the shape stretches along the z-axis (prolate), squashes along it (oblate), or tilts the stretch into the xy-plane.

{% include figure.liquid loading="eager" path="assets/img/blog/ellipsoid_anisotropy.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Surfaces whose radius is set by \(r(\hat{\mathbf{r}}) = f^{(0)} Y_0^0 + \sum_m f_m^{(2)} Y_2^m\). With only \(f^{(0)}\) (left), the radius is constant — a sphere. Turning on different type-2 coefficients deforms the sphere into ellipsoids. Red = stretched outward, blue = compressed inward relative to the base sphere (gray wireframe)." %}

The defining property of spherical harmonics is how they transform under rotations. When we rotate the coordinate system by $R \in SO(3)$, the spherical harmonics of degree $\ell$ mix among themselves according to the Wigner-D matrix $D^{(\ell)}(R)$.

> **Spherical harmonic transformation.** If $\hat{\mathbf{r}}$ is a unit direction vector, evaluating a degree-$\ell$ spherical harmonic at the rotated direction $R^{-1}\hat{\mathbf{r}}$ is equivalent to mixing the spherical harmonics at the original direction:
>
> $$Y_\ell^m(R^{-1}\hat{\mathbf{r}}) = \sum_{m'=-\ell}^{\ell} D^{(\ell)}_{mm'}(R) \, Y_\ell^{m'}(\hat{\mathbf{r}})$$
>
> Rotating a spherical harmonic of degree $\ell$ produces a linear combination of *same-degree* harmonics. Spherical harmonics of different degrees never mix — they transform independently.[^physicsh]
{: .block-lemma }

For each degree $\ell$, the $2\ell + 1$ spherical harmonics $\{Y_\ell^{-\ell}, Y_\ell^{-\ell+1}, \ldots, Y_\ell^{\ell}\}$ span the $(2\ell+1)$-dimensional carrier space of the $\ell$-th irrep. Any linear combination of degree-$\ell$ spherical harmonics can be written as a vector of $2\ell+1$ coefficients, and when we rotate the coordinate system, these coefficients transform by multiplication with the Wigner-D matrix.

In equivariant neural networks, spherical harmonics encode directional information between atoms: evaluating them at the direction from atom $i$ to atom $j$ gives an equivariant geometric feature that the network can process.

### Spherical Tensors

With spherical harmonics providing our basis, we can now define the feature vectors used in equivariant neural networks.

> **Spherical tensor.** A **spherical tensor**[^sphericaltensor] of degree $\ell$ is a $(2\ell+1)$-dimensional vector $\mathbf{f}^{(\ell)} \in \mathbb{R}^{2\ell+1}$ that transforms under rotation $R$ according to the Wigner-D matrix:
>
> $$\mathbf{f}^{(\ell)} \mapsto D^{(\ell)}(R) \, \mathbf{f}^{(\ell)}$$
>
> The carrier space is $\mathbb{R}^{2\ell+1}$—the same space spanned by the degree-$\ell$ spherical harmonics. The components are indexed by order $m \in \{-\ell, \ldots, \ell\}$, mirroring the spherical harmonic indices.
{: .block-definition }

Spherical tensors live in the carrier spaces of $SO(3)$ irreps, the same spaces spanned by spherical harmonics of each degree. Layers that operate on these features are therefore **spherical equivariant layers**. A natural example is the vector of all degree-$\ell$ spherical harmonics evaluated at a direction $\hat{\mathbf{r}}$, which we write as $$Y^{(\ell)}(\hat{\mathbf{r}}) = (Y_\ell^{-\ell}(\hat{\mathbf{r}}), \ldots, Y_\ell^{\ell}(\hat{\mathbf{r}}))^T \in \mathbb{R}^{2\ell+1}$$.

Modern equivariant networks maintain features as **direct sums** of spherical tensors of multiple degrees. A direct sum (denoted $\oplus$) concatenates vectors that transform independently, with each block carrying its own transformation rule:

$$\mathbf{f} = \mathbf{f}^{(0)} \oplus \mathbf{f}^{(1)} \oplus \mathbf{f}^{(2)} \oplus \cdots \oplus \mathbf{f}^{(L)}$$

where $L$ is the maximum degree used in the network. Each component $\mathbf{f}^{(\ell)}$ may have multiple "channels"—for instance, we might have 64 independent type-1 vectors at each node, giving a type-1 feature of shape $(3, 64)$. The total feature at a node is the concatenation of all these components, and under rotation, each component transforms independently according to its type.

This multi-type structure is essential for expressivity. Type-0 features alone would give an invariant network that cannot predict vector quantities such as forces. Type-1 features alone would limit the model's ability to represent more complex angular dependencies. Including multiple types up to a maximum degree $L$ lets the model represent rich angular functions while maintaining exact equivariance.

---

## Clebsch-Gordan Tensor Products

Neural network layers need to combine features, such as a node representation with edge geometry. Equivariant layers need this combination to preserve transformation laws. The Clebsch-Gordan tensor product is the operation that does so.

Think of the layer as operating on features organized by type: type-0 scalars, type-1 vectors, type-2 tensors, and so on. A standard MLP would destroy this structure, so the equivariant analogue of a mixing layer is a CG tensor product. It takes a multi-type feature, mixes contributions across types according to the CG coefficients, and produces a new multi-type feature. Panel (a) below shows the compact view; panel (b) reveals the cross-type connections performed inside one layer.

{% include figure.liquid loading="eager" path="assets/img/blog/cg_network.png" class="img-fluid rounded z-depth-1" zoomable=true caption="CG tensor products as neural network layers. (a) Features organized by type (0 through 3) pass through successive layers. (b) Inside each layer, CG tensor products mix features across types — for example, combining a type-1 and type-2 input to produce type-2 and type-3 outputs. This cross-type mixing is what gives equivariant networks their expressivity." %}

### The Idea

The ordinary tensor product of two spherical tensors is already equivariant, but its output space is *not* organized as a direct sum of irreps. The **Clebsch-Gordan (CG) tensor product** fixes that organization problem by applying a change of basis that splits the tensor product space into irreducible components.

The key insight is:
1. The naive tensor product gives a valid equivariant representation, but it's **reducible**
2. A change-of-basis transformation can decompose this into a direct sum of **irreps**
3. The CG coefficients define exactly this change of basis

{% include figure.liquid loading="eager" path="assets/img/blog/cg_tensor_product.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The Clebsch-Gordan tensor product for two ℓ=1 vectors. Two 3D input vectors x⁽¹⁾ and y⁽¹⁾ are combined via the tensor product into a 3×3 object. The CG transform applies a change-of-basis matrix C·(·)·C⁻¹ to decompose this into a direct sum of irreps: z⁽⁰⁾ (1D scalar, green), z⁽¹⁾ (3D vector, blue), and z⁽²⁾ (5D quadrupole, red). Grid lines indicate the dimension of each component." %}

### The Naive Tensor Product

Consider two spherical tensors $\mathbf{x}^{(\ell_1)}$ and $\mathbf{y}^{(\ell_2)}$ of types $\ell_1$ and $\ell_2$. Recall that a type-$\ell$ feature has $2\ell + 1$ components, indexed by order $m$ ranging from $-\ell$ to $\ell$. We write $x^{(\ell_1)}_{m_1}$ for the $m_1$-th component of $\mathbf{x}^{(\ell_1)}$.

The **tensor product** $\otimes$ computes all pairwise products of their components:

$$(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})_{m_1, m_2} = x^{(\ell_1)}_{m_1} \cdot y^{(\ell_2)}_{m_2}$$

This produces a $(2\ell_1 + 1) \times (2\ell_2 + 1)$-dimensional object. For example, combining two type-1 vectors ($\ell_1 = \ell_2 = 1$, each with 3 components) gives $3 \times 3 = 9$ values.

How does this tensor product transform under rotation? When both inputs are rotated, the outer product matrix transforms by left- and right-multiplication with Wigner-D matrices:

$$\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)} \;\mapsto\; D^{(\ell_1)}(R)\,\bigl(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)}\bigr)\,D^{(\ell_2)}(R)^T$$

To treat this as a single feature vector (which is more natural for neural networks), we **vectorize** the matrix—stacking its columns into a vector $$\operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})$$ of dimension $$(2\ell_1+1)(2\ell_2+1)$$. Using the standard identity $$\operatorname{vec}(ABC) = (C^T \otimes A)\,\operatorname{vec}(B)$$ for the Kronecker product $$\otimes$$, the transformation rule becomes:

$$\begin{aligned}
R \cdot \operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})
&= \underbrace{\bigl(D^{(\ell_2)}(R) \otimes D^{(\ell_1)}(R)\bigr)}_{\text{new group representation}}\;\operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})
\end{aligned}$$

The Kronecker product $$D^{(\ell_2)}(R) \otimes D^{(\ell_1)}(R)$$ is a new, larger representation matrix of dimension $$(2\ell_1+1)(2\ell_2+1)$$. This is still a valid equivariant representation—it respects the group structure—but it is **reducible**: it can be decomposed into smaller, independent blocks.

### The Change-of-Basis Solution

The vectorized tensor product's components do not transform independently by type; they are mixed by the large Kronecker-product matrix. For neural-network layers, we want features organized as a **direct sum** of irreps, where each block transforms by its own Wigner-D matrix.

Representation theory guarantees that any reducible representation can be block-diagonalized by a change of basis. There exists an orthogonal change-of-basis matrix $$\mathbf{C}$$ such that:

$$\begin{aligned}
& R \cdot \operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)}) \\[4pt]
&\quad= \mathbf{C}^{-1}\begin{bmatrix} D^{(\ell'_1)}(R) & 0 & 0 & \cdots \\ 0 & D^{(\ell'_2)}(R) & 0 & \cdots \\ 0 & 0 & D^{(\ell'_3)}(R) & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix} \mathbf{C}\;\operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})
\end{aligned}$$

where the output degrees $$\ell'$$ range from $$\lvert\ell_1 - \ell_2\rvert$$ to $$\ell_1 + \ell_2$$. For our $$\ell_1 = \ell_2 = 1$$ example, the 9D tensor product decomposes into $$D^{(0)} \oplus D^{(1)} \oplus D^{(2)}$$ (dimensions $$1 + 3 + 5 = 9$$).

This motivates the definition of the **Clebsch-Gordan (CG) tensor product** $$\otimes_{\text{cg}}$$, which applies the change of basis directly:

$$\operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes_{\text{cg}} \mathbf{y}^{(\ell_2)}) \;=\; \mathbf{C}\;\operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes \mathbf{y}^{(\ell_2)})$$

Because $$\mathbf{C}$$ absorbs the change of basis, the CG tensor product transforms by the block-diagonal matrix directly—each output block is a spherical tensor that transforms by a single Wigner-D matrix:

$$\begin{aligned}
& R \cdot \operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes_{\text{cg}} \mathbf{y}^{(\ell_2)}) \\[4pt]
&\quad= \begin{bmatrix} D^{(\ell'_1)}(R) & 0 & 0 & \cdots \\ 0 & D^{(\ell'_2)}(R) & 0 & \cdots \\ 0 & 0 & D^{(\ell'_3)}(R) & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix} \operatorname{vec}(\mathbf{x}^{(\ell_1)} \otimes_{\text{cg}} \mathbf{y}^{(\ell_2)})
\end{aligned}$$

This is the format the network needs: a direct sum of irreps with independently transforming blocks. The entries of $$\mathbf{C}$$ are the **Clebsch-Gordan coefficients** $$C_{(\ell_1, m_1),(\ell_2, m_2)}^{(\ell, m)}$$, where the subscript specifies the two input components and the superscript specifies the output component.

> **CG tensor product (component form).** The CG tensor product combines two spherical tensors into a direct sum of irreps:
>
> $$(\mathbf{x}^{(\ell_1)} \otimes_{\text{cg}} \mathbf{y}^{(\ell_2)})^{(\ell)}_m = \sum_{m_1=-\ell_1}^{\ell_1} \sum_{m_2=-\ell_2}^{\ell_2} C_{(\ell_1, m_1),(\ell_2, m_2)}^{(\ell, m)} \, x^{(\ell_1)}_{m_1} \, y^{(\ell_2)}_{m_2}$$
>
> The output degrees range from $$\lvert\ell_1 - \ell_2\rvert$$ to $$\ell_1 + \ell_2$$. Each output block is a spherical tensor that transforms by its own Wigner-D matrix.
{: .block-definition }

> **Example: $$\ell_1 = \ell_2 = 1$$.** For two vectors, the CG tensor product decomposes the $3 \times 3 = 9$ components into three irreps:
> - A **type-0** (1D scalar): the dot product $\mathbf{x} \cdot \mathbf{y}$
> - A **type-1** (3D vector): the cross product $\mathbf{x} \times \mathbf{y}$
> - A **type-2** (5D): the traceless symmetric outer product
{: .block-example }

This is what allows us to stack multiple layers while maintaining equivariance throughout.

### Computational Considerations

CG tensor products are expensive. A naive implementation has complexity $O(L^6)$ where $L$ is the maximum degree, which becomes prohibitive for large $L$. This has motivated substantial research into more efficient implementations. Modern approaches reduce the complexity to $O(L^3)$ through techniques such as aligning features with edge vectors (reducing $SO(3)$ operations to simpler $SO(2)$ operations, as introduced by eSCN), computing only selected output types rather than all possible outputs, and exploiting sparsity patterns in the CG coefficients.

---

## General Architectural Framework

This section assembles the components from previous sections into a complete neural network layer.

### Building Blocks

Most modern equivariant architectures for 3D atomic systems share a common framework built from the components above. The input is a set of atoms indexed by $i = 1, \ldots, N$, each with a 3D position $\mathbf{r}_i \in \mathbb{R}^3$ and an atom type (element identity). The output might be a scalar property such as energy, vector quantities such as forces, or more complex tensorial properties.

The network begins by embedding atom types into type-0 (scalar) features. This resembles word embeddings in NLP, except the resulting features are rotation-invariant.

The geometric information enters through the edges of the graph: for each pair of neighboring atoms $i$ and $j$, we compute the relative position vector, its length, and its direction. The direction is encoded using spherical harmonics, providing an equivariant representation of the edge geometry.

The core of the network consists of **equivariant message passing layers**. In each layer, node features are updated by aggregating information from neighboring nodes. Let $$\mathbf{h}_j^{(\ell)}$$ denote the type-$$\ell$$ spherical tensor at node $$j$$, with distance $$r_{ij} = \lVert \mathbf{r}_j - \mathbf{r}_i \rVert$$ and direction $$\hat{\mathbf{r}}_{ij} = (\mathbf{r}_j - \mathbf{r}_i) / r_{ij}$$ for each neighboring pair.

> **Equivariant message.** The message from node $$j$$ to node $$i$$ is:
>
> $$\mathbf{m}_{ij}^{(\ell_{\text{out}})} = W(r_{ij}) \left( \mathbf{h}_j^{(\ell_{\text{in}})} \otimes_{\text{cg}} Y^{(\ell_f)}(\hat{\mathbf{r}}_{ij}) \right)^{(\ell_{\text{out}})}$$
>
> $$Y^{(\ell_f)}(\hat{\mathbf{r}}_{ij})$$ encodes the edge direction as a degree-$$\ell_f$$ spherical harmonic (the "filter" degree), $$\otimes_{\text{cg}}$$ combines neighbor features with this directional encoding, and $$W(r_{ij})$$ is a learnable **radial function** — rotation-invariant because it depends only on distance, not direction.
{: .block-definition }

The messages are then summed over all neighbors $$\mathcal{N}(i)$$ to update the node features: $$\mathbf{h}_i^{(\ell)} \leftarrow \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(\ell)}$$. Summation is a natural choice for aggregation because it commutes with rotation.

### Nonlinearities and Output

Nonlinear activations require care. Applying a pointwise nonlinearity such as ReLU directly to the components of a type-$\ell$ feature (for $\ell > 0$) breaks equivariance, because the nonlinearity does not commute with the Wigner-D rotation matrices. Several workarounds preserve the rotation law.

The simplest approach is to apply nonlinearities only to type-0 (scalar) features, which are invariant and can be processed with any standard activation function. For higher-type features, we use only linear transformations within each layer, relying on the CG tensor products between layers to provide the necessary nonlinear mixing.

A more sophisticated approach is **gated nonlinearity**, where we use a scalar quantity (such as the norm of a higher-type feature) to gate the feature: $\mathbf{h}^{(\ell)} \leftarrow \sigma(\lVert\mathbf{h}^{(\ell)}\rVert) \cdot \mathbf{h}^{(\ell)}$, where $\sigma$ is a standard activation function (e.g., sigmoid or SiLU). This preserves equivariance because the norm $\lVert\mathbf{h}^{(\ell)}\rVert$ is rotation-invariant, and multiplying by a scalar preserves the transformation properties.

The output layer depends on the target quantity. For invariant quantities such as total energy, the model uses type-0 features and sums over atoms to obtain one scalar. For equivariant quantities such as atomic forces, the model can either use type-1 features directly or compute forces as the negative gradient of the predicted energy: $\mathbf{F} = -\nabla E$. The gradient approach guarantees conservative forces, which matters for molecular dynamics, but it requires backpropagation through the energy prediction.

```
Input: Positions + Atom Types
         ↓
    Embedding Layer (atom types → type-0 features)
         ↓
    ┌─────────────────────────────────────┐
    │  Equivariant Message Passing Layer  │
    │  ├─ Spherical harmonic edge embed   │
    │  ├─ CG tensor product               │
    │  ├─ Radial weighting                │
    │  └─ Neighbor aggregation            │
    └─────────────────────────────────────┘
         ↓ (repeat N times)
    Nonlinearity (gated or scalar-only)
         ↓
    Output Layer (type-0 for energy, type-1 for forces)
```

---

## Modern Architectures

The mathematical framework above is the common language behind many 3D atomic neural networks. Tensor Field Networks (Thomas et al., 2018) made the basic recipe explicit: spherical tensor features, spherical-harmonic edge information, radial functions, and CG tensor products. Later models such as NequIP, MACE, Equiformer, and eSCN changed the engineering details, but they still revolve around the same question: how do we mix atomic features without breaking the rotation law?

For this tutorial, the important point is not the leaderboard history. Modern architectures mainly differ in where they spend compute: attention versus message passing, higher body-order interactions versus cheaper edge-aligned $$SO(2)$$ operations, and strict equivariance versus approximate equivariance for speed. Once the irreps, Wigner-D matrices, and CG tensor products are clear, those design choices become much easier to read.

---

## References

- Thomas, N., et al. (2018). Tensor Field Networks. [arXiv:1802.08219](https://arxiv.org/abs/1802.08219).
- Batzner, S., et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. [Nature Communications](https://doi.org/10.1038/s41467-022-29939-5).
- Batatia, I., et al. (2022). MACE: Higher Order Equivariant Message Passing Neural Networks. [NeurIPS 2022](https://arxiv.org/abs/2206.07697).
- Liao, Y.-L. & Smidt, T. (2023). Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs. [ICLR 2023](https://arxiv.org/abs/2206.11990).
- Passaro, S. & Zitnick, C. L. (2023). Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs. [ICML 2023](https://arxiv.org/abs/2302.03655).
- Tang, S. (2025). A Complete Guide to Spherical Equivariant Graph Transformers. [arXiv:2512.13927](https://arxiv.org/abs/2512.13927).
- Bekkers, E. (2024). Geometric Deep Learning Lecture Series. [UvA](https://uvagedl.github.io/).
- Duval, A., et al. (2023). A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems. [arXiv:2312.07511](https://arxiv.org/abs/2312.07511).

---

[^carrier]: There does not seem to be a universally agreed-upon terminology for the vector space on which a group representation acts. Different authors use "carrier space," "representation space," or simply "the space $V$." I use "carrier space" throughout this post to emphasize that this is the space that "carries" the group action—i.e., the space whose elements get transformed by the representation matrices.

[^irreps]: The term "irreps" is short for "irreducible representations"—representations that cannot be decomposed into smaller independent blocks. They are the atomic building blocks from which all representations can be constructed. In some of the literature, features built from irreps are called "irreps features" and the corresponding layers "irreps-based equivariant layers."

[^realsh]: We use real spherical harmonics here, which are real-valued linear combinations of the complex spherical harmonics. Most equivariant neural network libraries use real spherical harmonics because they avoid complex arithmetic while maintaining all the essential transformation properties.

[^indexing]: Throughout this post, subscript indices on a matrix symbol denote its entries (e.g., $D^{(\ell)}_{mm'}(R)$ is the $(m, m')$-th entry of $D^{(\ell)}(R)$), and subscript indices on a vector symbol denote its components (e.g., $x^{(\ell)}_m$ is the $m$-th component of $\mathbf{x}^{(\ell)}$).

[^sphericaltensor]: The term "spherical tensor" comes from physics, where it refers to a set of quantities that transform under rotations according to a Wigner-D matrix of a specific degree. In the equivariant neural network literature, these are also called "steerable features" (because we can predict—or "steer"—exactly how they change under any rotation) or simply "type-$\ell$ features."

[^physicsh]: From a physics standpoint, spherical harmonics are the eigenstates of angular momentum operators acting on the function space of the sphere. Angular momentum operators describe infinitesimal rotations, which is why the transformation of spherical harmonics under finite rotations is so well-behaved.
