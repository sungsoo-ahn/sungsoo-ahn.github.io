---
layout: post
title: "Equivariant Neural Networks for 3D Atomic Systems"
date: 2025-01-02
last_updated: 2025-01-02
description: "A comprehensive guide to equivariant neural networks for 3D data, covering group theory, spherical harmonics, irreducible representations, and the general architectural framework."
order: 2
categories: [gnn]
tags: [geometric-deep-learning, equivariance, spherical-harmonics]
toc:
  sidebar: left
---

## Introduction

Equivariant neural networks have revolutionized machine learning for 3D atomic systems, enabling neural networks to operate on molecular and materials data while respecting physical symmetries. This post provides a comprehensive introduction to the mathematical foundations and general architectural framework that underlies modern approaches such as Tensor Field Networks, MACE, EquiFormerV2, and eSEN.

The central insight is that we can build neural networks whose intermediate representations transform predictably under 3D rotations and translations. This property, called **equivariance**, allows models to generalize across all possible orientations of the input, dramatically improving data efficiency and physical consistency. When a molecule is rotated in space, an equivariant network's internal representations rotate accordingly, and any predicted vector quantities (like forces) rotate in the same way. This is in contrast to invariant networks, which discard directional information entirely, or non-equivariant networks, which must learn rotational symmetry from data augmentation.

The mathematical machinery required to build such networks draws from group theory, representation theory, and the theory of spherical harmonics. While these topics may seem abstract at first, they provide a principled and elegant framework for constructing neural networks that respect the fundamental symmetries of 3D space.

---

## Mathematical Foundations

### Groups and Symmetries

Mathematics is fundamentally about decorating a set of elements with additional structure—typically algebraic or topological structures. A **space** is a set endowed with a structure defining the relations among its elements. In geometric deep learning, we are interested in two types of spaces: the space of data (such as 3D atom-wise coordinates of a molecule) and the space of transformations (such as 3D rotations acting on that data).

The language of **groups** provides the natural framework for describing symmetries. A group $(G, \cdot)$ is a set $G$ equipped with a binary operation $\cdot$ that satisfies four fundamental axioms: closure (combining two group elements yields another group element), associativity (the order of applying the operation doesn't matter as long as the sequence is preserved), the existence of an identity element, and the existence of inverses for every element. These axioms capture the essential properties we expect from symmetry transformations—we can compose them, undo them, and there's always a "do nothing" transformation.

A group is called **Abelian** (or commutative) if the order of composition doesn't matter: $a \cdot b = b \cdot a$ for all elements. The integers under addition form an Abelian group. However, the group of 3D rotations is **non-Abelian**—rotating first around the x-axis and then around the y-axis gives a different result than rotating first around the y-axis and then around the x-axis. This non-commutativity is a fundamental feature of 3D rotations and has important implications for how we construct equivariant neural networks.

For 3D molecular systems, several groups are particularly important. The **special orthogonal group** $SO(3)$ consists of all 3D rotations—these are $3 \times 3$ orthogonal matrices with determinant $+1$. The **orthogonal group** $O(3)$ additionally includes reflections (matrices with determinant $-1$), which is relevant when dealing with chiral molecules. The **special Euclidean group** $SE(3)$ combines rotations with translations, representing all rigid motions in 3D space. Most equivariant architectures for atomic systems are designed to be $SE(3)$-equivariant or $E(3)$-equivariant (the latter including reflections).

### Group Representations

A **representation** of a group $G$ on a vector space $V$ is a way of realizing the abstract group elements as concrete linear transformations (matrices) acting on vectors. Formally, it is a homomorphism $\rho: G \to \text{GL}(V)$ satisfying $\rho(g_1 \cdot g_2) = \rho(g_1) \rho(g_2)$. This means that composing two group elements and then applying the representation gives the same result as applying the representations separately and then composing the resulting matrices.

Representations are the key to understanding how neural network features should transform under symmetry operations. If we want our feature vectors to transform in a predictable way when the input is rotated, we need to specify which representation governs that transformation. The simplest representation is the **trivial representation**, where every group element maps to the identity matrix—this describes quantities that don't change under rotation, i.e., scalars. The **standard representation** of $SO(3)$ uses the $3 \times 3$ rotation matrices themselves—this describes how ordinary 3D vectors transform.

But these are just two examples from an infinite family of representations. The theory of **irreducible representations** tells us that any representation can be decomposed into fundamental building blocks that cannot be further reduced. For $SO(3)$, the irreducible representations (irreps) are labeled by non-negative integers $\ell = 0, 1, 2, \ldots$, and the $\ell$-th irrep has dimension $2\ell + 1$. The $\ell = 0$ irrep is the trivial representation (scalars), the $\ell = 1$ irrep is the standard representation (vectors), and higher values of $\ell$ correspond to higher-order tensors with increasingly complex transformation properties.

---

## Spherical Harmonics and Steerable Features

### Spherical Harmonics

The **spherical harmonics** $Y_\ell^m(\theta, \phi)$ are special functions defined on the sphere that form a complete orthonormal basis for square-integrable functions on $S^2$. They are defined as:

$$Y_\ell^m(\theta, \phi) = N_\ell^m P_\ell^m(\cos\theta) e^{im\phi}$$

where $\ell \geq 0$ is the degree, $-\ell \leq m \leq \ell$ is the order, $N_\ell^m$ is a normalization constant, and $P_\ell^m$ is the associated Legendre polynomial. For each degree $\ell$, there are $2\ell + 1$ spherical harmonics corresponding to the different values of $m$.

The crucial property of spherical harmonics is how they transform under rotations. When we rotate the coordinate system by a rotation $R \in SO(3)$, the spherical harmonics of degree $\ell$ mix among themselves according to a $(2\ell+1) \times (2\ell+1)$ matrix called the **Wigner-D matrix** $D^{(\ell)}(R)$:

$$Y_\ell^m(R^{-1}\hat{\mathbf{r}}) = \sum_{m'=-\ell}^{\ell} D^{(\ell)}_{mm'}(R) Y_\ell^{m'}(\hat{\mathbf{r}})$$

This means that spherical harmonics of different degrees don't mix under rotation—they transform independently according to their respective irreducible representations. The degree-0 spherical harmonic ($Y_0^0$) is just a constant and doesn't change under rotation. The three degree-1 spherical harmonics ($Y_1^{-1}, Y_1^0, Y_1^1$) transform like the components of a 3D vector. The five degree-2 spherical harmonics transform like the independent components of a traceless symmetric $3 \times 3$ matrix.

| Degree $\ell$ | Dimension | Physical Interpretation |
|---------------|-----------|------------------------|
| 0 | 1 | Scalar (invariant) |
| 1 | 3 | Vector (dipole) |
| 2 | 5 | Quadrupole |
| 3 | 7 | Octupole |

In equivariant neural networks, spherical harmonics serve two purposes. First, they provide a way to encode directional information about the relative positions of atoms. Given a displacement vector $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ between atoms $i$ and $j$, we can project it onto the unit sphere and compute the spherical harmonics $Y_\ell^m(\hat{\mathbf{r}}_{ij})$ to obtain an equivariant encoding of the direction. Second, the transformation properties of spherical harmonics tell us exactly how to construct feature vectors that transform correctly under rotations.

### Steerable Features

A **steerable feature** of type $\ell$ is a $(2\ell+1)$-dimensional vector that transforms under rotation $R$ according to the Wigner-D matrix: $\mathbf{f}^{(\ell)} \mapsto D^{(\ell)}(R) \mathbf{f}^{(\ell)}$. The term "steerable" comes from the fact that we can predict exactly how the feature will change under any rotation—we can "steer" it to any orientation by applying the appropriate Wigner-D matrix.

Modern equivariant networks maintain features as direct sums of steerable features of multiple types:

$$\mathbf{f} = \mathbf{f}^{(0)} \oplus \mathbf{f}^{(1)} \oplus \mathbf{f}^{(2)} \oplus \cdots \oplus \mathbf{f}^{(L)}$$

where $L$ is the maximum degree used in the network. Each component $\mathbf{f}^{(\ell)}$ may have multiple "channels"—for instance, we might have 64 independent type-1 vectors at each node, giving a type-1 feature of shape $(3, 64)$. The total feature at a node is the concatenation of all these components, and under rotation, each component transforms independently according to its type.

This multi-type structure is essential for expressivity. Using only type-0 (scalar) features would give us an invariant network that cannot predict vector quantities like forces. Using only type-1 (vector) features would limit our ability to represent more complex angular dependencies. By including multiple types up to some maximum degree $L$, we can represent arbitrarily complex angular functions while maintaining exact equivariance.

---

## Clebsch-Gordan Tensor Products

### Combining Steerable Features

The **Clebsch-Gordan (CG) tensor product** is the fundamental operation that allows us to combine steerable features while preserving equivariance. The challenge is that when we take the ordinary tensor product of two irreducible representations, the result is generally **reducible**—it can be decomposed into a direct sum of irreps. The CG tensor product performs this decomposition explicitly.

Mathematically, the tensor product of two irreps decomposes as:

$$D^{(\ell_1)} \otimes D^{(\ell_2)} = \bigoplus_{\ell = |\ell_1 - \ell_2|}^{\ell_1 + \ell_2} D^{(\ell)}$$

For example, when we combine two type-1 (vector) features, the result contains type-0, type-1, and type-2 components: $D^{(1)} \otimes D^{(1)} = D^{(0)} \oplus D^{(1)} \oplus D^{(2)}$. The type-0 output corresponds to the dot product of the two vectors (a scalar), the type-1 output corresponds to the cross product (another vector), and the type-2 output corresponds to the traceless symmetric part of the outer product (a quadrupole).

The **Clebsch-Gordan coefficients** $C^{\ell_1 \ell_2 \ell}_{m_1 m_2 m}$ are the numerical coefficients that define exactly how to combine the components. Given input features $\mathbf{x}^{(\ell_1)}$ and $\mathbf{y}^{(\ell_2)}$, the output component of type $\ell$ is:

$$z^{(\ell)}_m = \sum_{m_1, m_2} C^{\ell_1 \ell_2 \ell}_{m_1 m_2 m} \, x^{(\ell_1)}_{m_1} \, y^{(\ell_2)}_{m_2}$$

The CG tensor product is **equivariant by construction**: if we rotate both input features using their respective Wigner-D matrices, the output features rotate according to their Wigner-D matrices. This is the key property that allows us to build deep networks that maintain equivariance through multiple layers.

### Computational Considerations

A significant challenge with CG tensor products is their computational cost. A naive implementation has complexity $O(L^6)$ where $L$ is the maximum degree, which becomes prohibitive for large $L$. This has motivated substantial research into more efficient implementations. Modern approaches reduce the complexity to $O(L^3)$ through techniques such as aligning features with edge vectors (reducing $SO(3)$ operations to simpler $SO(2)$ operations), computing only selected output types rather than all possible outputs, and exploiting sparsity patterns in the CG coefficients.

---

## General Architectural Framework

### Building Blocks

All modern equivariant architectures for 3D atomic systems share a common framework built from the mathematical components described above. The input to such a network is typically a point cloud consisting of atomic positions $\{\mathbf{r}_i\}$ and atom types (element identities). The output might be a scalar property like energy, vector quantities like forces, or more complex tensorial properties.

The network begins by embedding the atom types into type-0 (scalar) features—this is similar to word embeddings in NLP, but the resulting features are rotation-invariant. The geometric information enters through the edges of the graph: for each pair of neighboring atoms $i$ and $j$, we compute the relative position vector $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, its length $r_{ij} = \|\mathbf{r}_{ij}\|$, and its direction $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij} / r_{ij}$. The direction is encoded using spherical harmonics $Y_\ell^m(\hat{\mathbf{r}}_{ij})$, providing an equivariant representation of the edge geometry.

The core of the network consists of **equivariant message passing layers**. In each layer, node features are updated by aggregating information from neighboring nodes. The message from node $j$ to node $i$ combines the neighbor's features with the geometric information about the edge:

$$\mathbf{m}_{ij}^{(\ell_{\text{out}})} = W(r_{ij}) \left( \mathbf{h}_j^{(\ell_{\text{in}})} \otimes_{\text{CG}} Y^{(\ell_f)}(\hat{\mathbf{r}}_{ij}) \right)^{(\ell_{\text{out}})}$$

Here, $W(r_{ij})$ is a learnable function that depends only on the distance (not the direction), making it rotation-invariant. This radial function is typically implemented as an MLP applied to a radial basis expansion of the distance, or as a linear combination of radial basis functions. The CG tensor product combines the neighbor features with the spherical harmonic edge embeddings, and the superscript indicates which output type we extract. The messages are then aggregated (typically summed) over all neighbors to update the node features:

$$\mathbf{h}_i^{(\ell)} \leftarrow \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(\ell)}$$

Summation is a natural choice for aggregation because it commutes with rotation—rotating all the messages and then summing gives the same result as summing and then rotating.

### Nonlinearities and Output

Applying nonlinear activation functions to steerable features requires care. If we apply a pointwise nonlinearity like ReLU directly to the components of a type-$\ell$ feature (for $\ell > 0$), we break equivariance because the nonlinearity doesn't commute with the Wigner-D rotation matrices. There are several solutions to this problem.

The simplest approach is to apply nonlinearities only to type-0 (scalar) features, which are invariant and can be processed with any standard activation function. For higher-type features, we use only linear transformations within each layer, relying on the CG tensor products between layers to provide the necessary nonlinear mixing. A more sophisticated approach is **gated nonlinearity**, where we use a scalar quantity (such as the norm of a higher-type feature) to gate the feature: $\mathbf{h}^{(\ell)} \leftarrow \sigma(\|\mathbf{h}^{(\ell)}\|) \cdot \mathbf{h}^{(\ell)}$. This preserves equivariance because the norm is invariant and multiplying by a scalar preserves the transformation properties.

For the output layer, the approach depends on what quantity we want to predict. For invariant quantities like total energy, we use only the type-0 features and sum over all atoms to get a single scalar. For equivariant quantities like forces on each atom, we can either use the type-1 features directly or compute forces as the negative gradient of the predicted energy: $\mathbf{F} = -\nabla E$. The gradient approach guarantees energy conservation (the forces are conservative), which is important for molecular dynamics simulations, but requires backpropagation through the energy prediction.

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

## Design Considerations

When designing or choosing an equivariant architecture, several key decisions affect the trade-off between accuracy, computational cost, and physical properties of the predictions.

The **maximum degree $L$** determines how many types of features the network maintains. Higher $L$ allows the network to represent more complex angular dependencies—for instance, $L=0$ can only represent isotropic (spherically symmetric) functions, while $L=2$ can represent quadrupolar angular dependencies. However, the number of feature dimensions grows as $(L+1)^2$, and the cost of CG tensor products grows even faster. In practice, values of $L$ between 1 and 4 are common, with higher values used when angular resolution is critical (such as for predicting anisotropic properties).

The choice of **radial basis functions** affects how the network represents distance-dependent interactions. Common choices include Gaussian radial basis functions centered at different distances, Bessel functions (which arise naturally from solving the Schrödinger equation), and learnable radial networks. An important consideration is the behavior at the **cutoff distance**—the distance beyond which atoms are not considered neighbors. If the radial functions don't go smoothly to zero at the cutoff, the potential energy surface will have discontinuities, causing problems in molecular dynamics simulations. Polynomial envelope functions that ensure smooth cutoffs are often used to address this.

For **force prediction**, there are two main approaches. Direct prediction uses type-1 features to output forces directly, which is computationally efficient but may violate energy conservation—the predicted forces might not correspond to the gradient of any potential energy function. Computing forces as the gradient of energy ($\mathbf{F} = -\nabla E$) guarantees conservative forces but requires backpropagation through the energy prediction, roughly doubling the computational cost. The choice depends on the application: for single-point predictions, direct forces may be acceptable, but for molecular dynamics simulations where energy conservation is important, gradient-based forces are preferred.

---

## Modern Architectures

The general framework described above has been implemented in numerous architectures, each introducing innovations to improve accuracy, efficiency, or specific properties. **Tensor Field Networks** (2018) introduced the basic framework of steerable features and CG tensor products. **NequIP** (2021) demonstrated that this approach could achieve state-of-the-art accuracy for molecular dynamics. **MACE** (2022) introduced higher body-order message passing, allowing the network to capture many-body interactions more efficiently. **Equiformer** (2022) incorporated attention mechanisms from Transformers into the equivariant framework. **EquiFormerV2** (2024) dramatically improved efficiency by reducing $SO(3)$ convolutions to $SO(2)$. **eSEN** (2025) focused on producing smooth potential energy surfaces suitable for stable molecular dynamics simulations.

These architectures differ in their specific implementations, but they all build on the same mathematical foundation: spherical harmonics for encoding directional information, irreducible representations for organizing features by transformation type, CG tensor products for combining features equivariantly, and invariant radial functions for encoding distance information.

---

## Summary

Equivariant neural networks for 3D atomic systems represent a beautiful synthesis of group theory, representation theory, and deep learning. The key ideas are:

1. **Symmetry as a constraint**: By building networks that respect the rotational symmetry of 3D space, we obtain models that generalize across all orientations without data augmentation.

2. **Irreducible representations**: Features are organized by their transformation type (scalar, vector, quadrupole, etc.), with spherical harmonics providing the natural basis.

3. **Clebsch-Gordan tensor products**: These operations combine features of different types while maintaining equivariance, enabling deep networks with complex nonlinear transformations.

4. **Separation of radial and angular**: Distance information is encoded through invariant radial functions, while directional information is encoded through equivariant spherical harmonics.

5. **Message passing on graphs**: Local atomic environments are processed through neighbor aggregation, with the framework extending naturally to any point cloud or graph structure.

This framework has enabled remarkable advances in molecular property prediction, force field development, and materials discovery, and continues to be an active area of research as new architectures push the boundaries of accuracy and efficiency.

---

## References

- Thomas, N., et al. (2018). Tensor Field Networks. [arXiv:1802.08219](https://arxiv.org/abs/1802.08219)
- Batzner, S., et al. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications.
- Batatia, I., et al. (2022). MACE: Higher Order Equivariant Message Passing Neural Networks. [NeurIPS 2022](https://arxiv.org/abs/2206.07697)
- Liao, Y.-L., et al. (2024). EquiformerV2: Improved Equivariant Transformer. [ICLR 2024](https://arxiv.org/abs/2306.12059)
- Bronstein, M. M., et al. (2021). Geometric Deep Learning. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
- Duval, A., et al. (2023). A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems. [arXiv:2312.07511](https://arxiv.org/abs/2312.07511)
