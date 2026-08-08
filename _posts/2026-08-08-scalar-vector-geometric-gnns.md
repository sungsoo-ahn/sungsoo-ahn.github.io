---
layout: post
title: "Scalar and Vector Geometric Graph Networks"
date: 2026-08-08
last_updated: 2026-08-08
description: "How geometric graph networks move from invariant distances and angles to equivariant coordinate updates and vector channels, and what directional information buys."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [geometric-deep-learning]
lecture_paths: [ml4mol, gdl]
tags: [geometric-deep-learning, equivariance, molecular-graphs, schnet, egnn, painn]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the scalar-to-vector storyline from my Machine
  Learning for Molecules and Geometric Deep Learning lectures. The distinction
  is about hidden representations, not scientific ambition: scalar models can
  encode rich geometry, while vector models keep selected directions explicit
  throughout the computation.</em>
</p>

A molecular graph has two kinds of structure. Its edges say which atoms interact, and its coordinates say how those atoms are arranged in space. An ordinary graph neural network sees the first structure. A geometric graph network must use the second without making its prediction depend on an arbitrary orientation of the molecule.

There are two direct ways to meet this requirement. A **scalar geometric network** converts coordinates into invariant numbers such as distances, bond angles, and torsion angles, then processes those numbers with an ordinary message-passing network. A **vector geometric network** also maintains features that rotate with the molecule. The scalar route removes orientation before learning; the vector route lets orientation flow through the hidden state under a controlled transformation law.

The distinction is easy to overstate. A complete collection of distances can determine a point cloud up to rigid transformations and reflection. An invariant energy model can produce equivariant forces by differentiation. Conversely, a vector channel does not guarantee that a network has captured every geometric relation. The practical question is which geometric correlations the architecture makes easy to represent.

## Geometry fixes the transformation law

Let a geometric graph have node positions $$\mathbf{x}_i\in\mathbb{R}^3$$ and initial node features $$\mathbf{h}_i$$. A Euclidean transformation sends every position to

$$
\mathbf{x}_i' = \mathbf{Q}\mathbf{x}_i + \mathbf{b},
$$

where $$\mathbf{Q}\in\mathbb{R}^{3\times 3}$$ is orthogonal, so $$\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}$$, and $$\mathbf{b}\in\mathbb{R}^3$$ is a translation. Allowing $$\det(\mathbf{Q})=-1$$ includes reflections as well as rotations.

The target determines how a prediction should transform. Total energy, atom type, and charge are scalars. They should not change:

$$
E(\mathbf{Q}\mathbf{X}+\mathbf{b})=E(\mathbf{X}).
$$

Force, velocity, and displacement are vectors. They should rotate or reflect with the coordinates:

$$
\mathbf{F}_i(\mathbf{Q}\mathbf{X}+\mathbf{b})
=\mathbf{Q}\mathbf{F}_i(\mathbf{X}).
$$

The first property is **invariance**; the second is **equivariance**. Both prevent the model from relearning the same physical configuration in every orientation. They differ only because the outputs mean different things.

Relative displacements remove translation immediately:

$$
\mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i,
\qquad
\mathbf{r}_{ij}'=\mathbf{Q}\mathbf{r}_{ij}.
$$

The displacement is equivariant. Its length is invariant because orthogonal transformations preserve inner products:

$$
(r_{ij}')^2
=(\mathbf{Q}\mathbf{r}_{ij})^{\mathsf T}(\mathbf{Q}\mathbf{r}_{ij})
=\mathbf{r}_{ij}^{\mathsf T}\mathbf{r}_{ij}
=r_{ij}^{2}.
$$

Every architecture in this post starts from these two facts: relative vectors transform predictably, and inner products of relative vectors do not change.

## Scalarization turns geometry into numbers

**Scalarization** maps transformation-sensitive coordinates to invariant scalars. The distance between two atoms is the simplest example. Three atoms define a bond angle through an inner product, and four atoms define a torsion angle through the relative orientation of two bond planes.

{% include figure.liquid loading="eager" path="assets/img/blog/svgnn_geometric_scalars.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Distance, bond angle, and torsion are invariant scalarizations of a geometric neighborhood. They discard the global pose while retaining progressively higher-order relations among two, three, and four atoms. Original figure." %}

For atoms $$i$$, $$j$$, and $$k$$, define unit directions $$\hat{\mathbf{r}}_{ji}=\mathbf{r}_{ji}/r_{ji}$$ and $$\hat{\mathbf{r}}_{jk}=\mathbf{r}_{jk}/r_{jk}$$. Their bond angle satisfies

$$
\cos\theta_{ijk}
=\hat{\mathbf{r}}_{ji}^{\mathsf T}\hat{\mathbf{r}}_{jk}.
$$

Rotating both directions by $$\mathbf{Q}$$ leaves the inner product unchanged. Torsion angles follow the same principle with plane normals. Given consecutive bonds, construct

$$
\mathbf{n}_1=\mathbf{r}_{ij}\times\mathbf{r}_{jk},
\qquad
\mathbf{n}_2=\mathbf{r}_{jk}\times\mathbf{r}_{k\ell},
$$

then compare $$\mathbf{n}_1$$ and $$\mathbf{n}_2$$. A signed torsion additionally uses the bond direction to define orientation. Whether that sign should survive reflection depends on the physical target and the chosen symmetry group.

Scalarization is attractive because the rest of the network can use unconstrained scalar operations. Multilayer perceptrons, componentwise nonlinearities, attention weights, and sum aggregation all preserve invariance when their inputs are invariant.

## Distance-conditioned message passing

SchNet made the radial version of this idea explicit (<span id="cite-schutt2017"></span>[Schütt et al., 2017](#ref-schutt2017)). Each atom begins with an embedding determined by its element. A continuous filter network maps the distance $$r_{ij}$$ to a learned weight, and a message-passing layer updates atom $$i$$ as

$$
\mathbf{h}_i^{(t+1)}
=\mathbf{h}_i^{(t)}
+\sum_{j\in\mathcal{N}(i)}
\mathbf{h}_j^{(t)}\odot
\mathbf{W}^{(t)}(r_{ij}).
$$

Here $$\odot$$ denotes channel-wise multiplication, and $$\mathbf{W}^{(t)}$$ is a learnable function of distance. SchNet expands $$r_{ij}$$ in radial basis functions before applying the filter network. The radial expansion turns a single number into a smooth feature vector, making it easier to learn interactions that vary sharply at short range and decay toward a cutoff.

The layer is invariant because atom features and filter weights are invariant. It still represents geometry: two carbon atoms at 1.4 angstroms can send a different message from two carbon atoms at 3 angstroms. What it does not expose in one layer is how two edges around the same center are oriented relative to each other.

A distance-based energy model can nevertheless predict equivariant forces. Let the network produce an invariant potential energy $$\widehat E(\mathbf{X})$$, and define

$$
\widehat{\mathbf{F}}_i(\mathbf{X})
=-\nabla_{\mathbf{x}_i}\widehat E(\mathbf{X}).
$$

Differentiate the invariance relation $$\widehat E(\mathbf{Q}\mathbf{X}+\mathbf{b})=\widehat E(\mathbf{X})$$. Because $$\mathbf{Q}^{-\mathsf T}=\mathbf{Q}$$ for an orthogonal matrix,

$$
\widehat{\mathbf{F}}_i(\mathbf{Q}\mathbf{X}+\mathbf{b})
=\mathbf{Q}\widehat{\mathbf{F}}_i(\mathbf{X}).
$$

The force inherits equivariance from the invariant scalar energy. It is also conservative by construction, since it is a gradient. This route is often the right one for molecular dynamics, where energy conservation matters more than producing a vector in one forward pass.

## Direction can remain scalar

Distance alone does not mean that scalar models are restricted to pairwise physics. DimeNet attaches hidden states to directed edges and lets messages interact through bond angles (<span id="cite-gasteiger2020"></span>[Gasteiger et al., 2020](#ref-gasteiger2020)). A message on directed edge $$j\to i$$ receives information from an incoming edge $$k\to j$$ together with the distance $$r_{ji}$$ and angle $$\theta_{kji}$$:

$$
\mathbf{m}_{ji}^{(t+1)}
=U^{(t)}\!\left(
\mathbf{m}_{ji}^{(t)},
\sum_{k\in\mathcal{N}(j)\setminus\{i\}}
M^{(t)}\!\left(
\mathbf{m}_{kj}^{(t)},r_{ji},r_{kj},\theta_{kji}
\right)
\right).
$$

All quantities passed to $$M^{(t)}$$ are invariant. The hidden state remains scalar, but the update knows whether two bonds are aligned, orthogonal, or bent. GemNet extends this directional message-passing view with interactions designed to capture richer geometric correlations and establishes universality results for its setting (<span id="cite-klicpera2021"></span>[Klicpera et al., 2021](#ref-klicpera2021)).

A two-neighbor example shows exactly what the angle contributes. Place a center atom at the origin and two neighbors at unit distance:

$$
\mathbf{r}_1=(1,0),
\qquad
\mathbf{r}_2=(\cos\theta,\sin\theta).
$$

A one-layer radial update at the center sees the multiset $$\{\!\{1,1\}\!\}$$ for every value of $$\theta$$. The sum of the two directions carries the missing correlation:

$$
\begin{aligned}
\left\lVert\mathbf{r}_1+\mathbf{r}_2\right\rVert^2
&=(\mathbf{r}_1+\mathbf{r}_2)^{\mathsf T}
(\mathbf{r}_1+\mathbf{r}_2)\\
&=2+2\cos\theta.
\end{aligned}
$$

The norm equals $$\sqrt{3}$$ at $$60^\circ$$ and zero at $$180^\circ$$. An angle feature supplies $$\cos\theta$$ directly; a vector channel can form the directional sum and scalarize it later.

{% include figure.liquid loading="eager" path="assets/img/blog/svgnn_directional_ambiguity.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Two neighborhoods can have the same center--neighbor distance multiset but different bond angles. Their vector sums have different norms, so either an explicit angle or an equivariant vector channel separates them in one local update. Original figure." %}

The example isolates a property of the *local computation*, not a theorem that distances are fundamentally incomplete. If the model receives every pairwise distance, then the neighbor--neighbor distance obeys

$$
r_{12}^2=r_1^2+r_2^2-2r_1r_2\cos\theta,
$$

so it determines the angle. A sparse radial GNN may need additional layers to make that distance available at the center, while a directional architecture exposes the three-body relation immediately. Geometry features change the path by which information enters the model.

## Vector channels keep orientation alive

A vector geometric network carries two feature types at each node. Scalar channels $$\mathbf{s}_i\in\mathbb{R}^{C_s}$$ remain unchanged under $$\mathbf{Q}$$. Vector channels $$\mathbf{V}_i\in\mathbb{R}^{3\times C_v}$$ contain $$C_v$$ ordinary three-dimensional vectors and transform as

$$
\mathbf{s}_i' = \mathbf{s}_i,
\qquad
\mathbf{V}_i'=\mathbf{Q}\mathbf{V}_i.
$$

The network may freely apply nonlinear functions to scalars. Vector operations must commute with the transformation. Safe operations include linear combinations of vectors, multiplication by invariant scalar gates, and scalarization through norms or inner products.

{% include figure.liquid loading="eager" path="assets/img/blog/svgnn_scalar_vector_channels.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Scalar channels remain fixed while vector channels rotate with the coordinates. Norms and inner products turn vectors into invariants, and invariant functions can gate vectors without changing their transformation law. Original figure." %}

A componentwise ReLU is not safe for a vector. In general,

$$
\operatorname{ReLU}(\mathbf{Q}\mathbf{v})
\neq \mathbf{Q}\operatorname{ReLU}(\mathbf{v}),
$$

because rotating a vector mixes its Cartesian components before the nonlinearity clips them. A scalar gate is safe:

$$
\mathbf{v}'=g(\mathbf{s},\lVert\mathbf{v}\rVert)\mathbf{v}.
$$

The gate is invariant and only changes the vector's magnitude. If the input rotates, the output rotates by the same matrix.

Vector channels are directional memory. A node can accumulate where its neighbors lie, preserve that information across layers, and use it to construct a vector target directly. Scalar channels still do most of the flexible nonlinear work; vectors provide an orientation-aware route between geometric input and output.

## EGNN derives vectors from coordinate differences

The E(n)-Equivariant Graph Neural Network (EGNN) builds an equivariant layer from relative coordinates and invariant weights (<span id="cite-satorras2021"></span>[Satorras et al., 2021](#ref-satorras2021)). It first computes an invariant edge message

$$
\mathbf{m}_{ij}
=\phi_e\!\left(
\mathbf{h}_i,
\mathbf{h}_j,
\lVert\mathbf{x}_i-\mathbf{x}_j\rVert^2,
\mathbf{a}_{ij}
\right),
$$

where $$\mathbf{a}_{ij}$$ is an optional invariant edge attribute. The layer then updates coordinates with a weighted sum of relative vectors:

$$
\mathbf{x}_i^{+}
=\mathbf{x}_i
+C\sum_{j\neq i}
(\mathbf{x}_i-\mathbf{x}_j)\,
\phi_x(\mathbf{m}_{ij}).
$$

The scalar $$\phi_x(\mathbf{m}_{ij})$$ decides how strongly neighbor $$j$$ pushes or pulls node $$i$$. The node feature update remains invariant:

$$
\mathbf{h}_i'
=\phi_h\!\left(
\mathbf{h}_i,
\sum_{j\neq i}\mathbf{m}_{ij}
\right).
$$

{% include figure.liquid loading="eager" path="assets/img/blog/svgnn_egnn_coordinate_update.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An EGNN layer computes invariant edge messages from features and squared distances, then uses scalar message weights to combine relative vectors. The resulting displacement transforms with the geometry, so the updated coordinate is equivariant. Original figure." %}

The equivariance proof fits in one line once the ingredients are separated. Write the transformed input as $$\widetilde{\mathbf{x}}_i=\mathbf{Q}\mathbf{x}_i+\mathbf{b}$$. Squared distances and messages remain unchanged, while

$$
(\widetilde{\mathbf{x}}_i-\widetilde{\mathbf{x}}_j)
=\mathbf{Q}(\mathbf{x}_i-\mathbf{x}_j).
$$

Therefore the displacement becomes

$$
\Delta\widetilde{\mathbf{x}}_i
=C\sum_{j\neq i}
\mathbf{Q}(\mathbf{x}_i-\mathbf{x}_j)\phi_x(\mathbf{m}_{ij})
=\mathbf{Q}\Delta\mathbf{x}_i.
$$

Adding the displacement gives $$\widetilde{\mathbf{x}}_i^{+}=\mathbf{Q}\mathbf{x}_i^{+}+\mathbf{b}$$, as required.

For a concrete update, let $$\mathbf{x}_i=(0,0)$$, $$\mathbf{x}_1=(1,0)$$, and $$\mathbf{x}_2=(0,1)$$. If $$C=1$$ and the two invariant weights are $$1/2$$ and $$1/4$$, then

$$
\Delta\mathbf{x}_i
=\frac{1}{2}(\mathbf{x}_i-\mathbf{x}_1)
+\frac{1}{4}(\mathbf{x}_i-\mathbf{x}_2)
=\left(-\frac{1}{2},-\frac{1}{4}\right).
$$

Rotating the three input points by $$90^\circ$$ rotates this displacement to $$\left(1/4,-1/2\right)$$ without changing either weight. The learned functions only choose invariant coefficients; geometry supplies the directions.

## PaiNN stores vectors as hidden features

EGNN updates coordinates themselves. PaiNN instead keeps dedicated scalar and vector feature channels for each atom (<span id="cite-schutt2021"></span>[Schütt et al., 2021](#ref-schutt2021)). A simplified message illustrates the division of labor:

$$
\begin{aligned}
\mathbf{m}_{i}^{s}
&=\sum_{j\in\mathcal{N}(i)}
\phi_s(\mathbf{s}_j,r_{ij}),\\
\mathbf{M}_{i}^{v}
&=\sum_{j\in\mathcal{N}(i)}
\left[
\mathbf{V}_j\mathbf{W}_{vv}(\mathbf{s}_j,r_{ij})
+\hat{\mathbf{r}}_{ij}\,\mathbf{w}_{vs}(\mathbf{s}_j,r_{ij})^{\mathsf T}
\right].
\end{aligned}
$$

The first line sends invariant scalar messages. The second combines existing vector features with the equivariant direction $$\hat{\mathbf{r}}_{ij}$$. Every learned coefficient depends only on invariant inputs, so the entire vector message rotates with the molecule.

PaiNN couples the channels by scalarizing vectors through norms and inner products, then using the resulting invariants to update scalars and gate vectors. This pattern lets orientation influence nonlinear scalar processing without applying an invalid componentwise nonlinearity to a vector.

Dedicated vector features are especially natural for dipole moments and forces. A readout can combine vector channels directly instead of recovering direction only through an energy gradient. The gradient route remains preferable when the target must be conservative; direct vector prediction is useful when no scalar potential defines the quantity or when one forward pass matters.

## Four representative design points

The scalar/vector distinction is a spectrum of where geometry enters, not a contest between two mutually exclusive families.

| Model | Hidden geometric state | Geometric input per update | Main consequence |
|---|---|---|---|
| SchNet | scalar node channels | pair distance $$r_{ij}$$ | simple invariant energy model; forces can come from the energy gradient |
| DimeNet / GemNet | scalar directed-edge channels | distances plus angles and richer multi-atom relations | exposes directional correlations while keeping hidden features invariant |
| EGNN | invariant node channels plus coordinates | squared distances and relative displacement vectors | updates point positions equivariantly with a small set of constrained operations |
| PaiNN | scalar and vector node channels | distances, unit directions, and vector features | carries directional memory and predicts tensorial quantities directly |

SchNet is not “non-geometric”; distance is geometry. DimeNet is not vector-valued merely because it uses directions; it scalarizes those directions into angles. EGNN uses vectors but does not maintain a general hierarchy of tensor types. PaiNN maintains vector features but still relies on scalar channels for flexible nonlinear transformations.

The comparison should also not be read as a universal accuracy ranking. A radial model may be sufficient when the target is dominated by pair distances and data are abundant. An angle-aware scalar model can encode local chemistry efficiently. Coordinate updates are natural for point-cloud dynamics and structure generation. Vector channels help when orientation must persist across several interactions or reach a vector output directly.

## What directional information buys

Directional information buys a shorter computational path to geometric correlations. In the bent-versus-linear example, a radial center update needs information about the edge between the two neighbors or additional message-passing steps. An angle feature or vector sum separates the structures immediately. The advantage is not that direction contains information absent from every possible distance representation. It is that the architecture makes the relevant relation local and explicit.

Direction also aligns hidden features with equivariant targets. If the model predicts a force, a vector channel already lives in the correct output space. The network learns how much to combine available directions instead of learning an unconstrained Cartesian mapping and hoping data enforce the rotation law.

These benefits can improve data efficiency, but they impose constraints and cost. Directional scalar models enumerate triplets or quadruplets around an atom. Vector models store three numbers per vector channel and restrict their nonlinearities. Coordinate updates may be undesirable when coordinates are observations that should remain fixed rather than latent states to refine.

Symmetry remains a design choice. The constructions above use $$E(3)$$ or $$O(3)$$ behavior, which treats mirror images according to reflection symmetry. A property that changes sign under reflection requires parity-aware feature types, not only ordinary invariant scalars and polar vectors. The architecture should match the target's transformation law rather than adopt the largest familiar symmetry by default.

Scalar and vector networks therefore solve the same problem at different points in the computation. Scalar models answer, “Which invariant geometric measurements should we expose?” Vector models answer, “Which directions should the hidden state preserve?” Good geometric modeling often uses both answers: invariant scalars for expressive nonlinear processing, and equivariant vectors for the directional structure that should not be discarded.

---

## References

- <span id="ref-schutt2017"></span>Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., & Müller, K.-R. (2017). SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions. *NeurIPS 2017*. [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html). <a href="#cite-schutt2017" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-gasteiger2020"></span>Gasteiger, J., Groß, J., & Günnemann, S. (2020). Directional Message Passing for Molecular Graphs. *ICLR 2020*. [OpenReview](https://openreview.net/forum?id=B1eWbxStPH). <a href="#cite-gasteiger2020" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-klicpera2021"></span>Klicpera, J., Becker, F., & Günnemann, S. (2021). GemNet: Universal Directional Graph Neural Networks for Molecules. *NeurIPS 2021*. [OpenReview](https://openreview.net/forum?id=HS_sOaxS9K-). <a href="#cite-klicpera2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-satorras2021"></span>Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. *ICML 2021*. [PMLR](https://proceedings.mlr.press/v139/satorras21a.html). <a href="#cite-satorras2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-schutt2021"></span>Schütt, K. T., Unke, O. T., & Gastegger, M. (2021). Equivariant Message Passing for the Prediction of Tensorial Properties and Molecular Spectra. *ICML 2021*. [PMLR](https://proceedings.mlr.press/v139/schutt21a.html). <a href="#cite-schutt2021" class="reversefootnote" role="doc-backlink">↩</a>
