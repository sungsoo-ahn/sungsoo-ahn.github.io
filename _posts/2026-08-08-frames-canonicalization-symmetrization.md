---
layout: post
title: "Frames, Canonicalization, and Symmetrization"
date: 2026-08-08
last_updated: 2026-08-08
description: "How local frames, canonicalization, frame averaging, and probabilistic symmetrization turn arbitrary neural networks into geometric models—and why continuity is the central difficulty."
abstract: >
  A frame can express geometry in invariant coordinates, choose a canonical pose, or reduce an infinite symmetry average to a finite computation. These uses share one idea but have different failure modes.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [geometric-deep-learning]
lecture_paths: [gdl]
tags: [equivariance, canonicalization, frame-averaging, local-frames, geometric-deep-learning]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the storyline of my 2025 Geometric Deep Learning lecture on frame-based models. It continues the language of group actions and equivariance introduced in <a href="{% post_url 2026-08-08-symmetry-equivariance-geometric-data %}">Symmetry and Equivariance for Geometric Data</a>, while deriving the frame constructions from first principles.</em>
</p>

An equivariant network is usually built from equivariant layers. Every intermediate feature transforms according to a prescribed representation, and each operation preserves that rule. Frames offer another route: transform the input into one or more reference poses, apply an ordinary neural network there, then transform or average the outputs.

This architecture-agnostic recipe is appealing because the backbone can be an MLP, Transformer, or pretrained model with no built-in geometric symmetry. But the apparent simplicity moves the hard problem elsewhere. A model must choose a pose continuously, or it must average over enough poses to make the choice irrelevant. Symmetric objects make a unique continuous choice impossible.

The same tension appears locally and globally. A local frame turns vectors near an edge into invariant scalar coordinates, where an ordinary nonlinear network can process them. A global frame chooses a coordinate system for an entire point cloud. Frame averaging keeps several such choices instead of trusting one. Probabilistic symmetrization replaces a finite unweighted set by a learned distribution. Each step spends more computation to reduce the instability of choosing a single pose.

## Symmetry identifies an orbit, not a preferred pose

Let a group $$G$$ act on an input space $$\mathcal{X}$$. We write $$g\cdot x$$ for the transformed input. For a three-dimensional point cloud, $$G$$ might be the rotation group $$SO(3)$$, the orthogonal group $$O(3)$$ including reflections, or the Euclidean group with translations.

The **orbit** of an input $$x$$ is the set of all descriptions related by the group:

$$
G\cdot x=\{g\cdot x:g\in G\}.
$$

A rotation-invariant target is constant on this orbit. A rotation-equivariant target is not constant, but its values along the orbit are determined by one value and an output representation $$\rho_Y$$:

$$
f(g\cdot x)=\rho_Y(g)f(x).
$$

The set of orbits is the **quotient space** $$\mathcal{X}/G$$. Passing to the quotient discards pose while retaining shape. Canonicalization tries to implement this abstract quotient by selecting one concrete representative from every orbit.

{% include figure.liquid loading="eager" path="assets/img/blog/framesym_orbit_quotient.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An orbit contains every pose of the same geometric object. Canonicalization chooses one representative so an ordinary backbone can operate on shape rather than pose; an equivariant output later restores the original pose. Original diagram." %}

To make the construction precise, let $$h(x)\in G$$ denote the pose that carries a canonical representative to $$x$$. The canonicalized input is

$$
c(x)=h(x)^{-1}\cdot x.
$$

The pose map must itself be equivariant:

$$
h(g\cdot x)=g\,h(x).
$$

This condition makes the canonicalized input invariant:

$$
c(g\cdot x)
=h(g\cdot x)^{-1}\cdot(g\cdot x)
=h(x)^{-1}g^{-1}g\cdot x
=c(x).
$$

An arbitrary backbone $$\phi$$ can now produce an invariant prediction

$$
F_{\mathrm{inv}}(x)=\phi(c(x)).
$$

For an equivariant prediction, apply the output representation to restore the pose:

$$
F_{\mathrm{eq}}(x)
=\rho_Y(h(x))\,\phi(c(x)).
$$

Substituting $$h(g\cdot x)=gh(x)$$ immediately gives $$F_{\mathrm{eq}}(g\cdot x)=\rho_Y(g)F_{\mathrm{eq}}(x)$$. Kaba et al. use this factorization to learn the canonicalizer with a small equivariant network while leaving the main backbone unrestricted (<span id="cite-kaba2023"></span>[Kaba et al., 2023](#ref-kaba2023)).

## A unique canonical pose cannot always vary continuously

Canonicalization fails first at objects with nontrivial symmetry. The **stabilizer** of $$x$$ is

$$
G_x=\{g\in G:g\cdot x=x\}.
$$

If $$x$$ is a perfectly regular tetrahedron, for example, several rotations leave it unchanged after the vertices are treated as an unordered set. Suppose a single-valued pose map $$h$$ existed at such an input and satisfied the equivariance condition. For any non-identity $$s\in G_x$$,

$$
h(x)=h(s\cdot x)=s\,h(x).
$$

Left multiplication of a group element is free, so this equality would imply $$s=e$$, a contradiction. A unique equivariant pose cannot be assigned to an object that does not itself have a unique pose.

Even inputs with trivial stabilizers can approach a symmetric input. A deterministic choice must then jump somewhere. Principal-component analysis (PCA) makes the problem concrete. Center a point cloud $$\mathbf{X}\in\mathbb{R}^{N\times d}$$ by subtracting its mean $$\boldsymbol{\mu}$$ and form the covariance

$$
\boldsymbol{\Sigma}
=\sum_{i=1}^{N}
(\mathbf{x}_i-\boldsymbol{\mu})
(\mathbf{x}_i-\boldsymbol{\mu})^{\mathsf T}.
$$

When the eigenvalues are distinct, its eigenvectors define principal axes, though each axis still has an arbitrary sign. When two eigenvalues coincide, any orthonormal basis of their shared eigenspace is valid. Near that tie, a tiny perturbation can exchange the eigenvalue order or rotate the selected axes by a large angle.

{% include figure.liquid loading="eager" path="assets/img/blog/framesym_canonicalization_discontinuity.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="PCA gives a stable main axis to an elongated point cloud, but nearly equal eigenvalues make that axis sensitive to tiny perturbations. At an exact tie every basis of the repeated eigenspace is valid, so any deterministic convention must jump along some path through shape space. Original diagram." %}

This is not only a flaw of PCA. Dym et al. show that for common group actions there is no efficiently computable unweighted frame that preserves continuity for every function being averaged (<span id="cite-dym2024"></span>[Dym et al., 2024](#ref-dym2024)). The geometric obstruction is that a global continuous choice of representative—a continuous section of the quotient map—often does not exist.

Exact equivariance and numerical stability are therefore separate properties. A canonicalizer can obey the symmetry equation everywhere it is defined yet change abruptly near a degenerate configuration. A smooth backbone composed with that canonicalizer can inherit the discontinuity.

## Local frames avoid choosing one pose for the whole object

A local frame restricts the choice to a neighborhood, an edge, or a small geometric motif. This is enough to convert vector information into invariant scalars without canonicalizing the entire structure.

Consider two non-collinear relative vectors $$\mathbf{r},\mathbf{s}\in\mathbb{R}^{3}$$. We can construct a right-handed orthonormal frame by Gram–Schmidt:

$$
\mathbf{e}_1=\frac{\mathbf{r}}{\lVert\mathbf{r}\rVert},
$$

$$
\widetilde{\mathbf{e}}_2
=\mathbf{s}-(\mathbf{e}_1^{\mathsf T}\mathbf{s})\mathbf{e}_1,
\qquad
\mathbf{e}_2
=\frac{\widetilde{\mathbf{e}}_2}
{\lVert\widetilde{\mathbf{e}}_2\rVert},
$$

$$
\mathbf{e}_3=\mathbf{e}_1\times\mathbf{e}_2.
$$

Collect the axes as columns of $$\mathbf{E}=[\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3]$$. Under a rotation $$\mathbf{R}\in SO(3)$$, both relative vectors rotate, and the frame transforms as $$\mathbf{E}\mapsto\mathbf{R}\mathbf{E}$$.

Any geometric vector $$\mathbf{v}$$ can be **scalarized** into its coordinates in this frame:

$$
\mathbf{s}_{\mathbf{v}}
=\mathbf{E}^{\mathsf T}\mathbf{v}
=\begin{bmatrix}
\mathbf{e}_1^{\mathsf T}\mathbf{v}\\
\mathbf{e}_2^{\mathsf T}\mathbf{v}\\
\mathbf{e}_3^{\mathsf T}\mathbf{v}
\end{bmatrix}.
$$

These coordinates are rotation invariant because

$$
(\mathbf{R}\mathbf{E})^{\mathsf T}(\mathbf{R}\mathbf{v})
=\mathbf{E}^{\mathsf T}\mathbf{v}.
$$

An ordinary nonlinear network can process $$\mathbf{s}_{\mathbf{v}}$$ along with distances, atom types, and other scalar features. If it outputs coefficients $$\mathbf{a}$$, **vectorization** reconstructs a geometric vector:

$$
\mathbf{v}'=\mathbf{E}\mathbf{a}
=a_1\mathbf{e}_1+a_2\mathbf{e}_2+a_3\mathbf{e}_3.
$$

The reconstructed vector rotates with the input because the frame does. ClofNet uses this scalarize–process–vectorize pattern to build expressive $$SE(3)$$-equivariant graph networks from complete local frames (<span id="cite-du2022"></span>[Du et al., 2022](#ref-du2022)).

{% include figure.liquid loading="eager" path="assets/img/blog/framesym_local_frame.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A local orthonormal frame turns a vector into three rotation-invariant coordinates through dot products. An unrestricted nonlinear network processes the coordinates, and vectorization combines its scalar outputs with the rotating frame axes to recover an equivariant vector. Original diagram." %}

The construction has boundaries. It becomes undefined when $$\mathbf{r}=\mathbf{0}$$ or when $$\mathbf{r}$$ and $$\mathbf{s}$$ are parallel, because the second Gram–Schmidt vector vanishes. Local-frame methods must choose geometric inputs that avoid these degeneracies, provide fallbacks, or retain several possible frames. Frames defined at neighboring nodes can also disagree; a model that mixes their coordinates needs to encode the transition from one frame to another.

The cross product introduces a second qualification. Under an improper rotation $$\mathbf{R}\in O(3)$$ with determinant $$-1$$,

$$
(\mathbf{R}\mathbf{e}_1)\times(\mathbf{R}\mathbf{e}_2)
=\det(\mathbf{R})\,\mathbf{R}(\mathbf{e}_1\times\mathbf{e}_2).
$$

The third axis is a pseudovector. A right-handed cross-product frame is naturally $$SO(3)$$ equivariant, not automatically $$O(3)$$ equivariant. Reflection-equivariant models must track parity or include reflected frames rather than treating every vector component as the same type.

## Group averaging symmetrizes an arbitrary function

Canonicalization evaluates the backbone once. Group averaging takes the opposite approach: evaluate it at every pose and remove the pose dependence by integration.

Assume for now that $$G$$ is finite or compact, so it has a normalized invariant measure $$\mu$$. For any function $$\phi:\mathcal{X}\to\mathcal{Y}$$, define the equivariant symmetrization

$$
(\mathcal{S}\phi)(x)
=\int_G
\rho_Y(g)\,
\phi(g^{-1}\cdot x)
\,d\mu(g).
$$

The invariant case uses the trivial output representation, $$\rho_Y(g)=1$$. No property of $$\phi$$ is required. Equivariance follows from a change of variables. For $$h\in G$$, set $$g=hk$$:

$$
\begin{aligned}
(\mathcal{S}\phi)(h\cdot x)
&=\int_G \rho_Y(g)\,
\phi(g^{-1}h\cdot x)\,d\mu(g)\\
&=\int_G \rho_Y(hk)\,
\phi(k^{-1}\cdot x)\,d\mu(k)\\
&=\rho_Y(h)(\mathcal{S}\phi)(x).
\end{aligned}
$$

The measure is unchanged because Haar measure is invariant under group multiplication. For a finite group, the integral is the average over all elements.

Continuous rotations make exact group averaging expensive, while the full Euclidean group is noncompact and has no normalized Haar probability measure. Geometric models usually remove translation first by centering coordinates, then average over a compact rotation or orthogonal group. Monte Carlo integration over rotations is possible, but many backbone evaluations may be needed for a low-variance result.

## Frame averaging replaces the group by a data-dependent set

A **frame** is a set-valued pose map $$\mathcal{F}:\mathcal{X}\to 2^G$$ that transforms equivariantly:

$$
\mathcal{F}(g\cdot x)=g\mathcal{F}(x)
=\{gh:h\in\mathcal{F}(x)\}.
$$

Instead of averaging over all of $$G$$, frame averaging uses the finite set $$\mathcal{F}(x)$$:

$$
(\mathcal{A}_{\mathcal{F}}\phi)(x)
=\frac{1}{\lvert\mathcal{F}(x)\rvert}
\sum_{h\in\mathcal{F}(x)}
\rho_Y(h)\,
\phi(h^{-1}\cdot x).
$$

The proof of equivariance is the same change-of-variables argument as for group averaging. The essential condition is not that each pose be canonical. It is that the entire set transform consistently.

PCA gives a common frame construction for a centered point cloud. If $$\mathbf{U}=[\mathbf{u}_1,\ldots,\mathbf{u}_d]$$ contains principal axes with distinct eigenvalues, each eigenvector has two sign choices. Keeping all $$2^d$$ signed bases produces an $$O(d)$$ frame; keeping only bases with determinant $$+1$$ produces the orientation-preserving subset for $$SO(d)$$. Averaging removes the arbitrary sign convention. Puny et al. show that such frame averages can turn unrestricted backbones into invariant or equivariant universal approximators under their stated assumptions (<span id="cite-puny2022"></span>[Puny et al., 2022](#ref-puny2022)).

{% include figure.liquid loading="eager" path="assets/img/blog/framesym_averaging_recipes.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Group averaging evaluates every pose, frame averaging evaluates a finite equivariant set, and canonicalization evaluates one selected pose. All three can wrap an unrestricted backbone, but they trade computation against the stability of pose selection. Original diagram." %}

Frame averaging resolves a finite sign ambiguity but not every degeneracy. When PCA eigenvalues repeat, the admissible bases form a continuous family inside the repeated eigenspace. Selecting a finite subset can still jump under small perturbations. Exact equivariance at each input does not guarantee that the averaged network is continuous across these jumps.

The number of evaluations is also direct computational overhead. A three-dimensional PCA frame with all sign combinations requires eight backbone passes for $$O(3)$$. Sampling one frame reduces the cost to one pass, as stochastic frame-averaging models do, but introduces estimator variance. Symmetry may then hold in expectation or in distribution rather than for every finite random draw.

## Probabilistic symmetrization makes the frame weighted

Canonicalization, finite frame averaging, and group averaging can be written with one conditional distribution over poses. Let $$q(dg\mid x)$$ be a probability measure on $$G$$ that transforms by pushforward:

$$
q(\,\cdot\mid h\cdot x)=h_{\#}q(\,\cdot\mid x).
$$

This means that transforming the input by $$h$$ left-multiplies every sampled pose by $$h$$. The symmetrized function is

$$
F_q(x)
=\mathbb{E}_{g\sim q(\cdot\mid x)}
\left[
\rho_Y(g)\phi(g^{-1}\cdot x)
\right].
$$

A point mass at one equivariant pose gives deterministic canonicalization. A uniform distribution on a finite frame gives frame averaging. Haar measure gives group averaging. A learned weighted distribution gives **probabilistic symmetrization**.

Kim et al. parameterize the pose distribution with a small equivariant network and train it jointly with an arbitrary backbone (<span id="cite-kim2023"></span>[Kim et al., 2023](#ref-kim2023)). The distribution can concentrate computation on useful poses instead of weighting every frame equally. Dym et al.'s weighted-frame construction provides a complementary theoretical point: carefully chosen weights can preserve continuity where finite unweighted frames cannot.

In practice, the expectation is estimated from samples. If the sampler obeys the pushforward law, the estimator is equivariant in distribution and the expectation is exactly equivariant. A finite independent Monte Carlo estimate need not give identical numerical outputs for two transformed copies unless their randomness is coupled consistently. More samples reduce this variance but recover the same computation–accuracy tradeoff as group averaging.

## Universality does not settle efficiency or stability

Symmetrization preserves the functions we care about. The operator $$\mathcal{S}$$ is linear and idempotent:

$$
\mathcal{S}(a\phi+b\psi)
=a\mathcal{S}\phi+b\mathcal{S}\psi,
\qquad
\mathcal{S}(\mathcal{S}\phi)=\mathcal{S}\phi.
$$

An already equivariant target $$f^{\star}$$ is a fixed point, so $$\mathcal{S}f^{\star}=f^{\star}$$. If an unrestricted backbone family can approximate $$f^{\star}$$ on a compact domain, symmetrizing that family does not remove the target from its closure. This projection argument is the source of many universality results for averaging and canonicalization.

Universality says little about the cost of reaching a useful approximation. Group averaging can require many evaluations. Finite frame averaging multiplies backbone cost by the frame size. Deterministic canonicalization is cheap but can hand the backbone a discontinuous canonical target. Local frames are efficient and preserve fine geometric information, but they become singular when the vectors defining a frame are degenerate. Learned and probabilistic frames add parameters and sampling variance while improving adaptability.

The practical choice is therefore governed by the geometry of the ambiguous inputs. If degeneracies are excluded by construction, one stable local frame may be enough. If a small finite ambiguity such as eigenvector signs dominates, frame averaging is attractive. If symmetric configurations are common or near-symmetries matter physically, a weighted or probabilistic frame can trade several evaluations for smoother behavior. A built-in equivariant architecture remains preferable when exact symmetry at every evaluation and predictable runtime matter more than reusing an unrestricted backbone.

Frames do not eliminate the work of equivariant design. They concentrate it in the pose mechanism. The quotient view tells us what should be forgotten, local frames tell us how to express geometry without forgetting it, and symmetrization tells us how to make an arbitrary function respect the group. The discontinuities reveal the price of asking one coordinate system to represent a space in which some objects have no unique orientation.

---

## References

<ol class="bibliography">
  <li id="ref-du2022">Du, W., Zhang, H., Du, Y., Meng, Q., Chen, W., Zheng, N., Shao, B., &amp; Liu, T.-Y. (2022). <a href="https://proceedings.mlr.press/v162/du22e.html">SE(3) equivariant graph neural networks with complete local frames</a>. <em>Proceedings of the 39th International Conference on Machine Learning</em>, 5583–5608. <a href="#cite-du2022">↩</a></li>
  <li id="ref-puny2022">Puny, O., Atzmon, M., Smith, E. J., Misra, I., Grover, A., Ben-Hamu, H., &amp; Lipman, Y. (2022). <a href="https://arxiv.org/abs/2110.03336">Frame averaging for invariant and equivariant network design</a>. <em>International Conference on Learning Representations</em>. <a href="#cite-puny2022">↩</a></li>
  <li id="ref-kaba2023">Kaba, S.-O., Mondal, A. K., Zhang, Y., Bengio, Y., &amp; Ravanbakhsh, S. (2023). <a href="https://proceedings.mlr.press/v202/kaba23a.html">Equivariance with learned canonicalization functions</a>. <em>Proceedings of the 40th International Conference on Machine Learning</em>, 15546–15566. <a href="#cite-kaba2023">↩</a></li>
  <li id="ref-kim2023">Kim, J., Nguyen, T. D., Suleymanzade, A., An, H., &amp; Hong, S. (2023). <a href="https://arxiv.org/abs/2306.02866">Learning probabilistic symmetrization for architecture agnostic equivariance</a>. <em>Advances in Neural Information Processing Systems</em>, 36. <a href="#cite-kim2023">↩</a></li>
  <li id="ref-dym2024">Dym, N., Lawrence, H., &amp; Siegel, J. W. (2024). <a href="https://proceedings.mlr.press/v235/dym24a.html">Equivariant frames and the impossibility of continuous canonicalization</a>. <em>Proceedings of the 41st International Conference on Machine Learning</em>, 12228–12267. <a href="#cite-dym2024">↩</a></li>
</ol>
