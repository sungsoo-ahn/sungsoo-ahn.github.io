---
layout: post
title: "Depth, Over-Smoothing, and Over-Squashing in Graph Networks"
date: 2026-08-08
last_updated: 2026-08-08
description: "Why deeper graph networks face under-reaching, over-smoothing, and over-squashing—and how topology determines which remedy helps."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [graph-learning]
lecture_paths: [ml4mol, gdl]
tags: [graph-neural-networks, message-passing, over-smoothing, over-squashing, graph-rewiring]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the deep-GNN storyline from my 2025 Machine
  Learning for Molecules and Geometric Deep Learning lectures. It continues
  the message-passing abstraction developed in <a href="{% post_url 2026-08-08-graph-neural-networks-message-passing %}">Graph Neural Networks as Learnable Message Passing</a>.</em>
</p>

A message-passing graph neural network communicates one edge per layer. Two nodes separated by ten edges cannot interact through a five-layer network. The obvious response is to add depth.

Depth solves this reach problem and exposes two harder failures. Repeated propagation can make node representations indistinguishable, which is called **over-smoothing**. Long-range signals can also be compressed through fixed-width states and narrow graph cuts, which is called **over-squashing**. These failures are not synonyms. One removes contrast by mixing too much; the other loses sensitivity because too much information must cross too little capacity.

The distinction changes how we design a remedy. Residual connections can preserve local features but leave a topological bottleneck untouched. Rewiring can shorten long paths but accelerate unwanted smoothing. A wider network can carry more information but increase parameters and sensitivity. Deeper graph learning is therefore not a contest to stack the most layers. It is a problem of matching communication range, feature dynamics, and graph topology to the dependency structure of the task.

## Depth sets the communication radius

Let $$\mathbf{h}_v^{(\ell)} \in \mathbb{R}^d$$ denote the state of node $$v$$ after layer $$\ell$$. A message-passing layer has the form

$$
\mathbf{m}_v^{(\ell)}
=
\operatorname{AGG}
\left\{
M_\ell\!\left(
\mathbf{h}_v^{(\ell)},
\mathbf{h}_u^{(\ell)},
\mathbf{e}_{uv}
\right)
: u \in \mathcal{N}(v)
\right\},
$$

$$
\mathbf{h}_v^{(\ell+1)}
=
U_\ell\!\left(
\mathbf{h}_v^{(\ell)},
\mathbf{m}_v^{(\ell)}
\right).
$$

Here, $$M_\ell$$ constructs a message across edge $$(u,v)$$, $$\operatorname{AGG}$$ combines the unordered incoming messages, and $$U_\ell$$ updates the receiving node. One layer can only use one-hop neighbors. By induction, $$\mathbf{h}_v^{(L)}$$ can depend only on nodes at graph distance at most $$L$$ from $$v$$.

This $$L$$-hop neighborhood is the model's **receptive field** at node $$v$$. The term comes from vision, where stacking local convolutions expands the region of an image that can influence one output pixel. On a graph, the receptive field depends on topology. It grows linearly on a path, quadratically on a two-dimensional grid, and exponentially with depth on a regular tree until it reaches the graph boundary.

Insufficient depth causes **under-reaching**: the target depends on a node outside the receptive field. Consider a molecular bond graph. A two-layer model can couple an atom to atoms at most two bonds away. It cannot directly represent a property whose relevant dependency spans a longer conjugated path. The same issue appears in algorithmic tasks where a node must know whether a distant marked node exists.

Adding layers expands the receptive field, but reachability only says that a signal *can* arrive. It does not say that the signal remains identifiable or influential when it does. The three resulting regimes are shown below.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnfail_three_failures.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Depth creates three distinct communication regimes. Too few layers cause under-reaching; repeated mixing can erase feature contrast through over-smoothing; and an expanding receptive field can compress many signals through a narrow cut, causing over-squashing. Original diagram." %}

Under-reaching has a direct architectural fix: increase communication range. Over-smoothing and over-squashing require us to ask what happens *inside* that larger receptive field.

## Over-smoothing is a diffusion toward low-frequency features

Many graph layers combine a node with an average of its neighbors. Repeating this operation resembles diffusion: local differences shrink, and the feature field becomes smoother across edges. This is useful when adjacent nodes should have similar representations. It becomes over-smoothing when the diffusion erases distinctions required by the target.

A linear graph convolution makes the mechanism explicit. Add self-loops to the adjacency matrix $$\mathbf{A}$$ and call the result $$\widetilde{\mathbf{A}} = \mathbf{A}+\mathbf{I}$$. Let $$\widetilde{\mathbf{D}}$$ be its degree matrix. The normalized propagation matrix is

$$
\mathbf{S}
=
\widetilde{\mathbf{D}}^{-1/2}
\widetilde{\mathbf{A}}
\widetilde{\mathbf{D}}^{-1/2}.
$$

If we temporarily remove learned weights and nonlinearities, an $$L$$-layer network reduces to

$$
\mathbf{H}^{(L)} = \mathbf{S}^L\mathbf{H}^{(0)}.
$$

Because $$\mathbf{S}$$ is symmetric, it has an orthonormal eigenbasis:

$$
\mathbf{S}
=
\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\mathsf{T}}.
$$

The propagated features become

$$
\mathbf{H}^{(L)}
=
\mathbf{U}\boldsymbol{\Lambda}^{L}
\mathbf{U}^{\mathsf{T}}\mathbf{H}^{(0)}.
$$

Each eigenvector is a graph-frequency pattern, and its coefficient is multiplied by $$\lambda_k^L$$. Components with $$\lvert\lambda_k\rvert < 1$$ decay geometrically with depth. The dominant low-frequency components survive, so neighboring node features become increasingly aligned. Under conditions on the graph spectrum and layer weights, Oono and Suzuki (2020) formalize an exponential approach toward a low-dimensional subspace that mainly retains connected-component and degree information (<span id="cite-oono2020"></span>[Oono & Suzuki, 2020](#ref-oono2020)).

The same behavior can be measured without diagonalizing $$\mathbf{S}$$. Let $$\mathbf{L}=\mathbf{D}-\mathbf{A}$$ be the combinatorial graph Laplacian. The **Dirichlet energy**

$$
\mathcal{E}(\mathbf{H})
=
\operatorname{tr}
\left(
\mathbf{H}^{\mathsf{T}}\mathbf{L}\mathbf{H}
\right)
=
\frac{1}{2}
\sum_{u\in V}
\sum_{v\in\mathcal{N}(u)}
\left\lVert
\mathbf{h}_u-\mathbf{h}_v
\right\rVert_2^2
$$

measures variation across edges. Low energy means adjacent features are similar. In the toy linear diffusion above, repeated propagation suppresses feature variation and drives the energy down.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnfail_smoothing_diffusion.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Repeated linear graph averaging damps differences between neighboring node features. In the toy diffusion, the Dirichlet energy \(\mathcal{E}(\mathbf{H})=\operatorname{tr}(\mathbf{H}^{\mathsf{T}}\mathbf{L}\mathbf{H})\) decays as the feature field approaches a low-frequency limit. Original diagram." %}

Real GNNs include learned weights, nonlinearities, normalization, and attention, so their Dirichlet energy need not decrease monotonically. Over-smoothing is still the right diagnosis when deeper representations lose node-level separation because propagation repeatedly favors low-frequency structure. It is not the right diagnosis for every failure of a deep model. Optimization can fail while node features remain distinct, and a task can fail because distant signals never arrive even when local features remain sharp.

Over-smoothing is also task-dependent. In a homophilic citation graph, papers connected by citations often share a topic, so smoothing can remove noise. In a heterophilic graph, an edge may connect unlike classes, and the same diffusion mixes evidence that should stay separate. A low Dirichlet energy is therefore a geometric statistic, not a certificate of a useful representation.

## Residual paths preserve multiple neighborhood scales

The most reliable anti-smoothing methods give a node access to less-propagated features. A residual layer keeps the previous state:

$$
\mathbf{H}^{(\ell+1)}
=
\mathbf{H}^{(\ell)}
+ F_\ell\!\left(\mathbf{H}^{(\ell)},G\right).
$$

The identity path helps optimization and lets information bypass one smoothing step. A residual connection does not remove diffusion from $$F_\ell$$, however. Repeated residual layers can still drift toward smooth representations.

Jumping Knowledge networks expose every intermediate scale to the readout. Instead of forcing the final prediction to use only $$\mathbf{h}_v^{(L)}$$, they combine

$$
\mathbf{h}_v^{(1)},\,
\mathbf{h}_v^{(2)},\,
\ldots,\,
\mathbf{h}_v^{(L)}.
$$

The readout can then select a shallow representation for a node whose useful evidence is local and a deeper representation for a node that needs a broader neighborhood. Xu et al. (2018) motivate this design through the connection between neighborhood aggregation and random-walk influence distributions (<span id="cite-xu2018"></span>[Xu et al., 2018](#ref-xu2018)).

GCNII makes the bypass more explicit by injecting the initial features at every layer and keeping each learned transformation close to the identity. In simplified notation,

$$
\mathbf{H}^{(\ell+1)}
=
\sigma\!\left[
\left(
(1-\alpha)\mathbf{S}\mathbf{H}^{(\ell)}
+\alpha\mathbf{H}^{(0)}
\right)
\left(
(1-\beta_\ell)\mathbf{I}
+\beta_\ell\mathbf{W}^{(\ell)}
\right)
\right].
$$

The coefficient $$\alpha$$ preserves input information, while $$\beta_\ell$$ controls how far the feature transformation departs from the identity. Chen et al. (2020) use this construction to train substantially deeper graph convolutions without the degradation of a vanilla GCN (<span id="cite-chen2020"></span>[Chen et al., 2020](#ref-chen2020)).

Normalization attacks the symptom more directly. PairNorm rescales representations so that their total pairwise distance does not collapse (<span id="cite-zhao2020"></span>[Zhao & Akoglu, 2020](#ref-zhao2020)). Energy regularization similarly penalizes unwanted loss of variation. These methods can keep embeddings separated, but separation alone does not make them informative. A model can maintain large pairwise distances in directions unrelated to the label. Optimizing an over-smoothing metric and improving the downstream task are different objectives.

## Over-squashing is a sensitivity bottleneck

Over-squashing appears when a large receptive field feeds a fixed-size state through too few communication channels. Node features need not become similar. The problem is that the receiving node becomes insensitive to individual distant inputs.

A binary tree gives an exact example. Suppose each internal node stores the mean of its two children:

$$
\mathbf{h}_{\text{parent}}
=
\frac{1}{2}
\left(
\mathbf{h}_{\text{left}}
+\mathbf{h}_{\text{right}}
\right).
$$

At depth $$r$$, the root receives the average of $$2^r$$ leaves:

$$
\mathbf{h}_{\text{root}}
=
2^{-r}
\sum_{u\in\text{leaves}}
\mathbf{h}_u.
$$

The sensitivity to one leaf is

$$
\frac{\partial\mathbf{h}_{\text{root}}}
{\partial\mathbf{h}_u}
=
2^{-r}\mathbf{I}.
$$

Every extra level doubles the number of leaves and halves the direct influence of each leaf. The root can know an aggregate, but it cannot preserve the identity and content of exponentially many leaf signals in a fixed-width vector.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnfail_tree_squashing.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A depth-\(r\) binary tree has \(2^r\) leaves inside the root's receptive field. Under recursive mean aggregation, the root sensitivity to any single leaf is \(2^{-r}\), so reach grows while individual influence vanishes. Original diagram." %}

Alon and Yahav (2021) used this bottleneck view to explain why message-passing networks struggle on tasks that require matching information across distant tree leaves (<span id="cite-alon2021"></span>[Alon & Yahav, 2021](#ref-alon2021)). The argument extends beyond mean aggregation. A distant input affects a node through products of layer Jacobians along graph walks. When the number of relevant inputs grows faster than path capacity, or when normalized aggregation repeatedly attenuates each path, the Jacobian from a distant source to the target becomes small.

Increasing depth does not repair this attenuation merely by creating more walks. Di Giovanni et al. (2023) analyze width, depth, and topology in a common sensitivity framework. Their results show that width can mitigate over-squashing at the cost of greater overall sensitivity, while extra depth eventually faces vanishing-gradient effects. The largest obstruction comes from graph topology, especially node pairs with high commute time (<span id="cite-digiovanni2023"></span>[Di Giovanni et al., 2023](#ref-digiovanni2023)).

Commute time is the expected number of random-walk steps needed to travel from node $$u$$ to node $$v$$ and return. It is proportional to **effective resistance**, the electrical resistance obtained by treating every graph edge as a unit resistor:

$$
\operatorname{Res}(u,v)
=
\frac{\tau(u,v)}{2\lvert E\rvert}.
$$

Here, $$\tau(u,v)$$ is commute time and $$\lvert E\rvert$$ is the number of edges. Two nodes can have modest shortest-path distance but high effective resistance if all routes between them share a narrow cut. Effective resistance therefore detects global bottlenecks that hop count alone misses.

## Curvature localizes topological bottlenecks

A dumbbell graph—two dense communities connected by one bridge—shows why topology dominates. Many signals from the left community must cross one edge to influence the right community. The bridge is not long, but it has low path redundancy. Deleting it disconnects the graph.

Discrete curvature gives a local language for this geometry. In a positively curved region, nearby nodes have overlapping neighborhoods and many short alternative routes. Across a negatively curved edge, the neighborhoods expand away from each other with few shared triangles or four-cycles. Topping et al. (2022) introduce an edge-based combinatorial curvature and connect highly negative curvature to sensitivity bottlenecks in message passing (<span id="cite-topping2022"></span>[Topping et al., 2022](#ref-topping2022)).

Curvature is a diagnostic, not a synonym for over-squashing. Over-squashing is defined by task-relevant sensitivity. Curvature describes topology without knowing the target. A negatively curved bridge is harmful when distant information must cross it; the same bridge is a useful locality prior when the task should ignore the other community.

## Rewiring trades locality for communication

If topology creates the bottleneck, one response is to separate the **input graph** from the **computational graph**. The input graph records observed relations. The computational graph records which nodes exchange messages. Adding edges across a narrow cut creates alternative routes and lowers effective resistance.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnfail_rewiring_tradeoffs.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Adding shortcuts across a graph bottleneck creates more routes for long-range information and lowers effective resistance. The same rewiring adds messages, weakens the original locality prior, and can accelerate over-smoothing. Original diagram." %}

Rewiring is not free. A denser graph increases time and memory per layer. It makes unrelated nodes interact earlier and may erase the meaning of graph distance. It can also worsen over-smoothing because diffusion now has more routes. Full self-attention takes the limiting approach of connecting every node pair, reducing graph distance to one but typically paying quadratic cost and requiring positional or structural encodings to recover the original graph.

Dynamic rewiring tries to retain distance as an inductive bias. DRew adds connections as layers progress, so nodes at distance $$r$$ begin communicating only when the network reaches the corresponding scale. Its delay mechanism can also use a less-smoothed state from the time when the source information first became available. Gutteridge et al. (2023) report that this progressive construction improves several long-range benchmarks without making every distant pair interact from the first layer (<span id="cite-gutteridge2023"></span>[Gutteridge et al., 2023](#ref-gutteridge2023)).

Hierarchical pooling offers another route. A coarsened graph can turn a long path into a short path between supernodes, much as image pyramids communicate over large spatial scales. The tradeoff moves into the pooling rule: an early merge can discard which fine-scale node contributed the signal. Global graph states provide a cheap communication hub, but compress the entire graph through one shared vector—the same bottleneck in a different location.

Molecular graphs make the choice concrete. A bond graph is a good computational prior for short-range covalent interactions. It is a poor communication graph for two atoms that are distant in bond count but close in three-dimensional space. Radius graphs, long-range electrostatic modules, or global attention can add the missing physical routes. Connecting every atom to every other atom may be unnecessary when the target is predominantly local. The right computational graph follows the interactions that matter, not graph density as an end in itself.

## Diagnose the failure before choosing the remedy

Depth alone does not identify why a graph network fails. The useful diagnostics correspond to the three mechanisms.

For under-reaching, compare model depth with the graph distances over which the target depends. For over-smoothing, track feature contrast across layers using Dirichlet energy, pairwise distances, or class separation. For over-squashing, inspect Jacobian sensitivity from distant sources, effective resistance, graph cuts, or curvature-based bottleneck scores.

The intervention should then match the evidence. Skip connections, initial-feature injection, and multi-scale readouts preserve less-smoothed information. Width increases channel capacity but raises computational cost and can amplify sensitivity. Rewiring, pooling, global tokens, and attention create shorter communication paths but modify the locality prior. Regularization can stabilize a diagnostic without solving the task.

The central tension is unavoidable. Local message passing is efficient because it ignores most node pairs. Long-range tasks ask the model to recover some of those ignored interactions. A successful deep graph network does not remove locality indiscriminately. It decides which information should travel far, how quickly it should travel, and how much detail must survive the trip.

## References

- <span id="ref-xu2018"></span>Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K., & Jegelka, S. (2018). Representation Learning on Graphs with Jumping Knowledge Networks. [Proceedings of ICML](https://proceedings.mlr.press/v80/xu18c.html). <a href="#cite-xu2018" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-oono2020"></span>Oono, K., & Suzuki, T. (2020). Graph Neural Networks Exponentially Lose Expressive Power for Node Classification. [ICLR](https://openreview.net/forum?id=S1ldO2EFPr). <a href="#cite-oono2020" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-chen2020"></span>Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. (2020). Simple and Deep Graph Convolutional Networks. [Proceedings of ICML](https://proceedings.mlr.press/v119/chen20v.html). <a href="#cite-chen2020" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-zhao2020"></span>Zhao, L., & Akoglu, L. (2020). PairNorm: Tackling Oversmoothing in GNNs. [ICLR](https://openreview.net/forum?id=rkecl1rtwB). <a href="#cite-zhao2020" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-alon2021"></span>Alon, U., & Yahav, E. (2021). On the Bottleneck of Graph Neural Networks and its Practical Implications. [ICLR](https://openreview.net/forum?id=i80OPhOCVH2). <a href="#cite-alon2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-topping2022"></span>Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2022). Understanding Over-Squashing and Bottlenecks on Graphs via Curvature. [ICLR](https://openreview.net/forum?id=7UmjRGzp-A). <a href="#cite-topping2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-digiovanni2023"></span>Di Giovanni, F., Giusti, L., Barbero, F., Luise, G., Liò, P., & Bronstein, M. M. (2023). On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology. [Proceedings of ICML](https://proceedings.mlr.press/v202/di-giovanni23a.html). <a href="#cite-digiovanni2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-gutteridge2023"></span>Gutteridge, B., Dong, X., Bronstein, M. M., & Di Giovanni, F. (2023). DRew: Dynamically Rewired Message Passing with Delay. [Proceedings of ICML](https://proceedings.mlr.press/v202/gutteridge23a.html). <a href="#cite-gutteridge2023" class="reversefootnote" role="doc-backlink">↩</a>
