---
layout: post
title: "What Can a Graph Neural Network Distinguish?"
date: 2026-08-08
last_updated: 2026-08-08
description: "Graph neural network expressivity through graph isomorphism, multiset aggregation, the Weisfeiler--Leman test, its blind spots, and the tradeoffs behind stronger models."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [graph-learning]
lecture_paths: [ml4mol, gdl]
tags: [graph-neural-networks, expressivity, weisfeiler-leman, graph-isomorphism, subgraph-gnns]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the expressivity storyline from my Machine
  Learning for Molecules and Geometric Deep Learning lectures. The central
  question is not which graph architecture has the longest list of features.
  It is which graph distinctions a representation preserves, which ones it
  erases, and whether the distinctions it preserves are useful for the task.</em>
</p>

A graph neural network can fail even when optimization succeeds. Two different graphs may receive exactly the same representation for every possible choice of the network parameters. No additional data, wider hidden state, or longer training can recover information that the architecture has already erased.

The cleanest way to study this failure is through graph isomorphism. A graph representation should ignore arbitrary node names but should, ideally, separate graphs with different connectivity. These requirements pull in opposite directions. Ignoring order creates equivalence classes; making those classes too large collapses genuinely different graphs.

The Weisfeiler--Leman (WL) test makes that tension concrete. Its simplest form repeatedly colors each node using its current color and the multiset of neighboring colors. Standard message-passing neural networks perform the same kind of local computation, so the test gives them a sharp expressivity ceiling. The ceiling also tells us how to go beyond message passing: compare tuples of nodes, expose substructures, process multiple subgraphs, or supply structural positions. Each remedy adds information, computation, and inductive bias. None guarantees a better model.

## A representation should identify graphs, not node names

Let $$G=(V,E,\mathbf{X})$$ be a graph with node set $$V$$, edge set $$E$$, and node-feature matrix $$\mathbf{X}$$. A permutation matrix $$\mathbf{P}$$ relabels the nodes. In matrix form, the relabeled graph has features and adjacency matrix

$$
\mathbf{X}'=\mathbf{P}\mathbf{X},
\qquad
\mathbf{A}'=\mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf T}.
$$

A graph-level representation $$f$$ must be invariant to this relabeling:

$$
f(\mathbf{P}\mathbf{X},\mathbf{P}\mathbf{A}\mathbf{P}^{\mathsf T})
=f(\mathbf{X},\mathbf{A}).
$$

This equation says that the graph is the combinatorial object, not the arrays used to store it. Two graphs $$G$$ and $$H$$ are **isomorphic** if some bijection $$\varphi:V(G)\to V(H)$$ preserves node features and adjacency. Isomorphic graphs must receive the same graph-level representation.

Expressivity concerns the converse. If two graphs are not isomorphic, does the representation assign them different values? The desired logic is shown below: relabelings should collapse, while structurally different graphs should separate.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnexpr_representation_equivalence.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A graph representation must identify relabelings of the same graph, because node indices carry no meaning. Expressivity asks whether that invariance also collapses structurally different graphs. Original figure." %}

No practical predictor needs to solve graph isomorphism as an end in itself. The test is useful because it exposes architectural collisions without referring to a particular dataset. If an architecture maps $$G$$ and $$H$$ to the same vector for all parameters, it also cannot assign them different target values. Any target function that separates the pair lies outside the architecture's function class.

The reverse conclusion does not hold. An architecture that *can* distinguish two graphs may still learn not to distinguish them. Expressivity describes what the model class can represent, not what stochastic gradient descent will find or what finite data can support.

## Message passing compresses a multiset

A message-passing neural network (MPNN) updates node $$v$$ from its current representation and the representations of its neighbors. One layer has the form

$$
\begin{aligned}
\mathbf{m}_{v}^{(t)}
&=\operatorname{AGG}^{(t)}\!\left(
\left\{\!\left\{
M^{(t)}\!\left(\mathbf{h}_{v}^{(t-1)},
\mathbf{h}_{u}^{(t-1)},\mathbf{e}_{uv}\right)
:u\in\mathcal{N}(v)
\right\}\!\right\}\right),\\
\mathbf{h}_{v}^{(t)}
&=U^{(t)}\!\left(\mathbf{h}_{v}^{(t-1)},\mathbf{m}_{v}^{(t)}\right).
\end{aligned}
$$

Here $$\mathcal{N}(v)$$ is the neighborhood of $$v$$, $$\mathbf{e}_{uv}$$ is an optional edge feature, and the double braces denote a multiset. A multiset records both the elements and their multiplicities. The aggregation function must ignore neighbor order, because an ordering of $$\mathcal{N}(v)$$ is as arbitrary as the global node indices.

The aggregation step is therefore a compression problem. A non-injective aggregator maps different multisets to the same output. Mean aggregation cannot distinguish $$\{\!\{\mathbf{x},\mathbf{y}\}\!\}$$ from $$\{\!\{\mathbf{x},\mathbf{x},\mathbf{y},\mathbf{y}\}\!\}$$ because the multiplicities cancel. Max aggregation records whether a large coordinate appears but not how often it appears. These losses can make nodes with different neighborhoods indistinguishable before the update function sees them.

An injective multiset function avoids that particular loss. For multisets of bounded size drawn from a countable domain, a learned transformation followed by summation can represent an injective encoding. This observation motivates the Graph Isomorphism Network (GIN) update of <span id="cite-xu2019"></span>[Xu et al., 2019](#ref-xu2019):

$$
\mathbf{h}_{v}^{(t)}
=\operatorname{MLP}^{(t)}\!\left(
(1+\epsilon^{(t)})\mathbf{h}_{v}^{(t-1)}
+\sum_{u\in\mathcal{N}(v)}\mathbf{h}_{u}^{(t-1)}
\right).
$$

The sum retains neighbor multiplicity, while the multilayer perceptron can assign a new code to each distinct combination. The coefficient $$1+\epsilon^{(t)}$$ keeps the center node distinguishable from its neighbors. Under the assumptions used in the expressivity theorem, this construction reaches the ceiling of ordinary message passing. It does not remove that ceiling.

## Color refinement is the discrete analogue

The one-dimensional Weisfeiler--Leman test, usually called **1-WL** or **color refinement**, turns the multiset computation into a deterministic graph procedure. The method starts with a color $$c_v^{(0)}$$ for every node. The initial color may encode an atom type, a categorical node label, or one common symbol when the graph is unlabeled. At iteration $$t$$, it computes

$$
c_v^{(t)}
=\operatorname{HASH}\!\left(
c_v^{(t-1)},
\left\{\!\left\{c_u^{(t-1)}:u\in\mathcal{N}(v)\right\}\!\right\}
\right).
$$

The hash is injective: two nodes receive the same new color exactly when their previous colors match and their neighbor-color multisets match. We compare two graphs by running this refinement jointly, so equal signatures receive equal color names across both graphs. If their color histograms differ at any iteration, the graphs cannot be isomorphic. If the histograms remain equal after the colors stabilize, the test returns no conclusion. The graphs may still be non-isomorphic.

The five-node path gives a complete worked example. Suppose every node starts with color $$a$$. After one round, the two endpoints receive one color because they each see the multiset $$\{\!\{a\}\!\}$$. The three internal nodes receive another because they see $$\{\!\{a,a\}\!\}$$. After the second round, the center separates from the two nodes next to the endpoints. No further refinement is possible.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnexpr_wl_refinement.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Color refinement on a five-node path. The first round separates nodes by degree; the second round separates the center because its two neighbors have a different color from the neighbors seen by the other internal nodes. Original figure." %}

The colors have no semantics by themselves. Color $$d$$ is not numerically larger than color $$e$$. Each color is a lossless name for a rooted neighborhood pattern revealed so far. After $$t$$ rounds, a node color summarizes the unfolding of its neighborhood out to $$t$$ hops, while merging patterns that 1-WL cannot tell apart.

This procedure originated in the graph-isomorphism work of <span id="cite-wl1968"></span>[Weisfeiler and Leman, 1968](#ref-wl1968). It is only a heuristic for isomorphism: a mismatch certifies non-isomorphism, but a match does not certify isomorphism. For neural networks, that incompleteness is the useful part. It identifies pairs that local symmetric aggregation cannot separate.

## Standard MPNNs inherit the 1-WL ceiling

The connection between MPNNs and 1-WL follows by induction. Suppose two nodes have the same 1-WL color at iteration $$t-1$$. Their previous colors match, and their multisets of neighboring colors match. If node states are functions of those colors, then the two nodes also have matching states and matching multisets of neighboring states. Applying the same message, aggregation, and update functions gives the same state at iteration $$t$$.

The base case holds when the initial node state is a function of the initial label. Therefore,

> **The 1-WL ceiling.** If 1-WL assigns two nodes the same color at every round, a standard MPNN with shared, permutation-invariant local aggregation assigns them the same hidden state at the corresponding layers. If 1-WL cannot distinguish two graphs, neither can such an MPNN followed by a permutation-invariant readout.
{: .block-lemma }

<span id="cite-morris2019"></span>[Morris et al., 2019](#ref-morris2019) and Xu et al. established this correspondence from complementary directions. The upper bound applies to every choice of weights, not merely to a poorly trained network. With injective aggregation and readout, an MPNN can match 1-WL on the graph family covered by the assumptions. GIN was designed to realize this matching case.

The result explains why changing the MLP depth or hidden dimension cannot fix every collision. Those choices improve the functions computed *within* the message-passing template. They do not change the equivalence relation imposed by repeated multiset aggregation.

The assumptions matter. Continuous node attributes, informative edge attributes, geometric coordinates, or externally supplied identifiers can separate a pair that is indistinguishable as an unlabeled graph. The theorem then applies to the enriched input. The information did not emerge from message passing; it entered through the features.

## A six-cycle looks like two triangles

Consider a cycle of six nodes and a disjoint union of two triangles. The graphs are not isomorphic: one is connected and triangle-free, while the other has two connected components and two triangles. Yet every node has degree two.

Start 1-WL with the same color on every node. At the first iteration, every node sees two neighbors of that color, so every node receives the same new color. The same argument repeats forever. Both graphs always have a histogram containing six copies of one color.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnexpr_regular_collision.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A six-cycle and two disjoint triangles are non-isomorphic, but 1-WL cannot distinguish them from uniform initial features. Every node always sees two neighbors with the same color, so standard message passing also produces identical graph representations. Original figure." %}

The collision is stronger than a finite-depth receptive-field problem. Adding layers does not help, because the coloring has already reached a fixed point. A standard MPNN with uniform initial features gives every node the same state at every layer and produces the same multiset of six node states for both graphs.

This pair exposes several missing graph properties at once. The representation does not detect connectedness, distinguish a six-cycle from a three-cycle, or count triangles. It does not follow that MPNNs can never learn any signal correlated with those properties. Real graphs may contain node or edge features that break the symmetry. The claim is exact for the unlabeled pair, and that is enough to prove a limitation of the architecture.

Regular graphs produce many such examples. When all nodes begin identically and every node has the same degree, the first refinement cannot create different colors. This is why a model can encode increasingly complicated local functions yet remain blind to a simple global difference.

## Stronger models change what gets compared

Going beyond 1-WL requires more than a stronger function inside the same node-wise update. A successful remedy changes the object being represented or adds structural information that the original update could not derive. The main strategies can be understood through that single principle.

{% include figure.liquid loading="eager" path="assets/img/blog/gnnexpr_remedy_tradeoff.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Four routes beyond ordinary message passing enlarge the unit of comparison or enrich its coordinates. The added distinctions come with tuple states, preprocessing, repeated graph evaluations, or the need to handle positional ambiguities. Original figure." %}

### Compare tuples instead of individual nodes

Higher-order WL tests color ordered tuples of nodes. A $$k$$-tuple records relations among $$k$$ positions, including which entries coincide and which pairs are adjacent. Refinement then replaces one coordinate at a time and aggregates the resulting tuple colors. Two nodes considered separately may look identical, while a pair reveals whether they lie on the same cycle or how they relate to the rest of the graph.

Higher-order GNNs imitate this computation by storing features on tuples rather than only on nodes. The construction of Morris et al. gives a hierarchy with provable increases in distinguishing power. Its cost is equally structural. A graph with $$n$$ nodes has $$n^k$$ ordered $$k$$-tuples. Even sparse variants must decide which tuples and tuple neighborhoods are worth retaining.

The higher-order view is appropriate when the target depends on relations that cannot be reduced to independent node neighborhoods. It is wasteful when a node-level statistic already determines the label. Increasing $$k$$ changes both the expressivity class and the computational object.

### Expose selected substructures

Another route tells the network which motifs occur. A **substructure encoding** can attach to each node the number of triangles, cycles, paths, or domain-specific patterns that contain it. The six-cycle and the two triangles separate immediately once triangle counts are available.

Graph Substructure Networks of <span id="cite-bouritsas2021"></span>[Bouritsas et al., 2021](#ref-bouritsas2021) formalize this idea by augmenting message passing with subgraph-isomorphism counts. The architecture remains permutation equivariant because relabeling the graph relabels the node-wise counts. Its advantage comes from preprocessing information that 1-WL cannot compute.

The motif bank is an inductive bias, not free expressivity. Triangle counts help when rings or local cycles matter, but another task may depend on long-range separation or a different pattern. Counting general subgraphs is also computationally hard, so practical systems use a small bank and bounded motif size. The model distinguishes more graphs along axes chosen by the designer.

### Represent the graph through a bag of subgraphs

Subgraph GNNs run a base network on several related views of the input, then aggregate across those views. A node-based policy might delete one node at a time, mark one node at a time, or extract one ego-network per node. Two graphs that collide under whole-graph message passing can yield different collections of subgraphs.

Equivariant Subgraph Aggregation Networks of <span id="cite-bevilacqua2022"></span>[Bevilacqua et al., 2022](#ref-bevilacqua2022) make the symmetry explicit: the subgraphs form an unordered collection, and the nodes within them also permute. Applying shared functions and invariant aggregations across these axes preserves the graph's relabeling symmetry while exceeding 1-WL for suitable selection policies.

This route trades one large blind spot for repeated computation. Deleting every node creates $$n$$ graph views; richer policies create more. Sampling can control the cost, but the resulting representation may lose the clean guarantee of exhaustive aggregation.

### Supply structural positions

Position can break symmetries that local neighborhoods preserve. A graph Transformer, for example, has no reason to know whether two nodes are adjacent unless adjacency, distance, or spectral information enters the input or attention score. Common encodings use shortest-path distance, random-walk statistics, or eigenvectors of the graph Laplacian. GraphGPS combines such structural encodings with local message passing and global attention (<span id="cite-rampasek2022"></span>[Rampášek et al., 2022](#ref-rampasek2022)).

Laplacian eigenvectors act like graph-dependent coordinates, but they are not canonical node IDs. An eigenvector can change sign without changing the eigenproblem. An eigenspace with repeated eigenvalue can rotate to another basis. A network that consumes raw eigenvectors without respecting these ambiguities can assign different outputs to two equivalent numerical decompositions. SignNet and BasisNet address these symmetries explicitly (<span id="cite-lim2023"></span>[Lim et al., 2023](#ref-lim2023)).

Random node identifiers can make a network highly expressive as well, but they change the symmetry contract. Fixed IDs make the answer depend on an arbitrary labeling. Averaging over randomized IDs can restore invariance in expectation, at the price of variance and repeated evaluation. Structural positions are most useful when they expose relevant geometry while preserving, or carefully recovering, the invariance the graph problem requires.

## Expressivity is not model quality

An expressivity result is a statement about possible distinctions. Model quality depends on which distinctions support the target, how easily the model learns them, and how well they transfer beyond the training sample.

The difference can be stated with a target function $$y(G)$$. If an architecture always maps $$G$$ and $$H$$ to the same representation but $$y(G)\neq y(H)$$, the architecture has irreducible approximation error on that pair. Greater expressivity can remove this obstruction. If instead $$y(G)=y(H)$$, separating the pair provides no necessary signal. The extra degrees of freedom may fit incidental structure that does not generalize.

Computational cost changes the comparison further. Node-wise message passing stores $$O(n)$$ node states and usually communicates along $$O(|E|)$$ edges per layer. Full pair or tuple representations can require quadratic or higher memory. Motif counts move work into preprocessing. Subgraph methods multiply the number of forward passes. Positional encodings add eigendecompositions, distance computations, or Monte Carlo features. Sparse approximations reduce these costs by choosing which distinctions to retain.

The right design question is therefore narrower than “Which model is most expressive?” Ask which graph pairs the task must separate. Then choose the cheapest representation that exposes those differences while preserving the required symmetry. For molecular property prediction, ring counts, stereochemistry, and three-dimensional geometry may matter more than arbitrary graph-isomorphism corner cases. For a social network, global position or community structure may dominate small motifs. The target decides which blind spots are harmful.

The WL hierarchy remains useful precisely because it makes blind spots explicit. It turns an informal claim that a GNN “captures structure” into a falsifiable question: which structures survive the representation? Once that question is answered, accuracy, efficiency, optimization, and generalization still have to be earned.

---

## References

- <span id="ref-wl1968"></span>Weisfeiler, B., & Leman, A. (1968). The reduction of a graph to canonical form and the algebra which appears therein. *Nauchno-Technicheskaya Informatsia*, 2(9), 12--16. [English translation](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf). <a href="#cite-wl1968" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-xu2019"></span>Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful Are Graph Neural Networks? *ICLR 2019*. [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km). <a href="#cite-xu2019" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-morris2019"></span>Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. *AAAI 2019*. [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/4384). <a href="#cite-morris2019" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bouritsas2021"></span>Bouritsas, G., Frasca, F., Zafeiriou, S., & Bronstein, M. M. (2021). Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting. *ICLR 2021*. [OpenReview](https://openreview.net/forum?id=LT0gkQt1h7). <a href="#cite-bouritsas2021" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-bevilacqua2022"></span>Bevilacqua, B., Frasca, F., Lim, D., Srinivasan, B., Cai, C., Balamurugan, G., Bronstein, M. M., & Maron, H. (2022). Equivariant Subgraph Aggregation Networks. *ICLR 2022*. [OpenReview](https://openreview.net/forum?id=6buz2fR0Nw9). <a href="#cite-bevilacqua2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-rampasek2022"></span>Rampášek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a General, Powerful, Scalable Graph Transformer. *NeurIPS 2022*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html). <a href="#cite-rampasek2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-lim2023"></span>Lim, D., Robinson, J. D., Zhao, L., Smidt, T., Sra, S., Maron, H., & Jegelka, S. (2023). Sign and Basis Invariant Networks for Spectral Graph Representation Learning. *ICLR 2023*. [OpenReview](https://openreview.net/forum?id=Q-UHqMorzil). <a href="#cite-lim2023" class="reversefootnote" role="doc-backlink">↩</a>
