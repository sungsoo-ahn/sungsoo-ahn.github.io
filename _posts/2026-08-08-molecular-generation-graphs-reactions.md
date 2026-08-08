---
layout: post
title: "Generating Molecular Graphs and Chemical Reactions"
date: 2026-08-08
last_updated: 2026-08-08
description: "Molecular graph generation and reaction modeling viewed as constrained structured prediction, from representation and symmetry to synthesis-aware evaluation."
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [molecular-science]
lecture_paths: [ml4mol]
tags: [molecular-generation, graph-generative-models, reaction-prediction, retrosynthesis, discrete-diffusion]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>Note: This post develops the molecule-generation and reaction storyline
    from my Machine Learning for Molecules lecture, treating the two as sides
    of constrained graph modeling. The first proposes chemical objects;
    the second asks how those objects can change or be made. The emphasis is on
    the representations, symmetries, and evaluation decisions that connect the
    two.</em>
</p>

## A Molecule Is Not an Arbitrary Graph

The usual statement of molecular generation sounds simple: learn a distribution $$p_\theta(G)$$ from observed molecular graphs, then sample new graphs with desired properties. The difficulty is concentrated in the word *graph*. An image remains an image when a few pixels are wrong. A molecular graph can cease to describe a molecule when one bond is wrong, one charge is omitted, or one stereocenter is inverted.

Write a molecular graph as $$G=(V,E)$$. Each node carries an atom type and possibly formal charge, aromaticity, or chirality. Each edge carries a bond type and possibly stereochemical information. Validity couples these labels. Neutral carbon and oxygen allow different valence patterns; aromatic bonds participate in rings; disconnected components may represent salts or reagents rather than one compound. A decoder that predicts each label accurately in isolation can still produce an inconsistent combination.

The representation also contains symmetries that the model should not mistake for chemical differences. Ethanol has an underlying heavy-atom graph $$\mathrm{C{-}C{-}O}$$. It can be serialized as `CCO` or `OCC`, and its nodes can be indexed in six possible orders, but these choices do not create six molecules. For every node permutation $$\pi$$, a graph distribution should satisfy

$$
p_\theta(G)=p_\theta(\pi G),
$$

where $$\pi G$$ consistently permutes the node and edge tensors. A string model encounters a related many-to-one problem because a molecule can have multiple valid SMILES strings. Canonicalization chooses one spelling; randomized SMILES exposes several spellings; neither changes the underlying chemical object.

{% include figure.liquid loading="eager" path="assets/img/blog/molgenrxn_representation_contract.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A molecular representation must ignore irrelevant serialization choices while preserving chemically decisive labels. Ethanol can be written as `CCO` or `OCC`, but a neutral oxygen assigned three single bonds is graph-shaped rather than chemically valid; validity is a coupled constraint across atom and bond attributes." %}

This tension drives architecture. Sequential models make constraints easier to check after every edit but introduce an arbitrary construction order. Parallel models respect permutation symmetry more naturally but must coordinate many node and edge decisions at once. Motif-based models move the constraint boundary again: they assemble larger valid pieces, at the cost of deciding in advance which pieces exist.

## Autoregressive Generation Turns a Graph Into a History

An autoregressive generator chooses a sequence of actions $$a_1,\ldots,a_T$$ such as add an atom, attach a bond, close a ring, or stop. Given an ordering $$\pi$$, the chain rule gives

$$
p_\theta(G,\pi)
=\prod_{t=1}^{T}p_\theta(a_t\mid a_1,\ldots,a_{t-1}).
$$

The partial molecule is the state. Its advantage is immediate: an action grammar can mask choices that violate allowed valences or attachment rules. If the current carbon already has four bond-order units, another bond can be forbidden. Errors remain possible, but validity can be defended locally rather than repaired after a full adjacency matrix appears.

The price is that the model learns histories, not only graphs. The correct graph likelihood requires summing over all action sequences that produce it,

$$
p_\theta(G)=\sum_{\pi\in\Pi(G)}p_\theta(G,\pi),
$$

which is usually intractable. Training therefore selects or samples an ordering. Breadth-first order shortens the range of edge decisions because a new node is likely to connect near the current frontier. It does not remove the ordering dependence. GraphRNN made this sequential node-and-edge decomposition explicit for general graph generation (<span id="cite-you2018"></span>[You et al., 2018](#ref-you2018)).

Autoregression also creates exposure bias. During training, the next action is conditioned on a correct partial graph. During sampling, it is conditioned on the model's own earlier choices. An unlikely bond added at step 5 may force awkward actions at steps 6 through 20. Hard masks prevent some invalid paths, but they cannot recognize every unstable ring system, implausible functional-group combination, or poor synthetic decision.

Motif generation reduces the horizon by treating rings or functional fragments as larger tokens. A benzene ring can be added in one decision instead of six atom additions and six bond decisions. Junction-tree generation formalizes this idea by first building a tree of chemical substructures and then assembling the detailed molecular graph (<span id="cite-jin2018"></span>[Jin et al., 2018](#ref-jin2018)). The tradeoff is structural: a vocabulary that makes common chemistry easy can make genuinely new motifs unreachable. “Validity by construction” is always validity relative to the construction grammar.

## Parallel Generation Moves the Burden to Joint Consistency

A one-shot graph generator predicts a node tensor and edge tensor together. A discrete diffusion model makes this prediction iterative: it corrupts categorical atom and bond labels, then learns to reverse the corruption. At a noisy step, each candidate edge may be absent, single, double, aromatic, or masked; each node may carry a noisy atom label. A permutation-equivariant network updates all of them without choosing a canonical node order.

Discrete state spaces matter here. Relaxing an adjacency matrix to continuous values can simplify optimization, but a value of $$0.4$$ is not a chemical bond. The decoder eventually has to return to categorical decisions, where small continuous errors can change connectivity. Discrete denoising keeps the intermediate variables in the same categorical language as the output. DiGress is a representative construction: a Markov chain corrupts node and edge categories, and a graph transformer predicts the clean graph needed for reverse transitions (<span id="cite-vignac2023"></span>[Vignac et al., 2023](#ref-vignac2023)).

{% include figure.liquid loading="eager" path="assets/img/blog/molgenrxn_generation_strategies.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Autoregressive models can validate each partial molecule, but sampling is serial and depends on an action order. One-shot and discrete-denoising models update many categorical variables in parallel and avoid a canonical construction history, but size, connectivity, and valence must become consistent jointly." %}

Parallel prediction does not automatically solve graph generation. The model still has to choose graph size or represent absent nodes, preserve symmetry while distinguishing atoms with different environments, and couple distant decisions such as ring closure. Independent edge logits may assign too many bonds to one atom. Enforcing valence after sampling can raise the validity rate while distorting the learned distribution: the postprocessor, not the model, decides which errors survive.

The two paradigms therefore place the same difficulty in different locations. Autoregressive generation serializes a global constraint problem into locally checkable steps. Parallel generation preserves graph symmetry but asks the network and reverse process to coordinate global constraints. A useful comparison reports not only final validity and speed, but also how constraints were imposed, which failures were filtered, and whether filtering changed diversity.

## Conditional Design Is More Than Adding a Property Label

Unconditional generation imitates a molecular dataset. Design asks for a conditional distribution $$p(G\mid\mathbf{y})$$, where $$\mathbf{y}$$ might contain solubility, binding affinity, band gap, toxicity, or a required scaffold. Conditioning can enter as an embedding supplied to every decoding step, as a conditional denoiser, or as guidance from a property predictor. Formally,

$$
p(G\mid\mathbf{y})\propto p(\mathbf{y}\mid G)p(G).
$$

The prior $$p(G)$$ keeps proposals near learned chemistry; the likelihood-like term rewards the requested property. Stronger guidance shifts probability toward the condition but can push the model outside the region where the property predictor is trustworthy.

Consider asking for high lipophilicity using a learned proxy. A generator may discover that extending hydrocarbon chains increases the score. It can then produce repetitive, insoluble molecules that satisfy the scalar proxy while violating the broader design intent. A binding predictor can be exploited through unusual charged groups or structures far from its training domain. Conditional success must therefore include uncertainty, domain of applicability, and competing properties—not merely the value of the optimized oracle.

Constraints also differ in kind. “Contains this scaffold” can often be enforced exactly by freezing a subgraph. “Has activity below 10 nM” is a noisy experimental claim mediated by a predictor. “Can be synthesized in three steps from available stock” requires a route, not a graph-local property. Treating all three as interchangeable conditioning vectors hides where evidence comes from.

## A Reaction Is a Small Edit Embedded in a Large Graph

Forward reaction prediction receives reactants and reagents and predicts products. Retrosynthesis reverses the question: given a desired product, propose precursor sets that could make it. These are not inverse functions. Several conditions can transform the same reactants into different products, and many precursor sets can lead to the same target.

For many recorded single-step reactions, most atoms and bonds are unchanged. The transformation is concentrated in a small **reaction center**. This suggests an edit representation

$$
G_{\mathrm{product}}=\operatorname{Apply}(G_{\mathrm{reactants}},\Delta),
$$

where $$\Delta$$ contains bond deletions, bond additions, and bond-order changes. In a schematic nucleophilic substitution of bromoethane by hydroxide, the heavy-atom edit set deletes the carbon–bromine bond and adds a carbon–oxygen bond. Carbon skeleton atoms persist; the leaving group becomes a separate bromide component.

{% include figure.liquid loading="eager" path="assets/img/blog/molgenrxn_reaction_edits_mapping.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="In a schematic substitution, the product differs from the reactants by a small edit set: delete the carbon–bromine bond and add a carbon–oxygen bond. Atom mapping identifies persistent atoms, but equivalent atoms can exchange labels without changing the unlabeled chemical graph, so mapping is supervision rather than chemical ground truth." %}

Template-based models store reaction patterns that specify a local environment and edit. Templates supply a strong chemical grammar and interpretable precedent, but their precision competes with coverage. A narrow pattern applies safely to few substrates; a broad pattern retrieves more cases but may ignore selectivity. Template-free graph models instead predict likely reaction centers, enumerate a small set of edited products, and rank them. This sparse-edit strategy avoids enumerating all possible product graphs; an early example learned reaction-center scores and ranked resulting candidates with a graph network (<span id="cite-jin2017"></span>[Jin et al., 2017](#ref-jin2017)).

Sequence models take another route: serialize reactants and products as SMILES and treat reaction prediction as translation. This removes explicit templates and can capture long-range context. The Molecular Transformer showed that attention-based sequence prediction could handle forward reactions and produce useful uncertainty estimates (<span id="cite-schwaller2019"></span>[Schwaller et al., 2019](#ref-schwaller2019)). Yet a syntactically fluent product can still violate atom conservation, choose the wrong stereoisomer, or omit a minor component. Graph edits and strings emphasize different structure; neither representation supplies missing conditions, yields, or experimental context.

## Atom Mapping Is Necessary—and Not Unique

To learn edits, we need to know which reactant atom corresponds to which product atom. An atom mapping is a partial bijection between conserved atoms on the two sides. Once mapping is fixed, a bond present only on the reactant side is a deletion, and a bond present only on the product side is an addition.

But the chemical graph may admit automorphisms: permutations that leave it unchanged. In a carboxylate group, the two oxygen positions can be equivalent under resonance and symmetry at the level used by a dataset. Swapping their map numbers produces the same unlabeled structure but a different labeled correspondence. A mapping pipeline that makes inconsistent choices across examples can turn one chemical transformation into several apparent edit patterns. A model is then penalized for disagreeing with bookkeeping even when its product graph is correct.

Modern mapping systems can infer correspondences from reaction data without handcrafted reaction rules. Attention-derived mapping, for example, revealed that a reaction language model had learned atom correspondences implicitly (<span id="cite-schwaller2021"></span>[Schwaller et al., 2021](#ref-schwaller2021)). This is powerful preprocessing, but it should remain visible in evaluation. Report whether scoring compares mapped edits, canonicalized product graphs, or chemically equivalent products; these questions have different answers under symmetry.

Atom mapping can also leak the solution. If product-informed mappings or reaction-center labels are available to a model at test time, the task is no longer ordinary forward prediction from raw reactants. The safest pipeline separates information used to normalize training data from information the deployed model could actually observe.

## Retrosynthesis Turns One Edit Into a Search Problem

A one-step retrosynthesis model proposes precursor sets for a target. A synthesis plan recursively applies such disconnections until every leaf is an available starting material. If each target has $$b$$ plausible disconnections and a route has depth $$d$$, naive enumeration grows like $$b^d$$. The most accurate single-step predictor is not automatically the best planner: it may rank ten nearly identical disconnections highly while missing the one that leads to short, purchasable branches.

Planning therefore combines learned proposals with search. A policy prioritizes disconnections; a value or rollout estimate judges whether branches are likely to reach building blocks; filters reject implausible forward reactions. Neural-guided tree search demonstrated how learned reaction policies and symbolic search can work together for multistep planning (<span id="cite-segler2018"></span>[Segler et al., 2018](#ref-segler2018)).

The route is still a hypothesis. Patent datasets overrepresent successful reactions and often omit failed conditions. Reagent identity may be inconsistently separated from reactants. Yields from different laboratories are not directly comparable, and a step that works on milligram scale may fail at scale-up. A route with known reaction classes but unavailable starting materials is not actionable. Synthesis constraints should therefore enter generation early—through reaction-based construction, purchasable fragments, or route-aware objectives—rather than appear only as a final scalar penalty.

## Validity and Novelty Are the Beginning of Evaluation

Validity asks whether software can parse and sanitize a generated structure. Uniqueness asks whether samples repeat. Novelty asks whether they are absent from a reference set. These are necessary diagnostics, but each admits a weak solution. A model can generate many valid carbon chains, achieve high uniqueness through small substitutions, and appear novel because the training database is finite. None of this demonstrates useful chemical design.

Benchmarks such as GuacaMol expanded evaluation toward distribution matching and goal-directed, multi-objective tasks (<span id="cite-brown2019"></span>[Brown et al., 2019](#ref-brown2019)). A serious evaluation should continue further:

- **Canonicalize before counting.** Different SMILES strings or atom orders should not inflate uniqueness and novelty. Near-duplicate scaffolds should be measured separately from exact duplicates.

- **Compare distributions, not only averages.** Atom types, ring systems, scaffold frequencies, molecular sizes, and property distributions expose mode dropping that a mean score hides.

- **Audit conditional calibration.** Measure success across requested property ranges, including hard or rare conditions, and use predictors that were not optimized jointly with the generator.

- **Respect reaction equivalence.** Product accuracy should account for canonicalization, atom-mapping ambiguity, stereochemistry, and multiple plausible products. Retrosynthesis needs top-$$k$$ precursor accuracy, diversity of disconnections, and route-level success.

- **Test synthesizability with routes.** Heuristic synthetic-accessibility scores are cheap filters, not proofs. Route planners should use available building blocks and report search budget, route length, precedent, and forward-validation confidence.

- **Reserve the final claim for experiments.** Predicted affinity, a plausible route, and a stable structure are evidence filters. Synthesis, characterization, and measured function are the test.

{% include figure.liquid loading="eager" path="assets/img/blog/molgenrxn_evaluation_funnel.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Evaluation becomes progressively more expensive as it approaches the scientific claim. Parsing, novelty, and property scores filter proposals; route planning tests practical access; only synthesis and measurement establish whether the designed molecule exists and has the intended behavior." %}

This final distinction matters because common objectives can disagree. Biasing generation toward structures that a planner can synthesize may reduce the best predicted property score. Gao and Coley showed that molecules favored by generative benchmarks can be difficult for synthesis planners and that synthetic-complexity bias trades against the primary objective (<span id="cite-gao2020"></span>[Gao & Coley, 2020](#ref-gao2020)). That is not an evaluation nuisance. It is the real multi-objective structure of molecular discovery.

## Generation and Reaction Modeling Belong in One Loop

Molecular graph generation proposes what might exist. Reaction prediction asks how graphs transform. Retrosynthetic planning asks whether a sequence of those transformations reaches available matter. Keeping the tasks separate makes each benchmark cleaner, but it also creates the easiest failure mode in molecular AI: an impressive proposal with no credible path to a flask.

The durable abstraction is constrained structured prediction. The representation must quotient out irrelevant orderings while retaining charge, bond order, and stereochemistry. The generator must coordinate local valence with global connectivity. The reaction model must locate sparse edits without confusing atom-map bookkeeping for chemistry. The planner must turn plausible local steps into a globally feasible route. Evaluation must follow the same chain all the way to experiment.

The most useful molecular generator is therefore not the one that produces the largest number of novel valid graphs. It is the one whose proposals survive increasingly realistic constraints without collapsing to the training set—and whose failures remain legible enough to guide the next design cycle.

---

## References

<span id="ref-you2018"></span>You, J., Ying, R., Ren, X., Hamilton, W., & Leskovec, J. (2018). [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models](https://proceedings.mlr.press/v80/you18a.html). *Proceedings of the 35th International Conference on Machine Learning*. [↩](#cite-you2018)

<span id="ref-jin2018"></span>Jin, W., Barzilay, R., & Jaakkola, T. (2018). [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://proceedings.mlr.press/v80/jin18a.html). *Proceedings of the 35th International Conference on Machine Learning*. [↩](#cite-jin2018)

<span id="ref-vignac2023"></span>Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., & Frossard, P. (2023). [DiGress: Discrete Denoising Diffusion for Graph Generation](https://openreview.net/forum?id=UaAD-Nu86WX). *International Conference on Learning Representations*. [↩](#cite-vignac2023)

<span id="ref-jin2017"></span>Jin, W., Coley, C., Barzilay, R., & Jaakkola, T. (2017). [Predicting Organic Reaction Outcomes with Weisfeiler–Lehman Network](https://proceedings.neurips.cc/paper/2017/hash/ced556cd9f9c0c8315cfbe0744a3baf0-Abstract.html). *Advances in Neural Information Processing Systems, 30*. [↩](#cite-jin2017)

<span id="ref-schwaller2019"></span>Schwaller, P., Laino, T., Gaudin, T., Bolgar, P., Hunter, C. A., Bekas, C., & Lee, A. A. (2019). [Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction](https://doi.org/10.1021/acscentsci.9b00576). *ACS Central Science, 5*(9), 1572–1583. [↩](#cite-schwaller2019)

<span id="ref-schwaller2021"></span>Schwaller, P., Hoover, B., Reymond, J.-L., Strobelt, H., & Laino, T. (2021). [Extraction of Organic Chemistry Grammar from Unsupervised Learning of Chemical Reactions](https://doi.org/10.1126/sciadv.abe4166). *Science Advances, 7*(15), eabe4166. [↩](#cite-schwaller2021)

<span id="ref-segler2018"></span>Segler, M. H. S., Preuss, M., & Waller, M. P. (2018). [Planning Chemical Syntheses with Deep Neural Networks and Symbolic AI](https://www.nature.com/articles/nature25978). *Nature, 555*, 604–610. [↩](#cite-segler2018)

<span id="ref-brown2019"></span>Brown, N., Fiscato, M., Segler, M. H. S., & Vaucher, A. C. (2019). [GuacaMol: Benchmarking Models for de Novo Molecular Design](https://doi.org/10.1021/acs.jcim.8b00839). *Journal of Chemical Information and Modeling, 59*(3), 1096–1108. [↩](#cite-brown2019)

<span id="ref-gao2020"></span>Gao, W., & Coley, C. W. (2020). [The Synthesizability of Molecules Proposed by Generative Models](https://doi.org/10.1021/acs.jcim.0c00174). *Journal of Chemical Information and Modeling, 60*(12), 5714–5723. [↩](#cite-gao2020)
