---
layout: post
title: "Human–AI Co-Discovery of a State-of-the-Art Crystal Structure Prediction Algorithm"
date: 2026-06-19
last_updated: 2026-06-21
description: "How a human–AI co-scientist loop produced MaskGXT, a state-of-the-art deep learning algorithm for crystal structure prediction."
post_type: research
authors: ["Kiyoung Seong", "Sungsoo Ahn"]
categories: [machine-learning]
tags: [ai-scientist, generative-models, crystal-structure-prediction, materials, masked-generative-models]
toc:
  sidebar: left
related_posts: false
published: true
---

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_hero.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>An AI co-scientist discovers a new CSP algorithm.</strong> The co-scientist transferred MaskGIT's masked-generation idea from vision to crystal structure prediction: fill in the sites of a crystal lattice through iterative unmasking." %}

AI co-scientists may be ready to change deep learning research. In our recent
work, we show that a human–AI co-scientist loop can produce a state-of-the-art
algorithm for **crystal structure prediction**[^csp]: generating plausible crystal
structures from chemical compositions.

This story sits within a rapidly growing wave of AI-for-science systems. Recent
systems such as FunSearch (<span id="cite-romeraparedes2024"></span>[Romera-Paredes et al., 2024](#ref-romeraparedes2024)),
AlphaEvolve (<span id="cite-novikov2025"></span>[Novikov et al., 2025](#ref-novikov2025)), The AI Scientist
(<span id="cite-lu2024"></span>[Lu et al., 2024](#ref-lu2024); <span id="cite-yamada2025"></span>[Yamada et al., 2025](#ref-yamada2025)), AIDE
(<span id="cite-jiang2025"></span>[Jiang et al., 2025](#ref-jiang2025)), and Google's AI co-scientist
(<span id="cite-gottweis2025"></span>[Gottweis et al., 2025](#ref-gottweis2025)) show that language-model agents can
search programs, generate hypotheses,
run experiments, and organize scientific reasoning. However, for deep learning
research, the strongest demonstrations have still mostly looked like program
search, hypothesis generation, code-level experimentation, or automated research
workflow construction. What remains rare is a clear case where a human–AI
co-scientist loop yields a reusable **deep learning algorithm**: a modeling
formulation, objective, or sampler that improves a real scientific ML benchmark.

That is what we tried to demonstrate with **MaskGXT**, the **Masked Generative
Crystal Transformer**. MaskGXT treats a crystal as a sequence of discrete tokens:
lattice parameters, fractional coordinates, space group, and Wyckoff
positions.[^wyckoff] It learns to complete the missing tokens with a transformer.
It transfers the masked-generation principle behind
**MaskGIT** (<span id="cite-chang2022"></span>[Chang et al., 2022](#ref-chang2022)) from
computer vision to crystals, then adapts it to periodic coordinates,
crystallographic symmetry, and polymorph coverage.

The broader lesson is not full autonomy. It is that fast evaluation can turn
**research taste into an empirical loop**: humans propose risky model-level
directions, and the co-scientist tests, combines, and refines them at scale.
MaskGXT is one example of that loop producing a state-of-the-art deep learning
algorithm.

Code is available for both the
[AI co-scientist search loop](https://github.com/kiyoung98/NanoCSP-agent) and
[MaskGXT](https://github.com/kiyoung98/maskgxt).

## The result: a new state of the art

A single MaskGXT model reaches the best match rate on the standard MP-20 and
MPTS-52 benchmarks. Its largest advantage appears in polymorph-aware evaluation,
where the model must recover multiple structures that can arise from the same
chemical composition (<span id="cite-martirossyan2025"></span>[Martirossyan et al., 2025](#ref-martirossyan2025)).[^metre]

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_results_bars.svg" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>MaskGXT benchmark results.</strong> Bars summarize the main paper's filtered standard-CSP scores and held-out METRe scores; higher is better for MR and METRe, lower is better for RMSE and cRMSE. MaskGXT wins the match-rate and METRe columns, with its largest margin on the MP-20 polymorph split." %}

On the MP-20 polymorph split, MaskGXT raises METRe from **70.87% to 79.06%**.
This is not a random benchmark victory.

CSP has become one of the most active testbeds for generative modeling in
materials science. Early learned crystal generators such as
CDVAE (<span id="cite-xie2022"></span>[Xie et al., 2022](#ref-xie2022)) and DiffCSP
(<span id="cite-jiao2023"></span>[Jiao et al., 2023](#ref-jiao2023)) made periodic coordinate generation a
benchmark problem. FlowMM (<span id="cite-miller2024"></span>[Miller et al., 2024](#ref-miller2024)), OMatG
(<span id="cite-hoellmer2025"></span>[Hoellmer et al., 2025](#ref-hoellmer2025)), CrystalFlow
(<span id="cite-luo2025"></span>[Luo et al., 2025](#ref-luo2025)), and MCFlow
(<span id="cite-seong2026"></span>[Seong et al., 2026](#ref-seong2026)) developed continuous flow, Riemannian,
stochastic-interpolant, and multimodal formulations.

A large line of symmetry-aware methods---including
DiffCSP++ (<span id="cite-jiao2024"></span>[Jiao et al., 2024](#ref-jiao2024)), WyCryst
(<span id="cite-zhu2024"></span>[Zhu et al., 2024](#ref-zhu2024)), Wyckoff Transformer
(<span id="cite-kazeev2025"></span>[Kazeev et al., 2025](#ref-kazeev2025)), WyckoffDiff
(<span id="cite-kelvinius2025"></span>[Kelvinius et al., 2025](#ref-kelvinius2025)), SymmCD
(<span id="cite-levy2025"></span>[Levy et al., 2025](#ref-levy2025)), and CrystalFormer
(<span id="cite-cao2025"></span>[Cao et al., 2025](#ref-cao2025))---explicitly modeled space groups, Wyckoff
positions, or crystallographic constraints. Recent materials-generation and
lightweight-transformer models such as MatterGen (<span id="cite-zeni2025"></span>[Zeni et al., 2025](#ref-zeni2025))
and Crystalite (<span id="cite-veljkovic2026"></span>[Veljković et al., 2026](#ref-veljkovic2026)) add further
competitive baselines. In other words, MaskGXT was not improving an
underexplored toy benchmark. It was competing against several years of
domain-specific architecture and generative-model design.

## How the co-scientist loop worked

{% include video.liquid path="assets/video/maskgxt_agent_anim.mp4" class="img-fluid rounded z-depth-1 mx-auto d-block" poster="assets/img/blog/maskgxt_agent.png" autoplay=true loop=true muted=true controls=true caption="<strong>The AI co-scientist loop.</strong> A tree-structured search organizes candidate CSP methods; each node is a complete generative model. The tree grows as idea, draft, debug, and improve operators are applied, while human input enters only sparsely as high-level mechanisms or objectives." %}

The co-scientist framework itself was not new. We mostly reused the ingredients
that appear across recent AI-scientist systems: a tree of candidate ideas,
operators for proposing, drafting, debugging, and improving candidates,
executable experiments, and score-based selection. The difference was the human
interface. We allowed humans to intervene during the search with high-level
mechanisms and objectives, then let the loop turn those hints into code,
experiments, and further search.

The search target was also different. Many ML-agent workflows start with the
broad modeling family already fixed and then search for better code,
hyperparameters, or implementation details. Here, we made the **generative
methodology itself** the object of search. Rather than restricting the loop to
known CSP models, we used it to explore frameworks from other fields that had
credible mechanisms for crystals.

The search was organized as a tree. Each node was a complete candidate method:
proposed, implemented, trained under a fixed budget, and evaluated on validation
METRe within the co-scientist loop. Scores determined which branches to expand,
debug, or abandon. The exploration covered fourteen cross-domain frameworks,
including autoregressive transformers from language modeling, masked generative
transformers from vision, and state-space interpolants from sequence modeling.
The MaskGIT branch became the strongest lineage and was then developed into
MaskGXT.

Explore the full search tree below.

<div class="row justify-content-center my-4">
  <div class="col-12">
    <div class="position-relative">
      <iframe
        src="https://kiyoung98.github.io/NanoCSP-agent/"
        title="MaskGXT search tree"
        loading="lazy"
        class="img-fluid rounded z-depth-1 d-block w-100"
        style="height: 70vh; border: 0;"
        allowfullscreen></iframe>
      <a href="https://kiyoung98.github.io/NanoCSP-agent/" target="_blank" rel="noopener"
         class="btn btn-light position-absolute"
         style="top: 0.5rem; right: 0.5rem; padding: 0.15rem 0.5rem; font-size: 0.7rem; line-height: 1.4; opacity: 0.9;">Open ↗</a>
    </div>
  </div>
</div>

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_trajectory.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>The research trajectory toward MaskGXT.</strong> Validation METRe against the number of trials; the black step line is the running best. The three shaded bands are the search stages, with the per-candidate budget escalating from 2h to 12h training and then 30m of sampling tuning." %}

The trajectory matters. This was not one impressive chat response, but a long
empirical process: propose a mechanism, write runnable code, train it, inspect
the result, preserve what worked, and try again. Across roughly five hundred
trials, research ideas became measurable bets rather than prose suggestions.

## Where human research taste entered

The masked generative formulation and most of the model code emerged from the
automated search. Before the human-guided refinements, the best branch was
already competitive with the previous state of the art. We then supplied four
missing mechanisms or objectives:

- **Symmetry tokens.** We suggested representing crystallographic symmetry
  explicitly, pointing to DiffCSP++ ([Jiao et al., 2024](#ref-jiao2024)) and
  WyFormer ([Kazeev et al., 2025](#ref-kazeev2025)).
- **Symmetry-preserving augmentation.** We suggested orbit permutation over a
  symmetry-based atom ordering, pointing to MCFlow
  ([Seong et al., 2026](#ref-seong2026)).
- **Stratified sampling.** We supplied the mechanism that non-i.i.d. sampling
  could improve polymorph coverage.[^iid]
- **Sub-bin precision.** We supplied only the objective---recover the precision
  lost by discretization. The continuous offset mechanism was then developed
  within the loop.

This boundary is important. MaskGXT was neither a fully autonomous discovery nor
a conventional human-designed method. Human input entered at a few bottlenecks as
research taste; the co-scientist loop converted those hints into code,
experiments, ablations, and further improvements.

**Autonomy is not the central claim. Scientific leverage is.**

## The resulting algorithm: MaskGXT

{% include figure.liquid loading="eager" path="assets/img/blog/maskgxt_overview.png" class="img-fluid rounded z-depth-1 mx-auto d-block" zoomable=true caption="<strong>How MaskGXT works.</strong> (a) Tokenizing a crystal: one space group token, six lattice tokens, and five tokens per atom site. (b) Training reconstructs randomly masked tokens. (c) Sampling branches over space groups to cover polymorphs, then greedily unmasks the rest." %}

The transferred core is a discrete masked formulation. MaskGXT represents the
space group, lattice, fractional coordinates, Wyckoff positions, and atom sites
as a sequence of tokens. During training it masks a random subset and learns to
reconstruct them, as MaskGIT reconstructs masked image tokens.

A direct transfer would not work. The final formulation adds several
crystal-specific mechanisms:

- **Periodic ordinal smoothing** treats nearby coordinate bins as similar and
  makes bins near 0 and 1 neighbors.
- **Symmetry tokens and symmetry-preserving augmentation** expose the model to
  crystallographic structure and equivalent crystal descriptions.
- **Sub-bin regression** restores continuous geometric precision inside each
  discrete coordinate bin.
- **Confidence-ranked greedy decoding** produces a high-probability structure
  from the finite token space.
- **Space-group-stratified sampling** branches over likely symmetries, producing
  diverse polymorph candidates instead of redundant draws.

The important scientific idea is therefore not “apply MaskGIT unchanged.” It is
“transfer a useful modeling principle, then discover the domain adaptations that
make the transfer work.”

## Our take: start using AI scientists now

### Start with one problem, not all of science

We think the usual question---“Can AI autonomously do science?”---is too broad to
guide research practice. A more useful question is: **which part of a current
problem can be converted into a closed loop of proposal, implementation,
evaluation, and revision?**

This result does not show that AI can solve materials science in general. It
shows something narrower: when a problem has code, data, and a meaningful score,
a co-scientist loop can become a practical research tool. We would start with
those concrete bottlenecks rather than wait for a system that can address every
scientific problem at once.

### Evaluation may be the real infrastructure

We suspect AI co-scientist workflows are strongest when evaluation is fast enough
to turn **research taste into an empirical loop**. CSP offered fixed data,
executable models, training runs measured in hours, and a validation metric
reasonably aligned with the final objective. This allowed the loop to test many
ideas without asking a human to judge every intermediate result.

For many areas of science, we expect the decisive bottleneck to be neither idea
generation nor coding. It may be the design of **fast evaluation protocols that
approximate the ultimate scientific goal well enough to guide search**. Better
surrogates, small-scale experiments, predictive scaling laws, and carefully
designed proxy tasks may matter as much as better foundation models.

A powerful model paired with a misleading metric may optimize the wrong thing
faster. A merely competent co-scientist workflow paired with a well-designed loop
may discover something useful.

### Co-scientist loops can give humans more room to be novel

There is a common concern that AI will make research more incremental because it
is trained on existing work. We believe that the opposite can happen.
Current AI systems are extremely effective at exploiting and combining existing
ideas: they can implement unfamiliar methods, debug interactions, run ablations,
and search through many combinations much faster than a human researcher.

That lowers the cost of trying an idea that may fail. Humans can spend more of
their limited attention on unusual abstractions, cross-domain hypotheses, and
new objectives, while the co-scientist loop performs the large amount of
exploitation needed to determine whether those ideas survive implementation and
evaluation.

In this division of labor, AI does not replace research taste. It makes research
taste cheaper to exercise.

### The human role may shift toward goal and loop design

If implementation and empirical search become increasingly automatable, the
human role moves upward: choosing a worthwhile problem, constructing an aligned
metric, supplying missing domain mechanisms, recognizing misleading evidence,
and deciding when the objective itself should change.

The main bottleneck may eventually be human attention. One run can create
hundreds of hypotheses, code variants, logs, plots, and failures. Useful systems
will need to maintain a compact research state: what was tried, why it failed,
what mechanisms remain unexplored, and which decisions truly require human
judgment.

## Conclusion

MaskGXT is one case study, not evidence that current systems can autonomously
solve arbitrary scientific problems. But it is evidence for a more immediate and
actionable claim: **human–AI co-scientist workflows may already be capable enough
to contribute to discovery when we design the research loop correctly.**

The practical next step, in our view, is not to wait for an AI that can solve all
of science. It is to look at the problems already on our desks and ask which
parts can be made searchable now.

These workflows become useful long before they can replace scientists. They
become useful when they multiply our ability to try, evaluate, combine, and
refine ideas.

## References

- <span id="ref-cao2025"></span>Cao, Z., Luo, X., Lv, J. & Wang, L. (2025). Space Group Informed Transformer for Crystalline Materials Generation. [Science Bulletin](https://doi.org/10.1016/j.scib.2025.09.035). <a href="#cite-cao2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-chang2022"></span>Chang, H., Zhang, H., Jiang, L., Liu, C. & Freeman, W. T. (2022). MaskGIT: Masked Generative Image Transformer. [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html). <a href="#cite-chang2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-gottweis2025"></span>Gottweis, J., et al. (2025). Towards an AI Co-Scientist. [arXiv:2502.18864](https://arxiv.org/abs/2502.18864). <a href="#cite-gottweis2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-hoellmer2025"></span>Hoellmer, P., et al. (2025). Open Materials Generation with Stochastic Interpolants. [ICML 2025](https://proceedings.mlr.press/v267/hollmer25a.html). <a href="#cite-hoellmer2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-jiang2025"></span>Jiang, Z., et al. (2025). AIDE: AI-Driven Exploration in the Space of Code. [arXiv:2502.13138](https://arxiv.org/abs/2502.13138). <a href="#cite-jiang2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-jiao2023"></span>Jiao, R., et al. (2023). Crystal Structure Prediction by Joint Equivariant Diffusion (DiffCSP). [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html). <a href="#cite-jiao2023" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-jiao2024"></span>Jiao, R., Huang, W., Liu, Y., Zhao, D. & Liu, Y. (2024). Space Group Constrained Crystal Generation (DiffCSP++). [ICLR 2024](https://openreview.net/forum?id=jkvZ7v4OmP). <a href="#cite-jiao2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kazeev2025"></span>Kazeev, N., et al. (2025). Wyckoff Transformer: Generation of Symmetric Crystals (WyFormer). [ICML 2025](https://proceedings.mlr.press/v267/kazeev25a.html). <a href="#cite-kazeev2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-kelvinius2025"></span>Kelvinius, F. E., et al. (2025). WyckoffDiff: A Generative Diffusion Model for Crystal Symmetry. [ICML 2025](https://proceedings.mlr.press/v267/ekstrom-kelvinius25a.html). <a href="#cite-kelvinius2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-levy2025"></span>Levy, D., et al. (2025). SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models. [ICLR 2025](https://openreview.net/forum?id=xnssGv9rpW). <a href="#cite-levy2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-lu2024"></span>Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J. & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. [arXiv:2408.06292](https://arxiv.org/abs/2408.06292). <a href="#cite-lu2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-luo2025"></span>Luo, X., Wang, Z., Lv, J., Wang, L., Wang, Y. & Ma, Y. (2025). CrystalFlow: A Flow-Based Generative Model for Crystalline Materials. [Nature Communications](https://doi.org/10.1038/s41467-025-64364-4). <a href="#cite-luo2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-martirossyan2025"></span>Martirossyan, M. M., et al. (2025). All That Structure Matches Does Not Glitter (METRe). [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/390a00871e5593fcf8717f83b2c1395f-Abstract-Datasets_and_Benchmarks_Track.html). <a href="#cite-martirossyan2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-miller2024"></span>Miller, B. K., Chen, R. T. Q., Sriram, A. & Wood, B. M. (2024). FlowMM: Generating Materials with Riemannian Flow Matching. [ICML 2024](https://proceedings.mlr.press/v235/miller24a.html). <a href="#cite-miller2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-novikov2025"></span>Novikov, A., et al. (2025). AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery. [arXiv:2506.13131](https://arxiv.org/abs/2506.13131). <a href="#cite-novikov2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-romeraparedes2024"></span>Romera-Paredes, B., et al. (2024). Mathematical Discoveries from Program Search with Large Language Models. [Nature](https://doi.org/10.1038/s41586-023-06924-6). <a href="#cite-romeraparedes2024" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-seong2026"></span>Seong, K., Ahn, S., Han, S. & Park, C. (2026). Multimodal Crystal Flow: Any-to-Any Modality Generation for Unified Crystal Modeling (MCFlow). [arXiv:2602.20210](https://arxiv.org/abs/2602.20210). <a href="#cite-seong2026" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-veljkovic2026"></span>Veljković, T. H., Rosenthal, J., Lončarić, I. & van de Meent, J.-W. (2026). Crystalite: A Lightweight Transformer for Efficient Crystal Modeling. [arXiv:2604.02270](https://arxiv.org/abs/2604.02270). <a href="#cite-veljkovic2026" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-xie2022"></span>Xie, T., Fu, X., Ganea, O.-E., Barzilay, R. & Jaakkola, T. (2022). Crystal Diffusion Variational Autoencoder for Periodic Material Generation (CDVAE). [ICLR 2022](https://openreview.net/forum?id=03RLpj-tc_). <a href="#cite-xie2022" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-yamada2025"></span>Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J. & Ha, D. (2025). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search. [arXiv:2504.08066](https://arxiv.org/abs/2504.08066). <a href="#cite-yamada2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-zeni2025"></span>Zeni, C., et al. (2025). MatterGen: A Generative Model for Inorganic Materials Design. [Nature](https://doi.org/10.1038/s41586-025-08628-5). <a href="#cite-zeni2025" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-zhu2024"></span>Zhu, R., Nong, W., Yamazaki, S. & Hippalgaonkar, K. (2024). WyCryst: Wyckoff Inorganic Crystal Generator Framework. [Matter](https://doi.org/10.1016/j.matt.2024.05.042). <a href="#cite-zhu2024" class="reversefootnote" role="doc-backlink">↩</a>

---

[^csp]: In this post, CSP means the benchmark version of crystal structure
    prediction: given a chemical composition, generate plausible crystal
    structures for that composition. It is narrower than exhaustive
    first-principles search over all possible structures.

[^wyckoff]: A **space group** describes the symmetry operations of a crystal.
    **Wyckoff positions** describe symmetry-equivalent sites inside that space
    group, so they provide a compact way to represent crystallographic
    constraints.

[^iid]: I.i.d. means independent and identically distributed. Here, non-i.i.d.
    sampling means deliberately splitting generation across likely space groups
    rather than drawing every sample from the same unconstrained distribution.

[^metre]: A composition can crystallize into several stable structures, or
    *polymorphs*. Match rate asks for one correct structure; METRe
    (match-everyone-to-reference) is stricter, rewarding recovery of *all* of a
    composition's polymorphs. The polymorph split is designed to stress this
    setting, and it is where MaskGXT's margin is widest. cRMSE combines coverage
    and geometric error by assigning unmatched references the matching tolerance.
