---
layout: post
title: "Three-Dimensional Molecular Generation and Optimization"
date: 2026-08-08
last_updated: 2026-08-08
description: "How generative models respect molecular geometry, how conditional guidance turns sampling into design, and why oracle scores must survive synthesis and experiment."
abstract: >
  Three-dimensional molecular design couples a symmetric geometric sampling problem to a constrained, multi-objective optimization problem. A credible workflow must preserve chemical structure, control oracle exploitation, and close the loop with synthesis and measurement.
post_type: tutorial
authors: ["Sungsoo Ahn"]
categories: [molecular-science]
lecture_paths: [ml4mol]
tags: [three-dimensional-generation, conformer-generation, molecular-optimization, diffusion-models, synthesizability]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
  <em>This post develops the three-dimensional generation and molecular-optimization storyline from my 2025 Machine Learning for Molecules lecture. The geometry of diffusion and flow on symmetric spaces is developed in <a href="{% post_url 2026-08-08-geometric-flow-matching-manifolds %}">Geometric Flow Matching on Manifolds</a>, while molecular property oracles are discussed in <a href="{% post_url 2026-08-08-molecular-data-property-prediction %}">Molecular Data and Property Prediction Across 1D, 2D, and 3D</a>.</em>
</p>

Generating a plausible molecule and optimizing a useful molecule are different problems. A generative model learns where molecular data live. An optimizer deliberately searches toward rare regions with high predicted reward. The first problem punishes leaving the data distribution; the second problem often demands it.

Three-dimensional structure makes this tension sharper. Coordinates contain redundant global translations and rotations. A molecular graph can admit many conformers, whose relative populations matter more than one best geometry. If the graph itself is generated, atom identities, bonds, chirality, coordinates, charge, and sometimes protonation state must agree. A property predictor then scores these objects under its own assumptions and domain limitations.

The result is not one monolithic inverse-design problem. It is a pipeline of probability spaces and approximations. The most important discipline is to say which variable is being generated, which symmetries identify equivalent states, which oracle supplies the reward, and which physical or experimental gate makes the final claim credible.

## Conformer generation keeps the molecule fixed

Represent a molecule by a graph

$$
G=(\mathbf{Z},\mathbf{B})
$$

and coordinates

$$
\mathbf{R}=(\mathbf{r}_1,\ldots,\mathbf{r}_N)\in\mathbb{R}^{N\times 3}.
$$

The atom types $$\mathbf{Z}$$ and bond structure $$\mathbf{B}$$ define connectivity. **Conformer generation** models

$$
p(\mathbf{R}\mid G),
$$

the distribution of geometries for that fixed graph. Rotation about single bonds can create several low-energy basins; rings, chirality, and steric interactions constrain which combinations are accessible. A useful output is therefore an ensemble with realistic coverage and weights, not simply the coordinate set closest to one reference structure.

**Molecule generation** models a larger joint distribution,

$$
p(G,\mathbf{R}),
$$

or factorizes it as $$p(G)p(\mathbf{R}\mid G)$$. Now the model may change atom count, element identity, bonds, and geometry. The generated coordinates and inferred bonds must describe the same chemical object. Bonding purely by distance can fail for aromaticity, formal charge, metals, or unusual valence, so joint graph–geometry models need explicit consistency checks or a carefully specified bond perception procedure.

{% include figure.liquid loading="eager" path="assets/img/blog/mol3dopt_two_tasks.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Conformer generation samples new coordinates while keeping atoms and bonds fixed. Full molecule generation may change atom identities, connectivity, and coordinates, so it operates on a larger state space with additional chemical-validity constraints. Original diagram." %}

The distinction controls evaluation. Conformer methods are compared by ensemble coverage, precision, RMSD after alignment, energy distributions, and sometimes Boltzmann-weighted observables. Molecule generators are evaluated by chemical validity, uniqueness, novelty, graph–geometry consistency, stability after relaxation, and downstream property distributions. Mixing these metrics makes a model appear to solve a problem it was never asked to represent.

## Three-dimensional probability lives modulo rigid motion

If $$\mathbf{Q}\in SO(3)$$ and $$\mathbf{t}\in\mathbb{R}^3$$, then

$$
\mathbf{R}'
=\mathbf{R}\mathbf{Q}^{\mathsf T}
+\mathbf{1}\mathbf{t}^{\mathsf T}
$$

describes the same isolated molecular geometry in a different laboratory frame. A physical density must assign equivalent probability to every such representation:

$$
p(\mathbf{R}\mid G)
=p(\mathbf{R}'\mid G).
$$

Translation is often removed by centering the coordinates,

$$
\widetilde{\mathbf{r}}_i
=\mathbf{r}_i
-\frac{1}{N}\sum_j\mathbf{r}_j.
$$

Rotation remains as a symmetry. A Cartesian diffusion model can use an equivariant denoiser or score, so rotating the noisy molecule rotates the predicted coordinate update. EDM applies this idea while jointly denoising coordinates and categorical atom features (<span id="cite-hoogeboom2022"></span>[Hoogeboom et al., 2022](#ref-hoogeboom2022)). Permutations of indistinguishable atom labels introduce another symmetry that the graph architecture must respect.

The density is invariant, but the coordinate score is equivariant:

$$
\nabla_{\mathbf{R}'}\log p(\mathbf{R}'\mid G)
=\left(\nabla_{\mathbf{R}}\log p(\mathbf{R}\mid G)\right)\mathbf{Q}^{\mathsf T}.
$$

This is not decorative data augmentation. It prevents the generator from spending capacity on arbitrary frame conventions and ensures that a rotated noisy state receives a correspondingly rotated denoising direction.

The same logic applies to continuous flows. A velocity field must be tangent to the centered coordinate subspace and equivariant under rotation. The [geometric flow-matching post]({% post_url 2026-08-08-geometric-flow-matching-manifolds %}) develops this construction for Euclidean, rotational, and product manifolds.

## Cartesian and torsional models choose different state spaces

Cartesian generation perturbs every atomic coordinate. It is flexible: the model can change bond lengths, bond angles, torsions, and global shape. That flexibility makes the prior simple—usually centered Gaussian noise—but asks the network to relearn strong local chemical constraints. Independent coordinate noise readily creates distorted bonds and atomic clashes at intermediate times.

Internal-coordinate methods instead parameterize bond lengths, angles, and dihedral angles. For a fixed graph whose local geometry is supplied or separately generated, much conformational flexibility lies in $$m$$ rotatable torsions,

$$
\boldsymbol{\tau}
=(\tau_1,\ldots,\tau_m)
\in\mathbb{T}^m,
$$

where each angle belongs to a circle and the full space is a torus. Torsional diffusion places its stochastic process on this periodic space and rotates whole molecular fragments about bonds (<span id="cite-jing2022"></span>[Jing et al., 2022](#ref-jing2022)). Bond lengths and most local angles remain fixed by construction.

{% include figure.liquid loading="eager" path="assets/img/blog/mol3dopt_coordinates_torsions.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Cartesian models diffuse all coordinates and must learn rigid-motion handling and local chemical geometry. Torsional models diffuse periodic dihedral angles, rotating fragments while preserving the supplied bond lengths and angles. Original diagram." %}

The restriction is both strength and limitation. Torsional generation is efficient for flexible organic molecules whose graph and local stereochemistry are known. It is less natural when rings change conformation, bonds form or break, coordination geometry varies, or local bond lengths respond strongly to environment. Cartesian methods are more general, but usually need stronger equivariant architectures, geometric regularization, or relaxation.

Latent 3D models add a second compromise. An encoder compresses graph and geometry into invariant and equivariant latent variables; diffusion or flow is performed in the smoother latent space; a decoder reconstructs the molecule. Sampling can be cheaper, but validity now depends on decoder coverage. A latent point that is numerically nearby need not decode to a chemically nearby structure.

## Diffusion and flow learn a prior before optimization begins

A diffusion model defines a noising path from molecular data toward a simple reference distribution, then learns the reverse score or denoising field. A flow model learns a time-dependent velocity that transports the reference distribution toward data. In either case, the unconditional model represents a prior over plausible molecules:

$$
p_\theta(x),
$$

where $$x$$ may be a conformer, a graph–coordinate pair, or a latent representation.

The prior matters during design. Direct optimization of a property predictor over unconstrained coordinates can create meaningless adversarial structures. Searching through a learned generator restricts proposals toward regions that resemble its training data. This is useful, but it is not a guarantee: a generator may reproduce dataset biases, miss rare but feasible chemistry, or assign probability to structures that pass representation-level validity while failing quantum relaxation.

Conditional generation changes the target to

$$
p_\theta(x\mid c),
$$

where $$c$$ can be a desired property, a protein pocket, a scaffold, a fragment arrangement, or a symmetry constraint. Conditioning may be trained directly when paired data are available. Protein-specific 3D generators, for example, condition ligand placement and chemistry on a target pocket rather than optimizing a ligand in isolation.

When an unconditional score $$s_\theta(x,t)$$ and a differentiable condition model $$p_\phi(c\mid x_t)$$ are available, guidance uses

$$
s_{\mathrm{guided}}(x_t,t)
=s_\theta(x_t,t)
+w\nabla_{x_t}\log p_\phi(c\mid x_t).
$$

The first term follows the learned molecular prior; the second bends the reverse process toward the condition. Classifier-free guidance learns conditional and unconditional predictions in one model and combines them without an external gradient.

{% include figure.liquid loading="eager" path="assets/img/blog/mol3dopt_guided_generation.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="An unconditional diffusion or flow model supplies a plausibility prior. Conditioning or guidance adds a direction associated with a property, pocket, or scaffold, bending the sampling path toward desired candidates while retaining some pressure toward the molecular data distribution. Original diagram." %}

Increasing $$w$$ usually improves the surrogate condition at the cost of diversity and distributional fidelity. More guidance is not universally better. If the condition model is least accurate off distribution, strong guidance drives sampling precisely where its gradients are least trustworthy.

## Goal-directed design is search under an imperfect oracle

Let $$R(x)$$ be a computational oracle: a property predictor, docking score, quantum calculation, synthesis score, or weighted combination. Goal-directed design seeks candidates with large reward under a limited number of oracle evaluations.

Several strategies fit this template. A genetic algorithm mutates and recombines molecular representations. Reinforcement learning treats construction as a sequence of actions. Latent optimization encodes a molecule as $$z$$, fits a surrogate $$\widehat R(z)$$, searches continuous latent space, and decodes promising points. The continuous molecular representation of Gómez-Bombarelli et al. established this influential encoder–predictor–decoder pattern (<span id="cite-gomez2018"></span>[Gómez-Bombarelli et al., 2018](#ref-gomez2018)).

Latent optimization succeeds only where three maps remain aligned:

$$
x\xrightarrow{\text{encode}}z,
\qquad
z\xrightarrow{\text{predict}}\widehat R,
\qquad
z\xrightarrow{\text{decode}}x'.
$$

A large predicted reward is useless if the decoder is invalid or discontinuous there. Retraining on high-scoring samples can move the latent model toward the search distribution, but can also narrow diversity and amplify oracle artifacts.

GFlowNets pursue a different goal: sample diverse objects with probability proportional to a positive reward rather than collapse onto one maximizer (<span id="cite-bengio2021"></span>[Bengio et al., 2021](#ref-bengio2021)):

$$
p(x)\propto R(x)^{1/T}.
$$

The temperature $$T$$ controls selectivity. This objective is attractive when many chemically distinct modes should be tested because the oracle is uncertain or batch experiments benefit from diversity. It still inherits the reward's misspecification and the construction space's limitations.

Oracle budget is a critical axis. An approach that finds a high score after one million predictor queries may be irrelevant when each query is a costly docking calculation or assay. The Practical Molecular Optimization benchmark shows that rankings can change substantially when algorithms share a fixed query budget (<span id="cite-gao2022"></span>[Gao et al., 2022](#ref-gao2022)). Reported results should therefore include the entire best-so-far curve, repeated seeds, and the number and cost of oracle evaluations—not only the best molecule found.

## Multi-objective design is not one scalar in disguise

Real candidates must balance potency, selectivity, solubility, permeability, metabolic stability, toxicity, novelty, and synthesis cost. Writing

$$
R(x)=\sum_{k=1}^K w_k R_k(x)
$$

is convenient, but the weights encode a policy decision and can hide tradeoffs. A small change in normalization can reorder every candidate. Hard constraints such as forbidden substructures or maximum toxicity are not always well represented by a soft penalty.

The Pareto view is more transparent. Candidate $$x$$ dominates $$y$$ if it is no worse on every objective and better on at least one. The nondominated set exposes the frontier of tradeoffs rather than pretending that one weighting is uniquely correct. In practice, one can combine hard feasibility filters, calibrated uncertainty bounds, and Pareto ranking, then let downstream experiments or domain experts choose among diverse frontier candidates.

Three-dimensional objectives also depend on conformer choice. A docking score for one pose is not a molecular property independent of geometry. Fair evaluation may require generating multiple conformers, accounting for protonation and tautomer states, relaxing the complex, and aggregating across poses. Optimizing the easiest pose to score can exploit the docking pipeline rather than improve binding.

## Oracle exploitation is the default, not an edge case

Optimization induces distribution shift. A predictor trained on ordinary molecules is queried on candidates selected specifically to maximize its output. Even an unbiased in-distribution predictor becomes optimistically biased after selecting the top of a large pool, because positive errors are preferentially retained.

If

$$
\widehat R(x)=R(x)+\epsilon(x),
$$

then maximizing $$\widehat R$$ selects both true reward and favorable error. The optimizer may exploit fingerprint shortcuts, docking clashes, unstable charge states, strained geometries, or regions where the uncertainty model is overconfident.

Several defenses work together:

- use ensembles or calibrated predictive distributions and penalize uncertainty;
- cap distance from the training domain or require similarity to validated chemistry;
- rescore finalists with an independent method that has different failure modes;
- adversarially audit repeated motifs and pathological geometries;
- reserve a hidden oracle or prospective evaluation that never guides optimization;
- optimize batches for diversity rather than near-duplicates around one apparent maximum.

Uncertainty is not a universal shield. Ensemble members sharing data and architecture can agree on the same wrong extrapolation. Domain checks, physical relaxation, and independent calculations remain necessary.

## Synthesizability belongs inside the search loop

Representation-level validity asks whether a graph satisfies formal valence rules. Synthesizability asks whether available reagents and reactions provide a plausible route under practical constraints. These are far apart. Gao and Coley found that high-scoring generative proposals can be difficult for a synthesis planner to realize (<span id="cite-gao2020"></span>[Gao & Coley, 2020](#ref-gao2020)).

A heuristic synthetic-accessibility score is useful for filtering, but it is only a proxy. Stronger approaches generate through known fragments or reaction templates, query a retrosynthesis planner during optimization, or construct a sequence of reactions whose terminal product is the candidate. Reaction-constrained generation narrows chemical space, yet every generated object comes with at least one proposed route. The remaining questions—yield, selectivity, reagent availability, protection chemistry, and scale—still require evaluation.

{% include figure.liquid loading="eager" path="assets/img/blog/mol3dopt_optimization_funnel.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="A realistic design loop applies independent gates: generate diverse structures, balance property and uncertainty objectives, establish a plausible synthesis route, and test compounds experimentally. Failed assays are information that should update both the oracle and the proposal distribution. Original diagram." %}

Synthesizability also interacts with diversity. A generator that repeatedly decorates one easy scaffold may score well on route success but offer little scientific breadth. Reports should distinguish graph diversity, scaffold diversity, three-dimensional shape diversity, and diversity among actually synthesizable candidates.

## Experimental validation closes the epistemic loop

The computational endpoint should be a batch of hypotheses, not a declaration of discovery. Final candidates need structural sanitization, charge and stereochemistry assignment, conformer generation, quantum or force-field relaxation, independent rescoring, retrosynthetic review, and experimental prioritization. Each stage should record attrition rather than silently replace failed candidates.

Prospective validation is the decisive test because it prevents information leakage from benchmark construction. Synthesize a preregistered or otherwise frozen batch, measure the relevant endpoints with appropriate controls and replicates, and report success rate as well as the best success. Negative results reveal where the property model, generator, synthesis model, or assay assumptions were wrong.

The next iteration should learn from all tested compounds, including failures. Active discovery is a sequential decision problem: update predictive uncertainties, revise objective thresholds, and choose a new diverse batch with high value of information. The most useful molecule is not always the one with the highest predicted score; it may be the one that most clearly distinguishes competing hypotheses while remaining feasible to make.

Three-dimensional molecular generation gives us a geometrically legal way to propose structures. Conditional diffusion, flows, latent search, reinforcement learning, and GFlowNets give us different mechanisms for moving toward goals. None removes the central difficulty: optimization magnifies every weakness in the oracle and every omission in the representation. A credible design system therefore treats symmetry, chemical validity, diversity, uncertainty, synthesis, and experiment as one connected chain of evidence.

---

## References

<ol class="bibliography">
  <li id="ref-hoogeboom2022">Hoogeboom, E., Garcia Satorras, V., Vignac, C., &amp; Welling, M. (2022). <a href="https://proceedings.mlr.press/v162/hoogeboom22a.html">Equivariant diffusion for molecule generation in 3D</a>. <em>Proceedings of the 39th International Conference on Machine Learning</em>, 8867–8887. <a href="#cite-hoogeboom2022">↩</a></li>
  <li id="ref-jing2022">Jing, B., Corso, G., Chang, J., Barzilay, R., &amp; Jaakkola, T. S. (2022). <a href="https://openreview.net/forum?id=w6fj2r62r_H">Torsional diffusion for molecular conformer generation</a>. <em>Advances in Neural Information Processing Systems</em>, 35. <a href="#cite-jing2022">↩</a></li>
  <li id="ref-gomez2018">Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., &amp; Aspuru-Guzik, A. (2018). <a href="https://doi.org/10.1021/acscentsci.7b00572">Automatic chemical design using a data-driven continuous representation of molecules</a>. <em>ACS Central Science</em>, 4(2), 268–276. <a href="#cite-gomez2018">↩</a></li>
  <li id="ref-bengio2021">Bengio, E., Jain, M., Korablyov, M., Precup, D., &amp; Bengio, Y. (2021). <a href="https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html">Flow network based generative models for non-iterative diverse candidate generation</a>. <em>Advances in Neural Information Processing Systems</em>, 34. <a href="#cite-bengio2021">↩</a></li>
  <li id="ref-gao2022">Gao, W., Fu, T., Sun, J., &amp; Coley, C. W. (2022). <a href="https://proceedings.neurips.cc/paper_files/paper/2022/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html">Sample efficiency matters: A benchmark for practical molecular optimization</a>. <em>Advances in Neural Information Processing Systems</em>, 35. <a href="#cite-gao2022">↩</a></li>
  <li id="ref-gao2020">Gao, W., &amp; Coley, C. W. (2020). <a href="https://doi.org/10.1021/acs.jcim.0c00174">The synthesizability of molecules proposed by generative models</a>. <em>Journal of Chemical Information and Modeling</em>, 60(12), 5714–5723. <a href="#cite-gao2020">↩</a></li>
</ol>

---

*Figure provenance.* All four `mol3dopt_` diagrams are original SVG illustrations generated by `scripts/generate_mol3dopt_figures.py`. They synthesize standard geometric-generation identities and optimization safeguards described in the cited primary literature; no third-party artwork is reproduced.
