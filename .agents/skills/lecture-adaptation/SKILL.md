---
name: lecture-adaptation
description: Adapt rights-cleared lecture decks into source-faithful technical blog posts. Use when creating or revising a post whose authoritative source is a lecture PDF or PPTX, when slide-coverage manifests are required, or when extracting and attributing lecture figures.
---

# Lecture Adaptation

Turn a lecture deck into a coherent technical article without changing its
scientific argument. Apply `../blog-writing/SKILL.md` for prose,
`../academic-writing/SKILL.md` for derivations, and the figure and Jekyll
skills for visual and rendering details.

## Preconditions

- Confirm that the author has the right to republish the deck's embedded
  figures. If reuse authority is unclear, pause direct extraction and use the
  license-checking workflow in `../download-paper-figures/SKILL.md`.
- Treat the PPTX as the visual source of truth and the lecture deck as the
  content source of truth.
- Maintain the durable deck manifest used by `scripts/validate_blog.py`.

## Source-Faithful Workflow

When the author asks for a post to follow an existing lecture deck, the deck is
the content authority. The house references set the standard for prose,
captions, equation-to-explanation rhythm, and rendering; they do not authorize a
new thesis, a reordered curriculum, or additional technical branches.

1. Inventory every slide before outlining the post. Mark each slide as
   substantive, logistical, transition-only, or repeated recap.
2. Preserve every substantive concept, equation, example, method, caveat,
   historical item, and comparison in the deck's scientific order. Collapse
   only literal repetition.
3. Derive H2 and H3 order from the deck's section and topic sequence. Do not
   preserve a previous synthetic post's headings merely because they are
   polished.
4. Reuse existing prose only when it can be traced to the deck. Delete running
   examples, calculations, claims, and conclusions introduced by an earlier
   draft when they are absent from the source.
5. Add only the connective prose and local definitions needed to turn slide
   fragments into a readable article. Do not add content to meet a word-count,
   figure-count, or derivation-depth target.
6. Maintain an agent-facing slide coverage record. Every substantive slide must
   map to prose, an equation, a table, or a figure before the post is complete.
7. Preserve deliberate cross-domain juxtapositions. Do not split one deck into
   separately optimized essays when its sequence uses the domains to expose a
   shared discovery, measurement, or modeling pattern.
8. Preserve the deck's rhetorical destination as well as its technical topics.
   If a historical survey ends in a methodological critique, open question, or
   author judgment, do not stop the article at the last architecture or equation.
   That change of register is part of the source argument. In particular, a
   closing audit of real-world relevance, graph or data semantics, benchmark
   culture, or foundation-model evidence sets the claim boundary for the methods
   that precede it and remains scientific content even without new equations.
9. Distinguish literal repetition from deliberate structural analogy. When a
   deck re-derives the same construction in Euclidean, manifold, and discrete
   settings, retain each version and make the shared roles explicit. The repeated
   pattern is the lesson; it is not a recap to collapse.
10. Treat a source-ordered dataset or benchmark survey as substantive when it
    establishes changing scientific contracts. If the sequence moves from
    scalar labels to forces, spectra, periodic structures, trajectories, or
    other distinct inputs and outputs, retain those transitions and explain
    what each dataset makes observable. Attach later claims about universality,
    scaling, discovery, or physical simulation to the matching data and target
    regime. Do not compress the survey into a short preamble merely because it
    contains fewer derivations than later method slides.
11. Preserve a deck's organizing axis when it classifies methods by the object
    they alter. Spatial reduction, time coarse-graining, biased exploration,
    learned dynamics, and postprocessing may all be called acceleration, but
    they preserve different distributions and observables. Keep the source
    categories explicit instead of rewriting them as a flat method catalog.
12. When the deck treats a representation as a construction grammar, explain
    the actions, constraints, and ambiguities induced by that grammar before
    moving to model comparisons. Matrix entries, motifs, strings, edits, and
    mechanistic paths can encode the same endpoint while defining different
    learning problems; do not relegate this progression to notation or merge it
    into one generic representation paragraph.
13. Preserve a deck's hierarchy of learned scientific objects and downstream
    computations. A state, functional, operator, field, and basis expansion may
    describe related physics while defining different supervision, symmetry,
    inference, and solver contracts. When the source moves between these
    objects, state what is predicted and what calculation still follows; do not
    flatten the sequence into a catalog of machine-learning architectures.
14. Retain an opening map of adjacent tasks when later methods cross its
    boundaries. Sequence encoding, structure prediction, ensemble generation,
    inverse design, property prediction, and simulation may initially appear as
    orientation, yet later architectures can change from one target to another.
    Use the map to make those shifts legible instead of deleting it as generic
    background and presenting the later models as a single task lineage.
15. Preserve a source-defined design pipeline when different models occupy
    successive stages. Generation, inverse prediction, forward validation,
    affinity or property scoring, filtering, synthesis, and experiment are not
    interchangeable benchmarks. Name the object passed at each boundary and
    retain the source's reported experimental endpoint; do not extract only the
    generative architecture and discard the selection or validation chain.
16. When a deck introduces measurement modalities before a predictive model,
    preserve that observation layer. Sequence, bulk counts, single-cell counts,
    spatial expression, images, and experimental phenotypes expose different
    projections of the underlying system. Explain which measurement defines
    the model's input and target before using broad language such as simulation,
    response, state, or digital twin.
17. For a course-introduction deck, separate logistics from scientific
    orientation slide by slide. Remove grading, schedule, staff, and project
    administration, but retain the motivating applications, domain inventory,
    definition of the field, and architectural question when they form a
    coherent scientific argument. “Introduction” is not itself a reason to omit
    a slide.
18. Preserve a source's formal dependency chain. When a sequence such as group,
    action, stabilizer, coset, quotient, representation, and equivariance is
    cumulative, each definition is substantive because it supplies an object
    used by the next construction. Explain that role in the surrounding prose;
    do not compress the sequence into a glossary or retain only its endpoint.
19. Preserve the argument encoded by a chronological method survey. A sequence
    of architectures can show successive changes in locality, pooling,
    expressivity, depth, long-range communication, evaluation, and scale rather
    than merely list papers by year. Retain the source's comparisons and author
    judgments, and connect each method to the limitation or question it changes.
20. When a deck derives one operator through two different principles, preserve
    both routes and the source's pivot between them. State the object held fixed,
    the assumptions and limitations of each route, and why the second viewpoint
    is introduced. Do not merge the formulas into one polished derivation or
    omit the critique that makes their comparison meaningful.
21. Do not classify a slide as disposable merely because it is titled “recap.”
    A recap is substantive when it compares guarantees, costs, or representation
    families and uses that comparison to motivate the next part. Collapse only
    literal repetition; retain synthesis slides that change the organizing axis
    or pose the next scientific question. When a new method is defined as a
    modification of the recapped baseline, retain enough of the baseline's
    notation and layer structure to state that architectural delta exactly.
22. Preserve a source's progression of representational sufficiency. When
    successive methods add distances, angles, torsions, local frames, or vector
    channels, explain which geometric ambiguity each construction resolves and
    retain the source's accuracy, force, and efficiency comparisons. Do not
    compress the sequence into a binary scalar-versus-vector taxonomy.
23. Track the scope and index set of a construction when the source reuses one
    name at several scales. An edge-local frame, an atom-wise frame, and a
    global input-dependent frame solve different transport or symmetrization
    problems. State what selects each object, what it acts on, and where it is
    averaged or reconstructed; shared terminology does not make the roles
    interchangeable.
24. Do not detach a representation-theory prelude from the architecture it
    enables. Carry group, action, representation, irrep type, basis function,
    tensor-product coefficient, and selection rule into the model's actual
    feature indices, filters, nonlinearities, and outputs. The abstract sequence
    is substantive because it determines which learned operations are legal.
25. Preserve a method survey's bottleneck--remedy pairs. Adaptive neighbor
    selection, explicit body order, sparse tensor products, richer activations,
    and cross-type normalization improve different axes even when their
    benchmark tables look comparable. Name the exact operation changed, retain
    its cost or stability motivation, and attach the reported evidence to that
    change rather than to a generic claim of architectural progress.
26. Preserve the source's proof strategy and the objects it moves between.
    When a derivation proceeds from path increments to observables, conditional
    expectations, generators, adjoints, and densities, retain those transitions
    and the assumptions used at each one. Do not replace the lecture's proof
    with a shorter familiar argument that reaches the same final equation but
    erases why the intermediate objects were introduced.
27. Keep conditional and marginal constructions distinct throughout a
    generative-model derivation. Name the conditional path, marginal path,
    conditional field, marginal field, conditioning posterior, and endpoint
    convention whenever the source uses them. Preserve the expectation or
    gradient identity that connects the levels; shared notation or Gaussian
    examples do not make the conditional and marginal objects interchangeable.
28. When a source generalizes an algorithm to a new geometry, preserve its
    replacement dictionary and operational sequence. State which Euclidean
    primitive becomes a tangent-space, metric, geodesic, exponential-map,
    logarithmic-map, transport, gradient, or divergence operation, and retain
    the intrinsic-coordinate or ambient-embedding choice that makes each
    operation implementable. Do not present a manifold algorithm as the same
    formula with renamed symbols.
29. When a source presents continuous and discrete versions of one generative
    construction, preserve the analogy between their mathematical objects and
    equations. Map vector fields to jump-rate vectors, continuity equations to
    Kolmogorov equations, and conditional-to-marginal posterior expectations to
    their discrete counterparts. Introduce coordinate factorization at the
    source's computational bottleneck rather than treating the two versions as
    unrelated method summaries.
30. Keep an infinitesimal generator distinct from the finite-dimensional
    coefficient used to parameterize it. Define the generator by its action on
    test functions, derive how flow, diffusion, or jump coefficients enter that
    action, and preserve the conditional-to-marginal identity at both the
    operator and coefficient levels. Do not rename every generator coefficient
    a velocity or omit the operator family that makes the learning target
    meaningful.
31. When a source surveys one scientific domain across representation learning,
    property or structure prediction, dynamics, and design, preserve the input,
    output, conditioning information, and supervision of each task family.
    Reused encoders, geometric modules, or generative objectives do not make the
    tasks interchangeable. Keep the source's task map visible as the method
    survey crosses its boundaries.
32. When a source organizes generative methods by the object sampled, retain
    that sample unit and its time semantics. Independent equilibrium states,
    lag-conditioned transitions, complete trajectories, discrete sequences,
    geometric backbones, and joint sequence--structure states are different
    targets even when each uses diffusion or flow matching. State what is
    conditioned, what one sample contains, and whether temporal order is part
    of the learned law.
33. Preserve the changing axes in a foundation-model chronology. For every
    consequential step, retain the tokenization, context length, training
    corpus and species coverage, measurement modalities, learning objective,
    and evaluated task that the source uses to distinguish it. Model names and
    parameter counts alone do not explain what was scaled or what new claim the
    model supports.
34. Preserve application sections as model-to-observable handoffs rather than
    shortening them into a list of domains. State which generated or predicted
    object enters which downstream calculation, relaxation, simulation, or
    experiment, and which physical observable is finally evaluated. The
    application often defines the scientific purpose and limitations of the
    upstream model output.
35. Do not publish a whole-slide screenshot as a blog figure. Transcribe slide
    prose as prose, equations as MathJax, and tables as Markdown or HTML. Keep
    only the visual object that carries information which text cannot replace:
    a plot, scientific image, architecture, photograph, or semantic diagram.
    Extract that object from the source PPTX rather than cropping the PDF or a
    rendered slide. When several native PowerPoint shapes form one comparison,
    export the deliberate semantic group; do not preserve the title, footer,
    or surrounding slide canvas. Verify every retained image against the actual
    PPTX object tree and record its slide, source media or shape, extraction
    method, and role in the lecture manifest.

In this mode, source fidelity overrides the general depth heuristics below. A
short deck may produce a shorter article; a long deck may exceed a 45-minute
read. Completeness means faithful coverage, not convergence toward a standard
tutorial length.

## Completion Workflow

1. Inventory and classify every slide before outlining.
2. Map every substantive slide to prose, native math/table content, or a figure.
3. Draft in the deck's scientific order and remove unsupported legacy material.
4. Extract visual objects from the PPTX; never publish slide or PDF-region crops.
5. Record slide, source object, extraction method, role, and reuse status in the
   lecture manifest.
6. Run `python3 scripts/validate_blog.py` and resolve manifest/post mismatches.
7. Inspect the rendered article for caption clarity, visual legibility, and
   reader-facing coherence.

Keep historical, post-specific review notes under `docs/agent-audits/`, not in
the skill directory or `.agents` root.

## Reader-Facing Notes

- Keep the opening note to one or two reader-facing sentences.
- Name the lecture source briefly when provenance is useful: “Adapted from my
  2025 Geometric Deep Learning lectures.”
- Use the remaining sentence to state the article's central question or point.
- Do not describe slide reuse, lecture-storyline production, chapter ownership,
  validation work, or the writing process. Those are editorial facts, not reasons
  for a reader to continue.
