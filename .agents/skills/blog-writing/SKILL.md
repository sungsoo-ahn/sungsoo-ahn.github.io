---
name: blog-writing
description: Write and edit blog posts with a direct, opinionated style — closer to a conference talk than a textbook. Use when drafting or revising blog posts.
---

# Blog Writing Style

Rules for writing and editing blog posts. Blog prose is direct and opinionated — closer to a conference talk than a textbook.

## Prose Task Mode

For blog-writing tasks, suppress the normal coding-agent response style. The
post is the output, not a report about the work.

Do not:

- report what you inspected, changed, or verified inside the post;
- organize the prose around tasks completed;
- convert the argument into documentation;
- optimize for exhaustive coverage;
- expose planning, validation, or implementation steps;
- add headings only to make the material scannable;
- conclude every section with a summary.

Write as the author speaking to the reader. Keep only the context, examples,
equations, figures, and caveats that help the intended reader understand the
argument. Do not use a word-count target as a definition of completeness. Stop
when the post has answered its central question at the depth its reader needs.

The quality checklist below diagnoses the argument that belongs in the post. It
is not a content checklist, and it must not cause new sections or background to
be added solely for coverage. When the requested deliverable is prose, return
the prose without appending a coding-agent completion report unless the user
explicitly asks for one.

## Prose Style

- **Lead with the point, then justify** — "The BO approximation separates electrons from nuclei" before "because nuclei are 1836x heavier." Don't make readers wait through setup to learn the result.
- **Cut throat-clearing openers** — delete filler like "The equation says:", "What does X look like?", "The fundamental challenge is clear:", "The methods described above form the backbone of...". Just state the content.
- **Don't restate what was just shown** — if the math already demonstrated a property, don't add a sentence restating it in words.
- **Drop dramatic qualifiers** — "fundamental physical flaw", "radically different approach", "enormous complexity" → just state the facts. Let the reader judge significance.
- **Avoid self-aware scaffolding sentences** — delete sentences like "The boundary of this claim matters", "This distinction is important", or "This raises an interesting question". Replace them with the concrete claim, comparison, or evidence.
- **Merge redundant statements** — if two consecutive sentences say the same thing in different words, combine into one.
- **Break up stacked parentheticals** — a sentence with three em-dash clauses should be split into separate sentences.

## Clarity

- One idea per sentence, one topic per paragraph
- Cut filler words: "very", "quite", "somewhat", "really", "basically"
- Eliminate redundancy: "completely eliminate" → "eliminate", "first introduce" → "introduce"
- Quantify claims: "improves significantly" → "improves by 5%"
- Avoid passive voice when the actor matters: "The model was trained" → "We trained the model"
- Fix undefined pronouns: "We applied it. It improved." → "We applied Algorithm 1. Accuracy improved by 5%."
- Follow every equation with an explanation of its terms

## Structure and Readability

- **No wall-of-text paragraphs.** If a paragraph exceeds ~8 lines or contains multiple distinct ideas, split it.
- **Sentences under ~40 words.** Split chains of em-dashes or nested parentheticals into separate sentences.
- **Don't bury key points.** Important results belong at the start of a paragraph or in a boxed definition, not mid-paragraph.
- **Signpost transitions.** Before a derivation or definition, add a setup sentence explaining what's coming and why.
- **Don't repeat yourself.** If a point (e.g., "the variance problem") appears in multiple places, state it fully once and back-reference elsewhere.
- **Figures near their discussion.** Place figures immediately after the text that introduces them, not paragraphs later.
- **Notation introductions need breathing room.** Don't introduce 3+ new symbols in one paragraph with no prose between equations.

## Comparative Tutorial Revision

For long technical tutorials, compare the draft against the site's named house
references before revising it. Review the posts one at a time so that lessons
from one comparison can improve the criteria used for the next.

1. State the draft's central question and section-level argument in one short
   outline. If the outline is only a list of topics, the post is still a survey.
2. Compare the draft with the reference posts on argument, derivation depth,
   examples, equation-to-prose rhythm, figures, evidence, and voice.
3. Promote only topic-independent lessons into this skill. Keep post-specific
   findings in an agent-facing audit log.
4. Revise with the updated criteria, then repeat the comparison until no major
   explanatory gap remains.

### Source-Faithful Lecture Adaptation

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
   That change of register is part of the source argument.
9. Distinguish literal repetition from deliberate structural analogy. When a
   deck re-derives the same construction in Euclidean, manifold, and discrete
   settings, retain each version and make the shared roles explicit. The repeated
   pattern is the lesson; it is not a recap to collapse.
10. Treat a source-ordered dataset or benchmark survey as substantive when it
    establishes changing scientific contracts. If the sequence moves from
    scalar labels to forces, spectra, periodic structures, trajectories, or
    other distinct inputs and outputs, retain those transitions and explain
    what each dataset makes observable. Do not compress the survey into a short
    preamble merely because it contains fewer derivations than later method
    slides.
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

In this mode, source fidelity overrides the general depth heuristics below. A
short deck may produce a shorter article; a long deck may exceed a 45-minute
read. Completeness means faithful coverage, not convergence toward a standard
tutorial length.

### Depth Without Padding

- A 30--45 minute technical tutorial usually needs 4,500--6,500 substantive
  words, adjusted for mathematical density. Treat this range as a diagnostic,
  not a reason to add background or repeated summaries. Do not apply this range
  to a source-faithful lecture adaptation when the deck determines the scope.
- Preserve a coherent H2 storyline. Add H3 subsections when a derivation,
  worked example, or case study needs room; do not add headings merely to make
  a post longer.
- Replace compressed surveys with explanatory sequences: motivate the object,
  define it, derive the central relation, work through a concrete case, and
  state the approximation or failure boundary.
- A displayed equation is not depth by itself. Show the non-obvious steps and
  use at least one limiting case, numerical calculation, or toy construction to
  make the result operational.
- Preserve the lecture's scientific order when requested, but make the links
  between sections explicit so the post reads as one argument.
- In an application-domain tutorial, use a running scientific scenario when
  several stages act on the same object. Carrying one molecule, material, or
  experiment through the chain makes assumptions and lost information visible;
  unrelated examples can leave a long post feeling like an abstract survey.
- Whenever a model score changes a scientific workflow, identify the decision,
  the population being filtered, and the denominator behind the reported
  success rate. A benchmark metric alone does not complete the argument.
- When an optimizer selects candidates by a noisy learned score, quantify
  selection-induced optimism. Carry one finite pool through oracle scoring,
  maximization or ranking, and independent evaluation; report the query budget
  and best-so-far curve rather than only the winner. Random-test accuracy does
  not establish accuracy on the adaptively selected tail.
- When a tutorial explains confidence or uncertainty scores, define the random
  variable being predicted, the conditioning or alignment convention, the
  spatial or statistical granularity, the calibration population, and the
  decision the score supports. Construct a case where two legitimate confidence
  summaries disagree. “High confidence” is incomplete without its scope.
- When constraints are imposed by action masks, projection, repair, rejection,
  canonicalization, or downstream filters, distinguish the raw model
  distribution, the transformed or accepted distribution, and the physical
  target distribution. Work a finite sample through the transformation and
  report every denominator, collision, and unreachable mode. Postprocessing is
  part of the generative method, not a metric-neutral cleanup step.
- When a tutorial compares dataset splits or generalization claims, define the
  independent unit, the equivalence relation that makes records dependent or
  near-duplicate, and the deployment population before naming a split. Carry
  one small family of related records through row-wise and group-, structure-,
  or time-aware partitions, then state which estimand changes and where leakage
  remains. A split label such as “random” or “scaffold” is not a generalization
  argument by itself.
- When comparing related architectures, carry one controlled input through the
  alternatives and compute the intermediate outputs. Hold the task and data
  fixed so the reader can see exactly which normalization, aggregation,
  parameterization, or information loss caused the difference; separate toy
  examples for each model often reproduce a catalog rather than an argument.
- When a tutorial distinguishes failure modes with similar symptoms, organize
  the explanation as a differential diagnosis. Give each mechanism an
  observable diagnostic and a matched intervention, then include at least one
  counterfactual where a remedy for one mechanism leaves another unchanged or
  makes it worse. A remedy list without these contrasts encourages readers to
  treat distinct causes as synonyms.
- When comparing representations, separate information-theoretic completeness
  from computational accessibility. State whether the encoded variables
  determine the desired quantity in principle, then show how many interactions,
  message-passing steps, or reconstruction operations are needed to expose it
  to the predictor. “The information is present” does not mean it is locally or
  efficiently available to the architecture being discussed.
- When coordinates contain an arbitrary basis, gauge, or unit-cell choice,
  name the full equivalence action and carry one nontrivial change of basis
  through every dependent object. Transform coordinates, neighborhoods,
  features, densities, and tensor outputs together, then verify the claimed
  invariant or covariant quantity numerically. Periodic wrapping or rotation
  invariance alone does not establish independence from the stored basis.
- When claiming that a representation “captures” a concept, separate
  decodability, accessibility to a capacity-controlled probe, use by a
  fine-tuned predictor, and stability under a matched intervention. Compare
  against nuisance baselines and construct close counterfactuals where family,
  metadata, or global similarity stays fixed while the target mechanism changes.
  A cluster plot or high probe score alone does not establish mechanism.
- When a mature companion post already covers nearby material, give the new post
  a different organizing question, derivation, or running example. Cross-link
  prerequisite machinery instead of re-teaching it, and audit the pair for
  duplicated exposition; a series should deepen by composition, not paraphrase.
  Describe this relationship in reader-facing terms ("For the PDE derivation,
  see...") rather than project-management language such as "division of labor"
  or "this chapter owns...".
- When a post connects an ML architecture to a downstream scientific workflow,
  expose the interface contract between them. State exactly what mathematical
  object the architecture supplies, what the downstream calculation assumes,
  which guarantees cross the boundary, and which properties still depend on
  data coverage, numerical choices, or physical approximations. Carry one case
  across the boundary; do not let an architectural symmetry claim silently turn
  into a claim of simulation, experimental, or decision validity.
- When one learned model validates or filters another, audit the independence of
  evidence. Name shared training data, architectures, representations, and
  physical assumptions, then construct a correlated-error case where both
  models agree but an orthogonal check fails. Internal self-consistency,
  independent computational evidence, and experimental evidence support
  different claims and should not be counted as interchangeable votes.
- When a model predicts an intermediate object consumed by a solver, propagate
  a controlled prediction perturbation through the downstream derivative,
  diagonalization, fixed point, or estimator. Quantify conditioning and compare
  the final observable or convergence behavior. Small entrywise or pointwise
  error does not establish correctness after an ill-conditioned computation.
- When model outputs are fed back through an iterative numerical process,
  separate model or surface error, discretization error, transient or mixing
  error, and estimator variance. Carry one solvable system through changes in
  model parameters, step size, and trajectory length, and show which diagnostic
  responds to each change. Static accuracy, bounded trajectories, converged
  integration, and a correct observable are distinct claims.
- When comparing stochastic processes or generative transports, name the level
  at which two objects agree: sample paths, conditional transition kernels,
  finite-dimensional joint laws, one-time marginals, or endpoints. Demonstrate
  the claimed equality on one controlled process and give a witness for a level
  that does not agree. Similar samples at each time do not imply the same
  coupling, path measure, likelihood, or dynamics.
- When measurements are destructive or unpaired, distinguish a change in
  population marginals from trajectories of individual entities. Construct two
  couplings with the same observed before/after marginals but different
  transitions, then name the lineage, longitudinal, randomization, or modeling
  assumptions needed to choose between them. Cross-sectional agreement does not
  identify dynamics or individual counterfactual response.
- When calling learning targets, objectives, or parameterizations equivalent,
  state the exact equivalence: invertible target information, a shared
  population minimizer, identical gradients up to a parameter-independent
  constant, or identical weighted losses. Derive the conversion and transform
  the time-dependent loss weight with it. Then identify what finite capacity,
  sampling, conditioning, optimization, or numerical integration can still make
  different; algebraic convertibility does not imply equal training behavior.
- When a tutorial gives multiple derivations of one method, keep the output and
  controlled example fixed across the routes. Maintain an assumption ledger,
  prove the exact overlap, and mark the approximation, restriction, or added
  input where the formulas diverge. Similar-looking final updates do not make
  their premises, guarantees, transfer behavior, or failure modes equivalent.
- When symmetry, conservation, or consistency is obtained through randomized
  averaging, separate per-sample, coupled-sample, distributional, and
  expectation-level guarantees. Work one finite estimator, compute its residual
  and variance as sample count changes, and state whether transformed inputs
  share randomness. An exact expectation does not make one independent Monte
  Carlo evaluation exactly equivariant or conservative.
- When a learned or hand-built representation is used as the state of a
  dynamical model, audit Markov sufficiency rather than geometric compactness or
  reconstruction alone. Merge a finite pair of microstates with different
  outgoing transition laws, compute how the coarse transition depends on the
  hidden mixture or entry history, and show which state refinement, lag change,
  or memory variable repairs the claim. A visually coherent cluster need not be
  a valid dynamical state.

## Quality Checklist (apply before finalizing)

### Substance

- Every paragraph must contain a concrete claim, result, argument, or necessary explanation.
- If the opening sentence can be deleted without losing information, delete or rewrite it.
- Replace vague praise ("important", "powerful", "compelling", "robust") with the specific mechanism, result, or consequence.
- Support novelty, causality, and superiority claims with evidence, numbers, comparisons, or citations.

### Relevance and Precision

- Every sentence should advance the paragraph's purpose.
- Remove background the intended reader already knows.
- Tie abstract nouns to specific objects, mechanisms, experiments, or numbers.
- Check pronouns such as "this", "it", and "these results" for clear antecedents.

### Formulaic Rhetoric

Flag these patterns; do not ban them when they carry real technical contrast:

- "It is important to note that ..."
- "This highlights/underscores/demonstrates ..."
- "Not only X, but also Y."
- "X is not merely A; it is B."
- "In today's rapidly evolving landscape ..."
- "Taken together", "Overall", or "In summary"
- A final sentence that merely restates the paragraph.
- A short dramatic sentence that tells the reader what to feel: "This matters." "The implication is clear."

### Structure and Rhythm

- Avoid making every paragraph follow the same claim -> explanation -> summary template.
- Watch for sections that are suspiciously equal in length or structure.
- Vary sentence length and syntax when several consecutive sentences sound alike.
- Use transitions only when the logical relation requires them.
- End each paragraph where the argument ends, not with a manufactured conclusion.

### Language

- Prefer direct verbs over noun phrases: "evaluate" instead of "conduct an evaluation of".
- Cut 10-20% of words when meaning survives.
- Use passive voice only when the actor is unknown, irrelevant, or intentionally de-emphasized.
- Remove redundant adjective pairs such as "novel and innovative" or "clear and evident".
- Keep technical terms only when they are necessary and precise.

### Voice

- Ask whether the passage sounds like something the author would actually say or write.
- Preserve legitimate uncertainty instead of converting everything into confident declarative prose.
- Include the author's actual judgment, not only a polished synthesis of conventional observations.
- Compare the passage against two or three known non-AI passages by the author when voice is uncertain.

### Lecture Adaptation Notes

- Keep the opening note to one or two reader-facing sentences.
- Name the lecture source briefly when provenance is useful: “Adapted from my
  2025 Geometric Deep Learning lectures.”
- Use the remaining sentence to state the article's central question or point.
- Do not describe slide reuse, lecture-storyline production, chapter ownership,
  validation work, or the writing process. Those are editorial facts, not reasons
  for a reader to continue.

## Coherence Checklist (apply when reviewing)

- Every forward reference ("Part 3 defines this") — verify the target actually contains what you claim
- Every backward reference ("from Part 2") — verify the source
- Every cited paper — verify it appears in the References section
- Every figure file referenced — verify it exists in `assets/img/blog/`
- Notation consistent throughout: same symbol = same meaning, same formatting convention
