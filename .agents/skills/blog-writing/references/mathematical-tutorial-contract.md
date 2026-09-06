# Mathematical tutorial contract

Read for mathematical tutorials and substantial derivation revisions.
Adapt the assumed background to the request. The usual audience is an ML
researcher with probability and multivariate calculus; introduce specialized
measure theory, physics, geometry, or control concepts when needed.

## Argument and rigor

Organize around the mathematical question. Use a small construction or worked
example when it exposes the mechanism; a finite-to-continuous development is
useful when the limit is the conceptual difficulty, not a required outline.

Prepare the reader before a theorem or definition. Show the step where an
assumption, normalization, conditioning operation, cancellation, or approximation
does the work. Keep notation stable and state dimensions when products could be
ambiguous. Interpret the result without repeating every equation in prose.

Distinguish exact identities, modeling assumptions, approximations, and learned
surrogates. Label discretize-and-limit arguments heuristic unless regularity and
convergence are justified. Check consequential signs, constants, index
conventions, and limiting cases independently.

For competing derivations, identify the assumptions under which they agree.
For impossibility or expressivity claims, state the quantifiers and give an
appropriate witness. For scientific pipelines, explain the predicted object,
the downstream computation, and which guarantees survive that handoff.
Use these checks when the topic requires them; do not add unrelated branches.

## Stochastic-process tutorials

When comparing processes, identify the probability law, conditioning variables,
and the level of equality: endpoints, marginals, transitions, or path laws.
Equal marginals do not establish an equal coupling or dynamics.

Distinguish a measure from its density and the reference measure used. Do not
assign an ordinary probability mass to one exact continuous path. Fixed-time
overlap does not establish absolute continuity of full path laws.

Explain the scaling that makes a continuous limit finite and name the law under
which a stochastic integral has zero expectation. A scalar or constant-diffusion
case often suffices to expose the mechanism; introduce singular covariances and
other extensions only when the central argument needs them.

## Revision and figures

Preserve the source's argument and the post's intended scope. Use the house
references to assess explanatory depth during substantial revisions. A running
example should make the abstraction usable, not introduce an unsupported result.

Use figures for a geometric transformation, conservation law, comparison, or
other step that is difficult to see in algebra. Choose their count by purpose.
Verify derivations and cross-references, then inspect substantial rendered
changes at desktop and narrow widths. Revisit specific unresolved gaps rather
than repeating a fixed number of review passes.
