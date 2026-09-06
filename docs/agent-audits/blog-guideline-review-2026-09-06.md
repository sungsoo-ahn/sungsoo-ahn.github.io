# Blog guideline review — 2026-09-06

This is a record of a completed editorial task, not an instruction file.

## Scope

Reviewed all 12 tracked posts and their 83 originally referenced image assets.
Read the prose and inspected figure contact sheets, including representative
frames of animations. Rewrote passages with unsupported generalizations,
repeated summaries, unclear assumptions, or mismatched figure descriptions.
Preserved article identities, authors, selection flags, and editorial-status
labels; set `last_updated` to 2026-09-06. No post was promoted to human-reviewed.

The three untracked August 30 drafts, `.agents/asbs-blog/`, `temp_prompt.md`,
and the pre-existing Sass changes were left untouched. Lecture source registries
and durable migration manifests were preserved. This was not a new deck
extraction or a fresh slide-coverage audit.

The revised blog-writing guidance informed the claim-specific prose edits;
blog-figures informed editable exports, label spacing, accessibility, and
explicit separation of illustrative values from measurements. Site-validation
guided the rendered mobile, equation, and dark-theme checks.

## Substantive revisions

| Post | Main changes |
| --- | --- |
| Spherical equivariant layers | Distinguished feature-vector basis changes from conjugating representation matrices; qualified harmonic rotation and scalar-force claims. |
| Quantum chemistry / DFT | Distinguished coordinate dimensionality from representation cost; corrected observable expectations and qualified Kohn–Sham, functional, and numerical-error claims. |
| Fokker–Planck | Labeled Euler–Maruyama transitions as approximations; added regularity/integrability caveats and normalized the smoothing illustration. |
| Electrocatalysis | Dated historical estimates, narrowed catalyst claims, marked schematic numbers, and distinguished intermediate-state energies from activation barriers. |
| Protein design | Separated computational consistency from experimental success; corrected contig syntax and confidence-metric claims; made the example funnel internally consistent and explicitly hypothetical. |
| Ensembles and thermostats | Restored variable-particle counting factors, corrected chemical-potential interpretation and GCMC deletion acceptance, and qualified thermostat/barostat sampling claims. |
| GFlowNets | Made log rewards versus positive weights explicit; corrected the shared-terminal example and matched every displayed path flow and probability. |
| Path measures | Corrected Girsanov's drift-square terms, AIS ratio orientation, backward-time discretization, and finite-step/mixing claims; replaced the ordinary Brownian-path-density argument with a finite-grid derivation. |
| Adsorption / classical DFT | Distinguished instantaneous particle count from uptake; explained variable-N counting and rejection-state sampling; required more than a small fixed-point update for convergence. |
| Enhanced sampling | Qualified fixed-bias reweighting, distinguished exponential-work estimates from log free-energy estimates, and replaced repeated conclusions with concrete checks. |
| MaskGXT | Limited conclusions to the reported benchmarks, preserved numerical results, acknowledged the MPTS-52 RMSE comparison, and separated conceptual illustration from evidence. |
| MADField | Specified inference-only speedups, average-precision and uptake-error metrics, pressure-conditioned calls, distribution-shift limits, and the studied simulation regime. |

## Figures and rendering

- Added explicit descriptive alt text to all 80 original figure includes. The
  revised pages contain 87 figure includes because one benchmark image is now
  eight panels.
- Reworked the CG decomposition, Gaussian smoothing, cDFT iteration,
  self-consistency, and protein-design funnel diagrams with fewer crossings,
  clearer labels, and figure audits.
- Redrew three GFlowNet worked examples as editable SVGs with PNG previews.
  Removed their three superseded PNGs from the active asset tree:
  `gflownet/fig_example_forward.png`, `fig_example_backward.png`, and
  `fig_flow_matching.png`. They remain recoverable from Git history.
- Corrected electrocatalysis labels, the water-return arrow, proton/electron
  bookkeeping, legend overlap, and the misleading ideal-path comparison.
- Exported eight independently labeled MaskGXT benchmark panels, stacked on
  phones and paired on wider screens. Kept a linked combined-chart export.
  The benchmark data arrays were not changed.
- Used opaque backgrounds for revised dark-text diagrams so their labels
  remain readable in the site's dark theme. Kept the existing purple palette.
- Wrapped wide comparison tables in local scrolling containers. Replaced
  literal inline-math pipes that Kramdown misread as table delimiters, and
  split the paired Girsanov SDE display and two long inline derivations into
  shorter equation lines.
- Updated eight figure generators along with their outputs; normalized SVG
  trailing whitespace after generation. No shared theme policy was changed.

## Verification

- `python3 scripts/validate_blog.py`: passed, without unused-asset warnings.
- `python3 scripts/validate_agent_hygiene.py`: passed.
- Figure-style and instruction-hygiene unit tests: 14 passed.
- Independent numerical checks: Gaussian likelihood/Girsanov signs, exact
  two-state AIS normalization and ratio direction, fully mixed finite-step
  weight variance, ideal-gas GCMC detailed balance, and rejection-state bias.
  These are checks of worked examples, not proofs of general SDE results.
- Isolated Jekyll build: passed; existing Sass deprecation warnings remain.
- Browser checks: all 12 posts at 1280-pixel desktop and 390-pixel mobile
  widths returned HTTP 200, with no page-level horizontal overflow, broken
  blog images, missing image descriptions, MathJax errors, unparsed equation
  delimiters, or JavaScript page errors. The last two edited tutorials were
  checked again after their final build.
- Inspected revised figures and captions in rendered pages, selected diagrams
  in dark mode, and the responsive benchmark panels at an intermediate
  820-pixel width. Confirmed the dark-background contrast fixes visually.
- Eight edited figure generators passed Python compilation; `git diff --check`
  passed. The changes were not committed or pushed in this task.

The app-browser connection was unavailable. Rendering was checked using an
isolated headless Chrome process and a temporary local build, without changing
the user's browser session or restarting the preview on port 4000. External
embedded demos were excluded from the layout check. This pass does not certify
every animation frame, external demo, paper result, or manuscript print layout.

## Primary references checked

- [Neal, Annealed Importance Sampling](https://arxiv.org/html/physics/9803008):
  invariant kernels, reversal, support, and normalization identities.
- [Vargas et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3716db40060004d0629d4051b2c57ab-Paper-Conference.pdf):
  Propositions 2.1–2.2, backward-time conventions, and compatible reference paths.
- [Goodman, stochastic-calculus notes](https://math.nyu.edu/faculty/goodman/teaching/StochCalc2012/notes/Week10.pdf):
  change-of-measure conditions and Girsanov's formula.
- [GROMACS pressure coupling](https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html#pressure-coupling):
  compressibility and Berendsen fluctuation limitations.
- [RFdiffusion README](https://github.com/RosettaCommons/RFdiffusion/blob/main/README.md?plain=1):
  contig and chain-break syntax.
- [Zitnick et al., 2020](https://arxiv.org/html/2010.09435): historical energy
  storage estimates and catalyst-screening context.
- [MaskGXT preprint](https://arxiv.org/html/2606.22866v1) and
  [MADField preprint](https://arxiv.org/html/2606.21284v1): numerical benchmarks,
  metric definitions, experimental scope, and limitations.
