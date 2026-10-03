---
layout: page
title: "Research map: editing guide"
permalink: /research-map/editing/
nav: false
---

The **Research** section replaces the homepage’s selected highlights, after Members. The map reads `_data/publications.yml` for bibliographic metadata, `_data/research_map.yml` for paper analyses and connections, and `_data/research_map_keywords.yml` for the overcomplete keyword pool. The [complete analysis](/research-map/analysis/) and [keyword index](/research-map/keywords/) retain the detail outside the interactive map.

## Edit labels and categories

A work's `map_label` is its short station nickname. Use the paper's stated method shorthand when one is available, such as GEGL, GEEL, or LAMP. Papers with two introduced methods can show both, such as G-MF / G-BP. Use a descriptive nickname for studies without a named method. Its `label`, summaries, and bibliographic title remain unchanged. Taxonomy `map_label` supplies a short row or concept name; a relationship's `map_label` supplies its short spoke name. Paper hover cards show full scientific titles, author names in publication order, and all bibliographic editions. Connection hover cards use the short `map_label` and `map_explanation`. The complete analysis retains the original explanations and evidence.

Edit `memberships` to change a work's categories. Each membership names a theme, its role (`central`, `used`, `evaluation`, or `parallel`), a source index, and an evidence locator. Each work requires a non-parallel major domain and a non-parallel major methodology. The overview uses domains; methodologies appear in paper hover cards. General ML is divided into **Graphical models** (classical BP, elimination, and partition-function inference) and **Deep learning** (general neural learning, sampling, graph models, language reasoning, control, and robustness). Graph and language methods remain in the methodology annotations; their application domains determine their other memberships.

The 412-keyword pool is intentionally larger than the plotted taxonomy. Edit keyword labels and aliases in `_data/research_map_keywords.yml`, and work assignments in `_data/research_map.yml`. Keywords are retained for analysis and future editing; they are not diagram filters and do not automatically create connections. The `sim2real` alias supports reviewing the simulation/proxy-to-reality gap; the draft does not claim verified real-robot transfer.

## Contribution symbols

Colors identify domains; station shapes identify one main research contribution. Domain colors use short line samples in the legend and paper hover cards, so they remain distinct from contribution symbols. The eight families are Generative modeling (circle), Symmetry (hexagon), Representations (square), Learning & inference (+), Search & optimization (triangle), Reasoning (×), Agents (star), and Benchmarks (diamond). Both persistent legends are sorted by first appearance in the complete collection.

Edit `contribution_categories` for labels and shapes, `map_contribution` for a work’s single family, and `map_contribution_reason` for its short rationale. The generated analysis records the rationale; hover cards show the contribution name and full method annotations. Classify the central novelty from the existing contribution analysis and reviewed arXiv source. Using a flow model, an LLM, or equivariance alone does not determine the shape. QHFlow and QHFlow2 both use Symmetry by the author’s choice; Symmetric replay also uses Symmetry. CORE-PO and veracity inference use Reasoning; MT-Mol and INDIBATOR use Agents. VibeProteinBench and RL4CO use Benchmarks. All copies of a paper use the same symbol.

**Electronic structure** contains GPWNO, QHFlow, and QHFlow2, replacing their Materials/Molecules appearances. Materials now covers crystals, porous materials, and catalysts. Original experimental descriptions and sources remain intact. The former combined Materials & electronic structure keyword remains in the overcomplete pool without an active domain mapping.

## Routes and layout

Only papers are stations. Every major domain route connects adjacent publications within its complete chronological sequence. For A, B, C, the only segments are A–B and B–C. Panning clips the existing routes and never adds A–C.

The figure uses a shared chronological x coordinate for every paper. A deterministic sweep follows the earliest eligible arXiv or acceptance date and reserves room only for the fixed topmost label of each paper. Spacing is variable: label widths require 12 pixels of clear space, station centers stay at least 44 pixels apart, and compressed time gaps contribute at most 80 pixels. The temporal contribution is 80 pixels per year. Earlier works always precede later ones across domains; Blossom-BP (2015) appears before GEGL (2020). Repeated stations are staggered by 16 pixels between appearances. Lower copies remain unnamed and reserve no text width. Space is reserved where a link would cross a primary label or any station. This is an ordered timeline, not a proportional calendar scale.

SVG station symbols are about 20 pixels across, routes are 4 pixels thick, and station hit targets are 32 pixels across. Rows are 68 pixels tall; the current collection occupies six fixed rows. Each paper’s nickname and first venue/year appear only on its fixed topmost copy when the label fits inside the viewport; lower duplicate stations retain their colors, links, and interactions. All copies retain hover and keyboard details; full titles and all editions are available on hover or keyboard focus.

There are no category headings inside the figure. A persistent, non-interactive legend below it shows every category in a distinct color, sorted by the first publication in that category. Figure height and lane positions remain fixed while panning, including lanes without a station in the current viewport. The earliest General ML line starts in the center; newer application lines fan above and below it as their first papers enter the chronology. This widening arrangement preserves actual theme routes rather than inventing parent–child relationships between unrelated papers.

Interval partitioning allows categories with non-overlapping full date spans to share a lane while retaining independent colors and routes. Weather & cosmology and Materials currently share a lane, as do the older graphical-model work and later deep-learning work. Packing and lane order use the complete collection and remain stable while navigating.

The map opens at the newest end. Left/right buttons pan up to 360 CSS pixels at one scale, with overlapping windows on narrow screens so papers cannot fall between navigation steps. Horizontal scrolling is also supported. Arrow keys move the same distance, Home reaches the oldest papers, and End reaches the newest. Visible grey, solid straight links named **Same paper** continuously join consecutive visible appearances of repeated papers. These links sit behind stations, keep clear of labels, and become stronger on hover. Hover or keyboard focus highlights their corresponding paper and reveals the link name.

Taxonomy `color` references the site palette. Optional `display_color` provides the distinct category hue. Route colors are adjusted for at least 3:1 contrast on light and dark backgrounds. The runtime uses native SVG. The Python generator invokes the same Node layout model for exact static/runtime geometry and label wrapping. Arial advance metrics with a safety margin determine label widths before rendering; labels retain their 12-pixel and 10-pixel font sizes. Legacy `x`, `y`, and `overview` fields are optional and ignored.

## Application-domain review

Domain assignments follow each paper's actual tasks, with source locators retained in `memberships`. Molecular graph generation and molecular language-model work stay under Molecules. Protein-interface language models stay under Proteins, perturbation reasoning under Cells, and crystal co-scientist work under Materials. Generic routing and independent-set optimization, generic graph architectures, and general language reasoning belong to Deep learning.

The review added missing evaluation memberships:

- [EPIC, Section 4.1](https://arxiv.org/html/2306.01310v3): molecular classification on NCI1, BZR, COX2, Mutagenicity, BBBP, BACE, and HIV; biological graph classification on PROTEINS and ENZYMES.
- [Wavelet diffusion, Sections 4.1 and 4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/427f20d90386fd27804f1831d6a3d48f-Paper-Conference.pdf): QM9 molecular generation alongside generic graph generation.
- [Non-backtracking GNN, Section 5.1](https://arxiv.org/html/2310.07430v2): Peptides-func and Peptides-struct alongside vision and generic node-classification tasks.
- [Node diffusion, Section 6.2](https://arxiv.org/html/2302.10506v5): protein–protein interaction (PPI) classification alongside citation networks and graph algorithmic reasoning.

These generic architecture papers retain a Deep learning membership; application evaluation memberships create additional appearances only where supported by experiments. Classical BP and factor-transformation work is assigned to Graphical models.

## Conceptual connections

Curated `relationships` have a name, explanation, and evidence from both endpoints. `documented` means the shared mechanism is supported by those sources. `interpretive` means a conceptual parallel; its explanation should retain the material differences. Neither status asserts citation or historical influence.

Clicking a paper replaces the overview with its conceptual neighbors. Both supported mechanisms and interpretive parallels appear. Generic domain or methodology membership does not create a spoke. Concept themes with `map_role: detail` and two or three papers contribute only the chronological segments incident to the selected paper. Broad concept memberships alone do not create edges.

Multiple reasons between a pair share one spoke. The map shows each short explanation without an evidence panel; the canonical annotations and complete analysis retain every reason and source. Short concept names appear on spokes on desktop and beneath peer cards on narrow screens. Interpretive-only spokes are dashed; there are no arrowheads. Earlier peers appear left of the selected paper, later peers right; narrow screens use above/below. Clicking a peer recenters the graph. **Back** restores the original overview offset and clicked appearance. The selected paper links to its reviewed arXiv version, or its publisher source when no arXiv version is available.

The display explanations follow [ASD-STE100-style descriptive writing](https://www.asd-ste100.org/STE_faq.html): active voice, one point per sentence, consistent technical names, and at most 25 words per sentence. Edit a relationship’s `map_explanation` for its display text; retain `explanation` for the detailed analysis. Small concept groups use the taxonomy’s `map_description`. The short text states the shared idea and the differences between methods.

Shared URLs retain only `rm_paper` and `rm_offset` (distance from the newest end, in 120-pixel navigation units). Obsolete lens, filter, search, and zoom parameters are ignored and removed. Unrelated URL parameters are preserved.

## Chronology and sources

Each work records candidate dates and selects the earliest eligible one:

```yaml
chronology:
  date: "2024-02-05"
  venue: arXiv
  basis: arxiv_version
  source: https://arxiv.org/abs/2402.05965v1
  version: 2402.05965v1
  author: Sungsoo Ahn
  events:
    - date: "2024-02-05"
      venue: arXiv
      basis: arxiv_version
      source: https://arxiv.org/abs/2402.05965v1
      version: 2402.05965v1
      author: Sungsoo Ahn
    - date: "2024-05-01"
      venue: ICML
      basis: conference_notification
      source: https://icml.cc/Conferences/2024/Dates
```

Conference papers and preprints follow the same rule: the earlier of **the first arXiv version listing Sungsoo Ahn as an author** and **the conference's acceptance notification**. Version dates come from arXiv's version-specific `updated` timestamp, not the base record's original `published` timestamp when authorship changed. Conference dates come from official calendars or calls for papers. They retain the calendar's displayed date, without converting an AoE deadline into a Korean date. Conference and year labels still follow the bibliography, independent of the ordering date. Titles and identifiers break ties at the same date. A theme still connects only adjacent papers after sorting.

The first-version author lists were checked for all 68 arXiv-backed works. Three papers require later versions: [Odd-cycle BP v2](https://arxiv.org/abs/1306.1167v2), January 1, 2018; [RL4CO v4](https://arxiv.org/abs/2306.17100v4), June 21, 2024; and [Antibody decoupling v2](https://arxiv.org/abs/2402.05982v2), May 27, 2024. Earlier versions are recorded in `excluded_arxiv_versions` with source links and the authorship reason.

For journals, a verified publication month (`basis: journal_month`) remains a fallback when neither an eligible arXiv version nor an independently verified acceptance date is available. HoliMol uses this fallback. The archived NIPS 2015 calendar omits the notification date, so Blossom BP currently uses its verified arXiv posting; the CFP's review-end date is not silently treated as an acceptance announcement. The NIPS 2016 notification window and both KDD 2025 notification rounds fall after the eligible arXiv versions, so their missing exact paper-specific dates do not affect the earliest event. These evidence boundaries are retained in each chronology's `note`.

ASSD retains its bibliographic discrepancy: the [official TMLR archive](https://jmlr.org/tmlr/papers/) gives January 2025, while the homepage bibliography lists 2024. Its map date now follows the first coauthored arXiv version in May 2024; the bibliography and stable identifiers are preserved.

When editing dates, keep the top-level selected event consistent with the earliest `events` entry. The validator rejects later selections, missing version pins, and unverified authorship fields.

Prefer arXiv links already present in the bibliography and pin the version reviewed. When only partial evidence is available, set `review_status: partial`, explain the boundary in `coverage_note`, and limit claims accordingly. HoliMol, DND, and STGG remain partial reviews. Sources and chronology do not use OpenReview.

## Regenerate and check

After editing canonical YAML:

```bash
uv run python scripts/research_map.py --write-reports
uv run python scripts/research_map.py
uv run python -m unittest discover -s tests -p 'test_research_map.py'
npm run test:research-map
npm run layout:research-map
```

The generator updates the analysis, keyword index, linked static fallback, and persistent legend. The read-only validator checks coverage, identifiers, chronology, evidence, source versions, and generated drift. Jekyll renders the browser JSON directly from canonical data. A failed JSON request leaves the linked static map visible.

With the existing local preview running:

```bash
PLAYWRIGHT_CHROMIUM_CHANNEL=chrome npm run check:research-map:browser
```

The browser checker accepts `--url` and `--screenshots`, and covers fixed-scale navigation, conceptual focus, history, keyboard access, mobile and dark layouts, and fallback rendering. It does not restart the preview server.
