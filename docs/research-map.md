---
layout: page
title: "Research map: editing guide"
permalink: /research-map/editing/
nav: false
---

The **Research** section replaces the homepage’s selected highlights, after Members. The map reads `_data/publications.yml` for bibliographic metadata, `_data/research_map.yml` for paper analyses and connections, and `_data/research_map_keywords.yml` for the overcomplete keyword pool. The [complete analysis](/research-map/analysis/) and [keyword index](/research-map/keywords/) retain the detail outside the interactive map.

## Edit labels and categories

A work's `map_label` is its short station nickname. Use the paper's stated method shorthand when one is available, such as GEGL, GEEL, or LAMP. Papers with two introduced methods can show both, such as G-MF / G-BP. Use a descriptive nickname for studies without a named method. Its `label`, summaries, and bibliographic title remain unchanged. Taxonomy `map_label` supplies a short domain or concept name; a relationship's `map_label` supplies its short spoke name. Paper hover cards show full scientific titles, author names in publication order, and all bibliographic editions. Connection hover cards use the short `map_label` and `map_explanation`. The complete analysis retains the original explanations and evidence.

Edit `memberships` to change a work's categories. Each membership names a theme, its role (`central`, `used`, `evaluation`, or `parallel`), a source index, and an evidence locator. Each work requires a non-parallel major domain and a non-parallel major methodology. The overview uses domains; methodologies appear in paper hover cards. General ML is divided into **Graphical models** (classical BP, elimination, and partition-function inference) and **Deep learning** (general neural learning, sampling, graph models, language reasoning, control, and robustness). Graph and language methods remain in the methodology annotations; their application domains determine their other memberships.

The 412-keyword pool is intentionally larger than the plotted taxonomy. Edit keyword labels and aliases in `_data/research_map_keywords.yml`, and work assignments in `_data/research_map.yml`. Keywords are retained for analysis and future editing; they are not diagram filters and do not automatically create connections. The `sim2real` alias supports reviewing the simulation/proxy-to-reality gap; the draft does not claim verified real-robot transfer.

## Contribution symbols

Colors identify domains; station shapes identify one main research contribution. Domain colors use short line samples in the legend and paper hover cards, so they remain distinct from contribution symbols. The eight families are Generative modeling (circle), Symmetry (hexagon), Representations (square), Learning & inference (+), Search & optimization (triangle), Reasoning (×), Agents (star), and Benchmarks (diamond). Both persistent legends are sorted by first appearance in the complete collection.

Edit `contribution_categories` for labels and shapes, `map_contribution` for a work’s single family, and `map_contribution_reason` for its short rationale. The generated analysis records the rationale; hover cards show the contribution name and full method annotations. Classify the central novelty from the existing contribution analysis and reviewed arXiv source. Using a flow model, an LLM, or equivariance alone does not determine the shape. QHFlow and QHFlow2 both use Symmetry by the author’s choice; Symmetric replay also uses Symmetry. CORE-PO and veracity inference use Reasoning; MT-Mol and INDIBATOR use Agents. VibeProteinBench and RL4CO use Benchmarks. GFN approaches use Search & optimization: LS-GFN, LED-GFN, PBP-GFN, RxnFlow, and Adaptive Teachers. The station, focus cards, and tooltip use the same symbol.

**Electronic structure** contains GPWNO, QHFlow, and QHFlow2, replacing their Materials/Molecules appearances. Materials now covers crystals, porous materials, and catalysts. Original experimental descriptions and sources remain intact. The former combined Materials & electronic structure keyword remains in the overcomplete pool without an active domain mapping.

## Routes and layout

Only papers are stations. Every major domain route connects adjacent publications within its complete chronological sequence. For A, B, C, the only segments are A–B and B–C. Panning clips the existing routes and never adds A–C.

The figure uses a shared chronological x coordinate for every paper and one station per work. A deterministic sweep follows the earliest eligible arXiv or acceptance date and reserves room for each nickname and venue label. Spacing is variable; compressed time gaps contribute up to 72 pixels at 64 pixels per year. Earlier works always precede later ones across domains; Blossom-BP (2015) appears before GEGL (2020). This is an ordered timeline, not a proportional calendar scale.

Domains supply soft vertical preferences rather than fixed rows. Stations and labels use the available space above and below each line. A constrained force pass uses edge springs, station repulsion, and line repulsion to open crowded areas. Horizontal coordinates stay pinned, vertical movement is limited to 28 pixels, and labels and routing corridors retain clear space. The figure is 300 pixels tall and keeps the same full-collection geometry throughout navigation. A paper in multiple domains is one interchange; its station uses its primary domain color and all relevant domain routes meet there.

Routing favors horizontal runs and 45-degree transitions, with short rounded corners. An obstacle visibility graph avoids labels and unrelated stations. The cost penalizes bends, large detours, crossings, and close parallel runs; a second pass compares each route with all other routes and accepts only a lower-cost path. Tracks use separate entry and exit positions beneath each shared station symbol, ordered consistently by domain. A little extra horizontal room at steep interchanges gives the lines space to turn. Tracks that connect the same pair of papers stay separate. A background casing leaves a small gap in the lower line at unavoidable crossings; there are no line jumps. Routes still connect only adjacent papers within each complete chronological theme sequence.

For a crowded interchange, an optional work-level `map_layout` adjusts the station and caption after automatic packing and the force pass:

```yaml
map_layout:
  y: 164
  label_side: above
  label_gap: 16
```

`y` is a vertical display position inside the 300-pixel figure. `label_side` is optional (`above` or `below`); `label_gap` adds 0–24 pixels between the station and caption. These are display hints, independent of scientific categories and chronology. Unhinted station heights remain unchanged. Local horizontal reflow reserves room around adjusted stations and their labels while keeping chronological order. Use the rendered preview to check both the nickname and venue label; the automated geometry checks catch collisions and line intrusions.

SVG station symbols are about 20 pixels across, routes are 4 pixels thick, and station hit targets are 32 pixels across. Each paper’s nickname and first venue/year appear above or below its station when the label fits inside the viewport. Full titles, authors, all domains, contributions, methods, and bibliographic editions are available on hover or keyboard focus.

The diagram keeps one scale and a consistent legend below it. Domain swatches use distinct colored line segments; contribution families use monochrome shape markers. Both legends follow the first appearance of their categories. The map opens at the newest end. Left/right buttons pan up to 360 CSS pixels, with overlapping windows on narrow screens so papers cannot fall between navigation steps. Horizontal scrolling is also supported. Arrow keys move the same distance, Home reaches the oldest papers, and End reaches the newest.

Taxonomy `color` references the site palette. Optional `display_color` provides the distinct category hue. Route colors are adjusted for at least 3:1 contrast on light and dark backgrounds. The runtime uses native SVG. The Python generator invokes the same Node layout model for exact static/runtime geometry and label wrapping. Arial advance metrics with a safety margin determine label widths before rendering; labels retain their 12-pixel and 10-pixel font sizes. Legacy `x`, `y`, and `overview` fields are optional and ignored.

## Application-domain review

Domain assignments follow each paper's actual tasks, with source locators retained in `memberships`. Molecular graph generation and molecular language-model work stay under Molecules. Protein-interface language models stay under Proteins, perturbation reasoning under Cells, and crystal co-scientist work under Materials. Generic routing and independent-set optimization, generic graph architectures, and general language reasoning belong to Deep learning.

The review added missing evaluation memberships:

- [EPIC, Section 4.1](https://arxiv.org/html/2306.01310v3): molecular classification on NCI1, BZR, COX2, Mutagenicity, BBBP, BACE, and HIV; biological graph classification on PROTEINS and ENZYMES.
- [Wavelet diffusion, Sections 4.1 and 4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/427f20d90386fd27804f1831d6a3d48f-Paper-Conference.pdf): QM9 molecular generation alongside generic graph generation.
- [Non-backtracking GNN, Section 5.1](https://arxiv.org/html/2310.07430v2): Peptides-func and Peptides-struct alongside vision and generic node-classification tasks.
- [Node diffusion, Section 6.2](https://arxiv.org/html/2302.10506v5): protein–protein interaction (PPI) classification alongside citation networks and graph algorithmic reasoning.

These generic architecture papers retain a Deep learning membership; application evaluation memberships add domain routes at their shared stations only where supported by experiments. Classical BP and factor-transformation work is assigned to Graphical models.

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
