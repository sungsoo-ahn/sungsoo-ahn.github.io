---
layout: page
title: "Research map: editing guide"
permalink: /research-map/editing/
nav: false
---

The **Research** section replaces the homepage’s selected highlights, after Members. The map reads `_data/publications.yml` for bibliographic metadata, `_data/research_map.yml` for paper analyses and connections, and `_data/research_map_keywords.yml` for the overcomplete keyword pool. The [complete analysis](/research-map/analysis/) and [keyword index](/research-map/keywords/) retain the detail outside the interactive map.

## Edit labels and categories

A work's `map_label` is its short station nickname. Use the paper's stated method shorthand when one is available, such as GEGL, GEEL, or LAMP. Papers with two introduced methods can show both, such as G-MF / G-BP. Use a descriptive nickname for studies without a named method. Its `label`, summaries, and bibliographic title remain unchanged. Taxonomy `map_label` supplies a short domain or concept name. Paper hover cards show the full scientific title, author list, bibliographic editions, and original abstract from `_data/publications.yml`. Abstract-source comments record the arXiv or publisher/conference/university record used; TeX formatting is normalized to readable text. Domain-line hover cards compare the two papers’ contributions. Shared-idea line hovers explain the specific parallel and link to both papers. Evidence remains in the complete analysis.

Edit `memberships` to change a work's categories. Each membership names a theme, its role (`central`, `used`, `evaluation`, or `parallel`), a source index, and an evidence locator. Each work requires a non-parallel major domain and a non-parallel major methodology. The overview uses domains; methodology annotations remain in the analysis. **Graphical models** covers classical BP, elimination, and partition-function inference. **General ML** covers general learning, sampling, graph models, language reasoning, control, and robustness. General ML retains the stable `d_deep` identifier and its existing paper assignments. Graph and language methods remain in the methodology annotations; their application domains determine their other memberships.

The 412-keyword pool is intentionally larger than the plotted taxonomy. Edit keyword labels and aliases in `_data/research_map_keywords.yml`, and work assignments in `_data/research_map.yml`. Keywords are retained for analysis and future editing; they are not diagram filters and do not automatically create connections. The `sim2real` alias supports reviewing the simulation/proxy-to-reality gap; the draft does not claim verified real-robot transfer.

## Contribution annotations

Every paper uses the same circular station marker. Colors identify domains. Desktop description boxes use a light tint of their station’s primary domain color, with every domain named in its own route color to the right of the title. Narrow screens retain the domain legend below the narrow vertical map, sorted by first appearance in the complete collection. The complete analysis retains contribution families: Generative modeling, Symmetry, Representations, Learning & inference, Search & optimization, Reasoning, Agents, and Benchmarks.

Edit `contribution_categories` for labels, `map_contribution` for a work’s main family, and `map_contribution_reason` for its short rationale. Shape values remain as inactive metadata. The generated analysis records the rationale; paper hover cards use the original abstract. Classify the central novelty from the existing contribution analysis and reviewed arXiv source. Using a flow model, an LLM, or equivariance alone does not determine the contribution. QHFlow and QHFlow2 both use Symmetry by the author’s choice; Symmetric replay also uses Symmetry. CORE-PO and veracity inference use Reasoning; MT-Mol and INDIBATOR use Agents. VibeProteinBench and RL4CO use Benchmarks. GFN approaches use Search & optimization: LS-GFN, LED-GFN, PBP-GFN, RxnFlow, and Adaptive Teachers. Optional `map_secondary_contributions` lists additional contribution-category IDs for the analysis; keep them distinct from the main `map_contribution`. These annotations are curated from each paper’s contribution analysis, not inferred automatically from methods or keywords. MaskGXT / HACO retains Agents as its main contribution and also shows Generative modeling. Edit `summary` for the one-sentence desktop box description and a publication’s `abstract` for its hover text. Long abstracts scroll inside a viewport-bounded card; hovering does not fetch external data.

**Electronic structure** contains GPWNO, QHFlow, and QHFlow2, replacing their Materials/Molecules appearances. Materials now covers crystals, porous materials, and catalysts. Original experimental descriptions and sources remain intact. The former combined Materials & electronic structure keyword remains in the overcomplete pool without an active domain mapping.

## Routes and layout

Only papers are stations. Every major domain route connects adjacent publications within its complete chronological sequence. For A, B, C, the only segments are A–B and B–C. Scrolling never adds A–C.

The map is **always vertical**, with the newest papers at the top and the oldest at the bottom. Ordinary page scrolling traverses all 74 works at one scale. Chronology uses the earliest eligible arXiv or conference acceptance date. Spacing is variable and reserves room for descriptions; this is an ordered timeline, not a proportional calendar scale. Blossom-BP (2015) appears below GEGL (2020).

The build first packs stations using a chronological sweep and domain preferences, then retains the cleaned cross-domain placement in a 300-pixel-wide vertical network. A constrained force pass uses edge springs and station/line repulsion. Routing favors vertical runs and 45-degree transitions with short rounded corners. The cost penalizes bends, detours, crossings, and close parallel tracks. Separate entry and exit positions keep parallel domain routes apart at shared stations. Background casings leave clear gaps at unavoidable crossings; there are no line jumps. Every paper has one station, including papers in several domains.

An optional work-level `map_layout` refines the initial cross-domain placement:

```yaml
map_layout:
  y: 164
  label_side: above
  label_gap: 16
```

`y` is the cross-domain coordinate inside the 300-pixel network. The optional `label_side` and `label_gap` affect the initial packing. These display hints do not change scientific categories or chronology. Use the rendered preview to assess the final vertical station and route positions.

At browser widths of **992 CSS pixels and above**, lightly tinted side boxes show the nickname, first conference/year, one-sentence summary, and domain names. Captions alternate sides, with a thin grey leader connecting each box to its circle. Domain names use their own route colors and replace the desktop legend. Spacing starts at 52 pixels and grows for compressed time gaps and to leave at least 16 pixels between boxes in the same column.

**Below 992 pixels**, the boxes and leaders are hidden. Each circle has a nearby paper nickname and conference/year instead. The generator selects the clearer side of the circle, wraps longer names, and keeps labels inside the network. A small background-colored text halo separates letters from crossing routes. The stable domain legend appears below the figure, sorted by first appearance. There are no navigation buttons, layout toggles, or filters.

Both label treatments use the **same SVG, stations, connections, and figure height**. Resizing changes only label visibility; it does not rearrange papers or reset the reading position. The SVG remains usable without JavaScript. Circles are 18 pixels across, hit targets are 32 pixels across, and domain routes are 4 pixels thick. Full titles, authors, bibliographic editions, and original abstracts are available on hover or keyboard focus. Circles, nearby nicknames, and desktop caption nicknames link directly to papers.

Taxonomy `color` references the site palette; optional `display_color` supplies the distinct category hue. Route colors meet at least 3:1 contrast on light and dark backgrounds. The Python generator runs the Node layout model at build time and emits one native SVG with both label treatments. The browser adds hover and focus behavior without rerunning force simulation or routing. Arial advance metrics determine label widths before rendering. Legacy `x`, `y`, and `overview` fields are optional and ignored.

## Application-domain review

Domain assignments follow each paper's actual tasks, with source locators retained in `memberships`. Molecular graph generation and molecular language-model work stay under Molecules. Protein-interface language models stay under Proteins, perturbation reasoning under Cells, and crystal co-scientist work under Materials. Generic routing and independent-set optimization, generic graph architectures, and general language reasoning belong to General ML.

The review added missing evaluation memberships:

- [EPIC, Section 4.1](https://arxiv.org/html/2306.01310v3): molecular classification on NCI1, BZR, COX2, Mutagenicity, BBBP, BACE, and HIV; biological graph classification on PROTEINS and ENZYMES.
- [Wavelet diffusion, Sections 4.1 and 4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/427f20d90386fd27804f1831d6a3d48f-Paper-Conference.pdf): QM9 molecular generation alongside generic graph generation.
- [Non-backtracking GNN, Section 5.1](https://arxiv.org/html/2310.07430v2): Peptides-func and Peptides-struct alongside vision and generic node-classification tasks.
- [Node diffusion, Section 6.2](https://arxiv.org/html/2302.10506v5): protein–protein interaction (PPI) classification alongside citation networks and graph algorithmic reasoning.

These generic architecture papers retain a General ML membership; application evaluation memberships add domain routes at their shared stations only where supported by experiments. Classical BP and factor-transformation work is assigned to Graphical models.

## Conceptual connections

Curated `relationships` have a name, explanation, and evidence from both endpoints. `documented` means the shared mechanism is supported by those sources. `interpretive` means a conceptual parallel; its explanation should retain the material differences. Neither status asserts citation or historical influence.

Add a relationship’s optional `map_idea` to select it for the dashed connections and name a specific shared mechanism. The initial 14 pairs include **Local search improvement operators** between GEGL and LS-GFN: molecular mutation/crossover and trajectory backtracking/reconstruction refine generated candidates before reuse in training. Broad domain, methodology, and keyword overlap does not create an idea link.

`idea_window_years: 5` limits endpoint publication dates to five calendar years, inclusive, using the same earliest eligible arXiv/acceptance chronology as the main map. All selected idea connections stay visible as thin, dashed grey lines behind the solid domain routes. Hover or keyboard focus strengthens the relevant dashed line and highlights its endpoints. Hovering a paper highlights its incident connections; its card contains only bibliographic information and the abstract. Hovering or focusing an idea line shows its name, `map_explanation`, and links to the two papers. The explanations distinguish the two implementations without claiming citation or influence.

Idea paths are routed at build time around unrelated stations in the vertical network. The browser reuses the generated paths for highlighting and line hovers. The remaining canonical relationships, concept groups, keyword pool, and full analysis stay editable. Short explanations retain ASD-STE100-style descriptive writing: active voice, one point per sentence, and consistent technical names.

## Paper links and section copy

Each station is a native link, including its icon and nickname, and opens in a new tab. The JS model’s `paperUrl` helper supplies destinations for both the interactive and generated static map. It prefers the bibliography’s arXiv ID, then a reviewed arXiv source normalized to an abstract-page link. When neither exists, it uses a non-OpenReview publisher page or reviewed source. The six current exceptions are BEAG, HoliMol, Wave-GD, DRIMA, DND, and STGG. Static SVG titles include the full title, authors, venues, and abstract; their accessible link names remain the concise paper titles.

The Research introduction and small attribution live in `_includes/research_map.liquid`. The exact two-sentence introduction precedes the figure; the muted 11-pixel credit below the figure links to [Kirill Neklyudov’s homepage](https://necludov.github.io/).

Traversal uses ordinary page scrolling at every width. Obsolete map offset, paper, lens, filter, search, and zoom parameters are removed. Unrelated URL parameters are preserved.

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

The generator updates the analysis, keyword index, linked static fallback, and mobile legend. The read-only validator checks coverage, identifiers, chronology, evidence, source versions, and generated drift. Jekyll renders the browser JSON directly from canonical data. A failed JSON request leaves the linked static map visible.

With the existing local preview running:

```bash
PLAYWRIGHT_CHROMIUM_CHANNEL=chrome npm run check:research-map:browser
```

The browser checker accepts `--url` and `--screenshots`. It checks vertical layout at all widths, the 992-pixel label breakpoint, stable geometry, label clearance, all paper links and complete abstract hovers, desktop leaders, persistent grey idea lines, keyboard/touch access, light/dark themes, fast setup, and fallback rendering. It does not restart the preview server.
