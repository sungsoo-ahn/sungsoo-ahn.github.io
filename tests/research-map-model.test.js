const test = require("node:test");
const assert = require("node:assert/strict");
const model = require("../assets/js/research-map-model.js");
const theme = (id, kind = "domain") => ({ id, kind, label: id, map_label: id });
const work = (id, date, themes, title = id) => ({
  annotations: { id, label: id, map_label: id, chronology: { date }, memberships: themes.map((theme) => ({ theme, role: "central" })) },
  publications: [{ id, title, venue: "ICML", year: Number(date.slice(0, 4)) }],
});
const fixture = (works, taxonomy, relationships = []) => model.prepare({ works, taxonomy, relationships });
const data = fixture(
  [
    {
      ...work("a", "2024-05-07", ["d_a", "c_small", "c_broad"]),
      publications: [
        { id: "a", year: 2024, title: "A" },
        { id: "a_journal", year: 2025, title: "A" },
      ],
    },
    work("b", "2024-07-21", ["d_a", "c_broad"]),
    work("c", "2024-12-10", ["d_a", "d_b", "c_small", "c_broad"]),
    work("d", "2025-01", ["d_b", "c_small", "c_broad"]),
  ],
  [theme("d_a"), theme("d_b"), theme("c_small", "concept"), theme("c_broad", "concept")],
  [
    { id: "r1", from: "d", to: "a", label: "Rare-event dynamics", map_label: "Rare transitions", status: "documented" },
    { id: "r2", from: "a", to: "c", label: "Different analogy", map_label: "c_small", status: "interpretive" },
  ]
);

test("navigation state keeps only offset and ignores obsolete paper selections", () => {
  assert.deepEqual(model.sanitizeState({ paper: "a_journal", offset: 3, lens: "method", tags: ["d_a"] }), { offset: 3 });
  for (const offset of [-2, Infinity, "invalid", undefined]) assert.equal(model.sanitizeState({ offset }).offset, 0);
});

test("URLs round-trip panning, clean old paper and filter parameters, and preserve unrelated values", () => {
  const original = "https://example.com/?utm_source=friend&rm_paper=a&rm_lens=method&rm_tags=d_a&rm_zoom=2#research-map";
  const url = model.writeUrl(original, { offset: 5.123456 });
  assert.deepEqual([...url.searchParams.keys()], ["utm_source", "rm_offset"]);
  assert.deepEqual(model.readUrl(url), { offset: 5.123 });
  assert.equal(url.hash, "#research-map");
  assert.equal(model.writeUrl(url, { offset: 0 }).searchParams.size, 1);
});

test("paper links prefer bibliographic arXiv and use reviewed sources without OpenReview", () => {
  assert.equal(
    model.paperUrl({
      publications: [{ id: "a", arxiv: "2606.22866", html: "https://publisher.example/paper" }],
      sources: [{ url: "https://arxiv.org/html/2606.22866v1" }],
    }),
    "https://arxiv.org/abs/2606.22866"
  );
  for (const url of ["https://arxiv.org/html/1306.1167v2", "https://arxiv.org/pdf/1306.1167v2.pdf", "https://arxiv.org/abs/1306.1167v2"])
    assert.equal(model.paperUrl({ publications: [{ id: "a" }], sources: [{ url }] }), "https://arxiv.org/abs/1306.1167v2");
  assert.equal(
    model.paperUrl({ publications: [{ id: "a", html: "https://publisher.example/paper" }], sources: [{ url: "https://github.com/author/project" }] }),
    "https://publisher.example/paper"
  );
  assert.equal(
    model.paperUrl({
      publications: [{ id: "a", html: "https://openreview.net/forum?id=a" }],
      sources: [{ url: "https://github.com/author/project" }],
    }),
    "https://github.com/author/project"
  );
});

test("major routes contain only sequential papers in full conference chronology", () => {
  const rows = model.timelineRows(data);
  assert.deepEqual(
    rows[0].connections.map((edge) => [edge.from, edge.to]),
    [
      ["a", "b"],
      ["b", "c"],
    ]
  );
  assert.deepEqual(
    rows[1].connections.map((edge) => [edge.from, edge.to]),
    [["c", "d"]]
  );
  assert.ok(
    rows
      .flatMap((row) => row.connections)
      .every((edge) => edge.label && model.compareChronology(data.workById.get(edge.from), data.workById.get(edge.to)) < 0)
  );
  assert.deepEqual(model.chronologyKey(data.workById.get("d")), ["2025-01-01", "d", "d"]);
});

test("equal conference dates use full titles then identifiers, independently of labels and geometry", () => {
  const corpus = fixture(
    [work("z", "2024-07-21", ["d_a"], "Alpha"), work("b", "2024-07-21", ["d_a"], "Beta"), work("a", "2024-07-21", ["d_a"], "Alpha")],
    [theme("d_a")]
  );
  assert.deepEqual(
    model.timelineRows(corpus)[0].stations.map((paper) => paper.id),
    ["a", "z", "b"]
  );
});

test("a multi-domain paper is one interchange with stable coordinates during navigation", () => {
  const layout = model.timelineLayout(data);
  assert.equal(layout.stations.length, data.works.length);
  const shared = layout.stationByInstance.get("c");
  assert.deepEqual(shared.themes.map((theme) => theme.id).sort(), ["d_a", "d_b"]);
  assert.equal(layout.routes.filter((route) => route.to === "c").length, 1);
  assert.equal(layout.routes.filter((route) => route.from === "c").length, 1);
  const sequence = [...data.works].sort(model.compareChronology);
  sequence.slice(1).forEach((paper, i) => assert.ok(layout.xById.get(paper.id) > layout.xById.get(sequence[i].id)));
  const original = layout.stations.map((station) => [station.id, station.x, station.y]);
  for (const width of [128, 390, 900]) {
    for (const offset of [0, 2, 1000]) model.timelineViewport(layout, offset, width);
    assert.deepEqual(
      layout.stations.map((station) => [station.id, station.x, station.y]),
      original
    );
  }
});

test("route colors meet graphic contrast on both backgrounds", () => {
  for (const color of ["#CC7D5F", "#ffffff", "#000000", "#986FDC", "#ffbd40"]) {
    const swatches = model.colorSwatches(color);
    assert.ok(model.contrast(swatches.light, "#ffffff") >= 3.1);
    assert.ok(model.contrast(swatches.dark, "#21192b") >= 3.1);
  }
});
