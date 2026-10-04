import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import yaml from "js-yaml";
import model from "../assets/js/research-map-model.js";
const source = yaml.load(fs.readFileSync(new URL("../_data/research_map.yml", import.meta.url), "utf8"));
const pubs = yaml.load(fs.readFileSync(new URL("../_data/publications.yml", import.meta.url), "utf8"));
const data = model.prepare({
  ...source,
  works: source.works.map((annotations) => ({ annotations, publications: annotations.publication_ids.map((id) => pubs.find((p) => p.id === id)) })),
});
const layout = model.timelineLayout(data);
const geometry = (value) => value.lanes.map((lane) => [lane.id, lane.stations.map((station) => [station.id, station.x, station.y])]);

test("full corpus uses 74 shared stations and eight distinct chronological routes", () => {
  assert.equal(data.works.length, 74);
  assert.equal(layout.rows.length, 8);
  assert.equal(layout.height, 300);
  assert.ok(layout.width < 4033, "shared stations and flexible paths reduce the previous track width");
  assert.equal(layout.stationByInstance.size, 74);
  assert.equal(new Set(layout.stations.map((station) => station.work.id)).size, 74);
  assert.equal(layout.rows.find((row) => row.theme.id === "d_deep").stations.length, 29);
  assert.equal(layout.rows.find((row) => row.theme.id === "d_graphical").stations.length, 6);
  for (const removed of ["d_control", "d_graphs", "d_language", "d_general"]) assert.ok(!data.themeById.has(removed));
  assert.ok(layout.xById.get("ahn2015minimum") < layout.xById.get("ahn2020guiding"));
  const catflow = layout.stationByInstance.get("kim2026catflow");
  assert.deepEqual(catflow.themes.map((theme) => theme.id).sort(), ["d_materials", "d_molecules"]);
});

test("packing ignores source order and legacy coordinates, preserves chronology and clear labels", () => {
  const moved = { ...data, works: [...data.works].reverse().map((work) => ({ ...work, x: 1, y: 1, overview: false })) };
  assert.deepEqual(geometry(model.timelineLayout(moved)), geometry(layout));
  for (const [i, station] of layout.stations.entries()) {
    const label = model.labelBounds(station);
    assert.ok(label.top >= 8 && label.bottom <= layout.height - 8);
    for (const prior of layout.stations.slice(0, i)) {
      const other = model.labelBounds(prior);
      assert.ok(
        label.right <= other.left || other.right <= label.left || label.bottom <= other.top || other.bottom <= label.top,
        `${station.id} overlaps ${prior.id}'s label`
      );
      assert.ok(Math.hypot(station.x - prior.x, station.y - prior.y) >= 32, "hit targets have separate centers");
      assert.ok(!model.segmentHitsBox({ x: station.x - 11, y: station.y }, { x: station.x + 11, y: station.y }, other), "station avoids prior label");
    }
  }
  for (const row of layout.rows) {
    assert.equal(row.connections.length, row.stations.length - 1);
    row.connections.forEach((edge, i) => assert.deepEqual([edge.from, edge.to], [row.stations[i].id, row.stations[i + 1].id]));
  }
});

test("targeted display hints keep the busy green route flat and preserve other station heights", () => {
  const automatic = model.timelineLayout({ ...data, works: data.works.map(({ map_layout, ...work }) => work) });
  for (const station of layout.stations) {
    const hint = station.work.map_layout;
    if (hint) {
      assert.equal(station.y, hint.y);
      assert.equal(station.labelSide, hint.label_side);
      const bounds = model.labelBounds(station);
      assert.ok(bounds.top >= 8 && bounds.bottom <= layout.height - 8);
    } else assert.equal(station.y, automatic.stationByInstance.get(station.id).y, station.id);
  }
  const trunk = ["bu2024tackling", "jang2024pessimistic", "berto2025rlco", "woo2024iterated"].map((id) => layout.stationByInstance.get(id).y);
  assert.ok(Math.max(...trunk) - Math.min(...trunk) < 8, "UCom2, PBP-GFN, RL4CO and iEFM follow a nearly flat green trunk");
  assert.ok(layout.stationByInstance.get("seong2025transition").y > Math.max(...trunk) + 25, "TPS-DPS leaves room below the trunk");
});

test("routed lines avoid labels and unrelated stations while preserving left-to-right order", () => {
  for (const route of layout.routes) {
    for (const [point, id] of [
      [route.points[0], route.from],
      [route.points.at(-1), route.to],
    ]) {
      const { x, y } = model.stationPort(layout.stationByInstance.get(id), route.theme.id);
      assert.deepEqual(point, { x, y });
      assert.ok(Math.abs(y - layout.stationByInstance.get(id).y) <= 8, "line ends inside the shared station backplate");
    }
    for (const [i, point] of route.points.entries()) {
      assert.ok(point.y >= 8 && point.y <= layout.height - 8);
      if (!i) continue;
      const previous = route.points[i - 1];
      assert.ok(point.x >= previous.x, "base routes do not reverse chronology");
      for (const station of layout.stations) {
        assert.equal(model.segmentHitsBox(previous, point, model.labelBounds(station, 2)), false, `${route.id} crosses ${station.id}'s label`);
        if (![route.from, route.to].includes(station.work.id))
          assert.equal(model.segmentHitsBox(previous, point, model.nodeBounds(station, 2)), false, `${route.id} crosses ${station.id}'s symbol`);
      }
    }
  }
  assert.ok(
    layout.routes.some((route) => route.points.some((point, i) => i && point.y !== route.points[i - 1].y)),
    "domain routes can bend"
  );
});

test("route simplification favors clean metro angles and avoids excessive bends", () => {
  const bends = layout.routes.reduce((sum, route) => sum + route.points.length - 2, 0);
  assert.ok(bends < 240, `${bends} bends should remain below the original 431-bend routing`);
  let total = 0,
    regular = 0;
  for (const route of layout.routes)
    for (let i = 1; i < route.points.length; i++) {
      const dx = Math.abs(route.points[i].x - route.points[i - 1].x),
        dy = Math.abs(route.points[i].y - route.points[i - 1].y),
        length = Math.hypot(dx, dy);
      total += length;
      if (dx < 1e-5 || dy < 1e-5 || Math.abs(dx - dy) < 1e-5) regular += length;
    }
  assert.ok(regular / total > 0.95, "at least 95% of route length follows horizontal, vertical or 45-degree runs");
});

test("parallel domain tracks stay distinct without line jumps", () => {
  assert.ok(
    layout.routes.every((route) => !route.path.includes("A")),
    "crossings do not introduce semicircular jumps"
  );
  for (const station of layout.stations) {
    const ports = station.themes.map((theme) => model.stationPort(station, theme.id).y);
    assert.equal(new Set(ports).size, ports.length, "domains have separate entry and exit positions at shared stations");
  }
  const pairs = new Map();
  for (const route of layout.routes) {
    const key = `${route.from}/${route.to}`;
    if (!pairs.has(key)) pairs.set(key, []);
    pairs.get(key).push(route.path);
  }
  for (const paths of pairs.values()) assert.equal(new Set(paths).size, paths.length, "parallel domain routes remain individually visible");
});

test("sorting uses earliest eligible arXiv or acceptance dates and excludes earlier uncoauthored versions", () => {
  for (const work of data.works) {
    const dates = work.chronology.events.map((event) => (event.date.length === 7 ? `${event.date}-01` : event.date));
    assert.equal(model.chronologyKey(work)[0], dates.sort()[0], work.id);
  }
  for (const [id, date, version] of [
    ["ahn2018maximum", "2018-01-01", "1306.1167v2"],
    ["berto2025rlco", "2024-06-21", "2306.17100v4"],
    ["kim2024decoupled", "2024-05-27", "2402.05982v2"],
  ]) {
    const chronology = data.workById.get(id).chronology;
    assert.equal(chronology.date, date);
    assert.equal(chronology.version, version);
    assert.ok(chronology.excluded_arxiv_versions.length);
  }
  assert.equal(data.workById.get("ahn2019variational").chronology.date, "2019-03-02");
  assert.equal(data.workById.get("kim2024improving").chronology.date, "2024-05-01");
  assert.equal(data.workById.get("oh2026sctrilemma").chronology.date, "2026-09-24");
});

test("panning exposes every work including older publications, keeps figure height and creates no shortcuts", () => {
  for (const width of [220, 640]) {
    const seen = new Set(),
      originalRoutes = layout.rows.flatMap((row) => row.connections.map((edge) => edge.id));
    const max = model.timelineViewport(layout, 0, width).maxOffset;
    for (let offset = 0; offset <= max + 1; offset++) {
      const view = model.timelineViewport(layout, offset, width);
      for (const lane of view.lanes) {
        assert.equal(view.lanes.length, layout.lanes.length);
        assert.deepEqual([...lane.visibleThemes].sort(), [...new Set(lane.visibleStations.map((station) => station.theme.id))].sort());
        lane.visibleStations.forEach((station) => seen.add(station.work.id));
      }
    }
    assert.equal(seen.size, 74);
    assert.deepEqual(
      layout.rows.flatMap((row) => row.connections.map((edge) => edge.id)),
      originalRoutes
    );
  }
  assert.ok(
    model
      .timelineViewport(layout, 1000, 640)
      .lanes.flatMap((lane) => lane.visibleStations)
      .some((station) => station.work.year < 2024)
  );
});

test("new application classifications have checked experimental locators and distinct legend hues", () => {
  for (const [id, domain, locator] of [
    ["heo2024epic", "d_molecules", "BBBP"],
    ["cho2023multiresolution", "d_molecules", "QM9"],
    ["park2024nonbacktracking", "d_bio", "Peptides"],
    ["jang2023diffusion", "d_bio", "PPI"],
  ]) {
    const member = data.workById.get(id).memberships.find((member) => member.theme === domain);
    assert.ok(member && member.role === "evaluation" && member.locator.includes(locator), id);
  }
  for (const id of ["berto2025rlco", "bu2024tackling", "ahn2020learning", "jang2025selftraining", "kim2022what"]) {
    assert.ok(
      data.workById.get(id).memberships.some((member) => member.theme === "d_deep"),
      id
    );
  }
  assert.equal(new Set(layout.rows.map((row) => row.theme.display_color)).size, 8);
  const sequence = [...data.works].sort(model.compareChronology);
  for (let i = 1; i < sequence.length; i++) assert.ok(layout.xById.get(sequence[i].id) > layout.xById.get(sequence[i - 1].id));
});

test("every paper links directly to a recorded source and keeps secondary contributions off its station", () => {
  const fallbacks = [];
  for (const work of data.works) {
    const link = model.paperUrl(work);
    assert.match(link, /^https:\/\//);
    assert.ok(!link.includes("openreview.net"));
    if (!link.startsWith("https://arxiv.org/abs/")) fallbacks.push(work.map_label);
    const categories = model.contributionsFor(work, data);
    assert.equal(categories[0].id, work.map_contribution);
    assert.equal(new Set(categories.map((category) => category.id)).size, categories.length);
  }
  assert.deepEqual(fallbacks.sort(), ["BEAG", "DND", "DRIMA", "HoliMol", "STGG", "Wave-GD"].sort());
  const mask = data.workById.get("seong2026discovering");
  assert.deepEqual(
    model.contributionsFor(mask, data).map((category) => category.id),
    ["agents", "generation"]
  );
  assert.equal(model.contributionFor(mask, data).shape, "star");
  assert.equal(source.relationships.length, 92, "curated connections remain in canonical analysis data");
});

test("one editable contribution per work preserves authored symmetry choices and distinct domain targets", () => {
  assert.equal(data.contribution_categories.length, 8);
  assert.equal(new Set(data.contribution_categories.map((category) => category.shape)).size, 8);
  for (const work of data.works) {
    assert.ok(model.contributionFor(work, data));
    assert.ok(work.map_contribution_reason.length > 10);
    assert.ok(work.authors.length && work.authors.some((author) => author.includes("Sungsoo Ahn")));
    const station = layout.stationByInstance.get(work.id);
    assert.equal(station.work.id, work.id);
    assert.equal(station.primaryLabel, true);
  }
  for (const id of ["kim2026machine", "kim2025highorder", "kim2024gaussian"]) {
    assert.deepEqual(
      data.workById
        .get(id)
        .memberships.filter((member) => member.theme.startsWith("d_"))
        .map((member) => member.theme),
      ["d_electronic"]
    );
  }
  for (const id of ["kim2026machine", "kim2025highorder"]) assert.equal(model.contributionFor(data.workById.get(id), data).shape, "hexagon");
  const view = model.timelineViewport(layout, 0, 900);
  assert.ok(new Set(view.lanes.flatMap((lane) => lane.visibleStations.map((station) => station.work.id))).size >= 15);
  const ordered = model.contributionLegend(data);
  const firsts = ordered.map((category) => data.works.filter((work) => work.map_contribution === category.id).sort(model.compareChronology)[0]);
  firsts.slice(1).forEach((work, i) => assert.ok(model.compareChronology(firsts[i], work) < 0));
});
