/* Read-only diagnostic; station positions are derived from canonical data. */
import yaml from "js-yaml";
import fs from "node:fs";
import model from "../assets/js/research-map-model.js";
const read = (name) => yaml.load(fs.readFileSync(new URL(`../_data/${name}.yml`, import.meta.url), "utf8"));
const jsonMode = process.argv.includes("--json");
const input = jsonMode ? JSON.parse(fs.readFileSync(0, "utf8")) : null;
const source = input ? input.data : read("research_map"),
  pubs = input ? input.publications : read("publications");
const data = model.prepare({
  ...source,
  works: source.works.map((annotations) => ({ annotations, publications: annotations.publication_ids.map((id) => pubs.find((p) => p.id === id)) })),
});
const layout = model.timelineLayout(data);
const vertical = model.verticalTimelineLayout(data, layout);
if (jsonMode) {
  fs.writeFileSync(
    1,
    JSON.stringify({
      lanes: layout.lanes,
      rows: layout.rows,
      width: layout.width,
      height: layout.height,
      stations: layout.stations,
      positions: Object.fromEntries([...layout.stationByInstance].map(([id, station]) => [id, station.x])),
      x_by_id: Object.fromEntries(layout.xById),
      paper_urls: Object.fromEntries(data.works.map((work) => [work.id, model.paperUrl(work)])),
      ideas: model.ideaLayout(data, layout),
      vertical: {
        width: vertical.width,
        height: vertical.height,
        stations: vertical.stations,
        rows: vertical.rows,
        ideas: model.ideaLayout(data, vertical, true),
      },
    })
  );
  process.exit(0);
}
const gaps = layout.stations.slice(1).map((station, i) => station.x - layout.stations[i].x);
console.log(
  `${layout.rows.length} domains, ${layout.stations.length} shared stations; ${layout.width} × ${layout.height}px canvas, ${Math.min(...gaps).toFixed(1)}px minimum chronological spacing.`
);
console.log(`${layout.routes.length} sequential connections, ${layout.routes.reduce((sum, route) => sum + route.points.length - 2, 0)} bends.`);
console.log(
  `Vertical desktop: ${vertical.width} × ${vertical.height}px, ${vertical.routes.reduce((sum, route) => sum + route.points.length - 2, 0)} bends.`
);
