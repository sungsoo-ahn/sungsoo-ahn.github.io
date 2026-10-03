/* Pure layout, conceptual connections and URL state, shared with the checks. */
(function (root) {
  "use strict";
  const SPACING = 120,
    PADDING = 72,
    MAP_HEIGHT = 300;
  const legacyParameters = ["rm_view", "rm_lens", "rm_tags", "rm_expand", "rm_search", "rm_interpretive", "rm_zoom", "rm_compact"];
  // Arial advances at 12px. Additive metrics and a safety margin keep the
  // linked fallback and browser positions identical without font-load timing.
  const FONT_WIDTHS = [
    3.334, 3.334, 4.26, 6.674, 6.674, 10.67, 8.004, 2.292, 3.997, 3.997, 4.67, 7.008, 3.334, 3.997, 3.334, 3.334, 6.674, 6.674, 6.674, 6.674, 6.674,
    6.674, 6.674, 6.674, 6.674, 6.674, 3.334, 3.334, 7.008, 7.008, 7.008, 6.674, 12.182, 8.004, 8.004, 8.667, 8.667, 8.004, 7.331, 9.334, 8.667,
    3.334, 6, 8.004, 6.674, 9.997, 8.667, 9.334, 8.004, 9.334, 8.667, 8.004, 7.331, 8.667, 8.004, 11.327, 8.004, 8.004, 7.331, 3.334, 3.334, 3.334,
    5.631, 6.674, 3.997, 6.674, 6.674, 6, 6.674, 6.674, 3.334, 6.674, 6.674, 2.667, 2.667, 6, 2.667, 9.997, 6.674, 6.674, 6.674, 6.674, 3.997, 6,
    3.334, 6.674, 6, 8.667, 6, 6, 6, 4.008, 3.118, 4.008, 7.008,
  ];
  function textWidth(text, size = 12) {
    return (
      ([...String(text)].reduce((sum, char) => {
        const code = char.codePointAt(0);
        return sum + (code >= 32 && code <= 126 ? FONT_WIDTHS[code - 32] : char === "²" ? 4 : 12);
      }, 0) *
        size) /
      12
    );
  }
  function nicknameLines(text) {
    const lines = [""];
    for (const word of text.split(/\s+/)) {
      const next = `${lines.at(-1)} ${word}`.trim();
      if (lines.at(-1) && textWidth(next) > 106 && lines.length < 2) lines.push(word);
      else lines[lines.length - 1] = next;
    }
    return lines;
  }
  function firstVenue(work) {
    const edition = work.publications[0];
    return `${edition.venue || ""} ${edition.year || ""}`.trim();
  }
  function stationSymbol(shape) {
    const cross = "-3.5,-9 3.5,-9 3.5,-3.5 9,-3.5 9,3.5 3.5,3.5 3.5,9 -3.5,9 -3.5,3.5 -9,3.5 -9,-3.5 -3.5,-3.5";
    const shapes = {
      circle: ["circle", { r: 9 }],
      hexagon: ["polygon", { points: "-9,0 -4.5,-8 4.5,-8 9,0 4.5,8 -4.5,8" }],
      square: ["rect", { x: -8, y: -8, width: 16, height: 16 }],
      cross: ["polygon", { points: cross }],
      diagonal_cross: ["polygon", { points: cross, transform: "rotate(45)" }],
      triangle: ["polygon", { points: "0,-9 9,7 -9,7" }],
      diamond: ["polygon", { points: "0,-10 9,0 0,10 -9,0" }],
      star: [
        "polygon",
        {
          points: Array.from({ length: 10 }, (_, i) => {
            const angle = ((-90 + i * 36) * Math.PI) / 180,
              radius = i % 2 ? 5.8 : 10;
            return `${(Math.cos(angle) * radius).toFixed(3)},${(Math.sin(angle) * radius).toFixed(3)}`;
          }).join(" "),
        },
      ],
    };
    const [tag, attributes] = shapes[shape] || shapes.circle;
    return { tag, attributes };
  }
  function contributionFor(work, data) {
    return data.contributionById.get(work.map_contribution);
  }
  function contributionLegend(data) {
    return data.contribution_categories
      .map((category) => {
        const first = data.works.filter((work) => work.map_contribution === category.id).sort(compareChronology)[0];
        return { category, first };
      })
      .filter((item) => item.first)
      .sort((a, b) => compareChronology(a.first, b.first))
      .map((item) => item.category);
  }
  function prepare(raw) {
    const works = raw.works.map(({ annotations, publications }) => {
      const editions = [...publications].sort((a, b) => a.year - b.year || a.id.localeCompare(b.id));
      return { ...annotations, publications: editions, title: editions[0].title, authors: editions[0].authors || [], year: editions[0].year };
    });
    return {
      ...raw,
      contribution_categories: raw.contribution_categories || [],
      contributionById: new Map((raw.contribution_categories || []).map((category) => [category.id, category])),
      works,
      workById: new Map(works.map((work) => [work.id, work])),
      themeById: new Map(raw.taxonomy.map((theme) => [theme.id, theme])),
      aliases: new Map(works.flatMap((work) => work.publications.map((publication) => [publication.id, work.id]))),
    };
  }
  function sanitizeState(input, data) {
    const offset = Number(input.offset);
    return {
      paper: data.aliases.get(input.paper) || (data.workById.has(input.paper) ? input.paper : null),
      offset: Number.isFinite(offset) ? Math.max(0, offset) : 0,
    };
  }
  function readUrl(url, data) {
    const params = new URL(url).searchParams;
    return sanitizeState({ paper: params.get("rm_paper"), offset: params.get("rm_offset") }, data);
  }
  function writeUrl(url, state) {
    const result = new URL(url);
    legacyParameters.forEach((parameter) => result.searchParams.delete(parameter));
    if (state.paper) result.searchParams.set("rm_paper", state.paper);
    else result.searchParams.delete("rm_paper");
    if (state.offset > 0) result.searchParams.set("rm_offset", String(Math.round(state.offset * 1000) / 1000));
    else result.searchParams.delete("rm_offset");
    return result;
  }
  function chronologyKey(work) {
    const date = work.chronology?.date || String(work.year || 0).padStart(4, "0");
    return [
      date.length === 4 ? `${date}-01-01` : date.length === 7 ? `${date}-01` : date,
      (work.title || work.label || work.id).toLowerCase(),
      work.id,
    ];
  }
  function compareChronology(a, b) {
    const left = chronologyKey(a),
      right = chronologyKey(b);
    for (let i = 0; i < left.length; i++) {
      if (left[i] < right[i]) return -1;
      if (left[i] > right[i]) return 1;
    }
    return 0;
  }
  function mapRole(theme) {
    return theme.map_role || (theme.kind === "concept" ? "detail" : "major");
  }
  function chronologicalConnection(connection, data) {
    const a = data.workById.get(connection.from),
      b = data.workById.get(connection.to);
    const [from, to] = compareChronology(a, b) < 0 ? [a.id, b.id] : [b.id, a.id];
    return { ...connection, from, to, sameDate: chronologyKey(a)[0] === chronologyKey(b)[0] };
  }
  function timelineRows(data) {
    return data.taxonomy
      .filter((theme) => theme.kind === "domain" && mapRole(theme) === "major")
      .map((theme) => {
        const stations = data.works.filter((work) => work.memberships.some((member) => member.theme === theme.id)).sort(compareChronology);
        const connections = stations.slice(1).map((to, index) => ({
          id: `${theme.id}_${stations[index].id}_${to.id}`,
          from: stations[index].id,
          to: to.id,
          label: theme.label,
          theme,
          layer: "major",
          sameDate: chronologyKey(stations[index])[0] === chronologyKey(to)[0],
        }));
        return {
          theme,
          stations,
          connections,
          start: stations.length ? chronologyKey(stations[0])[0] : null,
          end: stations.length ? chronologyKey(stations.at(-1))[0] : null,
        };
      })
      .filter((row) => row.stations.length);
  }
  // Domain spans supply a soft vertical order. Stations may move away from it
  // to use empty space; no domain owns a permanent row.
  function domainAnchors(rows, data) {
    const order = (a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end) || a.theme.id.localeCompare(b.theme.id);
    const bands = [];
    for (const row of [...rows].sort(order)) {
      const band = bands.filter((item) => item.end < row.start).sort((a, b) => a.count - b.count)[0];
      if (band) {
        band.rows.push(row);
        band.end = row.end;
        band.count += row.stations.length;
      } else bands.push({ rows: [row], end: row.end, count: row.stations.length });
    }
    const center = Math.floor(bands.length / 2),
      slots = [center];
    for (let step = 1; slots.length < bands.length; step++) {
      if (center - step >= 0) slots.push(center - step);
      if (center + step < bands.length) slots.push(center + step);
    }
    const anchors = new Map();
    bands.forEach((band, i) =>
      band.rows.forEach((row) =>
        anchors.set(row.theme.id, bands.length === 1 ? MAP_HEIGHT / 2 : 56 + (slots[i] * (MAP_HEIGHT - 112)) / (bands.length - 1))
      )
    );
    return anchors;
  }
  function labelBounds(station, padding = 0) {
    const extent =
      station.labelSide === "below" ? [14, 44 + (station.labelLines.length - 1) * 13] : [-43 - (station.labelLines.length - 1) * 13, -13];
    return {
      left: station.x - station.labelHalf - padding,
      right: station.x + station.labelHalf + padding,
      top: station.y + (station.labelSide === "above" ? -1 : 1) * (station.labelGap || 0) + extent[0] - padding,
      bottom: station.y + (station.labelSide === "above" ? -1 : 1) * (station.labelGap || 0) + extent[1] + padding,
    };
  }
  function positionLabel(station) {
    station.labelBaselines = station.labelLines.map((_, i) =>
      station.labelSide === "above" ? -30 - station.labelGap - (station.labelLines.length - 1 - i) * 13 : 27 + station.labelGap + i * 13
    );
    station.metaY = station.labelSide === "above" ? -17 - station.labelGap : 40 + station.labelGap + (station.labelLines.length - 1) * 13;
  }
  function nodeBounds(station, padding = 0) {
    return { left: station.x - 13 - padding, right: station.x + 13 + padding, top: station.y - 13 - padding, bottom: station.y + 13 + padding };
  }
  function verticalOverlap(a, b, gap = 0) {
    return a.top < b.bottom + gap && b.top < a.bottom + gap;
  }
  function segmentHitsBox(a, b, box) {
    let low = 0,
      high = 1;
    for (const [axis, first, last] of [
      ["x", "left", "right"],
      ["y", "top", "bottom"],
    ]) {
      const delta = b[axis] - a[axis];
      if (Math.abs(delta) < 1e-8) {
        if (a[axis] <= box[first] || a[axis] >= box[last]) return false;
      } else {
        const t1 = (box[first] - a[axis]) / delta,
          t2 = (box[last] - a[axis]) / delta;
        low = Math.max(low, Math.min(t1, t2));
        high = Math.min(high, Math.max(t1, t2));
        if (low >= high - 1e-8) return false;
      }
    }
    return low < high;
  }
  function simplifyPoints(points) {
    const result = [];
    for (const point of points) {
      while (result.length > 1) {
        const a = result.at(-2),
          b = result.at(-1);
        if (Math.abs((b.x - a.x) * (point.y - b.y) - (b.y - a.y) * (point.x - b.x)) > 1e-6) break;
        result.pop();
      }
      result.push(point);
    }
    return result;
  }
  function stationPort(station, themeId) {
    const index = station.themes.findIndex((theme) => theme.id === themeId);
    const gap = station.themes.length === 2 ? 10 : 8;
    return { id: station.id, x: station.x, y: station.y + (index - (station.themes.length - 1) / 2) * gap };
  }
  function routePoints(a, b, stations, preferred, offset = 0, priorRoutes = [], existing = null, octilinear = true) {
    const boxes = stations
      .flatMap((station) => [labelBounds(station, 5), ...(station.id !== a.id && station.id !== b.id ? [nodeBounds(station, 5)] : [])])
      .filter((box) => box.right > a.x && box.left < b.x);
    const clear = (p, q) => !boxes.some((box) => segmentHitsBox(p, q, box));
    const inside = (p) => boxes.some((box) => p.x > box.left && p.x < box.right && p.y > box.top && p.y < box.bottom);
    const segments = priorRoutes.flatMap((route) =>
      route.points.slice(1).map((q, i) => [route.points[i], q, route.from === a.id || route.to === b.id])
    );
    const congestion = (p, q) => {
      const dx = q.x - p.x,
        dy = q.y - p.y,
        length = Math.hypot(dx, dy),
        ux = dx / length,
        uy = dy / length;
      let cost = 0;
      for (const [c, d, sharedTrack] of segments) {
        if (Math.max(c.x, d.x) < p.x || Math.min(c.x, d.x) > q.x) continue;
        const otherLength = Math.hypot(d.x - c.x, d.y - c.y);
        const sine = Math.abs(ux * (d.y - c.y) - uy * (d.x - c.x)) / otherLength;
        if (sine < 0.12) {
          const project = (r) => (r.x - p.x) * ux + (r.y - p.y) * uy;
          const first = Math.max(0, Math.min(project(c), project(d))),
            last = Math.min(length, Math.max(project(c), project(d)));
          const distance = Math.abs((c.x - p.x) * uy - (c.y - p.y) * ux);
          const gap = sharedTrack ? 7.5 : 11;
          if (last > first && distance < gap) cost += (last - first) * (1 - distance / gap) * 6;
        } else {
          const cross = crossingPoint(p, q, c, d);
          if (cross && ![a, b].some((station) => Math.hypot(station.x - cross.x, station.y - cross.y) < 20)) cost += 50;
        }
      }
      return cost;
    };
    const shoulder = Math.min(20, (b.x - a.x) / 3);
    const candidates = [a, b, { x: a.x + shoulder, y: a.y + offset }, { x: b.x - shoulder, y: b.y + offset }];
    for (const box of boxes) {
      for (const x of [box.left - 1, box.right + 1]) for (const y of [box.top - 1, box.bottom + 1]) candidates.push({ x, y });
    }
    for (const x of [...new Set(boxes.flatMap((box) => [box.left - 1, box.right + 1]))]) for (const y of [a.y, b.y]) candidates.push({ x, y });
    // A few clean horizontal rails provide room to separate parallel routes.
    for (const y of [...new Set([a.y + offset, b.y + offset, preferred, ...Array.from({ length: 12 }, (_, i) => 14 + i * 24)])])
      for (const x of [a.x + shoulder, b.x - shoulder]) candidates.push({ x, y });
    const nodes = [
      ...new Map(
        candidates
          .filter((p) => p.x >= a.x && p.x <= b.x && p.y >= 8 && p.y <= MAP_HEIGHT - 8 && !inside(p))
          .map((p) => [`${p.x}/${p.y}`, { x: p.x, y: p.y }])
      ).values(),
    ].sort((p, q) => p.x - q.x || p.y - q.y);
    const start = nodes.findIndex((p) => p.x === a.x && p.y === a.y),
      end = nodes.findIndex((p) => p.x === b.x && p.y === b.y);
    const states = nodes.map(() => []);
    states[start] = [{ cost: 0, point: nodes[start], angle: 0, previous: null, path: [] }];
    const patterns = (p, q) => {
      if (!octilinear) return [[p, q]];
      const dx = q.x - p.x,
        dy = q.y - p.y,
        shift = Math.abs(dy),
        direction = Math.sign(dy);
      if (shift < 1e-6) return [[p, q]];
      if (shift <= dx)
        return [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const x = p.x + (dx - shift) * fraction;
          return simplifyPoints(
            [p, { x, y: p.y }, { x: x + shift, y: q.y }, q].filter(
              (point, i, list) => !i || Math.hypot(point.x - list[i - 1].x, point.y - list[i - 1].y) > 1e-6
            )
          );
        });
      return [
        [p, { x: p.x, y: q.y - direction * dx }, q],
        [p, { x: q.x, y: p.y + direction * dx }, q],
      ].map(simplifyPoints);
    };
    for (let i = start + 1; i <= end; i++) {
      const q = nodes[i],
        options = [];
      for (let j = start; j < i; j++) {
        const p = nodes[j];
        if (!states[j].length || q.x - p.x < 0.5) continue;
        for (const path of patterns(p, q)) {
          if (path.slice(1).some((point, k) => !clear(path[k], point))) continue;
          const legs = path.slice(1).map((point, k) => ({ from: path[k], to: point, angle: Math.atan2(point.y - path[k].y, point.x - path[k].x) }));
          const first = legs[0].angle,
            last = legs.at(-1).angle;
          const trackCost =
            legs.reduce(
              (sum, leg) =>
                sum +
                congestion(leg.from, leg.to) +
                Math.hypot(leg.to.x - leg.from.x, leg.to.y - leg.from.y) +
                Math.abs((leg.from.y + leg.to.y) / 2 - (a.y + b.y) / 2) * (leg.to.x - leg.from.x) * 0.006,
              0
            ) +
            (legs.length - 1) * 32;
          for (const prior of states[j]) {
            const bend = prior.previous && Math.abs(first - prior.angle) > 0.03 ? 32 : 0;
            const departure = j === start && Math.abs(first) > 0.15 ? 14 : 0;
            const arrival = i === end && Math.abs(last) > 0.15 ? 14 : 0;
            options.push({ cost: prior.cost + trackCost + bend + departure + arrival, point: q, angle: last, previous: prior, path: path.slice(1) });
          }
        }
      }
      options.sort((p, q) => p.cost - q.cost);
      for (const candidate of options) {
        if (!states[i].some((prior) => Math.abs(prior.angle - candidate.angle) < 0.08)) states[i].push(candidate);
        if (states[i].length === 3) break;
      }
    }
    const winner = states[end][0];
    if (!winner && octilinear) return routePoints(a, b, stations, preferred, offset, priorRoutes, existing, false);
    if (!winner) throw new Error(`No clear route from ${a.id} to ${b.id}`);
    const parts = [];
    for (let item = winner; item; item = item.previous) parts.unshift(item.path);
    const result = simplifyPoints([{ x: a.x, y: a.y }, ...parts.flat()]);
    result[0] = { x: a.x, y: a.y };
    result[result.length - 1] = { x: b.x, y: b.y };
    if (existing) {
      const score = (points) => {
        const legs = points
          .slice(1)
          .map((point, i) => ({ from: points[i], to: point, angle: Math.atan2(point.y - points[i].y, point.x - points[i].x) }));
        return (
          legs.reduce(
            (sum, leg) =>
              sum +
              congestion(leg.from, leg.to) +
              Math.hypot(leg.to.x - leg.from.x, leg.to.y - leg.from.y) +
              Math.abs((leg.from.y + leg.to.y) / 2 - (a.y + b.y) / 2) * (leg.to.x - leg.from.x) * 0.006,
            0
          ) +
          (legs.length - 1) * 32 +
          (Math.abs(legs[0].angle) > 0.15 ? 14 : 0) +
          (Math.abs(legs.at(-1).angle) > 0.15 ? 14 : 0)
        );
      };
      if (score(result) >= score(existing) - 0.001) return existing;
    }
    return result;
  }
  function crossingPoint(a, b, c, d) {
    const dx = b.x - a.x,
      dy = b.y - a.y,
      ex = d.x - c.x,
      ey = d.y - c.y,
      den = dx * ey - dy * ex;
    if (Math.abs(den) < 1e-7) return null;
    const t = ((c.x - a.x) * ey - (c.y - a.y) * ex) / den,
      u = ((c.x - a.x) * dy - (c.y - a.y) * dx) / den;
    return t > 1e-5 && t < 1 - 1e-5 && u > 1e-5 && u < 1 - 1e-5 ? { x: a.x + t * dx, y: a.y + t * dy, t, u } : null;
  }
  function routePath(points) {
    const format = (point) => `${Number(point.x.toFixed(3))},${Number(point.y.toFixed(3))}`;
    let path = `M${format(points[0])}`;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1],
        b = points[i],
        length = Math.hypot(b.x - a.x, b.y - a.y);
      if (i === points.length - 1) path += `L${format(b)}`;
      else {
        const c = points[i + 1],
          nextLength = Math.hypot(c.x - b.x, c.y - b.y),
          radius = Math.min(7, length / 3, nextLength / 3);
        const before = { x: b.x - ((b.x - a.x) * radius) / length, y: b.y - ((b.y - a.y) * radius) / length };
        const after = { x: b.x + ((c.x - b.x) * radius) / nextLength, y: b.y + ((c.y - b.y) * radius) / nextLength };
        path += `L${format(before)}Q${format(b)} ${format(after)}`;
      }
    }
    return path;
  }
  // Constrained force layout: horizontal coordinates and label sides are pinned.
  // Edge springs and edge-to-edge repulsion can move papers only a little vertically.
  function relaxStations(stations, edges) {
    const byId = new Map(stations.map((station) => [station.id, station]));
    const original = new Map(stations.map((station) => [station.id, station.y]));
    const pairs = [...new Map(edges.map((edge) => [`${edge.from}/${edge.to}`, [byId.get(edge.from), byId.get(edge.to)]])).values()];
    const valid = (station) => {
      const label = labelBounds(station, 3),
        node = nodeBounds(station, 3);
      if (label.top < 8 || label.bottom > MAP_HEIGHT - 8) return false;
      for (const [a, b] of pairs) {
        if (a !== station && b !== station) continue;
        if (((b.y > a.y + 8 && a.labelSide === "below") || (b.y < a.y - 8 && a.labelSide === "above")) && b.x < labelBounds(a).right + 28)
          return false;
        if (((a.y < b.y - 8 && b.labelSide === "above") || (a.y > b.y + 8 && b.labelSide === "below")) && b.x < a.x + b.labelHalf + 28) return false;
      }
      const overlap = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
      return stations.every(
        (other) =>
          other === station ||
          (!overlap(label, labelBounds(other, 3)) &&
            !overlap(label, nodeBounds(other, 3)) &&
            !overlap(node, labelBounds(other, 3)) &&
            Math.hypot(station.x - other.x, station.y - other.y) >= 32)
      );
    };
    for (let iteration = 0; iteration < 100; iteration++) {
      const forces = new Map(stations.map((station) => [station.id, (original.get(station.id) - station.y) * 0.24]));
      const push = (station, value) => forces.set(station.id, forces.get(station.id) + value);
      for (let i = 0; i < stations.length; i++)
        for (let j = 0; j < i; j++) {
          const a = stations[i],
            b = stations[j],
            dx = a.x - b.x,
            dy = a.y - b.y,
            distance = Math.hypot(dx, dy);
          if (distance >= 64) continue;
          const force = (dy / distance) * (64 - distance) * 0.7;
          push(a, force);
          push(b, -force);
        }
      for (const [a, b] of pairs) {
        const spring = (b.y - a.y) * 0.075;
        push(a, spring);
        push(b, -spring);
      }
      for (let i = 0; i < pairs.length; i++)
        for (let j = 0; j < i; j++) {
          const [a, b] = pairs[i],
            [c, d] = pairs[j];
          const left = Math.max(a.x, c.x) + 24,
            right = Math.min(b.x, d.x) - 24;
          if (right <= left) continue;
          const midpoint = (left + right) / 2,
            t = (midpoint - a.x) / (b.x - a.x),
            u = (midpoint - c.x) / (d.x - c.x);
          const side = Math.sign(a.y + (b.y - a.y) * t - (c.y + (d.y - c.y) * u)) || (i % 2 ? 1 : -1);
          for (let k = 0; k < 5; k++) {
            const x = left + ((right - left) * (k + 0.5)) / 5,
              t = (x - a.x) / (b.x - a.x),
              u = (x - c.x) / (d.x - c.x);
            const gap = side * (a.y + (b.y - a.y) * t - (c.y + (d.y - c.y) * u));
            if (gap >= 18) continue;
            const force = side * Math.min(18, 18 - gap) * 0.16;
            push(a, force * (1 - t));
            push(b, force * t);
            push(c, -force * (1 - u));
            push(d, -force * u);
          }
        }
      for (const station of stations) {
        const previous = station.y,
          delta = Math.max(-1.5, Math.min(1.5, forces.get(station.id)));
        station.y = Math.max(original.get(station.id) - 28, Math.min(original.get(station.id) + 28, previous + delta));
        if (!valid(station)) station.y = previous;
      }
    }
    // Project blocked routing corridors back to their known-clear geometry.
    // This keeps a force pass from closing a narrow gap between two labels.
    for (let repair = 0; repair < stations.length; repair++) {
      const blocked = [];
      for (const [a, b] of pairs) {
        try {
          routePoints(a, b, stations, (a.y + b.y) / 2);
        } catch (error) {
          if (!error.message.startsWith("No clear route")) throw error;
          blocked.push([a.x, b.x]);
        }
      }
      if (!blocked.length) break;
      let changed = false;
      for (const station of stations)
        if (blocked.some(([left, right]) => station.x >= left - 80 && station.x <= right + 80) && station.y !== original.get(station.id)) {
          station.y = original.get(station.id);
          changed = true;
        }
      if (!changed) break;
    }
    stations.forEach((station) => (station.y = Number(station.y.toFixed(3))));
  }
  function timelineLayout(data) {
    const rows = timelineRows(data),
      anchors = domainAnchors(rows, data),
      sequence = [...data.works].sort(compareChronology);
    const stations = [],
      stationByInstance = new Map(),
      xById = new Map(),
      lastByTheme = new Map();
    let previous = PADDING - 20,
      previousDate;
    for (const work of sequence) {
      const members = work.memberships.filter((m) => anchors.has(m.theme)),
        themes = members.map((m) => data.themeById.get(m.theme)).sort((a, b) => anchors.get(a.id) - anchors.get(b.id) || a.id.localeCompare(b.id));
      const primary = members.find((m) => m.role === "central") || members[0];
      const preferred = themes.reduce((sum, theme) => sum + anchors.get(theme.id), 0) / Math.max(1, themes.length);
      const labelLines = nicknameLines(work.map_label || work.label),
        metaText = firstVenue(work),
        labelWidth = Math.max(...labelLines.map((line) => textWidth(line)), textWidth(metaText, 10));
      const date = Date.parse(chronologyKey(work)[0]),
        elapsed = previousDate === undefined ? 0 : (date - previousDate) / 86400000;
      const seed = Math.max(PADDING, previous + Math.max(20, Math.min(72, (elapsed * 64) / 365.2425)));
      let best;
      const ys = [...new Set([preferred, ...Array.from({ length: 25 }, (_, i) => 52 + i * 8)])];
      for (const y of ys)
        for (const labelSide of ["above", "below"]) {
          const station = {
            work,
            theme: data.themeById.get(primary?.theme) || themes[0],
            themes,
            id: work.id,
            x: seed,
            y,
            primaryLabel: true,
            labelDx: 0,
            labelSide,
            labelGap: 0,
            labelLines,
            metaText,
            labelWidth,
            labelHalf: labelWidth / 2 + 4,
          };
          const label = labelBounds(station),
            circle = nodeBounds(station);
          if (label.top < 8 || label.bottom > MAP_HEIGHT - 8) continue;
          for (const prior of stations) {
            const priorLabel = labelBounds(prior),
              priorCircle = nodeBounds(prior);
            if (verticalOverlap(label, priorLabel, 12)) station.x = Math.max(station.x, priorLabel.right + station.labelHalf + 16);
            if (verticalOverlap(label, priorCircle, 4)) station.x = Math.max(station.x, priorCircle.right + station.labelHalf + 8);
            if (verticalOverlap(circle, priorLabel, 4)) station.x = Math.max(station.x, priorLabel.right + 21);
            if (Math.abs(y - prior.y) < 40) station.x = Math.max(station.x, prior.x + Math.max(44, prior.labelHalf + 36, station.labelHalf + 36));
          }
          for (const theme of themes) {
            const prior = lastByTheme.get(theme.id);
            if (!prior) continue;
            if ((y > prior.y + 8 && prior.labelSide === "below") || (y < prior.y - 8 && prior.labelSide === "above"))
              station.x = Math.max(station.x, labelBounds(prior).right + 28);
            if ((prior.y < y - 8 && labelSide === "above") || (prior.y > y + 8 && labelSide === "below"))
              station.x = Math.max(station.x, prior.x + station.labelHalf + 28);
          }
          const continuity =
            themes.reduce((sum, theme) => sum + (lastByTheme.has(theme.id) ? Math.abs(y - lastByTheme.get(theme.id).y) : 0), 0) /
            Math.max(1, themes.length);
          const score = station.x - seed + Math.abs(y - preferred) * 0.6 + continuity * 0.2 + (labelSide === "below" ? 3 : 0);
          if (!best || score < best.score) best = { station, score };
        }
      const station = best.station;
      positionLabel(station);
      station.x = Number(station.x.toFixed(3));
      station.y = Number(station.y.toFixed(3));
      stations.push(station);
      stationByInstance.set(work.id, station);
      xById.set(work.id, station.x);
      themes.forEach((theme) => lastByTheme.set(theme.id, station));
      previous = station.x;
      previousDate = date;
    }
    const routes = rows.flatMap((row) => row.connections.map((connection) => ({ ...connection, preferred: anchors.get(row.theme.id) })));
    relaxStations(stations, routes);
    // Curated display hints refine crowded areas after the automatic packing.
    // Preserve the rest of the vertical placement, then reflow only the space
    // needed to keep captions and station symbols clear.
    for (const station of stations)
      if (station.work.map_layout) {
        const hint = station.work.map_layout;
        station.y = hint.y;
        station.labelSide = hint.label_side || station.labelSide;
        station.labelGap = hint.label_gap || 0;
      }
    let hintPadding = 0;
    const placed = [],
      lastPlacedByTheme = new Map();
    for (const station of stations) {
      station.x += hintPadding;
      const originalX = station.x,
        label = labelBounds(station),
        circle = nodeBounds(station);
      for (const prior of placed) {
        if (!station.work.map_layout && !prior.work.map_layout) continue;
        const priorLabel = labelBounds(prior),
          priorCircle = nodeBounds(prior);
        if (verticalOverlap(label, priorLabel, 12)) station.x = Math.max(station.x, priorLabel.right + station.labelHalf + 16);
        if (verticalOverlap(label, priorCircle, 4)) station.x = Math.max(station.x, priorCircle.right + station.labelHalf + 8);
        if (verticalOverlap(circle, priorLabel, 4)) station.x = Math.max(station.x, priorLabel.right + 21);
        if (Math.abs(station.y - prior.y) < 40) station.x = Math.max(station.x, prior.x + Math.max(44, prior.labelHalf + 36, station.labelHalf + 36));
      }
      for (const theme of station.themes) {
        const prior = lastPlacedByTheme.get(theme.id);
        if (!prior || (!station.work.map_layout && !prior.work.map_layout)) continue;
        if ((station.y > prior.y + 8 && prior.labelSide === "below") || (station.y < prior.y - 8 && prior.labelSide === "above"))
          station.x = Math.max(station.x, labelBounds(prior).right + 28);
        if ((prior.y < station.y - 8 && station.labelSide === "above") || (prior.y > station.y + 8 && station.labelSide === "below"))
          station.x = Math.max(station.x, prior.x + station.labelHalf + 28);
      }
      hintPadding += station.x - originalX;
      positionLabel(station);
      placed.push(station);
      station.themes.forEach((theme) => lastPlacedByTheme.set(theme.id, station));
    }
    // Spend horizontal space only where steep branches leave a shared station.
    // The earlier packing and force pass still determine the vertical placement.
    let junctionPadding = 0;
    const previousByTheme = new Map();
    for (const station of stations) {
      station.x += junctionPadding;
      let extra = 0;
      for (const theme of station.themes) {
        const previous = previousByTheme.get(theme.id);
        if (!previous || Math.max(previous.themes.length, station.themes.length) < 2) continue;
        const dy = Math.abs(stationPort(station, theme.id).y - stationPort(previous, theme.id).y);
        if (dy > 30) extra = Math.max(extra, Math.min(56, 28 + dy * 0.35) - (station.x - previous.x));
      }
      station.x = Number((station.x + extra).toFixed(3));
      junctionPadding += extra;
      xById.set(station.id, station.x);
      station.themes.forEach((theme) => previousByTheme.set(theme.id, station));
    }
    const pairs = new Map();
    routes.forEach((route) => {
      const key = `${route.from}/${route.to}`;
      if (!pairs.has(key)) pairs.set(key, []);
      pairs.get(key).push(route);
    });
    const routed = [];
    const ordered = [...routes].sort((a, b) => xById.get(b.to) - xById.get(b.from) - (xById.get(a.to) - xById.get(a.from)));
    for (const route of ordered) {
      const peers = pairs.get(`${route.from}/${route.to}`),
        i = peers.indexOf(route);
      route.points = routePoints(
        stationPort(stationByInstance.get(route.from), route.theme.id),
        stationPort(stationByInstance.get(route.to), route.theme.id),
        stations,
        route.preferred,
        (i - (peers.length - 1) / 2) * 7,
        routed
      );
      routed.push(route);
    }
    // Relax route congestion once more against the full set, accepting only a
    // shorter, simpler or less congested result for each line.
    for (const route of [...routes].sort((a, b) => b.points.length - a.points.length)) {
      const peers = pairs.get(`${route.from}/${route.to}`),
        i = peers.indexOf(route);
      route.points = routePoints(
        stationPort(stationByInstance.get(route.from), route.theme.id),
        stationPort(stationByInstance.get(route.to), route.theme.id),
        stations,
        route.preferred,
        (i - (peers.length - 1) / 2) * 7,
        routes.filter((other) => other !== route),
        route.points
      );
    }
    routes.forEach((route) => (route.path = routePath(route.points)));
    rows.forEach((row) => (row.connections = routes.filter((route) => route.theme.id === row.theme.id)));
    const width = Math.ceil(Math.max(PADDING, ...stations.map((station) => station.x + station.labelHalf)) + PADDING);
    return { rows, routes, stations, height: MAP_HEIGHT, width, xById, stationByInstance, lanes: [{ id: "network", rows, stations }] };
  }

  function timelineViewport(layout, offset, viewportWidth) {
    const width = Math.max(1, viewportWidth),
      maxOffset = Math.max(0, layout.width - width) / SPACING;
    const clamped = Math.max(0, Math.min(maxOffset, Number(offset) || 0)),
      left = (maxOffset - clamped) * SPACING,
      right = left + width;
    const lanes = layout.lanes.map((lane) => {
      const stations = lane.stations.filter((station) => station.x >= left + 16 && station.x <= right - 16);
      return { ...lane, visibleStations: stations, visibleThemes: new Set(stations.map((station) => station.theme.id)) };
    });
    return { lanes, offset: clamped, maxOffset, left, right };
  }
  function detailConnections(paper, data) {
    if (!paper || !data.workById.has(paper)) return [];
    const connections = data.relationships
      .filter((relation) => [relation.from, relation.to].includes(paper))
      .map((relation) => chronologicalConnection({ ...relation, themes: [], layer: "detail" }, data));
    for (const theme of data.taxonomy.filter((theme) => theme.kind === "concept" && mapRole(theme) === "detail")) {
      const members = data.works.filter((work) => work.memberships.some((member) => member.theme === theme.id)).sort(compareChronology);
      if (members.length < 2 || members.length > 3 || !members.some((work) => work.id === paper)) continue;
      const parallel = members.some((work) => work.memberships.some((member) => member.theme === theme.id && member.role === "parallel"));
      for (let i = 1; i < members.length; i++) {
        if (![members[i - 1].id, members[i].id].includes(paper)) continue;
        connections.push(
          chronologicalConnection(
            {
              id: `detail_${theme.id}_${members[i - 1].id}_${members[i].id}`,
              from: members[i - 1].id,
              to: members[i].id,
              label: theme.label,
              map_label: theme.map_label || theme.label,
              themes: [theme.id],
              status: parallel ? "interpretive" : "local",
              layer: "detail",
              group: members.map((work) => work.id),
              explanation: theme.description,
              map_explanation: theme.map_description || theme.description,
            },
            data
          )
        );
      }
    }
    return connections;
  }
  function focusGraph(paper, data) {
    const selected = data.workById.get(paper);
    if (!selected) return { selected: null, connections: [], earlier: [], later: [] };
    const peers = new Map();
    for (const reason of detailConnections(paper, data)) {
      const peer = data.workById.get(reason.from === paper ? reason.to : reason.from);
      if (!peers.has(peer.id))
        peers.set(peer.id, {
          id: `focus_${paper}_${peer.id}`,
          from: reason.from,
          to: reason.to,
          peer,
          reasons: [],
          layer: "detail",
          sameDate: reason.sameDate,
        });
      peers.get(peer.id).reasons.push(reason);
    }
    const connections = [...peers.values()]
      .sort((a, b) => compareChronology(a.peer, b.peer))
      .map((connection) => {
        const concepts = new Map();
        connection.reasons.forEach((reason) => {
          const name = reason.map_label || reason.label;
          if (!concepts.has(name)) concepts.set(name, { label: name, reasons: [] });
          concepts.get(name).reasons.push(reason);
        });
        return {
          ...connection,
          concepts: [...concepts.values()],
          label: [...concepts.keys()].join(" · "),
          status: connection.reasons.every((reason) => reason.status === "interpretive") ? "interpretive" : "documented",
        };
      });
    return {
      selected,
      connections,
      earlier: connections.filter((connection) => compareChronology(connection.peer, selected) < 0),
      later: connections.filter((connection) => compareChronology(connection.peer, selected) > 0),
    };
  }
  function contrast(a, b) {
    const luminance = (hex) => {
      const values = hex
        .match(/[a-f\d]{2}/gi)
        .map((value) => parseInt(value, 16) / 255)
        .map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4));
      return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722;
    };
    const left = luminance(a),
      right = luminance(b);
    return (Math.max(left, right) + 0.05) / (Math.min(left, right) + 0.05);
  }
  function colorSwatches(hex) {
    const base = hex.match(/[a-f\d]{2}/gi).map((value) => parseInt(value, 16));
    const adjust = (background, target) => {
      let result = hex;
      for (let i = 0; contrast(result, background) < 3.1 && i < 20; i++)
        result =
          "#" +
          base
            .map((value) =>
              Math.round(value + ((target - value) * (i + 1)) / 20)
                .toString(16)
                .padStart(2, "0")
            )
            .join("");
      return result;
    };
    return { light: adjust("#ffffff", 0), dark: adjust("#21192b", 255) };
  }
  const model = {
    SPACING,
    PADDING,
    MAP_HEIGHT,
    labelBounds,
    nodeBounds,
    segmentHitsBox,
    routePath,
    textWidth,
    nicknameLines,
    firstVenue,
    stationSymbol,
    stationPort,
    contributionFor,
    contributionLegend,
    prepare,
    sanitizeState,
    readUrl,
    writeUrl,
    chronologyKey,
    compareChronology,
    mapRole,
    chronologicalConnection,
    timelineRows,
    timelineLayout,
    timelineViewport,
    detailConnections,
    focusGraph,
    contrast,
    colorSwatches,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = model;
  else root.ResearchMapModel = model;
})(globalThis);
