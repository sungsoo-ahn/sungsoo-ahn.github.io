/* Pure layout, conceptual connections and URL state, shared with the checks. */
(function (root) {
  "use strict";
  const SPACING = 120,
    PADDING = 72,
    ROW_HEIGHT = 68,
    STATION_Y = 56;
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
  // Pack compatible category spans, then fan new lanes above and below the
  // earliest lane. The global chronological sweep keeps dates aligned across rows.
  function timelineLayout(data) {
    const rows = timelineRows(data),
      index = new Map(data.taxonomy.map((theme, i) => [theme.id, i]));
    const lanes = [];
    const order = (a, b) =>
      a.start < b.start ? -1 : a.start > b.start ? 1 : a.end < b.end ? -1 : a.end > b.end ? 1 : a.theme.id < b.theme.id ? -1 : 1;
    for (const row of [...rows].sort(order)) {
      const compatible = lanes.filter((lane) => lane.end < row.start).sort((a, b) => a.count - b.count || a.index - b.index);
      if (compatible.length) {
        const lane = compatible[0];
        lane.rows.push(row);
        lane.end = row.end;
        lane.count += row.stations.length;
        lane.index = Math.min(lane.index, index.get(row.theme.id));
      } else lanes.push({ rows: [row], end: row.end, count: row.stations.length, index: index.get(row.theme.id) });
    }
    lanes.sort((a, b) => (a.rows[0].start < b.rows[0].start ? -1 : a.rows[0].start > b.rows[0].start ? 1 : a.index - b.index));
    const center = Math.floor(lanes.length / 2),
      slots = [center];
    for (let distance = 1; slots.length < lanes.length; distance++) {
      if (center - distance >= 0) slots.push(center - distance);
      if (center + distance < lanes.length) slots.push(center + distance);
    }
    const lanesByWork = new Map(),
      xById = new Map();
    lanes.forEach((lane, index) => {
      lane.slot = slots[index];
      lane.id = `lane_${lane.rows.map((row) => row.theme.id).join("_")}`;
      lane.stations = lane.rows
        .flatMap((row) => row.stations.map((work) => ({ work, theme: row.theme, id: `${row.theme.id}/${work.id}` })))
        .sort((a, b) => compareChronology(a.work, b.work));
      lane.stations.forEach((station) => {
        const { work } = station;
        if (!lanesByWork.has(work.id)) lanesByWork.set(work.id, []);
        lanesByWork.get(work.id).push(station);
      });
    });
    const stationByInstance = new Map();
    lanes.sort((a, b) => a.slot - b.slot);
    lanes.forEach((lane, index) =>
      lane.stations.forEach((station) => {
        station.lane = lane.id;
        station.y = index * ROW_HEIGHT + STATION_Y;
        station.labelDx = 0;
        stationByInstance.set(station.id, station);
      })
    );
    // Small, stable staggering makes repeated appearances distinguishable.
    // Only the topmost copy reserves text space; all copies keep their symbols.
    for (const copies of lanesByWork.values()) {
      copies.sort((a, b) => a.y - b.y);
      copies.forEach((station, i) => {
        station.dx = (i - (copies.length - 1) / 2) * 16;
        station.primaryLabel = i === 0;
        station.labelDx = 0;
        station.labelLines = nicknameLines(station.work.map_label || station.work.label);
        station.metaText = firstVenue(station.work);
        station.labelWidth = Math.max(...station.labelLines.map((line) => textWidth(line)), textWidth(station.metaText, 10));
        station.labelHalf = station.labelWidth / 2 + 3;
      });
    }
    const sequence = [...data.works].sort(compareChronology),
      stations = [...stationByInstance.values()];
    let extent = PADDING;
    function place() {
      const lastOnLane = new Map(),
        lastLabelOnLane = new Map(),
        placed = [],
        links = [];
      const range = ({ from: a, to: b }, top, bottom) => {
        const low = Math.max(top, a.y),
          high = Math.min(bottom, b.y);
        if (low > high) return null;
        const xs = [low, high].map((y) => a.x + ((b.x - a.x) * (y - a.y)) / (b.y - a.y));
        return [Math.min(...xs), Math.max(...xs)];
      };
      let previous = PADDING - 8,
        previousDate;
      for (const work of sequence) {
        const date = Date.parse(chronologyKey(work)[0]),
          copies = lanesByWork.get(work.id) || [],
          elapsed = previousDate === undefined ? 0 : (date - previousDate) / 86400000;
        const candidates = [
          PADDING,
          previous + Math.max(8, Math.min(80, (elapsed * 80) / 365.2425)) - Math.min(0, ...copies.map((station) => station.dx)),
          ...copies.flatMap((station) => {
            const last = lastOnLane.get(station.lane),
              lastLabel = lastLabelOnLane.get(station.lane);
            return [
              PADDING - station.dx,
              ...(last ? [last.x + 44 - station.dx] : []),
              ...(station.primaryLabel && lastLabel ? [lastLabel.x + lastLabel.labelHalf + station.labelHalf + 12 - station.dx] : []),
            ];
          }),
        ];
        // Reserve space only where a straight identity segment crosses another
        // paper's text or station. This avoids stretching unrelated rows.
        for (const station of copies) {
          for (const link of links) {
            const label = range(link, station.y - 54, station.y - 13),
              circle = range(link, station.y - 11, station.y + 11);
            if (label && station.primaryLabel) candidates.push(label[1] + station.labelHalf + 8 - station.dx);
            if (circle) candidates.push(circle[1] + 24 - station.dx);
          }
        }
        const relativeLinks = identityConnections(copies.map((station) => ({ ...station, x: station.dx })));
        for (const link of relativeLinks) {
          for (const station of placed) {
            const label = range(link, station.y - 54, station.y - 13),
              circle = range(link, station.y - 11, station.y + 11);
            if (label && station.primaryLabel) candidates.push(station.x + station.labelHalf + 8 - label[0]);
            if (circle) candidates.push(station.x + 24 - circle[0]);
          }
          for (const prior of links) {
            const low = Math.max(link.from.y, prior.from.y),
              high = Math.min(link.to.y, prior.to.y),
              a = range(link, low, high),
              b = range(prior, low, high);
            if (a && b) candidates.push(b[1] + 24 - a[0]);
          }
        }
        const x = Math.max(...candidates);
        xById.set(work.id, Math.round(x * 1000) / 1000);
        copies.forEach((station) => {
          station.x = Math.round((x + station.dx) * 1000) / 1000;
          lastOnLane.set(station.lane, station);
          if (station.primaryLabel) lastLabelOnLane.set(station.lane, station);
        });
        previous = Math.max(x, ...copies.map((station) => station.x));
        previousDate = date;
        placed.push(...copies);
        links.push(...identityConnections(copies));
      }
      extent = Math.max(...stations.map((station) => Math.max(station.x, station.x + station.labelDx))) + PADDING;
    }
    place();
    const width = Math.ceil(extent);
    return { lanes, rows, width, xById, stationByInstance };
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
  function identityConnections(stations, paper) {
    const groups = new Map();
    for (const station of stations) {
      if (paper !== undefined && station.work.id !== paper) continue;
      if (!groups.has(station.work.id)) groups.set(station.work.id, []);
      groups.get(station.work.id).push(station);
    }
    return [...groups].flatMap(([work, copies]) => {
      copies.sort((a, b) => a.y - b.y);
      return copies.slice(1).map((to, index) => ({
        id: `identity_${copies[index].id}_${to.id}`,
        from: copies[index],
        to,
        work,
        label: "Same paper",
        layer: "identity",
      }));
    });
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
    ROW_HEIGHT,
    STATION_Y,
    textWidth,
    nicknameLines,
    firstVenue,
    stationSymbol,
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
    identityConnections,
    detailConnections,
    focusGraph,
    contrast,
    colorSwatches,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = model;
  else root.ResearchMapModel = model;
})(globalThis);
