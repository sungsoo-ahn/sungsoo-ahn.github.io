(function () {
  "use strict";
  const root = document.querySelector(".research-map");
  if (!root) return;
  const model = window.ResearchMapModel;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const vector = (tag, attributes = {}) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
    return node;
  };
  const scroller = root.querySelector(".rm-overview .rm-timeline-scroll"),
    track = root.querySelector(".rm-overview .rm-timeline"),
    navigation = root.querySelector(".rm-navigation"),
    earlier = navigation.querySelector(".rm-earlier"),
    later = navigation.querySelector(".rm-later");
  let data, state, layout, viewport, tooltip;
  let hoverAnchor, hoverPaper, hoverConnection, hideTimer, frame;
  const nickname = (work) => work.map_label || work.label;
  const venueYears = (work) => work.publications.map((p) => `${p.venue} ${p.year}`).join(" · ");
  const firstVenue = model.firstVenue;
  const contributionMark = (work, className = "rm-station") => {
    const category = model.contributionFor(work, data),
      symbol = model.stationSymbol(category?.shape);
    return vector(symbol.tag, { ...symbol.attributes, class: className, "data-shape": category?.shape || "circle" });
  };
  const themed = (node, theme) => {
    const swatches = model.colorSwatches(theme.display_color || data.colors[theme.color].hex);
    node.classList.add("rm-themed");
    node.style.setProperty("--rm-color-light", swatches.light);
    node.style.setProperty("--rm-color-dark", swatches.dark);
    return node;
  };
  const viewportWidth = () => Math.max(1, scroller.clientWidth);
  const panStep = () => Math.min(3, Math.max(1, Math.floor((viewportWidth() - 64) / model.SPACING)));
  const persist = () => history.replaceState({}, "", model.writeUrl(location.href, state));

  function hideTooltip() {
    clearTimeout(hideTimer);
    hoverAnchor?.removeAttribute("aria-describedby");
    hoverAnchor = hoverPaper = hoverConnection = null;
    if (tooltip) tooltip.hidden = true;
    updateHighlights();
  }
  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideTooltip, 180);
  }
  function positionTooltip(event) {
    if (!hoverAnchor || tooltip.hidden) return;
    const anchor = hoverAnchor.getBoundingClientRect(),
      box = tooltip.getBoundingClientRect();
    const x = Number.isFinite(event?.clientX) ? event.clientX : anchor.left + anchor.width / 2;
    const y = Number.isFinite(event?.clientY) ? event.clientY : anchor.bottom;
    let left = x + 14,
      top = y + 14;
    if (left + box.width > innerWidth - 12) left = x - box.width - 14;
    if (top + box.height > innerHeight - 12) top = y - box.height - 14;
    tooltip.style.left = `${Math.max(12, Math.min(left, innerWidth - box.width - 12))}px`;
    tooltip.style.top = `${Math.max(12, Math.min(top, innerHeight - box.height - 12))}px`;
  }
  function showTooltip(anchor, item, kind, event) {
    clearTimeout(hideTimer);
    hoverAnchor?.removeAttribute("aria-describedby");
    hoverAnchor = anchor;
    anchor.setAttribute("aria-describedby", tooltip.id);
    tooltip.replaceChildren();
    hoverPaper = kind === "paper" ? item.id : null;
    hoverConnection = kind === "connection" ? item : null;
    if (kind === "paper") {
      tooltip.append(
        element("div", item.title, "rm-tooltip-title"),
        element("div", item.summary, "rm-tooltip-summary"),
        element("div", item.authors.join(", "), "rm-tooltip-authors"),
        element("div", venueYears(item), "rm-tooltip-meta")
      );
      const contribution = model.contributionFor(item, data);
      if (contribution) {
        const caption = element("div", undefined, "rm-tooltip-contribution"),
          symbol = vector("svg", { viewBox: "-12 -12 24 24", "aria-hidden": "true" });
        symbol.append(contributionMark(item, "rm-symbol"));
        const primary = element("strong", contribution.label);
        primary.dataset.contribution = contribution.id;
        caption.append(symbol, primary);
        for (const category of model.contributionsFor(item, data).slice(1)) {
          const secondary = element("span", category.label, "rm-tooltip-secondary");
          secondary.dataset.contribution = category.id;
          caption.append(secondary);
        }
        tooltip.append(caption);
      }
      const chips = element("div", undefined, "rm-tooltip-themes");
      for (const member of item.memberships) {
        const theme = data.themeById.get(member.theme);
        if (!["domain", "method"].includes(theme.kind)) continue;
        const chip = themed(element("span", undefined, "rm-tooltip-theme"), theme);
        chip.append(element("i"), element("span", theme.label));
        chips.append(chip);
      }
      tooltip.append(chips);
    } else {
      tooltip.append(element("div", item.label, "rm-tooltip-title"));
      for (const id of [item.from, item.to]) {
        const work = data.workById.get(id);
        tooltip.append(element("div", `${nickname(work)} · ${firstVenue(work)}`, "rm-tooltip-meta"));
      }
    }
    tooltip.hidden = false;
    updateHighlights();
    positionTooltip(event);
  }
  function bindHover(node, item, kind) {
    node.addEventListener("pointerenter", (event) => showTooltip(node, item, kind, event));
    node.addEventListener("pointermove", (event) => positionTooltip(event));
    node.addEventListener("pointerleave", scheduleHide);
    node.addEventListener("focus", () => {
      if (node.matches(".rm-overview .rm-work")) {
        const frame = scroller.getBoundingClientRect(),
          mark = node.querySelector(".rm-hit").getBoundingClientRect();
        const left = frame.left + 16;
        if (mark.left < left) scroller.scrollLeft -= left - mark.left;
        else if (mark.right > frame.right - 16) scroller.scrollLeft += mark.right - frame.right + 16;
        updateViewport();
      }
      showTooltip(node, item, kind);
    });
    node.addEventListener("blur", scheduleHide);
  }
  function updateHighlights() {
    root.querySelectorAll(".rm-work").forEach((node) => {
      node.dataset.highlighted = String(
        node.dataset.work === hoverPaper || Boolean(hoverConnection && [hoverConnection.from, hoverConnection.to].includes(node.dataset.work))
      );
    });
    root.querySelectorAll(".rm-connection").forEach((node) => {
      node.dataset.highlighted = String(
        node.dataset.connection === hoverConnection?.id || Boolean(hoverPaper && [node.dataset.from, node.dataset.to].includes(hoverPaper))
      );
    });
  }
  function connectionNode(connection, path) {
    const group = vector("g", {
      class: "rm-connection",
      "data-connection": connection.id,
      "data-from": connection.from,
      "data-to": connection.to,
      "data-status": connection.status || "major",
    });
    group.append(
      vector("path", { class: "rm-connection-casing", d: path, "aria-hidden": "true" }),
      vector("path", { class: "rm-connection-path", d: path }),
      vector("path", { class: "rm-connection-hit", d: path })
    );
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "img");
    group.setAttribute("aria-label", connection.label);
    bindHover(group, connection, "connection");
    return group;
  }
  function buildOverview() {
    track.style.setProperty("--rm-track-width", `${layout.width}px`);
    const svg = vector("svg", { width: layout.width, height: layout.height, class: "rm-network" });
    for (const themeRow of layout.rows) {
      const route = themed(vector("g", { class: "rm-route", "data-theme": themeRow.theme.id }), themeRow.theme);
      for (const connection of themeRow.connections) route.append(connectionNode(connection, connection.path));
      svg.append(route);
    }
    for (const station of layout.stations) {
      const { work, theme, x, y } = station;
      const group = themed(
        vector("a", {
          class: "rm-work",
          href: model.paperUrl(work),
          target: "_blank",
          rel: "external nofollow noopener",
          tabindex: "0",
          transform: `translate(${x},${y})`,
          "data-work": work.id,
          "data-instance": station.id,
          "data-theme": theme.id,
          "data-themes": station.themes.map((item) => item.id).join(","),
          "data-contribution": work.map_contribution,
          "data-primary-label": "true",
          "data-x": x,
          "data-y": y,
          "data-label-side": station.labelSide,
          "aria-label": `${nickname(work)}, ${venueYears(work)}`,
        }),
        theme
      );
      group.append(
        vector("circle", { class: "rm-hit", r: 16 }),
        vector("circle", { class: "rm-station-backplate", r: 11, "aria-hidden": "true" }),
        contributionMark(work)
      );
      const label = vector("text", { class: "rm-work-label", "text-anchor": "middle" });
      station.labelLines.forEach((line, i) => {
        const span = vector("tspan", { x: 0, y: station.labelBaselines[i] });
        span.textContent = line;
        label.append(span);
      });
      const meta = vector("text", { class: "rm-work-meta", "text-anchor": "middle", y: station.metaY });
      meta.textContent = station.metaText;
      group.append(label, meta);
      group.dataset.labelWidth = station.labelWidth + 8;
      group.dataset.labelX = x;
      bindHover(group, work, "paper");
      svg.append(group);
    }
    track.append(svg);
  }
  function updateViewport() {
    if (!data) return;
    const max = Math.max(0, layout.width - viewportWidth());
    state.offset = Math.max(0, max - scroller.scrollLeft) / model.SPACING;
    viewport = model.timelineViewport(layout, state.offset, viewportWidth());
    const visibleStations = new Set(viewport.lanes[0].visibleStations.map((station) => station.id));
    track.querySelectorAll(".rm-work").forEach((node) => {
      const visible = visibleStations.has(node.dataset.instance);
      node.toggleAttribute("hidden", !visible);
      const x = Number(node.dataset.labelX),
        half = Number(node.dataset.labelWidth) / 2;
      const clipped = x - half < viewport.left + 4 || x + half > viewport.right - 4;
      for (const selector of [".rm-work-label", ".rm-work-meta"]) node.querySelector(selector).toggleAttribute("hidden", !visible || clipped);
    });
    track.querySelectorAll(".rm-connection").forEach((node) => {
      const a = layout.stationByInstance.get(node.dataset.from),
        b = layout.stationByInstance.get(node.dataset.to);
      node.toggleAttribute("hidden", a.x > viewport.right || b.x < viewport.left);
    });
    earlier.disabled = state.offset >= viewport.maxOffset - 0.01;
    later.disabled = state.offset <= 0.01;
    if (hoverAnchor?.closest("[hidden]")) hideTooltip();
    else positionTooltip();
    persist();
  }
  function setOffset(offset) {
    const max = Math.max(0, layout.width - viewportWidth());
    state.offset = Math.max(0, Math.min(max / model.SPACING, offset));
    scroller.scrollLeft = max - state.offset * model.SPACING;
    updateViewport();
  }
  earlier.addEventListener("click", () => {
    hideTooltip();
    setOffset(state.offset + panStep());
  });
  later.addEventListener("click", () => {
    hideTooltip();
    setOffset(state.offset - panStep());
  });
  scroller.addEventListener("scroll", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(updateViewport);
  });
  scroller.addEventListener("keydown", (event) => {
    const offsets = { ArrowLeft: state.offset + panStep(), ArrowRight: state.offset - panStep(), Home: Infinity, End: 0 };
    if (!(event.key in offsets)) return;
    event.preventDefault();
    hideTooltip();
    setOffset(offsets[event.key]);
  });
  window.addEventListener("popstate", () => {
    if (!data) return;
    hideTooltip();
    state = model.readUrl(location.href);
    setOffset(state.offset);
  });
  window.addEventListener("scroll", () => positionTooltip(), { passive: true });
  root.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
  async function initialize() {
    try {
      const response = await fetch(root.dataset.source);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = model.prepare(await response.json());
      state = model.readUrl(location.href);
      layout = model.timelineLayout(data);
      tooltip = element("div", undefined, "rm-tooltip");
      tooltip.id = "research-map-tooltip";
      tooltip.setAttribute("role", "tooltip");
      tooltip.hidden = true;
      tooltip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
      tooltip.addEventListener("pointerleave", scheduleHide);
      tooltip.addEventListener("focusin", () => clearTimeout(hideTimer));
      tooltip.addEventListener("focusout", scheduleHide);
      root.append(tooltip);
      buildOverview();
      root.querySelector(".rm-static").hidden = true;
      root.querySelector(".rm-interactive").hidden = false;
      root.querySelector(".rm-load-status").hidden = true;
      navigation.hidden = false;
      setOffset(state.offset);
      let previousWidth = scroller.clientWidth;
      new ResizeObserver(() => {
        if (scroller.clientWidth !== previousWidth) {
          previousWidth = scroller.clientWidth;
          setOffset(state.offset);
        }
      }).observe(root);
      root.dataset.ready = "true";
    } catch (error) {
      root.querySelector(".rm-load-status").textContent = "Interactive map unavailable.";
      console.error("Research map:", error);
    }
  }
  initialize();
})();
