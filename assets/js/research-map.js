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
  let data, state, viewport, tooltip, mapWidth;
  let stations = [],
    connections = [];
  let hoverAnchor, hoverPaper, hoverConnection, hideTimer, frame;
  let previousLeft, previousWidth;
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
  const persist = () => {
    const url = model.writeUrl(location.href, state).href;
    if (url !== location.href) history.replaceState({}, "", url);
  };

  function hideTooltip() {
    clearTimeout(hideTimer);
    if (!hoverAnchor) return;
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
    stations.forEach(({ node, work }) => {
      const highlighted = String(work.id === hoverPaper || Boolean(hoverConnection && [hoverConnection.from, hoverConnection.to].includes(work.id)));
      if (node.dataset.highlighted !== highlighted) node.dataset.highlighted = highlighted;
    });
    connections.forEach(({ node, connection }) => {
      const highlighted = String(
        connection.id === hoverConnection?.id || Boolean(hoverPaper && [connection.from, connection.to].includes(hoverPaper))
      );
      if (node.dataset.highlighted !== highlighted) node.dataset.highlighted = highlighted;
    });
  }
  function bindOverview() {
    // Geometry is computed by the generator. Reuse its SVG rather than solving
    // and rendering the same layout again on the browser's main thread.
    mapWidth = Number(track.querySelector(".rm-network").getAttribute("width"));
    stations = [...track.querySelectorAll(".rm-static-paper")].map((node) => {
      const work = data.workById.get(node.dataset.work);
      node.classList.replace("rm-static-paper", "rm-work");
      node.setAttribute("tabindex", "0");
      node.querySelector("title")?.remove();
      bindHover(node, work, "paper");
      return {
        node,
        work,
        x: Number(node.dataset.x),
        labelX: Number(node.dataset.labelX),
        labelHalf: Number(node.dataset.labelWidth) / 2,
        labels: [...node.querySelectorAll(".rm-work-label, .rm-work-meta")],
      };
    });
    const stationById = new Map(stations.map((station) => [station.node.dataset.instance, station]));
    connections = [...track.querySelectorAll(".rm-connection")].map((node) => {
      const theme = data.themeById.get(node.closest(".rm-route").dataset.theme);
      const connection = { id: node.dataset.connection, from: node.dataset.from, to: node.dataset.to, label: theme.label };
      node.querySelector("title")?.remove();
      node.append(vector("path", { class: "rm-connection-hit", d: node.querySelector(".rm-connection-path").getAttribute("d") }));
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "img");
      node.setAttribute("aria-label", connection.label);
      bindHover(node, connection, "connection");
      return { node, connection, fromX: stationById.get(connection.from).x, toX: stationById.get(connection.to).x };
    });
  }
  function updateViewport() {
    if (!mapWidth) return;
    const width = viewportWidth(),
      max = Math.max(0, mapWidth - width);
    const left = Math.max(0, Math.min(max, scroller.scrollLeft));
    state.offset = (max - left) / model.SPACING;
    // A programmatic pan also emits a scroll event. Process a position once.
    if (left === previousLeft && width === previousWidth) {
      persist();
      return;
    }
    previousLeft = left;
    previousWidth = width;
    viewport = { left, right: left + width, maxOffset: max / model.SPACING };
    stations.forEach(({ node, x, labelX, labelHalf, labels }) => {
      const visible = x >= viewport.left + 16 && x <= viewport.right - 16;
      if (node.hasAttribute("hidden") === visible) node.toggleAttribute("hidden", !visible);
      const labelsHidden = !visible || labelX - labelHalf < viewport.left + 4 || labelX + labelHalf > viewport.right - 4;
      labels.forEach((label) => {
        if (label.hasAttribute("hidden") !== labelsHidden) label.toggleAttribute("hidden", labelsHidden);
      });
    });
    connections.forEach(({ node, fromX, toX }) => {
      const hidden = fromX > viewport.right || toX < viewport.left;
      if (node.hasAttribute("hidden") !== hidden) node.toggleAttribute("hidden", hidden);
    });
    earlier.disabled = state.offset >= viewport.maxOffset - 0.01;
    later.disabled = state.offset <= 0.01;
    if (hoverAnchor?.closest("[hidden]")) hideTooltip();
    else positionTooltip();
    persist();
  }
  function setOffset(offset) {
    const max = Math.max(0, mapWidth - viewportWidth());
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
    if (!mapWidth) return;
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
      tooltip = element("div", undefined, "rm-tooltip");
      tooltip.id = "research-map-tooltip";
      tooltip.setAttribute("role", "tooltip");
      tooltip.hidden = true;
      tooltip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
      tooltip.addEventListener("pointerleave", scheduleHide);
      tooltip.addEventListener("focusin", () => clearTimeout(hideTimer));
      tooltip.addEventListener("focusout", scheduleHide);
      root.append(tooltip);
      bindOverview();
      root.querySelector(".rm-static").classList.replace("rm-static", "rm-interactive");
      root.querySelector(".rm-load-status").hidden = true;
      navigation.hidden = false;
      setOffset(state.offset);
      let observedWidth = scroller.clientWidth;
      new ResizeObserver(() => {
        if (scroller.clientWidth !== observedWidth) {
          observedWidth = scroller.clientWidth;
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
