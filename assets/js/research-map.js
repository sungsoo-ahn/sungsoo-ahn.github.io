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
  const figure = root.querySelector(".rm-vertical"),
    legend = root.querySelector(".rm-legend");
  let data, tooltip;
  let stations = [],
    connections = [],
    captions = [],
    ideas = [];
  let hoverAnchor, hoverPaper, hoverConnection, hideTimer;
  const nickname = (work) => work.map_label || work.label;
  const venueYears = (work) => work.publications.map((p) => `${p.venue} ${p.year}`).join(" · ");
  const firstVenue = model.firstVenue;

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
  function viewportBottom() {
    const top = innerWidth < 992 ? legend.getBoundingClientRect().top : innerHeight;
    return top > 0 && top < innerHeight ? top : innerHeight;
  }
  function positionTooltip(event) {
    if (!hoverAnchor || tooltip.hidden) return;
    const anchor = (hoverAnchor.querySelector(".rm-hit") || hoverAnchor).getBoundingClientRect(),
      bottom = viewportBottom();
    if (anchor.bottom < 0 || anchor.top > bottom) {
      hideTooltip();
      return;
    }
    const x = Number.isFinite(event?.clientX) ? event.clientX : anchor.left + anchor.width / 2;
    const y = Math.max(24, Math.min(bottom - 24, Number.isFinite(event?.clientY) ? event.clientY : anchor.top + anchor.height / 2));
    tooltip.style.maxHeight = `${bottom * 0.76}px`;
    let box = tooltip.getBoundingClientRect();
    const rightFits = x + 18 + box.width <= innerWidth - 12,
      leftFits = x - box.width - 18 >= 12;
    let left = rightFits ? x + 18 : x - box.width - 18,
      top = y + 18;
    if (!rightFits && !leftFits) {
      // A full abstract can be taller than either side of its station. Reserve
      // a clear vertical side and scroll the card rather than cover the link.
      const above = y - 30,
        below = bottom - y - 30,
        useAbove = above > below;
      tooltip.style.maxHeight = `${Math.min(bottom * 0.76, Math.max(above, below))}px`;
      box = tooltip.getBoundingClientRect();
      left = x - box.width / 2;
      top = useAbove ? y - box.height - 18 : y + 18;
    } else if (top + box.height > bottom - 12) top = y - box.height - 18;
    tooltip.style.left = `${Math.max(12, Math.min(left, innerWidth - box.width - 12))}px`;
    tooltip.style.top = `${Math.max(12, Math.min(top, bottom - box.height - 12))}px`;
  }
  function showTooltip(anchor, item, kind, event) {
    clearTimeout(hideTimer);
    hoverAnchor?.removeAttribute("aria-describedby");
    hoverAnchor = anchor;
    anchor.setAttribute("aria-describedby", tooltip.id);
    tooltip.replaceChildren();
    hoverPaper = kind === "paper" ? item.id : null;
    hoverConnection = kind === "paper" ? null : item;
    if (kind === "paper") {
      tooltip.append(
        element("div", item.title, "rm-tooltip-title"),
        element("div", item.authors.join(", "), "rm-tooltip-authors"),
        element("div", venueYears(item), "rm-tooltip-meta"),
        element("div", item.abstract, "rm-tooltip-abstract")
      );
    } else {
      tooltip.append(element("div", item.label, "rm-tooltip-title"));
      if (kind === "idea") tooltip.append(element("div", item.explanation, "rm-tooltip-idea-explanation"));
      for (const id of [item.from, item.to]) {
        const work = data.workById.get(id);
        tooltip.append(paperLink(work));
        if (kind === "connection") tooltip.append(element("div", work.contribution || work.summary, "rm-tooltip-approach"));
      }
    }
    tooltip.hidden = false;
    updateHighlights();
    positionTooltip(event);
  }
  function paperLink(work) {
    const link = element("a", `${nickname(work)} · ${firstVenue(work)}`, "rm-tooltip-paper");
    link.href = model.paperUrl(work);
    link.target = "_blank";
    link.rel = "external nofollow noopener";
    return link;
  }
  function bindHover(node, item, kind, pointerNode = node) {
    let pointer;
    const pointerPosition = (event) => (pointer = { clientX: event.clientX, clientY: event.clientY });
    pointerNode.addEventListener("pointerenter", (event) => showTooltip(pointerNode, item, kind, pointerPosition(event)));
    pointerNode.addEventListener("pointermove", (event) => positionTooltip(pointerPosition(event)));
    pointerNode.addEventListener("pointerleave", () => {
      pointer = null;
      scheduleHide();
    });
    node.addEventListener("focus", () => {
      const mark = (node.querySelector(".rm-hit") || node).getBoundingClientRect(),
        top = document.body.classList.contains("fixed-top-nav") ? 73 : 16,
        bottom = viewportBottom();
      if (mark.top < top) window.scrollBy(0, mark.top - top);
      else if (mark.bottom > bottom - 24) window.scrollBy(0, mark.bottom - bottom + 24);
      showTooltip(node, item, kind, pointer);
    });
    node.addEventListener("blur", scheduleHide);
  }
  function updateHighlights() {
    const selected = hoverPaper ? data.ideas.filter((idea) => [idea.from, idea.to].includes(hoverPaper)) : [];
    const related = new Set(selected.flatMap((idea) => [idea.from, idea.to]));
    const ids = new Set(selected.map((idea) => idea.id));
    ideas.forEach(({ node, idea }) => {
      node.dataset.highlighted = String(ids.has(idea.id) || hoverConnection?.id === idea.id);
    });
    stations.forEach(({ node, work }) => {
      const highlighted = String(
        related.has(work.id) || work.id === hoverPaper || Boolean(hoverConnection && [hoverConnection.from, hoverConnection.to].includes(work.id))
      );
      if (node.dataset.highlighted !== highlighted) node.dataset.highlighted = highlighted;
    });
    captions.forEach(({ node, work }) => {
      const highlighted = String(
        related.has(work.id) || work.id === hoverPaper || Boolean(hoverConnection && [hoverConnection.from, hoverConnection.to].includes(work.id))
      );
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
    const works = [...figure.querySelectorAll(".rm-static-paper")].map((node) => {
      const work = data.workById.get(node.dataset.work);
      node.classList.replace("rm-static-paper", "rm-work");
      node.setAttribute("tabindex", "0");
      node.querySelector("title")?.remove();
      bindHover(node, work, "paper");
      return { node, work };
    });
    const lines = [...figure.querySelectorAll(".rm-connection")].map((node) => {
      const theme = data.themeById.get(node.closest(".rm-route").dataset.theme);
      const connection = { id: node.dataset.connection, from: node.dataset.from, to: node.dataset.to, label: theme.label };
      node.querySelector("title")?.remove();
      node.append(vector("path", { class: "rm-connection-hit", d: node.querySelector(".rm-connection-path").getAttribute("d") }));
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "img");
      node.setAttribute("aria-label", connection.label);
      bindHover(node, connection, "connection");
      return { node, connection };
    });
    const descriptions = [...figure.querySelectorAll(".rm-paper-caption")].map((node) => {
      const work = data.workById.get(node.dataset.work);
      bindHover(node.querySelector(".rm-caption-link"), work, "paper", node);
      return { node, work };
    });
    const concepts = [...figure.querySelectorAll(".rm-idea")].map((node) => {
      const idea = data.ideaById.get(node.dataset.idea);
      node.querySelector("title")?.remove();
      node.setAttribute("tabindex", "0");
      bindHover(node, idea, "idea");
      return { node, idea };
    });
    return { stations: works, connections: lines, captions: descriptions, ideas: concepts };
  }
  window.addEventListener("scroll", () => positionTooltip(), { passive: true });
  window.addEventListener("resize", () => {
    // CSS changes only the label treatment. Station geometry and scroll
    // position remain the same across the desktop breakpoint.
    if (hoverAnchor?.closest(".rm-paper-caption") && innerWidth < 992) hideTooltip();
    else positionTooltip();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
  async function initialize() {
    try {
      const response = await fetch(root.dataset.source);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = model.prepare(await response.json());
      tooltip = element("div", undefined, "rm-tooltip");
      tooltip.id = "research-map-tooltip";
      tooltip.setAttribute("role", "tooltip");
      tooltip.hidden = true;
      tooltip.addEventListener("pointerenter", () => clearTimeout(hideTimer));
      tooltip.addEventListener("pointerleave", scheduleHide);
      tooltip.addEventListener("focusin", () => clearTimeout(hideTimer));
      tooltip.addEventListener("focusout", scheduleHide);
      root.append(tooltip);
      root.querySelector(".rm-static").classList.replace("rm-static", "rm-interactive");
      ({ stations, connections, captions, ideas } = bindOverview());
      root.dataset.layout = "vertical";
      root.querySelector(".rm-load-status").hidden = true;
      const url = model.writeUrl(location.href, { offset: 0 }).href;
      if (url !== location.href) history.replaceState({}, "", url);
      root.dataset.ready = "true";
    } catch (error) {
      root.querySelector(".rm-load-status").textContent = "Interactive map unavailable.";
      console.error("Research map:", error);
    }
  }
  initialize();
})();
