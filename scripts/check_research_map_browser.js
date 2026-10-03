/* Integration checks against an existing preview; never starts or restarts it. */
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
(async () => {
  const argument = (name, fallback) => {
    const i = process.argv.indexOf(name);
    return i < 0 ? fallback : process.argv[i + 1];
  };
  const url = argument("--url", "http://127.0.0.1:4000/"),
    screenshots = argument("--screenshots", null);
  if (screenshots) fs.mkdirSync(screenshots, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHROMIUM_CHANNEL } : {}),
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1100 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const root = page.locator("#research-map"),
    scroller = root.locator(".rm-overview .rm-timeline-scroll");
  const idle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const snapshot = async (name) => {
    if (screenshots) {
      await root.scrollIntoViewIfNeeded();
      await root.screenshot({ path: path.join(screenshots, `${name}.png`) });
    }
  };
  const load = async (target = url) => {
    await page.goto(target, { waitUntil: "networkidle" });
    await page.waitForSelector('#research-map[data-ready="true"]');
    await root.scrollIntoViewIfNeeded();
    await idle();
  };
  const scrollTo = async (offset) => {
    await page.evaluate((offset) => {
      const node = document.querySelector(".rm-overview .rm-timeline-scroll");
      node.scrollLeft = Math.max(0, node.scrollWidth - node.clientWidth - offset * 120);
    }, offset);
    await idle();
  };
  const focusMetrics = () =>
    page.evaluate(() => {
      const model = ResearchMapModel,
        data = model.prepare(window.rmRaw),
        state = model.readUrl(location.href, data),
        graph = model.focusGraph(state.paper, data);
      const box = document.querySelector(".rm-focus-graph").getBoundingClientRect(),
        center = document.querySelector(".rm-focus-center").getBoundingClientRect(),
        mobile = matchMedia("(max-width:900px)").matches;
      const labels = [...document.querySelectorAll(mobile ? ".rm-mobile-concepts" : ".rm-spoke-names")].map((node) => node.getBoundingClientRect());
      const collisions = labels.some((a, i) =>
        labels
          .slice(i + 1)
          .some((b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1)
      );
      const overflow = labels.some((rect) => rect.left < box.left - 1 || rect.right > box.right + 1);
      return {
        expected: graph.connections.map((edge) => edge.peer.id).sort(),
        actual: [...document.querySelectorAll(".rm-focus-peer")].map((node) => node.dataset.peer).sort(),
        expectedLabels: graph.connections.flatMap((edge) => edge.concepts.map((concept) => concept.label)).sort(),
        actualLabels: [...document.querySelectorAll(mobile ? ".rm-mobile-concepts .rm-concept-name" : ".rm-spoke-names .rm-concept-name")]
          .map((node) => node.textContent)
          .sort(),
        lines: document.querySelectorAll(".rm-focus-lines .rm-connection").length,
        placement: [...document.querySelectorAll(".rm-focus-peer")].every((node) => {
          const rect = node.getBoundingClientRect(),
            peer = data.workById.get(node.dataset.peer);
          return model.compareChronology(peer, graph.selected) < 0
            ? mobile
              ? rect.bottom < center.top
              : rect.right < center.left
            : mobile
              ? rect.top > center.bottom
              : rect.left > center.right;
        }),
        collisions,
        overflow,
      };
    });
  const checkFocus = async () => {
    await idle();
    const metrics = await focusMetrics();
    assert.deepEqual(metrics.actual, metrics.expected);
    assert.deepEqual(metrics.actualLabels, metrics.expectedLabels);
    assert.equal(metrics.lines, metrics.expected.length);
    assert.ok(metrics.placement && !metrics.collisions && !metrics.overflow, JSON.stringify(metrics));
    assert.equal(await root.locator(".rm-overview").isVisible(), false);
    assert.equal(await root.locator(".rm-navigation").isVisible(), false);
    assert.equal(await root.locator(".rm-back").isVisible(), true);
    assert.equal(await root.locator("marker").count(), 0);
    await root.locator(".rm-focus-center").hover();
    const expected = await page.evaluate(() => {
      const data = ResearchMapModel.prepare(window.rmRaw),
        state = ResearchMapModel.readUrl(location.href, data),
        work = data.workById.get(state.paper);
      return { authors: work.authors.join(", "), shape: ResearchMapModel.contributionFor(work, data).shape };
    });
    assert.equal(await root.locator(".rm-tooltip-authors").innerText(), expected.authors);
    assert.equal(await root.locator(".rm-focus-center .rm-symbol").getAttribute("data-shape"), expected.shape);
    await page.keyboard.press("Escape");
  };
  try {
    await load();
    assert.equal(await root.locator("h2").innerText(), "Research");
    assert.equal(await page.getByRole("heading", { name: "Selected Highlights", exact: true }).count(), 0);
    assert.equal(await page.locator(".post article > .publications").count(), 0);
    assert.ok(
      await root.evaluate((node) => node.previousElementSibling?.textContent.includes("Alumni")),
      "Research replaces the old highlights after members"
    );
    await page.evaluate(async () => {
      window.rmRaw = await (await fetch(document.querySelector("#research-map").dataset.source)).json();
    });
    assert.equal(await root.locator(".rm-overview .rm-network").count(), 1);
    assert.equal(await root.locator(".rm-overview .rm-work").count(), 74);
    assert.equal(
      await root
        .locator("input, select, .rm-toolbar, .rm-filter-panel, .rm-paper-list, .rm-details, .rm-about, .rm-caption, .rm-timeline-axis")
        .count(),
      0
    );
    assert.equal(await root.locator("button:visible").count(), 2);
    assert.equal(await root.locator(".rm-later").isDisabled(), true);
    assert.equal(await root.locator(".rm-earlier").isDisabled(), false);
    const legendBefore = await root.locator(".rm-legend").innerHTML();
    const figureHeight = await scroller.evaluate((node) => node.getBoundingClientRect().height);
    assert.equal(await root.locator(".rm-domain-legend .rm-legend-item").count(), 8);
    assert.equal(await root.locator(".rm-contribution-legend .rm-legend-item").count(), 8);
    assert.equal(await page.getByRole("link", { name: "View all publications", exact: true }).count(), 0);
    const newestCount = await root
      .locator(".rm-overview .rm-work:not([hidden])")
      .evaluateAll((nodes) => new Set(nodes.map((node) => node.dataset.work)).size);
    assert.ok(newestCount >= 15, `newest window shows ${newestCount} papers`);
    assert.deepEqual(await root.locator(".rm-domain-legend .rm-legend-item").evaluateAll((nodes) => nodes.map((node) => node.dataset.theme)), [
      "d_graphical",
      "d_deep",
      "d_molecules",
      "d_bio",
      "d_electronic",
      "d_geoscience",
      "d_materials",
      "d_cells",
    ]);
    assert.equal(await root.locator(".rm-row-heading, .rm-row-category").count(), 0);
    const initialScroll = await scroller.evaluate((node) => node.scrollLeft);
    await root.locator(".rm-earlier").click();
    await idle();
    assert.equal(await scroller.evaluate((node) => node.scrollLeft), initialScroll - 360);
    await root.locator(".rm-later").click();
    await idle();
    assert.equal(await scroller.evaluate((node) => node.scrollLeft), initialScroll);
    const geometry = await page.evaluate(() => {
      const runtime = [...document.querySelectorAll(".rm-overview .rm-work")],
        fallback = [...document.querySelectorAll(".rm-static .rm-static-paper")];
      const canonical = new Map(fallback.map((node) => [node.dataset.instance, [Number(node.dataset.x), Number(node.dataset.y)]]));
      return {
        same: runtime.every((node) => canonical.get(node.dataset.instance).join() === [Number(node.dataset.x), Number(node.dataset.y)].join()),
        symbols: runtime.every((node) => {
          const fallback = document.querySelector(`.rm-static-paper[data-instance="${node.dataset.instance}"] .rm-station`);
          return fallback.outerHTML === node.querySelector(".rm-station").outerHTML && node.querySelector(".rm-hit").getAttribute("r") === "16";
        }),
        nicknames: runtime.filter((node) => node.dataset.primaryLabel === "true").length === 74,
        lines: [...document.querySelectorAll(".rm-overview .rm-connection-path")].every((node) => getComputedStyle(node).strokeWidth === "4px"),
        shared:
          document.querySelectorAll('.rm-overview .rm-work[data-work="kim2026catflow"]').length === 1 &&
          document.querySelector('.rm-overview .rm-work[data-work="kim2026catflow"]').dataset.themes.split(",").length === 2,
        noJumps: [...document.querySelectorAll(".rm-overview .rm-connection")].every(
          (node) => !node.querySelector(".rm-connection-path").getAttribute("d").includes("A") && node.querySelector(".rm-connection-casing")
        ),
        height: document.querySelector(".rm-overview .rm-network").getAttribute("height") === "300",
      };
    });
    assert.ok(
      geometry.same && geometry.symbols && geometry.nicknames && geometry.lines && geometry.shared && geometry.noJumps && geometry.height,
      JSON.stringify(geometry)
    );
    await snapshot("desktop-overview");
    const seen = new Set();
    const maxOffset = await scroller.evaluate((node) => (node.scrollWidth - node.clientWidth) / 120);
    for (let offset = 0; offset <= Math.ceil(maxOffset); offset++) {
      await scrollTo(offset);
      const visible = await root.locator(".rm-overview .rm-work:not([hidden])").evaluateAll((nodes) => nodes.map((node) => node.dataset.work));
      visible.forEach((id) => seen.add(id));
      const labelDefects = await root.evaluate((node) => {
        const works = [...node.querySelectorAll(".rm-overview .rm-work:not([hidden])")];
        const named = works.filter((work) => !work.querySelector(".rm-work-label").hasAttribute("hidden"));
        const duplicate = new Set(named.map((work) => work.dataset.work)).size !== named.length;
        const labels = named.flatMap((work) =>
          [...work.querySelectorAll(".rm-work-label, .rm-work-meta")].map((label) => label.getBoundingClientRect())
        );
        const overlap = labels.some((a, i) =>
          labels
            .slice(i + 1)
            .some((b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1)
        );
        return { duplicate, overlap };
      });
      assert.deepEqual(labelDefects, { duplicate: false, overlap: false }, `labels collide at offset ${offset}`);
      const collisions = await root.evaluate((node) => {
        const labels = [...node.querySelectorAll(".rm-overview .rm-work:not([hidden]) :is(.rm-work-label, .rm-work-meta):not([hidden])")].map(
          (label) => ({ text: label.textContent, rect: label.getBoundingClientRect() })
        );
        const hits = [];
        for (const line of node.querySelectorAll(".rm-overview .rm-connection:not([hidden]) .rm-connection-path")) {
          const length = line.getTotalLength(),
            matrix = line.getScreenCTM();
          for (let position = 0; position <= length; position += 2) {
            const point = line.getPointAtLength(position).matrixTransform(matrix);
            for (const { text, rect } of labels)
              if (point.x >= rect.left - 2 && point.x <= rect.right + 2 && point.y >= rect.top - 2 && point.y <= rect.bottom + 2)
                hits.push([line.closest("g").dataset.connection, text]);
          }
        }
        return hits;
      });
      assert.deepEqual(collisions, [], `routed lines intersect labels at offset ${offset}`);
      assert.equal(await scroller.evaluate((node) => node.getBoundingClientRect().height), figureHeight);
      assert.equal(await root.locator(".rm-legend").innerHTML(), legendBefore);
    }
    assert.equal(seen.size, 74);
    await scroller.focus();
    await scroller.press("Home");
    await idle();
    assert.equal(await scroller.evaluate((node) => node.scrollLeft), 0);
    assert.equal(await root.locator(".rm-earlier").isDisabled(), true);
    assert.ok(
      await root
        .locator(".rm-overview .rm-work:not([hidden]) .rm-work-meta")
        .evaluateAll((nodes) => nodes.some((node) => /20(?:1\d|2[0-3])/.test(node.textContent)))
    );
    await snapshot("desktop-oldest");
    await scroller.press("End");
    await idle();
    assert.equal(await scroller.evaluate((node) => node.scrollLeft), initialScroll);
    const reveal = async (id) => {
      await page.evaluate((id) => {
        const layout = ResearchMapModel.timelineLayout(ResearchMapModel.prepare(window.rmRaw));
        const node = document.querySelector(".rm-overview .rm-timeline-scroll");
        node.scrollLeft = Math.max(0, Math.min(node.scrollWidth - node.clientWidth, layout.xById.get(id) - node.clientWidth / 2));
      }, id);
      await idle();
    };
    await reveal("seong2025transition");
    await page.mouse.move(1100, 50);
    await snapshot("desktop-busy-interchanges");
    await reveal("kim2026catflow");
    const repeated = root.locator('.rm-overview .rm-work[data-work="kim2026catflow"]:not([hidden])').first();
    assert.equal(await root.locator(".rm-overview .rm-identity-path").count(), 0);
    assert.equal(await root.locator('.rm-overview .rm-work[data-work="kim2026catflow"]').count(), 1);
    await repeated.hover();
    await idle();
    assert.equal(await root.locator('.rm-overview .rm-work[data-work="kim2026catflow"][data-highlighted="true"]:not([hidden])').count(), 1);
    assert.ok((await root.locator(".rm-tooltip").innerText()).includes("Cat"));
    assert.match(await root.locator(".rm-tooltip-authors").innerText(), /Sungsoo Ahn/);
    await snapshot("desktop-authors");
    await snapshot("desktop-interchange");
    await page.keyboard.press("Escape");
    const hoverablePoint = (nodes) => {
      for (const node of nodes) {
        const path = node.querySelector(".rm-connection-path"),
          length = path.getTotalLength(),
          matrix = path.getScreenCTM();
        const frame = node.closest(".rm-timeline-scroll").getBoundingClientRect();
        for (let distance = 0; distance <= length; distance += 3) {
          const point = path.getPointAtLength(distance).matrixTransform(matrix);
          if (
            point.x > frame.left + 20 &&
            point.x < frame.right - 20 &&
            document.elementFromPoint(point.x, point.y)?.closest(".rm-connection") === node
          )
            return { x: point.x, y: point.y };
        }
      }
      return null;
    };
    const routePoint = await root
      .locator('.rm-overview .rm-route[data-theme="d_molecules"] .rm-connection:not([hidden])')
      .evaluateAll(hoverablePoint);
    assert.ok(routePoint, "a routed line remains available for hover");
    await page.mouse.move(routePoint.x, routePoint.y);
    await idle();
    assert.ok((await root.locator(".rm-tooltip").innerText()).includes("Molecules"));
    await page.keyboard.press("Escape");
    await reveal("kim2024local");
    const parallels = root.locator('.rm-overview .rm-connection[data-from="kim2024local"][data-to="jang2024learning"]');
    assert.equal(await parallels.count(), 3);
    for (const route of await parallels.all()) {
      const point = await route.evaluateAll(hoverablePoint);
      assert.ok(point, "each of three parallel domain tracks has an independent hover target");
      await page.mouse.move(point.x, point.y);
      await idle();
      assert.equal(await route.getAttribute("data-highlighted"), "true");
      assert.equal(await root.locator(".rm-tooltip-title").innerText(), await route.getAttribute("aria-label"));
      await page.keyboard.press("Escape");
    }
    await reveal("park2026learning");
    const paper = root.locator('.rm-overview .rm-work[data-work="park2026learning"]:not([hidden])').first();
    await paper.focus();
    await idle();
    const savedOffset = await scroller.evaluate((node) => node.scrollLeft),
      instance = await paper.getAttribute("data-instance");
    await paper.press("Enter");
    await checkFocus();
    await page.keyboard.press("Escape");
    await snapshot("desktop-focus");
    assert.match(await root.locator(".rm-focus-center a").getAttribute("href"), /^https:\/\/arxiv\.org\//);
    const concept = root.locator(".rm-spoke-names .rm-concept-name").first();
    await concept.hover();
    await idle();
    const displayExplanation = await page.evaluate(() => window.rmRaw.relationships.find((relation) => relation.id === "r058").map_explanation);
    assert.equal(await root.locator(".rm-tooltip details, .rm-tooltip summary, .rm-tooltip a").count(), 0);
    assert.equal(await root.locator(".rm-tooltip-reason p").first().innerText(), displayExplanation);
    await snapshot("desktop-connection");
    await page.keyboard.press("Escape");
    const peerId = await root.locator(".rm-focus-peer .rm-focus-paper").first().getAttribute("data-work");
    await root.locator(".rm-focus-peer .rm-focus-paper").first().click();
    await checkFocus();
    assert.equal(await root.locator(".rm-focus-center").getAttribute("data-work"), peerId);
    await root.locator(".rm-back").click();
    await idle();
    assert.equal(await scroller.evaluate((node) => node.scrollLeft), savedOffset);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.instance), instance);
    await page.keyboard.press("Escape");
    // Exercise all 74 focused layouts for label overlap and complete reasons.
    const workIds = await page.evaluate(() => window.rmRaw.works.map((work) => work.annotations.id));
    for (const id of workIds) {
      await page.evaluate((id) => {
        const u = new URL(location.href);
        u.searchParams.set("rm_paper", id);
        history.replaceState({}, "", u);
        dispatchEvent(new PopStateEvent("popstate"));
      }, id);
      await checkFocus();
    }
    await page.evaluate(() => {
      const u = new URL(location.href);
      u.searchParams.set("rm_paper", "park2026learning");
      u.searchParams.set("rm_offset", "7");
      u.searchParams.set("rm_lens", "method");
      u.searchParams.set("rm_tags", "missing");
      history.replaceState({}, "", u);
      dispatchEvent(new PopStateEvent("popstate"));
    });
    await checkFocus();
    await root.locator(".rm-back").click();
    await idle();
    assert.equal(new URL(page.url()).searchParams.get("rm_lens"), null);
    assert.equal(new URL(page.url()).searchParams.get("rm_tags"), null);
    assert.equal(new URL(page.url()).searchParams.get("rm_offset"), "7");
    await page.goBack();
    await idle();
    await checkFocus();
    await page.setViewportSize({ width: 768, height: 1100 });
    await idle();
    await checkFocus();
    await snapshot("tablet-focus");
    await page.setViewportSize({ width: 390, height: 1100 });
    await idle();
    await checkFocus();
    await page.keyboard.press("Escape");
    await snapshot("mobile-focus");
    await root.locator(".rm-mobile-concepts .rm-concept-name").first().click();
    await idle();
    assert.equal(await root.locator(".rm-tooltip details, .rm-tooltip summary, .rm-tooltip a").count(), 0);
    assert.equal(await root.locator(".rm-tooltip-reason p").first().innerText(), displayExplanation);
    await snapshot("mobile-connection");
    await root.locator(".rm-back").click();
    await idle();
    await scroller.focus();
    await scroller.press("End");
    await idle();
    await page.keyboard.press("Escape");
    await snapshot("mobile-overview");
    const mobileSeen = new Set();
    const mobileHeight = await scroller.evaluate((node) => node.getBoundingClientRect().height);
    for (let step = 0; step < 60; step++) {
      (await root.locator(".rm-overview .rm-work:not([hidden])").evaluateAll((nodes) => nodes.map((node) => node.dataset.work))).forEach((id) =>
        mobileSeen.add(id)
      );
      assert.equal(await scroller.evaluate((node) => node.getBoundingClientRect().height), mobileHeight);
      assert.equal(await root.locator(".rm-legend").innerHTML(), legendBefore);
      if (await root.locator(".rm-earlier").isDisabled()) break;
      await root.locator(".rm-earlier").click();
      await idle();
    }
    assert.equal(mobileSeen.size, 74, "every paper must be reachable using just the arrows on mobile");
    await scroller.focus();
    await scroller.press("End");
    await idle();
    await page.keyboard.press("Escape");
    await reveal("seong2025transition");
    await page.mouse.move(380, 50);
    await snapshot("mobile-busy-interchanges");
    await scroller.press("End");
    await idle();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.evaluate(() => {
      document.documentElement.dataset.theme = "dark";
    });
    await idle();
    await snapshot("mobile-dark");
    const colors = await page.evaluate(() => {
      const toHex = (rgb) =>
        "#" +
        rgb
          .match(/\d+/g)
          .slice(0, 3)
          .map((v) => Number(v).toString(16).padStart(2, "0"))
          .join("");
      return [...document.querySelectorAll(".rm-domain-legend .rm-legend-item")].map((node) =>
        ResearchMapModel.contrast(toHex(getComputedStyle(node).color), "#21192b")
      );
    });
    assert.ok(colors.every((value) => value >= 3.1));
    await page.setViewportSize({ width: 1280, height: 1100 });
    await idle();
    await snapshot("desktop-dark");
    await reveal("seong2025transition");
    await page.mouse.move(1100, 50);
    await snapshot("desktop-dark-busy-interchanges");

    assert.deepEqual(errors, []);
    const noJs = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 390, height: 1100 } });
    await noJs.goto(url, { waitUntil: "networkidle" });
    assert.equal(await noJs.locator(".rm-static-paper").count(), 74);
    assert.equal(await noJs.locator(".rm-static .rm-identity-path").count(), 0);
    assert.equal(await noJs.locator(".rm-static").isVisible(), true);
    assert.equal(await noJs.locator(".rm-navigation").isVisible(), false);
    assert.ok(await noJs.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    if (screenshots) await noJs.locator("#research-map").screenshot({ path: path.join(screenshots, "mobile-no-js.png") });
    const failed = await browser.newPage();
    await failed.route("**/assets/json/research-map.json", (route) => route.abort());
    await failed.goto(url, { waitUntil: "networkidle" });
    assert.equal(await failed.locator(".rm-static").isVisible(), true);
    assert.equal(await failed.locator(".rm-load-status").innerText(), "Interactive map unavailable.");
    console.log(
      "Research-map browser checks passed: Research placement, compact spacing, contribution symbols, author hovers, fixed height and legends, chronology, navigation, all 74 focused layouts, shared stations and clear crossing gaps, concise connections, history, keyboard, mobile/tablet/dark, and static/error fallback."
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
