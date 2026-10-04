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
  const page = await browser.newPage({ viewport: { width: 390, height: 1100 }, hasTouch: true });
  const client = await page.context().newCDPSession(page);
  await client.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  await page.addInitScript(() => {
    Object.defineProperty(window, "ResearchMapModel", {
      configurable: true,
      set(model) {
        model.timelineLayout =
          model.verticalTimelineLayout =
          model.ideaLayout =
            () => {
              throw new Error("Layout must be computed at build time");
            };
        const prepare = model.prepare;
        model.prepare = (...args) => {
          window.rmHydrationStart = performance.now();
          return prepare(...args);
        };
        Object.defineProperty(window, "ResearchMapModel", { value: model, configurable: true });
      },
    });
    new MutationObserver(() => {
      if (document.querySelector('#research-map[data-ready="true"]') && !window.rmHydrationMs)
        window.rmHydrationMs = performance.now() - window.rmHydrationStart;
    }).observe(document, { subtree: true, attributes: true, attributeFilter: ["data-ready"] });
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const root = page.locator("#research-map"),
    figure = root.locator(".rm-vertical");
  const idle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const paper = (id) => figure.locator(`.rm-work[data-work="${id}"]`);
  const reveal = async (id) => {
    await page.mouse.move(0, 0);
    await page.keyboard.press("Escape");
    await paper(id).evaluate((node) => {
      const circle = node.querySelector(".rm-hit").getBoundingClientRect();
      scrollBy(0, circle.top + circle.height / 2 - innerHeight / 2);
    });
    await idle();
  };
  const hoverStation = async (id) => {
    const box = await paper(id).locator(".rm-station").boundingBox();
    assert.ok(box);
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await idle();
  };
  const snapshot = async (name) => {
    if (screenshots) {
      await idle();
      await page.screenshot({ path: path.join(screenshots, `${name}.png`), animations: "disabled" });
    }
  };
  const load = async (target = url) => {
    const response = await page.goto(target, { waitUntil: "networkidle" });
    await page.evaluate(
      (markup) => {
        window.rmCanonical = new DOMParser().parseFromString(markup, "text/html");
      },
      await response.text()
    );
    await page.waitForSelector('#research-map[data-ready="true"]');
    await idle();
  };
  const checkPaperActivation = async (id, activate) => {
    const destination = await paper(id).getAttribute("href");
    await page.context().route(destination, (route) => route.fulfill({ contentType: "text/html", body: "Paper link check" }));
    const opened = page.waitForEvent("popup");
    await activate();
    const popup = await opened;
    await popup.waitForLoadState("domcontentloaded");
    assert.equal(popup.url(), destination);
    await popup.close();
    await page.context().unroute(destination);
  };
  try {
    await load();
    const hydrationMs = await page.evaluate(() => window.rmHydrationMs);
    assert.ok(hydrationMs < 250, `setup took ${hydrationMs}ms`);
    console.log(`Vertical map setup at 4× CPU slowdown: ${hydrationMs}ms`);
    await client.send("Emulation.setCPUThrottlingRate", { rate: 1 });
    assert.equal(await root.locator("h2").innerText(), "Research");
    assert.equal(
      await root.locator(".rm-introduction").innerText(),
      "We develop structured and probabilistic machine learning to infer, predict, and design molecular and material systems. Our goal is to expand what scientists can learn from experiments and simulations, and what they can investigate with that knowledge."
    );
    assert.equal(await root.locator(".rm-credit a").getAttribute("href"), "https://necludov.github.io/");
    assert.equal(await root.locator(".rm-network").count(), 1);
    assert.equal(await root.locator(".rm-horizontal, .rm-navigation, button, input, select, .rm-row-heading, .rm-identity-path").count(), 0);
    assert.equal(await figure.locator(".rm-work").count(), 74);
    assert.equal(await figure.locator(".rm-connection").count(), 88);
    assert.equal(await figure.locator(".rm-idea:not([hidden])").count(), 14);
    assert.equal(await figure.locator(".rm-station:not(circle)").count(), 0);
    assert.equal(await page.getByRole("link", { name: "View all publications", exact: true }).count(), 0);
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
    assert.equal(await root.locator('.rm-domain-legend [data-theme="d_deep"] span').innerText(), "General ML");
    const corpus = await page.evaluate(async () => {
      const raw = await (await fetch(document.querySelector("#research-map").dataset.source)).json();
      const data = ResearchMapModel.prepare(raw);
      return data.works.map((work) => ({
        id: work.id,
        title: work.title,
        abstract: work.abstract,
        venues: work.publications.map((edition) => `${edition.venue} ${edition.year}`).join(" · "),
        authors: work.authors.join(", "),
        url: ResearchMapModel.paperUrl(work),
      }));
    });
    const geometry = await figure
      .locator(".rm-work")
      .evaluateAll((nodes) => nodes.map((node) => [node.dataset.work, node.dataset.x, node.dataset.y]));
    const figureHeight = await figure.evaluate((node) => node.getBoundingClientRect().height);
    const legend = await root.locator(".rm-legend").innerHTML();
    assert.ok(
      Number(geometry.find(([id]) => id === "ahn2015minimum")[2]) > Number(geometry.find(([id]) => id === "ahn2020guiding")[2]),
      "Blossom-BP precedes GEGL in newest-first chronology"
    );
    for (const width of [320, 390, 768, 900, 991, 992, 1280]) {
      await page.setViewportSize({ width, height: 1100 });
      await reveal("seong2025transition");
      assert.equal(await root.getAttribute("data-layout"), "vertical");
      assert.equal(await figure.isVisible(), true);
      assert.equal(await figure.locator(".rm-paper-caption:visible").count(), width >= 992 ? 74 : 0);
      assert.equal(await figure.locator(".rm-compact-label:visible").count(), width < 992 ? 148 : 0);
      assert.equal(await root.locator(".rm-legend").isVisible(), width < 992);
      assert.equal(await figure.evaluate((node) => node.getBoundingClientRect().height), figureHeight);
      assert.deepEqual(
        await figure.locator(".rm-work").evaluateAll((nodes) => nodes.map((node) => [node.dataset.work, node.dataset.x, node.dataset.y])),
        geometry
      );
      assert.equal(await root.locator(".rm-legend").innerHTML(), legend);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `page overflow at ${width}px`);
      if (width < 992) {
        const legendBox = await root.locator(".rm-legend").boundingBox();
        assert.ok(Math.abs(legendBox.y + legendBox.height - 1100) < 1, `legend sticks to the bottom at ${width}px`);
        const collisions = await figure.evaluate((node) => {
          const svg = node.querySelector("svg").getBoundingClientRect();
          const labels = [...node.querySelectorAll(".rm-compact-label")].map((label) => ({
            id: label.closest("a").dataset.work,
            rect: label.getBoundingClientRect(),
          }));
          const circles = [...node.querySelectorAll(".rm-hit")].map((circle) => ({
            id: circle.closest("a").dataset.work,
            rect: circle.getBoundingClientRect(),
          }));
          const overlap = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
          return labels.flatMap(({ id, rect }, i) => {
            const hits = [];
            if (rect.left < svg.left || rect.right > svg.right) hits.push(`${id} is clipped`);
            for (const circle of circles) if (id !== circle.id && overlap(rect, circle.rect)) hits.push(`${id} covers ${circle.id}`);
            for (const other of labels.slice(i + 1)) if (id !== other.id && overlap(rect, other.rect)) hits.push(`${id} overlaps ${other.id}`);
            return hits;
          });
        });
        assert.deepEqual(collisions, [], `compact labels at ${width}px`);
      }
      await snapshot(`vertical-${width}-busy`);
    }
    // The breakpoint changes only text presentation, retaining the reading position.
    await page.setViewportSize({ width: 991, height: 1100 });
    await reveal("kim2024local");
    const before = await paper("kim2024local").locator(".rm-hit").boundingBox();
    await page.setViewportSize({ width: 992, height: 1100 });
    await idle();
    const after = await paper("kim2024local").locator(".rm-hit").boundingBox();
    assert.ok(Math.abs(before.y - after.y) < 1);
    assert.equal(await figure.locator(".rm-connection-hit").count(), 88, "resizing does not bind duplicate targets");
    // Check all native paper links and complete hover cards in the compact view.
    await page.setViewportSize({ width: 390, height: 1100 });
    for (const work of corpus) {
      await reveal(work.id);
      assert.equal(await paper(work.id).getAttribute("href"), work.url);
      assert.equal(await paper(work.id).getAttribute("target"), "_blank");
      assert.match(await paper(work.id).getAttribute("rel"), /noopener/);
      await hoverStation(work.id);
      assert.equal(await root.locator(".rm-tooltip-title").innerText(), work.title);
      assert.equal(await root.locator(".rm-tooltip-authors").innerText(), work.authors);
      assert.equal(await root.locator(".rm-tooltip-meta").innerText(), work.venues);
      assert.equal(await root.locator(".rm-tooltip-abstract").innerText(), work.abstract);
      assert.equal(await root.locator(".rm-tooltip-ideas, .rm-tooltip-contribution, .rm-tooltip-summary").count(), 0);
      const card = await root.locator(".rm-tooltip").boundingBox();
      assert.ok(card.x >= 0 && card.y >= 0 && card.x + card.width <= 390 && card.y + card.height <= 1100);
      const legendBox = await root.locator(".rm-legend").boundingBox();
      assert.ok(card.y + card.height <= legendBox.y, "paper hover keeps the sticky legend readable");
      const circle = await paper(work.id).locator(".rm-station").boundingBox();
      assert.equal(
        await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.closest(".rm-work")?.dataset.work, {
          x: circle.x + circle.width / 2,
          y: circle.y + circle.height / 2,
        }),
        work.id,
        "hover card keeps icon clickable"
      );
      await page.keyboard.press("Escape");
    }
    // Keyboard focus scrolls a paper clear of the bottom legend.
    await reveal("seong2025transition");
    await paper("seong2025transition").evaluate((node) => {
      const mark = node.querySelector(".rm-hit").getBoundingClientRect();
      scrollBy(0, mark.top - innerHeight + 20);
    });
    await paper("seong2025transition").focus();
    await idle();
    const focusedMark = await paper("seong2025transition").locator(".rm-hit").boundingBox();
    const focusedLegend = await root.locator(".rm-legend").boundingBox();
    assert.ok(focusedMark.y + focusedMark.height <= focusedLegend.y - 23, "focused paper is above the sticky legend");
    await page.keyboard.press("Escape");
    // The legend belongs to the map: it is off-screen before the section and
    // resumes normal flow below the final paper instead of covering the footer.
    await page.evaluate(() => scrollTo(0, 0));
    await idle();
    assert.ok((await root.locator(".rm-legend").boundingBox()).y > 1100);
    await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
    await idle();
    const endLegend = await root.locator(".rm-legend").boundingBox();
    const endFrame = await root.locator(".rm-map-frame").boundingBox();
    assert.ok(endLegend.y + endLegend.height < 1100);
    assert.ok(Math.abs(endLegend.y + endLegend.height - endFrame.y - endFrame.height) < 1);
    await reveal("seong2026discovering");
    await hoverStation("seong2026discovering");
    await snapshot("vertical-mobile-paper-hover");
    await checkPaperActivation("seong2026discovering", () => paper("seong2026discovering").locator(".rm-station").click());
    await page.keyboard.press("Escape");
    await checkPaperActivation("seong2026discovering", () => paper("seong2026discovering").locator(".rm-work-label").click());
    await page.keyboard.press("Escape");
    await checkPaperActivation("seong2026discovering", () => paper("seong2026discovering").locator(".rm-station").tap());
    await paper("seong2026discovering").focus();
    await checkPaperActivation("seong2026discovering", () => paper("seong2026discovering").press("Enter"));
    // Desktop captions remain attached to circles and preserve domain colors.
    await page.setViewportSize({ width: 1280, height: 1100 });
    await reveal("kim2024local");
    const caption = figure.locator('.rm-paper-caption[data-work="kim2024local"]');
    await caption.locator(".rm-caption-summary").hover();
    assert.equal(await figure.locator('.rm-paper-caption[data-work="ahn2020guiding"]').getAttribute("data-highlighted"), "true");
    await snapshot("vertical-desktop-paper-hover");
    await page.keyboard.press("Escape");
    const leaderGaps = await figure.evaluate((node) =>
      [...node.querySelectorAll(".rm-paper-caption")].map((caption) => {
        const circle = node.querySelector(`.rm-work[data-work="${caption.dataset.work}"] .rm-hit`).getBoundingClientRect();
        const leader = caption.querySelector(".rm-caption-leader").getBoundingClientRect();
        const end = caption.dataset.side === "left" ? leader.right : leader.left;
        return Math.abs(end - (circle.left + circle.width / 2)) + Math.abs(leader.top + leader.height / 2 - circle.top - circle.height / 2);
      })
    );
    assert.ok(
      leaderGaps.every((gap) => gap < 2),
      "caption leaders terminate at their circles"
    );
    const domains = figure.locator('.rm-paper-caption[data-work="kim2026catflow"] .rm-caption-domain');
    assert.equal(await domains.count(), 2);
    const idea = figure.locator('.rm-idea[data-idea="r031"]');
    const point = await idea.evaluate((node) => {
      const path = node.querySelector(".rm-idea-hit"),
        matrix = path.getScreenCTM();
      for (let distance = 20; distance < path.getTotalLength() - 20; distance += 8) {
        const point = path.getPointAtLength(distance).matrixTransform(matrix);
        if (point.y > 75 && point.y < innerHeight - 20 && document.elementFromPoint(point.x, point.y)?.closest(".rm-idea") === node)
          return { x: point.x, y: point.y };
      }
      return null;
    });
    assert.ok(point, "shared-idea line is reachable");
    await page.mouse.move(point.x, point.y);
    assert.equal(await root.locator(".rm-tooltip-title").innerText(), "Local search improvement operators");
    assert.match(await root.locator(".rm-tooltip-idea-explanation").innerText(), /mutation and crossover.*backtracking/s);
    assert.equal(await idea.locator(".rm-idea-path").evaluate((node) => getComputedStyle(node).stroke), "rgb(146, 146, 146)");
    await snapshot("vertical-desktop-idea-hover");
    await page.keyboard.press("Escape");
    await page.evaluate(() => {
      document.documentElement.dataset.theme = "dark";
    });
    await reveal("seong2025transition");
    await snapshot("vertical-desktop-dark");
    await page.setViewportSize({ width: 390, height: 1100 });
    await snapshot("vertical-mobile-dark");
    await load(
      `${url.split("?")[0].split("#")[0]}?rm_paper=seong2026discovering&rm_offset=7&rm_lens=method&rm_tags=missing&utm_source=check#research-map`
    );
    assert.equal(new URL(page.url()).search, "?utm_source=check");
    assert.equal(await root.getAttribute("data-layout"), "vertical");
    assert.deepEqual(errors, []);
    // Native fallback uses exactly the same vertical figure and responsive labels.
    const noJs = await browser.newPage({ javaScriptEnabled: false, viewport: { width: 390, height: 1100 } });
    await noJs.goto(url, { waitUntil: "networkidle" });
    assert.equal(await noJs.locator(".rm-vertical .rm-static-paper").count(), 74);
    assert.equal(await noJs.locator(".rm-compact-label:visible").count(), 148);
    assert.equal(await noJs.locator(".rm-paper-caption:visible").count(), 0);
    assert.equal(await noJs.locator(".rm-idea:not([hidden])").count(), 14);
    assert.ok(await noJs.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await noJs.locator('.rm-static-paper[data-work="seong2025transition"]').scrollIntoViewIfNeeded();
    const staticLegend = await noJs.locator(".rm-legend").boundingBox();
    assert.ok(Math.abs(staticLegend.y + staticLegend.height - 1100) < 1, "sticky legend works without JavaScript");
    await noJs.setViewportSize({ width: 1280, height: 1100 });
    assert.equal(await noJs.locator(".rm-vertical .rm-paper-caption:visible").count(), 74);
    assert.equal(await noJs.locator(".rm-compact-label:visible").count(), 0);
    await noJs.close();
    const failed = await browser.newPage({ viewport: { width: 390, height: 1100 } });
    await failed.route("**/assets/json/research-map.json", (route) => route.abort());
    await failed.goto(url, { waitUntil: "networkidle" });
    assert.equal(await failed.locator(".rm-static").isVisible(), true);
    assert.equal(await failed.locator(".rm-load-status").innerText(), "Interactive map unavailable.");
    assert.equal(await failed.locator(".rm-compact-label:visible").count(), 148);
    await failed.close();
    console.log(
      "Research-map browser checks passed: vertical at every width, 992px label breakpoint, stable geometry, clear compact labels, all 74 paper links and abstract hovers, desktop leaders, persistent grey idea lines, keyboard/touch activation, light/dark, fast setup, and static/error fallback."
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
