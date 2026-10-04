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
  const page = await browser.newPage({ viewport: { width: 1280, height: 1100 }, hasTouch: true });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const root = page.locator("#research-map"),
    scroller = root.locator(".rm-overview .rm-timeline-scroll");
  const idle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const snapshot = async (name) => {
    if (screenshots) {
      await root.scrollIntoViewIfNeeded();
      const capture = name.includes("paper-summary") ? page : root;
      await capture.screenshot({ path: path.join(screenshots, `${name}.png`) });
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
  try {
    await load();
    assert.equal(await root.locator("h2").innerText(), "Research");
    assert.equal(
      await root.locator(".rm-introduction").innerText(),
      "We develop structured and probabilistic machine learning to infer, predict, and design molecular and material systems. Our goal is to expand what scientists can learn from experiments and simulations, and what they can investigate with that knowledge."
    );
    assert.equal(await root.locator(".rm-credit a").getAttribute("href"), "https://necludov.github.io/");
    assert.equal(await root.locator(".rm-credit").evaluate((node) => getComputedStyle(node).fontSize), "11px");

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
        .locator(
          "input, select, .rm-toolbar, .rm-filter-panel, .rm-paper-list, .rm-details, .rm-about, .rm-caption, .rm-timeline-axis, .rm-focus, .rm-back"
        )
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
        const node = document.querySelector(".rm-overview .rm-timeline-scroll"),
          station = document.querySelector(`.rm-overview .rm-work[data-work="${id}"]`);
        node.scrollLeft = Math.max(0, Math.min(node.scrollWidth - node.clientWidth, Number(station.dataset.x) - node.clientWidth / 2));
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
    await repeated.locator(".rm-station").hover();
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
    const corpus = await page.evaluate(() => {
      const data = ResearchMapModel.prepare(window.rmRaw);
      return data.works.map((work) => ({
        id: work.id,
        title: work.title,
        summary: work.summary,
        authors: work.authors.join(", "),
        url: ResearchMapModel.paperUrl(work),
        contributions: ResearchMapModel.contributionsFor(work, data).map((category) => category.id),
      }));
    });
    for (const work of corpus) {
      await reveal(work.id);
      const paper = root.locator(`.rm-overview .rm-work[data-work="${work.id}"]`);
      assert.equal(await paper.getAttribute("href"), work.url);
      assert.equal(await paper.getAttribute("target"), "_blank");
      assert.match(await paper.getAttribute("rel"), /noopener/);
      assert.equal(await root.locator(`.rm-static-paper[data-work="${work.id}"]`).getAttribute("href"), work.url);
      await paper.locator(".rm-station").hover();
      assert.equal(await root.locator(".rm-tooltip-title").innerText(), work.title);
      assert.equal(await root.locator(".rm-tooltip-summary").innerText(), work.summary);
      assert.equal(await root.locator(".rm-tooltip-authors").innerText(), work.authors);
      assert.deepEqual(
        await root.locator(".rm-tooltip-contribution [data-contribution]").evaluateAll((nodes) => nodes.map((node) => node.dataset.contribution)),
        work.contributions
      );
      await page.keyboard.press("Escape");
    }
    const mask = root.locator('.rm-overview .rm-work[data-work="seong2026discovering"]');
    const checkMaskHover = async () => {
      await reveal("seong2026discovering");
      await mask.locator(".rm-station").hover();
      assert.equal(await mask.locator(".rm-station").getAttribute("data-shape"), "star");
      assert.deepEqual(
        await root.locator(".rm-tooltip-contribution [data-contribution]").evaluateAll((nodes) => nodes.map((node) => node.textContent)),
        ["Agents", "Generative modeling"]
      );
      const tooltip = await root.locator(".rm-tooltip").boundingBox();
      assert.ok(tooltip.x >= 0 && tooltip.x + tooltip.width <= (await page.viewportSize()).width, "hover card fits the viewport");
    };
    await checkMaskHover();
    await snapshot("desktop-paper-summary");
    const checkPaperActivation = async (paper, activate) => {
      const destination = await paper.getAttribute("href");
      await page.context().route(destination, (route) => route.fulfill({ contentType: "text/html", body: "Paper link check" }));
      const opened = page.waitForEvent("popup");
      await activate();
      const popup = await opened;
      await popup.waitForLoadState("domcontentloaded");
      assert.equal(popup.url(), destination);
      assert.equal(await root.locator(".rm-overview").isVisible(), true);
      await popup.close();
      await page.context().unroute(destination);
    };
    await checkPaperActivation(mask, () => mask.locator(".rm-station").click());
    await checkPaperActivation(mask, () => mask.locator(".rm-work-label").click());
    await mask.focus();
    assert.ok((await root.locator(".rm-tooltip-summary").innerText()).length > 0);
    await checkPaperActivation(mask, () => mask.press("Enter"));
    await reveal("yoon2024breadthfirst");
    const beag = root.locator('.rm-overview .rm-work[data-work="yoon2024breadthfirst"]');
    assert.match(await beag.getAttribute("href"), /proceedings\.mlr\.press/);
    await checkPaperActivation(beag, () => beag.click());
    await load(
      `${url.split("?")[0].split("#")[0]}?rm_paper=seong2026discovering&rm_offset=7&rm_lens=method&rm_tags=missing&utm_source=check#research-map`
    );
    assert.equal(new URL(page.url()).searchParams.get("rm_paper"), null);
    assert.equal(new URL(page.url()).searchParams.get("rm_lens"), null);
    assert.equal(new URL(page.url()).searchParams.get("rm_tags"), null);
    assert.equal(new URL(page.url()).searchParams.get("rm_offset"), "7");
    assert.equal(new URL(page.url()).searchParams.get("utm_source"), "check");
    assert.equal(await root.locator(".rm-overview").isVisible(), true);
    await page.goBack();
    await page.waitForSelector('#research-map[data-ready="true"]');
    await page.setViewportSize({ width: 768, height: 1100 });
    await idle();
    await checkMaskHover();
    await snapshot("tablet-paper-summary");
    await page.setViewportSize({ width: 390, height: 1100 });
    await idle();
    await checkMaskHover();
    await snapshot("mobile-paper-summary");
    await checkPaperActivation(mask, () => mask.locator(".rm-station").tap());
    await page.keyboard.press("Escape");
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
    await checkMaskHover();
    await snapshot("mobile-dark-paper-summary");
    await page.keyboard.press("Escape");
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
    assert.equal(
      await noJs.locator('.rm-static-paper[data-work="seong2026discovering"]').getAttribute("href"),
      corpus.find((work) => work.id === "seong2026discovering").url
    );
    assert.match(await noJs.locator('.rm-static-paper[data-work="seong2026discovering"] title').textContent(), /Generative modeling/);
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
      "Research-map browser checks passed: Research placement, compact spacing, contribution symbols, author hovers, fixed height and legends, chronology, navigation, all 74 paper links and summaries, primary and secondary contributions, shared stations and clear crossing gaps, legacy URLs, keyboard and touch activation, mobile/tablet/dark, and static/error fallback."
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
