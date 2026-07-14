#!/usr/bin/env node
/* Capture desktop and mobile review snapshots for hidden kUPS MD pages. */

const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const pages = [
  ["index", ""],
  ["01", "post-01-initialization"],
  ["02", "post-02-integrators"],
  ["03", "post-03-errors"],
  ["04", "post-04-thermostats"],
  ["05", "post-05-barostats"],
  ["06", "post-06-trajectory-length"],
  ["07", "post-07-observables"],
  ["08", "post-08-free-energies"],
  ["09", "post-09-estimators"],
  ["10", "post-10-umbrella-sampling"],
  ["11", "post-11-enhanced-sampling"],
  ["12", "post-12-mlip-capstone"],
];

const viewports = [
  ["desktop", { width: 1440, height: 1200 }],
  ["mobile", { width: 390, height: 1200, isMobile: true }],
];

function parseArgs(argv) {
  const args = {
    baseUrl: "https://sungsoo-ahn.github.io",
    outputDir: "snapshots/kups-md-pages",
    posts: pages.map(([post]) => post),
  };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--base-url") {
      args.baseUrl = argv[++index];
    } else if (arg === "--output-dir") {
      args.outputDir = argv[++index];
    } else if (arg === "--posts") {
      args.posts = argv[++index].split(",").map((item) => {
        if (item === "index") {
          return item;
        }
        return item.padStart(2, "0");
      });
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  fs.mkdirSync(args.outputDir, { recursive: true });
  const selectedPages = pages.filter(([post]) => args.posts.includes(post));
  if (selectedPages.length === 0) {
    throw new Error("No matching kUPS posts selected");
  }

  const browser = await chromium.launch();
  const results = [];
  try {
    for (const [post, slug] of selectedPages) {
      for (const [label, viewport] of viewports) {
        const context = await browser.newContext({ viewport });
        const page = await context.newPage();
        const baseUrl = args.baseUrl.replace(/\/$/, "");
        const url = slug
          ? `${baseUrl}/kups-md-tutorials/${slug}/`
          : `${baseUrl}/kups-md-tutorials/`;
        const response = await page.goto(url, {
          waitUntil: "networkidle",
          timeout: 45000,
        });
        if (!response || response.status() >= 400) {
          throw new Error(`${url} returned HTTP ${response ? response.status() : "none"}`);
        }
        await page.screenshot({
          path: path.join(args.outputDir, `post-${post}-${label}.png`),
          fullPage: true,
        });
        results.push({
          post,
          viewport: label,
          url,
          status: response.status(),
          title: await page.title(),
        });
        await context.close();
      }
    }
  } finally {
    await browser.close();
  }
  const manifestPath = path.join(args.outputDir, "manifest.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(results, null, 2)}\n`);
  console.log(`Captured ${results.length} snapshots in ${args.outputDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
