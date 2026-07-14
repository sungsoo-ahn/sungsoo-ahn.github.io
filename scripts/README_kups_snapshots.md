# kUPS Page Snapshots

Use `capture_kups_snapshots.js` to review hidden kUPS MD tutorial pages at
desktop and mobile widths before they are made public.

```bash
npm install
npx playwright install chromium
node scripts/capture_kups_snapshots.js \
  --posts 01,02,03,04,05,06,07,08,09,10,11,12 \
  --output-dir ../kups-md-tutorials/snapshots/pages \
  --base-url https://sungsoo-ahn.github.io
```

On Linux, Playwright may also need system browser libraries:

```bash
sudo npx playwright install-deps
```

The script writes `post-XX-desktop.png`, `post-XX-mobile.png`, and
`manifest.json` to the output directory. Use a local Jekyll server by replacing
`--base-url` with the local URL.
