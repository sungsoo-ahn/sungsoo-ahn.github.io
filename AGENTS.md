# AGENTS.md

## Project Overview

Academic homepage for Sungsoo Ahn (KAIST Graduate School of AI), built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme.

**Live site:** https://sungsoo-ahn.github.io

## Common Commands

```bash
# Local development (requires Homebrew Ruby, not system Ruby)
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve  # Opens at http://localhost:4000

# Blog validation
python3 scripts/validate_blog.py

# Agent instruction and skill validation
python3 scripts/validate_agent_hygiene.py

# Python package management (uses uv)
uv sync                                # Install/update dependencies

# Update structured content
uv run python scripts/update_publications.py --check  # Validate publication YAML
uv run python scripts/update_members.py               # From members.xlsx
uv run python scripts/update_cv.py                    # Sync YAML -> LaTeX -> website PDF
uv run python scripts/update_cv.py --check --no-compile  # Check CV drift
```

**Note:** This project requires Bundler 4.x which is not compatible with macOS system Ruby. Use Homebrew Ruby (`/opt/homebrew/opt/ruby/bin/`) instead.

**Important:** Do not kill and restart the Jekyll server on every file edit — this disconnects the user's browser. Leave the server running while editing. Only restart (kill + serve) when the user explicitly asks to open/preview the site.

## Opening the Blog Preview

When the user asks to show/open/preview the blog:

1. Check whether Jekyll is already serving:
   ```bash
   lsof -iTCP:4000 -sTCP:LISTEN -n -P
   ```
2. If nothing is listening, start the server and leave it running:
   ```bash
   /opt/homebrew/opt/ruby/bin/bundle exec jekyll serve --host 127.0.0.1 --port 4000
   ```
3. Open the blog directly:
   ```bash
   open http://127.0.0.1:4000/blog/
   ```

If the server is already running, only run the `open` command. Do not restart the server unless the user explicitly asks.

## Data Sources

External Excel file synced from Dropbox:

- `~/SPML Dropbox/SPML/administration/members.xlsx` → Lab members

Override the workbook location with `SPML_MEMBERS_XLSX` when the Dropbox root
differs. Do not hard-code another user's home directory in scripts or tracked
metadata.

## Key Files

| File                       | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `_data/publications.yml`   | Canonical publications for website and CV   |
| `_data/cv_content.yml`     | Structured source for CV sections           |
| `cv/cv.tex`                | CV template with generated SYNC blocks      |
| `assets/pdf/cv.pdf`        | Generated CV served by the homepage         |
| `_pages/about.md`          | Homepage content                            |
| `_pages/people.md`         | Lab members (auto-generated)                |
| `_pages/publications.md`   | Publications page                           |
| `_config.yml`              | Site configuration                          |
| `_data/socials.yml`        | Social links                                |
| `assets/img/prof_pic.jpg`  | Profile photo                               |
| `_pages/teaching.md`       | Teaching page (links to course sites)       |
| `_data/courses.yml`        | Course metadata (links to standalone sites) |

## Skills

Writing style and rendering rules are managed as skills:

- `/blog-writing` — direct, opinionated prose style for blog posts
- `/lecture-adaptation` — source-faithful lecture-to-blog workflow
- `/academic-writing` — top-down, rigorous style for papers and teaching notes
- `/jekyll-writing` — MathJax/KaTeX rendering rules for this Jekyll site
- `/generate-blog-figures` — matplotlib figure generation workflow
- `/download-paper-figures` — incorporating figures from academic papers
  Folder-specific guidelines (frontmatter, figures, audience) are in `_posts/AGENTS.md`.
  Blog metadata, figure paths, footnote IDs, and asset drift are checked by `scripts/validate_blog.py`, which also runs in pre-commit and CI.

Lecture notes live in standalone course repos (e.g., `protein-ai-s26`), each with their own AGENTS.md and skills.

### Instruction-file ownership

- `AGENTS.md` files are the canonical repository and directory instructions.
- `.agents/skills/*/SKILL.md` files are the canonical skill definitions.
- `CLAUDE.md`, `_posts/CLAUDE.md`, and `.claude/skills/` are thin adapters to
  those canonical files. Keep policy in one place; do not duplicate it in the
  adapters.
- `.agents/lecture-adaptation/*.json` are durable workflow manifests consumed by
  validation scripts, not agent instructions. Preserve them as data until the
  active lecture migration is complete.
- Keep credentials and broad tool permission histories out of the repository.
  Machine-local agent settings belong in ignored files.

## Architecture

- **Framework:** Jekyll with al-folio theme
- **Hosting:** GitHub Pages (auto-deploy on push to main)
- **Content updates:** YAML drives publications/CV; Python imports members from Excel
