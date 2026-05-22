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

# If port 4000 is already in use:
lsof -ti:4000 | xargs kill -9

# Python package management (uses uv)
uv sync                                # Install/update dependencies

# Update content from Excel files
uv run python scripts/update_publications.py  # From papers.xlsx
uv run python scripts/update_members.py       # From members.xlsx
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

Excel files synced from Dropbox:

- `~/Sungsahn0215 Dropbox/SPML/administration/papers.xlsx` → Publications
- `~/Sungsahn0215 Dropbox/SPML/administration/members.xlsx` → Lab members

## Key Files

| File                       | Purpose                                     |
| -------------------------- | ------------------------------------------- |
| `_bibliography/papers.bib` | Publications (auto-generated)               |
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
- `/academic-writing` — top-down, rigorous style for papers and teaching notes
- `/jekyll-writing` — MathJax/KaTeX rendering rules for this Jekyll site
- `/generate-blog-figures` — matplotlib figure generation workflow
- `/download-paper-figures` — incorporating figures from academic papers
  Folder-specific guidelines (frontmatter, figures, audience) are in `_posts/AGENTS.md`.
  Blog metadata, figure paths, footnote IDs, and asset drift are checked by `scripts/validate_blog.py`, which also runs in pre-commit and CI.

Lecture notes live in standalone course repos (e.g., `protein-ai-s26`), each with their own AGENTS.md and skills.

## Architecture

- **Framework:** Jekyll with al-folio theme
- **Hosting:** GitHub Pages (auto-deploy on push to main)
- **Content updates:** Python scripts parse Excel → generate Markdown/BibTeX
