# CLAUDE.md

## Project Overview

Academic homepage for Sungsoo Ahn (KAIST Graduate School of AI), built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme.

**Live site:** https://sungsoo-ahn.github.io

## Common Commands

```bash
# Local development (requires Homebrew Ruby, not system Ruby)
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve  # Opens at http://localhost:4000

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

## Data Sources

Excel files synced from Dropbox:
- `~/Sungsahn0215 Dropbox/SPML/administration/papers.xlsx` → Publications
- `~/Sungsahn0215 Dropbox/SPML/administration/members.xlsx` → Lab members

## Key Files

| File | Purpose |
|------|---------|
| `_bibliography/papers.bib` | Publications (auto-generated) |
| `_pages/about.md` | Homepage content |
| `_pages/people.md` | Lab members (auto-generated) |
| `_pages/publications.md` | Publications page |
| `_config.yml` | Site configuration |
| `_data/socials.yml` | Social links |
| `assets/img/prof_pic.jpg` | Profile photo |
| `_pages/teaching.md` | Teaching page (lists courses) |
| `_teaching/` | Lecture notes collection |

## Blog Post Writing Guidelines

- **Introduce concepts intuitively before formal notation.** When a concept (e.g., equivariance) first appears, explain it in plain language with a concrete example. Do not use formal symbols (e.g., group $G$, element $g$) until they have been properly defined in their own section.
- **Motivate mathematical abstractions from concrete examples.** When introducing an abstraction like a group, start from the concrete transformations it describes (e.g., rotations) and show how the abstract definition captures their properties naturally.

## Architecture

- **Framework:** Jekyll with al-folio theme
- **Hosting:** GitHub Pages (auto-deploy on push to main)
- **Content updates:** Python scripts parse Excel → generate Markdown/BibTeX
