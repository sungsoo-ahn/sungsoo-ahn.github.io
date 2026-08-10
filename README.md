# Sungsoo Ahn's Homepage

Personal academic homepage built with [al-folio](https://github.com/alshedivat/al-folio) theme.

**Live site:** https://sungsoo-ahn.github.io

## Quick Update

```bash
source .venv/bin/activate

# Validate publications after editing _data/publications.yml
uv run python scripts/update_publications.py --check

# Update lab members from members.xlsx
uv run python scripts/update_members.py

# Regenerate and compile the CV
uv run python scripts/update_cv.py

# Commit and push
git add -A && git commit -m "Update content" && git push
```

External data source (synced from Dropbox):

- `~/SPML Dropbox/SPML/administration/members.xlsx`

Set `SPML_MEMBERS_XLSX` to override this location on another machine.

## CV Maintenance

The CV follows a single-source workflow inspired by
[academic-homepage-cv-sync](https://github.com/kw1jjang/academic-homepage-cv-sync):

- Edit education, employment, service, talks, courses, awards, and grants in
  `_data/cv_content.yml`.
- Edit all publication metadata in `_data/publications.yml`. The website reads
  this file directly, and the CV generator reads the same records.
- Run `uv run python scripts/update_cv.py`. This regenerates the protected
  `SYNC` blocks in `cv/cv.tex`, regenerates `cv/publications.tex`, compiles
  `cv/cv.pdf`, and copies it to `assets/pdf/cv.pdf` for the website.

Do not edit content inside `% SYNC:...:BEGIN/END` markers by hand. Content
outside those markers, including the LaTeX design and contact header, remains
hand-maintained.

Useful variants:

```bash
uv run python scripts/update_cv.py --no-compile        # update LaTeX only
uv run python scripts/update_cv.py --check --no-compile # verify committed output
uv run python scripts/update_publications.py --check    # validate publication YAML
uv run python -m unittest discover tests                # regression tests
```

## Local Development

```bash
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve
```

Then open http://localhost:4000

Before pushing blog changes, run:

```bash
python3 scripts/validate_agent_hygiene.py
python3 scripts/validate_blog.py
python3 scripts/validate_kups_pages.py
```

## File Structure

```
_data/publications.yml    # Canonical publications for the website and CV
_data/cv_content.yml      # Structured source for non-publication CV sections
cv/cv.tex                 # CV template with generated SYNC blocks
_pages/about.md           # Homepage content
_pages/people.md          # Lab members (auto-generated from members.xlsx)
_pages/publications.md    # Publications page
_data/socials.yml         # Social links
docs/palette.md           # Official color palette
assets/img/prof_pic.jpg   # Profile photo
scripts/                  # Update scripts
```

## License

Based on [al-folio](https://github.com/alshedivat/al-folio) theme (MIT License).
