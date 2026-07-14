# Sungsoo Ahn's Homepage

Personal academic homepage built with [al-folio](https://github.com/alshedivat/al-folio) theme.

**Live site:** https://sungsoo-ahn.github.io

## Quick Update

```bash
source .venv/bin/activate

# Update publications from papers.xlsx
python scripts/update_publications.py

# Update lab members from members.xlsx
python scripts/update_members.py

# Commit and push
git add -A && git commit -m "Update content" && git push
```

Data sources (synced from Dropbox):

- `~/Sungsahn0215 Dropbox/SPML/administration/papers.xlsx`
- `~/Sungsahn0215 Dropbox/SPML/administration/members.xlsx`

## Local Development

```bash
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve
```

Then open http://localhost:4000

Before pushing blog changes, run:

```bash
python3 scripts/validate_blog.py
python3 scripts/validate_kups_pages.py
```

## File Structure

```
_bibliography/papers.bib  # Publications (auto-generated from papers.xlsx)
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
