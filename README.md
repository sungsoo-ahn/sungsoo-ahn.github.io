# Sungsoo Ahn's Homepage

Personal academic homepage built with [al-folio](https://github.com/alshedivat/al-folio) theme.

**Live site:** https://sungsoo-ahn.github.io

## Updating Publications

Publications are synced from the SPML lab Excel file. To update:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the update script
python scripts/update_publications.py

# Commit and push
git add -A
git commit -m "Update publications"
git push
```

The script reads from:
```
~/Sungsahn0215 Dropbox/SPML/administration/papers.xlsx
```

And generates `_bibliography/papers.bib` with:
- Conference and journal papers (excludes workshop papers and preprints)
- Venue abbreviations (NeurIPS, ICML, ICLR, etc.)
- Links to arXiv, code, and project pages
- "Selected" flag for spotlight/oral papers and recent top-venue papers

## Local Development

To run the site locally:

```bash
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000 in your browser.

## File Structure

```
_bibliography/papers.bib  # Publications (auto-generated)
_data/socials.yml         # Social links (email, Google Scholar, LinkedIn)
_news/                    # News announcements
_pages/about.md           # Homepage content
_pages/people.md          # Lab members
_pages/publications.md    # Publications page
assets/img/prof_pic.jpg   # Profile photo
assets/pdf/cv.pdf         # CV file
scripts/                  # Utility scripts
```

## Adding/Editing Content

### Update profile photo
Replace `assets/img/prof_pic.jpg`

### Update CV
Replace `assets/pdf/cv.pdf`

### Add news
Create a new file in `_news/` following the existing format:
```yaml
---
layout: post
date: 2025-01-01 00:00:00+0900
inline: true
related_posts: false
---

Your news content here.
```

### Update lab members
Edit `_pages/people.md`

### Update social links
Edit `_data/socials.yml`

## License

Based on [al-folio](https://github.com/alshedivat/al-folio) theme (MIT License).
