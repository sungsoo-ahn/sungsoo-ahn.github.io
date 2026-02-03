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

# Update content from Excel files
source .venv/bin/activate
python scripts/update_publications.py  # From papers.xlsx
python scripts/update_members.py       # From members.xlsx
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

## Architecture

- **Framework:** Jekyll with al-folio theme
- **Hosting:** GitHub Pages (auto-deploy on push to main)
- **Content updates:** Python scripts parse Excel → generate Markdown/BibTeX

## Blog Writing Tips

### Content Structure
- Start with a roadmap table mapping sections to "why it's needed" — forces clear narrative dependency
- Lead with concrete examples before abstract formulas
- Introduce every symbol before first use; do a dedicated pass for undefined symbols at the end
- Collapse related works into paragraphs rather than giving each its own subsection
- Footnotes work well for alternate terminology without cluttering the main text

### Figures
- Write a single Python script with separate functions per figure — easier to regenerate individually
- Use a consistent color palette across all figures with matching text colors for equation-figure correspondence
- Expect multiple rounds of visual iteration per figure; screenshots are the feedback loop

### Jekyll/LaTeX Pitfalls
- Use `$$...$$` (not `$...$`) for inline math containing multiple underscores — markdown interprets underscores as italics
- In inline math, use `\lvert...\rvert` instead of `|...|` for absolute values/norms — markdown interprets `|` as table column delimiters. Same applies to bra-ket notation: use `\mid` instead of `|` (e.g., `$$\langle \Psi \mid \hat{O} \mid \Psi \rangle$$`). Display math on its own line is not affected.
- Add `overflow-y: visible` and padding to `mjx-container` CSS to prevent equation clipping
- Homebrew Ruby, not system Ruby, for bundler

### Writing Quality
- Replace domain jargon with plain-language descriptions (readers likely don't know "quadrupole")
- When referencing publications, add to a References section as you cite, not at the end

### Prose Flow
- **Lead with the point, then justify** — e.g., "The BO approximation separates electrons from nuclei" before "because nuclei are 1836× heavier." Don't make readers wait through the setup to learn the result.
- **Show the destination before the derivation** — when presenting a key equation, state it upfront so readers know what the subsequent steps are building toward, then derive it.
- **Cut throat-clearing openers** — delete filler like "The equation says:", "What does X look like?", "The fundamental challenge is clear:", "The methods described above form the backbone of...". Just state the content.
- **Don't restate what was just shown** — if the math already demonstrated a property (e.g., determinant enforces antisymmetry), don't add a sentence restating it in words.
- **Break up stacked parentheticals** — a sentence with three em-dash clauses (e.g., "Ψ is the unknown — ..., E is the energy — ..., H is the operator — ...") should be split into separate sentences.
- **Merge redundant statements** — if two consecutive sentences say the same thing in different words (e.g., "exponentially large space" then "function from R^3N to C"), combine into one.
- **Be specific in section intros** — "neural networks have been applied at several points" → name the specific targets (XC functional, Hamiltonian, density).
- **Define terms next to first use** — if a symbol appears in an equation, define it immediately after, not "defined below." For symbols defined far earlier, add a brief reminder.
- **Drop dramatic qualifiers** — "fundamental physical flaw", "radically different approach", "enormous complexity" → just state the facts. Let the reader judge significance.

### CSS for Academic Blogs
- Distinct h2/h3 styling (border-bottom on h2, different font sizes) for visual hierarchy
- Explicit table borders — al-folio defaults are too subtle
- These are one-time additions that carry over to future posts

### Workflow
- Keep the Jekyll server running and check rendered output frequently
- Commit at natural checkpoints (not after every small edit)
