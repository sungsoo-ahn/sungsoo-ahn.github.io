#!/usr/bin/env python3
"""
Script to generate papers.bib from the SPML lab papers.xlsx file.
Usage: python scripts/update_publications.py
"""

import pandas as pd
import re
from pathlib import Path

# Configuration
EXCEL_PATH = Path.home() / "SPML Dropbox/SPML/administration/papers.xlsx"
OUTPUT_PATH = Path(__file__).parent.parent / "_bibliography/papers.bib"

# Venue abbreviations
VENUE_ABBR = {
    "NeurIPS": "NeurIPS",
    "ICML": "ICML",
    "ICLR": "ICLR",
    "CVPR": "CVPR",
    "IJCAI": "IJCAI",
    "AISTATS": "AISTATS",
    "EMNLP": "EMNLP",
    "ACL": "ACL",
    "TMLR": "TMLR",
    "IEEE TIT": "IEEE TIT",
    "JSTAT": "JSTAT",
}

CO_FIRST_MARK = "*"
CO_CORRESPONDING_MARK = "†"
HIGHLIGHT_PRESENTATION_TYPES = {"spotlight", "oral"}
AUTHOR_OVERRIDES = {
    "DNACHUNKER: Learnable Tokenization for DNA Language Models": "Taewon Kim, Jihwan Shin, Hyomin Kim, Youngmok Jung, Jonghoon Lee, Won-Chul Lee, Insu Han, Sungsoo Ahn",
    "Non-backtracking Graph Neural Networks": "Seonghyun Park, Narae Ryu, Gahee Kim, Dongyeop Woo, Se-Young Yun, Sungsoo Ahn",
}
CO_CORRESPONDING_OVERRIDES = {
    "Machine Learning Hamiltonians are Accurate Energy-Force Predictors": ["Sungbin Lim", "Sungsoo Ahn"],
    "Riemannian MeanFlow": ["Kirill Neklyudov", "Sungsoo Ahn"],
}
FIELD_OVERRIDES = {
    ("Adaptive Teachers for Amortized Samplers", "ICLR"): {
        "arxiv": "2410.01432",
        "code": "https://github.com/alstn12088/adaptive-teacher",
        "html": "https://openreview.net/forum?id=BdmVgLMvaf",
    },
    ("Boltz is a Strong Baseline for Atom-level Representation Learning", "-"): {
        "code": None,
    },
    ("Bucket-Renormalization for Approximate Inference", "ICML"): {
        "html": "https://proceedings.mlr.press/v80/ahn18a.html",
    },
    ("Bucket-Renormalization for Approximate Inference", "JSTAT"): {
        "html": "https://iopscience.iop.org/article/10.1088/1742-5468/ab3218",
    },
    ("Gauging Variational Inference", "NeurIPS"): {
        "year": 2017,
    },
    ("Graph Generation with K^2 Trees", "ICLR"): {
        "code": "https://github.com/yunhuijang/hggt",
    },
    ("Learning Collective Variables from BioEmu with Time-Lagged Generation", "ICLR"): {
        "website": None,
    },
}

def clean_author_name(name):
    """Clean author name for BibTeX key generation."""
    name = normalize_name(name)
    name = re.sub(r'\*|†', '', name)  # Remove special markers
    return name

def normalize_name(name):
    """Normalize spreadsheet names for matching and BibTeX output."""
    name = str(name).replace('\xa0', ' ').strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'^and\s+', '', name)
    return name

def split_names(names_str):
    """Split comma-delimited author fields from the spreadsheet."""
    if not names_str or pd.isna(names_str):
        return []
    names = []
    for raw_name in str(names_str).split(','):
        name = normalize_name(raw_name)
        if name:
            names.append(name)
    return names

def generate_bibtex_key(row):
    """Generate a unique BibTeX key from the paper data."""
    first_author = clean_author_name(row['First Author'].split(',')[0].split(' ')[-1].lower())
    year = str(get_field(row, 'year', row['Year']))
    # Take first significant word from title
    title_words = row['Title'].lower().split()
    skip_words = {'a', 'an', 'the', 'for', 'of', 'in', 'on', 'to', 'via', 'with', 'and'}
    title_word = next((w for w in title_words if w not in skip_words), title_words[0])
    title_word = re.sub(r'[^a-z]', '', title_word)
    return f"{first_author}{year}{title_word}"

def format_authors_bibtex(authors_str):
    """Format authors string for BibTeX."""
    authors = split_names(authors_str)
    return ' and '.join(authors)

def format_authors_with_contribution_marks(row):
    """Format authors and mark co-first/co-corresponding authors."""
    authors = split_names(AUTHOR_OVERRIDES.get(row['Title'], row['Authors']))
    co_first_authors = {
        clean_author_name(name)
        for name in split_names(row.get('First Author', ''))
    }
    co_corresponding_authors = {
        clean_author_name(name)
        for name in split_names(row.get('Last Author', ''))
    }
    for name in CO_CORRESPONDING_OVERRIDES.get(row['Title'], []):
        co_corresponding_authors.add(clean_author_name(name))

    mark_co_first = len(co_first_authors) > 1
    mark_co_corresponding = len(co_corresponding_authors) > 1

    marked_authors = []
    for author in authors:
        clean_author = clean_author_name(author)
        marked_author = clean_author
        if mark_co_first and clean_author in co_first_authors:
            marked_author += CO_FIRST_MARK
        if mark_co_corresponding and clean_author in co_corresponding_authors:
            marked_author += CO_CORRESPONDING_MARK
        marked_authors.append(marked_author)

    return ' and '.join(marked_authors)

def get_entry_type(row):
    """Determine BibTeX entry type."""
    pub_type = row['Type']
    if pub_type == 'Journal':
        return 'article'
    elif pub_type == 'Preprint':
        return 'article'  # Use article type for preprints to display "arXiv, year"
    else:  # Conference, Workshop
        return 'inproceedings'

def get_venue_name(venue, pub_type):
    """Get venue name for BibTeX (uses acronym)."""
    return venue  # Use acronym directly (ICLR, NeurIPS, etc.)

def get_field(row, field, default=None):
    """Return a row field after applying local publication overrides."""
    override = FIELD_OVERRIDES.get((row['Title'], row['Conference/Journal/Workshop']), {})
    if field in override:
        return override[field]
    return default

def get_presentation_type(row):
    """Return highlighted presentation type for display."""
    presentation_type = normalize_name(row.get('Presentation Type', ''))
    if presentation_type.lower() in HIGHLIGHT_PRESENTATION_TYPES:
        return presentation_type
    return None

def normalize_title(title):
    """Normalize titles for matching duplicate publication records."""
    title = str(title).replace('\xa0', ' ').strip().lower()
    return re.sub(r'\s+', ' ', title)

def build_highlighted_presentations(df):
    """Collect spotlight/oral presentation records by title."""
    highlighted_presentations = {}
    for _, row in df.iterrows():
        if row['Type'] == 'Workshop':
            continue
        presentation_type = get_presentation_type(row)
        if not presentation_type:
            continue
        title = normalize_title(row['Title'])
        highlighted_presentations.setdefault(title, []).append({
            'type': presentation_type,
            'venue': normalize_name(row['Conference/Journal/Workshop']),
        })
    return highlighted_presentations

def get_presentation_info(row, highlighted_presentations):
    """Return presentation metadata, including duplicate workshop records."""
    presentation_type = get_presentation_type(row)
    if presentation_type:
        return presentation_type, None

    matches = highlighted_presentations.get(normalize_title(row['Title']), [])
    if matches:
        match = matches[0]
        return match['type'], match['venue']

    return None, None

def paper_to_bibtex(row, seen_keys, highlighted_presentations):
    """Convert a paper row to BibTeX entry."""
    # Skip workshop papers
    if row['Type'] == 'Workshop':
        return None
    # Include preprints only if they have an arxiv link
    if row['Type'] == 'Preprint':
        if not (pd.notna(row.get('Arxiv link')) and row['Arxiv link']):
            return None

    key = generate_bibtex_key(row)
    # Handle duplicate keys
    if key in seen_keys:
        seen_keys[key] += 1
        key = f"{key}{chr(ord('a') + seen_keys[key] - 1)}"
    else:
        seen_keys[key] = 1

    entry_type = get_entry_type(row)
    venue = row['Conference/Journal/Workshop']
    year = get_field(row, 'year', row['Year'])

    # Build BibTeX entry
    lines = [f"@{entry_type}{{{key},"]
    lines.append(f'  title={{{row["Title"]}}},')
    lines.append(f'  author={{{format_authors_with_contribution_marks(row)}}},')

    if entry_type == 'inproceedings':
        lines.append(f'  booktitle={{{get_venue_name(venue, row["Type"])}}},')
    elif entry_type == 'article':
        if row['Type'] == 'Preprint':
            lines.append('  journal={arXiv},')
        else:
            lines.append(f'  journal={{{get_venue_name(venue, row["Type"])}}},')

    lines.append(f'  year={{{year}}},')

    # Add abbreviation
    abbr = VENUE_ABBR.get(venue, venue)
    lines.append(f'  abbr={{{abbr}}},')

    # Add highlighted presentation history
    presentation_type, presentation_venue = get_presentation_info(row, highlighted_presentations)
    if presentation_type:
        lines.append(f'  presentation={{{presentation_type}}},')
    if presentation_venue:
        lines.append(f'  presentation_venue={{{presentation_venue}}},')

    # Add links
    paper_link = get_field(row, 'html', row.get('Paper link'))
    arxiv_link = get_field(row, 'arxiv', row.get('Arxiv link'))
    code_link = get_field(row, 'code', row.get('Code link'))
    website_link = get_field(row, 'website', row.get('Project page link'))

    if pd.notna(paper_link) and paper_link:
        lines.append(f'  html={{{paper_link}}},')
    if pd.notna(arxiv_link) and arxiv_link:
        lines.append(f'  arxiv={{{str(arxiv_link).replace("https://arxiv.org/abs/", "")}}},')
    if pd.notna(code_link) and code_link:
        lines.append(f'  code={{{code_link}}},')
    if pd.notna(website_link) and website_link and website_link != '-':
        lines.append(f'  website={{{website_link}}},')

    # Mark selected papers (from "Selected" column in Excel)
    selected = row.get('Selected', '')
    if pd.notna(selected):
        selected_str = str(selected).strip().lower()
        if selected_str in ['true', 'yes', '1', '1.0', 'o'] or selected_str.startswith('1'):
            lines.append('  selected={true},')

    lines.append('}')
    return '\n'.join(lines)

def main():
    print(f"Reading papers from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Reverse order (last row in Excel appears first)
    df = df.iloc[::-1]
    highlighted_presentations = build_highlighted_presentations(df)

    # Generate BibTeX entries
    seen_keys = {}
    entries = []

    for _, row in df.iterrows():
        entry = paper_to_bibtex(row, seen_keys, highlighted_presentations)
        if entry:
            entries.append(entry)

    # Write to file
    output = "---\n---\n\n" + "\n\n".join(entries) + "\n"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output)

    print(f"Generated {len(entries)} BibTeX entries")
    print(f"Output written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
