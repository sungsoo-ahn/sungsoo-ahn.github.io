#!/usr/bin/env python3
"""
Script to generate papers.bib from the SPML lab papers.xlsx file.
Usage: python scripts/update_publications.py
"""

import pandas as pd
import re
from pathlib import Path

# Configuration
EXCEL_PATH = Path.home() / "Sungsahn0215 Dropbox/SPML/administration/papers.xlsx"
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

def clean_author_name(name):
    """Clean author name for BibTeX key generation."""
    name = name.strip()
    name = re.sub(r'\*|†', '', name)  # Remove special markers
    return name

def generate_bibtex_key(row):
    """Generate a unique BibTeX key from the paper data."""
    first_author = clean_author_name(row['First Author'].split(',')[0].split(' ')[-1].lower())
    year = str(row['Year'])
    # Take first significant word from title
    title_words = row['Title'].lower().split()
    skip_words = {'a', 'an', 'the', 'for', 'of', 'in', 'on', 'to', 'via', 'with', 'and'}
    title_word = next((w for w in title_words if w not in skip_words), title_words[0])
    title_word = re.sub(r'[^a-z]', '', title_word)
    return f"{first_author}{year}{title_word}"

def format_authors_bibtex(authors_str):
    """Format authors string for BibTeX."""
    authors_str = re.sub(r'\*|†', '', authors_str)  # Remove markers
    authors = [a.strip() for a in authors_str.split(',')]
    return ' and '.join(authors)

def get_entry_type(row):
    """Determine BibTeX entry type."""
    pub_type = row['Type']
    if pub_type == 'Journal':
        return 'article'
    elif pub_type == 'Preprint':
        return 'misc'
    else:  # Conference, Workshop
        return 'inproceedings'

def get_venue_full_name(venue, pub_type):
    """Get full venue name for BibTeX."""
    venue_names = {
        "NeurIPS": "Advances in Neural Information Processing Systems",
        "ICML": "International Conference on Machine Learning",
        "ICLR": "International Conference on Learning Representations",
        "CVPR": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "IJCAI": "International Joint Conference on Artificial Intelligence",
        "AISTATS": "International Conference on Artificial Intelligence and Statistics",
        "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
        "ACL": "Annual Meeting of the Association for Computational Linguistics",
    }
    return venue_names.get(venue, venue)

def paper_to_bibtex(row, seen_keys):
    """Convert a paper row to BibTeX entry."""
    # Skip preprints and workshop papers for main bibliography
    if row['Type'] == 'Preprint':
        return None
    if row['Type'] == 'Workshop':
        return None  # Skip workshop papers

    key = generate_bibtex_key(row)
    # Handle duplicate keys
    if key in seen_keys:
        seen_keys[key] += 1
        key = f"{key}{chr(ord('a') + seen_keys[key] - 1)}"
    else:
        seen_keys[key] = 1

    entry_type = get_entry_type(row)
    venue = row['Conference/Journal/Workshop']
    year = row['Year']

    # Build BibTeX entry
    lines = [f"@{entry_type}{{{key},"]
    lines.append(f'  title={{{row["Title"]}}},')
    lines.append(f'  author={{{format_authors_bibtex(row["Authors"])}}},')

    if entry_type == 'inproceedings':
        lines.append(f'  booktitle={{{get_venue_full_name(venue, row["Type"])}}},')
    elif entry_type == 'article':
        lines.append(f'  journal={{{venue}}},')

    lines.append(f'  year={{{year}}},')

    # Add abbreviation
    abbr = VENUE_ABBR.get(venue, venue)
    lines.append(f'  abbr={{{abbr}}},')

    # Add links
    if pd.notna(row.get('Paper link')) and row['Paper link']:
        lines.append(f'  html={{{row["Paper link"]}}},')
    if pd.notna(row.get('Arxiv link')) and row['Arxiv link']:
        lines.append(f'  arxiv={{{row["Arxiv link"].replace("https://arxiv.org/abs/", "")}}},')
    if pd.notna(row.get('Code link')) and row['Code link']:
        lines.append(f'  code={{{row["Code link"]}}},')
    if pd.notna(row.get('Project page link')) and row['Project page link'] and row['Project page link'] != '-':
        lines.append(f'  website={{{row["Project page link"]}}},')

    # Mark selected papers (spotlight, oral, or recent top venues)
    presentation = row.get('Presentation Type', '')
    if presentation in ['Spotlight', 'Oral'] or (year >= 2024 and venue in ['NeurIPS', 'ICML', 'ICLR']):
        lines.append('  selected={true},')

    lines.append('}')
    return '\n'.join(lines)

def main():
    print(f"Reading papers from: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Sort by year (descending) then by venue
    df = df.sort_values(['Year', 'Conference/Journal/Workshop'], ascending=[False, True])

    # Generate BibTeX entries
    seen_keys = {}
    entries = []

    for _, row in df.iterrows():
        entry = paper_to_bibtex(row, seen_keys)
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
