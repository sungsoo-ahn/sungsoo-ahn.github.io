#!/usr/bin/env python3
"""
Generate cv/publications.tex from _bibliography/papers.bib.
Usage: uv run python scripts/update_cv.py
"""

import re
from pathlib import Path

import bibtexparser

# Paths
BIB_PATH = Path(__file__).parent.parent / "_bibliography/papers.bib"
OUTPUT_PATH = Path(__file__).parent.parent / "cv/publications.tex"

# Which abbr values belong to which category
CONFERENCE_ABBRS = {
    "NeurIPS", "ICML", "ICLR", "CVPR", "IJCAI", "AISTATS",
    "EMNLP", "ACL", "KDD",
}
JOURNAL_ABBRS = {"TMLR", "JSTAT", "IEEE TIT"}
PREPRINT_ABBR = "-"

# Full venue names for conferences
CONFERENCE_FULL = {
    "NeurIPS": "Conference on Neural Information Processing Systems (NeurIPS)",
    "ICML": "International Conference on Machine Learning (ICML)",
    "ICLR": "International Conference on Learning Representations (ICLR)",
    "CVPR": "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    "IJCAI": "International Joint Conference on Artificial Intelligence (IJCAI)",
    "AISTATS": "International Conference on Artificial Intelligence and Statistics (AISTATS)",
    "EMNLP": "Empirical Methods in Natural Language Processing (EMNLP)",
    "ACL": "Annual Meeting of the Association for Computational Linguistics (ACL)",
    "KDD": "Knowledge Discovery and Data Mining (KDD)",
}

# Full venue names for journals
JOURNAL_FULL = {
    "TMLR": "Transactions of Machine Learning Research (TMLR)",
    "JSTAT": "Journal of Statistical Mechanics: Theory and Experiment",
    "IEEE TIT": "IEEE Transactions on Information Theory",
}

# Conference priority for sorting within the same year (lower = earlier)
CONF_PRIORITY = {
    "ICLR": 0, "ICML": 1, "ACL": 2, "IJCAI": 3, "KDD": 4,
    "EMNLP": 5, "NeurIPS": 6, "CVPR": 7, "AISTATS": 8,
}

MY_NAME = "Sungsoo Ahn"


def parse_bib(path: Path) -> list[dict]:
    """Parse papers.bib, stripping the YAML front matter."""
    text = path.read_text()
    # Strip YAML front matter (---\n---\n)
    text = re.sub(r"^---\s*\n---\s*\n", "", text)
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    bib_db = bibtexparser.loads(text, parser=parser)
    return bib_db.entries


def get_abbr(entry: dict) -> str:
    return entry.get("abbr", "").strip()


def get_year(entry: dict) -> int:
    try:
        return int(entry.get("year", "0"))
    except ValueError:
        return 0


def sort_key(entry: dict) -> tuple:
    """Sort: year descending, then conference priority ascending."""
    abbr = get_abbr(entry)
    priority = CONF_PRIORITY.get(abbr, 99)
    return (-get_year(entry), priority)


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters in text (but not commands)."""
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s


def format_author(name: str) -> str:
    """Format a single author name for LaTeX output.

    Handles:
    - Bold own name with \\me{}
    - Convert * to $^*$ and † to $^\\dagger$
    """
    name = name.strip()

    # Detect markers
    markers = ""
    clean = name
    for char in ["*", "\u2020"]:  # * and †
        if char in clean:
            clean = clean.replace(char, "").strip()
            if char == "*":
                markers += "$^*$"
            else:
                markers += r"$^\dagger$"

    # Check if this is our name
    if clean == MY_NAME:
        return rf"\me{{{clean}}}{markers}"
    return f"{clean}{markers}"


def format_authors(author_str: str) -> str:
    """Format the full author string."""
    # bibtexparser joins with ' and '
    authors = [a.strip() for a in author_str.split(" and ")]
    formatted = [format_author(a) for a in authors]

    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def format_venue_conference(entry: dict) -> str:
    """Format venue string for a conference entry."""
    abbr = get_abbr(entry)
    booktitle = entry.get("booktitle", "").strip()

    # ACL Findings special case: use booktitle directly
    if "Findings" in booktitle:
        full = "Annual Meeting of the Association for Computational Linguistics (ACL) Findings"
    elif abbr == "KDD":
        full = "Knowledge Discovery and Data Mining (KDD) Datasets and Benchmarks"
    else:
        full = CONFERENCE_FULL.get(abbr, booktitle)

    return rf"In \textit{{{full}}}"


def format_venue_journal(entry: dict) -> str:
    """Format venue string for a journal entry."""
    abbr = get_abbr(entry)
    full = JOURNAL_FULL.get(abbr, entry.get("journal", abbr))

    parts = [rf"In \textit{{{full}}}"]

    # Add volume/number/pages if present
    vol = entry.get("volume", "")
    num = entry.get("number", "")
    pages = entry.get("pages", "")
    if vol:
        detail = vol
        if num:
            detail += f"({num})"
        if pages:
            detail += f", {pages}"
        parts.append(detail)

    return ", ".join(parts)


def format_venue_preprint(entry: dict) -> str:
    """Format venue string for a preprint entry."""
    return r"\textit{arXiv}"


def format_arxiv_link(entry: dict) -> str:
    """Format arxiv link if present."""
    arxiv = entry.get("arxiv", "").strip()
    if arxiv:
        return rf" \href{{https://arxiv.org/abs/{arxiv}}}{{[arXiv]}}"
    return ""


def format_annotation(entry: dict) -> str:
    """Format annotation (award) if present."""
    ann = entry.get("annotation", "").strip()
    if ann:
        ann = tex_escape(ann)
        return rf", \textcolor{{WineRed}}{{{ann}}}"
    return ""


def format_entry(entry: dict, category: str) -> str:
    """Format a single bib entry as a LaTeX \\item line."""
    title = entry.get("title", "").strip()
    authors = format_authors(entry.get("author", ""))
    year = entry.get("year", "").strip()

    if category == "conference":
        venue = format_venue_conference(entry)
    elif category == "journal":
        venue = format_venue_journal(entry)
    else:
        venue = format_venue_preprint(entry)

    arxiv = format_arxiv_link(entry)
    annotation = format_annotation(entry)

    return rf"\item {authors}, {title}, {venue}, {year}{annotation}.{arxiv}"


def main():
    entries = parse_bib(BIB_PATH)

    conferences = []
    journals = []
    preprints = []

    for e in entries:
        abbr = get_abbr(e)
        if abbr in CONFERENCE_ABBRS:
            conferences.append(e)
        elif abbr in JOURNAL_ABBRS:
            journals.append(e)
        elif abbr == PREPRINT_ABBR and e.get("arxiv", "").strip():
            preprints.append(e)
        # Skip workshops and preprints without arxiv

    conferences.sort(key=sort_key)
    journals.sort(key=sort_key)
    preprints.sort(key=sort_key)

    lines = []

    # Conference section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Conference}")
    lines.append(r"\begin{enumerate}")
    for e in conferences:
        lines.append(format_entry(e, "conference"))
    lines.append(r"\end{enumerate}")
    lines.append("")

    # Journal section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Journal}")
    lines.append(r"\begin{enumerate}")
    for e in journals:
        lines.append(format_entry(e, "journal"))
    lines.append(r"\end{enumerate}")
    lines.append("")

    # Preprint section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Preprint}")
    lines.append(r"\begin{enumerate}")
    for e in preprints:
        lines.append(format_entry(e, "preprint"))
    lines.append(r"\end{enumerate}")

    output = "\n".join(lines) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output)

    print(f"Conference: {len(conferences)}, Journal: {len(journals)}, Preprint: {len(preprints)}")
    print(f"Total: {len(conferences) + len(journals) + len(preprints)} entries")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
