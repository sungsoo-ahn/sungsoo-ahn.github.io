#!/usr/bin/env python3
"""
Generate cv/publications.tex from _bibliography/papers.bib.
Usage: uv run python scripts/update_cv.py
"""

import re
from pathlib import Path

import bibtexparser
import yaml

# Paths
BIB_PATH = Path(__file__).parent.parent / "_bibliography/papers.bib"
OUTPUT_PATH = Path(__file__).parent.parent / "cv/publications.tex"
OVERRIDES_PATH = Path(__file__).parent.parent / "cv/publication_overrides.yml"

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
    "CVPR": "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    "IJCAI": "International Joint Conference on Artificial Intelligence (IJCAI)",
    "AISTATS": "International Conference on Artificial Intelligence and Statistics (AISTATS)",
    "EMNLP": "Empirical Methods in Natural Language Processing (EMNLP)",
    "ACL": "Annual Meeting of the Association for Computational Linguistics (ACL)",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)",
}

# Full venue names for journals
JOURNAL_FULL = {
    "TMLR": "Transactions of Machine Learning Research (TMLR)",
    "JSTAT": "Journal of Statistical Mechanics: Theory and Experiment",
    "IEEE TIT": "IEEE Transactions on Information Theory",
}

# Reverse calendar order within the same year (later conferences first).
CONF_PRIORITY = {
    "NeurIPS": 0, "EMNLP": 1, "KDD": 2, "IJCAI": 3, "ACL": 4,
    "ICML": 5, "CVPR": 6, "AISTATS": 7, "ICLR": 8,
}

MY_NAME = "Sungsoo Ahn"

UNICODE_TEX = {
    "André": r"Andr\'{e}",
    "Gómez": r"G\'{o}mez",
}


def parse_bib(path: Path) -> list[dict]:
    """Parse papers.bib, stripping the YAML front matter."""
    text = path.read_text()
    # Strip YAML front matter (---\n---\n)
    text = re.sub(r"^---\s*\n---\s*\n", "", text)
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    bib_db = bibtexparser.loads(text, parser=parser)
    return bib_db.entries


def load_overrides(path: Path) -> dict[str, dict]:
    """Load citation-keyed CV rendering overrides."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def apply_overrides(entries: list[dict], overrides: dict[str, dict]) -> list[dict]:
    """Merge rendering-only overrides without changing the bibliography."""
    known_ids = {entry["ID"] for entry in entries}
    unknown_ids = sorted(set(overrides) - known_ids)
    if unknown_ids:
        raise ValueError(f"Unknown citation keys in {OVERRIDES_PATH}: {unknown_ids}")

    for entry in entries:
        override = overrides.get(entry["ID"], {})
        if not isinstance(override, dict):
            raise ValueError(f"Override for {entry['ID']} must be a mapping")
        entry.update(override)
    return entries


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


def preprint_sort_key(entry: dict) -> tuple:
    """Sort preprints by year and then exact submission timestamp when supplied."""
    return (-get_year(entry), -int(entry.get("sort_order", "0")))


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters in text (but not commands)."""
    for source, target in UNICODE_TEX.items():
        s = s.replace(source, target)
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
    clean = tex_escape(clean)
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

    if entry.get("venue"):
        full = entry["venue"].strip()
    # ACL Findings special case: use the official anthology volume title.
    elif "Findings" in booktitle:
        full = f"Findings of the Association for Computational Linguistics: {abbr} {entry['year']}"
    elif abbr == "KDD":
        full = "ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), Datasets and Benchmarks Track"
    else:
        full = CONFERENCE_FULL.get(abbr, booktitle)

    return rf"in \textit{{{full}}}"


def format_venue_journal(entry: dict) -> str:
    """Format venue string for a journal entry."""
    abbr = get_abbr(entry)
    full = entry.get("venue", JOURNAL_FULL.get(abbr, entry.get("journal", abbr)))

    parts = [rf"\textit{{{full}}}"]

    # Add volume/number/pages if present
    override_details = entry.get("details", "").strip()
    if override_details:
        parts.append(override_details)
        return ", ".join(parts)

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


def format_paper_link(entry: dict, category: str) -> str:
    """Prefer an official paper page, falling back to arXiv."""
    official = entry.get("paper_url", entry.get("html", "")).strip()
    arxiv = entry.get("arxiv", "").strip()
    if category != "preprint" and official:
        official = official.replace("&", r"\&")
        return rf" \href{{{official}}}{{[paper]}}"
    if arxiv:
        return rf" \href{{https://arxiv.org/abs/{arxiv}}}{{[arXiv]}}"
    return ""


def format_annotation(entry: dict) -> str:
    """Format annotation (award) if present."""
    ann = entry.get("annotation", "").strip()
    if not ann and entry.get("presentation", "").strip():
        ann = f"{entry['presentation'].strip().lower()} presentation"
    if ann:
        ann = tex_escape(ann)
        return rf", \textcolor{{WineRed}}{{{ann}}}"
    return ""


def format_entry(entry: dict, category: str) -> str:
    """Format a single bib entry as a LaTeX \\item line."""
    title = entry.get("title_tex", entry.get("title", "")).strip()
    authors = format_authors(entry.get("author", ""))
    year = entry.get("year", "").strip()

    if category == "conference":
        venue = format_venue_conference(entry)
    elif category == "journal":
        venue = format_venue_journal(entry)
    else:
        venue = format_venue_preprint(entry)

    paper_link = format_paper_link(entry, category)
    annotation = format_annotation(entry)

    return rf"\item {authors}, {title}, {venue}, {year}{annotation}.{paper_link}"


def main():
    entries = apply_overrides(parse_bib(BIB_PATH), load_overrides(OVERRIDES_PATH))

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
    preprints.sort(key=preprint_sort_key)

    lines = []

    # Conference section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Conference Papers}")
    lines.append(r"\begin{enumerate}")
    for e in conferences:
        lines.append(format_entry(e, "conference"))
    lines.append(r"\end{enumerate}")
    lines.append("")

    # Journal section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Journal Articles}")
    lines.append(r"\begin{enumerate}")
    for e in journals:
        lines.append(format_entry(e, "journal"))
    lines.append(r"\end{enumerate}")
    lines.append("")

    # Preprint section
    lines.append(r"\vspace{0.5\baselineskip}")
    lines.append(r"\textsc{Preprints}")
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
