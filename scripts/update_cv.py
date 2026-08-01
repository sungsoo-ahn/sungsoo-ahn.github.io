#!/usr/bin/env python3
"""Synchronize the structured CV data, LaTeX sources, and website PDF.

Usage:
    uv run python scripts/update_cv.py
    uv run python scripts/update_cv.py --no-compile
    uv run python scripts/update_cv.py --check --no-compile
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Paths
PUBLICATIONS_PATH = Path(__file__).parent.parent / "_data/publications.yml"
OUTPUT_PATH = Path(__file__).parent.parent / "cv/publications.tex"
CV_DATA_PATH = Path(__file__).parent.parent / "_data/cv_content.yml"
CV_TEX_PATH = Path(__file__).parent.parent / "cv/cv.tex"
CV_PDF_PATH = Path(__file__).parent.parent / "cv/cv.pdf"
WEB_PDF_PATH = Path(__file__).parent.parent / "assets/pdf/cv.pdf"
SOURCE_HASH_PATH = Path(__file__).parent.parent / "cv/source.sha256"

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


def load_publications(path: Path = PUBLICATIONS_PATH) -> list[dict]:
    """Load the canonical YAML publication records."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def get_abbr(entry: dict) -> str:
    return entry.get("abbr", "").strip()


def get_year(entry: dict) -> int:
    try:
        return int(entry.get("year", "0"))
    except (TypeError, ValueError):
        return 0


def sort_key(entry: dict) -> tuple:
    """Sort: year descending, then conference priority ascending."""
    abbr = get_abbr(entry)
    priority = CONF_PRIORITY.get(abbr, 99)
    return (-get_year(entry), priority)


def preprint_sort_key(entry: dict) -> tuple:
    """Sort preprints by year and then exact submission timestamp when supplied."""
    return (-get_year(entry), -int(entry.get("cv_sort_order", "0")))


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters in text (but not commands)."""
    for source, target in UNICODE_TEX.items():
        s = s.replace(source, target)
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s


def tex_escape_plain(value: object) -> str:
    """Escape human-readable YAML text for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "—": "---",
    }
    return "".join(replacements.get(char, char) for char in str(value))


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


def format_authors(authors: list[str]) -> str:
    """Format the full author list."""
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
    venue = entry.get("venue", "").strip()

    if entry.get("cv_venue"):
        full = entry["cv_venue"].strip()
    elif abbr == "KDD":
        full = "ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), Datasets and Benchmarks Track"
    else:
        full = CONFERENCE_FULL.get(abbr, venue)

    return rf"in \textit{{{full}}}"


def format_venue_journal(entry: dict) -> str:
    """Format venue string for a journal entry."""
    abbr = get_abbr(entry)
    full = entry.get("cv_venue", JOURNAL_FULL.get(abbr, entry.get("venue", abbr)))

    parts = [rf"\textit{{{full}}}"]

    # Add volume/number/pages if present
    override_details = entry.get("cv_details", "").strip()
    if override_details:
        parts.append(override_details)
        return ", ".join(parts)

    return ", ".join(parts)


def format_venue_preprint(entry: dict) -> str:
    """Format venue string for a preprint entry."""
    return r"\textit{arXiv}"


def format_paper_link(entry: dict, category: str) -> str:
    """Prefer an official paper page, falling back to arXiv."""
    official = entry.get("html", "").strip()
    arxiv = entry.get("arxiv", "").strip()
    if category != "preprint" and official:
        official = official.replace("&", r"\&")
        return rf" \href{{{official}}}{{[paper]}}"
    if arxiv:
        return rf" \href{{https://arxiv.org/abs/{arxiv}}}{{[arXiv]}}"
    return ""


def format_annotation(entry: dict) -> str:
    """Format annotation (award) if present."""
    ann = entry.get("cv_annotation", "").strip()
    if not ann and entry.get("presentation", "").strip():
        ann = f"{entry['presentation'].strip().lower()} presentation"
    if ann:
        ann = tex_escape(ann)
        return rf", \textcolor{{WineRed}}{{{ann}}}"
    return ""


def format_entry(entry: dict, category: str) -> str:
    """Format a single bib entry as a LaTeX \\item line."""
    title = entry.get("cv_title_tex", entry.get("title", "")).strip()
    authors = format_authors(entry.get("authors", []))
    year = str(entry.get("year", "")).strip()

    if category == "conference":
        venue = format_venue_conference(entry)
    elif category == "journal":
        venue = format_venue_journal(entry)
    else:
        venue = format_venue_preprint(entry)

    paper_link = format_paper_link(entry, category)
    annotation = format_annotation(entry)

    return rf"\item {authors}, {title}, {venue}, {year}{annotation}.{paper_link}"


def build_publications() -> tuple[str, tuple[int, int, int]]:
    """Render publications.tex and return its category counts."""
    entries = load_publications()
    conferences = [entry for entry in entries if entry.get("type") == "conference"]
    journals = [entry for entry in entries if entry.get("type") == "journal"]
    preprints = [
        entry
        for entry in entries
        if entry.get("type") == "preprint" and str(entry.get("arxiv", "")).strip()
    ]

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

    counts = (len(conferences), len(journals), len(preprints))
    return "\n".join(lines) + "\n", counts


def load_cv_data(path: Path = CV_DATA_PATH) -> dict:
    """Load and validate the structured, hand-maintained CV sections."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")

    list_sections = ("education", "employment", "service", "talks", "courses", "awards")
    for section in list_sections:
        entries = data.get(section)
        if not isinstance(entries, list):
            raise ValueError(f"{path}: {section} must be a list")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: {section}[{index}] must be a mapping")
            missing = {"title", "organization", "date"} - set(entry)
            if missing:
                raise ValueError(f"{path}: {section}[{index}] is missing {sorted(missing)}")
            url = entry.get("organization_url")
            if url and not str(url).startswith(("https://", "http://")):
                raise ValueError(f"{path}: {section}[{index}] has an invalid organization_url")

    grants = data.get("grants")
    if not isinstance(grants, dict):
        raise ValueError(f"{path}: grants must be a mapping")
    for group in ("research", "computing"):
        entries = grants.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"{path}: grants.{group} must be a list")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: grants.{group}[{index}] must be a mapping")
            missing = {"title", "organization", "date", "detail"} - set(entry)
            if missing:
                raise ValueError(
                    f"{path}: grants.{group}[{index}] is missing {sorted(missing)}"
                )
    return data


def render_organization(entry: dict) -> str:
    raw_organization = str(entry["organization"])
    organization = tex_escape_plain(raw_organization)
    url = entry.get("organization_url")
    if not url:
        return organization
    safe_url = str(url).replace("%", r"\%").replace("#", r"\#")
    link_text = str(entry.get("organization_link_text", raw_organization))
    if not raw_organization.startswith(link_text):
        raise ValueError("organization_link_text must be a prefix of organization")
    suffix = tex_escape_plain(raw_organization[len(link_text):])
    return rf"\href{{{safe_url}}}{{{tex_escape_plain(link_text)}}}{suffix}"


def render_entry(entry: dict, command: str) -> str:
    return (
        rf"\{command}{{{tex_escape_plain(entry['title'])}}}"
        rf"{{{render_organization(entry)}}}"
        rf"{{{tex_escape_plain(entry['date'])}}}"
    )


def render_entry_list(entries: list[dict], command: str, environment: str = "enumerate") -> str:
    lines = [rf"\begin{{{environment}}}"]
    lines.extend(render_entry(entry, command) for entry in entries)
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def render_compact_entry_list(entries: list[dict]) -> str:
    options = "[itemsep=0pt,topsep=0pt,parsep=0pt,partopsep=0pt]"
    lines = [rf"\begin{{enumerate}}{options}"]
    lines.extend(render_entry(entry, "cventry") for entry in entries)
    lines.append(r"\end{enumerate}")
    return "\n".join(lines)


def render_talks(entries: list[dict]) -> str:
    return "\n".join(
        (
            r"\begingroup",
            r"\setlength{\emergencystretch}{2em}",
            render_entry_list(entries, "cventry"),
            r"\endgroup",
        )
    )


def render_grants(grants: dict[str, list[dict]]) -> str:
    lines = [r"\begingroup", r"\setlength{\emergencystretch}{2em}"]
    options = "[itemsep=0pt,topsep=0pt,parsep=0pt,partopsep=0pt]"
    for group, label in (("research", "Research"), ("computing", "Computing")):
        lines.extend((rf"\noindent\textsc{{{label}}}", rf"\begin{{enumerate}}{options}"))
        for entry in grants[group]:
            lines.append(
                rf"\grantentry{{{tex_escape_plain(entry['title'])}}}"
                rf"{{{tex_escape_plain(entry['date'])}}}"
                rf"{{{tex_escape_plain(entry['organization'])}}}"
                rf"{{{tex_escape_plain(entry['detail'])}}}"
            )
        lines.append(r"\end{enumerate}")
        if group == "research":
            lines.append("")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def render_cv_sections(data: dict) -> dict[str, str]:
    return {
        "EDUCATION": render_entry_list(data["education"], "cvunnumberedentry", "itemize"),
        "EMPLOYMENT": render_entry_list(data["employment"], "cvunnumberedentry", "itemize"),
        "SERVICE": render_entry_list(data["service"], "cventry"),
        "TALKS": render_talks(data["talks"]),
        "COURSES": render_compact_entry_list(data["courses"]),
        "AWARDS": render_compact_entry_list(data["awards"]),
        "GRANTS": render_grants(data["grants"]),
    }


def replace_generated_section(content: str, name: str, generated: str) -> str:
    """Replace exactly one protected LaTeX SYNC block."""
    begin = f"% SYNC:{name}:BEGIN"
    end = f"% SYNC:{name}:END"
    if content.count(begin) != 1 or content.count(end) != 1:
        raise ValueError(f"Expected exactly one {begin}/{end} marker pair in {CV_TEX_PATH}")
    begin_index = content.index(begin) + len(begin)
    end_index = content.index(end, begin_index)
    return content[:begin_index] + "\n" + generated.rstrip() + "\n" + content[end_index:]


def expected_cv_source(current: str, data: dict) -> str:
    for name, generated in render_cv_sections(data).items():
        current = replace_generated_section(current, name, generated)
    return current


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def compile_cv() -> None:
    """Compile the LaTeX CV and copy the successful PDF to the website."""
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk is required to compile the CV; use --no-compile to skip it")
    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", CV_TEX_PATH.name],
        cwd=CV_TEX_PATH.parent,
        check=True,
    )
    if not CV_PDF_PATH.exists():
        raise RuntimeError(f"Expected {CV_PDF_PATH} after LaTeX compilation")
    WEB_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CV_PDF_PATH, WEB_PDF_PATH)
    SOURCE_HASH_PATH.write_text(source_digest() + "\n", encoding="utf-8")


def source_digest() -> str:
    """Hash the generated LaTeX inputs used to build the committed PDF."""
    digest = hashlib.sha256()
    for path in (CV_TEX_PATH, OUTPUT_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def check_generated_files(publications: str, cv_source: str) -> list[str]:
    stale = []
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text(encoding="utf-8") != publications:
        stale.append(str(OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)))
    if CV_TEX_PATH.read_text(encoding="utf-8") != cv_source:
        stale.append(str(CV_TEX_PATH.relative_to(CV_TEX_PATH.parent.parent)))
    if not CV_PDF_PATH.exists() or not WEB_PDF_PATH.exists():
        stale.append("cv/cv.pdf or assets/pdf/cv.pdf (missing)")
    elif CV_PDF_PATH.read_bytes() != WEB_PDF_PATH.read_bytes():
        stale.append("assets/pdf/cv.pdf (does not match cv/cv.pdf)")
    if not SOURCE_HASH_PATH.exists():
        stale.append("cv/source.sha256 (missing; compile the CV once)")
    elif SOURCE_HASH_PATH.read_text(encoding="utf-8").strip() != source_digest():
        stale.append("cv/cv.pdf (generated from older LaTeX sources)")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true", help="update sources without compiling")
    parser.add_argument("--check", action="store_true", help="fail if committed generated files are stale")
    args = parser.parse_args(argv)

    data = load_cv_data()
    publications, counts = build_publications()
    current_cv = CV_TEX_PATH.read_text(encoding="utf-8")
    cv_source = expected_cv_source(current_cv, data)

    if args.check:
        stale = check_generated_files(publications, cv_source)
        if stale:
            print("CV synchronization check failed. Regenerate these files:", file=sys.stderr)
            for path in stale:
                print(f"  - {path}", file=sys.stderr)
            return 1
        print("CV sources and website PDF are synchronized.")
        return 0

    publications_changed = write_if_changed(OUTPUT_PATH, publications)
    cv_changed = write_if_changed(CV_TEX_PATH, cv_source)
    if not args.no_compile:
        compile_cv()

    conference_count, journal_count, preprint_count = counts
    total = sum(counts)
    print(
        f"Publications: {conference_count} conference, {journal_count} journal, "
        f"{preprint_count} preprint ({total} total)"
    )
    print(f"Updated sources: publications={publications_changed}, cv={cv_changed}")
    if args.no_compile:
        print("Skipped PDF compilation.")
    else:
        print(f"Compiled {CV_PDF_PATH} and copied it to {WEB_PDF_PATH}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
