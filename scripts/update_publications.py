#!/usr/bin/env python3
"""Validate the canonical YAML publication data used by the website and CV.

Publication edits no longer require a generated BibTeX file. Edit
``_data/publications.yml`` directly, then run this command and regenerate the
CV with ``scripts/update_cv.py``.
"""

import argparse
import re
from pathlib import Path

import yaml


PUBLICATIONS_PATH = Path(__file__).parent.parent / "_data/publications.yml"
PUBLICATION_TYPES = {"conference", "journal", "preprint"}
REQUIRED_FIELDS = {"id", "type", "title", "authors", "venue", "year", "abbr"}
URL_FIELDS = {"html", "code", "code2", "website", "video"}
OPTIONAL_FIELDS = {
    "presentation",
    "presentation_venue",
    "arxiv",
    "code_label",
    "code2_label",
    "selected",
    "annotation",
    "award",
    "abstract",
    "preview",
    "pdf",
    "cv_title_tex",
    "cv_venue",
    "cv_details",
    "cv_annotation",
    "cv_sort_order",
}
ALLOWED_FIELDS = REQUIRED_FIELDS | URL_FIELDS | OPTIONAL_FIELDS
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")
ARXIV_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def load_publications(path: Path = PUBLICATIONS_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of publications")
    return data


def validate_publications(publications: list[dict]) -> list[str]:
    """Return actionable schema and consistency errors."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    previous_year: int | None = None

    for index, publication in enumerate(publications, start=1):
        label = f"publication #{index}"
        if not isinstance(publication, dict):
            problems.append(f"{label} must be a mapping")
            continue

        publication_id = publication.get("id", f"#{index}")
        label = str(publication_id)
        missing = REQUIRED_FIELDS - set(publication)
        if missing:
            problems.append(f"{label}: missing required fields {sorted(missing)}")
            continue

        unknown = set(publication) - ALLOWED_FIELDS
        if unknown:
            problems.append(f"{label}: unknown fields {sorted(unknown)}")

        if not isinstance(publication_id, str) or not ID_PATTERN.fullmatch(publication_id):
            problems.append(f"{label}: id must match {ID_PATTERN.pattern}")
        elif publication_id in seen_ids:
            problems.append(f"{label}: duplicate id")
        seen_ids.add(str(publication_id))

        publication_type = publication["type"]
        if publication_type not in PUBLICATION_TYPES:
            problems.append(f"{label}: type must be one of {sorted(PUBLICATION_TYPES)}")
        if publication_type == "preprint" and publication["abbr"] != "-":
            problems.append(f"{label}: preprints must use abbr: '-' ")
        if publication_type != "preprint" and publication["abbr"] == "-":
            problems.append(f"{label}: non-preprints cannot use abbr: '-' ")

        authors = publication["authors"]
        if not isinstance(authors, list) or not authors or not all(
            isinstance(author, str) and author.strip() for author in authors
        ):
            problems.append(f"{label}: authors must be a non-empty list of names")
        elif any("†" in author for author in authors):
            dagger_count = sum("†" in author for author in authors)
            if dagger_count < 2:
                problems.append(f"{label}: † must mark at least two corresponding authors")

        year = publication["year"]
        if not isinstance(year, int) or year < 1900 or year > 2100:
            problems.append(f"{label}: year must be an integer between 1900 and 2100")
        elif previous_year is not None and year > previous_year:
            problems.append(f"{label}: entries must be ordered by descending year")
        if isinstance(year, int):
            previous_year = year

        for field in URL_FIELDS:
            value = publication.get(field)
            if value and (not isinstance(value, str) or not value.startswith(("https://", "http://"))):
                problems.append(f"{label}: {field} must be an HTTP(S) URL")

        arxiv = publication.get("arxiv")
        if arxiv is not None and not ARXIV_PATTERN.fullmatch(str(arxiv)):
            problems.append(f"{label}: arxiv must be an identifier such as 2606.22866")
        if "selected" in publication and not isinstance(publication["selected"], bool):
            problems.append(f"{label}: selected must be true or false")
        if "cv_sort_order" in publication and not isinstance(publication["cv_sort_order"], int):
            problems.append(f"{label}: cv_sort_order must be an integer timestamp")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate without modifying files (currently the default behavior)",
    )
    parser.parse_args(argv)

    publications = load_publications()
    problems = validate_publications(publications)
    if problems:
        print(f"Publication validation failed with {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    counts = {
        publication_type: sum(pub["type"] == publication_type for pub in publications)
        for publication_type in sorted(PUBLICATION_TYPES)
    }
    selected = sum(bool(pub.get("selected")) for pub in publications)
    print(
        f"Validated {len(publications)} publications: "
        f"{counts['conference']} conference, {counts['journal']} journal, "
        f"{counts['preprint']} preprint; {selected} selected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
