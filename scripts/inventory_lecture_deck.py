#!/usr/bin/env python3
"""Create an agent-facing slide and media inventory from a lecture PPTX.

The inventory is deliberately source-oriented: it records slide order, all text
runs, embedded media relationships, and a conservative initial classification.
An editor must review the classifications before declaring coverage complete.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import posixpath
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


TEXT_TAG = "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
REL_TAG = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
REL_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

LOGISTICAL_RE = re.compile(
    r"\b(course information|course schedule|important notice|notice|announcement|"
    r"homework|project|team building|presentation date)\b",
    re.IGNORECASE,
)
TRANSITION_RE = re.compile(
    r"^(part\s+\d+|today(?:'s lecture)?|lecture overview|overview|end of (?:slide|section|lecture))\b",
    re.IGNORECASE,
)
RECAP_RE = re.compile(r"^(recap|contents so far|summary)\b", re.IGNORECASE)


def slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    if not match:
        raise ValueError(f"not a slide XML path: {name}")
    return int(match.group(1))


def classify(title: str) -> str:
    if LOGISTICAL_RE.search(title):
        return "logistical"
    if TRANSITION_RE.search(title):
        return "transition-only"
    if RECAP_RE.search(title):
        return "repeated-recap"
    return "substantive"


def relationship_map(archive: zipfile.ZipFile, slide_name: str) -> dict[str, str]:
    slide_path = PurePosixPath(slide_name)
    rel_name = str(slide_path.parent / "_rels" / f"{slide_path.name}.rels")
    if rel_name not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rel_name))
    mapping: dict[str, str] = {}
    for rel in root.findall(REL_TAG):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            mapping[rel_id] = posixpath.normpath(posixpath.join(slide_path.parent, target))
    return mapping


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def pdf_pages(pdf: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    pages: list[dict[str, object]] = []
    for number, page in enumerate(result.stdout.split("\f"), start=1):
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if not lines:
            continue
        pages.append({"slide": number, "title": lines[0], "text": lines})
    return pages


def align_pages(
    pages: list[dict[str, object]], ppt_slides: list[dict[str, object]]
) -> list[int | None]:
    """Align PDF pages to PPTX slides while allowing hidden/versioned slides."""

    n, m = len(pages), len(ppt_slides)
    gap = -0.35
    dp = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    choice = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        choice[i][0] = "page-gap"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        choice[0][j] = "ppt-gap"

    for i in range(1, n + 1):
        left = normalize_title(str(pages[i - 1]["title"]))
        for j in range(1, m + 1):
            right = normalize_title(str(ppt_slides[j - 1]["title"]))
            ratio = difflib.SequenceMatcher(None, left, right).ratio()
            match_score = (2.0 * ratio) - 0.75 if ratio >= 0.35 else -1.5
            candidates = {
                "match": dp[i - 1][j - 1] + match_score,
                "page-gap": dp[i - 1][j] + gap,
                "ppt-gap": dp[i][j - 1] + gap,
            }
            move = max(candidates, key=candidates.get)
            dp[i][j] = candidates[move]
            choice[i][j] = move

    aligned: list[int | None] = [None] * n
    i, j = n, m
    while i or j:
        move = choice[i][j]
        if move == "match":
            left = normalize_title(str(pages[i - 1]["title"]))
            right = normalize_title(str(ppt_slides[j - 1]["title"]))
            if difflib.SequenceMatcher(None, left, right).ratio() >= 0.35:
                aligned[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == "page-gap":
            i -= 1
        else:
            j -= 1
    return aligned


def inventory(pptx: Path, pdf: Path, deck_id: str) -> dict[str, object]:
    with zipfile.ZipFile(pptx) as archive:
        slide_names = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=slide_number,
        )
        media_cache: dict[str, dict[str, object]] = {}
        ppt_slides: list[dict[str, object]] = []

        for slide_name in slide_names:
            root = ET.fromstring(archive.read(slide_name))
            text_runs = [
                node.text.strip()
                for node in root.iter(TEXT_TAG)
                if node.text and node.text.strip()
            ]
            title = text_runs[0] if text_runs else f"Slide {slide_number(slide_name)}"
            rels = relationship_map(archive, slide_name)
            media_targets: list[str] = []
            for node in root.iter():
                embed = node.attrib.get(REL_EMBED)
                if not embed or embed not in rels:
                    continue
                target = rels[embed]
                if not target.startswith("ppt/media/") or target not in archive.namelist():
                    continue
                payload = archive.read(target)
                digest = hashlib.sha256(payload).hexdigest()
                if digest not in media_cache:
                    media_cache[digest] = {
                        "sha256": digest,
                        "pptx_path": target,
                        "extension": Path(target).suffix.lower(),
                        "bytes": len(payload),
                        "slides": [],
                    }
                media_cache[digest]["slides"].append(slide_number(slide_name))
                media_targets.append(digest)

            ppt_slides.append(
                {
                    "slide": slide_number(slide_name),
                    "title": title,
                    "text": text_runs,
                    "media_sha256": list(dict.fromkeys(media_targets)),
                }
            )

    pages = pdf_pages(pdf)
    alignment = align_pages(pages, ppt_slides)
    slides: list[dict[str, object]] = []
    for page, ppt_index in zip(pages, alignment, strict=True):
        ppt_slide = ppt_slides[ppt_index] if ppt_index is not None else None
        slides.append(
            {
                "slide": page["slide"],
                "pptx_slide": ppt_slide["slide"] if ppt_slide else None,
                "title": page["title"],
                "classification": classify(str(page["title"])),
                "coverage": "pending",
                "text": page["text"],
                "media_sha256": ppt_slide["media_sha256"] if ppt_slide else [],
                "notes": "",
            }
        )

    return {
        "deck_id": deck_id,
        "source_pdf": str(pdf),
        "source_pptx": str(pptx),
        "slide_count": len(slides),
        "review_status": "pending",
        "slides": slides,
        "media": sorted(media_cache.values(), key=lambda item: item["pptx_path"]),
        "figure_coverage": {
            "unique_substantive": None,
            "reused": 0,
            "duplicate": 0,
            "decorative": 0,
            "scientifically_redundant": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--pptx", required=True, type=Path)
    parser.add_argument("--deck-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = inventory(args.pptx, args.pdf, args.deck_id)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
