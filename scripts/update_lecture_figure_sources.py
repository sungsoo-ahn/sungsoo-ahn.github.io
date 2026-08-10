#!/usr/bin/env python3
"""Build and validate the source registry for reused lecture figures.

The registry separates three layers:

* ``sources`` contains deduplicated paper, project, dataset, and web metadata;
* ``slides`` contains evidence extracted from the source decks; and
* ``figures`` maps every published asset to its deck, slide, and audited source.

Running ``--sync`` updates deck-derived fields and applies manual audit decisions
from the curated assignment file. Exact byte-for-byte duplicates inherit one
audited decision while retaining asset-level traceability. ``--require-complete``
is the final gate: it rejects pending figures and incomplete source metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"
MANIFEST_DIR = ROOT / ".agents" / "lecture-adaptation"
REGISTRY_PATH = ROOT / "_data" / "lecture_figure_sources.json"
ASSIGNMENTS_PATH = MANIFEST_DIR / "figure-source-assignments.json"

DECK_PREFIXES = ("ml4mol-2025-", "gdl-2025-")
STATUS_VALUES = {
    "pending",
    "verified",
    "lecture-original",
    "composite",
    "ambiguous",
    "unresolved",
}
CONFIDENCE_VALUES = {"high", "medium", "low", "unresolved"}
LICENSE_STATUS_VALUES = {"open", "restricted", "unknown", "not-applicable"}
CURATED_FIGURE_FIELDS = {
    "status",
    "source_ids",
    "confidence",
    "evidence",
    "reader_note",
    "notes",
}

FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')
URL_RE = re.compile(r"https?://[^\s<>\]\)\}\"']+")
CITATION_HINT_RE = re.compile(
    r"(?:\bet\s+al\.?\b|\barxiv\b|\bbiorxiv\b|\bmedrxiv\b|\bdoi\b|"
    r"\b(?:19|20)\d{2}\b)",
    re.IGNORECASE,
)


def normalize_url(value: str) -> str:
    """Remove punctuation artifacts without rewriting the actual target."""

    value = value.strip().rstrip(".,;:!?")
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def parse_figure_includes(post_path: Path) -> dict[str, dict[str, str]]:
    text = post_path.read_text(encoding="utf-8")
    figures: dict[str, dict[str, str]] = {}
    for match in FIGURE_RE.finditer(text):
        attrs = {
            item.group("key"): item.group("value")
            for item in ATTR_RE.finditer(match.group("attrs"))
        }
        path = attrs.get("path")
        if not path:
            continue
        # A deck can deliberately reuse one visual on multiple slides, and the
        # article may mirror that repetition with context-specific captions.
        # The source registry is asset-based, so retain the first include here.
        figures.setdefault(path, attrs)
    return figures


def external_slide_urls(pptx_path: Path, pptx_slide: int) -> list[str]:
    rel_path = f"ppt/slides/_rels/slide{pptx_slide}.xml.rels"
    if not pptx_path.exists():
        return []
    try:
        with ZipFile(pptx_path) as archive:
            if rel_path not in archive.namelist():
                return []
            root = ET.fromstring(archive.read(rel_path))
    except (BadZipFile, OSError, ET.ParseError):
        return []

    urls: set[str] = set()
    for relation in root:
        target = relation.attrib.get("Target", "")
        if relation.attrib.get("TargetMode") != "External":
            continue
        if target.startswith(("http://", "https://")):
            urls.add(normalize_url(target))
    return sorted(urls)


def citation_candidates(text_lines: list[str]) -> list[str]:
    candidates: list[str] = []
    for line in text_lines:
        normalized = " ".join(line.split())
        if not normalized or normalized.isdigit():
            continue
        if CITATION_HINT_RE.search(normalized):
            candidates.append(normalized)
    return list(dict.fromkeys(candidates))


def deck_slide_evidence(manifest: dict) -> dict[int, dict]:
    pptx_path = Path(manifest.get("source_pptx", ""))
    evidence: dict[int, dict] = {}
    for slide in manifest.get("slides", []):
        slide_number = slide["slide"]
        text_lines = slide.get("text", [])
        urls = {normalize_url(url) for url in URL_RE.findall(" ".join(text_lines))}
        pptx_slide = slide.get("pptx_slide")
        if pptx_slide:
            urls.update(external_slide_urls(pptx_path, pptx_slide))
        evidence[slide_number] = {
            "deck_id": manifest["deck_id"],
            "slide": slide_number,
            "pptx_slide": pptx_slide,
            "title": slide.get("title", ""),
            "url_candidates": sorted(urls),
            "citation_candidates": citation_candidates(text_lines),
        }
    return evidence


def reused_assets(manifest: dict) -> dict[str, dict]:
    """Return published assets with logical lecture slide numbers."""

    pptx_to_logical = {
        slide.get("pptx_slide"): slide["slide"]
        for slide in manifest.get("slides", [])
        if slide.get("pptx_slide")
    }
    assets: dict[str, dict] = {}
    source_media_by_asset: dict[str, set[str]] = defaultdict(set)
    logical_slides_by_media: dict[str, list[int]] = {}
    for media in manifest.get("media", []):
        logical_slides = sorted(
            {
                pptx_to_logical.get(number, number)
                for number in media.get("slides", [])
            }
        )
        logical_slides_by_media[media["pptx_path"]] = logical_slides
        if media.get("reuse_status") != "reused":
            continue
        path = media["asset_path"]
        source_media_by_asset[path].add(media["pptx_path"])
        assets[path] = {
            "slides": logical_slides,
            "extraction_method": "pptx-media-extraction",
        }
    for figure in manifest.get("pptx_figures", []):
        if figure.get("reuse_status") != "reused":
            continue
        path = figure["asset_path"]
        if path in assets:
            figure_sources = set(figure.get("source_media_paths", []))
            if not figure_sources.intersection(source_media_by_asset[path]):
                raise ValueError(f"conflicting published figure record for {path}")
        figure_slides = {figure["slide"]}
        if figure["extraction_method"] == "pptx-media-copy":
            for source_path in figure.get("source_media_paths", []):
                figure_slides.update(logical_slides_by_media.get(source_path, []))
        assets[path] = {
            "slides": sorted(figure_slides),
            "extraction_method": figure["extraction_method"],
        }
    for region in manifest.get("pdf_regions", []):
        if region.get("reuse_status") != "reused":
            continue
        path = region["asset_path"]
        assets[path] = {
            "slides": [region["slide"]],
            "extraction_method": "pdf-slide-crop",
        }
    return assets


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "schema_version": 1,
            "reuse_basis": (
                "Figures are republished from author-cleared lecture decks. "
                "Original-source licenses are recorded separately and do not "
                "replace that deck-level reuse authority."
            ),
            "sources": {},
            "slides": {},
            "figures": {},
        }
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def load_assignments() -> dict:
    """Load the hand-curated audit layer kept outside generated site data."""

    if not ASSIGNMENTS_PATH.exists():
        return {"schema_version": 1, "sources": {}, "figures": {}}
    return json.loads(ASSIGNMENTS_PATH.read_text(encoding="utf-8"))


def sync_registry(registry: dict, assignments: dict) -> dict:
    generated_slides: dict[str, dict] = {}
    generated_figures: dict[str, dict] = {}
    curated_figures = assignments.get("figures", {})

    for manifest_path in sorted(MANIFEST_DIR.glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        deck_id = manifest.get("deck_id", "")
        if not deck_id.startswith(DECK_PREFIXES):
            continue
        post_path = ROOT / manifest["published_post"]
        includes = parse_figure_includes(post_path)
        slide_evidence = deck_slide_evidence(manifest)
        assets = reused_assets(manifest)
        if set(includes) != set(assets):
            missing = sorted(set(assets) - set(includes))
            extra = sorted(set(includes) - set(assets))
            raise ValueError(
                f"{deck_id}: manifest/post figure mismatch; missing={missing}, extra={extra}"
            )

        for number, evidence in slide_evidence.items():
            generated_slides[f"{deck_id}:{number}"] = evidence

        for asset_path, asset in assets.items():
            slide_keys = [f"{deck_id}:{number}" for number in asset["slides"]]
            candidate_urls = sorted(
                {
                    url
                    for key in slide_keys
                    for url in generated_slides[key]["url_candidates"]
                }
            )
            candidate_citations = list(
                dict.fromkeys(
                    candidate
                    for key in slide_keys
                    for candidate in generated_slides[key]["citation_candidates"]
                )
            )
            generated_figures[asset_path] = {
                "deck_id": deck_id,
                "slides": asset["slides"],
                "slide_keys": slide_keys,
                "post": str(post_path.relative_to(ROOT)),
                "caption": includes[asset_path].get("caption", ""),
                "extraction_method": asset["extraction_method"],
                "reuse_basis": "rights-cleared-lecture-deck",
                "content_sha256": hashlib.sha256(
                    (ROOT / asset_path).read_bytes()
                ).hexdigest(),
                "candidate_urls": candidate_urls,
                "candidate_citations": candidate_citations,
                "status": "pending",
                "source_ids": [],
                "confidence": "unresolved",
                "evidence": [],
                "reader_note": "",
                "notes": "",
            }
            if asset_path in curated_figures:
                generated_figures[asset_path].update(curated_figures[asset_path])

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for asset_path, figure in generated_figures.items():
        exact_groups[figure["content_sha256"]].append(asset_path)
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        members.sort()
        representative = members[0]
        audited = [
            asset_path
            for asset_path in members
            if asset_path in curated_figures
            and generated_figures[asset_path]["status"] != "pending"
        ]
        inherited_from = audited[0] if audited else ""
        for asset_path in members:
            if asset_path != representative:
                generated_figures[asset_path]["exact_duplicate_of"] = representative
            if not inherited_from or asset_path in curated_figures:
                continue
            source = generated_figures[inherited_from]
            target = generated_figures[asset_path]
            for field in (
                "status",
                "source_ids",
                "confidence",
                "reader_note",
                "notes",
            ):
                target[field] = source[field]
            target["evidence"] = [
                f"Byte-for-byte duplicate of {inherited_from}; source audit inherited."
            ] + list(source["evidence"])
            target["source_audit_inherited_from"] = inherited_from

    registry["schema_version"] = 1
    registry["slides"] = dict(sorted(generated_slides.items()))
    registry["figures"] = dict(sorted(generated_figures.items()))
    registry["sources"] = dict(sorted(assignments.get("sources", {}).items()))
    return registry


def validate_registry(
    registry: dict, assignments: dict, require_complete: bool
) -> list[str]:
    errors: list[str] = []
    sources = registry.get("sources", {})
    figures = registry.get("figures", {})
    curated_figures = assignments.get("figures", {})

    if assignments.get("schema_version") != 1:
        errors.append("figure-source assignments must use schema_version 1")
    if set(assignments) - {"schema_version", "sources", "figures"}:
        unexpected = sorted(set(assignments) - {"schema_version", "sources", "figures"})
        errors.append(f"figure-source assignments: unexpected fields {unexpected}")
    unknown_assignments = sorted(set(curated_figures) - set(figures))
    for asset_path in unknown_assignments:
        errors.append(f"curated figure {asset_path}: not present in lecture manifests")

    source_urls: dict[str, str] = {}
    for source_id, source in sources.items():
        for field in ("label", "url", "kind", "license_status", "license", "verified_on"):
            if not source.get(field):
                errors.append(f"source {source_id}: missing {field}")
        if source.get("license_status") not in LICENSE_STATUS_VALUES:
            errors.append(
                f"source {source_id}: invalid license_status {source.get('license_status')}"
            )
        if source.get("url") and not source["url"].startswith(("http://", "https://")):
            errors.append(f"source {source_id}: invalid URL {source['url']}")
        if source.get("url"):
            normalized_url = normalize_url(source["url"])
            if normalized_url in source_urls:
                errors.append(
                    f"source {source_id}: URL duplicates source "
                    f"{source_urls[normalized_url]}"
                )
            source_urls[normalized_url] = source_id

    for asset_path, curated in curated_figures.items():
        unexpected = sorted(set(curated) - CURATED_FIGURE_FIELDS)
        if unexpected:
            errors.append(f"curated figure {asset_path}: unexpected fields {unexpected}")

    for asset_path, figure in figures.items():
        status = figure.get("status")
        confidence = figure.get("confidence")
        source_ids = figure.get("source_ids", [])
        if not (ROOT / asset_path).exists():
            errors.append(f"figure {asset_path}: asset does not exist")
        if status not in STATUS_VALUES:
            errors.append(f"figure {asset_path}: invalid status {status}")
        if confidence not in CONFIDENCE_VALUES:
            errors.append(f"figure {asset_path}: invalid confidence {confidence}")
        if not re.fullmatch(r"[0-9a-f]{64}", figure.get("content_sha256", "")):
            errors.append(f"figure {asset_path}: invalid or missing content_sha256")
        for source_id in source_ids:
            if source_id not in sources:
                errors.append(f"figure {asset_path}: unknown source {source_id}")
        if status in {"verified", "composite"} and not source_ids:
            errors.append(f"figure {asset_path}: {status} record has no source")
        if status in {"verified", "composite", "ambiguous"} and confidence == "unresolved":
            errors.append(f"figure {asset_path}: {status} record has unresolved confidence")
        if status in {"verified", "composite"} and not figure.get("evidence"):
            errors.append(f"figure {asset_path}: {status} record needs audit evidence")
        if status == "ambiguous" and (not source_ids or not figure.get("reader_note")):
            errors.append(
                f"figure {asset_path}: ambiguous record needs a candidate source "
                "and reader note"
            )
        if status == "unresolved" and (
            source_ids or confidence != "unresolved" or not figure.get("reader_note")
        ):
            errors.append(
                f"figure {asset_path}: unresolved record must have no source, "
                "unresolved confidence, and a reader note"
            )
        if status == "lecture-original" and source_ids:
            errors.append(f"figure {asset_path}: lecture-original record has a source")
        if status in {"lecture-original", "ambiguous", "unresolved"} and not figure.get("notes"):
            errors.append(f"figure {asset_path}: {status} record needs an audit note")
        if require_complete and status == "pending":
            errors.append(f"figure {asset_path}: provenance audit is pending")

    figures_by_hash: dict[str, list[str]] = defaultdict(list)
    for asset_path, figure in figures.items():
        figures_by_hash[figure.get("content_sha256", "")].append(asset_path)
    for members in figures_by_hash.values():
        curated_members = [path for path in members if path in curated_figures]
        decisions = {
            (
                figures[path].get("status"),
                tuple(figures[path].get("source_ids", [])),
                figures[path].get("confidence"),
            )
            for path in curated_members
            if figures[path].get("status") != "pending"
        }
        if len(decisions) > 1:
            errors.append(
                "exact duplicate figures have conflicting curated audits: "
                + ", ".join(sorted(curated_members))
            )

    return errors


def report(registry: dict) -> None:
    counts: dict[str, int] = {}
    hash_counts: dict[str, int] = defaultdict(int)
    inherited = 0
    for figure in registry.get("figures", {}).values():
        status = figure.get("status", "missing")
        counts[status] = counts.get(status, 0) + 1
        hash_counts[figure.get("content_sha256", "")] += 1
        inherited += int(bool(figure.get("source_audit_inherited_from")))
    print(f"sources: {len(registry.get('sources', {}))}")
    print(f"figures: {len(registry.get('figures', {}))}")
    print(f"exact duplicate groups: {sum(count > 1 for count in hash_counts.values())}")
    print(f"source audits inherited by exact duplicate: {inherited}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync", action="store_true", help="synchronize deck-derived records")
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail when the generated registry differs from manifests or assignments",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if any figure remains pending",
    )
    parser.add_argument("--report", action="store_true", help="print registry counts")
    args = parser.parse_args()

    registry = load_registry()
    assignments = load_assignments()
    drift_error = ""
    if args.sync:
        registry = sync_registry(registry, assignments)
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif args.check:
        expected = sync_registry(
            json.loads(json.dumps(registry)),
            assignments,
        )
        if expected != registry:
            drift_error = (
                "lecture figure source registry is out of date; run "
                "python3 scripts/update_lecture_figure_sources.py --sync"
            )
        registry = expected

    errors = validate_registry(registry, assignments, args.require_complete)
    if drift_error:
        errors.append(drift_error)
    if args.report or errors:
        report(registry)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
