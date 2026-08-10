#!/usr/bin/env python3
"""Record PPTX-native lecture figures after a post has been migrated.

Direct-media asset names use ``sNN-imageMM.ext``.  The command verifies that
each source media item is actually related to the declared PowerPoint slide,
then replaces PDF-region publication records with explicit PPTX provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')
DIRECT_MEDIA_RE = re.compile(r"/s(?P<slide>\d+)-image(?P<image>\d+)\.[^/]+$")
SLIDE_PREFIX_RE = re.compile(r"/s(?P<slide>\d+)-[^/]+$")
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def figures_in_post(post_path: Path) -> list[str]:
    text = post_path.read_text(encoding="utf-8")
    figures: list[str] = []
    for match in FIGURE_RE.finditer(text):
        attrs = {
            item.group("key"): item.group("value")
            for item in ATTR_RE.finditer(match.group("attrs"))
        }
        if attrs.get("path"):
            figures.append(attrs["path"])
    if len(figures) != len(set(figures)):
        raise SystemExit(f"{post_path}: duplicate figure paths are not supported")
    return figures


def slide_media(archive: ZipFile, pptx_slide: int) -> set[str]:
    rel_path = f"ppt/slides/_rels/slide{pptx_slide}.xml.rels"
    if rel_path not in archive.namelist():
        return set()
    root = ET.fromstring(archive.read(rel_path))
    resolved: set[str] = set()
    base = posixpath.dirname(f"ppt/slides/slide{pptx_slide}.xml")
    for rel in root.findall(f"{{{REL_NS}}}Relationship"):
        target = rel.attrib.get("Target", "")
        if rel.attrib.get("TargetMode") == "External":
            continue
        candidate = posixpath.normpath(posixpath.join(base, target))
        if candidate.startswith("ppt/media/"):
            resolved.add(candidate)
    return resolved


def parse_roles(values: list[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--role must use ASSET_PATH=ROLE")
        asset_path, role = value.split("=", 1)
        roles[asset_path] = role
    return roles


def slide_from_asset_path(asset_path: str) -> int:
    match = SLIDE_PREFIX_RE.search(asset_path)
    if not match:
        raise SystemExit(
            f"{asset_path}: cannot infer the logical slide from an sNN- filename"
        )
    return int(match.group("slide"))


def exact_related_media(
    archive: ZipFile, asset_path: str, related_media: set[str]
) -> str:
    asset_sha = hashlib.sha256(Path(asset_path).read_bytes()).hexdigest()
    candidates = sorted(
        media_path
        for media_path in related_media
        if hashlib.sha256(archive.read(media_path)).hexdigest() == asset_sha
    )
    if len(candidates) != 1:
        raise SystemExit(
            f"{asset_path}: expected one byte-identical related PPTX media source, "
            f"found {candidates}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--post", required=True, type=Path)
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="Override the default diagram role with ASSET_PATH=ROLE",
    )
    parser.add_argument(
        "--relink-media",
        action="store_true",
        help=(
            "Update the manifest's media inventory instead of adding duplicate "
            "pptx_figures records. This requires every published asset to be an "
            "exact copy of one related PowerPoint media object."
        ),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    roles = parse_roles(args.role)
    paths = figures_in_post(args.post)
    expected_prefix = f"assets/img/blog/lectures/{manifest['deck_id']}/"
    pptx_path = Path(manifest["source_pptx"])
    logical_to_pptx = {
        int(item["slide"]): int(item.get("pptx_slide") or item["slide"])
        for item in manifest["slides"]
    }

    records: list[dict] = []
    with ZipFile(pptx_path) as archive:
        archive_names = set(archive.namelist())
        for asset_path in paths:
            if not asset_path.startswith(expected_prefix):
                raise SystemExit(
                    f"{asset_path}: expected the deck prefix {expected_prefix}"
                )
            match = DIRECT_MEDIA_RE.search(asset_path)
            slide = slide_from_asset_path(asset_path)
            pptx_slide = logical_to_pptx.get(slide)
            if pptx_slide is None:
                raise SystemExit(f"{asset_path}: slide {slide} is absent from manifest")
            related_media = slide_media(archive, pptx_slide)
            source_media = exact_related_media(archive, asset_path, related_media)
            if match:
                stem = f"image{int(match.group('image'))}"
                named_candidates = sorted(
                    name
                    for name in archive_names
                    if name.startswith(f"ppt/media/{stem}.")
                )
                if named_candidates != [source_media]:
                    raise SystemExit(
                        f"{asset_path}: filename identifies {named_candidates}, but "
                        f"the byte-identical related source is {source_media}"
                    )
            records.append(
                {
                    "slide": slide,
                    "pptx_slide": pptx_slide,
                    "asset_path": asset_path,
                    "extraction_method": "pptx-media-copy",
                    "content_role": roles.get(asset_path, "diagram"),
                    "source_media_paths": [source_media],
                    "reuse_status": "reused",
                }
            )

    unknown_roles = sorted(set(roles) - set(paths))
    if unknown_roles:
        raise SystemExit(f"--role paths are absent from the post: {unknown_roles}")

    if args.relink_media:
        if manifest.get("pptx_figures"):
            raise SystemExit(
                "--relink-media cannot replace an existing non-empty pptx_figures list"
            )
        asset_by_media = {
            record["source_media_paths"][0]: record["asset_path"]
            for record in records
        }
        if len(asset_by_media) != len(records):
            raise SystemExit(
                "--relink-media requires a distinct PowerPoint media object for "
                "every published figure"
            )
        media_by_path = {item["pptx_path"]: item for item in manifest["media"]}
        missing_media = sorted(set(asset_by_media) - set(media_by_path))
        if missing_media:
            raise SystemExit(f"manifest media inventory is missing: {missing_media}")
        for media_path, item in media_by_path.items():
            if media_path in asset_by_media:
                item["asset_path"] = asset_by_media[media_path]
                item["reuse_status"] = "reused"
            elif item.get("reuse_status") == "reused":
                item["reuse_status"] = "not-published"
                item.pop("asset_path", None)
        manifest.pop("pptx_figures", None)
    else:
        manifest["pptx_figures"] = records
    manifest.pop("pdf_regions", None)
    manifest["asset_migration_status"] = "pptx-native-complete"
    manifest["figure_coverage"] = {
        "unique_substantive_figures": len(records),
        "reused_figures": len(records),
        "redundant_figures": 0,
        "reuse_ratio": 1.0,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"recorded {len(records)} PPTX-native figures in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
