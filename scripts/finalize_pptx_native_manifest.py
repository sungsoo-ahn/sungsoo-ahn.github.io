#!/usr/bin/env python3
"""Record PPTX-native lecture figures after a post has been migrated.

Direct-media asset names use ``sNN-imageMM.ext``.  The command verifies that
each source media item is actually related to the declared PowerPoint slide,
then replaces PDF-region publication records with explicit PPTX provenance.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')
DIRECT_MEDIA_RE = re.compile(r"/s(?P<slide>\d+)-image(?P<image>\d+)\.[^/]+$")
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
            if not match:
                raise SystemExit(
                    f"{asset_path}: cannot infer source; use sNN-imageMM.ext for "
                    "direct media and record PowerPoint shape exports manually"
                )
            slide = int(match.group("slide"))
            pptx_slide = logical_to_pptx.get(slide)
            if pptx_slide is None:
                raise SystemExit(f"{asset_path}: slide {slide} is absent from manifest")
            stem = f"image{int(match.group('image'))}"
            candidates = sorted(
                name
                for name in archive_names
                if name.startswith(f"ppt/media/{stem}.")
            )
            if len(candidates) != 1:
                raise SystemExit(
                    f"{asset_path}: expected one PPTX media source, found {candidates}"
                )
            source_media = candidates[0]
            if source_media not in slide_media(archive, pptx_slide):
                raise SystemExit(
                    f"{asset_path}: {source_media} is not related to PPTX slide "
                    f"{pptx_slide}"
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
