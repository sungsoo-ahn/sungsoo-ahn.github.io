#!/usr/bin/env python3
"""Inspect PowerPoint-native objects used by a lecture slide.

The published lecture posts must not use rendered PDF pages as figures.  This
utility exposes the objects that actually live in the source PPTX so an editor
can decide which content belongs in prose/MathJax and which picture or grouped
diagram should be exported as a figure.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def relationship_map(archive: zipfile.ZipFile, pptx_slide: int) -> dict[str, str]:
    rel_path = f"ppt/slides/_rels/slide{pptx_slide}.xml.rels"
    if rel_path not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read(rel_path))
    mapping: dict[str, str] = {}
    for relation in root:
        rel_id = relation.attrib.get("Id")
        target = relation.attrib.get("Target")
        if not rel_id or not target:
            continue
        mapping[rel_id] = posixpath.normpath(
            posixpath.join("ppt/slides", target)
        )
    return mapping


def non_visual_properties(element: ET.Element) -> ET.Element | None:
    for path in (
        "./p:nvPicPr/p:cNvPr",
        "./p:nvSpPr/p:cNvPr",
        "./p:nvGrpSpPr/p:cNvPr",
        "./p:nvGraphicFramePr/p:cNvPr",
    ):
        found = element.find(path, NS)
        if found is not None:
            return found
    return None


def transform(element: ET.Element) -> dict[str, int | float | bool]:
    xfrm = element.find("./p:spPr/a:xfrm", NS)
    if xfrm is None:
        xfrm = element.find("./p:grpSpPr/a:xfrm", NS)
    if xfrm is None:
        return {}
    result: dict[str, int | float | bool] = {}
    offset = xfrm.find("a:off", NS)
    extent = xfrm.find("a:ext", NS)
    if offset is not None:
        result.update(x=int(offset.attrib["x"]), y=int(offset.attrib["y"]))
    if extent is not None:
        result.update(
            width=int(extent.attrib["cx"]), height=int(extent.attrib["cy"])
        )
    if "rot" in xfrm.attrib:
        result["rotation_degrees"] = int(xfrm.attrib["rot"]) / 60000
    if xfrm.attrib.get("flipH") in {"1", "true"}:
        result["flip_horizontal"] = True
    if xfrm.attrib.get("flipV") in {"1", "true"}:
        result["flip_vertical"] = True
    return result


def source_crop(element: ET.Element) -> dict[str, int] | None:
    crop = element.find("./p:blipFill/a:srcRect", NS)
    if crop is None:
        return None
    return {
        edge: int(crop.attrib.get(edge, "0"))
        for edge in ("l", "t", "r", "b")
    }


def object_record(
    element: ET.Element, rels: dict[str, str], z_order: int
) -> dict[str, object]:
    properties = non_visual_properties(element)
    tag = element.tag.rsplit("}", 1)[-1]
    texts = [
        node.text.strip()
        for node in element.findall(".//a:t", NS)
        if node.text and node.text.strip()
    ]
    media_paths: list[str] = []
    for blip in element.findall(".//a:blip", NS):
        rel_id = blip.attrib.get(f"{{{NS['r']}}}embed")
        if rel_id and rel_id in rels:
            target = rels[rel_id]
            if target.startswith("ppt/media/"):
                media_paths.append(target)
    record: dict[str, object] = {
        "z_order": z_order,
        "kind": {
            "pic": "picture",
            "sp": "shape",
            "grpSp": "group",
            "graphicFrame": "graphic-frame",
        }.get(tag, tag),
        "shape_id": int(properties.attrib["id"]) if properties is not None else None,
        "shape_name": properties.attrib.get("name", "") if properties is not None else "",
        "description": properties.attrib.get("descr", "") if properties is not None else "",
        "text": texts,
        "contains_math": bool(element.findall(".//m:oMath", NS)),
        "media_paths": list(dict.fromkeys(media_paths)),
        "transform": transform(element),
    }
    crop = source_crop(element)
    if crop is not None:
        record["source_crop"] = crop
    if tag == "grpSp":
        children = [
            child
            for child in element
            if child.tag.rsplit("}", 1)[-1] in {"pic", "sp", "grpSp", "graphicFrame"}
        ]
        record["children"] = [
            object_record(child, rels, index) for index, child in enumerate(children)
        ]
    return record


def resolve_pptx_slide(manifest: dict, logical_slide: int) -> int:
    match = next(
        (item for item in manifest["slides"] if item["slide"] == logical_slide),
        None,
    )
    if match is None:
        raise SystemExit(f"logical slide {logical_slide} is not present in manifest")
    return int(match.get("pptx_slide") or logical_slide)


def inventory_slide(pptx_path: Path, pptx_slide: int) -> dict[str, object]:
    """Return the native object inventory for one physical PPTX slide."""
    slide_path = f"ppt/slides/slide{pptx_slide}.xml"
    with zipfile.ZipFile(pptx_path) as archive:
        if slide_path not in archive.namelist():
            raise SystemExit(f"missing {slide_path} in source PPTX")
        root = ET.fromstring(archive.read(slide_path))
        rels = relationship_map(archive, pptx_slide)
        shape_tree = root.find(".//p:spTree", NS)
        children = [] if shape_tree is None else [
            child
            for child in shape_tree
            if child.tag.rsplit("}", 1)[-1]
            in {"pic", "sp", "grpSp", "graphicFrame"}
        ]
        return {
            "pptx_slide": pptx_slide,
            "objects": [
                object_record(child, rels, index)
                for index, child in enumerate(children)
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slide", required=True, type=int, help="Logical lecture slide")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pptx_slide = resolve_pptx_slide(manifest, args.slide)
    result = inventory_slide(Path(manifest["source_pptx"]), pptx_slide)
    result.update(deck_id=manifest["deck_id"], logical_slide=args.slide)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
