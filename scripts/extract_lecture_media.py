#!/usr/bin/env python3
"""Extract rights-cleared media from an inventoried lecture deck for web use."""

from __future__ import annotations

import argparse
import json
import posixpath
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


PASSTHROUGH = {".gif", ".svg", ".webp"}
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_slides(value: str) -> set[int]:
    slides: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            slides.update(range(start, end + 1))
        else:
            slides.add(int(part))
    return slides


def render_pdf_crop(
    *,
    archive: zipfile.ZipFile,
    manifest: dict,
    media_path: str,
    canonical_page: int,
    output_path: Path,
    temp_root: Path,
) -> bool:
    """Recover unsupported legacy media by cropping its rendered PDF slide."""

    slide_record = next(item for item in manifest["slides"] if item["slide"] == canonical_page)
    pptx_slide = slide_record["pptx_slide"]
    slide_path = f"ppt/slides/slide{pptx_slide}.xml"
    rels_path = f"ppt/slides/_rels/slide{pptx_slide}.xml.rels"
    slide_root = ET.fromstring(archive.read(slide_path))
    rels_root = ET.fromstring(archive.read(rels_path))
    rels = {item.attrib["Id"]: item.attrib["Target"] for item in rels_root}

    boxes: list[tuple[int, int, int, int]] = []
    for picture in slide_root.findall(".//p:pic", NS):
        blip = picture.find(".//a:blip", NS)
        if blip is None:
            continue
        rel_id = blip.attrib.get(f"{{{NS['r']}}}embed")
        target = rels.get(rel_id, "")
        resolved = posixpath.normpath(posixpath.join("ppt/slides", target))
        if resolved != media_path:
            continue
        offset = picture.find(".//a:xfrm/a:off", NS)
        extent = picture.find(".//a:xfrm/a:ext", NS)
        if offset is None or extent is None:
            continue
        boxes.append(
            (
                int(offset.attrib["x"]),
                int(offset.attrib["y"]),
                int(extent.attrib["cx"]),
                int(extent.attrib["cy"]),
            )
        )
    if not boxes:
        return False

    presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
    slide_size = presentation.find("p:sldSz", NS)
    if slide_size is None:
        return False
    slide_width = int(slide_size.attrib["cx"])
    slide_height = int(slide_size.attrib["cy"])
    x, y, width, height = max(boxes, key=lambda item: item[2] * item[3])

    page_root = temp_root / f"pdf-page-{canonical_page:03d}"
    page_png = page_root.with_suffix(".png")
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(canonical_page),
            "-l",
            str(canonical_page),
            "-r",
            "180",
            "-singlefile",
            "-png",
            manifest["source_pdf"],
            str(page_root),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dimensions = subprocess.run(
        ["magick", "identify", "-format", "%w %h", str(page_png)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    page_width, page_height = map(int, dimensions)
    crop_x = round(x / slide_width * page_width)
    crop_y = round(y / slide_height * page_height)
    crop_width = max(1, round(width / slide_width * page_width))
    crop_height = max(1, round(height / slide_height * page_height))
    subprocess.run(
        [
            "magick",
            str(page_png),
            "-crop",
            f"{crop_width}x{crop_height}+{crop_x}+{crop_y}",
            "+repage",
            "-strip",
            "-quality",
            "88",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slides", required=True, help="Comma-separated pages/ranges")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_pages = parse_slides(args.slides)
    selected_hashes: set[str] = set()
    first_page: dict[str, int] = {}
    for slide in manifest["slides"]:
        if slide["slide"] not in selected_pages:
            continue
        for digest in slide["media_sha256"]:
            selected_hashes.add(digest)
            first_page[digest] = min(first_page.get(digest, slide["slide"]), slide["slide"])

    media_by_hash = {item["sha256"]: item for item in manifest["media"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(manifest["source_pptx"]) as archive, tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for digest in sorted(selected_hashes, key=lambda item: (first_page[item], media_by_hash[item]["pptx_path"])):
            item = media_by_hash[digest]
            source_name = Path(item["pptx_path"])
            source_suffix = source_name.suffix.lower()
            output_suffix = source_suffix if source_suffix in PASSTHROUGH else ".webp"
            output_name = f"s{first_page[digest]:02d}-{source_name.stem}{output_suffix}"
            output_path = args.output_dir / output_name
            temporary_source = temp_root / source_name.name
            temporary_source.write_bytes(archive.read(item["pptx_path"]))
            if source_suffix in PASSTHROUGH:
                shutil.copyfile(temporary_source, output_path)
            else:
                try:
                    subprocess.run(
                        [
                            "magick",
                            str(temporary_source),
                            "-auto-orient",
                            "-strip",
                            "-resize",
                            "2400x2400>",
                            "-quality",
                            "88",
                            str(output_path),
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
                    output_path.unlink(missing_ok=True)
                    recovered = render_pdf_crop(
                        archive=archive,
                        manifest=manifest,
                        media_path=item["pptx_path"],
                        canonical_page=first_page[digest],
                        output_path=output_path,
                        temp_root=temp_root,
                    )
                    if not recovered:
                        print(
                            f"warning: unsupported deck media {item['pptx_path']} "
                            f"({source_suffix}); keeping it in the inventory but not publishing it"
                        )
                        continue
                    print(
                        f"recovered unsupported deck media {item['pptx_path']} "
                        f"from PDF slide {first_page[digest]}"
                    )
            item["asset_path"] = str(output_path.relative_to(args.repo_root))
            item["reuse_status"] = "candidate"
            extracted += 1

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"extracted {extracted} assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
