#!/usr/bin/env python3
"""Build review sheets for replacing PDF lecture crops with PPTX objects.

Each row compares the currently published PDF-derived asset with every raster
media object actually related to that PowerPoint slide.  The output is for
editorial review only; it never writes published assets.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from inventory_lecture_pptx_objects import (
    NS,
    object_record,
    relationship_map,
    resolve_pptx_slide,
)


ROOT = Path(__file__).resolve().parents[1]
CANVAS_WIDTH = 1800
ROW_HEIGHT = 520
LEFT_WIDTH = 780
PADDING = 18


def flatten_media(records: list[dict]) -> list[str]:
    paths: list[str] = []
    for record in records:
        paths.extend(record.get("media_paths", []))
        paths.extend(flatten_media(record.get("children", [])))
    return list(dict.fromkeys(paths))


def contain(image: Image.Image, width: int, height: int) -> Image.Image:
    converted = image.convert("RGBA")
    background = Image.new("RGBA", converted.size, "white")
    background.alpha_composite(converted)
    flattened = background.convert("RGB")
    flattened.thumbnail((width, height), Image.Resampling.LANCZOS)
    return flattened


def open_image(payload: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    except Exception:
        return None


def slide_records(archive: zipfile.ZipFile, pptx_slide: int) -> list[dict]:
    slide_path = f"ppt/slides/slide{pptx_slide}.xml"
    root = ET.fromstring(archive.read(slide_path))
    rels = relationship_map(archive, pptx_slide)
    shape_tree = root.find(".//p:spTree", NS)
    children = [] if shape_tree is None else [
        child
        for child in shape_tree
        if child.tag.rsplit("}", 1)[-1]
        in {"pic", "sp", "grpSp", "graphicFrame"}
    ]
    return [object_record(child, rels, index) for index, child in enumerate(children)]


def draw_row(
    canvas: Image.Image,
    row_index: int,
    logical_slide: int,
    current_path: Path,
    candidates: list[tuple[str, Image.Image]],
) -> None:
    top = row_index * ROW_HEIGHT
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.rectangle((0, top, CANVAS_WIDTH, top + ROW_HEIGHT), fill="#f7f8fa")
    draw.text((PADDING, top + 8), f"Slide {logical_slide}: current PDF asset", fill="black", font=font)
    current = contain(Image.open(current_path), LEFT_WIDTH - 2 * PADDING, ROW_HEIGHT - 62)
    canvas.paste(current, (PADDING, top + 42))

    right_x = LEFT_WIDTH
    right_width = CANVAS_WIDTH - right_x
    draw.text((right_x, top + 8), "Visible PPTX raster media", fill="black", font=font)
    count = max(1, len(candidates))
    cols = min(3, count)
    rows = math.ceil(count / cols)
    cell_w = (right_width - PADDING * (cols + 1)) // cols
    cell_h = (ROW_HEIGHT - 56 - PADDING * rows) // rows
    for index, (name, image) in enumerate(candidates):
        col = index % cols
        row = index // cols
        x = right_x + PADDING + col * (cell_w + PADDING)
        y = top + 42 + row * (cell_h + PADDING)
        thumb = contain(image, cell_w, cell_h - 24)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + cell_h - 20), name, fill="#333333", font=font)
    draw.line((0, top + ROW_HEIGHT - 1, CANVAS_WIDTH, top + ROW_HEIGHT - 1), fill="#c8ccd2", width=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    regions = [
        region
        for region in manifest.get("pdf_regions", [])
        if region.get("reuse_status") == "reused"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, Path, list[tuple[str, Image.Image]]]] = []
    with zipfile.ZipFile(manifest["source_pptx"]) as archive:
        for region in regions:
            logical_slide = int(region["slide"])
            pptx_slide = resolve_pptx_slide(manifest, logical_slide)
            records = slide_records(archive, pptx_slide)
            candidates: list[tuple[str, Image.Image]] = []
            for media_path in flatten_media(records):
                image = open_image(archive.read(media_path))
                if image is not None:
                    candidates.append((Path(media_path).name, image))
            rows.append(
                (
                    logical_slide,
                    ROOT / region["asset_path"],
                    candidates,
                )
            )

    for sheet_index in range(0, len(rows), args.rows_per_sheet):
        batch = rows[sheet_index : sheet_index + args.rows_per_sheet]
        canvas = Image.new(
            "RGB",
            (CANVAS_WIDTH, ROW_HEIGHT * len(batch)),
            "white",
        )
        for row_index, row in enumerate(batch):
            draw_row(canvas, row_index, *row)
        output = args.output_dir / f"sheet-{sheet_index // args.rows_per_sheet + 1:02d}.png"
        canvas.save(output)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
