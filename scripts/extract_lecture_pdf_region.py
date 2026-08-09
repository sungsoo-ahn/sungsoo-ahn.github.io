#!/usr/bin/env python3
"""Extract a source-faithful region from a canonical lecture PDF page."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--page", required=True, type=int)
    parser.add_argument(
        "--crop",
        required=True,
        help="Normalized x,y,width,height fractions in the rendered slide",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fractions = [float(item) for item in args.crop.split(",")]
    if len(fractions) != 4 or any(item < 0 for item in fractions):
        raise SystemExit("--crop must contain four nonnegative fractions")
    x, y, width, height = fractions
    if x + width > 1 or y + height > 1:
        raise SystemExit("crop extends beyond the slide")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        page_root = Path(temp_dir) / "page"
        page_png = page_root.with_suffix(".png")
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                str(args.page),
                "-l",
                str(args.page),
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
        geometry = (
            f"{round(width * page_width)}x{round(height * page_height)}"
            f"+{round(x * page_width)}+{round(y * page_height)}"
        )
        subprocess.run(
            [
                "magick",
                str(page_png),
                "-crop",
                geometry,
                "+repage",
                "-strip",
                "-resize",
                "2400x2400>",
                "-quality",
                "88",
                str(args.output),
            ],
            check=True,
        )
    record = {
        "slide": args.page,
        "crop": fractions,
        "asset_path": str(args.output),
        "reuse_status": "candidate",
    }
    regions = manifest.setdefault("pdf_regions", [])
    regions[:] = [item for item in regions if item.get("asset_path") != str(args.output)]
    regions.append(record)
    regions.sort(key=lambda item: (item["slide"], item["asset_path"]))
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"extracted slide {args.page} region to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
