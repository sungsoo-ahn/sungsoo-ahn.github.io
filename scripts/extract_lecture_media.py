#!/usr/bin/env python3
"""Extract rights-cleared media from an inventoried lecture deck for web use."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


PASSTHROUGH = {".gif", ".svg", ".webp"}


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
                )
            item["asset_path"] = str(output_path.relative_to(args.repo_root))
            item["reuse_status"] = "candidate"
            extracted += 1

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"extracted {extracted} assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
