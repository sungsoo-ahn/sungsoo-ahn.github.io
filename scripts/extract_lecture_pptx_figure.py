#!/usr/bin/env python3
"""Extract one published lecture figure from its source PowerPoint deck.

Unmodified browser-safe media can be copied byte-for-byte from ``ppt/media``.
Cropped pictures, EMF artwork, and native PowerPoint groups are rendered by
PowerPoint's own ``save as picture`` command.  PDF rendering is deliberately
not supported.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path


POWERPOINT_TEMP = Path.home() / (
    "Library/Containers/com.microsoft.Powerpoint/Data/tmp/codex-lecture-export"
)


def resolve_pptx_slide(manifest: dict, logical_slide: int) -> int:
    match = next(
        (item for item in manifest["slides"] if item["slide"] == logical_slide),
        None,
    )
    if match is None:
        raise SystemExit(f"logical slide {logical_slide} is not present in manifest")
    return int(match.get("pptx_slide") or logical_slide)


def copy_media(pptx_path: Path, media_path: str, output: Path) -> None:
    if not media_path.startswith("ppt/media/"):
        raise SystemExit("--media must name an item under ppt/media/")
    with zipfile.ZipFile(pptx_path) as archive:
        if media_path not in archive.namelist():
            raise SystemExit(f"missing {media_path} in source PPTX")
        payload = archive.read(media_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def export_shape(
    pptx_path: Path, pptx_slide: int, shape_name: str, output: Path
) -> None:
    POWERPOINT_TEMP.mkdir(parents=True, exist_ok=True)
    temporary = POWERPOINT_TEMP / f"{uuid.uuid4().hex}.png"
    script = r'''
on run argv
  set deckPath to item 1 of argv
  set slideNumber to (item 2 of argv) as integer
  set shapeName to item 3 of argv
  set outputPath to item 4 of argv
  tell application "Microsoft PowerPoint"
    open POSIX file deckPath
    set deckDoc to active presentation
    set targetShape to shape shapeName of slide slideNumber of deckDoc
    save as picture targetShape picture type save as PNG file file name outputPath
    close deckDoc
  end tell
end run
'''
    try:
        subprocess.run(
            [
                "osascript",
                "-",
                str(pptx_path),
                str(pptx_slide),
                shape_name,
                str(temporary),
            ],
            input=script,
            text=True,
            check=True,
        )
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise SystemExit(
                "PowerPoint did not create the requested figure; check the slide "
                "and shape name"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slide", required=True, type=int, help="Logical lecture slide")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--media", help="Exact ppt/media path to copy")
    source.add_argument("--shape", help="PowerPoint shape name to export")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pptx_path = Path(manifest["source_pptx"])
    if not pptx_path.exists():
        raise SystemExit(f"source PPTX does not exist: {pptx_path}")
    pptx_slide = resolve_pptx_slide(manifest, args.slide)
    if args.media:
        copy_media(pptx_path, args.media, args.output)
        method = "pptx-media-copy"
    else:
        export_shape(pptx_path, pptx_slide, args.shape, args.output)
        method = "pptx-picture-export"
    print(
        json.dumps(
            {
                "logical_slide": args.slide,
                "pptx_slide": pptx_slide,
                "asset_path": str(args.output),
                "extraction_method": method,
                "source_media_path": args.media or "",
                "source_shape_name": args.shape or "",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
