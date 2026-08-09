#!/usr/bin/env python3
"""Finalize slide coverage and published-media status for one lecture post."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ASSET_RE = re.compile(r"assets/img/blog/lectures/[^\s\"')>]+")


def parse_ranges(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--post", required=True, type=Path)
    parser.add_argument("--covered", required=True)
    parser.add_argument("--logistical", default="")
    parser.add_argument("--transition", default="")
    parser.add_argument("--recap", default="")
    parser.add_argument("--unique-substantive", required=True, type=int)
    parser.add_argument("--redundant", required=True, type=int)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assignments = {
        "covered": parse_ranges(args.covered),
        "omitted-logistical": parse_ranges(args.logistical),
        "omitted-transition": parse_ranges(args.transition),
        "omitted-repeated-recap": parse_ranges(args.recap),
    }
    assigned: set[int] = set()
    for status, slides in assignments.items():
        overlap = assigned & slides
        if overlap:
            raise SystemExit(f"slides assigned more than once: {sorted(overlap)}")
        assigned.update(slides)
        for slide in manifest["slides"]:
            if slide["slide"] in slides:
                slide["coverage"] = status

    expected = {slide["slide"] for slide in manifest["slides"]}
    if assigned != expected:
        raise SystemExit(f"unassigned slides: {sorted(expected - assigned)}")

    post_text = args.post.read_text(encoding="utf-8")
    referenced = set(ASSET_RE.findall(post_text))
    reused = 0
    for item in manifest["media"]:
        asset = item.get("asset_path")
        if asset and asset in referenced:
            item["reuse_status"] = "reused"
            reused += 1
        else:
            item["reuse_status"] = "not-published"
            item.pop("asset_path", None)

    if reused + args.redundant != args.unique_substantive:
        raise SystemExit(
            "unique figure accounting must equal reused + scientifically redundant: "
            f"{args.unique_substantive} != {reused} + {args.redundant}"
        )
    ratio = reused / args.unique_substantive if args.unique_substantive else 1.0
    manifest["figure_coverage"] = {
        "unique_substantive": args.unique_substantive,
        "reused": reused,
        "scientifically_redundant": args.redundant,
        "reuse_ratio": round(ratio, 4),
    }
    manifest["published_post"] = str(args.post)
    manifest["review_status"] = "complete" if ratio >= 0.8 else "incomplete"
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{manifest['deck_id']}: slides={len(expected)} reused={reused}/"
        f"{args.unique_substantive} status={manifest['review_status']}"
    )
    return 0 if manifest["review_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
