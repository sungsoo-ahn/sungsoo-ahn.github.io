#!/usr/bin/env python3
"""Stage hidden kUPS tutorial pages as final blog posts.

This script prepares the final public publication migration without changing
the live site by default. It copies hidden `_pages/kups-md-post-XX-*.md` drafts
into an output directory using final `_posts/YYYY-MM-DD-*.md` filenames and
rewrites front matter fields that must differ between hidden drafts and public
posts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "_pages"
POSTS_DIR = ROOT / "_posts"
INDEX_PATH = PAGES_DIR / "kups-md-tutorials.md"
POSTS = tuple(f"{post:02d}" for post in range(1, 13))
SERIES = "kups-md-tutorials"


@dataclass(frozen=True)
class PreparedPost:
    post: str
    source: Path
    destination: Path
    public_blockers: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hidden kUPS pages as staged final _posts files.",
    )
    parser.add_argument(
        "--publication-date",
        required=True,
        help="Shared final publication date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where staged _posts files are written. If omitted, the "
            "script performs a dry run and prints the planned destinations."
        ),
    )
    return parser.parse_args()


def split_frontmatter(text: str, path: Path) -> tuple[list[str], str]:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML front matter")
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError(f"{path}: malformed YAML front matter")
    return parts[1].strip("\n").splitlines(), parts[2].lstrip("\n")


def scalar_value(line: str) -> str:
    _, value = line.split(":", 1)
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def get_frontmatter_value(lines: list[str], key: str) -> str | None:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return scalar_value(line)
    return None


def rewrite_frontmatter(lines: list[str], *, publication_date: str) -> list[str]:
    rewritten: list[str] = []
    skip_keys = {"permalink", "nav", "nav_order"}
    for line in lines:
        key = (
            line.split(":", 1)[0].strip()
            if ":" in line and not line.startswith(" ")
            else ""
        )
        if key in skip_keys:
            continue
        if key == "date":
            rewritten.append(f"date: {publication_date}")
        elif key == "last_updated":
            rewritten.append(f"last_updated: {publication_date}")
        else:
            rewritten.append(line)
    return rewritten


def rewrite_author_note(body: str, *, post: str) -> str:
    final_note = (
        '<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">\n'
        "<em>Note: This post is part of the kUPS Molecular Dynamics Tutorials "
        "series for ML researchers who already know MLIPs and the equations of "
        "motion. Corrections and replication issues should be tracked in "
        '<a href="https://github.com/sungsoo-ahn/kups-md-tutorials">'
        "sungsoo-ahn/kups-md-tutorials</a>; the corresponding source, "
        "configuration, notebook, results, figures, and self-review artifacts "
        f"are linked below for Post {post}.</em>\n"
        "</p>"
    )
    return re.sub(
        r'<p style="color: #666; font-size: 0\.9em; margin-bottom: 1\.5em;">\n'
        r"<em>Note:.*?</em>\n</p>",
        final_note,
        body,
        count=1,
        flags=re.DOTALL,
    )


def public_blockers(text: str) -> tuple[str, ...]:
    blockers = []
    checks = {
        "contains non-final article language": "This page is not the final article",
        "contains hidden navigation language": "intentionally hidden from site navigation",
        "contains hidden draft language": "hidden draft",
    }
    for label, phrase in checks.items():
        if phrase in text:
            blockers.append(label)
    return tuple(blockers)


def source_page(post: str) -> Path:
    matches = sorted(PAGES_DIR.glob(f"kups-md-post-{post}-*.md"))
    if len(matches) != 1:
        raise ValueError(
            f"{PAGES_DIR.relative_to(ROOT)}: expected one hidden page for post "
            f"{post}, found {len(matches)}"
        )
    return matches[0]


def destination_name(source: Path, *, publication_date: str) -> str:
    slug = source.stem
    return f"{publication_date}-{slug}.md"


def validate_publication_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("--publication-date must use YYYY-MM-DD") from exc
    return parsed.isoformat()


def prepare_posts(
    *,
    publication_date: str,
    output_dir: Path | None,
) -> list[PreparedPost]:
    prepared: list[PreparedPost] = []
    target_dir = output_dir if output_dir is not None else POSTS_DIR
    for post in POSTS:
        source = source_page(post)
        text = source.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text, source)
        if get_frontmatter_value(frontmatter, "series") != SERIES:
            raise ValueError(f"{source.relative_to(ROOT)}: unexpected series")
        if get_frontmatter_value(frontmatter, "layout") != "post":
            raise ValueError(f"{source.relative_to(ROOT)}: layout must be post")
        if get_frontmatter_value(frontmatter, "nav") != "false":
            raise ValueError(
                f"{source.relative_to(ROOT)}: hidden draft must set nav: false"
            )

        destination = target_dir / destination_name(
            source,
            publication_date=publication_date,
        )
        rewritten = (
            "---\n"
            + "\n".join(
                rewrite_frontmatter(frontmatter, publication_date=publication_date)
            )
            + "\n---\n\n"
            + rewrite_author_note(body, post=post)
        )
        if output_dir is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rewritten, encoding="utf-8")
        prepared.append(
            PreparedPost(
                post=post,
                source=source,
                destination=destination,
                public_blockers=public_blockers(rewritten),
            )
        )
    return prepared


def index_uses_hidden_pages() -> bool:
    return 'assign postlist = site.pages | where: "series", "kups-md-tutorials"' in (
        INDEX_PATH.read_text(encoding="utf-8")
    )


def main() -> int:
    args = parse_args()
    try:
        publication_date = validate_publication_date(args.publication_date)
        prepared = prepare_posts(
            publication_date=publication_date,
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    mode = "staged" if args.output_dir is not None else "dry run"
    print(f"kUPS publication preparation {mode}: {len(prepared)} posts")
    for item in prepared:
        blocker_note = (
            f" [public blockers: {', '.join(item.public_blockers)}]"
            if item.public_blockers
            else ""
        )
        print(
            "- "
            f"post-{item.post}: {item.source.relative_to(ROOT)} -> "
            f"{item.destination.relative_to(ROOT) if item.destination.is_relative_to(ROOT) else item.destination}"
            f"{blocker_note}"
        )
    blocked = [item for item in prepared if item.public_blockers]
    if blocked:
        print(
            "public release still blocked: staged posts contain non-final or "
            "hidden-draft body language that must be revised after production "
            "diagnostics."
        )
    if index_uses_hidden_pages():
        print(
            "index update needed: change the kUPS index postlist assignment "
            "from site.pages to site.posts after final posts are reviewed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
