#!/usr/bin/env python3
"""Validate hidden kUPS MD tutorial pages and exported assets."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "_pages"
BLOG_IMG_DIR = ROOT / "assets" / "img" / "blog"
EXPORT_DIR = ROOT / "assets" / "json" / "kups-md-tutorials"
SERIES = "kups-md-tutorials"
POSTS = tuple(f"{post:02d}" for post in range(1, 13))

FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')


@dataclass(frozen=True)
class Finding:
    path: Path
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}: {self.message}"


def parse_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def parse_frontmatter(text: str) -> tuple[dict[str, str], str] | None:
    if not text.startswith("---\n"):
        return None
    try:
        _, frontmatter, body = text.split("---", 2)
    except ValueError:
        return None

    data: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if not line.strip() or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data, body


def page_for_post(post: str) -> Path | None:
    candidates = sorted(PAGES_DIR.glob(f"kups-md-post-{post}-*.md"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_page(post: str, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    parsed = parse_frontmatter(text)
    if parsed is None:
        return [Finding(path, "missing YAML frontmatter")]

    frontmatter, body = parsed
    expected = {
        "layout": "post",
        "post_type": "tutorial",
        "series": SERIES,
        "series_order": str(int(post)),
        "related_posts": "false",
        "nav": "false",
    }
    for key, value in expected.items():
        if frontmatter.get(key) != value:
            findings.append(Finding(path, f"{key} must be {value!r}"))

    permalink = frontmatter.get("permalink", "")
    if not permalink.startswith(f"/{SERIES}/post-{post}-") or not permalink.endswith("/"):
        findings.append(Finding(path, "permalink must use hidden kUPS post URL"))

    if "toc" not in frontmatter:
        findings.append(Finding(path, "missing toc frontmatter"))
    if "Note:" not in body or "intentionally hidden from site navigation" not in body:
        findings.append(Finding(path, "missing hidden-draft author note"))

    required_links = (
        f"configs/post-{post}/smoke.json",
        f"configs/post-{post}/full.json",
        f"notebooks/post-{post}",
        f"results/post-{post}/smoke/",
        f"results/post-{post}/full/",
        f"reviews/post-{post}.md",
    )
    for fragment in required_links:
        if fragment not in body:
            findings.append(Finding(path, f"missing source link fragment: {fragment}"))

    figures = list(FIGURE_RE.finditer(text))
    if not figures:
        findings.append(Finding(path, "missing figure include"))
    for match in figures:
        attrs = {
            attr.group("key"): attr.group("value")
            for attr in ATTR_RE.finditer(match.group("attrs"))
        }
        figure_path = attrs.get("path")
        if not figure_path:
            findings.append(Finding(path, "figure include missing path"))
            continue
        if f"kups_md_post{post}_" not in figure_path:
            findings.append(Finding(path, f"figure path does not match post {post}: {figure_path}"))
        if not (ROOT / figure_path).exists():
            findings.append(Finding(path, f"figure asset is missing: {figure_path}"))
        png_path = (ROOT / figure_path).with_suffix(".png")
        if figure_path.endswith(".svg") and not png_path.exists():
            findings.append(Finding(path, f"PNG companion is missing: {png_path.relative_to(ROOT)}"))
        caption = attrs.get("caption", "")
        if "$" in caption:
            findings.append(Finding(path, "figure caption uses dollar math delimiters"))

    if "This page is not the final article" not in body:
        findings.append(Finding(path, "missing explicit non-final status"))
    if "rendered desktop and mobile page snapshots" not in body:
        findings.append(Finding(path, "missing rendered snapshot blocker"))
    return findings


def validate_exported_assets() -> list[Finding]:
    findings: list[Finding] = []
    manifest_path = EXPORT_DIR / "manifest.json"
    if not manifest_path.exists():
        return [Finding(EXPORT_DIR, "missing kUPS export manifest")]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile") != "full":
        findings.append(Finding(manifest_path, "export manifest profile must be full"))
    if not manifest.get("source_git_revision"):
        findings.append(Finding(manifest_path, "export manifest missing source git revision"))

    for post in POSTS:
        post_dir = EXPORT_DIR / f"post-{post}" / "full"
        if not post_dir.exists():
            findings.append(Finding(post_dir, "missing exported full-profile data directory"))
            continue
        if not (post_dir / "manifest.json").exists():
            findings.append(Finding(post_dir, "missing exported provenance manifest"))
        if not any(path.name.endswith("_summary.json") for path in post_dir.iterdir()):
            findings.append(Finding(post_dir, "missing exported summary JSON"))
        if not any(BLOG_IMG_DIR.glob(f"kups_md_post{post}_*.svg")):
            findings.append(Finding(BLOG_IMG_DIR, f"missing SVG figure for post {post}"))
        if not any(BLOG_IMG_DIR.glob(f"kups_md_post{post}_*.png")):
            findings.append(Finding(BLOG_IMG_DIR, f"missing PNG figure for post {post}"))
    return findings


def main() -> int:
    findings: list[Finding] = []
    for post in POSTS:
        page = page_for_post(post)
        if page is None:
            findings.append(Finding(PAGES_DIR, f"expected exactly one hidden page for post {post}"))
            continue
        findings.extend(validate_page(post, page))
    findings.extend(validate_exported_assets())

    if findings:
        print("kUPS hidden page validation failed:")
        for finding in findings:
            print(f"- {finding.format()}")
        return 1

    print("kUPS hidden page validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
