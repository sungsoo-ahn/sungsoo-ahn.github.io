#!/usr/bin/env python3
"""Validate blog post metadata, figure links, and local writing rules."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "_posts"
BLOG_IMG_DIR = ROOT / "assets" / "img" / "blog"

REQUIRED_FRONTMATTER = {
    "layout",
    "title",
    "date",
    "last_updated",
    "description",
    "authors",
    "categories",
    "tags",
    "toc",
    "related_posts",
}

FIGURE_RE = re.compile(r'{%\s*include\s+figure\.liquid\b(?P<attrs>.*?)%}')
ATTR_RE = re.compile(r'(?P<key>[\w-]+)="(?P<value>[^"]*)"')
FOOTNOTE_USE_RE = re.compile(r"\[\^([^\]]+)\](?!:)")
FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:", re.MULTILINE)
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")


@dataclass
class Finding:
    path: Path
    message: str
    severity: str = "error"

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


def parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def validate_post(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8")
    parsed = parse_frontmatter(text)
    if parsed is None:
        return [Finding(path, "missing YAML frontmatter")]

    frontmatter, body = parsed
    missing = sorted(REQUIRED_FRONTMATTER - set(frontmatter))
    if missing:
        findings.append(Finding(path, f"missing required frontmatter: {', '.join(missing)}"))

    if frontmatter.get("layout") != "post":
        findings.append(Finding(path, 'frontmatter layout must be "post"'))

    filename_date = DATE_PREFIX_RE.match(path.name)
    post_date = frontmatter.get("date")
    if filename_date and post_date and filename_date.group(1) != post_date[:10]:
        findings.append(Finding(path, f"filename date {filename_date.group(1)} differs from frontmatter date {post_date}"))

    created = parse_date(frontmatter.get("date", ""))
    updated = parse_date(frontmatter.get("last_updated", ""))
    if not created:
        findings.append(Finding(path, "date must use YYYY-MM-DD format"))
    if not updated:
        findings.append(Finding(path, "last_updated must use YYYY-MM-DD format"))
    if created and updated and updated < created:
        findings.append(Finding(path, "last_updated must not be earlier than date"))

    if "series" in frontmatter and "series_order" not in frontmatter:
        findings.append(Finding(path, "series posts must define series_order"))
    if "series_order" in frontmatter and "series" not in frontmatter:
        findings.append(Finding(path, "series_order requires series"))

    for match in FIGURE_RE.finditer(text):
        attrs = {attr.group("key"): attr.group("value") for attr in ATTR_RE.finditer(match.group("attrs"))}
        figure_path = attrs.get("path")
        if not figure_path:
            findings.append(Finding(path, "figure include missing path attribute"))
            continue
        if not (ROOT / figure_path).exists():
            findings.append(Finding(path, f"figure path does not exist: {figure_path}"))

        caption = attrs.get("caption", "")
        if "$" in caption:
            findings.append(Finding(path, f"figure caption uses dollar math delimiters: {figure_path}"))
        direct_license_source = re.search(r"\b(Wikimedia Commons|public domain|CC BY|Labster Theory)\b", caption)
        source_wording = re.search(r"\b(From|Figure from)\s+[A-Z][^.;\"]+", caption)
        adapted_wording = re.search(r"\b(Redrawn from|Adapted from|Data adapted from)\b", caption)
        if source_wording and not adapted_wording and not direct_license_source:
            findings.append(
                Finding(
                    path,
                    f'figure caption should use "Adapted from" or "Redrawn from" when the figure is redrawn: {figure_path}',
                    "warning",
                )
            )

    footnote_ids = set(FOOTNOTE_USE_RE.findall(body)) | set(FOOTNOTE_DEF_RE.findall(body))
    bad_footnote_ids = sorted(identifier for identifier in footnote_ids if "-" in identifier)
    if bad_footnote_ids:
        findings.append(Finding(path, f"footnote IDs must not contain hyphens: {', '.join(bad_footnote_ids)}"))

    used = set(FOOTNOTE_USE_RE.findall(body))
    defined = set(FOOTNOTE_DEF_RE.findall(body))
    missing_defs = sorted(used - defined)
    if missing_defs:
        findings.append(Finding(path, f"footnotes used but not defined: {', '.join(missing_defs)}"))

    return findings


def validate_assets() -> list[Finding]:
    findings: list[Finding] = []
    referenced: set[Path] = set()
    for post in sorted(POSTS_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")):
        text = post.read_text(encoding="utf-8")
        for match in FIGURE_RE.finditer(text):
            attrs = {attr.group("key"): attr.group("value") for attr in ATTR_RE.finditer(match.group("attrs"))}
            figure_path = attrs.get("path")
            if figure_path:
                referenced.add(ROOT / figure_path)

    all_blog_images = {
        image
        for image in BLOG_IMG_DIR.rglob("*")
        if image.is_file() and image.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
    }
    unused = sorted(all_blog_images - referenced)
    if unused:
        names = ", ".join(str(path.relative_to(ROOT)) for path in unused[:12])
        if len(unused) > 12:
            names += f", ... ({len(unused)} total)"
        findings.append(Finding(BLOG_IMG_DIR, f"unused blog images: {names}", "warning"))
    return findings


def main() -> int:
    findings: list[Finding] = []
    for path in sorted(POSTS_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")):
        findings.extend(validate_post(path))
    findings.extend(validate_assets())

    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]

    if errors:
        print("Blog validation failed:")
        for finding in errors:
            print(f"- {finding.format()}")
        if warnings:
            print("\nWarnings:")
            for finding in warnings:
                print(f"- {finding.format()}")
        return 1

    if warnings:
        print("Blog validation passed with warnings:")
        for finding in warnings:
            print(f"- {finding.format()}")
        return 0

    print("Blog validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
