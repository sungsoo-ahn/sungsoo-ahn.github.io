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
FOUNDATIONS_PAGE = PAGES_DIR / "kups-md-foundations.md"
PUBLICATION_STATUSES = {"draft", "ready"}

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

    publication_status = frontmatter.get("publication_status")
    if (
        publication_status is not None
        and publication_status not in PUBLICATION_STATUSES
    ):
        findings.append(
            Finding(
                path,
                "publication_status must be 'draft' or 'ready' when present",
            )
        )

    required_links = (
        "https://github.com/sungsoo-ahn/kups-md-tutorials",
        f"kups-tutorial run {post}",
        f"kups-tutorial verify {post}",
    )
    for fragment in required_links:
        if fragment not in body:
            findings.append(Finding(path, f"missing source link fragment: {fragment}"))

    if "\\[" in body or "\\]" in body:
        findings.append(
            Finding(
                path,
                "display math uses raw \\[...\\] delimiters; use $$...$$ so Jekyll preserves it",
            )
        )

    if re.search(r"^#{1,6}\s+current\s+(?:status|state)\b", body, re.IGNORECASE | re.MULTILINE):
        findings.append(
            Finding(
                path,
                "reader-facing article contains a development 'Current Status/State' heading",
            )
        )

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

    return findings


def validate_foundations_page(path: Path) -> list[Finding]:
    if not path.exists():
        return [Finding(path, "missing MD/JAX foundations lesson")]
    text = path.read_text(encoding="utf-8")
    parsed = parse_frontmatter(text)
    if parsed is None:
        return [Finding(path, "missing YAML frontmatter")]

    frontmatter, body = parsed
    expected = {
        "layout": "post",
        "post_type": "tutorial",
        "series": SERIES,
        "series_order": "0",
        "related_posts": "false",
        "nav": "false",
        "permalink": f"/{SERIES}/foundations/",
    }
    findings = [
        Finding(path, f"{key} must be {value!r}")
        for key, value in expected.items()
        if frontmatter.get(key) != value
    ]
    if frontmatter.get("publication_status") not in PUBLICATION_STATUSES:
        findings.append(Finding(path, "foundations publication_status must be draft or ready"))
    for fragment in (
        "https://github.com/sungsoo-ahn/kups-md-tutorials",
        "verify-notebooks --posts 00",
        "kups-notebooks/post-00/",
        "kups_md_post00_atomic_trajectory.svg",
    ):
        if fragment not in body:
            findings.append(Finding(path, f"missing foundations fragment: {fragment}"))
    if "\\[" in body or "\\]" in body:
        findings.append(Finding(path, "foundations display math must use $$ delimiters"))
    if re.search(r"^#{1,6}\s+current\s+(?:status|state)\b", body, re.IGNORECASE | re.MULTILINE):
        findings.append(Finding(path, "foundations contains a development status heading"))

    figure_path = BLOG_IMG_DIR / "kups_md_post00_atomic_trajectory.svg"
    if not figure_path.exists():
        findings.append(Finding(figure_path, "missing foundations SVG"))
    if not figure_path.with_suffix(".png").exists():
        findings.append(Finding(figure_path.with_suffix(".png"), "missing foundations PNG poster"))
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

    notebook_manifest = EXPORT_DIR / "notebook-cells.json"
    if not notebook_manifest.exists():
        findings.append(Finding(notebook_manifest, "missing notebook-cell manifest"))
    else:
        notebook_data = json.loads(notebook_manifest.read_text(encoding="utf-8"))
        foundation_cells = {
            cell.get("cell_id")
            for cell in notebook_data.get("cells", [])
            if cell.get("post") == "00"
        }
        expected_cells = {
            "post00-setup",
            "post00-energy-to-force",
            "post00-jax-trajectory",
            "post00-kups-state",
        }
        if foundation_cells != expected_cells:
            findings.append(
                Finding(
                    notebook_manifest,
                    "foundations notebook cells are missing or stale: "
                    f"expected {sorted(expected_cells)}, found {sorted(foundation_cells)}",
                )
            )
    return findings


def main() -> int:
    findings: list[Finding] = []
    findings.extend(validate_foundations_page(FOUNDATIONS_PAGE))
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
