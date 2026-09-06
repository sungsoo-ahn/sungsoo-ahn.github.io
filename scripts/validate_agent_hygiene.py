#!/usr/bin/env python3
"""Validate canonical Codex instructions, skill metadata, and local references."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IGNORES = {"/.claude/settings.local.json", "/.env", "/.env.*"}
PATH_PREFIXES = (
    ".agents/", ".github/", "_data/", "_pages/", "_posts/",
    "assets/", "cv/", "scripts/", "tests/", "docs/",
)
OPTIONAL_FIELDS = {"license", "compatibility", "metadata", "allowed-tools"}


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---[ \t]*(?:\n|\Z)", text, re.DOTALL)
    if not match:
        raise ValueError("missing or unterminated YAML frontmatter")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return metadata, text[match.end():]


def without_fences(text: str) -> str:
    """Ignore executable examples, including tilde and longer backtick fences."""
    result = []
    fence_char, fence_length = "", 0
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if match:
            marker, rest = match.groups()
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
                continue
            if marker[0] == fence_char and len(marker) >= fence_length and not rest.strip():
                fence_char, fence_length = "", 0
                continue
        if not fence_char:
            result.append(line)
    return "\n".join(result)


def validate_references(path: Path, root: Path) -> list[str]:
    """Check Markdown links and concrete root-relative inline code paths."""
    text = without_fences(path.read_text(encoding="utf-8"))
    findings = []
    destinations = re.findall(
        r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)", text
    )
    destinations += re.findall(
        r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", text, re.MULTILINE
    )
    for destination in destinations:
        destination = destination.removeprefix("<").removesuffix(">")
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        local = unquote(parsed.path)
        candidate = root / local.lstrip("/") if local.startswith("/") else path.parent / local
        if not candidate.exists():
            findings.append(f"{path.relative_to(root)}: missing linked resource {destination}")

    for value in re.findall(r"`([^`\n]+)`", text):
        if not value.startswith(PATH_PREFIXES):
            continue
        if any(token in value for token in ("<", ">", "*", "YYYY", "{", " ")) or "\n" in value:
            continue
        candidate = value.rstrip("/.,:;")
        if not (root / candidate).exists():
            findings.append(f"{path.relative_to(root)}: missing referenced path {candidate}")
    return findings


def validate(root: Path = ROOT) -> list[str]:
    root = Path(root)
    findings = []
    instruction_files = []
    for guide in (root / "AGENTS.md", root / "_posts" / "AGENTS.md"):
        if not guide.is_file():
            findings.append(f"missing canonical guide: {guide.relative_to(root)}")
        else:
            instruction_files.append(guide)

    skill_root = root / ".agents" / "skills"
    skills = sorted(skill_root.glob("*/SKILL.md"))
    if not skills:
        findings.append(".agents/skills: no skills found")
    if skill_root.is_dir():
        for folder in sorted(skill_root.iterdir()):
            if folder.is_dir() and any(folder.iterdir()) and not (folder / "SKILL.md").is_file():
                findings.append(f"{folder.relative_to(root)}: missing SKILL.md")

    for skill in skills:
        instruction_files.append(skill)
        instruction_files.extend(sorted(skill.parent.glob("references/**/*.md")))
        try:
            metadata, body = parse_frontmatter(skill)
        except ValueError as exc:
            findings.append(f"{skill.relative_to(root)}: {exc}")
            continue
        prefix = str(skill.relative_to(root))
        unknown = set(metadata) - {"name", "description"} - OPTIONAL_FIELDS
        if unknown:
            findings.append(f"{prefix}: unsupported frontmatter fields {sorted(map(str, unknown))}")
        name = metadata.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
            findings.append(f"{prefix}: name must use lowercase letters, digits, and hyphens (max 64 characters)")
        if name != skill.parent.name:
            findings.append(f"{prefix}: name must match its directory")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip() or len(description) > 1024:
            findings.append(f"{prefix}: description must be a nonempty string (max 1024 characters)")
        if "metadata" in metadata and not isinstance(metadata["metadata"], dict):
            findings.append(f"{prefix}: metadata must be a mapping")
        if len(body.splitlines()) > 500:
            findings.append(f"{prefix}: exceeds the 500-line skill body budget")

    for path in instruction_files:
        findings.extend(validate_references(path, root))
        if "/Users/" in path.read_text(encoding="utf-8"):
            findings.append(f"{path.relative_to(root)}: contains a machine-specific home path")
    sources = root / ".agents" / "third-party" / "sources.md"
    if sources.is_file():
        findings.extend(validate_references(sources, root))

    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        findings.append("missing .gitignore")
    else:
        ignore_lines = {
            line.strip() for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        missing = sorted(REQUIRED_IGNORES - ignore_lines)
        if missing:
            findings.append(f".gitignore: missing local-secret patterns {missing}")
    # Machine-local ignored settings are not repository instructions or audit input.
    return findings


def main() -> int:
    findings = validate()
    if findings:
        print("Agent hygiene validation failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Agent hygiene validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
