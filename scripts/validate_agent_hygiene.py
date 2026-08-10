#!/usr/bin/env python3
"""Validate canonical repository instructions and skill adapters."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SKILLS = ROOT / ".agents" / "skills"
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
CANONICAL_GUIDES = (ROOT / "AGENTS.md", ROOT / "_posts" / "AGENTS.md")
ADAPTER_GUIDES = {
    ROOT / "CLAUDE.md": "AGENTS.md",
    ROOT / "_posts" / "CLAUDE.md": "_posts/AGENTS.md",
}
REQUIRED_IGNORES = {
    "/.claude/settings.local.json",
    "/.env",
    "/.env.*",
}
PATH_PREFIXES = (
    ".agents/",
    ".claude/",
    ".github/",
    "_data/",
    "_pages/",
    "_posts/",
    "assets/",
    "cv/",
    "scripts/",
    "tests/",
)


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    try:
        _, raw, body = text.split("---", 2)
    except ValueError as exc:
        raise ValueError("unterminated YAML frontmatter") from exc

    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body


def validate_literal_paths(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8")
    for value in re.findall(r"`([^`]+)`", text):
        if not value.startswith(PATH_PREFIXES):
            continue
        if any(token in value for token in ("<", ">", "*", "YYYY")):
            continue
        candidate = value.rstrip("/.,:;")
        if not (ROOT / candidate).exists():
            findings.append(f"{path.relative_to(ROOT)}: missing referenced path {candidate}")
    return findings


def validate() -> list[str]:
    findings: list[str] = []

    for guide in CANONICAL_GUIDES:
        if not guide.is_file():
            findings.append(f"missing canonical guide: {guide.relative_to(ROOT)}")
            continue
        findings.extend(validate_literal_paths(guide))

    for adapter, target in ADAPTER_GUIDES.items():
        if not adapter.is_file():
            findings.append(f"missing adapter guide: {adapter.relative_to(ROOT)}")
        elif target not in adapter.read_text(encoding="utf-8"):
            findings.append(f"{adapter.relative_to(ROOT)}: must point to {target}")

    canonical_names = {path.parent.name for path in CANONICAL_SKILLS.glob("*/SKILL.md")}
    adapter_names = {path.parent.name for path in CLAUDE_SKILLS.glob("*/SKILL.md")}
    if canonical_names != adapter_names:
        findings.append(
            "skill trees differ: "
            f"canonical-only={sorted(canonical_names - adapter_names)}, "
            f"adapter-only={sorted(adapter_names - canonical_names)}"
        )

    instruction_files = [*CANONICAL_GUIDES, *ADAPTER_GUIDES]
    instruction_files.extend(ROOT.glob(".github/agents/*.agent.md"))

    for name in sorted(canonical_names):
        canonical = CANONICAL_SKILLS / name / "SKILL.md"
        adapter = CLAUDE_SKILLS / name / "SKILL.md"
        instruction_files.append(canonical)
        if adapter.is_file():
            instruction_files.append(adapter)
        try:
            metadata, _ = parse_frontmatter(canonical)
        except ValueError as exc:
            findings.append(f"{canonical.relative_to(ROOT)}: {exc}")
            continue
        if set(metadata) != {"name", "description"}:
            findings.append(
                f"{canonical.relative_to(ROOT)}: frontmatter must contain only name and description"
            )
        if metadata.get("name") != name:
            findings.append(f"{canonical.relative_to(ROOT)}: name must match its directory")
        if not metadata.get("description"):
            findings.append(f"{canonical.relative_to(ROOT)}: description is required")
        if len(canonical.read_text(encoding="utf-8").splitlines()) > 500:
            findings.append(f"{canonical.relative_to(ROOT)}: exceeds the 500-line skill budget")

        if not adapter.is_file():
            continue
        try:
            adapter_metadata, adapter_body = parse_frontmatter(adapter)
        except ValueError as exc:
            findings.append(f"{adapter.relative_to(ROOT)}: {exc}")
            continue
        if adapter_metadata != metadata:
            findings.append(f"{adapter.relative_to(ROOT)}: metadata differs from canonical skill")
        expected_target = f"../../../.agents/skills/{name}/SKILL.md"
        if expected_target not in adapter_body:
            findings.append(f"{adapter.relative_to(ROOT)}: must point to {expected_target}")

    for path in instruction_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text:
            findings.append(f"{path.relative_to(ROOT)}: contains a machine-specific home path")

    ignore_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_ignores = sorted(REQUIRED_IGNORES - ignore_lines)
    if missing_ignores:
        findings.append(f".gitignore: missing local-secret patterns {missing_ignores}")

    local_settings = ROOT / ".claude" / "settings.local.json"
    if local_settings.is_file():
        settings_text = local_settings.read_text(encoding="utf-8")
        if re.search(r"(?i)password|passwd|api[_-]?key|access[_-]?token", settings_text):
            findings.append(".claude/settings.local.json: contains credential-like text")
        if any(command in settings_text for command in ("rm -rf", "git push", "git rm")):
            findings.append(".claude/settings.local.json: contains broad destructive permissions")

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
