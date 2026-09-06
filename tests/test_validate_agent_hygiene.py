"""Instruction validation uses isolated repositories, never local settings."""
import tempfile
import unittest
from pathlib import Path

from scripts import validate_agent_hygiene as hygiene


class HygieneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("AGENTS.md", "# Repository\n")
        self.write("_posts/AGENTS.md", "# Posts\n")
        self.write(".gitignore", "\n".join(sorted(hygiene.REQUIRED_IGNORES)))
        self.skill = ".agents/skills/example/SKILL.md"
        self.write(self.skill, "---\nname: example\ndescription: An example skill.\n---\n# Example\n")

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def test_codex_only_tree_and_ignored_settings(self):
        self.write(".claude/settings.local.json", "not even JSON; password; git push")
        self.assertEqual(hygiene.validate(self.root), [])

    def test_multiline_yaml_optional_metadata(self):
        self.write(self.skill, """---
name: example
description: >-
  Handle an example
  with a multiline description.
metadata:
  short-description: Example
  nested:
    version: 2
license: MIT
compatibility: Python
allowed-tools: Read
---
# Example
""")
        self.assertEqual(hygiene.validate(self.root), [])

    def test_invalid_metadata(self):
        cases = (
            ("name: [", "invalid YAML"),
            ("- example", "must be a YAML mapping"),
            ("name: Wrong Name\ndescription: test", "name must use lowercase"),
            ("name: different\ndescription: test", "name must match"),
            ("name: example\ndescription: []", "description must be"),
            ("name: example\ndescription: test\nmetadata: invalid", "metadata must be"),
        )
        for yaml_text, expected in cases:
            with self.subTest(expected=expected):
                self.write(self.skill, f"---\n{yaml_text}\n---\nBody\n")
                self.assertTrue(any(expected in issue for issue in hygiene.validate(self.root)))

    def test_local_links_relative_root_and_reference_style(self):
        self.write(".agents/skills/example/references/notes.md", "Notes\n")
        self.write("docs/a note.md", "Note\n")
        self.write("AGENTS.md", """[skill](.agents/skills/example/SKILL.md#example)
[root](/docs/a%20note.md)
[spaced](<docs/a note.md>)
[alias]: .agents/skills/example/references/notes.md "Notes"
[external](https://example.com/missing)
[anchor](#non-path-anchor)
[mail](mailto:test@example.com)
""")
        self.assertEqual(hygiene.validate(self.root), [])
        self.write(".agents/skills/example/references/notes.md", "[missing](absent.md)")
        self.assertTrue(any("missing linked resource absent.md" in issue for issue in hygiene.validate(self.root)))

    def test_missing_root_inline_path_and_reference_definition(self):
        self.write("AGENTS.md", "`scripts/absent.py`\n[alias]: absent.md\n")
        findings = hygiene.validate(self.root)
        self.assertTrue(any("missing referenced path scripts/absent.py" in x for x in findings))
        self.assertTrue(any("missing linked resource absent.md" in x for x in findings))

    def test_examples_are_not_paths_to_validate(self):
        self.write("AGENTS.md", """# Examples
```python
`scripts/example.py`
[not a real link](missing.md)
```
~~~~markdown
[not a real link](missing.md)
~~~~
`assets/img/<slug>/figure.svg` and `scripts/*.py`
""")
        self.assertEqual(hygiene.validate(self.root), [])

    def test_orphan_resources_and_required_ignores(self):
        self.write(".agents/skills/orphan/references/notes.md", "Notes")
        self.write(".gitignore", "/.env\n")
        findings = hygiene.validate(self.root)
        self.assertTrue(any("orphan: missing SKILL.md" in x for x in findings))
        self.assertTrue(any("missing local-secret patterns" in x for x in findings))


if __name__ == "__main__":
    unittest.main()
