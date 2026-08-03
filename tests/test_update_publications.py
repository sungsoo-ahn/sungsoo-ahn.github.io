import unittest
from pathlib import Path

import yaml

import scripts.update_publications as publications


class UpdatePublicationsTest(unittest.TestCase):
    def test_canonical_publications_are_valid(self):
        entries = publications.load_publications()
        self.assertEqual(publications.validate_publications(entries), [])
        self.assertEqual(len(entries), 74)
        self.assertEqual(sum(entry["selected"] for entry in entries if "selected" in entry), 12)

    def test_jekyll_reads_yaml_instead_of_bibtex(self):
        publications_page = Path("_pages/publications.md").read_text(encoding="utf-8")
        selected_include = Path("_includes/selected_papers.liquid").read_text(encoding="utf-8")
        config = yaml.safe_load(Path("_config.yml").read_text(encoding="utf-8"))
        self.assertIn("site.data.publications", publications_page)
        self.assertIn("site.data.publications", selected_include)
        self.assertNotIn("jekyll/scholar", config["plugins"])
        self.assertNotIn("scholar", config)
        self.assertFalse(Path("_bibliography/papers.bib").exists())

    def test_publication_ids_are_renderable_from_related_content(self):
        entries = publications.load_publications()
        ids = {entry["id"] for entry in entries}
        self.assertEqual(len(ids), len(entries))
        self.assertIn("seong2026discovering", ids)


if __name__ == "__main__":
    unittest.main()
