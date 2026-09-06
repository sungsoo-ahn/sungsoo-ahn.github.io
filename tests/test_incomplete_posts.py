"""Check unlisted-post metadata and real Jekyll discovery surfaces in isolation.

Run with Homebrew Ruby on PATH on macOS. Only temporary fixtures are generated;
the user's posts, assets, and running preview are never rewritten.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.validate_blog import validate_post


ROOT = Path(__file__).resolve().parents[1]
DRAFT_META = "tags: [incomplete, stochastic-control]\ndraft: true\nsitemap: false\nnoindex: true\n"
POST_META = """layout: post
title: {title}
date: {date}
last_updated: 2026-09-06
description: A short test article.
post_type: tutorial
editorial_status: ai-generated
authors: [Sungsoo Ahn]
categories: [generative-modeling]
toc: false
related_posts: false
"""


class IncompleteMetadataTests(unittest.TestCase):
    def findings(self, visibility):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "2026-08-30-example.md"
            metadata = POST_META.format(title="Example", date="2026-08-30")
            path.write_text(f"---\n{metadata}{visibility}---\n\nExample.\n", encoding="utf-8")
            return [finding.message for finding in validate_post(path)]

    def test_complete_visibility_bundle_passes(self):
        self.assertEqual(self.findings(DRAFT_META), [])

    def test_incomplete_tag_requires_each_visibility_override(self):
        for line in ("draft: true\n", "sitemap: false\n", "noindex: true\n"):
            with self.subTest(line=line):
                findings = self.findings(DRAFT_META.replace(line, ""))
                self.assertTrue(any("incomplete posts require" in item for item in findings))

    def test_unpublished_would_break_direct_preview(self):
        self.assertIn("incomplete posts must remain readable; omit published: false",
                      self.findings(DRAFT_META + "published: false\n"))

    def test_ready_post_without_overrides_passes(self):
        self.assertEqual(self.findings("tags: [stochastic-control]\n"), [])


class IncompleteRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="incomplete-post-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.source = Path(cls.temp.name) / "source"
        cls.destination = Path(cls.temp.name) / "site"
        for directory in ("_layouts", "_includes", "_posts", "_scripts", "_data"):
            (cls.source / directory).mkdir(parents=True, exist_ok=True)
        for directory, names in {
            "_layouts": ["page.liquid", "post-type-archive.liquid", "archive.liquid", "post.liquid"],
            "_includes": ["blog-category-nav.liquid", "latest_posts.liquid", "related_posts.liquid", "metadata.liquid"],
            "_scripts": ["search.liquid.js"],
            "_data": ["blog_categories.yml"],
        }.items():
            for name in names:
                shutil.copy2(ROOT / directory / name, cls.source / directory / name)
        (cls.source / "_layouts/default.html").write_text(
            '<html><head>{% include metadata.liquid %}</head><body>{{ content }}</body></html>', encoding="utf-8")
        cls.write_page("index.md", "permalink: /\nlatest_posts:\n  limit: 1\n", "{% include latest_posts.liquid %}")
        shutil.copy2(ROOT / "_pages/blog.md", cls.source / "blog.md")
        for name in ("all", "selected", "tutorials", "incomplete"):
            shutil.copy2(ROOT / f"_pages/blog-{name}.md", cls.source / f"{name}.md")
        for slug, date in (("public-older", "2026-02-01"), ("public-newer", "2026-03-01"), ("unfinished-example", "2026-08-30")):
            metadata = POST_META.format(title=slug, date=date)
            visibility = DRAFT_META if slug == "unfinished-example" else "tags: [stochastic-control]\nselected: true\n"
            if slug != "unfinished-example":
                metadata = metadata.replace("editorial_status: ai-generated\n", "")
            if slug == "public-older":
                metadata = metadata.replace("related_posts: false", "related_posts: true")
            (cls.source / f"_posts/{date}-{slug}.md").write_text(
                f"---\n{metadata}{visibility}---\n\nBody for {slug}.\n", encoding="utf-8")
        cls.build()

    @classmethod
    def write_page(cls, name, metadata, body):
        (cls.source / name).write_text(f"---\nlayout: default\n{metadata}---\n{body}\n", encoding="utf-8")

    @classmethod
    def build(cls, destination=None):
        ruby = """
require 'jekyll'
config = Jekyll.configuration({
  'source' => ARGV[0], 'destination' => ARGV[1], 'url' => 'https://example.test',
  'permalink' => '/blog/:year/:title/', 'include' => ['_scripts'],
  'plugins' => ['jekyll-feed', 'jekyll-sitemap', 'jekyll-archives-v2', 'jekyll-toc'],
  'posts_in_search' => true, 'related_blog_posts' => {'enabled' => true, 'max_related' => 1},
  'jekyll-archives' => {'posts' => {'enabled' => ['year', 'categories'],
    'permalinks' => {'year' => '/blog/:year/', 'categories' => '/blog/category/:name/'}}}
})
Jekyll::Site.new(config).process
"""
        env = os.environ.copy()
        env["JEKYLL_ENV"] = "production"
        result = subprocess.run(["bundle", "exec", "ruby", "-e", ruby, str(cls.source), str(destination or cls.destination)],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def rendered(self, path):
        return (self.destination / path).read_text(encoding="utf-8")

    def test_public_discovery_excludes_incomplete(self):
        for path in ("index.html", "blog/index.html", "blog/2026/public-older/index.html", "blog/all/index.html", "blog/selected/index.html",
                     "blog/type/tutorial/index.html", "blog/2026/index.html",
                     "blog/category/generative-modeling/index.html", "assets/js/search-data.js", "feed.xml", "sitemap.xml"):
            with self.subTest(path=path):
                output = self.rendered(path)
                self.assertNotIn("unfinished-example", output)
                self.assertNotIn("/blog/incomplete/", output)
                self.assertIn("public-newer", output)

    def test_counts_and_limits_apply_after_filtering(self):
        self.assertIn("All posts 2", self.rendered("blog/all/index.html"))
        self.assertIn("Tutorials 2", self.rendered("blog/all/index.html"))
        self.assertIn("Tutorials 2", self.rendered("blog/index.html"))
        self.assertNotIn("public-older", self.rendered("index.html"))
        self.assertIn("public-newer", self.rendered("index.html"))

    def test_unlisted_index_and_direct_post_remain_readable(self):
        index = self.rendered("blog/incomplete/index.html")
        self.assertIn("unfinished-example", index)
        self.assertNotIn("public-newer", index)
        for output in (index, self.rendered("blog/2026/unfinished-example/index.html")):
            self.assertRegex(output, r'<meta name="robots" content="noindex, follow">')
        self.assertIn("Incomplete draft.", self.rendered("blog/2026/unfinished-example/index.html"))

    def test_ready_post_returns_to_public_surfaces(self):
        # Exercise the documented transition without changing the shared build.
        post = self.source / "_posts/2026-08-30-unfinished-example.md"
        original = post.read_text(encoding="utf-8")
        ready = Path(self.temp.name) / "ready"
        try:
            post.write_text(original.replace(DRAFT_META, "tags: [stochastic-control]\n"), encoding="utf-8")
            self.build(ready)
            for path in ("blog/index.html", "blog/all/index.html", "assets/js/search-data.js", "feed.xml", "sitemap.xml"):
                self.assertIn("unfinished-example", (ready / path).read_text(encoding="utf-8"))
            self.assertNotIn("unfinished-example", (ready / "blog/incomplete/index.html").read_text(encoding="utf-8"))
            self.assertNotIn('content="noindex, follow"', (ready / "blog/2026/unfinished-example/index.html").read_text(encoding="utf-8"))
        finally:
            post.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
