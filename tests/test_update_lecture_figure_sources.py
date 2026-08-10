import tempfile
import unittest
from pathlib import Path

from scripts.update_lecture_figure_sources import parse_figure_includes, reused_assets


class ReusedAssetsTests(unittest.TestCase):
    def test_curated_figure_overrides_matching_media_inventory_record(self):
        path = "assets/img/blog/lectures/example/s04-image2.webp"
        manifest = {
            "slides": [{"slide": 4, "pptx_slide": 5}],
            "media": [
                {
                    "pptx_path": "ppt/media/image2.png",
                    "slides": [5],
                    "asset_path": path,
                    "reuse_status": "reused",
                }
            ],
            "pptx_figures": [
                {
                    "slide": 4,
                    "asset_path": path,
                    "source_media_paths": ["ppt/media/image2.png"],
                    "extraction_method": "pptx-picture-export",
                    "reuse_status": "reused",
                }
            ],
        }

        self.assertEqual(
            reused_assets(manifest)[path],
            {"slides": [4], "extraction_method": "pptx-picture-export"},
        )

    def test_unrelated_duplicate_records_remain_an_error(self):
        path = "assets/img/blog/lectures/example/s04-image2.webp"
        manifest = {
            "media": [
                {
                    "pptx_path": "ppt/media/image2.png",
                    "slides": [5],
                    "asset_path": path,
                    "reuse_status": "reused",
                }
            ],
            "pptx_figures": [
                {
                    "slide": 4,
                    "asset_path": path,
                    "source_media_paths": ["ppt/media/image3.png"],
                    "extraction_method": "pptx-picture-export",
                    "reuse_status": "reused",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "conflicting published figure record"):
            reused_assets(manifest)

    def test_media_copy_inherits_all_logical_slides_using_source_media(self):
        path = "assets/img/blog/lectures/example/s04-image2.webp"
        manifest = {
            "slides": [
                {"slide": 4, "pptx_slide": 5},
                {"slide": 23, "pptx_slide": 24},
            ],
            "media": [
                {
                    "pptx_path": "ppt/media/image2.png",
                    "slides": [5, 24],
                    "reuse_status": "not-published",
                }
            ],
            "pptx_figures": [
                {
                    "slide": 4,
                    "asset_path": path,
                    "source_media_paths": ["ppt/media/image2.png"],
                    "extraction_method": "pptx-media-copy",
                    "reuse_status": "reused",
                }
            ],
        }

        self.assertEqual(reused_assets(manifest)[path]["slides"], [4, 23])

    def test_duplicate_post_include_keeps_first_contextual_caption(self):
        include = (
            '{% include figure.liquid path="assets/img/blog/example.png" '
            'caption="CAPTION" %}'
        )
        with tempfile.TemporaryDirectory() as directory:
            post_path = Path(directory) / "post.md"
            post_path.write_text(
                "\n".join(
                    [
                        include.replace("CAPTION", "First context"),
                        include.replace("CAPTION", "Second context"),
                    ]
                ),
                encoding="utf-8",
            )

            figures = parse_figure_includes(post_path)

        self.assertEqual(figures["assets/img/blog/example.png"]["caption"], "First context")


if __name__ == "__main__":
    unittest.main()
