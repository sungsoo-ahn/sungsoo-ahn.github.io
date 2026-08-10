import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.finalize_pptx_native_manifest import (
    exact_related_media,
    slide_from_asset_path,
)


class FinalizePptxNativeManifestTests(unittest.TestCase):
    def test_semantic_asset_name_retains_slide_number(self):
        self.assertEqual(
            slide_from_asset_path(
                "assets/img/blog/lectures/example/s16-model-framework.png"
            ),
            16,
        )

    def test_exact_related_media_uses_content_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_path = root / "s16-model-framework.png"
            asset_path.write_bytes(b"published figure")
            pptx_path = root / "deck.pptx"
            with ZipFile(pptx_path, "w") as archive:
                archive.writestr("ppt/media/image20.png", b"other figure")
                archive.writestr("ppt/media/image21.png", b"published figure")
            with ZipFile(pptx_path) as archive:
                source = exact_related_media(
                    archive,
                    str(asset_path),
                    {"ppt/media/image20.png", "ppt/media/image21.png"},
                )

        self.assertEqual(source, "ppt/media/image21.png")


if __name__ == "__main__":
    unittest.main()
