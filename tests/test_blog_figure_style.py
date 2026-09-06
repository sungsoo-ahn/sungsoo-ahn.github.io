"""Check exported artifacts and audit behavior without writing site assets."""
import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts import blog_figure_style as bfs


class FigureStyleTests(unittest.TestCase):
    def setUp(self):
        self.rc = matplotlib.rc_context()
        self.rc.__enter__()
        self.addCleanup(self.rc.__exit__, None, None, None)
        self.addCleanup(plt.close, "all")
        bfs.use_publication_style()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, fig, path, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bfs.save_publication_figure(fig, path, **kwargs)

    def test_fixed_dimensions_and_editable_text_despite_global_tight_crop(self):
        for width in (bfs.COLUMN, 4.2):
            with self.subTest(width=width):
                matplotlib.rcParams["savefig.bbox"] = "tight"
                fig, ax = plt.subplots(figsize=(width, 2.3), layout="constrained")
                ax.plot([0, 1], [0, 1], color=bfs.PURPLE)
                ax.set_xlabel("Time (s)")
                engine, canvas = fig.get_layout_engine(), fig.canvas
                pdf, svg, png = self.save(fig, self.root / f"plot-{width}.svg",
                                         target_width_in=width, dpi=200, close=False)
                media = re.search(rb"/MediaBox\s*\[\s*([\d.\s-]+)\]", pdf.read_bytes())
                self.assertIsNotNone(media)
                bounds = list(map(float, media.group(1).split()))
                self.assertAlmostEqual(bounds[2] - bounds[0], width * 72, places=4)
                self.assertAlmostEqual(bounds[3] - bounds[1], 2.3 * 72, places=4)
                self.assertNotIn(b"/Subtype /Type3", pdf.read_bytes())
                vector = ElementTree.parse(svg).getroot()
                self.assertAlmostEqual(float(vector.attrib["width"].removesuffix("pt")), width * 72, places=4)
                self.assertTrue(vector.findall(".//{http://www.w3.org/2000/svg}text"))
                with Image.open(png) as raster:
                    self.assertLessEqual(abs(raster.width - width * 200), 1)
                    self.assertLessEqual(abs(raster.height - 2.3 * 200), 1)
                self.assertEqual(matplotlib.rcParams["savefig.bbox"], "tight")
                self.assertIs(fig.get_layout_engine(), engine)
                self.assertIs(fig.canvas, canvas)

    def test_invalid_width_fails_before_output(self):
        fig = plt.figure(figsize=(bfs.COLUMN, 2))
        for width in (4.0, 0, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.save(fig, self.root / "absent" / "figure", target_width_in=width)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_audit_detects_width_overlap_small_and_clipped_text(self):
        fig = plt.figure(figsize=(3.25, 2))
        fig.text(.3, .5, "First", fontsize=9)
        fig.text(.3, .5, "Second", fontsize=9)
        fig.text(.5, .2, "Tiny", fontsize=4)
        fig.text(1.01, .5, "Outside", fontsize=9)
        canvas = fig.canvas
        codes = {issue.code for issue in bfs.audit_figure(fig, target_width_in=4)}
        self.assertEqual(codes, {"width", "small-text", "clipped-text", "text-overlap"})
        self.assertIs(fig.canvas, canvas)

    def test_intentional_overlap_excludes_only_named_artist(self):
        fig = plt.figure(figsize=(3.25, 2))
        fig.text(.2, .5, "Label")
        intentional = fig.text(.2, .5, "Intentional overlay")
        self.assertTrue(bfs.audit_figure(fig))
        self.assertEqual(bfs.audit_figure(fig, ignore=[intentional]), [])
        fig.text(.2, .5, "Accidental overlay")
        self.assertTrue(any(x.code == "text-overlap" for x in bfs.audit_figure(fig, ignore=[intentional])))

    def test_hidden_axes_legends_ticks_are_not_drawn_text(self):
        fig, ax = plt.subplots(figsize=(4.2, 3), layout="constrained")
        ax.plot([0, 1], [0, 1], label="Legend that is hidden")
        ax.legend().set_visible(False)
        ax.set_xticks([-5, 0, 1, 5], labels=["Offscreen" * 20, "0", "1", "Offscreen" * 20])
        ax.set_xlim(0, 1)
        ax.text(.5, .5, "Hidden" * 30, visible=False)
        hidden = fig.add_axes([.1, .1, .2, .2])
        hidden.text(0, 0, "Hidden" * 30)
        hidden.set_visible(False)
        self.assertEqual(bfs.audit_figure(fig), [])

    def test_multiline_colorbar_shared_axes_and_arrow_are_valid(self):
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), layout="constrained", sharey=True)
        for ax in axes:
            heat = ax.imshow(np.arange(16).reshape(4, 4), cmap="Purples")
            ax.set_xlabel("A label with\nan intentional line break")
        fig.colorbar(heat, ax=axes, label="Value")
        self.assertEqual(bfs.audit_figure(fig), [])

        diagram, ax = plt.subplots(figsize=(3.25, 2.3), layout="constrained")
        ax.set_axis_off()
        ax.text(.4, .4, "Middle", fontsize=9)
        ax.annotate("Start", xy=(.85, .85), xytext=(.05, .05),
                    arrowprops={"arrowstyle": "->"}, fontsize=9)
        # The arrow passes near Middle, but its bounding box is not text.
        self.assertEqual(bfs.audit_figure(diagram), [])

    def test_legacy_web_export_still_returns_svg_png_and_closes(self):
        bfs.use_blog_style()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        number = fig.number
        with contextlib.redirect_stdout(io.StringIO()):
            svg, png = bfs.save_svg_png(fig, self.root / "legacy.png", dpi=100)
        self.assertEqual((svg.suffix, png.suffix), (".svg", ".png"))
        self.assertTrue(svg.is_file() and png.is_file())
        self.assertFalse(plt.fignum_exists(number))
        self.assertEqual(matplotlib.rcParams["savefig.bbox"], "tight")
        # Tight web export still crops the canvas; print export above does not.
        with Image.open(png) as raster:
            self.assertNotEqual(raster.size, (400, 300))


if __name__ == "__main__":
    unittest.main()
