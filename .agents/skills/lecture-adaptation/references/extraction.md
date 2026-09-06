# Lecture visual extraction and manifests

For source-deck reuse, the figure must represent the actual visible PowerPoint
object. Existing manifest contracts remain authoritative for active migrations.

## Extract the semantic object

Inspect the actual slide object tree. Package media can include hidden,
duplicate, unused, or alternate assets.

- Copy an unmodified browser-safe picture from `ppt/media` when it matches
  what the slide displays.
- Export the exact picture or semantic shape group when crops, masks, rotations,
  or native shapes affect the visible figure.
- Retain a composite as one scientific figure rather than counting every
  embedded fragment independently.
- Transcribe prose, equations, and tables into Markdown/HTML/MathJax.
  Do not publish a full-slide screenshot or a PDF-region crop in a native
  extraction workflow.
- Treat EMF and other unsupported formats as extraction problems. Use an
  appropriate object-preserving renderer; do not omit required content silently.

Use existing helpers, inspecting their CLI before use:
[inventory](../../../../scripts/inventory_lecture_pptx_objects.py),
[extraction](../../../../scripts/extract_lecture_pptx_figure.py), and
[manifest finalization](../../../../scripts/finalize_pptx_native_manifest.py).

## Preserve the data contract

Keep `.agents/lecture-adaptation/*.json` at their current paths. Record the
slide, published asset, source media or shape, method, role, and reuse status.
Existing native extraction methods are `pptx-media-copy`,
`pptx-picture-export`, and `pptx-shape-group-export`.

Track unique substantive visuals, duplicates, and decorative omissions without
a figure quota. Existing completed native migrations require exact
manifest-to-post asset agreement and no reused PDF-region records.

Use [validate_blog.py](../../../../scripts/validate_blog.py) and
[update_lecture_figure_sources.py](../../../../scripts/update_lecture_figure_sources.py)
for the actual schema/check behavior. Run the latter with `--check` to inspect
drift; regeneration is a separate action.

## Attribution

Separate visual origin from the paper explaining a method. Keep confirmed
paper/project credits, label lecturer-made diagrams accurately, and record
uncertain origin honestly. Confirmed reuse rights apply to the supplied deck,
not automatically to other images from its cited sources.
