# Generate Blog Figures

Create matplotlib figure scripts for blog posts following the project's visual conventions.

## File Structure

- Script: `scripts/generate_<postname>_figures.py`
- Output: `assets/img/blog/<prefix>_<name>.png`
- One function per figure (e.g., `generate_energy_cycle_figure(output_path)`)
- Module-level color palette constants
- `if __name__ == '__main__':` block that calls all functions with correct output paths

## Visual Style

Reference existing scripts for patterns:
- `scripts/generate_dft_figures.py` — flowchart style using `FancyBboxPatch` / `FancyArrowPatch`
- `scripts/generate_fokker_planck_figures.py` — data plots with color palettes, helper functions

### Color Palette

Define module-level constants. Typical palette:
- Slate blue for primary curves/boxes: `#5b7fa5`, fill `#dce8f4`
- Warm amber for highlights: `#e8a030`, `#fff3e0`
- Green for optimal/output: `#4caf50`, `#e0f2e9`
- Red for warnings/errors: `#d32f2f`
- Text: `#263238`

### Flowcharts

Use `FancyBboxPatch` for boxes, `FancyArrowPatch` for arrows:
```python
box = FancyBboxPatch(
    (cx - bw/2, cy - bh/2), bw, bh,
    boxstyle=f'round,pad={rounding}',
    facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
ax.add_patch(box)
```

### Data Plots

Use helper functions for axis styling:
```python
def _style_axis(ax, xlim, ylim, xlabel=None, ylabel=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ...
```

## Output Settings

- PNG format, `bbox_inches='tight'`, `facecolor='white'`
- Diagrams/flowcharts: 200 DPI
- Data plots: 150 DPI
- Print confirmation after each save: `print(f"Saved {name} to {output_path}")`

## Workflow

1. Write the full script with all figure functions
2. Run with `uv run python scripts/generate_<postname>_figures.py`
3. Take screenshots of outputs, iterate on visual quality
4. Expect multiple rounds of refinement per figure
5. All data should be hardcoded (approximate/illustrative values, not copyrighted)

## Embedding in Blog Posts

Use the Liquid include:
```liquid
{% include figure.liquid loading="eager" path="assets/img/blog/your_figure.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Your caption here." %}
```
