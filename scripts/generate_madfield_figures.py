"""Generate figures for the MADField blog post.

Currently provides the Cambridge Structural Database (CSD) MOF-growth chart used
in the post introduction.

Data source: the CSD "MOF subset" (broad definition, 137,167 entries; CSD 2026.1
update), queried directly from the locally installed Cambridge Structural
Database with the CSD Python API (ccdc 3.7) and binned by each entry's
deposition year.  This gives one internally consistent definition across all
years.  The same subset total (137,167) is reported by the CCDC
(https://support.ccdc.cam.ac.uk/support/solutions/articles/103000306242-how-many-mofs-are-there-in-the-csd-).
2026 is partial (DB snapshot from early 2026), so the series is plotted through
the last complete year, 2025.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from blog_figure_style import (
    PURPLE,
    PURPLE_STRONG,
    MUTED,
    use_blog_style,
    style_axis,
    save_svg_png,
    label_box,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "img" / "blog"

# (year, cumulative MOFs in CSD MOF subset, MOFs deposited that year).
# Counted locally from MOF_subset.gcd (137,167 refcodes, CSD 2026.1) by
# deposition year; cumulative starts from 444 entries deposited before 1981.
CSD_MOF_COUNTS = [
    (1981, 474, 30), (1982, 545, 71), (1983, 617, 72), (1984, 671, 54),
    (1985, 779, 108), (1986, 862, 83), (1987, 990, 128), (1988, 1109, 119),
    (1989, 1286, 177), (1990, 1434, 148), (1991, 1648, 214), (1992, 1869, 221),
    (1993, 2097, 228), (1994, 2350, 253), (1995, 2605, 255), (1996, 3028, 423),
    (1997, 3449, 421), (1998, 3947, 498), (1999, 4580, 633), (2000, 5338, 758),
    (2001, 6636, 1298), (2002, 8055, 1419), (2003, 9775, 1720), (2004, 12002, 2227),
    (2005, 14825, 2823), (2006, 17993, 3168), (2007, 21450, 3457), (2008, 25152, 3702),
    (2009, 29036, 3884), (2010, 34134, 5098), (2011, 41712, 7578), (2012, 47500, 5788),
    (2013, 56919, 9419), (2014, 63657, 6738), (2015, 71773, 8116), (2016, 78838, 7065),
    (2017, 86179, 7341), (2018, 92936, 6757), (2019, 101051, 8115), (2020, 107755, 6704),
    (2021, 113681, 5926), (2022, 119552, 5871), (2023, 125287, 5735), (2024, 131304, 6017),
    (2025, 136836, 5532),
]


def csd_mof_growth(output_path: Path) -> None:
    years = [r[0] for r in CSD_MOF_COUNTS]
    cumulative = [r[1] for r in CSD_MOF_COUNTS]
    annual = [r[2] for r in CSD_MOF_COUNTS]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Annual additions as faint bars (context), cumulative as the headline curve.
    ax.bar(years, annual, width=0.78, color=PURPLE, alpha=0.18, zorder=1,
           label="Added that year")
    ax.fill_between(years, cumulative, color=PURPLE, alpha=0.12, zorder=2)
    ax.plot(years, cumulative, color=PURPLE_STRONG, lw=2.6, zorder=3,
            label="Cumulative total")

    # End-point marker + label (end of the continuous annual series).
    ax.scatter([years[-1]], [cumulative[-1]], s=46, color=PURPLE_STRONG,
               edgecolors="white", linewidths=1.4, zorder=5)
    ax.annotate(
        f"{cumulative[-1]:,}\nby 2025",
        xy=(years[-1], cumulative[-1]),
        xytext=(years[-1] - 1.2, cumulative[-1] - 28000),
        ha="right", va="center", fontsize=9.5, color=PURPLE_STRONG,
        fontweight="semibold", bbox=label_box(alpha=0.9, pad=2.0),
    )

    ax.set_ylim(0, 150000)
    ax.set_xlim(1980, 2026)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v else "0"))
    style_axis(
        ax,
        xlabel="Year",
        ylabel="MOF structures in the CSD",
        title="MOFs deposited in the Cambridge Structural Database",
        grid=True,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=MUTED)

    save_svg_png(fig, output_path)


def main() -> None:
    csd_mof_growth(ASSETS / "madfield_csd_mofs.svg")


if __name__ == "__main__":
    main()
