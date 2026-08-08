"""Generate original figures for the chemistry/physics foundations blog post.

The four figures are conceptual syntheses drawn from standard equations, not
reproductions of the source lecture slides or third-party artwork.  Curves use
dimensionless toy parameters chosen for clarity; they are not measured data.
The editable SVG files are the publication assets and the PNG files are QA
previews.  Design and implementation: Sungsoo Ahn / OpenAI, 2026.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import blog_figure_style as bfs


OUTPUT_DIR = ROOT / "assets" / "img" / "blog"
bfs.use_blog_style()


def representation_tradeoff() -> None:
    """Show how representations retain progressively richer structure."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = ["String", "2D graph", "3D coordinates", "Periodic crystal"]
    x = np.arange(4)
    information = np.array([0.22, 0.45, 0.72, 0.94])
    cost = np.array([0.12, 0.28, 0.61, 0.88])

    ax.plot(x, information, color=bfs.PURPLE, marker="o", ms=7)
    ax.plot(x, cost, color=bfs.AMBER, marker="o", ms=7)
    ax.fill_between(x, cost, information, color=bfs.PURPLE_LIGHT, alpha=0.9)
    for i, label in enumerate(labels):
        ax.text(i, -0.07, label, ha="center", va="top", color=bfs.TEXT, fontsize=10)

    ax.text(2.98, information[-1] + 0.075, "geometry retained", color=bfs.PURPLE,
            ha="right", fontweight="semibold", bbox=bfs.label_box())
    ax.text(2.35, 0.62, "modeling burden", color=bfs.AMBER,
            ha="center", fontweight="semibold", bbox=bfs.label_box())
    ax.text(0.04, 0.73, "No representation is universally best:\nretain the structure the target depends on.",
            transform=ax.transAxes, ha="left", va="top", color=bfs.MUTED, fontsize=10.5)
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.12, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Relative amount")
    bfs.clean_axes(ax)
    bfs.save_svg_png(fig, OUTPUT_DIR / "cpml_representation_tradeoff.svg")


def energy_and_force() -> None:
    """Connect an electronic energy curve to nuclear force."""
    r = np.linspace(0.72, 3.4, 500)
    depth, width, equilibrium = 1.0, 1.85, 1.35
    energy = depth * (1 - np.exp(-width * (r - equilibrium))) ** 2 - depth
    r0 = 1.86
    e0 = depth * (1 - np.exp(-width * (r0 - equilibrium))) ** 2 - depth
    derivative = 2 * depth * width * np.exp(-width * (r0 - equilibrium)) * (
        1 - np.exp(-width * (r0 - equilibrium))
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(r, energy, color=bfs.PURPLE, lw=2.7)
    ax.axvline(equilibrium, color=bfs.SPINE, lw=1.0, ls="--")
    tangent_x = np.array([r0 - 0.32, r0 + 0.32])
    ax.plot(tangent_x, e0 + derivative * (tangent_x - r0), color=bfs.AMBER, lw=2.0)
    ax.scatter([r0], [e0], s=58, color=bfs.AMBER, edgecolor="white", lw=1.2, zorder=5)
    ax.annotate(r"force $F=-dE/dr$", xy=(r0, e0), xytext=(2.42, -0.26),
                color=bfs.AMBER, fontweight="semibold",
                arrowprops={"arrowstyle": "-|>", "color": bfs.AMBER, "lw": 1.3})
    ax.text(equilibrium, -1.08, "equilibrium", ha="center", va="top", color=bfs.MUTED)
    ax.text(2.7, -0.93, "bond dissociation", ha="center", color=bfs.MUTED)
    bfs.style_axis(ax, xlabel="Internuclear distance  r", ylabel="Electronic energy  E(r)")
    ax.set_xlim(0.72, 3.4)
    ax.set_ylim(-1.15, 0.55)
    ax.set_yticks([])
    bfs.save_svg_png(fig, OUTPUT_DIR / "cpml_energy_force.svg")


def boltzmann_landscape() -> None:
    """Show how one energy landscape produces temperature-dependent populations."""
    q = np.linspace(-2.7, 2.7, 800)
    energy = 0.22 * (q**2 - 2.2) ** 2 + 0.10 * q
    temperatures = [(0.22, bfs.PURPLE, "low temperature"),
                    (0.62, bfs.AMBER, "high temperature")]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.12})
    ax0.plot(q, energy, color=bfs.MUTED, lw=2.5)
    ax0.fill_between(q, energy, energy.max() + 0.2, color=bfs.PURPLE_SOFT, alpha=0.75)
    ax0.text(-1.5, 0.18, "basin A", ha="center", color=bfs.PURPLE_STRONG,
             fontweight="semibold")
    ax0.text(1.45, 0.35, "basin B", ha="center", color=bfs.PURPLE_STRONG,
             fontweight="semibold")
    ax0.text(0.0, 1.04, "barrier", ha="center", color=bfs.MUTED)
    bfs.style_axis(ax0, ylabel="Potential energy  U(q)")
    ax0.set_yticks([])
    ax0.spines["bottom"].set_visible(False)
    ax0.tick_params(bottom=False)

    for temp, color, label in temperatures:
        probability = np.exp(-(energy - energy.min()) / temp)
        probability /= np.trapezoid(probability, q)
        ax1.plot(q, probability, color=color, lw=2.5, label=label)
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    bfs.style_axis(ax1, xlabel="Configuration coordinate  q", ylabel="Probability density  p(q)")
    ax1.set_yticks([])
    bfs.save_svg_png(fig, OUTPUT_DIR / "cpml_boltzmann_landscape.svg")


def thermodynamics_and_kinetics() -> None:
    """Separate endpoint free-energy difference from activation barrier."""
    x = np.linspace(0, 1, 500)
    baseline = -0.48 * (3 * x**2 - 2 * x**3)
    low = baseline + 0.62 * np.sin(np.pi * x) ** 2
    high = baseline + 1.16 * np.sin(np.pi * x) ** 2

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(x, low, color=bfs.TEAL, lw=2.6, label="lower barrier pathway")
    ax.plot(x, high, color=bfs.PURPLE, lw=2.6, label="higher barrier pathway")
    ax.scatter([0, 1], [0, -0.48], s=58, color=[bfs.AMBER, bfs.GREEN],
               edgecolor="white", lw=1.2, zorder=5)
    ax.annotate("", xy=(0.965, -0.48), xytext=(0.965, 0.0),
                arrowprops={"arrowstyle": "<->", "color": bfs.GREEN, "lw": 1.4})
    ax.text(0.91, -0.24, "same ΔG", color=bfs.GREEN, fontweight="semibold",
            ha="right", va="center", bbox=bfs.label_box())
    peak_x = x[np.argmax(high)]
    peak_y = high.max()
    ax.annotate("different ΔG‡", xy=(peak_x, peak_y), xytext=(0.72, 1.03),
                color=bfs.PURPLE, fontweight="semibold", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": bfs.PURPLE, "lw": 1.3})
    ax.text(0, -0.09, "reactant", ha="center", va="top", color=bfs.AMBER,
            fontweight="semibold")
    ax.text(1, -0.56, "product", ha="center", va="top", color=bfs.GREEN,
            fontweight="semibold")
    ax.legend(frameon=False, loc="upper left")
    bfs.style_axis(ax, xlabel="Reaction coordinate", ylabel="Free energy")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.68, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    bfs.save_svg_png(fig, OUTPUT_DIR / "cpml_thermodynamics_kinetics.svg")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    representation_tradeoff()
    energy_and_force()
    boltzmann_landscape()
    thermodynamics_and_kinetics()


if __name__ == "__main__":
    main()
