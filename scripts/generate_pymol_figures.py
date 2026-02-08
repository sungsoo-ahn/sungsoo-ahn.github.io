"""Generate protein structure figures using PyMOL renderings and real PDB data.

Replaces synthetic matplotlib protein figures with PyMOL ray-traced renderings
of actual PDB structures (1UBQ, 4HHB) for publication-quality appearance.

Saves to: assets/img/teaching/protein-ai/
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "img" / "teaching" / "protein-ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_CACHE = Path(tempfile.gettempdir()) / "pymol_protein_ai"
PDB_CACHE.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ===========================================================================
# Helper functions
# ===========================================================================

def init_pymol():
    """Initialize PyMOL in headless mode."""
    import pymol
    from pymol import cmd
    pymol.finish_launching(["pymol", "-cq"])
    return cmd


def reset_pymol(cmd):
    """Reset PyMOL state for a new figure."""
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)


def fetch_pdb(cmd, pdb_id):
    """Fetch a PDB structure, using cache if available."""
    cached = PDB_CACHE / f"{pdb_id}.cif"
    if cached.exists():
        cmd.load(str(cached), pdb_id)
    else:
        cmd.fetch(pdb_id, path=str(PDB_CACHE))


def parse_calpha_coords(pdb_id, chain="A"):
    """Parse Cα coordinates from a cached PDB/CIF file, returns (N,3) array."""
    # Try mmCIF first, then PDB
    cif_path = PDB_CACHE / f"{pdb_id}.cif"
    pdb_path = PDB_CACHE / f"{pdb_id}.pdb"

    coords = []
    if cif_path.exists():
        with open(cif_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    parts = line.split()
                    # mmCIF ATOM records: check atom name and chain
                    if len(parts) >= 18:
                        atom_name = parts[3]
                        chain_id = parts[6]
                        if atom_name == "CA" and chain_id == chain:
                            x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                            coords.append([x, y, z])
    elif pdb_path.exists():
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No Cα atoms found for {pdb_id} chain {chain}")
    return np.array(coords)


def pca_project_2d(coords):
    """PCA-project 3D coordinates to 2D."""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Use top 2 eigenvectors (largest eigenvalues are last)
    proj = centered @ eigvecs[:, -2:]
    return proj


# ===========================================================================
# Figure 1: PDB Ribbon Example (for L0)
# ===========================================================================

def render_pdb_ribbon(cmd):
    """PyMOL cartoon of 1UBQ (ubiquitin) colored by secondary structure."""
    print("  Rendering pdb_ribbon_example.png ...")
    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")

    cmd.hide("everything")
    cmd.show("cartoon", "1UBQ")
    cmd.remove("solvent")

    # Color by secondary structure
    cmd.color("firebrick", "ss h")    # helices - red
    cmd.color("marine", "ss s")       # strands - blue
    cmd.color("gray70", "ss l+''")    # loops - gray

    # Cartoon styling
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_oval_width", 0.25)
    cmd.set("cartoon_loop_radius", 0.15)

    cmd.orient("1UBQ")
    cmd.zoom("1UBQ", 2)
    cmd.ray(900, 900)

    tmp_path = str(PDB_CACHE / "pdb_ribbon_raw.png")
    cmd.png(tmp_path, dpi=150)
    time.sleep(1)

    # Add legend strip with matplotlib
    from PIL import Image
    img = Image.open(tmp_path)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="white")
    ax.imshow(img)
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="#c0392b", linewidth=4, label=r"$\alpha$-helix"),
        Line2D([0], [0], color="#2980b9", linewidth=4, label=r"$\beta$-strand"),
        Line2D([0], [0], color="#95a5a6", linewidth=4, label="Loop/coil"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#bdc3c7")
    ax.set_title("Ubiquitin (PDB: 1UBQ)\nCartoon Representation", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pdb_ribbon_example.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved pdb_ribbon_example.png")


# ===========================================================================
# Figure 2: Protein Structure Levels (for L1)
# ===========================================================================

def render_protein_structure_levels(cmd):
    """4-panel composite: primary (matplotlib) + secondary/tertiary/quaternary (PyMOL)."""
    print("  Rendering protein_structure_levels.png ...")
    from PIL import Image
    from matplotlib.patches import FancyBboxPatch

    # --- Panel B: Secondary structure (PyMOL) ---
    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")
    cmd.hide("everything")
    cmd.show("cartoon", "1UBQ")
    cmd.remove("solvent")
    cmd.color("firebrick", "ss h")
    cmd.color("marine", "ss s")
    cmd.color("gray70", "ss l+''")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    # Zoom to show helix+sheet region
    cmd.orient("1UBQ")
    cmd.ray(600, 600)
    tmp_sec = str(PDB_CACHE / "structure_secondary.png")
    cmd.png(tmp_sec, dpi=150)
    time.sleep(1)

    # --- Panel C: Tertiary structure (PyMOL, rainbow) ---
    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")
    cmd.hide("everything")
    cmd.show("cartoon", "1UBQ")
    cmd.remove("solvent")
    cmd.spectrum("count", "rainbow", "1UBQ")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.orient("1UBQ")
    cmd.ray(600, 600)
    tmp_tert = str(PDB_CACHE / "structure_tertiary.png")
    cmd.png(tmp_tert, dpi=150)
    time.sleep(1)

    # --- Panel D: Quaternary structure (PyMOL, 4HHB by chain) ---
    reset_pymol(cmd)
    fetch_pdb(cmd, "4HHB")
    cmd.hide("everything")
    cmd.show("cartoon", "4HHB")
    cmd.remove("solvent")
    cmd.color("firebrick", "chain A")
    cmd.color("marine", "chain B")
    cmd.color("forest", "chain C")
    cmd.color("orange", "chain D")
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.orient("4HHB")
    cmd.ray(600, 600)
    tmp_quat = str(PDB_CACHE / "structure_quaternary.png")
    cmd.png(tmp_quat, dpi=150)
    time.sleep(1)

    # --- Composite figure ---
    img_sec = Image.open(tmp_sec)
    img_tert = Image.open(tmp_tert)
    img_quat = Image.open(tmp_quat)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=150, facecolor="white")

    # Panel A: Primary structure (matplotlib sequence boxes)
    ax = axes[0]
    seq = "MQIFVK"
    for i, aa in enumerate(seq):
        box = FancyBboxPatch((i * 0.9 + 0.1, 0.8), 0.7, 0.9,
                             boxstyle="round,pad=0.05",
                             facecolor="#e8f4fd", edgecolor="#2196F3", linewidth=1.5)
        ax.add_patch(box)
        ax.text(i * 0.9 + 0.45, 1.25, aa, fontsize=14, fontweight="bold",
                ha="center", va="center", color="#2196F3")
    ax.set_xlim(-0.2, 5.8)
    ax.set_ylim(0, 2.5)
    ax.set_title("Primary\n(Sequence)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Panel B: Secondary
    axes[1].imshow(img_sec)
    axes[1].set_title("Secondary\n(Local folds)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel C: Tertiary
    axes[2].imshow(img_tert)
    axes[2].set_title("Tertiary\n(3D fold)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel D: Quaternary
    axes[3].imshow(img_quat)
    axes[3].set_title("Quaternary\n(Subunit assembly)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    fig.suptitle("Four Levels of Protein Structure", fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "protein_structure_levels.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved protein_structure_levels.png")


# ===========================================================================
# Figure 3: ProteinMPNN k-NN Graph (for L9)
# ===========================================================================

def render_proteinmpnn_graph(cmd):
    """k-NN graph illustration using real 1UBQ Cα coordinates (PCA-projected)."""
    print("  Rendering proteinmpnn_graph.png ...")

    # Ensure PDB is downloaded
    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")
    cmd.save(str(PDB_CACHE / "1UBQ_for_parse.pdb"), "1UBQ", format="pdb")
    time.sleep(0.5)

    # Parse real Cα coordinates
    coords_3d = parse_calpha_coords("1UBQ", chain="A")
    # Use first 30 residues for clarity
    coords_3d = coords_3d[:30]
    coords = pca_project_2d(coords_3d)
    n_res = len(coords)
    x, y = coords[:, 0], coords[:, 1]

    # Compute pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    k = 5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, facecolor="white")

    # Left: backbone chain
    axes[0].plot(x, y, "o-", color="#3498db", markersize=8, linewidth=1.5, alpha=0.7)
    for i in range(n_res):
        axes[0].annotate(str(i + 1), (x[i], y[i]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="#2c3e50")
    axes[0].set_title("Protein Backbone (chain connectivity)", fontsize=12, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    # Right: k-NN graph (spatial neighbors)
    highlight = 15  # Highlight one residue
    dist_knn = dist.copy()
    np.fill_diagonal(dist_knn, np.inf)
    knn_idx = np.argsort(dist_knn[highlight])[:k]

    # Draw all edges faintly
    for i in range(n_res):
        neighbors = np.argsort(dist_knn[i])[:k]
        for j in neighbors:
            axes[1].plot([x[i], x[j]], [y[i], y[j]], "-", color="#bdc3c7", linewidth=0.5, alpha=0.4)

    # Highlight selected residue's neighbors
    for j in knn_idx:
        axes[1].plot([x[highlight], x[j]], [y[highlight], y[j]], "-",
                     color="#e74c3c", linewidth=2, alpha=0.8)

    axes[1].plot(x, y, "o", color="#3498db", markersize=8, alpha=0.7)
    axes[1].plot(x[highlight], y[highlight], "o", color="#e74c3c", markersize=12, zorder=5)
    axes[1].plot(x[knn_idx], y[knn_idx], "o", color="#e67e22", markersize=10, zorder=4)

    for i in range(n_res):
        axes[1].annotate(str(i + 1), (x[i], y[i]), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color="#2c3e50")

    axes[1].set_title(f"k-NN Graph (k={k}, spatial neighbors)", fontsize=12, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10,
               label=f"Query residue ({highlight + 1})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=10,
               label=f"k={k} nearest neighbors"),
        Line2D([0], [0], color="#bdc3c7", linewidth=1, label="All k-NN edges"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle("From Backbone to Graph: ProteinMPNN's Input Representation",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "proteinmpnn_graph.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved proteinmpnn_graph.png")


# ===========================================================================
# Figure 4: SE(3) Frame on Residue (for L8)
# ===========================================================================

def render_se3_frame_residue(cmd):
    """PyMOL sticks + CGO coordinate frame arrows on a residue."""
    print("  Rendering se3_frame_residue.png ...")
    from pymol import cgo
    from PIL import Image

    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")
    cmd.remove("solvent")

    # Show sticks for residues 25-29
    cmd.hide("everything")
    cmd.show("sticks", "1UBQ and resi 25-29")
    cmd.show("cartoon", "1UBQ and resi 20-35")
    cmd.set("cartoon_transparency", 0.7, "1UBQ")
    cmd.set("stick_radius", 0.15)

    # Color the highlighted residue 27
    cmd.color("gray70", "1UBQ")
    cmd.color("skyblue", "1UBQ and resi 27 and name N")
    cmd.color("red", "1UBQ and resi 27 and name CA")
    cmd.color("green", "1UBQ and resi 27 and name C")
    cmd.color("yellow", "1UBQ and resi 27 and name O")

    # Get atom coordinates for residue 27
    n_coords = cmd.get_coords("1UBQ and resi 27 and name N", 1)
    ca_coords = cmd.get_coords("1UBQ and resi 27 and name CA", 1)
    c_coords = cmd.get_coords("1UBQ and resi 27 and name C", 1)

    if n_coords is not None and ca_coords is not None and c_coords is not None:
        n_pos = n_coords[0]
        ca_pos = ca_coords[0]
        c_pos = c_coords[0]

        # Compute local frame
        e1 = c_pos - ca_pos
        e1 = e1 / np.linalg.norm(e1)
        v = n_pos - ca_pos
        e3 = np.cross(e1, v)
        e3 = e3 / np.linalg.norm(e3)
        e2 = np.cross(e3, e1)

        # Draw CGO arrows for each axis
        arrow_len = 3.0
        arrow_rad = 0.08
        cone_rad = 0.25
        cone_len = 0.6

        axis_colors = [
            ([1.0, 0.2, 0.2], e1, "$e_1$"),   # red
            ([0.2, 0.8, 0.2], e2, "$e_2$"),   # green
            ([0.2, 0.4, 1.0], e3, "$e_3$"),   # blue
        ]

        for color, axis, label in axis_colors:
            start = ca_pos.tolist()
            end = (ca_pos + axis * arrow_len).tolist()
            cone_start = end
            cone_end = (ca_pos + axis * (arrow_len + cone_len)).tolist()

            obj = [
                cgo.CYLINDER,
                *start, *end,
                arrow_rad,
                *color, *color,
                cgo.CONE,
                *cone_start, *cone_end,
                cone_rad, 0.0,
                *color, *color,
                1.0, 0.0,
            ]
            cmd.load_cgo(obj, f"arrow_{label}")

    cmd.zoom("1UBQ and resi 25-29", 5)
    cmd.set("ray_opaque_background", 1)
    cmd.ray(900, 900)

    tmp_path = str(PDB_CACHE / "se3_frame_raw.png")
    cmd.png(tmp_path, dpi=150)
    time.sleep(1)

    # Post-process: add labels
    img = Image.open(tmp_path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor="white")
    ax.imshow(img)
    ax.axis("off")

    # Add text labels for axes and atoms
    ax.set_title("Local Coordinate Frame on a Residue\n(Ubiquitin, Residue 27)",
                 fontsize=12, fontweight="bold")

    # Legend for axes
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", linewidth=3, label="$e_1$ (C$\\alpha$ → C)"),
        Line2D([0], [0], color="#2ecc71", linewidth=3, label="$e_2$ (perpendicular)"),
        Line2D([0], [0], color="#3498db", linewidth=3, label="$e_3$ (normal)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#bdc3c7")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "se3_frame_residue.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved se3_frame_residue.png")


# ===========================================================================
# Figure 5: Diffusion Reverse Process on Protein (for L8)
# ===========================================================================

def render_diffusion_reverse_protein(cmd):
    """Reverse diffusion from noise to protein structure using real 1UBQ Cα coords."""
    print("  Rendering diffusion_reverse_protein.png ...")
    np.random.seed(42)

    # Ensure PDB is available
    reset_pymol(cmd)
    fetch_pdb(cmd, "1UBQ")
    cmd.save(str(PDB_CACHE / "1UBQ_for_parse.pdb"), "1UBQ", format="pdb")
    time.sleep(0.5)

    # Parse real Cα coordinates
    coords_3d = parse_calpha_coords("1UBQ", chain="A")
    coords_3d = coords_3d[:30]  # First 30 residues for clarity
    clean_2d = pca_project_2d(coords_3d)

    timesteps = [1.0, 0.7, 0.4, 0.1, 0.0]
    labels = ["$t=1.0$\n(Pure Noise)", "$t=0.7$", "$t=0.4$", "$t=0.1$", "$t=0.0$\n(Generated Protein)"]
    n_res = len(clean_2d)

    fig, axes = plt.subplots(1, 5, figsize=(14, 3), dpi=150, facecolor="white")

    for idx, (t, label) in enumerate(zip(timesteps, labels)):
        ax = axes[idx]
        noise = np.random.randn(n_res, 2) * t * 12
        coords = clean_2d * (1 - t) + noise * t

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_res))
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=30, alpha=0.8,
                   edgecolors="k", linewidth=0.3, zorder=3)
        if t < 0.8:
            for i in range(n_res - 1):
                ax.plot([coords[i, 0], coords[i + 1, 0]],
                        [coords[i, 1], coords[i + 1, 1]],
                        "-", color=colors[i], linewidth=1.5, alpha=0.6)

        ax.set_title(label, fontsize=10, fontweight="bold")
        lim = max(np.abs(clean_2d).max(), 12) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Reverse Diffusion: From Noise to Protein Structure",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diffusion_reverse_protein.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved diffusion_reverse_protein.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Initializing PyMOL...")
    cmd = init_pymol()

    print("\nGenerating protein structure figures with PyMOL...\n")

    render_pdb_ribbon(cmd)
    render_protein_structure_levels(cmd)
    render_proteinmpnn_graph(cmd)
    render_se3_frame_residue(cmd)
    render_diffusion_reverse_protein(cmd)

    cmd.quit()

    print("\nAll PyMOL-based figures generated successfully!")
    print(f"Output directory: {OUT_DIR}")
