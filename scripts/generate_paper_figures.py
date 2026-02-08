"""Generate redrawn/conceptual figures for Protein AI lecture notes.

These are original figures inspired by concepts from published papers.
Data is approximate/illustrative and clearly labeled as such.
Saves to: assets/img/teaching/protein-ai/

Note: Protein structure figures (pdb_ribbon_example, protein_structure_levels,
proteinmpnn_graph, se3_frame_residue, diffusion_reverse_protein) have been
moved to generate_pymol_figures.py which uses PyMOL renderings of real PDB data.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "img" / "teaching" / "protein-ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ===========================================================================
# Helper functions (shared with generate_all_figures.py)
# ===========================================================================
def helix_segment(n, start, direction):
    rise, radius, rpt = 1.5, 2.3, 3.6
    omega = 2 * np.pi / rpt
    d = direction / np.linalg.norm(direction)
    arb = np.array([1.,0.,0.]) if abs(d[0]) < 0.9 else np.array([0.,1.,0.])
    e1 = np.cross(d, arb); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return np.array([start + d*rise*i + e1*radius*np.cos(omega*i) + e2*radius*np.sin(omega*i) for i in range(n)])

def strand_segment(n, start, direction):
    rise = 3.3
    d = direction / np.linalg.norm(direction)
    arb = np.array([0.,1.,0.]) if abs(d[1]) < 0.9 else np.array([0.,0.,1.])
    perp = np.cross(d, arb); perp /= np.linalg.norm(perp)
    return np.array([start + d*rise*i + perp*0.8*((-1)**i) for i in range(n)])

def loop_segment(n, start, end):
    t = np.linspace(0, 1, n)
    mid = 0.5*(start+end); diff = end-start
    arb = np.array([1.,0.,0.]) if abs(diff[0]) < 0.9*np.linalg.norm(diff) else np.array([0.,1.,0.])
    perp = np.cross(diff, arb)
    if np.linalg.norm(perp) > 1e-6: perp /= np.linalg.norm(perp)
    bulge = perp * np.linalg.norm(diff) * 0.35 * (np.random.rand()-0.3)
    return np.array([(1-ti)**2*start + 2*(1-ti)*ti*(mid+bulge) + ti**2*end for ti in t])

def make_protein_coords(n_res=76):
    segs = []
    c = np.array([0.,0.,0.])
    for kind, n, d in [
        ('s',7,[1.,0.,0.]),('l',5,None),('h',11,[0.2,1.,0.1]),('l',3,None),
        ('s',6,[-1.,0.1,0.]),('l',3,None),('h',6,[-0.3,-1.,0.2]),('l',3,None),
        ('s',5,[1.,-0.2,-0.1]),('l',4,None),('s',8,[-1.,0.3,0.1]),('l',3,None),
        ('h',8,[0.1,0.8,-0.5]),('l',3,None)]:
        if kind == 's':
            seg = strand_segment(n, c, np.array(d)); segs.append(seg); c = seg[-1]
        elif kind == 'h':
            seg = helix_segment(n, c, np.array(d)); segs.append(seg); c = seg[-1]
        else:
            offsets = [[3.,8.,2.],[-5.,4.,-3.],[-2.,-6.,5.],[4.,-5.,-4.],
                       [2.,6.,3.],[3.,3.,-5.],[-4.,2.,4.]]
            off = offsets.pop(0) if offsets else [3.,3.,3.]
            target = c + np.array(off)
            seg = loop_segment(n, c, target); segs.append(seg); c = seg[-1]
    total = sum(len(s) for s in segs)
    if total < n_res:
        seg = strand_segment(n_res-total, c, np.array([-1.,-0.1,0.2]))
        segs.append(seg)
    return np.concatenate(segs, axis=0)[:n_res]

def dist_matrix(coords):
    diff = coords[:,np.newaxis,:] - coords[np.newaxis,:,:]
    return np.sqrt(np.sum(diff**2, axis=-1))


# ===========================================================================
# LECTURE 9 FIGURES (ProteinMPNN)
# ===========================================================================
print("Generating Lecture 9 (ProteinMPNN) figures...")

# --- Figure 1: ProteinMPNN Sequence Recovery Bar Chart ---
def proteinmpnn_recovery():
    """Bar chart comparing sequence recovery rates across methods.
    Data adapted from Dauparas et al., 2022, Science.
    """
    methods = ['Rosetta\nFixBB', 'StructGNN', 'GraphTrans', 'GVP', 'ProteinMPNN\n(no noise)', 'ProteinMPNN']
    recovery = [32.9, 35.9, 36.1, 39.2, 45.7, 52.4]
    colors = ['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor='white')
    bars = ax.bar(range(len(methods)), recovery, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, recovery):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Sequence Recovery Rate (%)', fontsize=12)
    ax.set_title('Inverse Folding: Sequence Recovery Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=52.4, color='#2ecc71', linestyle='--', alpha=0.3, linewidth=1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'proteinmpnn_recovery.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved proteinmpnn_recovery.png")

proteinmpnn_recovery()



# --- Figure 3: Design Pipeline Flowchart ---
def design_pipeline():
    """Computational protein design pipeline: RFDiffusion -> ProteinMPNN -> AlphaFold."""
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=150, facecolor='white')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Box positions and labels
    boxes = [
        (0.5, 0.5, 'Design\nSpecification', '#ecf0f1', '#95a5a6'),
        (2.8, 0.5, 'RFDiffusion', '#fce4ec', '#e91e63'),
        (5.1, 0.5, 'ProteinMPNN', '#e8f5e9', '#4CAF50'),
        (7.4, 0.5, 'AlphaFold2', '#e8f4fd', '#2196F3'),
        (9.7, 0.5, 'Experimental\nValidation', '#fff3e0', '#FF9800'),
    ]

    for x, y, label, facecolor, edgecolor in boxes:
        box = FancyBboxPatch((x - 0.9, y - 0.6), 1.8, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold',
                color='#2c3e50')

    # Arrows between boxes
    arrow_style = "Simple,tail_width=3,head_width=12,head_length=8"
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.95
        x2 = boxes[i+1][0] - 0.95
        y_mid = boxes[i][1]
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=2))

    # Labels below arrows
    arrow_labels = [
        (1.65, -0.3, 'target\nshape'),
        (3.95, -0.3, 'backbone\ncoordinates'),
        (6.25, -0.3, 'candidate\nsequences'),
        (8.55, -0.3, 'TM-score\n> 0.8?'),
    ]
    for x, y, label in arrow_labels:
        ax.text(x, y, label, ha='center', va='top', fontsize=8, color='#7f8c8d', style='italic')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'design_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved design_pipeline.png")

design_pipeline()


# ===========================================================================
# LECTURE 8 FIGURES (RFDiffusion)
# ===========================================================================


# ===========================================================================
# LECTURE 5 FIGURES (Generative Models)
# ===========================================================================
print("Generating Lecture 5 (Generative Models) figures...")

# --- Figure 6: VAE Graphical Model ---
def vae_graphical_model():
    """Graphical model / architecture diagram for VAE."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor='white')
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-1, 3.5)
    ax.axis('off')

    # Draw the graphical model (left side)
    # Nodes
    node_style = dict(fontsize=14, fontweight='bold', ha='center', va='center',
                      bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='#2c3e50', linewidth=2))

    # Generative model (p)
    ax.text(1.0, 2.5, '$z$', **node_style)
    ax.text(1.0, 0.5, '$x$', **{**node_style,
            'bbox': dict(boxstyle='circle,pad=0.3', facecolor='#e8f4fd', edgecolor='#2196F3', linewidth=2)})
    ax.annotate('', xy=(1.0, 0.95), xytext=(1.0, 2.05),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    ax.text(1.4, 1.5, r'$p_\theta(x|z)$', fontsize=11, color='#2196F3')
    ax.text(1.0, 3.2, 'Generative\nModel', fontsize=10, ha='center', fontweight='bold', color='#2c3e50')

    # Inference model (q)
    ax.text(3.5, 2.5, '$z$', **node_style)
    ax.text(3.5, 0.5, '$x$', **{**node_style,
            'bbox': dict(boxstyle='circle,pad=0.3', facecolor='#e8f5e9', edgecolor='#4CAF50', linewidth=2)})
    ax.annotate('', xy=(3.5, 2.05), xytext=(3.5, 0.95),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2, linestyle='dashed'))
    ax.text(3.9, 1.5, r'$q_\phi(z|x)$', fontsize=11, color='#4CAF50')
    ax.text(3.5, 3.2, 'Inference\nModel', fontsize=10, ha='center', fontweight='bold', color='#2c3e50')

    # Architecture diagram (right side)
    # Encoder box
    enc_box = FancyBboxPatch((5.5, 0.0), 1.3, 1.0, boxstyle="round,pad=0.1",
                             facecolor='#e8f5e9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(enc_box)
    ax.text(6.15, 0.5, 'Encoder\n$q_\\phi$', ha='center', va='center', fontsize=9, fontweight='bold')

    # Latent space
    ax.text(7.5, 0.5, '$\\mu, \\sigma$', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3e0', edgecolor='#FF9800', linewidth=1.5))

    # z sample
    ax.text(8.3, 0.5, '$z$', fontsize=14, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='#2c3e50', linewidth=2))

    # Decoder box
    dec_box = FancyBboxPatch((8.8, 0.0), 1.3, 1.0, boxstyle="round,pad=0.1",
                             facecolor='#e8f4fd', edgecolor='#2196F3', linewidth=2)
    ax.add_patch(dec_box)
    ax.text(9.45, 0.5, 'Decoder\n$p_\\theta$', ha='center', va='center', fontsize=9, fontweight='bold')

    # Input/output labels
    ax.text(5.2, 0.5, '$x$', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(10.4, 0.5, '$\\hat{x}$', fontsize=12, ha='center', va='center', fontweight='bold')

    # Arrows
    for (x1, y1, x2, y2) in [(5.4, 0.5, 5.5, 0.5), (6.8, 0.5, 7.1, 0.5),
                               (7.9, 0.5, 8.1, 0.5), (8.5, 0.5, 8.8, 0.5),
                               (10.1, 0.5, 10.3, 0.5)]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # ELBO label
    ax.text(7.5, -0.7, r'$\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$',
            fontsize=10, ha='center', va='center', color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4ec', edgecolor='#e74c3c', linewidth=1))

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'vae_graphical_model.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved vae_graphical_model.png")

vae_graphical_model()


# --- Figure 7: DDPM Forward-Reverse Process ---
def ddpm_forward_reverse():
    """Illustrate the DDPM forward and reverse processes with 2D data."""
    np.random.seed(42)

    # 2D "data distribution" - a circle with some structure
    n_points = 200
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    clean_x = 2 * np.cos(theta) + np.random.randn(n_points) * 0.15
    clean_y = 2 * np.sin(theta) + np.random.randn(n_points) * 0.15

    timesteps_fwd = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), dpi=150, facecolor='white')

    # Forward process (top row)
    for idx, t in enumerate(timesteps_fwd):
        noise = np.random.randn(n_points, 2) * t * 3
        x_noisy = clean_x * (1-t) + noise[:, 0]
        y_noisy = clean_y * (1-t) + noise[:, 1]

        axes[0, idx].scatter(x_noisy, y_noisy, c='#3498db', s=10, alpha=0.6)
        axes[0, idx].set_xlim(-5, 5); axes[0, idx].set_ylim(-5, 5)
        axes[0, idx].set_aspect('equal')
        axes[0, idx].set_xticks([]); axes[0, idx].set_yticks([])
        axes[0, idx].set_title(f'$t = {t:.2f}$', fontsize=10, fontweight='bold')
        if idx == 0:
            axes[0, idx].set_ylabel('Forward\n(add noise)', fontsize=11, fontweight='bold', color='#e74c3c')

    # Reverse process (bottom row) - same but reversed
    for idx, t in enumerate(reversed(timesteps_fwd)):
        noise = np.random.randn(n_points, 2) * t * 3
        x_noisy = clean_x * (1-t) + noise[:, 0]
        y_noisy = clean_y * (1-t) + noise[:, 1]

        axes[1, idx].scatter(x_noisy, y_noisy, c='#2ecc71', s=10, alpha=0.6)
        axes[1, idx].set_xlim(-5, 5); axes[1, idx].set_ylim(-5, 5)
        axes[1, idx].set_aspect('equal')
        axes[1, idx].set_xticks([]); axes[1, idx].set_yticks([])
        t_rev = timesteps_fwd[-(idx+1)]
        axes[1, idx].set_title(f'$t = {1-t_rev:.2f}$', fontsize=10, fontweight='bold')
        if idx == 0:
            axes[1, idx].set_ylabel('Reverse\n(denoise)', fontsize=11, fontweight='bold', color='#2ecc71')

    # Arrows between columns
    for row in range(2):
        for col in range(4):
            x_pos = (col + 0.5) / 5 + 0.04
            y_pos = 0.72 if row == 0 else 0.28
            fig.text(x_pos + 0.14, y_pos, '→',
                     fontsize=16, color='#7f8c8d', fontweight='bold',
                     ha='center', va='center', transform=fig.transFigure)

    fig.suptitle('Denoising Diffusion: Forward and Reverse Processes', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ddpm_forward_reverse.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved ddpm_forward_reverse.png")

ddpm_forward_reverse()


# ===========================================================================
# LECTURE 6 FIGURES (Protein Language Models)
# ===========================================================================
print("Generating Lecture 6 (Protein Language Models) figures...")

# --- Figure 8: Masked Language Modeling for Proteins ---
def mlm_protein_illustration():
    """Illustrate masked language modeling on a protein sequence."""
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=150, facecolor='white')
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1.5, 3)
    ax.axis('off')

    # Original sequence
    sequence = list("MVLSPADKTNVK")
    masked_positions = [2, 5, 9]

    # Draw original sequence
    y_orig = 2.2
    ax.text(-0.3, y_orig, 'Input:', fontsize=10, fontweight='bold', va='center', ha='right', color='#2c3e50')
    for i, aa in enumerate(sequence):
        if i in masked_positions:
            color = '#e74c3c'
            facecolor = '#fce4ec'
            label = '[MASK]'
            fontsize = 8
        else:
            color = '#2c3e50'
            facecolor = '#ecf0f1'
            label = aa
            fontsize = 12
        box = FancyBboxPatch((i * 1.1 + 0.2, y_orig - 0.35), 0.8, 0.7,
                             boxstyle="round,pad=0.05", facecolor=facecolor, edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(i * 1.1 + 0.6, y_orig, label, fontsize=fontsize, fontweight='bold',
                ha='center', va='center', color=color)

    # Transformer block
    trans_box = FancyBboxPatch((2.5, 0.6), 8, 0.6, boxstyle="round,pad=0.1",
                               facecolor='#fff3e0', edgecolor='#FF9800', linewidth=2)
    ax.add_patch(trans_box)
    ax.text(6.5, 0.9, 'Transformer Encoder (ESM-2)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#FF9800')

    # Arrows from input to transformer
    for i in range(len(sequence)):
        ax.annotate('', xy=(i * 1.1 + 0.6, 1.2), xytext=(i * 1.1 + 0.6, y_orig - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#bdc3c7', lw=0.8))

    # Predictions below
    y_pred = -0.5
    ax.text(-0.3, y_pred, 'Predict:', fontsize=10, fontweight='bold', va='center', ha='right', color='#2c3e50')
    predictions = {'L': 0.82, 'A': 0.91, 'N': 0.76}  # mock probabilities
    pred_idx = 0
    for i in range(len(sequence)):
        if i in masked_positions:
            aa = list(predictions.keys())[pred_idx]
            prob = list(predictions.values())[pred_idx]
            pred_idx += 1
            box = FancyBboxPatch((i * 1.1 + 0.2, y_pred - 0.35), 0.8, 0.7,
                                 boxstyle="round,pad=0.05", facecolor='#e8f5e9', edgecolor='#4CAF50', linewidth=1.5)
            ax.add_patch(box)
            ax.text(i * 1.1 + 0.6, y_pred + 0.05, aa, fontsize=12, fontweight='bold',
                    ha='center', va='center', color='#4CAF50')
            ax.text(i * 1.1 + 0.6, y_pred - 0.2, f'p={prob:.2f}', fontsize=7,
                    ha='center', va='center', color='#7f8c8d')
            # Arrow from transformer
            ax.annotate('', xy=(i * 1.1 + 0.6, y_pred + 0.4), xytext=(i * 1.1 + 0.6, 0.6),
                        arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

    fig.suptitle('Masked Language Modeling for Protein Sequences', fontsize=13, fontweight='bold', y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'mlm_protein_illustration.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved mlm_protein_illustration.png")

mlm_protein_illustration()


# --- Figure 9: ESM Contact Prediction (Attention ~ Structure) ---
def esm_contact_prediction():
    """Show that attention maps from protein LMs correlate with structural contacts."""
    np.random.seed(42)
    n = 50

    # Generate a fake contact map
    coords = make_protein_coords(n)
    D = dist_matrix(coords)
    contacts = (D < 8.0).astype(float)
    np.fill_diagonal(contacts, 0)

    # Generate a "predicted" attention map that correlates with contacts
    attention = contacts * (0.5 + 0.5 * np.random.rand(n, n))
    attention += np.random.rand(n, n) * 0.15
    attention = (attention + attention.T) / 2
    np.fill_diagonal(attention, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, facecolor='white')

    # Left: attention map
    im0 = axes[0].imshow(attention, cmap='Reds', aspect='equal', origin='upper')
    axes[0].set_title('ESM-2 Attention Map\n(averaged over heads)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Residue $j$', fontsize=11)
    axes[0].set_ylabel('Residue $i$', fontsize=11)
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label='Attention weight')

    # Right: true contact map
    im1 = axes[1].imshow(contacts, cmap='Blues', aspect='equal', origin='upper')
    axes[1].set_title('True Structural Contacts\n(8 Å threshold)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residue $j$', fontsize=11)
    axes[1].set_ylabel('Residue $i$', fontsize=11)
    cb = fig.colorbar(im1, ax=axes[1], shrink=0.8)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['No contact', 'Contact'])

    fig.suptitle('Protein Language Model Attention Correlates with 3D Contacts',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'esm_contact_prediction.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved esm_contact_prediction.png")

esm_contact_prediction()


# ===========================================================================
# LECTURE 4 FIGURES (Transformers & GNNs)
# ===========================================================================
print("Generating Lecture 4 (Transformers & GNNs) figures...")

# --- Figure 10: Transformer Architecture ---
def transformer_architecture():
    """Simplified Transformer architecture diagram."""
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150, facecolor='white')
    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 13)
    ax.axis('off')

    # Components from bottom to top
    components = [
        (3, 0.5, 'Input\nEmbedding', '#ecf0f1', '#95a5a6'),
        (3, 2.0, 'Positional\nEncoding', '#e8f4fd', '#2196F3'),
        (3, 3.5, 'Multi-Head\nSelf-Attention', '#fff3e0', '#FF9800'),
        (3, 5.0, 'Add & Norm', '#f3e5f5', '#9C27B0'),
        (3, 6.5, 'Feed-Forward\nNetwork', '#e8f5e9', '#4CAF50'),
        (3, 8.0, 'Add & Norm', '#f3e5f5', '#9C27B0'),
        (3, 10.0, 'Linear +\nSoftmax', '#fce4ec', '#e91e63'),
    ]

    for x, y, label, facecolor, edgecolor in components:
        box = FancyBboxPatch((x - 1.5, y - 0.55), 3, 1.1,
                             boxstyle="round,pad=0.1",
                             facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold',
                color='#2c3e50')

    # Arrows
    for i in range(len(components) - 1):
        y1 = components[i][1] + 0.55
        y2 = components[i+1][1] - 0.55
        if i == 5:  # skip to output
            y2 = components[i+1][1] - 0.55
        ax.annotate('', xy=(3, y2), xytext=(3, y1),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    # Nx bracket for the transformer block
    ax.annotate('', xy=(5.5, 3.0), xytext=(5.5, 8.5),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.5))
    ax.text(6.2, 5.75, '×N\nlayers', fontsize=11, fontweight='bold', va='center',
            color='#2c3e50')

    # Residual connection arrows
    # Around attention block
    ax.annotate('', xy=(1.2, 5.0), xytext=(1.2, 3.0),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1, linestyle='dashed',
                                connectionstyle='arc3,rad=-0.5'))
    # Around FFN block
    ax.annotate('', xy=(1.2, 8.0), xytext=(1.2, 6.0),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1, linestyle='dashed',
                                connectionstyle='arc3,rad=-0.5'))

    # Input/output labels
    ax.text(3, -0.5, 'Token Sequence', fontsize=10, ha='center', color='#7f8c8d', style='italic')
    ax.text(3, 10.9, 'Output Probabilities', fontsize=10, ha='center', color='#7f8c8d', style='italic')

    ax.set_title('Transformer Architecture', fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'transformer_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved transformer_architecture.png")

transformer_architecture()


# ===========================================================================
# LECTURE 0 & 1 FIGURES
# ===========================================================================
print("Generating Lecture 0 & 1 figures...")

# --- Figure 12: Amino Acid Properties (for L1) ---
def amino_acid_properties():
    """Grouped chart of amino acid properties by category."""
    categories = {
        'Nonpolar\n(hydrophobic)': (['G', 'A', 'V', 'L', 'I', 'P', 'F', 'M', 'W'], '#e74c3c'),
        'Polar\n(uncharged)': (['S', 'T', 'C', 'Y', 'N', 'Q'], '#3498db'),
        'Positive\ncharge': (['K', 'R', 'H'], '#2ecc71'),
        'Negative\ncharge': (['D', 'E'], '#f39c12'),
    }

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor='white')

    x_pos = 0
    group_centers = []
    all_x = []
    all_colors = []
    all_labels = []

    for group_name, (aas, color) in categories.items():
        positions = []
        for aa in aas:
            all_x.append(x_pos)
            all_colors.append(color)
            all_labels.append(aa)
            positions.append(x_pos)
            x_pos += 1
        group_centers.append((np.mean(positions), group_name, color))
        x_pos += 0.8  # gap between groups

    # Draw bars (height = rough molecular weight, normalized)
    mw_approx = {
        'G': 75, 'A': 89, 'V': 117, 'L': 131, 'I': 131, 'P': 115, 'F': 165, 'M': 149, 'W': 204,
        'S': 105, 'T': 119, 'C': 121, 'Y': 181, 'N': 132, 'Q': 146,
        'K': 146, 'R': 174, 'H': 155,
        'D': 133, 'E': 147
    }

    heights = [mw_approx.get(l, 100) for l in all_labels]
    bars = ax.bar(all_x, heights, color=all_colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel('Molecular Weight (Da)', fontsize=11)

    # Group labels
    for cx, name, color in group_centers:
        ax.text(cx, -25, name, ha='center', va='top', fontsize=9, fontweight='bold',
                color=color, transform=ax.get_xaxis_transform())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('The 20 Standard Amino Acids Grouped by Chemical Properties',
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'amino_acid_properties.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved amino_acid_properties.png")

amino_acid_properties()


print("\nAll paper/concept figures generated successfully!")
print(f"Output directory: {OUT_DIR}")
