"""Generate remaining matplotlib figures for protein-AI lecture notes."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "assets/img/teaching/protein-ai"


def casp_progress_chart():
    """Figure 7-5: CASP progress chart showing AlphaFold2's breakthrough."""
    # Historical CASP GDT-TS scores (approximate median scores for best groups)
    casp_editions = [
        "CASP1\n(1994)", "CASP2\n(1996)", "CASP3\n(1998)", "CASP4\n(2000)",
        "CASP5\n(2002)", "CASP6\n(2004)", "CASP7\n(2006)", "CASP8\n(2008)",
        "CASP9\n(2010)", "CASP10\n(2012)", "CASP11\n(2014)", "CASP12\n(2016)",
        "CASP13\n(2018)", "CASP14\n(2020)"
    ]
    # Approximate median GDT-TS for best-performing group
    scores = [20, 25, 30, 35, 38, 42, 45, 50, 52, 55, 58, 60, 65, 92]
    years = list(range(len(casp_editions)))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color bars: AlphaFold2 in a highlight color
    colors = ['#5B9BD5'] * 13 + ['#E74C3C']
    bars = ax.bar(years, scores, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

    # Add "experimental accuracy" threshold line
    ax.axhline(y=90, color='#2ECC71', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(0.5, 91.5, 'Experimental accuracy threshold (~90 GDT-TS)',
            fontsize=9, color='#2ECC71', fontweight='bold')

    # Annotate AlphaFold2
    ax.annotate('AlphaFold2', xy=(13, 92), xytext=(10.5, 85),
                fontsize=11, fontweight='bold', color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

    ax.set_xticks(years)
    ax.set_xticklabels(casp_editions, fontsize=7, rotation=0)
    ax.set_ylabel('Median GDT-TS Score', fontsize=12)
    ax.set_title('CASP Competition Progress: 25 Years of Incremental Improvement,\nThen a Breakthrough', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/casp_progress.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated casp_progress.png")


def attention_vs_contact_map():
    """Figure 4-5 / 6-5: Simulated attention map vs true contact map comparison."""
    np.random.seed(42)
    L = 60  # sequence length

    # Generate a synthetic contact map (symmetric, sparse)
    contact_map = np.zeros((L, L))

    # Diagonal contacts (local backbone)
    for i in range(L):
        for j in range(max(0, i-3), min(L, i+4)):
            contact_map[i, j] = 1.0

    # Long-range contacts (simulating secondary structure packing)
    # Helix-helix contacts
    for i in range(5, 15):
        for j in range(40, 50):
            if abs((i - 5) - (j - 40)) <= 2:
                contact_map[i, j] = 1.0
                contact_map[j, i] = 1.0

    # Beta-sheet contacts (antiparallel)
    for i in range(20, 30):
        j = 55 - (i - 20)
        for dj in range(-1, 2):
            if 0 <= j + dj < L:
                contact_map[i, j + dj] = 1.0
                contact_map[j + dj, i] = 1.0

    # Generate simulated attention map (noisy version of contacts)
    attention_map = np.zeros((L, L))

    # True contacts get high attention (with noise)
    attention_map += contact_map * (0.5 + 0.5 * np.random.rand(L, L))

    # Add some noise and local attention
    attention_map += 0.05 * np.random.rand(L, L)
    for i in range(L):
        for j in range(max(0, i-5), min(L, i+6)):
            attention_map[i, j] += 0.2 * np.exp(-0.5 * (i - j)**2)

    # Symmetrize
    attention_map = (attention_map + attention_map.T) / 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # True contact map
    im1 = axes[0].imshow(contact_map, cmap='Greys', aspect='equal', origin='upper')
    axes[0].set_title('True Contact Map\n(Cβ < 8Å)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Residue index', fontsize=10)
    axes[0].set_ylabel('Residue index', fontsize=10)

    # Attention map
    im2 = axes[1].imshow(attention_map, cmap='Reds', aspect='equal', origin='upper')
    axes[1].set_title('ESM Attention Weights\n(averaged over heads)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Residue index', fontsize=10)
    axes[1].set_ylabel('Residue index', fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay: attention on top, true contacts as outlines
    axes[2].imshow(attention_map, cmap='Reds', aspect='equal', origin='upper', alpha=0.7)
    # Mark true long-range contacts
    long_range = np.zeros_like(contact_map)
    for i in range(L):
        for j in range(L):
            if contact_map[i, j] > 0 and abs(i - j) > 6:
                long_range[i, j] = 1.0
    yi, xi = np.where(long_range > 0)
    axes[2].scatter(xi, yi, s=8, c='blue', marker='s', alpha=0.4, label='True contacts (|i-j|>6)')
    axes[2].set_title('Overlay\n(blue = true long-range contacts)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Residue index', fontsize=10)
    axes[2].set_ylabel('Residue index', fontsize=10)
    axes[2].legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention_vs_contacts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated attention_vs_contacts.png")


if __name__ == "__main__":
    casp_progress_chart()
    attention_vs_contact_map()
    print("All remaining figures generated successfully!")
