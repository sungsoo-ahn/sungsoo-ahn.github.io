"""Generate all matplotlib figures for Protein AI lecture notes.

Produces figures for Lectures 0-9 and saves them to:
  assets/img/teaching/protein-ai/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "img" / "teaching" / "protein-ai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ===========================================================================
# Helper: generate realistic protein-like Cα coordinates
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
# LECTURE 0 FIGURES
# ===========================================================================
print("Generating Lecture 0 figures...")
coords76 = make_protein_coords(76)
D = dist_matrix(coords76)

# 0-1: Contact map heatmap
fig, ax = plt.subplots(figsize=(6,5), dpi=150, facecolor='white')
contacts = (D < 8.0).astype(float)
im = ax.imshow(contacts, cmap='Blues', origin='upper', aspect='equal')
ax.set_xlabel('Residue index $j$', fontsize=12)
ax.set_ylabel('Residue index $i$', fontsize=12)
ax.set_title(r'Protein Contact Map (8 $\AA$ threshold)', fontsize=13, fontweight='bold')
cb = fig.colorbar(im, ax=ax, shrink=0.82); cb.set_ticks([0,1]); cb.set_ticklabels(['No contact','Contact'])
fig.tight_layout(); fig.savefig(OUT_DIR/'contact_map_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved contact_map_heatmap.png")

# 0-4: Distance matrix visualization
fig, ax = plt.subplots(figsize=(6,5), dpi=150, facecolor='white')
im = ax.imshow(D, cmap='viridis', origin='upper', aspect='equal')
ax.set_xlabel('Residue index $j$', fontsize=12)
ax.set_ylabel('Residue index $i$', fontsize=12)
ax.set_title(r'C$\alpha$ Distance Matrix', fontsize=13, fontweight='bold')
cb = fig.colorbar(im, ax=ax, shrink=0.82); cb.set_label(r'Distance ($\AA$)', fontsize=11)
fig.tight_layout(); fig.savefig(OUT_DIR/'distance_matrix_vis.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved distance_matrix_vis.png")

# ===========================================================================
# LECTURE 1 FIGURES
# ===========================================================================
print("Generating Lecture 1 figures...")

# 1-1: One-hot vs BLOSUM encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
onehot = np.eye(20)

# BLOSUM62 (upper triangle, standard order ACDEFGHIKLMNPQRSTVWY)
blosum62 = np.array([
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2],
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3],
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2,-1, 0,-1,-2,-3,-2],
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3],
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3],
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2],
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1],
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2],
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1],
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1],
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2],
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3],
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1],
    [-1,-3,-2,-1,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2],
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2],
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2],
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1],
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2],
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7],
])

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, facecolor='white')
im0 = axes[0].imshow(onehot, cmap='Blues', aspect='equal')
axes[0].set_title('One-Hot Encoding', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Amino Acid Index'); axes[0].set_ylabel('Amino Acid Index')
axes[0].set_xticks(range(0,20,2)); axes[0].set_yticks(range(0,20,2))
axes[0].set_xticklabels([AMINO_ACIDS[i] for i in range(0,20,2)], fontsize=8)
axes[0].set_yticklabels([AMINO_ACIDS[i] for i in range(0,20,2)], fontsize=8)

im1 = axes[1].imshow(blosum62, cmap='RdBu_r', aspect='equal', vmin=-5, vmax=11)
axes[1].set_title('BLOSUM62 Substitution Matrix', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Amino Acid Index'); axes[1].set_ylabel('Amino Acid Index')
axes[1].set_xticks(range(0,20,2)); axes[1].set_yticks(range(0,20,2))
axes[1].set_xticklabels([AMINO_ACIDS[i] for i in range(0,20,2)], fontsize=8)
axes[1].set_yticklabels([AMINO_ACIDS[i] for i in range(0,20,2)], fontsize=8)
fig.colorbar(im1, ax=axes[1], shrink=0.82, label='Substitution Score')

fig.tight_layout(); fig.savefig(OUT_DIR/'onehot_vs_blosum.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved onehot_vs_blosum.png")

# 1-2: Ramachandran plot
n_residues = 500
# Alpha helix cluster
phi_h = np.random.normal(-60, 12, n_residues//3)
psi_h = np.random.normal(-45, 12, n_residues//3)
# Beta sheet cluster
phi_b = np.random.normal(-120, 15, n_residues//3)
psi_b = np.random.normal(130, 15, n_residues//3)
# Left-handed helix (small cluster)
phi_l = np.random.normal(60, 15, n_residues//6)
psi_l = np.random.normal(45, 15, n_residues//6)
# Random coil scatter
phi_c = np.random.uniform(-180, 180, n_residues//6)
psi_c = np.random.uniform(-180, 180, n_residues//6)

fig, ax = plt.subplots(figsize=(6, 5), dpi=150, facecolor='white')
ax.scatter(phi_h, psi_h, s=8, alpha=0.5, c='#e74c3c', label=r'$\alpha$-helix')
ax.scatter(phi_b, psi_b, s=8, alpha=0.5, c='#3498db', label=r'$\beta$-sheet')
ax.scatter(phi_l, psi_l, s=8, alpha=0.5, c='#2ecc71', label='Left-handed helix')
ax.scatter(phi_c, psi_c, s=5, alpha=0.2, c='#95a5a6', label='Coil')
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
ax.set_ylabel(r'$\psi$ (degrees)', fontsize=12)
ax.set_title('Ramachandran Plot', fontsize=13, fontweight='bold')
ax.set_xticks([-180,-120,-60,0,60,120,180]); ax.set_yticks([-180,-120,-60,0,60,120,180])
ax.axhline(0, color='gray', lw=0.5, alpha=0.3); ax.axvline(0, color='gray', lw=0.5, alpha=0.3)
ax.legend(fontsize=9, markerscale=2)
ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(OUT_DIR/'ramachandran_plot.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved ramachandran_plot.png")

# ===========================================================================
# LECTURE 2 FIGURES
# ===========================================================================
print("Generating Lecture 2 figures...")

# 2-3: Gradient descent on 2D loss landscape
from matplotlib.colors import LogNorm

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
# Rosenbrock-like surface
Z = (1-X)**2 + 5*(Y-X**2)**2 + 0.5

fig, ax = plt.subplots(figsize=(6, 5), dpi=150, facecolor='white')
ax.contour(X, Y, Z, levels=np.logspace(-0.5, 3, 30), cmap='viridis', alpha=0.7)
ax.contourf(X, Y, Z, levels=np.logspace(-0.5, 3, 30), cmap='viridis', alpha=0.3)

# Simulate gradient descent trajectory
lr = 0.002
pos = np.array([-2.5, 2.5])
trajectory = [pos.copy()]
for _ in range(500):
    gx = -2*(1-pos[0]) + 5*2*(pos[1]-pos[0]**2)*(-2*pos[0])
    gy = 5*2*(pos[1]-pos[0]**2)
    pos = pos - lr * np.array([gx, gy])
    trajectory.append(pos.copy())
    if np.linalg.norm([gx, gy]) < 0.01: break
traj = np.array(trajectory)

ax.plot(traj[:,0], traj[:,1], 'r.-', markersize=3, linewidth=1, alpha=0.8, label='Gradient descent path')
ax.plot(traj[0,0], traj[0,1], 'ro', markersize=8, label='Start')
ax.plot(1, 1, 'r*', markersize=15, label='Minimum')
ax.set_xlabel('$w_1$', fontsize=12); ax.set_ylabel('$w_2$', fontsize=12)
ax.set_title('Gradient Descent on Loss Landscape', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT_DIR/'gradient_descent.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved gradient_descent.png")

# ===========================================================================
# LECTURE 3 FIGURES
# ===========================================================================
print("Generating Lecture 3 figures...")

# 3-1: Overfitting illustration (train vs val loss)
epochs = np.arange(1, 101)
train_loss = 2.0 * np.exp(-0.05*epochs) + 0.05 + 0.02*np.random.randn(100)
val_loss_good = 2.0 * np.exp(-0.04*epochs) + 0.15 + 0.03*np.random.randn(100)
# Make val loss increase after epoch 40
val_loss = val_loss_good.copy()
val_loss[40:] = val_loss[40] + 0.008*(epochs[40:]-40) + 0.03*np.random.randn(60)

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor='white')
ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training loss', alpha=0.8)
ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation loss', alpha=0.8)
ax.axvline(40, color='gray', linestyle='--', alpha=0.5)
ax.annotate('Best model\n(early stopping)', xy=(40, val_loss[39]), xytext=(55, 0.8),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
            ha='center', color='gray')
ax.fill_betweenx([0, 2.5], 40, 100, alpha=0.05, color='red')
ax.text(70, 1.8, 'Overfitting\nregion', fontsize=11, color='red', alpha=0.6, ha='center')
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training vs Validation Loss (Overfitting)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.set_ylim(0, 2.5)
fig.tight_layout(); fig.savefig(OUT_DIR/'overfitting_curves.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved overfitting_curves.png")

# 3-3: Learning rate schedules
steps = np.arange(0, 1000)
base_lr = 1e-3

# Step decay
step_lr = np.where(steps < 300, base_lr, np.where(steps < 600, base_lr*0.1, base_lr*0.01))

# Cosine annealing
cosine_lr = 0.5 * base_lr * (1 + np.cos(np.pi * steps / 1000))

# Warmup + cosine decay
warmup_steps = 100
warmup_cosine_lr = np.where(steps < warmup_steps,
    base_lr * steps / warmup_steps,
    0.5 * base_lr * (1 + np.cos(np.pi * (steps - warmup_steps) / (1000 - warmup_steps))))

fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor='white')
ax.plot(steps, step_lr, '-', linewidth=2, label='Step Decay', alpha=0.8)
ax.plot(steps, cosine_lr, '-', linewidth=2, label='Cosine Annealing', alpha=0.8)
ax.plot(steps, warmup_cosine_lr, '-', linewidth=2, label='Warmup + Cosine', alpha=0.8)
ax.set_xlabel('Training Step', fontsize=12); ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Learning Rate Schedules', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.set_yscale('log')
ax.set_ylim(1e-6, 2e-3)
fig.tight_layout(); fig.savefig(OUT_DIR/'lr_schedules.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved lr_schedules.png")

# ===========================================================================
# LECTURE 5 FIGURES
# ===========================================================================
print("Generating Lecture 5 figures...")

# 5-2: VAE latent space visualization
n_clusters = 5
points_per_cluster = 60
colors_ae = []
colors_vae = []
ae_points = []
vae_points = []

cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i in range(n_clusters):
    angle = 2*np.pi*i/n_clusters
    # AE: scattered, no structure
    ae_pts = np.random.randn(points_per_cluster, 2) * 2 + np.array([np.cos(angle)*4, np.sin(angle)*4]) * np.random.rand()
    ae_points.append(ae_pts)
    # VAE: well-clustered
    vae_pts = np.random.randn(points_per_cluster, 2) * 0.4 + np.array([np.cos(angle)*2.5, np.sin(angle)*2.5])
    vae_points.append(vae_pts)
    colors_ae.extend([cluster_colors[i]] * points_per_cluster)
    colors_vae.extend([cluster_colors[i]] * points_per_cluster)

ae_all = np.concatenate(ae_points)
vae_all = np.concatenate(vae_points)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150, facecolor='white')
axes[0].scatter(ae_all[:,0], ae_all[:,1], c=colors_ae, s=15, alpha=0.6)
axes[0].set_title('Autoencoder Latent Space', fontsize=13, fontweight='bold')
axes[0].set_xlabel('$z_1$'); axes[0].set_ylabel('$z_2$')
axes[0].set_xlim(-6,6); axes[0].set_ylim(-6,6)

axes[1].scatter(vae_all[:,0], vae_all[:,1], c=colors_vae, s=15, alpha=0.6)
axes[1].set_title('VAE Latent Space', fontsize=13, fontweight='bold')
axes[1].set_xlabel('$z_1$'); axes[1].set_ylabel('$z_2$')
axes[1].set_xlim(-6,6); axes[1].set_ylim(-6,6)

# Add legend
for i, label in enumerate(['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']):
    axes[1].scatter([], [], c=cluster_colors[i], label=label, s=30)
axes[1].legend(fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(OUT_DIR/'vae_latent_space.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved vae_latent_space.png")

# 5-3: Diffusion forward process (progressive noise on 2D protein shape)
np.random.seed(123)
t_vals = np.linspace(0, 2*np.pi, 50)
# Heart-like shape as a simple 2D "protein"
clean_x = 16 * np.sin(t_vals)**3
clean_y = 13*np.cos(t_vals) - 5*np.cos(2*t_vals) - 2*np.cos(3*t_vals) - np.cos(4*t_vals)
clean_x /= 8; clean_y /= 8

timesteps = [0, 0.2, 0.5, 0.8, 1.0]
fig, axes = plt.subplots(1, 5, figsize=(14, 3), dpi=150, facecolor='white')
for idx, t in enumerate(timesteps):
    noise = np.random.randn(50, 2) * t * 1.5
    x_noisy = clean_x + noise[:,0]
    y_noisy = clean_y + noise[:,1]
    axes[idx].scatter(x_noisy, y_noisy, c='steelblue', s=20, alpha=0.7)
    axes[idx].set_xlim(-5, 5); axes[idx].set_ylim(-5, 5)
    axes[idx].set_aspect('equal')
    axes[idx].set_title(f'$t = {t:.1f}$', fontsize=12, fontweight='bold')
    axes[idx].set_xticks([]); axes[idx].set_yticks([])
    if idx == 0:
        axes[idx].set_xlabel(r'$\mathbf{x}_0$ (clean)', fontsize=10)
    elif idx == len(timesteps)-1:
        axes[idx].set_xlabel(r'$\mathbf{x}_T$ (noise)', fontsize=10)

fig.suptitle('Diffusion Forward Process: Progressive Noise Addition', fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout(); fig.savefig(OUT_DIR/'diffusion_forward.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved diffusion_forward.png")

# 5-4: Diffusion noise schedule
T = 1000
t = np.arange(T)
# Linear schedule
beta = np.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)
sqrt_alpha_bar = np.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)

fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor='white')
ax.plot(t, beta, '-', linewidth=2, label=r'$\beta_t$ (noise variance)', alpha=0.8)
ax.plot(t, sqrt_alpha_bar, '-', linewidth=2, label=r'$\sqrt{\bar{\alpha}_t}$ (signal)', alpha=0.8)
ax.plot(t, sqrt_one_minus_alpha_bar, '-', linewidth=2, label=r'$\sqrt{1-\bar{\alpha}_t}$ (noise)', alpha=0.8)
ax.set_xlabel('Timestep $t$', fontsize=12); ax.set_ylabel('Value', fontsize=12)
ax.set_title('Diffusion Noise Schedule', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
fig.tight_layout(); fig.savefig(OUT_DIR/'diffusion_noise_schedule.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved diffusion_noise_schedule.png")

# ===========================================================================
# LECTURE 6 FIGURES
# ===========================================================================
print("Generating Lecture 6 figures...")

# 6-2: ESM-2 model sizes bar chart
model_names = ['ESM-2\n8M', 'ESM-2\n35M', 'ESM-2\n150M', 'ESM-2\n650M', 'ESM-2\n3B', 'ESM-2\n15B']
params = [8, 35, 150, 650, 3000, 15000]  # in millions
# Simulated contact prediction accuracy
contact_acc = [0.32, 0.45, 0.58, 0.68, 0.73, 0.76]

fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor='white')
bars = ax1.bar(range(len(model_names)), params, color='steelblue', alpha=0.7, label='Parameters (M)')
ax1.set_yscale('log')
ax1.set_ylabel('Parameters (millions)', fontsize=12, color='steelblue')
ax1.set_xlabel('Model', fontsize=12)
ax1.set_xticks(range(len(model_names))); ax1.set_xticklabels(model_names, fontsize=9)
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(range(len(model_names)), contact_acc, 'ro-', linewidth=2, markersize=8, label='Contact Precision')
ax2.set_ylabel('Long-range Contact Precision (L/5)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0.2, 0.85)

ax1.set_title('ESM-2 Model Family: Scale vs Performance', fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(OUT_DIR/'esm2_model_sizes.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved esm2_model_sizes.png")

# ===========================================================================
# LECTURE 8 FIGURES
# ===========================================================================
print("Generating Lecture 8 figures...")

# 8-3: IGSO(3) distribution
angles = np.linspace(0, np.pi, 200)

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150, facecolor='white')
sigmas = [0.1, 0.3, 0.5, 1.0, 2.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sigmas)))
for sigma, color in zip(sigmas, colors):
    # Approximate IGSO(3) density: (1 - cos(angle)) * exp(-angle^2 / (2*sigma^2))
    density = (1 - np.cos(angles)) * np.exp(-angles**2 / (2*sigma**2))
    density /= np.sum(density) * (angles[1] - angles[0])  # normalize
    ax.plot(angles * 180/np.pi, density, '-', linewidth=2, color=color, label=f'$\\sigma = {sigma}$')

ax.set_xlabel('Rotation angle (degrees)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('IGSO(3) Distribution for Different Noise Levels', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 180)
fig.tight_layout(); fig.savefig(OUT_DIR/'igso3_distribution.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(f"  Saved igso3_distribution.png")

print("\nAll matplotlib figures generated successfully!")
print(f"Output directory: {OUT_DIR}")
