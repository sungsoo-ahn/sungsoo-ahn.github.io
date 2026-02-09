# Protein & AI Course (Spring 2026) — Synthesis Report

Generated: 2026-02-10

## Summary

10 notes processed (4 preliminary + 6 lectures). All notes are now consistent in cross-references, section numbering, and notation.

**Total changes made:** 5 edits across 4 files.

---

## Per-Note Reports

### Note P1: Introduction to Machine Learning with Linear Regression

**Status:** processed
**Changes made:** none
**Concepts introduced:** 14
**Notation symbols introduced:** 9

---

### Note P2: Protein Representations and Neural Network Architectures

**Status:** processed
**Changes made:**
- Line 286: Fixed cross-reference "Lecture 6" to "Lecture 3" for protein language models

**Concepts introduced:** 22
**Notation symbols introduced:** 10

---

### Note P3: Training Neural Networks for Protein Science

**Status:** processed
**Changes made:** none
**Concepts introduced:** 14
**Notation symbols introduced:** 4

---

### Note P4: Case Study: Predicting Protein Solubility

**Status:** processed
**Changes made:** none
**Concepts introduced:** 6
**Notation symbols introduced:** 0

---

### Note L1: Transformers & Graph Neural Networks for Proteins

**Status:** processed
**Changes made:** none
**Concepts introduced:** 16
**Notation symbols introduced:** 11

---

### Note L2: Generative Models: VAEs and Diffusion for Proteins

**Status:** processed
**Changes made:** none
**Concepts introduced:** 11
**Notation symbols introduced:** 8

---

### Note L3: Protein Language Models

**Status:** processed
**Changes made:**
- Added section numbers (1-14) to all content sections — was the only lecture without `## N. Title` format
- Line 328: Removed false cross-reference "(introduced in Preliminary Note 1)" for Swish activation — Swish is not defined in any prior note; the footnote provides an inline definition

**Concepts introduced:** 18
**Notation symbols introduced:** 0

---

### Note L4: AlphaFold: Protein Structure Prediction

**Status:** processed
**Changes made:**
- Line 18: Fixed cross-reference "Lecture 2" to "Lecture 1" for transformers and attention mechanisms
- Line 125: Renumbered duplicate subsection `### 2.2` to `### 2.3`; cascaded `### 2.3` to `### 2.4`

**Concepts introduced:** 10
**Notation symbols introduced:** 0

---

### Note L5: RFDiffusion: De Novo Protein Structure Generation

**Status:** processed
**Changes made:**
- Fixed roadmap table: added missing "Key Takeaways" row (section 13) and updated "Exercises" to section 14

**Issues noted (not fixed — minor):**
- Line 822: "message-passing layer" lacks back-reference to Lecture 1
- Line 994: "Triangular updates" lack back-reference to Lecture 4
- Line 1493: "DDPM" acronym not expanded on first use
- Line 1510: "ProteinMPNN" mentioned without noting it is Lecture 6
- Line 1622: "von Mises distributions" used in Exercise 10 without definition

**Concepts introduced:** 23
**Notation symbols introduced:** 0

---

### Note L6: ProteinMPNN: Inverse Folding and Sequence Design

**Status:** processed
**Changes made:** none — all cross-references correct, section numbering consistent

**Concepts introduced:** 13
**Notation symbols introduced:** 0

---

## Concept Index

Alphabetical listing of all tracked concepts with source note and usage.

| Concept | Defined In | Also Used In |
|---------|-----------|-------------|
| Activation Function | P2 | L1 |
| Adam Optimizer | P3 | P4 |
| Artificial Neuron | P2 | |
| Attention Mechanism | L1 | L3, L4, L5, L6 |
| AUC-ROC | P4 | |
| Autoregressive Decoding | L6 | |
| Backpropagation | P3 | P2 |
| Bias-Variance Tradeoff | P3 | |
| Binary Classification | P2 | P4 |
| Binary Cross-Entropy | P3 | P4 |
| C-alpha Atom | P2 | L1, L6 |
| Catastrophic Forgetting | L3 | |
| Chain Rule | P3 | |
| Class Imbalance | P4 | |
| Classifier Guidance | L2 | |
| Classifier-Free Guidance | L2 | L5 |
| Co-evolution | L1 | L4 |
| Computational Graph | P1 | P3 |
| Contact Map | P2 | L1 |
| Convolutional Neural Network | P2 | P4 |
| Cross-Entropy Loss | P3 | L2 |
| Dataset and DataLoader | P3 | P4 |
| Dihedral Angles | P2 | |
| Diffusion Model | L2 | L5 |
| Distance Matrix | P2 | L1 |
| Early Stopping | P3, P4 | |
| ELBO | L2 | |
| Embedding Extraction | L3 | |
| Epoch | P3 | |
| ESM | L1 | L3, L4, L6 |
| ESM-2 | L3 | |
| ESMFold | L3 | L6 |
| Evoformer | L4 | |
| F1 Score | P4 | |
| FAPE Loss | L4 | |
| FASTA Format | P2 | |
| Fine-Tuning | L3 | |
| Forward Process | L2 | L5 |
| Fully Connected Layer | P2 | |
| Function Approximation | P1 | |
| GAT | L1 | |
| GCN | L1 | |
| GELU | P2 | L1 |
| Generalization | P1 | P3, P4 |
| GPU Acceleration | P1 | |
| Gradient | P1 | P3 |
| Gradient Descent | P1 | P2, P3 |
| Graph Neural Network | P2 | L1, L6 |
| IGSO3 Distribution | L5 | |
| Invariant Point Attention | L1 | L4, L5 |
| Inverse Folding | L6 | |
| KL Divergence | L2 | |
| Layer Normalization | L1 | L3, L4 |
| Learned Embedding | P2 | P4, L1 |
| Learning Rate | P1 | P3 |
| Linear Regression | P1 | P2 |
| LoRA | L3 | |
| Loss Function | P1 | P2, P3 |
| Machine Learning | P1 | |
| Masked Language Modeling | L1 | L3 |
| Mean Squared Error | P1 | P3, L2 |
| Message Passing | P2 | L1, L5, L6 |
| Mini-Batch Training | P3 | |
| Motif Scaffolding | L5 | L6 |
| MPNN Framework | L1 | L6 |
| Multi-class Classification | P2 | |
| Multi-Head Attention | L1 | L3, L4 |
| Multiple Sequence Alignment | L1 | L3, L4 |
| nn.Module | P2 | P3 |
| Noise Schedule | L2 | L5 |
| One-Hot Encoding | P1 | P2, L1, L2 |
| Overfitting | P3 | P4 |
| Parameters | P1 | P2 |
| PDB Format | P2 | |
| Positional Encoding | L1 | L2 |
| Precision | P4 | |
| Protein Language Model | L3 | L4, L6 |
| Recall | P4 | |
| Regression | P2 | |
| ReLU | P2 | |
| Reparameterization Trick | L2 | |
| Residual Connection | L1 | L3, L4 |
| Reverse Process | L2 | L5 |
| Rigid Body Frame | L4, L5 | L6 |
| Rotational Diffusion | L5 | |
| Scaled Dot-Product Attention | L1 | |
| Score Function | L2 | |
| SE(3) Equivariance | L1 | L4, L5, L6 |
| Self-Supervised Learning | L3 | |
| Sequence Recovery Rate | L6 | |
| Sequence-Identity Split | P4 | |
| Sigmoid | P2 | |
| Softmax | P2 | L1 |
| Stochastic Gradient Descent | P3 | |
| Temperature Sampling | L6 | |
| Tensor | P1 | P2, P3 |
| Train-Validation-Test Split | P3 | |
| Training Loop | P3 | P4 |
| Transformer | P2 | L1, L2 |
| Transformer Block | L1 | L3, L4 |
| Universal Approximation Theorem | P2 | |
| Variational Autoencoder | L2 | |
| Zero-Shot Prediction | L3 | |

**Total tracked concepts: ~100**

---

## Notation Table

| Symbol | Meaning | Source | Conflicts |
|--------|---------|--------|-----------|
| $$f^*$$ | True unknown function | P1 | None |
| $$f_\theta$$ | Parameterized model | P1 | None |
| $$\theta$$ | Learnable parameters | P1 | None |
| $$\mathbf{W}$$ | Weight matrix | P1 | None |
| $$b$$ | Bias term | P1 | None |
| $$\hat{y}$$ | Model prediction | P1 | None |
| $$L$$ | Loss function | P1 | Overloaded: also protein length (P2+) |
| $$\eta$$ | Learning rate | P1 | None |
| $$\nabla_\theta L$$ | Parameter gradient | P1 | None |
| $$L$$ | Protein sequence length | P2 | Overloaded with loss (P1) — context disambiguates |
| $$d$$ | Embedding dimension | P2 | None |
| $$D_{ij}$$ | Pairwise C-alpha distance | P2 | None |
| $$\mathbf{r}_i$$ | 3D coordinate of residue i | P2 | None |
| $$C_{ij}$$ | Binary contact indicator | P2 | None |
| $$\phi, \psi$$ | Backbone dihedral angles | P2 | None |
| $$\sigma$$ | Activation function | P2 | None |
| $$\mathbf{h}$$ | Hidden representation | P2 | None |
| $$B$$ | Batch size | P3 | None |
| $$L_{\text{BCE}}$$ | Binary cross-entropy | P3 | None |
| $$L_{\text{CE}}$$ | Cross-entropy loss | P3 | None |
| $$C$$ | Number of classes | P3 | None |
| $$q_i, k_j, v_j$$ | Query, key, value vectors | L1 | None |
| $$\alpha_{ij}$$ | Attention weight | L1 | None |
| $$d_k$$ | Key/query dimension | L1 | None |
| $$h$$ | Number of attention heads | L1 | Overloaded: also hidden representation (P2) |
| $$\mathcal{N}(i)$$ | Node neighborhood | L1 | None |
| $$h_i^{(\ell)}$$ | Node feature at layer l | L1 | None |
| $$e_{ij}$$ | Edge feature | L1 | None |
| $$x_0$$ | Clean data point | L2 | None |
| $$x_t$$ | Noisy data at timestep t | L2 | None |
| $$\beta_t$$ | Per-step noise variance | L2 | None |
| $$\bar{\alpha}_t$$ | Cumulative noise product | L2 | None |
| $$\epsilon_\theta$$ | Noise prediction network | L2 | None |
| $$z$$ | Latent code (VAE) | L2 | None |
| $$T$$ | Total diffusion timesteps | L2 | Overloaded: also temperature (L6) — context disambiguates |

**Known overloads (not conflicts):**
- $$L$$: loss function (P1) vs. protein length (P2+) — always clear from context
- $$h$$: hidden representation (P2) vs. number of heads (L1) — different scopes
- $$T$$: diffusion timesteps (L2) vs. temperature (L6) — different notes

**Minor typographic inconsistency:**
- Translation vector in frames: $$\vec{t}_i$$ (L5) vs. $$\mathbf{t}_i$$ (L4, L6) — arrow vs. bold notation

---

## Gap Report

### Concepts used without formal definition anywhere in the curriculum

| Concept | Where Used | Severity |
|---------|-----------|----------|
| **Dropout** | L1 (code), L3 (code), L4 (code) | Medium — used extensively in code without explanation. A one-sentence definition in P3 or L1 would suffice. |
| **Logistic regression** | L3, Section 5 | Low — mentioned as a downstream model example |
| **UMAP / t-SNE** | L3, Exercise 1 | Low — exercise context, students can look up |
| **Spearman correlation** | L3, Exercise 2 | Low — exercise context |
| **Von Mises distribution** | L5, Exercise 10 | Low — exercise context |
| **DDPM (acronym)** | L5 | Low — referenced to Ho et al. but acronym not expanded |

### Unresolved minor issues

| File | Line | Issue |
|------|------|-------|
| L5 | 822 | "message-passing layer" lacks back-reference to Lecture 1 |
| L5 | 994 | "Triangular updates" lack back-reference to Lecture 4 |
| L5 | 1510 | "ProteinMPNN" mentioned without "(Lecture 6)" |
| L4, L5 | 20 | "## Introduction" is unnumbered (unlike sections 1-10/14) — this is a deliberate preamble pattern shared by L3, L4, L6 |

### No missing figures detected

All figure references use `relative_url` filter and point to files in `assets/img/teaching/protein-ai/`.

---

## Unified Bibliography

References appear across notes. Major papers cited:

### Required Reading (cited in 2+ notes)
- Jumper et al. (2021) "Highly accurate protein structure prediction with AlphaFold" — L4, L5, L6
- Vaswani et al. (2017) "Attention Is All You Need" — L1, L3, L4
- Lin et al. (2023) "Evolutionary-scale prediction..." (ESM-2/ESMFold) — L1, L3, L4
- Watson et al. (2023) "De novo design of protein structure and function with RFDiffusion" — L5, L6
- Dauparas et al. (2022) "Robust deep learning-based protein sequence design..." (ProteinMPNN) — L5, L6
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models" — L2, L5
- Kingma & Welling (2014) "Auto-Encoding Variational Bayes" — L2
- Gilmer et al. (2017) "Neural Message Passing for Quantum Chemistry" — L1, L6

### Supplementary (cited in 1 note)
- Anfinsen (1973) — L4
- Levinthal (1968) — L4
- Baek et al. (2021) RoseTTAFold — L4, L5
- Devlin et al. (2019) BERT — L1, L3, L4
- Kipf & Welling (2017) GCN — L1
- Velickovic et al. (2018) GAT — L1
- Hu et al. (2022) LoRA — L3
- Rives et al. (2021) ESM — L1, L3
- Nichol & Dhariwal (2021) Improved DDPM — L2, L5
- Mirdita et al. (2022) ColabFold — L4

---

## Verification

- `jekyll build --incremental` passed after all edits (no Liquid errors)
- All cross-references verified against frontmatter `lecture_number` fields
- Section numbering now consistent across all 10 notes
