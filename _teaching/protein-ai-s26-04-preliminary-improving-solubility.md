---
layout: post
title: "Case Study: Predicting Protein Solubility"
date: 2026-03-03
description: "An end-to-end case study—building sequence-based (1D-CNN) and structure-based (MLP) solubility predictors, combining both modalities, and learning to evaluate honestly with sequence-identity splits, class weighting, and early stopping."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 4
preliminary: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Preliminary Note 4 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. It applies everything from Preliminary Notes 1--3 in a complete case study. You should work through this note before the first in-class lecture.</em></p>

## Introduction

In the previous three notes you learned what machine learning is, how to convert protein data into numerical features, how neural networks transform those features into predictions, and how the training loop adjusts weights to reduce a loss function.
Now we bring everything together in a single, end-to-end project: predicting whether a protein will be soluble when expressed in *E. coli*.

We tackle this problem from two angles.
First, we build a **sequence-based** model: a 1D convolutional neural network (CNN) that scans the amino acid sequence for local patterns predictive of solubility.
Second, we build a **structure-based** model: an MLP that takes numerical features computed from the protein's 3D coordinates.
Comparing the two reveals what kind of information each approach captures --- and what it misses.

Along the way, we discover problems --- misleading evaluation, class imbalance, overfitting --- and fix them.
Each section follows the same arc: *observe a problem → understand why it happens → introduce a technique that addresses it → show the improvement*.

By the end, you will have working solubility predictors from both sequence and structure, and a practical toolkit for diagnosing and fixing the most common training problems in protein machine learning.

### Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | The Solubility Prediction Problem | Why this problem matters and what makes it amenable to ML |
| 2 | Sequence-Based Approach: 1D-CNN | How convolutions detect local sequence patterns; a CNN solubility classifier |
| 3 | Structure-Based Approach | Computing structural features from 3D coordinates; an MLP classifier; combining both modalities |
| 4 | Data Preparation | Dataset/DataLoader setup, train/val/test splitting |
| 5 | Training and Evaluation | Training script, evaluation metrics beyond accuracy, precision-recall |
| 6 | Evaluating Properly: Sequence-Identity Splits | Why random splits overestimate performance, and how to fix it |
| 7 | Handling Class Imbalance | Weighted loss functions for imbalanced datasets |
| 8 | Knowing When to Stop: Early Stopping | Detecting the overfitting point and saving the best model |
| 9 | Debugging and Reproducibility | NaN detection, shape checks, single-batch overfit test, seed setting |

### Prerequisites

This note assumes you have worked through Preliminary Notes 1--3: tensors, neural network architectures, loss functions, optimizers, the training loop, data loading, and validation.

---

## 1. The Solubility Prediction Problem

### Why Solubility Prediction Matters

Expressing recombinant proteins is a core technique in structural biology, biotechnology, and therapeutic development.
When a target protein aggregates into inclusion bodies instead of dissolving in the cytoplasm, downstream applications --- crystallography, assays, drug formulation --- become much harder or impossible.
A computational model that predicts solubility from sequence and structure can guide construct design and save weeks of experimental effort.

### What Makes This Problem Amenable to Machine Learning?

Solubility is influenced by sequence-level properties: amino acid composition, charge distribution, hydrophobicity patterns, and the presence of certain sequence motifs.
These patterns are learnable from data.

This is a **binary classification** task: given a protein sequence, predict whether it will be soluble (1) or insoluble (0).
We use the tools from Preliminary Note 3: binary cross-entropy loss, the `ProteinDataset` class, and the training loop.

---

## 2. Sequence-Based Approach: 1D-CNN

A simple approach to predicting solubility is to summarize the sequence as amino acid composition --- a 20-dimensional vector counting the fraction of each amino acid type --- and feed it to an MLP.
That approach treats the sequence as a *bag of amino acids*: it knows how much alanine a protein contains, but not where that alanine is.
Yet the *position* of amino acids matters.
A cluster of five hydrophobic residues in a row is a strong signal for a transmembrane helix (likely insoluble), while the same five residues scattered throughout the sequence may have no effect.

To detect such **local patterns**, we need an architecture that looks at neighboring positions together: the **convolutional neural network (CNN)**.

### 2.1 How 1D Convolution Works

<div class="col-sm-10 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/mermaid/s26-04-conv1d-sliding.png' | relative_url }}" alt="1D convolution sliding window">
    <div class="caption mt-1">A 1D convolution with kernel size 5 and padding 2. The kernel slides along the input sequence, computing a weighted sum at each position. With appropriate padding, the output length equals the input length.</div>
</div>

A **1D convolution** slides a small window (called a **filter** or **kernel**) along the sequence, computing a weighted sum at each position.

Consider a protein sequence of length $$L$$, where each position has been embedded into a $$d$$-dimensional vector (so the input is a matrix of shape $$L \times d$$).
A convolutional filter of **kernel size** $$k$$ and $$C_{\text{out}}$$ output channels computes:

$$
y_i^{(c)} = \text{ReLU}\!\left(\sum_{c'=1}^{d} \sum_{j=0}^{k-1} W_{c,c',j} \cdot x_{i+j}^{(c')} + b_c\right)
$$

where $$c$$ indexes the output channel, $$c'$$ indexes the input channel (embedding dimension), and $$j$$ indexes the position within the kernel window.
The filter slides from position 1 to position $$L$$, producing one output value per position per output channel.

**Padding** controls whether the output length matches the input length.
With `padding = (k-1)/2` (for odd $$k$$), the output length equals the input length:

$$
L_{\text{out}} = L_{\text{in}} + 2 \cdot \text{padding} - k + 1
$$

For example, `kernel_size=5` with `padding=2` gives $$L_{\text{out}} = L + 4 - 4 = L$$.

The key properties of convolution are:

- **Local receptive field.** Each output depends on only $$k$$ neighboring inputs. A kernel of size 5 captures motifs spanning five consecutive amino acids --- appropriate for charge clusters, hydrophobic stretches, and turn signals.
- **Weight sharing.** The same filter weights are applied at every position. The network doesn't need to learn separately that "FFFFF at position 10" and "FFFFF at position 200" are the same hydrophobic stretch.
- **Multiple filters.** We use many filters in parallel (e.g., 128), each learning to detect a different pattern. One filter might activate for hydrophobic stretches; another for charged clusters.
- **Stacking layers.** A second convolutional layer on top of the first sees combinations of first-layer patterns, detecting higher-level motifs spanning $$2k - 1 = 9$$ positions.

After convolution, **global average pooling** aggregates over all positions into a single vector.
When sequences are padded, we use a masked average to ignore padding positions:

$$
\bar{\mathbf{h}}^{(c)} = \frac{1}{L_{\text{valid}}} \sum_{i=1}^{L} m_i \cdot h_i^{(c)}
$$

where $$m_i \in \{0, 1\}$$ is the padding mask and $$L_{\text{valid}} = \sum_i m_i$$ is the number of real residues.
The resulting fixed-size vector is then fed to a linear layer for classification.
This makes the model invariant to sequence length --- it works for 50-residue proteins and 500-residue proteins alike.

### 2.2 The 1D-CNN Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceCNN(nn.Module):
    """
    1D-CNN for predicting protein solubility from amino acid sequence.

    Architecture:
    1. Embedding: map each amino acid index to a learned 64-dim vector
    2. Two Conv1d layers: detect local sequence motifs
    3. Global average pooling: aggregate over the full sequence
    4. Linear output: predict soluble (1) vs. insoluble (0)
    """

    def __init__(self, vocab_size=21, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()

        # Embedding layer: integers → continuous vectors
        # padding_idx=0 ensures the padding token always maps to a zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 1D convolutions detect local patterns in the sequence
        # kernel_size=5 means each filter looks at 5 consecutive amino acids
        # padding=2 preserves the sequence length after convolution
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len) — integer-encoded amino acids

        # Step 1: Embed amino acids → (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Step 2: Rearrange for Conv1d → (batch, embed_dim, seq_len)
        # Conv1d expects channels before sequence length
        x = x.transpose(1, 2)

        # Step 3: Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))    # → (batch, hidden_dim, seq_len)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))    # → (batch, hidden_dim, seq_len)

        # Step 4: Global average pooling over the sequence dimension
        # Use the mask to ignore padding positions
        if mask is not None:
            mask = mask.unsqueeze(1)             # → (batch, 1, seq_len)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)                   # → (batch, hidden_dim)

        # Step 5: Classify → (batch, num_classes)
        x = self.fc(x)
        return x
```

Note the `transpose(1, 2)` in step 2: PyTorch's `Conv1d` expects the input shape to be `(batch, channels, length)`, but the embedding produces `(batch, length, channels)`.

---

## 3. Structure-Based Approach: MLP on Structural Features

Sequence tells you *what* amino acids are present and in what order.
Structure tells you *how they are arranged in space*.
A protein might have a perfectly benign amino acid composition yet be insoluble because its fold exposes a large hydrophobic surface, or because it has unusually few internal contacts and tends to unfold.

For proteins with known or predicted structures[^predicted], we can extract numerical features from the 3D coordinates and feed them to an MLP.

[^predicted]: In practice, you can use AlphaFold (Lecture 4) to predict the structure for any sequence. This means the structure-based approach is available even when no experimental structure exists.

### 3.1 Structural Features from Coordinates

In Preliminary Note 2, you learned to extract $$\text{C}_\alpha$$ coordinates from PDB files.
From these coordinates, we can compute several features that are informative for solubility.

**Distance matrix.**
The $$\text{C}_\alpha$$ distance matrix is an $$L \times L$$ matrix where entry $$(i, j)$$ is the Euclidean distance between residues $$i$$ and $$j$$:

$$
D_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|_2
$$

**Contact map.**
A contact map is a binary version of the distance matrix: two residues are "in contact" if their $$\text{C}_\alpha$$ atoms are within a threshold distance (typically 8 Å):

$$
C_{ij} = \begin{cases} 1 & \text{if } D_{ij} < 8\text{ Å and } |i - j| > 3 \\ 0 & \text{otherwise} \end{cases}
$$

The condition $$|i - j| > 3$$ excludes trivial contacts between neighbors along the chain.

From the distance matrix and contact map, we compute four summary statistics:

| Feature | Formula | What It Captures |
|---|---|---|
| **Contact density** | Mean contacts per residue | How tightly packed the structure is |
| **Relative contact order** | Mean sequence separation of contacts, normalized by $$L$$ | Whether the fold involves mostly local or long-range contacts |
| **Radius of gyration** | RMS distance of $$\text{C}_\alpha$$ atoms from the centroid | Overall compactness of the protein |
| **Secondary structure fractions** | Fraction of helix / sheet / coil residues | Structural composition |

Each feature has a biological rationale for its connection to solubility:

- **Contact density:** Tightly packed proteins have more stabilizing intramolecular contacts, making them less likely to expose hydrophobic patches that cause aggregation. Higher contact density generally correlates with greater stability and solubility.

- **Relative contact order (RCO):** Higher RCO means the native fold depends on long-range contacts (residues far apart in sequence coming together in space). Such proteins fold more slowly and are more prone to kinetic trapping in misfolded, aggregation-prone states.

- **Radius of gyration ($$R_g$$):** $$R_g = \sqrt{\frac{1}{L}\sum_{i=1}^{L} \lVert \mathbf{r}_i - \bar{\mathbf{r}} \rVert^2}$$ measures overall compactness. Compact proteins (low $$R_g$$ relative to their length) expose less hydrophobic surface area to the solvent, reducing the driving force for aggregation.

- **Secondary structure fractions:** Proteins dominated by beta-sheets (especially those with exposed edge strands) are more aggregation-prone than helical proteins, because edge strands can form intermolecular beta-sheet contacts.

```python
import numpy as np
import torch

def compute_structural_features(ca_coords, sequence):
    """
    Compute structural features from Calpha coordinates.

    Args:
        ca_coords: (L, 3) array of Calpha positions in Angstroms.
        sequence: str, the amino acid sequence.

    Returns:
        features: (num_features,) tensor of structural features.
    """
    L = len(ca_coords)

    # Distance matrix: (L, L)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # (L, L, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))         # (L, L)

    # Contact map: exclude trivial chain neighbors (|i-j| <= 3)
    seq_sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    contact_map = (dist_matrix < 8.0) & (seq_sep > 3)

    # Feature 1: Contact density (mean contacts per residue)
    contact_density = contact_map.sum() / L

    # Feature 2: Relative contact order
    contacts_i, contacts_j = np.where(contact_map)
    if len(contacts_i) > 0:
        contact_order = np.mean(np.abs(contacts_i - contacts_j)) / L
    else:
        contact_order = 0.0

    # Feature 3: Radius of gyration
    centroid = ca_coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=-1)))

    # Feature 4: Amino acid composition
    aa_types = 'ACDEFGHIKLMNPQRSTVWY'
    aa_comp = np.array([sequence.count(aa) / L for aa in aa_types])

    # Combine all features
    structural = np.array([contact_density, contact_order, rg, float(L)])
    all_features = np.concatenate([structural, aa_comp])

    return torch.tensor(all_features, dtype=torch.float32)
```

This function produces a 24-dimensional feature vector: 4 structural statistics plus 20 amino acid composition values.
Including both sequence and structure features lets the model learn which information source is more predictive.

### 3.2 The Structure-Based MLP

Since the structural features are a fixed-size vector (regardless of protein length), we can use the same MLP architecture from Preliminary Note 2:

```python
class StructureMLP(nn.Module):
    """
    MLP for predicting protein solubility from structural features.

    Architecture:
    1. Input: 24-dim feature vector (4 structural + 20 composition)
    2. Two hidden layers with ReLU activation
    3. Linear output: predict soluble (1) vs. insoluble (0)
    """

    def __init__(self, input_dim=24, hidden_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, input_dim)
        return self.net(x)
```

### 3.3 Combining Sequence and Structure

If both sequence and structure carry useful but different signals, why not use both?
A **multimodal model** processes each input type with its own encoder and then combines the resulting representations for a joint prediction.

```python
class CombinedModel(nn.Module):
    """
    Multimodal model combining a sequence CNN and a structural MLP.

    The two branches produce independent representations,
    which are concatenated and fed to a shared classification head.
    """

    def __init__(self, vocab_size=21, embed_dim=64, cnn_dim=128,
                 struct_dim=24, mlp_dim=64, num_classes=2):
        super().__init__()

        # Branch 1: Sequence CNN (produces cnn_dim-dimensional representation)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, cnn_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)

        # Branch 2: Structure MLP (produces mlp_dim-dimensional representation)
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
        )

        # Shared classification head on the concatenated representations
        self.classifier = nn.Linear(cnn_dim + mlp_dim, num_classes)

    def forward(self, sequence, struct_features, mask=None):
        # Branch 1: encode the sequence
        x = self.embedding(sequence).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        if mask is not None:
            m = mask.unsqueeze(1)
            x = (x * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)                   # → (batch, cnn_dim)

        # Branch 2: encode the structural features
        s = self.struct_encoder(struct_features)  # → (batch, mlp_dim)

        # Combine and classify
        combined = torch.cat([x, s], dim=1)       # → (batch, cnn_dim + mlp_dim)
        return self.classifier(combined)
```

The key idea is **late fusion**: each branch first builds its own representation from its input modality, then the representations are concatenated before the final classification layer.
Formally, if the sequence encoder produces $$\mathbf{h}_{\text{seq}} \in \mathbb{R}^{d_1}$$ and the structure encoder produces $$\mathbf{h}_{\text{struct}} \in \mathbb{R}^{d_2}$$, the combined representation is:

$$
\mathbf{h}_{\text{combined}} = [\mathbf{h}_{\text{seq}}; \mathbf{h}_{\text{struct}}] \in \mathbb{R}^{d_1 + d_2}
$$

The classification layer $$\mathbf{W} \in \mathbb{R}^{C \times (d_1 + d_2)}$$ then learns to weight each modality appropriately.
If structure features are uninformative for a particular protein, the classifier can assign near-zero weights to the $$\mathbf{h}_{\text{struct}}$$ dimensions and rely on sequence alone.

### 3.4 Comparing the Three Approaches

| | Sequence CNN | Structure MLP | Combined |
|---|---|---|---|
| **Input** | Raw amino acid sequence | $$\text{C}_\alpha$$ coordinates + composition | Both |
| **What it captures** | Local sequence motifs | Global structural properties | Both signal types |
| **Availability** | Any protein | Requires 3D structure | Requires 3D structure |
| **Parameters** | ~100K | ~5K | ~105K |

In practice, the combined model typically outperforms either individual approach because the two input modalities carry complementary information.
Sequence motifs and structural compactness are correlated but not redundant --- a protein can have a "normal" amino acid composition yet be insoluble due to an unusual fold.

In the main lectures, we will see architectures (GNNs, transformers) that can process the full 3D structure directly, rather than relying on hand-crafted structural features.

---

## 4. Data Preparation

Both models need a training dataset split into train, validation, and test sets.
The procedure follows Preliminary Note 3 exactly: use `train_test_split` with `stratify` to maintain class balance, wrap sequences in a `ProteinDataset`, and create `DataLoader` objects with `shuffle=True` for training and `shuffle=False` for evaluation.

For the structure-based model, precompute structural features for the entire dataset using `compute_structural_features` (Section 3.1), stack them into a tensor, and split using the same train/val/test indices.
Precomputation avoids redundant coordinate parsing during training.

---

## 5. Training and Evaluation

### The Training Script

We combine the `train_one_epoch` and `evaluate` functions from Preliminary Note 3 into a complete training pipeline with validation monitoring.

```python
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Full training pipeline with validation monitoring."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training phase ---
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # --- Validation phase ---
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        # --- Logging ---
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}")

    # Load the best model before returning
    model.load_state_dict(torch.load('best_model.pt'))
    return model

# Instantiate each model and inspect its size
seq_model = SequenceCNN()
struct_model = StructureMLP()
combined_model = CombinedModel()

for name, m in [("SequenceCNN", seq_model), ("StructureMLP", struct_model),
                ("CombinedModel", combined_model)]:
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: {n_params:,} parameters")

# Train with validation monitoring (example with the sequence model)
trained_model = train_model(seq_model, train_loader, val_loader, epochs=50)
```

### Evaluation: Beyond Accuracy

A single accuracy number rarely tells the full story.
For a solubility dataset where 70% of proteins are soluble, a model that *always* predicts "soluble" achieves 70% accuracy while being completely useless.
We need a richer set of metrics.

```python
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

def evaluate_classifier(model, test_loader, device):
    """Evaluate a binary classifier with multiple metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['sequence'].to(device)
            mask = batch['mask'].to(device)
            y = batch['label']

            logits = model(x, mask)                         # Raw scores
            probs = F.softmax(logits, dim=-1)               # Probabilities
            preds = logits.argmax(dim=-1)                   # Predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())     # P(soluble)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    return accuracy, precision, recall, f1, auc
```

### Understanding the Metrics

All classification metrics are defined in terms of four counts: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

The $$F_1$$ score is the harmonic mean of precision and recall --- it is high only when *both* are high.

**AUC-ROC** (Area Under the Receiver Operating Characteristic curve) plots the true positive rate (recall) against the false positive rate ($$FP / (FP + TN)$$) as the classification threshold varies from 0 to 1, and computes the area under this curve.
An AUC of 1.0 means the model achieves perfect separation at some threshold; 0.5 means no better than random.

| Metric | Question It Answers | Protein Example |
|---|---|---|
| **Accuracy** | What fraction of all predictions are correct? | 85% of solubility predictions correct |
| **Precision** | Of positive predictions, what fraction truly are? | Of proteins predicted soluble, how many truly are? |
| **Recall** | Of true positives, what fraction did we detect? | Of truly soluble proteins, how many did we find? |
| **F1 Score** | Harmonic mean of precision and recall | Balance between missing soluble proteins and wasting experiments |
| **AUC-ROC** | How well does the model separate classes across all thresholds? | Overall ability to distinguish soluble from insoluble |

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/precision_recall_curves.png' | relative_url }}" alt="Precision-recall tradeoff">
    <div class="caption mt-1">Precision-recall curves for two models and a random baseline. The tradeoff between precision and recall is controlled by the classification threshold.</div>
</div>

The precision-recall tradeoff deserves special attention.
In a drug discovery setting, where expressing each candidate is expensive, a biologist might want **high precision**: "I only want to express proteins that are very likely to be soluble."
By raising the classification threshold from 0.5 to 0.8, we predict fewer proteins as soluble but are more confident in those predictions.

Conversely, in a high-throughput screening setting with thousands of candidates, a biologist might prefer **high recall**: "I don't want to miss any potentially soluble protein."
Lowering the threshold to 0.3 captures more true positives at the cost of more false positives.

The AUC-ROC summarizes this tradeoff across all possible thresholds.
An AUC of 1.0 means perfect separation; 0.5 means the model is no better than random.

### Analyzing the Loss Curves

After training for 50 epochs, examine the loss curves.

<div class="col-sm-9 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/overfitting_curves.png' | relative_url }}" alt="Training vs validation loss showing overfitting">
    <div class="caption mt-1"><strong>Training and validation loss curves.</strong> Training loss decreases steadily, but validation loss begins increasing after ~40 epochs --- the model is memorizing the training data rather than learning generalizable patterns.</div>
</div>

If the training loss decreases smoothly but the validation loss rises after approximately 40 epochs, the model is overfitting.
The gap between training accuracy (~98%) and validation accuracy (~72%) confirms this.
Sections 7 and 8 address this problem with weighted losses and early stopping.

---

## 6. Evaluating Properly: Sequence-Identity Splits

### The Problem

After training, the solubility predictor achieves 85% validation accuracy.
Impressive?
Not necessarily.
We need to examine *how* we split the data.

If we used a random train/validation/test split, there is a high probability that some test proteins are closely related (>90% sequence identity) to proteins in the training set.
These homologous proteins almost certainly share the same solubility status.
The model can score well by memorizing similar sequences rather than learning true sequence-to-solubility patterns.
This is **data leakage** --- the test set contains information that was effectively available during training.

### The Solution: Sequence-Identity Splits

The fix: cluster all proteins by sequence identity --- commonly at a 30% or 40% threshold --- and split the data at the **cluster** level, not the individual protein level.
This ensures that no test protein is closely related to any training protein.

```python
import subprocess
import numpy as np

def create_sequence_identity_splits(fasta_file, identity_threshold=0.3, train_ratio=0.8):
    """Split proteins into train/val/test sets respecting sequence identity.

    Requires MMseqs2 to be installed (https://github.com/soedinglab/MMseqs2).
    """
    # Step 1: Cluster proteins at the specified identity threshold
    subprocess.run([
        'mmseqs', 'easy-cluster',
        fasta_file, 'clusters', 'tmp',
        '--min-seq-id', str(identity_threshold)
    ])

    # Step 2: Parse cluster assignments
    clusters = parse_cluster_file('clusters_cluster.tsv')

    # Step 3: Shuffle and split clusters (not individual proteins)
    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * 0.1)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    # Step 4: Collect protein IDs from the assigned clusters
    train_ids = [pid for c in train_clusters for pid in clusters[c]]
    val_ids = [pid for c in val_clusters for pid in clusters[c]]
    test_ids = [pid for c in test_clusters for pid in clusters[c]]

    return train_ids, val_ids, test_ids
```

### The Reality Check

When we retrain our model using sequence-identity splits instead of random splits, the test accuracy typically drops by 5--15 percentage points.
This drop reflects the true difficulty of the task: predicting solubility for proteins that are genuinely different from anything in the training set.

The random-split accuracy was a mirage.
The sequence-identity-split accuracy is the honest answer.
Any paper that reports performance without controlling for sequence similarity should be read with skepticism.

A word of caution: even 30% sequence identity splits may not be sufficient for all tasks.
Proteins from the same CATH[^cath] superfamily can share structural features despite having diverged below 30% identity.
For the most rigorous evaluation, consider splitting at the fold or superfamily level.

[^cath]: CATH is a hierarchical classification of protein domain structures: **C**lass (secondary structure content), **A**rchitecture (spatial arrangement), **T**opology (fold), and **H**omologous superfamily.

---

## 7. Handling Class Imbalance

### The Problem

After switching to sequence-identity splits, we notice another issue: the model's performance on **insoluble** proteins is much worse than on soluble ones.
Looking at the data, we find that 70% of our dataset is soluble and only 30% is insoluble.
The model has learned a shortcut: predicting "soluble" for everything gives 70% accuracy with no effort.

### Weighted Loss Functions

The simplest correction: assign higher weights to underrepresented classes, so that misclassifying an insoluble protein incurs a larger penalty:

```python
def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency weights for class-balanced training."""
    counts = torch.bincount(labels.flatten(), minlength=num_classes).float()
    weights = 1.0 / (counts + 1)                    # Inverse frequency
    weights = weights / weights.sum() * num_classes  # Normalize
    return weights

# Apply to our solubility dataset
class_weights = compute_class_weights(train_labels, num_classes=2)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

With weighted loss, the model is penalized more heavily for misclassifying the minority class (insoluble proteins).
This forces it to pay attention to features that distinguish insoluble proteins rather than defaulting to "soluble."

### Effect on the Model

After applying class weights, the overall accuracy may drop slightly (the model can no longer cheat by always predicting the majority class), but the **F1 score and recall for the minority class improve substantially**.
This is the metric that matters: a model that correctly identifies insoluble proteins is far more useful than one that achieves high accuracy by ignoring them.

---

## 8. Knowing When to Stop: Early Stopping

### The Problem

Even with regularization (dropout in our model), there comes a point when continued training hurts more than it helps.
Validation loss may start rising again after reaching its best value, indicating that the model is beginning to overfit to the training data.

### Early Stopping

**Early stopping** is a form of regularization based on *time* rather than architecture.
The idea: monitor validation performance during training and stop when it stops improving.

Why does this work as regularization?
In the early phases of training, the model learns general, transferable patterns.
As training continues, it gradually begins to memorize training-specific noise.
The point at which validation performance peaks is the sweet spot between underfitting and overfitting.

```python
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """Call once per epoch. Returns True if this is a new best model."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True   # New best — save checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # No improvement
```

Integrating early stopping into the training loop:

```python
early_stopping = EarlyStopping(patience=15)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)

    if early_stopping.step(val_loss):
        torch.save(model.state_dict(), 'best_model.pt')

    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))
```

The **patience** parameter controls how long to wait for improvement.
For protein models with small datasets (and therefore noisy validation estimates), a patience of 10 to 20 epochs is common.

---

## 9. Debugging and Reproducibility

### Debugging Neural Networks

Neural networks can fail silently.
The code runs, the loss decreases, but predictions are useless.
A systematic debugging checklist:

1. **Check for NaN gradients** — iterate over `model.named_parameters()` and check `torch.isnan(param.grad).any()`. NaN gradients indicate numerical instability (often from a learning rate that is too large).

2. **Verify output range** — run a forward pass with `torch.no_grad()` and print `output.min()` / `output.max()`. For logits, values should be in a reasonable range (roughly $$[-10, 10]$$).

3. **Check shapes** — print input and output shapes at every stage. Shape mismatches (especially after `transpose` or `unsqueeze`) are the most common bug.

4. **Single-batch overfit test** — train on a single batch for 200 steps. If the loss does not approach zero, there is a bug in the architecture, loss function, or data pipeline. This is the single most important debugging technique.

### Reproducibility

Set all random seeds (`random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) at the start of every experiment.
For full GPU determinism, additionally set `torch.backends.cudnn.deterministic = True` (may slow training by 10--20%).

### A Practical Checklist

Before declaring a model "trained," verify the following:

1. Training loss decreases steadily over epochs.
2. Validation loss decreases initially, then plateaus (not increases --- that signals overfitting).
3. The model can perfectly overfit a single batch (sanity check for bugs).
4. Gradients are finite (no NaN or Inf values).
5. Metrics on the test set are consistent with validation set metrics.
6. Results are reproducible when the same seed is used.

---

## Key Takeaways

1. **Solubility prediction** is a representative binary classification task that exercises every component of the ML pipeline: data preparation, model architecture, training, and evaluation.

2. **Sequence and structure provide complementary signals.** A 1D-CNN captures local sequence motifs; an MLP on structural features captures global 3D properties. A combined model leverages both.

3. **CNNs detect local patterns** by sliding learned filters along the sequence. Weight sharing makes them efficient, and stacking layers lets them capture progressively larger motifs.

4. **Evaluation metrics** must go beyond accuracy. Precision, recall, F1, and AUC-ROC tell a more complete story, especially for imbalanced datasets.

5. **Sequence-identity splits are mandatory** for honest evaluation of protein models. Random splits systematically overestimate performance due to data leakage from homologous sequences.

6. **Address class imbalance** with weighted losses. High accuracy on an imbalanced dataset is meaningless if the model ignores the minority class.

7. **Early stopping** saves the best model and prevents wasted computation. Use a patience of 10--20 epochs for protein tasks.

8. **Systematic debugging** catches silent failures. The single-batch overfit test is the most important sanity check.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 6--8. Available at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems*, 32.

3. Rao, R., Bhatt, N., Lu, A., Johnson, J., Ott, M., Auli, M., Russ, C., & Sander, C. (2019). Evaluating protein transfer learning with TAPE. *Advances in NeurIPS*. (Best practices for protein ML evaluation, including sequence identity splits.)

4. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., & Fergus, R. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15).
