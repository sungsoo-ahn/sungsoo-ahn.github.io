---
layout: post
title: "Case Study: Predicting Protein Solubility"
date: 2026-03-03
description: "An end-to-end case study—building, training, and honestly evaluating a solubility predictor, including sequence-identity splits, class imbalance, early stopping, and debugging."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 4
preliminary: true
toc:
  sidebar: left
related_posts: false
mermaid:
  enabled: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Preliminary Note 4 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. It applies everything from Preliminary Notes 1--3 in a complete case study. You should work through this note before the first in-class lecture.</em></p>

## Introduction

In the previous three notes you learned what machine learning is, how to represent protein data as tensors, how neural networks transform those tensors into predictions, and how the training loop adjusts weights to reduce a loss function.
Now we bring everything together in a single, end-to-end project: predicting whether a protein will be soluble when expressed in *E. coli*.

This note is structured as a narrative.
We start by building and training a baseline model using the tools from Notes 1--3.
Then, step by step, we discover problems --- misleading evaluation, class imbalance, overfitting --- and fix them.
Each section follows the same arc: *observe a problem → understand why it happens → introduce a technique that addresses it → show the improvement*.

By the end, you will have a working solubility predictor and a practical toolkit for diagnosing and fixing the most common training problems in protein machine learning.

### Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | The Solubility Prediction Problem | Why this problem matters and what makes it amenable to ML |
| 2 | Model Architecture and Data Preparation | A 1D-CNN classifier, data splitting, Dataset/DataLoader setup |
| 3 | Training and Evaluation | Training script, evaluation metrics beyond accuracy, precision-recall |
| 4 | Evaluating Properly: Sequence-Identity Splits | Why random splits overestimate performance, and how to fix it |
| 5 | Handling Class Imbalance | Weighted loss functions for imbalanced datasets |
| 6 | Knowing When to Stop: Early Stopping | Detecting the overfitting point and saving the best model |
| 7 | Debugging and Reproducibility | NaN detection, shape checks, single-batch overfit test, seed setting |

### Prerequisites

This note assumes you have worked through Preliminary Notes 1--3: tensors, neural network architectures, loss functions, optimizers, the training loop, data loading, and validation.

---

## 1. The Solubility Prediction Problem

### Why Solubility Prediction Matters

Expressing recombinant proteins is a core technique in structural biology, biotechnology, and therapeutic development.
When a target protein aggregates into inclusion bodies instead of dissolving in the cytoplasm, downstream applications --- crystallography, assays, drug formulation --- become much harder or impossible.
A computational model that predicts solubility from sequence alone can guide construct design and save weeks of experimental effort.

### What Makes This Problem Amenable to Machine Learning?

Solubility is influenced by sequence-level properties: amino acid composition, charge distribution, hydrophobicity patterns, and the presence of certain sequence motifs.
These patterns are learnable from data.

This is a **binary classification** task: given a protein sequence, predict whether it will be soluble (1) or insoluble (0).
We use the tools from Preliminary Note 3: binary cross-entropy loss, the `ProteinDataset` class, and the training loop.

---

## 2. Model Architecture and Data Preparation

### The Model Architecture

We build a **1D convolutional neural network** (CNN) that processes amino acid embeddings.
The architecture reflects domain knowledge: convolutional layers with a kernel size of 5 can detect patterns spanning five consecutive amino acids, which is appropriate for capturing local sequence motifs like charge clusters or hydrophobic stretches.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinSolubilityClassifier(nn.Module):
    """
    A 1D-CNN for predicting protein solubility from amino acid sequence.

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

### Preparing the Data

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load a solubility dataset (e.g., from the SOLpro or eSOL databases)
df = pd.read_csv('solubility_data.csv')
print(f"Dataset size: {len(df)} proteins")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Split into train / validation / test sets
# Stratify by label to maintain class balance in each split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'],
                                     random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'],
                                   random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Create Dataset and DataLoader objects (ProteinDataset from Note 3)
from torch.utils.data import DataLoader

train_dataset = ProteinDataset(train_df['sequence'].tolist(),
                               train_df['label'].tolist())
val_dataset = ProteinDataset(val_df['sequence'].tolist(),
                             val_df['label'].tolist())
test_dataset = ProteinDataset(test_df['sequence'].tolist(),
                              test_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

---

## 3. Training and Evaluation

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

# Instantiate the model and inspect its size
model = ProteinSolubilityClassifier()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Train with validation monitoring
trained_model = train_model(model, train_loader, val_loader, epochs=50)
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

| Metric | Question It Answers | Protein Example |
|---|---|---|
| **Accuracy** | What fraction of all predictions are correct? | 85% of solubility predictions correct |
| **Precision** | Of positive predictions, what fraction truly are? | Of proteins predicted soluble, how many truly are? |
| **Recall** | Of true positives, what fraction did we detect? | Of truly soluble proteins, how many did we find? |
| **F1 Score** | Harmonic mean of precision and recall | Balance between missing soluble proteins and wasting experiments |
| **AUC-ROC** | How well does the model separate classes across all thresholds? | Overall ability to distinguish soluble from insoluble |

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
Sections 5 and 6 address this problem with early stopping and other techniques.

---

## 4. Evaluating Properly: Sequence-Identity Splits

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

## 5. Handling Class Imbalance

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

## 6. Knowing When to Stop: Early Stopping

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

## 7. Debugging and Reproducibility

### Debugging Neural Networks

Neural networks can fail silently.
The code runs, the loss decreases, but predictions are useless.
Systematic debugging is essential.

```python
# 1. Check for NaN gradients (sign of numerical instability)
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"WARNING: NaN gradient detected in {name}")

# 2. Verify that outputs are in a sensible range
with torch.no_grad():
    sample_output = model(sample_input)
    print(f"Output range: [{sample_output.min():.3f}, {sample_output.max():.3f}]")

# 3. Confirm that input and output shapes match expectations
print(f"Input shape:  {sample_input.shape}")
print(f"Output shape: {model(sample_input).shape}")

# 4. Sanity check: can the model overfit a single batch?
# If it cannot, there is likely a bug in the architecture or loss
small_batch_x, small_batch_y = next(iter(train_loader))
for step in range(200):
    pred = model(small_batch_x.to(device))
    loss = criterion(pred, small_batch_y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
# Loss should approach 0. If it does not, debug your model.
```

### Reproducibility

Science requires reproducibility.
Set all random seeds at the start of every experiment:

```python
import random
import numpy as np

def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism on GPU (may reduce performance slightly):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

Full determinism on GPU can slow training by 10--20%.
For exploratory experiments, setting the Python, NumPy, and PyTorch seeds is usually sufficient.
Reserve full determinism for final reported results.

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

2. **Evaluation metrics** must go beyond accuracy. Precision, recall, F1, and AUC-ROC tell a more complete story, especially for imbalanced datasets.

3. **Sequence-identity splits are mandatory** for honest evaluation of protein models. Random splits systematically overestimate performance due to data leakage from homologous sequences.

4. **Address class imbalance** with weighted losses. High accuracy on an imbalanced dataset is meaningless if the model ignores the minority class.

5. **Early stopping** saves the best model and prevents wasted computation. Use a patience of 10--20 epochs for protein tasks.

6. **Systematic debugging** catches silent failures. The single-batch overfit test is the most important sanity check.

7. **Improvement is iterative.** Each technique provides incremental gains. The combination of sequence-identity splits, class weighting, and early stopping yields a model that is substantially more honest and more useful than the naive baseline.

---

## Exercises

All exercises use the protein solubility predictor from this note as their starting point.

### Exercise 1: Data Leakage Experiment

Create two sets of train/test splits for the solubility dataset:
- (a) Random splits (proteins assigned uniformly at random)
- (b) Sequence identity splits at 30% using MMseqs2 or CD-HIT

Train the same model on both splits and compare test accuracy, F1, and AUC.
By how many percentage points does the random split overestimate performance?

### Exercise 2: Class Imbalance Experiment

Construct a version of the solubility dataset where 90% of examples are soluble and 10% are insoluble.
Train the model with:
- (a) Standard cross-entropy (unweighted)
- (b) Inverse-frequency weighted cross-entropy

Compare not just accuracy but also precision, recall, and F1-score for the *insoluble* class.
Why is accuracy a misleading metric for imbalanced tasks?

### Exercise 3: Matthews Correlation Coefficient

Implement **Matthews Correlation Coefficient** (MCC), a metric that is informative even when classes are severely imbalanced:

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

where $$TP$$, $$TN$$, $$FP$$, and $$FN$$ are the counts of true positives, true negatives, false positives, and false negatives respectively.
Add this metric to the `evaluate_classifier` function and compare it to accuracy on a dataset where 90% of proteins are soluble.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 6--8. Available at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems*, 32.

3. Rao, R., Bhatt, N., Lu, A., Johnson, J., Ott, M., Auli, M., Russ, C., & Sander, C. (2019). Evaluating protein transfer learning with TAPE. *Advances in NeurIPS*. (Best practices for protein ML evaluation, including sequence identity splits.)

4. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., & Fergus, R. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15).
