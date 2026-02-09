---
layout: post
title: "Training Neural Networks for Protein Science"
date: 2026-03-03
description: "Loss functions, optimizers, the training loop, data loading, validation, and a complete protein solubility prediction case study."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 3
preliminary: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Preliminary Note 3 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. It continues directly from Preliminary Note 2 (Machine Learning and Neural Network Fundamentals). By the end of this note, you will be able to train a neural network end-to-end and evaluate it on a protein prediction task.</em></p>

## Introduction

In Preliminary Note 2 you learned how to build neural networks: tensors for data, autograd for gradients, and `nn.Module` for architecture.
But a network fresh off the assembly line knows nothing --- its weights are random numbers, and its predictions are meaningless.
This note is about taking that raw network and making it *learn*.

We will cover every component of the training process: the loss functions that quantify mistakes, the optimizers that correct them, the training loop that orchestrates the process, and the data loading machinery that feeds proteins to the network efficiently.
We end with a complete case study --- predicting protein solubility in *E. coli* --- that ties every piece together into a working system you can adapt for your own projects.

### Roadmap

| Section | Topic | Why You Need It |
|---|---|---|
| 1 | Loss Functions | Different prediction tasks require different ways of measuring error |
| 2 | Mini-Batch Training and Optimizers | Why we train on batches, what "stochastic" means, and the algorithms that turn gradients into weight updates |
| 3 | The Training Loop | The four-step cycle that turns data into knowledge |
| 4 | Data Loading for Proteins | Efficient batching, shuffling, and handling of variable-length sequences |
| 5 | Validation and Overfitting | How to detect when your model is memorizing rather than learning |
| 6 | Case Study: Protein Solubility Prediction | A complete, end-to-end project integrating everything from this note |
| 7 | Debugging and Reproducibility | Practical tips for when things go wrong |

### Prerequisites

This note assumes familiarity with Preliminary Note 2: tensors, `nn.Module`, activation functions, autograd, and the basic idea of gradient descent.

---

## 1. Loss Functions: Measuring Mistakes

Before the model can learn, we need a way to quantify how wrong its predictions are.
The **loss function** (also called a **cost function** or **objective function**) produces a single number: zero means perfect predictions, and larger values mean worse predictions.
The choice of loss function depends on the type of prediction task.

We introduced $$L_{\text{MSE}}$$ briefly in Preliminary Note 2 as our first loss function. Here we examine it alongside the classification losses in a systematic treatment.

### Mean Squared Error (MSE) for Regression

MSE is the standard loss for **regression** tasks --- predicting continuous values.
In protein science, this means predicting binding affinity or melting temperature; in a general setting, it means predicting a house's sale price or a person's age from a photograph.
Let $$y_i$$ be the true value and $$\hat{y}_i(\theta)$$ be the model's prediction for example $$i$$ (which depends on the current parameters $$\theta$$), with $$n$$ examples in total:

$$
L_{\text{MSE}}(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i(\theta))^2
$$

Squaring the error penalizes large mistakes heavily.
A prediction that is off by 10 degrees contributes 100 to the sum, while one that is off by 1 degree contributes only 1.
This makes $$L_{\text{MSE}}$$ sensitive to outliers --- a single wildly mispredicted protein can dominate the loss.

### Binary Cross-Entropy (BCE) for Binary Classification

BCE is designed for **binary classification** --- tasks with two categories, such as predicting whether a protein is soluble versus insoluble, or whether an email is spam versus not spam.
Let $$y_i \in \{0, 1\}$$ be the true label and $$\hat{y}_i(\theta) \in (0, 1)$$ be the predicted probability:

$$
L_{\text{BCE}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\bigl[y_i \log(\hat{y}_i(\theta)) + (1 - y_i)\log(1 - \hat{y}_i(\theta))\bigr]
$$

The intuition comes from probability theory.
We are asking: "how surprised would we be by the true label, given our predicted probability?"
When the true label is 1 and we predict $$\hat{y} = 0.99$$, the loss is $$-\log(0.99) \approx 0.01$$ --- we are barely surprised.
When we predict $$\hat{y} = 0.01$$, the loss is $$-\log(0.01) \approx 4.6$$ --- we are very surprised.
This logarithmic penalty grows without bound as the predicted probability approaches the wrong extreme, creating a strong signal to correct confident mistakes.

### Cross-Entropy (CE) for Multi-Class Classification

CE generalizes BCE to **multi-class classification** --- tasks with more than two categories, such as predicting which enzyme class a protein belongs to, or recognizing which of 10 digits appears in a handwritten image.
Let $$C$$ be the number of classes, $$y_c \in \{0, 1\}$$ be the indicator for class $$c$$, and $$\hat{y}_c(\theta)$$ be the predicted probability for class $$c$$:

$$
L_{\text{CE}}(\theta) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c(\theta))
$$

In practice, only one $$y_c$$ is 1 (the true class), so this simplifies to $$-\log(\hat{y}_{\text{true class}}(\theta))$$.
The model is rewarded for assigning high probability to the correct class and penalized (logarithmically) for low probability.

From an information-theoretic perspective, cross-entropy measures the number of extra "bits" of surprise when using the model's predicted distribution instead of the true distribution.
A perfect model has zero cross-entropy with the data.

### Using Loss Functions in PyTorch

```python
import torch
import torch.nn as nn

# Regression: predict melting temperature
criterion = nn.MSELoss()

# Binary classification: soluble vs. insoluble
# BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
# (your model outputs raw scores, not probabilities)
criterion = nn.BCEWithLogitsLoss()

# Multi-class classification: predict secondary structure (H, E, C)
# CrossEntropyLoss combines softmax + CE for numerical stability
# (your model outputs raw scores, called "logits")
criterion = nn.CrossEntropyLoss()
```

A practical note: PyTorch's `BCEWithLogitsLoss` and `CrossEntropyLoss` accept **logits** (raw, unbounded scores) rather than probabilities.
They apply sigmoid or softmax internally, which is more numerically stable than applying these functions yourself and then computing the log.
This means your model's output layer should *not* include a final sigmoid or softmax --- let the loss function handle it.

---

## 2. Mini-Batch Training and Optimizers

The loss function tells us how wrong we are.
The **optimizer** tells us how to improve.
But before we can discuss optimization algorithms, we need to address a more fundamental question: how much data should we use to compute each gradient update?

### The SGD Update Rule

The simplest optimizer is **gradient descent**: update each weight by taking a step in the direction that reduces the loss:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

Here $$\theta_t$$ represents the current parameter values, $$\eta$$ is the **learning rate** (a small positive number controlling step size), $$L(\theta_t)$$ is the loss function from Section 1 ($$L_{\text{MSE}}$$, $$L_{\text{CE}}$$, etc.) evaluated at the current parameters, and $$\nabla_\theta L(\theta_t)$$ is the gradient of the loss with respect to the parameters.

The learning rate is one of the most important hyperparameters[^hyperparameter] in training.
Too small, and learning is painfully slow.
Too large, and training becomes unstable --- the loss oscillates wildly or diverges to infinity.

[^hyperparameter]: A hyperparameter is a setting chosen by the practitioner before training begins (like learning rate, batch size, or number of layers), as opposed to a parameter learned during training (like the weights of a linear layer).

### Mini-Batch Training: Why Not Use All the Data?

Suppose your training set contains 50,000 proteins.
To compute the gradient of the loss over all 50,000 proteins, you would need to run every protein through the network, compute each individual loss, average them, and then backpropagate through the entire computation graph --- all before taking a single step to update the weights.
This is called **full-batch gradient descent**, and it has three problems.

First, it is slow.
You process the entire dataset for a single weight update.
If training takes 1,000 updates to converge, you must scan the full dataset 1,000 times --- and each scan requires holding all intermediate activations in memory.

Second, it is memory-intensive.
For large protein datasets, storing the activations of all 50,000 proteins simultaneously exceeds the memory of any GPU.

Third, the gradient you compute is *too* accurate.
This sounds paradoxical, but a perfect gradient points exactly toward the minimum of the training loss, which may not coincide with the minimum of the true (generalization) loss.
Some noise in the gradient direction actually helps the optimizer explore the loss landscape and avoid sharp, narrow minima that generalize poorly.

The opposite extreme --- computing the gradient from a **single random protein** --- solves the memory and speed problems but introduces too much noise.
The gradient from one protein may point in a completely different direction than the gradient from another.
It also wastes the GPU's parallelism: modern GPUs are designed to process many data points simultaneously, and feeding them one at a time leaves thousands of cores idle.

**Mini-batch stochastic gradient descent** is the standard compromise.
At each training step, we sample a random subset of $$B$$ proteins (the **mini-batch**) from the training set, compute the average loss over that subset, and update the weights using its gradient:

$$
\nabla_\theta L \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(\mathbf{x}_i, y_i; \theta)
$$

This is an unbiased estimate of the full-batch gradient --- its expectation over all possible mini-batches equals the true gradient --- but each individual estimate is noisy.

The word **stochastic** in "stochastic gradient descent" refers to this randomness: at each step, the mini-batch is a random sample, so the gradient is a random variable.
The `shuffle=True` flag in PyTorch's DataLoader is what makes SGD stochastic --- it randomizes which proteins end up in which mini-batch at each epoch.

**Batch size** controls the noise-accuracy tradeoff:

- **Small batches (16--32)** produce noisier gradient estimates. This noise acts as implicit regularization, helping the model generalize. Small batches also use less GPU memory, allowing larger models or longer sequences.
- **Large batches (256--512)** produce smoother, more accurate gradients that converge faster per step. However, each step requires more computation, and the smoother optimization path can lead the model into sharp minima that generalize worse[^sharp-minima].
- **A common starting point** for protein tasks is a batch size of 32 or 64. If your GPU has memory to spare, try 128; if you are running out of memory, drop to 16.

[^sharp-minima]: The relationship between batch size and generalization is an active area of research. The prevailing view is that small-batch training finds "flatter" minima in the loss landscape, which tend to generalize better. See Keskar et al. (2017), "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima."

One **epoch** means one complete pass through the training set.
If the dataset has 50,000 proteins and the batch size is 32, one epoch consists of $$\lceil 50{,}000 / 32 \rceil = 1{,}563$$ mini-batch updates.
After each epoch, the DataLoader reshuffles the dataset, so mini-batches are different across epochs.

### Beyond SGD: Momentum and Adaptive Methods

Vanilla SGD can oscillate in loss landscapes where the surface curves much more steeply in one direction than another.
Several extensions address this.

**Momentum** adds a "velocity" term that accumulates a running average of recent gradients (with a decay factor $$\beta$$, typically 0.9).
The intuition is physical: a ball rolling down the loss landscape builds speed in consistent directions and dampens oscillations where gradients flip sign.

**Adam** [3] adapts the learning rate individually for each parameter by tracking both the mean and variance of recent gradients.
Parameters with consistently large gradients take smaller steps; parameters with small or noisy gradients take larger steps.
Adam works well out of the box for most problems and is the recommended starting point for protein AI projects.

**AdamW** [6] fixes a subtle issue where Adam's adaptive scaling interacts incorrectly with weight decay regularization.
For most practical purposes, AdamW is the preferred optimizer.

```python
# SGD with momentum — simple, interpretable, good for fine-tuning
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam — good default, adaptive learning rates per parameter
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# AdamW — Adam with correct weight decay (the recommended choice)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## 3. The Training Loop: Four Steps, Repeated

Training unfolds as a repeated cycle of four steps.
Each iteration of this cycle processes one **batch** of training examples (a subset of the full dataset).
One pass through the entire dataset is called an **epoch**.

**Step 1: Forward pass.**
Feed a batch of proteins through the model to produce predictions.
Data flows forward through the network, layer by layer.

**Step 2: Compute loss.**
Compare predictions to true labels using the loss function.
This produces a single scalar measuring how wrong we are on this batch.

**Step 3: Backward pass.**
Call `loss.backward()` to compute gradients for all parameters.
Each gradient answers: "how should this weight change to reduce the loss?"

**Step 4: Update weights.**
The optimizer uses the gradients to adjust the weights.
We have now learned from this batch.

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one pass through the dataset."""
    model.train()   # Enable training mode (activates dropout, etc.)
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # Move data to the same device as the model (CPU or GPU)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Step 1: Forward pass — compute predictions
        predictions = model(batch_x)

        # Step 2: Compute loss — measure prediction error
        loss = criterion(predictions, batch_y)

        # Step 3: Backward pass — compute gradients
        optimizer.zero_grad()   # Clear gradients from the previous batch!
        loss.backward()         # Compute new gradients

        # Optional: clip gradients to prevent exploding values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step 4: Update weights — apply gradient descent
        optimizer.step()

        total_loss += loss.item()   # .item() extracts a Python float

    avg_loss = total_loss / len(dataloader)
    return avg_loss
```

A critical detail: `optimizer.zero_grad()` must be called before each backward pass.
By default, PyTorch *accumulates* gradients --- calling `.backward()` multiple times adds to the existing `.grad` values rather than replacing them.
Without zeroing, gradients from previous batches would contaminate the current update[^accumulation].

[^accumulation]: Gradient accumulation is sometimes used intentionally. When GPU memory is too small for a large batch, you can run several small forward/backward passes, accumulate their gradients, and then call `optimizer.step()` once. This simulates training with a larger effective batch size.

---

## 4. Data Loading: Feeding Proteins to Neural Networks

Training neural networks requires showing them thousands or millions of examples.
Getting data from disk into the model efficiently is a surprisingly important engineering problem --- and proteins make it harder than usual.

### Why Data Loading Matters

PyTorch separates data loading into two abstractions: the **Dataset**, which defines how to access a single example, and the **DataLoader**, which groups examples into mini-batches and manages shuffling, padding, and parallel loading.
This separation is a deliberate design decision.

Why not just load everything into a single tensor?
Proteins make this impractical for three reasons.

**Variable lengths.**
A 50-residue peptide and a 1,000-residue enzyme cannot be naively stacked into a single tensor.
Each protein must be individually encoded and padded to a common length within each batch, and the padding length should change from batch to batch to avoid wasting computation.

**Large datasets.**
UniProt contains over 200 million protein sequences.
Even a curated training set of 100,000 proteins may not fit in GPU memory simultaneously.
The DataLoader streams batches from disk or CPU memory to the GPU on demand, so only one batch needs to reside on the GPU at any time.

**GPU efficiency.**
As we discussed in Section 2, mini-batch training requires data to arrive in consistent batches of a fixed size.
The DataLoader handles this automatically: it groups proteins into batches, shuffles the ordering each epoch (making SGD stochastic), and optionally loads data in parallel using background worker processes.

The pipeline looks like this:

```
Raw data (CSV, FASTA)  →  Dataset (encode one protein)  →  DataLoader (batch, shuffle, pad)  →  Model
```

### The `Dataset` Class

PyTorch's `Dataset` class defines how to access individual examples.
You implement two methods:

- `__len__`: returns the total number of examples.
- `__getitem__`: returns one example by its index.

Here is a dataset for protein sequences:

```python
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    """A dataset of protein sequences and their labels."""

    def __init__(self, sequences, labels, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

        # Map each of the 20 standard amino acids to an integer (1–20)
        # Index 0 is reserved for padding
        self.aa_to_idx = {aa: i + 1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Convert amino acid characters to integers
        # Truncate sequences longer than max_len
        encoded = torch.zeros(self.max_len, dtype=torch.long)
        for i, aa in enumerate(seq[:self.max_len]):
            encoded[i] = self.aa_to_idx.get(aa, 0)  # Unknown amino acids → 0

        # Create a mask: 1 for real positions, 0 for padding
        seq_len = min(len(seq), self.max_len)
        mask = torch.zeros(self.max_len, dtype=torch.float)
        mask[:seq_len] = 1.0

        return {
            'sequence': encoded,        # Shape: (max_len,)
            'mask': mask,               # Shape: (max_len,)
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### The `DataLoader`: Batching and Shuffling

The `DataLoader` wraps a dataset and handles three important tasks:

1. **Batching**: groups individual examples into batches for efficient GPU computation.
2. **Shuffling**: randomizes the order of examples each epoch so the model does not learn spurious ordering patterns.
3. **Parallel loading**: uses multiple worker processes to prepare the next batch while the GPU trains on the current one.

```python
# Create dataset objects
train_dataset = ProteinDataset(train_sequences, train_labels)
val_dataset = ProteinDataset(val_sequences, val_labels)

# Wrap in DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,        # Process 32 proteins at a time
    shuffle=True,         # Randomize order each epoch (important for training)
    num_workers=4,        # Use 4 parallel processes for data loading
    pin_memory=True       # Faster CPU → GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,        # Larger batches are fine for evaluation (no gradients stored)
    shuffle=False         # Keep a consistent order for reproducible evaluation
)

# Iterate through batches
for batch in train_loader:
    sequences = batch['sequence']   # Shape: (32, max_len)
    masks = batch['mask']           # Shape: (32, max_len)
    labels = batch['label']         # Shape: (32,)
    # ... feed to model ...
```

### Handling Variable-Length Sequences

Proteins range from tens to thousands of amino acids.
Padding every sequence to the length of the longest protein in the entire dataset wastes computation and memory.
A **custom collate function** can pad each batch only to its own maximum length:

```python
def collate_proteins(batch):
    """Pad sequences in a batch to the length of the longest sequence in that batch."""
    # Find the maximum length in this specific batch
    max_len = max(len(item['sequence']) for item in batch)

    sequences = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item['sequence'])
        sequences[i, :seq_len] = item['sequence']
        masks[i, :seq_len] = 1.0
        labels[i] = item['label']

    return {'sequence': sequences, 'mask': masks, 'label': labels}

# Pass the custom collate function to the DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_proteins)
```

This reduces wasted computation when batches contain only short sequences.

---

## 5. Validation and Overfitting

### The Train/Validation/Test Split

Before training, we divide our data into three non-overlapping subsets, each serving a distinct purpose:

- **Training set** (~80%): the data the model learns from. The model sees these examples during gradient updates.
- **Validation set** (~10%): used to monitor generalization *during* training. We evaluate on this set after each epoch to detect overfitting and to select hyperparameters (learning rate, model size, etc.).
- **Test set** (~10%): used *once*, after all training and hyperparameter selection is complete, to report the final performance estimate. This set must never influence any decision during model development.

Why three sets instead of two?
If we use the validation set to choose hyperparameters (which we always do), the model's performance on the validation set is no longer an unbiased estimate of true generalization.
We may have inadvertently overfit to the validation set by choosing hyperparameters that happen to work well on it.
The test set provides an independent, unbiased estimate.

```python
from sklearn.model_selection import train_test_split

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'],
                                     random_state=42)

# Second split: 50/50 of temp → 10% validation, 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'],
                                   random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### Detecting Overfitting

Training loss alone can be misleading.
A model might memorize the training examples perfectly (achieving near-zero training loss) without learning patterns that generalize to new proteins.
This is **overfitting** --- the central failure mode of machine learning.

The classic signature: training loss decreases steadily, but **validation loss starts increasing** after some point.
The gap between training and validation performance grows over time.

```python
@torch.no_grad()   # No gradient computation needed during evaluation
def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a held-out dataset."""
    model.eval()    # Disable dropout, use running statistics for batch norm
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        total_loss += loss.item()
        all_predictions.append(predictions.cpu())
        all_labels.append(batch_y.cpu())

    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    return avg_loss, all_predictions, all_labels
```

### What Overfitting Looks Like

The figure below shows a concrete example from our solubility predictor (Section 6).

<div class="col-sm-9 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/overfitting_curves.png' | relative_url }}" alt="Training vs validation loss showing overfitting">
    <div class="caption mt-1"><strong>Training and validation loss curves illustrating overfitting.</strong> Training loss decreases steadily, but validation loss begins increasing after ~40 epochs --- the model is memorizing the training data rather than learning generalizable patterns.</div>
</div>

When we train the solubility classifier for 100 epochs, training loss decreases smoothly toward zero --- the model is classifying the training proteins with high confidence.
But the validation loss tells a different story: it decreases initially, plateaus around epoch 30--40, and then starts *increasing*.
Training accuracy reaches 98%, while validation accuracy stalls at 72%.
The 26-percentage-point gap means the model is spending most of its capacity on memorization.

In general, loss curves fall into four patterns:

- **Good**: both curves decrease and stay close together. The model is learning patterns that generalize.
- **Mild overfitting**: training loss keeps decreasing, validation loss plateaus. The model has learned what it can but is starting to memorize noise.
- **Severe overfitting**: training loss approaches zero, validation loss *increases*. The model is memorizing training data at the expense of generalization.
- **Underfitting**: both curves are high and flat. The model is too simple to capture the patterns in the data.

### Why Protein Models Are Especially Prone to Overfitting

Protein datasets are typically small relative to model capacity.
A dataset of 5,000 proteins with a model containing 500,000 parameters means there are 100 parameters per training example --- plenty of room for the model to memorize each protein individually instead of learning general patterns.

Additionally, protein sequences have rich internal structure --- motifs, repeats, compositional biases --- that a model can latch onto as "shortcuts" for the training set without these shortcuts being predictive on new data.
This mirrors the problem in image classification where models sometimes learn to recognize the background (grass behind cows, snow behind wolves) rather than the object itself.

The moment when validation loss stops improving and starts rising is the point of best generalization.
Saving the model at that point --- and discarding later, overfit versions --- is the idea behind **early stopping**, which we discuss in the optional Preliminary Note 4.

### A Complete Training Script

Here is a production-quality training script that combines all the pieces:

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
```

The optional Preliminary Note 4 extends this with learning rate scheduling, early stopping, regularization techniques, and other improvements that address overfitting systematically.

---

## 6. Case Study: Predicting Protein Solubility

Let us bring every concept together in a complete, end-to-end application: predicting whether a protein will be soluble when expressed in *E. coli*.

### Why Solubility Prediction Matters

Expressing recombinant proteins is a core technique in structural biology, biotechnology, and therapeutic development.
When a target protein aggregates into inclusion bodies instead of dissolving in the cytoplasm, downstream applications --- crystallography, assays, drug formulation --- become much harder or impossible.
A computational model that predicts solubility from sequence alone can guide construct design and save weeks of experimental effort.

What makes this problem amenable to machine learning?
Solubility is influenced by sequence-level properties: amino acid composition, charge distribution, hydrophobicity patterns, and the presence of certain sequence motifs.
These patterns are learnable from data.

### The Model Architecture

We build a **1D convolutional neural network** (CNN) that processes amino acid embeddings.
The architecture reflects domain knowledge: convolutional layers with a kernel size of 5 can detect patterns spanning five consecutive amino acids, which is appropriate for capturing local sequence motifs like charge clusters or hydrophobic stretches.

```python
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

# Create Dataset and DataLoader objects
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

### Training

```python
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

| Metric | Question It Answers | Protein Example | General Example |
|---|---|---|---|
| **Accuracy** | What fraction of all predictions are correct? | 85% of solubility predictions correct | 95% of spam classifications correct |
| **Precision** | Of positive predictions, what fraction truly are? | Of proteins predicted soluble, how many truly are? | Of emails flagged spam, how many truly are? |
| **Recall** | Of true positives, what fraction did we detect? | Of truly soluble proteins, how many did we find? | Of actual spam emails, how many did we catch? |
| **F1 Score** | Harmonic mean of precision and recall | Balance between missing soluble proteins and wasting expression experiments | Balance between missing spam and annoying users with false flags |
| **AUC-ROC** | How well does the model separate classes across all thresholds? | Overall ability to distinguish soluble from insoluble | Overall ability to distinguish spam from legitimate email |

The precision-recall tradeoff deserves special attention.
In a drug discovery setting, where expressing each candidate is expensive, a biologist might want **high precision**: "I only want to express proteins that are very likely to be soluble."
By raising the classification threshold from 0.5 to, say, 0.8, we predict fewer proteins as soluble but are more confident in those predictions.
The same logic applies in medical diagnosis: a doctor screening for a rare disease might want high precision to avoid unnecessary invasive follow-up procedures.

Conversely, in a high-throughput screening setting with thousands of candidates, a biologist might prefer **high recall**: "I don't want to miss any potentially soluble protein."
Lowering the threshold to 0.3 captures more true positives at the cost of more false positives.
In the medical setting, a cancer screening program would prefer high recall: missing a true case is far worse than ordering extra tests.

The AUC-ROC summarizes this tradeoff across all possible thresholds.
An AUC of 1.0 means perfect separation; 0.5 means the model is no better than random.

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

1. **Loss functions** quantify prediction errors. MSE for regression, BCE for binary classification, CE for multi-class. Always use PyTorch's numerically stable versions (`BCEWithLogitsLoss`, `CrossEntropyLoss`).

2. **Optimizers** turn gradients into weight updates. SGD with momentum is simple and interpretable; AdamW is the recommended default. The learning rate is the single most impactful hyperparameter.

3. **Training** is a four-step loop --- forward pass, loss computation, backward pass, weight update --- repeated across many batches and epochs. Don't forget `optimizer.zero_grad()` before each backward pass.

4. **Data loading** with `Dataset` and `DataLoader` handles batching, shuffling, and parallel processing. Custom collate functions manage variable-length protein sequences efficiently.

5. **Validation** is essential for detecting overfitting. Always split data into train/validation/test sets. The test set should only be used for final evaluation --- never for making decisions during training.

6. **Evaluation** must use multiple metrics. Accuracy alone is misleading for imbalanced datasets. Understand the precision-recall tradeoff in the context of your biological application.

7. **Optional next step**: Preliminary Note 4 takes the solubility predictor from this note and systematically improves it using regularization, learning rate schedules, sequence-identity splits, and other techniques. It is recommended but not required before proceeding to Lecture 1.

---

## Exercises

These exercises reinforce the concepts from this note.
Each one can be completed in a single Python script or Jupyter notebook.

### Exercise 1: Gradient Accumulation

Implement **gradient accumulation** to simulate a batch size of 128 when your GPU can only fit 32 samples at a time.
The idea: run four forward/backward passes (each with 32 samples), accumulate the gradients, and then call `optimizer.step()` once.

*Key detail:* You should call `optimizer.zero_grad()` only once per effective batch (every 4 mini-batches), not at every step.

### Exercise 2: Optimizer Comparison

Train the `ProteinSolubilityClassifier` from Section 6 three times, each with a different optimizer:
- SGD with momentum 0.9
- Adam with default settings
- AdamW with weight decay 0.01

Plot the training and validation loss curves for all three on the same graph.
Which optimizer converges fastest?
Which achieves the lowest final validation loss?

### Exercise 3: Custom Metric — Matthews Correlation Coefficient

Implement **Matthews Correlation Coefficient** (MCC), a metric that is informative even when classes are severely imbalanced:

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

where $$TP$$, $$TN$$, $$FP$$, and $$FN$$ are the counts of true positives, true negatives, false positives, and false negatives respectively.
Add this metric to the `evaluate_classifier` function and compare it to accuracy on a dataset where 90% of proteins are soluble.

### Exercise 4: Per-Residue Secondary Structure Prediction

Build a 3-layer MLP that predicts secondary structure for each residue in a protein sequence.
The model should take a one-hot encoded sequence of shape `(batch, length, 20)` and output three probabilities (helix, sheet, coil) for each position, giving output shape `(batch, length, 3)`.

*Hints:*
- Use `nn.CrossEntropyLoss()` with the input reshaped to `(batch * length, 3)`.
- Think carefully about how to handle padding positions in the loss computation (see the variable-length sequences section).

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 6--8. Available at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems*, 32.

3. Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

4. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., ... & Fergus, R. (2021). "Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences." *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

5. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323(6088), 533--536.

6. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *Proceedings of ICLR*. (The paper introducing AdamW.)

7. PyTorch Documentation. [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/).
