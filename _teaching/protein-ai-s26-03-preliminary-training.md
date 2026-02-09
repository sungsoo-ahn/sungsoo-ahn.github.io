---
layout: post
title: "Training Neural Networks for Protein Science"
date: 2026-03-02
description: "Loss functions, optimizers, the training loop, data loading, validation, overfitting, and backpropagation—everything you need to train a protein model."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 3
preliminary: true
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;"><em>This is Preliminary Note 3 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim at KAIST. It continues directly from Preliminary Notes 1 and 2. By the end of this note, you will understand every component of the training process and be ready for the case study in Preliminary Note 4.</em></p>

## Introduction

In Preliminary Notes 1 and 2 you learned how to represent protein data as tensors and how neural networks transform those tensors into predictions.
But a network fresh off the assembly line knows nothing --- its weights are random numbers, and its predictions are meaningless.
This note is about taking that raw network and making it *learn*.

We cover every component of the training process: the loss functions that quantify mistakes, the optimizers that correct them, the training loop that orchestrates the process, and the data loading machinery that feeds proteins to the network efficiently.
We then discuss validation, overfitting, and the bias-variance tradeoff that governs model design.
The note concludes with an advanced section on backpropagation for those who want to understand how gradients flow through multi-layer networks.

Preliminary Note 4 applies all of these components in a complete case study: predicting protein solubility in *E. coli*.

### Roadmap

| Section | Topic | Why You Need It |
|---|---|---|
| 1 | Loss Functions | Different prediction tasks require different ways of measuring error |
| 2 | Mini-Batch Training and Optimizers | Why we train on batches, what "stochastic" means, and the algorithms that turn gradients into weight updates |
| 3 | The Training Loop | The four-step cycle that turns data into knowledge |
| 4 | Data Loading for Proteins | Efficient batching, shuffling, and handling of variable-length sequences |
| 5 | Validation, Overfitting, and the Bias-Variance Tradeoff | How to detect when your model is memorizing rather than learning |
| 6 | Backpropagation (Advanced) | How gradients flow through multi-layer networks via the chain rule |

### Prerequisites

This note assumes familiarity with Preliminary Notes 1 and 2: tensors, `nn.Module`, activation functions, autograd, gradient descent, and protein features.

---

## 1. Loss Functions: Measuring Mistakes

In Preliminary Note 2, we built a neural network that takes amino acid composition features as input and outputs scores for protein solubility classes.
But the network's weights are random --- its output is meaningless noise.
To make it learn, we need a way to quantify *how wrong* its predictions are for each training example, so that gradient descent can push the weights in the right direction.

The **loss function** (also called a **cost function** or **objective function**) does exactly this: it produces a single number measuring prediction quality.
Zero means perfect predictions; larger values mean worse predictions.
The choice of loss function depends on the type of prediction task --- solubility classification needs a different loss than melting temperature regression.

We introduced $$L_{\text{MSE}}$$ briefly in Preliminary Note 1 as our first loss function. Here we examine it alongside the classification losses in a systematic treatment.

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

### Gradient Descent

The simplest optimizer is **(full-batch) gradient descent**: compute the loss over the *entire* training set, then update each weight by taking a step in the direction that reduces it:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

Here $$\theta_t$$ represents the current parameter values, $$\eta$$ is the **learning rate** (a small positive number controlling step size), $$L(\theta_t)$$ is the loss function from Section 1 evaluated over *all* training examples, and $$\nabla_\theta L(\theta_t)$$ is its gradient with respect to the parameters.
This is called "full-batch" because the gradient uses every example in the dataset.
As we will see next, this is impractical for real datasets --- we need the *stochastic* variant.

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

Second, it is memory-intensive.
For large protein datasets, storing the activations of all 50,000 proteins simultaneously exceeds the memory of any GPU.

Third, the gradient you compute is *too* accurate.
This sounds paradoxical, but a perfect gradient points exactly toward the minimum of the training loss, which may not coincide with the minimum of the true (generalization) loss.
Some noise in the gradient direction actually helps the optimizer explore the loss landscape and avoid sharp, narrow minima that generalize poorly.

**Mini-batch stochastic gradient descent** is the standard compromise.
At each training step, we sample a random subset of $$B$$ proteins (the **mini-batch**) from the training set, compute the average loss over that subset, and update the weights using its gradient:

$$
\nabla_\theta L \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(\mathbf{x}_i, y_i; \theta)
$$

The word **stochastic** in "stochastic gradient descent" refers to this randomness: at each step, the mini-batch is a random sample, so the gradient is a random variable.
The `shuffle=True` flag in PyTorch's DataLoader is what makes SGD stochastic --- it randomizes which proteins end up in which mini-batch at each epoch.

**Batch size** controls the noise-accuracy tradeoff:

- **Small batches (16--32)** produce noisier gradient estimates. This noise acts as implicit regularization, helping the model generalize. Small batches also use less GPU memory, allowing larger models or longer sequences.
- **Large batches (256--512)** produce smoother, more accurate gradients that converge faster per step. However, each step requires more computation, and the smoother optimization path can lead the model into sharp minima that generalize worse.
- **A common starting point** for protein tasks is a batch size of 32 or 64. If your GPU has memory to spare, try 128; if you are running out of memory, drop to 16.

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
Mini-batch training requires data to arrive in consistent batches of a fixed size.
The DataLoader handles this automatically: it groups proteins into batches, shuffles the ordering each epoch (making SGD stochastic), and optionally loads data in parallel using background worker processes.

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

## 5. Validation, Overfitting, and the Bias-Variance Tradeoff

### The Bias-Variance Tradeoff

Before discussing how to detect overfitting in practice, let us formalize the tension introduced in Preliminary Note 1.
Why not simply use the most powerful model available?

The answer involves a fundamental tradeoff.

**Bias** refers to error caused by a model being too simple to capture the true patterns in the data.
A linear model predicting solubility from just the protein's length has high bias: it systematically misses the real relationship because the true function is far more complex than a straight line.

**Variance** refers to error caused by a model being too sensitive to the specific training data.
A very complex model might fit the training data perfectly, including its noise and idiosyncrasies, but produce wildly different predictions when trained on a different random sample.

The sweet spot lies between these extremes.
We want a model complex enough to capture the true patterns (low bias) but constrained enough to avoid fitting noise (low variance).
In practice, this means:

- **Too simple** (high bias): the model underfits --- training performance is already poor.
- **Too complex** (high variance): the model overfits --- training performance is excellent, but validation performance is much worse.
- **Just right**: both training and validation performance are good, and they are close to each other.

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

<div class="col-sm-9 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/overfitting_curves.png' | relative_url }}" alt="Training vs validation loss showing overfitting">
    <div class="caption mt-1"><strong>Training and validation loss curves illustrating overfitting.</strong> Training loss decreases steadily, but validation loss begins increasing after ~40 epochs --- the model is memorizing the training data rather than learning generalizable patterns.</div>
</div>

In general, loss curves fall into four patterns:

- **Good**: both curves decrease and stay close together. The model is learning patterns that generalize.
- **Mild overfitting**: training loss keeps decreasing, validation loss plateaus. The model has learned what it can but is starting to memorize noise.
- **Severe overfitting**: training loss approaches zero, validation loss *increases*. The model is memorizing training data at the expense of generalization.
- **Underfitting**: both curves are high and flat. The model is too simple to capture the patterns in the data.

### Why Protein Models Are Especially Prone to Overfitting

Protein datasets are typically small relative to model capacity.
A dataset of 5,000 proteins with a model containing 500,000 parameters means there are 100 parameters per training example --- plenty of room for the model to memorize each protein individually instead of learning general patterns.

The moment when validation loss stops improving and starts rising is the point of best generalization.
Saving the model at that point --- and discarding later, overfit versions --- is the idea behind **early stopping**, which we discuss in Preliminary Note 4 alongside other practical techniques for addressing overfitting.

---

## 6. Backpropagation (Advanced)

*This section is optional for a first reading. It explains how PyTorch computes gradients through multi-layer networks. You can safely skip it and return later.*

In Preliminary Note 1 we saw PyTorch compute gradients for a simple linear model.
But what about deeper networks with many layers, where the output of one layer feeds into the next?
To compute how a weight in an early layer affects the final loss, we need the **chain rule** from calculus.

### The Chain Rule

We need $$\nabla_\theta L(\theta)$$ --- the derivative of the loss with respect to every parameter in $$\theta$$.
But a parameter in an early layer does not appear directly in the loss formula; it influences the loss through a chain of intermediate computations: $$\theta_k \to z \to a \to \cdots \to L$$.
The chain rule lets us decompose this dependency.
For a parameter $$\theta_k$$ that affects the loss through an intermediate variable $$z$$:

$$
\frac{\partial L}{\partial \theta_k} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial \theta_k}
$$

In words: to find how $$\theta_k$$ affects $$L$$, multiply how $$z$$ affects $$L$$ by how $$\theta_k$$ affects $$z$$.
Applied recursively backward through the network --- from the loss, through each layer, all the way to the first parameter --- this gives us $$\nabla_\theta L(\theta)$$.
This recursive backward application of the chain rule is the **backpropagation** algorithm[^backprop].

[^backprop]: Backpropagation was popularized for neural network training by Rumelhart, Hinton, and Williams in 1986, though the mathematical idea of reverse-mode automatic differentiation predates it.

### The Computation Graph

The following diagram shows a simple computation graph and how gradients flow backward through it during backpropagation.

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/mermaid/s26-03-preliminary-training_diagram_0.png' | relative_url }}" alt="s26-03-preliminary-training_diagram_0">
</div>

### PyTorch Autograd

The remarkable thing about PyTorch is that you never need to implement backpropagation yourself.
You define only the forward computation, and PyTorch automatically builds a computational graph that tracks every operation.
When you call `.backward()`, it traverses this graph in reverse, computing all gradients.

Here is a simple example to see autograd at work:

```python
# Create a tensor and tell PyTorch to track operations on it
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward computation: y_i = x_i^2 + 3*x_i
y = x ** 2 + 3 * x

# The loss must be a scalar (single number) for .backward()
z = y.sum()

# Backward pass: compute dz/dx for each element of x
z.backward()

# Gradients are stored in the .grad attribute
print(x.grad)  # tensor([7., 9.])
```

Let us verify this by hand.
We have $$z = \sum_i (x_i^2 + 3x_i)$$, so the partial derivative is $$\frac{\partial z}{\partial x_i} = 2x_i + 3$$.
For $$x_1 = 2$$: $$2(2) + 3 = 7$$.
For $$x_2 = 3$$: $$2(3) + 3 = 9$$.
PyTorch computed exactly these values --- automatically.

---

## Key Takeaways

1. **Loss functions** quantify prediction errors. MSE for regression, BCE for binary classification, CE for multi-class. Always use PyTorch's numerically stable versions (`BCEWithLogitsLoss`, `CrossEntropyLoss`).

2. **Optimizers** turn gradients into weight updates. SGD with momentum is simple and interpretable; AdamW is the recommended default. The learning rate is the single most impactful hyperparameter.

3. **Training** is a four-step loop --- forward pass, loss computation, backward pass, weight update --- repeated across many batches and epochs. Don't forget `optimizer.zero_grad()` before each backward pass.

4. **Data loading** with `Dataset` and `DataLoader` handles batching, shuffling, and parallel processing. Custom collate functions manage variable-length protein sequences efficiently.

5. **The bias-variance tradeoff** governs model design: too simple models underfit (high bias), too complex models overfit (high variance). The train/validation/test split is essential for detecting overfitting.

6. **Backpropagation** uses the chain rule to compute gradients through multi-layer networks. PyTorch automates this entirely --- you only define the forward pass.

7. **Next up**: Preliminary Note 4 applies all of these components in a complete case study --- predicting protein solubility --- including evaluation, sequence-identity splits, class imbalance, and debugging.

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapters 6--8. Available at [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/).

2. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems*, 32.

3. Kingma, D. P. & Ba, J. (2015). "Adam: A Method for Stochastic Optimization." *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

4. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., ... & Fergus, R. (2021). "Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences." *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

5. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning Representations by Back-Propagating Errors." *Nature*, 323(6088), 533--536.

6. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *Proceedings of ICLR*. (The paper introducing AdamW.)

7. PyTorch Documentation. [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/).
