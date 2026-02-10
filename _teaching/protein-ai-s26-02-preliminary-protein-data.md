---
layout: post
title: "Protein Features and Neural Networks"
date: 2026-03-01
description: "How to turn protein data into numerical features—file formats, sequence encodings, PyTorch tensors—and the neural network architectures that learn representations from them."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 2
preliminary: true
toc:
  sidebar: left
related_posts: false
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>This is Preliminary Note 2 for the Protein &amp; Artificial Intelligence course (Spring 2026), co-taught by <a href="https://sungsoo-ahn.github.io">Prof. Sungsoo Ahn</a> and Prof. Homin Kim at KAIST. It builds on Preliminary Note 1 (Introduction to Machine Learning with Linear Regression). Now that you know what tensors and gradients are, this note answers two questions: how do you turn protein data into numerical features that a neural network can process, and what neural network architectures learn useful representations from those features?</em>
</p>

## Introduction

Proteins are not tensors.
They are amino acid sequences stored in text files and three-dimensional structures stored in coordinate files.
Before any model can learn from protein data, you must solve a translation problem: convert biological data into numerical **features** that a neural network can process.

A note on terminology: **features** are the numerical inputs you construct from raw data --- one-hot encodings, amino acid compositions, and the like.
**Representations** are the internal vectors that a neural network learns in its hidden layers.
Features are hand-crafted; representations are learned.
This note covers both sides: how to build features from protein files, and the neural network architectures that learn representations from them.

### Roadmap

| Section | What You Will Learn | Why It Is Needed |
|---------|---------------------|------------------|
| [Protein File Formats](#1-protein-file-formats) | FASTA and PDB parsing with Biopython and Biotite | Raw biological data must be loaded before it can be encoded |
| [Protein Features](#2-protein-features) | One-hot encoding, PyTorch tensors | Every downstream model needs numerical features as input |
| [Neural Network Architectures](#3-neural-network-architectures) | Neurons, activations, layers, depth, learned representations, `nn.Module` | The function families that learn representations from features and produce predictions |
| [Task Formulations](#4-task-formulations) | Regression, classification, sequence-to-sequence | Different biological questions require different output formats |

### Prerequisites

This note assumes familiarity with Preliminary Note 1: tensors, gradient descent, and the learning cycle (model → loss → gradients → update).

---

## 1. Protein File Formats

Two file formats dominate protein data: FASTA for sequences and PDB for 3D structures.
These are the raw materials from which you build datasets.

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/pdb_ribbon_example.png' | relative_url }}" alt="Simplified ribbon representation of a protein structure">
    <div class="caption mt-1"><strong>Protein 3D structure.</strong> A simplified ribbon representation showing the three main secondary structure elements: alpha-helices (red), beta-strands (blue), and loops/coils (gray). Real structures from the Protein Data Bank encode the precise 3D coordinates of every atom.</div>
</div>

### 1.1 FASTA: The Universal Sequence Format

FASTA is the simplest bioinformatics file format.
Each entry consists of a header line starting with `>`, followed by one or more lines of amino acid sequence:

```
>sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK
MGKIIGIDLGTTNSCVAIMDGTTPRVLENAEGDRTTPSIIAYTQDGETLVGQPAKRQAVT
NPQNTLFAIKRLIGRRFQDEEVQRDVSIMPFKIIAADNGDAWVEVKGQKMAPPQISAEVL
```

The header in this example follows UniProt[^uniprot] conventions.
`sp` indicates Swiss-Prot (the manually curated portion of UniProt).
`P0A6Y8` is the accession number --- a unique identifier for this protein.
`DNAK_ECOLI` is the entry name.
Different databases use different header conventions, so always check the source before writing a parser.

[^uniprot]: UniProt (Universal Protein Resource) is the most comprehensive protein sequence database, containing over 200 million entries. Swiss-Prot is its curated subset with roughly 570,000 entries.

### 1.2 Parsing FASTA with Biopython

Biopython is the standard library for biological file parsing in Python.
A few lines of code are enough to load every sequence from a FASTA file into a dictionary keyed by accession:

```python
from Bio import SeqIO

def load_fasta(filepath):
    """Load all sequences from a FASTA file into a dictionary."""
    sequences = {}
    for record in SeqIO.parse(filepath, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Example usage
seqs = load_fasta("proteins.fasta")
for name, seq in list(seqs.items())[:3]:
    print(f"{name}: {len(seq)} residues, starts with {seq[:10]}...")
```
<div class="caption mt-1"><code>SeqIO.parse()</code> returns an iterator of <code>SeqRecord</code> objects, each containing the sequence and metadata. The iterator design reads one record at a time rather than loading the entire file into memory, which matters when processing databases with millions of entries.</div>

Good parsers read records lazily — one at a time — rather than loading the entire file into memory.
This matters when processing databases with millions of entries.

### 1.3 PDB: The Format for 3D Structures

While FASTA captures the *sequence* of a protein, the **PDB format** captures its *structure*: the three-dimensional positions of every atom.
PDB files use a fixed-width text format[^pdb-history] with a specific column layout:

[^pdb-history]: The PDB file format dates back to 1971, when data was stored on 80-column punch cards. The fixed-width layout reflects this heritage. A newer format, mmCIF, is gradually replacing PDB for large structures, but PDB remains the most widely used format for single-chain proteins.

```
PDB ATOM Record Format (80-column fixed-width):
┌──────┬───────┬──────┬─────┬───┬─────┬────────────────────────────┬──────┬──────┬───┐
│Record│ Atom# │ Name │ Res │Chn│Res# │     X        Y        Z    │ Occ. │B-fac │Elm│
│ Type │       │      │     │ ID│     │                            │      │      │   │
├──────┼───────┼──────┼─────┼───┼─────┼────────────────────────────┼──────┼──────┼───┤
│ATOM  │     1 │  N   │ MET │ A │   1 │  27.340  24.430   2.614   │ 1.00 │ 9.67 │ N │
│ATOM  │     2 │  CA  │ MET │ A │   1 │  26.266  25.413   2.842   │ 1.00 │10.38 │ C │
│ATOM  │     3 │  C   │ MET │ A │   1 │  26.913  26.639   3.531   │ 1.00 │ 9.62 │ C │
└──────┴───────┴──────┴─────┴───┴─────┴────────────────────────────┴──────┴──────┴───┘
 Col:  1-6    7-11   13-16  18-20  22  23-26       31-38  39-46  47-54  55-60 61-66 77-78
```
<div class="caption mt-1">The fixed-width column layout of a PDB ATOM record. Each line contains the atom serial number, atom name, residue name, chain identifier, residue sequence number, (x, y, z) Cartesian coordinates in Ångströms, occupancy, temperature factor, and element symbol.</div>

The key columns are:

| Columns | Content | Example |
|---------|---------|---------|
| 1--6 | Record type | `ATOM` |
| 13--16 | Atom name | `N`, `CA`, `C`, `O` |
| 18--20 | Residue name (3-letter code) | `MET`, `ALA`, `GLY` |
| 22 | Chain identifier | `A` |
| 31--54 | X, Y, Z coordinates (Angstroms) | `27.340  24.430   2.614` |

The **$$\text{C}_\alpha$$** (alpha-carbon) atom is central to protein machine learning.
Every standard amino acid has exactly one $$\text{C}_\alpha$$, located at the backbone's central carbon.
The $$\text{C}_\alpha$$ trace --- one point per residue --- provides a simplified yet informative representation of protein structure.

### 1.4 Parsing PDB with Biotite

Biotite[^biotite] is a modern Python library for structural biology that makes it straightforward to load a PDB file and extract $$\text{C}_\alpha$$ coordinates.

[^biotite]: Biotite is an alternative to Biopython's `Bio.PDB` module, offering a more Pythonic API and better integration with NumPy arrays.

```python
import biotite.structure.io.pdb as pdb
import biotite.structure as struc

def load_pdb(filepath, chain='A'):
    """Load a protein structure from a PDB file."""
    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure(model=1)
    structure = structure[struc.filter_amino_acids(structure)]
    if chain:
        structure = structure[structure.chain_id == chain]
    return structure

# Load Ubiquitin (PDB ID: 1UBQ) and extract Calpha coordinates
structure = load_pdb("1ubq.pdb")
ca_mask = structure.atom_name == "CA"
ca_coords = structure.coord[ca_mask]
print(f"Calpha coordinates shape: {ca_coords.shape}")  # (76, 3)
```
<div class="caption mt-1">After extraction, <code>ca_coords</code> is a standard NumPy array of shape <code>(76, 3)</code> — one row per residue, three columns for x, y, z in Ångströms.</div>

The result is a coordinate array of shape $$(L, 3)$$ — one row per residue, three columns for $$x$$, $$y$$, $$z$$ in Ångströms.
These coordinates are ready to be converted into structural features like distance matrices and contact maps (see the case study in Preliminary Note 4).

### 1.5 Bridging Sequence and Structure

Sometimes you need to extract the amino acid sequence from a structure file --- to verify consistency or because only the PDB is available.
This requires mapping three-letter residue codes (used in PDB files) to single-letter codes (used in FASTA files and sequence models):

```python
import biotite.structure as struc

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def get_sequence_from_structure(structure):
    """Extract the amino acid sequence from a Biotite AtomArray."""
    residue_ids, residue_names = struc.get_residues(structure)
    return ''.join(AA_3TO1.get(name, 'X') for name in residue_names)

seq = get_sequence_from_structure(structure)
print(f"Ubiquitin sequence ({len(seq)} residues): {seq[:20]}...")
```
<div class="caption mt-1">The fallback to <code>'X'</code> handles non-standard amino acids — modified residues, selenomethionine (used in X-ray crystallography), and other variants that frequently appear in real-world PDB files.</div>

Non-standard amino acids — modified residues, selenomethionine, and other variants common in experimental structures — are mapped to a single "unknown" token rather than being silently dropped.

---

## 2. Protein Features

<div class="col-sm-10 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/amino_acid_properties.png' | relative_url }}" alt="The 20 standard amino acids grouped by chemical properties">
    <div class="caption mt-1"><strong>The 20 standard amino acids.</strong> Amino acids are grouped by their side-chain chemistry: nonpolar/hydrophobic (red), polar uncharged (blue), positively charged (green), and negatively charged (orange). Bar heights show approximate molecular weights. These chemical differences determine how each amino acid contributes to protein folding and function.</div>
</div>

The amino acid sequence is the primary structure of a protein --- the linear chain of residues encoded by the gene.
Because sequencing is cheap and fast, sequence data is far more abundant than structure data: UniProt contains over 200 million sequences, while the Protein Data Bank has roughly 200,000 experimentally determined structures.

To feed a protein into a neural network, we must convert its sequence into numerical **features**.
This section introduces one-hot encoding and shows how to package protein features as PyTorch tensors.

### 2.1 One-Hot Encoding

The most straightforward feature is a **one-hot encoding**[^onehot].
Each amino acid at position $$i$$ becomes a binary vector of length 20, with a single 1 indicating which residue is present:

[^onehot]: One-hot encoding is also called "dummy encoding" or "indicator encoding" in the statistics literature.

$$
\mathbf{x}_i \in \{0, 1\}^{20}, \quad \sum_{j=1}^{20} x_{ij} = 1
$$

A full protein of length $$L$$ becomes a feature matrix of shape $$(L, 20)$$.

```python
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(sequence: str) -> torch.Tensor:
    """One-hot encode a protein sequence as a PyTorch tensor."""
    encoding = torch.zeros(len(sequence), 20)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1.0
    return encoding

# Example: encode the first 5 residues of hemoglobin alpha
enc = one_hot_encode("MVLSP")
print(enc.shape)  # torch.Size([5, 20])
```

One-hot encoding preserves the full sequence --- every position and every residue identity.
Its limitation is that it treats every amino acid as equally different from every other.
**Learned embeddings** (Section 3.5) address this by replacing each one-hot vector with a trainable continuous vector.

### 2.2 From Features to PyTorch Tensors

A neural network does not process one protein at a time.
Training requires **batches** --- groups of proteins processed together for computational efficiency.
But proteins have different lengths, so we must **pad** every sequence to a common length before stacking them into a tensor.

After one-hot encoding, each protein of length $$L$$ is a matrix of shape $$(L, 20)$$.
We choose a fixed maximum length $$L_{\max}$$, pad shorter proteins with zero vectors, and truncate longer ones.
Stacking $$B$$ proteins gives a 3D tensor of shape $$(B, L_{\max}, 20)$$.
For an MLP, which expects a flat vector as input, we **flatten** each protein's matrix into a single vector of dimension $$L_{\max} \times 20$$.

```python
from torch.utils.data import TensorDataset, DataLoader

# Suppose we have protein sequences and their solubility labels
sequences = ["MGKIIGIDLG...", "MSKGEELFTG...", "MVLSPADKTN..."]
labels = [1, 1, 0]  # 1 = soluble, 0 = insoluble

# One-hot encode and pad to a fixed maximum length
max_len = 100  # Truncate longer proteins, pad shorter ones with zeros
features = torch.zeros(len(sequences), max_len, 20)
for i, seq in enumerate(sequences):
    L = min(len(seq), max_len)
    features[i, :L] = one_hot_encode(seq[:L])
labels = torch.tensor(labels, dtype=torch.long)

print(features.shape)  # torch.Size([3, 100, 20]) — 3 proteins, padded

# Flatten each protein's matrix into a single vector for the MLP
features_flat = features.view(len(sequences), -1)
print(features_flat.shape)  # torch.Size([3, 2000]) — ready for MLP input

# Wrap in a dataset and data loader
dataset = TensorDataset(features_flat, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop iterates over batches
for batch_features, batch_labels in loader:
    # batch_features shape: (batch_size, 2000)
    # batch_labels shape:   (batch_size,)
    pass  # feed to neural network (Section 3)
```

A data loader handles shuffling, batching, and iterating: each iteration yields a batch of flattened feature vectors and their corresponding labels, ready to be passed through a neural network.
Note that flattening discards the sequential structure --- the MLP treats position 1 and position 100 as unrelated inputs.
Preliminary Note 4 introduces convolutional networks that exploit the sequential ordering.

---

## 3. Neural Network Architectures

A feature vector is not a prediction.
To go from a one-hot encoded sequence to a solubility score, we need a function --- and a linear model $$\hat{y} = \mathbf{W}\mathbf{x} + b$$ is limited to straight-line relationships.
Neural networks overcome this by composing simple operations into powerful function approximators that learn internal **representations** --- abstract, task-relevant encodings of the input --- in their hidden layers.

### 3.1 The Single Neuron

The fundamental unit is the **artificial neuron**.
It takes multiple input features, computes a weighted sum, adds a bias, and applies a nonlinear function:

$$
\text{output} = \sigma(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)
$$

Consider predicting protein solubility from sequence features.
The input features $$x_1, x_2, \ldots, x_n$$ are numerical values derived from the protein sequence.
The weights $$w_1, w_2, \ldots, w_n$$ determine how much each feature contributes to the solubility score.
The bias $$b$$ shifts the decision boundary, and the function $$\sigma$$ is called an **activation function**; it introduces nonlinearity, allowing the neuron to model relationships that are not straight lines.

When $$\sigma$$ is the sigmoid function, the single neuron computes:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

This is exactly **logistic regression** --- the classification analogue of the linear regression we saw in Preliminary Note 1.
The neuron's output is a probability, and the **decision boundary** is the set of points where $$\mathbf{w}^T \mathbf{x} + b = 0$$ (i.e., where the output probability is exactly 0.5).
In 2D feature space, this boundary is a line; in the high-dimensional space of our flattened sequence features, it is a hyperplane.
Points on one side are classified as soluble, points on the other as insoluble.

A single neuron can only learn linear decision boundaries.
This is sufficient for linearly separable problems but fails when the boundary between classes is curved or disconnected --- which is why we need multiple layers.

<div class="col-sm mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/mermaid/s26-02-preliminary-protein-data_diagram_0.png' | relative_url }}" alt="s26-02-preliminary-protein-data_diagram_0">
</div>

### 3.2 Layers: Many Neurons in Parallel

A single neuron is limited.
But arrange many neurons in parallel --- each receiving the same input features but with *different* weights --- and you get a **layer**.
With 64 neurons processing a $$d$$-dimensional input, you get a 64-dimensional **representation** --- 64 different weighted combinations of the input features.
This can be written compactly as a matrix equation:

$$
\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})
$$

Tracing the dimensions explicitly:

$$
\underbrace{\mathbf{W}}_{64 \times d} \underbrace{\mathbf{x}}_{d \times 1} + \underbrace{\mathbf{b}}_{64 \times 1} = \underbrace{\mathbf{z}}_{64 \times 1} \xrightarrow{\sigma} \underbrace{\mathbf{h}}_{64 \times 1}
$$

Each row of $$\mathbf{W}$$ is one neuron's weight vector. Row $$k$$ computes the dot product $$\mathbf{w}_k^T \mathbf{x} + b_k$$, producing one scalar. Stack 64 such scalars and you get the 64-dimensional pre-activation vector $$\mathbf{z}$$, which becomes $$\mathbf{h}$$ after applying $$\sigma$$ element-wise.

This is a **fully connected layer** (also called a **dense layer** or **linear layer**).
The total number of parameters is $$64d + 64$$ (weights plus biases).
For our padded protein sequences with $$d = L_{\max} \times 20 = 2{,}000$$, that is already $$128{,}064$$ parameters in a single layer.

The function $$\sigma$$ is called an **activation function** --- a nonlinear function applied element-wise after each linear transformation.
Without it, stacking layers would be pointless: a linear transformation followed by another linear transformation is just a single linear transformation.
The most common choice is **ReLU**: $$\text{ReLU}(z) = \max(0, z)$$, which simply zeros out negative values and passes positive values through unchanged.
Other activations include **sigmoid** ($$\sigma(z) = 1/(1 + e^{-z})$$, used at the output for binary probabilities), **GELU** (a smooth variant of ReLU used in transformers), and **softmax** (normalizes a vector into a probability distribution, used for multi-class output).

### 3.3 Why Depth Matters: The Power of Composition

<div class="col-sm-8 mt-3 mb-3 mx-auto">
    <img class="img-fluid rounded" src="{{ '/assets/img/teaching/protein-ai/mermaid/s26-02-preliminary-protein-data_diagram_1.png' | relative_url }}" alt="A two-hidden-layer MLP for solubility prediction">
    <div class="caption mt-1"><strong>A two-hidden-layer MLP for solubility prediction.</strong> The flattened padded sequence is transformed through two hidden layers of decreasing width (64, 32), each applying a linear transformation followed by an activation function. The output layer produces two scores (soluble vs. insoluble).</div>
</div>

The power of neural networks comes from stacking multiple layers.
An $$n$$-layer network, called a **multi-layer perceptron (MLP)**, composes $$n$$ transformations:

$$
\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\mathbf{h}_l = \sigma(\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l), \quad l = 2, \ldots, n
$$

$$
\hat{y} = \mathbf{W}_{n+1} \mathbf{h}_n + \mathbf{b}_{n+1}
$$

Each hidden layer takes the previous layer's output $$\mathbf{h}_{l-1}$$ as input, applies a linear transformation followed by a nonlinear activation, and produces a new representation $$\mathbf{h}_l$$.
The final layer typically has no activation (for regression) or a sigmoid/softmax (for classification).
Deeper networks can represent complex functions efficiently because each layer builds more abstract features from the previous layer's output.

For a protein solubility predictor, this compositional hierarchy might look like:

- **Layer 1** detects which positions and amino acid identities are informative, extracting basic patterns from the flattened sequence.
- **Layer 2** combines these into higher-level patterns: local composition trends, position-specific signals.
- **Output layer** maps these abstract representations to a solubility prediction.

In practice, deeper networks are not always better.
Very deep networks can be harder to train (gradients may vanish or explode as they propagate through many layers).
Techniques like residual connections, normalization layers, and careful initialization have made training deep networks practical.

### 3.4 Beyond Flat Features: Specialized Architectures

The fully connected networks above treat the input as a flat vector of features --- every input element connects to every neuron.
This works for fixed-size feature vectors, but proteins have **sequential structure** that a flat vector ignores.

**Convolutional Neural Networks (CNNs)** exploit this structure by sliding a small filter along the sequence, detecting local patterns like charge clusters or hydrophobic stretches.
We build a 1D-CNN for solubility prediction in the case study (Preliminary Note 4).

**Transformers** and **Graph Neural Networks (GNNs)** are more powerful architectures that we cover in Lecture 1.
Transformers use attention to let every position attend to every other position --- the backbone of protein language models (Lecture 3) and AlphaFold (Lecture 4).
GNNs operate on graph structures where residues are nodes and edges connect spatial neighbors --- the basis of ProteinMPNN (Lecture 6).

The key takeaway: **the choice of architecture encodes an inductive bias** --- an assumption about the structure of the problem --- and matching the architecture to the data is one of the most important design decisions in protein AI.

### 3.5 Learned Representations: Embeddings

One-hot encodings are hand-crafted --- we designed them before training.
The alternative is to let the network learn its own representations directly from data.

An **embedding layer** maps each amino acid index to a trainable vector of dimension $$d$$.
These vectors start random and are adjusted during training so that amino acids with similar roles get similar representations --- all without any explicit biochemical supervision.

```python
import torch.nn as nn

# 21 possible tokens (20 amino acids + 1 for unknown), each mapped to a 64-dim vector
embedding = nn.Embedding(num_embeddings=21, embedding_dim=64)

# Input: integer-encoded sequence (batch of 2 proteins, length 5)
indices = torch.tensor([[0, 2, 3, 5, 10],
                         [1, 4, 6, 8, 12]])
representations = embedding(indices)  # Shape: (2, 5, 64)
```

After training, the learned representations often reveal biologically meaningful structure: hydrophobic residues cluster together, charged residues form their own group, and aromatic residues occupy a distinct region.
This is the strategy behind **protein language models** such as ESM, which we cover in Lecture 3.

### 3.6 `nn.Module`: PyTorch's Building Block

In PyTorch, every neural network component inherits from `nn.Module`.
This base class provides machinery for tracking parameters, moving to GPU, saving and loading models, and more.
A custom network specifies two things: what layers exist, and how data flows through them.

```python
import torch
import torch.nn as nn

class SolubilityPredictor(nn.Module):
    """A feedforward network for protein solubility prediction."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # First fully connected layer: input features → hidden representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Activation function
        self.relu = nn.ReLU()
        # Second fully connected layer: hidden representation → prediction
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        h = self.fc1(x)    # Features → hidden representation
        h = self.relu(h)   # Nonlinear activation
        out = self.fc2(h)  # Representation → prediction
        return out

# Flattened padded sequences: max_len=100 × 20 amino acids = 2000-dim input
model = SolubilityPredictor(input_dim=2000, hidden_dim=64, output_dim=2)

# Pass a batch of 32 proteins through the model
features = torch.randn(32, 2000)  # 32 proteins, flattened padded sequences
predictions = model(features)    # Shape: (32, 2) — scores for [soluble, insoluble]
print(predictions.shape)
```
<div class="caption mt-1">The <code>__init__</code> method defines what layers exist; <code>forward</code> defines how data flows through them. Note the distinction: <code>x</code> is the input <strong>features</strong>, <code>h</code> is the hidden <strong>representation</strong> (what the network learns), and <code>out</code> is the prediction. PyTorch handles the backward pass automatically — you only specify the forward computation.</div>

The data flow follows the pattern from Section 3.3: input features pass through a linear layer and activation to produce a hidden representation, then a second linear layer maps that representation to a prediction.
PyTorch handles the backward pass (gradient computation) automatically — you only specify the forward computation.

### 3.7 Common Layer Types

The most common layer types in protein AI are linear layers, activations, normalization, dropout, and embeddings.

```python
# --- Linear layer ---
# Computes y = Wx + b. The fundamental building block.
nn.Linear(in_features=20, out_features=64)

# --- Activation functions ---
nn.ReLU()        # max(0, x) — simple, effective, the default choice
nn.GELU()        # Smooth approximation of ReLU, used in transformer models
nn.Sigmoid()     # Squashes output to (0, 1), useful for binary probabilities
nn.Softmax(dim=-1)  # Normalizes a vector to sum to 1 (probability distribution)

# --- Normalization layers ---
# Stabilize training by normalizing intermediate activations
nn.LayerNorm(normalized_shape=64)    # Normalizes across features (used in transformers)
nn.BatchNorm1d(num_features=64)      # Normalizes across the batch dimension

# --- Dropout ---
# Randomly zeros out neurons during training to prevent overfitting
nn.Dropout(p=0.1)  # Each neuron has a 10% chance of being turned off per forward pass

# --- Embedding layer ---
# Maps discrete tokens (like amino acid indices) to continuous vectors
# 21 possible tokens (20 amino acids + 1 padding), each mapped to a 64-dim vector
nn.Embedding(num_embeddings=21, embedding_dim=64)
```
<div class="caption mt-1">PyTorch provides pre-built implementations of all these layer types. For simple architectures with no branching, <code>nn.Sequential</code> provides a compact shorthand. To count parameters: <code>sum(p.numel() for p in model.parameters())</code>. To save/load weights: <code>torch.save(model.state_dict(), path)</code> and <code>model.load_state_dict(torch.load(path))</code>.</div>

---

## 4. Task Formulations

Different biological questions require different mathematical formulations.
Getting this mapping right is the first design decision in any project.

| Formulation | Output | Protein Example | General Example |
|---|---|---|---|
| **Regression** | A continuous number | Sequence → melting temperature (62.5 °C) | Photo → person's age (34 years) |
| **Binary classification** | One of two categories | Sequence → soluble / insoluble | Email → spam / not spam |
| **Multi-class classification** | One of $$C$$ categories | Sequence → enzyme class (oxidoreductase) | Handwritten digit → 0--9 |
| **Multi-label classification** | Multiple labels per protein | Sequence → {kinase, membrane, signaling} | Photo → {outdoor, sunny, beach} |
| **Sequence-to-sequence** | One output per position | Sequence → secondary structure (H/E/C) per residue | Sentence → part-of-speech tag per word |

**Regression** predicts a continuous number --- melting temperature, binding affinity --- and measures error as the gap between prediction and ground truth.
**Binary classification** (soluble vs. insoluble, enzyme vs. non-enzyme) outputs a probability between 0 and 1 and applies a threshold to decide.
**Multi-class classification** extends this to $$C$$ categories, outputting a full probability distribution.
**Multi-label classification** handles proteins with multiple simultaneous labels (e.g., both kinase *and* membrane-bound) by predicting each label independently.
**Sequence-to-sequence** tasks produce one output per residue --- secondary structure (helix/sheet/coil) or disorder prediction.

---

## Key Takeaways

1. **FASTA** stores sequences; **PDB** stores 3D structures. Biopython handles sequence parsing; Biotite handles structure parsing. These are the entry points to any protein ML pipeline.

2. **Features** are hand-crafted numerical inputs: one-hot encoding converts each amino acid into a binary vector, preserving the full sequence. Features are converted to PyTorch tensors for training.

3. **Representations** are what neural networks learn internally. Embedding layers learn amino acid representations from data; hidden layers learn task-specific representations. Protein language models (Lecture 3) produce powerful learned representations.

4. **Neural networks** are compositions of simple layers: linear transformations followed by nonlinear activations. Depth enables hierarchical representation learning --- each layer builds more abstract representations from the previous layer's output.

5. **Task formulations** map biological questions to mathematical outputs: regression for continuous values, classification for categories, sequence-to-sequence for per-residue predictions.

6. **Next up**: Preliminary Note 3 puts these features and architectures to work in a complete training pipeline --- loss functions, optimizers, and validation.

---

## References

1. Cock, P.J., Antao, T., Chang, J.T., et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*, 25(11), 1422--1423.

2. Kunzmann, P. and Hamacher, K. (2018). Biotite: a unifying open source computational biology framework in Python. *BMC Bioinformatics*, 19, 346.

3. Berman, H.M., Westbrook, J., Feng, Z., et al. (2000). The Protein Data Bank. *Nucleic Acids Research*, 28(1), 235--242.

4. Rives, A., Meier, J., Sercu, T., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

