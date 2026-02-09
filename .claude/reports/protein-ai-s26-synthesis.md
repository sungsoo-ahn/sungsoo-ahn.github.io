# Protein & AI Lecture Notes — Synthesis Report (Iteration 1)

**Date:** 2026-02-10
**Scope:** Preliminary Notes P1–P4
**Focus:** Expand math/concepts, trim redundant code, add figures, fix notation

---

## Per-Note Reports

### Note P1: Introduction to Machine Learning with Linear Regression

**Status:** processed
**Changes made:**
- Added formal optimization definition: theta* = argmin L(theta)
- Expanded MSE gradient derivation step-by-step
- Added geometric intuition for gradient direction (steepest ascent/descent)
- Added computational graph worked example: forward/backward pass
- Trimmed tensor creation code (from 24 lines to 6)
- Trimmed tensor operations code (from 12 lines to 6)
- Added tensor dimensions diagram (scalar to vector to matrix to 3D tensor)
- Added computational graph diagram
- Changed loss notation from L to calligraphic L to avoid conflict with sequence length L

**Concepts introduced:** 14
**Notation symbols introduced:** 9

---

### Note P2: Protein Features and Neural Networks

**Status:** processed
**Changes made:**
- Bridged one-hot encoding limitation to learned embeddings (equal-distance problem)
- Added logistic regression / decision boundary explanation for single neuron
- Added activation function derivative formulas (sigmoid, ReLU)
- Added quantitative vanishing gradient explanation (0.25^10 approx 10^-6)
- Added formal UAT statement with Cybenko 1989 citation
- Added explicit dimension analysis for fully connected layer
- Trimmed nn.Sequential and parameter management sections (from 30 lines to 3)
- Added activation functions figure (sigmoid, ReLU, GELU with derivatives)

**Concepts introduced:** 22
**Notation symbols introduced:** 10

---

### Note P3: Training Neural Networks for Protein Science

**Status:** processed
**Changes made:**
- Added maximum-likelihood derivation of cross-entropy (BCE = negative log-likelihood)
- Formalized momentum update rule with equations
- Formalized Adam optimizer with full update equations (m_t, v_t, bias correction)
- Added formal bias-variance decomposition: Error = Bias^2 + Variance + Noise
- Added 2-layer backpropagation worked example (chain rule through W^(1))
- Trimmed Dataset class (from 30 lines to 3)
- Trimmed collate function (from 15 lines to 3)
- Trimmed evaluate function (from 20 lines to 3)
- Added momentum comparison figure (SGD vs SGD+momentum)
- Added bias-variance tradeoff figure (classic U-curve)
- Clarified relationship between per-example loss ell and aggregate loss calL
- Changed loss notation from L to calL

**Concepts introduced:** 14
**Notation symbols introduced:** 4

---

### Note P4: Case Study: Predicting Protein Solubility

**Status:** processed
**Changes made:**
- Rewrote 1D convolution formula with multi-channel notation
- Added padding formula (L_out = L_in + 2*padding - k + 1)
- Added masked global average pooling formula
- Expanded structural features with biological rationale (contact density, RCO, Rg, SS)
- Added formal metric definitions: precision, recall, F1, AUC-ROC
- Added formal late fusion treatment (concatenation of representations)
- Trimmed data preparation code (from 30 lines to 3)
- Trimmed debugging code (from 25 lines to checklist)
- Added 1D convolution sliding window diagram
- Added precision-recall tradeoff figure

**Concepts introduced:** 6
**Notation symbols introduced:** 0

---

## Cross-Note Consistency

### Notation Table

| Symbol | Meaning | Source | Conflicts |
|--------|---------|--------|-----------|
| theta | Learnable parameters | P1 S1 | None |
| calL | Loss function | P1 S4 | Resolved: was L, conflicted with sequence length |
| L | Sequence length | P2 S2.1 | None (after fix) |
| eta | Learning rate | P1 S4 | None |
| B | Mini-batch size | P3 S2 | None |
| n | Full dataset size | P1 S1 | None |
| C | Number of classes | P3 S1 | Minor: also used as C_out for conv channels in P4 |
| d | Embedding/feature dimension | P2 S2.2 | None |
| W | Weight matrix | P1 S4 | None |
| b | Bias term | P1 S4 | None |
| ell | Per-example loss | P3 S2 | None; relationship to calL clarified |

### Issues Resolved

1. CRITICAL: L overloaded for loss and sequence length - Changed to calL for loss across all notes
2. HIGH: ell vs calL relationship unclear - Added explicit definition
3. All cross-references verified correct
4. All concept dependencies satisfied (each concept defined before first use)

### Remaining Minor Issues

- C used for both number of classes (P3) and output channels (P4, with subscript). Not confusing in context.
- Secondary structure fractions mentioned in P4 feature table but not computed in the code. Acceptable as conceptual.

---

## New Figures Added

| Figure | Type | Note | Description |
|--------|------|------|-------------|
| s26-01-tensor-dimensions.png | Mermaid | P1 | Scalar to vector to matrix to 3D tensor |
| s26-01-computational-graph.png | Mermaid | P1 | Forward/backward pass for loss computation |
| activation_functions.png | matplotlib | P2 | Sigmoid, ReLU, GELU with derivatives |
| momentum_comparison.png | matplotlib | P3 | SGD vs SGD+momentum on elongated loss landscape |
| bias_variance_tradeoff.png | matplotlib | P3 | Classic U-shaped bias/variance/total error |
| s26-04-conv1d-sliding.png | Mermaid | P4 | 1D convolution sliding window |
| precision_recall_curves.png | matplotlib | P4 | Precision-recall tradeoff with use cases |

---

## Code-to-Concept Ratio

| Note | Before | After | Change |
|------|--------|-------|--------|
| P1 | ~45 lines of code | ~25 lines | -44% |
| P2 | ~100 lines of code | ~70 lines | -30% |
| P3 | ~85 lines of code | ~45 lines | -47% |
| P4 | ~130 lines of code | ~95 lines | -27% |

Math/prose expanded in all notes; code trimmed to essential examples.

---

## Jekyll Writing Rules Updated

- Changed default Mermaid layout preference from flowchart TD to flowchart LR
- Formalized rule: export Mermaid as images via mmdc, not fenced code blocks
