---
layout: post
title: "ML Basics for Biology Students"
date: 2026-03-01
description: "A gentle introduction to machine learning concepts for students with a biology background."
course: "2026-spring-protein-ai"
course_title: "Protein & Artificial Intelligence"
course_semester: "Spring 2026"
lecture_number: 0
preliminary: true
toc:
  sidebar: left
related_posts: false
---

*This is a preliminary note for the Protein & Artificial Intelligence course (Spring 2026), co-taught by Prof. Sungsoo Ahn and Prof. Homin Kim.*

## What is Machine Learning?

Machine learning (ML) is a subfield of artificial intelligence that focuses on building systems that learn from data. Instead of explicitly programming rules, we provide examples and let the algorithm discover patterns.

## Supervised Learning

In supervised learning, we have a dataset of input-output pairs $\{(x_i, y_i)\}_{i=1}^{N}$, and we want to learn a function $f: X \to Y$ that maps inputs to outputs.

For example, given a protein sequence (input), we might want to predict its function (output).

### Loss Functions

We measure how well our model performs using a **loss function** $\mathcal{L}(f(x), y)$. Common choices include:

- **Mean Squared Error (MSE):** $\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(f(x_i) - y_i)^2$
- **Cross-Entropy Loss:** $\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} y_i \log f(x_i)$

### Gradient Descent

To minimize the loss, we iteratively update our model parameters $\theta$ in the direction of steepest descent:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the **learning rate**.

## Key Takeaways

1. ML learns patterns from data rather than following explicit rules.
2. Supervised learning requires labeled training examples.
3. We optimize model parameters by minimizing a loss function via gradient descent.

## Further Reading

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
