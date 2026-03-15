---
layout: post
title: "Generative Flow Networks for ML Researchers"
date: 2026-03-14
last_updated: 2026-03-15
description: "An introduction to GFlowNets from the perspective of probabilistic ML — sampling proportionally to rewards, training objectives, and connections to MaxEnt RL, variational inference, and diffusion models."
order: 1
categories: [science]
tags: [generative-models, reinforcement-learning, variational-inference, sampling]
toc:
  sidebar: left
related_posts: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post introduces Generative Flow Networks (GFlowNets) for ML researchers who already think in terms of probability distributions and sampling. I present GFlowNets from my own perspective as an outsider from RL and probabilistic ML, heavily drawing from a talk I gave at KAIST in June 2024. I recommend Emmanuel Bengio's <a href="https://youtu.be/H2w-TrAzrBA">Introduction to GFlowNets</a> and Yoshua Bengio's GFlowNet Tutorial for the original framing. Corrections are welcome.</em>
</p>

## Introduction

Suppose you want to generate a new molecule that binds to a target protein. You have a reward function $$R(x)$$ that scores how well molecule $$x$$ binds. The standard RL approach maximizes expected reward — but this converges to a single high-reward molecule and ignores other good candidates.

In scientific discovery, diversity is essential. Our proxy reward functions are imprecise — the top-scoring molecule under the proxy may fail experimentally. We need to cast a wide net: generate many diverse candidates that score well, then filter in the lab.

GFlowNets address this by sampling objects **proportionally to their reward**. Instead of finding $$x^* = \arg\max_x R(x)$$, a GFlowNet learns a policy that generates $$x$$ with probability proportional to $$\exp R(x)$$. Concretely, if molecule A has reward 4 and molecule B has reward 2, A is sampled $$e^4 / e^2 \approx 7.4$$ times more often than B — but B is still generated regularly. This is an energy-based (Boltzmann) distribution, the same type of distribution that appears in statistical mechanics and Bayesian inference.

### Roadmap

| Section | What It Explains |
|---------|-----------------|
| **The Goal** | Sampling from energy-based distributions over combinatorial objects |
| **GFlowNet Basics** | DAG states, forward/backward policies, flows |
| **Training Objectives** | Trajectory balance, detailed balance, sub-trajectory balance |
| **The Training Loop** | Off-policy training, replay buffers, exploration, backward policy design |
| **Connections** | MaxEnt RL, hierarchical variational inference, diffusion models |
| **Applications** | Molecules, biological sequences, Bayesian structure learning |

---

## Part I: The Goal

### Sampling from Energy-Based Distributions

GFlowNet is a generative framework for sampling combinatorial objects from an energy-based (Boltzmann) distribution:

> **Target distribution.** The goal is to learn a policy $$p_\theta(x)$$ that approximates
>
> $$p^*(x) = \frac{\exp R(x)}{Z}, \qquad Z = \sum_{x \in \mathcal{X}} \exp R(x)$$
>
> where $$R(x)$$ is a reward function (or negative energy), $$\mathcal{X}$$ is a finite set of objects, and $$Z$$ is the partition function — a normalizing constant that ensures the probabilities sum to one.
{: .block-definition }

We can evaluate $$R(x)$$ for any given object $$x$$ (e.g., by running a docking simulation), but we cannot enumerate all objects to compute $$Z$$. Unlike VAEs or diffusion models, the model learns from the reward function directly — not from a dataset of samples from $$p^*(x)$$.

### Amortized Sampling

MCMC methods (Metropolis-Hastings, Langevin dynamics, HMC) can sample from $$p^*(x)$$ without knowing $$Z$$, but each run produces a single correlated chain of samples. When we need to sample from the same distribution repeatedly — or from many related distributions — rerunning MCMC from scratch every time is wasteful.

GFlowNets perform **amortized sampling**: they invest upfront computation to train a neural network (the forward policy), and then sampling is a single forward pass through the network — fast and parallelizable. This is the same idea behind amortized variational inference, where an encoder network replaces per-datapoint optimization. The upfront training cost is large, but inference at deployment time is cheap.

For probabilistic ML researchers, GFlowNets can be understood as amortized MCMC for combinatorial spaces: train once, sample forever, with the learned policy replacing the Markov chain.

### Why Not RL?

RL maximizes expected cumulative reward — it finds the single best action sequence (or a narrow set of near-optimal ones). GFlowNets solve a different problem: sampling proportionally to the reward. The two are related but distinct:

- **Similar:** Both use interactive training with a reward function and learn a policy through trial and error.
- **Different:** RL converges to the mode; GFlowNets converge to the full distribution.

For scientific discovery, mode-seeking is dangerous. If the reward function is a learned proxy for binding affinity, its top-scoring molecule may not actually bind well in the lab. We want many diverse candidates so that even if some fail experimentally, others succeed.

Maximum entropy RL comes closer — it augments the reward with an entropy bonus that encourages the policy to spread probability mass across trajectories. GFlowNets turn out to be equivalent to MaxEnt RL with a specific reward shaping, but they target a distribution over terminal *objects* rather than trajectories. We will make this connection precise in Part V.

---

## Part II: GFlowNet Basics

### DAG of States

GFlowNets construct objects step by step, like assembling a molecule atom by atom. The construction process is represented as a directed acyclic graph (DAG) — a graph with directed edges and no cycles, so every path eventually terminates. The DAG has three types of nodes:

- There is a single **initial state** $$s_0$$ (e.g., an empty molecule).
- **Intermediate states** represent partially constructed objects (e.g., a molecule with some atoms added).
- **Terminal states** $$x \in \mathcal{X}$$ are the completed objects.

Each edge $$(s_{t-1}, s_t)$$ in the DAG represents an action — adding an atom, appending an amino acid, or placing a node. Multiple trajectories can lead to the same terminal object $$x$$, since different construction orders can produce the same result.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_dag_molecules.png" class="img-fluid rounded z-depth-1" zoomable=true caption="A DAG for molecule construction. Each node is a partially built molecule; each edge adds a fragment. Terminal states (large circles) are completed molecules. Redrawn from Bengio et al., <a href='https://arxiv.org/abs/2302.00615'>GFlowNet Foundations</a> (Figure 2b)." %}

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_dag_abstract.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Abstract view of the same structure. The blue node is the initial state \(s_0\); pink nodes are terminal states \(x \in \mathcal{X}\). Intermediate white nodes are partially constructed objects." %}

### Forward Policy

The **forward policy** $$p_\mathrm{F}(s_t \mid s_{t-1})$$ is a learned distribution over next states given the current state — a neural network that looks at the current partial object and decides what to add next. A **trajectory** $$\tau = (s_0, s_1, \ldots, s_T = x)$$ is a complete path from the initial state to a terminal state. Its probability is the product of all transition probabilities along the path:

$$p_\mathrm{F}(\tau) = \prod_{t=1}^{T} p_\mathrm{F}(s_t \mid s_{t-1})$$

The probability of generating a specific object $$x$$ is the sum over all trajectories that end at $$x$$:

$$p_\mathrm{F}(x) = \sum_{\tau \in \mathcal{T}(x)} p_\mathrm{F}(\tau)$$

where $$\mathcal{T}(x)$$ is the set of all trajectories that terminate at $$x$$. For example, a molecule with three atoms A, B, C can be built as A→B→C or A→C→B or B→A→C, and so on — all producing the same molecule $$x$$. The total probability of generating $$x$$ is the sum over all these construction orders.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_forward_policy.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The forward policy constructs objects step by step through a DAG. A trajectory (highlighted) is a path from the initial state to a terminal state. The probability of an object is the sum over all trajectories ending at it." %}

### The Key Difficulty

We want $$p_\mathrm{F}(x) \approx p^*(x)$$ for every object $$x$$, but computing $$p_\mathrm{F}(x)$$ requires summing over all trajectories $$\mathcal{T}(x)$$. For a molecule with 20 atoms, the number of construction orders can be astronomically large — this sum is intractable.

GFlowNets sidestep this by replacing **object-level matching** with **trajectory-level matching**. Instead of asking "does the probability of each object $$x$$ match $$p^*(x)$$?", we ask "does the probability of each trajectory under the forward policy match a target trajectory distribution?" If trajectory probabilities match, object probabilities match automatically — because $$p_\mathrm{F}(x) = \sum_\tau p_\mathrm{F}(\tau)$$, and if each $$p_\mathrm{F}(\tau)$$ matches the corresponding target, the sum matches too.

### Backward Policy

To define a target distribution over trajectories, we need a way to decompose each terminal object $$x$$ into a trajectory. The **backward policy** $$p_\mathrm{B}(s_{t-1} \mid s_t)$$ does this: given a state, it assigns probabilities to its parent states in the DAG. Starting from a terminal state $$x$$ and repeatedly sampling parents, we trace a path back to the initial state $$s_0$$ — a trajectory in reverse.

The backward policy, combined with the reward, defines the target distribution over trajectories:

$$p_\mathrm{B}(\tau) \propto \exp R(x) \prod_{t=1}^{T} p_\mathrm{B}(s_{t-1} \mid s_t)$$

A trajectory leading to a high-reward terminal state gets high probability; the backward policy determines how that probability is split among the different construction orders for $$x$$. The backward policy can be fixed (e.g., uniform over parents) or learned jointly with the forward policy.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_backward_policy.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The backward policy decomposes a terminal state into a trajectory by reversing the construction. Training matches the forward and backward trajectory distributions, which implies matching the marginal distributions over objects." %}

### A Worked Example: Uniform Backward Policy

Consider a simple DAG with three terminal states $$x_1, x_2, x_3$$ with unnormalized target weights $$\exp R(x_1) = 4$$, $$\exp R(x_2) = 2$$, $$\exp R(x_3) = 1$$. With a uniform backward policy — and assuming each state has exactly one parent, so $$p_\mathrm{B} = 1$$ on every edge — each terminal state has exactly one backward trajectory, and each trajectory's target probability is proportional to $$\exp R(x)$$: $$p_\mathrm{B}(\tau_1) \propto 4$$, $$p_\mathrm{B}(\tau_2) \propto 2$$, $$p_\mathrm{B}(\tau_3) \propto 1$$. The total is $$4 + 2 + 1 = 7$$, so the forward policy must route 4/7 of its probability toward $$x_1$$, 2/7 toward $$x_2$$, and 1/7 toward $$x_3$$.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_example_forward.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Uniform backward policy example. Left: backward policy with all probabilities equal to 1. Center: three trajectories with probabilities proportional to terminal rewards. Right: the forward policy that matches these trajectory probabilities." %}

### A Worked Example: Non-Uniform Backward Policy

What if the backward policy is non-uniform? Suppose $$x_2$$ has two parents $$s_1$$ and $$s_2$$, and we set $$p_\mathrm{B}(s_1 \mid x_2) = p_\mathrm{B}(s_2 \mid x_2) = 0.5$$. Now there are four trajectories instead of three, because $$x_2$$ can be reached via either $$s_1$$ or $$s_2$$. The backward policy splits $$x_2$$'s weight of $$\exp R(x_2) = 2$$ across the two paths: each trajectory through $$x_2$$ gets target probability proportional to $$2 \times 0.5 = 1$$. The forward policy adjusts: it now routes more probability through $$s_2$$, because $$s_2$$ serves as a waypoint to both $$x_2$$ and $$x_3$$.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_example_backward.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Non-uniform backward policy. The 0.5 split at \(x_2\) creates four trajectories instead of three. The forward policy adapts: \(s_2\) now receives more probability (2/7 vs 1/7) because it serves as a path to both \(x_2\) and \(x_3\)." %}

### Flows

We want $$p_\mathrm{F}(\tau) \approx p_\mathrm{B}(\tau)$$, but there is a problem: $$p_\mathrm{B}(\tau)$$ is defined only up to a normalizing constant (the $$\propto$$ in the equation above). We do not know the partition function $$Z$$, so we cannot compute the actual probability $$p_\mathrm{B}(\tau)$$.

The solution is to work with **unnormalized densities** instead of probabilities. We multiply both sides by their respective normalizing constants and match the unnormalized quantities directly:

$$f_\mathrm{F}(\tau) = Z_\theta \prod_{t=1}^{T} p_\mathrm{F}(s_t \mid s_{t-1}) \approx \exp R(x) \prod_{t=1}^{T} p_\mathrm{B}(s_{t-1} \mid s_t) = f_\mathrm{B}(\tau)$$

Here $$Z_\theta$$ is a trainable scalar — the model's estimate of the partition function. The left side $$f_\mathrm{F}(\tau)$$ is the forward flow (unnormalized forward probability), and the right side $$f_\mathrm{B}(\tau)$$ is the backward flow (unnormalized backward probability). These trajectory-wise unnormalized densities are called **flows**, hence the name Generative *Flow* Network.

---

## Part III: Training Objectives

### Trajectory Balance (TB)

The trajectory balance objective[^tb] directly enforces the flow-matching condition for complete trajectories. If $$f_\mathrm{F}(\tau) = f_\mathrm{B}(\tau)$$, then $$\log(f_\mathrm{F} / f_\mathrm{B}) = 0$$. Squaring this log-ratio gives a loss that is zero when the flows match and positive otherwise:

> **Trajectory Balance.** For a trajectory $$\tau = (s_0, \ldots, s_T = x)$$:
>
> $$\mathcal{L}_\mathrm{TB}(\tau) = \left(\log \frac{Z_\theta \prod_{t=1}^{T} p_\mathrm{F}(s_t \mid s_{t-1})}{\exp R(x) \prod_{t=1}^{T} p_\mathrm{B}(s_{t-1} \mid s_t)}\right)^2$$
>
> The loss is zero when the forward flow $$Z_\theta \prod p_\mathrm{F}$$ equals the backward flow $$\exp R(x) \prod p_\mathrm{B}$$ for every trajectory.
{: .block-definition }

[^tb]: Malkin et al., "Trajectory balance: Improved credit assignment in GFlowNets," NeurIPS 2022.

TB is the simplest objective. It trains a single scalar $$Z_\theta$$ plus the forward and backward policies. The downside is credit assignment: a single reward signal at the terminal state must propagate back through the entire construction sequence. For long trajectories, this makes learning slow — the gradient carries information about the full trajectory, and early transitions receive weak signal.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_flow_matching.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Trajectory balance in action. From left: backward policy and rewards; backward flows \(f_\mathrm{B}(\tau) = \exp R(x) \prod p_\mathrm{B}\); forward flows \(f_\mathrm{F}(\tau) = Z_\theta\, p_\mathrm{F}(\tau)\) with \(Z_\theta = 7\); the resulting forward policy." %}

### Detailed Balance (DB)

TB applies to entire trajectories, which can be long. The detailed balance objective[^db] breaks this down to individual transitions — one edge at a time. The idea is borrowed from the detailed balance condition in Markov chain theory: at equilibrium, the probability flux along each edge must be equal in both directions.

> **Detailed Balance.** For each edge $$(s_{t-1}, s_t)$$:
>
> $$\mathcal{L}_\mathrm{DB}(s_{t-1}, s_t) = \left(\log \frac{f_\theta(s_{t-1}) \, p_\mathrm{F}(s_t \mid s_{t-1})}{f_\theta(s_t) \, p_\mathrm{B}(s_{t-1} \mid s_t)}\right)^2$$
>
> where $$f_\theta(s)$$ is a learned **state flow** — the total flow through state $$s$$. Boundary conditions: $$f_\theta(s_0) = Z_\theta$$ and $$f_\theta(x) = \exp R(x)$$ for terminal states.
{: .block-definition }

[^db]: Bengio et al., "GFlowNet foundations," JMLR 2023. Deleu et al., "Bayesian structure learning with generative flow networks," UAI 2022.

DB provides local credit assignment — each transition gets its own loss signal, so early transitions receive direct feedback rather than waiting for the terminal reward. The trade-off is learning the state flow function $$f_\theta(s)$$, an additional neural network that estimates how much total flow passes through each intermediate state.

{% include figure.liquid loading="eager" path="assets/img/blog/gflownet/fig_detailed_balance.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Detailed balance operates at individual edges. From left: backward policy; forward policy; trajectory flows; the DB condition at a single edge — the product of state flow and transition probability must match in both directions." %}

### Sub-Trajectory Balance (SubTB)

TB enforces balance over the full trajectory (global but weak signal); DB enforces balance over single edges (local but requires learning state flows). Sub-trajectory balance[^subtb] interpolates between the two by enforcing balance on sub-trajectories of arbitrary length $$\ell$$. For a sub-trajectory $$(s_i, s_{i+1}, \ldots, s_{i+\ell})$$:

> **Sub-Trajectory Balance.** For states $$s_i, \ldots, s_{i+\ell}$$ along a trajectory:
>
> $$\mathcal{L}_\mathrm{SubTB}(s_i, \ldots, s_{i+\ell}) = \left(\log \frac{f_\theta(s_i) \prod_{t=i+1}^{i+\ell} p_\mathrm{F}(s_t \mid s_{t-1})}{f_\theta(s_{i+\ell}) \prod_{t=i+1}^{i+\ell} p_\mathrm{B}(s_{t-1} \mid s_t)}\right)^2$$
>
> where $$f_\theta(s)$$ is the learned state flow. Setting $$\ell = 1$$ recovers DB; setting $$i = 0$$, $$\ell = T$$, and using the boundary conditions $$f_\theta(s_0) = Z_\theta$$, $$f_\theta(x) = \exp R(x)$$ recovers TB.
{: .block-definition }

[^subtb]: Madan et al., "Learning GFlowNets from partial episodes for improved convergence and stability," ICML 2023.

In practice, SubTB sums losses over all sub-trajectories of all lengths within a sampled trajectory, optionally weighting shorter sub-trajectories more heavily. This interpolates between local credit assignment (DB) and global credit assignment (TB), and often trains more stably than either alone.

### Flow Matching

An alternative to the balance conditions is **flow matching** (not to be confused with the flow matching used in continuous normalizing flows). This is the original training objective from the first GFlowNet paper.[^fm] It enforces flow conservation at each intermediate state: the total incoming flow must equal the total outgoing flow, like water in a pipe network. This is conceptually clean but requires summing over all parents and children of each state, which can be expensive for states with many neighbors.

[^fm]: Bengio et al., "Flow network based generative models for non-iterative diverse candidate generation," NeurIPS 2021.

### Why These Objectives Are Surprisingly Easy to Optimize

What struck me when I first studied GFlowNets is how simple the training actually is. Look at the TB loss again: it is a squared log-ratio between two quantities that we can both compute for any given trajectory. This is a regression problem — we are fitting the forward flow to match the backward flow, and the loss is just MSE in log-space.

Compare this to on-policy RL, where you must collect fresh trajectories with your current policy, estimate advantages with high-variance baselines, and carefully tune clipping ratios and entropy bonuses to keep training stable. GFlowNet training looks more like offline RL or behavioral cloning: given a dataset of trajectories (from a replay buffer), minimize a well-defined regression loss. There is no policy gradient, no REINFORCE estimator, no reward-to-go. The gradient of $$\mathcal{L}_\mathrm{TB}$$ with respect to the policy parameters is a straightforward backpropagation through the log-probabilities — the same kind of computation you would do in a supervised sequence model.

This is, in my opinion, the most underappreciated aspect of GFlowNets. The problem (sampling from an energy-based distribution over combinatorial objects) sounds hard, but the training reduces to something much closer to supervised learning than to RL. The difficulty shifts from optimization to exploration: the loss is easy to minimize on any given trajectory, but you need to find the right trajectories to train on.

---

## Part IV: The Training Loop

### Off-Policy Training with Replay

GFlowNet training follows an RL-style loop:

1. **Collect** trajectories $$\{\tau^{(b)}\}_{b=1}^{B_1}$$ using the current forward policy $$p_\mathrm{F}$$.
2. **Store** these trajectories in a replay buffer $$\mathcal{B}$$.
3. **Train** the forward policy on a batch $$\{\tau^{(b)}\}_{b=1}^{B_2}$$ sampled from the replay buffer, minimizing $$\mathcal{L}_\mathrm{TB}$$, $$\mathcal{L}_\mathrm{DB}$$, or $$\mathcal{L}_\mathrm{SubTB}$$.

This is an **off-policy** algorithm — we train on replayed trajectories, not just fresh ones. "Off-policy" means the trajectories used for training were not necessarily generated by the current version of the policy; they may come from an earlier version stored in the replay buffer. Replay matters because reward evaluation is often expensive (e.g., docking simulations take seconds per molecule). Replaying past trajectories extracts more learning signal per reward evaluation.

In practice, most implementations mix **on-policy** samples (fresh trajectories from the current $$p_\mathrm{F}$$) with replayed trajectories. The on-policy samples ensure the model keeps exploring new regions of the DAG, while the replay buffer provides a stable training distribution and prevents the model from forgetting high-reward regions it discovered earlier.

### Exploration

A pure on-policy GFlowNet only visits states reachable under its current forward policy. If the policy has not yet discovered a high-reward region, it never trains on trajectories leading there. This is the same exploration challenge that plagues RL.

Common strategies include $$\epsilon$$-greedy exploration (with probability $$\epsilon$$, take a uniformly random action instead of sampling from $$p_\mathrm{F}$$), tempering the forward policy (raising the temperature to flatten the distribution and encourage more random choices), and prioritized replay buffers that oversample high-reward trajectories so the model gets more training signal from the best discoveries. Some recent work combines GFlowNets with local search:[^localsearch] generate a candidate with the forward policy, improve it with local perturbations (e.g., swapping one atom for another), and add the improved candidate to the replay buffer.

[^localsearch]: Kim et al., "Local Search GFlowNets," ICLR 2024.

### Backward Policy Design

The backward policy $$p_\mathrm{B}$$ is a design choice with real consequences. It determines how the target distribution over trajectories decomposes reward across construction paths.

**Uniform** backward policy assigns equal probability to all parent states. This is the simplest option and works well when the DAG has uniform branching. But in DAGs where some states have many parents, the uniform backward policy spreads reward thinly across many trajectories, making each one a weak training signal.

**Learned** backward policies are trained jointly with the forward policy. The backward policy can concentrate reward on fewer trajectories, giving the forward policy a stronger signal. The downside is more parameters and potential instability from co-adaptation.

**Pessimistic backward policy (PBP)** biases the backward policy toward trajectories that are hard for the current forward policy — trajectories where $$p_\mathrm{F}(\tau) / p_\mathrm{B}(\tau)$$ is small. This focuses training on the forward policy's weakest regions, similar to hard example mining in supervised learning.

### Evaluation

GFlowNet performance is measured along two axes:

- **Reward quality:** average reward of the top-$$k$$ generated samples, or the fraction of samples exceeding a reward threshold.
- **Diversity:** number of distinct modes discovered. This is domain-specific — for molecules, it might be the number of structurally distinct scaffolds; for sequences, the number of distinct high-affinity families.

A good GFlowNet achieves high reward *and* high diversity. A model that finds one excellent molecule and generates it repeatedly has failed — we need many distinct candidates for downstream experimental validation.

---

## Part V: Connections to Existing Frameworks

GFlowNets connect to RL, variational inference, and diffusion models through precise mathematical equivalences.

### GFlowNet as Maximum Entropy RL

Standard RL maximizes expected cumulative reward $$\mathbb{E}[\sum_t r_t]$$, which leads to deterministic optimal policies. Maximum entropy (MaxEnt) RL adds an entropy bonus $$\mathcal{H}(\pi)$$ that penalizes the policy for being too concentrated, encouraging it to spread probability across multiple good trajectories:

$$\pi^*_\mathrm{MaxEnt} = \arg\max_\pi \mathbb{E}_\tau \left[\sum_{t=1}^T r(s_{t-1}, s_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_{t-1}))\right]$$

where $$r(s_{t-1}, s_t)$$ is the per-step reward, $$\mathcal{H}(\pi(\cdot \mid s))$$ is the entropy of the policy at state $$s$$, and $$\alpha > 0$$ controls the trade-off between reward and entropy. The optimal policy samples trajectories proportionally to exponentiated cumulative reward:

$$\pi^*_\mathrm{MaxEnt}(\tau) \propto \exp\left(\frac{1}{\alpha}\sum_{t=1}^T r(s_{t-1}, s_t)\right)$$

This looks similar to GFlowNets, but there is a subtle mismatch. MaxEnt RL optimizes a distribution over **trajectories**, while GFlowNets target a distribution over **terminal states**. Multiple trajectories can lead to the same terminal object $$x$$, so MaxEnt RL assigns probability to $$x$$ proportional to the *sum* of exponentiated rewards over all trajectories ending at $$x$$. This is not $$\exp R(x)$$ unless the per-step rewards are carefully structured.

The precise connection:[^maxentrl] GFlowNet is equivalent to MaxEnt RL with a specific reward shaping. Setting the per-step reward to

$$r(s_{t-1}, s_t) = \log p_\mathrm{B}(s_{t-1} \mid s_t)$$

at intermediate steps and $$r(s_{T-1}, x) = R(x) + \log p_\mathrm{B}(s_{T-1} \mid x)$$ at the terminal step makes the MaxEnt RL objective identical to the GFlowNet trajectory balance condition. The backward policy acts as reward shaping that removes the redundancy of endpoint-sharing trajectories.

[^maxentrl]: Deleu et al., "Discrete probabilistic inference as control in multi-path environments," AISTATS 2024.

In my view, this connection does not diminish GFlowNets. The primary contribution is the *problem formulation* — sampling proportionally to $$\exp R(x)$$ over combinatorial objects — and the insight that the backward policy naturally resolves the trajectory-redundancy problem that makes naive MaxEnt RL fail.

### GFlowNet as Hierarchical Variational Inference

Variational inference (VI) approximates an intractable target distribution $$p$$ by optimizing a tractable approximation $$q$$ to minimize $$D_\mathrm{KL}(q \| p)$$. GFlowNet training can be viewed as VI with trajectories as the latent variables.[^hvi]

The mapping is:

| VI concept | GFlowNet counterpart |
|------------|---------------------|
| Latent variable | Trajectory $$\tau$$ |
| Observed variable | Terminal object $$x$$ |
| Variational distribution $$q$$ | Forward policy $$p_\mathrm{F}(\tau)$$ |
| Target distribution $$p$$ | $$p_\mathrm{B}(\tau) \propto \exp R(x) \prod_t p_\mathrm{B}(s_{t-1} \mid s_t)$$ |
| ELBO | Trajectory balance loss |

When training on-policy (sampling trajectories from $$p_\mathrm{F}$$), the TB gradient equals the KL divergence gradient:

$$\nabla_\theta \mathbb{E}_{\tau \sim p_\mathrm{F}}[\mathcal{L}_\mathrm{TB}(\tau)] = \nabla_\theta D_\mathrm{KL}(p_\mathrm{F} \| p_\mathrm{B})$$

[^hvi]: Malkin et al., "GFlowNets and variational inference," ICLR 2023.

This perspective explains a practical advantage: GFlowNets can train off-policy while standard VI cannot. In VI, the gradient estimator requires samples from the current variational distribution $$q$$ — you must re-sample every time you update $$q$$. In GFlowNets, the TB loss is a squared log-ratio that can be evaluated on any trajectory, regardless of how it was generated. This is why replay buffers work: old trajectories from previous policy versions are still valid training data.

### GFlowNet as Diffusion Model

For continuous state spaces, the forward and backward policies become continuous-time stochastic processes — and the connection to diffusion models becomes direct.[^diffusion] The correspondence reverses the naming convention:

- The GFlowNet **backward policy** $$\leftrightarrow$$ the diffusion model's **forward process** (adding noise, corrupting data toward a simple distribution).
- The GFlowNet **forward policy** $$\leftrightarrow$$ the diffusion model's **reverse process** (denoising, generating data from noise).

[^diffusion]: Zhang et al., "Unifying Generative Models with GFlowNets and Beyond," arXiv 2022. Sendera et al., "Improved off-policy training of diffusion samplers," arXiv 2024.

The naming is confusing because GFlowNets and diffusion models use "forward" and "backward" in opposite senses. In GFlowNets, "forward" means constructing the object; in diffusion models, "forward" means destroying it.

The key difference is the training signal. Diffusion models learn from data samples — they observe $$x \sim p_\text{data}$$ and learn to reverse the noise process. GFlowNets learn from an energy function — they can evaluate $$R(x)$$ for any $$x$$ but have no dataset. This makes GFlowNets applicable when we have an energy function but no samples from the target distribution — the typical scientific discovery setting.

A related line of work studies **diffusion samplers** — diffusion models trained to sample from Boltzmann distributions using energy functions rather than data.[^diffusionsampler] The GFlowNet objectives (TB, DB, SubTB) have continuous-time counterparts: in the limit of infinitesimal discretization steps, they converge to SDEs and path space measures that are natural objects in stochastic control theory. This means GFlowNet training objectives and diffusion sampler objectives are not merely analogous — they are asymptotically equivalent, and practitioners can choose between discrete-time (GFlowNet-style) and continuous-time (SDE-based) formulations based on the problem structure.

[^diffusionsampler]: Berner, Richter, Sendera, Rector-Brooks, and Malkin, "From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training," TMLR 2026.

---

## Part VI: Applications

GFlowNets have been applied to several domains where diverse, high-quality candidates are more valuable than a single optimum:

- **Molecule design**[^fm] — the canonical application. Molecules are built step by step by adding fragments, with a GNN policy and binding affinity as the reward. Drug discovery needs diverse scaffolds because proxy rewards are imprecise.
- **Biological sequence design**[^seqdesign] — generating protein, DNA, or RNA sequences token by token, with binding activity as the reward. GFlowNets can jump between distant regions of sequence space in a single trajectory, unlike MCMC samplers that make local edits.
- **Bayesian structure learning**[^bayesdag] — sampling causal DAGs proportionally to their posterior probability. Each DAG is constructed by adding edges, and the reward is the marginal likelihood. GFlowNets capture uncertainty over causal structures rather than returning a single MAP graph.
- **Combinatorial optimization** — finding diverse high-quality solutions to problems like maximum independent set, using GNN-parameterized policies.

[^seqdesign]: Jain et al., "Biological sequence design with GFlowNets," ICML 2022.
[^bayesdag]: Deleu et al., "Bayesian structure learning with generative flow networks," UAI 2022.

Standard synthetic benchmarks include HyperGrid (multi-modal reward on a grid) and bag generation (constructing multisets with trajectory redundancy).

---

## FAQ

**Does GFlowNet work better than RL?**
For reward maximization, no — that is not its goal. For generating diverse high-reward candidates, GFlowNets outperform RL in several domains including molecular design and combinatorial optimization.

**Do we really need energy-based sampling? Why not RL with diversity regularization?**
It depends on whether you need the *correct* distribution or just a diverse set. If you need samples from the Boltzmann distribution (e.g., for computing expectations in statistical mechanics or Bayesian posterior inference), diversity regularization does not give the right answer — it produces a heuristically diverse set with no guarantees about the relative probabilities. If diversity is a soft goal rather than a distributional requirement, adding diversity bonuses to RL is a simpler alternative.

**Isn't GFlowNet just variational inference?**
In my view, yes — with two practical innovations that make it work for constructive generation. First, the DAG structure provides a natural factorization of the variational distribution over trajectories. Second, the squared log-ratio loss enables off-policy training, so we can reuse past trajectories rather than requiring fresh samples from the current variational distribution. The contribution is the problem formulation — using RL-style interactive training to sample from energy-based distributions over structured objects — more than the specific algorithm.

---

## Closing Thoughts

GFlowNets address a specific gap: amortized sampling from energy-based distributions over combinatorial objects when we have access to the energy function but no dataset. Train a constructive policy once, sample forever — with the flow-matching objectives reducing what sounds like a hard RL problem to something closer to regression.

What I find most compelling is the combination of a well-defined probabilistic target (the Boltzmann distribution) with a training procedure that is practical and stable. The framework connects to MaxEnt RL, variational inference, and diffusion models, but the training itself is simpler than any of these — no policy gradients, no ELBO tricks, no score matching. Just match the flows.

I think GFlowNets are in a somewhat undeserved state of neglect. The core idea — sampling diverse candidates proportionally to a reward — is exactly what real scientific discovery pipelines need. But "discovery" is hard to benchmark. We can only evaluate GFlowNets on proxy tasks (docking scores, learned oracles), and these proxies have limited impact on whether the community takes the method seriously. A strong result on a proxy benchmark does not prove that GFlowNets would work in a real drug discovery campaign, and running that campaign is a multi-year, multi-million-dollar effort that no ML lab can do alone.

There is also a practical gap. Real-world molecular design does not start from scratch — it starts from massive databases of known compounds, assay results, and pretrained models. A realistic discovery pipeline would pretrain a generative model on this offline data (learning the chemistry), then fine-tune with GFlowNet objectives to steer generation toward a specific target with feedback from experiments. This pretrain-then-finetune loop is expensive to set up and even harder to benchmark rigorously, because it requires the full pipeline: data curation, pretraining, reward model, GFlowNet fine-tuning, and experimental validation.

I believe people will revisit GFlowNets when the infrastructure for AI-driven scientific discovery matures — when molecular optimization with feedback loops becomes routine rather than heroic, and when benchmarking these pipelines end-to-end becomes tractable. The algorithm is ready. The ecosystem is not, yet.
