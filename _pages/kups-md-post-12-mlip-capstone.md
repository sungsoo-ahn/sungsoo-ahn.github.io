---
layout: post
permalink: /kups-md-tutorials/post-12-mlip-capstone/
title: "What Changes When the Potential Is a Machine-Learned Interatomic Potential?"
date: 2026-07-14
last_updated: 2026-08-01
description: "Run a hash-pinned MACE-MPA-0 export through kUPS, inspect real NVT and NVE trajectories, and separate execution evidence from model-validity claims."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 12
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for MLIP-aware machine-learning researchers."
series_order: 12
categories: [science]
tags: [molecular-dynamics, machine-learned-potentials, mace, aluminum, kups]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: ready
collapse_code: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This executable article is hidden from site navigation until the twelve-part series is staged for publication. Its numbers come from real kUPS trajectories using a revision- and hash-pinned Tojax export of MACE-MPA-0. The page does not claim reference accuracy or model uncertainty because this experiment has neither reference labels nor a model ensemble.</em>
</p>

## A Learned Potential Is Still Part of the MD Method

Replacing Lennard--Jones with a neural network changes the force provider. It
does not repeal molecular dynamics.

The initial state can still be biased. A timestep can still be unstable. A
thermostat can still hide poor equilibration. A short trajectory can still
give a precise-looking but unsupported mean. An umbrella estimator can still
fail from weak overlap, and nonequilibrium work can still fail from missing
tails. On top of those familiar problems, a learned potential adds a new
dependency: the answer now depends on an external model artifact whose
behavior is not summarized by its filename.

That is the capstone lesson. An MLIP is not an oracle passed into an otherwise
finished workflow. The exported computation, integration settings, sampled
configurations, and downstream estimator form one method. They must be
reviewed together.<sup id="cite-validation"><a href="#ref-validation">1</a></sup>

This post therefore makes a deliberately narrow claim. I ran a pinned Tojax
export of `mace-mpa-0-medium.model` inside the public kUPS MD application for
three fcc-Al deployment regimes. The CPU smoke path and GPU production path
both executed the same model bytes. The resulting trajectories have sensible
short-run temperature, force, geometry, and energy-conservation diagnostics.
That establishes an executable and inspectable simulation path. It does not
establish density-functional-theory accuracy, training-domain support,
long-time stability, transport properties, or uncertainty calibration.

The distinction is easy to say and easy to lose. A job that returns an HDF5
file proves that software ran. A small NVE drift proves that one numerical
setup conserved its own learned Hamiltonian for the tested time. Neither fact
proves that the learned Hamiltonian is the right physical one.

## Freeze the Computation That Actually Ran

"We used MACE" is incomplete provenance. MACE names an architecture and
software ecosystem, not a unique force function.<sup id="cite-mace"><a href="#ref-mace">2</a></sup>
The deployed object in this tutorial is a serialized Tojax archive. Its
repository, revision, and content hash are part of the scientific method:

<div class="table-responsive" markdown="1">

| Field | Production value |
|---|---|
| upstream model | `mace-mpa-0-medium.model` |
| deployed artifact | `mace-mpa-0-medium_32.zip` |
| artifact repository | `CuspAI/kUPS-mace-jax` |
| repository revision | `aa54c05695b6509f588d04d664be70b28cf3138c` |
| artifact SHA-256 | `728762228338782ab961e9dc689ffbe7b51690fcf7cd8b4ef3c63c37ec6cd78c` |
| exporter / kUPS backend | Tojax / `TojaxPotentialConfig` |
| kUPS entry point | `kups.application.simulations.md.run` |
| kUPS / JAX versions | 1.0.3 / 0.10.2 |

</div>

The runner downloads that exact Hugging Face revision and computes SHA-256
before launching MD. A mismatch stops the run. The `32` archive is the
float32 model export; the isolated worker also enables JAX x64 because the
serialized graph uses int64 indices. That detail is not cosmetic: disabling
x64 truncates the graph-index type and the serialized computation rejects the
call. The produced HDF5 arrays record their actual dtypes independently of the
model-export label.

The model identity is exposed directly in the notebook rather than buried in
a setup file:

{% include kups-notebooks/post-12/setup.html %}

{% include kups-notebooks/post-12/protocol.html %}

The compact artifacts behind this page are all public-reviewable:

- [smoke configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/smoke.json)
- [production configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-12/full.json)
- [executable notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/notebooks/post-12-mlip-capstone.ipynb)
- [smoke summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/smoke/mlip_summary.json)
- [production summary](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/mlip_summary.json)
- [production sample table](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/mlip_samples.csv)
- [provenance manifest](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/results/post-12/full/manifest.json)
- [kUPS/Tojax worker](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/src/kups_md_tutorials/mlip_capstone_worker.py)
- [figure-generation source](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/scripts/generate_post12_figures.py)
- [review record](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/reviews/post-12.md)

This provenance is more specific than a model card citation, but it answers a
different question. A model card describes a release. The revision and hash
identify the bytes that supplied every force in this experiment.

## The CPU Smoke Run Is an Execution Test

A smoke test should be cheap enough to run in a clean notebook and strict
enough to catch a fake execution path. Here it uses one conventional four-atom
fcc Al cell at 300 K. It launches one BAOAB Langevin trajectory and one
velocity-Verlet NVE trajectory through kUPS, storing eight frames from each.
The model download and SHA check are identical to production, but JAX is
forced to the CPU.

The notebook does not load the committed smoke JSON and call that execution.
It creates fresh outputs under `notebook-runs/`, invokes both isolated workers,
and then runs the same verifier used by the command line:

{% include kups-notebooks/post-12/smoke-run.html %}

The committed smoke record contains two real runs and 16 stored frames. The
observed backend is CPU, the potential backend is Tojax, and the artifact hash
matches the production artifact. Its short temperature trace is not useful
for thermodynamics—the NVT warmup is only one femtosecond—but that is not the
smoke test's job. It proves that a clean CPU environment can download,
deserialize, execute, store, read, and verify the model.

This separation prevents two common mistakes. First, a CPU smoke run is not
silently relabeled as GPU evidence. Second, production is not made notebook
interactive just to prove reproducibility. The small path is for rapid
interface checks; the larger path is separately recorded and reviewed on the
hardware required by its claim.

## Three Regimes Are Tests, Not Domain Labels

The production system is a 2 by 2 by 2 repetition of the conventional fcc Al
cell: 32 atoms at a reference lattice constant of 4.05 Å. I test three initial
cells:

1. `ambient_fcc`: zero strain at a 300 K thermostat setpoint;
2. `compressed_fcc`: isotropic strain of -0.08 at 300 K;
3. `hot_expanded_fcc`: isotropic strain of +0.06 at 900 K.

These names describe what was constructed. They do not say that one case is
"in domain" or another is "extrapolative." Making that claim would require a
defined support metric, suitable calibration data, or an ensemble of models.
This tutorial has none of those. Renaming a large displacement
"extrapolation" would create evidence by vocabulary.

Each regime has three independent random seeds. For each seed, the workflow
launches one NVT and one NVE simulation, giving 18 trajectories in total. The
NVT branch uses BAOAB Langevin dynamics, a 0.5 fs timestep, friction
coefficient 0.01 fs$$^{-1}$$, 500 fs of warmup, and 120 fs of measured
production. The NVE branch uses velocity Verlet for 120 fs at the same 0.5 fs
timestep. Both store a frame every 5 fs, so each trajectory contributes 24
frames and the full experiment contributes 432.

The NVE branch is independently initialized. It is not a continuation of the
NVT endpoint. That choice keeps the integration diagnostic simple, but it
also narrows its meaning. This experiment does not ask whether a particular
thermostatted configuration retains its energy after handoff. It asks whether
independently initialized trajectories in each constructed cell show a trend
in total energy under the learned force provider and the selected timestep.

Every worker reported `gpu:NVIDIA RTX A5000`; none fell back to CPU. The full
workflow took 560.464 seconds. A successful run requires more than a zero exit
code: the verifier checks the requested and observed device, model metadata,
number of trajectories, frame counts, required HDF5 datasets, raw trajectory
hashes, replica uncertainty fields, compact-file hashes, and absence of any
GPU blocking reason.

## A Thermostat Setpoint Is Not an Equilibrium Result

The first production attempt used only 50 fs of NVT warmup with a 100 fs
Langevin relaxation scale. The temperatures were predictably low: about 241 K,
260 K, and 693 K for targets of 300 K, 300 K, and 900 K. Nothing had crashed.
The thermostat field in the configuration said the desired temperatures. The
trajectories simply had not relaxed long enough.

That failure was useful because it exposed a bad protocol before prose turned
it into a result. The final run uses 500 fs of warmup—five relaxation times—
before the 120 fs measurement window. The resulting replica-aggregated
temperatures are:

<div class="table-responsive" markdown="1">

| Regime | Target (K) | Measured NVT mean ± replica SEM (K) | Absolute difference (K) |
|---|---:|---:|---:|
| ambient fcc | 300 | 304.4 ± 14.5 | 4.4 |
| compressed fcc | 300 | 304.4 ± 19.0 | 4.4 |
| hot expanded fcc | 900 | 831.9 ± 39.1 | 68.1 |

</div>

The two 300 K means are close to their setpoints relative to their
three-replica spread. The hot case remains lower than 900 K by about 68 K,
roughly 1.7 reported SEMs. I do not "fix" that difference by reporting the
setpoint in place of the measurement. Three short replicas are too little data
for a tight ensemble claim, and 120 fs is especially short for a production
materials calculation. The honest conclusion is narrower: longer warmup
removed the obvious transient, while the hot regime still deserves a longer
equilibration and sampling study before any temperature-dependent observable
is trusted.

The shaded bands in the temperature panel are frame-wise SEM across the three
replicas. They are useful for seeing disagreement between independent seeds,
but they do not correct for time correlation or turn 24 frames into 24
independent samples. Post 6's lesson still applies: stored-frame count and
effective sample count are different quantities.

{% include kups-notebooks/post-12/production-evidence.html %}

## NVE Drift Answers One Narrow Numerical Question

For an autonomous NVE system, the total energy should have bounded numerical
error when the force is conservative and the timestep is appropriate. For
each replica I fit a straight line to total energy per atom against time, take
the absolute slope, and report the mean and SEM across replicas. I also report
the full energy span. The production diagnostics are:

<div class="table-responsive" markdown="1">

| Regime | Absolute drift ± SEM (meV/atom/ps) | Energy span ± SEM (meV/atom) |
|---|---:|---:|
| ambient fcc | 0.00126 ± 0.00045 | 0.00321 ± 0.00007 |
| compressed fcc | 0.00446 ± 0.00289 | 0.00819 ± 0.00045 |
| hot expanded fcc | 0.00504 ± 0.00169 | 0.00624 ± 0.00212 |

</div>

All three drifts are small on this 0.12 ps horizon. The compressed and hot
cases have larger mean absolute slopes than ambient fcc, but three replicas
and a very short time series do not support a precise ordering. More
importantly, these are self-consistency numbers. They compare the numerical
trajectory with conservation of the Tojax MACE Hamiltonian that generated it.
They do not compare that Hamiltonian with density-functional theory or an
experiment.

This is why "the NVE drift is small" and "the potential is accurate" are
different sentences. A smooth but biased potential can conserve its own
energy perfectly. Conversely, a good potential can appear unstable when a
timestep is too large, precision is mishandled, or a neighbor update creates a
discontinuity. Energy drift is an excellent deployment diagnostic precisely
because it catches several numerical failures. It is not a reference-error
metric.

The short horizon matters too. A slope of 0.005 meV/atom/ps estimated over
0.12 ps should not be extrapolated linearly to nanoseconds. Longer trajectories
can encounter geometries absent from this test, reveal slow systematic trends,
or change the apparent slope as bounded oscillations average out. The capstone
keeps the duration beside the number so the unit does not imply evidence at a
timescale that was never simulated.

## Forces and Geometry Describe the Trajectory, Not Its Accuracy

The remaining diagnostics come directly from the NVT HDF5 arrays. The maximum
force is computed per frame from kUPS `position_gradients` and then averaged.
The nearest-neighbor distance uses the periodic minimum-image convention. RMS
displacement is measured from the initial lattice with the same minimum-image
wrapping. Potential energy is reported per atom.

<div class="table-responsive" markdown="1">

| Regime | Potential energy ± SEM (eV/atom) | Mean max force ± SEM (eV/Å) | Mean nearest neighbor ± SEM (Å) | RMS displacement ± SEM (Å) |
|---|---:|---:|---:|---:|
| ambient fcc | -3.70402 ± 0.00048 | 1.086 ± 0.052 | 2.6906 ± 0.0002 | 0.1903 ± 0.0004 |
| compressed fcc | -3.37802 ± 0.00194 | 1.703 ± 0.074 | 2.5345 ± 0.0011 | 0.1136 ± 0.0059 |
| hot expanded fcc | -3.51522 ± 0.00306 | 1.856 ± 0.158 | 2.6336 ± 0.0095 | 0.3340 ± 0.0157 |

</div>

The trends are physically interpretable at the level of the simulated model.
Compression shortens the measured nearest-neighbor distance and raises the
force scale relative to ambient fcc. The hot expanded trajectory has the
largest displacement and largest mean maximum force. These observations help
identify which regimes exercise the deployed computation differently.

They are not force errors. There are no reference forces in the table. The
potential energy values are not energy errors for the same reason. Calling the
largest force a failure would also be unjustified: high forces may be a
reasonable response to a compressed or thermally displaced structure. A
reference comparison on selected trajectory frames is the next validation
step, not an interpretation that can be recovered from the trajectory alone.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post12_mlip_diagnostics.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Post 12 production diagnostics from 18 real kUPS/Tojax MACE trajectories on NVIDIA RTX A5000 GPUs. Lines are means across three independent replicas and shaded bands show frame-wise replica SEM. The force, displacement, and drift panels describe this deployed model and protocol; they are not reference-accuracy or model-uncertainty estimates." %}

The figure is regenerated from the committed 432-row CSV and summary JSON.
The notebook only displays that canonical asset, so re-executing a presentation
cell cannot silently replace the reviewed plot with notebook-local state.

## One Checkpoint Cannot Give Epistemic Uncertainty

The earlier draft of this tutorial plotted synthetic "uncertainty" beside
synthetic force error. That panel was easy to draw and scientifically empty.
The production workflow uses one serialized MACE checkpoint. There is no
committee, posterior sample, calibrated distance score, or reference residual
model. The summary therefore records
`epistemic_uncertainty_available: false`.

The ± values in this article are independent-replica SEMs. For a scalar metric
$$m_r$$ from replica $$r$$, the reported value is

$$
\operatorname{SEM}(m)
=\frac{s(m_1,m_2,m_3)}{\sqrt{3}},
$$

where $$s$$ is the sample standard deviation. This quantifies variation across
three random initializations under one model and one protocol. It does not
quantify uncertainty in the potential-energy surface. If all three replicas
share the same model bias, their SEM can be tiny while the prediction is
wrong.

The same discipline applies to extrapolation. A hot or compressed cell may be
more challenging, but difficulty is not a numerical extrapolation score. A
defensible support claim could come from a calibrated model ensemble, a
distance in an appropriate local-environment representation, explicit
training-set coverage analysis, or reference labels on selected frames. Each
choice answers a slightly different question and must itself be validated.

For a real application, I would sample frames from all three regimes—not just
the ambient trajectory—and obtain reference energies, forces, and stresses.
I would stratify by force magnitude, local coordination, and time rather than
randomly selecting adjacent frames. I would compare more than one model or
checkpoint where possible. Only then would force RMSE, energy error, stress
error, or uncertainty coverage belong in this article.

## The Earlier Eleven Posts Still Apply

The capstone does not replace the earlier tutorials. It turns each one into a
question about the learned force provider.

Initialization still defines the ensemble you can reach. A periodic fcc cell,
cell strain, initial positions, masses, momenta, seed, and center-of-mass
treatment all precede the first neural-network evaluation. If two MLIP runs
start from different distributions, model comparison is already confounded.

Integrator analysis still separates discretization from force error. The 0.5
fs timestep used here produces small short-run NVE drift, but a timestep scan
would be needed to show convergence. A model that is more accurate but much
stiffer may require a smaller step. Runtime per step and stable timestep belong
in the same deployment discussion.

Error decomposition becomes more important, not less. Precision, timestep,
finite sampling, initialization, and potential bias can all move a reported
observable. "MLIP error" is not a useful residual bucket when the simulation
protocol itself has not been tested.

Thermostats and barostats control distributions generated by the supplied
forces. They cannot make an inaccurate Hamiltonian physically correct. The
warmup failure in this post is the simplest example: even a valid setpoint did
not make the short transient an equilibrated sample. For NPT work, pressure
and cell distributions would need the same scrutiny, plus reference stress and
equation-of-state checks.

Trajectory length still controls effective information. Neural-network force
evaluation may make long runs expensive, but expense does not reduce
autocorrelation. Three short replicas are useful engineering evidence. They are
not a substitute for a convergence study matched to the slow observable.

Observable estimation still begins with a definition and an estimator. An RDF,
coordination number, diffusion coefficient, or spectrum can be estimated with
tiny statistical error for the wrong potential. The trajectory must support
both the statistical estimator and the model-validity claim in the regions
that dominate it.

Free-energy methods add another layer. WHAM, MBAR, or reweighting can be
implemented correctly while reconstructing the free energy of a biased learned
Hamiltonian. Reference checks matter especially in barriers and low-probability
regions because a small energy bias changes probabilities exponentially.

Umbrella and enhanced-sampling methods actively visit configurations that an
unbiased trajectory may rarely see. That is their purpose and their MLIP risk.
Zero-bias parity, force-gradient checks, overlap, work accounting, and model
support all belong in the same record. A bias can accelerate a trajectory out
of the region where the force provider has been checked.

The resulting hierarchy is simple:

1. prove that the intended model bytes ran through the intended engine;
2. prove numerical stability and ensemble behavior for the chosen protocol;
3. quantify sampling error for the observable;
4. validate model accuracy on configurations relevant to that observable;
5. narrow the claim to the weakest supported link.

Skipping directly to step four is a mistake, but so is stopping after step one.
This tutorial completes the first two links for a small teaching protocol and
leaves the missing reference/model-support evidence explicit.

## Raw Trajectories Can Stay Large Without Becoming Untraceable

The 18 HDF5 files live under ignored `runs/` directories. They are not copied
into the website repository. Instead, the production manifest records, for
every trajectory:

- regime, replica, ensemble, integrator, and seed;
- requested device, observed JAX devices, and default backend;
- input CIF path and SHA-256;
- raw HDF5 path, SHA-256, byte count, frame count, and atom count;
- required dataset names, shapes, and dtypes;
- elapsed worker time and the scalar diagnostics derived from that file.

The compact CSV retains every stored frame's time, temperature, potential and
total energy per atom, maximum force, nearest-neighbor distance, and RMS
displacement. The manifest hashes both the CSV and summary. Verification
therefore detects a changed compact result even when the large raw trajectory
is unavailable in a fresh checkout.

This is not archival magic. A hash cannot reconstruct a deleted HDF5 file.
It does make the evidence chain falsifiable: a retained raw file can be checked
against the manifest, and a compact table cannot drift without failing its
recorded digest. For long-term scientific preservation, raw trajectories or a
documented subset should also be stored in an appropriate data repository.

## What This Experiment Does Not Establish

The strongest part of a validation report is often the boundary around its
claim. This capstone leaves several boundaries in place:

- No DFT or experimental labels were evaluated, so there is no accuracy claim.
- One model checkpoint ran, so there is no epistemic-uncertainty claim.
- No support metric was calibrated, so there is no in-domain/extrapolation
  classification.
- The NVE horizon is 0.12 ps, so there is no long-time stability claim.
- The NVT measurement is 0.12 ps with three replicas, so there is no converged
  thermodynamic or transport observable.
- NVE starts independently rather than from equilibrated NVT endpoints, so
  there is no thermostat-to-microcanonical handoff result.
- Only fcc Al cells with isotropic strain were tested. Defects, surfaces,
  liquids, multiple elements, reactions, and extreme coordination are outside
  this experiment.
- The run records adaptive neighbor-capacity handling through the actual kUPS
  execution, but this article does not turn that runtime mechanism into a
  calibrated model-risk score.

None of these limits makes the run pointless. They locate the next experiment.
For a melting-temperature claim, extend cell size and time, test phase-specific
equilibration, validate liquid and solid configurations, and quantify finite-
size and sampling error. For a defect migration barrier, validate forces and
energies along the transition region and combine that with window overlap. For
mechanical response, include stress references and strain-path convergence.

The claim should determine the validation set. A universal checklist can
prevent omissions, but it cannot decide which configurations matter.

## Reproduce the Evidence

Install the GPU and model-download extras on a CUDA machine, then run the two
profiles separately:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync --locked --extra gpu --extra mlff

uv run kups-tutorial run 12 --profile smoke
uv run kups-tutorial verify 12 --profile smoke

uv run kups-tutorial run 12 --profile full
uv run kups-tutorial verify 12 --profile full

uv run python scripts/generate_post12_figures.py
uv run kups-tutorial verify-notebooks --posts 12 --timeout 240
```

The smoke command forces CPU execution. The full configuration requests GPU,
and verification rejects it if any trajectory reports a non-GPU default
backend. Both paths reject an artifact whose SHA-256 differs from the pinned
value.

When adapting the workflow to another MLIP, freeze the following before
interpreting a trajectory:

<div class="table-responsive" markdown="1">

| Question | Minimum evidence |
|---|---|
| Which force function ran? | artifact repository, immutable revision, file hash, exporter, backend |
| Did the intended engine run? | entry point, engine version, raw trajectory schema and hashes |
| Which hardware executed it? | requested and observed devices; explicit fallback failure |
| Is integration stable? | timestep scan, NVE drift and span at relevant configurations |
| Is the ensemble equilibrated? | warmup study, trace review, replica agreement, effective samples |
| Is the model accurate where used? | reference labels selected from deployment trajectories |
| Does uncertainty mean model uncertainty? | defined estimator, calibration target, coverage test |
| Does the observable converge? | estimator diagnostics, autocorrelation, replica/length stability |
| Did biasing change the support problem? | overlap/work checks plus validation in biased regions |

</div>

The learned potential changes what must be frozen and validated. It does not
change the standard of evidence. A trustworthy MLIP simulation is still a
chain of explicit, testable claims—from bytes, to forces, to dynamics, to
sampling, to the observable that appears in the paper.

## References

1. <span id="ref-validation"></span>Morrow, J. D., Gardner, J. L. A. & Deringer, V. L. (2023). How to validate machine-learned interatomic potentials. *The Journal of Chemical Physics*, 158, 121501. [DOI](https://doi.org/10.1063/5.0139611) <a href="#cite-validation" class="reversefootnote" role="doc-backlink">↩</a>
2. <span id="ref-mace"></span>Batatia, I. et al. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *NeurIPS Workshop*. [arXiv](https://arxiv.org/abs/2206.07697) <a href="#cite-mace" class="reversefootnote" role="doc-backlink">↩</a>
3. Batatia, I. et al. (2025). A foundation model for atomistic materials chemistry. *The Journal of Chemical Physics*, 163, 184110. [DOI](https://doi.org/10.1063/5.0291759)
4. CuspAI. *kUPS documentation*. [Project documentation](https://cusp-ai-oss.github.io/kUPS/)
5. CuspAI. *Tojax: Export JAX computations and load them anywhere*. [Source repository](https://github.com/cusp-ai-oss/tojax)
