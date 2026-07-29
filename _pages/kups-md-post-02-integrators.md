---
layout: post
permalink: /kups-md-tutorials/post-02-integrators/
title: "What Does an MD Integrator Actually Approximate?"
date: 2026-07-14
last_updated: 2026-07-29
description: "What velocity Verlet preserves, what a finite timestep changes, and how reversibility and energy error expose the discrete map."
post_type: tutorial
authors: ["Sungsoo Ahn"]
order: 2
series: kups-md-tutorials
series_title: "kUPS Molecular Dynamics Tutorials"
series_description: "Executable molecular-dynamics practice for ML researchers who are new to simulation."
series_order: 2
categories: [science]
tags: [molecular-dynamics, integrators, kups, reproducibility]
toc:
  sidebar: left
related_posts: false
nav: false
publication_status: draft
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Part 2 of the kUPS Molecular Dynamics Tutorials. The executable example is maintained in <a href="https://github.com/sungsoo-ahn/kups-md-tutorials">sungsoo-ahn/kups-md-tutorials</a>.</em>
</p>


The equation of motion is continuous, but an MD trajectory is not. Every saved
state came from a discrete map: positions and momenta at one time were
converted into positions and momenta at the next time. The integrator is that
map. The velocity-Verlet family became central to molecular simulation because
it gives a simple reversible, second-order update for Hamiltonian dynamics
(<span id="cite-verlet1967"></span>[Verlet, 1967](#ref-verlet1967)).

For ML researchers who already know

$$\dot{\mathbf{r}}_i=\mathbf{v}_i,\qquad m_i\dot{\mathbf{v}}_i=\mathbf{F}_i,$$

the practical question is not whether Newton's equation is correct. It is what
the finite timestep update preserves, what it distorts, and which diagnostics
can tell the difference between bounded discretization error and real
simulation drift.

This article uses a dimensionless harmonic oscillator as a microscope. That
choice is deliberately small. A one-dimensional oscillator is not a production
molecular simulation, but it has the important property that the exact
trajectory is known. Because the reference solution is analytic, the diagnostic
can expose the numerical map itself without confusing integrator error with
force-field error, neighbor-list rebuilds, thermostat coupling, barostat motion,
finite-size effects, or MLIP extrapolation.

The first tutorial fixed an initial state. This second tutorial asks what
happens when that state is advanced. The answer is not "the computer solves
Newton's equation." The computer applies a map with a timestep. That map may be
time-reversible or not. It may be symplectic or not. It may conserve a nearby
shadow Hamiltonian while the reported physical energy oscillates. It may show a
small bounded error for a very long time, or it may inject energy until a toy
oscillator explodes.

Those distinctions matter in real MD. A trajectory can have energy oscillations
that are acceptable for a symplectic integrator at a chosen timestep. It can
also have systematic drift from a timestep that is too large, discontinuous
forces, neighbor-list mistakes, precision problems, or a learned potential that
is being evaluated outside its training distribution. The point of this page is
to make the integrator part precise before later posts add those other failure
modes.

## What Is the Discrete Object?

The continuous equations define a flow. Give the exact flow an initial state
and a time interval, and it returns the state at that later time. A numerical
integrator replaces that exact flow with a discrete update. For a timestep dt,
the code applies the same update repeatedly:

$$x_{n+1}, v_{n+1} = \Phi_{\Delta t}(x_n, v_n).$$

The map Phi_dt is the real computational object. Its properties are not
guaranteed merely because the differential equation has good properties. The
exact Hamiltonian flow conserves phase-space volume and energy. A generic
finite-difference update may not. It can distort phase space, amplify energy,
or fail to retrace its path when velocities are reversed.

This is why integrator names matter. "Velocity Verlet" does not just mean
"second order." It means a particular composition of position drifts and
momentum kicks for separable Hamiltonians. In one common form,

$$v_{n+1/2} = v_n + \frac{\Delta t}{2m} F(x_n),$$

$$x_{n+1} = x_n + \Delta t\, v_{n+1/2},$$

$$v_{n+1} = v_{n+1/2} + \frac{\Delta t}{2m} F(x_{n+1}).$$

This update asks for forces at the old and new positions and stores velocities
at integer steps. Other Verlet-family conventions store half-step velocities or
arrange the kick and drift operations differently. Those conventions can be
equivalent after a change of variables, but they are not interchangeable in
code unless the stored state and force-update convention are handled carefully.

For a separable Hamiltonian, velocity Verlet is time-reversible and symplectic.
Time-reversible means that if the map is applied forward, velocities are
reversed, and the same map is applied again, the state returns to the start up
to numerical precision. Symplectic means the map preserves the Hamiltonian
phase-space geometry. Informally, it does not squeeze or expand the canonical
position-momentum volume in the way dissipative or unstable maps can.

These properties do not make every timestep acceptable. They explain why the
error often has a particular shape. A stable symplectic method can show bounded
energy error rather than monotonic energy drift, because it nearly conserves a
modified energy. That modified energy is often called a shadow Hamiltonian. The
simulation is still approximate, but the approximation is structured.
This is the geometric-numerical-integration perspective behind shadow-energy
diagnostics (<span id="cite-leimkuhler2004"></span>[Leimkuhler & Reich,
2004](#ref-leimkuhler2004); <span id="cite-hairer2006"></span>[Hairer et al.,
2006](#ref-hairer2006)).

## Why Use a Harmonic Oscillator?

The diagnostic fixes a harmonic oscillator with mass 1.0, angular frequency
1.0, initial position 1.0, and initial velocity 0.0. The exact solution is
simple:

$$x(t) = \cos(t), \qquad v(t) = -\sin(t).$$

The exact energy is 0.5 for this initial condition. This gives three useful
checks. The phase-space orbit should stay close to the unit circle. The energy
error should reveal whether the map has bounded oscillatory behavior or
monotonic growth. A forward/backward check should expose reversibility.

The current diagnostic fixes:

| Choice | Full value | Why it matters |
|---|---:|---|
| system | harmonic oscillator | exact reference trajectory is known |
| mass | 1.0 | dimensionless controlled example |
| angular frequency | 1.0 | sets the natural timescale |
| initial position | 1.0 | starts at the turning point |
| initial velocity | 0.0 | makes the phase-space orbit easy to inspect |
| timesteps | 0.02, 0.05, 0.1, 0.2 | exposes timestep sensitivity |
| steps per run | 2000 | separates bounded error from growth |

The oscillator is not used because molecular systems are harmonic. It is used
because good numerical behavior should be visible in the cleanest case before
it is trusted in a messy one. If an update rule fails this test, it should not
be trusted as a production MD integrator. If it passes, that is not a complete
validation; it means the map has the expected structure in a controlled case.

The full profile runs four timesteps: 0.02, 0.05, 0.1, and 0.2, each for 2000
steps. The final simulated time therefore grows with dt: 40, 100, 200, and 400.
This is useful because it shows how local timestep error accumulates over many
applications of the map. The smoke profile uses fewer steps and a smaller
timestep set so CI can catch broken logic quickly.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post02_phase_space.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Velocity Verlet follows the exact oscillator orbit closely in phase space. The closed orbit tests the integrator map without force-field or sampling noise." %}

## What Does Velocity Verlet Preserve?

The most important lesson is not that velocity Verlet is accurate at every
timestep. It is that its errors are structured. In the full diagnostic, velocity
Verlet's maximum relative energy error stays small across the sweep:
approximately 1.0e-4 at dt = 0.02, 6.25e-4 at dt = 0.05, 2.5e-3 at dt = 0.1,
and 1.0e-2 at dt = 0.2. The error grows with timestep, as it should. But it
does not explode over 2000 steps.

The signed final energy drift is also small compared with the explicit Euler
control: about -2.77e-5 at dt = 0.02, -7.73e-5 at dt = 0.05, -8.60e-4 at
dt = 0.1, and -4.93e-3 at dt = 0.2. The larger timestep is visibly worse, but
the result is still a bounded-error story for this oscillator, not runaway
energy injection.

The reversibility check is even sharper. The full profile records velocity
Verlet reversibility errors at roundoff scale: roughly 3.6e-15, 2.8e-15,
3.7e-15, and 7.8e-16 across the timestep sweep. That supports the claim that
the implemented velocity-Verlet map is reversible for this separable test
problem. It is not merely producing a plausible-looking trajectory.

This is why plotting only position versus time is usually too weak. A curve can
look reasonable while the map violates the structure one expected. Integrator
diagnostics should ask about phase-space geometry, energy behavior, and
reversibility before the article claims that a timestep is acceptable.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post02_energy_error.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="The timestep sweep separates bounded velocity-Verlet error from explicit Euler's runaway energy. The negative control shows that solving the same differential equation does not give every discrete map the same stability." %}

## Why Keep Explicit Euler?

Explicit Euler is not included as a serious production MD method. It is a
negative control. It takes the derivative at the current state and steps
forward directly. For the oscillator, that simple update moves the state
outward in phase space. Energy grows instead of oscillating around the correct
value.

The full results make the contrast unambiguous. At dt = 0.02, explicit Euler
already ends with energy about 1.1126 instead of 0.5 after 2000 steps. At
dt = 0.05, the final energy is about 73.745. At dt = 0.1, it is about
2.196e8. At dt = 0.2, it is about 5.83e33. The log-scale figure has to span
many orders of magnitude because this negative control is supposed to fail.

That failure is useful pedagogically. It prevents a common misunderstanding:
"small timestep" does not make any update rule conceptually safe. Reducing dt
helps, but the map still has a structure. A non-symplectic unstable update can
accumulate qualitatively wrong behavior even for a simple oscillator. A stable
MD workflow needs both an appropriate timestep and an appropriate integrator.

The negative control also clarifies what a diagnostic should not do. If a plot
only reports final energy drift for one chosen timestep, a bad update may look
acceptable by accident over a short run. A sweep across timesteps, with an
exact reference and a reversibility check, gives a much stronger picture.

## What Does Bounded Energy Error Mean?

Energy conservation is often used as an MD sanity check, but it is easy to use
too bluntly. A velocity-Verlet trajectory at finite timestep does not reproduce
the exact continuous energy at every step. Instead, for stable timesteps in
Hamiltonian systems, the reported energy typically oscillates around the true
energy while the numerical trajectory follows a nearby modified Hamiltonian.

That is very different from monotonic drift. Bounded oscillation says the
integrator is making a structured discretization error. Drift says something is
systematically adding or removing energy. In real MD, drift can come from too
large a timestep, discontinuous forces, poor neighbor-list settings, mixed
precision, constraint tolerances, thermostat/barostat coupling, or a learned
potential with noisy forces. Later posts separate those mechanisms. This page
only isolates the integrator map.

The difference matters for choosing a timestep. If dt is made smaller,
velocity-Verlet energy oscillations shrink in this test. But dt also controls
cost: halving the timestep roughly doubles the number of force evaluations for
the same physical time. In production MD, a timestep is a compromise between
accuracy, stability, and compute budget. The right diagnostic is not "does the
trajectory run?" It is "does the timestep keep the relevant conserved or
controlled quantities within a scientifically acceptable error?"

For MLIP simulations, this question becomes sharper. A learned potential may
have force noise or extrapolation artifacts that interact with the integrator.
If the timestep is too large, a trajectory can leave the model's reliable
region. If the model has discontinuities or noisy derivatives, even a familiar
integrator can show drift. One should not blame "the MLIP" before checking the
plain integrator/timestep behavior, and one should not blame "the timestep"
before checking the force model.

## What Does Reversibility Test?

A reversible map has a simple operational test. Start from a state. Apply the
map forward for N steps. Reverse velocities. Apply the same map for N steps.
Reverse velocities again. For an exactly reversible map in exact arithmetic,
the result should be the starting state. In floating-point arithmetic, the
error should be near roundoff for a clean test.

This diagnostic is not a proof of correctness for every system. It is a useful
implementation check. A wrong force-update convention, an accidental use of a
new force at the wrong point, or a state variable stored at the wrong half step
can break reversibility. The forward trajectory may still look plausible for a
short time, so the explicit reversibility check catches a class of bugs that
ordinary plots can miss.

Reversibility is also conceptually important for equilibrium sampling. Many
thermostats, Markov-chain moves, and detailed-balance arguments rely on
time-reversal structure. The deterministic integrator is only one part of those
algorithms, but if its state convention is unclear, the sampling argument can
become unclear too. This is one reason production MD packages are careful about
velocity definitions, half-step updates, and restart files.

{% include figure.liquid loading="eager" path="assets/img/blog/kups_md_post02_reversibility.svg" class="img-fluid rounded z-depth-1" zoomable=true caption="Forward integration followed by velocity reversal returns to the initial state at floating-point scale. Reversibility tests the implemented update sequence, not the physical accuracy of the force model." %}

## How Do Force Evaluations Fit In?

Velocity Verlet is cheap in a specific sense: for a force that depends only on
positions, it needs one new force evaluation per step after the initial force is
known. This matters because force evaluation dominates cost in atomistic
simulation. In classical pair potentials it may mean neighbor-list traversal.
In MLIP simulations it may mean a neural network evaluation, graph construction,
and GPU memory traffic.

The update convention affects how forces are reused. A common implementation
stores the old force, updates positions with a half-step velocity, computes a
new force at the new positions, and finishes the velocity update. Restart files
and analysis code must know which velocity is stored. A half-step velocity and
a full-step velocity are not the same observable. If a code writes one but a
post-processing script assumes the other, kinetic-energy and temperature
diagnostics can be shifted.

This page does not benchmark force calls because the oscillator force is
trivial. The point is to name the bookkeeping. In later tutorials, the same
map-level thinking applies when the force comes from kUPS, neighbor lists,
precision policies, or MACE. The integrator is not independent of force
evaluation; it is the schedule by which force evaluations are requested and
used.

## Run the Example

The smoke profile follows the same protocol with a smaller CPU workload:

```bash
git clone https://github.com/sungsoo-ahn/kups-md-tutorials
cd kups-md-tutorials
uv sync
uv run kups-tutorial run 02 --profile smoke
uv run kups-tutorial verify 02 --profile smoke
```

The repository also contains the [full configuration](https://github.com/sungsoo-ahn/kups-md-tutorials/blob/main/configs/post-02/full.json), [notebook](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/notebooks), and [recorded results](https://github.com/sungsoo-ahn/kups-md-tutorials/tree/main/results/post-02/full).

## References

- <span id="ref-verlet1967"></span>Verlet, L. (1967). Computer "experiments" on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules. *Physical Review*, 159(1), 98-103. <a href="#cite-verlet1967" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-leimkuhler2004"></span>Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press. <a href="#cite-leimkuhler2004" class="reversefootnote" role="doc-backlink">↩</a>
- <span id="ref-hairer2006"></span>Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. <a href="#cite-hairer2006" class="reversefootnote" role="doc-backlink">↩</a>
