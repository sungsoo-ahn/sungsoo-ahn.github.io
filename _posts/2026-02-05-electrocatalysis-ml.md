---
layout: post
title: "Introduction to Heterogeneous Electrocatalysis"
date: 2026-02-05
last_updated: 2026-02-11
description: "An introduction to heterogeneous electrocatalysis — the energy storage problem, why oxides matter, the solid-liquid interface, and the complexities of real catalyst design."
order: 1
categories: [science]
tags: [electrocatalysis, renewable-energy, machine-learning, density-functional-theory]
toc:
  sidebar: left
related_posts: false
published: true
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post introduces heterogeneous electrocatalysis — the problem setting that motivates large-scale catalyst design efforts like the Open Catalyst project. The first half follows <a href="https://arxiv.org/abs/2010.09435">Zitnick et al. (2020)</a>, which covers the energy storage problem, the Sabatier principle, and the idealized metal-surface picture. The second half draws on <a href="https://arxiv.org/abs/2206.08917">Tran et al. (2023)</a> and <a href="https://arxiv.org/abs/2509.17862">Shuaibi et al. (2025)</a> to explain why real catalysts — oxides, disordered materials, surfaces in liquid — are far more complex. I wrote this as a bridge between my previous posts on <a href="/blog/2026/quantum-chemistry-dft/">DFT</a> and <a href="/blog/2026/spherical-equivariant-layers/">equivariant GNNs</a>, and the application domain where these methods have the most impact. Corrections are welcome.</em>
</p>

## Introduction

Here is the problem: design a catalyst[^catalyst] material that achieves a target adsorption energy[^adsorption-energy] for a given chemical reaction.

Why this matters: the Sabatier principle (Section 4) says that the best catalyst for a reaction binds intermediates[^intermediate] at a specific, known strength — not too strongly, not too weakly. Scaling relations (Section 5) further reduce the problem: a single number, the adsorption energy of one key intermediate, determines catalyst performance. The optimal value of that number is known from theory. What is *not* known is which material achieves it.

The search space is enormous. Catalyst surfaces are built from ~40 candidate metals in alloys of 1–3 elements, cut along different crystal facets,[^facet] with multiple binding sites[^binding-site] per surface. Combined with 82 relevant adsorbate[^adsorbate] molecules, the number of candidate configurations runs into the billions. Evaluating each one requires a DFT relaxation[^dft-relaxation] — an iterative quantum-mechanical simulation costing hours to days per candidate. Exhaustive evaluation is infeasible.

This is a natural problem for ML: learn a surrogate that maps material structure to adsorption energy, then search or generate candidates that hit the target. The rest of this post explains where the target comes from, why the search space is structured the way it is, and what makes this problem scientifically important.

### Roadmap

| Section | What It Explains |
|---------|-----------------|
| **The Energy Storage Problem** | Why electrocatalysis matters: the bottleneck for grid-scale renewable energy |
| **Fuel Cells and Electrolyzers** | The devices that need catalysts — and why platinum is too expensive |
| **Electrocatalysis** | How reactions happen on surfaces: adsorbates, intermediates, energy barriers |
| **The Sabatier Principle** | Why optimal binding strength exists — and the volcano plot |
| **Scaling Relations and the Search Space** | Why one adsorption energy suffices as a descriptor, and the structure of the design space |
| **Why Oxides Matter** | The oxygen evolution reaction, the iridium problem, and why oxide catalysts are the frontier |
| **Beyond Ideal Surfaces** | Oxide-specific complexity, the solid-liquid interface, compositional diversity, and where scaling relations fail |
| **Machine Learning for Catalyst Discovery** | Surrogate models, generative design, and the progression from energy prediction to structure generation |

---

## The Energy Storage Problem

Renewable electricity from solar and wind is intermittent. Solar output peaks at midday, but demand peaks in the evening — a mismatch known as the **duck curve**.[^duck-curve] Grid-scale energy storage is the missing piece for full renewable adoption.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_duck_curve.png" class="img-fluid rounded z-depth-1" zoomable=true caption="California hourly energy demand. The green area shows wind and solar generation peaking midday while total demand (black) peaks in the evening — the 'duck curve.' The gap must be filled by other sources or storage. From Zitnick et al. (2020)." %}

Several storage technologies exist, each with trade-offs:

- **Pumped-storage hydropower (PSH):** 70–80% round-trip efficiency, but requires specific geography (two reservoirs at different elevations). Already accounts for 95% of grid storage worldwide — and most good sites are taken.
- **Batteries:** 60–95% round-trip efficiency, but cost-prohibitive at the scale needed for multi-day or seasonal storage. Lithium-ion costs have fallen dramatically, yet storing a full day of U.S. electricity demand (~100 TWh) in batteries remains economically impractical.
- **Hydrogen energy storage (HES):** Use excess electricity to split water (electrolysis),[^electrolysis] store the hydrogen, and convert it back to electricity in a fuel cell[^fuel-cell] when needed. Round-trip efficiency is lower (~35%), but hydrogen can be stored in bulk at low cost — underground caverns, pressurized tanks, or converted to methane for existing natural gas infrastructure.

The efficiency gap is real: HES wastes roughly two-thirds of the input energy.[^hes-efficiency] But efficiency is not the only constraint — what matters at grid scale is the total cost of stored energy. Zitnick et al. (2020) estimate HES at \$113/MWh, competitive with batteries for multi-day storage where the low cost of bulk hydrogen storage offsets the efficiency loss.

The bottleneck is not storage capacity. It is the **catalyst**. Both the electrolyzer (splitting water) and the fuel cell (recombining hydrogen with oxygen) require electrocatalysts[^electrocatalyst] to drive their reactions at practical rates. The dominant catalyst is platinum, which is scarce and expensive. Reducing or replacing platinum is the key to making HES economically viable.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_energy_cycle.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The hydrogen energy storage cycle. Renewable electricity powers an electrolyzer that splits water into hydrogen. The hydrogen can be stored directly or converted to methane via methanation with captured CO2. Fuel cells convert stored fuel back to electricity for the grid. From Zitnick et al. (2020)." %}

---

## Fuel Cells and Electrolyzers

The previous section identified the catalyst as the bottleneck for hydrogen energy storage. But where exactly does catalysis happen, and what reactions need to be catalyzed? The answer lies in two devices: the **electrolyzer**, which converts electricity into hydrogen, and the **fuel cell**, which converts hydrogen back into electricity. Understanding their internal structure reveals the precise chemical reactions that a catalyst must accelerate — and why those reactions are hard.

A **proton exchange membrane (PEM) fuel cell** converts hydrogen and oxygen into electricity and water. It has three layers:

- **Anode:**[^anode-cathode] Hydrogen gas arrives and is split into protons and electrons: $$\text{H}_2 \rightarrow 2\text{H}^+ + 2e^-$$.
- **Membrane:** A polymer electrolyte[^electrolyte] that conducts protons (H$$^+$$) but blocks electrons, forcing them through an external circuit — producing useful electrical current.
- **Cathode:** Oxygen combines with the protons and electrons to form water: $$\tfrac{1}{2}\text{O}_2 + 2\text{H}^+ + 2e^- \rightarrow \text{H}_2\text{O}$$.

An **electrolyzer** runs the same reactions in reverse: apply a voltage to split water into hydrogen and oxygen. Both devices need a catalyst at each electrode[^electrode] to make the reactions proceed fast enough to be practical.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_fuel_cell.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="Schematic of a PEM fuel cell. Hydrogen enters at the anode (left, red), is split into protons and electrons. Protons pass through the membrane (gold center), electrons flow through an external circuit (top) to power a load. At the cathode (right, blue), oxygen combines with protons and electrons to form water. From Zitnick et al. (2020)." %}

### The Platinum Problem

Platinum is the best-known catalyst for both the hydrogen oxidation reaction (anode) and the oxygen reduction reaction[^orr] (cathode). It sits near the peak of the activity volcano (explained below) — binding reactants strongly enough to catalyze the reaction, but weakly enough to release the products.

The problem is cost and scarcity:

- Platinum accounts for over **40% of fuel cell capital costs** when including the support structures needed for adequate power density (Zitnick et al., 2020).
- To supply 35 TWh/day of electricity via HES would require ~2,000 metric tons of platinum. Known world reserves are ~70,000 metric tons.[^pt-reserves]
- A survey of automotive fuel cell experts found that **76% identified platinum cost** as the primary barrier to reducing fuel cell costs.

Research suggests a 90% reduction in platinum loading may be achievable, and entirely platinum-free catalysts are an active area of research. But finding them requires searching an enormous space of candidate materials — and evaluating each candidate is computationally expensive.

---

## Electrocatalysis

**Heterogeneous catalysis**[^heterogeneous] involves a reaction between a solid surface (the catalyst) and gas- or liquid-phase reactants. The catalyst is not consumed — it provides a surface where reactants can adsorb,[^adsorption] react, and desorb as products.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_adsorption_types.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="Potential energy as a function of distance between an adsorbate and a surface. Physisorption (gold) is a weak van der Waals attraction. Chemisorption (blue) involves forming a chemical bond — the deeper well corresponds to the adsorption energy that determines catalytic activity. From Zitnick et al. (2020)." %}

### The Oxygen Reduction Reaction

The cathode reaction in a fuel cell — the **oxygen reduction reaction (ORR)** — is the harder half to catalyze and the primary target for improvement. Consider the dissociative pathway[^dissociative] on a metal surface, where $$*$$ denotes a binding site on the catalyst:

$$\tfrac{1}{2}\text{O}_2 + * \;\longrightarrow\; *\text{O}$$

$$*\text{O} + \text{H}^+ + e^- \;\longrightarrow\; *\text{OH}$$

$$*\text{OH} + \text{H}^+ + e^- \;\longrightarrow\; \text{H}_2\text{O} + *$$

Each step involves an **adsorbate** ($$*$$O or $$*$$OH) bound to the catalyst surface. The reaction proceeds through these intermediate states, and the catalyst is regenerated at the end — the binding site $$*$$ is freed.

### Gibbs Free Energy and Reaction Barriers

Each step has a **Gibbs free energy change** $$\Delta G$$, which determines whether the step is thermodynamically favorable ($$\Delta G < 0$$) or requires energy input ($$\Delta G > 0$$). The **Gibbs free energy** is:

$$G = H - TS$$

where $$H$$ is enthalpy,[^enthalpy] $$T$$ is temperature, and $$S$$ is entropy.[^gibbs]

A catalyst works by lowering the activation energy[^activation-energy] barriers between steps — not by changing the overall thermodynamics (the total $$\Delta G$$ from reactants to products is fixed), but by providing an alternative pathway with lower barriers.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_activation_energy.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Energy diagram for a single reaction step: O2 dissociating on a surface. The activation energy (0.62 eV) is the barrier the system must overcome. The reaction free energy (−1.5 eV) is the net energy change. A catalyst lowers the activation energy without changing the net free energy. From Zitnick et al. (2020)." %}

The key quantity for catalyst screening is the **adsorption energy**: how strongly each intermediate binds to the surface. If binding is too strong, the intermediate cannot desorb (the catalyst is "poisoned"). If binding is too weak, the intermediate never forms in the first place.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_gibbs_energy.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Gibbs free energy diagram for the dissociative ORR pathway on Pt(111) (blue) and Ni(111) (green). Pt has moderate energy barriers at each step — close to ideal. Ni binds oxygen too strongly: the initial steps drop much further, but the activation barriers between steps are correspondingly larger. From Zitnick et al. (2020)." %}

---

## The Sabatier Principle

> **The Sabatier principle.** The optimal catalyst binds reaction intermediates neither too strongly nor too weakly. Too strong, and products cannot desorb. Too weak, and reactants cannot adsorb. Maximum activity occurs at an intermediate binding strength.
{: .block-definition }

This qualitative principle becomes quantitative through the **Brønsted-Evans-Polanyi (BEP) relations**: across a family of catalysts, the activation energy $$E_a$$ for a given elementary step is linearly related to the reaction energy $$\Delta E$$:

$$E_a = \alpha \, \Delta E + \beta$$

where $$\alpha$$ and $$\beta$$ are constants specific to the reaction step.[^bep] The reaction rate follows the Arrhenius equation:[^arrhenius]

$$r \propto \exp\!\left(-\frac{E_a}{k_B T}\right)$$

Combining these: as adsorption energy becomes more negative (stronger binding), $$E_a$$ decreases for adsorption steps but increases for desorption steps. The overall rate is limited by the slowest step. On the strong-binding side, desorption is rate-limiting — the rate decreases as binding strengthens. On the weak-binding side, adsorption is rate-limiting — the rate decreases as binding weakens. The result is a **volcano plot**: reaction rate versus adsorption energy traces an inverted-V shape, with the optimum at the peak.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_volcano_plot.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="Volcano plot for the ORR. Catalytic activity (log scale) vs. oxygen adsorption energy. Platinum and palladium sit near the peak. Metals to the left (Fe, W, Ni) bind too strongly; metals to the right (Ag, Au) bind too weakly. The two branches correspond to different rate-limiting steps. From Nørskov et al. (2004), as presented in Zitnick et al. (2020)." %}

The volcano plot is the central organizing principle of electrocatalysis. It reduces the problem of finding a good catalyst to a one-dimensional search: find a material whose adsorption energy places it at the volcano peak. The optimal adsorption energy is known from theory — what remains is to find a material that achieves it.

---

## Scaling Relations and the Search Space

### Why One Binding Energy Suffices

The ORR involves multiple intermediates ($$*$$O, $$*$$OH, $$*$$OOH), each with its own binding energy. In principle, optimizing the catalyst requires tuning all of them independently. In practice, they are correlated.

The $$*$$OH and $$*$$OOH binding energies are **linearly correlated** across all known catalysts, with a constant offset of ~3.2 eV:

$$\Delta G_{*\text{OOH}} \approx \Delta G_{*\text{OH}} + 3.2 \;\text{eV}$$

This **scaling relation** arises because both $$*$$OH and $$*$$OOH bond to the surface through their oxygen atom — the catalyst surface "sees" essentially the same binding chemistry. The hydrogen atoms point away and contribute little to the interaction.

The consequence: a single descriptor — e.g., $$\Delta G_{*\text{OH}}$$ — determines a catalyst's position on the volcano. This is what makes the search tractable despite the high-dimensional space of possible materials.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_scaling_relations.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="2D volcano plot showing catalytic activity (color, red = highest) as a function of OH and OOH binding free energies. The dashed red line is the scaling relation — all known catalysts (dots) cluster along it. The activity peak (red region) sits slightly off the line, meaning the scaling relation itself limits achievable performance. Pt(111) is labeled near the peak. From Zitnick et al. (2020)." %}

### The Constraint — and the Open Challenge

The scaling relation is both a gift and a curse. It simplifies the search to one dimension, but it also imposes a **fundamental limit**: because all known catalysts lie on (or near) the scaling line, they are all constrained to the same trade-off. The ideal catalyst — one that binds $$*$$OH at the volcano peak while also binding $$*$$OOH at its independently optimal value — would need to break the scaling relation. This remains an open challenge. Strategies include nanostructured surfaces, alloys with specific local environments, and single-atom catalysts, but none have definitively broken the linear scaling.

### The Design Space

Even with the simplification to a single descriptor, the space of candidate catalyst configurations is vast:

- **~40 metals** can appear in a catalyst, in combinations of 1–3 elements (yielding over 10,000 compositions before considering ratios).
- Each composition can be cut along multiple **crystal facets** — (100), (110), (111), and others — exposing different surface arrangements.
- Each surface has multiple **binding sites**: an adsorbate can bond to 1 atom (atop), 2 atoms (bridge), or 3 atoms (hollow).
- **82 adsorbate molecules** are relevant intermediates of reactions important for renewable energy.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_catalyst_surface.png" class="img-fluid rounded z-depth-1" zoomable=true caption="3D rendering of a catalyst surface with adsorbate molecules (red and white atoms) bound at different sites on a close-packed metal surface (gray). The adsorbates are small compared to the surface — their binding energy depends on the local arrangement of surface atoms. From Zitnick et al. (2020)." %}

{% include figure.liquid loading="eager" path="assets/img/blog/ec_catalyst_types.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Types of catalyst materials in the OC20 search space. From left: pure metal, multi-metallic alloy, intermetallic compound (ordered), overlayer (thin film on bulk), and high-entropy alloy (5+ elements, disordered). Each type has different surface chemistry. From Zitnick et al. (2020)." %}

{% include figure.liquid loading="eager" path="assets/img/blog/ec_miller_indices.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Crystal facets exposed by cutting along different planes (Miller indices). Top: the (100), (110), and (111) cutting planes through a cubic unit cell. Bottom: the resulting surface arrangements — (111) is the most densely packed. Different facets expose different binding site geometries. From Zitnick et al. (2020)." %}

{% include figure.liquid loading="eager" path="assets/img/blog/ec_adsorbates.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="The 82 adsorbate molecules in the OC20 dataset, grouped by composition: O and H (top), small molecules with one carbon (C1), larger molecules with two or more carbons (C2), and nitrogen-containing molecules (N). These are intermediates of reactions relevant to renewable energy storage. From Zitnick et al. (2020)." %}

Combining all factors, there are on the order of **1,000 candidate configurations per catalyst composition**, and billions of total (composition, facet, site, adsorbate) combinations to evaluate. Each evaluation requires a DFT relaxation — an $$O(n^3)$$ iterative simulation taking hours to days. This is the search problem that motivates ML approaches to catalyst design.

---

## Why Oxides Matter

The previous sections focused on the fuel cell cathode (the ORR) and used pure metals — especially platinum — as running examples. The other half of the hydrogen energy cycle is water splitting, and there the bottleneck is a different reaction: the **oxygen evolution reaction (OER)**.[^oer]

$$2\text{H}_2\text{O} \;\longrightarrow\; \text{O}_2 + 4\text{H}^+ + 4e^-$$

The OER is kinetically sluggish — it requires forming an O–O bond through a sequence of four proton-coupled electron transfers, each with its own energy barrier. It is the primary source of efficiency loss in electrolyzers and the main target for catalyst improvement on the water-splitting side.

Metal oxides are the dominant catalyst class for the OER. The reason is stability: water splitting typically operates under strongly acidic conditions to reduce gas solubility and improve proton conductivity. Under these conditions, most pure metals dissolve. Oxides survive. The best-known stable and active OER catalyst is **iridium oxide (IrO$$_2$$)** — but iridium is rarer and more expensive than platinum (Tran et al., 2023). Finding cheaper multi-component oxide catalysts that match IrO$$_2$$ in both activity and acid stability is a central goal of electrocatalysis research.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_oer_workflow.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The OER catalyst discovery workflow. (a) Select a bulk oxide structure. (b) Enumerate surface terminations and identify the most stable one via surface Pourbaix diagrams. (c) Place adsorbate intermediates. (d) Relax the structure and compute adsorption energy. Steps (a)–(b) are unique to oxides — on metals, the surface is determined by the facet alone. From Tran et al. (2023)." %}

### Why Oxides Are Harder Than Metals

Oxide electrocatalysts introduce at least five layers of complexity absent from pure metal surfaces (Tran et al., 2023):

1. **Multiple polymorphs.** A given oxide composition (e.g., TiO$$_2$$) can crystallize in several distinct structures (rutile, anatase, brookite), each with different surface chemistry. All must be screened.
2. **Surface terminations.** Cutting a crystal along a given plane can expose different atomic layers. A rutile (110) surface has at least three possible terminations,[^termination] each presenting different atoms to the adsorbate. On metals, the facet determines the surface; on oxides, the termination adds another degree of freedom.
3. **Oxygen vacancies.** Surface oxygen atoms can be removed — by thermal treatment, electrochemical reduction, or solvent dissolution — leaving behind vacancy defects that serve as active sites. The number and arrangement of vacancies affects both activity and selectivity.
4. **Active site ambiguity.** It is often unclear which surface site is catalytically active, and multiple competing reaction mechanisms may operate simultaneously.
5. **Stronger electron correlation.** Standard DFT functionals (GGA) are less accurate for oxides because of strong electron–electron interactions in transition metal d-orbitals. Hubbard U corrections[^hubbard-u] or more expensive hybrid functionals are needed.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_oxide_terminations.png" class="img-fluid rounded z-depth-1 mx-auto d-block" max-width="550px" zoomable=true caption="Surface terminations of rutile (110). (a) Three possible terminations (T1, T2, T3) obtained by cutting at different depths. The dashed blue box marks the surface unit cell. (b) Labeled surface oxygen sites — removing any subset of these creates vacancy defects, each configuration with different catalytic properties. From Tran et al. (2023)." %}

### Adsorbate Binding on Oxides

On a metal surface, adsorbates bind to metal atoms at well-defined sites (atop, bridge, hollow). Oxides add a second class of binding interactions: adsorbates can bind to **surface oxygen atoms** rather than to metal atoms. This enables reactions that have no analogue on metals.

The **Mars-van Krevelen (MvK) mechanism**[^mvk] is the most important example. In MvK, an incoming adsorbate reacts with a lattice oxygen atom on the surface — forming a new intermediate that desorbs and leaves behind an oxygen vacancy. The vacancy is later replenished by oxygen from the next adsorbate. The catalyst surface itself participates as a reactant, cycling between oxidized and reduced states.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_oxide_adsorbates.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="Adsorbate placement strategies on oxide surfaces. Top row: adsorbates bind to undercoordinated surface metals at lattice oxygen positions (including vacancy sites). Bottom row: adsorbates bind to existing surface oxygen to form new intermediates — e.g., CO on surface O forms CO2, monatomic O on surface O forms a dimer. This second class of binding is unique to oxides. From Tran et al. (2023)." %}

---

## Beyond Ideal Surfaces

The picture so far — volcano plots, scaling relations, well-defined crystal facets — applies to idealized single-crystal surfaces in vacuum. Real electrocatalysts operate under conditions that break these simplifications in several ways.

### The Solid-Liquid Interface

Real electrocatalysis happens in liquid, not vacuum. The catalyst surface is immersed in an electrolyte — typically water with dissolved ions. This changes the physics in ways that gas-phase models cannot capture (Shuaibi et al., 2025):

- **Solvent stabilization.** Water molecules form hydrogen bonds with adsorbed intermediates, shifting their binding energies. An intermediate that binds too weakly in vacuum may be stabilized enough by the solvent to become catalytically relevant.
- **The electrical double layer.** At an electrified interface, ions in the electrolyte rearrange to screen the surface charge, forming a structured layer whose properties govern charge transfer kinetics. The structure of this double layer depends on applied potential, ion concentration, and solvent identity.
- **Specific ion adsorption.** Ions from the electrolyte can adsorb directly on the catalyst surface, blocking active sites or modifying the local electronic environment.

The quantity that captures these effects is the **solvation energy**: the difference between a species' adsorption energy in the solvated environment and in vacuum. A catalyst that looks optimal in gas-phase DFT calculations may perform differently when solvent effects shift binding energies by tenths of an eV — comparable to the width of the volcano peak.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_solid_liquid_overview.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The OC25 dataset models catalysis at solid-liquid interfaces: catalyst surfaces with explicit solvent molecules and ions, spanning 88 elements and multiple solvent types. From Shuaibi et al. (2025)." %}

### Compositional Complexity

The design space grows dramatically beyond pure metals and binary alloys.

**Heteroatom doping.** Introducing foreign atoms — nitrogen, sulfur, phosphorus, boron — into a host material modifies the electronic structure of nearby active sites. Nitrogen-doped carbon catalysts are a well-known example. Dopant concentration, spatial distribution, and the specific coordination environment all affect activity. Even for a single host material, this creates a large combinatorial space.

**High-entropy materials.** These contain five or more principal elements in near-equimolar ratios — high-entropy alloys (HEAs) or high-entropy oxides. The vast configurational space produces local electronic environments not achievable in simpler compositions. Properties emerge from collective interactions among elements (the "cocktail effect"), not from individual elemental contributions. This makes decomposition-based reasoning — "element A contributes property X" — unreliable.

### Structural Disorder and Dynamic Surfaces

Many high-performing electrocatalysts lack long-range crystalline order. Amorphous metal oxides and hydroxides used for oxygen evolution have no single repeating unit cell — they present a distribution of local coordination environments, each with potentially different catalytic activity. Standard DFT approaches assume periodic boundary conditions and a well-defined slab; modeling amorphous surfaces requires large supercells or ensemble sampling to capture the structural diversity.

The problem is compounded by **surface reconstruction under operating conditions**. Atoms migrate, oxidation states change, and the electrochemically active phase may differ from the as-synthesized material. Oxide surfaces are particularly prone to this: partial dissolution by the solvent creates vacancy defects, and the surface can restructure in response to applied potential. The structure you model is not necessarily the structure that catalyzes.

Oxides also exhibit **magnetic polymorphism** — the same crystal structure can have different magnetic configurations (ferromagnetic, antiferromagnetic, nonmagnetic), each with different surface energies and adsorption properties (Tran et al., 2023). In semiconducting oxides, a further complication arises from **charge self-compensation**: when vacancies or dopants perturb the electron count, the surface can thermodynamically prefer to reconstruct — breaking or forming bonds — rather than promote electrons into the conduction band. This means a vacancy on one side of a slab can trigger geometric rearrangement on the other side, an effect that is long-ranged and difficult to capture with local models.

### Scaling Relations on Complex Materials

The linear scaling relations that simplify catalyst screening on metals (the $$\Delta G_{*\text{OOH}} \approx \Delta G_{*\text{OH}} + 3.2$$ eV relation from Section 5) were established on close-packed metal surfaces with uniform binding sites. On oxide surfaces with diverse site types, the picture is noisier.

{% include figure.liquid loading="eager" path="assets/img/blog/ec_oxide_scaling.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Scaling relations on oxide surfaces. (A) OOH* vs OH* binding energies. (B) O* vs OH* binding energies. Red points: OC22 dataset (oxides). Blue points: literature values. The linear correlations still exist but with larger scatter (R² ≈ 0.6–0.8) and higher MAE compared to metals. On materials with diverse site types — high-entropy alloys, doped carbons, amorphous oxides — the correlations weaken further. From Tran et al. (2023)." %}

Tran et al. (2023) find that the linear correlations persist on oxides — slopes are within 0.15 eV of literature values for metals — but with substantially more scatter (R$$^2$$ ≈ 0.6 for $$*$$OOH vs. $$*$$OH, compared to >0.9 on metals). The increased scatter means that the one-dimensional volcano picture becomes less predictive: two oxides with similar $$\Delta G_{*\text{OH}}$$ can have meaningfully different $$\Delta G_{*\text{OOH}}$$ values. On even more disordered materials — high-entropy alloys, amorphous oxides — the correlations are expected to weaken further, and screening based on a single descriptor becomes unreliable.

When scaling relations break down, the full multi-dimensional binding-energy space re-emerges. Each intermediate must be evaluated independently, and the tractable one-dimensional search becomes a high-dimensional optimization problem — precisely the setting where learned surrogates have the most to contribute.

---

## Machine Learning for Catalyst Discovery

The previous sections establish the problem: find a catalyst material whose adsorption energy sits at the volcano peak, from a search space of billions of candidates, where each evaluation costs hours of DFT compute. ML enters as a way to make this search tractable. The approaches fall into two generations — surrogate energy models that accelerate evaluation, and generative models that propose candidates directly.

### Surrogate Energy Models

The first generation of ML models for catalysis learns to predict DFT energies and forces from atomic structure, replacing the expensive quantum-mechanical calculation with a fast forward pass. These are **machine learning interatomic potentials (MLIPs)**[^mlip] — they take a configuration of atoms as input and predict the total energy and per-atom forces, enabling structure relaxation at a fraction of the DFT cost.

The key architectural insight is that catalyst systems are inherently three-dimensional and obey physical symmetries: rotating or translating the entire system should not change the predicted energy. This motivates **equivariant graph neural networks** (covered in a [previous post](/blog/2026/spherical-equivariant-layers/)), which build these symmetries into the model architecture rather than learning them from data.

The progression of models on the [OC20 benchmark](https://opencatalystproject.org/) illustrates how architectural advances translate to accuracy gains:

- **SchNet** (Schütt et al., 2018) and **DimeNet++** (Gasteiger et al., 2020) established the invariant GNN baseline — message-passing networks that use interatomic distances (and angles, for DimeNet++) as features.
- **GemNet-OC** (Gasteiger et al., 2022) introduced two-hop message passing with dihedral angle information, achieving strong performance on both energy and force prediction.
- **EquiformerV2** (Liao & Smidt, 2024) combined equivariant Transformers with higher-order spherical harmonics representations, achieving state-of-the-art results on OC20 — energy MAE of 0.21 eV and force MAE of 0.013 eV/Å.

### From Prediction to Screening: AdsorbML

A trained MLIP does not directly solve the catalyst design problem — it accelerates a single energy evaluation. The **AdsorbML** pipeline (Lan et al., 2023) connects the surrogate model to the actual screening workflow:

1. For a given (surface, adsorbate) pair, generate many candidate initial configurations by random placement.
2. Relax each configuration using the ML surrogate instead of DFT.
3. Rank by predicted energy and run DFT only on the top-$$k$$ lowest-energy candidates to validate.

This reduces DFT cost by orders of magnitude while maintaining high success rates — EquiformerV2 within AdsorbML identifies the correct lowest-energy configuration ~84% of the time on OC20 (Lan et al., 2023). The pipeline has been applied to screen thousands of bimetallic alloy surfaces for the hydrogen evolution reaction, with a validated adsorption energy MAE of 0.12 eV across the screened space.

### Generative Catalyst Design

Surrogate models accelerate evaluation but still require someone to propose candidates. The search is enumerate-then-filter: list configurations, predict energies, pick the best. An alternative is to **generate** promising candidates directly — to learn a model that, given a target adsorption energy or reaction intermediate, outputs a catalyst structure likely to achieve it.

This is a harder problem. A catalyst structure is a coupled system: the surface slab (a periodic crystal cut along a specific facet) and the adsorbate (a molecule at a specific position and orientation on that surface). The two are not independent — the adsorbate's preferred binding site depends on the surface geometry, and the surface may reconstruct in response to the adsorbate.

**CatFlow** (Kim et al., 2026) addresses this coupling through a flow matching framework that co-generates the slab and adsorbate jointly. The key idea is a **factorized primitive-cell representation**: rather than generating the full slab (which may contain hundreds of atoms in a repeated pattern), CatFlow generates only the primitive cell and encodes the surface orientation explicitly. This reduces the number of learnable variables by ~9x on average while preserving the interface geometry that determines catalytic activity. Generated structures show accurate adsorption energy distributions and sit closer to DFT-relaxed local minima compared to autoregressive and sequential baselines.

### Open Challenges

Several gaps remain between current ML capabilities and practical catalyst discovery:

- **Out-of-distribution generalization.** Models trained on OC20 (metals) transfer imperfectly to oxides (OC22) or solvated interfaces (OC25). Fine-tuning helps, but the chemical diversity of real catalyst spaces — high-entropy alloys, amorphous oxides, doped carbons — remains undersampled.
- **Long-range interactions.** Local message-passing GNNs struggle with the long-range electrostatics and magnetic interactions that dominate in semiconducting oxides (Section 7). Extending effective interaction ranges without prohibitive cost is an active research direction.
- **Beyond adsorption energy.** Current screening pipelines optimize a single thermodynamic descriptor. Real catalyst selection requires stability under operating conditions (Pourbaix analysis), synthesis feasibility, cost, and selectivity — none of which are captured by adsorption energy alone.

---

## References

- Zitnick, C. L., Chanussot, L., Das, A., Goyal, S., Heras-Domingo, J., Ho, C., ... & Ulissi, Z. W. (2020). An introduction to electrocatalyst design using machine learning for renewable energy storage. *arXiv preprint arXiv:2010.09435*.
- Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M., Riviere, M., ... & Zitnick, C. L. (2021). Open Catalyst 2020 (OC20) dataset and community challenges. *ACS Catalysis*, 11(10), 6059-6072.
- Nørskov, J. K., Rossmeisl, J., Logadottir, A., Lindqvist, L., Kitchin, J. R., Bligaard, T., & Jónsson, H. (2004). Origin of the overpotential for oxygen reduction at a fuel-cell cathode. *J. Phys. Chem. B*, 108(46), 17886-17892.
- Nørskov, J. K., Bligaard, T., Rossmeisl, J., & Christensen, C. H. (2009). Towards the computational design of solid catalysts. *Nature Chemistry*, 1, 37-46.
- Tran, R., Lan, J., Shuaibi, M., Wood, B. M., Goyal, S., Das, A., ... & Zitnick, C. L. (2023). The Open Catalyst 2022 (OC22) dataset and challenges for oxide electrocatalysts. *ACS Catalysis*, 13(5), 3066-3084.
- Shuaibi, M., Choubisa, H., Engel, M., Wood, B. M., Musielewicz, J., Comer, B., ... & Zitnick, C. L. (2025). Open Catalyst 2025 (OC25): dataset and models for the solid-liquid interface. *arXiv preprint arXiv:2509.17862*.
- Gasteiger, J., Becker, F., & Günnemann, S. (2022). GemNet-OC: developing graph neural networks for large and diverse molecular simulation datasets. *Transactions on Machine Learning Research*.
- Liao, Y.-L., & Smidt, T. (2024). EquiformerV2: improved equivariant Transformer for scaling to higher-degree representations. *ICLR 2024*.
- Lan, J., Palizhati, A., Shuaibi, M., Wood, B. M., Wander, B., Das, A., ... & Zitnick, C. L. (2023). AdsorbML: a leap in efficiency for adsorption energy calculations using generalizable machine learning potentials. *npj Computational Materials*, 9, 172.
- Kim, M., Kim, N., Kim, H., & Ahn, S. (2026). CatFlow: co-generation of slab-adsorbate systems via flow matching. *arXiv preprint arXiv:2602.05372*.

---

[^catalyst]: A **catalyst** is a substance that increases the rate of a chemical reaction without being consumed. It works by lowering the energy barrier (activation energy) that reactants must overcome, providing an alternative reaction pathway. The catalyst participates in intermediate steps but is regenerated at the end.

[^adsorption-energy]: The **adsorption energy** (or binding energy) measures how strongly a molecule sticks to a surface. More negative values mean stronger binding. It is computed as the energy difference between the combined system (molecule on surface) and the separated components (clean surface + isolated molecule).

[^intermediate]: A reaction **intermediate** is a molecule that is produced in one step of a multi-step reaction and consumed in a subsequent step. It exists transiently on the catalyst surface. For the oxygen reduction reaction, the intermediates are $$*$$O and $$*$$OH, where $$*$$ denotes binding to the surface.

[^facet]: A **crystal facet** is the flat surface exposed when a crystal is cut along a specific plane. Different cuts expose different arrangements of atoms. Facets are labeled by **Miller indices** — e.g., (111), (110), (100) — which describe the orientation of the cutting plane relative to the crystal lattice.

[^binding-site]: A **binding site** is a specific location on the catalyst surface where an adsorbate molecule can attach. On a close-packed metal surface, common sites include: **atop** (directly above one atom), **bridge** (between two atoms), and **hollow** (in the center of three atoms). Each site type produces a different adsorption energy.

[^adsorbate]: An **adsorbate** is a molecule or atom that has adhered to a surface. In electrocatalysis, adsorbates are the reaction intermediates sitting on the catalyst surface — for example, an oxygen atom ($$*$$O) or a hydroxyl group ($$*$$OH) bonded to a metal surface.

[^dft-relaxation]: A **DFT relaxation** (or geometry optimization) is an iterative simulation that finds the lowest-energy arrangement of atoms. Starting from an initial guess, each step computes quantum-mechanical forces on every atom using density functional theory (see [previous post](/blog/2026/quantum-chemistry-dft/)), then nudges the atoms downhill. This repeats for 50–400 steps until the forces converge to near zero.

[^duck-curve]: The "duck curve" refers to the shape of net electricity demand (demand minus solar generation) over a day: it dips at midday when solar peaks, then rises sharply in the evening. First identified by the California Independent System Operator (CAISO).

[^electrolysis]: **Electrolysis** is the process of using electrical energy to drive a non-spontaneous chemical reaction. In water electrolysis, an applied voltage splits water into hydrogen and oxygen: $$2\text{H}_2\text{O} \rightarrow 2\text{H}_2 + \text{O}_2$$.

[^fuel-cell]: A **fuel cell** is an electrochemical device that converts chemical energy (from a fuel like hydrogen) directly into electrical energy, without combustion. Unlike a battery, it does not store energy internally — it produces electricity continuously as long as fuel is supplied.

[^hes-efficiency]: Zitnick et al. (2020) estimate HES round-trip efficiency at ~35% (AC to AC), compared to 70–80% for pumped hydro and 60–95% for batteries. The losses occur in both the electrolyzer (~70% efficient) and the fuel cell (~60% efficient): 0.7 × 0.6 ≈ 0.42, minus additional transmission and conversion losses.

[^electrocatalyst]: An **electrocatalyst** is a catalyst that operates at an electrode surface in an electrochemical cell. It accelerates reactions involving electron transfer — such as the splitting of water or the reduction of oxygen. The "electro-" prefix distinguishes it from catalysts for purely thermal or gas-phase reactions.

[^anode-cathode]: The **anode** is the electrode where oxidation occurs (loss of electrons); the **cathode** is where reduction occurs (gain of electrons). A mnemonic: **a**node = **a**way (electrons flow away from it).

[^electrolyte]: An **electrolyte** is a substance that conducts ions but not electrons. In a PEM fuel cell, the polymer membrane serves as the electrolyte — it allows protons (H$$^+$$) to pass through while forcing electrons to travel through the external circuit.

[^electrode]: An **electrode** is a conductor through which electric current enters or leaves an electrochemical cell. In a fuel cell, the two electrodes are the anode (fuel side) and cathode (oxygen side).

[^orr]: The **oxygen reduction reaction (ORR)** is the cathode half-reaction in a fuel cell: $$\text{O}_2 + 4\text{H}^+ + 4e^- \rightarrow 2\text{H}_2\text{O}$$. It is kinetically sluggish — the main source of efficiency loss in fuel cells — and the primary motivation for catalyst research.

[^pt-reserves]: As of 2020. South Africa holds approximately 80% of known platinum reserves.

[^heterogeneous]: **Heterogeneous catalysis** means the catalyst and reactants are in different phases — typically a solid catalyst with gas- or liquid-phase reactants. This is in contrast to **homogeneous catalysis**, where catalyst and reactants are in the same phase (e.g., both dissolved in solution).

[^adsorption]: **Adsorption** is the adhesion of atoms or molecules to a surface (not to be confused with *absorption*, which is uptake into the bulk). In **chemisorption**, the adsorbate forms a chemical bond with the surface. In **physisorption**, the interaction is weaker (van der Waals forces only).

[^dissociative]: In a **dissociative pathway**, a molecule breaks apart as it adsorbs — e.g., O$$_2$$ splits into two separate O atoms on the surface. The alternative is an **associative pathway**, where the molecule adsorbs intact and breaks apart in a later step.

[^enthalpy]: **Enthalpy** ($$H$$) is the total heat content of a system at constant pressure: $$H = U + PV$$, where $$U$$ is internal energy, $$P$$ is pressure, and $$V$$ is volume. Changes in enthalpy ($$\Delta H$$) measure the heat released or absorbed by a reaction.

[^gibbs]: The **Gibbs free energy** ($$G = H - TS$$) determines the direction of spontaneous processes at constant temperature and pressure. A reaction with $$\Delta G < 0$$ is thermodynamically favorable (exergonic); $$\Delta G > 0$$ requires energy input (endergonic).

[^activation-energy]: The **activation energy** ($$E_a$$) is the minimum energy that reactants must have to undergo a reaction. Even thermodynamically favorable reactions ($$\Delta G < 0$$) may proceed slowly if the activation energy is high. A catalyst provides an alternative pathway with a lower $$E_a$$.

[^bep]: The **BEP relation** is an empirical observation: across a family of related catalysts, activation energy is linearly related to reaction energy. Proposed independently by Brønsted (1928) and Evans & Polanyi (1938). The slope $$\alpha$$ is typically between 0 and 1.

[^arrhenius]: The **Arrhenius equation** says that reaction rate increases exponentially as the activation energy decreases or temperature increases. $$k_B$$ is Boltzmann's constant. At room temperature ($$k_B T \approx 0.025$$ eV), even small changes in $$E_a$$ (tenths of an eV) cause large changes in rate.

[^oer]: The **oxygen evolution reaction (OER)** is the anode half-reaction in water splitting: $$2\text{H}_2\text{O} \rightarrow \text{O}_2 + 4\text{H}^+ + 4e^-$$. It is the reverse of the ORR and is kinetically even more demanding — forming the O–O bond requires coordinating four electron transfers.

[^termination]: A **surface termination** is the specific atomic layer exposed when a crystal is cut. On a binary oxide like TiO$$_2$$, cutting along the (110) plane can expose a Ti-rich, O-rich, or mixed layer depending on where the cut is made. Each termination has different catalytic properties.

[^hubbard-u]: The **Hubbard U correction** adds an empirical on-site Coulomb repulsion to specific orbitals (usually transition metal d-orbitals) to compensate for the tendency of standard DFT functionals to over-delocalize electrons. Typical U values range from 3–6 eV depending on the metal.

[^mvk]: The **Mars-van Krevelen (MvK) mechanism** is a catalytic cycle in which lattice atoms from the catalyst surface participate directly in the reaction. An adsorbate reacts with a surface oxygen, the product desorbs (creating a vacancy), and the vacancy is replenished by oxygen from a subsequent reactant. Common in oxide and carbide catalysts.

[^mlip]: A **machine learning interatomic potential (MLIP)** is a model that predicts the potential energy surface of an atomic system — total energy as a function of atomic positions — learned from quantum-mechanical (typically DFT) training data. Once trained, it can compute energies and forces orders of magnitude faster than DFT, enabling rapid structure relaxation and molecular dynamics.

<style>
.footnote-tooltip {
  position: absolute;
  background: #263238;
  color: #fff;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.82em;
  line-height: 1.45;
  max-width: 360px;
  z-index: 1000;
  pointer-events: none;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  opacity: 0;
  transition: opacity 0.15s;
}
.footnote-tooltip.visible { opacity: 1; }
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  var tooltip = document.createElement('div');
  tooltip.className = 'footnote-tooltip';
  document.body.appendChild(tooltip);

  document.querySelectorAll('a.footnote').forEach(function(link) {
    var fnId = link.getAttribute('href').substring(1);
    var fnLi = document.getElementById(fnId);
    if (!fnLi) return;

    // Get text content, strip the back-arrow
    var clone = fnLi.cloneNode(true);
    clone.querySelectorAll('.reversefootnote').forEach(function(el) { el.remove(); });
    var text = clone.textContent.trim();
    if (!text) return;

    link.addEventListener('mouseenter', function(e) {
      tooltip.textContent = text;
      tooltip.classList.add('visible');
      var rect = link.getBoundingClientRect();
      var tipW = tooltip.offsetWidth;
      var left = rect.left + rect.width / 2 - tipW / 2 + window.scrollX;
      if (left < 8) left = 8;
      if (left + tipW > window.innerWidth - 8) left = window.innerWidth - tipW - 8;
      tooltip.style.left = left + 'px';
      tooltip.style.top = (rect.top + window.scrollY - tooltip.offsetHeight - 6) + 'px';
    });

    link.addEventListener('mouseleave', function() {
      tooltip.classList.remove('visible');
    });
  });
});
</script>
