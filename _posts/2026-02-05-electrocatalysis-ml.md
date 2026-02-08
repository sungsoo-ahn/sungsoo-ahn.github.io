---
layout: post
title: "Electrocatalysis for ML Researchers"
date: 2026-02-05
last_updated: 2026-02-08
description: "An introduction to heterogeneous electrocatalysis — the energy storage problem, catalyst design, and why it's a compelling search problem for machine learning."
order: 1
categories: [science]
tags: [electrocatalysis, renewable-energy, machine-learning, density-functional-theory]
toc:
  sidebar: left
related_posts: false
published: false
---

<p style="color: #666; font-size: 0.9em; margin-bottom: 1.5em;">
<em>Note: This post introduces electrocatalysis from the perspective of an ML researcher. The primary reference is Zitnick et al. (2020), a tutorial paper from the Open Catalyst team that motivates the OC20 dataset and benchmark. I wrote this as a bridge between my previous posts on <a href="/blog/2026/quantum-chemistry-dft/">DFT</a> and <a href="/blog/2026/spherical-equivariant-layers/">equivariant GNNs</a>, and the application domain where these methods have the most impact. Figures are from Zitnick et al. (2020) unless otherwise noted. Corrections are welcome.</em>
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

## References

- Zitnick, C. L., Chanussot, L., Das, A., Goyal, S., Heras-Domingo, J., Ho, C., ... & Ulissi, Z. W. (2020). An introduction to electrocatalyst design using machine learning for renewable energy storage. *arXiv preprint arXiv:2010.09435*.
- Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M., Riviere, M., ... & Zitnick, C. L. (2021). Open Catalyst 2020 (OC20) dataset and community challenges. *ACS Catalysis*, 11(10), 6059-6072.
- Nørskov, J. K., Rossmeisl, J., Logadottir, A., Lindqvist, L., Kitchin, J. R., Bligaard, T., & Jónsson, H. (2004). Origin of the overpotential for oxygen reduction at a fuel-cell cathode. *J. Phys. Chem. B*, 108(46), 17886-17892.
- Nørskov, J. K., Bligaard, T., Rossmeisl, J., & Christensen, C. H. (2009). Towards the computational design of solid catalysts. *Nature Chemistry*, 1, 37-46.

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
