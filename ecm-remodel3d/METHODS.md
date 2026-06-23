# Methods — A discrete fiber-network model of cell-driven ECM self-organization

A reproducible computational platform in which simply-placed cells secrete,
contract, and plastically remodel a discrete collagen network, so that complex
divergent reticular architectures (e.g. spleen white-pulp–like webs) **emerge**
from mechanics rather than being prescribed by the seeding pattern.

> **Units.** All quantities are in **reduced (non-dimensional) units**: lengths in
> µm, stiffnesses relative to the reference tensile modulus `k_s = 1`, forces in
> units of `k_s · µm`, and time in discrete *remodeling steps*. The model is **not
> yet calibrated** to a specific experiment; mapping the reduced moduli to Pa and
> the traction to nN and steps to hours is deferred (see §10). This choice is
> deliberate: the present aim is the *qualitative emergence and its control*, not
> quantitative prediction.

---

## 1. Conceptual framework

Real connective-tissue architecture (the reticular/FRC collagen scaffold of
splenic white pulp; tendon straps; the matrix tracks between explants) is not
drawn cell-by-cell. It is the **mechanical self-organization** of a fibrous,
**mechanically nonlinear** extracellular matrix under cell traction:

1. cells **secrete** collagen locally;
2. cells **contract** (integrin-mediated traction), pulling the surrounding
   matrix into aligned, tensed **struts**;
3. because collagen **bears tension but buckles in compression** and
   **strain-stiffens**, cell forces are transmitted over long range and focus
   into divergent tracks (asters) and inter-cell bridges;
4. loaded collagen is **stiffened/crosslinked** and **protected from turnover**,
   while unloaded collagen is **degraded** — so the structure becomes **plastic
   (permanent)**.

The platform encodes (1)–(4) and measures the emergent architecture
quantitatively.

---

## 2. Discrete network representation

The matrix is a graph embedded in ℝ³ (2D studies fix one coordinate; §6):

- **Nodes** `x_i ∈ ℝ³`, `i = 1…N`.
- **Edges** `e = (i,j) ∈ 𝓔`: *segment* edges along each fiber backbone, plus
  *crosslink* edges between nearby nodes of different fibers. Each edge carries a
  **rest length** `ℓ⁰_e`, an initial rest length `ℓ⁰_{e,0}`, and a **plastic
  stiffness gain** `κ_e` (initially `κ_e = 1`).
- **Bending triples** `(a,c,b)`: consecutive node triples along a fiber backbone
  (free hinges at crosslink junctions).
- **Constraints**: a node may be **fixed** (Dirichlet) if near the domain
  boundary or chosen as a sparse gel-adhesion anchor.

Current length and unit direction of an edge:

```
ℓ_e = |x_j − x_i|,   û_e = (x_j − x_i)/ℓ_e,   ε_e = (ℓ_e − ℓ⁰_e)/ℓ⁰_e
```

`ε_e` is the axial strain.

---

## 3. Constitutive model (passive matrix mechanics)

### 3.1 Axial response — tension-bearing, buckling, strain-stiffening

The hallmark of collagen/fibrin networks is a **strongly asymmetric** axial
response: fibers carry tension but **buckle** (almost no stiffness) under
compression, and **stiffen** with tensile strain. We use a smooth tangent
stiffness

```
k_e(ε) = κ_e · k_s · [ r_b + (1 − r_b)·½(1 + tanh(ε / w)) ] · (1 + α·⟨ε⟩₊)     (1)
```

with `⟨ε⟩₊ = max(ε,0)`. Terms:

| symbol | meaning | default |
|---|---|---|
| `k_s` | reference tensile modulus | 1 |
| `r_b` (`buckle_ratio`) | compression/tension stiffness ratio (buckling) | 0.02 |
| `w` (`buckle_width`) | smoothing width of the tension/compression switch | 0.03 |
| `α` (`stiffen_alpha`) | strain-stiffening coefficient | 6 |
| `κ_e` (`k_scale`) | plastic reinforcement gain (§7) | 1 → ≤ 8 |

The `tanh` switch makes `k_e ≈ r_b·k_s` in compression and `k_e ≈ k_s(1+αε)` in
tension, with a smooth transition that removes the stiffness discontinuity at
`ε = 0` (which would otherwise make the relaxation oscillate). The **axial
tension** and the resulting nodal forces are

```
T_e = k_e(ε_e) · (ℓ_e − ℓ⁰_e)                                                   (2)
F_i ←  +T_e û_e ,   F_j ←  −T_e û_e        (tension pulls endpoints together)    (3)
```

This is the negative gradient of a nonlinear axial potential `U_axial = Σ_e ψ(ε_e)`
whose tangent modulus is `k_e`; the solver (§5) is force-based, so (2)–(3) are the
operative definition.

### 3.2 Bending — semiflexible backbone

Fibers resist bending along their backbone (worm-like-chain–like). For each
backbone triple `(a,c,b)` we use a discrete curvature penalty

```
U_bend = ½ κ_b Σ |x_a + x_b − 2x_c|²,                                           (4)
```

giving momentum-conserving restoring forces (Laplacian smoothing)

```
F_c ← +κ_b (x_a + x_b − 2x_c),   F_a ← −½κ_b(…),   F_b ← −½κ_b(…)               (5)
```

with `κ_b = k_bend = 0.12`. Bending rigidity is essential: a purely central-force
(buckling) network is **sub-isostatic** and has floppy zero-energy modes; the
bending term stabilizes them, as in the established theory of biopolymer network
mechanics.

### 3.3 Crosslinks and anchoring

- **Crosslinks**: permanent edges created between any two deposited nodes closer
  than `r_x` (`crosslink_dist ≈ 3.5–4.2 µm`), modeled as axial springs (1)–(3)
  with rest length set to their initial separation. These knit the deposited
  fibers into a connected network.
- **Boundary anchoring**: nodes within `anchor_margin = 8 µm` of a wall are fixed,
  representing attachment of the matrix to the surrounding gel; this gives cell
  traction something to pull against (without it the matrix collapses to points).
- **Gel-adhesion anchors**: a sparse random fraction `adhesion_frac = 0.05` of
  nodes is pinned, representing matrix/cell adhesion to the cured GelMA, so
  contraction condenses the matrix into struts instead of collapsing it.

---

## 4. Active cell traction

Each contractile cell `m` (a spheroid is a cluster of such cells) at position
`c_m` exerts on every free node `i` a force directed **toward** the cell:

```
f_{i,m} = T₀ · exp(−d_{im}/λ) · ramp(d_{im}) · d̂_{im} ,   for d_{im} < R         (6)
d_{im} = |c_m − x_i| ,   d̂_{im} = (c_m − x_i)/d_{im}
ramp(d) = clip( (d − a_c)/a_c , 0, 1 )
```

Parameters: traction magnitude `T₀ = traction` (≈1.1–1.6), decay length
`λ = reach_decay` (≈9–18 µm), interaction radius `R = reach` (≈13–34 µm), cell
radius `a_c = cell_radius = 9 µm`. The exponential models force decay with
distance; the `ramp` **tapers traction to zero inside the cell body**, so nodes
settle on a stable peri-cellular shell rather than collapsing through the cell
(a physically required regularization). The total active force on node `i` is
`F_i^{cell} = Σ_m f_{i,m}`. This is a phenomenological representation of
integrin-mediated contractility; it is the discrete analogue of an active
contractile body force.

---

## 5. Mechanical equilibrium

Because ECM remodeling is slow, inertia is negligible and we seek **quasi-static
mechanical equilibrium**: the configuration `{x_i}` at which the net force on
every free node vanishes,

```
F_i^total = F_i^axial + F_i^bend + F_i^cell = 0 ,   ∀ free i.                    (7)
```

### 5.1 FIRE solver with Jacobi preconditioning

(7) is a stiff, partly-floppy minimization (buckled regions are nearly
unconstrained). We solve it with the **FIRE** algorithm (Fast Inertial
Relaxation Engine; Bitzek et al., 2006) acting on **diagonally-preconditioned**
forces. The per-node Jacobi stiffness `K_i` (sum of incident tangent stiffnesses
from axial, bending, and traction terms) defines the preconditioned acceleration

```
a_i = F_i / K_i .                                                                (8)
```

FIRE integrates an overdamped-inertial dynamics with velocity mixing

```
v ← (1 − γ) v + γ |v| â ,                                                        (9)
```

adaptive time step (`dt → 1.1·dt`, `γ → 0.99·γ` while power `P = F·v > 0`;
`v → 0`, `dt → dt/2`, `γ → γ₀` when `P ≤ 0`), and a per-step displacement cap
`max_step`. Convergence: `max_i |F_i| < f_tol = 1.5×10⁻²`. The preconditioning
(8) is what makes a single step size stable across the orders-of-magnitude
spread of edge stiffnesses (buckled `r_b·k_s` vs stiffened `k_s(1+αε)·κ_e`).

### 5.2 Planar (2D) constraint

For 2D studies we set `planar = True`, which zeros the z-component of
acceleration and velocity each iteration, confining motion to the `z = const`
plane while keeping the identical 3D force law — i.e. a genuine plane-strain-like
restriction, not a separate 2D model.

---

## 6. ECM secretion and growth over time

Cells **secrete** collagen locally (`build_deposited_network`): each cell deposits
`fibers_per_cell` short straight fibers of length `ℓ_f` (subdivided into segments
of length `ℓ_s`), at positions drawn **uniformly in a disk** of radius `ρ_s`
(`secretion_radius`) about the cell — `r = ρ_s √U`, `θ ∼ U[0,2π)` — with random
in-plane orientation. Because only cell-neighborhoods are filled, **cell-free
regions stay empty**, so the pore structure is set by the seeding geometry.

**Gradual growth.** Each fiber is assigned a **secretion time** `b ∈ U[0,1)`; an
edge’s birth is `b_e = max(b of its two endpoints’ fibers)`. The time course
(`grow_and_remodel`) reveals the matrix progressively: at step `s = 1…n_steps`
the **active** sub-network is `{e : b_e ≤ s/n_steps}`. At `t = 0` only the seeded
cells exist (no ECM); each step a new fraction of fibers has been secreted, is
relaxed (§5) and remodeled (§7), with reinforcement/compaction persisting across
steps. This reproduces *cells laying down and maturing ECM over time*, with the
**seeding pattern visible at t = 0**.

---

## 7. Plastic (mechanosensitive) remodeling

After each equilibration we read the per-edge tension `T_e` and apply
tension-driven rules (`remodel` / `grow_and_remodel`), defined by **adaptive
percentile thresholds** of the current tension distribution (robust to the
absolute force scale):

```
τ_hi = P_{p_r}( {T_e : T_e > 0} ) ,     τ_lo = P_{p_s}( {|T_e|} )                (10)
```

with `p_r = reinforce_pct ≈ 58–65`, `p_s = slack_pct ≈ 30–45`.

**(a) Tensional reinforcement & compaction** — for *loaded* edges `T_e ≥ τ_hi`:

```
κ_e ← min( κ_e (1 + r_r), κ_max )                  (stiffening / crosslinking)   (11)
ℓ⁰_e ← max( ℓ⁰_e (1 − c_c), f_min · ℓ⁰_{e,0} )      (tensional compaction)        (12)
```

`r_r = reinforce_rate ≈ 0.3`, `κ_max = 8`, `c_c = compaction ≈ 0.05`,
`f_min = min_rest_frac = 0.55`. (11) models tension-induced collagen recruitment
and lysyl-oxidase–type crosslinking that stiffen loaded fibers; (12) models
active shortening/compaction of the matrix along tension lines.

**(b) Mechanically-protected turnover** — *slack* edges `|T_e| ≤ τ_lo` accumulate
a slack age; an edge is **degraded (removed)** if it stays slack for `≥ s_slack`
steps **and** is not yet reinforced (`κ_e < κ_protect = 1.4`):

```
slack_age_e ← (slack_age_e + 1) if slack else 0
remove e   if   slack_age_e ≥ s_slack  and  κ_e < κ_protect                      (13)
```

This is the **“use it or lose it”** rule: mechanically loaded collagen is
protected from proteolysis, unloaded collagen is turned over — pruning the
network down to its load-bearing struts. Because `κ_e` and `ℓ⁰_e` changes
persist, the remodeled track is **plastic** (it survives cell removal; §9.2).

---

## 8. Quantitative morphometrics

All operate on the (reinforced) network projected to the plane. Let `m_e` be an
edge midpoint, `û_e` its unit direction, `θ_e` its in-plane angle, weights = edge
length `ℓ_e`.

- **Nematic order** (global alignment): `S = | Σ_e ℓ_e e^{2iθ_e} | / Σ_e ℓ_e ∈
  [0,1]` (0 isotropic, 1 perfectly aligned).
- **Radial alignment** about a center `c` (aster character):
  `A_r = ⟨ |û_e · r̂_e| ⟩_ℓ`, `r̂_e = (m_e − c)/|m_e − c|` (0.5 isotropic in 2D, 1
  fully radial).
- **Pore size**: `P(x) = min_e dist(x, segment_e)` sampled by Monte Carlo over the
  domain → pore-radius distribution (median, spread).
- **Branch points**: number of nodes of the **reinforced** sub-network with degree
  `≥ 3` (divergent junctions).
- **Coverage**: fraction of domain points within a radius (≈12 µm) of a reinforced
  strut — how far the strut web *spans*.
- **Connectivity**: largest-connected-component fraction of the reinforced
  sub-network (union–find).
- **Radial density profile**: reinforced fiber length per annulus area vs distance
  from a center (compartmental organization).

These turn the images into numbers that can be tracked over time or compared
across conditions.

---

## 9. Emergence, scoring, and validation

### 9.1 Emergent structure and its control (spheroid number / spacing)

With a **uniform background matrix** and **simply-placed spheroids**, contraction
+ protected turnover yield an **emergent** divergent strut-and-pore reticular web
(radial asters around each spheroid, bridges between them) — *not* prescribed by
the seeding. Layouts are auto-scored for **spleen-likeness** by a normalized
composite of the reticular readouts:

```
Φ = 0.34·connectivity + 0.33·coveragẽ + 0.33·branching̃                          (14)
```

(`~` = min–max normalization across the compared layouts). **Findings:** fewer /
wider-spaced spheroids produce the most connected spanning web (wide-spacing
connectivity 0.83 vs tight 0.52; 2-spheroid composite 0.78 highest in a fixed
domain), because too many or too-close spheroids fragment the matrix into local
clumps.

### 9.2 Internal validation (regression tests)

The model ships executable checks (`verify_mechanics.py`, `verify_remodel.py`):

1. **Equilibrium** — FIRE reaches force balance (`max|F| < f_tol`).
2. **Single-cell radial tension** — around one contractile cell, tension-weighted
   fiber radiality (0.98) ≫ unweighted (0.68): tension is carried by the fibers
   pointing at the cell (a radial aster), as observed experimentally.
3. **Nonlinearity is necessary** — a contractile pair aligns a collagen track only
   with the buckling/stiffening law (band alignment 0.50) and **not** with a
   linear control matrix (0.40 ≈ isotropic 0.44): force focusing requires the
   nonlinear constitutive response.
4. **Plastic memory** — after the cells are removed, the remodeled matrix retains
   **residual self-stress** ≈ 4000–10000 (reduced units) vs ≈ 10–40 for a single
   elastic equilibrium — a 100–400× difference, robust across random seeds: the
   track is a permanent, pre-stressed structure.

### 9.3 Face validity — reproduced phenomena

Without fitting, the model reproduces well-documented results: contractile cells
generate **radial collagen alignment (asters)** and **aligned matrix bridges
between two cell foci** (the classic explant traction-structuring of Stopak &
Harris); **long-range force transmission and track formation require fiber
nonlinearity** (tension-stiffening + microbuckling); **cell density controls
matrix compaction / pore size**; and **mechanically loaded collagen is reinforced
and protected while unloaded collagen is turned over** (tensional homeostasis).

---

## 10. Assumptions and limitations

1. **Reduced units / not calibrated.** Moduli, traction, and time are in reduced
   units; mapping to Pa, nN, and hours (e.g. via collagen shear modulus, measured
   cell tractions, and turnover rates) is future work.
2. **Quasi-static, athermal.** No inertia, no viscous/poroelastic fluid coupling,
   no thermal fluctuations; each frame is a mechanical equilibrium.
3. **Phenomenological traction & remodeling.** Cell contractility is an imposed
   force field, not a feedback model of stress-fibers/focal adhesions;
   reinforcement/turnover use adaptive percentile rules, not explicit LOX/MMP
   kinetics.
4. **Fiber interactions simplified.** Crosslinks are permanent springs; there is
   no explicit fiber–fiber steric/friction contact or excitable severing/branching
   beyond deposition + crosslinking.
5. **Mostly 2D.** The reported studies use the planar constraint; the force law is
   3D and the framework extends to 3D at higher cost.
6. **Single matrix species.** One generic collagen-like fiber type; no distinction
   of collagen I/III, fibronectin, basement membrane, etc.
7. **No active cell migration in the emergence studies** (cells are fixed foci);
   contact-guided migration exists in other modules but is off here.

None of these affect the qualitative claims (emergence, its dependence on
nonlinearity, and its control by spheroid number/spacing), which are the scope of
this work.

---

## 11. Numerical reproducibility

Every study is **deterministic** given its RNG seed (network deposition, anchor
selection, Monte-Carlo metrics all seeded). Default parameters are listed in
§3–§7 and in the parameter table below; the exact values per study are in the
`run_*.py` scripts. Convergence is enforced by `f_tol`; the FIRE constants are the
standard Bitzek et al. values.

### Parameter summary (defaults)

| Group | Symbol (code) | Value | Meaning |
|---|---|---|---|
| Axial | `k_s` (k_stretch) | 1 | reference tensile modulus |
| | `r_b` (buckle_ratio) | 0.02 | compression/tension stiffness ratio |
| | `w` (buckle_width) | 0.03 | tension/compression switch width |
| | `α` (stiffen_alpha) | 6 | strain-stiffening |
| Bending | `κ_b` (k_bend) | 0.12 | backbone bending stiffness |
| Traction | `T₀` (traction) | 1.1–1.6 | contractile force magnitude / node |
| | `R` (reach) | 13–34 µm | traction interaction radius |
| | `λ` (reach_decay) | 9–18 µm | traction decay length |
| | `a_c` (cell_radius) | 9 µm | cell body (traction taper) |
| Anchoring | `anchor_margin` | 8 µm | boundary fixed band |
| | `adhesion_frac` | 0.05 | sparse gel anchors |
| Deposition | `ρ_s` (secretion_radius) | 8–32 µm | secretion disk radius |
| | `ℓ_f` (fiber_len) | 18 µm | deposited fiber length |
| | `ℓ_s` (seg_len) | 6 µm | segment length |
| | `r_x` (crosslink_dist) | 3.5–4.2 µm | crosslink distance |
| Solver | `f_tol` | 1.5×10⁻² | force-balance tolerance |
| | `max_step` | 0.6 µm | per-iteration displacement cap |
| | `max_iter` | 1100–4000 | iteration cap |
| Remodeling | `p_r` (reinforce_pct) | 58–65 | reinforce tension percentile |
| | `r_r` (reinforce_rate) | 0.3 | stiffening rate / step |
| | `κ_max` (max_k_scale) | 8 | reinforcement cap |
| | `c_c` (compaction) | 0.04–0.06 | rest-length compaction / step |
| | `f_min` (min_rest_frac) | 0.55 | compaction floor |
| | `p_s` (slack_pct) | 30–45 | slack tension percentile |
| | `s_slack` (slack_steps) | 2–3 | slack steps before turnover |
| | `κ_protect` (protect_k) | 1.4 | turnover-protection threshold |

---

## 12. Representative literature

*(Standard, real references grounding each modeling choice; verify formatting/page
numbers before submission.)*

**Solver.**
Bitzek, Koskinen, Gähler, Moseler, Gumbsch. *Structural Relaxation Made Simple.*
Phys. Rev. Lett. **97**, 170201 (2006). (FIRE)

**Nonlinear fibrous-matrix mechanics & long-range force transmission.**
Storm, Pastore, MacKintosh, Lubensky, Janmey. *Nonlinear elasticity in biological
gels.* Nature **435**, 191 (2005).
Head, Levine, MacKintosh. *Distinct regimes of elastic response… (Mikado
networks).* Phys. Rev. E **68**, 061907 (2003).
Wang, Abhilash, Chen, Wells, Shenoy. *Long-range force transmission in fibrous
matrices…* Biophys. J. **107**, 2592 (2014).
Notbohm, Lesman, Rosakis, Tirrell, Ravichandran. *Microbuckling of fibrin…* J. R.
Soc. Interface **12**, 20150320 (2015).
Han et al. *Cell contraction induces long-ranged stress stiffening in the
extracellular matrix.* PNAS **115**, 4075 (2018).

**Cell traction structuring of matrix.**
Stopak, Harris. *Connective tissue morphogenesis by fibroblast traction.* Dev.
Biol. **90**, 383 (1982).

**Tensional homeostasis / mechanosensitive remodeling.**
Humphrey, Dufresne, Schwartz. *Mechanotransduction and extracellular matrix
homeostasis.* Nat. Rev. Mol. Cell Biol. **15**, 802 (2014).

**Splenic / lymphoid reticular (FRC) networks.**
Novkovic et al. *Topological small-world organization of the fibroblastic
reticular cell network…* PLoS Biol. **14**, e1002515 (2016).
Link et al. *Fibroblastic reticular cells in lymph nodes regulate the homeostasis
of naive T cells.* Nat. Immunol. **8**, 1255 (2007).
