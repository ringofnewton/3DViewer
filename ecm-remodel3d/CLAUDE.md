# ECM simulation — project memory

## Canonical design decision (user preference — keep)

**The INTEGRATED, single-network simulation is the best and most natural ECM
design. Always build on it; do not split the model into separate, disconnected
demos.**

The reference implementation is the *one coupled simulation* where every
mechanism runs on the **same evolving network, together**:

1. **Continuous secretion (NOT one-shot)** — cells keep synthesizing collagen
   every step at a kinetic rate (tapering toward a soft cap); deposition follows
   the cells as they migrate. Do NOT secrete everything up front and then stop —
   that was a modeling bug the user flagged.
2. **Mechanics** — cells apply traction; the network relaxes (force balance / PBD).
3. **Mechanochemical turnover** — tension-saturating (Hill) synthesis on loaded
   fibers + strain-PROTECTED MMP degradation (slack fibers lose mass and are
   cleaved). Runs continuously → the matrix is a DYNAMIC STEADY STATE, constantly
   renewed; stop synthesis and the matrix decays. Mirrors `ecm_mechanochem.py`.
4. **Migration** — cells sense stiffness (durotaxis) and signal each other
   (chemotaxis) to move.
5. **Aggregation** — cells cluster along the reinforced inter-cell bridges they
   build (cell–cell separation keeps them a cluster, not a collapsed point).

Realism rule: ECM kinetics must be CONTINUOUS (ongoing synthesis ⇄ degradation,
dynamic equilibrium), never a single burst followed by a frozen network.

Implementations of this canonical design:
- `ecm_simulation.html` — the live, interactive browser version (the one the user
  approved as "가장 좋고 자연스럽다").
- `coupled_sim.py` — the Python 3-D version (migration + mechanosensing +
  mechanochemistry; control = durotaxis only disperses, treatment = +chemotaxis
  aggregates).
- `build_integrated_gif.py` → `output_validation/figures/ecm_integrated.gif` — the
  mobile-viewable render of the integrated model.

**Do NOT** go back to separate, independent toy demos for the *simulation* itself.
Rheology / KS-EMD / MCMC are downstream **measurement & inference** tools applied
to this model's output — those may live separately (`research_interactive.html`,
`rheology_periodic.py`, `distribution_compare.py`, `mcmc_infer.py`), but the
morphogenesis simulation must stay unified.

## Working notes
- Develop on branch `claude/html-review-software-options-8ls41y`.
- Keep everything PRIVATE — do not deploy / publish (GitHub Pages auto-deploy is
  disabled in `.github/workflows/pages.yml`). HTML is the user's own study/review
  material.
- Mobile can't open HTML attachments → deliver dynamic results as **GIFs** (they
  play inline). Desktop → interactive HTML.
- Prefer **light mode** for HTML the user reviews.
- numpy-only (no scipy) in the Python pipeline.
