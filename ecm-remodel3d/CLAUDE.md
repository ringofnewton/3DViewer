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
3. **Mechanochemical turnover (with a structural baseline)** — tension-saturating
   (Hill) synthesis on loaded fibers + strain-protected MMP degradation that
   relaxes a fiber toward a **baseline k_base (~0.3), NOT to zero**. The network
   the cells build PERSISTS and stays visible; loaded fibers mature into bright
   bridges; only a few unloaded fibers are slowly turned over. Do NOT degrade
   unloaded fibers to nothing — that made the whole network vanish (user flagged
   "다 분해된다"). Turnover is a slow modulation on a persistent structure, not
   dissolution. Mirrors `ecm_mechanochem.py`.
4. **Migration** — cells sense stiffness (durotaxis) and signal each other
   (chemotaxis) to move.
5. **Aggregation** — cells cluster along the reinforced inter-cell bridges they
   build (cell–cell separation keeps them a cluster, not a collapsed point).

Realism rule: ECM kinetics must be CONTINUOUS (ongoing synthesis ⇄ degradation,
dynamic equilibrium), never a single burst followed by a frozen network.

### NOT a substrate-attached model
Bulk matrix model (cells embedded in / contracting a fiber network), NOT cells
crawling on a dish bottom (2D substrate culture). Keep it that way.

### Pattern mode = GelMA photopatterning (the real experiment & chosen mechanism)

Fabrication idea: cell suspension in GelMA precursor, then PHOTO-CURE only chosen
foci → rigid anchor points (거점); the desired structure emerges between them.
**User's confirmed mechanism: cells DUROTAXIS to the rigid photo-cured anchors
(long-range rigidity sensing) and bridge adjacent anchors into reinforced struts.**
So you DESIGN the topology by WHERE YOU PHOTO-CURE.

Pattern mode in `ecm_simulation.html` (triangle/square/pentagon/hub+spokes):
- uniform GelMA background anchored ONLY at the cured foci (box edges left free),
- cured foci = pinned + stiff (k=3) rigid anchors, excluded from turnover,
- a uniform CELL SUSPENSION (NOT seeded at foci),
- cells durotaxis up a stiffness field that includes a LONG-RANGE anchor term
  (~200/(1+r²/λ²)) so they home to distant rigid anchors (validated: mean
  cell→nearest-anchor 80→28 px), then traction + compaction build struts between
  adjacent anchors.
`build_durotaxis_gif.py` → `ecm_durotaxis.gif` shows it. The "cells seeded AT foci"
variant (`build_patterns_fig.py` / `run_inverse.py`) is a DIFFERENT fabrication
(cell patterning, not photopatterning) — keep both, but the photopatterning +
durotaxis version is the one matching the real experiment.

Honest caveats: uniform cells WITHOUT anchor-homing just condense the gel
everywhere (no pattern) — the anchor-directed durotaxis is essential. Strut-vs-
target selectivity is moderate (radial bursts at anchors + bridges), matching the
reference image's character. Keep the current design/visuals; develop this further.

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
