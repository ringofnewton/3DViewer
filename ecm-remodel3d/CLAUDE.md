# ECM simulation — project memory

## Canonical purpose (user-confirmed, after a reset)

**The purpose is ECM NETWORK DESIGN via GelMA photopatterning** — design the ECM
structure you want by choosing where to photo-cure the gel. The realistic,
canonical interactive tool is **`ecm_simulation.html`**:

- GelMA = a uniform soft fiber network with embedded cells.
- **Photo-cure a design** (line / ring / Y-branch / grid / star / spiral / ✎draw-
  your-own) → the fibers along that design crosslink into a dense, stiff ECM
  network (the designed structure). The rest stays soft background.
- **Culture (▶)** = cells locally deposit ECM and mature the cured structure
  (contact guidance / local reinforcement only). The design thickens over time.
- Keep the ECM-network rendering (fibers coloured by reinforcement, light→dark).

### DEPRECATED as unrealistic (do NOT re-add) — user flagged "엉망"
The user found these UNREALISTIC; they were removed and must not come back into
the main simulation:
- **chemotaxis / cell–cell aggregation** ("자석" effect — cells rushing together),
- **long-range cell migration / durotaxis homing to distant anchors** (gave blobs),
- coupled "morphogenesis" where cells migrate+aggregate to self-organize topology.
Cells in the canonical tool only do LOCAL contact-guided ECM deposition. No
chemotaxis, no long-range homing, no aggregation.

Removed artifacts (churn from the unrealistic phase): build_aggregation_gif.py,
build_durotaxis_gif.py, build_integrated_gif.py, build_patterns_fig.py,
build_cure_compare.py and their figures. (coupled_sim.py / ecm_cells.py chemotaxis
remain only as old research scripts, NOT the main sim.)

### Kept (realistic) supporting pieces
- `build_arbitrary_fig.py` → ecm_arbitrary.png: arbitrary shapes ARE achievable by
  curing the outline as a continuous guide (verified: star 81% / spiral 100% /
  wave 97% on-target).
- `build_formation_gif.py` → ecm_formation.gif: cells secreting an ECM network.
- Research validation (rheology / KS-EMD / MCMC) lives separately and is unaffected.

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

### Arbitrary designed shapes — what actually works (corrected after a mistake)

I twice overclaimed that sparse point-anchors trace arbitrary shapes — they do
NOT. Two empirically-verified facts:
- **Sparse point anchors + bulk contraction/proximity reinforcement → a tangled
  proximity-graph BLOB** that does not follow non-convex outlines (chords cut
  across). This was wrong to present as shape-following.
- **What DOES trace an arbitrary outline** (`build_arbitrary_fig.py`, verified):
  photo-cure the outline as a CONTINUOUS guide; cells home to it (durotaxis) and
  DEPOSIT ECM LOCALLY along it, with **NO bulk contraction** (bulk contraction is
  what caused the blob). Result: star 81% / spiral 100% / wave 97% of reinforced
  ECM on the designed outline. This is contact-guidance / durotaxis on a patterned
  scaffold — a real mechanism.

Rule of thumb: arbitrary shapes come from the CURED GUIDE (continuous photopattern
of the outline), with cells following & reinforcing it — NOT from sparse-anchor
self-organization. Do not claim a figure "follows the shape" without checking the
on-target fraction and looking at the image.

The interactive `ecm_simulation.html` pattern mode now implements THIS guide-
tracing mechanism (superseding the old discrete-anchor version): pick an outline
(triangle/square/pentagon/hub/star/spiral/wave) or **✎ draw your own** on the
canvas; the outline is cured as a continuous guide, cells durotaxis to it and
deposit ECM along it (no bulk contraction). Verified: valid JS + runs clean under
a DOM mock for presets and draw mode.

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
