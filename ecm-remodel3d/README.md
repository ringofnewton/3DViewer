# ecm-remodel3d — ECM remodeling analysis pipeline

A **publication-oriented** pipeline that turns a time series of cell + ECM-fiber
geometry into **quantitative structural metrics**, **publication figures**, and
**ParaView / web-viewer 3D output**.

This is the "산출물 ①" (quantitative analysis) layer from the *ECM Remodeling
Simulation Stack* decision map — the part that fills a paper's Methods and
Results. It ships with a synthetic generator so the whole thing runs end-to-end
**with no simulator and no GPU**, then is designed so you drop in real
PhysiCell + PhysiMeSS output and change nothing downstream.

```
PhysiCell + PhysiMeSS  ─┐
   (or synthetic.py)    ├─►  cells_t*.csv / fibers_t*.csv  ─►  metrics  ─►  figures (paper)
                        │                                   └─►  VTP      (ParaView)
                        │                                   └─►  JSON     (web viewer)
```

## Interactive 3D viewer (web, no Blender)

A self-contained three.js viewer renders the result in the browser with the look
you'd get from an offline renderer — **screen-space ambient occlusion** for depth,
**translucent subsurface cells** with **organic noise-displaced (bumpy) surfaces**,
environment-lit reflections, and ACES tone mapping. Fibers are instanced tubes
colored by state/tension; orbit to rotate, scroll to zoom.

```bash
python make_track_scene.py         # cells' collagen tracks (plastic remodel)  ← recommended
python make_viewer_scene.py        # OR a grown network with many cells
cd viewer && python serve.py       # serves + opens the browser automatically
```

> Do **not** double-click `index.html`. It uses ES modules + fetches `scene.json`,
> which browsers block on `file://` (you'd get a blank page). `serve.py` serves it
> over http — the page shows an on-screen hint if you open it the wrong way.

`viewer/index.html` loads `viewer/scene.json` (schema in `ecm_pipeline/web_export.py`),
so you can point it at any result — grown network, mechanics track, or remodeled
matrix — by exporting that scene to `viewer/scene.json`. Toggles for cells/fibers,
cell opacity, and AO are in the panel.

## Two ECM models

| Model | `run_demo.py` (rods) | `run_growth.py` (network) |
|---|---|---|
| Fiber | independent straight segment (PhysiMeSS-style) | curved, branching, **crosslinked filament** |
| Topology | scattered "buzz-cut" sticks | connected **web / mesh** |
| Largest connected component | ~0.02 | **~0.64** |
| Use | fast cell–fiber mechanics | realistic ECM **architecture & growth** |

Real collagen ECM is a *percolating network*, not loose rods. `run_growth.py`
models the actual assembly process — nucleation → tip elongation (with
persistence, so fibers curve) → branching → crosslinking → traction bundling →
degradation — and writes a growth montage so you can watch the matrix form. See
`ecm_pipeline/ecm_network.py`.

## Mechanics: cell arrangement → collagen patterning (morphogenesis)

The predictive core. Cells don't paint collagen — they *pull* on it, and the
matrix deforms to mechanical equilibrium. `run_mechanics.py` places contractile
cells in a passive collagen network (a quasi-2D slab, the canonical explant/gel
geometry) and relaxes the network under cell traction with a measured-collagen
constitutive law: axial springs that **bear tension but buckle in compression**
(strongly asymmetric), **strain-stiffening**, and **filament bending**. The
solver is FIRE (robust force-only minimization to true force balance).

The aligned, tensed collagen "track" between neighboring cells then **emerges
from the physics** — reproducing the classic Stopak & Harris traction result.
Three studies are produced in `output_mech/figures/`:

- **A** two cells form a tensed aligned track — and the buckling nonlinearity is
  required: a linear control builds no track (band alignment ≈ 0.51 vs 0.42,
  isotropic 0.33).
- **B** cell spacing tunes the track (alignment + how far force is transmitted).
- **C** the same three cells in a line vs a triangle give different collagen
  scaffolds — different morphogenetic templates.

```bash
python run_mechanics.py     # the morphogenesis studies (figures in output_mech/)
python verify_mechanics.py  # asserts the physics: convergence, radial single-cell
                            # tension, and nonlinear-only track alignment
```

### Plastic remodeling: tracks that outlast the cells

A single equilibrium is elastic — remove the cells and it springs back. Real
morphogenesis is *plastic*: `run_remodel.py` alternates equilibrium with
tension-driven remodeling (reinforce + compact loaded fibers, degrade
persistently slack ones, "use it or lose it"). Over ~10 steps the dense matrix
is pruned to its load-bearing collagen track and that track is reinforced into a
permanent, pre-stressed structure.

We prove the plasticity by removing the cells and measuring **residual
self-stress** — the tension the matrix holds with no cells pulling. The compacted
track stays pre-stressed (plastic ≈ 4000–10000) while an elastic control relaxes
to nothing (≈ 10–40), a 100–400× difference that is robust across seeds.

```bash
python run_remodel.py       # remodeling montage, build-up curves, permanence test
python verify_remodel.py    # asserts pruning, reinforcement, and residual self-stress
```

### 2D morphology over time (cell arrangement → collagen shape)

`run_morphology2d.py` is the simplest way to *see* the idea: a flat collagen
sheet with contractile cells, evolved plastically along a time axis. It plots a
grid (rows = cell arrangements: pair / line / triangle / ring, columns = time)
so you watch the ECM mature from a faint isotropic mesh into reinforced collagen
tracks — and the tracks' shape depends on where the cells are. A line of cells
builds the most aligned (anisotropic) matrix; a ring builds the most isotropic.

```bash
python run_morphology2d.py  # output_morph2d/: morphology_grid.png + morphology_metrics.png
```

### Cell density / DLP pattern → ECM pore morphology

`run_culture.py` reproduces the open-coarse vs dense-fine network contrast (the
naive vs fibrotic spleen morphology) from **cell density**, and shows
**DLP-printed seeding patterns** templating the ECM. Mechanism: cells secrete
ECM locally and then contract it into struts around cell-free pores, so cell
spacing sets the pore size. Matches the GelMA + DLP workflow — cells are seeded
only where the printer cures the gel (`ecm_pipeline/seeding.py`).

```bash
python run_culture.py   # output_culture/: density_morphology, density_poresize, dlp_patterns
```

Result: median pore ~19 µm (low density, open network) → ~2 µm (high density,
dense network); ring / stripe / block print masks give ring / stripe / block ECM.

### Spheroid time-course + quantitative analysis

`run_spheroid.py` drops a contractile spheroid (a cluster of cells) into a simple
isotropic collagen gel and follows the remodeling along a time axis: the matrix
condenses into radial collagen tracks (an aster), which mature into permanent
struts. Two spheroids build an aligned collagen bridge between them. The emphasis
is quantitative — `ecm_pipeline/morpho_metrics.py` measures, per time step:

- radial alignment (0.64 isotropic → radial), reinforced-fiber fraction,
- total reinforced collagen, median pore radius, strut-network connectivity,
- reinforced-ECM density vs distance from the spheroid (rim compaction).

```bash
python run_spheroid.py   # output_spheroid/: timecourse, metrics, two-spheroid bridge
```

### Spleen white-pulp mimic vs control

`run_whitepulp.py` answers "how should cells be arranged to reproduce the spleen
white-pulp reticular skeleton?" — a bounded compartment with a central arteriole,
B-cell follicles, and a denser marginal-zone ring, all knit by a uniform FRC
meshwork. It compares that to the same number of randomly seeded cells (the
control) and quantifies the difference (radial density profile, pore size,
connectivity). Rendered light-mode with a continuous blue→red reinforcement
colour scale at constant line width.

```bash
python run_whitepulp.py   # output_whitepulp/: whitepulp_vs_control, whitepulp_metrics
```

Rendering note: `ecm_pipeline/render.py` supports both `theme="light"` and
`"dark"`, and encodes plastic reinforcement (collagen bundling into struts) as a
continuous colour — not as discrete line thickness.

### Grow ECM into any seeding pattern, over time

`run_patterns.py` seeds cells along an arbitrary geometric pattern; they deposit
collagen and condense it, so over time the ECM matures into that pattern (faint
diffuse matrix → sharp reinforced struts along the design):

1. **regular grid** of aligned seeds → a lattice ECM,
2. **geometric shapes** — circle / triangle / square outlines → shaped ECM,
3. **complex motif** (honeycomb) → patterned ECM.

Any pattern is just a set of strokes: `seeding.along_segments`,
`circle_outline`, `polygon_outline`, `grid_lattice`, `hex_lattice_segments`.
Give me the coordinates (or an image) of a specific design — e.g. a lab logo —
and it seeds that exactly.

```bash
python run_patterns.py   # output_patterns/: patterns_grid, patterns_shapes, patterns_complex
```

Why the nonlinearity matters for *prediction*: collagen transmits cell force over
long range only because compressed fibers buckle and tense fibers stiffen,
focusing load into tracks. A linear material diffuses force and predicts no
tracks — so getting this constitutive law right is what makes the simulation
agree with experiment. See `ecm_pipeline/ecm_mechanics.py`.

## Quick start

```bash
pip install -r requirements.txt
python run_mechanics.py # cell arrangement → collagen tracks (morphogenesis)
python run_growth.py    # grow a connected ECM network + growth montage
python run_demo.py      # fast scattered-rod version
```

Outputs land in `output/` and `output/figures/`:

| File | What it is |
|---|---|
| `metrics_timeseries.csv` | every structural metric per frame — the table behind the figures |
| `figures/fig1_time_series.png` | volume fraction, alignment, contact, connectivity vs time |
| `figures/fig2_orientation_rose.png` | fiber orientation distribution (polar) |
| `figures/fig3_pore_size.png` | pore-size distribution |
| `figures/fig4_remodeling.png` | fiber birth/death events (turnover) |
| `figures/fig5_persistence.png` | per-cell migration persistence |
| `figures/fig6_scene3d.png` | 3D preview of cells + fibers |
| `fibers_final.vtp`, `cells_final.vtp` | open in **ParaView** (Tube filter on fibers, Glyph/sphere on cells) |
| `scene_final.json` | feed to a **three.js / vtk.js** web viewer |

## Using real PhysiCell + PhysiMeSS output

A parser is included. Point it at a PhysiCell output folder and it produces the
same metrics, figures, and 3D export as the demo:

```bash
python parse_physicell.py /path/to/PhysiCell/output  output_real
```

PhysiCell writes one MultiCellDS `outputNNNNNNNN.xml` per frame naming a
`_cells.mat` matrix plus a `<labels>` map. PhysiMeSS stores ECM fibers as
agents in that **same** matrix, separated by `cell_type`, with the fiber axis in
the standard `orientation` field and a length in custom data — so a fiber's
endpoints are `position ± 0.5·length·orientation`. The parser
(`ecm_pipeline/physicell_parser.py`) reads all of that into the CSV schema below.

If your PhysiMeSS build uses different custom-data label names, adjust
`ParseConfig` (`fiber_type_id`, `length_field`, `phenotype_field`, …) — that is
the only place names live.

**Verify the real-data path without a PhysiCell install:**

```bash
python verify_parser.py      # synthetic → MultiCellDS fixture → parser → CSV, asserted
```

This writes genuine MultiCellDS xml + `.mat` fixtures (`ecm_pipeline/fixture.py`),
parses them back, and checks cell/fiber counts, fiber lengths (matched to ~1e-14 µm),
and phenotype/state labels survive the round trip.

### The data contract

Every analysis function reads only these plain CSVs, so the parser is the *only*
format-specific piece; swap in any simulator that emits them:

```
cells_t{frame:03d}.csv : id, x, y, z, radius, phenotype, vx, vy, vz
fibers_t{frame:03d}.csv: id, x1, y1, z1, x2, y2, z2, radius, state, birth
tracks.csv             : cell_id, frame, x, y, z
events.csv             : frame, step, births, deaths
```

IDs are kept stable across frames (cells followed for tracks; fiber ID set
changes give births/deaths), and phenotype / fiber-state / orientation /
velocity are all preserved — exactly the per-agent data a cinematic Blender
render needs later for color, material, and motion.

## Metrics implemented (`ecm_pipeline/metrics.py`)

These mirror the decision map's *정량분석 metric* table:

1. **ECM volume fraction** — Σ(π r² L) / domain³. (Collagen-thin fibers give
   values ~1e-5; the *trend* matters, not the absolute.)
2. **Fiber anisotropy / alignment index** — length-weighted orientation tensor;
   `(3·λ₁ − 1)/2` ∈ [0 isotropic, 1 perfectly aligned].
3. **Local alignment index** — same tensor computed per-cell neighborhood.
4. **Pore-size distribution** — Monte-Carlo nearest-fiber distance sampling.
5. **Cell–ECM contact frequency** — point-to-segment distance < cell+fiber radius.
6. **Migration persistence** — net displacement / path length, per cell track.
7. **Remodeling rate** — fiber birth/death events per snapshot.
8. **Network connectivity** — union-find on fiber endpoints within a crosslink
   distance; reports component count and largest-component fraction.

## What the synthetic model demonstrates

`ecm_pipeline/synthetic.py` is a small agent model reproducing the four MVP
behaviors from the decision map, so the metrics show real emergent trends:

- **MVP1** producing cells nucleate fibers (n_fibers 120→359 over the run)
- **MVP2** degrading cells / MMP delete nearby fibers (death events)
- **MVP3** contractile cells rotate nearby fibers toward a traction axis
  (global alignment index rises ~0.09 → 0.23)
- **MVP4** cells migrate biased along local fiber orientation; tracks recorded

It is a **stand-in for testing the analysis**, not a validated biological model.
Replace it with PhysiCell + PhysiMeSS for real science.

## Module layout

```
ecm-remodel3d/
  run_demo.py              # end-to-end orchestrator (synthetic data)
  parse_physicell.py       # same pipeline driven by REAL PhysiCell/PhysiMeSS output
  verify_parser.py         # round-trip test of the real-data path
  requirements.txt
  run_growth.py            # grow ECM as a connected curved/branching network
  run_mechanics.py         # cell arrangement → collagen tracks (morphogenesis)
  run_remodel.py           # plastic remodeling: tracks that outlast the cells
  run_morphology2d.py      # 2D: cell arrangement → ECM morphology over a time axis
  run_culture.py           # cell density / DLP pattern → ECM pore morphology
  run_spheroid.py          # spheroid in gel → time-course remodeling + quantification
  run_whitepulp.py         # spleen white-pulp mimic vs random control
  run_patterns.py          # grow ECM into a chosen seeding pattern, over time
  make_track_scene.py      # export the cells' collagen tracks to viewer/scene.json
  make_viewer_scene.py     # export a grown network to viewer/scene.json
  viewer/index.html        # three.js 3D viewer (SSAO, translucent organic cells)
  verify_mechanics.py      # regression guard for the mechanics predictions
  verify_remodel.py        # regression guard for plastic remodeling
  ecm_pipeline/
    synthetic.py           # MVP1–4 scattered-rod generator (fast, PhysiMeSS-style)
    ecm_network.py         # growing branching crosslinked ECM NETWORK (web/mesh)
    ecm_mechanics.py       # fiber-network mechanics under cell traction (FIRE solver)
    ecm_remodel.py         # plastic, time-evolving remodeling driven by tension
    seeding.py             # cell seeding patterns + DLP masks + spheroids
    morpho_metrics.py      # quantitative ECM-architecture metrics (2D)
    render.py              # shared 2D renderer (light/dark + gradient colour scale)
    web_export.py          # export a scene to JSON for the three.js viewer
    physicell_parser.py    # MultiCellDS xml + .mat → CSV schema  (← real data in)
    fixture.py             # write synthetic frames AS MultiCellDS, for verification
    metrics.py             # the 8 structural metrics + CSV loaders
    export_vtk.py          # dependency-free VTP (ParaView) + scene JSON (web) writers
    figures.py             # matplotlib publication figures (300 dpi)
  output/                  # generated (figures tracked, bulk data gitignored)
```

## Where this sits in the full stack & next steps

- **This repo** = quantitative analysis + export (the paper's body).
- **ParaView** = open the `.vtp` for debugging / volume / threshold views.
- **Blender** = `scene_final.json` or VTP → publication-quality movie (next module).
- **Web viewer** = `scene_final.json` → three.js/vtk.js interactive supplementary
  figure (this is the `3DViewer` repo's natural home).

Suggested follow-ups: (a) a real PhysiCell/PhysiMeSS parser into the CSV schema;
(b) `render/blender_render.py`; (c) wire `scene_final.json` into the existing
web viewer.
