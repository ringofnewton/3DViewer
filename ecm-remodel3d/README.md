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
  verify_mechanics.py      # regression guard for the mechanics predictions
  ecm_pipeline/
    synthetic.py           # MVP1–4 scattered-rod generator (fast, PhysiMeSS-style)
    ecm_network.py         # growing branching crosslinked ECM NETWORK (web/mesh)
    ecm_mechanics.py       # fiber-network mechanics under cell traction (FIRE solver)
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
