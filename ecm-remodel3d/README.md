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

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py
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

## The data contract (how to plug in real PhysiCell/PhysiMeSS output)

Every analysis function reads only these plain CSVs, so the *only* thing you
replace is the generator:

```
cells_t{frame:03d}.csv : id, x, y, z, radius, phenotype, vx, vy, vz
fibers_t{frame:03d}.csv: id, x1, y1, z1, x2, y2, z2, radius, state, birth
tracks.csv             : cell_id, frame, x, y, z
events.csv             : frame, step, births, deaths
```

PhysiMeSS already represents ECM fibers as line segments (2D) / cylinders (3D)
with two endpoints — that maps directly onto `x1..z2`. Write a parser of the
PhysiCell `output/*.xml` + `.mat` (cells) and the PhysiMeSS fiber output into
the schema above, and `run_demo.py` produces the same figures from real data.

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
  run_demo.py              # end-to-end orchestrator
  requirements.txt
  ecm_pipeline/
    synthetic.py           # MVP1–4 synthetic generator  (← swap for PhysiCell parser)
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
