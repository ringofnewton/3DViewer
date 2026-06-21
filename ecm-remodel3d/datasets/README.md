# Reference datasets for distribution-level validation

These are **digitized-histogram-format** reference distributions for the
goodness-of-fit test in `distribution_compare.py` (Item 2 of the research
roadmap). They let the model's emergent distributions be compared to experimental
data at the *distribution* level (KS statistic + Wasserstein-1 / EMD), not just by
matching a mean or a range.

## File format

Plain CSV, one row per histogram bar, `#` for comments:

```
bin_low , bin_high , count
```

`count` is the number of measured objects in that bin. The **total count sets the
statistical power** of a KS test against the reference (more measured objects →
smaller detectable difference). This is exactly the output you get from digitizing
a published histogram with a tool such as WebPlotDigitizer, so a real digitization
is a **drop-in replacement** — no code change.

The loader and the binned (weighted-ECDF) KS/EMD live in
`ecm_pipeline/dist_stats.py`:
`load_histogram_csv`, `ks_2samp_weighted`, `wasserstein1_weighted`,
`sample_from_histogram`.

## Files

| file | observable | units | status |
|------|-----------|-------|--------|
| `collagen_pore_um.csv`   | pore size      | µm | wired into `distribution_compare.py` (A) |
| `fiber_diameter_nm.csv`  | fiber diameter | nm | format example (model needs a diameter observable to use it) |

## Provenance / honesty

The bin frequencies here are a **literature-informed reconstruction** — chosen to
reproduce the central tendency and right-skew reported for fibrillar collagen /
reticular networks by confocal- and electron-microscopy microstructure studies
(pore sizes O(1–80 µm), fibril diameters O(20–300 nm), both right-skewed /
log-normal-like). They are **not pixel-digitized from one specific figure**.

To make the goodness-of-fit a true *data-vs-model* validation, replace a file's
rows with a real digitization of a published histogram (same three columns). The
comparison code, statistics, and figures then report against real data unchanged.
