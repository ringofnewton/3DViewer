"""Unit calibration: map the model's reduced units to SI (µm, nN, Pa, hours).

The pipeline runs in reduced units (METHODS §units): lengths in µm, stiffness
relative to a reference `k_s = 1`, forces in `k_s·µm`, time in remodeling steps.
This module fixes the conversion factors from literature anchors and exposes a
consistency check between two *independent* measurables — single-fiber stiffness
and single-cell traction — which reveals what `k_s = 1` actually represents.

Literature anchors below are **order-of-magnitude** values from the collagen /
cell-mechanics literature; confirm exact numbers against primary sources before
publication. References (representative): single collagen fibril modulus ~0.1–1
GPa (Wenger 2007; Shen 2008); single-fibroblast total traction ~10–100 nN
(Munevar 2001; Legant 2010); reconstituted collagen gel plateau shear modulus
~1–100 Pa at 1–3 mg/mL (Storm 2005; Licup 2015); matrix remodeling/contraction
over hours (Grinnell 2003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ecm_mechanics as mech

UM = 1e-6  # one model length unit (µm) in metres


@dataclass
class Anchors:
    E_fibril_Pa: float = 1.0e8     # single collagen fibril Young's modulus (lower end)
    fiber_radius_um: float = 0.12  # model fiber radius (NetworkParams default)
    seg_len_um: float = 7.0        # segment rest length (build_random_network default)
    cell_traction_nN: float = 50.0 # single-fibroblast total traction
    step_hours: float = 2.0        # one remodeling step
    # network reference for the reduced traction sum
    traction: float = 1.5
    reach: float = 40.0
    reach_decay: float = 22.0
    cell_radius: float = 9.0
    domain: float = 200.0


def reduced_traction_per_cell(a: Anchors, seed: int = 1) -> float:
    """Total reduced traction magnitude a single cell exerts on a representative
    network (the model's own traction field summed over reachable nodes)."""
    net = mech.build_random_network(a.domain, n_fibers=240, fiber_len=70.0,
                                    seg_len=a.seg_len_um, crosslink_dist=4.5,
                                    seed=seed, slab_thickness=26.0)
    c = a.domain / 2.0
    cell = np.array([c, c, c])
    d = np.linalg.norm(net.nodes - cell, axis=1)
    m = np.where((d > a.cell_radius) & (d < a.reach),
                 a.traction * np.exp(-(d - a.cell_radius) / a.reach_decay), 0.0)
    return float(m.sum())


def calibrate(a: Anchors | None = None, seed: int = 1) -> dict:
    a = a or Anchors()
    A = np.pi * (a.fiber_radius_um * UM) ** 2          # fiber cross-section (m²)
    k_s_SI = a.E_fibril_Pa * A / (a.seg_len_um * UM)   # single-segment spring (N/m)
    sum_m = reduced_traction_per_cell(a, seed)

    # Anchor A — stiffness from a single fibril → predict cell traction
    force_unit_A = k_s_SI * UM                          # reduced force unit (N), k_s_red=1
    pred_traction_nN = sum_m * force_unit_A / 1e-9

    # Anchor B — measured cell traction → imply k_s and effective fiber modulus
    force_unit_B = (a.cell_traction_nN * 1e-9) / sum_m  # N
    k_s_SI_B = force_unit_B / UM
    E_eff_Pa = k_s_SI_B * (a.seg_len_um * UM) / A
    stress_unit_Pa = force_unit_B / (UM ** 2)

    return dict(
        sum_m=sum_m,
        k_s_SI_fibril=k_s_SI,
        force_unit_A_nN=force_unit_A / 1e-9,
        pred_traction_nN=pred_traction_nN,
        discrepancy=pred_traction_nN / a.cell_traction_nN,
        force_unit_B_nN=force_unit_B / 1e-9,
        k_s_SI_B=k_s_SI_B,
        E_eff_Pa=E_eff_Pa,
        stress_unit_Pa=stress_unit_Pa,
        length_unit_um=1.0,
        step_hours=a.step_hours,
    )
