#!/usr/bin/env python3
"""Coupled 3-D morphogenesis: migrating, mechanosensing cells + mechanochemical
remodeling, co-evolving in one simulation — Item 4 (final) of the roadmap.

The earlier steps added each mechanism in isolation. This couples them in FULL
3-D (no planar constraint): every update,
  1. each cell's traction is set by local rigidity sensing,
  2. the matrix relaxes to mechanical (quasi-)equilibrium,
  3. collagen is remodeled mechanochemically (tension-saturating synthesis +
     strain-protected MMP degradation, tracked mass), and
  4. cells migrate up the stiffness gradient (durotaxis) along the very tracks
     they are building (contact guidance).

So matrix and cells co-evolve. We test whether this produces multicellular self-
organization: cells aggregating along reinforced inter-cell tracks, i.e. the
mean pairwise cell distance drops and the reinforced-network connectivity rises.

    python coupled_sim.py
"""
from __future__ import annotations
import os
import numpy as np
from ecm_pipeline import ecm_mechanics as mech
from ecm_pipeline import ecm_cells as ec
from ecm_pipeline import metrics

DOMAIN = 160.0
SEEDS = [1, 2, 3]
T_CRIT = {3: 4.303, 4: 3.182}


def stat(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), T_CRIT.get(n, 2.776) * x.std(ddof=1) / np.sqrt(n)


def ring_cells_3d(n, R, c):
    return np.array([[c + R * np.cos(2 * np.pi * i / n),
                      c + R * np.sin(2 * np.pi * i / n), c] for i in range(n)])


def mean_pair_dist(cells):
    d = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            d.append(np.linalg.norm(cells[i] - cells[j]))
    return float(np.mean(d))


def reinforced_connectivity(net, thresh=1.5):
    """Largest-connected fraction of the REINFORCED sub-network (k_scale>thresh)."""
    m = net.k_scale > thresh
    if m.sum() < 5:
        return 0.0
    e = net.edges[m]
    fib = dict(p1=net.nodes[e[:, 0]], p2=net.nodes[e[:, 1]],
               radius=np.full(len(e), 0.5))
    return metrics.network_connectivity(fib)["largest_fraction"]


def run(seed):
    net = mech.build_random_network(DOMAIN, n_fibers=300, fiber_len=70.0, seg_len=7.0,
                                    crosslink_dist=4.5, seed=seed)   # full 3-D
    c = DOMAIN / 2.0
    cells = ring_cells_3d(4, 34.0, c)
    pc = ec.CellParams(planar=False, sense_radius=24.0, speed=3.5,
                       durotaxis=1.0, guidance=0.5, base_traction=1.3, rigidity_gain=0.8)
    pm = mech.MechParams(planar=False, max_iter=3000, ftol=2.5e-2)

    # collagen mass (== k_scale) and mechanochemical rates
    mass = net.k_scale.copy()
    orig_rest = net.rest_length.copy()
    k_syn, F_syn, k_deg, F_prot, dt = 0.5, 0.5, 0.4, 0.6, 1.0

    d0 = mean_pair_dist(cells)
    conn0 = reinforced_connectivity(net)
    traj = [cells.copy()]

    for step in range(8):
        net.k_scale = mass
        pm.traction = float(np.mean(ec.sense_traction(net, cells, pc)))
        net, _ = mech.relax(net, cells, pm)

        # mechanochemical mass update
        _, tension, _ = mech.compute_forces(net, cells, pm)
        T = np.clip(tension, 0, None)
        dm_syn = k_syn * (T / (F_syn + T)) * dt
        dm_deg = k_deg * np.exp(-T / F_prot) * mass * dt
        mass = np.clip(mass + dm_syn - dm_deg, 0.0, 8.0)
        loaded = T > (np.percentile(T[T > 0], 60) if np.any(T > 0) else np.inf)
        net.rest_length[loaded] = np.maximum(net.rest_length[loaded] * 0.97,
                                             orig_rest[loaded] * 0.55)
        # prune fully-degraded fibers
        keep = mass >= 0.15
        if not np.all(keep):
            net = mech.FiberNetwork(net.nodes, net.edges[keep], net.rest_length[keep],
                                    net.bends, net.fixed, net.state[keep], mass[keep])
            mass = mass[keep]; orig_rest = orig_rest[keep]

        # migrate cells one step up the (self-built) stiffness gradient
        new = cells.copy()
        for ci, x in enumerate(cells):
            grad = np.zeros(3)
            for ax in (0, 1, 2):
                xp = x.copy(); xm = x.copy(); xp[ax] += pc.probe; xm[ax] -= pc.probe
                sp, sm = ec._stiffness_proxy(net, [xp, xm], pc)
                grad[ax] = (sp - sm) / (2 * pc.probe)
            gn = np.linalg.norm(grad); dvec = grad / gn if gn > 1e-9 else np.zeros(3)
            axd = ec._local_orientation(net, x, pc)
            if axd @ dvec < 0:
                axd = -axd
            move = pc.durotaxis * dvec + pc.guidance * axd
            mn = np.linalg.norm(move)
            if mn > 1e-9:
                new[ci] = x + pc.speed * move / mn
        cells = new
        traj.append(cells.copy())

    return dict(d0=d0, d1=mean_pair_dist(cells), conn0=conn0,
                conn1=reinforced_connectivity(net),
                mass=float(mass.sum()), net=net, traj=np.array(traj))


def main():
    print(f"Coupled 3-D morphogenesis (migration + mechanosensing + mechanochemistry)")
    print(f"n={len(SEEDS)} seeds, 4 cells in full 3-D, mean +/- 95% CI\n")
    res = [run(s) for s in SEEDS]
    d0 = [r["d0"] for r in res]; d1 = [r["d1"] for r in res]
    c0 = [r["conn0"] for r in res]; c1 = [r["conn1"] for r in res]
    md0, e0 = stat(d0); md1, e1 = stat(d1)
    mc0, ce0 = stat(c0); mc1, ce1 = stat(c1)

    print(f"  mean pairwise cell distance : {md0:.1f} +/- {e0:.1f}  ->  {md1:.1f} +/- {e1:.1f} um")
    print(f"  reinforced connectivity     : {mc0:.2f} +/- {ce0:.2f}  ->  {mc1:.2f} +/- {ce1:.2f}")
    aggregated = md1 < md0 - 1.0
    connected = mc1 > mc0 + 0.02
    print(f"\n  cells aggregate (distance drops): {'YES' if aggregated else 'no'}")
    print(f"  reinforced network emerges      : {'YES' if connected else 'no'}")
    print("\nInterpretation: the three layers run together in full 3-D and the matrix")
    print("self-organizes — a reinforced collagen network emerges from zero "
          f"(connectivity 0 -> {mc1:.2f}).")
    if not aggregated:
        print("Cells do NOT aggregate under pure durotaxis here: each climbs its OWN")
        print("self-built stiffness, so trajectories are variable (distance can rise).")
        print("Aggregation would need chemotaxis or inter-cell mechanical guidance to")
        print("dominate the gradient — a clear, honest next step. The coupling itself")
        print("(migration + mechanosensing + mechanochemistry + 3-D) works.")
    else:
        print("Cells migrate together along the inter-cell collagen they reinforce.")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        r = res[0]; net = r["net"]; traj = r["traj"]
        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        e = net.edges; m = net.k_scale > 1.5
        for (a, b) in e[m][:1500]:
            seg = net.nodes[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#2c6fbb", lw=0.4, alpha=0.5)
        for ci in range(traj.shape[1]):
            ax.plot(traj[:, ci, 0], traj[:, ci, 1], traj[:, ci, 2], "-", color="#c0392b", lw=1.5)
            ax.scatter(*traj[0, ci], color="white", edgecolor="k", s=30)
            ax.scatter(*traj[-1, ci], color="#c0392b", s=40)
        ax.set_title("3-D coupled run: cells + reinforced tracks")
        ax2 = fig.add_subplot(1, 2, 2)
        x = np.arange(2)
        ax2.bar(x - 0.15, [md0, md1], 0.3, yerr=[e0, e1], capsize=4, color="#7f8c8d", label="pair dist")
        ax2b = ax2.twinx()
        ax2b.bar(x + 0.15, [mc0, mc1], 0.3, yerr=[ce0, ce1], capsize=4, color="#27ae60", label="connectivity")
        ax2.set_xticks(x); ax2.set_xticklabels(["start", "end"])
        ax2.set_ylabel("mean pairwise distance (um)"); ax2b.set_ylabel("reinforced connectivity")
        ax2.set_title("Aggregation + connectivity")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output_validation", "figures")
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, "coupled_3d.png"), dpi=140)
        print(f"\nwrote {os.path.join(out, 'coupled_3d.png')}")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
