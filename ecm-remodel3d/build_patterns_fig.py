#!/usr/bin/env python3
"""4-panel 'inverse design' figure from the CURRENT integrated sim (continuous
kinetics): cell foci in chosen layouts (triangle / square / pentagon / hub+spokes)
-> the matrix self-organizes reinforced STRUTS as bridges between foci, matching
the designed (dashed) target topology. Soft->strut reinforcement colour.

    python build_patterns_fig.py
"""
from __future__ import annotations
import os, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_patterns.png")
W = H = 440.0
NODECAP, FIBCAP = 3600, 2200
px, py, fx, fib, cells, grid, GS, foci, edges = [], [], [], [], [], {}, 12, [], []


def gk(a, b): return (a, b)
def rebuild():
    grid.clear()
    for i in range(len(px)): grid.setdefault(gk(int(px[i]/GS), int(py[i]/GS)), []).append(i)
def near(qx, qy, d):
    ix, iy, r, d2 = int(qx/GS), int(qy/GS), [], d*d
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for n in grid.get(gk(ix+a, iy+b), ()):
                if (px[n]-qx)**2+(py[n]-qy)**2 < d2: r.append(n)
    return r
def addN(nx, ny):
    px.append(nx); py.append(ny); fx.append(nx < 6 or ny < 6 or nx > W-6 or ny > H-6); return len(px)-1
def addF(i, j):
    if i == j: return
    L0 = math.hypot(px[i]-px[j], py[i]-py[j])
    if L0 < 0.5: return
    fib.append({"i": i, "j": j, "L0": L0, "r0": L0, "k": 0.5})


def pattern(p):
    cx, cy, R, f, e = W/2, H/2, min(W, H)*0.32, [], []
    def ring(n, off):
        for i in range(n):
            a = -math.pi/2 + 2*math.pi*i/n + off
            f.append((cx+R*math.cos(a), cy+R*math.sin(a)))
    if p == "tri": ring(3, 0); e = [(0, 1), (1, 2), (2, 0)]
    elif p == "sq": ring(4, math.pi/4); e = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif p == "pent": ring(5, 0); e = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    elif p == "hub":
        f.append((cx, cy))
        for i in range(5):
            a = -math.pi/2 + 2*math.pi*i/5; f.append((cx+R*math.cos(a), cy+R*math.sin(a)))
        e = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    return f, e


def seed_bg(nf):
    for _ in range(nf):
        x0, y0 = 10+random.random()*(W-20), 10+random.random()*(H-20)
        ang, ln = random.random()*2*math.pi, 40+random.random()*55; sn = max(3, int(ln/6)); prev = addN(x0, y0)
        for s in range(1, sn+1):
            nx = max(4, min(W-4, x0+ln*s/sn*math.cos(ang))); ny = max(4, min(H-4, y0+ln*s/sn*math.sin(ang)))
            nj = addN(nx, ny); addF(prev, nj); prev = nj
    rebuild()
    for i in range(len(px)):
        for m in near(px[i], py[i], 5):
            if m > i: addF(i, m)


def reset(p):
    global foci, edges
    px.clear(); py.clear(); fx.clear(); fib.clear(); cells.clear(); grid.clear()
    foci, edges = pattern(p)
    seed_bg(150)
    for i, (fxp, fyp) in enumerate(foci):
        for _ in range(5):
            a, rr = random.random()*2*math.pi, random.random()*10
            cells.append({"x": fxp+rr*math.cos(a), "y": fyp+rr*math.sin(a), "home": i})
    rebuild()


def synth():
    rate = 0.9*(1-len(fib)/FIBCAP)
    if rate <= 0 or len(px) > NODECAP: return
    reach, seg = min(W, H)*0.12, 5.0
    for cell in cells:
        if random.random() > rate: continue
        nn = near(cell["x"], cell["y"], reach)
        frm = nn[int(random.random()*len(nn))] if nn else addN(cell["x"]+(random.random()-0.5)*5, cell["y"]+(random.random()-0.5)*5)
        ang = random.random()*2*math.pi; nx, ny = px[frm]+seg*math.cos(ang), py[frm]+seg*math.sin(ang)
        if nx < 5 or ny < 5 or nx > W-5 or ny > H-5: continue
        nj = addN(nx, ny); addF(frm, nj)
def pbd():
    for o in fib:
        i, j = o["i"], o["j"]; dx = px[j]-px[i]; dy = py[j]-py[i]; L = math.hypot(dx, dy)
        if L < 1e-4: continue
        s = 0.5*min(0.9, 0.25+0.22*o["k"])*(L-o["L0"])/L
        if not fx[i]: px[i] += s*dx; py[i] += s*dy
        if not fx[j]: px[j] -= s*dx; py[j] -= s*dy
def trac():
    reach, pull = min(W, H)*0.18, 0.06
    for cell in cells:
        for n in near(cell["x"], cell["y"], reach):
            if fx[n]: continue
            dx = cell["x"]-px[n]; dy = cell["y"]-py[n]; r = math.hypot(dx, dy) or 1
            if r < 8: continue
            w = pull*(1-r/reach); px[n] += w*dx/r; py[n] += w*dy/r
def kin():
    ks, Fs, kd, Fp, kb = 0.10, 0.05, 0.03, 0.03, 0.30
    for f in range(len(fib)-1, -1, -1):
        o = fib[f]; L = math.hypot(px[o["j"]]-px[o["i"]], py[o["j"]]-py[o["i"]]); t = max((L-o["L0"])/o["L0"], 0.0)
        o["k"] += ks*t/(Fs+t) - kd*max(o["k"]-kb, 0.0)*math.exp(-t/Fp)
        o["k"] = min(3.5, max(0.0, o["k"]))
        if o["k"] > 1.8: o["L0"] = max(0.62*o["r0"], o["L0"]*0.992)  # gentle compaction of true struts
        if o["k"] < 0.4 and random.random() < 0.0015: fib.pop(f)
def cohere():
    for cell in cells:
        fxp, fyp = foci[cell["home"]]
        cell["x"] += 0.10*(fxp-cell["x"])+(random.random()-0.5)*0.5
        cell["y"] += 0.10*(fyp-cell["y"])+(random.random()-0.5)*0.5
def run(p, steps=360):
    random.seed(3); reset(p)
    for s in range(steps):
        rebuild(); synth()
        for _ in range(2): pbd(); trac()
        if s % 3 == 0: kin()
        if s % 4 == 0: cohere()
    return ([(px[o["i"]], py[o["i"]], px[o["j"]], py[o["j"]], o["k"]) for o in fib],
            [(fxp, fyp) for (fxp, fyp) in foci], list(edges))


def main():
    panels = [("tri", "triangle (3 foci)"), ("sq", "square (4 foci)"),
              ("pent", "pentagon (5 foci)"), ("hub", "hub + 5 spokes")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2)); fig.patch.set_facecolor("white")
    for ax, (p, title) in zip(axes, panels):
        fibers, fc, ed = run(p)
        segs = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2, k) in fibers]
        cols = [cm.coolwarm(min(1.0, k/3.0)) for *_, k in fibers]
        lws = [0.4+min(1.4, k*0.45) for *_, k in fibers]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws))
        for (a, b) in ed:
            ax.plot([fc[a][0], fc[b][0]], [fc[a][1], fc[b][1]], "--", color="#e0a000", lw=1.6, alpha=0.9)
        ax.scatter([f[0] for f in fc], [f[1] for f in fc], s=70, c="#155e63",
                   edgecolors="white", linewidths=1.0, zorder=5)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=13)
    sm = cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["soft", "strut"]); cb.set_label("reinforcement (soft→strut)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
