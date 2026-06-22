#!/usr/bin/env python3
"""Honest test: does the ECM follow a designed outline?
Spiral target, two routes:
  LEFT  sparse point anchors (what I showed before) -> cells bridge NEAREST
        anchors = proximity graph -> chords/blob, does NOT follow the outline.
  RIGHT continuous cure ALONG the outline (direct photopatterning) -> the cured
        stiff fibers ARE the shape; cells reinforce along it.
"""
from __future__ import annotations
import os, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_cure_compare.png")
W = H = 440.0
px, py, fx, cured, fib, cells, grid, GS, path = [], [], [], [], [], [], {}, 12, []


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
    px.append(nx); py.append(ny); fx.append(False); cured.append(False); return len(px)-1
def addF(i, j):
    if i == j: return
    L0 = math.hypot(px[i]-px[j], py[i]-py[j])
    if L0 < 0.5: return
    fib.append({"i": i, "j": j, "L0": L0, "r0": L0, "k": 0.5})


def spiral_path():
    pts, t = [], 0.0
    while t < 3.2*math.pi:
        r = 22 + 17*t; pts.append((220+r*math.cos(t), 220+r*math.sin(t))); t += 0.18
    return pts
def dist_to_path(qx, qy, pts):
    best = 1e9
    for k in range(len(pts)-1):
        ax, ay = pts[k]; bx, by = pts[k+1]; dx = bx-ax; dy = by-ay; L2 = dx*dx+dy*dy
        u = ((qx-ax)*dx+(qy-ay)*dy)/L2 if L2 else 0.0; u = max(0, min(1, u))
        d = math.hypot(qx-(ax+u*dx), qy-(ay+u*dy))
        if d < best: best = d
    return best


def seed_gel(nf):
    for _ in range(nf):
        x0, y0 = 8+random.random()*(W-16), 8+random.random()*(H-16)
        ang, ln = random.random()*2*math.pi, 40+random.random()*55; sn = max(3, int(ln/6)); prev = addN(x0, y0)
        for s in range(1, sn+1):
            nx = max(4, min(W-4, x0+ln*s/sn*math.cos(ang))); ny = max(4, min(H-4, y0+ln*s/sn*math.sin(ang)))
            nj = addN(nx, ny); addF(prev, nj); prev = nj
    rebuild()
    for i in range(len(px)):
        for m in near(px[i], py[i], 5):
            if m > i: addF(i, m)


def reset(mode):
    global path
    px.clear(); py.clear(); fx.clear(); cured.clear(); fib.clear(); cells.clear(); grid.clear()
    path = spiral_path()
    seed_gel(220)
    if mode == "sparse":
        anchors = path[::8]                       # sparse points along the spiral
        for i in range(len(px)):
            for (ax, ay) in anchors:
                if (px[i]-ax)**2+(py[i]-ay)**2 < 15*15: fx[i] = True; cured[i] = True
    else:                                         # continuous cure along the whole outline
        for i in range(len(px)):
            if dist_to_path(px[i], py[i], path) < 9: fx[i] = True; cured[i] = True
    for o in fib:
        if cured[o["i"]] and cured[o["j"]]: o["k"] = 3.0
    for _ in range(55):
        cells.append({"x": 8+random.random()*(W-16), "y": 8+random.random()*(H-16)})
    rebuild()


def pbd():
    for o in fib:
        i, j = o["i"], o["j"]; dx = px[j]-px[i]; dy = py[j]-py[i]; L = math.hypot(dx, dy)
        if L < 1e-4: continue
        s = 0.5*min(0.9, 0.25+0.22*o["k"])*(L-o["L0"])/L
        if not fx[i]: px[i] += s*dx; py[i] += s*dy
        if not fx[j]: px[j] -= s*dx; py[j] -= s*dy
def trac():
    reach, pull = min(W, H)*0.10, 0.05
    for cell in cells:
        for n in near(cell["x"], cell["y"], reach):
            if fx[n]: continue
            dx = cell["x"]-px[n]; dy = cell["y"]-py[n]; r = math.hypot(dx, dy) or 1
            if r < 6: continue
            w = pull*(1-r/reach); px[n] += w*dx/r; py[n] += w*dy/r
def kin():
    ks, Fs, kd, Fp, kb = 0.10, 0.04, 0.025, 0.03, 0.30
    for f in range(len(fib)-1, -1, -1):
        o = fib[f]
        if cured[o["i"]] and cured[o["j"]]: continue
        L = math.hypot(px[o["j"]]-px[o["i"]], py[o["j"]]-py[o["i"]]); t = max((L-o["L0"])/o["L0"], 0.0)
        o["k"] += ks*t/(Fs+t) - kd*max(o["k"]-kb, 0.0)*math.exp(-t/Fp)
        o["k"] = min(3.5, max(0.0, o["k"]))
        if o["k"] > 1.8: o["L0"] = max(0.62*o["r0"], o["L0"]*0.992)
        if o["k"] < 0.4 and random.random() < 0.0015: fib.pop(f)
def nstiff():
    nk = [0.0]*len(px)
    for o in fib: nk[o["i"]] += o["k"]; nk[o["j"]] += o["k"]
    return nk
def migrate():
    rebuild(); nk = nstiff(); sense = min(W, H)*0.16; sp = min(W, H)*0.007
    def S(qx, qy):
        s = 0.0
        for n in near(qx, qy, sense): s += nk[n]
        return s
    for cell in cells:
        p = sense*0.6
        gx = S(cell["x"]+p, cell["y"])-S(cell["x"]-p, cell["y"]); gy = S(cell["x"], cell["y"]+p)-S(cell["x"], cell["y"]-p)
        gn = math.hypot(gx, gy); ra = random.random()*6.283
        if gn > 1e-6: mx, my = gx/gn+0.2*math.cos(ra), gy/gn+0.2*math.sin(ra)
        else: mx, my = math.cos(ra), math.sin(ra)
        mn = math.hypot(mx, my) or 1
        cell["x"] = max(6, min(W-6, cell["x"]+sp*mx/mn)); cell["y"] = max(6, min(H-6, cell["y"]+sp*my/mn))


def run(mode, steps=340):
    random.seed(5); reset(mode)
    for _ in range(steps):
        rebuild()
        for _ in range(2): pbd(); trac()
        kin(); migrate()
    return [(px[o["i"]], py[o["i"]], px[o["j"]], py[o["j"]], o["k"]) for o in fib]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2)); fig.patch.set_facecolor("white")
    for ax, (mode, title) in zip(axes, [("sparse", "SPARSE anchors → proximity blob (does NOT follow)"),
                                        ("line", "CONTINUOUS cure along outline → follows")]):
        fibers = run(mode)
        segs = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2, k) in fibers]
        ax.add_collection(LineCollection(segs, colors=[cm.coolwarm(min(1.0, k/3.0)) for *_, k in fibers],
                                         linewidths=[0.4+min(1.4, k*0.45) for *_, k in fibers]))
        ax.plot([p[0] for p in path], [p[1] for p in path], "--", color="#e0a000", lw=1.2, alpha=0.7)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
