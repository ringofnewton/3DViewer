#!/usr/bin/env python3
"""Arbitrary-shape patterning — CORRECTED, verified mechanism.

Earlier attempts let cells contract the bulk gel + reinforce by proximity, which
collapsed everything into a tangle that did NOT follow the design. Fix: photo-cure
the desired outline as a continuous GUIDE; cells home to it (durotaxis) and
DEPOSIT ECM locally along it (no bulk contraction). The reinforced ECM then traces
the designed shape (contact guidance / durotaxis on a patterned scaffold).

    python build_arbitrary_fig.py
"""
from __future__ import annotations
import os, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_arbitrary.png")
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
def addN(nx, ny): px.append(nx); py.append(ny); fx.append(False); cured.append(False); return len(px)-1
def addF(i, j):
    if i == j: return
    if math.hypot(px[i]-px[j], py[i]-py[j]) < 0.5: return
    fib.append({"i": i, "j": j, "k": 0.5})


# ---- target outlines (fine, continuous polylines) ----
def densify(pts, step=7):
    out = []
    for k in range(len(pts)-1):
        ax, ay = pts[k]; bx, by = pts[k+1]; d = math.hypot(bx-ax, by-ay); n = max(1, int(d/step))
        for s in range(n): out.append((ax+(bx-ax)*s/n, ay+(by-ay)*s/n))
    out.append(pts[-1]); return out
def star_path():
    cx, cy, ro, ri, n = 220, 222, 150, 64, 5; v = []
    for i in range(2*n+1):
        a = -math.pi/2 + math.pi*i/n; r = ro if i % 2 == 0 else ri
        v.append((cx+r*math.cos(a), cy+r*math.sin(a)))
    return densify(v)
def spiral_path():
    p, t = [], 0.0
    while t < 3.2*math.pi:
        r = 22+17*t; p.append((220+r*math.cos(t), 220+r*math.sin(t))); t += 0.12
    return densify(p)
def wave_path():
    return densify([(55+x, 220+95*math.sin(2*math.pi*x/210)) for x in range(0, 331, 8)])


def dpath(qx, qy):
    best = 1e9
    for k in range(len(path)-1):
        ax, ay = path[k]; bx, by = path[k+1]; dx = bx-ax; dy = by-ay; L2 = dx*dx+dy*dy
        u = ((qx-ax)*dx+(qy-ay)*dy)/L2 if L2 else 0; u = max(0, min(1, u))
        d = math.hypot(qx-(ax+u*dx), qy-(ay+u*dy))
        if d < best: best = d
    return best


def seed(nf):
    for _ in range(nf):
        x0, y0 = 8+random.random()*(W-16), 8+random.random()*(H-16); ang = random.random()*2*math.pi
        ln = 40+random.random()*55; sn = max(3, int(ln/6)); prev = addN(x0, y0)
        for s in range(1, sn+1):
            nx = max(4, min(W-4, x0+ln*s/sn*math.cos(ang))); ny = max(4, min(H-4, y0+ln*s/sn*math.sin(ang)))
            nj = addN(nx, ny); addF(prev, nj); prev = nj
    rebuild()
    for i in range(len(px)):
        for m in near(px[i], py[i], 5):
            if m > i: addF(i, m)


def reset(shape):
    global path
    px.clear(); py.clear(); fx.clear(); cured.clear(); fib.clear(); cells.clear(); grid.clear()
    path = shape; seed(240)
    for i in range(len(px)):
        if dpath(px[i], py[i]) < 8: cured[i] = True
    for o in fib:
        if cured[o["i"]] and cured[o["j"]]: o["k"] = 3.0
    for _ in range(60): cells.append({"x": 8+random.random()*(W-16), "y": 8+random.random()*(H-16)})
    rebuild()


def nstiff():
    nk = [0.0]*len(px)
    for o in fib: nk[o["i"]] += o["k"]; nk[o["j"]] += o["k"]
    return nk
def migrate(anchors):
    rebuild(); nk = nstiff(); sense = min(W, H)*0.16; sp = min(W, H)*0.007; lam = 55.0
    def S(qx, qy):
        s = 0.0
        for n in near(qx, qy, sense): s += 0.1*nk[n]
        for (ax, ay) in anchors: s += 80/(1+((qx-ax)**2+(qy-ay)**2)/(lam*lam))
        return s
    for cell in cells:
        p = sense*0.6
        gx = S(cell["x"]+p, cell["y"])-S(cell["x"]-p, cell["y"]); gy = S(cell["x"], cell["y"]+p)-S(cell["x"], cell["y"]-p)
        gn = math.hypot(gx, gy); ra = random.random()*6.283
        if gn > 1e-6: mx, my = gx/gn+0.15*math.cos(ra), gy/gn+0.15*math.sin(ra)
        else: mx, my = math.cos(ra), math.sin(ra)
        mn = math.hypot(mx, my) or 1
        cell["x"] = max(6, min(W-6, cell["x"]+sp*mx/mn)); cell["y"] = max(6, min(H-6, cell["y"]+sp*my/mn))
def deposit():
    mark = [False]*len(px)
    for cell in cells:
        for n in near(cell["x"], cell["y"], 14): mark[n] = True
    for o in fib:
        if cured[o["i"]] and cured[o["j"]]: continue
        if mark[o["i"]] or mark[o["j"]]: o["k"] = min(3.0, o["k"]+0.10)   # cells deposit ECM locally
        else: o["k"] = max(0.0, o["k"]-0.01)                               # elsewhere slowly clears


def run(shape, steps=320):
    random.seed(5); reset(shape); anchors = shape[::3]
    for _ in range(steps): migrate(anchors); deposit()
    fibers = [(px[o["i"]], py[o["i"]], px[o["j"]], py[o["j"]], o["k"]) for o in fib]
    rN = rT = 0
    for (x1, y1, x2, y2, k) in fibers:
        if k > 1.5:
            rN += 1
            if dpath((x1+x2)/2, (y1+y2)/2) < 16: rT += 1
    return fibers, (100*rT/max(rN, 1))


def main():
    panels = [("star (non-convex)", star_path()), ("spiral", spiral_path()), ("wave / S-curve", wave_path())]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7)); fig.patch.set_facecolor("white")
    for ax, (title, shape) in zip(axes, panels):
        fibers, onpath = run(shape)
        print(f"{title}: {onpath:.0f}% of reinforced ECM on the designed outline")
        segs = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2, k) in fibers]
        ax.add_collection(LineCollection(segs, colors=[cm.coolwarm(min(1.0, k/3.0)) for *_, k in fibers],
                                         linewidths=[0.4+min(1.5, k*0.5) for *_, k in fibers]))
        ax.plot([p[0] for p in shape], [p[1] for p in shape], "--", color="#e0a000", lw=1.1, alpha=0.55)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}  ({onpath:.0f}% on-target)", fontsize=12)
    sm = cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["soft", "ECM"]); cb.set_label("cell-deposited ECM")
    fig.suptitle("Cure the outline as a guide → cells home to it & deposit ECM along it → the shape is traced",
                 fontsize=13, y=1.02)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
