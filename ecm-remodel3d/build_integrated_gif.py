#!/usr/bin/env python3
"""GIF of the INTEGRATED ECM morphogenesis sim with CONTINUOUS kinetics.

Same coupled model on one network, with realistic turnover (not one-shot
secretion): cells CONTINUOUSLY synthesize collagen, the matrix undergoes
tension-saturating (Hill) synthesis + strain-PROTECTED MMP degradation (dynamic
steady state), while cells sense (durotaxis), signal (chemotaxis), migrate and
aggregate, reinforcing the inter-cell bridges. Mobile-viewable.

    python build_integrated_gif.py
"""
from __future__ import annotations
import os, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "output_validation", "figures", "ecm_integrated.gif")
W = H = 440.0
NODECAP, FIBCAP, NCELL = 1800, 1000, 7
random.seed(7)

px, py, fx, fib, cells, grid, GS, frame = [], [], [], [], [], {}, 12, 0


def gkey(a, b): return (a, b)
def rebuild():
    grid.clear()
    for i in range(len(px)):
        grid.setdefault(gkey(int(px[i]/GS), int(py[i]/GS)), []).append(i)
def near(qx, qy, d):
    ix, iy, r, d2 = int(qx/GS), int(qy/GS), [], d*d
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for n in grid.get(gkey(ix+a, iy+b), ()):
                if (px[n]-qx)**2 + (py[n]-qy)**2 < d2:
                    r.append(n)
    return r
def add_node(nx, ny):
    px.append(nx); py.append(ny); fx.append(nx < 6 or ny < 6 or nx > W-6 or ny > H-6)
    return len(px)-1
def add_fiber(i, j, xl):
    if i == j: return
    L0 = math.hypot(px[i]-px[j], py[i]-py[j])
    if L0 < 0.5: return
    fib.append({"i": i, "j": j, "L0": L0, "k": 0.5, "xl": xl})


def reset():
    global frame
    px.clear(); py.clear(); fx.clear(); fib.clear(); cells.clear(); grid.clear(); frame = 0
    cx, cy, R = W/2, H/2, min(W, H)*0.30
    for c in range(NCELL):
        a = 2*math.pi*c/NCELL + (random.random()-0.5)*0.6; rr = R*(0.7+0.5*random.random())
        ex, ey = cx+rr*math.cos(a), cy+rr*math.sin(a)
        cells.append({"x": ex, "y": ey, "tx": [ex], "ty": [ey]})
    rebuild()


def synthesize():
    rate = 0.9*(1 - len(fib)/FIBCAP)
    if rate <= 0 or len(px) > NODECAP: return
    reach, seg = min(W, H)*0.12, 5.0
    for cell in cells:
        if random.random() > rate: continue
        nn = near(cell["x"], cell["y"], reach)
        frm = nn[int(random.random()*len(nn))] if nn else \
            add_node(cell["x"]+(random.random()-0.5)*5, cell["y"]+(random.random()-0.5)*5)
        ang = random.random()*2*math.pi
        nx, ny = px[frm]+seg*math.cos(ang), py[frm]+seg*math.sin(ang)
        if nx < 5 or ny < 5 or nx > W-5 or ny > H-5: continue
        nj = add_node(nx, ny); add_fiber(frm, nj, False)
        for m in near(nx, ny, 6):
            if m != nj and m != frm:
                add_fiber(nj, m, True); break


def pbd():
    for o in fib:
        i, j = o["i"], o["j"]; dx = px[j]-px[i]; dy = py[j]-py[i]; L = math.hypot(dx, dy)
        if L < 1e-4: continue
        s = 0.5*min(0.9, 0.25+0.22*o["k"])*(L-o["L0"])/L
        if not fx[i]: px[i] += s*dx; py[i] += s*dy
        if not fx[j]: px[j] -= s*dx; py[j] -= s*dy
def traction():
    reach, pull = min(W, H)*0.13, 0.05
    for cell in cells:
        for n in near(cell["x"], cell["y"], reach):
            if fx[n]: continue
            dx = cell["x"]-px[n]; dy = cell["y"]-py[n]; r = math.hypot(dx, dy) or 1
            if r < 8: continue
            w = pull*(1-r/reach); px[n] += w*dx/r; py[n] += w*dy/r
def kinetics():
    ks, Fs, kd, Fp, kb = 0.10, 0.02, 0.025, 0.03, 0.30
    for f in range(len(fib)-1, -1, -1):
        o = fib[f]; L = math.hypot(px[o["j"]]-px[o["i"]], py[o["j"]]-py[o["i"]])
        t = max((L-o["L0"])/o["L0"], 0.0)
        o["k"] += ks*t/(Fs+t) - kd*max(o["k"]-kb, 0.0)*math.exp(-t/Fp)  # relax to baseline (persists)
        o["k"] = min(3.5, max(0.0, o["k"]))
        if o["k"] < 0.4 and random.random() < 0.0015: fib.pop(f)        # slow turnover, network stays
def compact():
    used = set()
    for o in fib: used.add(o["i"]); used.add(o["j"])
    mp = {}; nx = []; ny = []; nf = []
    for i in range(len(px)):
        if i in used:
            mp[i] = len(nx); nx.append(px[i]); ny.append(py[i]); nf.append(fx[i])
    for o in fib: o["i"] = mp[o["i"]]; o["j"] = mp[o["j"]]
    px[:] = nx; py[:] = ny; fx[:] = nf
def node_stiff():
    nk = [0.0]*len(px)
    for o in fib: nk[o["i"]] += o["k"]; nk[o["j"]] += o["k"]
    return nk
def migrate(chemo_w):
    rebuild(); nk = node_stiff(); sense = min(W, H)*0.14; lam = min(W, H)*0.34; speed = min(W, H)*0.006
    def S(qx, qy): return sum(nk[n] for n in near(qx, qy, sense))
    moved = []
    for c, cell in enumerate(cells):
        p = sense*0.6
        gx = S(cell["x"]+p, cell["y"])-S(cell["x"]-p, cell["y"])
        gy = S(cell["x"], cell["y"]+p)-S(cell["x"], cell["y"]-p)
        gn = math.hypot(gx, gy) or 1; dux, duy = gx/gn, gy/gn
        hx = hy = 0.0
        for j, oc in enumerate(cells):
            if j == c: continue
            ddx = oc["x"]-cell["x"]; ddy = oc["y"]-cell["y"]; r = math.hypot(ddx, ddy) or 1
            wg = math.exp(-r/lam)/lam; hx += ddx/r*wg; hy += ddy/r*wg
        hn = math.hypot(hx, hy) or 1; hx /= hn; hy /= hn
        ra = random.random()*6.283
        mx = dux+chemo_w*hx+0.35*math.cos(ra); my = duy+chemo_w*hy+0.35*math.sin(ra)
        mn = math.hypot(mx, my) or 1
        moved.append((max(10, min(W-10, cell["x"]+speed*mx/mn)),
                      max(10, min(H-10, cell["y"]+speed*my/mn))))
    for c, cell in enumerate(cells): cell["x"], cell["y"] = moved[c]
    ms = 26
    for _ in range(4):
        for a in range(len(cells)):
            for b in range(a+1, len(cells)):
                dx = cells[b]["x"]-cells[a]["x"]; dy = cells[b]["y"]-cells[a]["y"]; r = math.hypot(dx, dy) or 1
                if r < ms:
                    pu = (ms-r)/2/r
                    cells[a]["x"] -= pu*dx; cells[a]["y"] -= pu*dy
                    cells[b]["x"] += pu*dx; cells[b]["y"] += pu*dy
    for cell in cells: cell["tx"].append(cell["x"]); cell["ty"].append(cell["y"])


def step(chemo_w=1.0):
    global frame
    frame += 1
    rebuild(); synthesize()
    for _ in range(2): pbd(); traction()
    if frame % 3 == 0: kinetics()
    if frame % 4 == 0: migrate(chemo_w)
    if frame % 24 == 0: compact()


def main():
    reset()
    fig, ax = plt.subplots(figsize=(6, 6)); fig.patch.set_facecolor("white")
    capture = []; nsteps = 540
    for s in range(nsteps):
        step()
        if s % 12 == 0 or s == nsteps-1:
            capture.append(([(px[o["i"]], py[o["i"]], px[o["j"]], py[o["j"]], o["k"]) for o in fib],
                            [(c["x"], c["y"]) for c in cells],
                            [(list(c["tx"]), list(c["ty"])) for c in cells],
                            len(fib), sum(1 for o in fib if o["k"] > 1.5)))

    def col(k):
        t = min(1, k/3.0)
        return (0xcf/255+(0x1e/255-0xcf/255)*t, 0xe0/255+(0x3a/255-0xe0/255)*t,
                0xf5/255+(0x8a/255-0xf5/255)*t)

    def draw(fi):
        ax.clear(); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#f7fafc")
        fibers, cpos, trails, nf, nr = capture[fi]
        segs = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2, k) in fibers]
        ax.add_collection(LineCollection(segs, colors=[col(k) for *_, k in fibers],
                                         linewidths=[0.5+min(2.2, k*0.7) for *_, k in fibers]))
        for (tx, ty) in trails:
            ax.plot(tx, ty, color="#db2777", lw=1.2, alpha=0.55)
        for (cx, cy) in cpos:
            ax.scatter(cx, cy, s=110, c="#f97316", edgecolors="white", linewidths=1.6, zorder=5)
        ax.set_title("Continuous turnover: secretion ⇄ strain-protected degradation\n"
                     f"fibers {nf} (dynamic steady state) · reinforced bridges {nr}",
                     fontsize=12, color="#1f2933")

    anim = FuncAnimation(fig, draw, frames=len(capture), interval=150, blit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=7))
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB, {len(capture)} frames)")


if __name__ == "__main__":
    main()
