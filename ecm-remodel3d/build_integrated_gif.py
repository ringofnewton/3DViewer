#!/usr/bin/env python3
"""GIF of the INTEGRATED ECM morphogenesis sim (same coupled model, one network).

Formation (cells secrete + crosslink) -> mechanics (traction) -> remodeling
(tensed fibers reinforced) -> migration (durotaxis + chemotaxis) -> aggregation,
ALL on one evolving network. Mirrors ecm_simulation.html. Mobile-viewable.

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
NODECAP, FORMEND, NCELL = 720, 150, 7
random.seed(7)

px, py, fx, fib, cells, grid, GS, frame, phase = [], [], [], [], [], {}, 12, 0, "form"


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
    global frame, phase
    px.clear(); py.clear(); fx.clear(); fib.clear(); cells.clear(); grid.clear()
    frame = 0; phase = "form"
    cx, cy, R = W/2, H/2, min(W, H)*0.24
    for c in range(NCELL):
        a = 2*math.pi*c/NCELL; ex, ey = cx+R*math.cos(a), cy+R*math.sin(a)
        tips = []
        for _ in range(3):
            d = 2*math.pi*random.random(); root = add_node(ex, ey)
            tips.append({"last": root, "dx": math.cos(d), "dy": math.sin(d),
                         "life": 34+int(random.random()*26)})
        cells.append({"x": ex, "y": ey, "tx": [ex], "ty": [ey], "tips": tips})
    rebuild()


def grow():
    rebuild()
    for cell in cells:
        nt = []
        for tp in cell["tips"]:
            rd = random.random()*2*math.pi
            ndx = 0.82*tp["dx"]+0.18*math.cos(rd); ndy = 0.82*tp["dy"]+0.18*math.sin(rd)
            nn = math.hypot(ndx, ndy) or 1; ndx /= nn; ndy /= nn
            nx = px[tp["last"]]+4.5*ndx; ny = py[tp["last"]]+4.5*ndy
            if nx < 5 or ny < 5 or nx > W-5 or ny > H-5 or tp["life"] <= 0 or len(px) > NODECAP:
                continue
            nj = add_node(nx, ny); add_fiber(tp["last"], nj, False)
            for m in near(nx, ny, 6):
                if m != nj and m != tp["last"]:
                    add_fiber(nj, m, True); break
            tp["last"] = nj; tp["dx"] = ndx; tp["dy"] = ndy; tp["life"] -= 1; nt.append(tp)
            if random.random() < 0.10 and len(nt)+len(cell["tips"]) < 10:
                ang = math.atan2(ndy, ndx)+(random.random()-0.5)*1.4
                nt.append({"last": nj, "dx": math.cos(ang), "dy": math.sin(ang),
                           "life": 18+int(random.random()*16)})
        cell["tips"] = nt


def pbd():
    for o in fib:
        i, j = o["i"], o["j"]; dx = px[j]-px[i]; dy = py[j]-py[i]; L = math.hypot(dx, dy)
        if L < 1e-4: continue
        s = 0.5*min(0.9, 0.25+0.22*o["k"])*(L-o["L0"])/L; cx = s*dx; cy = s*dy
        if not fx[i]: px[i] += cx; py[i] += cy
        if not fx[j]: px[j] -= cx; py[j] -= cy
def traction():
    reach, pull = min(W, H)*0.13, 0.05
    for cell in cells:
        for n in near(cell["x"], cell["y"], reach):
            if fx[n]: continue
            dx = cell["x"]-px[n]; dy = cell["y"]-py[n]; r = math.hypot(dx, dy) or 1
            if r < 8: continue
            w = pull*(1-r/reach); px[n] += w*dx/r; py[n] += w*dy/r
def remodel():
    for f in range(len(fib)-1, -1, -1):
        o = fib[f]; L = math.hypot(px[o["j"]]-px[o["i"]], py[o["j"]]-py[o["i"]])
        strain = (L-o["L0"])/o["L0"]
        o["k"] = min(3.2, o["k"]+0.05) if strain > 0.015 else max(0, o["k"]-0.018)
        if o["k"] < 0.05 and o["xl"]: fib.pop(f)
def node_stiff():
    nk = [0.0]*len(px)
    for o in fib: nk[o["i"]] += o["k"]; nk[o["j"]] += o["k"]
    return nk
def migrate(chemo_w):
    rebuild(); nk = node_stiff(); sense = min(W, H)*0.14; lam = min(W, H)*0.34
    speed = min(W, H)*0.006
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
            wgt = math.exp(-r/lam)/lam; hx += ddx/r*wgt; hy += ddy/r*wgt
        hn = math.hypot(hx, hy) or 1; hx /= hn; hy /= hn
        ra = random.random()*6.283; rng = 0.35
        mx = dux+chemo_w*hx+rng*math.cos(ra); my = duy+chemo_w*hy+rng*math.sin(ra)
        mn = math.hypot(mx, my) or 1
        moved.append((max(10, min(W-10, cell["x"]+speed*mx/mn)),
                      max(10, min(H-10, cell["y"]+speed*my/mn))))
    for c, cell in enumerate(cells): cell["x"], cell["y"] = moved[c]
    minSep = 26
    for _ in range(4):
        for a in range(len(cells)):
            for b in range(a+1, len(cells)):
                dx = cells[b]["x"]-cells[a]["x"]; dy = cells[b]["y"]-cells[a]["y"]
                r = math.hypot(dx, dy) or 1
                if r < minSep:
                    pu = (minSep-r)/2/r
                    cells[a]["x"] -= pu*dx; cells[a]["y"] -= pu*dy
                    cells[b]["x"] += pu*dx; cells[b]["y"] += pu*dy
    for cell in cells:
        cell["tx"].append(cell["x"]); cell["ty"].append(cell["y"])


def step(chemo_w=1.4):
    global frame, phase
    frame += 1
    if len(px) < NODECAP and phase == "form": grow()
    if frame > FORMEND or len(px) >= NODECAP: phase = "morph"
    for _ in range(2): pbd(); traction()
    if frame % 6 == 0: remodel()
    if phase == "morph" and frame % 4 == 0: migrate(chemo_w)


def main():
    reset()
    fig, ax = plt.subplots(figsize=(6, 6)); fig.patch.set_facecolor("white")
    capture = []
    nsteps = 430
    for s in range(nsteps):
        step()
        if s % 10 == 0 or s == nsteps-1:
            capture.append(([(px[o["i"]], py[o["i"]], px[o["j"]], py[o["j"]], o["k"]) for o in fib],
                            [(c["x"], c["y"]) for c in cells],
                            [(list(c["tx"]), list(c["ty"])) for c in cells],
                            phase, len(fib), sum(1 for o in fib if o["k"] > 1.5)))

    def col(k):
        t = min(1, k/3.0)
        return (0xcf/255+(0x1e/255-0xcf/255)*t, 0xe0/255+(0x3a/255-0xe0/255)*t,
                0xf5/255+(0x8a/255-0xf5/255)*t)

    def draw(fi):
        ax.clear(); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#f7fafc")
        segs, cols, lws = [], [], []
        fibers, cpos, trails, ph, nf, nr = capture[fi]
        for (x1, y1, x2, y2, k) in fibers:
            segs.append([(x1, y1), (x2, y2)]); cols.append(col(k)); lws.append(0.5+min(2.2, k*0.7))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws))
        for (tx, ty) in trails:
            ax.plot(tx, ty, color="#db2777", lw=1.2, alpha=0.55)
        for (cx, cy) in cpos:
            ax.scatter(cx, cy, s=110, c="#f97316", edgecolors="white", linewidths=1.6, zorder=5)
        lab = "FORMATION: cells build the network" if ph == "form" else \
              "MORPHOGENESIS: sense · migrate · remodel · aggregate"
        ax.set_title(f"{lab}\nfibers {nf} · reinforced bridges {nr}", fontsize=12, color="#1f2933")

    anim = FuncAnimation(fig, draw, frames=len(capture), interval=160, blit=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=6))
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB, {len(capture)} frames)")


if __name__ == "__main__":
    main()
