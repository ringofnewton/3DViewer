#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E10.5 마우스 배아 foregut - 비장 원기(splenic anlage) 융기 구조 / 일러스트용 베이스 메시 생성기.

과학적 정량 재현이 아니라 "그림 그리기 좋은 형태"가 목표라서,
- 각 기관의 상대 위치/방향은 발생학 교과서 수준으로 맞추고
- 크기와 융기 정도는 그림에서 읽히도록 살짝 과장했습니다.

좌표계(빌드 좌표):
    +X = 배아의 왼쪽(left)
    +Y = 머리쪽(cranial)
    +Z = 배쪽(ventral)        ->  -Z = 등쪽(dorsal)
    1 unit = 100 um  (위장 지름 약 2.3 unit = 230 um)

내보낼 때 루트 노드에 Y축 -90도 회전을 넣어서, 뷰어 기본 카메라
(camera-orbit="45deg 65deg")가 좌측-등쪽(비장이 융기하는 면)을 보도록 했습니다.
Blender에서 축을 원래대로 쓰고 싶으면 루트 노드 회전만 0으로 지우면 됩니다.

의존성 없음(표준 라이브러리만). 실행:
    python3 tools/foregut_model.py --out models/e105_foregut_spleen.glb --cell 0.13
"""

import argparse
import json
import math
import os
import struct
from array import array

# ----------------------------------------------------------------------------
# 튜닝 파라미터 - 일러스트 느낌은 대부분 여기만 만져도 바뀝니다.
# ----------------------------------------------------------------------------
PARAMS = {
    "cell": 0.13,            # 복셀 크기(작을수록 고해상도 / 느림)
    "smooth_passes": 3,      # 라플라시안 스무딩 횟수
    "smooth_factor": 0.55,

    "stomach_fat": 1.0,      # 위(stomach) 팽대 정도
    "gc_bulge": 1.0,         # 대만곡(greater curvature) 좌-등쪽 돌출

    "spleen_length": 1.0,    # 비장 원기 두상 융기의 길이 배율
    "spleen_thickness": 1.0, # 융기 두께 배율
    "spleen_offset": 0.20,   # 장간막 표면에서 왼쪽으로 얼마나 더 튀어나오는지

    "meso_thickness": 0.40,  # 등쪽위간막 두께
    "meso_bulge": 1.0,       # 장간막이 왼쪽으로 부풀어 오르는 정도

    "show_liver": True,
    "show_lungs": True,
    "show_aorta": True,
    "meso_alpha": 0.55,      # 등쪽위간막 투명도(비장 융기가 비쳐 보이게)
}

ISO = 0.377          # 단일 blob의 표면이 정확히 반지름 r에 오도록 맞춘 임계값
BLEND_P = 3          # soft-union 지수 (클수록 hard union)
ISO_P = ISO ** BLEND_P

# sRGB 색 (일러스트 팔레트)
COLORS = {
    "esophagus":  "#e3b49b",
    "stomach":    "#e0997a",
    "duodenum":   "#e3b49b",
    "lung":       "#8fc4e8",
    "pancreas":   "#f0c95f",
    "liver":      "#a9515c",
    "meso":       "#c6dcd2",
    "spleen":     "#8e63c6",
    "aorta":      "#9b3a44",
}


# ----------------------------------------------------------------------------
# 작은 벡터 헬퍼
# ----------------------------------------------------------------------------
def vadd(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def vsub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def vmul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def vdot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vnorm(a):
    m = math.sqrt(vdot(a, a))
    if m < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / m, a[1] / m, a[2] / m)


def lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def catmull(pts, t):
    """제어점 리스트를 지나는 Catmull-Rom 스플라인. t in [0,1]."""
    n = len(pts) - 1
    s = max(0.0, min(1.0, t)) * n
    i = min(int(s), n - 1)
    f = s - i
    p0 = pts[max(i - 1, 0)]
    p1 = pts[i]
    p2 = pts[i + 1]
    p3 = pts[min(i + 2, n)]
    f2, f3 = f * f, f * f * f
    out = []
    for c in range(3):
        out.append(0.5 * ((2 * p1[c]) +
                          (-p0[c] + p2[c]) * f +
                          (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * f2 +
                          (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * f3))
    return tuple(out)


def catmull1(vals, t):
    """스칼라(반지름 등) 보간."""
    pts = [(v, 0.0, 0.0) for v in vals]
    return catmull(pts, t)[0]


def hex2lin(h):
    """sRGB hex -> linear RGB (glTF의 factor/COLOR_0은 linear 공간)."""
    h = h.lstrip("#")
    out = []
    for i in range(3):
        c = int(h[i * 2:i * 2 + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


# ----------------------------------------------------------------------------
# 메타볼 프리미티브 (캡슐 = 선분 + 반지름, 선택적 이방성)
# ----------------------------------------------------------------------------
class Blob(object):
    __slots__ = ("ax", "ay", "az", "bx", "by", "bz", "r", "R", "col",
                 "nx", "ny", "nz", "k")

    def __init__(self, a, b, r, col, axis=None, k=1.0):
        self.ax, self.ay, self.az = a
        self.bx, self.by, self.bz = b
        self.r = r
        self.R = r * 1.9                 # 영향 반경
        self.col = col
        ax_ = axis or (0.0, 0.0, 1.0)
        self.nx, self.ny, self.nz = ax_
        self.k = k                       # axis 방향으로 눌러 납작하게(>1)

    def falloff(self, px, py, pz):
        dx = px - self.ax
        dy = py - self.ay
        dz = pz - self.az
        ex = self.bx - self.ax
        ey = self.by - self.ay
        ez = self.bz - self.az
        L2 = ex * ex + ey * ey + ez * ez
        if L2 > 1e-12:
            t = (dx * ex + dy * ey + dz * ez) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            dx -= ex * t
            dy -= ey * t
            dz -= ez * t
        if self.k != 1.0:
            s = (dx * self.nx + dy * self.ny + dz * self.nz) * (self.k - 1.0)
            dx += self.nx * s
            dy += self.ny * s
            dz += self.nz * s
        d2 = dx * dx + dy * dy + dz * dz
        R2 = self.R * self.R
        if d2 >= R2:
            return 0.0
        u = 1.0 - d2 / R2
        return u * u * u


def add_tube(blobs, pts, radii, col, samples=None, axis=None, k=1.0):
    """제어점 스플라인을 따라 캡슐 체인을 깔아준다."""
    length = 0.0
    for i in range(len(pts) - 1):
        length += math.sqrt(vdot(vsub(pts[i + 1], pts[i]), vsub(pts[i + 1], pts[i])))
    if samples is None:
        samples = max(4, int(length / 0.28))
    prev = catmull(pts, 0.0)
    prev_r = catmull1(radii, 0.0)
    for i in range(1, samples + 1):
        t = i / float(samples)
        cur = catmull(pts, t)
        cur_r = catmull1(radii, t)
        blobs.append(Blob(prev, cur, 0.5 * (prev_r + cur_r), col, axis, k))
        prev, prev_r = cur, cur_r


def add_ball(blobs, c, r, col, axis=None, k=1.0):
    blobs.append(Blob(c, c, r, col, axis, k))


# ----------------------------------------------------------------------------
# 복셀화 + Surface Nets (마칭큐브보다 코드가 짧고 결과가 매끈함)
# ----------------------------------------------------------------------------
def voxelize(blobs, cell):
    minx = min(min(b.ax, b.bx) - b.R for b in blobs) - cell * 2
    miny = min(min(b.ay, b.by) - b.R for b in blobs) - cell * 2
    minz = min(min(b.az, b.bz) - b.R for b in blobs) - cell * 2
    maxx = max(max(b.ax, b.bx) + b.R for b in blobs) + cell * 2
    maxy = max(max(b.ay, b.by) + b.R for b in blobs) + cell * 2
    maxz = max(max(b.az, b.bz) + b.R for b in blobs) + cell * 2

    nx = int((maxx - minx) / cell) + 2
    ny = int((maxy - miny) / cell) + 2
    nz = int((maxz - minz) / cell) + 2
    field = array('f', bytes(4 * nx * ny * nz))

    for b in blobs:
        R = b.R
        R2 = R * R
        ax, ay, az = b.ax, b.ay, b.az
        ex, ey, ez = b.bx - ax, b.by - ay, b.bz - az
        L2 = ex * ex + ey * ey + ez * ez
        kk = b.k
        bnx, bny, bnz = b.nx, b.ny, b.nz
        # 이방성으로 눌린 축 방향으로는 영향 반경이 줄지만, 넉넉히 R로 잡는다.
        i0 = max(0, int((min(ax, b.bx) - R - minx) / cell))
        i1 = min(nx - 1, int((max(ax, b.bx) + R - minx) / cell) + 1)
        j0 = max(0, int((min(ay, b.by) - R - miny) / cell))
        j1 = min(ny - 1, int((max(ay, b.by) + R - miny) / cell) + 1)
        k0 = max(0, int((min(az, b.bz) - R - minz) / cell))
        k1 = min(nz - 1, int((max(az, b.bz) + R - minz) / cell) + 1)

        for kz in range(k0, k1 + 1):
            pz = minz + kz * cell
            for jy in range(j0, j1 + 1):
                py = miny + jy * cell
                base = (kz * ny + jy) * nx
                for ix in range(i0, i1 + 1):
                    px = minx + ix * cell
                    dx = px - ax
                    dy = py - ay
                    dz = pz - az
                    if L2 > 1e-12:
                        t = (dx * ex + dy * ey + dz * ez) / L2
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        dx -= ex * t
                        dy -= ey * t
                        dz -= ez * t
                    if kk != 1.0:
                        s = (dx * bnx + dy * bny + dz * bnz) * (kk - 1.0)
                        dx += bnx * s
                        dy += bny * s
                        dz += bnz * s
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < R2:
                        u = 1.0 - d2 / R2
                        u3 = u * u * u          # falloff
                        field[base + ix] += u3 * u3 * u3   # soft-union (p=3)
    return field, (nx, ny, nz), (minx, miny, minz)


CORNER = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
          (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
EDGES = [(c, c | b) for b in (1, 2, 4) for c in range(8) if not (c & b)]


def surface_nets(field, dims, origin, cell):
    nx, ny, nz = dims
    ox, oy, oz = origin
    cellvert = {}
    verts = []
    cube = 1.0 / 3.0

    for kz in range(nz - 1):
        for jy in range(ny - 1):
            b00 = (kz * ny + jy) * nx
            b10 = (kz * ny + jy + 1) * nx
            b01 = ((kz + 1) * ny + jy) * nx
            b11 = ((kz + 1) * ny + jy + 1) * nx
            for ix in range(nx - 1):
                v = (field[b00 + ix], field[b00 + ix + 1],
                     field[b10 + ix], field[b10 + ix + 1],
                     field[b01 + ix], field[b01 + ix + 1],
                     field[b11 + ix], field[b11 + ix + 1])
                inside = 0
                for s in v:
                    if s >= ISO_P:
                        inside += 1
                if inside == 0 or inside == 8:
                    continue
                sx = sy = sz = 0.0
                n = 0
                for (ca, cb) in EDGES:
                    va, vb = v[ca], v[cb]
                    ia = va >= ISO_P
                    ib = vb >= ISO_P
                    if ia == ib:
                        continue
                    fa = va ** cube if va > 0.0 else 0.0
                    fb = vb ** cube if vb > 0.0 else 0.0
                    denom = (fb - fa)
                    t = 0.5 if abs(denom) < 1e-9 else (ISO - fa) / denom
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    pa, pb = CORNER[ca], CORNER[cb]
                    sx += pa[0] + (pb[0] - pa[0]) * t
                    sy += pa[1] + (pb[1] - pa[1]) * t
                    sz += pa[2] + (pb[2] - pa[2]) * t
                    n += 1
                if n == 0:
                    continue
                cellvert[(ix, jy, kz)] = len(verts)
                verts.append((ox + (ix + sx / n) * cell,
                              oy + (jy + sy / n) * cell,
                              oz + (kz + sz / n) * cell))

    tris = []
    get = cellvert.get

    def quad(a, b, c, d, flip):
        if flip:
            tris.append((a, c, b))
            tris.append((a, d, c))
        else:
            tris.append((a, b, c))
            tris.append((a, c, d))

    for kz in range(nz):
        for jy in range(ny):
            base = (kz * ny + jy) * nx
            for ix in range(nx):
                s0 = field[base + ix]
                in0 = s0 >= ISO_P
                # +X 방향 에지
                if ix + 1 < nx and jy >= 1 and kz >= 1:
                    if in0 != (field[base + ix + 1] >= ISO_P):
                        a = get((ix, jy, kz))
                        b = get((ix, jy - 1, kz))
                        c = get((ix, jy - 1, kz - 1))
                        d = get((ix, jy, kz - 1))
                        if None not in (a, b, c, d):
                            quad(a, b, c, d, not in0)
                # +Y 방향 에지
                if jy + 1 < ny and ix >= 1 and kz >= 1:
                    if in0 != (field[((kz * ny) + jy + 1) * nx + ix] >= ISO_P):
                        a = get((ix, jy, kz))
                        b = get((ix, jy, kz - 1))
                        c = get((ix - 1, jy, kz - 1))
                        d = get((ix - 1, jy, kz))
                        if None not in (a, b, c, d):
                            quad(a, b, c, d, not in0)
                # +Z 방향 에지
                if kz + 1 < nz and ix >= 1 and jy >= 1:
                    if in0 != (field[(((kz + 1) * ny) + jy) * nx + ix] >= ISO_P):
                        a = get((ix, jy, kz))
                        b = get((ix - 1, jy, kz))
                        c = get((ix - 1, jy - 1, kz))
                        d = get((ix, jy - 1, kz))
                        if None not in (a, b, c, d):
                            quad(a, b, c, d, not in0)
    return verts, tris


# ----------------------------------------------------------------------------
# 메시 후처리
# ----------------------------------------------------------------------------
def smooth(verts, tris, passes, factor):
    if passes <= 0:
        return verts
    adj = [set() for _ in verts]
    for (a, b, c) in tris:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    adj = [tuple(s) for s in adj]
    cur = list(verts)
    for _ in range(passes):
        nxt = []
        for i, p in enumerate(cur):
            nb = adj[i]
            if not nb:
                nxt.append(p)
                continue
            sx = sy = sz = 0.0
            for j in nb:
                q = cur[j]
                sx += q[0]; sy += q[1]; sz += q[2]
            m = float(len(nb))
            nxt.append((p[0] + (sx / m - p[0]) * factor,
                        p[1] + (sy / m - p[1]) * factor,
                        p[2] + (sz / m - p[2]) * factor))
        cur = nxt
    return cur


def signed_volume(verts, tris):
    vol = 0.0
    for (a, b, c) in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        vol += (pa[0] * (pb[1] * pc[2] - pb[2] * pc[1])
                - pa[1] * (pb[0] * pc[2] - pb[2] * pc[0])
                + pa[2] * (pb[0] * pc[1] - pb[1] * pc[0]))
    return vol / 6.0


def orient(verts, tris):
    """닫힌 메시의 법선이 바깥을 향하도록 전체 winding 정렬."""
    if signed_volume(verts, tris) < 0:
        return [(a, c, b) for (a, b, c) in tris]
    return tris


def vertex_normals(verts, tris):
    nrm = [[0.0, 0.0, 0.0] for _ in verts]
    for (a, b, c) in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        n = vcross(vsub(pb, pa), vsub(pc, pa))   # 면적 가중
        for i in (a, b, c):
            nrm[i][0] += n[0]; nrm[i][1] += n[1]; nrm[i][2] += n[2]
    return [vnorm(tuple(n)) if vdot(tuple(n), tuple(n)) > 1e-20 else (0.0, 1.0, 0.0)
            for n in nrm]


def vertex_colors(verts, blobs):
    """각 blob의 기여도로 색을 섞어서 기관 경계가 부드럽게 넘어가게."""
    cache = {}
    for b in blobs:
        if b.col not in cache:
            cache[b.col] = hex2lin(b.col)
    out = []
    for (px, py, pz) in verts:
        r = g = bl = w = 0.0
        for b in blobs:
            f = b.falloff(px, py, pz)
            if f <= 0.0:
                continue
            f = f * f
            c = cache[b.col]
            r += c[0] * f; g += c[1] * f; bl += c[2] * f
            w += f
        if w <= 0.0:
            out.append((0.8, 0.8, 0.8))
        else:
            out.append((r / w, g / w, bl / w))
    return out


def build_blob_mesh(blobs, cell, name, alpha=1.0, double_sided=False,
                    passes=None, factor=None):
    field, dims, origin = voxelize(blobs, cell)
    verts, tris = surface_nets(field, dims, origin, cell)
    tris = orient(verts, tris)
    verts = smooth(verts, tris,
                   PARAMS["smooth_passes"] if passes is None else passes,
                   PARAMS["smooth_factor"] if factor is None else factor)
    return {
        "name": name,
        "verts": verts,
        "tris": tris,
        "normals": vertex_normals(verts, tris),
        "colors": vertex_colors(verts, blobs),
        "alpha": alpha,
        "double_sided": double_sided,
    }


# ----------------------------------------------------------------------------
# 해부: E10.5 foregut 계통
#   인두-식도 -> 기관/폐아 -> 위(방추형 팽대) -> 십이지장
#   + 등쪽/배쪽 췌장아, 간아, 등쪽위간막, 그 안의 비장 원기
# ----------------------------------------------------------------------------
def foregut_blobs():
    P = PARAMS
    B = []
    ce = COLORS["esophagus"]
    cs = COLORS["stomach"]
    cd = COLORS["duodenum"]
    cl = COLORS["lung"]
    cp = COLORS["pancreas"]

    # 식도 / 인두쪽 foregut (등쪽으로 살짝 치우침)
    add_tube(B, [(0.0, 10.2, -0.35), (0.0, 8.2, -0.35),
                 (0.0, 6.0, -0.25), (0.0, 4.6, -0.05), (0.0, 3.8, 0.0)],
             [0.52, 0.48, 0.45, 0.47, 0.52], ce)

    if P["show_lungs"]:
        # 기관: 식도 배쪽으로 분리되어 내려감
        add_tube(B, [(0.0, 7.7, -0.3), (0.0, 7.0, 0.35),
                     (0.0, 6.2, 0.85), (0.0, 5.6, 1.05)],
                 [0.40, 0.40, 0.40, 0.38], cl)
        # 좌 주기관지 + 폐아 (좌폐는 단엽)
        add_tube(B, [(0.0, 5.6, 1.05), (0.75, 5.15, 1.2), (1.45, 4.6, 1.3)],
                 [0.36, 0.34, 0.34], cl)
        add_ball(B, (1.68, 4.35, 1.32), 0.46, cl)
        # 우 주기관지 + 폐아 (우폐는 다엽 -> 곁싹 하나 더)
        add_tube(B, [(0.0, 5.6, 1.05), (-0.8, 5.2, 1.2), (-1.55, 4.75, 1.3)],
                 [0.36, 0.35, 0.34], cl)
        add_ball(B, (-1.85, 4.5, 1.35), 0.46, cl)
        add_ball(B, (-1.75, 5.25, 1.0), 0.40, cl)

    # 위: 방추형 팽대. 대만곡이 좌-등쪽을 향한다.
    fat = P["stomach_fat"]
    add_tube(B, [(0.0, 3.8, 0.0), (0.15, 3.0, -0.10), (0.30, 2.0, -0.20),
                 (0.30, 1.0, -0.15), (0.15, 0.2, 0.02), (0.0, -0.45, 0.18)],
             [0.55, 0.86 * fat, 1.14 * fat, 1.12 * fat, 0.84 * fat, 0.58], cs)
    # 대만곡 능선(좌-등쪽)을 만들어 주는 덧살 - 여기에 등쪽위간막이 붙는다.
    gb = P["gc_bulge"]
    add_ball(B, (0.80 * gb, 2.65, -0.62 * gb), 0.70, cs)
    add_ball(B, (0.95 * gb, 1.90, -0.78 * gb), 0.76, cs)
    add_ball(B, (0.92 * gb, 1.15, -0.76 * gb), 0.72, cs)
    add_ball(B, (0.70 * gb, 0.45, -0.58 * gb), 0.60, cs)

    # 십이지장: 배-오른쪽으로 돌아 내려감
    add_tube(B, [(0.0, -0.45, 0.18), (-0.30, -1.25, 0.50), (-0.52, -2.2, 0.80),
                 (-0.30, -3.4, 0.62), (0.05, -4.5, 0.18)],
             [0.58, 0.50, 0.47, 0.46, 0.44], cd)

    # 등쪽 췌장아: 십이지장 등쪽에서 등-좌측으로 돌출
    add_tube(B, [(0.10, -0.95, 0.10), (0.22, -1.15, -0.60), (0.34, -1.30, -1.30)],
             [0.34, 0.32, 0.32], cp)
    add_ball(B, (0.38, -1.36, -1.55), 0.42, cp)
    # 배쪽 췌장아: 간아 옆 배쪽으로 작게
    add_tube(B, [(-0.34, -1.45, 0.55), (-0.55, -1.68, 1.05)], [0.26, 0.26], cp)
    add_ball(B, (-0.60, -1.78, 1.20), 0.32, cp)
    return B


def liver_blobs():
    c = COLORS["liver"]
    B = []
    add_tube(B, [(-0.42, -1.45, 0.70), (-0.55, -1.95, 1.35), (-0.60, -2.3, 1.8)],
             [0.40, 0.44, 0.5], c)
    add_ball(B, (0.05, -2.65, 2.55), 1.45, c)
    add_ball(B, (-1.55, -2.45, 2.25), 1.25, c)
    add_ball(B, (1.55, -2.50, 2.20), 1.20, c)
    add_ball(B, (0.0, -3.85, 2.45), 1.30, c)
    add_ball(B, (-0.95, -3.45, 3.05), 0.95, c)
    add_ball(B, (0.95, -3.40, 3.00), 0.95, c)
    return B


def aorta_blobs():
    c = COLORS["aorta"]
    B = []
    # 등쪽대동맥: 두측은 좌우 한 쌍, 미측은 정중선에서 융합
    add_tube(B, [(0.62, 6.5, -4.20), (0.60, 4.5, -4.25), (0.45, 2.6, -4.30),
                 (0.20, 1.4, -4.30), (0.0, 0.7, -4.30)],
             [0.23, 0.25, 0.25, 0.25, 0.26], c)
    add_tube(B, [(-0.62, 6.5, -4.20), (-0.60, 4.5, -4.25), (-0.45, 2.6, -4.30),
                 (-0.20, 1.4, -4.30), (0.0, 0.7, -4.30)],
             [0.23, 0.25, 0.25, 0.25, 0.26], c)
    add_tube(B, [(0.0, 0.9, -4.30), (0.0, -1.5, -4.28), (0.0, -4.0, -4.20)],
             [0.28, 0.28, 0.26], c)
    return B


# --- 등쪽위간막(dorsal mesogastrium) : 얇은 시트 ---------------------------
# u : 두측(0) -> 미측(1),  v : 위 대만곡(0) -> 등쪽 체벽/대동맥(1)
MESO_GUT = [(0.05, 4.60, -0.45), (0.35, 3.70, -0.62), (0.90, 2.85, -0.86),
            (1.20, 1.90, -1.02), (1.10, 0.95, -0.94), (0.60, 0.10, -0.66),
            (0.15, -0.85, -0.36), (-0.05, -2.00, -0.18), (-0.10, -3.20, -0.28)]
MESO_ROOT = [(0.10, 4.40, -3.30), (0.12, 3.60, -3.40), (0.15, 2.80, -3.50),
             (0.15, 1.90, -3.56), (0.15, 1.00, -3.56), (0.12, 0.10, -3.50),
             (0.10, -0.85, -3.44), (0.05, -2.00, -3.38), (0.00, -3.20, -3.28)]

# 위(stomach) 높이(u~0.35)에서 간막이 가장 크게 좌측으로 부풀고, 그 안에서 비장이 자란다.
U_SPLEEN = 0.35


def _gauss(u, mu, sg):
    return math.exp(-((u - mu) ** 2) / (2.0 * sg * sg))


def meso_span(u):
    """두미 방향 위치에 따른 간막의 '두께 배율'(양 끝은 얇게)."""
    return 0.45 + 0.55 * math.sin(math.pi * min(1.0, max(0.0, u))) ** 0.5


def meso_depth(u):
    """간막의 깊이(위->등쪽 체벽). 식도/십이지장 쪽은 짧고 위(stomach)에서 가장 깊다."""
    return 0.30 + 0.70 * math.sin(math.pi * min(1.0, max(0.0, u))) ** 0.45


def meso_point(u, v):
    a = catmull(MESO_GUT, u)
    b = lerp3(a, catmull(MESO_ROOT, u), meso_depth(u))
    p = lerp3(a, b, v)
    # 좌측으로 부풀리기(비장이 자라나는 쪽). 두미방향으로 가운데가 가장 크게.
    amp = PARAMS["meso_bulge"] * (0.10 + 0.62 * _gauss(u, U_SPLEEN, 0.24))
    bulge = amp * math.sin(math.pi * max(0.0, min(1.0, v)) ** 0.85)
    sag = 0.30 * math.sin(math.pi * max(0.0, min(1.0, v))) * (0.25 + 0.75 * u)
    return (p[0] + bulge, p[1], p[2] + sag)


def meso_normal(u, v, h=1e-3):
    du = vsub(meso_point(min(1.0, u + h), v), meso_point(max(0.0, u - h), v))
    dv = vsub(meso_point(u, min(1.0, v + h)), meso_point(u, max(0.0, v - h)))
    n = vnorm(vcross(dv, du))
    return n if n[0] >= 0.0 else vmul(n, -1.0)   # 항상 배아의 왼쪽을 향하게


def mesogastrium_mesh():
    """양면 오프셋한 얇은 판. u,v 격자 -> 위/아래면 + 테두리."""
    nu, nv = 72, 30
    t = PARAMS["meso_thickness"] * 0.5
    col = hex2lin(COLORS["meso"])
    top, bot = [], []
    for i in range(nu + 1):
        u = i / float(nu)
        for j in range(nv + 1):
            # v를 -0.06까지 넣어 위(stomach) 안쪽에 살짝 묻히게, 1.02까지 체벽쪽으로
            v = -0.06 + (1.08) * j / float(nv)
            p = meso_point(u, v)
            n = meso_normal(u, max(0.0, min(1.0, v)))
            # 가장자리로 갈수록 얇아지게(자연스러운 막 느낌)
            taper = math.sin(math.pi * min(1.0, max(0.02, (v + 0.06) / 1.08))) ** 0.30
            tt = t * (0.30 + 0.70 * taper) * meso_span(u)
            top.append(vadd(p, vmul(n, tt)))
            bot.append(vsub(p, vmul(n, tt)))

    stride = nv + 1
    verts = top + bot
    off = len(top)
    tris = []
    for i in range(nu):
        for j in range(nv):
            a = i * stride + j
            b = (i + 1) * stride + j
            c = (i + 1) * stride + j + 1
            d = i * stride + j + 1
            tris.append((a, b, c)); tris.append((a, c, d))
            a2, b2, c2, d2 = a + off, b + off, c + off, d + off
            tris.append((a2, c2, b2)); tris.append((a2, d2, c2))
    # 테두리 봉합
    def rim(pairs):
        for (p, q) in pairs:
            tris.append((p, q, q + off))
            tris.append((p, q + off, p + off))
    edge = []
    for i in range(nu):
        edge.append((i * stride + nv, (i + 1) * stride + nv))          # 체벽쪽
        edge.append(((i + 1) * stride, i * stride))                    # 위쪽
    for j in range(nv):
        edge.append((j, j + 1))                                        # 두측
        edge.append((nu * stride + j + 1, nu * stride + j))            # 미측
    rim(edge)
    tris = orient(verts, tris)
    return {
        "name": "DorsalMesogastrium",
        "verts": verts,
        "tris": tris,
        "normals": vertex_normals(verts, tris),
        "colors": [col] * len(verts),
        "alpha": PARAMS["meso_alpha"],
        "double_sided": True,
    }


def spleen_blobs():
    """비장 원기: 등쪽위간막의 '왼쪽 면'에 방추형 능선으로 융기."""
    P = PARAMS
    c = COLORS["spleen"]
    B = []
    u0 = U_SPLEEN - 0.155 * P["spleen_length"]
    u1 = U_SPLEEN + 0.175 * P["spleen_length"]
    v_s = 0.26
    n_ref = meso_normal(0.5, v_s)
    steps = 16
    prev = None
    for i in range(steps + 1):
        s = i / float(steps)
        u = u0 + (u1 - u0) * s
        n = meso_normal(u, v_s)
        p = vadd(meso_point(u, v_s), vmul(n, P["spleen_offset"]))
        # 방추형: 가운데가 두껍고 양 끝이 뾰족
        r = (0.16 + 0.40 * math.sin(math.pi * (s ** 1.25)) ** 0.7) * P["spleen_thickness"]
        if prev is not None:
            B.append(Blob(prev[0], p, 0.5 * (prev[1] + r), c, n_ref, 1.45))
        prev = (p, r)
    return B


# ----------------------------------------------------------------------------
# GLB 출력
# ----------------------------------------------------------------------------
def write_glb(path, meshes, root_rotation=None, root_name="E105_foregut_spleen"):
    bin_parts = []
    offset = [0]
    bufferviews = []
    accessors = []

    def add_view(data, target):
        while offset[0] % 4:
            bin_parts.append(b"\x00")
            offset[0] += 1
        bufferviews.append({"buffer": 0, "byteOffset": offset[0],
                            "byteLength": len(data), "target": target})
        bin_parts.append(data)
        offset[0] += len(data)
        return len(bufferviews) - 1

    gmeshes = []
    materials = []
    nodes = []
    for m in meshes:
        verts, tris, nrm, cols = m["verts"], m["tris"], m["normals"], m["colors"]
        pos = bytearray()
        nor = bytearray()
        col = bytearray()
        mn = [1e30] * 3
        mx = [-1e30] * 3
        for i, p in enumerate(verts):
            pos += struct.pack("<3f", *p)
            nor += struct.pack("<3f", *nrm[i])
            c = cols[i]
            col += struct.pack("<4f", c[0], c[1], c[2], 1.0)
            for k in range(3):
                if p[k] < mn[k]: mn[k] = p[k]
                if p[k] > mx[k]: mx[k] = p[k]
        idx = bytearray()
        for t in tris:
            idx += struct.pack("<3I", *t)

        vpos = add_view(bytes(pos), 34962)
        vnor = add_view(bytes(nor), 34962)
        vcol = add_view(bytes(col), 34962)
        vidx = add_view(bytes(idx), 34963)
        n = len(verts)
        a_pos = len(accessors)
        accessors.append({"bufferView": vpos, "componentType": 5126, "count": n,
                          "type": "VEC3", "min": mn, "max": mx})
        a_nor = len(accessors)
        accessors.append({"bufferView": vnor, "componentType": 5126, "count": n,
                          "type": "VEC3"})
        a_col = len(accessors)
        accessors.append({"bufferView": vcol, "componentType": 5126, "count": n,
                          "type": "VEC4"})
        a_idx = len(accessors)
        accessors.append({"bufferView": vidx, "componentType": 5125,
                          "count": len(tris) * 3, "type": "SCALAR"})

        mat = {
            "name": m["name"] + "_mat",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, m.get("alpha", 1.0)],
                "metallicFactor": 0.0,
                "roughnessFactor": m.get("roughness", 0.62),
            },
            "doubleSided": bool(m.get("double_sided", False)),
        }
        if m.get("alpha", 1.0) < 1.0:
            mat["alphaMode"] = "BLEND"
        materials.append(mat)

        gmeshes.append({"name": m["name"], "primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nor, "COLOR_0": a_col},
            "indices": a_idx, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": m["name"], "mesh": len(gmeshes) - 1})

    root = {"name": root_name, "children": list(range(len(nodes)))}
    if root_rotation:
        root["rotation"] = list(root_rotation)
    nodes.insert(0, root)
    for nd in nodes[1:]:
        pass
    root["children"] = list(range(1, len(nodes)))

    binary = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0",
                  "generator": "3DViewer foregut_model.py (E10.5 illustration base)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": gmeshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": bufferviews,
        "buffers": [{"byteLength": len(binary)}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    bn = binary
    while len(bn) % 4:
        bn += b"\x00"
    total = 12 + 8 + len(js) + 8 + len(bn)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(bn), 0x004E4942))
        f.write(bn)
    return total


# ----------------------------------------------------------------------------
def build_all(cell=None, verbose=True):
    cell = cell or PARAMS["cell"]
    meshes = []

    def log(*a):
        if verbose:
            print(*a, flush=True)

    log("  foregut ...")
    meshes.append(build_blob_mesh(foregut_blobs(), cell, "Foregut_Stomach_Duodenum"))
    if PARAMS["show_liver"]:
        log("  liver bud ...")
        meshes.append(build_blob_mesh(liver_blobs(), cell * 1.25, "LiverBud"))
    log("  dorsal mesogastrium ...")
    meshes.append(mesogastrium_mesh())
    log("  splenic anlage ...")
    meshes.append(build_blob_mesh(spleen_blobs(), cell * 0.85,
                                  "SpleenAnlage", passes=2))
    if PARAMS["show_aorta"]:
        log("  dorsal aorta (ref) ...")
        meshes.append(build_blob_mesh(aorta_blobs(), cell * 1.3, "DorsalAorta_ref"))
    return meshes


ROOT_ROT_Y_MINUS90 = (0.0, -0.7071067811865476, 0.0, 0.7071067811865476)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/e105_foregut_spleen.glb")
    ap.add_argument("--cell", type=float, default=PARAMS["cell"])
    ap.add_argument("--no-view-rotation", action="store_true",
                    help="뷰어용 루트 회전 없이 해부 좌표계 그대로 출력")
    args = ap.parse_args()

    PARAMS["cell"] = args.cell
    print("building (cell=%.3f) ..." % args.cell)
    meshes = build_all(args.cell)
    out = args.out
    d = os.path.dirname(out)
    if d:
        try:
            os.makedirs(d)
        except OSError:
            pass
    size = write_glb(out, meshes,
                     None if args.no_view_rotation else ROOT_ROT_Y_MINUS90)
    tv = sum(len(m["verts"]) for m in meshes)
    tt = sum(len(m["tris"]) for m in meshes)
    for m in meshes:
        print("  %-28s %6d verts %7d tris" % (m["name"], len(m["verts"]), len(m["tris"])))
    print("total: %d verts / %d tris -> %s (%.2f MB)" % (tv, tt, out, size / 1048576.0))


if __name__ == "__main__":
    main()
