#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
foregut_model.py 결과를 브라우저 없이 바로 확인하기 위한 초간단 소프트웨어 렌더러.
(z-buffer + 램버트 + 반투명 2패스, 표준 라이브러리만 사용)

    python3 tools/preview_render.py --cell 0.15 --out docs
"""
import argparse
import math
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foregut_model as fm


def write_png(path, w, h, buf):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += buf[y * w * 3:(y + 1) * w * 3]
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def lin2srgb(c):
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    s = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return int(s * 255 + 0.5)


def render(meshes, direction, size=760, fov_deg=26.0, bg=(0.055, 0.06, 0.075)):
    W = H = size
    pix = [bg[0]] * (W * H * 3)
    for i in range(W * H):
        pix[i * 3] = bg[0]; pix[i * 3 + 1] = bg[1]; pix[i * 3 + 2] = bg[2]
    zbuf = [1e30] * (W * H)

    allv = [p for m in meshes for p in m["verts"]]
    cx = sum(p[0] for p in allv) / len(allv)
    cy = sum(p[1] for p in allv) / len(allv)
    cz = sum(p[2] for p in allv) / len(allv)
    rad = max(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)
              for p in allv)
    fov = math.radians(fov_deg)
    dist = rad / math.sin(fov * 0.5) * 0.95

    d = fm.vnorm(direction)
    eye = (cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist)
    fwd = fm.vnorm((cx - eye[0], cy - eye[1], cz - eye[2]))
    right = fm.vnorm(fm.vcross(fwd, (0.0, 1.0, 0.0)))
    up = fm.vcross(right, fwd)
    f = 1.0 / math.tan(fov * 0.5)


    # 라이트는 카메라 기준(좌상단 키 + 우측 필)이라 모든 시점에서 형태가 읽힌다.
    L1 = fm.vnorm(tuple(-fwd[i] * 0.55 + up[i] * 0.72 - right[i] * 0.45 for i in range(3)))
    L2 = fm.vnorm(tuple(-fwd[i] * 0.35 + right[i] * 0.85 - up[i] * 0.25 for i in range(3)))

    def project(p):
        rx = p[0] - eye[0]; ry = p[1] - eye[1]; rz = p[2] - eye[2]
        vz = rx * fwd[0] + ry * fwd[1] + rz * fwd[2]
        if vz <= 0.02:
            return None
        vx = rx * right[0] + ry * right[1] + rz * right[2]
        vy = rx * up[0] + ry * up[1] + rz * up[2]
        return (W * 0.5 + (vx * f / vz) * W * 0.5,
                H * 0.5 - (vy * f / vz) * H * 0.5, vz)

    def shade(n, col, view_dir):
        d1 = n[0] * L1[0] + n[1] * L1[1] + n[2] * L1[2]
        d2 = n[0] * L2[0] + n[1] * L2[1] + n[2] * L2[2]
        d1 = d1 if d1 > 0 else 0.0
        d2 = d2 if d2 > 0 else 0.0
        rim = 1.0 - abs(n[0] * view_dir[0] + n[1] * view_dir[1] + n[2] * view_dir[2])
        k = 0.20 + 0.86 * d1 + 0.26 * d2 + 0.16 * rim ** 3
        return (col[0] * k, col[1] * k, col[2] * k)

    def draw(mesh, alpha, write_z):
        verts = mesh["verts"]; nrm = mesh["normals"]; cols = mesh["colors"]
        proj = [project(p) for p in verts]
        shaded = [shade(nrm[i], cols[i], fwd) for i in range(len(verts))]
        for (ia, ib, ic) in mesh["tris"]:
            pa, pb, pc = proj[ia], proj[ib], proj[ic]
            if pa is None or pb is None or pc is None:
                continue
            area = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if area == 0.0:
                continue
            back = area > 0.0            # 화면 좌표 y가 뒤집혀 있으므로 부호 반전
            if back and alpha >= 1.0:
                continue
            x0 = max(0, int(min(pa[0], pb[0], pc[0])))
            x1 = min(W - 1, int(max(pa[0], pb[0], pc[0])) + 1)
            y0 = max(0, int(min(pa[1], pb[1], pc[1])))
            y1 = min(H - 1, int(max(pa[1], pb[1], pc[1])) + 1)
            if x1 < x0 or y1 < y0:
                continue
            ca, cb, cc = shaded[ia], shaded[ib], shaded[ic]
            inv = 1.0 / area
            for y in range(y0, y1 + 1):
                py = y + 0.5
                row = y * W
                for x in range(x0, x1 + 1):
                    px = x + 0.5
                    w0 = ((pb[0] - pa[0]) * (py - pa[1]) - (pb[1] - pa[1]) * (px - pa[0])) * inv
                    w1 = ((pc[0] - pb[0]) * (py - pb[1]) - (pc[1] - pb[1]) * (px - pb[0])) * inv
                    w2 = ((pa[0] - pc[0]) * (py - pc[1]) - (pa[1] - pc[1]) * (px - pc[0])) * inv
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        if not (w0 <= 0 and w1 <= 0 and w2 <= 0):
                            continue
                    la, lb, lc = w1, w2, w0
                    z = la * pa[2] + lb * pb[2] + lc * pc[2]
                    idx = row + x
                    if z >= zbuf[idx]:
                        continue
                    r = la * ca[0] + lb * cb[0] + lc * cc[0]
                    g = la * ca[1] + lb * cb[1] + lc * cc[1]
                    b = la * ca[2] + lb * cb[2] + lc * cc[2]
                    o = idx * 3
                    if alpha >= 1.0:
                        pix[o] = r; pix[o + 1] = g; pix[o + 2] = b
                        zbuf[idx] = z
                    else:
                        a = alpha * (0.75 if back else 1.0)
                        pix[o] = pix[o] * (1 - a) + r * a
                        pix[o + 1] = pix[o + 1] * (1 - a) + g * a
                        pix[o + 2] = pix[o + 2] * (1 - a) + b * a
                        if write_z:
                            zbuf[idx] = z

    opaque = [m for m in meshes if m.get("alpha", 1.0) >= 1.0]
    trans = [m for m in meshes if m.get("alpha", 1.0) < 1.0]
    for m in opaque:
        draw(m, 1.0, True)
    for m in trans:
        draw(m, m.get("alpha", 1.0), False)

    buf = bytearray(W * H * 3)
    for i in range(W * H * 3):
        buf[i] = lin2srgb(pix[i])
    return W, H, buf


VIEWS = {
    # 빌드 좌표계: +X=left, +Y=cranial, +Z=ventral
    "1_left_dorsal": (0.80, 0.32, -0.72),   # 뷰어 기본 시점(비장 융기가 정면)
    "2_left":        (1.00, 0.18, 0.05),
    "3_dorsal":      (0.10, 0.22, -1.00),
    "4_ventral":     (0.35, 0.20, 1.00),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=0.15)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--size", type=int, default=760)
    ap.add_argument("--views", default="")
    args = ap.parse_args()

    fm.PARAMS["cell"] = args.cell
    meshes = fm.build_all(args.cell)
    try:
        os.makedirs(args.out)
    except OSError:
        pass
    wanted = args.views.split(",") if args.views else list(VIEWS)
    for name in wanted:
        w, h, buf = render(meshes, VIEWS[name], size=args.size)
        p = os.path.join(args.out, "preview_%s.png" % name)
        write_png(p, w, h, buf)
        print("wrote", p)


if __name__ == "__main__":
    main()
