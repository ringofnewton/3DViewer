"""Periodic (Lees-Edwards) shear rheology of the fiber network.

The clamped-boundary `rheology_test.py` leaves a residual force imbalance for a
sub-isostatic network (open side walls), so its modulus is only order-of-
magnitude. This module removes the wall artifact entirely with a fully periodic
box and the Lees-Edwards convention for simple shear, and reads the modulus from
the *virial* stress (an exact, boundary-free stress estimator).

Lees-Edwards: the box is periodic in x,y,z; a y-image is additionally offset by
gamma*L in x. The minimum-image bond vector therefore picks up the shear, so the
imposed strain is felt by every bond, not just the ones near a plate.

Virial shear stress (athermal, at force balance):
    sigma_xy = (1/V) * sum_bonds  f_x * r_y
where f = tension * dir is the force the bond exerts and r is its min-image
vector. G = sigma_xy / gamma in the linear regime; the prestress-stiffening
G(gamma) curve is the nonlinear hallmark of collagen (Storm 2005, Licup 2015).

Constitutive law is shared with `ecm_mechanics` (`_axial_stiffness`) so the
periodic modulus is comparable to the rest of the pipeline.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import ecm_mechanics as mech


@dataclass
class PeriodicNetwork:
    nodes: np.ndarray        # (N,3) wrapped into [0,L)^3
    edges: np.ndarray        # (E,2) int
    edge_off: np.ndarray     # (E,3) int image offset of node j relative to node i
    rest_length: np.ndarray  # (E,)
    bends: np.ndarray        # (B,3) prev,center,next  (same-image backbone only)
    k_scale: np.ndarray      # (E,)
    L: float
    n_backbone: int = 0      # edges[:n_backbone] are fiber backbone, rest are crosslinks

    def copy(self):
        return PeriodicNetwork(self.nodes.copy(), self.edges.copy(),
                               self.edge_off.copy(), self.rest_length.copy(),
                               self.bends.copy(), self.k_scale.copy(), self.L,
                               self.n_backbone)

    def is_crosslink(self):
        m = np.zeros(len(self.edges), bool)
        m[self.n_backbone:] = True
        return m


def _le_min_image(d, L, gamma):
    """Lees-Edwards minimum image of displacement(s) d (..,3) in box L, shear gamma.

    Order matters: resolve y first (it carries the x shear offset), then x, z.
    """
    d = d.copy()
    ny = np.round(d[..., 1] / L)
    d[..., 1] -= ny * L
    d[..., 0] -= ny * gamma * L
    nx = np.round(d[..., 0] / L)
    d[..., 0] -= nx * L
    nz = np.round(d[..., 2] / L)
    d[..., 2] -= nz * L
    return d


def _le_wrap(nodes, L, gamma):
    """Lees-Edwards-consistent wrap into the box.

    A y-image crossing must carry the shear offset gamma*L in x (plain np.mod does
    not), otherwise wrapping a node injects a spurious ~gamma*L bond stretch — the
    forces are *not* invariant under naive wrapping at finite shear.
    """
    n = nodes.copy()
    ny = np.floor(n[:, 1] / L)
    n[:, 1] -= ny * L
    n[:, 0] -= ny * gamma * L
    n[:, 0] = np.mod(n[:, 0], L)
    n[:, 2] = np.mod(n[:, 2], L)
    return n


def build_periodic_network(L=120.0, n_fibers=300, fiber_len=70.0, seg_len=6.0,
                           crosslink_dist=4.5, seed=0):
    """Fully periodic 3D Mikado network. Bonds store a fixed integer image offset
    so topology is stable under shear; crosslinks use the minimum image."""
    rng = np.random.default_rng(seed)
    nodes, edges, bends = [], [], []

    for _ in range(n_fibers):
        c = rng.uniform(0, L, size=3)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        n_seg = max(2, int(fiber_len / seg_len))
        start = c - 0.5 * fiber_len * d
        pts = np.array([start + d * seg_len * i for i in range(n_seg + 1)])
        base = len(nodes)
        nodes.extend(pts.tolist())
        for i in range(len(pts) - 1):
            edges.append((base + i, base + i + 1))
        for i in range(1, len(pts) - 1):
            bends.append((base + i - 1, base + i, base + i + 1))

    nodes = np.array(nodes)
    edges = list(edges)

    # backbone bond image offsets from the UNWRAPPED coords, then wrap nodes
    off = []
    for (a, b) in edges:
        off.append(np.round((nodes[b] - nodes[a]) / L).astype(int))
    nodes = np.mod(nodes, L)

    n_backbone = len(edges)
    # crosslinks under the (unsheared) minimum image
    cell = crosslink_dist
    buckets: dict[tuple, list[int]] = {}
    keys = np.floor(nodes / cell).astype(int)
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(tuple(k), []).append(i)
    added = set()
    n_xkey = int(np.ceil(L / cell))
    for k, idxs in buckets.items():
        cand = []
        for dk in itertools.product((-1, 0, 1), repeat=3):
            nb = ((k[0] + dk[0]) % n_xkey, (k[1] + dk[1]) % n_xkey, (k[2] + dk[2]) % n_xkey)
            cand += buckets.get(nb, [])
        for a in idxs:
            for b in cand:
                if b == a:
                    continue
                key = (min(a, b), max(a, b))
                if key in added:
                    continue
                dd = _le_min_image((nodes[b] - nodes[a])[None, :], L, 0.0)[0]
                if dd @ dd < cell * cell:
                    edges.append((a, b))
                    off.append(np.round((nodes[b] - nodes[a] - dd) / L).astype(int))
                    added.add(key)

    edges = np.array(edges)
    off = np.array(off)
    base_disp = _le_min_image(nodes[edges[:, 1]] + off * L - nodes[edges[:, 0]], L, 0.0)
    # floor crosslink rest length to avoid stiffness blow-ups from near-coincident
    # node pairs (tiny rest -> huge strain -> FIRE instability)
    rest = np.maximum(np.linalg.norm(base_disp, axis=1), 1e-3)
    rest[n_backbone:] = np.maximum(rest[n_backbone:], 1.0)
    bends = np.array(bends) if bends else np.zeros((0, 3), int)
    return PeriodicNetwork(nodes, edges, off, rest, bends,
                           np.ones(len(edges)), L, n_backbone)


def _forces(net: PeriodicNetwork, gamma, p: mech.MechParams):
    """Periodic nodal forces + per-bond tension + min-image bond vectors."""
    nodes, L = net.nodes, net.L
    N = len(nodes)
    F = np.zeros((N, 3))
    Kdiag = np.full(N, 1e-6)
    i, j = net.edges[:, 0], net.edges[:, 1]

    raw = nodes[j] + net.edge_off * L - nodes[i]
    d = _le_min_image(raw, L, gamma)
    Lij = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    dirv = d / Lij[:, None]
    strain = (Lij - net.rest_length) / net.rest_length
    k = mech._axial_stiffness(strain, p) * net.k_scale
    tension = k * (Lij - net.rest_length)
    fvec = tension[:, None] * dirv
    np.add.at(F, i, fvec)
    np.add.at(F, j, -fvec)
    np.add.at(Kdiag, i, k)
    np.add.at(Kdiag, j, k)

    if len(net.bends):
        a, c, b = net.bends[:, 0], net.bends[:, 1], net.bends[:, 2]
        da = _le_min_image(nodes[a] - nodes[c], L, gamma)
        db = _le_min_image(nodes[b] - nodes[c], L, gamma)
        lap = da + db
        np.add.at(F, c, p.k_bend * lap)
        np.add.at(F, a, -0.5 * p.k_bend * lap)
        np.add.at(F, b, -0.5 * p.k_bend * lap)
        np.add.at(Kdiag, c, 2.0 * p.k_bend)
        np.add.at(Kdiag, a, 0.5 * p.k_bend)
        np.add.at(Kdiag, b, 0.5 * p.k_bend)

    return F, tension, dirv, Lij, Kdiag


def relax(net: PeriodicNetwork, gamma, p: mech.MechParams | None = None):
    """FIRE relaxation of a periodic network at fixed shear gamma (all nodes free)."""
    p = p or mech.MechParams()
    net = net.copy()
    v = np.zeros_like(net.nodes)
    dt, alpha, n_pos = p.fire_dt, p.fire_a0, 0
    fmax = np.inf
    it = 0
    for it in range(p.max_iter):
        F, tension, dirv, Lij, Kdiag = _forces(net, gamma, p)
        acc = F / Kdiag[:, None]
        power = float(np.sum(F * v))
        vnorm, anorm = np.linalg.norm(v), np.linalg.norm(acc)
        if anorm > 1e-12:
            v = (1 - alpha) * v + alpha * vnorm * (acc / anorm)
        if power > 0:
            n_pos += 1
            if n_pos > p.fire_nmin:
                dt = min(dt * 1.1, p.fire_dtmax); alpha *= 0.99
        else:
            n_pos = 0; v[:] = 0.0; dt = max(dt * 0.5, 1e-3); alpha = p.fire_a0
        v = v + dt * acc
        disp = dt * v
        mag = np.linalg.norm(disp, axis=1)
        big = mag > p.max_step
        if np.any(big):
            disp[big] *= (p.max_step / mag[big])[:, None]
        net.nodes = np.mod(net.nodes + disp, net.L)
        fmax = float(np.max(np.linalg.norm(F, axis=1)))
        if fmax < p.ftol:
            break
    return net, dict(iters=it + 1, converged=fmax < p.ftol, final_max_force=fmax)


# --------------------------------------------------------------------------- #
# Floppy-mode-aware solver: trust-region Newton (Steihaug truncated-CG)
#
# Sub-isostatic fiber networks have *floppy modes* (zero-stiffness collective
# motions) plus, under shear, strongly stiffening tension modes. That stiffness
# spread (a near-singular, indefinite Hessian) is what makes FIRE — a single
# inertial timescale — oscillate and stall at high strain. The fix used here is
# the textbook robust second-order method for exactly this Hessian:
#
#   * matrix-free Hessian-vector products by finite difference of the analytic
#     gradient g = -F (no sparse Hessian assembled — numpy-only), and
#   * a Steihaug-Toint truncated CG that, the moment it meets a direction of
#     non-positive curvature dᵀHd <= 0 (a floppy / buckling mode), stops dividing
#     by ~0 and steps to the trust-region boundary along that direction.
#
# That single branch is the "floppy-mode awareness": the zero-stiffness modes are
# handled by the trust region rather than blowing the step up.
# --------------------------------------------------------------------------- #
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_U = 0.5 * (_GL_X + 1.0)          # Gauss-Legendre nodes on [0,1]
_GL_WU = 0.5 * _GL_W                 # ... and weights


def energy(net: PeriodicNetwork, gamma, p: mech.MechParams):
    """Total elastic energy (nonlinear axial + bending) at shear `gamma`.

    Consistent with `_forces`: F = -dE/dx exactly (the bond tension is the
    derivative of the per-bond axial energy, integrated here by 8-pt Gauss-
    Legendre over strain so the strain-stiffening / buckling law is honoured).
    This is the merit function for the trust-region ratio.
    """
    nodes, L = net.nodes, net.L
    i, j = net.edges[:, 0], net.edges[:, 1]
    raw = nodes[j] + net.edge_off * L - nodes[i]
    d = _le_min_image(raw, L, gamma)
    Lij = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    strain = (Lij - net.rest_length) / net.rest_length
    # U_axial = sum_bonds L0^2 * strain^2 * ∫_0^1 k_axial(strain*u) * u du
    sq = strain[:, None] * _GL_U[None, :]                 # (E, nq) strain samples
    kq = mech._axial_stiffness(sq, p) * net.k_scale[:, None]
    integ = np.sum(_GL_WU[None, :] * kq * _GL_U[None, :], axis=1)
    U_ax = float(np.sum(net.rest_length ** 2 * strain ** 2 * integ))
    U_bend = 0.0
    if len(net.bends):
        a, c, b = net.bends[:, 0], net.bends[:, 1], net.bends[:, 2]
        da = _le_min_image(nodes[a] - nodes[c], L, gamma)
        db = _le_min_image(nodes[b] - nodes[c], L, gamma)
        lap = da + db
        U_bend = 0.25 * p.k_bend * float(np.sum(lap * lap))
    return U_ax + U_bend


def _to_boundary(pvec, d, Delta):
    """Largest tau >= 0 with ||pvec + tau d|| = Delta (trust-region boundary)."""
    a = float(d @ d)
    b = 2.0 * float(pvec @ d)
    c = float(pvec @ pvec) - Delta * Delta
    disc = max(b * b - 4.0 * a * c, 0.0)
    return (-b + np.sqrt(disc)) / (2.0 * a) if a > 1e-30 else 0.0


def _steihaug_cg(g, hess_vec, Delta, tol, max_cg=120):
    """Truncated CG for  min_p  g·p + ½ p·H·p   s.t. ||p|| <= Delta.

    Returns (p, hit_boundary). Negative/zero-curvature directions (floppy modes)
    are taken to the trust-region boundary — this is what FIRE cannot do.
    """
    n = len(g)
    p = np.zeros(n)
    r = g.copy()                       # residual g + H p at p=0
    d = -r
    rr = float(r @ r)
    if np.sqrt(rr) <= tol:
        return p, False
    for _ in range(min(n, max_cg)):
        Hd = hess_vec(d)
        dHd = float(d @ Hd)
        if dHd <= 1e-12 * float(d @ d):        # floppy / negative-curvature mode
            return p + _to_boundary(p, d, Delta) * d, True
        alpha = rr / dHd
        p_next = p + alpha * d
        if np.linalg.norm(p_next) >= Delta:    # crossed the trust boundary
            return p + _to_boundary(p, d, Delta) * d, True
        r_next = r + alpha * Hd
        rr_next = float(r_next @ r_next)
        if np.sqrt(rr_next) <= tol:
            return p_next, False
        d = -r_next + (rr_next / rr) * d
        p, r, rr = p_next, r_next, rr_next
    return p, False


def relax_newton(net: PeriodicNetwork, gamma, p: mech.MechParams | None = None,
                 tol=None, max_outer=150, fire_warm=2500, max_cg=120, verbose=False):
    """Floppy-mode-aware trust-region Newton relaxation at fixed shear `gamma`.

    A short FIRE pre-relaxation drops into the basin cheaply; Newton then polishes
    to a tight force balance where FIRE stalls (high strain). Returns
    (relaxed_net, info) with the same info keys as `relax`, plus method.
    """
    p = p or mech.MechParams()
    tol = tol if tol is not None else p.ftol
    work = net.copy()

    if fire_warm:
        pw = mech.MechParams(**dict(p.__dict__))
        pw.max_iter, pw.ftol = fire_warm, tol
        work, _ = relax(work, gamma, pw)

    x = work.nodes.astype(float).copy()
    shape = x.shape

    def force_at(xx):
        work.nodes = xx
        return _forces(work, gamma, p)[0]

    def grad_at(xx):
        return -force_at(xx).reshape(-1)

    def energy_at(xx):
        work.nodes = xx
        return energy(work, gamma, p)

    g = grad_at(x)
    U = energy_at(x)
    fmax = float(np.max(np.linalg.norm(force_at(x), axis=1)))
    # trust radius capped near the rest-length scale: larger steps overshoot a
    # stiff network and make the radius oscillate.
    Delta, Dmax = 0.3, 1.5
    best_x, best_fmax = x.copy(), fmax       # the iterate need not be monotone in fmax
    outer = 0
    for outer in range(max_outer):
        if fmax < tol:
            break
        gloc = g

        def hess_vec(v):
            nv = np.linalg.norm(v)
            if nv < 1e-30:
                return np.zeros_like(v)
            h = 1e-6 / nv                       # probe displacement ~1e-6 um
            return (grad_at(x + h * v.reshape(shape)) - gloc) / h

        gnorm = float(np.linalg.norm(gloc))
        cg_tol = min(0.5, np.sqrt(max(gnorm, 1e-30))) * gnorm   # inexact-Newton forcing
        step, hit = _steihaug_cg(gloc, hess_vec, Delta, cg_tol, max_cg)

        x_trial = x + step.reshape(shape)
        U_trial = energy_at(x_trial)
        Hp = hess_vec(step)
        pred = -(float(gloc @ step) + 0.5 * float(step @ Hp))
        actual = U - U_trial
        rho = actual / pred if pred > 1e-30 else (1.0 if actual > 0 else -1.0)

        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and hit:
            Delta = min(2.0 * Delta, Dmax)
        accept = rho > 0.1 and actual > 0
        if accept:                             # accept
            x = x_trial
            g = grad_at(x)
            U = U_trial
            fmax = float(np.max(np.linalg.norm(force_at(x), axis=1)))
            if fmax < best_fmax:
                best_fmax, best_x = fmax, x.copy()
        if verbose:
            print(f"   nt{outer:3d} D={Delta:8.4f} |s|={np.linalg.norm(step):8.4f} "
                  f"hit={int(hit)} pred={pred:+.2e} act={actual:+.2e} rho={rho:+.2f} "
                  f"fmax={fmax:.4f} best={best_fmax:.4f} {'ACC' if accept else 'rej'}")
        if Delta < 1e-10:
            break

    work.nodes = _le_wrap(best_x, work.L, gamma)
    fmax = float(np.max(np.linalg.norm(_forces(work, gamma, p)[0], axis=1)))
    return work, dict(iters=outer + 1, converged=fmax < tol,
                      final_max_force=fmax, method="newton-tr")


def virial_shear_stress(net: PeriodicNetwork, gamma, p: mech.MechParams):
    """sigma_xy from the bond virial at the current configuration."""
    F, tension, dirv, Lij, _ = _forces(net, gamma, p)
    fvec = (tension[:, None] * dirv)          # bond force vector (on node j side)
    rvec = dirv * Lij[:, None]                # min-image bond vector
    V = net.L ** 3
    sigma_xy = float(np.sum(fvec[:, 0] * rvec[:, 1]) / V)
    return sigma_xy


def shear_modulus(net: PeriodicNetwork, gamma=0.02, p: mech.MechParams | None = None):
    """Relax at shear gamma and return G = sigma_xy / gamma (+ convergence info)."""
    p = p or mech.MechParams(traction=0.0, max_iter=8000, ftol=8e-3)
    relaxed, info = relax(net, gamma, p)
    sigma = virial_shear_stress(relaxed, gamma, p)
    return sigma / gamma, sigma, info


def shear_sweep(net: PeriodicNetwork, gammas, p: mech.MechParams | None = None,
                solver="fire"):
    """Quasi-static incremental shear: relax at each gamma starting from the
    previous strained configuration (physical loading path, monotonic, and far
    easier to converge than re-relaxing from the undeformed state each time).

    `solver`: "fire" (inertial, fast) or "newton" (floppy-mode-aware trust-region
    Newton — converges where FIRE stalls at high strain).

    Returns dict(gamma, sigma, fmax, converged) arrays.
    """
    p = p or mech.MechParams(traction=0.0, max_iter=12000, ftol=1.5e-2)
    cur = net.copy()
    sig, fm, cv = [], [], []
    prev_g = 0.0
    for g in gammas:
        # affine pre-shift for the strain increment: a near-equilibrium starting
        # guess that lets the solver converge in far fewer iterations (standard
        # for Lees-Edwards stepping).
        cur.nodes[:, 0] = np.mod(cur.nodes[:, 0] + (g - prev_g) * cur.nodes[:, 1], cur.L)
        prev_g = g
        cur, info = (relax_newton(cur, g, p) if solver == "newton"
                     else relax(cur, g, p))
        sig.append(virial_shear_stress(cur, g, p))
        fm.append(info["final_max_force"]); cv.append(info["converged"])
    return dict(gamma=np.asarray(gammas, float), sigma=np.array(sig),
                fmax=np.array(fm), converged=np.array(cv))
