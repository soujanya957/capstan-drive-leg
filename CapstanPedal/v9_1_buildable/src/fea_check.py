"""Linear-elastic FEA of the most loaded printed parts, under the same design loads as check_strength.py.

Each part is meshed from its CAD (gmsh, 2nd-order tetrahedra) and solved with scikit-fem. Bearings and
pins load their bores as a cosine pressure over the half they bear on; clamped faces are fixed. Loads:
  shin, thigh, crank:  the 4-bar's statics at the motor-limited foot force (28 N*m per output), standing
                       and crouched (the larger of that and the hop, as in check_strength.py)
  gantry, front wall:  the gantry / wall bearing loads of check_strength.py, swept round 8 directions
  motor plate:         each motor's rope pull and torque reaction on its stator screws, plus drum A's
                       shaft load on its two 6806s, swept round: with the gantry at 28 N*m, without it at
                       the 14 N*m limit those builds already need for drum A's nose
Material PETG, isotropic (E 2000 MPa, nu 0.38). Rated on von Mises against 45 MPa (along the layers)
and on the stress across the layers (the print axis) against 25 MPa. Stresses within 1.5 mm of a
fixed face are left out (the clamp is idealised there); the hottest 0.1 % of the remaining volume is
also left out, so a single bad element can't set the result. Target: SF >= 2.

Not modelled: contact, the bolted sandwich's slip (the shin is run both as a bonded sandwich and as
the core alone), fatigue, impact, creep, layer adhesion beyond the 25 MPa rating.
Run: python fea_check.py [part ...]      (needs gmsh, scikit-fem, pyamg)
"""
import math
import os
import sys
import tempfile

import gmsh
import numpy as np
import pyamg
from build123d import Pos, export_step
from skfem import Basis, ElementTetP2, ElementVector, FacetBasis, Functional, LinearForm, MeshTet, asm
from skfem.helpers import dot
from skfem.models.elasticity import lame_parameters, linear_elasticity

import generate as g

E, NU = 2000.0, 0.38
PETG_XY, PETG_Z = 45.0, 25.0
TAU = g.MOTOR_TAU * g.RATIO          # v9: 31.5 N*m at 9:1 (v8: 28)
TARGET = 2.0
CLAMP_BAND = 1.5
NO_GANTRY_SCALE = 0.5          # builds without the gantry are limited to 14 N*m on output A (check_strength.py)
results = []


# --- statics (as check_strength.py) ----------------------------------------------------------
def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def leg_statics(qt, qs, F):
    k, f, c, d = (np.array(p, float) for p in g.leg4_points(qt, qs))
    o = np.array(g.OUT, float)
    u = unit(d - c)
    fc = -cross2(f - k, F) / cross2(d - k, u)
    fk = -(F + fc * u)
    return fc * u, fk, cross2(c - o, fc * u), cross2(k - o, fk)


def motor_limited(qt, qs):
    best = (0.0, None)
    for ang in np.arange(60.0, 121.0, 2.0):
        Fu = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        _, _, ta, tb = leg_statics(qt, qs, Fu)
        s = TAU * 1000.0 / max(abs(ta), abs(tb))
        if s > best[0]:
            best = (s, Fu * s)
    return best[1]


LEG_CASES = {}
for tag, (r, phi) in {"standing": (259.8, -90.0), "crouched": (200.0, -90.0)}.items():
    q = g.leg4_ik(r, phi)
    F = motor_limited(*q)
    fcu, fk, _, _ = leg_statics(*q, F)
    LEG_CASES[tag] = (q, F, fcu, fk)


def to_local(v, psi):
    """Model XZ vector -> link-local xy (to_y maps local (x, y) to model (x, -y) after Rot(psi))."""
    x, y = v[0], -v[1]
    c, s = math.cos(math.radians(-psi)), math.sin(math.radians(-psi))
    return np.array([c * x - s * y, s * x + c * y, 0.0])


# --- meshing and solving ---------------------------------------------------------------------
def mesh_of(solid, h):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "part.step")
        export_step(solid, path)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, _, conn = gmsh.model.mesh.getElements(3)
        gmsh.finalize()
    idx = np.zeros(int(tags.max()) + 1, int)
    idx[tags.astype(int)] = np.arange(len(tags))
    t = idx[np.concatenate(conn).astype(int)].reshape(-1, 4).T
    used = np.unique(t)
    remap = -np.ones(len(tags), int)
    remap[used] = np.arange(len(used))
    return MeshTet(xyz.reshape(-1, 3)[used].T.copy(), remap[t])


class Part:
    def __init__(self, name, solid, h, print_axis):
        self.name, self.print_axis = name, np.asarray(print_axis, float)
        self.m = mesh_of(solid, h)
        self.e = ElementVector(ElementTetP2())
        self.ib = Basis(self.m, self.e)
        lam, mu = lame_parameters(E, NU)
        self.lam, self.mu = lam, mu
        self.K = asm(linear_elasticity(lam, mu), self.ib)
        bf = self.m.boundary_facets()
        self.bf = bf
        self.fc = self.m.p[:, self.m.facets[:, bf]].mean(axis=1)               # facet centroids
        # rigid-body modes for the AMG preconditioner
        loc = self.ib.doflocs
        self.B = np.zeros((self.K.shape[0], 6))
        cx, cy, cz = self.ib.split_indices()
        x, y, z = loc
        for c, idx in enumerate((cx, cy, cz)):
            self.B[idx, c] = 1.0
        self.B[cx, 3], self.B[cy, 3] = -y[cx], x[cy]
        self.B[cy, 4], self.B[cz, 4] = -z[cy], y[cz]
        self.B[cz, 5], self.B[cx, 5] = -x[cz], z[cx]
        print(f"[{name}] {self.m.t.shape[1]} tets, {self.K.shape[0]} dofs", flush=True)

    def bore(self, centre, axis, r, span, tol=0.35):
        """Boundary facets on the cylinder of radius r about the axis through centre, within span along it."""
        a = np.asarray(axis, float)
        rel = self.fc - np.asarray(centre, float)[:, None]
        along = a @ rel
        rad = np.linalg.norm(rel - np.outer(a, along), axis=0)
        sel = (abs(rad - r) < tol) & (along > span[0] - 0.01) & (along < span[1] + 0.01)
        return self.bf[sel]

    def face(self, axis, level, centre, r, tol=0.05):
        """Boundary facets on the plane (axis . x = level), within r of centre (projected)."""
        a = np.asarray(axis, float)
        rel = self.fc - np.asarray(centre, float)[:, None]
        along = a @ self.fc
        rad = np.linalg.norm(rel - np.outer(a, a @ rel), axis=0)
        return self.bf[(abs(along - level) < tol) & (rad < r)]

    def bearing_load(self, facets, centre, axis, F):
        """Force F (N, vector) from a pin or bearing on the part, as a cosine pressure on the half of the
        bore it bears on."""
        if len(facets) == 0:
            raise ValueError(f"{self.name}: no facets for a load at {centre}")
        F = np.asarray(F, float)
        Fn = np.linalg.norm(F)
        if Fn == 0:
            return np.zeros(self.K.shape[0])
        fu, a, c = F / Fn, np.asarray(axis, float), np.asarray(centre, float)
        fb = FacetBasis(self.m, self.e, facets=facets)

        def weight(x):
            rel = x - c[:, None, None]
            rad = rel - a[:, None, None] * np.einsum("i,ijk->jk", a, rel)
            rad = rad / np.maximum(np.linalg.norm(rad, axis=0), 1e-9)
            return np.maximum(0.0, np.einsum("i,ijk->jk", fu, rad))

        total = Functional(lambda w: weight(w.x)).assemble(fb)
        if total <= 0:
            raise ValueError(f"{self.name}: load at {centre} finds no bearing face")
        return LinearForm(lambda v, w: dot(fu[:, None, None] * weight(w.x), v) * (Fn / total)).assemble(fb)

    def spread_load(self, facets, F):
        """Force F spread uniformly over the facets (a clamped or screwed face)."""
        fb = FacetBasis(self.m, self.e, facets=facets)
        area = Functional(lambda w: 1.0 + 0 * w.x[0]).assemble(fb)
        F = np.asarray(F, float)
        return LinearForm(lambda v, w: dot(F[:, None, None] + 0 * w.x, v) / area).assemble(fb)

    def solve(self, fixed, loads):
        D = self.ib.get_dofs(facets=np.concatenate(fixed)).all()
        out = []
        I = np.setdiff1d(np.arange(self.K.shape[0]), D)
        Kc = self.K[I][:, I].tocsr()
        ml = pyamg.smoothed_aggregation_solver(Kc, B=self.B[I], max_coarse=2000)       # once per part
        for tag, f in loads:
            fc_ = f[I]
            u_ = np.zeros(self.K.shape[0])
            res = []
            u_[I] = ml.solve(fc_, tol=1e-8, accel="cg", maxiter=1500, residuals=res)
            print(f"    {tag}: {len(res)} CG iterations, residual {res[-1] / res[0]:.1e}", flush=True)
            if res[-1] / res[0] > 1e-5:
                raise RuntimeError(f"{self.name}: {tag} did not converge")
            out.append((tag, u_))
        return out, D

    def stresses(self, u, fixed):
        """Per element: peak von Mises, peak tensile stress along the print axis, element volume."""
        ub = self.ib.interpolate(u)
        gr = ub.grad                                                       # (3, 3, nel, nqp)
        eps = 0.5 * (gr + np.swapaxes(gr, 0, 1))
        tr = np.einsum("ii...->...", eps)
        sig = 2 * self.mu * eps + self.lam * np.eye(3)[:, :, None, None] * tr
        dev = sig - np.eye(3)[:, :, None, None] * np.einsum("ii...->...", sig) / 3
        vm = np.sqrt(1.5 * np.einsum("ij...,ij...->...", dev, dev)).max(axis=1)
        a = self.print_axis
        sz = np.einsum("i,ij...,j->...", a, sig, a).max(axis=1)
        vol = self.ib.dx.sum(axis=1)
        # leave out elements next to the fixed faces
        fp = np.unique(self.m.facets[:, np.concatenate(fixed)])
        cen = self.m.p[:, self.m.t].mean(axis=1)
        near = np.zeros(cen.shape[1], bool)
        for k in range(0, len(fp), 400):
            d = np.linalg.norm(cen[:, :, None] - self.m.p[:, fp[k:k + 400]][:, None, :], axis=0).min(axis=1)
            near |= d < CLAMP_BAND + 1.0
        return vm, sz, vol, near, cen


def peak(values, vol, mask, frac=0.001):
    """Largest value once the hottest `frac` of the volume (outside the mask) is left out."""
    v, w = values[~mask], vol[~mask]
    o = np.argsort(v)[::-1]
    cum = np.cumsum(w[o])
    k = np.searchsorted(cum, frac * w.sum())
    return v[o][min(k, len(v) - 1)], v.max()


def rate(part, fixed, loads, note=""):
    sols, _ = part.solve(fixed, loads)
    worst = None
    for tag, u in sols:
        vm, sz, vol, near, cen = part.stresses(u, fixed)
        vm_p, vm_max = peak(vm, vol, near)
        sz_p, _ = peak(sz, vol, near)
        sf = min(PETG_XY / vm_p, PETG_Z / max(sz_p, 1e-9))
        k = np.argmax(np.where(near, 0, vm))
        umax = np.abs(u[part.ib.nodal_dofs]).max()
        row = (part.name, tag, vm_p, sz_p, sf, umax, cen[:, k])
        print(f"  {tag:22s} von Mises {vm_p:6.1f} MPa (raw max {vm_max:6.1f})  across layers {sz_p:5.1f} MPa  "
              f"SF {sf:4.1f}  max disp {umax:5.2f} mm  hottest at ({cen[0, k]:.0f}, {cen[1, k]:.0f}, {cen[2, k]:.0f})",
              flush=True)
        if worst is None or sf < worst[4]:
            worst = row
    results.append(worst + (note,))
    return worst


def sweep(n=8):
    return [math.radians(360.0 * k / n) for k in range(n)]


# --- parts -----------------------------------------------------------------------------------
def shin_parts():
    lx, ly = g.polar(g.L4_C, 90.0 - g.L4_DELTA)
    core = g.leg4_shin()
    sandwich = core + Pos(0, 0, -g.SHIN_PLATE_T) * g.leg4_shin_plate(False) + Pos(0, 0, 8.0) * g.leg4_shin_plate(True)
    return (lx, ly), {"Shin, core alone": core, "Shin, bonded sandwich": sandwich}


def run_shin():
    (lx, ly), parts = shin_parts()
    z = (0, 0, 1)
    for name, solid in parts.items():
        p = Part(name, solid, 1.6, z)
        fixed = [p.bore((0, 0, 0), z, g.KNEE_BRG_POCKET, (0.0, g.KNEE_BRG[2]))]       # knee 636 (pinned)
        loads = []
        for tag, ((qt, qs), F, fcu, fk) in LEG_CASES.items():
            psi = 90.0 - qs
            k, f, c, d = (np.array(v_) for v_ in g.leg4_points(qt, qs))
            dl = to_local(d - k, psi)
            assert np.allclose(dl[:2], (lx, ly), atol=0.05), (dl, lx, ly)
            fl = p.bearing_load(p.bore((lx, ly, 0), z, g.PIN_BRG_POCKET, (0.0, 8.0)), (lx, ly, 0), z, to_local(fcu, psi))
            fl += p.bearing_load(p.bore((0, -g.L4_L2, 0), z, g.L4_CLAW_R[0], (0.0, 8.0)), (0, -g.L4_L2, 0), z, to_local(F, psi))
            loads.append((f"{tag} {np.linalg.norm(F):.0f} N", fl))
        print(f"[{name}]", flush=True)
        rate(p, fixed, loads, "knee fixed; coupler + foot loads")


def run_thigh():
    _, phi_b = -g.dir_deg(g.OUT, g.MA), -g.dir_deg(g.OUT, g.MB)
    qt0 = g.L4_Q0["thigh"]
    solid = g.leg4_thigh((phi_b + 180.0 - (90.0 - qt0)) % 60.0)
    z = (0, 0, 1)
    p = Part("Thigh", solid, 2.0, z)
    fixed = [p.bore((0, 0, 0), z, 27.1, (0.0, 8.0))]
    for k_ in range(6):
        x, y = g.polar(35.0, (phi_b + 180.0 - (90.0 - qt0)) % 60.0 + 60 * k_)
        fixed.append(p.bore((x, y, 0), z, g.M3_CLR / 2, (0.0, 8.0)))
    loads = []
    for tag, ((qt, qs), F, fcu, fk) in LEG_CASES.items():
        fl = p.bearing_load(p.bore((0, -g.L4_L1, 0), z, g.KNEE_PIN_CLR / 2, (0.0, 8.0)), (0, -g.L4_L1, 0), z, to_local(-fk, 90.0 - qt))
        loads.append((f"{tag} knee {np.linalg.norm(fk):.0f} N", fl))
    print("[Thigh]", flush=True)
    rate(p, fixed, loads, "hub bore + 6 bolt holes fixed; knee load")


def run_crank():
    phi_a = -g.dir_deg(g.OUT, g.MA)
    qs0 = g.L4_Q0["shin"]
    phase = (phi_a + 330.0 - (90.0 - (qs0 + 180.0 + g.L4_DELTA))) % 60.0
    solid = g.leg4_crank(phase, stub=True)
    z = (0, 0, 1)
    p = Part("Crank (with stub)", solid, 1.6, z)
    fixed = [p.bore((0, 0, 0), z, g.A_ARM_SPIGOT_R + 0.1, (0.0, 2.5))]
    for k_ in range(6):
        x, y = g.polar(24.0, phase + 60 * k_)
        fixed.append(p.bore((x, y, 0), z, g.M3_CLR / 2, (0.0, 8.0)))
    loads = []
    for tag, ((qt, qs), F, fcu, fk) in LEG_CASES.items():
        psi = 90.0 - (qs + 180.0 + g.L4_DELTA)
        fl = p.bearing_load(p.bore((0, -g.L4_C, 0), z, g.PIN_CLR / 2, (0.0, 8.0)), (0, -g.L4_C, 0), z, to_local(-fcu, psi))
        loads.append((f"{tag} pin {np.linalg.norm(fcu):.0f} N", fl))
    print("[Crank]", flush=True)
    rate(p, fixed, loads, "bolt circle fixed; coupler pin load, stub unsupported")


def gantry_load():
    y_crank = g.L4_Y["Leg4_Crank"] + 4.0
    y_b2 = g.B2_Y + g.B30[2] / 2
    y_g = g.GANTRY_Y[0] + g.GANTRY_BRG[2] / 2
    FC = max(np.linalg.norm(c[2]) for c in LEG_CASES.values())
    F = max(np.linalg.norm(c[1]) for c in LEG_CASES.values())
    wall_y = sum(g.WALL_BRG_Y) / 2
    share = (125.0 - wall_y) / (y_g - wall_y)
    return max(FC * (y_crank - y_b2) / (y_g - y_b2), share * F), share, F, FC


def run_gantry():
    Fg, *_ = gantry_load()
    y = (0, 1, 0)
    p = Part("Gantry plate", g.gantry(), 1.6, y)
    ox, oz = g.OUT
    g0 = g.GANTRY_Y[0]
    fixed = [p.bore((x, 0, z), y, g.M4_CLR / 2, (g0, g.GANTRY_Y[1])) for x, z in g.GANTRY_POSTS]
    fixed += [p.face(y, g0, (x, g0, z), g.GANTRY_POST_R) for x, z in g.GANTRY_POSTS]
    sel = p.bore((ox, 0, oz), y, (g.GANTRY_BRG[1] + g.PRESS) / 2, (g0, g0 + g.GANTRY_BRG[2]))
    loads = [(f"{math.degrees(a):3.0f} deg {Fg:.0f} N", p.bearing_load(sel, (ox, 0, oz), y, (Fg * math.cos(a), 0, Fg * math.sin(a))))
             for a in sweep()]
    print("[Gantry plate]", flush=True)
    rate(p, fixed, loads, "posts' end faces + bores fixed; 6903 load swept")


def run_wall():
    Fg, share, F, FC = gantry_load()
    FK = max(np.linalg.norm(c[3]) for c in LEG_CASES.values())
    Fw = max(FK, (1 - share) * F) + FC                                          # as check_strength.py (conservative)
    T = TAU / (g.BIG_PITCH_R / 1000.0)
    y = (0, 1, 0)
    p = Part("Front wall", g.front_plate(), 2.0, y)
    y0, y1 = g.FRONT_Y0, g.FRONT_Y1
    ox, oz = g.OUT
    fixed = [p.bore((x, 0, z), y, g.M4_CLR / 2, (y0, y1)) for x, z in g.POSTS]
    fixed += [p.face(y, y0, (x, y0, z), g.POST_R) for x, z in g.POSTS]
    s6808 = p.bore((ox, 0, oz), y, (g.B40[1] + g.PRESS) / 2, g.WALL_BRG_Y)
    tips = []
    for mx, mz in (g.MA, g.MB):
        sel = p.bore((mx, 0, mz), y, (g.B15[1] + g.PRESS) / 2, (y0, y1 - 1.0))
        tips.append(p.bearing_load(sel, (mx, 0, mz), y, T * unit((ox - mx, 0, oz - mz))))
    ins = [p.bore((x, 0, z), y, g.M4_INSERT / 2, (y1 - 7.0, y1)) for x, z in g.GANTRY_POSTS]
    loads = []
    for a in sweep():
        d = np.array([math.cos(a), 0, math.sin(a)])
        f = p.bearing_load(s6808, (ox, 0, oz), y, Fw * d) + sum(tips)
        for (x, z), s in zip(g.GANTRY_POSTS, ins):
            f += p.bearing_load(s, (x, 0, z), y, Fg / 2 * d)
        loads.append((f"{math.degrees(a):3.0f} deg {Fw:.0f}+{Fg:.0f} N", f))
    print("[Front wall]", flush=True)
    rate(p, fixed, loads, "post ends fixed; 6808 + gantry swept, rope pull on both tips")


def a_shaft_reactions(gantry):
    """Drum A as a continuous beam on its supports (y): the plate's two 6806 (A1, A2), B2 (drum B's
    bearing on A's tube, carried by the wall) and the gantry; the crank load at the crank. Returns the
    two plate reactions per newton of crank load."""
    ys = [g.A_R1_Y + 3.5, g.A_R2_Y + 3.5, g.B2_Y + 3.5] + ([g.GANTRY_Y[0] + 3.5] if gantry else [])
    yl = g.L4_Y["Leg4_Crank"] + 4.0
    n = 400
    xs = np.linspace(min(ys) - 1, max(ys + [yl]) + 1, n + 1)
    # Euler-Bernoulli beam FE, EI = 1 (uniform; only the reactions matter)
    K = np.zeros((2 * n + 2, 2 * n + 2))
    for i in range(n):
        L = xs[i + 1] - xs[i]
        k = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                      [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]) / L ** 3
        K[2 * i:2 * i + 4, 2 * i:2 * i + 4] += k
    f = np.zeros(2 * n + 2)
    f[2 * np.argmin(abs(xs - yl))] = 1.0
    sup = [2 * np.argmin(abs(xs - y_)) for y_ in ys]
    free = [i for i in range(2 * n + 2) if i not in sup]
    u = np.zeros(2 * n + 2)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    R = K @ u - f
    return -R[sup[0]], -R[sup[1]]


def run_plate():
    FC = max(np.linalg.norm(c[2]) for c in LEG_CASES.values())
    T = TAU / (g.BIG_PITCH_R / 1000.0)
    t_screw = TAU / g.RATIO / 4 / (g.STATOR_BC / 2000.0)                     # N per stator screw, tangential
    y = (0, 1, 0)
    p = Part("Motor plate", g.back_plate(), 2.0, y)
    y0, y1 = g.BACK_Y0, g.BACK_Y1
    ox, oz = g.OUT
    fixed = [p.bore((x, 0, z), y, g.M4_CLR / 2, (y0, y1)) for x, z in g.POSTS]
    fixed += [p.face(y, y1, (x, y1, z), g.POST_R) for x, z in g.POSTS]
    rope = np.zeros(p.K.shape[0])
    for mx, mz in (g.MA, g.MB):
        pull = T * unit((ox - mx, 0, oz - mz))
        for k_ in range(4):
            dx, dz = g.polar(g.STATOR_BC / 2, g.STATOR_PHASE + 90 * k_)
            sel = p.bore((mx + dx, 0, mz + dz), y, g.M4_CLR / 2, (y0, y1))
            tang = t_screw * unit((-dz, 0, dx))
            rope += p.bearing_load(sel, (mx + dx, 0, mz + dz), y, pull / 4 + tang)
    s1 = p.bore((ox, 0, oz), y, (g.B30[1] + g.PRESS) / 2, (g.A_R1_Y, g.A_R1_Y + g.B30[2]))
    s2 = p.bore((ox, 0, oz), y, (g.B30[1] + g.PRESS) / 2, (g.A_R2_Y, g.A_R2_Y + g.B30[2]))
    loads = []
    # with the gantry at 28 N*m; without it at the 14 N*m limit those builds need for drum A's nose
    for gantry, scale in ((True, 1.0), (False, NO_GANTRY_SCALE)):
        r1, r2 = a_shaft_reactions(gantry)
        print(f"[Motor plate] {'with' if gantry else 'without'} the gantry, drum A's plate bearings carry "
              f"{r1:+.2f} / {r2:+.2f} x the crank load", flush=True)
        for a in sweep():
            d = np.array([math.cos(a), 0, math.sin(a)])
            f = rope + p.bearing_load(s1, (ox, 0, oz), y, scale * FC * r1 * d) + p.bearing_load(s2, (ox, 0, oz), y, scale * FC * r2 * d)
            loads.append((f"{'' if gantry else 'no '}gantry {math.degrees(a):3.0f} deg", f))
    rate(p, fixed, loads, "post ends fixed; rope pull + torque on the stators, drum A swept; no gantry at 14 N*m")


def run_end_bracket():
    """v9: the hub-end bracket with its lightening windows. Held at its four ears (the corner posts' bolts);
    loaded on the six hub through-bolt holes by the pod's hub reaction (the quadruped's continuous-beam
    reaction at the motor-limited foot force, 606 N at 28 N*m scaled to this ratio), swept round, plus the
    ab/ad drive's torque (MOTOR_TAU x RATIO about the hub axis) as tangential loads on the bolts."""
    x = (1, 0, 0)
    b = g.end_bracket(front=False)
    p = Part("End bracket", b, 1.8, x)
    x0, x1 = g.END_X
    fixed = []
    for px, pz in g.POSTS[0::2]:
        fixed.append(p.bore((px, 0, pz), (0, 1, 0), g.M4_CLR / 2, (g.BACK_Y0 - 9.0, g.FRONT_Y1 + 9.0)))
    holes = [p.bore((x0, yy, zz), x, g.M4_CLR / 2, (0.0, x1 - x0)) for yy, zz in g.AB_HUB_SCREWS]
    F = 606.0 * TAU / 28.0
    T = TAU * 1000.0                                                           # N*mm
    r = g.AB_HUB_R
    loads = []
    for a in sweep():
        d = np.array([0.0, math.cos(a), math.sin(a)])
        f = np.zeros(p.K.shape[0])
        n = len(g.AB_HUB_SCREWS)
        for (yy, zz), h in zip(g.AB_HUB_SCREWS, holes):
            t = np.array([0.0, -(zz - g.AB_Z), yy - g.AB_Y]) / r                # tangential, about +X
            f += p.bearing_load(h, (x0, yy, zz), x, F / n * d + T / (n * r) * t)
        loads.append((f"{math.degrees(a):3.0f} deg {F:.0f} N + {TAU:.1f} N*m", f))
    print("[End bracket]", flush=True)
    rate(p, fixed, loads, "ears held on the post bolts; hub reaction swept + ab/ad torque on the 8 hub screws")


RUNS = {"end_bracket": run_end_bracket, "shin": run_shin, "thigh": run_thigh, "crank": run_crank, "gantry": run_gantry, "wall": run_wall, "plate": run_plate}

if __name__ == "__main__":
    which = sys.argv[1:] or list(RUNS)
    for tag, (q, F, fcu, fk) in LEG_CASES.items():
        print(f"{tag}: foot {np.linalg.norm(F):.0f} N, coupler {np.linalg.norm(fcu):.0f} N, knee {np.linalg.norm(fk):.0f} N")
    for w in which:
        RUNS[w]()
    print(f"\n{'part':26s} {'worst case':26s} {'vM':>6s} {'layers':>7s} {'SF':>5s} {'disp':>6s}")
    for name, tag, vm, sz, sf, umax, at, note in results:
        print(f"{'ok  ' if sf >= TARGET else 'LOW '}{name:22s} {tag:26s} {vm:6.1f} {sz:7.1f} {sf:5.1f} {umax:6.2f}  {note}")
    print("\nLinear-elastic FEA, isotropic PETG; MPa and mm. Not a substitute for a load test.")
    # the gantry variant is the one built: a no-gantry case is reported, not required (as check_strength.py)
    assert all(r[4] >= TARGET for r in results if not r[1].startswith("no gantry")), "a part is below SF 2"
